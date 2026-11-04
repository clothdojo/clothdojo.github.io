"""Single-process rollout worker for the LingBot-VA flatten EEF debug rollout.

Bridges the LingBot-VA EEF action space into the existing Isaac RGB + Newton rollout
loop. The policy (16-dim episode-initial-relative EEF deltas with achieved-pose
semantics) runs in a separate py3.12 policy server process; this worker converts the
returned targets to absolute attach-frame poses, solves IK against the live Newton
state, and forwards the resulting 14-dim position commands to the standard runtime.

The default panel follows the Isaac RGB dual-appearance (dualtex) evaluation
protocol: unified state pools record a visual profile per state, so one asset run
evaluates both in-distribution ("train") and held-out ("test_all") texture-pool
appearances. cam_high frames the policy actually consumed are saved next to each
trajectory for review, because the generic render-review path re-renders without
texture layouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
ACTION_DIM = 16
LEFT_DELTA = slice(0, 7)
RIGHT_DELTA = slice(8, 15)
LEFT_GRIP = 7
RIGHT_GRIP = 15


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply xyzw quaternions, batched on the last axis."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    xyz = (
        left[..., 3:4] * right[..., :3]
        + right[..., 3:4] * left[..., :3]
        + np.cross(left[..., :3], right[..., :3])
    )
    scalar = left[..., 3:4] * right[..., 3:4] - np.sum(
        left[..., :3] * right[..., :3], axis=-1, keepdims=True
    )
    value = np.concatenate((xyz, scalar), axis=-1)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    return np.concatenate((-value[:3], value[3:4]))


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by an xyzw quaternion."""

    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return rotation @ np.asarray(vector, dtype=np.float64)

class LingBotPolicyAdapter:
    """Rollout Policy protocol backed by the py3.12 LingBot-VA policy server.

    The engine is configured with ``action_horizon == exec_horizon == 1`` so the
    policy observes every control step, exactly like the upstream client loop; the
    autoregressive chunk refill happens inside the policy action queue.
    Requests and responses travel as length-prefixed pickles on one TCP connection.
    """

    def __init__(self, endpoint: str) -> None:
        host, port = endpoint.rsplit(":", 1)
        self._address = (host, int(port))
        self._socket: socket.socket | None = None

    def _connect(self) -> socket.socket:
        if self._socket is None:
            self._socket = socket.create_connection(self._address, timeout=3600)
        return self._socket

    @staticmethod
    def _recv_exact(sock: socket.socket, count: int) -> bytes:
        chunks = []
        remaining = count
        while remaining:
            chunk = sock.recv(min(remaining, 1 << 20))
            if not chunk:
                raise ConnectionError("lingbot policy server closed the connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _request(self, payload: dict) -> dict:
        data = pickle.dumps(payload, protocol=5)
        sock = self._connect()
        sock.sendall(struct.pack("<Q", len(data)) + data)
        (length,) = struct.unpack("<Q", self._recv_exact(sock, 8))
        response = pickle.loads(self._recv_exact(sock, length))
        if "error" in response:
            raise RuntimeError("lingbot policy server: " + str(response["error"]))
        return response

    def identity(self) -> dict:
        return self._request({"identity": True})["identity"]

    def reset(self, seed: int, *, mode: str, exec_horizon: int) -> None:
        if mode != "chunk" or exec_horizon != 1:
            raise ValueError("LingBot-VA debug rollout requires chunk mode with exec_horizon 1")
        self._request({"reset": int(seed)})

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
        *,
        inference_delay_steps: int = 0,
    ) -> np.ndarray:
        if inference_delay_steps:
            raise ValueError("the debug rollout does not use RTC delays")
        payload: dict = {"task": prompt}
        for key in CAMERA_KEYS:
            images = np.asarray(observation[key])
            if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(f"expected uint8 [batch, H, W, 3] images for {key}, got {images.shape}")
            payload[key] = np.ascontiguousarray(images)
        action = np.asarray(self._request(payload)["action"], dtype=np.float64)
        if action.shape != (1, ACTION_DIM) or not np.isfinite(action).all():
            raise ValueError(f"unexpected LingBot-VA action: {action.shape}")
        return action[:, None, :]


def _pool_visual_profiles(pool):
    """Read per-state visual profiles recorded in dual-appearance pools."""

    profiles = pool.metadata.get("visual_profiles")
    if profiles is None:
        return None
    if (
        not isinstance(profiles, list)
        or len(profiles) != pool.state_count
        or any(not isinstance(name, str) or not name for name in profiles)
    ):
        raise ValueError("pool visual_profiles metadata is malformed")
    return tuple(profiles)


class _ObservationRecorder:
    """Policy wrapper saving the cam_high frames the policy actually consumed."""

    def __init__(self, inner, destinations, *, stride):
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError("observation stride must be a positive integer")
        self._inner = inner
        self._stride = stride
        self._calls = 0
        self._destinations = [str(destination) for destination in destinations]
        self._encoders = [None] * len(self._destinations)

    def _encode(self, index, image):
        import av

        encoder = self._encoders[index]
        if encoder is None:
            container = av.open(self._destinations[index], mode="w")
            stream = container.add_stream("libx264", rate=10)
            stream.height, stream.width = int(image.shape[0]), int(image.shape[1])
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "veryfast", "crf": "23"}
            encoder = self._encoders[index] = (container, stream)
        container, stream = encoder
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    def identity(self):
        return self._inner.identity()

    def reset(self, seed, *, mode, exec_horizon):
        return self._inner.reset(seed, mode=mode, exec_horizon=exec_horizon)

    def infer(self, observation, prompt, *, inference_delay_steps=0):
        if self._calls % self._stride == 0:
            frames = np.asarray(observation["observation.images.cam_high"])
            if len(frames) != len(self._encoders):
                raise ValueError("cam_high frame count does not match recorded worlds")
            for index, frame in enumerate(frames):
                self._encode(index, frame)
        self._calls += 1
        return self._inner.infer(
            observation, prompt, inference_delay_steps=inference_delay_steps
        )

    def close(self):
        for encoder in self._encoders:
            if encoder is None:
                continue
            container, stream = encoder
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._encoders = []


class EEFCommandEpisode:
    """Episode wrapper converting 16-dim EEF targets into 14-dim runtime commands.

    Action semantics follow ``clothdojo_lingbot_va_eef_actions_v1``: translation
    deltas and ``init_quat^-1 * target_quat`` relative to the episode-initial
    attach frame (Piper semantic TCP); gripper channels are normalized open
    commands. Targets are converted to end-effector body frames through the
    attach offsets of the robot profile before IK.
    """

    action_dim = ACTION_DIM

    def __init__(self, episode, runtime) -> None:
        from clothdojo.teleop.ik import DualArmIKController

        if int(episode.world_count) != 1:
            raise ValueError("the EEF IK bridge supports a single world")
        self.episode = episode
        self.ik = DualArmIKController(runtime, iterations=20, rotation_weight=10.0)
        # The v30 EEF-delta labels use the Piper semantic TCP frame: link6 composed
        # with translate(0, 0, 0.13503) and an extra Rz(+90 deg) axis convention
        # (verified 2026-09-15 against base2 joints: label deltas round-trip through
        # the bridge at 4e-5 m / 1e-5 rad only with this rotation; the workstation
        # USD attach frame carries identity rotation). The translation lies on the
        # z axis, so the position offset is unchanged by the extra rotation.
        self._offsets = tuple(
            (np.asarray(frame.position, dtype=np.float64),
             np.asarray(frame.rotation_xyzw, dtype=np.float64))
            for frame in runtime.scene.robot_profile.attach_frames
        )
        self._init_attach: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._last_targets: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self.diagnostics: list[dict] = []
        self._step_index = 0
        self._ik_fallbacks = 0

    @property
    def world_count(self) -> int:
        return self.episode.world_count

    def observe(self) -> dict[str, np.ndarray]:
        return self.episode.observe()

    def synchronize(self) -> None:
        self.episode.synchronize()

    def _body_poses(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        left_pos, left_quat, right_pos, right_quat = self.ik.measured_targets()
        return (left_pos, left_quat), (right_pos, right_quat)

    def _attach_pose(self, body: tuple[np.ndarray, np.ndarray],
                     offset: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        position, quaternion = body
        offset_pos, offset_quat = offset
        return (
            np.asarray(position, dtype=np.float64) + _quat_rotate(quaternion, offset_pos),
            _quat_multiply(quaternion, offset_quat),
        )

    def begin(self) -> None:
        """Record the episode-initial attach poses after the state is restored."""

        self.ik.reset_from_state()
        bodies = self._body_poses()
        self._init_attach = {
            "left": self._attach_pose(bodies[0], self._offsets[0]),
            "right": self._attach_pose(bodies[1], self._offsets[1]),
        }
        self._last_targets = None
        self.diagnostics = []
        self._step_index = 0
        self._ik_fallbacks = 0

    def step(self, eef_actions: np.ndarray) -> None:
        actions = np.asarray(eef_actions, dtype=np.float64)
        if actions.shape != (1, ACTION_DIM):
            raise ValueError(f"EEF actions must have shape (1, {ACTION_DIM})")
        if self._init_attach is None:
            raise RuntimeError("call begin() before the first step")
        delta = actions[0]

        bodies = self._body_poses()
        achieved = {
            "left": self._attach_pose(bodies[0], self._offsets[0]),
            "right": self._attach_pose(bodies[1], self._offsets[1]),
        }
        targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        record = {"step": self._step_index}
        for side, delta_slice in (("left", LEFT_DELTA), ("right", RIGHT_DELTA)):
            init_pos, init_quat = self._init_attach[side]
            target_pos = init_pos + delta[delta_slice][:3]
            target_quat = _quat_multiply(init_quat, delta[delta_slice][3:7])
            targets[side] = (target_pos, target_quat)
            if self._last_targets is not None:
                previous_pos, previous_quat = self._last_targets[side]
                achieved_pos, achieved_quat = achieved[side]
                relative = _quat_multiply(_quat_conjugate(previous_quat), achieved_quat)
                record[f"{side}_pos_err"] = float(
                    np.linalg.norm(achieved_pos - previous_pos)
                )
                record[f"{side}_rot_err"] = float(
                    2.0 * np.arcsin(min(1.0, float(np.linalg.norm(relative[:3]))))
                )
        if self._last_targets is not None:
            self.diagnostics.append(record)
        self._last_targets = targets

        matrices = {}
        for side, index in (("left", 0), ("right", 1)):
            target_pos, target_quat = targets[side]
            offset_pos, offset_quat = self._offsets[index]
            body_quat = _quat_multiply(target_quat, _quat_conjugate(offset_quat))
            body_pos = target_pos - _quat_rotate(body_quat, offset_pos)
            from clothdojo.teleop.ik import _pose_matrix

            matrices[side] = _pose_matrix(body_pos, body_quat)
        self.ik.sync_from_state()
        self.ik.set_targets(matrices["left"], matrices["right"])
        arm = np.asarray(self.ik.solve().numpy(), dtype=np.float32).reshape(12)
        if self.ik.solution_status() != 0:
            self._ik_fallbacks += 1
        gripper = np.asarray((delta[LEFT_GRIP], delta[RIGHT_GRIP]), dtype=np.float32)
        command = np.concatenate(
            (arm[:6], gripper[:1], arm[6:], gripper[1:])
        ).astype(np.float32)
        if not np.isfinite(command).all():
            raise FloatingPointError("non-finite bridged joint command")
        self.episode.step(command[None, :])
        self._step_index += 1


def evaluate(args, isaac_app=None) -> None:
    import torch

    from clothdojo.assets import GarmentLibrary
    from clothdojo.evaluation.metric import load_canonical_template
    from clothdojo.evaluation.protocol import load_evaluation_protocol
    from clothdojo.evaluation.scoring import score_evaluation_trajectory
    from clothdojo.evaluation.templates import resolve_canonical_templates
    from clothdojo.rendering import policy_observation_settings
    from clothdojo.rollout.engine import run_episode
    from clothdojo.rollout.initial_states import (
        InitialStatePool,
        load_initial_state_collection,
        validate_pool_compatibility,
    )
    from clothdojo.rollout.newton import NewtonTrajectoryEpisode
    from clothdojo.rollout.scenario import load_rollout_scenario
    from clothdojo.rollout.scheduling import select_assets
    from clothdojo.rollout.trajectory_collection import atomic_write_json, policy_seed_for_batch
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

    torch.set_num_threads(1)
    project = Path(args.project).resolve()
    assets_root = project / "assets"
    states = Path(args.states)
    if not states.is_absolute():
        states = project / states
    templates = Path(args.templates)
    if not templates.is_absolute():
        templates = project / templates
    meta = load_initial_state_collection(states)
    selected = select_assets(
        meta["assets"], seen_assets=args.seen_assets, unseen_assets=args.unseen_assets
    )
    first_pool = InitialStatePool.load(states / selected[0]["asset_id"] / "states.npz")
    pool_profiles = _pool_visual_profiles(first_pool)
    state_indexes = None
    if args.state_indexes is not None:
        state_indexes = [
            int(token) for token in args.state_indexes.split(",") if token.strip()
        ]
        if not state_indexes:
            raise ValueError("state-indexes must list at least one state")
        for index in state_indexes:
            if not 0 <= index < first_pool.state_count:
                raise ValueError(f"state index {index} outside the pool")
    if pool_profiles is not None:
        chosen = (
            state_indexes
            if state_indexes is not None
            else list(range(min(args.limit_per_asset, first_pool.state_count)))
        )
        selected_profiles = tuple(dict.fromkeys(pool_profiles[index] for index in chosen))
    else:
        selected_profiles = None
    adapter = LingBotPolicyAdapter(args.policy_endpoint)
    identity = adapter.identity()
    print(json.dumps({"phase": "policy_loaded", "identity": identity}), flush=True)

    flatten_protocol = load_evaluation_protocol(project / "configs/evaluation/shirt_flatten.yaml")
    resolved_templates = resolve_canonical_templates(
        templates,
        [item["asset_id"] for item in selected],
        target_settings=flatten_protocol.target.to_json_dict(),
    )
    scenario = replace(
        load_rollout_scenario(project / "configs/rollout/shirt_flatten.yaml"),
        action_horizon=1,
        steps=args.steps,
    )
    if scenario.prompt != "flatten the shirt":
        raise ValueError(f"unexpected scenario prompt: {scenario.prompt!r}")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectories").mkdir(exist_ok=True)
    plan = {
        "policy": identity,
        "scenario": scenario.to_json_dict(),
        "execution": {"mode": "chunk", "exec_horizon": 1, "note": "per-step LingBot select_action"},
        "assets": selected,
        "states": str(states),
        "storage_kind": "render_only_v1",
        "observation": policy_observation_settings(
            cloth_visual="rgb",
            visual_profile=None if selected_profiles else args.visual_profile,
            visual_profiles=selected_profiles,
            visual_seed=args.visual_seed,
        ),
        "template_sha256": {asset: value[1] for asset, value in resolved_templates.items()},
    }
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("debug output already belongs to a different plan")
    atomic_write_json(plan_path, plan)
    protocols = (
        {"flatten": replace(flatten_protocol, horizon_seconds=args.steps / 50)}
        if args.steps >= 150
        else {}
    )
    atomic_write_json(
        output / "protocols.json",
        {name: protocol.to_json_dict() for name, protocol in protocols.items()},
    )

    results = []
    for item in selected:
        asset_id = item["asset_id"]
        print(json.dumps({"phase": "building_scene", "asset": asset_id}), flush=True)
        pool_path = states / asset_id / "states.npz"
        pool = InitialStatePool.load(pool_path)
        asset_profiles = _pool_visual_profiles(pool)
        if state_indexes is not None:
            for index in state_indexes:
                if not 0 <= index < pool.state_count:
                    raise ValueError(f"state index {index} outside the pool: {asset_id}")
            indexes = list(state_indexes)
        else:
            indexes = list(range(min(args.limit_per_asset, pool.state_count)))
        if asset_profiles is not None:
            episode_profiles = tuple(asset_profiles[index] for index in indexes)
            if not set(episode_profiles) <= set(selected_profiles):
                raise ValueError(f"pool visual profiles differ from the plan: {asset_id}")
        else:
            episode_profiles = (args.visual_profile,) * len(indexes)
        count = len(indexes)
        garment = GarmentLibrary(assets_root).converted(asset_id)
        validate_pool_compatibility(pool, garment_path=garment, particle_count=pool.particle_count)
        with pool_path.open("rb") as stream:
            pool_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if pool_hash != item["sha256"]:
            raise ValueError(f"initial state pool changed: {asset_id}")
        scene = build_scene(
            assets_root,
            task_name=scenario.task_name,
            spec=ShirtFlattenSpec(garment_id=asset_id, world_count=count),
            device="cuda:0",
            visuals=False,
            cloth_visual="white",
        )
        runtime = ShirtFlattenRuntime(scene)
        from clothdojo.rollout.isaac import IsaacCameraObservation

        camera = IsaacCameraObservation(
            isaac_app, runtime, asset_path=garment, assets_root=assets_root,
            visual_profile=None if asset_profiles is not None else args.visual_profile,
            visual_profiles=(
                tuple(dict.fromkeys(episode_profiles))
                if asset_profiles is not None
                else None
            ),
            visual_seed=args.visual_seed,
        )
        runtime.reset()
        runtime.restore_particles(
            np.ascontiguousarray(pool.particle_q[indexes]),
            np.ascontiguousarray(pool.particle_qd[indexes]),
        )
        scenarios = tuple(
            replace(scenario, garment_id=asset_id, seed=int(pool.seeds[i])) for i in indexes
        )
        targets = tuple(output / "trajectories" / f"{asset_id}_{i:03d}" for i in indexes)
        appearances = camera.begin_episodes(
            tuple(path.name for path in targets),
            visual_profiles=episode_profiles if asset_profiles is not None else None,
        )
        seed = policy_seed_for_batch(np.asarray(pool.seeds[indexes], dtype=np.int32))
        if any(path.exists() for path in targets):
            raise FileExistsError("use a fresh output for the debug rollout")
        metadata = tuple(
            {
                "asset_id": asset_id,
                "policy_kind": "lingbot_va_debug",
                "policy": identity,
                "policy_seed": seed,
                "replicate": 0,
                "cloth_visual": "rgb",
                "observation": plan["observation"],
                "visual_profile": episode_profiles[i],
                "visual_layout": appearances[i],
                "initial_state": {
                    "index": indexes[i],
                    "seed": int(pool.seeds[indexes[i]]),
                    "pool_sha256": pool_hash,
                },
            }
            for i in range(count)
        )
        with NewtonTrajectoryEpisode(
            runtime,
            targets,
            scenarios,
            assets_root=assets_root,
            observation_provider=camera,
            trajectory_metadata=metadata,
            producer="clothdojo_lingbot_debug",
            training_eligible=False,
            storage_kind="render_only_v1",
        ) as episode:
            bridge = EEFCommandEpisode(episode, runtime)
            bridge.begin()
            print(json.dumps({"phase": "running", "asset": asset_id, "steps": args.steps}), flush=True)
            recorder = _ObservationRecorder(
                adapter,
                tuple(path_ / "obs_cam_high.mp4" for path_ in targets),
                stride=args.obs_stride,
            )
            try:
                report = run_episode(
                    bridge, recorder, scenarios[0],
                    mode="chunk", exec_horizon=1, policy_seed=seed,
                )
            finally:
                recorder.close()
            pos_errors = [row["left_pos_err"] for row in bridge.diagnostics]
            right_errors = [row["right_pos_err"] for row in bridge.diagnostics]
            bridge_summary = {
                "tracked_steps": len(bridge.diagnostics),
                "left_pos_err_median": float(np.median(pos_errors)) if pos_errors else None,
                "left_pos_err_p90": float(np.percentile(pos_errors, 90)) if pos_errors else None,
                "right_pos_err_median": float(np.median(right_errors)) if right_errors else None,
                "ik_fallbacks": bridge._ik_fallbacks,
            }
        for index, path in enumerate(targets):
            atomic_write_json(path / "eef_bridge_diagnostics.json", bridge.diagnostics)
            row = {
                "asset_id": asset_id,
                "initial_state_index": indexes[index],
                "visual_profile": episode_profiles[index],
                "path": str(path),
                "split": item.get("split"),
                "execution": report.to_json_dict(),
                "eef_bridge": bridge_summary,
            }
            for name, protocol in protocols.items():
                scored = score_evaluation_trajectory(
                    path,
                    load_canonical_template(resolved_templates[asset_id][0]),
                    protocol,
                    assets_root=assets_root,
                )
                row[name] = scored.result.to_json_dict()
            results.append(row)
        atomic_write_json(output / "results.json", results)
        print(json.dumps({"asset": asset_id, "completed_cases": len(results)}), flush=True)
        camera.close()
        del episode, camera, runtime, scene
        torch.cuda.empty_cache()
    atomic_write_json(output / "status.json", {"status": "complete", "cases": len(results)})


def main() -> None:
    from clothdojo.assets.visual_benchmark import VISUAL_PROFILES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-endpoint", default="127.0.0.1:8731")
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", default=".")
    parser.add_argument(
        "--states", default="data/states/unified_common_100_100_flatten_dualtex_v1",
        help="initial-state pool; dual-appearance pools carry per-state visual profiles",
    )
    parser.add_argument(
        "--templates", default="data/benchmarks/unified_common_100_100_templates_v1",
    )
    parser.add_argument(
        "--state-indexes", default=None,
        help="comma-separated pool state indexes to roll out, e.g. 0,4",
    )
    parser.add_argument("--seen-assets", type=int, default=1)
    parser.add_argument("--unseen-assets", type=int, default=1)
    parser.add_argument("--limit-per-asset", type=int, default=1)
    parser.add_argument("--obs-stride", type=int, default=5)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--visual-profile", choices=VISUAL_PROFILES, default="train")
    parser.add_argument("--visual-seed", type=int, default=20260911)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    args = parser.parse_args()
    if args.limit_per_asset < 1:
        parser.error("limit-per-asset must be positive")
    from clothdojo.rendering import policy_observation_settings

    policy_observation_settings(
        cloth_visual="rgb", visual_profile=args.visual_profile, visual_seed=args.visual_seed
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from clothdojo.rollout.isaac import isaac_application

    with isaac_application(args.gpu) as app:
        try:
            evaluate(args, app)
        except BaseException as error:
            import traceback

            from clothdojo.rollout.trajectory_collection import atomic_write_json

            traceback.print_exc()
            output = Path(args.output).resolve()
            output.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output / "status.json", {
                "status": "failed", "error": str(error), "traceback": traceback.format_exc(),
            })
            raise


if __name__ == "__main__":
    main()
