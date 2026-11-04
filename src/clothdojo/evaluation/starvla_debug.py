"""Single-process rollout worker for the MR0 StarVLA flatten debug rollout.

Bridges the MR0 StarVLA absolute 20-D action-frame EEF action space into the
existing Isaac RGB + Newton rollout loop. The policy (QwenMMDiT serving a
16-step chunk of per-arm absolute [xyz, rot6d, gripper] targets in the
piper_arm_kinematics action frame) runs in a separate py3.11 policy server
process on an H100 node; this worker converts the returned targets to
world-frame link6 poses through the fixed arm-base transforms plus the
constant +90 degree z rotation between the action frame and the Newton
link6 body frame, solves IK against the live Newton state, and forwards the
resulting 14-D position commands to the standard runtime.

The default panel follows the Isaac RGB dual-appearance (dualtex) evaluation
protocol: unified state pools record a visual profile per state, so one asset
run evaluates both in-distribution ("train") and held-out ("test_all")
texture-pool appearances. cam_high frames the policy actually consumed are
saved next to each trajectory for review, because the generic render-review
path re-renders without texture layouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from clothdojo.rollout.starvla import StarVLAEFCommandEpisode, StarVLAPolicyAdapter


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
        return self._inner.identity

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
    adapter = StarVLAPolicyAdapter(args.policy_endpoint)
    identity = adapter.identity
    print(json.dumps({"phase": "policy_loaded", "identity": identity}), flush=True)

    flatten_protocol = load_evaluation_protocol(project / "configs/evaluation/shirt_flatten.yaml")
    resolved_templates = resolve_canonical_templates(
        templates,
        [item["asset_id"] for item in selected],
        target_settings=flatten_protocol.target.to_json_dict(),
    )
    scenario = replace(
        load_rollout_scenario(project / "configs/rollout/shirt_flatten.yaml"),
        action_horizon=adapter.action_horizon,
        steps=args.steps,
    )
    if scenario.prompt != "flatten the shirt":
        raise ValueError(f"unexpected scenario prompt: {scenario.prompt!r}")
    if not 1 <= args.exec_horizon <= adapter.action_horizon:
        raise ValueError("exec-horizon must be in [1, action_horizon]")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectories").mkdir(exist_ok=True)
    plan = {
        "policy": identity,
        "scenario": scenario.to_json_dict(),
        "execution": {
            "mode": "chunk",
            "exec_horizon": args.exec_horizon,
            "note": "per-request StarVLA predict_action chunks",
        },
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
                "policy_kind": "mr0_starvla_debug",
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
            producer="clothdojo_starvla_debug",
            training_eligible=False,
            storage_kind="render_only_v1",
        ) as episode:
            bridge = StarVLAEFCommandEpisode(episode, runtime)
            bridge.begin()
            print(
                json.dumps(
                    {
                        "phase": "running",
                        "asset": asset_id,
                        "steps": args.steps,
                        "frame_check": bridge.frame_check,
                    }
                ),
                flush=True,
            )
            recorder = _ObservationRecorder(
                adapter,
                tuple(path_ / "obs_cam_high.mp4" for path_ in targets),
                stride=args.obs_stride,
            )
            try:
                report = run_episode(
                    bridge,
                    recorder,
                    scenarios[0],
                    mode="chunk",
                    exec_horizon=args.exec_horizon,
                    policy_seed=seed,
                )
            finally:
                recorder.close()
            left_errors = [row["left_pos_err"] for row in bridge.diagnostics]
            right_errors = [row["right_pos_err"] for row in bridge.diagnostics]
            bridge_summary = {
                "tracked_steps": len(bridge.diagnostics),
                "left_pos_err_median": float(np.median(left_errors)) if left_errors else None,
                "left_pos_err_p90": float(np.percentile(left_errors, 90)) if left_errors else None,
                "right_pos_err_median": float(np.median(right_errors)) if right_errors else None,
                "right_pos_err_p90": (
                    float(np.percentile(right_errors, 90)) if right_errors else None
                ),
                "ik_fallbacks": bridge._ik_fallbacks,
                "frame_check": bridge.frame_check,
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
    parser.add_argument("--policy-endpoint", default="127.0.0.1:8741")
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", default=".")
    parser.add_argument(
        "--states", default="data/states/unified_common_100_100_flatten_dualtex_v1",
        help="initial-state pool; dual-appearance pools carry per-state visual profiles",
    )
    parser.add_argument(
        "--templates", default="data/benchmarks/unified_common_100_100_templates_v2",
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
    parser.add_argument("--exec-horizon", type=int, default=1)
    parser.add_argument("--visual-profile", choices=VISUAL_PROFILES, default="train")
    parser.add_argument("--visual-seed", type=int, default=20260911)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    args = parser.parse_args()
    if args.limit_per_asset < 1:
        parser.error("limit-per-asset must be positive")
    if args.exec_horizon < 1:
        parser.error("exec-horizon must be positive")
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
