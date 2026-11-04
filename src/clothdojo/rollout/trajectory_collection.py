"""One trajectory collection runner shared by rollout and evaluation plans."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import traceback
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.rendering import POLICY_RENDER_SIZE, policy_observation_settings
from clothdojo.rollout.initial_states import InitialStatePool, validate_pool_compatibility
from clothdojo.rollout.policy import RemotePolicy
from clothdojo.rollout.scenario import load_rollout_scenario
from clothdojo.rollout.scheduling import run_policy_asset_processes, run_policy_asset_workers
from clothdojo.trajectory import TrajectoryReader
from clothdojo.trajectory.schema import RENDER_ONLY_STORAGE_KIND

_BASE_METADATA_KEYS = frozenset(
    {
        "asset_id",
        "policy_kind",
        "policy",
        "policy_seed",
        "cloth_visual",
        "observation",
        "execution",
        "parallelism",
        "initial_state",
        "replicate",
        "visual_layout",
    }
)


@dataclass(frozen=True, slots=True)
class TrajectoryAssetJob:
    """All cases for one asset and one immutable initial-state pool."""

    asset_id: str
    pool: str
    pool_sha256: str
    state_count: int
    replicates: tuple[int | None, ...] = (None,)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("trajectory asset id must not be empty")
        if (
            isinstance(self.state_count, bool)
            or not isinstance(self.state_count, int)
            or self.state_count < 1
        ):
            raise ValueError("trajectory state count must be positive")
        if not self.replicates:
            raise ValueError("trajectory job must contain at least one replicate")
        concrete = [value for value in self.replicates if value is not None]
        if len(concrete) != len(set(concrete)) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in concrete
        ):
            raise ValueError("trajectory replicates must be unique non-negative integers")
        if None in self.replicates and self.replicates != (None,):
            raise ValueError("the unreplicated trajectory layout cannot mix with replicates")
        overlap = _BASE_METADATA_KEYS & set(self.metadata)
        if overlap:
            raise ValueError(f"trajectory job metadata contains reserved keys: {sorted(overlap)}")


@dataclass(frozen=True, slots=True)
class TrajectoryCollectionSpec:
    """Execution and publication settings common to every asset worker."""

    scenario_path: str
    assets_root: str
    trajectories_root: str
    expected_checkpoint_sha256: str | None
    policy_identity: dict[str, Any]
    execution: str
    exec_horizon: int
    rtc_delay_steps: int
    horizon_steps: int
    batch_size: int
    producer: str
    training_eligible: bool
    compressed: bool = True
    cloth_visual: str = "nocs"
    storage_kind: str = "full"
    visual_profile: str | None = None
    visual_profiles: tuple[str, ...] | None = None
    visual_seed: int = 0
    record_video: bool = False
    robot_id: str = "piper"

    def __post_init__(self) -> None:
        from clothdojo.robots import robot_ids

        if self.robot_id not in robot_ids():
            raise ValueError(f"unknown rollout robot {self.robot_id!r}")
        self.observation_settings()
        if not isinstance(self.record_video, bool):
            raise TypeError("record_video must be a bool")
        if self.record_video and self.execution != "chunk":
            raise ValueError("observation video requires chunk execution")
        if self.execution not in {"chunk", "rtc"}:
            raise ValueError("trajectory execution must be 'chunk' or 'rtc'")
        for name, value in (
            ("exec_horizon", self.exec_horizon),
            ("horizon_steps", self.horizon_steps),
            ("batch_size", self.batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"trajectory {name} must be a positive integer")
        if (
            isinstance(self.rtc_delay_steps, bool)
            or not isinstance(self.rtc_delay_steps, int)
            or self.rtc_delay_steps < 0
        ):
            raise ValueError("trajectory rtc_delay_steps must be a non-negative integer")
        if self.execution == "chunk" and self.rtc_delay_steps:
            raise ValueError("chunk trajectories do not accept an RTC delay")
        if self.execution == "rtc" and self.rtc_delay_steps >= self.exec_horizon:
            raise ValueError("trajectory RTC delay must be smaller than exec_horizon")
        if not self.producer:
            raise ValueError("trajectory producer must not be empty")
        if not isinstance(self.training_eligible, bool):
            raise TypeError("trajectory training_eligible must be a bool")
        if not isinstance(self.compressed, bool):
            raise TypeError("trajectory compressed must be a bool")
        if self.storage_kind not in {"full", RENDER_ONLY_STORAGE_KIND}:
            raise ValueError("trajectory storage_kind must be 'full' or render_only_v1")
        if self.storage_kind == RENDER_ONLY_STORAGE_KIND and self.training_eligible:
            raise ValueError("render-only trajectories cannot be training eligible")

    def observation_settings(self) -> dict[str, object]:
        settings = policy_observation_settings(
            cloth_visual=self.cloth_visual, visual_profile=self.visual_profile,
            visual_profiles=self.visual_profiles,
            visual_seed=self.visual_seed,
        )
        transport = self.policy_identity.get("transport_image_size")
        if transport is not None:
            settings["transport_width_height"] = list(transport)
        return settings


def collect_trajectories(
    jobs: list[TrajectoryAssetJob],
    *,
    gpus: tuple[str, ...],
    policy_endpoints: tuple[str, ...],
    spec: TrajectoryCollectionSpec,
) -> None:
    """Collect every missing trajectory with one asset worker per simulation GPU."""

    pending = [job for job in jobs if asset_has_missing_trajectories(job, spec=spec)]
    if not pending:
        return
    worker_count = min(len(gpus), len(pending))
    runner = (
        run_policy_asset_processes
        if spec.cloth_visual == "rgb"
        else run_policy_asset_workers
    )
    runner(
        pending,
        gpus=gpus[:worker_count],
        policy_endpoints=policy_endpoints[:worker_count],
        worker_target=_trajectory_worker,
        common_args=(spec,),
        process_kind="trajectory collection",
    )


def collect_trajectory_index(
    jobs: list[TrajectoryAssetJob],
    *,
    spec: TrajectoryCollectionSpec,
    partial: bool = False,
) -> list[dict[str, Any]]:
    """Build the canonical flat trajectory index, validating provenance on the way."""

    root = Path(spec.trajectories_root)
    trajectories = []
    for job in jobs:
        for replicate in job.replicates:
            for index in range(job.state_count):
                relative = Path("trajectories") / trajectory_name(
                    job.asset_id, index, replicate
                )
                path = root / trajectory_name(job.asset_id, index, replicate)
                if not path.is_dir():
                    if partial:
                        continue
                    raise RuntimeError(f"trajectory is missing: {path}")
                try:
                    validate_existing_trajectory(
                        path,
                        job,
                        spec=spec,
                        initial_state_index=index,
                        replicate=replicate,
                        validate_arrays=False,
                    )
                    reader = TrajectoryReader(path)
                    metadata = reader.manifest.metadata
                except (OSError, RuntimeError, TypeError, ValueError, KeyError, EOFError):
                    if partial:
                        continue
                    raise
                item = {
                    "asset_id": job.asset_id,
                    "initial_state_index": index,
                    "seed": metadata["initial_state"]["seed"],
                    "policy_seed": metadata["policy_seed"],
                    "path": relative.as_posix(),
                    "action_count": reader.manifest.action_count,
                    "execution_report": metadata.get("execution_report", {}),
                }
                if replicate is not None:
                    item["replicate"] = replicate
                trajectories.append(item)
    return trajectories


def asset_has_missing_trajectories(
    job: TrajectoryAssetJob,
    *,
    spec: TrajectoryCollectionSpec,
) -> bool:
    root = Path(spec.trajectories_root)
    missing = False
    for replicate in job.replicates:
        for index in range(job.state_count):
            path = root / trajectory_name(job.asset_id, index, replicate)
            if not path.is_dir():
                missing = True
                continue
            validate_existing_trajectory(
                path,
                job,
                spec=spec,
                initial_state_index=index,
                replicate=replicate,
                # Published trajectory directories are installed atomically.
                # Resume planning only needs identity/provenance here; loading
                # every particle array makes a completed RGB evaluation scan
                # hundreds of GiB before it can proceed to scoring.
                validate_arrays=False,
            )
    return missing


def validate_existing_trajectory(
    path: Path,
    job: TrajectoryAssetJob,
    *,
    spec: TrajectoryCollectionSpec,
    initial_state_index: int,
    replicate: int | None,
    validate_arrays: bool = True,
) -> None:
    reader = TrajectoryReader(path)
    if spec.record_video and not (path / "policy_obs.mp4").is_file():
        raise ValueError(f"existing trajectory is missing its observation video: {path}")
    if validate_arrays:
        if reader.is_render_only:
            reader.validate_render_data()
        else:
            reader.validate_training_data()
    expected_render_only = spec.storage_kind == RENDER_ONLY_STORAGE_KIND
    if reader.is_render_only != expected_render_only:
        raise ValueError(f"existing trajectory has different storage kind: {path}")
    metadata = reader.manifest.metadata
    expected = {
        "producer": spec.producer,
        "training_eligible": spec.training_eligible,
        "asset_id": job.asset_id,
        "initial_state_index": initial_state_index,
        "pool_sha256": job.pool_sha256,
        "policy": spec.policy_identity,
        "replicate": replicate,
        "execution": execution_settings(
            spec.execution,
            spec.exec_horizon,
            rtc_delay_steps=spec.rtc_delay_steps,
        ),
        "cloth_visual": spec.cloth_visual,
        "observation": spec.observation_settings(),
        **job.metadata,
    }
    actual = {
        "producer": metadata.get("producer"),
        "training_eligible": metadata.get("training_eligible"),
        "asset_id": metadata.get("asset_id"),
        "initial_state_index": metadata.get("initial_state", {}).get("index"),
        "pool_sha256": metadata.get("initial_state", {}).get("pool_sha256"),
        "policy": metadata.get("policy"),
        "replicate": metadata.get("replicate"),
        "execution": metadata.get("execution"),
        "cloth_visual": metadata.get("cloth_visual"),
        "observation": metadata.get("observation"),
        **{key: metadata.get(key) for key in job.metadata},
    }
    if spec.cloth_visual == "rgb":
        from dataclasses import asdict

        from clothdojo.data.isaac_render import sample_visual_layout

        visual_profile = spec.visual_profile
        if spec.visual_profiles is not None:
            pool = InitialStatePool.load(job.pool)
            recorded = pool.metadata.get("visual_profiles")
            if not isinstance(recorded, list) or len(recorded) != pool.state_count:
                raise ValueError("mixed-profile pool requires one visual profile per state")
            visual_profile = recorded[initial_state_index]

        expected["visual_layout"] = json.loads(json.dumps(asdict(sample_visual_layout(
            spec.visual_seed, 0, visual_profile=visual_profile,
            source_episode_id=trajectory_name(job.asset_id, initial_state_index, replicate),
        ))))
        actual["visual_layout"] = metadata.get("visual_layout")
    if actual != expected:
        raise ValueError(f"existing trajectory has different provenance: {path}")
    if reader.manifest.action_count != spec.horizon_steps:
        raise ValueError(f"existing trajectory has the wrong horizon: {path}")


def trajectory_name(asset_id: str, index: int, replicate: int | None) -> str:
    suffix = "" if replicate is None else f"_r{replicate:03d}"
    return f"{asset_id}_{index:06d}{suffix}"


def policy_seed_for_batch(seeds: np.ndarray, replicate: int | None = None) -> int:
    """Derive one stable Torch seed for an ordered world batch."""

    values = np.asarray(seeds, dtype=np.int32)
    if values.ndim != 1 or not len(values) or np.any(values < 0):
        raise ValueError("batch seeds must be a non-empty non-negative int32 vector")
    if replicate is not None and replicate < 0:
        raise ValueError("replicate must be non-negative")
    if replicate is None and len(values) == 1:
        return int(values[0])
    payload = values.astype("<i4", copy=False).tobytes()
    if replicate is not None:
        payload += np.asarray(replicate, dtype="<i4").tobytes()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def execution_settings(
    mode: str,
    exec_horizon: int,
    *,
    rtc_delay_steps: int = 0,
) -> dict[str, int | str]:
    return {
        "mode": mode,
        "exec_horizon": int(exec_horizon),
        **(
            {
                "scheduler": "fixed_logical_delay",
                "inference_delay_steps": int(rtc_delay_steps),
            }
            if mode == "rtc"
            else {}
        ),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _trajectory_worker(job_queue, result_queue, gpu, endpoint, spec) -> None:
    if gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cpu" if gpu == "cpu" else "cuda:0"
    application = nullcontext(None)
    if spec.cloth_visual == "rgb":
        from clothdojo.rollout.isaac import isaac_application

        application = isaac_application(gpu)
    with application as app:
        try:
            _trajectory_jobs(job_queue, result_queue, gpu, endpoint, device, spec, app)
        finally:
            # Kit's fast shutdown exits the process immediately. Deliver every
            # completion/error before it can interrupt Queue's feeder thread.
            result_queue.close()
            result_queue.join_thread()


def _trajectory_jobs(job_queue, result_queue, gpu, endpoint, device, spec, app) -> None:
    while True:
        job = job_queue.get()
        if job is None:
            return
        try:
            _run_asset(job, endpoint=endpoint, device=device, spec=spec, isaac_app=app)
        except Exception as exc:
            traceback.print_exc()
            result_queue.put(
                {
                    "status": "error",
                    "asset_id": job.asset_id,
                    "gpu": gpu,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return
        result_queue.put({"status": "complete", "asset_id": job.asset_id, "gpu": gpu})


def _run_asset(
    job: TrajectoryAssetJob,
    *,
    endpoint: str,
    device: str,
    spec: TrajectoryCollectionSpec,
    isaac_app=None,
) -> None:
    from clothdojo.assets import GarmentLibrary
    from clothdojo.rollout.render import ShirtFlattenCameraObservation
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

    pool = InitialStatePool.load(job.pool)
    scenario = replace(load_rollout_scenario(spec.scenario_path), garment_id=job.asset_id)
    assets = Path(spec.assets_root)
    garment = GarmentLibrary(assets).converted(job.asset_id)
    validate_pool_compatibility(pool, garment_path=garment, particle_count=pool.particle_count)
    from clothdojo.rollout.starvla import StarVLAPolicyAdapter, is_starvla_endpoint

    if is_starvla_endpoint(endpoint):
        policy = StarVLAPolicyAdapter(
            endpoint,
            expected_checkpoint_sha256=spec.expected_checkpoint_sha256,
            action_horizon=scenario.action_horizon,
        )
    else:
        policy = RemotePolicy(
            endpoint,
            expected_checkpoint_sha256=spec.expected_checkpoint_sha256,
            expected_image_preprocessing=spec.policy_identity.get(
                "image_preprocessing", "none"
            ),
            action_horizon=scenario.action_horizon,
        )
    if policy.identity != spec.policy_identity:
        raise ValueError("trajectory worker policy identity differs from the collection plan")
    world_count = min(spec.batch_size, job.state_count)
    scene = build_scene(
        assets,
        robot_id=spec.robot_id,
        task_name=scenario.task_name,
        spec=ShirtFlattenSpec(garment_id=job.asset_id, world_count=world_count),
        device=device,
        visuals=spec.cloth_visual != "rgb",
        cloth_visual="white" if spec.cloth_visual == "rgb" else spec.cloth_visual,
    )
    runtime = ShirtFlattenRuntime(scene)
    validate_pool_compatibility(
        pool,
        garment_path=garment,
        particle_count=runtime.grasp.particle_ranges()[0][1]
        - runtime.grasp.particle_ranges()[0][0],
    )
    if spec.cloth_visual == "rgb":
        from clothdojo.rollout.isaac import IsaacCameraObservation

        camera = IsaacCameraObservation(
            isaac_app, runtime, asset_path=garment, assets_root=assets,
            visual_profile=spec.visual_profile, visual_profiles=spec.visual_profiles,
            visual_seed=spec.visual_seed,
        )
    else:
        camera = ShirtFlattenCameraObservation(
            scene, width=POLICY_RENDER_SIZE[0], height=POLICY_RENDER_SIZE[1],
        )
    try:
        _run_asset_cases(job, spec, runtime, pool, scenario, policy, camera, world_count)
    finally:
        if spec.cloth_visual == "rgb":
            camera.close()


def _record_semantic_chunks(writers, report, identity) -> None:
    """Persist per-chunk semantic subtask selections into each world manifest."""

    records = report.semantic_records
    if not records:
        return
    contract = identity.get("semantic_subtask") or {}
    subtasks = [str(item) for item in contract.get("subtasks", [])]
    for world, writer in enumerate(writers):
        if writer is None:
            continue
        writer.update_metadata(
            "semantic_subtask",
            {
                "schema": "clothdojo_semantic_chunk_log_v1",
                "overall_prompt": str(contract.get("overall_prompt", "")),
                "subtasks": subtasks,
                "chunks": [
                    {
                        "request_index": int(record["request_index"]),
                        "requested_at_step": int(record["requested_at_step"]),
                        "subtask": str(record["subtask"][world]),
                        "candidate_nll": [
                            float(value) for value in record["candidate_nll"][world]
                        ],
                    }
                    for record in records
                ],
            },
        )


def _run_asset_cases(job, spec, runtime, pool, scenario, policy, camera, world_count) -> None:
    from clothdojo.rollout.engine import run_episode
    from clothdojo.rollout.newton import NewtonTrajectoryEpisode

    root = Path(spec.trajectories_root)
    for replicate in job.replicates:
        for batch_start in range(0, job.state_count, world_count):
            active = list(range(batch_start, min(batch_start + world_count, job.state_count)))
            world_indices = active + [active[-1]] * (world_count - len(active))
            targets = [
                root / trajectory_name(job.asset_id, index, replicate) for index in active
            ]
            missing = []
            for index, target in zip(active, targets, strict=True):
                if target.is_dir():
                    validate_existing_trajectory(
                        target,
                        job,
                        spec=spec,
                        initial_state_index=index,
                        replicate=replicate,
                    )
                else:
                    missing.append(index)
            if not missing:
                continue
            staging_by_index = {
                index: root
                / f".{trajectory_name(job.asset_id, index, replicate)}.{uuid.uuid4().hex}.tmp"
                for index in missing
            }
            policy_seed = policy_seed_for_batch(
                np.asarray(pool.seeds[world_indices], dtype=np.int32), replicate
            )
            runtime.reset()
            runtime.restore_particles(
                np.ascontiguousarray(pool.particle_q[world_indices]),
                np.ascontiguousarray(pool.particle_qd[world_indices]),
            )
            cold_robot = pool.metadata.get("cold_robot_state")
            if cold_robot is not None:
                if cold_robot.get("velocity") != "zero":
                    raise ValueError("cold robot initial states require zero velocities")
                joints = np.asarray(cold_robot["joint_q"], dtype=np.float32)
                opened = np.asarray(cold_robot["gripper_open_state"], dtype=np.float32)
                if joints.ndim != 2 or len(joints) != pool.state_count:
                    raise ValueError("cold robot joint states must match pool states")
                if (opened.shape != (pool.state_count, 2) or not np.isfinite(opened).all()
                        or np.any(opened < .95) or np.any(opened > 1)):
                    raise ValueError("cold robot initial states must have both grippers open")
                runtime.restore_rigid_state_batch(
                    np.ascontiguousarray(joints[world_indices]),
                    np.zeros((world_count, runtime.model.joint_dof_count // world_count), dtype=np.float32),
                )
                runtime.grasp.previous_open.assign(np.ascontiguousarray(opened[world_indices]))
                for view in (runtime.scene.arms, runtime.scene.grippers):
                    view.set_attribute("joint_target_q", runtime.control,
                                       view.get_attribute("joint_q", runtime.state_0))
            appearances = [{} for _ in world_indices]
            if spec.cloth_visual == "rgb":
                case_ids = tuple(
                    trajectory_name(job.asset_id, index, replicate) for index in world_indices
                )
                profiles = None
                if spec.visual_profiles is not None:
                    recorded = pool.metadata.get("visual_profiles")
                    if not isinstance(recorded, list) or len(recorded) != pool.state_count:
                        raise ValueError("mixed-profile pool requires one visual profile per state")
                    profiles = tuple(recorded[index] for index in world_indices)
                    if any(profile not in spec.visual_profiles for profile in profiles):
                        raise ValueError("initial-state visual profile is outside the run contract")
                layouts = camera.begin_episodes(case_ids, visual_profiles=profiles)
                appearances = [{"visual_layout": layout} for layout in layouts]
            scenarios = tuple(
                replace(scenario, seed=int(pool.seeds[index])) for index in world_indices
            )
            outputs = tuple(
                staging_by_index.get(index) if world < len(active) else None
                for world, index in enumerate(world_indices)
            )
            metadata = tuple(
                {
                    "asset_id": job.asset_id,
                    "policy_kind": policy.identity["backend"],
                    "policy": policy.identity,
                    "policy_seed": policy_seed,
                    "cloth_visual": spec.cloth_visual,
                    "observation": spec.observation_settings(),
                    **appearances[world],
                    "execution": execution_settings(
                        spec.execution,
                        spec.exec_horizon,
                        rtc_delay_steps=spec.rtc_delay_steps,
                    ),
                    "parallelism": {
                        "worlds": world_count,
                        "active_worlds": len(active),
                        "batch_start": batch_start,
                    },
                    "initial_state": {
                        "pool_sha256": job.pool_sha256,
                        "index": index,
                        "seed": int(pool.seeds[index]),
                        **({"source_frame": pool.metadata["source_frames"][index],
                            "restoration": "cloth_and_robot_positions_zero_velocities_released"}
                           if "cold_robot_state" in pool.metadata else {}),
                    },
                    **({"replicate": replicate} if replicate is not None else {}),
                    **job.metadata,
                }
                for world, index in enumerate(world_indices)
            )
            try:
                batch_started = time.perf_counter()
                with NewtonTrajectoryEpisode(
                    runtime,
                    outputs,
                    scenarios,
                    assets_root=spec.assets_root,
                    observation_provider=camera,
                    trajectory_metadata=metadata,
                    producer=spec.producer,
                    training_eligible=spec.training_eligible,
                    compressed=spec.compressed,
                    storage_kind=spec.storage_kind,
                ) as episode:
                    from clothdojo.rollout.observation_video import record_policy_observations

                    active_episode = episode
                    if policy.identity.get("action_space") == "eef_abs_20d_action_frame":
                        from clothdojo.rollout.starvla import StarVLAEFCommandEpisode

                        active_episode = StarVLAEFCommandEpisode(episode, runtime)
                        active_episode.begin()
                    elif policy.identity.get("observation_horizon") == 2:
                        from clothdojo.rollout.diffusion import AdjacentObservationEpisode

                        active_episode = AdjacentObservationEpisode(
                            episode, spec.exec_horizon
                        )

                    recording = (
                        record_policy_observations(policy, outputs, fps=50 / spec.exec_horizon)
                        if spec.record_video else nullcontext(policy)
                    )
                    with recording as active_policy:
                        report = run_episode(
                            active_episode, active_policy, scenario,
                            mode=spec.execution,
                            exec_horizon=spec.exec_horizon,
                            policy_seed=policy_seed,
                            rtc_delay_steps=spec.rtc_delay_steps,
                        )
                        _record_semantic_chunks(
                            episode.writers, report, policy.identity
                        )
                batch_wall_seconds = time.perf_counter() - batch_started
                for index, target in zip(active, targets, strict=True):
                    staging = staging_by_index.get(index)
                    if staging is None:
                        continue
                    manifest_path = staging / "manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["metadata"]["execution_report"] = report.to_json_dict()
                    if spec.record_video:
                        manifest["metadata"]["observation_video"] = {
                            "path": "policy_obs.mp4",
                            "source": "policy_input",
                            "fps": 50 / spec.exec_horizon,
                            "views": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                        }
                    manifest["metadata"]["trajectory_publication"] = {
                        "batch_wall_seconds": batch_wall_seconds,
                        "finalize_seconds": max(
                            0.0, batch_wall_seconds - report.timings.wall_seconds
                        ),
                    }
                    atomic_write_json(manifest_path, manifest)
                    os.replace(staging, target)
            finally:
                for staging in staging_by_index.values():
                    shutil.rmtree(staging, ignore_errors=True)
