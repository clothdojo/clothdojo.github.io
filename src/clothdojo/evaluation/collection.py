"""Trajectory-first multi-GPU evaluation over a fixed initial-state pool."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from clothdojo.assets import GarmentLibrary
from clothdojo.evaluation.metric import AIRBORNE_METRIC, file_sha256, load_canonical_template
from clothdojo.evaluation.protocol import EvaluationProtocol, load_evaluation_protocol
from clothdojo.evaluation.results import (
    FORMAL_BENCHMARK_ASSET_SPLITS_SHA256,
    FORMAL_BENCHMARK_INITIAL_STATES_SHA256,
    FORMAL_BENCHMARK_PANEL_ID,
    FORMAL_BENCHMARK_SEEN_ASSETS,
    FORMAL_BENCHMARK_SELECTION_SHA256,
    FORMAL_BENCHMARK_STATES_PER_ASSET,
    FORMAL_BENCHMARK_UNSEEN_ASSETS,
    benchmark_asset_splits_sha256,
    load_evaluation_condition,
    summarize_paired_conditions,
    validate_formal_evaluation_metadata,
    write_evaluation_cases,
)
from clothdojo.evaluation.scoring import score_evaluation_trajectory
from clothdojo.evaluation.templates import (
    garment_topology_sha256,
    resolve_canonical_templates,
    validate_template_physics,
)
from clothdojo.rendering import policy_observation_settings
from clothdojo.rollout.initial_states import (
    InitialStatePool,
    load_initial_state_collection,
    validate_pool_compatibility,
)
from clothdojo.rollout.policy import RemotePolicy
from clothdojo.rollout.scenario import load_rollout_scenario
from clothdojo.rollout.scheduling import (
    policy_endpoints_for_workers,
    validate_policy_identities,
)
from clothdojo.rollout.trajectory_collection import (
    TrajectoryAssetJob,
    TrajectoryCollectionSpec,
    atomic_write_json,
    collect_trajectories,
    collect_trajectory_index,
    execution_settings,
)
from clothdojo.trajectory import TrajectoryReader
from clothdojo.trajectory.schema import RENDER_ONLY_STORAGE_KIND

EVALUATION_CONDITION_SCHEMA = "clothdojo_evaluation_condition"
_SCORING_TEMPLATE_CACHE: dict[tuple[str, str], Any] = {}
_FORMAL_TEMPLATES_META_SHA256 = "4d82be0c5accf48d8dcbae96f3505a3b85c805447fda22639f80a884753bdb9d"


def _policy_service_identity(
    endpoint: str,
    *,
    expected_checkpoint_sha256: str | None,
    expected_image_preprocessing: str,
    action_horizon: int,
) -> dict[str, Any]:
    """Fetch one policy identity, dispatching on the endpoint scheme."""

    from clothdojo.rollout.starvla import StarVLAPolicyAdapter, is_starvla_endpoint

    if is_starvla_endpoint(endpoint):
        return StarVLAPolicyAdapter(
            endpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            action_horizon=action_horizon,
        ).identity
    return RemotePolicy(
        endpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_image_preprocessing=expected_image_preprocessing,
        action_horizon=action_horizon,
    ).identity


def _validate_formal_inputs(
    states_root: Path,
    states_meta: dict[str, Any],
    templates_root: Path,
) -> dict[str, str]:
    if file_sha256(states_root / "meta.json") != FORMAL_BENCHMARK_INITIAL_STATES_SHA256:
        raise ValueError("benchmark run requires the frozen balanced-200 state panel")
    if file_sha256(templates_root / "meta.json") != _FORMAL_TEMPLATES_META_SHA256:
        raise ValueError("benchmark run requires the frozen balanced-200 templates")
    provenance = states_meta.get("selection_provenance")
    if not isinstance(provenance, dict) or provenance != {
        "panel_id": FORMAL_BENCHMARK_PANEL_ID,
        "selection_sha256": FORMAL_BENCHMARK_SELECTION_SHA256,
    }:
        raise ValueError("benchmark state collection is not the frozen balanced-200 panel")
    if (
        states_meta.get("seen_assets") != FORMAL_BENCHMARK_SEEN_ASSETS
        or states_meta.get("unseen_assets") != FORMAL_BENCHMARK_UNSEEN_ASSETS
    ):
        raise ValueError("benchmark panel must contain 100 seen and 100 unseen assets")
    assets = states_meta.get("assets")
    if not isinstance(assets, list) or len(assets) != (
        FORMAL_BENCHMARK_SEEN_ASSETS + FORMAL_BENCHMARK_UNSEEN_ASSETS
    ):
        raise ValueError("benchmark panel must contain exactly 200 assets")
    split_counts = {
        split: sum(item.get("split") == split for item in assets if isinstance(item, dict))
        for split in ("seen", "unseen")
    }
    if split_counts != {
        "seen": FORMAL_BENCHMARK_SEEN_ASSETS,
        "unseen": FORMAL_BENCHMARK_UNSEEN_ASSETS,
    }:
        raise ValueError("benchmark panel split metadata is invalid")
    if any(
        not isinstance(item, dict) or item.get("state_count") != FORMAL_BENCHMARK_STATES_PER_ASSET
        for item in assets
    ):
        raise ValueError("every benchmark asset requires exactly four initial states")
    asset_splits = {item["asset_id"]: item["split"] for item in assets}
    if benchmark_asset_splits_sha256(asset_splits) != FORMAL_BENCHMARK_ASSET_SPLITS_SHA256:
        raise ValueError("benchmark asset split mapping changed")
    pool_hashes = {}
    for item in assets:
        asset_id = item["asset_id"]
        digest = file_sha256(states_root / asset_id / "states.npz")
        if digest != item.get("sha256"):
            raise ValueError(f"benchmark initial-state pool changed for asset {asset_id}")
        pool_hashes[asset_id] = digest
    return pool_hashes


def score_evaluation_condition(
    condition: str | Path,
    protocol_path: str | Path,
    templates_path: str | Path,
    *,
    workers: int = 1,
    require_formal: bool = False,
    rescore_from: str | Path | None = None,
    rescore_templates_from: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Score saved trajectories; changing scoring requires the original protocol."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("evaluation scoring workers must be a positive integer")

    source = Path(condition).expanduser().resolve()
    root = source if output is None else Path(output).expanduser().resolve()
    if output is not None:
        if root == source or source in root.parents:
            raise ValueError("rescore output must be separate from the source condition")
        root.mkdir(parents=True, exist_ok=True)
    # Keep this inode after closing: unlinking allows concurrent callers to
    # hold locks on different files. Readers still use the publication marker.
    with (root / ".scoring.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"evaluation scoring is already running: {root}") from exc
        if output is not None:
            original_meta = load_evaluation_condition(source).meta
            provenance = {
                "path": str(source), "meta_sha256": file_sha256(source / "meta.json"),
                "results_sha256": file_sha256(source / "results.jsonl"),
            }
            if (root / "meta.json").exists():
                if _load_running_meta(root / "meta.json").get("rescore_source") != provenance:
                    raise ValueError("rescore output belongs to a different source")
            else:
                if any(p.name != ".scoring.lock" for p in root.iterdir()):
                    raise FileExistsError("rescore output is not empty")
                (root / "trajectories").symlink_to(
                    source / "trajectories", target_is_directory=True,
                )
                atomic_write_json(root / "meta.json", {
                    **original_meta, "status": "running", "phase": "scoring",
                    "rescore_source": provenance,
                })
        return _score_evaluation_condition_locked(
            root,
            protocol_path,
            templates_path,
            workers=workers,
            require_formal=require_formal,
            rescore_from=rescore_from,
            rescore_templates_from=rescore_templates_from,
        )


def _score_evaluation_condition_locked(
    root: Path,
    protocol_path: str | Path,
    templates_path: str | Path,
    *,
    workers: int,
    require_formal: bool,
    rescore_from: str | Path | None,
    rescore_templates_from: str | Path | None = None,
) -> dict[str, Any]:
    meta_path = root / "meta.json"
    meta = _load_running_meta(meta_path)
    require_formal = require_formal or (
        meta.get("benchmark", {}).get("panel_id") == FORMAL_BENCHMARK_PANEL_ID
    )
    protocol = load_evaluation_protocol(protocol_path)
    recorded_protocol = meta.get("protocol", {})
    collection_protocol = dict(meta.get("collection_protocol", recorded_protocol))
    collection_templates = dict(meta.get("collection_templates", meta.get("templates", {})))
    if rescore_templates_from is not None and rescore_from is None:
        raise ValueError("template replacement requires the original rescore source protocol")
    if rescore_from is None and protocol.fingerprint != recorded_protocol.get("fingerprint"):
        raise ValueError("evaluation protocol differs from the recorded condition")
    if rescore_from is not None:
        original = load_evaluation_protocol(rescore_from)
        if original.fingerprint != collection_protocol.get("fingerprint"):
            raise ValueError("rescore source protocol differs from the original collection")
        original_definition = original.to_json_dict()
        allowed_fields = {"metric", "success"}
        if (
            protocol.control_fps == original.control_fps
            and protocol.metric_hz == original.metric_hz
            and protocol.horizon_seconds <= original.horizon_seconds
        ):
            # A complete longer rollout is immutable evidence for any prefix.
            # Only shortening the hard horizon is allowed; sampling rates stay fixed.
            allowed_fields.add("timing")
        if rescore_templates_from is not None:
            old_target = original.target.to_json_dict()
            new_target = protocol.target.to_json_dict()
            if (
                old_target["method"] != "default_free_settle"
                or new_target["method"] != "large_support_free_settle_v1"
                or {k: v for k, v in old_target.items() if k != "method"}
                != {k: v for k, v in new_target.items() if k != "method"}
            ):
                raise ValueError(
                    "template rescoring only permits the versioned large-support correction"
                )
            allowed_fields.add("target")
        changed = [
            field for field, value in protocol.to_json_dict().items()
            if field not in allowed_fields and original_definition[field] != value
        ]
        if changed:
            raise ValueError(
                "rescoring may change only metric and success, plus a shorter horizon or target; "
                "changed: " + ", ".join(changed)
            )
        if (
            "definition" in collection_protocol
            and collection_protocol["definition"] != original_definition
        ):
            raise ValueError("recorded collection protocol definition differs")
        collection_protocol["definition"] = original_definition
    elif collection_protocol.get("fingerprint") == protocol.fingerprint:
        collection_protocol["definition"] = protocol.to_json_dict()
    scored_meta = {
        **meta,
        "protocol": _protocol_summary(protocol),
        "collection_protocol": collection_protocol,
        "scoring_protocol": protocol.to_json_dict(),
        "airborne_metric": {**AIRBORNE_METRIC, "sample_hz": protocol.metric_hz},
    }
    trajectories = meta.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != meta.get("case_count"):
        raise RuntimeError("evaluation condition does not contain every trajectory")
    templates_root = Path(templates_path).expanduser().resolve()
    if (
        require_formal
        and protocol.target.method == "large_support_free_settle_v1"
        and file_sha256(templates_root / "meta.json") != _FORMAL_TEMPLATES_META_SHA256
    ):
        raise ValueError("formal scoring requires the frozen large-support templates")
    resolved_templates = resolve_canonical_templates(
        templates_root,
        meta.get("templates", {}),
        target_settings=protocol.target.to_json_dict(),
    )
    original_templates = None
    if rescore_templates_from is not None:
        original_templates = resolve_canonical_templates(
            rescore_templates_from, collection_templates,
            target_settings=collection_protocol["definition"]["target"],
        )
    templates = {}
    for asset_id, expected_sha256 in meta.get("templates", {}).items():
        template_path, collection_sha256 = resolved_templates[asset_id]
        if original_templates is None and collection_sha256 != expected_sha256:
            raise ValueError(f"canonical template changed for asset {asset_id}")
        template = load_canonical_template(template_path)
        if template.asset_id != asset_id:
            raise ValueError("canonical template asset identity differs")
        if original_templates is not None:
            from clothdojo.sim.scene import ShirtFlattenSpec

            old_path, old_sha = original_templates[asset_id]
            if old_sha != collection_templates[asset_id]:
                raise ValueError("original canonical template differs from collection provenance")
            old_template = load_canonical_template(old_path)
            if (template.asset_sha256, template.topology_sha256) != (
                old_template.asset_sha256, old_template.topology_sha256
            ):
                raise ValueError("replacement template garment or topology differs")
            if template.source_kind != protocol.target.method:
                raise ValueError("replacement template method differs from scoring protocol")
            validate_template_physics(old_template, ShirtFlattenSpec(garment_id=asset_id))
            validate_template_physics(template, ShirtFlattenSpec(garment_id=asset_id))
        templates[asset_id] = (template_path, collection_sha256)
        _SCORING_TEMPLATE_CACHE[(str(template_path), collection_sha256)] = template
    scored_meta["templates"] = {a: value[1] for a, value in templates.items()}
    if original_templates is not None or "collection_templates" in meta:
        scored_meta["collection_templates"] = collection_templates
    if original_templates is not None:
        scored_meta["template_rescore"] = {
            "original_path": str(Path(rescore_templates_from).resolve()),
            "original_meta_sha256": file_sha256(Path(rescore_templates_from) / "meta.json"),
            "scoring_path": str(templates_root),
            "scoring_meta_sha256": file_sha256(templates_root / "meta.json"),
        }
    if require_formal:
        validate_formal_evaluation_metadata(scored_meta)

    score_jobs = []
    expected_keys = []
    seen = set()
    for item in trajectories:
        key = (item["asset_id"], item["initial_state_index"], item["replicate"])
        if key in seen:
            raise ValueError("evaluation trajectory index contains duplicate cases")
        seen.add(key)
        relative = Path(item["path"])
        if relative.parent != Path("trajectories"):
            raise ValueError("evaluation trajectory path is not flat")
        reader = TrajectoryReader(root / relative)
        metadata = reader.manifest.metadata
        identity_fields = [
            "condition_id",
            "policy",
            "execution",
            "cloth_visual",
            "observation",
        ]
        if "runtime" in meta:
            identity_fields.append("runtime")
        for field in identity_fields:
            if field not in meta or metadata.get(field) != meta[field]:
                raise ValueError(f"evaluation trajectory {field} differs from condition")
        initial_state = metadata.get("initial_state", {})
        if (
            initial_state.get("pool_sha256") != meta["initial_state_pools"][item["asset_id"]]
            or initial_state.get("seed") != item["seed"]
            or metadata.get("policy_seed") != item["policy_seed"]
        ):
            raise ValueError(
                "evaluation trajectory initial state or policy seed differs from index"
            )
        if (
            metadata.get("protocol_fingerprint") != collection_protocol.get("fingerprint")
            or metadata.get("template_sha256") != collection_templates[item["asset_id"]]
        ):
            raise ValueError("evaluation trajectory scoring provenance differs")
        expected_keys.append(key)
        template_path, template_sha256 = templates[item["asset_id"]]
        score_jobs.append(
            (root / relative, template_path, template_sha256, protocol,
             meta.get("assets_root", "assets"))
        )

    if workers == 1:
        cases = [_score_evaluation_job(job) for job in score_jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(score_jobs))) as executor:
            cases = list(executor.map(_score_evaluation_job, score_jobs, chunksize=1))
    for expected_key, result in zip(expected_keys, cases, strict=True):
        actual_key = (
            result.key.asset_id,
            result.key.initial_state_index,
            result.key.replicate,
        )
        if actual_key != expected_key:
            raise ValueError("evaluation trajectory identity differs from its index")

    # Two files cannot be replaced together. Mark the condition unreadable before
    # replacing results, then publish their matching scoring protocol atomically.
    # A failed publication never exposes new scores as belonging to the old one.
    atomic_write_json(meta_path, {**meta, "status": "running", "phase": "publishing_scores"})
    write_evaluation_cases(root / "results.jsonl", cases)
    scored_meta["status"] = "complete"
    scored_meta["phase"] = "complete"
    scored_meta.pop("error", None)
    atomic_write_json(meta_path, scored_meta)
    atomic_write_json(root / "summary.json", summarize_paired_conditions([root]))
    return {
        "compatible": True,
        "output": str(root),
        "condition_id": meta["condition_id"],
        "case_count": len(cases),
        "summary": str(root / "summary.json"),
        "training_eligible": False,
    }


def _protocol_summary(protocol: EvaluationProtocol) -> dict[str, Any]:
    return {
        "id": protocol.protocol_id,
        "stage": protocol.stage,
        "fingerprint": protocol.fingerprint,
        "horizon_steps": protocol.horizon_steps,
        "control_fps": protocol.control_fps,
        "metric_backend": "numpy_float64_reference",
    }


def run_evaluation_condition(
    scenario_path: str | Path,
    protocol_path: str | Path,
    initial_states_path: str | Path,
    templates_path: str | Path,
    policy_endpoints: tuple[str, ...],
    output: str | Path,
    *,
    condition_id: str,
    cloth_visual: str = "nocs",
    visual_profile: str | None = None,
    visual_profiles: tuple[str, ...] | None = None,
    visual_seed: int = 0,
    expected_checkpoint_sha256: str | None = None,
    image_preprocessing: str = "none",
    assets_root: str | Path = "assets",
    gpus: tuple[str, ...] = ("0",),
    execution: str = "chunk",
    exec_horizon: int = 10,
    rtc_delay_steps: int = 0,
    batch_size: int = 4,
    record_video: bool = False,
    defer_scoring: bool = False,
    score_workers: int = 1,
    trajectory_collector: Callable[
        [list[TrajectoryAssetJob], TrajectoryCollectionSpec, tuple[str, ...]], None
    ]
    | None = None,
    expected_policy_runtime_sha256: str | None = None,
    runtime_sha256: str | None = None,
    panel_id: str | None = None,
    shared_policy_endpoints: bool = False,
) -> dict[str, Any]:
    """Collect canonical trajectories, then score one immutable condition offline."""

    from clothdojo.sim.scene import WORLD_LAYOUT, WORLD_SPACING_M, ShirtFlattenSpec

    if not condition_id.strip():
        raise ValueError("condition_id must not be empty")
    observation = policy_observation_settings(
        cloth_visual=cloth_visual, visual_profile=visual_profile,
        visual_profiles=visual_profiles, visual_seed=visual_seed,
    )
    if not policy_endpoints or any(not endpoint.strip() for endpoint in policy_endpoints):
        raise ValueError("at least one policy endpoint is required")
    if not gpus or any(not gpu.strip() for gpu in gpus):
        raise ValueError("at least one simulation GPU is required")
    if execution not in {"chunk", "rtc"}:
        raise ValueError("execution must be 'chunk' or 'rtc'")
    for name, digest in (
        ("expected_policy_runtime_sha256", expected_policy_runtime_sha256),
        ("runtime_sha256", runtime_sha256),
    ):
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    for name, value in (
        ("batch_size", batch_size),
        ("score_workers", score_workers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(rtc_delay_steps, bool)
        or not isinstance(rtc_delay_steps, int)
        or rtc_delay_steps < 0
    ):
        raise ValueError("rtc_delay_steps must be a non-negative integer")

    scenario = load_rollout_scenario(scenario_path)
    protocol = load_evaluation_protocol(protocol_path)
    if scenario.task_name != protocol.task_name:
        raise ValueError("scenario task differs from evaluation protocol")
    if scenario.steps != protocol.horizon_steps:
        raise ValueError("scenario steps differ from the evaluation horizon")
    if not 1 <= exec_horizon <= scenario.action_horizon:
        raise ValueError("exec_horizon must be in [1, action_horizon]")
    logical_delay = rtc_delay_steps if execution == "rtc" else 0
    if execution == "rtc" and logical_delay >= exec_horizon:
        raise ValueError("RTC delay must be smaller than exec_horizon")
    if execution == "rtc" and exec_horizon + logical_delay > scenario.action_horizon:
        raise ValueError("RTC horizon plus delay exceeds the model action horizon")

    states_root = Path(initial_states_path).expanduser().resolve()
    states_meta = load_initial_state_collection(states_root)
    templates_root = Path(templates_path).expanduser().resolve()
    provenance = states_meta.get("selection_provenance", {})
    panel_id = panel_id or provenance.get("panel_id") or states_root.name
    if provenance.get("panel_id", panel_id) != panel_id:
        raise ValueError("benchmark panel differs from the initial-state collection")
    if panel_id == FORMAL_BENCHMARK_PANEL_ID:
        pool_hashes = _validate_formal_inputs(states_root, states_meta, templates_root)
    else:
        pool_hashes = {}
        for item in states_meta["assets"]:
            digest = file_sha256(states_root / item["asset_id"] / "states.npz")
            if digest != item["sha256"]:
                raise ValueError(f"benchmark initial-state pool changed for {item['asset_id']}")
            pool_hashes[item["asset_id"]] = digest
    assets = Path(assets_root).expanduser().resolve()
    garments = GarmentLibrary(assets)
    destination = Path(output).expanduser().resolve()
    selected_assets = list(states_meta["assets"])
    selected_assets.sort(key=lambda item: int(item["particle_count"]), reverse=True)
    resolved_templates = resolve_canonical_templates(
        templates_root,
        (item["asset_id"] for item in selected_assets),
        target_settings=protocol.target.to_json_dict(),
    )
    jobs: list[TrajectoryAssetJob] = []
    for item in selected_assets:
        pool_path = states_root / item["asset_id"] / "states.npz"
        pool = InitialStatePool.load(pool_path)
        count = item["state_count"]
        if count != pool.state_count:
            raise ValueError("initial-state count differs from the panel metadata")
        template_path, template_sha256 = resolved_templates[item["asset_id"]]
        template = load_canonical_template(template_path)
        if template.asset_id != item["asset_id"]:
            raise ValueError(f"canonical template asset mismatch: {template_path}")
        garment = garments.converted(item["asset_id"])
        validate_pool_compatibility(
            pool,
            garment_path=garment,
            particle_count=pool.particle_count,
        )
        if template.asset_sha256 != file_sha256(garment):
            raise ValueError("canonical template garment hash differs from the runtime asset")
        if template.topology_sha256 != garment_topology_sha256(garment):
            raise ValueError("canonical template topology differs from the runtime asset")
        physics_spec = ShirtFlattenSpec(garment_id=item["asset_id"])
        validate_template_physics(template, physics_spec)
        jobs.append(
            TrajectoryAssetJob(
                asset_id=item["asset_id"],
                pool=str(pool_path),
                pool_sha256=pool_hashes[item["asset_id"]],
                state_count=count,
                replicates=(0,),
                metadata={
                    "condition_id": condition_id,
                    "protocol_fingerprint": protocol.fingerprint,
                    "template_sha256": template_sha256,
                    **(
                        {
                            "runtime": {
                                "clothdojo_source_sha256": runtime_sha256,
                            }
                        }
                        if runtime_sha256 is not None
                        else {}
                    ),
                },
            )
        )

    worker_count = min(len(gpus), len(jobs))
    active_endpoints = policy_endpoints_for_workers(
        policy_endpoints, worker_count, allow_shared=shared_policy_endpoints,
    )
    identities = tuple(
        _policy_service_identity(
            endpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_image_preprocessing=image_preprocessing,
            action_horizon=scenario.action_horizon,
        )
        for endpoint in active_endpoints
    )
    policy_identity = validate_policy_identities(
        identities,
        execution=execution,
        worlds_per_batch=batch_size,
    )
    observation["transport_width_height"] = list(
        policy_identity["transport_image_size"]
    )
    if (
        expected_policy_runtime_sha256 is not None
        and policy_identity.get("clothdojo_runtime_sha256")
        != expected_policy_runtime_sha256
    ):
        raise ValueError("policy service ClothDojo runtime differs from the coordinator")
    asset_splits = {item["asset_id"]: item["split"] for item in selected_assets}
    state_counts = {job.asset_id: job.state_count for job in jobs}
    counts = set(state_counts.values())
    requested = {
        "condition_id": condition_id,
        "protocol": _protocol_summary(protocol),
        "scenario": scenario.to_json_dict(),
        "cloth_visual": cloth_visual,
        "observation": observation,
        "initial_states_sha256": file_sha256(states_root / "meta.json"),
        "initial_state_pools": {job.asset_id: job.pool_sha256 for job in jobs},
        "templates": {
            job.asset_id: job.metadata["template_sha256"] for job in jobs
        },
        "policy": policy_identity,
        **(
            {"runtime": {"clothdojo_source_sha256": runtime_sha256}}
            if runtime_sha256 is not None
            else {}
        ),
        "execution": execution_settings(
            execution,
            exec_horizon,
            rtc_delay_steps=logical_delay,
        ),
        "benchmark": {
            "panel_id": panel_id,
            "selection_sha256": provenance.get("selection_sha256"),
            "asset_splits_sha256": benchmark_asset_splits_sha256(asset_splits),
            "seen_assets": sum(split == "seen" for split in asset_splits.values()),
            "unseen_assets": sum(split == "unseen" for split in asset_splits.values()),
            "states_per_asset": next(iter(counts)) if len(counts) == 1 else None,
            "asset_splits": asset_splits,
        },
        "replicates": 1,
        "parallelism": {
            "worlds_per_batch": batch_size,
            "simulation": {
                "engine": "newton-worlds",
                "world_layout": WORLD_LAYOUT,
                "spacing_m": WORLD_SPACING_M,
            },
            "rendering": "isaac_rtx" if cloth_visual == "rgb" else "sensor-tiled-camera",
            "policy": "single-batch-request",
        },
        "trajectory_storage": {"kind": RENDER_ONLY_STORAGE_KIND, "npz_compressed": False},
        "case_count": sum(state_counts.values()),
        **({"record_video": True} if record_video else {}),
    }
    collection_spec = TrajectoryCollectionSpec(
        scenario_path=str(Path(scenario_path).expanduser().resolve()),
        assets_root=str(assets),
        trajectories_root=str(destination / "trajectories"),
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        policy_identity=policy_identity,
        execution=execution,
        exec_horizon=exec_horizon,
        rtc_delay_steps=logical_delay,
        horizon_steps=scenario.steps,
        batch_size=batch_size,
        producer="clothdojo_evaluation",
        record_video=record_video,
        training_eligible=False,
        storage_kind=RENDER_ONLY_STORAGE_KIND,
        compressed=False,
        cloth_visual=cloth_visual,
        visual_profile=visual_profile,
        visual_profiles=visual_profiles,
        visual_seed=visual_seed,
    )
    meta_path = destination / "meta.json"
    if meta_path.is_file():
        meta = _load_running_meta(meta_path)
        if {key: meta.get(key) for key in requested} != requested:
            raise ValueError("existing evaluation condition settings differ")
        if meta.get("status") == "complete":
            try:
                _validate_published_condition(destination, jobs, collection_spec)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError, EOFError):
                meta["status"] = "running"
                meta["phase"] = "scoring"
                atomic_write_json(meta_path, meta)
            else:
                return _summary(destination, meta, gpus)
    else:
        if destination.exists():
            if not destination.is_dir():
                raise FileExistsError(f"evaluation output is not a directory: {destination}")
            unexpected = [path for path in destination.iterdir() if path.name != "runtime"]
            if unexpected:
                raise FileExistsError(f"evaluation output is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "trajectories").mkdir()
        meta = {
            "schema": EVALUATION_CONDITION_SCHEMA,
            "status": "running",
            "phase": "trajectories",
            "training_eligible": False,
            "results": "results.jsonl",
            "trajectories": [],
            "collection_protocol": {
                **_protocol_summary(protocol),
                "definition": protocol.to_json_dict(),
            },
            "assets_root": str(assets),
            **requested,
        }
        atomic_write_json(meta_path, meta)

    try:
        meta["status"] = "running"
        meta["phase"] = "trajectories"
        meta.pop("error", None)
        atomic_write_json(meta_path, meta)
        if trajectory_collector is None:
            collect_trajectories(
                jobs,
                gpus=gpus,
                policy_endpoints=active_endpoints,
                spec=collection_spec,
            )
        else:
            trajectory_collector(jobs, collection_spec, active_endpoints)
        trajectories = collect_trajectory_index(jobs, spec=collection_spec)
        meta["trajectories"] = trajectories
        meta["phase"] = "scoring"
        atomic_write_json(meta_path, meta)
        if defer_scoring:
            return _summary(destination, meta, gpus)
        score_evaluation_condition(
            destination,
            protocol_path,
            templates_path,
            workers=score_workers,
            require_formal=panel_id == FORMAL_BENCHMARK_PANEL_ID,
        )
    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["trajectories"] = collect_trajectory_index(
            jobs, spec=collection_spec, partial=True
        )
        atomic_write_json(meta_path, meta)
        raise
    meta = _load_running_meta(meta_path)
    return _summary(destination, meta, gpus)


def _validate_published_condition(
    root: Path,
    jobs: list[TrajectoryAssetJob],
    spec: TrajectoryCollectionSpec,
) -> None:
    meta = _load_running_meta(root / "meta.json")
    trajectories = collect_trajectory_index(jobs, spec=spec)
    if trajectories != meta.get("trajectories"):
        raise ValueError("published evaluation trajectory index differs from the trajectories")
    published = load_evaluation_condition(root)
    expected = {
        (item["asset_id"], item["initial_state_index"], item["replicate"]): (
            item["seed"],
            item["policy_seed"],
        )
        for item in trajectories
    }
    actual = {
        (case.key.asset_id, case.key.initial_state_index, case.key.replicate): (
            case.initial_state_seed,
            case.policy_seed,
        )
        for case in published.cases
    }
    if actual != expected:
        raise ValueError("published evaluation case index differs from the trajectories")


def _load_running_meta(path: Path) -> dict[str, Any]:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read evaluation metadata {path}: {exc}") from exc
    if meta.get("schema") != EVALUATION_CONDITION_SCHEMA:
        raise ValueError("unsupported evaluation condition schema")
    return meta


def _summary(root: Path, meta: dict[str, Any], gpus) -> dict[str, Any]:
    result = {
        "compatible": True,
        "output": str(root),
        "condition_id": meta["condition_id"],
        "status": meta["status"],
        "phase": meta["phase"],
        "protocol_stage": meta["protocol"]["stage"],
        "case_count": meta["case_count"],
        "trajectory_count": len(meta.get("trajectories", [])),
        "checkpoint_sha256": meta["policy"]["checkpoint_sha256"],
        "training_eligible": False,
        "gpus": list(gpus),
    }
    performance = _performance_summary(root, meta.get("trajectories", []))
    if performance is not None:
        result["performance"] = performance
    return result


def _score_evaluation_job(job) -> Any:
    trajectory, template_path, template_sha256, protocol, assets_root = job
    key = (str(template_path), template_sha256)
    template = _SCORING_TEMPLATE_CACHE.get(key)
    if template is None:
        template = load_canonical_template(template_path)
        _SCORING_TEMPLATE_CACHE[key] = template
    return score_evaluation_trajectory(
        trajectory, template, protocol, assets_root=assets_root,
    ).result


def _performance_summary(root: Path, trajectories) -> dict[str, float | int] | None:
    batches = []
    seen = set()
    for item in trajectories:
        try:
            metadata = TrajectoryReader(root / item["path"]).manifest.metadata
            parallelism = metadata["parallelism"]
            key = (
                item["asset_id"],
                item["replicate"],
                parallelism["batch_start"],
                metadata["policy_seed"],
            )
            if key in seen:
                continue
            seen.add(key)
            performance = dict(metadata["execution_report"]["performance"])
            performance["active_world_count"] = parallelism["active_worlds"]
            performance["trajectory_publication"] = metadata.get(
                "trajectory_publication", {}
            )
            batches.append(performance)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, EOFError):
            continue
    if not batches:
        return None
    total_seconds = sum(float(item["wall_seconds"]) for item in batches)
    result: dict[str, float | int] = {
        "batch_count": len(batches),
        "episode_worker_seconds": total_seconds,
        "trajectory_finalize_seconds": sum(
            float(item["trajectory_publication"].get("finalize_seconds", 0.0))
            for item in batches
        ),
    }
    for phase in ("policy_session_reset", "observation", "policy_end_to_end", "execution"):
        phase_seconds = sum(float(item["phases"][phase]["total_seconds"]) for item in batches)
        result[f"{phase}_seconds"] = phase_seconds
        result[f"{phase}_fraction"] = phase_seconds / total_seconds
    unattributed_seconds = sum(
        float(item.get("unattributed_seconds", item.get("engine_overhead_seconds", 0.0)))
        for item in batches
    )
    overlapped_seconds = sum(
        float(item.get("overlapped_phase_seconds", 0.0)) for item in batches
    )
    result["unattributed_seconds"] = unattributed_seconds
    result["unattributed_fraction"] = unattributed_seconds / total_seconds
    result["overlapped_phase_seconds"] = overlapped_seconds
    result["overlapped_phase_fraction"] = overlapped_seconds / total_seconds
    return result
