"""Adaptive per-asset success-quota rollout collection."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from clothdojo.evaluation.metric import load_canonical_template
from clothdojo.evaluation.protocol import (
    GEOMETRIC_SUCCESS_CRITERION,
    load_evaluation_protocol,
)
from clothdojo.evaluation.scoring import (
    score_render_selection_trajectory,
    score_rollout_trajectory,
)
from clothdojo.rollout.scheduling import run_policy_asset_workers
from clothdojo.rollout.trajectory_collection import (
    TrajectoryAssetJob,
    TrajectoryCollectionSpec,
    _run_asset,
    atomic_write_json,
    trajectory_name,
    validate_existing_trajectory,
)
from clothdojo.trajectory import (
    TrajectoryReader,
    compact_render_trajectory,
    publish_render_trajectory,
)
from clothdojo.trajectory.schema import RENDER_ONLY_STORAGE_KIND

SUCCESS_SELECTION_SCHEMA = "clothdojo_success_selection"
SUCCESS_SELECTION_FORMAT_VERSION = 1
SUCCESS_CRITERION = GEOMETRIC_SUCCESS_CRITERION
SUCCESS_RANKING = (
    "success_at_end_desc",
    "terminal_combined_distance_asc",
    "best_observed_combined_distance_asc",
    "confirmed_step_asc",
    "initial_state_index_asc",
)


@dataclass(frozen=True, slots=True)
class SuccessQuotaSpec:
    protocol_path: str
    protocol_fingerprint: str
    template_paths: dict[str, tuple[str, str]]
    quota: int
    initial_attempts: int
    candidates_root: str
    selections_root: str

    def __post_init__(self) -> None:
        for name, value in (("quota", self.quota), ("initial_attempts", self.initial_attempts)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"success {name} must be a positive integer")
        if self.quota > self.initial_attempts:
            raise ValueError("success quota cannot exceed initial attempts")
        if not self.template_paths:
            raise ValueError("success selection requires canonical templates")


def collect_success_trajectories(
    jobs: list[TrajectoryAssetJob],
    *,
    gpus: tuple[str, ...],
    policy_endpoints: tuple[str, ...],
    collection_spec: TrajectoryCollectionSpec,
    success_spec: SuccessQuotaSpec,
) -> None:
    """Run only unfinished assets and publish their selected render-only paths."""

    pending = [
        job
        for job in jobs
        if not _selection_is_complete(job, collection_spec, success_spec)
    ]
    if not pending:
        return
    worker_count = min(len(gpus), len(pending))
    run_policy_asset_workers(
        pending,
        gpus=gpus[:worker_count],
        policy_endpoints=policy_endpoints[:worker_count],
        worker_target=_success_worker,
        common_args=(collection_spec, success_spec),
        process_kind="success-quota rollout",
    )


def collect_success_trajectory_index(
    jobs: list[TrajectoryAssetJob],
    *,
    collection_spec: TrajectoryCollectionSpec,
    success_spec: SuccessQuotaSpec,
    partial: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return selected trajectories and assets that exhausted their state pool."""

    trajectories: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    published_spec = replace(
        collection_spec,
        training_eligible=False,
        storage_kind=RENDER_ONLY_STORAGE_KIND,
    )
    for job in jobs:
        try:
            record = _load_selection(job, collection_spec, success_spec)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            if partial:
                continue
            raise
        if record is None:
            if partial:
                continue
            raise RuntimeError(f"success selection is missing for asset {job.asset_id}")
        if record["status"] == "shortfall":
            shortfalls.append(
                {
                    "asset_id": job.asset_id,
                    "selected": len(record["selected"]),
                    "quota": success_spec.quota,
                    "attempted": record["attempted"],
                }
            )
        for rank, item in enumerate(record["selected"]):
            index = int(item["initial_state_index"])
            path = Path(collection_spec.trajectories_root) / trajectory_name(
                job.asset_id, index, None
            )
            validate_existing_trajectory(
                path,
                job,
                spec=published_spec,
                initial_state_index=index,
                replicate=None,
                validate_arrays=False,
            )
            reader = TrajectoryReader(path)
            if not reader.is_render_only:
                raise ValueError(f"selected trajectory is not render-only: {path}")
            metadata = reader.manifest.metadata
            trajectories.append(
                {
                    "asset_id": job.asset_id,
                    "initial_state_index": index,
                    "seed": metadata["initial_state"]["seed"],
                    "policy_seed": metadata["policy_seed"],
                    "path": (Path("trajectories") / path.name).as_posix(),
                    "action_count": reader.manifest.action_count,
                    "execution_report": metadata.get("execution_report", {}),
                    "selection_rank": rank,
                    "success": item["score"],
                }
            )
    return trajectories, shortfalls


def _success_worker(job_queue, result_queue, gpu, endpoint, collection_spec, success_spec):
    import os

    if gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cpu" if gpu == "cpu" else "cuda:0"
    while True:
        job = job_queue.get()
        if job is None:
            return
        try:
            _run_success_asset(
                job,
                endpoint=endpoint,
                device=device,
                collection_spec=collection_spec,
                success_spec=success_spec,
            )
        except Exception as exc:
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


def _run_success_asset(
    job: TrajectoryAssetJob,
    *,
    endpoint: str,
    device: str,
    collection_spec: TrajectoryCollectionSpec,
    success_spec: SuccessQuotaSpec,
) -> None:
    if _selection_is_complete(job, collection_spec, success_spec):
        shutil.rmtree(Path(success_spec.candidates_root) / job.asset_id, ignore_errors=True)
        return
    template_entry = success_spec.template_paths.get(job.asset_id)
    if template_entry is None:
        raise ValueError(f"canonical template is missing asset {job.asset_id}")
    protocol = load_evaluation_protocol(success_spec.protocol_path)
    if protocol.fingerprint != success_spec.protocol_fingerprint:
        raise ValueError("success protocol changed after collection planning")
    if collection_spec.horizon_steps != protocol.horizon_steps:
        raise ValueError("rollout and success protocol horizons differ")
    template = load_canonical_template(template_entry[0])

    candidate_root = Path(success_spec.candidates_root) / job.asset_id
    candidate_root.mkdir(parents=True, exist_ok=True)
    candidate_spec = replace(
        collection_spec,
        trajectories_root=str(candidate_root),
        training_eligible=False,
        storage_kind=RENDER_ONLY_STORAGE_KIND,
    )
    first_count = min(success_spec.initial_attempts, job.state_count)
    first_job = replace(job, state_count=first_count)
    _run_asset(first_job, endpoint=endpoint, device=device, spec=candidate_spec)
    scored = _score_candidates(first_job, candidate_spec, template, protocol)
    successful = [item for item in scored if item["score"]["success"]]
    if len(successful) < success_spec.quota and first_count < job.state_count:
        _run_asset(job, endpoint=endpoint, device=device, spec=candidate_spec)
        scored.extend(
            _score_candidates(
                job,
                candidate_spec,
                template,
                protocol,
                start=first_count,
            )
        )
        successful = [item for item in scored if item["score"]["success"]]

    selected = sorted(successful, key=_success_rank)[: success_spec.quota]
    final_root = Path(collection_spec.trajectories_root)
    final_root.mkdir(parents=True, exist_ok=True)
    for item in selected:
        name = trajectory_name(job.asset_id, item["initial_state_index"], None)
        candidate = candidate_root / name
        if TrajectoryReader(candidate).is_render_only:
            publish_render_trajectory(candidate, final_root / name)
        else:
            compact_render_trajectory(candidate, final_root / name)

    record = {
        "schema": SUCCESS_SELECTION_SCHEMA,
        "format_version": SUCCESS_SELECTION_FORMAT_VERSION,
        "status": "complete" if len(selected) == success_spec.quota else "shortfall",
        "asset_id": job.asset_id,
        "pool_sha256": job.pool_sha256,
        "protocol_fingerprint": success_spec.protocol_fingerprint,
        "template_sha256": template_entry[1],
        "quota": success_spec.quota,
        "initial_attempts": success_spec.initial_attempts,
        "maximum_attempts": job.state_count,
        "attempted": len(scored),
        "success_count": len(successful),
        "criterion": SUCCESS_CRITERION,
        "ranking": list(SUCCESS_RANKING),
        "selected": selected,
    }
    selection_path = _selection_path(job, success_spec)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_path, record)
    shutil.rmtree(candidate_root, ignore_errors=True)


def _score_candidates(job, spec, template, protocol, *, start: int = 0):
    scored = []
    root = Path(spec.trajectories_root)
    for index in range(start, job.state_count):
        path = root / trajectory_name(job.asset_id, index, None)
        reader = TrajectoryReader(path)
        if reader.is_render_only:
            scored_trajectory = score_render_selection_trajectory(
                path, template, protocol
            )
        else:
            scored_trajectory = score_rollout_trajectory(path, template, protocol)
        result = scored_trajectory.result
        score = result.to_json_dict()
        score["selection_gate"] = {
            "criterion": SUCCESS_CRITERION,
            "continuous_seconds": protocol.success_window_seconds,
            "continuous_samples": protocol.success_window_samples,
            "release_required": False,
            "speed_required": False,
        }
        scored.append(
            {
                "initial_state_index": index,
                "score": score,
            }
        )
    return scored


def _success_rank(item: dict[str, Any]) -> tuple[Any, ...]:
    score = item["score"]
    return (
        not score["success_at_end"],
        score["terminal"]["combined_distance"],
        score["best_observed"]["combined_distance"],
        score["confirmed_step"],
        item["initial_state_index"],
    )


def _selection_path(job: TrajectoryAssetJob, spec: SuccessQuotaSpec) -> Path:
    return Path(spec.selections_root) / f"{job.asset_id}.json"


def _selection_is_complete(job, collection_spec, success_spec) -> bool:
    try:
        return _load_selection(job, collection_spec, success_spec) is not None
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _load_selection(job, collection_spec, success_spec):
    path = _selection_path(job, success_spec)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    template = success_spec.template_paths[job.asset_id]
    expected = {
        "schema": SUCCESS_SELECTION_SCHEMA,
        "format_version": SUCCESS_SELECTION_FORMAT_VERSION,
        "asset_id": job.asset_id,
        "pool_sha256": job.pool_sha256,
        "protocol_fingerprint": success_spec.protocol_fingerprint,
        "template_sha256": template[1],
        "quota": success_spec.quota,
        "initial_attempts": success_spec.initial_attempts,
        "maximum_attempts": job.state_count,
        "criterion": SUCCESS_CRITERION,
        "ranking": list(SUCCESS_RANKING),
    }
    actual = {key: record.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"existing success selection settings differ: {path}")
    selected = record.get("selected")
    status = record.get("status")
    if status not in {"complete", "shortfall"} or not isinstance(selected, list):
        raise ValueError(f"success selection record is invalid: {path}")
    if len(selected) > success_spec.quota or (
        status == "complete" and len(selected) != success_spec.quota
    ):
        raise ValueError(f"success selection quota is inconsistent: {path}")
    indices = [item.get("initial_state_index") for item in selected]
    if len(indices) != len(set(indices)) or any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < job.state_count
        for index in indices
    ):
        raise ValueError(f"success selection indices are invalid: {path}")
    for item in selected:
        if item.get("score", {}).get("success") is not True:
            raise ValueError(f"success selection contains a failed trajectory: {path}")
        final = Path(collection_spec.trajectories_root) / trajectory_name(
            job.asset_id, item["initial_state_index"], None
        )
        if not final.is_dir():
            raise RuntimeError(f"selected render-only trajectory is missing: {final}")
    return record
