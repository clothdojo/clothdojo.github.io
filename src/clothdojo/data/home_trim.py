"""Remove the terminal operator-requested return-to-home phase from HIL data."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

TRIM_SCHEMA = "clothdojo_home_return_trim_collection"
TRIM_FORMAT_VERSION = 1


def trim_home_returns(
    source: str | Path,
    output: str | Path,
    *,
    tolerance: float = 1.0e-6,
    min_hold_frames: int = 20,
    min_output_actions: int = 50,
    workers: int = 8,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Copy original HIL trajectories, cutting at the first final arm-home command.

    The source may be an original-plus-mirror collection. Mirrors are deliberately
    ignored: callers regenerate them from this independently published collection.
    """

    source_root = Path(source).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"home-trim source does not exist: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"home-trim output already exists: {output_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("home-trim output must be outside the source collection")
    if tolerance < 0 or min_hold_frames < 1 or min_output_actions < 1:
        raise ValueError("home-trim thresholds must be non-negative/positive")
    if not 1 <= workers <= 32:
        raise ValueError("home-trim workers must be in [1, 32]")

    trajectories = []
    skipped_mirrors = 0
    skipped_incomplete = 0
    for trajectory in sorted((source_root / "trajectories").iterdir()):
        manifest_path = trajectory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete":
            skipped_incomplete += 1
            continue
        if isinstance(manifest.get("metadata", {}).get("augmentation"), dict):
            skipped_mirrors += 1
            continue
        trajectories.append((trajectory, manifest))
    if not trajectories:
        raise ValueError(f"home-trim source has no complete original trajectories: {source_root}")

    plans = [
        _plan_trajectory(
            trajectory,
            manifest,
            tolerance=tolerance,
            min_hold_frames=min_hold_frames,
            min_output_actions=min_output_actions,
        )
        for trajectory, manifest in trajectories
    ]
    summary = _summary(
        source_root,
        output_root,
        plans,
        tolerance=tolerance,
        min_hold_frames=min_hold_frames,
        min_output_actions=min_output_actions,
        skipped_mirrors=skipped_mirrors,
        skipped_incomplete=skipped_incomplete,
    )
    if plan_only:
        return summary

    staging = output_root.with_name(f".{output_root.name}.tmp-{uuid4().hex}")
    try:
        (staging / "trajectories" / "annotations").mkdir(parents=True)

        def publish(plan: dict[str, Any]) -> dict[str, Any]:
            source_trajectory = source_root / "trajectories" / plan["trajectory_id"]
            destination = staging / "trajectories" / plan["trajectory_id"]
            destination.mkdir()
            annotation = (
                source_root
                / "trajectories"
                / "annotations"
                / f'{plan["trajectory_id"]}.segments.json'
            )
            if plan["trimmed"] and annotation.is_file():
                raise ValueError(
                    "home-trim does not guess how reviewed segments cross the cut: "
                    f"{annotation}"
                )
            if annotation.is_file():
                shutil.copy2(
                    annotation,
                    staging / "trajectories" / "annotations" / annotation.name,
                )
            manifest = _read_json(source_trajectory / "manifest.json")
            if plan["trimmed"]:
                _slice_npz(
                    source_trajectory / "actions.npz",
                    destination / "actions.npz",
                    plan["output_action_count"],
                )
                _slice_npz(
                    source_trajectory / "states.npz",
                    destination / "states.npz",
                    plan["output_action_count"] + 1,
                )
                manifest["action_count"] = plan["output_action_count"]
                manifest["state_count"] = plan["output_action_count"] + 1
            else:
                shutil.copy2(source_trajectory / "actions.npz", destination / "actions.npz")
                shutil.copy2(source_trajectory / "states.npz", destination / "states.npz")
            manifest.setdefault("metadata", {})["home_return_trim"] = {
                "schema": "clothdojo_home_return_trim_v1",
                "source_manifest_sha256": plan["source_manifest_sha256"],
                "source_action_count": plan["source_action_count"],
                "cut_action_index": plan["output_action_count"],
                "removed_actions": plan["removed_actions"],
                "reason": plan["reason"],
                "detector": {
                    "signal": "final_sustained_per_arm_joint_target_zero_suffix",
                    "cut_rule": "earliest_left_or_right_suffix_start",
                    "tolerance": tolerance,
                    "min_hold_frames": min_hold_frames,
                },
            }
            _write_json(destination / "manifest.json", manifest)
            return {
                **plan,
                "output_manifest_sha256": _sha256(destination / "manifest.json"),
            }

        with ThreadPoolExecutor(max_workers=min(workers, len(plans))) as pool:
            published = list(pool.map(publish, plans))
        summary["trajectories"] = published
        summary["status"] = "complete"
        _write_json(staging / "home_trim.json", summary)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _plan_trajectory(
    trajectory: Path,
    manifest: dict[str, Any],
    *,
    tolerance: float,
    min_hold_frames: int,
    min_output_actions: int,
) -> dict[str, Any]:
    with np.load(trajectory / "actions.npz", allow_pickle=False) as archive:
        if "joint_target" not in archive:
            raise ValueError(f"joint_target is missing: {trajectory}")
        joint_target = np.asarray(archive["joint_target"])
        sample_index = np.asarray(archive["sample_index"])
    action_count = int(manifest.get("action_count", -1))
    if (
        joint_target.ndim != 2
        or joint_target.shape != (action_count, 12)
        or sample_index.shape != (action_count,)
        or not np.array_equal(sample_index, np.arange(action_count))
        or not np.isfinite(joint_target).all()
    ):
        raise ValueError(f"invalid canonical action archive: {trajectory}")
    left = _final_home_suffix_start(joint_target[:, :6], tolerance, min_hold_frames)
    right = _final_home_suffix_start(joint_target[:, 6:], tolerance, min_hold_frames)
    if left is None or right is None:
        cut = action_count
        reason = "no_confident_bilateral_home_suffix"
    else:
        cut = min(left, right)
        reason = "terminal_operator_home_return"
    if cut < min_output_actions:
        raise ValueError(
            f"home trim would leave fewer than {min_output_actions} actions: "
            f"{trajectory.name}: {cut}"
        )
    return {
        "trajectory_id": trajectory.name,
        "source_manifest_sha256": _sha256(trajectory / "manifest.json"),
        "source_action_count": action_count,
        "output_action_count": cut,
        "removed_actions": action_count - cut,
        "left_home_start": left,
        "right_home_start": right,
        "trimmed": cut < action_count,
        "reason": reason,
    }


def _final_home_suffix_start(
    joint_target: np.ndarray, tolerance: float, min_hold_frames: int
) -> int | None:
    at_home = np.max(np.abs(joint_target), axis=1) <= tolerance
    if len(at_home) < min_hold_frames or not np.all(at_home[-min_hold_frames:]):
        return None
    non_home = np.flatnonzero(~at_home)
    return int(non_home[-1] + 1) if len(non_home) else 0


def _slice_npz(source: Path, destination: Path, stop: int) -> None:
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name])[:stop] for name in archive.files}
    np.savez_compressed(destination, **arrays)


def _summary(
    source: Path,
    output: Path,
    plans: list[dict[str, Any]],
    **settings: Any,
) -> dict[str, Any]:
    return {
        "schema": TRIM_SCHEMA,
        "format_version": TRIM_FORMAT_VERSION,
        "status": "planned",
        "source": str(source),
        "output": str(output),
        "source_is_runtime_dependency": False,
        **settings,
        "input_original_trajectories": len(plans),
        "trimmed_trajectories": sum(plan["trimmed"] for plan in plans),
        "untrimmed_trajectories": sum(not plan["trimmed"] for plan in plans),
        "input_actions": sum(plan["source_action_count"] for plan in plans),
        "output_actions": sum(plan["output_action_count"] for plan in plans),
        "removed_actions": sum(plan["removed_actions"] for plan in plans),
        "minimum_output_actions": min(plan["output_action_count"] for plan in plans),
        "trajectories": plans,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON value must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
