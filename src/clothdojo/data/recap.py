"""Build an immutable RECAP episode plan from expert, rollout, and HIL data."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from clothdojo.collection.hil import HIL_REVIEW_SCHEMA
from clothdojo.data.annotations import ANNOTATION_SCHEMA, KNOWN_LABELS
from clothdojo.rollout.collection import ROLLOUT_COLLECTION_SCHEMA

RECAP_PLAN_SCHEMA = "clothdojo_recap_plan_v1"
RECAP_EXPORT_SCHEMA = "clothdojo_recap_export_v1"
RECAP_VALUE_SPLIT_SCHEMA = "clothdojo_recap_value_split_v1"
RECAP_RETURNS_SCHEMA = "clothdojo_recap_returns_v2"
RECAP_RETURNS_TAG = "fail4500_state_v2"
RECAP_PREFIX60_RETURNS_TAG = "fail3600_state_v3"
RECAP_PROMPT = "flatten the shirt"
RECAP_DATASETS = ("expert", "hil_rollout", "hil_correction")


def build_recap_plan(
    expert_root: str | Path,
    hil_rollout_root: str | Path,
    correction_root: str | Path,
) -> dict[str, Any]:
    """Validate source identities and select episodes under the agreed RECAP rules."""

    expert = Path(expert_root).expanduser().resolve()
    rollout = Path(hil_rollout_root).expanduser().resolve()
    corrections = Path(correction_root).expanduser().resolve()

    expert_episodes, expert_excluded = _expert_episodes(expert)
    rollout_episodes, rollout_excluded, breakpoint_keys = _rollout_episodes(rollout)
    correction_episodes, correction_excluded = _correction_episodes(
        corrections,
        rollout_root=rollout,
        breakpoint_keys=breakpoint_keys,
    )

    success_rollouts = sum(item["outcome"] == "success" for item in rollout_episodes)
    failure_rollouts = sum(item["outcome"] == "failure" for item in rollout_episodes)
    intervention_rollouts = sum(bool(item["intervention_points"]) for item in rollout_episodes)
    return {
        "schema": RECAP_PLAN_SCHEMA,
        "status": "ready",
        "task": {"task_name": "shirt_flatten", "prompt": RECAP_PROMPT},
        "semantics": {
            "expert_outcome": "all_complete_trajectories_are_success",
            "expert_bad_ranges": "metadata_only_no_cropping",
            "rollout_success_endpoint": "completion_state_excludes_actions_at_or_after_mark",
            "intervention_only_rollout_outcome": "failure_at_recorded_terminal_state",
            "unmarked_rollout": "excluded",
            "correction_use": "policy_only_force_positive_without_physical_concatenation",
            "duplicate_corrections": "include_all_complete_recordings_at_each_breakpoint",
        },
        "sources": {
            "expert": {"root": str(expert)},
            "hil_rollout": {
                "root": str(rollout),
                "meta_sha256": _sha256(rollout / "meta.json"),
                "review_sha256": _sha256(rollout / "hil_review.json"),
            },
            "hil_correction": {"root": str(corrections)},
        },
        "datasets": {
            "expert": {
                "uses": ["value", "policy"],
                "episodes": expert_episodes,
            },
            "hil_rollout": {
                "uses": ["value", "policy"],
                "episodes": rollout_episodes,
            },
            "hil_correction": {
                "uses": ["policy"],
                "force_positive": True,
                "episodes": correction_episodes,
            },
        },
        "excluded": {
            "expert": expert_excluded,
            "hil_rollout": rollout_excluded,
            "hil_correction": correction_excluded,
        },
        "counts": {
            "expert_success": len(expert_episodes),
            "expert_excluded": len(expert_excluded),
            "hil_rollout_success": success_rollouts,
            "hil_rollout_failure": failure_rollouts,
            "hil_rollout_with_intervention": intervention_rollouts,
            "hil_rollout_excluded_unmarked": sum(
                item["reason"] == "unmarked" for item in rollout_excluded
            ),
            "hil_rollout_excluded_initial_success": sum(
                item["reason"] == "success_at_initial_state"
                for item in rollout_excluded
            ),
            "hil_correction_policy_positive": len(correction_episodes),
            "hil_correction_excluded": len(correction_excluded),
        },
    }


def write_recap_plan(plan: dict[str, Any], output: str | Path) -> Path:
    """Atomically publish one fully validated plan without modifying source data."""

    if plan.get("schema") != RECAP_PLAN_SCHEMA or plan.get("status") != "ready":
        raise ValueError("only a ready RECAP plan can be written")
    path = Path(output).expanduser().resolve()
    _atomic_json(path, plan)
    return path


def prepare_recap_plan(
    expert_root: str | Path,
    hil_rollout_root: str | Path,
    correction_root: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Build a plan and optionally write it, returning a compact audit summary."""

    plan = build_recap_plan(expert_root, hil_rollout_root, correction_root)
    result: dict[str, Any] = {
        "compatible": True,
        "schema": plan["schema"],
        "status": plan["status"],
        "semantics": plan["semantics"],
        "counts": plan["counts"],
    }
    if output is not None:
        path = write_recap_plan(plan, output)
        result["output"] = str(path)
        result["plan_sha256"] = _sha256(path)
    return result


def prepare_recap_mirrored_plan(
    plan_path: str | Path,
    expert_root: str | Path,
    hil_rollout_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Expand a frozen RECAP plan with canonical original/mirror pairs."""

    from clothdojo.data.mirror import MIRROR_COLLECTION_SCHEMA, MIRROR_SUFFIX

    source_path, source_plan = _load_recap_plan(plan_path)
    plan = copy.deepcopy(source_plan)
    roots = {
        "expert": Path(expert_root).expanduser().resolve(),
        "hil_rollout": Path(hil_rollout_root).expanduser().resolve(),
    }
    role_counts: dict[str, dict[str, int]] = {}
    for role, root in roots.items():
        original_source = Path(source_plan["sources"][role]["root"]).expanduser().resolve()
        original_episodes = source_plan["datasets"][role]["episodes"]
        _validate_recap_sources(source_plan, role, original_source, original_episodes)

        augmentation_path = root / "augmentation.json"
        augmentation = _read_object(augmentation_path, f"{role} mirror augmentation")
        if (
            augmentation.get("schema") != MIRROR_COLLECTION_SCHEMA
            or augmentation.get("status") != "complete"
            or augmentation.get("source_is_runtime_dependency") is not False
        ):
            raise RuntimeError(f"RECAP mirror collection is not complete: {root}")
        raw_pairs = augmentation.get("trajectory_pairs")
        if not isinstance(raw_pairs, list) or any(not isinstance(item, dict) for item in raw_pairs):
            raise RuntimeError(f"RECAP mirror pairs are invalid: {root}")
        pairs = {item.get("source_id"): item for item in raw_pairs}
        expected_ids = {item["source_episode_id"] for item in original_episodes}
        if not expected_ids.issubset(pairs) or len(pairs) != len(raw_pairs):
            raise RuntimeError(
                f"RECAP mirror collection differs from the frozen {role} selection: "
                f"required={len(expected_ids)}, available={len(pairs)}"
            )

        expanded = []
        for original in original_episodes:
            source_id = original["source_episode_id"]
            pair = pairs[source_id]
            mirror_id = pair.get("mirror_id")
            if mirror_id != f"{source_id}{MIRROR_SUFFIX}":
                raise RuntimeError(f"invalid RECAP mirror identity: {source_id}")
            if pair.get("source_manifest_sha256") != original.get("manifest_sha256"):
                raise RuntimeError(f"RECAP mirror source changed after planning: {source_id}")
            for episode_id, expected_sha in (
                (source_id, pair.get("source_manifest_sha256")),
                (mirror_id, pair.get("mirror_manifest_sha256")),
            ):
                manifest = root / "trajectories" / episode_id / "manifest.json"
                if not isinstance(expected_sha, str) or _sha256(manifest) != expected_sha:
                    raise RuntimeError(f"RECAP mirror manifest differs: {role}:{episode_id}")

            original_copy = copy.deepcopy(original)
            original_copy["mirror_pair_id"] = source_id
            original_copy["is_mirror"] = False
            mirror = copy.deepcopy(original)
            mirror.update(
                {
                    "source_episode_id": mirror_id,
                    "path": f"trajectories/{mirror_id}",
                    "manifest_sha256": pair["mirror_manifest_sha256"],
                    "mirror_pair_id": source_id,
                    "mirror_of_source_episode_id": source_id,
                    "is_mirror": True,
                }
            )
            expanded.extend((original_copy, mirror))

        previous_source = copy.deepcopy(source_plan["sources"][role])
        plan["sources"][role] = {
            **previous_source,
            "root": str(root),
            "origin_root": str(original_source),
            "augmentation_sha256": _sha256(augmentation_path),
        }
        plan["datasets"][role]["episodes"] = expanded
        role_counts[role] = {
            "original": len(original_episodes),
            "mirror": len(original_episodes),
            "total": len(expanded),
        }

    plan["semantics"]["mirror_augmentation"] = (
        "canonical_state_action_bilateral_pairs_same_asset_and_split"
    )
    for key in (
        "expert_success",
        "hil_rollout_success",
        "hil_rollout_failure",
        "hil_rollout_metric_selected",
    ):
        if key in plan["counts"]:
            plan["counts"][key] *= 2
    plan["derived_from"] = {
        "plan": str(source_path),
        "plan_sha256": _sha256(source_path),
    }
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"mirrored RECAP plan already exists: {destination}")
    write_recap_plan(plan, destination)
    return {
        "compatible": True,
        "output": str(destination),
        "plan_sha256": _sha256(destination),
        "datasets": role_counts,
    }


def prepare_recap_value_split(
    plan_path: str | Path,
    output: str | Path,
    *,
    eval_seen_assets: int = 10,
    eval_unseen_assets: int = 10,
    seed: int = 260904,
) -> dict[str, Any]:
    """Freeze an asset-disjoint value train/eval split over a RECAP plan."""

    path, plan = _load_recap_plan(plan_path)
    for name, value in (
        ("eval_seen_assets", eval_seen_assets),
        ("eval_unseen_assets", eval_unseen_assets),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if eval_seen_assets + eval_unseen_assets < 1:
        raise ValueError("the value validation split must contain at least one asset")

    expert = plan["datasets"]["expert"]["episodes"]
    rollouts = plan["datasets"]["hil_rollout"]["episodes"]
    expert_assets = {str(item["asset_id"]) for item in expert}
    rollout_assets = {str(item["asset_id"]) for item in rollouts}
    seen_candidates = rollout_assets & expert_assets
    unseen_candidates = rollout_assets - expert_assets
    selected_seen = _select_assets(
        seen_candidates,
        eval_seen_assets,
        seed=seed,
        split="seen",
    )
    selected_unseen = _select_assets(
        unseen_candidates,
        eval_unseen_assets,
        seed=seed,
        split="unseen",
    )
    eval_assets = set(selected_seen) | set(selected_unseen)

    datasets = {
        name: _value_partition(episodes, eval_assets)
        for name, episodes in (("expert", expert), ("hil_rollout", rollouts))
    }
    split = {
        "schema": RECAP_VALUE_SPLIT_SCHEMA,
        "status": "ready",
        "plan": str(path),
        "plan_sha256": _sha256(path),
        "strategy": {
            "unit": "asset_id",
            "selection": "sha256_rank",
            "seed": seed,
            "seen_definition": "asset_present_in_expert_dataset",
            "policy_data_affected": False,
        },
        "assets": {
            "eligible_seen": len(seen_candidates),
            "eligible_unseen": len(unseen_candidates),
            "eval_seen": selected_seen,
            "eval_unseen": selected_unseen,
        },
        "datasets": datasets,
    }
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"RECAP value split already exists: {destination}")
    _atomic_json(destination, split)
    return {
        "compatible": True,
        "output": str(destination),
        "split_sha256": _sha256(destination),
        "eval_assets": len(eval_assets),
        "eval_seen_assets": len(selected_seen),
        "eval_unseen_assets": len(selected_unseen),
        "datasets": {
            name: value["counts"] for name, value in datasets.items()
        },
    }


def prepare_recap_value_trajectory_split(
    plan_path: str | Path,
    output: str | Path,
    *,
    eval_fraction: float = 0.1,
    seed: int = 260908,
) -> dict[str, Any]:
    """Freeze a stratified trajectory split while keeping mirror pairs together."""

    path, plan = _load_recap_plan(plan_path)
    if isinstance(eval_fraction, bool) or not isinstance(eval_fraction, (int, float)):
        raise TypeError("eval_fraction must be a number")
    if not 0.0 < float(eval_fraction) < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    datasets: dict[str, dict[str, Any]] = {}
    eval_groups_by_role: dict[str, list[str]] = {}
    strata_audit: dict[str, dict[str, int]] = {}
    for role in ("expert", "hil_rollout"):
        episodes = plan["datasets"][role]["episodes"]
        groups: dict[str, list[int]] = {}
        group_strata: dict[str, str] = {}
        for index, item in enumerate(episodes):
            pair_id = item.get("mirror_pair_id")
            if not isinstance(pair_id, str) or not pair_id:
                raise ValueError(
                    "trajectory split requires a mirrored RECAP plan with mirror_pair_id"
                )
            groups.setdefault(pair_id, []).append(index)
            stratum = ":".join(
                (
                    role,
                    str(item.get("outcome", "unknown")),
                    str(item.get("rollout_source", "default")),
                )
            )
            previous = group_strata.setdefault(pair_id, stratum)
            if previous != stratum:
                raise RuntimeError(f"mirror pair crosses trajectory strata: {role}:{pair_id}")

        for pair_id, indices in groups.items():
            members = [episodes[index] for index in indices]
            if len(indices) != 2 or {item.get("is_mirror") for item in members} != {False, True}:
                raise RuntimeError(f"incomplete original/mirror pair: {role}:{pair_id}")

        by_stratum: dict[str, list[str]] = {}
        for pair_id, stratum in group_strata.items():
            by_stratum.setdefault(stratum, []).append(pair_id)
        eval_groups: set[str] = set()
        for stratum, candidates in sorted(by_stratum.items()):
            if len(candidates) < 2:
                raise RuntimeError(
                    f"trajectory stratum cannot be split into train/eval: {stratum}"
                )
            count = min(
                len(candidates) - 1,
                max(1, round(len(candidates) * float(eval_fraction))),
            )
            ranked = sorted(
                candidates,
                key=lambda pair_id: hashlib.sha256(
                    f"{seed}:{stratum}:{pair_id}".encode()
                ).hexdigest(),
            )
            eval_groups.update(ranked[:count])
            strata_audit[stratum] = {
                "groups": len(candidates),
                "train_groups": len(candidates) - count,
                "eval_groups": count,
            }
        datasets[role] = _value_group_partition(episodes, eval_groups)
        eval_groups_by_role[role] = sorted(eval_groups)

    split = {
        "schema": RECAP_VALUE_SPLIT_SCHEMA,
        "status": "ready",
        "plan": str(path),
        "plan_sha256": _sha256(path),
        "strategy": {
            "unit": "mirror_pair_id",
            "selection": "stratified_sha256_rank",
            "seed": seed,
            "eval_fraction": float(eval_fraction),
            "strata": ["dataset", "outcome", "rollout_source"],
            "policy_data_affected": False,
        },
        "groups": {
            "eval_mirror_pair_ids": eval_groups_by_role,
            "strata": strata_audit,
        },
        "datasets": datasets,
    }
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"RECAP value split already exists: {destination}")
    _atomic_json(destination, split)
    return {
        "compatible": True,
        "output": str(destination),
        "split_sha256": _sha256(destination),
        "eval_fraction": float(eval_fraction),
        "datasets": {name: value["counts"] for name, value in datasets.items()},
    }


def _select_assets(
    candidates: set[str],
    count: int,
    *,
    seed: int,
    split: str,
) -> list[str]:
    if len(candidates) < count:
        raise ValueError(
            f"value validation needs {count} {split} assets, found {len(candidates)}"
        )
    ranked = sorted(
        candidates,
        key=lambda asset_id: hashlib.sha256(
            f"{seed}:{split}:{asset_id}".encode()
        ).hexdigest(),
    )
    return ranked[:count]


def _value_partition(
    episodes: list[dict[str, Any]], eval_assets: set[str]
) -> dict[str, Any]:
    train_indices = [
        index
        for index, item in enumerate(episodes)
        if item.get("asset_id") not in eval_assets
    ]
    eval_indices = [
        index
        for index, item in enumerate(episodes)
        if item.get("asset_id") in eval_assets
    ]
    if not train_indices:
        raise RuntimeError("value split removes every training episode")
    return {
        "train_episode_indices": train_indices,
        "eval_episode_indices": eval_indices,
        "train_source_episode_ids": [
            episodes[index]["source_episode_id"] for index in train_indices
        ],
        "eval_source_episode_ids": [
            episodes[index]["source_episode_id"] for index in eval_indices
        ],
        "counts": {
            "train_episodes": len(train_indices),
            "train_frames": sum(int(episodes[index]["action_count"]) for index in train_indices),
            "eval_episodes": len(eval_indices),
            "eval_frames": sum(int(episodes[index]["action_count"]) for index in eval_indices),
            "train_outcomes": _outcome_counts([episodes[index] for index in train_indices]),
            "eval_outcomes": _outcome_counts([episodes[index] for index in eval_indices]),
        },
    }


def _value_group_partition(
    episodes: list[dict[str, Any]], eval_groups: set[str]
) -> dict[str, Any]:
    train_indices = [
        index
        for index, item in enumerate(episodes)
        if str(item["mirror_pair_id"]) not in eval_groups
    ]
    eval_indices = [
        index
        for index, item in enumerate(episodes)
        if str(item["mirror_pair_id"]) in eval_groups
    ]
    if not train_indices or not eval_indices:
        raise RuntimeError("trajectory value split produced an empty partition")
    return {
        "train_episode_indices": train_indices,
        "eval_episode_indices": eval_indices,
        "train_source_episode_ids": [
            episodes[index]["source_episode_id"] for index in train_indices
        ],
        "eval_source_episode_ids": [
            episodes[index]["source_episode_id"] for index in eval_indices
        ],
        "counts": {
            "train_episodes": len(train_indices),
            "train_frames": sum(int(episodes[index]["action_count"]) for index in train_indices),
            "eval_episodes": len(eval_indices),
            "eval_frames": sum(int(episodes[index]["action_count"]) for index in eval_indices),
            "train_outcomes": _outcome_counts([episodes[index] for index in train_indices]),
            "eval_outcomes": _outcome_counts([episodes[index] for index in eval_indices]),
        },
    }


def inspect_recap_export(
    plan_path: str | Path,
    dataset: str,
    *,
    shard_index: int | None = None,
    shard_count: int | None = None,
    pair_member: str = "all",
) -> dict[str, Any]:
    """Validate one plan role and report its deterministic render workload."""

    path, plan = _load_recap_plan(plan_path)
    source_root, episodes = _recap_export_episodes(
        plan,
        dataset,
        shard_index=shard_index,
        shard_count=shard_count,
        pair_member=pair_member,
    )
    _validate_recap_sources(plan, dataset, source_root, episodes)
    result = {
        "compatible": True,
        "plan": str(path),
        "plan_sha256": _sha256(path),
        "dataset": dataset,
        "source": str(source_root),
        "output_format": "LeRobot v2.1",
        "output_episodes": len(episodes),
        "output_frames": sum(int(item["action_count"]) for item in episodes),
        "outcomes": _outcome_counts(episodes),
        "pair_member": pair_member,
    }
    if shard_count is not None:
        result["trajectory_shard"] = {"index": shard_index, "count": shard_count}
    return result


def export_recap_dataset(
    plan_path: str | Path,
    dataset: str,
    output: str | Path,
    *,
    assets_root: str | Path = "assets",
    device: str = "cuda:0",
    repo_id: str | None = None,
    image_writer_threads: int = 6,
    shard_index: int | None = None,
    shard_count: int | None = None,
    cloth_visual: str = "nocs",
    pair_member: str = "all",
) -> dict[str, Any]:
    """Render one RECAP role as LeRobot while preserving plan provenance."""

    from clothdojo.data.rollout import _export_collection_manifest, _NewtonTrajectoryCamera

    path, plan = _load_recap_plan(plan_path)
    source_root, episodes = _recap_export_episodes(
        plan,
        dataset,
        shard_index=shard_index,
        shard_count=shard_count,
        pair_member=pair_member,
    )
    _validate_recap_sources(plan, dataset, source_root, episodes)
    scenario = {
        "scenario_id": plan["task"]["task_name"],
        "prompt": plan["task"]["prompt"],
    }
    source_info = plan["sources"][dataset]
    provenance_root = Path(source_info.get("origin_root", source_root)).expanduser().resolve()
    source_meta = (
        _read_object(provenance_root / "meta.json", "HIL rollout metadata")
        if dataset == "hil_rollout"
        else {}
    )
    manifest = {
        "scenario": scenario,
        "policy": source_meta.get("policy", {}),
        "training_eligible": True,
        "trajectories": episodes,
    }
    selection = {
        "schema": RECAP_EXPORT_SCHEMA,
        "status": "complete",
        "plan": str(path),
        "plan_sha256": _sha256(path),
        "dataset": dataset,
        "uses": plan["datasets"][dataset]["uses"],
        "force_positive": plan["datasets"][dataset].get("force_positive", False),
        "source_root": str(source_root),
        "episodes": [
            {"episode_index": index, **item}
            for index, item in enumerate(episodes)
        ],
    }

    def camera_factory(asset_id: str, _initial_state: Any, profile: Any):
        return _NewtonTrajectoryCamera(
            asset_id,
            assets_root=assets_root,
            device=device,
            profile=profile,
            cloth_visual=cloth_visual,
        )

    rendering = {"backend": "newton", "cloth_visual": cloth_visual}
    rendering["pair_member"] = pair_member
    if shard_count is not None:
        rendering["trajectory_shard"] = {"index": shard_index, "count": shard_count}
    _export_collection_manifest(
        source_root,
        output,
        manifest=manifest,
        source_kind=f"recap_{dataset}",
        camera_factory=camera_factory,
        rendering=rendering,
        repo_id=repo_id,
        image_writer_threads=image_writer_threads,
        recap_selection=selection,
    )
    result = inspect_recap_export(
        path,
        dataset,
        shard_index=shard_index,
        shard_count=shard_count,
        pair_member=pair_member,
    )
    result["output"] = str(Path(output).expanduser().resolve())
    return result


def reuse_recap_export(
    plan_path: str | Path,
    dataset_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Reuse a role shard whose frozen episodes exactly match a new plan."""

    path, plan = _load_recap_plan(plan_path)
    root = Path(dataset_root).expanduser().resolve()
    recap_path = root / "meta" / "recap.json"
    recap = _read_object(recap_path, "source RECAP export")
    if (
        recap.get("schema") != RECAP_EXPORT_SCHEMA
        or recap.get("status") != "complete"
    ):
        raise RuntimeError(f"source RECAP export is incomplete: {root}")
    role = recap.get("dataset")
    if role not in {"expert", "hil_rollout"}:
        raise ValueError(f"RECAP value export cannot be reused: {root}")
    target_episodes = {
        item["source_episode_id"]: item
        for item in plan["datasets"][role]["episodes"]
    }
    source_episodes = recap.get("episodes")
    if not isinstance(source_episodes, list) or not source_episodes:
        raise RuntimeError(f"source RECAP export has no episodes: {root}")
    rebound = []
    for index, item in enumerate(source_episodes):
        if not isinstance(item, dict) or item.get("episode_index") != index:
            raise RuntimeError(f"source RECAP episode indices are invalid: {root}")
        source_id = item.get("source_episode_id")
        expected = target_episodes.get(source_id)
        if expected is None:
            raise RuntimeError(f"source episode is absent from target plan: {role}:{source_id}")
        for key in ("action_count", "asset_id", "outcome", "manifest_sha256"):
            if item.get(key) != expected.get(key):
                raise RuntimeError(f"source episode changed before reuse: {role}:{source_id}:{key}")
        rebound.append({"episode_index": index, **copy.deepcopy(expected)})

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"reused RECAP export already exists: {destination}")
    staging = destination.with_name(f".{destination.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"unfinished RECAP reuse staging exists: {staging}")
    try:
        method = _reuse_export_tree(root, staging)
        for old_return in (staging / "meta").glob("returns_*"):
            old_return.unlink()
        rebound_recap = copy.deepcopy(recap)
        rebound_recap.update(
            {
                "plan": str(path),
                "plan_sha256": _sha256(path),
                "source_root": plan["sources"][role]["root"],
                "episodes": rebound,
                "reused_from": {
                    "dataset_root": str(root),
                    "recap_sha256": _sha256(recap_path),
                    "method": method,
                },
            }
        )
        _atomic_json(staging / "meta" / "recap.json", rebound_recap)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "compatible": True,
        "output": str(destination),
        "dataset": role,
        "episodes": len(rebound),
        "frames": sum(int(item["action_count"]) for item in rebound),
        "plan_sha256": _sha256(path),
        "reused_from": str(root),
    }


def _reuse_export_tree(root: Path, staging: Path) -> str:
    """Reuse large payloads without copying on hard-link or symlink filesystems."""

    staging.mkdir(parents=True)
    probe = staging / ".hardlink_probe"
    try:
        os.link(root / "meta" / "recap.json", probe)
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EXDEV, errno.EOPNOTSUPP}:
            raise
        for source in root.iterdir():
            destination = staging / source.name
            if source.is_dir() and source.name in {"data", "videos"}:
                destination.symlink_to(source, target_is_directory=True)
            elif source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        return "symlink_payload_copy_metadata"
    else:
        probe.unlink()
        staging.rmdir()
        shutil.copytree(root, staging, copy_function=os.link)
        return "hard_link_without_video_reencoding"


def compute_recap_returns(
    plan_path: str | Path,
    dataset_roots: list[str | Path],
    *,
    tag: str = RECAP_RETURNS_TAG,
    c_fail_coef: float = 1.0,
    gamma: float = 1.0,
    failure_horizon_steps: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write raw RECAP reward/return sidecars for complete value-data exports."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    path, plan = _load_recap_plan(plan_path)
    roots = [Path(value).expanduser().resolve() for value in dataset_roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("dataset_roots must contain distinct RECAP exports")
    if not tag or any(not (char.isalnum() or char in "-_.") for char in tag):
        raise ValueError("RECAP returns tag must contain only letters, digits, '-', '_', or '.'")
    if isinstance(c_fail_coef, bool) or not isinstance(c_fail_coef, (int, float)):
        raise TypeError("c_fail_coef must be a positive number")
    if not float(c_fail_coef) > 0:
        raise ValueError("c_fail_coef must be positive")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be a number")
    if float(gamma) != 1.0:
        raise ValueError("the agreed ClothDojo RECAP return contract requires gamma=1.0")

    plan_sha256 = _sha256(path)
    records: list[tuple[Path, dict[str, Any], list[dict[str, Any]]]] = []
    coverage: dict[str, set[str]] = {"expert": set(), "hil_rollout": set()}
    seen_episode_keys: set[tuple[str, str]] = set()
    observed_max_episode_length = max(
        int(item["action_count"])
        for role in coverage
        for item in plan["datasets"][role]["episodes"]
    )
    if failure_horizon_steps is None:
        max_episode_length = observed_max_episode_length
    else:
        if isinstance(failure_horizon_steps, bool) or not isinstance(
            failure_horizon_steps, int
        ):
            raise TypeError("failure_horizon_steps must be an integer")
        if failure_horizon_steps < observed_max_episode_length:
            raise ValueError(
                "failure_horizon_steps must cover every selected trajectory: "
                f"requested={failure_horizon_steps}, observed={observed_max_episode_length}"
            )
        max_episode_length = failure_horizon_steps
    for root in roots:
        recap = _read_object(root / "meta" / "recap.json", "RECAP export metadata")
        if (
            recap.get("schema") != RECAP_EXPORT_SCHEMA
            or recap.get("status") != "complete"
            or recap.get("plan_sha256") != plan_sha256
        ):
            raise RuntimeError(f"RECAP export provenance differs from the plan: {root}")
        role = recap.get("dataset")
        if role not in coverage or "value" not in recap.get("uses", []):
            raise ValueError(f"returns require expert or HIL rollout value data: {root}")
        episodes = recap.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise RuntimeError(f"RECAP export episode provenance is missing: {root}")
        lerobot_episodes = _read_json_lines(root / "meta" / "episodes.jsonl")
        expected_indices = list(range(len(episodes)))
        if (
            [item.get("episode_index") for item in episodes] != expected_indices
            or [item.get("episode_index") for item in lerobot_episodes] != expected_indices
            or len(episodes) != len(lerobot_episodes)
        ):
            raise RuntimeError(f"RECAP export episode indices differ from LeRobot: {root}")
        for item, lerobot_item in zip(episodes, lerobot_episodes, strict=True):
            source_id = item.get("source_episode_id")
            outcome = item.get("outcome")
            action_count = item.get("action_count")
            key = (role, source_id)
            if (
                not isinstance(source_id, str)
                or outcome not in {"success", "failure"}
                or isinstance(action_count, bool)
                or not isinstance(action_count, int)
                or action_count < 1
            ):
                raise RuntimeError(f"invalid RECAP episode provenance: {root}: {item!r}")
            if action_count != lerobot_item.get("length"):
                raise RuntimeError(f"RECAP and LeRobot episode lengths differ: {root}:{source_id}")
            if key in seen_episode_keys:
                raise RuntimeError(f"RECAP episode appears in multiple exports: {key}")
            seen_episode_keys.add(key)
            coverage[role].add(source_id)
        records.append((root, recap, episodes))

    represented_roles = {recap["dataset"] for _, recap, _ in records}
    for role in represented_roles:
        expected = {
            str(item["source_episode_id"])
            for item in plan["datasets"][role]["episodes"]
        }
        if coverage[role] != expected:
            raise RuntimeError(
                f"RECAP {role} exports do not cover the frozen plan: "
                f"missing={len(expected - coverage[role])}, extra={len(coverage[role] - expected)}"
            )

    failure_reward = -(max_episode_length * float(c_fail_coef))
    global_return_min = -max_episode_length + failure_reward
    destinations = [root / "meta" / f"returns_{tag}.parquet" for root, _, _ in records]
    manifests = [root / "meta" / f"returns_{tag}.json" for root, _, _ in records]
    existing = [target for target in (*destinations, *manifests) if target.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"RECAP returns already exist: {existing[0]}")

    summaries = []
    for (root, recap, episodes), destination, manifest_path in zip(
        records, destinations, manifests, strict=True
    ):
        episode_columns = []
        frame_columns = []
        return_columns = []
        reward_columns = []
        prompt_columns = []
        for item in episodes:
            episode_index = int(item["episode_index"])
            length = int(item["action_count"])
            # Exported row t contains s_t, a_t; s_N is an unexported boundary.
            # Every actual action costs -1, then V(s_N) is 0 or -C_fail.
            # Keep N rows: G_t = -(N-t) + V(s_N), without inventing an action
            # or assigning the terminal value to the penultimate observation.
            remaining = np.arange(length, 0, -1, dtype=np.float32)
            returns = -remaining
            rewards = np.full(length, -1.0, dtype=np.float32)
            if item["outcome"] == "failure":
                returns += np.float32(failure_reward)
                # Fold the unexported boundary into the final transition so
                # reverse cumulative reward and n-step terminal targets agree.
                rewards[-1] += np.float32(failure_reward)
            episode_columns.append(np.full(length, episode_index, dtype=np.int64))
            frame_columns.append(np.arange(length, dtype=np.int64))
            return_columns.append(returns)
            reward_columns.append(rewards)
            prompt_columns.extend([RECAP_PROMPT] * length)
        table = pa.table(
            {
                "episode_index": pa.array(np.concatenate(episode_columns), type=pa.int64()),
                "frame_index": pa.array(np.concatenate(frame_columns), type=pa.int64()),
                "return": pa.array(np.concatenate(return_columns), type=pa.float32()),
                "reward": pa.array(np.concatenate(reward_columns), type=pa.float32()),
                "prompt": pa.array(prompt_columns, type=pa.string()),
            }
        )
        manifest = {
            "schema": RECAP_RETURNS_SCHEMA,
            "status": "complete",
            "plan": str(path),
            "plan_sha256": plan_sha256,
            "dataset": recap["dataset"],
            "tag": tag,
            "gamma": float(gamma),
            "c_fail_coef": float(c_fail_coef),
            "max_episode_length": max_episode_length,
            "observed_max_episode_length": observed_max_episode_length,
            "failure_horizon_source": (
                "explicit" if failure_horizon_steps is not None else "observed_max_episode_length"
            ),
            "failure_reward": failure_reward,
            "observation_alignment": "pre_action_state",
            "reward_alignment": "transition_cost_including_terminal_boundary",
            "action_reward": -1.0,
            "terminal_values": {"success": 0.0, "failure": failure_reward},
            "terminal_observation_exported": False,
            "normalization": {
                "mode": "divide_by_absolute_global_return_min",
                "return_min": global_return_min,
                "return_max": 0.0,
                "range": [-1.0, 0.0],
            },
            "episodes": len(episodes),
            "frames": table.num_rows,
            "outcomes": _outcome_counts(episodes),
        }
        _atomic_parquet(destination, table, pq)
        manifest["sidecar_sha256"] = _sha256(destination)
        _atomic_json(manifest_path, manifest)
        summaries.append(
            {
                "dataset_root": str(root),
                "dataset": recap["dataset"],
                "sidecar": str(destination),
                "manifest": str(manifest_path),
                "episodes": len(episodes),
                "frames": table.num_rows,
                "outcomes": manifest["outcomes"],
            }
        )
    return {
        "compatible": True,
        "schema": RECAP_RETURNS_SCHEMA,
        "tag": tag,
        "gamma": float(gamma),
        "max_episode_length": max_episode_length,
        "observed_max_episode_length": observed_max_episode_length,
        "failure_reward": failure_reward,
        "return_min": global_return_min,
        "return_max": 0.0,
        "exports": summaries,
    }


def _load_recap_plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    plan = _read_object(source, "RECAP plan")
    if plan.get("schema") != RECAP_PLAN_SCHEMA or plan.get("status") != "ready":
        raise RuntimeError(f"RECAP plan is not ready: {source}")
    task = plan.get("task")
    if task != {"task_name": "shirt_flatten", "prompt": RECAP_PROMPT}:
        raise RuntimeError(f"RECAP task identity changed: {source}")
    datasets = plan.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(RECAP_DATASETS):
        raise RuntimeError(f"RECAP datasets are incomplete: {source}")
    return source, plan


def _recap_export_episodes(
    plan: dict[str, Any],
    dataset: str,
    *,
    shard_index: int | None,
    shard_count: int | None,
    pair_member: str = "all",
) -> tuple[Path, list[dict[str, Any]]]:
    if dataset not in RECAP_DATASETS:
        raise ValueError(f"dataset must be one of {RECAP_DATASETS}, got {dataset!r}")
    sources = plan.get("sources")
    if not isinstance(sources, dict) or not isinstance(sources.get(dataset), dict):
        raise RuntimeError(f"RECAP source is missing: {dataset}")
    root_value = sources[dataset].get("root")
    if not isinstance(root_value, str) or not root_value:
        raise RuntimeError(f"RECAP source root is missing: {dataset}")
    source_root = Path(root_value).expanduser().resolve()
    raw = plan["datasets"][dataset].get("episodes")
    if not isinstance(raw, list) or not raw or any(not isinstance(item, dict) for item in raw):
        raise RuntimeError(f"RECAP dataset is empty or invalid: {dataset}")
    episodes = [dict(item) for item in raw]
    if pair_member not in {"all", "original", "mirror"}:
        raise ValueError("pair_member must be one of: all, original, mirror")
    if pair_member != "all":
        expected = pair_member == "mirror"
        if any("is_mirror" not in item for item in episodes):
            raise ValueError("pair_member filtering requires a mirrored RECAP plan")
        episodes = [item for item in episodes if item["is_mirror"] is expected]
        if not episodes:
            raise RuntimeError(f"RECAP {dataset} has no {pair_member} episodes")
    if shard_index is None and shard_count is None:
        return source_root, episodes
    if shard_index is None or shard_count is None:
        raise ValueError("shard_index and shard_count must be provided together")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    from clothdojo.data.rollout import _balanced_group_shards

    selected_assets = _balanced_group_shards(
        [(str(item["asset_id"]), int(item["action_count"])) for item in episodes],
        shard_count,
    )[shard_index]
    selected = [item for item in episodes if item.get("asset_id") in selected_assets]
    if not selected:
        raise ValueError(f"RECAP shard {shard_index}/{shard_count} is empty")
    return source_root, selected


def _validate_recap_sources(
    plan: dict[str, Any],
    dataset: str,
    source_root: Path,
    episodes: list[dict[str, Any]],
) -> None:
    sources = plan["sources"]
    rollout_source = sources["hil_rollout"]
    rollout_root = Path(
        rollout_source.get("origin_root", rollout_source["root"])
    ).expanduser().resolve()
    if _sha256(rollout_root / "meta.json") != rollout_source.get("meta_sha256"):
        raise RuntimeError("HIL rollout metadata changed after RECAP planning")
    source_scores = rollout_source.get("source_scores")
    if source_scores is None:
        if _sha256(rollout_root / "hil_review.json") != rollout_source.get("review_sha256"):
            raise RuntimeError("HIL review changed after RECAP planning")
    else:
        if not isinstance(source_scores, dict) or not source_scores:
            raise RuntimeError("RECAP rollout score provenance is invalid")
        for source_name, raw in source_scores.items():
            if not isinstance(raw, dict):
                raise RuntimeError(f"RECAP rollout score provenance is invalid: {source_name}")
            score_path = Path(str(raw.get("path", ""))).expanduser().resolve()
            if not score_path.is_file() or _sha256(score_path) != raw.get("sha256"):
                raise RuntimeError(f"RECAP rollout scores changed after planning: {source_name}")
    for item in episodes:
        relative = item.get("path")
        if not isinstance(relative, str) or Path(relative).parent != Path("trajectories"):
            raise RuntimeError(f"RECAP episode path must be flat: {relative!r}")
        manifest_path = source_root / relative / "manifest.json"
        if _sha256(manifest_path) != item.get("manifest_sha256"):
            raise RuntimeError(
                f"RECAP source manifest changed after planning: {item.get('source_episode_id')}"
            )
        if dataset == "expert":
            review = item.get("review")
            if not isinstance(review, dict):
                raise RuntimeError("expert RECAP review provenance is missing")
            annotation = (
                source_root
                / "trajectories"
                / "annotations"
                / f"{item['source_episode_id']}.segments.json"
            )
            expected = review.get("annotation_sha256")
            if expected is None and annotation.exists():
                raise RuntimeError(
                    f"expert annotation appeared after RECAP planning: {item['source_episode_id']}"
                )
            if expected is not None and _sha256(annotation) != expected:
                raise RuntimeError(
                    f"expert annotation changed after RECAP planning: {item['source_episode_id']}"
                )


def _outcome_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in episodes:
        outcome = item.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            raise RuntimeError("RECAP episode outcome is missing")
        result[outcome] = result.get(outcome, 0) + 1
    return result


def _expert_episodes(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectories_root = root / "trajectories"
    if not trajectories_root.is_dir():
        raise NotADirectoryError(f"expert trajectory directory is missing: {trajectories_root}")
    manifests = sorted(
        path
        for path in trajectories_root.glob("*/manifest.json")
        if path.parent.name != "annotations"
    )
    if not manifests:
        raise RuntimeError(f"no expert trajectories found: {trajectories_root}")

    episodes: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    known_ids = {path.parent.name for path in manifests}
    annotations_root = trajectories_root / "annotations"
    annotation_ids = {
        path.name.removesuffix(".segments.json")
        for path in annotations_root.glob("*.segments.json")
    }
    orphans = sorted(annotation_ids - known_ids)
    if orphans:
        raise RuntimeError(f"expert annotations do not match trajectories: {orphans}")

    for manifest_path in manifests:
        episode_id = manifest_path.parent.name
        manifest = _read_object(manifest_path, "expert manifest")
        if manifest.get("status") != "complete":
            excluded.append({"source_episode_id": episode_id, "reason": "incomplete"})
            continue
        action_count = _validated_action_count(manifest, episode_id)
        metadata = manifest.get("metadata")
        asset_id = metadata.get("asset_id") if isinstance(metadata, dict) else None
        if not isinstance(asset_id, str) or not asset_id:
            raise RuntimeError(f"expert asset_id is missing: {episode_id}")
        annotation_path = annotations_root / f"{episode_id}.segments.json"
        review = _expert_review(annotation_path, action_count)
        episodes.append(
            {
                "source_episode_id": episode_id,
                "path": f"trajectories/{episode_id}",
                "asset_id": asset_id,
                "action_start": 0,
                "action_end_exclusive": action_count,
                "action_count": action_count,
                "outcome": "success",
                "endpoint_state_index": action_count,
                "manifest_sha256": _sha256(manifest_path),
                "review": review,
            }
        )
    if not episodes:
        raise RuntimeError("no complete expert trajectories remain")
    return episodes, excluded


def _expert_review(path: Path, action_count: int) -> dict[str, Any]:
    if not path.is_file():
        return {
            "annotation_present": False,
            "review_status": "unreviewed",
            "annotation_sha256": None,
            "segments": [],
            "bad_ranges": [],
        }
    value = _read_object(path, "expert annotation")
    if value.get("schema") != ANNOTATION_SCHEMA:
        raise RuntimeError(f"annotation schema must be {ANNOTATION_SCHEMA}: {path}")
    review_status = value.get("review_status")
    if review_status is not None and not isinstance(review_status, str):
        raise RuntimeError(f"annotation review_status must be a string: {path}")
    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list):
        raise RuntimeError(f"annotation segments must be a list: {path}")
    segments = []
    bad_ranges = []
    for raw in raw_segments:
        if not isinstance(raw, dict) or raw.get("label") not in KNOWN_LABELS:
            raise RuntimeError(f"invalid annotation segment: {path}: {raw!r}")
        start = _integer(raw.get("start_frame"), "start_frame")
        end = _integer(raw.get("end_frame"), "end_frame")
        if start < 0 or end < start or end >= action_count:
            raise RuntimeError(f"annotation segment is outside the trajectory: {path}: {raw!r}")
        normalized = {"start_frame": start, "end_frame": end, "label": raw["label"]}
        segments.append(normalized)
        if raw["label"] == "bad":
            bad_ranges.append(
                {"action_start": start, "action_end_exclusive": end + 1}
            )
    return {
        "annotation_present": True,
        "review_status": review_status,
        "annotation_sha256": _sha256(path),
        "segments": segments,
        "bad_ranges": bad_ranges,
    }


def _rollout_episodes(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[tuple[str, int]]]:
    meta_path = root / "meta.json"
    review_path = root / "hil_review.json"
    meta = _read_object(meta_path, "HIL rollout metadata")
    if meta.get("schema") != ROLLOUT_COLLECTION_SCHEMA or meta.get("status") != "complete":
        raise RuntimeError(f"HIL rollout is not a complete canonical collection: {root}")
    review = _read_object(review_path, "HIL review")
    if review.get("schema") != HIL_REVIEW_SCHEMA:
        raise RuntimeError(f"HIL review schema must be {HIL_REVIEW_SCHEMA}: {review_path}")
    marks = review.get("marks")
    breakpoints = review.get("breakpoints")
    if not isinstance(marks, dict) or not isinstance(breakpoints, dict):
        raise RuntimeError(f"HIL review marks and breakpoints must be objects: {review_path}")

    raw_trajectories = meta.get("trajectories")
    if not isinstance(raw_trajectories, list) or not raw_trajectories:
        raise RuntimeError(f"HIL rollout trajectory index is empty: {meta_path}")
    indexed: dict[str, dict[str, Any]] = {}
    for item in raw_trajectories:
        if not isinstance(item, dict):
            raise RuntimeError(f"invalid HIL rollout trajectory entry: {item!r}")
        relative = item.get("path")
        if not isinstance(relative, str) or Path(relative).parent != Path("trajectories"):
            raise RuntimeError(f"invalid HIL rollout trajectory path: {relative!r}")
        episode_id = Path(relative).name
        if episode_id in indexed:
            raise RuntimeError(f"duplicate HIL rollout trajectory: {episode_id}")
        indexed[episode_id] = item
    unknown = sorted((set(marks) | set(breakpoints)) - set(indexed))
    if unknown:
        raise RuntimeError(f"HIL review refers to unknown trajectories: {unknown}")

    episodes = []
    excluded = []
    breakpoint_keys: set[tuple[str, int]] = set()
    for episode_id, item in indexed.items():
        manifest_path = root / str(item["path"]) / "manifest.json"
        manifest = _read_object(manifest_path, "HIL rollout manifest")
        if manifest.get("status") != "complete":
            raise RuntimeError(f"indexed HIL rollout is incomplete: {episode_id}")
        action_count = _validated_action_count(manifest, episode_id)
        if item.get("action_count") != action_count:
            raise RuntimeError(f"HIL rollout action count differs from its index: {episode_id}")
        asset_id = item.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            raise RuntimeError(f"HIL rollout asset_id is missing: {episode_id}")
        points = _breakpoints(breakpoints.get(episode_id), action_count, episode_id)
        breakpoint_keys.update((episode_id, point["sample_index"]) for point in points)
        mark = marks.get(episode_id)
        if mark is None and not points:
            excluded.append({"source_episode_id": episode_id, "reason": "unmarked"})
            continue
        if mark is not None:
            if not isinstance(mark, dict):
                raise RuntimeError(f"invalid HIL completion mark: {episode_id}")
            endpoint = _integer(mark.get("sample_index"), "completion sample_index")
            if not 0 <= endpoint <= action_count:
                raise RuntimeError(
                    f"HIL completion endpoint is outside [0,{action_count}]: {episode_id}"
                )
            if endpoint == 0:
                excluded.append(
                    {
                        "source_episode_id": episode_id,
                        "reason": "success_at_initial_state",
                    }
                )
                continue
            outcome = "success"
            action_end = endpoint
        else:
            endpoint = action_count
            outcome = "failure"
            action_end = action_count
        episodes.append(
            {
                "source_episode_id": episode_id,
                "path": str(item["path"]),
                "asset_id": asset_id,
                "action_start": 0,
                "action_end_exclusive": action_end,
                "source_action_count": action_count,
                "action_count": action_end,
                "outcome": outcome,
                "endpoint_state_index": endpoint,
                "completion_mark": mark,
                "intervention_points": points,
                "manifest_sha256": _sha256(manifest_path),
            }
        )
    if not episodes:
        raise RuntimeError("HIL review selects no rollouts")
    return episodes, excluded, breakpoint_keys


def _breakpoints(value: object, action_count: int, episode_id: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(f"invalid HIL breakpoint list: {episode_id}")
    points = []
    seen = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise RuntimeError(f"invalid HIL breakpoint: {episode_id}: {raw!r}")
        sample = _integer(raw.get("sample_index"), "breakpoint sample_index")
        created_at = raw.get("created_at")
        if not 0 <= sample < action_count or not isinstance(created_at, str) or not created_at:
            raise RuntimeError(f"HIL breakpoint is outside its rollout: {episode_id}:{sample}")
        if sample in seen:
            raise RuntimeError(f"duplicate HIL breakpoint: {episode_id}:{sample}")
        seen.add(sample)
        points.append({"sample_index": sample, "created_at": created_at})
    return points


def _correction_episodes(
    root: Path,
    *,
    rollout_root: Path,
    breakpoint_keys: set[tuple[str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectories_root = root / "trajectories"
    if not trajectories_root.is_dir():
        raise NotADirectoryError(f"HIL correction directory is missing: {trajectories_root}")
    rollout_meta_sha256 = _sha256(rollout_root / "meta.json")
    candidates: dict[tuple[str, int], list[dict[str, Any]]] = {}
    excluded: list[dict[str, Any]] = []
    for manifest_path in sorted(trajectories_root.glob("*/manifest.json")):
        correction_id = manifest_path.parent.name
        manifest = _read_object(manifest_path, "HIL correction manifest")
        if manifest.get("status") != "complete":
            excluded.append({"source_episode_id": correction_id, "reason": "incomplete"})
            continue
        action_count = _validated_action_count(manifest, correction_id)
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("source") != "human_intervention":
            raise RuntimeError(f"trajectory is not a HIL correction: {correction_id}")
        source_episode_id = metadata.get("source_episode_id")
        source_sample_index = metadata.get("source_sample_index")
        asset_id = metadata.get("asset_id")
        if not isinstance(source_episode_id, str) or not source_episode_id:
            raise RuntimeError(f"HIL correction source episode is missing: {correction_id}")
        source_sample_index = _integer(source_sample_index, "source_sample_index")
        if not isinstance(asset_id, str) or not asset_id:
            raise RuntimeError(f"HIL correction asset_id is missing: {correction_id}")
        key = (source_episode_id, source_sample_index)
        if key not in breakpoint_keys:
            excluded.append(
                {
                    "source_episode_id": correction_id,
                    "reason": "source_breakpoint_not_in_current_review",
                    "source_rollout_episode_id": source_episode_id,
                    "source_sample_index": source_sample_index,
                }
            )
            continue
        expected_root = metadata.get("source_rollout_root")
        if (
            not isinstance(expected_root, str)
            or Path(expected_root).expanduser().resolve() != rollout_root
        ):
            raise RuntimeError(f"HIL correction rollout root changed: {correction_id}")
        if metadata.get("source_run_id") != rollout_root.name:
            raise RuntimeError(f"HIL correction run id changed: {correction_id}")
        if metadata.get("source_rollout_meta_sha256") != rollout_meta_sha256:
            raise RuntimeError(f"HIL correction rollout metadata changed: {correction_id}")
        source_manifest = rollout_root / "trajectories" / source_episode_id / "manifest.json"
        if metadata.get("source_manifest_sha256") != _sha256(source_manifest):
            raise RuntimeError(f"HIL correction source manifest changed: {correction_id}")
        candidates.setdefault(key, []).append(
            {
                "source_episode_id": correction_id,
                "path": f"trajectories/{correction_id}",
                "asset_id": asset_id,
                "action_start": 0,
                "action_end_exclusive": action_count,
                "action_count": action_count,
                "source_rollout_episode_id": source_episode_id,
                "source_sample_index": source_sample_index,
                "outcome": "correction",
                "force_positive": True,
                "manifest_sha256": _sha256(manifest_path),
            }
        )

    selected = []
    for key in sorted(candidates):
        values = sorted(candidates[key], key=lambda item: item["source_episode_id"])
        selected.extend(values)
    selected.sort(key=lambda item: item["source_episode_id"])
    excluded.sort(key=lambda item: item["source_episode_id"])
    return selected, excluded


def _validated_action_count(manifest: dict[str, Any], episode_id: str) -> int:
    action_count = _integer(manifest.get("action_count"), "action_count")
    state_count = _integer(manifest.get("state_count"), "state_count")
    if action_count < 1 or state_count != action_count + 1:
        raise RuntimeError(
            f"trajectory must have positive N actions and N+1 states: {episode_id}"
        )
    return action_count


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {context}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be a JSON object: {path}")
    return value


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON lines: {path}: {exc}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise RuntimeError(f"JSON lines must contain objects: {path}")
    return values


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} must be an integer, got {value!r}")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"cannot hash source file: {path}: {exc}") from exc


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, table: Any, parquet_module: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        parquet_module.write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
