"""Guarded expert-replay plus correction fine-tuning from an SFT checkpoint."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.training.openpi_artifacts import (
    _read_jsonl,
    _read_object,
    _sha256,
    inspect_openpi_dataset,
    inspect_openpi_norm_stats,
)


def home_trim_anchor_indices(
    dataset_root: str | Path,
    trim_manifest: str | Path,
    *,
    action_horizon: int = 50,
    minimum_actions: int = 65,
) -> tuple[list[int], dict[str, Any]]:
    """Resolve a physical LeRobot aggregate into the frozen trimmed HIL view."""

    root = Path(dataset_root).expanduser().resolve()
    trim_path = Path(trim_manifest).expanduser().resolve()
    plan = _read_object(trim_path, "home-trim manifest")
    if (
        plan.get("schema") != "clothdojo_home_return_trim_collection"
        or plan.get("format_version") != 1
        or plan.get("status") != "complete"
    ):
        raise ValueError("offline SFT requires a complete home-trim manifest")
    if action_horizon < 1 or minimum_actions < action_horizon:
        raise ValueError("home-trim view requires minimum_actions >= action_horizon")

    original_limits = {}
    for record in plan.get("trajectories", []):
        identity = record.get("trajectory_id")
        source_count = record.get("source_action_count")
        limit = record.get("output_action_count")
        if (
            not isinstance(identity, str)
            or identity.endswith("__mirror")
            or type(source_count) is not int
            or type(limit) is not int
            or not 0 < limit <= source_count
            or identity in original_limits
        ):
            raise ValueError("home-trim trajectory record is invalid")
        if limit >= minimum_actions:
            original_limits[identity] = (source_count, limit)
    if not original_limits:
        raise ValueError("home-trim view has no eligible trajectories")

    episodes = _read_jsonl(root / "meta" / "episodes.jsonl", "LeRobot episodes")
    sidecar = _read_object(root / "meta" / "clothdojo.json", "dataset sidecar")
    provenance = sidecar.get("episodes")
    if not isinstance(provenance, list) or len(provenance) != len(episodes):
        raise ValueError("home-trim source dataset episode provenance is incomplete")
    expected = {
        identity: values
        for original, values in original_limits.items()
        for identity in (original, original + "__mirror")
    }
    starts, offset = [], 0
    for episode in episodes:
        starts.append(offset)
        offset += int(episode["length"])
    anchors, selected, seen = [], [], set()
    for episode, record, start in zip(episodes, provenance, starts, strict=True):
        identity = record.get("source_trajectory")
        if identity not in expected:
            continue
        if identity in seen:
            raise ValueError(f"home-trim dataset repeats trajectory identity: {identity}")
        seen.add(identity)
        source_count, limit = expected[identity]
        if (
            record.get("episode_index") != episode.get("episode_index")
            or record.get("action_start") != 0
            or record.get("source_action_count") != source_count
            or record.get("length") != source_count
            or episode.get("length") != source_count
        ):
            raise ValueError(f"home-trim source extent differs: {identity}")
        anchors.extend(range(start, start + limit - action_horizon + 1))
        selected.append(
            {
                "episode_index": episode["episode_index"],
                "source_trajectory": identity,
                "source_action_count": source_count,
                "output_action_count": limit,
            }
        )
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise ValueError(f"home-trim trajectories are missing from dataset: {missing}")
    return anchors, {
        "trim_manifest": str(trim_path),
        "trim_manifest_sha256": _sha256(trim_path),
        "selected_episodes": len(selected),
        "selected_frames": sum(item["output_action_count"] for item in selected),
        "full_h50_anchors": len(anchors),
        "minimum_actions": minimum_actions,
        "excluded_short_original_mirror_pairs": len(plan["trajectories"])
        - len(original_limits),
        "episodes": selected,
    }


def filter_action_jump_anchors(
    dataset_root: str | Path,
    anchors: list[int],
    *,
    max_arm_action_step_rad: float,
    action_horizon: int = 50,
) -> tuple[list[int], dict[str, Any]]:
    """Drop H-step anchors whose arm targets contain a discontinuous transition."""

    import pyarrow.parquet as pq

    root = Path(dataset_root).expanduser().resolve()
    rows = _read_jsonl(root / "meta" / "episodes.jsonl", "LeRobot episodes")
    total_frames = sum(int(row["length"]) for row in rows)
    if (
        isinstance(max_arm_action_step_rad, bool)
        or not math.isfinite(max_arm_action_step_rad)
        or max_arm_action_step_rad <= 0
    ):
        raise ValueError("max_arm_action_step_rad must be finite and positive")
    values = np.asarray(anchors, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0) or np.any(values >= total_frames):
        raise ValueError("action-jump filter received invalid anchor indices")
    keep = np.ones(len(values), dtype=bool)
    starts, offset = [], 0
    for row in rows:
        starts.append(offset)
        offset += int(row["length"])
    jump_count = 0
    affected_episodes = 0
    for row, start in zip(rows, starts, strict=True):
        episode_stop = start + int(row["length"])
        candidate_left = int(np.searchsorted(values, start, side="left"))
        candidate_right = int(np.searchsorted(values, episode_stop, side="left"))
        if candidate_left == candidate_right:
            continue
        episode = int(row["episode_index"])
        path = root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
        action = np.asarray(
            pq.read_table(path, columns=["action"], use_threads=False)["action"].to_pylist(),
            dtype=np.float32,
        )
        arms = np.concatenate((action[:, :6], action[:, 7:13]), axis=1)
        jumps = np.flatnonzero(
            np.max(np.abs(np.diff(arms, axis=0)), axis=1) > max_arm_action_step_rad
        ) + 1
        contributing_jumps = 0
        for frame in jumps:
            lower = start + max(0, int(frame) - action_horizon + 1)
            upper = start + min(int(frame), len(action) - action_horizon + 1)
            left = int(np.searchsorted(values, lower, side="left"))
            right = int(np.searchsorted(values, upper, side="left"))
            if right <= left:
                continue
            contributing_jumps += 1
            keep[left:right] = False
        jump_count += contributing_jumps
        affected_episodes += int(contributing_jumps > 0)
    filtered = values[keep].tolist()
    if not filtered:
        raise ValueError("action-jump filter removed every training anchor")
    return filtered, {
        "max_arm_action_step_rad": float(max_arm_action_step_rad),
        "action_jump_count": int(jump_count),
        "affected_episodes": int(affected_episodes),
        "input_anchors": len(values),
        "kept_anchors": len(filtered),
        "excluded_anchors": int((~keep).sum()),
    }


def inspect_offline_sft_inputs(cfg: Any) -> dict[str, Any]:
    """Bind an immutable SFT checkpoint, its norm stats, and weighted datasets."""

    from omegaconf import OmegaConf

    from clothdojo.rollout.artifacts import find_training_manifest, validate_sft_norm_binding
    from clothdojo.rollout.checkpoint import rlinf_openpi_weights_sha256
    settings = cfg.clothdojo.offline_sft
    if (
        cfg.clothdojo.get("action_chunk_boundary") != "full_only"
        or cfg.clothdojo.get("jax_reference") is not None
    ):
        raise ValueError("offline SFT requires full_only without a JAX override")
    model = cfg.actor.model
    if (
        model.model_type != "openpi_rlinf"
        or model.openpi.task != "sft"
        or model.num_action_chunks != 50
        or model.action_dim != 14
        or model.openpi.model_action_dim != 32
        or model.openpi.get("use_rlt", False)
        or model.get("is_lora", False)
        or model.get("add_value_head", False)
    ):
        raise ValueError("offline SFT requires the native H50/14D OpenPI_RLinf model")
    if cfg.runner.val_check_interval != -1:
        raise ValueError("embodied offline SFT uses paired rollout evaluation")

    checkpoint = Path(str(model.model_path)).expanduser().resolve()
    expected_weights = str(settings.initialization_weights_sha256)
    if len(expected_weights) != 64 or rlinf_openpi_weights_sha256(checkpoint) != expected_weights:
        raise ValueError("offline SFT initialization weights differ from the frozen hash")
    norm_root = Path(str(model.openpi_data.norm_stats_path)).expanduser().resolve()
    fresh_base = bool(settings.get("fresh_base", False))
    launch = None
    binding = None
    if not fresh_base:
        launch = _read_object(find_training_manifest(checkpoint), "SFT initialization manifest")
        binding = validate_sft_norm_binding(
            checkpoint,
            norm_root,
            openpi_config=str(model.openpi.config_name),
            action_horizon=50,
        )
        old_model = launch["resolved_config"]["actor"]["model"]
        for key in ("pi05", "num_action_chunks", "action_dim", "use_proprio"):
            if model.get(key) != old_model.get(key):
                raise ValueError(f"offline SFT changes initialization model field {key}")
        for key in (
            "model_action_dim",
            "paligemma_variant",
            "action_expert_variant",
            "max_token_len",
            "discrete_state_input",
        ):
            if model.openpi.get(key) != old_model["openpi"].get(key):
                raise ValueError(f"offline SFT changes initialization model field openpi.{key}")
        for key in ("adapt_to_pi", "extra_delta_transform", "default_prompt"):
            if model.openpi_data.get(key) != old_model["openpi_data"].get(key):
                raise ValueError(f"offline SFT changes initialization transform {key}")
        if launch.get("action_chunk_boundary") != "full_only":
            raise ValueError("offline SFT initialization must use full_only normalization")

    entries = OmegaConf.to_container(cfg.data.train_data_paths, resolve=True)
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("offline SFT requires explicit expert and correction sources")
    reports, roles, roots = [], set(), set()
    prompt = str(model.openpi_data.default_prompt)
    for entry in entries:
        required = {"dataset_path", "role", "weight"}
        allowed = required | {
            "home_trim_manifest", "home_trim_manifest_sha256", "max_arm_action_step_rad"
        }
        if (
            not isinstance(entry, dict)
            or not required.issubset(entry)
            or not set(entry) <= allowed
        ):
            raise ValueError("offline SFT source fields are invalid")
        role, weight = entry["role"], entry["weight"]
        root = Path(str(entry["dataset_path"])).expanduser().resolve()
        if (
            role not in {"expert", "correction"}
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
            or root in roots
        ):
            raise ValueError("offline SFT source role, weight, or path is invalid")
        roots.add(root)
        roles.add(role)
        report = inspect_openpi_dataset(root, expected_task=prompt)
        trim_path = entry.get("home_trim_manifest")
        trim_sha = entry.get("home_trim_manifest_sha256")
        if trim_path is None:
            if trim_sha is not None:
                raise ValueError("home-trim hash requires its manifest path")
            rows = _read_jsonl(root / "meta" / "episodes.jsonl", f"{role} episodes")
            anchors = []
            offset = 0
            for row in rows:
                length = int(row["length"])
                anchors.extend(range(offset, offset + max(0, length - 49)))
                offset += length
            report["full_h50_anchors"] = len(anchors)
        else:
            if (
                role != "correction"
                or not isinstance(trim_sha, str)
                or len(trim_sha) != 64
            ):
                raise ValueError("only correction sources may define a frozen home-trim view")
            anchors, view = home_trim_anchor_indices(root, trim_path)
            if view["trim_manifest_sha256"] != trim_sha:
                raise ValueError("home-trim manifest differs from its frozen hash")
            report["physical_episodes"] = report["episodes"]
            report["physical_frames"] = report["frames"]
            report["episodes"] = view["selected_episodes"]
            report["frames"] = view["selected_frames"]
            report["full_h50_anchors"] = view["full_h50_anchors"]
            report["home_trim_view"] = view
        threshold = entry.get("max_arm_action_step_rad")
        if threshold is not None:
            anchors, jump_filter = filter_action_jump_anchors(
                root, anchors, max_arm_action_step_rad=float(threshold)
            )
            report["unfiltered_full_h50_anchors"] = report["full_h50_anchors"]
            report["full_h50_anchors"] = len(anchors)
            report["action_jump_filter"] = jump_filter
        if role == "expert" and launch is not None:
            for key in ("info_sha256", "sidecar_sha256", "episodes_sha256", "tasks_sha256"):
                if report[key] != launch["dataset"][key]:
                    raise ValueError(f"expert replay differs from initialization: {key}")
        reports.append({**report, "role": role, "weight": float(weight)})
    if roles != {"expert", "correction"} or not math.isclose(
        sum(report["weight"] for report in reports), 1.0, abs_tol=1.0e-6
    ):
        raise ValueError("offline SFT weights must sum to one and include both roles")

    correction = next(report for report in reports if report["role"] == "correction")
    expected_correction = str(settings.correction_sidecar_sha256)
    if len(expected_correction) != 64 or correction["sidecar_sha256"] != expected_correction:
        raise ValueError("offline SFT correction dataset differs from the frozen sidecar")
    expert = next(report for report in reports if report["role"] == "expert")
    if fresh_base:
        data_kwargs = {
            key: model.openpi_data.get(key)
            for key in ("adapt_to_pi", "extra_delta_transform", "default_prompt")
        }
        norm = inspect_openpi_norm_stats(
            norm_root,
            dataset=expert,
            config_name=str(model.openpi.config_name),
            data_kwargs=data_kwargs,
            action_horizon=50,
            action_chunk_boundary="full_only",
        )
    else:
        norm = {
            **launch["norm_stats"],
            "root": str(norm_root),
            "stats_path": str(norm_root / "norm_stats.json"),
            "provenance_path": str(norm_root / "clothdojo.json"),
            "binding": binding,
        }
    return {
        "dataset": {
            "sources": reports,
            "frames": sum(report["frames"] for report in reports),
            "episodes": sum(report["episodes"] for report in reports),
        },
        "norm_stats": norm,
        "initialization_weights_sha256": expected_weights,
        "correction_sidecar_sha256": expected_correction,
        "balance_dataset_weights": False,
        "fresh_base": fresh_base,
    }
