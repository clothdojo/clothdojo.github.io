"""Data-contract checks for RLinf's native RECAP value trainer."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np

from clothdojo.data.recap import (
    RECAP_EXPORT_SCHEMA,
    RECAP_PREFIX60_RETURNS_TAG,
    RECAP_PROMPT,
    RECAP_RETURNS_SCHEMA,
    RECAP_RETURNS_TAG,
    RECAP_VALUE_SPLIT_SCHEMA,
)
from clothdojo.training.worker_environment import worker_environment_from_config

RECAP_IMAGE_KEY_MAP = {
    "base_0_rgb": "observation.images.cam_high",
    "base_1_rgb": "observation.images.cam_low",
    "left_wrist_0_rgb": "observation.images.cam_left_wrist",
    "right_wrist_0_rgb": "observation.images.cam_right_wrist",
}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_recap_value_checkpoint(model_path: str | Path | None) -> Path | None:
    """Resolve one full-weight checkpoint without restoring optimizer state."""

    if model_path is None:
        return None
    source = Path(model_path).expanduser().resolve()
    candidates = [source] if source.is_file() else [
        source / "model_state_dict" / "full_weights.pt",
        source / "actor" / "model_state_dict" / "full_weights.pt",
    ]
    matches = [candidate for candidate in candidates if candidate.is_file()]
    if len(matches) != 1 or matches[0].name != "full_weights.pt":
        raise FileNotFoundError(
            "RECAP Stage 2 requires exactly one full_weights.pt below model_path: "
            f"{source}"
        )
    return matches[0]


def inspect_recap_value_resume(
    resume_dir: str | Path,
    *,
    current_config: dict[str, Any],
    world_size: int,
    steps_per_epoch: int,
    max_steps: int,
) -> dict[str, Any]:
    """Validate an epoch-boundary native RLinf value-training resume."""

    source = Path(resume_dir).expanduser().resolve()
    if not source.is_dir() or source.parent.name != "checkpoints":
        raise FileNotFoundError(f"invalid RECAP resume checkpoint directory: {source}")
    try:
        completed_steps = int(source.name.removeprefix("global_step_"))
    except ValueError as exc:
        raise ValueError(f"RECAP resume path has no global step: {source}") from exc
    if source.name != f"global_step_{completed_steps}" or completed_steps < 1:
        raise ValueError(f"RECAP resume path has no global step: {source}")
    if completed_steps >= max_steps:
        raise ValueError(
            f"RECAP resume must extend the update budget: {completed_steps} >= {max_steps}"
        )
    if steps_per_epoch < 1 or completed_steps % steps_per_epoch:
        raise ValueError(
            "RECAP native worker does not checkpoint the DataLoader cursor; resume is only "
            f"safe at an epoch boundary, got step {completed_steps} of {steps_per_epoch}"
        )

    actor = source / "actor"
    dcp = actor / "dcp_checkpoint"
    metadata_path = dcp / ".metadata"
    full_weights = actor / "model_state_dict" / "full_weights.pt"
    if not metadata_path.is_file() or not full_weights.is_file():
        raise FileNotFoundError(f"RECAP resume checkpoint is incomplete: {source}")
    shards = sorted(dcp.glob("__*_0.distcp"))
    expected_shards = {f"__{rank}_0.distcp" for rank in range(world_size)}
    if {path.name for path in shards} != expected_shards or any(
        path.stat().st_size == 0 for path in shards
    ):
        raise RuntimeError(
            "RECAP resume requires one non-empty DCP shard per current rank: "
            f"found={len(shards)}, expected={world_size}"
        )

    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(str(dcp)).read_metadata()
    keys = set(metadata.state_dict_metadata)
    required_prefixes = {
        "model": "fsdp_checkpoint.model.",
        "optimizer": "fsdp_checkpoint.optimizers.state.",
        "scheduler": "fsdp_checkpoint.lr_schedulers.0.",
        "rng": "fsdp_checkpoint.rng.",
    }
    missing = [
        name
        for name, prefix in required_prefixes.items()
        if not any(key.startswith(prefix) for key in keys)
    ]
    if missing:
        raise RuntimeError(f"RECAP resume DCP is missing training state: {missing}")

    launch_path = source.parents[2] / "clothdojo_launch.json"
    launch = _read_object(launch_path, "RECAP source launch manifest")
    previous_config = launch.get("resolved_config")
    if not isinstance(previous_config, dict):
        raise RuntimeError(f"RECAP source launch has no resolved config: {launch_path}")

    def training_identity(config: dict[str, Any]) -> dict[str, Any]:
        identity = deepcopy(config)
        runner = identity.get("runner", {})
        identity["runner"] = {"task_type": runner.get("task_type")}
        optim = identity.get("actor", {}).get("optim", {})
        optim.pop("total_training_steps", None)
        return identity

    if training_identity(previous_config) != training_identity(current_config):
        raise ValueError(
            "RECAP resume changed the model, data, batching, optimizer, or numerical contract"
        )
    scheduler = str(current_config.get("actor", {}).get("optim", {}).get("lr_scheduler"))
    if scheduler != "constant":
        raise ValueError(
            "RECAP budget extension is only exact for the constant-with-warmup scheduler"
        )

    return {
        "path": str(source),
        "completed_steps": completed_steps,
        "resume_epoch": completed_steps // steps_per_epoch,
        "world_size": world_size,
        "dcp_shards": len(shards),
        "dcp_metadata_bytes": metadata_path.stat().st_size,
        "full_weights_bytes": full_weights.stat().st_size,
        "state": sorted(required_prefixes),
        "source_launch": str(launch_path),
        "source_runtime_sha256": launch.get("runtime_source_sha256"),
        "load": "native_rlinf_model_optimizer_scheduler_rng",
    }


def plan_recap_training_budget(
    *, train_samples: int, global_batch_size: int, max_epochs: int, max_steps: int
) -> dict[str, int | float]:
    """Resolve a bounded multi-epoch budget for the deterministic value loader."""

    if train_samples < global_batch_size or global_batch_size < 1:
        raise ValueError("RECAP value training split is smaller than one global batch")
    steps_per_epoch = train_samples // global_batch_size
    if max_epochs < 1:
        raise ValueError("RECAP max_epochs must be positive")
    epoch_step_budget = steps_per_epoch * max_epochs
    if not 1 <= max_steps <= epoch_step_budget:
        raise ValueError(
            "RECAP max_steps must fit within max_epochs: "
            f"{max_steps} not in [1, {epoch_step_budget}]"
        )
    return {
        "steps_per_epoch": steps_per_epoch,
        "max_epochs": max_epochs,
        "effective_steps": max_steps,
        "effective_epochs": max_steps / steps_per_epoch,
        "consumed_samples": max_steps * global_batch_size,
        "optimizer_samples_per_full_epoch": steps_per_epoch * global_batch_size,
        "dropped_samples_per_full_epoch": train_samples - steps_per_epoch * global_batch_size,
    }


def resolve_recap_value_selection(
    dataset_root: str | Path,
    split_path: str | Path,
    split: Literal["train", "val"],
    frame_stride: int = 1,
) -> dict[str, Any]:
    """Resolve one exported shard against the frozen value split."""

    if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError("RECAP value frame_stride must be a positive integer")

    root = Path(dataset_root).expanduser().resolve()
    split_source = Path(split_path).expanduser().resolve()
    recap = _read_object(root / "meta" / "recap.json", "RECAP export metadata")
    partition = _read_object(split_source, "RECAP value split")
    if recap.get("schema") != RECAP_EXPORT_SCHEMA or recap.get("status") != "complete":
        raise RuntimeError(f"RECAP export is not complete: {root}")
    if (
        partition.get("schema") != RECAP_VALUE_SPLIT_SCHEMA
        or partition.get("status") != "ready"
    ):
        raise RuntimeError(f"RECAP value split is not ready: {split_source}")
    if recap.get("plan_sha256") != partition.get("plan_sha256"):
        raise RuntimeError(f"RECAP export and value split use different plans: {root}")
    recorded_plan = Path(str(partition.get("plan", ""))).expanduser()
    if not recorded_plan.is_absolute():
        recorded_plan = split_source.parent / recorded_plan
    recorded_plan = recorded_plan.resolve()
    relocated_plans = (
        (split_source.parent / recorded_plan.name).resolve(),
        (split_source.parent / "plan.json").resolve(),
    )
    expected_plan_sha256 = partition.get("plan_sha256")
    plan_path = next(
        (
            candidate
            for candidate in dict.fromkeys((recorded_plan, *relocated_plans))
            if candidate.is_file() and _sha256(candidate) == expected_plan_sha256
        ),
        None,
    )
    if plan_path is None:
        raise RuntimeError(
            "RECAP value split plan changed or is missing: "
            f"recorded={recorded_plan}, relocated={relocated_plans}"
        )

    role = recap.get("dataset")
    if role not in {"expert", "hil_rollout"} or "value" not in recap.get("uses", []):
        raise ValueError(f"RECAP value data must be expert or HIL rollout: {root}")
    role_partition = partition.get("datasets", {}).get(role)
    if not isinstance(role_partition, dict):
        raise RuntimeError(f"RECAP value split is missing dataset role {role!r}")
    key = "train_source_episode_ids" if split == "train" else "eval_source_episode_ids"
    selected_values = role_partition.get(key)
    train_values = role_partition.get("train_source_episode_ids")
    eval_values = role_partition.get("eval_source_episode_ids")
    if not all(isinstance(value, list) for value in (selected_values, train_values, eval_values)):
        raise RuntimeError(f"RECAP value split has invalid episode lists for {role!r}")
    train_ids = {str(value) for value in train_values}
    eval_ids = {str(value) for value in eval_values}
    if train_ids & eval_ids:
        raise RuntimeError(f"RECAP value split overlaps train and eval for {role!r}")
    selected_ids = {str(value) for value in selected_values}

    episodes = recap.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError(f"RECAP export has no episode provenance: {root}")
    local_ids: list[str] = []
    frame_count = 0
    selected_frame_count = 0
    selected_sample_count = 0
    selected_episode_indices: list[int] = []
    for expected_index, episode in enumerate(episodes):
        if not isinstance(episode, dict) or episode.get("episode_index") != expected_index:
            raise RuntimeError(f"RECAP export episode indices are not contiguous: {root}")
        source_id = episode.get("source_episode_id")
        length = episode.get("action_count")
        if (
            not isinstance(source_id, str)
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length < 1
        ):
            raise RuntimeError(f"RECAP export has invalid episode provenance: {root}")
        local_ids.append(source_id)
        frame_count += length
        if source_id in selected_ids:
            selected_episode_indices.append(expected_index)
            selected_frame_count += length
            selected_sample_count += (length + frame_stride - 1) // frame_stride
    if len(local_ids) != len(set(local_ids)):
        raise RuntimeError(f"RECAP export contains duplicate source episodes: {root}")
    unknown = set(local_ids) - (train_ids | eval_ids)
    if unknown:
        raise RuntimeError(f"RECAP export contains episodes outside the frozen split: {root}")
    return {
        "dataset_root": str(root),
        "dataset": role,
        "split": split,
        "episodes": len(episodes),
        "frames": frame_count,
        "selected_episode_indices": selected_episode_indices,
        "selected_source_episode_ids": [local_ids[index] for index in selected_episode_indices],
        "selected_episodes": len(selected_episode_indices),
        "selected_frames": selected_frame_count,
        "selected_samples": selected_sample_count,
        "frame_stride": frame_stride,
    }


def inspect_recap_value_training(cfg: Any) -> dict[str, Any]:
    """Validate the frozen data, return targets, model, and GPU contract."""

    from omegaconf import OmegaConf
    from rlinf.data.datasets.recap.value_dataset import ValueDataset
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
        DEFAULT_VALUE_MODULE_PARAMETER_PATTERNS,
    )
    from rlinf.models.embodiment.value_model.recap.checkpoint_utils import (
        has_tokenizer_files,
    )

    required_dataset_parameters = {"episodes", "frame_stride", "image_key_map"}
    from inspect import signature

    missing_parameters = required_dataset_parameters - set(
        signature(ValueDataset.__init__).parameters
    )
    if missing_parameters or tuple(DEFAULT_VALUE_MODULE_PARAMETER_PATTERNS) != (
        "value_head",
        "model.value_head",
    ):
        raise RuntimeError(
            "the pinned RLinf ReCap extension patch is missing; "
            "run scripts/setup-environment.sh"
        )

    output = Path(str(cfg.runner.logger.log_path)).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"RECAP value training requires a new output: {output}")
    resume_dir = cfg.runner.get("resume_dir")
    for key, expected in {
        "critic_expert_variant": "gemma_100m", "num_bins": 201,
        "is_lora": False,
        "v_min": -1.0, "v_max": 0.0, "max_token_len": 128,
        "value_dropout": 0.0,
    }.items():
        if cfg.actor.model.get(key) != expected:
            raise ValueError(f"RECAP model contract requires {key}={expected}")
    stage = str(cfg.actor.model.get("training_stage", "full_finetune"))
    frozen_backbone = bool(cfg.actor.model.get("freeze_vision_encoder"))
    if frozen_backbone != bool(cfg.actor.model.get("freeze_vlm")):
        raise ValueError("RECAP staged training must freeze SigLIP and Gemma together")
    expected_frozen = {
        "value_modules": True,
        "full_finetune": False,
        "joint_finetune": False,
    }.get(stage)
    if expected_frozen is None or frozen_backbone != expected_frozen:
        raise ValueError(
            "RECAP training_stage must match the backbone freeze flags: "
            f"stage={stage!r}, frozen={frozen_backbone}"
        )
    checkpoint = resolve_recap_value_checkpoint(cfg.actor.model.get("model_path"))
    expects_checkpoint = stage == "joint_finetune"
    if (checkpoint is not None) != expects_checkpoint:
        raise ValueError(
            "RECAP joint_finetune must start from a full-weight Stage-1 checkpoint; "
            "other stages must start from the named pretrained backbones"
        )
    if cfg.actor.model.get("stop_gradient_to_vlm", False):
        raise ValueError("RECAP must backpropagate into both backbones")
    if str(cfg.actor.fsdp_config.sharding_strategy) != "full_shard":
        raise ValueError("RECAP requires FSDP full_shard")
    if not bool(cfg.data.normalize_to_minus_one_zero) or not bool(cfg.data.do_augment):
        raise ValueError("RECAP requires global normalization and training image augmentation")
    nodes = int(cfg.cluster.num_nodes)
    actor_placement = str(cfg.cluster.component_placement.actor)
    # RLINF interprets placement ranks globally over the selected cluster.
    devices_per_node = 8
    world_size = nodes * devices_per_node
    if nodes not in {1, 2} or actor_placement != "all":
        raise ValueError(
            "RECAP value training requires one or two eight-GPU nodes with actor placement "
            "'all', "
            f"got nodes={nodes}, actor={actor_placement!r}"
        )
    global_batch = int(cfg.actor.global_batch_size)
    micro_batch = int(cfg.actor.micro_batch_size)
    denominator = micro_batch * world_size
    if micro_batch < 1 or global_batch % denominator:
        raise ValueError("RECAP global batch must divide evenly over every GPU micro-batch")
    accumulation = global_batch // denominator
    if accumulation != 1:
        raise ValueError(f"RECAP training must not accumulate gradients, got {accumulation}")
    if bool(cfg.actor.fsdp_config.enable_gradient_accumulation):
        raise ValueError("RECAP training must disable FSDP gradient accumulation")
    if str(cfg.actor.model.model_type) != "recap_value_model":
        raise ValueError("RECAP value training requires model_type=recap_value_model")
    if str(cfg.actor.model.precision) != "fp32":
        raise ValueError("RECAP value training uses FP32 master weights")
    numerical = {
        "param_dtype": str(cfg.actor.fsdp_config.mixed_precision.param_dtype),
        "reduce_dtype": str(cfg.actor.fsdp_config.mixed_precision.reduce_dtype),
        "buffer_dtype": str(cfg.actor.fsdp_config.mixed_precision.buffer_dtype),
    }
    if numerical != {
        "param_dtype": "bf16",
        "reduce_dtype": "fp32",
        "buffer_dtype": "fp32",
    }:
        raise ValueError(f"unexpected RECAP mixed-precision contract: {numerical}")

    data_cfg = cfg.data
    if str(data_cfg.robot_type) != "clothdojo" or str(data_cfg.model_type) != "pi05":
        raise ValueError("RECAP value data requires robot_type=clothdojo and model_type=pi05")
    if int(data_cfg.action_horizon) != 1:
        raise ValueError("image-only value loading must use action_horizon=1")
    return_contracts = {
        RECAP_RETURNS_TAG: {"horizon": 4500, "return_min": -9000.0},
        RECAP_PREFIX60_RETURNS_TAG: {"horizon": 3600, "return_min": -7200.0},
    }
    returns_tag = str(data_cfg.tag)
    return_contract = return_contracts.get(returns_tag)
    if float(data_cfg.gamma) != 1.0 or return_contract is None:
        raise ValueError(
            f"RECAP value targets require gamma=1 and one of tags {sorted(return_contracts)}"
        )
    if (
        float(data_cfg.return_min) != return_contract["return_min"]
        or float(data_cfg.return_max) != 0.0
    ):
        raise ValueError(
            "RECAP return range does not match the configured failure horizon: "
            f"tag={returns_tag}, expected=[{return_contract['return_min']}, 0]"
        )
    image_key_map = dict(data_cfg.get("image_key_map", {}))
    if image_key_map != RECAP_IMAGE_KEY_MAP:
        raise ValueError(
            "RECAP value inputs must use the four configured cameras and no state: "
            f"{RECAP_IMAGE_KEY_MAP}"
        )

    data_root = Path(str(data_cfg.data_root)).expanduser().resolve()
    split_path = Path(str(data_cfg.value_split_path)).expanduser().resolve()
    frame_stride = int(data_cfg.get("frame_stride", 1))
    if frame_stride < 1 or 50 % frame_stride:
        raise ValueError("RECAP value frame_stride must be a positive divisor of 50")
    roots: list[Path] = []
    dataset_identities = []
    train_reports = []
    eval_reports = []
    entries = list(data_cfg.train_data_paths)
    if not entries:
        raise ValueError("RECAP value training has no datasets")
    for entry_index, entry_value in enumerate(entries):
        entry = dict(entry_value)
        if set(entry) - {"dataset_path", "type", "weight", "episodes"}:
            raise ValueError("per-shard data overrides would change the frozen RECAP contract")
        weight = float(entry.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("RECAP mixture weights must be finite and positive")
        root = Path(str(entry["dataset_path"])).expanduser()
        if not root.is_absolute():
            root = data_root / root
        root = root.resolve()
        if root in roots:
            raise ValueError(f"duplicate RECAP value dataset: {root}")
        roots.append(root)
        train_report = resolve_recap_value_selection(root, split_path, "train", frame_stride)
        eval_report = resolve_recap_value_selection(root, split_path, "val", frame_stride)
        train_report["weight"] = weight
        eval_report["weight"] = weight
        train_reports.append(train_report)
        eval_reports.append(eval_report)
        OmegaConf.update(
            data_cfg.train_data_paths[entry_index],
            "episodes",
            train_report["selected_episode_indices"],
            force_add=True,
        )
        returns_path = root / "meta" / f"returns_{data_cfg.tag}.parquet"
        returns_manifest_path = root / "meta" / f"returns_{data_cfg.tag}.json"
        if not returns_path.is_file():
            raise FileNotFoundError(f"RECAP returns sidecar is missing: {returns_path}")
        returns_manifest = _read_object(returns_manifest_path, "RECAP returns manifest")
        recap = _read_object(root / "meta" / "recap.json", "RECAP export metadata")
        expected_returns = {
            "schema": RECAP_RETURNS_SCHEMA,
            "status": "complete",
            "plan_sha256": recap.get("plan_sha256"),
            "tag": returns_tag,
            "gamma": 1.0,
            "c_fail_coef": 1.0,
            "max_episode_length": return_contract["horizon"],
            "failure_reward": -float(return_contract["horizon"]),
            "observation_alignment": "pre_action_state",
            "reward_alignment": "transition_cost_including_terminal_boundary",
            "action_reward": -1.0,
            "terminal_values": {
                "success": 0.0,
                "failure": -float(return_contract["horizon"]),
            },
            "terminal_observation_exported": False,
        }
        mismatches = {
            key: {"expected": expected, "actual": returns_manifest.get(key)}
            for key, expected in expected_returns.items()
            if returns_manifest.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"RECAP return contract differs at {root}: {mismatches}")
        if returns_manifest.get("sidecar_sha256") != _sha256(returns_path):
            raise RuntimeError(f"RECAP returns sidecar changed after generation: {returns_path}")
        validate_value_targets(
            root,
            str(data_cfg.tag),
            recap,
            failure_reward=-float(return_contract["horizon"]),
        )
        dataset_identities.append({
            "path": str(root), "recap_sha256": _sha256(root / "meta" / "recap.json"),
            "returns_sha256": returns_manifest["sidecar_sha256"],
            "info_sha256": _sha256(root / "meta" / "info.json"),
            "episodes_sha256": _sha256(root / "meta" / "episodes.jsonl"),
        })
    eval_entries = list(data_cfg.get("eval_data_paths", []))
    eval_roots = []
    populated_eval_entries = []
    for entry_index, entry_value in enumerate(eval_entries):
        entry = dict(entry_value)
        if set(entry) - {"dataset_path", "name", "episodes"}:
            raise ValueError("per-shard validation overrides change the frozen RECAP contract")
        root = Path(str(entry["dataset_path"])).expanduser()
        if not root.is_absolute():
            root = data_root / root
        eval_roots.append(root.resolve())
        selected_indices = eval_reports[entry_index]["selected_episode_indices"]
        if selected_indices:
            entry["episodes"] = selected_indices
            populated_eval_entries.append(entry)
    if eval_roots != roots:
        raise ValueError("RECAP eval_data_paths must repeat every train shard in the same order")
    # LeRobot interprets episodes=[] as an empty parquet file list and raises
    # during dataset construction. Keep full-shard validation above, then give
    # RLinf only the shards that actually contain held-out episodes.
    OmegaConf.update(
        data_cfg,
        "eval_data_paths",
        populated_eval_entries,
        force_add=True,
    )
    partition = _read_object(split_path, "RECAP value split")
    strategy = partition.get("strategy", {})
    split_unit = strategy.get("unit")
    if split_unit not in {"asset_id", "mirror_pair_id"}:
        raise ValueError(f"unsupported RECAP value split unit: {split_unit!r}")
    actual_assets: dict[str, set[str]] = {"train": set(), "val": set()}
    actual_groups: dict[str, dict[str, set[str]]] = {
        "train": {"expert": set(), "hil_rollout": set()},
        "val": {"expert": set(), "hil_rollout": set()},
    }
    for mode, reports in (("train", train_reports), ("val", eval_reports)):
        for role in ("expert", "hil_rollout"):
            selected = [source_id for report in reports if report["dataset"] == role
                        for source_id in report["selected_source_episode_ids"]]
            expected = partition["datasets"][role][
                "train_source_episode_ids" if mode == "train" else "eval_source_episode_ids"
            ]
            if len(selected) != len(set(selected)) or set(selected) != set(expected):
                raise RuntimeError(f"RECAP shards duplicate or omit {mode}/{role} episodes")
        for report in reports:
            recap = _read_object(
                Path(report["dataset_root"]) / "meta" / "recap.json", "RECAP export"
            )
            actual_assets[mode].update(recap["episodes"][i]["asset_id"]
                                       for i in report["selected_episode_indices"])
            actual_groups[mode][report["dataset"]].update(
                str(recap["episodes"][i].get("mirror_pair_id"))
                for i in report["selected_episode_indices"]
            )
    if split_unit == "asset_id":
        split_assets = set(partition["assets"]["eval_seen"]) | set(
            partition["assets"]["eval_unseen"]
        )
        if actual_assets["train"] & actual_assets["val"] or actual_assets["val"] != split_assets:
            raise RuntimeError("RECAP train/validation asset isolation changed")
    else:
        declared = partition.get("groups", {}).get("eval_mirror_pair_ids", {})
        for role in ("expert", "hil_rollout"):
            if actual_groups["train"][role] & actual_groups["val"][role]:
                raise RuntimeError(f"RECAP mirror pair crosses train/validation: {role}")
            if actual_groups["val"][role] != {str(value) for value in declared.get(role, [])}:
                raise RuntimeError(f"RECAP validation mirror-pair selection changed: {role}")
    train_frames = sum(item["selected_samples"] for item in train_reports)
    eval_frames = sum(item["selected_samples"] for item in eval_reports)
    if eval_frames < 1:
        raise ValueError("RECAP value validation split is empty")
    weights_by_role: dict[str, set[float]] = {"expert": set(), "hil_rollout": set()}
    for report in train_reports:
        weights_by_role[report["dataset"]].add(float(report["weight"]))
    if any(len(values) != 1 for values in weights_by_role.values()):
        raise ValueError("RECAP mixture weights must be uniform within each source role")
    weighted_samples = {
        role: sum(
            report["selected_samples"] * report["weight"]
            for report in train_reports if report["dataset"] == role
        )
        for role in weights_by_role
    }
    weighted_total = sum(weighted_samples.values())
    budget = plan_recap_training_budget(
        train_samples=train_frames,
        global_batch_size=global_batch,
        max_epochs=int(cfg.runner.max_epochs),
        max_steps=int(cfg.runner.max_steps),
    )
    run_steps = int(budget["effective_steps"])
    scheduler_steps = int(cfg.actor.optim.total_training_steps)
    if scheduler_steps != run_steps:
        raise ValueError(
            "RECAP scheduler must decay over the selected update budget: "
            f"{scheduler_steps} != {run_steps}"
        )
    if float(cfg.actor.optim.get("value_lr", 0.0)) <= 0.0:
        raise ValueError("RECAP optimizer requires a positive value_lr")
    if list(cfg.actor.optim.get("value_module_patterns", [])) != [
        "value_head",
        "multi_modal_proj",
        "experts.value",
    ]:
        raise ValueError("RECAP value_lr must cover the complete randomly initialized value stack")

    siglip = Path(str(cfg.actor.model.siglip_path)).expanduser().resolve()
    gemma = Path(str(cfg.actor.model.gemma3_path)).expanduser().resolve()
    tokenizer = Path(str(cfg.actor.model.tokenizer_path)).expanduser().resolve()
    if not siglip.is_dir():
        raise FileNotFoundError(f"RECAP SigLIP checkpoint is missing: {siglip}")
    if not gemma.is_dir():
        raise FileNotFoundError(f"RECAP Gemma checkpoint is missing: {gemma}")
    if tokenizer != gemma or not has_tokenizer_files(tokenizer):
        raise FileNotFoundError(f"RECAP Gemma tokenizer is missing: {tokenizer}")
    # Importing the decoder loads its native FFmpeg/PyTorch dependencies. A
    # package-presence check alone misses unusable shared-library paths.
    import torchcodec
    from torchcodec.decoders import VideoDecoder

    decoded = 0
    for root in roots:
        for camera in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"):
            videos = sorted(root.glob(f"videos/*/observation.images.{camera}/*.mp4"))
            video = next(iter(videos), None)
            if video is None:
                raise FileNotFoundError(f"RECAP camera video is missing: {root}:{camera}")
            decoder = VideoDecoder(str(video), seek_mode="approximate", num_ffmpeg_threads=1)
            frames = decoder.get_frames_at(indices=[0, len(decoder) - 1]).data
            if frames.shape[0] != 2 or frames.shape[1] != 3:
                raise RuntimeError(f"RECAP video decode shape is invalid: {video}")
            decoded += 1
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved RLINF config must be a mapping")
    resume = (
        inspect_recap_value_resume(
            str(resume_dir),
            current_config=resolved,
            world_size=world_size,
            steps_per_epoch=int(budget["steps_per_epoch"]),
            max_steps=run_steps,
        )
        if resume_dir is not None
        else None
    )
    return {
        "schema": "clothdojo_recap_value_launch_preflight",
        "format_version": 1,
        "compatible": True,
        "output": str(output),
        "accelerators": {"nodes": nodes, "gpus_per_node": devices_per_node, "gpus": world_size},
        "batching": {
            "micro_batch_size": micro_batch,
            "global_batch_size": global_batch,
            "gradient_accumulation": accumulation,
            **budget,
            "sampling": "rlinf_length_weighted_mixture_with_replacement",
        },
        "numerical_contract": numerical,
        "input_contract": {
            "views": list(RECAP_IMAGE_KEY_MAP),
            "prompt": True,
            "state": False,
            "actions": False,
            "aloha_conversion": False,
            "frame_stride": frame_stride,
            "sample_hz": 50 // frame_stride,
        },
        "returns": {
            "tag": returns_tag,
            "gamma": 1.0,
            "failure_reward": -float(return_contract["horizon"]),
            "return_min": return_contract["return_min"],
            "return_max": 0.0,
        },
        "split": {
            "path": str(split_path),
            "sha256": _sha256(split_path),
            "plan_sha256": partition["plan_sha256"],
            "train_frames": train_frames,
            "eval_frames": eval_frames,
            "shards": len(roots),
            "eval_shards": len(populated_eval_entries),
            "mixture_weights": {
                role: next(iter(values)) for role, values in weights_by_role.items()
            },
            "expected_sample_fraction": {
                role: value / weighted_total for role, value in weighted_samples.items()
            },
        },
        "training_stage": stage,
        "resume": resume,
        "initial_checkpoint": (
            {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
                "load": "strict_weights_only_new_optimizer",
            }
            if checkpoint is not None
            else None
        ),
        "backbones": {
            "siglip": str(siglip), "gemma3": str(gemma), "frozen": frozen_backbone,
        },
        "datasets": dataset_identities,
        "video_runtime": {"torchcodec": torchcodec.__version__, "decoded_videos": decoded},
        "worker_environment": worker_environment_from_config(cfg),
        "resolved_config": resolved,
    }


def validate_value_targets(
    root: Path,
    tag: str,
    recap: dict[str, Any],
    *,
    failure_reward: float = -4500.0,
) -> None:
    """Validate every frame's identity, prompt, action cost, and boundary return."""
    import pyarrow.parquet as pq

    table = pq.read_table(root / "meta" / f"returns_{tag}.parquet")
    columns = {
        name: table[name].to_numpy()
        for name in ("episode_index", "frame_index", "return", "reward")
    }
    if set(table["prompt"].unique().to_pylist()) != {RECAP_PROMPT}:
        raise RuntimeError(f"RECAP return prompt changed: {root}")
    cursor = 0
    for episode in recap["episodes"]:
        length = int(episode["action_count"])
        end = cursor + length
        boundary = 0.0 if episode["outcome"] == "success" else failure_reward
        expected = -np.arange(length, 0, -1, dtype=np.float32) + boundary
        expected_rewards = np.full(length, -1.0, dtype=np.float32)
        expected_rewards[-1] += boundary
        valid = (
            np.array_equal(columns["frame_index"][cursor:end], np.arange(length))
            and np.all(columns["episode_index"][cursor:end] == episode["episode_index"])
            and np.array_equal(columns["return"][cursor:end], expected)
            and np.array_equal(columns["reward"][cursor:end], expected_rewards)
        )
        if not valid:
            raise RuntimeError(
                f"RECAP observation/return alignment differs: {root}:{episode['source_episode_id']}"
            )
        cursor = end
    if cursor != table.num_rows:
        raise RuntimeError(f"RECAP returns have extra rows: {root}")


def write_recap_value_launch_manifest(cfg: Any, manifest: dict[str, Any]) -> Path:
    """Claim a fresh output directory and record the checked value run identity."""

    from clothdojo.runtime_identity import runtime_source_sha256

    manifest = {**manifest, "runtime_source_sha256": runtime_source_sha256()}
    output = Path(str(cfg.runner.logger.log_path)).expanduser().resolve()
    try:
        output.mkdir(parents=True)
    except FileExistsError as exc:
        raise FileExistsError(f"RECAP value training requires a new output: {output}") from exc
    path = output / "clothdojo_launch.json"
    temporary = output / ".clothdojo_launch.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
