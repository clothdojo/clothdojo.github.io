"""Human-correction data boundary for the existing OpenPI_RLinf SFT worker."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from clothdojo.training.openpi_artifacts import (
    _read_jsonl,
    _read_object,
    _sha256,
    _validate_features,
    inspect_openpi_dataset,
)


def _resolve_source_root(value: str) -> Path:
    """Map a frozen source path only when both project roots are explicit."""
    source = Path(value).expanduser().resolve()
    if source.exists():
        return source
    recorded_root = os.environ.get("CLOTHDOJO_SOURCE_PROJECT_ROOT")
    runtime_root = os.environ.get("CLOTHDOJO_PROJECT_ROOT")
    if not recorded_root or not runtime_root:
        return source
    recorded = Path(recorded_root).expanduser().resolve()
    runtime = Path(runtime_root).expanduser().resolve()
    try:
        relative = source.relative_to(recorded)
    except ValueError:
        return source
    mapped = (runtime / relative).resolve()
    if not mapped.is_relative_to(runtime):
        raise ValueError("Mapped correction source escapes the runtime project root")
    return mapped


def inspect_correction_dataset(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Accept only recorded human corrections bound to the supplied source plan."""
    info = _read_object(root / "meta/info.json", "correction info")
    export = _read_object(root / "meta/recap.json", "correction provenance")
    sidecar = _read_object(root / "meta/clothdojo.json", "correction sidecar")
    rows = _read_jsonl(root / "meta/episodes.jsonl", "correction episodes")
    tasks = _read_jsonl(root / "meta/tasks.jsonl", "correction task")
    _validate_features(info.get("features"), root)
    if info.get("codebase_version") != "v2.1" or info.get("fps") != 50:
        raise ValueError("HIL SFT requires LeRobot v2.1 at 50 Hz")
    if tasks != [{"task_index": 0, "task": "flatten the shirt"}]:
        raise ValueError("HIL correction task differs from flatten the shirt")
    if (
        sidecar.get("export_status") != "complete"
        or export.get("schema") != "clothdojo_recap_export_v1"
        or export.get("status") != "complete"
    ):
        raise ValueError("HIL correction export is incomplete or lacks source provenance")
    if export.get("dataset") != "hil_correction":
        raise ValueError("Autonomous rollout actions cannot enter HIL SFT")
    approved = {
        ep["source_episode_id"]: ep for ep in plan["datasets"]["hil_correction"]["episodes"]
    }
    episodes = export.get("episodes", [])
    if not episodes or len(episodes) != len(rows) or len(rows) != info.get("total_episodes"):
        raise ValueError("HIL correction episode counts differ")
    if sidecar.get("counts", {}).get("output_episodes") != len(rows):
        raise ValueError("HIL correction sidecar episode count differs")
    source_root = _resolve_source_root(export["source_root"])
    identifiers, lengths, included = set(), [], []
    for index, (episode, row) in enumerate(zip(episodes, rows, strict=True)):
        identity = episode["source_episode_id"]
        if identity in identifiers:
            raise ValueError("Duplicate correction episode identity")
        identifiers.add(identity)
        original_id = identity.removesuffix("__mirror")
        if original_id not in approved:
            raise ValueError(f"Unapproved correction source: {identity}")
        original = approved[original_id]
        count = episode["action_count"]
        if (
            type(count) is not int
            or count < 1
            or episode.get("episode_index") != index
            or row.get("episode_index") != index
            or row.get("length") != count
            or count != original["action_count"]
            or episode.get("action_start") != 0
            or episode.get("action_end_exclusive") != count
            or episode.get("asset_id") != original["asset_id"]
            or episode.get("outcome") != "correction"
        ):
            raise ValueError(f"Correction identity/extent differs: {identity}")
        trajectory = (source_root / episode["path"]).resolve()
        if not trajectory.is_relative_to(source_root):
            raise ValueError("Correction path escapes its source collection")
        manifest_path = trajectory / "manifest.json"
        if _sha256(manifest_path) != episode["manifest_sha256"]:
            raise ValueError(f"Correction manifest hash differs: {identity}")
        metadata = _read_object(manifest_path, "correction manifest").get("metadata", {})
        if (
            metadata.get("source") != "human_intervention"
            or metadata.get("control_source") != "vr_teleop"
            or metadata.get("action_semantics") != "runtime_target"
        ):
            raise ValueError(f"Correction is not a recorded human runtime target: {identity}")
        if identity == original_id:
            if episode["manifest_sha256"] != original["manifest_sha256"]:
                raise ValueError(f"Correction differs from approved plan: {identity}")
        else:
            augmentation = metadata.get("augmentation", {})
            if (
                augmentation.get("source_trajectory_id") != original_id
                or augmentation.get("source_manifest_sha256") != original["manifest_sha256"]
            ):
                raise ValueError(f"Mirror is not bound to the approved correction: {identity}")
        lengths.append(count)
        if count >= 50:
            included.append(index)
    frames = sum(lengths)
    if frames != info.get("total_frames") or frames != sidecar.get("counts", {}).get(
        "output_frames"
    ):
        raise ValueError("HIL correction frame counts differ")
    if not included:
        raise ValueError("HIL correction dataset has no complete H50 action chunks")
    return {
        "root": str(root),
        "episodes": len(rows),
        "frames": frames,
        "selected_episodes": included,
        "full_h50_anchors": sum(max(0, n - 49) for n in lengths),
        "excluded_short_episodes": [i for i, n in enumerate(lengths) if n < 50],
        "identities": sorted(identifiers),
        "info_sha256": _sha256(root / "meta/info.json"),
        "sidecar_sha256": _sha256(root / "meta/clothdojo.json"),
        "recap_sha256": _sha256(root / "meta/recap.json"),
    }


def inspect_hil_sft_inputs(cfg: Any) -> dict[str, Any]:
    """Keep old SFT normalization while checking the new supervised mixture."""
    from omegaconf import OmegaConf

    from clothdojo.rollout.artifacts import find_training_manifest, validate_sft_norm_binding
    from clothdojo.rollout.checkpoint import rlinf_openpi_weights_sha256

    settings = cfg.clothdojo.hil_sft
    if (
        cfg.clothdojo.get("action_chunk_boundary") != "full_only"
        or cfg.clothdojo.get("jax_reference") is not None
    ):
        raise ValueError("HIL SFT requires full_only without a JAX comparison override")
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
        raise ValueError("HIL SFT requires the native H50/14D OpenPI_RLinf SFT model")
    if cfg.runner.val_check_interval != -1:
        raise ValueError(
            "Embodied SFT validation is not implemented; use paired rollout evaluation"
        )
    plan_path = Path(str(settings.policy_plan)).expanduser().resolve()
    if _sha256(plan_path) != str(settings.policy_plan_sha256):
        raise ValueError("HIL source plan differs from its frozen hash")
    plan = _read_object(plan_path, "HIL source plan")
    if plan.get("schema") != "clothdojo_recap_plan_v1" or plan.get("status") != "ready":
        raise ValueError("HIL source plan must be ready")
    checkpoint = Path(str(model.model_path)).expanduser().resolve()
    expected = str(settings.initialization_weights_sha256)
    if len(expected) != 64 or rlinf_openpi_weights_sha256(checkpoint) != expected:
        raise ValueError("HIL SFT initialization weights differ from the frozen hash")
    launch = _read_object(find_training_manifest(checkpoint), "SFT initialization manifest")
    norm_root = Path(str(model.openpi_data.norm_stats_path)).expanduser().resolve()
    binding = validate_sft_norm_binding(
        checkpoint, norm_root, openpi_config=str(model.openpi.config_name), action_horizon=50
    )
    old_model = launch["resolved_config"]["actor"]["model"]
    for key in ("pi05", "num_action_chunks", "action_dim", "use_proprio"):
        if model.get(key) != old_model.get(key):
            raise ValueError(f"HIL SFT changes initialization model field {key}")
    for key in (
        "model_action_dim",
        "paligemma_variant",
        "action_expert_variant",
        "max_token_len",
        "discrete_state_input",
    ):
        if model.openpi.get(key) != old_model["openpi"].get(key):
            raise ValueError(f"HIL SFT changes initialization model field openpi.{key}")
    for key in ("adapt_to_pi", "extra_delta_transform", "default_prompt"):
        if model.openpi_data.get(key) != old_model["openpi_data"].get(key):
            raise ValueError(f"HIL SFT changes initialization transform {key}")
    if launch.get("action_chunk_boundary") != "full_only":
        raise ValueError("HIL SFT initialization must use full_only normalization")
    entries = OmegaConf.to_container(cfg.data.train_data_paths, resolve=True)
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("HIL SFT requires explicit expert and correction sources")
    reports, paths, roles, correction_ids = [], set(), set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"dataset_path", "role", "weight"}:
            raise ValueError("HIL source requires exactly dataset_path, role and weight")
        role, weight = entry["role"], entry["weight"]
        root = Path(entry["dataset_path"]).expanduser().resolve()
        if (
            role not in {"expert", "correction"}
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("HIL source role/weight is invalid")
        if root in paths:
            raise ValueError("HIL SFT repeats the same dataset path")
        paths.add(root)
        roles.add(role)
        if role == "expert":
            report = inspect_openpi_dataset(root)
            for key in ("info_sha256", "sidecar_sha256", "episodes_sha256", "tasks_sha256"):
                if report[key] != launch["dataset"][key]:
                    raise ValueError(f"Expert replay differs from initialization: {key}")
            rows = _read_jsonl(root / "meta/episodes.jsonl", "expert episodes")
            report["full_h50_anchors"] = sum(row["length"] - 49 for row in rows)
        else:
            report = inspect_correction_dataset(root, plan)
            if correction_ids.intersection(report["identities"]):
                raise ValueError("Correction sources overlap in episode identity")
            correction_ids.update(report["identities"])
        reports.append({**report, "role": role, "weight": float(weight)})
    if roles != {"expert", "correction"} or not math.isclose(
        sum(r["weight"] for r in reports), 1.0, abs_tol=1e-6
    ):
        raise ValueError("HIL weights must sum to one and include expert and correction")
    originals = {
        ep["source_episode_id"]
        for ep in plan["datasets"]["hil_correction"]["episodes"]
        if ep["action_count"] >= 50
    }
    required = originals | {identity + "__mirror" for identity in originals}
    selected = {
        identity for identity in correction_ids if identity.removesuffix("__mirror") in originals
    }
    if selected != required:
        raise ValueError(
            "HIL SFT requires every approved H50 correction and its mirror; "
            f"missing {len(required - selected)}"
        )
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
            "frames": sum(r["frames"] for r in reports),
            "episodes": sum(r["episodes"] for r in reports),
        },
        "norm_stats": norm,
        "initialization_weights_sha256": expected,
        "policy_plan_sha256": _sha256(plan_path),
        "balance_dataset_weights": False,
    }


class SourceDataset:
    """Carry source identity past transforms so logs count consumed examples."""

    def __init__(self, dataset: Any, role: str):
        self.dataset, self.role = dataset, role

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "hil_source": int(self.role == "correction")}


class HilSftLoader:
    def __init__(self, data_config: Any, loader: Any):
        self._data_config, self._data_loader = data_config, loader

    def data_config(self):
        return self._data_config

    def __iter__(self):
        from openpi.models.model import Observation

        for batch in self._data_loader:
            yield {
                "observation": Observation.from_dict(batch),
                "actions": batch["actions"],
                "hil_source": batch["hil_source"],
            }


def build_hil_sft_loader(cfg: Any, *, world_size: int, rank: int):
    """Reuse upstream sampling and ordinary OpenPI transforms, without CFG labels."""
    import torch
    from openpi import transforms
    from openpi.training.data_loader import TorchDataLoader, TransformedDataset
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        _validate_openpi_rlinf_model_shape,
    )
    from rlinf.data.datasets.recap.common import ReCapMixtureDataset
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    from clothdojo.data.lerobot import open_lerobot_dataset
    from clothdojo.training.cfg_data_runtime import CfgEpochSampler, EpochTaggedMixture
    from clothdojo.training.openpi import TrainingPrompt
    from clothdojo.training.openpi_rlinf_sft import (
        _EpochAdvancingSampler,
        _full_only_anchor_indices,
    )

    cache = Path(str(cfg.data.openpi_data_home)).expanduser().resolve()
    if not (cache / "big_vision/paligemma_tokenizer.model").is_file():
        raise FileNotFoundError(f"HIL SFT tokenizer is missing from its configured cache: {cache}")
    os.environ["OPENPI_DATA_HOME"] = str(cache)
    entries = cfg.data.train_data_paths
    config = get_openpi_config(
        str(cfg.actor.model.openpi.config_name),
        model_path=str(cfg.actor.model.model_path),
        batch_size=int(cfg.actor.micro_batch_size) * world_size,
        repo_id=str(entries[0].dataset_path),
        data_kwargs=cfg.actor.model.openpi_data,
    )
    _validate_openpi_rlinf_model_shape(cfg.actor.model, config)
    data_config = config.data.create(config.assets_dirs, config.model)
    if not data_config.norm_stats:
        raise ValueError("HIL SFT must load its initialization norm stats")
    datasets = []
    for entry in entries:
        root = Path(str(entry.dataset_path)).expanduser().resolve()
        raw = open_lerobot_dataset(
            str(root),
            root=root,
            delta_timestamps={
                key: [t / 50 for t in range(50)] for key in data_config.action_sequence_keys
            },
        )
        trim_manifest = entry.get("home_trim_manifest")
        if trim_manifest is None:
            anchors = _full_only_anchor_indices(raw, action_horizon=50)
        else:
            from clothdojo.training.offline_sft import home_trim_anchor_indices

            anchor_values, _ = home_trim_anchor_indices(root, str(trim_manifest))
            anchors = torch.as_tensor(anchor_values, dtype=torch.int64)
        threshold = entry.get("max_arm_action_step_rad")
        if threshold is not None:
            from clothdojo.training.offline_sft import filter_action_jump_anchors

            anchor_values, _ = filter_action_jump_anchors(
                root,
                anchors.tolist(),
                max_arm_action_step_rad=float(threshold),
            )
            anchors = torch.as_tensor(anchor_values, dtype=torch.int64)
        transformed = TransformedDataset(
            raw,
            [
                TrainingPrompt(str(cfg.actor.model.openpi_data.default_prompt)),
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.model_transforms.inputs,
            ],
        )
        datasets.append(
            (
                SourceDataset(
                    torch.utils.data.Subset(transformed, anchors.tolist()), str(entry.role)
                ),
                float(entry.weight),
            )
        )
    mixture = ReCapMixtureDataset(datasets, balance_dataset_weights=False, seed=int(cfg.actor.seed))
    tagged = EpochTaggedMixture(mixture)
    sampler = CfgEpochSampler(
        tagged,
        batch_size=int(cfg.actor.micro_batch_size),
        num_replicas=world_size,
        rank=rank,
        seed=int(cfg.actor.seed),
        shuffle=True,
        drop_last=True,
    )
    loader = TorchDataLoader(
        tagged,
        int(cfg.actor.micro_batch_size),
        sampler=_EpochAdvancingSampler(sampler),
        num_workers=int(cfg.data.num_workers),
        seed=int(cfg.actor.seed) + rank,
        framework="pytorch",
    )
    return HilSftLoader(data_config, loader), data_config
