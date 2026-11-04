"""Hierarchical semantic-subtask SFT for the public PyTorch Pi0.5 core.

The implementation follows the factorization used by the Pi0.5 paper while
reusing the two pieces already shipped in the pinned dependencies:

* OpenPI FAST's prefix-LM token masks and next-token objective.
* RLinf's Pi0.5 flow-matching action expert.

During training, the action expert attends to the teacher-forced semantic
subtask tokens.  At inference, candidate semantic texts are scored first and
the selected text is then inserted into the prefix used by the action expert.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@functools.lru_cache(maxsize=8)
def _sentencepiece_processor(path: str):
    import sentencepiece

    return sentencepiece.SentencePieceProcessor(model_file=path)


@dataclasses.dataclass(frozen=True)
class SemanticPromptTokenizer:
    """Build a prefix-LM sequence containing an overall task and subtask target."""

    tokenizer_path: str
    overall_prompt: str
    semantic_subtask: str
    max_token_len: int
    phase_index: int

    def __post_init__(self) -> None:
        for name, value in (
            ("overall_prompt", self.overall_prompt),
            ("semantic_subtask", self.semantic_subtask),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.max_token_len < 1:
            raise ValueError("max_token_len must be positive")
        if self.phase_index < 0:
            raise ValueError("phase_index must be non-negative")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        tokens, token_mask, ar_mask, loss_mask = encode_semantic_prompt(
            tokenizer_path=self.tokenizer_path,
            overall_prompt=self.overall_prompt,
            semantic_subtask=self.semantic_subtask,
            normalized_state=np.asarray(data["state"]),
            max_token_len=self.max_token_len,
        )
        data.pop("prompt", None)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
            "semantic_phase": np.asarray(self.phase_index, dtype=np.int64),
        }


def encode_semantic_prompt(
    *,
    tokenizer_path: str | Path,
    overall_prompt: str,
    semantic_subtask: str,
    normalized_state: np.ndarray,
    max_token_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tokenize the paper-style ``overall task -> semantic subtask`` target."""

    state = np.asarray(normalized_state)
    if state.ndim != 1:
        raise ValueError(f"normalized_state must be one-dimensional, got {state.shape}")
    discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    state_text = " ".join(map(str, discretized))
    overall = overall_prompt.strip().lower().replace("_", " ").replace("\n", " ")
    subtask = semantic_subtask.strip().lower().replace("_", " ").replace("\n", " ")
    prefix = f"Task: {overall}, State: {state_text};\nSubtask: "

    processor = _sentencepiece_processor(str(Path(tokenizer_path).expanduser().resolve()))
    prefix_tokens = processor.encode(prefix, add_bos=True)
    target_tokens = processor.encode(subtask, add_eos=True)
    tokens = prefix_tokens + target_tokens
    if len(tokens) > max_token_len:
        raise ValueError(
            "semantic prompt exceeds actor.model.openpi.max_token_len: "
            f"{len(tokens)} > {max_token_len}"
        )
    padding = max_token_len - len(tokens)
    return (
        np.asarray(tokens + [0] * padding, dtype=np.int64),
        np.asarray([True] * len(tokens) + [False] * padding),
        np.asarray(
            [0] * len(prefix_tokens) + [1] * len(target_tokens) + [0] * padding,
            dtype=np.int64,
        ),
        np.asarray([False] * len(prefix_tokens) + [True] * len(target_tokens) + [False] * padding),
    )


def _joint_semantic_action_losses(
    model: Any,
    observation: Any,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one shared VLM/action-expert forward and return both losses."""

    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import make_attn_mask

    required = (
        observation.tokenized_prompt,
        observation.tokenized_prompt_mask,
        observation.token_ar_mask,
        observation.token_loss_mask,
    )
    if any(value is None for value in required):
        raise ValueError("semantic SFT requires token, input, AR, and token-loss masks")

    batch_size = actions.shape[0]
    device = actions.device
    observation = model_module.preprocess_observation(observation, train=True)
    observation = model_module._observation_to_dtype(observation, model.embed_dtype)
    actions = actions.to(dtype=model.embed_dtype)

    noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
    time = (
        torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        .sample((batch_size,))
        .to(device=device, dtype=actions.dtype)
    )
    time = time * 0.999 + 0.001
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    target_velocity = noise - actions

    prefix_tokens, prefix_mask, _ = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation, x_t, time
    )
    image_token_count = prefix_tokens.shape[1] - observation.tokenized_prompt.shape[1]
    prefix_ar_mask = torch.cat(
        [
            torch.zeros(
                batch_size,
                image_token_count,
                dtype=torch.bool,
                device=device,
            ),
            observation.token_ar_mask.bool(),
        ],
        dim=1,
    )
    suffix_ar_mask = suffix_ar_mask.expand(batch_size, -1)
    input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
    ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=1)
    attention_mask = make_attn_mask(input_mask, ar_mask)
    positions = torch.cumsum(input_mask.int(), dim=1) - 1

    prefix_out, suffix_out = model.llm(
        [prefix_tokens, suffix_tokens],
        positions=positions,
        mask=attention_mask,
        adarms_cond=[None, adarms_cond],
    )[0]
    velocity = model.velocity_from_suffix(suffix_out[:, -model.action_horizon :])
    action_loss = torch.mean(torch.square(velocity - target_velocity), dim=-1)

    # A language token at position i predicts the token at i+1.  Decode only
    # supervised positions; materializing [B, max_len, vocab] is unnecessary.
    language_out = prefix_out[:, image_token_count:]
    target_ids = observation.tokenized_prompt[:, 1:].long()
    target_mask = observation.token_loss_mask[:, 1:].bool()
    selected_hidden = language_out[:, :-1][target_mask]
    selected_targets = target_ids[target_mask]
    if selected_targets.numel() == 0:
        raise ValueError("semantic SFT batch contains no supervised text tokens")
    embedding_weight = model.llm.embedder.embedding.weight
    logits = model.llm.embedder.decode(selected_hidden.to(embedding_weight.dtype))
    semantic_loss = F.cross_entropy(logits.float(), selected_targets, reduction="mean")
    return action_loss, semantic_loss


def install_semantic_sft(
    fsdp_model: Any,
    *,
    token_loss_weight: float,
    action_offset_weights: Sequence[float],
) -> None:
    """Install joint semantic-token and flow-matching SFT on one wrapper."""

    from rlinf.models.embodiment.openpi_rlinf.sft_action_model import (
        OpenPiPytorchSFTActionModel,
    )

    from clothdojo.training.openpi_rlinf_sft import _reduce_action_offset_loss

    if not math.isfinite(token_loss_weight) or token_loss_weight <= 0:
        raise ValueError("semantic token_loss_weight must be positive and finite")
    module = getattr(fsdp_model, "module", fsdp_model)
    if not isinstance(module, OpenPiPytorchSFTActionModel):
        raise TypeError(
            "semantic SFT requires OpenPiPytorchSFTActionModel, got "
            f"{type(module).__name__}"
        )
    if bool(module.rlt_cfg.use_rlt):
        raise ValueError("semantic SFT cannot be combined with legacy RLT")
    weights = tuple(float(value) for value in action_offset_weights)

    def semantic_sft_forward(wrapper: Any, data: Any) -> dict[str, torch.Tensor]:
        observation, actions = wrapper._unpack_sft_batch(data)
        observation = wrapper._observation_to_device(observation)
        actions = wrapper._actions_to_device(actions)
        per_offset_loss, token_loss = _joint_semantic_action_losses(
            wrapper.model, observation, actions
        )
        action_loss = _reduce_action_offset_loss(per_offset_loss, weights)
        return {
            "loss": action_loss + token_loss_weight * token_loss,
            "action_loss": action_loss,
            "semantic_token_loss": token_loss,
        }

    module.sft_forward = MethodType(semantic_sft_forward, module)


def _tokenize_semantic_batch(
    observation: Any,
    *,
    tokenizer_path: str | Path,
    overall_prompt: str,
    semantic_subtask: str,
    max_token_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = observation.state[..., :14].detach().float().cpu().numpy()
    encoded = [
        encode_semantic_prompt(
            tokenizer_path=tokenizer_path,
            overall_prompt=overall_prompt,
            semantic_subtask=semantic_subtask,
            normalized_state=item,
            max_token_len=max_token_len,
        )
        for item in state
    ]
    device = observation.state.device
    return tuple(
        torch.from_numpy(np.stack(values)).to(device=device)
        for values in zip(*encoded, strict=True)
    )


def _replace_observation_tokens(observation: Any, encoded: Sequence[torch.Tensor]) -> Any:
    return dataclasses.replace(
        observation,
        tokenized_prompt=encoded[0].long(),
        tokenized_prompt_mask=encoded[1].bool(),
        token_ar_mask=encoded[2].long(),
        token_loss_mask=encoded[3].bool(),
    )


def _semantic_sequence_nll(model: Any, observation: Any) -> torch.Tensor:
    """Return one mean next-token NLL per example for a candidate subtask."""

    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import make_attn_mask

    prepared = model_module.preprocess_observation(observation, train=False)
    prepared = model_module._observation_to_dtype(prepared, model.embed_dtype)
    embedded, input_mask, _ = model.embed_prefix(prepared)
    image_token_count = embedded.shape[1] - prepared.tokenized_prompt.shape[1]
    ar_mask = torch.cat(
        [
            torch.zeros(
                input_mask.shape[0],
                image_token_count,
                dtype=torch.bool,
                device=input_mask.device,
            ),
            prepared.token_ar_mask.bool(),
        ],
        dim=1,
    )
    attention_mask = make_attn_mask(input_mask, ar_mask)
    positions = torch.cumsum(input_mask.int(), dim=1) - 1
    prefix_out = model.llm(
        [embedded, None], positions=positions, mask=attention_mask
    )[0][0]
    language_out = prefix_out[:, image_token_count:]
    targets = prepared.tokenized_prompt[:, 1:].long()
    loss_mask = prepared.token_loss_mask[:, 1:].bool()
    batch_indices = torch.nonzero(loss_mask, as_tuple=True)[0]
    hidden = language_out[:, :-1][loss_mask]
    embedding_weight = model.llm.embedder.embedding.weight
    logits = model.llm.embedder.decode(hidden.to(embedding_weight.dtype))
    losses = F.cross_entropy(logits.float(), targets[loss_mask], reduction="none")
    sums = torch.zeros(input_mask.shape[0], device=losses.device, dtype=losses.dtype)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, batch_indices, losses)
    counts.scatter_add_(0, batch_indices, torch.ones_like(losses))
    return sums / counts.clamp_min(1)


@torch.no_grad()
def condition_on_predicted_subtask(
    wrapper: Any,
    observation: Any,
    *,
    tokenizer_path: str | Path,
    overall_prompt: str,
    subtasks: Sequence[str],
) -> tuple[Any, list[str], torch.Tensor]:
    """Select semantic text by candidate NLL and return action-conditioned inputs."""

    if len(subtasks) < 2 or len(set(subtasks)) != len(subtasks):
        raise ValueError("semantic inference requires distinct subtask candidates")
    max_token_len = int(wrapper.model._config.max_token_len)
    candidate_observations = []
    candidate_scores = []
    for subtask in subtasks:
        encoded = _tokenize_semantic_batch(
            observation,
            tokenizer_path=tokenizer_path,
            overall_prompt=overall_prompt,
            semantic_subtask=subtask,
            max_token_len=max_token_len,
        )
        candidate = _replace_observation_tokens(observation, encoded)
        candidate_observations.append(candidate)
        candidate_scores.append(_semantic_sequence_nll(wrapper.model, candidate))
    scores = torch.stack(candidate_scores, dim=1)
    selected = torch.argmin(scores, dim=1)

    def choose(field: str) -> torch.Tensor:
        values = torch.stack(
            [getattr(item, field) for item in candidate_observations], dim=1
        )
        index = selected[:, None, None].expand(-1, 1, values.shape[-1])
        return values.gather(1, index).squeeze(1)

    conditioned = dataclasses.replace(
        observation,
        tokenized_prompt=choose("tokenized_prompt"),
        tokenized_prompt_mask=choose("tokenized_prompt_mask"),
        token_ar_mask=choose("token_ar_mask"),
        token_loss_mask=choose("token_loss_mask"),
    )
    labels = [str(subtasks[index]) for index in selected.cpu().tolist()]
    return conditioned, labels, scores


def install_semantic_inference(
    wrapper: Any,
    *,
    tokenizer_path: str | Path,
    overall_prompt: str,
    subtasks: Sequence[str],
) -> None:
    """Install the semantic-text-first action inference interface."""

    settings = {
        "tokenizer_path": str(Path(tokenizer_path).expanduser().resolve()),
        "overall_prompt": str(overall_prompt),
        "subtasks": tuple(map(str, subtasks)),
    }
    wrapper._clothdojo_semantic_config = settings

    def condition(instance: Any, observation: Any):
        return condition_on_predicted_subtask(instance, observation, **settings)

    wrapper.condition_semantic_observation = MethodType(condition, wrapper)

    def predict(
        instance: Any,
        env_obs: dict[str, Any],
        mode: str = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs: Any,
    ):
        del compute_values, kwargs
        if mode != "eval":
            raise NotImplementedError("semantic Pi0.5 inference supports eval mode only")
        processed = instance.input_transform(
            instance._repack_env_obs(env_obs), transpose=False
        )
        observation = instance._observation_dict_to_device(processed)
        conditioned, labels, scores = instance.condition_semantic_observation(observation)
        actions, result = instance._predict_eval(conditioned, noise=noise, rng=rng)
        result["semantic_subtask"] = labels
        result["semantic_candidate_nll"] = scores
        return actions, result

    wrapper.predict_action_batch = MethodType(predict, wrapper)


class _PhaseDataset:
    def __init__(self, dataset: Any, phase_index: int):
        self.dataset = dataset
        self.phase_index = int(phase_index)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {**self.dataset[index], "semantic_phase": self.phase_index}


class SemanticSftLoader:
    def __init__(self, data_config: Any, loader: Any):
        self._data_config = data_config
        self._data_loader = loader

    def data_config(self) -> Any:
        return self._data_config

    def __iter__(self):
        from openpi.models.model import Observation

        for batch in self._data_loader:
            yield {
                "observation": Observation.from_dict(batch),
                "actions": batch["actions"],
                "semantic_phase": batch["semantic_phase"],
            }


def build_semantic_sft_loader(cfg: Any, *, world_size: int, rank: int):
    """Build a balanced, full-H50 loader over semantic subtask sources."""

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

    settings = cfg.clothdojo.semantic_sft
    entries = list(cfg.data.train_data_paths)
    cache = Path(str(cfg.data.openpi_data_home)).expanduser().resolve()
    tokenizer_path = cache / "big_vision/paligemma_tokenizer.model"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"semantic SFT tokenizer is missing: {tokenizer_path}")
    os.environ["OPENPI_DATA_HOME"] = str(cache)
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
        raise ValueError("semantic SFT must load mixture norm stats")

    model_transforms = []
    tokenize_count = 0
    for transform in data_config.model_transforms.inputs:
        if isinstance(transform, transforms.InjectDefaultPrompt):
            continue
        if isinstance(transform, transforms.TokenizePrompt):
            tokenize_count += 1
            continue
        model_transforms.append(transform)
    if tokenize_count != 1:
        raise TypeError("OpenPI model transform layout changed: expected one TokenizePrompt")

    datasets = []
    max_token_len = int(cfg.actor.model.openpi.max_token_len)
    overall_prompt = str(settings.overall_prompt)
    for phase_index, entry in enumerate(entries):
        root = Path(str(entry.dataset_path)).expanduser().resolve()
        raw = open_lerobot_dataset(
            str(root),
            root=root,
            delta_timestamps={
                key: [offset / 50 for offset in range(50)]
                for key in data_config.action_sequence_keys
            },
        )
        anchors = _full_only_anchor_indices(raw, action_horizon=50)
        semantic_transform = SemanticPromptTokenizer(
            tokenizer_path=str(tokenizer_path),
            overall_prompt=overall_prompt,
            semantic_subtask=str(entry.semantic_subtask),
            max_token_len=max_token_len,
            phase_index=phase_index,
        )
        transformed = TransformedDataset(
            raw,
            [
                TrainingPrompt(overall_prompt),
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    data_config.norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                semantic_transform,
                *model_transforms,
            ],
        )
        datasets.append(
            (
                _PhaseDataset(
                    torch.utils.data.Subset(transformed, anchors.tolist()), phase_index
                ),
                float(entry.weight),
            )
        )
    mixture = ReCapMixtureDataset(
        datasets, balance_dataset_weights=False, seed=int(cfg.actor.seed)
    )
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
    return SemanticSftLoader(data_config, loader), data_config


def semantic_sft_manifest(cfg: Any) -> dict[str, Any] | None:
    settings = cfg.get("clothdojo", {}).get("semantic_sft")
    if settings is None:
        return None
    return {
        "schema": "clothdojo_pi05_semantic_subtask",
        "format_version": 1,
        "overall_prompt": str(settings.overall_prompt),
        "subtasks": [str(entry.semantic_subtask) for entry in cfg.data.train_data_paths],
        "token_loss_weight": float(settings.token_loss_weight),
        "inference": {
            "selection": "candidate_sequence_mean_nll",
            "action_conditioning": "selected_semantic_subtask_tokens",
        },
    }


def inspect_semantic_sft_inputs(cfg: Any) -> dict[str, Any]:
    """Validate semantic sources and their shared normalization artifact."""

    from clothdojo.training.openpi_artifacts import _read_object, _sha256, inspect_openpi_dataset

    settings = cfg.clothdojo.semantic_sft
    if cfg.clothdojo.get("action_chunk_boundary") != "full_only":
        raise ValueError("semantic SFT requires full_only H50 supervision")
    if cfg.actor.model.openpi.get("use_rlt", False):
        raise ValueError("semantic SFT cannot be combined with legacy RLT")
    entries = list(cfg.data.train_data_paths)
    if len(entries) < 2:
        raise ValueError("semantic SFT requires at least two subtask sources")
    reports = []
    seen_subtasks: set[str] = set()
    total_weight = 0.0
    for entry in entries:
        keys = set(entry.keys())
        if keys != {"dataset_path", "semantic_subtask", "weight"}:
            raise ValueError(
                "semantic source requires exactly dataset_path, semantic_subtask, weight"
            )
        subtask = str(entry.semantic_subtask).strip().lower().rstrip(".")
        if not subtask or subtask in seen_subtasks:
            raise ValueError("semantic subtasks must be non-empty and unique")
        seen_subtasks.add(subtask)
        weight = float(entry.weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("semantic source weight must be positive and finite")
        total_weight += weight
        report = inspect_openpi_dataset(entry.dataset_path, expected_task=subtask)
        reports.append({**report, "semantic_subtask": subtask, "weight": weight})
    if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
        raise ValueError("semantic source weights must sum to one")
    if not str(settings.overall_prompt).strip():
        raise ValueError("semantic overall_prompt must be non-empty")
    token_weight = float(settings.token_loss_weight)
    if not math.isfinite(token_weight) or token_weight <= 0:
        raise ValueError("semantic token_loss_weight must be positive and finite")

    norm_root = Path(str(cfg.actor.model.openpi_data.norm_stats_path)).expanduser().resolve()
    stats_path = norm_root / "norm_stats.json"
    provenance_path = norm_root / "clothdojo.json"
    stats = _read_object(stats_path, "semantic mixture norm stats")
    provenance = _read_object(provenance_path, "semantic mixture norm provenance")
    if provenance.get("schema") != "clothdojo_openpi_mixture_norm_stats":
        raise ValueError("semantic SFT requires mixture normalization provenance")
    expected_sources = [
        {
            "dataset_root": report["root"],
            "dataset_info_sha256": report["info_sha256"],
            "dataset_sidecar_sha256": report["sidecar_sha256"],
        }
        for report in reports
    ]
    if provenance.get("sources") != expected_sources:
        raise ValueError("semantic mixture norm stats differ from training sources")
    values = stats.get("norm_stats")
    if not isinstance(values, dict) or set(values) != {"state", "actions"}:
        raise ValueError("semantic norm_stats.json must contain state and actions")
    return {
        "dataset": {
            "sources": reports,
            "episodes": sum(report["episodes"] for report in reports),
            "frames": sum(report["frames"] for report in reports),
        },
        "norm_stats": {
            "root": str(norm_root),
            "stats_path": str(stats_path),
            "stats_sha256": _sha256(stats_path),
            "provenance_path": str(provenance_path),
            "provenance_sha256": _sha256(provenance_path),
        },
        "semantic_sft": semantic_sft_manifest(cfg),
    }


def compute_mixture_norm_stats(
    dataset_roots: Sequence[str | Path],
    output_root: str | Path,
    *,
    block_size: int = 4096,
) -> dict[str, Any]:
    """Compute one exact transform-matched norm artifact over several datasets."""

    import openpi.shared.normalize as normalize
    import pyarrow as pa
    import pyarrow.parquet as pq
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.training.config import AssetsConfig, DataConfig, LeRobotAlohaDataConfig

    from clothdojo.training.openpi import _episode_ends, _float_matrix, _sha256

    if block_size < 1:
        raise ValueError("block_size must be positive")
    roots = [Path(value).expanduser().resolve() for value in dataset_roots]
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError("mixture normalization requires distinct dataset roots")
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"normalization output already exists: {output}")

    model_config = pi0_config.Pi0Config(pi05=True, discrete_state_input=True)
    factory = LeRobotAlohaDataConfig(
        repo_id=str(roots[0]),
        assets=AssetsConfig(assets_dir=str(output.parent), asset_id=output.name),
        base_config=DataConfig(prompt_from_task=True),
        repack_transforms=transforms.Group(),
        default_prompt="flatten and fold the shirt",
        adapt_to_pi=False,
        use_delta_joint_actions=True,
        action_sequence_keys=("action",),
    )
    data_config = factory.create(output.parent, model_config)
    horizon = int(model_config.action_horizon)
    offsets = np.arange(horizon, dtype=np.int64)
    running = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    source_records = []
    sampled_anchors = 0
    total_frames = 0
    total_episodes = 0
    for root in roots:
        info_path, sidecar_path = root / "meta/info.json", root / "meta/clothdojo.json"
        files = sorted((root / "data").glob("chunk-*/*.parquet"))
        if not files:
            raise FileNotFoundError(f"LeRobot parquet files are missing: {root / 'data'}")
        table = pq.read_table(files, columns=["observation.state", "action", "episode_index"])
        states = _float_matrix(table["observation.state"], pa)
        actions = _float_matrix(table["action"], pa)
        episode_indices = np.asarray(
            table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        episode_ends, episode_count = _episode_ends(episode_indices)
        anchors = np.flatnonzero(
            np.arange(len(states), dtype=np.int64) + horizon <= episode_ends
        )
        sampled_anchors += len(anchors)
        total_frames += len(states)
        total_episodes += episode_count
        for start in range(0, len(anchors), block_size):
            selected = anchors[start : start + block_size]
            transformed: dict[str, Any] = {
                "state": states[selected],
                "actions": actions[selected[:, None] + offsets],
                "images": {"cam_high": np.zeros((3, 1, 1), dtype=np.uint8)},
            }
            for transform in data_config.data_transforms.inputs:
                transformed = transform(transformed)
            for key in running:
                running[key].update(np.asarray(transformed[key]))
        source_records.append(
            {
                "dataset_root": str(root),
                "dataset_info_sha256": _sha256(info_path),
                "dataset_sidecar_sha256": _sha256(sidecar_path),
            }
        )
    output.mkdir(parents=True)
    normalize.save(output, {key: value.get_statistics() for key, value in running.items()})
    provenance = {
        "schema": "clothdojo_openpi_mixture_norm_stats",
        "format_version": 1,
        "sources": source_records,
        "config_name": "pi05_aloha_robotwin",
        "data_kwargs": {
            "adapt_to_pi": False,
            "extra_delta_transform": True,
            "default_prompt": "flatten and fold the shirt",
        },
        "action_horizon": horizon,
        "action_chunk_boundary": "full_only",
        "sampled_anchors": sampled_anchors,
        "frames": total_frames,
        "episodes": total_episodes,
    }
    (output / "clothdojo.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**provenance, "output_root": str(output)}
