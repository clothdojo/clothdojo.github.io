"""Reference OpenPI sampling and prompt binding without starting a trainer."""
from __future__ import annotations
from collections.abc import Iterator
from typing import Any
import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from clothdojo.training.openpi import TrainingPrompt


class _EpochAdvancingSampler:
    """Advance the wrapped OpenPI sampler whenever its infinite loader restarts."""

    def __init__(self, sampler: Sampler[int]) -> None:
        self._sampler = sampler
        self._next_epoch = 0

    def __iter__(self) -> Iterator[int]:
        self._sampler.set_epoch(self._next_epoch)
        self._next_epoch += 1
        return iter(self._sampler)

    def __len__(self) -> int:
        return len(self._sampler)

    def set_epoch(self, epoch: int) -> None:
        self._next_epoch = int(epoch)


class _MappedSampler(Sampler[int]):
    """Map positions from a distributed sampler to dataset-global indices."""

    def __init__(self, sampler: DistributedSampler[Any], indices: torch.Tensor) -> None:
        self._sampler = sampler
        self._indices = indices

    def __iter__(self) -> Iterator[int]:
        return (int(self._indices[position]) for position in self._sampler)

    def __len__(self) -> int:
        return len(self._sampler)

    def set_epoch(self, epoch: int) -> None:
        self._sampler.set_epoch(epoch)


def _openpi_torch_loader(data_loader: Any) -> Any:
    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None) or getattr(
        openpi_loader,
        "torch_loader",
        None,
    )
    if torch_loader is None:
        raise TypeError("OpenPI loader layout changed: inner torch DataLoader is missing")
    return torch_loader


def _set_training_prompt(data_loader: Any, prompt: str) -> None:
    """Override dataset task text before the official repack/tokenize pipeline."""

    from openpi import transforms
    from openpi.training.data_loader import TransformedDataset

    dataset = _openpi_torch_loader(data_loader).dataset
    if not isinstance(dataset, TransformedDataset):
        raise TypeError("OpenPI loader layout changed: expected TransformedDataset")
    # The official loader has not started its workers yet. Keep its dataset,
    # batching and transforms intact; only supply the configured task text first.
    dataset._transform = transforms.compose([TrainingPrompt(prompt), dataset._transform])


def _adapt_official_openpi_loader(
    data_loader: Any,
    *,
    seed: int,
    rank: int,
    action_horizon: int = 50,
    action_chunk_boundary: str = "repeat_last",
) -> None:
    """Bind actor seed and epoch advancement at OpenPI's guarded torch boundary."""

    from clothdojo.training.openpi import validate_action_chunk_boundary

    action_chunk_boundary = validate_action_chunk_boundary(action_chunk_boundary)

    torch_loader = _openpi_torch_loader(data_loader)
    batch_sampler = getattr(torch_loader, "batch_sampler", None)
    sampler = getattr(batch_sampler, "sampler", None)
    if not isinstance(sampler, DistributedSampler):
        raise TypeError(
            "OpenPI/RLinf SFT requires a DistributedSampler after process-group "
            f"initialization, got {type(sampler).__name__}"
        )
    sampler.seed = int(seed)
    adapted_sampler: Sampler[int] = sampler
    if action_chunk_boundary == "full_only":
        valid_indices = _full_only_anchor_indices(
            torch_loader.dataset,
            action_horizon=action_horizon,
        )
        distributed = DistributedSampler(
            valid_indices,
            num_replicas=sampler.num_replicas,
            rank=sampler.rank,
            shuffle=sampler.shuffle,
            seed=int(seed),
            drop_last=sampler.drop_last,
        )
        adapted_sampler = _MappedSampler(distributed, valid_indices)
    batch_sampler.sampler = _EpochAdvancingSampler(adapted_sampler)
    generator = getattr(torch_loader, "generator", None)
    if generator is not None:
        generator.manual_seed(int(seed) + int(rank))


def _full_only_anchor_indices(dataset: Any, *, action_horizon: int) -> torch.Tensor:
    """Return global frame indices whose H-step chunk stays in one episode."""

    if isinstance(action_horizon, bool) or action_horizon < 1:
        raise ValueError("action_horizon must be a positive integer")
    while hasattr(dataset, "_dataset"):
        dataset = dataset._dataset
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if not isinstance(episode_data_index, dict):
        raise TypeError(
            "OpenPI full_only requires the underlying LeRobot episode_data_index"
        )
    if "from" not in episode_data_index or "to" not in episode_data_index:
        raise ValueError("LeRobot episode_data_index is missing from/to arrays")
    starts = torch.as_tensor(episode_data_index["from"], dtype=torch.int64)
    stops = torch.as_tensor(episode_data_index["to"], dtype=torch.int64)
    if starts.ndim != 1 or stops.ndim != 1 or starts.shape != stops.shape:
        raise ValueError("LeRobot episode_data_index must contain aligned 1D from/to arrays")
    chunks = []
    previous_stop = 0
    for start_value, stop_value in zip(starts, stops, strict=True):
        start = int(start_value)
        stop = int(stop_value)
        if start < previous_stop or stop < start:
            raise ValueError("LeRobot episode ranges overlap or are reversed")
        previous_stop = stop
        if stop - start >= action_horizon:
            chunks.append(
                torch.arange(start, stop - action_horizon + 1, dtype=torch.int64)
            )
    if not chunks:
        raise ValueError(
            f"dataset has no full action chunks for action horizon {action_horizon}"
        )
    return torch.cat(chunks)
