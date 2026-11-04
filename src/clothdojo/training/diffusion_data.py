"""Balanced, task-unconditioned windows from existing LeRobot v2.1 episodes."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from clothdojo.rendering import CAMERA_NAMES

IMAGE_KEYS = tuple(f"observation.images.{name}" for name in CAMERA_NAMES)


def resize_images(images: torch.Tensor, size: int) -> torch.Tensor:
    """Use the existing policy transport's uint8 letterbox on both data and rollout."""
    from openpi_client.image_tools import resize_with_pad

    array = images.mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    array = resize_with_pad(array, size, size)
    return torch.from_numpy(np.array(array, copy=True)).permute(0, 3, 1, 2).float() / 255


def window_indices(anchor: int, length: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    # Actions start at the oldest observation, as required by upstream DiffusionPolicy.
    if not 0 <= anchor <= length - horizon + 1:
        raise ValueError("anchor must have a complete future action window")
    observations = np.maximum([anchor - 1, anchor], 0)
    actions = np.maximum(np.arange(anchor - 1, anchor - 1 + horizon), 0)
    return observations, actions


class DiffusionDataset(Dataset):
    """Read only three policy cameras; use upstream video decoding and Parquet layout.

    Episode-local windows never splice independent skills or cross correction boundaries.
    Sampling is uniform over valid anchors within each of the two equally weighted sources.
    No task names, phase labels, timestamps, or episode IDs are model inputs.
    """

    def __init__(self, project: Path, cfg: dict):
        self.cfg = cfg
        self.roots = [project / item for item in cfg["datasets"]]
        self.infos = []
        self.episodes = []
        self.cumulative = []
        self.provenance = []
        for root in self.roots:
            info = json.loads((root / "meta/info.json").read_text())
            if info["codebase_version"] != "v2.1" or info["fps"] != cfg["fps"]:
                raise ValueError(f"expected LeRobot v2.1 at {cfg['fps']} Hz: {root}")
            for key in ("action", "observation.state"):
                if info["features"][key]["shape"] != [14]:
                    raise ValueError(f"expected 14-dimensional {key}")
            episodes = [
                json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()
            ]
            counts = [max(0, ep["length"] - cfg["horizon"] + 2) for ep in episodes]
            self.infos.append(info)
            self.episodes.append(episodes)
            self.cumulative.append(np.cumsum(counts))
            self.provenance.append(
                {
                    "root": str(root),
                    "episodes": len(episodes),
                    "frames": info["total_frames"],
                    "valid_anchors": int(sum(counts)),
                    "metadata_sha256": {
                        name: hashlib.sha256((root / "meta" / name).read_bytes()).hexdigest()
                        for name in (
                            "info.json",
                            "episodes.jsonl",
                            "episodes_stats.jsonl",
                            "clothdojo.json",
                        )
                    },
                }
            )
        if len(self.roots) not in (1, 2) or any(
            len(values) == 0 or values[-1] == 0 for values in self.cumulative
        ):
            raise ValueError("diffusion requires one or two nonempty sources")

    def __len__(self):
        return sum(int(values[-1]) for values in self.cumulative)

    @functools.lru_cache(maxsize=8)  # noqa: B019 -- one dataset for each worker's lifetime
    def _episode_arrays(self, source: int, episode: int):
        import pyarrow.parquet as pq

        info = self.infos[source]
        path = self.roots[source] / info["data_path"].format(
            episode_chunk=episode // info["chunks_size"],
            episode_index=episode,
        )
        table = pq.read_table(path, columns=["observation.state", "action", "timestamp"])
        result = {
            key: np.asarray(table[key].to_pylist(), dtype=np.float32) for key in table.column_names
        }
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError(f"nonfinite data in {path}")
        return result

    def __getitem__(self, index):
        from lerobot.common.datasets.video_utils import decode_video_frames

        source, position = index
        cumulative = self.cumulative[source]
        ordinal = int(np.searchsorted(cumulative, position, side="right"))
        anchor = int(position - (cumulative[ordinal - 1] if ordinal else 0))
        episode = self.episodes[source][ordinal]
        episode_id = episode["episode_index"]
        observations, actions = window_indices(anchor, episode["length"], self.cfg["horizon"])
        arrays = self._episode_arrays(source, episode_id)
        result = {
            "observation.state": torch.from_numpy(arrays["observation.state"][observations].copy()),
            "action": torch.from_numpy(arrays["action"][actions].copy()),
            "action_is_pad": torch.tensor([anchor == 0] + [False] * (self.cfg["horizon"] - 1)),
        }
        info = self.infos[source]
        for key in IMAGE_KEYS:
            path = self.roots[source] / info["video_path"].format(
                episode_chunk=episode_id // info["chunks_size"],
                episode_index=episode_id,
                video_key=key,
            )
            images = decode_video_frames(
                path,
                arrays["timestamp"][observations].tolist(),
                tolerance_s=1e-4,
                backend="pyav",
            )
            result[key] = resize_images(images, self.cfg["image_size"])
        return result

    def normalization_stats(self):
        result = {}
        for key in ("observation.state", "action"):
            values = []
            for root in self.roots:
                with (root / "meta/episodes_stats.jsonl").open() as stream:
                    values.extend(json.loads(line)["stats"][key] for line in stream)
            result[key] = {
                "min": torch.tensor(
                    np.min([v["min"] for v in values], axis=0), dtype=torch.float32
                ),
                "max": torch.tensor(
                    np.max([v["max"] for v in values], axis=0), dtype=torch.float32
                ),
                # Upstream Normalize dispatches on the type of mean even for MIN_MAX.
                "mean": torch.zeros(14),
            }
            if not torch.all(result[key]["max"] > result[key]["min"]):
                raise ValueError(f"degenerate normalization: {key}")
        for key in IMAGE_KEYS:
            result[key] = {
                "mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
                "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
            }
        return result


class BalancedWindowSampler(Sampler):
    """Uniform draws per source; two sources receive equal per-rank batch shares."""

    def __init__(self, dataset, *, rank, start_step, steps, batch_size, seed):
        self.sizes = [int(v[-1]) for v in dataset.cumulative]
        if len(self.sizes) not in (1, 2) or batch_size % len(self.sizes):
            raise ValueError("micro batch must divide evenly across one or two sources")
        self.rank, self.start, self.steps = rank, start_step, steps
        self.batch_size, self.seed = batch_size, seed

    def __len__(self):
        return (self.steps - self.start) * self.batch_size

    def __iter__(self):
        for step in range(self.start, self.steps):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.rank, step]))
            batch = [
                (source, int(position))
                for source, size in enumerate(self.sizes)
                for position in rng.integers(size, size=self.batch_size // len(self.sizes))
            ]
            rng.shuffle(batch)
            yield from batch
