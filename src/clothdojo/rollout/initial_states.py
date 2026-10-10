"""Portable initial-state pools shared by rollout and evaluation."""
from __future__ import annotations
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
INITIAL_STATE_SCHEMA = "clothdojo_initial_state_pool"
INITIAL_STATE_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class InitialStatePool:
    particle_q: np.ndarray
    particle_qd: np.ndarray
    seeds: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        q = np.asarray(self.particle_q)
        qd = np.asarray(self.particle_qd)
        seeds = np.asarray(self.seeds)
        if q.ndim != 3 or q.shape[-1] != 3:
            raise ValueError(f"particle_q must have shape (states, particles, 3), got {q.shape}")
        if qd.shape != q.shape:
            raise ValueError("particle_qd shape must match particle_q")
        if q.dtype != np.float32 or qd.dtype != np.float32:
            raise TypeError("initial particle states must use float32")
        if seeds.shape != (len(q),) or seeds.dtype != np.int32:
            raise ValueError("seeds must be int32 with one value per state")
        if len(q) < 1 or len(np.unique(seeds)) != len(seeds):
            raise ValueError("initial-state pool must contain unique non-empty seeds")
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("initial-state pool contains non-finite values")
        if not isinstance(self.metadata, dict):
            raise TypeError("initial-state metadata must be a dict")

    @property
    def state_count(self) -> int:
        return len(self.seeds)

    @property
    def particle_count(self) -> int:
        return int(self.particle_q.shape[1])

    def select(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError("initial-state indices must be a vector")
        if np.any(values < 0) or np.any(values >= self.state_count):
            raise IndexError("initial-state index is outside the pool")
        return (
            np.ascontiguousarray(self.particle_q[values]),
            np.ascontiguousarray(self.particle_qd[values]),
            np.ascontiguousarray(self.seeds[values]),
        )

    def save(self, output: str | Path) -> Path:
        destination = Path(output).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"initial-state pool already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self.metadata,
            "schema": INITIAL_STATE_SCHEMA,
            "format_version": INITIAL_STATE_FORMAT_VERSION,
            "state_count": self.state_count,
            "particle_count": self.particle_count,
        }
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    particle_q=self.particle_q,
                    particle_qd=self.particle_qd,
                    seeds=self.seeds,
                    metadata_json=np.asarray(
                        json.dumps(payload, sort_keys=True, separators=(",", ":"))
                    ),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, source: str | Path) -> InitialStatePool:
        path = Path(source).expanduser().resolve()
        with np.load(path, allow_pickle=False) as data:
            required = {"particle_q", "particle_qd", "seeds", "metadata_json"}
            if set(data.files) != required:
                raise ValueError(f"initial-state fields differ: {sorted(data.files)}")
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata.pop("schema", None) != INITIAL_STATE_SCHEMA:
                raise ValueError("unsupported initial-state pool schema")
            if metadata.pop("format_version", None) != INITIAL_STATE_FORMAT_VERSION:
                raise ValueError("unsupported initial-state pool version")
            state_count = metadata.pop("state_count", None)
            particle_count = metadata.pop("particle_count", None)
            pool = cls(
                particle_q=np.asarray(data["particle_q"], dtype=np.float32),
                particle_qd=np.asarray(data["particle_qd"], dtype=np.float32),
                seeds=np.asarray(data["seeds"], dtype=np.int32),
                metadata=metadata,
            )
        if state_count != pool.state_count or particle_count != pool.particle_count:
            raise ValueError("initial-state metadata disagrees with its arrays")
        return pool


def validate_pool_compatibility(
    pool: InitialStatePool,
    *,
    garment_path: str | Path,
    particle_count: int,
) -> None:
    if pool.particle_count != int(particle_count):
        raise ValueError(
            "initial-state particle count differs from the garment: "
            f"{pool.particle_count} != {particle_count}"
        )
    expected = pool.metadata.get("garment_sha256")
    if not isinstance(expected, str) or _sha256(Path(garment_path)) != expected:
        raise ValueError("initial-state pool belongs to a different garment asset")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
