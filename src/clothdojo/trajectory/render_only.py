"""Lossless render-state compaction for selected rollout trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.trajectory.reader import TrajectoryReader
from clothdojo.trajectory.schema import (
    RENDER_ONLY_STATE_FIELDS,
    RENDER_ONLY_STORAGE_KIND,
    SAMPLE_INDEX_DTYPE,
    STATE_ARRAY_DTYPES,
    TrajectoryState,
)
from clothdojo.trajectory.writer import TrajectoryWriter

RENDER_SELECTION_SIGNALS_SCHEMA = "clothdojo_render_selection_signals"
RENDER_SELECTION_SIGNALS_FORMAT_VERSION = 1
RENDER_SELECTION_SIGNALS_FILE = "selection_signals.npz"


class RenderOnlyTrajectoryWriter(TrajectoryWriter):
    """Write exact rerender state plus tiny strict-selection signals directly."""

    def __init__(self, output_dir: str | Path, **kwargs) -> None:
        metadata = dict(kwargs.pop("metadata", {}) or {})
        if metadata.get("training_eligible") is not False:
            raise ValueError("direct render-only trajectories must not be training eligible")
        if "storage" in metadata:
            raise ValueError("render-only writer owns trajectory storage metadata")
        metadata["storage"] = {
            "kind": RENDER_ONLY_STORAGE_KIND,
            "generation": "direct",
            "retained_state_fields": list(RENDER_ONLY_STATE_FIELDS),
            "omitted_state_fields": [
                field for field in STATE_ARRAY_DTYPES if field not in RENDER_ONLY_STATE_FIELDS
            ],
            "physics_replay_eligible": False,
            "selection_signals": {
                "schema": RENDER_SELECTION_SIGNALS_SCHEMA,
                "format_version": RENDER_SELECTION_SIGNALS_FORMAT_VERSION,
                "path": RENDER_SELECTION_SIGNALS_FILE,
            },
        }
        if "state_fields" in kwargs:
            raise TypeError("render-only writer owns state_fields")
        super().__init__(
            output_dir,
            metadata=metadata,
            state_fields=RENDER_ONLY_STATE_FIELDS,
            **kwargs,
        )
        self._selection_speed = np.empty(self.manifest.max_actions + 1, dtype=np.float32)
        self._selection_released = np.empty(self.manifest.max_actions + 1, dtype=np.bool_)
        self._selection_signal_count = 0

    def write_initial_state(self, state: TrajectoryState) -> None:
        super().write_initial_state(state)
        self._record_selection_signal(state)

    def append_transition(self, action, next_state: TrajectoryState) -> None:
        super().append_transition(action, next_state)
        self._record_selection_signal(next_state)

    def close(self) -> None:
        if self._closed:
            return
        if self._selection_signal_count != self.manifest.state_count:
            raise RuntimeError("render-only selection signal count differs from trajectory")
        _save_npz(
            self.output_dir / RENDER_SELECTION_SIGNALS_FILE,
            {
                "schema": np.asarray(RENDER_SELECTION_SIGNALS_SCHEMA),
                "format_version": np.asarray(
                    RENDER_SELECTION_SIGNALS_FORMAT_VERSION, dtype=np.int32
                ),
                "sample_index": np.arange(
                    self._selection_signal_count, dtype=SAMPLE_INDEX_DTYPE
                ),
                "speed_rms_m_s": self._selection_speed[: self._selection_signal_count],
                "released": self._selection_released[: self._selection_signal_count],
            },
        )
        super().close()

    def abort(self) -> None:
        (self.output_dir / RENDER_SELECTION_SIGNALS_FILE).unlink(missing_ok=True)
        super().abort()

    def _record_selection_signal(self, state: TrajectoryState) -> None:
        sample_index = int(state.sample_index)
        if sample_index != self._selection_signal_count:
            raise ValueError("render-only selection signal timeline is not contiguous")
        velocities = np.asarray(state.particle_qd)
        if velocities.ndim != 2 or velocities.shape[1] != 3 or not np.isfinite(
            velocities
        ).all():
            raise ValueError("particle_qd must be finite with shape (particles, 3)")
        speed = np.sqrt(
            np.square(velocities.astype(np.float64)).sum(axis=1).mean()
        )
        attached = np.asarray(state.grasp_attached)
        if attached.shape != (2,):
            raise ValueError("grasp_attached must have shape (2,)")
        self._selection_speed[sample_index] = np.float32(speed)
        self._selection_released[sample_index] = not np.any(attached != 0)
        self._selection_signal_count += 1


def compact_render_trajectory(source: str | Path, output: str | Path) -> dict[str, Any]:
    """Publish a trajectory retaining every array needed for exact rerendering.

    Particle/robot poses, gripper state, actions, and all provenance remain exact.
    Dynamics and grasp-restoration arrays are omitted, so the result is explicitly
    unsuitable for simulation restart or trajectory-metric recomputation.
    """

    source_root = Path(source).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if source_root == destination:
        raise ValueError("render-only compaction requires a distinct output directory")
    source_reader = TrajectoryReader(source_root)
    source_reader.validate_training_data()
    source_manifest_path = source_root / "manifest.json"
    source_manifest_sha256 = _file_sha256(source_manifest_path)

    if destination.exists():
        reader = TrajectoryReader(destination)
        reader.validate_render_data()
        storage = reader.manifest.metadata.get("storage", {})
        if (
            storage.get("kind") != RENDER_ONLY_STORAGE_KIND
            or storage.get("source_manifest_sha256") != source_manifest_sha256
        ):
            raise FileExistsError(
                f"existing render-only trajectory has different provenance: {destination}"
            )
        return _summary(destination, reader)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        arrays = source_reader.state_arrays(RENDER_ONLY_STATE_FIELDS)
        payload = {
            "sample_index": np.arange(
                source_reader.manifest.state_count, dtype=SAMPLE_INDEX_DTYPE
            ),
            **arrays,
        }
        _save_npz(staging / "states.npz", payload)
        if source_reader.manifest.action_count:
            shutil.copyfile(source_root / "actions.npz", staging / "actions.npz")

        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        metadata = manifest.setdefault("metadata", {})
        source_storage = metadata.pop("storage", None)
        if source_storage is not None:
            if source_storage.get("kind") not in {"derived_overlay", "derived_overlay_v1"}:
                raise ValueError("source trajectory already declares a specialized storage kind")
            metadata["source_storage"] = source_storage
        retarget = metadata.get("retarget", {})
        metrics_name = retarget.get("metrics")
        if metrics_name is not None:
            if metrics_name != "retarget_metrics.npz":
                raise ValueError("unsupported retarget metrics path")
            shutil.copyfile(source_root / metrics_name, staging / metrics_name)
        metadata["training_eligible"] = False
        metadata["storage"] = {
            "kind": RENDER_ONLY_STORAGE_KIND,
            "retained_state_fields": list(RENDER_ONLY_STATE_FIELDS),
            "omitted_state_fields": [
                field for field in STATE_ARRAY_DTYPES if field not in RENDER_ONLY_STATE_FIELDS
            ],
            "source_manifest_sha256": source_manifest_sha256,
            "physics_replay_eligible": False,
        }
        _write_json(staging / "manifest.json", manifest)
        staged_reader = TrajectoryReader(staging)
        staged_reader.validate_render_data()
        os.replace(staging, destination)
        return _summary(destination, staged_reader)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def load_render_selection_signals(trajectory: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the tiny per-state signals used only for selection."""

    root = Path(trajectory).expanduser().resolve()
    reader = TrajectoryReader(root)
    storage = reader.manifest.metadata.get("storage", {})
    descriptor = storage.get("selection_signals") if isinstance(storage, dict) else None
    expected = {
        "schema": RENDER_SELECTION_SIGNALS_SCHEMA,
        "format_version": RENDER_SELECTION_SIGNALS_FORMAT_VERSION,
        "path": RENDER_SELECTION_SIGNALS_FILE,
    }
    if descriptor != expected:
        raise ValueError("render-only trajectory is missing strict-selection signals")
    path = root / RENDER_SELECTION_SIGNALS_FILE
    required = {"schema", "format_version", "sample_index", "speed_rms_m_s", "released"}
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != required:
            raise ValueError("render-only selection signal fields differ")
        if str(data["schema"]) != RENDER_SELECTION_SIGNALS_SCHEMA:
            raise ValueError("unsupported render-only selection signal schema")
        if int(data["format_version"]) != RENDER_SELECTION_SIGNALS_FORMAT_VERSION:
            raise ValueError("unsupported render-only selection signal version")
        result = {
            "sample_index": np.asarray(data["sample_index"]).copy(),
            "speed_rms_m_s": np.asarray(data["speed_rms_m_s"]).copy(),
            "released": np.asarray(data["released"]).copy(),
        }
    count = reader.manifest.state_count
    if (
        result["sample_index"].dtype != SAMPLE_INDEX_DTYPE
        or result["sample_index"].shape != (count,)
        or not np.array_equal(
            result["sample_index"], np.arange(count, dtype=SAMPLE_INDEX_DTYPE)
        )
    ):
        raise ValueError("render-only selection signal sample indices differ")
    speed = result["speed_rms_m_s"]
    released = result["released"]
    if (
        speed.dtype != np.float32
        or speed.shape != (count,)
        or not np.isfinite(speed).all()
        or np.any(speed < 0.0)
    ):
        raise ValueError("render-only speed signals are invalid")
    if released.dtype != np.bool_ or released.shape != (count,):
        raise ValueError("render-only release signals are invalid")
    for values in result.values():
        values.setflags(write=False)
    return result


def publish_render_trajectory(source: str | Path, output: str | Path) -> dict[str, Any]:
    """Atomically publish an immutable direct render-only candidate via hard links."""

    source_root = Path(source).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if source_root == destination:
        raise ValueError("render-only publication requires a distinct output directory")
    source_reader = TrajectoryReader(source_root)
    if not source_reader.is_render_only:
        raise ValueError("render-only publication requires a render-only source")
    source_reader.validate_render_data()
    load_render_selection_signals(source_root)
    source_manifest_sha256 = _file_sha256(source_root / "manifest.json")
    if destination.exists():
        reader = TrajectoryReader(destination)
        reader.validate_render_data()
        if _file_sha256(destination / "manifest.json") != source_manifest_sha256:
            raise FileExistsError(
                f"existing render-only trajectory has different provenance: {destination}"
            )
        return _summary(destination, reader)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        for path in source_root.iterdir():
            if not path.is_file():
                raise ValueError(f"render-only candidate contains an unexpected entry: {path}")
            os.link(path, staging / path.name)
        staged_reader = TrajectoryReader(staging)
        staged_reader.validate_render_data()
        load_render_selection_signals(staging)
        os.replace(staging, destination)
        return _summary(destination, staged_reader)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _summary(path: Path, reader: TrajectoryReader) -> dict[str, Any]:
    return {
        "output": str(path),
        "state_count": reader.manifest.state_count,
        "action_count": reader.manifest.action_count,
        "storage_kind": RENDER_ONLY_STORAGE_KIND,
    }


def _save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
