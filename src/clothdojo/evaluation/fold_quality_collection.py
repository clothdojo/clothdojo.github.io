"""Offline continuous fold-quality scoring for canonical trajectories."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

from clothdojo.evaluation.fold_quality import (
    FoldQualityMetric,
    load_fold_quality_triangles,
    prepare_fold_quality_reference,
)
from clothdojo.evaluation.fold_success import SUCCESS_HOLD_SECONDS, fold_success
from clothdojo.evaluation.metric import load_canonical_template
from clothdojo.trajectory import TrajectoryReader


@cache
def _metric() -> FoldQualityMetric:
    return FoldQualityMetric.load()


def _score_trajectory(job: tuple[str, str, int, str]) -> dict:
    path, assets_root, stride, reference_path = job
    reader = TrajectoryReader(path)
    manifest = reader.manifest
    if manifest.scene.task_name not in {"shirt_fold", "shirt_flatten_fold"}:
        raise ValueError(f"fold quality requires a fold-capable trajectory: {path}")
    asset = str(manifest.metadata["asset_id"])
    source = Path(reference_path)
    if source.is_dir():
        source = source / f"{asset}.npz"
    template = load_canonical_template(source)
    if template.asset_id != asset:
        raise ValueError(f"fold reference and trajectory asset differ: {path}")
    arrays = reader.state_arrays(("particle_q",), mmap=True)
    ref = prepare_fold_quality_reference(
        template.particle_q, load_fold_quality_triangles(assets_root, asset)
    )
    metric = _metric()
    final = manifest.state_count - 1
    steps = sorted(set(range(0, final + 1, stride)) | {final})
    scores = np.array(
        [
            metric.score(arrays["particle_q"][s], ref)
            for s in steps
        ]
    )
    times = np.array(steps) / manifest.control_fps
    peak = float(scores.max())
    outcome = fold_success(
        manifest.state_count, manifest.control_fps,
        lambda step: metric.score(arrays["particle_q"][step], ref),
        known_scores=dict(zip(steps, scores, strict=True)),
    )
    # Report every exactly tied maximum; no latest-frame tie preference.
    best_steps = [s for s, q in zip(steps, scores, strict=True) if q == peak]
    duration = float(times[-1])
    return {
        "trajectory_id": Path(path).name,
        "asset_id": asset,
        "metric": "fold_quality",
        "reference": str(source.resolve()),
        "success_threshold": metric.success_threshold,
        "success_hold_seconds": SUCCESS_HOLD_SECONDS,
        "success": outcome.success,
        "success_at_end": outcome.success_at_end,
        "success_witness_start_step": outcome.witness_start_step,
        "success_witness_end_step": outcome.witness_end_step,
        "control_fps": manifest.control_fps,
        "sample_stride": stride,
        "duration_seconds": duration,
        "steps": steps,
        "scores": scores.tolist(),
        "initial_score": float(scores[0]),
        "final_score": float(scores[-1]),
        "net_gain": float(scores[-1] - scores[0]),
        "peak_score": peak,
        "best_steps": best_steps,
        "end_drop": float(peak - scores[-1]),
        "time_mean_score": float(np.trapz(scores, times) / duration)
        if duration
        else float(scores[0]),
    }


def score_fold_quality_collection(
    source: str | Path,
    output: str | Path,
    *,
    reference: str | Path,
    assets_root: str | Path = "assets",
    sample_stride: int = 1,
    workers: int = 1,
) -> dict:
    """Score saved states on CPU using existing verified flat reference NPZs.

    Q summaries use sample_stride. Success always checks a full control-rate
    witness spanning 3 seconds, even when the reported curve is subsampled.
    Output is a new directory so earlier results are not overwritten.
    """

    if type(sample_stride) is not int or sample_stride < 1:
        raise ValueError("sample_stride must be a positive integer")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    root = Path(source).expanduser().resolve()
    if (root / "manifest.json").is_file():
        paths = [root]
    else:
        trajectories = root / "trajectories" if (root / "trajectories").is_dir() else root
        paths = sorted(p.parent for p in trajectories.glob("*/manifest.json"))
    if not paths:
        raise ValueError(f"no canonical trajectories found: {root}")
    destination = Path(output).expanduser().resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("score output must be outside the source collection")
    reference_path = Path(reference).expanduser().resolve()
    if not reference_path.exists():
        raise ValueError(f"fold reference does not exist: {reference_path}")
    jobs = [(str(p), str(Path(assets_root).expanduser().resolve()), sample_stride,
             str(reference_path)) for p in paths]
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    with (destination / "scores.jsonl").open("w") as stream:
        if workers == 1:
            for job in jobs:
                r = _score_trajectory(job)
                rows.append(r)
                stream.write(json.dumps(r) + "\n")
                stream.flush()
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for r in executor.map(_score_trajectory, jobs):
                    rows.append(r)
                    stream.write(json.dumps(r) + "\n")
                    stream.flush()
    result = {
        "trajectories": len(rows),
        "scored_states": sum(len(r["steps"]) for r in rows),
        "sample_stride": sample_stride,
        "scores": str(destination / "scores.jsonl"),
        "metric": "fold_quality",
        "success_threshold": _metric().success_threshold,
        "successes": sum(r["success"] for r in rows),
        "success_rate": sum(r["success"] for r in rows) / len(rows),
        "success_hold_seconds": SUCCESS_HOLD_SECONDS,
        "successes_at_end": sum(r["success_at_end"] for r in rows),
        "success_rate_at_end": sum(r["success_at_end"] for r in rows) / len(rows),
        "mean_final_score": float(np.mean([r["final_score"] for r in rows])),
        "semantics": (
            "Success requires Q >= threshold continuously for 3 s at saved control rate; "
            "success_at_end requires the final 3 s to pass."
        ),
    }
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
