"""Offline evaluation derived solely from canonical trajectories."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from clothdojo.evaluation.metric import (
    CanonicalTemplate,
    cloth_airborne_mask,
    factorized_cloth_distance,
)
from clothdojo.evaluation.protocol import EvaluationProtocol
from clothdojo.evaluation.results import (
    EvaluationCaseKey,
    EvaluationCaseResult,
)
from clothdojo.trajectory import TrajectoryReader, load_render_selection_signals

METRIC_NAMES = (
    "deformable_distance",
    "rigid_distance",
    "combined_distance",
    "speed_rms_m_s",
)


@dataclass(frozen=True, slots=True)
class ScoredTrajectory:
    result: EvaluationCaseResult
    sample_steps: np.ndarray
    curve: np.ndarray


def score_evaluation_trajectory(
    trajectory: str | Path,
    template: CanonicalTemplate,
    protocol: EvaluationProtocol,
    *,
    metric_batch_size: int = 32,
    assets_root: str | Path = "assets",
) -> ScoredTrajectory:
    """Score one complete trajectory without Newton, a policy or a GPU."""

    return _score_trajectory(
        trajectory,
        template,
        protocol,
        metric_batch_size=metric_batch_size,
        required_training_eligible=False,
        assets_root=assets_root,
    )


def score_rollout_trajectory(
    trajectory: str | Path,
    template: CanonicalTemplate,
    protocol: EvaluationProtocol,
    *,
    metric_batch_size: int = 32,
) -> ScoredTrajectory:
    """Score a canonical training rollout with the benchmark metric."""

    return _score_trajectory(
        trajectory,
        template,
        protocol,
        metric_batch_size=metric_batch_size,
        required_training_eligible=True,
    )


def score_render_selection_trajectory(
    trajectory: str | Path,
    template: CanonicalTemplate,
    protocol: EvaluationProtocol,
    *,
    metric_batch_size: int = 32,
) -> ScoredTrajectory:
    """Score a direct render-only candidate with the evaluation scorer."""

    return score_evaluation_trajectory(
        trajectory,
        template,
        protocol,
        metric_batch_size=metric_batch_size,
    )


def _score_trajectory(
    trajectory: str | Path,
    template: CanonicalTemplate,
    protocol: EvaluationProtocol,
    *,
    metric_batch_size: int,
    required_training_eligible: bool,
    assets_root: str | Path = "assets",
) -> ScoredTrajectory:

    if metric_batch_size < 1:
        raise ValueError("metric_batch_size must be positive")
    reader = TrajectoryReader(trajectory)
    manifest = reader.manifest
    manifest_task = getattr(getattr(manifest, "scene", None), "task_name", None)
    if manifest_task != "shirt_flatten_fold" and (manifest_task == "shirt_fold") != protocol.is_fold:
        raise ValueError("evaluation trajectory and protocol tasks differ")
    if manifest.control_fps != protocol.control_fps:
        raise ValueError("evaluation trajectory and protocol control_fps differ")
    if manifest.action_count < protocol.horizon_steps:
        raise ValueError("evaluation trajectory does not reach the protocol horizon")
    if manifest.state_count < protocol.horizon_steps + 1:
        raise ValueError("evaluation trajectory does not reach the protocol horizon state")
    metadata = manifest.metadata
    if metadata.get("training_eligible") is not required_training_eligible:
        kind = "training rollout" if required_training_eligible else "evaluation trajectory"
        raise ValueError(
            f"{kind} must be marked training_eligible="
            f"{str(required_training_eligible).lower()}"
        )
    if metadata.get("asset_id") != template.asset_id:
        raise ValueError("evaluation trajectory and canonical template asset differ")

    speed_rms_override = (
        load_render_selection_signals(trajectory)["speed_rms_m_s"]
        if reader.is_render_only else None
    )
    arrays = reader.state_arrays(
        ("particle_q",) if speed_rms_override is not None else ("particle_q", "particle_qd"),
        mmap=True,
    )
    if speed_rms_override is not None:
        speed_rms_override = np.asarray(speed_rms_override, dtype=np.float32)
        if (
            speed_rms_override.shape != (manifest.state_count,)
            or not np.isfinite(speed_rms_override).all()
            or np.any(speed_rms_override < 0.0)
        ):
            raise ValueError("trajectory speed RMS override is invalid")
    sample_steps = np.arange(
        0,
        protocol.horizon_steps + 1,
        protocol.metric_stride,
        dtype=np.int32,
    )
    if protocol.is_fold:
        return _score_fold_trajectory(
            arrays, template, protocol, metadata, sample_steps,
            assets_root=assets_root, speed_rms_override=speed_rms_override,
        )
    curve = np.empty((protocol.sample_count, len(METRIC_NAMES)), dtype=np.float32)
    airborne = np.empty(protocol.sample_count, dtype=np.float64)
    for start in range(0, protocol.sample_count, metric_batch_size):
        stop = min(start + metric_batch_size, protocol.sample_count)
        selection = sample_steps[start:stop]
        positions = arrays["particle_q"][selection]
        airborne[start:stop] = cloth_airborne_mask(positions, template)
        values = factorized_cloth_distance(
            positions,
            (
                np.zeros_like(positions)
                if speed_rms_override is not None
                else arrays["particle_qd"][selection]
            ),
            template,
            deformable_weight=protocol.deformable_weight,
            symmetry=protocol.symmetry,
            symmetry_center_xy=protocol.target.symmetry_center_xy,
        )
        curve[start:stop] = np.stack(
            (
                values.deformable_distance,
                values.rigid_distance,
                values.combined_distance,
                values.speed_rms_m_s,
            ),
            axis=1,
        )
        if speed_rms_override is not None:
            curve[start:stop, 3] = speed_rms_override[selection]

    replicate = metadata.get(
        "replicate",
        0 if required_training_eligible or speed_rms_override is not None else None,
    )
    key = EvaluationCaseKey(
        str(metadata["asset_id"]),
        _non_negative_integer(metadata["initial_state"]["index"], "initial_state index"),
        _non_negative_integer(replicate, "replicate"),
    )
    result = result_from_curve(
        curve,
        protocol,
        key=key,
        initial_state_seed=_non_negative_integer(
            metadata["initial_state"]["seed"], "initial_state seed"
        ),
        policy_seed=_non_negative_integer(metadata["policy_seed"], "policy seed"),
    )
    # Integrate floating indicators: NumPy boolean addition is logical OR,
    # which would incorrectly halve the duration of consecutive airborne frames.
    result = replace(
        result,
        airborne_seconds=float(np.trapz(airborne, dx=1.0 / protocol.metric_hz)),
    )
    sample_steps.setflags(write=False)
    curve.setflags(write=False)
    return ScoredTrajectory(result=result, sample_steps=sample_steps, curve=curve)


def _score_fold_trajectory(
    arrays: dict[str, np.ndarray],
    template: CanonicalTemplate,
    protocol: EvaluationProtocol,
    metadata: dict,
    sample_steps: np.ndarray,
    *,
    assets_root: str | Path,
    speed_rms_override: np.ndarray | None,
) -> ScoredTrajectory:
    """Sample Q diagnostics and certify success on the saved control-rate states."""
    from clothdojo.evaluation.fold_quality import (
        FoldQualityMetric,
        load_fold_quality_triangles,
        prepare_fold_quality_reference,
    )
    from clothdojo.evaluation.fold_success import earliest_fold_success

    reference = prepare_fold_quality_reference(
        template.particle_q,
        load_fold_quality_triangles(str(Path(assets_root).resolve()), template.asset_id),
    )
    metric = FoldQualityMetric.load()
    scores = np.asarray([
        metric.score(arrays["particle_q"][step], reference) for step in sample_steps
    ])
    outcome = earliest_fold_success(
        protocol.horizon_steps + 1,
        protocol.control_fps,
        lambda step: metric.score(arrays["particle_q"][step], reference),
        known_scores={int(step): float(q) for step, q in zip(sample_steps, scores, strict=True)},
    )
    if speed_rms_override is None:
        velocity = arrays["particle_qd"][sample_steps]
        speeds = np.sqrt(np.mean(np.sum(velocity.astype(np.float64) ** 2, axis=-1), axis=-1))
    else:
        speeds = speed_rms_override[sample_steps]
    best = int(scores.argmax())
    result = EvaluationCaseResult(
        key=EvaluationCaseKey(
            str(metadata["asset_id"]),
            _non_negative_integer(metadata["initial_state"]["index"], "initial_state index"),
            _non_negative_integer(metadata.get("replicate", 0), "replicate"),
        ),
        initial_state_seed=_non_negative_integer(
            metadata["initial_state"]["seed"], "initial_state seed"
        ),
        policy_seed=_non_negative_integer(metadata["policy_seed"], "policy seed"),
        success=outcome.success,
        success_at_end=outcome.success_at_end,
        confirmed_step=outcome.witness_end_step,
        terminal_deformable_distance=None,
        terminal_rigid_distance=None,
        terminal_combined_distance=None,
        terminal_speed_rms_m_s=float(speeds[-1]),
        best_observed_step=int(sample_steps[best]),
        best_observed_deformable_distance=None,
        best_observed_rigid_distance=None,
        best_observed_combined_distance=None,
        terminal_fold_quality=float(scores[-1]),
        best_observed_fold_quality=float(scores[best]),
        time_average_fold_quality=float(
            np.trapz(scores, dx=protocol.metric_stride) / protocol.horizon_steps
        ),
    )
    curve = np.column_stack((scores, speeds))
    sample_steps.setflags(write=False)
    curve.setflags(write=False)
    return ScoredTrajectory(result=result, sample_steps=sample_steps, curve=curve)


def result_from_curve(
    curve: np.ndarray,
    protocol: EvaluationProtocol,
    *,
    key: EvaluationCaseKey,
    initial_state_seed: int,
    policy_seed: int,
) -> EvaluationCaseResult:
    """Derive terminal, best-observed and absorbing success from one curve."""

    values = np.asarray(curve, dtype=np.float32)
    expected = (protocol.sample_count, len(METRIC_NAMES))
    if values.shape != expected or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"evaluation curve must be finite and non-negative with shape {expected}")
    deformable = values[:, 0]
    rigid = values[:, 1]
    combined = values[:, 2]
    confirmed = protocol.continuous_success_mask(deformable, rigid)
    success = bool(confirmed.any())
    confirmed_step = (
        int(confirmed.argmax()) * protocol.metric_stride if success else None
    )
    best = int(combined.argmin())
    return EvaluationCaseResult(
        key=key,
        initial_state_seed=int(initial_state_seed),
        policy_seed=int(policy_seed),
        success=success,
        success_at_end=bool(confirmed[-1]),
        confirmed_step=confirmed_step,
        terminal_deformable_distance=float(deformable[-1]),
        terminal_rigid_distance=float(rigid[-1]),
        terminal_combined_distance=float(combined[-1]),
        terminal_speed_rms_m_s=float(values[-1, 3]),
        best_observed_step=best * protocol.metric_stride,
        best_observed_deformable_distance=float(deformable[best]),
        best_observed_rigid_distance=float(rigid[best]),
        best_observed_combined_distance=float(combined[best]),
        time_average_combined_distance=float(
            np.trapz(combined, dx=protocol.metric_stride)
            / protocol.horizon_steps
        ),
    )


def _non_negative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
