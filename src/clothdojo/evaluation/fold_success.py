"""Exact sustained folding success on the saved control-rate state sequence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from clothdojo.evaluation.fold_quality import FoldQualityMetric

SUCCESS_HOLD_SECONDS = 3.0


@dataclass(frozen=True)
class FoldSuccess:
    success: bool
    success_at_end: bool
    witness_start_step: int | None
    witness_end_step: int | None


def fold_success(
    state_count: int,
    control_fps: float,
    score_at: Callable[[int], float],
    *,
    known_scores: Mapping[int, float] | None = None,
) -> FoldSuccess:
    """Require a passing run spanning >=3 s; at 50 Hz this means 151 states.

    The callback must score the requested saved state with the current Q model.
    Coarse probes can reject windows but never certify them: every state of a
    successful witness is checked. A witness need not be the earliest success.
    """
    if type(state_count) is not int or state_count < 1:
        raise ValueError("state_count must be a positive integer")
    if not np.isfinite(control_fps) or control_fps <= 0:
        raise ValueError("control_fps must be finite and positive")
    hold = SUCCESS_HOLD_SECONDS * control_fps
    span = int(round(hold))
    if span < 1 or not np.isclose(hold, span, rtol=0, atol=1e-9):
        raise ValueError("3 seconds must resolve to an exact positive control step count")
    metric = FoldQualityMetric.load()
    scores = dict(known_scores or {})
    if any(type(k) is not int or not 0 <= k < state_count or not np.isfinite(v)
           for k, v in scores.items()):
        raise ValueError("known scores must have valid state indices and finite Q")

    def passing(step):
        if step not in scores:
            scores[step] = float(score_at(step))
        return bool(metric.passes_threshold(scores[step]))

    def reject(start, end):
        """Return a failing index, or None only after checking the entire window."""
        failures = [s for s, q in scores.items()
                    if start <= s <= end and q < metric.success_threshold]
        if failures:
            return max(failures)
        # Probe late/coarse points first to reject impossible windows cheaply.
        # The final range always checks every remaining saved state.
        probes = dict.fromkeys([end, start, *range(start, end + 1, max(1, round(control_fps))),
                               *range(start, end + 1, max(1, round(control_fps / 5))),
                               *range(start, end + 1)])
        for step in probes:
            if not passing(step):
                return step
        return None

    last = state_count - 1
    if last < span:
        return FoldSuccess(False, False, None, None)
    if reject(last - span, last) is None:
        return FoldSuccess(True, True, last - span, last)
    start = 0
    while start + span <= last:
        failed = reject(start, start + span)
        if failed is None:
            return FoldSuccess(True, False, start, start + span)
        # Every window starting between start and failed contains this failure.
        start = failed + 1
    return FoldSuccess(False, False, None, None)


def earliest_fold_success(
    state_count: int,
    control_fps: float,
    score_at: Callable[[int], float],
    *,
    known_scores: Mapping[int, float] | None = None,
) -> FoldSuccess:
    """Return the earliest exact 3-s witness and terminal-success status.

    Unlike :func:`fold_success`, this function never accepts a later witness
    before excluding every earlier window.  Known failing scores and coarse
    probes are used only to skip windows that cannot succeed; a positive
    witness is still checked at every saved control-rate state.
    """
    if type(state_count) is not int or state_count < 1:
        raise ValueError("state_count must be a positive integer")
    if not np.isfinite(control_fps) or control_fps <= 0:
        raise ValueError("control_fps must be finite and positive")
    hold = SUCCESS_HOLD_SECONDS * control_fps
    span = int(round(hold))
    if span < 1 or not np.isclose(hold, span, rtol=0, atol=1e-9):
        raise ValueError("3 seconds must resolve to an exact positive control step count")
    metric = FoldQualityMetric.load()
    scores = dict(known_scores or {})
    if any(type(k) is not int or not 0 <= k < state_count or not np.isfinite(v)
           for k, v in scores.items()):
        raise ValueError("known scores must have valid state indices and finite Q")

    def passing(step: int) -> bool:
        if step not in scores:
            scores[step] = float(score_at(step))
        return bool(metric.passes_threshold(scores[step]))

    def reject(start: int, end: int) -> int | None:
        failures = [step for step, score in scores.items()
                    if start <= step <= end and score < metric.success_threshold]
        if failures:
            return max(failures)
        probes = dict.fromkeys([
            end,
            start,
            *range(start, end + 1, max(1, round(control_fps))),
            *range(start, end + 1, max(1, round(control_fps / 5))),
            *range(start, end + 1),
        ])
        for step in probes:
            if not passing(step):
                return step
        return None

    last = state_count - 1
    if last < span:
        return FoldSuccess(False, False, None, None)
    success_at_end = reject(last - span, last) is None
    start = 0
    while start + span <= last:
        failed = reject(start, start + span)
        if failed is None:
            return FoldSuccess(True, success_at_end, start, start + span)
        start = failed + 1
    return FoldSuccess(False, success_at_end, None, None)
