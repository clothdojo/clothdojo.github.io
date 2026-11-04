"""Offline metric results used only to prioritize human review."""
import json
import math
from functools import cache
from pathlib import Path

from clothdojo.evaluation.fold_success import SUCCESS_HOLD_SECONDS


@cache
def _fold_threshold() -> float:
    from clothdojo.evaluation.fold_quality import FoldQualityMetric

    return FoldQualityMetric.load().success_threshold


def released_samples(trajectory: Path, steps: list[int]) -> list[bool]:
    """True only when neither gripper has an active cloth attachment."""
    import numpy as np

    from clothdojo.trajectory import TrajectoryReader

    arrays = TrajectoryReader(trajectory).state_arrays(("grasp_attached",), mmap=True)
    return (~np.any(arrays["grasp_attached"][steps] != 0, axis=1)).tolist()


def load_metric_review(root: Path) -> dict:
    path = root / "review/metrics.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != "clothdojo_hil_metrics":
        return {}
    return value


def episode_metric_review(review: dict, episode_id: str, action_count: int) -> dict | None:
    row = review.get("episodes", {}).get(episode_id)
    if not isinstance(row, dict) or row.get("action_count") != action_count:
        return None
    metric = review.get("metric")
    thresholds = review.get("thresholds", {})
    # Old score files use a different scale; never reinterpret them with current Q.
    if metric == "fold_quality" and (
        thresholds.get("fold_quality_min") != _fold_threshold()
        or thresholds.get("success_hold_seconds") != SUCCESS_HOLD_SECONDS
        or row.get("success_hold_seconds") != SUCCESS_HOLD_SECONDS
        or type(row.get("success")) is not bool
    ):
        return None
    fields = (
        ("peak_score", "fold_quality_min")
        if metric == "fold_quality"
        else ("deformable_distance", "deformable_distance_max",
              "rigid_distance", "rigid_distance_max")
        if metric == "factorized_cloth_distance" else ()
    )
    if not fields:
        return None
    best_index = row.get("best_sample_index")
    if metric == "fold_quality" and (
        isinstance(best_index, bool) or not isinstance(best_index, int)
        or not 0 <= best_index <= action_count
    ):
        return None
    pairs = [(row.get(fields[i]), thresholds.get(fields[i+1])) for i in range(0, len(fields), 2)]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for pair in pairs for v in pair):
        return None
    attention = (not row["success"] if metric == "fold_quality"
                 else any(value > threshold for value, threshold in pairs))
    return {"metric": metric, "needs_attention": attention, "values": row,
            "thresholds": thresholds}


def fold_peak_review_row(score: dict, eligible: list[bool] | None = None) -> dict:
    """Recommend a completed 3 s hold, or the earliest maximum if none qualify."""
    threshold = _fold_threshold()
    if score.get("success_threshold") != threshold:
        raise ValueError("fold score must be recomputed with the current metric")
    steps, values = score["steps"], score["scores"]
    if not steps or len(steps) != len(values) or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid fold score series")
    if eligible is not None and len(eligible) != len(steps):
        raise ValueError("invalid eligibility mask")
    fps = score.get("control_fps")
    if (isinstance(fps, bool) or not isinstance(fps, (int, float))
        or not math.isfinite(fps) or fps <= 0 or steps != list(range(len(steps)))):
        raise ValueError("fold review requires every saved state and control_fps for the 3 s hold")
    span = round(SUCCESS_HOLD_SECONDS * fps)
    if span < 1 or not math.isclose(span, SUCCESS_HOLD_SECONDS * fps, abs_tol=1e-9):
        raise ValueError("fold hold duration must resolve to exact control steps")
    samples = sorted((s, v) for i, (s, v) in enumerate(zip(steps, values, strict=True))
                     if eligible is None or eligible[i])
    if not samples:
        return {"action_count": steps[-1], "recommendation_reason": "no_eligible_sample"}
    peak = max(v for _, v in samples)
    best_index = min(s for s, v in samples if v == peak)
    passing = None
    run_start = 0
    for i, value in enumerate(values):
        if value < threshold:
            run_start = i + 1
        elif i - run_start >= span and (eligible is None or eligible[i]):
            passing = (steps[i], value)
            break
    recommended_index, recommended_score = passing if passing else (best_index, peak)
    return {"action_count": steps[-1], "final_score": values[-1],
            "success": passing is not None, "success_hold_seconds": SUCCESS_HOLD_SECONDS,
            "peak_score": peak,
            "best_sample_index": best_index,
            "recommended_sample_index": recommended_index,
            "recommended_score": recommended_score,
            "recommendation_reason": "first_success" if passing else "best_unpassed",
            "recommendation_threshold": threshold,
            "recommendation_hold_seconds": SUCCESS_HOLD_SECONDS}


def flatten_review_row(steps: list[int], deformable: list[float], rigid: list[float],
                       thresholds: dict, deformable_weight: float = .65,
                       eligible: list[bool] | None = None, tolerance: float = 0.0) -> dict:
    """Recommend the earliest eligible near-optimum; screen joint success independently."""
    if not steps or len(steps) != len(deformable) or len(steps) != len(rigid):
        raise ValueError("invalid flatten score series")
    if not all(math.isfinite(v) for v in (*deformable, *rigid)):
        raise ValueError("nonfinite flatten score")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid tolerance")
    if eligible is not None and len(eligible) != len(steps):
        raise ValueError("invalid eligibility mask")
    candidates = [i for i in range(len(steps)) if eligible is None or eligible[i]]
    if not candidates:
        return {"action_count": steps[-1], "recommendation_reason": "no_eligible_sample"}
    combined = [deformable_weight*d + (1-deformable_weight)*r
                for d, r in zip(deformable, rigid, strict=True)]
    best = min(candidates, key=lambda i: (combined[i], steps[i]))
    passing = [i for i in candidates
               if deformable[i] <= thresholds["deformable_distance_max"]
               and rigid[i] <= thresholds["rigid_distance_max"]]
    index = min(passing, key=lambda i: steps[i]) if passing else best
    recommended = min((i for i in candidates if combined[i] <= combined[best] + tolerance),
                      key=lambda i: steps[i])
    # The screening pair is a joint-success witness; the recommendation may differ.
    return {"action_count": steps[-1], "deformable_distance": deformable[index],
            "rigid_distance": rigid[index], "combined_distance": combined[index],
            "terminal_deformable_distance": deformable[-1],
            "terminal_rigid_distance": rigid[-1], "best_sample_index": steps[best],
            "screening_sample_index": steps[index],
            "recommended_sample_index": steps[recommended],
            "recommended_score": combined[recommended],
            "recommended_deformable_distance": deformable[recommended],
            "recommended_rigid_distance": rigid[recommended],
            "recommendation_tolerance": tolerance,
            "recommendation_reason": ("near_best" if tolerance else "best") if passing
                else ("near_best_unpassed" if tolerance else "best_unpassed"),
            "recommendation_threshold": dict(thresholds)}
