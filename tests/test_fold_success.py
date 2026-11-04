import numpy as np
import pytest

from clothdojo.evaluation.fold_quality import FoldQualityMetric
from clothdojo.evaluation.fold_success import earliest_fold_success, fold_success


def _evaluate(flags, fps=50, known=None):
    threshold = FoldQualityMetric.load().success_threshold
    q = np.where(flags, threshold, threshold - 1)
    result = fold_success(len(q), fps, lambda i: q[i], known_scores=known)
    if result.success:
        start, end = result.witness_start_step, result.witness_end_step
        assert (end - start) / fps >= 3
        assert np.all(q[start:end + 1] >= threshold)
    return result


def test_150_states_are_only_2_98_seconds_and_151_are_three_seconds():
    assert not _evaluate(np.ones(150, bool)).success
    result = _evaluate(np.ones(151, bool))
    assert result.success and result.success_at_end


def test_any_interruption_resets_the_hold_even_between_one_second_samples():
    flags = np.ones(201, bool)
    flags[75] = False
    q = np.where(flags, FoldQualityMetric.load().success_threshold, -10)
    assert not _evaluate(flags, known={i: float(q[i]) for i in range(0, 201, 50)}).success


def test_success_survives_later_failure_but_end_success_does_not():
    flags = np.r_[np.zeros(3, bool), np.ones(151, bool), np.zeros(47, bool)]
    result = _evaluate(flags)
    assert result.success and not result.success_at_end
    assert result.witness_start_step == 3


def test_exact_search_matches_exhaustive_consecutive_run_for_random_sequences():
    rng = np.random.default_rng(43)
    for fps in (1, 2, 5):
        span = 3 * fps
        for _ in range(80):
            flags = rng.random(51) > rng.uniform(.02, .7)
            result = _evaluate(flags, fps)
            expected = any(flags[i:i + span + 1].all() for i in range(len(flags) - span))
            assert result.success == expected
            assert result.success_at_end == bool(flags[-span - 1:].all())


def test_invalid_scores_cannot_confirm_success():
    with pytest.raises(ValueError, match="finite"):
        fold_success(200, 50, lambda i: float("nan"))
    with pytest.raises(ValueError, match="control_fps"):
        fold_success(200, 0, lambda i: 1)


def test_earliest_search_does_not_prefer_a_later_terminal_witness():
    threshold = FoldQualityMetric.load().success_threshold
    flags = np.r_[
        np.zeros(5, bool),
        np.ones(151, bool),
        np.zeros(20, bool),
        np.ones(151, bool),
    ]
    q = np.where(flags, threshold, threshold - 1)
    result = earliest_fold_success(len(q), 50, lambda i: q[i])
    assert result.success and result.success_at_end
    assert result.witness_start_step == 5
    assert result.witness_end_step == 155


def test_earliest_search_matches_exhaustive_first_window():
    rng = np.random.default_rng(44)
    threshold = FoldQualityMetric.load().success_threshold
    for fps in (1, 2, 5):
        span = 3 * fps
        for _ in range(80):
            flags = rng.random(51) > rng.uniform(.02, .7)
            q = np.where(flags, threshold, threshold - 1)
            result = earliest_fold_success(len(q), fps, lambda i: q[i])
            starts = [i for i in range(len(flags) - span)
                      if flags[i:i + span + 1].all()]
            assert result.success == bool(starts)
            assert result.witness_start_step == (starts[0] if starts else None)
            assert result.witness_end_step == (starts[0] + span if starts else None)
