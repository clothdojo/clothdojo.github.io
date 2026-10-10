import pytest
from clothdojo.evaluation.fold_success import fold_success, earliest_fold_success
from clothdojo.evaluation.fold_quality import FoldQualityMetric

Q = FoldQualityMetric.load().success_threshold


@pytest.mark.parametrize(
    "count,fps,scores", [(0, 50, {}), (151, 0, {}), (151, 1.1, {}), (151, 50, {-1: Q})]
)
def test_invalid_inputs(count, fps, scores):
    with pytest.raises(ValueError):
        fold_success(count, fps, lambda i: Q, known_scores=scores)


def test_short_sequence_cannot_succeed():
    assert not fold_success(150, 50, lambda i: Q).success


def test_exact_three_second_witness():
    r = fold_success(151, 50, lambda i: Q)
    assert r.success and r.success_at_end
    assert (r.witness_start_step, r.witness_end_step) == (0, 150)


def test_one_failed_state_breaks_witness():
    assert not fold_success(151, 50, lambda i: Q - 1 if i == 75 else Q + 1).success


def test_earliest_witness_before_terminal_failure():
    r = earliest_fold_success(200, 50, lambda i: Q + 1 if i < 151 else Q - 1)
    assert r.success and not r.success_at_end
    assert (r.witness_start_step, r.witness_end_step) == (0, 150)
