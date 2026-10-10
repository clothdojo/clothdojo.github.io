import numpy as np
import pytest
from clothdojo.evaluation.fold_quality import (
    FoldQualityMetric,
    prepare_fold_quality_reference,
)


@pytest.mark.parametrize(
    "name", ["expert_terminal", "hil_completion", "hil_failure_terminal"]
)
def test_frozen_fixture_score(root, name):
    with np.load(
        root / f"scripts/data/fold_quality/{name}.npz", allow_pickle=False
    ) as z:
        ref = prepare_fold_quality_reference(z["flat"], z["triangles"])
        actual = FoldQualityMetric.load().score(z["q"], ref)
        assert actual == pytest.approx(float(z["expected_score"]), rel=1e-6, abs=1e-8)


@pytest.mark.parametrize(
    "values", [np.zeros(4), np.zeros((2, 4)), np.array([0, 0, 0, 0, np.nan])]
)
def test_bad_features(values):
    with pytest.raises(ValueError):
        FoldQualityMetric.load().score_features(values)


def test_threshold_inclusive():
    m = FoldQualityMetric.load()
    v = m.success_threshold
    assert bool(m.passes_threshold(v))
    assert not bool(m.passes_threshold(np.nextafter(v, -np.inf)))


def test_nonfinite_quality_rejected():
    with pytest.raises(ValueError):
        FoldQualityMetric.load().passes_threshold(np.inf)
