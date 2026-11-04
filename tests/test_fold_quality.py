from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clothdojo import cli
from clothdojo.evaluation.fold_quality import (
    FoldQualityMetric,
    _material_frame,
    fold_quality_features,
    prepare_fold_quality_reference,
)


def _cloth():
    nx, ny = 13, 9
    x, y = np.meshgrid(np.linspace(-0.4, 0.4, nx), np.linspace(-0.3, 0.3, ny))
    q = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.8)])
    tri = []
    for row in range(ny - 1):
        for col in range(nx - 1):
            i = row * nx + col
            tri.extend([(i, i + 1, i + nx), (i + 1, i + nx + 1, i + nx)])
    return q, np.array(tri)


def test_scores_are_independent_of_evaluation_order_and_reference_mutation():
    q, tri = _cloth()
    ref = prepare_fold_quality_reference(q, tri)
    metric = FoldQualityMetric.load()
    folded = q.copy()
    folded[:, 0] = np.abs(q[:, 0])
    folded[q[:, 0] < 0, 2] += 0.005
    states = [q, folded, folded + np.array([0, 0, 0.2])]
    first = [metric.score(s, ref) for s in states]
    # Caller mutations cannot change the prepared reference's mesh topology/geometry.
    q[:] = 99
    tri[:] = 0
    again = [metric.score(states[i], ref) for i in [2, 1]]
    assert again == pytest.approx([first[2], first[1]], abs=1e-12)
    vectors = np.stack([fold_quality_features(states[i], ref) for i in [1, 2]])
    assert metric.score_features(vectors) == pytest.approx(first[1:], abs=1e-12)
    assert all(np.isfinite(first))


def test_invalid_state_contracts_fail_instead_of_silent_clipping():
    q, tri = _cloth()
    ref = prepare_fold_quality_reference(q, tri)
    metric = FoldQualityMetric.load()
    with pytest.raises(ValueError, match="particle count"):
        metric.score(q[:-1], ref)
    with pytest.raises(ValueError, match="finite"):
        metric.score(q * np.nan, ref)
    with pytest.raises(ValueError, match="integers"):
        prepare_fold_quality_reference(q, tri.astype(float))


def test_cli_forwards_fold_quality_options(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        "clothdojo.evaluation.fold_quality_collection.score_fold_quality_collection",
        lambda *args, **kwargs: seen.update(args=args, kwargs=kwargs) or {"trajectories": 1},
    )
    cli.main(
        [
            "data",
            "score-fold-quality",
            "--source",
            "trajectory",
            "--output",
            "scores",
            "--reference",
            "references",
            "--sample-stride",
            "10",
            "--workers",
            "2",
        ]
    )
    assert seen == {
        "args": ("trajectory", "scores"),
        "kwargs": {"assets_root": "assets", "reference": "references",
                   "sample_stride": 10, "workers": 2},
    }
    assert json.loads(capsys.readouterr().out)["trajectories"] == 1


def test_fixed_material_frame_ignores_height_and_score_is_yaw_invariant():
    q, tri = _cloth()
    ref = prepare_fold_quality_reference(q, tri)
    folded = q.copy()
    folded[:, 0] = np.abs(q[:, 0])
    folded[q[:, 0] < 0, 2] += 0.005
    noisy = folded.copy()
    noisy[:, 2] += 0.001 * np.sin(np.arange(len(q)))
    assert np.array_equal(
        _material_frame(folded, tri, ref.material_weights)[:, :2],
        _material_frame(noisy, tri, ref.material_weights)[:, :2],
    )
    metric = FoldQualityMetric.load()
    for degrees in (37, 90, 180, 270):
        a = np.deg2rad(degrees)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        moved = folded.copy()
        moved[:, :2] = moved[:, :2] @ rot.T + [0.03, -0.02]
        assert metric.score(moved, ref) == pytest.approx(metric.score(folded, ref), abs=1e-8)


def test_success_uses_the_single_cutoff_inclusively():
    metric = FoldQualityMetric.load()
    cutoff = metric.success_threshold
    assert metric.passes_threshold([np.nextafter(cutoff, -np.inf), cutoff,
                              np.nextafter(cutoff, np.inf)]).tolist() == [False, True, True]
    with pytest.raises(ValueError, match="finite"):
        metric.passes_threshold(float("nan"))
    with pytest.raises(ValueError, match="shape"):
        metric.score_features(np.zeros(34))


@pytest.mark.parametrize("source", ["expert_terminal", "hil_completion", "hil_failure_terminal"])
def test_real_states_match_accepted_features_and_score(source):
    with np.load(Path(__file__).parent / "fixtures/fold_quality" / f"{source}.npz") as data:
        metric = FoldQualityMetric.load()
        reference = prepare_fold_quality_reference(data["flat"], data["triangles"])
        features = fold_quality_features(data["q"], reference)
        np.testing.assert_allclose(features, data["expected_features"], rtol=0, atol=1e-12)
        assert metric.score(data["q"], reference) == pytest.approx(
            float(data["expected_score"]), abs=1e-12
        )


def test_collection_uses_explicit_reference_and_final_not_peak(tmp_path, monkeypatch):
    from clothdojo.evaluation import fold_quality_collection as collection

    with np.load(Path(__file__).parent / "fixtures/fold_quality/expert_terminal.npz") as data:
        flat, folded, tri = data["flat"], data["q"], data["triangles"]
    source = tmp_path / "episode"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    reference_file = tmp_path / "reference.npz"
    reference_file.touch()
    manifest = SimpleNamespace(
        scene=SimpleNamespace(task_name="shirt_fold", robot_id="piper"),
        metadata={"asset_id": "shirt"}, state_count=2, control_fps=50,
    )
    monkeypatch.setattr(collection, "TrajectoryReader", lambda _: SimpleNamespace(
        manifest=manifest,
        state_arrays=lambda names, **kw: {"particle_q": np.stack([folded, flat])},
    ))
    monkeypatch.setattr(collection, "load_fold_quality_triangles", lambda *args: tri)
    monkeypatch.setattr(collection, "load_canonical_template", lambda _: SimpleNamespace(
        asset_id="shirt", particle_q=flat,
    ))
    result = collection.score_fold_quality_collection(
        source, tmp_path / "scored", reference=reference_file, sample_stride=100,
    )
    row = json.loads(Path(result["scores"]).read_text())
    assert row["steps"] == [0, 1]
    assert row["peak_score"] >= result["success_threshold"]
    assert row["final_score"] < result["success_threshold"]
    assert row["success"] is False
    assert result["successes"] == 0 and result["success_rate"] == 0
    assert row["reference"] == str(reference_file)
    with pytest.raises(FileExistsError):
        collection.score_fold_quality_collection(
            source, tmp_path / "scored", reference=reference_file,
        )
    monkeypatch.setattr(collection, "load_canonical_template", lambda _: SimpleNamespace(
        asset_id="wrong_asset", particle_q=flat,
    ))
    with pytest.raises(ValueError, match="asset differ"):
        collection._score_trajectory((str(source), "assets", 1, str(reference_file)))


@pytest.mark.parametrize("failed_step,ever,end", [(None, True, True), (75, False, False),
                                                 (175, True, False)])
def test_subsampled_curve_still_checks_every_state_of_hold(monkeypatch, failed_step, ever, end):
    from clothdojo.evaluation import fold_quality_collection as collection

    threshold = FoldQualityMetric.load().success_threshold
    states = np.full((201, 3, 3), threshold)
    if failed_step is not None:
        states[failed_step] = threshold - 1
    scored = set()

    def score(q, reference):
        scored.add(int(np.where(np.all(states == q, axis=(1, 2)))[0][0]))
        return float(q[0, 0])

    monkeypatch.setattr(collection, "_metric", lambda: SimpleNamespace(
        success_threshold=threshold, score=score,
    ))
    monkeypatch.setattr(collection, "TrajectoryReader", lambda _: SimpleNamespace(
        manifest=SimpleNamespace(scene=SimpleNamespace(task_name="shirt_fold", robot_id="piper"),
                                 metadata={"asset_id": "shirt"}, state_count=201, control_fps=50),
        state_arrays=lambda *a, **kw: {"particle_q": states},
    ))
    monkeypatch.setattr(collection, "load_canonical_template", lambda _: SimpleNamespace(
        asset_id="shirt", particle_q=states[0],
    ))
    monkeypatch.setattr(collection, "load_fold_quality_triangles", lambda *a: np.array([[0, 1, 2]]))
    monkeypatch.setattr(collection, "prepare_fold_quality_reference", lambda *a: None)
    row = collection._score_trajectory(("trajectory", "assets", 200, "reference.npz"))
    assert row["steps"] == [0, 200]
    assert all(q >= threshold for q in row["scores"])
    assert row["success"] is ever and row["success_at_end"] is end
    if failed_step is not None:
        assert failed_step in scored
