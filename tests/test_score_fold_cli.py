import json
import os
import subprocess
import sys
import numpy as np
import pytest


def cli(root, *args):
    return subprocess.run(
        [sys.executable, "-B", str(root / "scripts/score_fold.py"), *map(str, args)],
        cwd=root,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root / "src")),
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "name", ["expert_terminal", "hil_completion", "hil_failure_terminal"]
)
def test_cli_frozen_score(root, name):
    p = root / f"scripts/data/fold_quality/{name}.npz"
    r = cli(root, p)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    with np.load(p, allow_pickle=False) as z:
        assert d["score"] == pytest.approx(
            float(z["expected_score"]), rel=1e-6, abs=1e-8
        )
    assert d["frame_passes_threshold"] == (d["score"] >= d["threshold"])


def test_missing_input_exit(root, tmp_path):
    assert cli(root, tmp_path / "missing.npz").returncode != 0


def test_incomplete_archive_exit(root, tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, q=np.zeros((4, 3)))
    assert cli(root, p).returncode != 0


def test_default_matches_explicit(root):
    a = cli(root)
    b = cli(root, root / "scripts/data/fold_quality/expert_terminal.npz")
    assert a.returncode == b.returncode == 0
    assert json.loads(a.stdout) == json.loads(b.stdout)
