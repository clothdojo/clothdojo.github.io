import json
import pytest
from clothdojo.data.selection import first_selected_trajectory, approve_selected_export


def fixture(tmp):
    t = tmp / "trajectory"
    t.mkdir()
    (t / "manifest.json").write_text("{}")
    s = tmp / "selection.json"
    s.write_text(
        json.dumps(
            {
                "status": "complete",
                "selected": ["trajectory"],
                "protocol_fingerprint": "public-test-protocol",
            }
        )
    )
    (t / "score.json").write_text(
        json.dumps({"success": True, "protocol_fingerprint": "public-test-protocol"})
    )
    d = tmp / "dataset"
    (d / "meta").mkdir(parents=True)
    (d / "meta/clothdojo.json").write_text(
        json.dumps(
            {
                "schema": "clothdojo_lerobot_export",
                "export_status": "complete",
                "observation": {"cloth_visual": "rgb"},
                "training_eligible": False,
            }
        )
    )
    return s, t, d


def test_first_success(tmp_path):
    s, t, d = fixture(tmp_path)
    assert first_selected_trajectory(s) == t.resolve()


def test_incomplete_selection(tmp_path):
    s, t, d = fixture(tmp_path)
    s.write_text(json.dumps({"status": "running", "selected": ["trajectory"]}))
    with pytest.raises(ValueError):
        first_selected_trajectory(s)


def test_no_selected_trajectory(tmp_path):
    s = tmp_path / "selection.json"
    s.write_text(json.dumps({"status": "complete", "selected": []}))
    with pytest.raises(ValueError):
        first_selected_trajectory(s)


def test_missing_selected_manifest(tmp_path):
    s = tmp_path / "selection.json"
    s.write_text(json.dumps({"status": "complete", "selected": ["missing"]}))
    with pytest.raises(FileNotFoundError):
        first_selected_trajectory(s)


def test_export_approval_records_identity(tmp_path):
    s, t, d = fixture(tmp_path)
    approve_selected_export(s, t, d)
    x = json.loads((d / "meta/clothdojo.json").read_text())
    assert x["training_eligible"] is True
    assert len(x["selection"]["selection_sha256"]) == 64
    with pytest.raises(ValueError):
        approve_selected_export(s, t, d)


def test_protocol_mismatch_does_not_approve(tmp_path):
    s, t, d = fixture(tmp_path)
    (t / "score.json").write_text(
        json.dumps({"success": True, "protocol_fingerprint": "different"})
    )
    with pytest.raises(ValueError):
        approve_selected_export(s, t, d)
    assert (
        json.loads((d / "meta/clothdojo.json").read_text())["training_eligible"]
        is False
    )
