from types import SimpleNamespace

import pytest

from clothdojo.data import rollout


def test_export_uses_requested_asset_root_for_robot_lookup(tmp_path, monkeypatch):
    from clothdojo import trajectory

    source = tmp_path / "source"
    assets = tmp_path / "external-benchmark/assets"
    manifest = {
        "scenario": {},
        "trajectories": [{"path": "trajectories/example"}],
    }
    monkeypatch.setattr(rollout, "_load_canonical_collection",
                        lambda _: (manifest, "rollout_collection"))
    monkeypatch.setattr(trajectory, "TrajectoryReader", lambda _: SimpleNamespace(
        manifest=SimpleNamespace(scene=SimpleNamespace(robot_id="piper"))))

    class RobotLookupReached(Exception):
        pass

    def lookup(robot_id, *, assets_root):
        assert robot_id == "piper"
        assert assets_root == assets
        raise RobotLookupReached

    monkeypatch.setattr(rollout, "robot_profile", lookup)
    with pytest.raises(RobotLookupReached):
        rollout.export_rollout_collection(source, tmp_path / "rgb", assets_root=assets)
    assert not (tmp_path / "rgb").exists()
