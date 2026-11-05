"""Guard smoke sampling against losing the fixed state's appearance identity."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from clothdojo.rollout.initial_states import InitialStatePool


def test_smoke_subsets_state_and_appearance_without_changing_release(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("rgb_example", root / "scripts/rgb-policy.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.ROOT = tmp_path
    config = tmp_path / "configs/training/rgb_flatten_diffusion.yaml"
    config.parent.mkdir(parents=True)
    config.write_text((root / "configs/training/rgb_flatten_diffusion.yaml").read_text())
    work = tmp_path / "outputs/smoke"
    condition = work / "assets/benchmark/conditions/rgb-flatten"
    source = condition / "states/garment/states.npz"
    InitialStatePool(
        particle_q=np.zeros((4, 3, 3), dtype=np.float32),
        particle_qd=np.zeros((4, 3, 3), dtype=np.float32),
        seeds=np.arange(4, dtype=np.int32),
        metadata={
            "visual_profiles": ["train", "train", "test_all", "test_all"],
            "physical_state_indices": [0, 1, 0, 1],
        },
    ).save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    example.write(
        condition / "states/meta.json",
        {
            "assets": [
                {
                    "asset_id": "garment",
                    "path": "garment/states.npz",
                    "split": "seen",
                    "particle_count": 3,
                    "state_count": 4,
                    "sha256": digest,
                }
            ],
            "selection_provenance": {"panel_id": "released_rgb_panel"},
        },
    )
    example.write(condition / "scenario.json", {"action_horizon": 40, "steps": 3000})
    example.write(work / "download.json", {"smoke_only": True})
    example.prepare(work, True)
    selected = InitialStatePool.load(work / "smoke-state/garment/states.npz")
    assert selected.state_count == 1
    assert selected.metadata["visual_profiles"] == ["train"]
    assert selected.metadata["physical_state_indices"] == [0]
    assert selected.seeds.tolist() == [0]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    cfg = yaml.safe_load((work / "train.yaml").read_text())
    assert cfg["gradient_accumulation_steps"] == 1
    assert cfg["global_batch_size"] == cfg["micro_batch_size"] == 1
    assert isinstance(cfg["weight_decay"], float)
    scenario = json.loads((work / "scenario.json").read_text())
    assert scenario["steps"] == 3000
    assert scenario["action_horizon"] == 8


def test_evaluation_rejects_wrong_upstream_visual_version(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("rgb_example", root / "scripts/rgb-policy.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.ROOT = tmp_path
    expected = b"released version"
    row = {
        "path": "data/visual-assets/source.bin",
        "bytes": len(expected),
        "sha256": hashlib.sha256(expected).hexdigest(),
    }
    work = tmp_path / "outputs/example"
    example.write(work / "assets/upstream_visual_dependencies.json", [row])
    with pytest.raises(RuntimeError, match="1 missing"):
        example.check_visual_dependencies(work)
    path = tmp_path / row["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"released versioX")  # Same length; a size-only check would miss this.
    assert path.stat().st_size == len(expected)
    with pytest.raises(RuntimeError, match="1 different"):
        example.check_visual_dependencies(work)
    path.write_bytes(expected)
    example.check_visual_dependencies(work)
