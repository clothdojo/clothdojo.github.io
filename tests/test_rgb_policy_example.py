"""Guard smoke sampling against losing the fixed state's appearance identity."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from clothdojo.rollout.initial_states import InitialStatePool


@pytest.fixture
def example_inputs(tmp_path):
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
    return example, work, source, digest


def test_smoke_subsets_state_and_appearance_without_changing_release(example_inputs):
    example, work, source, digest = example_inputs
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


def test_assets_only_evaluation_does_not_fetch_training_data(example_inputs, monkeypatch):
    import huggingface_hub

    example, work, source, digest = example_inputs
    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **kwargs: calls.append(repo)
    )
    monkeypatch.setattr(
        sys, "argv", ["rgb-policy.py", "download", "--assets-only", "--smoke", "--work", str(work)]
    )
    example.main()
    assert calls == ["clothdojo/assets"]
    assert not (work / "dataset").exists()
    assert (work / "evaluate.json").is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    monkeypatch.setattr(sys, "argv", ["rgb-policy.py", "train", "--smoke", "--work", str(work)])
    with pytest.raises(ValueError, match="without --assets-only"):
        example.main()


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


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "--checkpoint", "model.pt"],
        ["train", "--model", "my_policy:Model"],
        ["train", "--assets-only"],
        ["download", "--check"],
    ],
)
def test_wrong_stage_options_fail_before_launch(arguments, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("rgb_example", root / "scripts/rgb-policy.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    monkeypatch.setattr(sys, "argv", ["rgb-policy.py", *arguments])
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2
