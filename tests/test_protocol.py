import numpy as np
import pytest
import yaml
from clothdojo.evaluation.protocol import load_evaluation_protocol, SuccessTracker


@pytest.mark.parametrize("task,steps", [("shirt_flatten", 3000), ("shirt_fold", 2000)])
def test_task_protocol(root, task, steps):
    p = load_evaluation_protocol(root / f"configs/evaluation/{task}.yaml")
    assert p.horizon_steps == steps


@pytest.mark.parametrize(
    "change", ["schema", "unknown", "missing", "bool_fps", "rate", "empty_id"]
)
def test_reject_bad_protocol(root, tmp_path, change):
    d = yaml.safe_load((root / "configs/evaluation/shirt_flatten.yaml").read_text())
    if change == "schema":
        d["schema"] = "other"
    elif change == "unknown":
        d["unknown"] = 1
    elif change == "missing":
        d.pop("target")
    elif change == "bool_fps":
        d["timing"]["control_fps"] = True
    elif change == "rate":
        d["timing"]["metric_hz"] = 11
    else:
        d["protocol_id"] = ""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(d))
    with pytest.raises((ValueError, KeyError)):
        load_evaluation_protocol(path)


def test_fingerprint_stable(root, tmp_path):
    path = root / "configs/evaluation/shirt_flatten.yaml"
    d = yaml.safe_load(path.read_text())
    p = tmp_path / "reordered.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=True))
    assert (
        load_evaluation_protocol(path).fingerprint
        == load_evaluation_protocol(p).fingerprint
    )


def test_continuous_success_is_absorbing(root):
    p = load_evaluation_protocol(root / "configs/evaluation/shirt_flatten.yaml")
    t = SuccessTracker(p, 1)
    for i in range(p.success_window_samples):
        t.update(
            step=i * p.metric_stride,
            deformable_distance=np.array([0.0]),
            rigid_distance=np.array([0.0]),
        )
    assert t.success[0]
    t.update(
        step=p.success_window_samples * p.metric_stride,
        deformable_distance=np.array([100.0]),
        rigid_distance=np.array([100.0]),
    )
    assert t.success[0]
