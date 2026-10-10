import numpy as np
import pytest
from clothdojo.data.mirror import mirror_action_arrays, mirror_state_arrays
from clothdojo.trajectory.schema import STATE_ARRAY_DTYPES


def action(n=2):
    return {
        "sample_index": np.arange(n),
        "joint_target": np.arange(n * 12, dtype=float).reshape(n, 12),
        "gripper_open_command": np.tile([0.0, 1.0], (n, 1)),
    }


@pytest.mark.parametrize("count", [1, 2, 5])
def test_action_mirror_involution(count):
    a = action(count)
    b = mirror_action_arrays(mirror_action_arrays(a))
    for k, v in a.items():
        np.testing.assert_allclose(b[k], v)


@pytest.mark.parametrize("shape", [(2, 11), (2, 13), (2, 2, 6)])
def test_invalid_action_shape(shape):
    a = action()
    a["joint_target"] = np.zeros(shape)
    with pytest.raises(ValueError):
        mirror_action_arrays(a)


def test_missing_action_field():
    a = action()
    a.pop("sample_index")
    with pytest.raises(ValueError):
        mirror_action_arrays(a)


def test_nonfinite_action():
    a = action()
    a["joint_target"][0, 0] = np.nan
    with pytest.raises(ValueError):
        mirror_action_arrays(a)


def test_full_state_involution(state_factory):
    a = state_factory()
    data = {k: np.expand_dims(getattr(a, k), 0) for k in STATE_ARRAY_DTYPES}
    data["sample_index"] = np.array([0])
    b = mirror_state_arrays(mirror_state_arrays(data))
    for k, v in data.items():
        np.testing.assert_allclose(b[k], v, atol=2e-6)


def test_missing_state_fields():
    with pytest.raises(ValueError):
        mirror_state_arrays({"sample_index": np.array([0])})
