import numpy as np
import pytest
from clothdojo.trajectory import TrajectoryWriter, TrajectoryReader


@pytest.mark.parametrize("compressed", [False, True])
def test_roundtrip_alignment(tmp_path, state_factory, action_factory, compressed):
    p = tmp_path / "episode"
    w = TrajectoryWriter(
        p, control_fps=50, sim_substeps=20, chunk_size=1, compressed=compressed
    )
    w.write_initial_state(state_factory(0))
    w.append_transition(action_factory(0), state_factory(1))
    w.close()
    r = TrajectoryReader(p)
    assert r.manifest.state_count == 2 and r.manifest.action_count == 1
    sample = list(r.iter_samples())[0]
    assert sample.state.sample_index == 0 and sample.action.sample_index == 0
    np.testing.assert_array_equal(list(r.iter_actions())[0].joint_target, np.arange(12))


def test_zero_action_episode(tmp_path, state_factory):
    p = tmp_path / "zero"
    w = TrajectoryWriter(p, control_fps=50, sim_substeps=20)
    w.write_initial_state(state_factory())
    w.close()
    r = TrajectoryReader(p)
    assert len(list(r.iter_states())) == 1
    assert list(r.iter_actions()) == []


def test_initial_index_rejected(tmp_path, state_factory):
    w = TrajectoryWriter(tmp_path / "bad", control_fps=50, sim_substeps=20)
    with pytest.raises(ValueError):
        w.write_initial_state(state_factory(1))


def test_transition_index_rejected(tmp_path, state_factory, action_factory):
    w = TrajectoryWriter(tmp_path / "bad", control_fps=50, sim_substeps=20)
    w.write_initial_state(state_factory())
    with pytest.raises(ValueError):
        w.append_transition(action_factory(1), state_factory(1))


def test_existing_output_is_preserved(tmp_path):
    p = tmp_path / "existing"
    p.mkdir()
    (p / "keep").write_text("preserved")
    with pytest.raises(FileExistsError):
        TrajectoryWriter(p, control_fps=50, sim_substeps=20)
    assert (p / "keep").read_text() == "preserved"


def test_incomplete_manifest_rejected(tmp_path):
    p = tmp_path / "incomplete"
    TrajectoryWriter(p, control_fps=50, sim_substeps=20)
    with pytest.raises(RuntimeError):
        TrajectoryReader(p)


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrajectoryReader(tmp_path / "missing")
