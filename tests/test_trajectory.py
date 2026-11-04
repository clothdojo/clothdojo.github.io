from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clothdojo.trajectory.newton import gripper_openness
from clothdojo.trajectory import (
    RenderOnlyTrajectoryWriter,
    SceneMetadata,
    TrajectoryAction,
    TrajectoryManifest,
    TrajectoryReader,
    TrajectoryState,
    TrajectoryWriter,
    compact_render_trajectory,
    load_render_selection_signals,
    publish_render_trajectory,
)


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        ([[0.02, 0.04]], [[0.5, 1.0]]),
        ([[0.02, -0.01, 0.04, -0.03]], [[0.5, 1.0]]),
    ],
)
def test_gripper_openness_supports_one_or_two_coordinates_per_side(
    positions, expected
) -> None:
    values = np.asarray(positions, dtype=np.float32)
    lower = np.zeros_like(values)
    upper = np.full_like(values, 0.04)

    np.testing.assert_allclose(gripper_openness(values, lower, upper), expected)


def test_gripper_openness_uses_authored_reversed_ur5e_semantics() -> None:
    values = np.asarray([[0.0, 0.4]], dtype=np.float32)
    limits = np.zeros_like(values)

    openness = gripper_openness(
        values,
        limits,
        limits,
        open_positions=(0.0,),
        closed_positions=(0.8,),
    )

    np.testing.assert_allclose(openness, [[1.0, 0.5]])


def test_manifest_reads_legacy_environment_without_reemitting_it() -> None:
    manifest = TrajectoryManifest.from_json_dict(
        {
            "control_fps": 50,
            "sim_substeps": 6,
            "units": "m",
            "coordinate_convention": "right_handed_z_up",
            "max_actions": 1,
            "scene": {
                "task_name": "shirt_flatten",
                "robot_name": "piper",
                "environment_id": "piper_shirt_flatten_v1",
                "environment_fingerprint": "obsolete",
            },
        }
    )

    assert manifest.scene.robot_id == "piper"
    assert manifest.to_json_dict()["scene"] == {
        "task_name": "shirt_flatten",
        "robot_id": "piper",
        "asset_fingerprints": {},
        "config_fingerprints": {},
    }


def test_manifest_infers_legacy_task_when_it_is_missing() -> None:
    scene = TrajectoryManifest.from_json_dict(
        {
            "control_fps": 50,
            "sim_substeps": 6,
            "units": "m",
            "coordinate_convention": "right_handed_z_up",
            "max_actions": 1,
            "scene": {"environment_id": "piper_shirt_flatten_v1"},
        }
    ).scene

    assert scene.robot_id == "piper"
    assert scene.task_name == "shirt_flatten"


def test_manifest_rejects_conflicting_legacy_task() -> None:
    with pytest.raises(ValueError, match="conflicting task"):
        SceneMetadata.from_json_dict(
            {
                "task_name": "shirt_fold",
                "environment_id": "piper_shirt_flatten_v1",
            }
        )


def test_manifest_rejects_unknown_legacy_environment() -> None:
    with pytest.raises(ValueError, match="unknown legacy"):
        SceneMetadata.from_json_dict({"environment_id": "unknown_scene_v1"})


def _state(index: int) -> TrajectoryState:
    return TrajectoryState(
        sample_index=index,
        joint_q=np.full((3,), index, dtype=np.float64),
        joint_qd=np.full((3,), index + 0.1, dtype=np.float64),
        body_q=np.full((2, 7), index, dtype=np.float64),
        body_qd=np.full((2, 6), index + 0.2, dtype=np.float64),
        particle_q=np.full((4, 3), index, dtype=np.float64),
        particle_qd=np.full((4, 3), index + 0.3, dtype=np.float64),
        gripper_open_state=np.full((2,), index, dtype=np.float64),
        grasp_attached=np.asarray([index % 2, 0], dtype=np.int64),
        grasp_previous_open=np.full((2,), index + 0.4, dtype=np.float64),
        grasp_pin_count=np.asarray([1, 0], dtype=np.int64),
        grasp_pins=np.asarray([[2, -1], [-1, -1]], dtype=np.int64),
        grasp_pin_local=np.full((2, 2, 3), index + 0.5, dtype=np.float64),
    )


def _action(index: int) -> TrajectoryAction:
    return TrajectoryAction(
        sample_index=index,
        joint_target=np.full((3,), index + 0.25, dtype=np.float64),
        gripper_open_command=np.full((2,), index + 0.5, dtype=np.float64),
    )


def _write_trajectory(path: Path, *, action_count: int = 3) -> Path:
    writer = TrajectoryWriter(
        path,
        control_fps=30.0,
        sim_substeps=20,
        max_actions=max(1, action_count),
        chunk_size=2,
    )
    writer.write_initial_state(_state(0))
    for index in range(action_count):
        writer.append_transition(_action(index), _state(index + 1))
    writer.close()
    return path


def test_writer_preserves_n_plus_one_alignment_and_reader_views(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory"
    writer = TrajectoryWriter(
        trajectory,
        control_fps=30.0,
        sim_substeps=20,
        chunk_size=2,
    )
    writer.write_initial_state(_state(0))
    for index in range(3):
        writer.append_transition(_action(index), _state(index + 1))

    assert (trajectory / ".chunks").is_dir()
    assert not (trajectory / "states.npz").exists()
    writer.close()

    reader = TrajectoryReader(trajectory)
    samples = list(reader.iter_samples())
    assert reader.manifest.state_count == 4
    assert reader.manifest.action_count == 3
    assert [sample.state.sample_index for sample in samples] == [0, 1, 2]
    assert [sample.action.sample_index for sample in samples] == [0, 1, 2]
    assert reader.terminal_state().sample_index == 3
    assert samples[0].state.particle_q.dtype == np.float32
    assert samples[0].state.grasp_pins.dtype == np.int32
    assert samples[0].action.joint_target.dtype == np.float32
    assert samples[0].state.particle_q.flags.writeable is False
    assert not (trajectory / ".chunks").exists()
    manifest = json.loads((trajectory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert "schema_version" not in manifest
    assert "state_profile" not in manifest


def test_reader_rejects_manifest_without_coordinate_frame(
    tmp_path: Path,
) -> None:
    trajectory = _write_trajectory(tmp_path / "trajectory")
    manifest_path = trajectory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("coordinate_frame")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsupported trajectory coordinate frame"):
        TrajectoryReader(trajectory)


def test_training_validation_excludes_unpaired_terminal_state(tmp_path: Path) -> None:
    trajectory = _write_trajectory(tmp_path / "trajectory")
    states_path = trajectory / "states.npz"
    with np.load(states_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["particle_q"][-1, 0, 0] = np.nan
    with states_path.open("wb") as file:
        np.savez_compressed(file, **payload)

    reader = TrajectoryReader(trajectory)
    reader.validate_training_data()
    assert np.isnan(reader.terminal_state().particle_q[0, 0])


def test_render_only_compaction_preserves_exact_render_inputs_and_actions(
    tmp_path: Path,
) -> None:
    source = _write_trajectory(tmp_path / "source")
    output = tmp_path / "render-only"
    original = TrajectoryReader(source)

    compact_render_trajectory(source, output)

    compact = TrajectoryReader(output)
    assert compact.is_render_only
    compact.validate_render_data()
    for field in ("joint_q", "body_q", "particle_q", "gripper_open_state"):
        assert np.array_equal(
            compact.state_arrays((field,))[field],
            original.state_arrays((field,))[field],
        )
    for field, values in original.action_arrays().items():
        assert np.array_equal(compact.action_arrays()[field], values)
    assert np.count_nonzero(compact.state_at(1).particle_qd) == 0
    assert set(np.load(output / "states.npz", allow_pickle=False).files) == {
        "sample_index",
        "joint_q",
        "body_q",
        "particle_q",
        "gripper_open_state",
    }
    with pytest.raises(ValueError, match="unavailable in a render-only"):
        compact.state_arrays(("particle_qd",))
    with pytest.raises(RuntimeError, match="cannot be used for dynamics"):
        compact.validate_training_data()
    assert compact.manifest.metadata["training_eligible"] is False


def test_direct_render_only_writer_records_selection_signals_and_hardlink_publishes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate"
    initial = _state(0)
    initial.particle_qd[:] = np.asarray([3.0, 4.0, 0.0], dtype=np.float32)
    initial.grasp_attached[:] = np.asarray([1, 0], dtype=np.int32)
    terminal = _state(1)
    terminal.particle_qd[:] = 0.0
    terminal.grasp_attached[:] = 0
    writer = RenderOnlyTrajectoryWriter(
        source,
        control_fps=50,
        sim_substeps=6,
        max_actions=1,
        metadata={"training_eligible": False},
    )
    writer.write_initial_state(initial)
    writer.append_transition(_action(0), terminal)
    writer.close()

    reader = TrajectoryReader(source)
    reader.validate_render_data()
    signals = load_render_selection_signals(source)
    assert reader.is_render_only
    assert set(np.load(source / "states.npz", allow_pickle=False).files) == {
        "sample_index",
        "joint_q",
        "body_q",
        "particle_q",
        "gripper_open_state",
    }
    assert signals["speed_rms_m_s"].tolist() == [5.0, 0.0]
    assert signals["released"].tolist() == [False, True]

    output = tmp_path / "published"
    publish_render_trajectory(source, output)

    assert (source / "states.npz").stat().st_ino == (output / "states.npz").stat().st_ino
    TrajectoryReader(output).validate_render_data()
    assert load_render_selection_signals(output)["released"].tolist() == [False, True]


def test_writer_rejects_shifted_action_index(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path / "trajectory", control_fps=30.0, sim_substeps=20)
    writer.write_initial_state(_state(0))

    with pytest.raises(ValueError, match="does not match source state"):
        writer.append_transition(_action(1), _state(1))

    writer.abort()
    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"


def test_writer_update_metadata_persists_into_manifest(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory"
    writer = TrajectoryWriter(trajectory, control_fps=30.0, sim_substeps=20)
    writer.write_initial_state(_state(0))
    writer.append_transition(_action(0), _state(1))
    writer.update_metadata(
        "semantic_subtask", {"chunks": [{"subtask": "fold the shirt"}]}
    )
    writer.close()

    manifest = TrajectoryManifest.from_json_dict(
        json.loads((trajectory / "manifest.json").read_text())
    )
    assert manifest.metadata["semantic_subtask"]["chunks"][0]["subtask"] == (
        "fold the shirt"
    )

    with pytest.raises(RuntimeError):
        writer.update_metadata("late", 1)
