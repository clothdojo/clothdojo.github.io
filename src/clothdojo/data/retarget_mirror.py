"""Reuse a UR5e retarget through numeric reflection and one actual action replay."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from clothdojo.data.retarget import (
    _attach_poses_from_body,
    _direction_angle,
    _quat_rotate,
    piper_attach_poses,
)
from clothdojo.data.retarget_dynamics import (
    _event_position_validation,
    _gripper_event_validation,
)
from clothdojo.robots import UR5E_2F85, robot_profile
from clothdojo.trajectory import TrajectoryReader

# USD joint zero frames: q' = offset - opposite_arm_q. The final pi keeps
# each Robotiq finger's own coordinate layout (no finger permutation).
_ARM_OFFSETS = np.asarray((np.pi, np.pi, 0.0, np.pi, 0.0, np.pi))


def mirror_arm_offsets(
    coordinates: list[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Choose one legal 2pi branch per arm joint for the entire trajectory.

    State, reference and command ranges participate together. Per-frame angle
    wrapping would introduce discontinuous drive commands and is not allowed.
    """
    values = np.concatenate([np.asarray(q).reshape(-1, 2, 6) for q in coordinates])
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("mirror arm coordinates must be finite and nonempty")
    lower = np.asarray(lower).reshape(2, 6)
    upper = np.asarray(upper).reshape(2, 6)
    reflected = -values[:, ::-1]
    offsets = np.empty((2, 6), dtype=np.float64)
    for side in range(2):
        for joint in range(6):
            low, high = reflected[:, side, joint].min(), reflected[:, side, joint].max()
            choices = []
            for turns in range(-2, 3):
                offset = _ARM_OFFSETS[joint] + turns * 2 * np.pi
                margin = min(low + offset - lower[side, joint], upper[side, joint] - high - offset)
                if margin >= 0:
                    choices.append((margin, offset))
            if not choices:
                raise ValueError(f"no continuous legal mirror branch for arm {side} joint {joint}")
            offsets[side, joint] = max(choices)[1]
    return offsets


def mirror_ur5e_q(q: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Reflect reference/initial coordinates with identity closure quaternions."""
    values = np.asarray(q, dtype=np.float32)
    if values.shape[-1] != 44:
        raise ValueError("UR5e + 2F85 reflection requires its 44-coordinate layout")
    result = values.reshape(*values.shape[:-1], 2, 22)[..., ::-1, :].copy()
    if not np.allclose(result[..., 14:], np.tile([0, 0, 0, 1], 2), atol=1e-6):
        raise ValueError("reflection requires identity closure quaternions")
    result[..., :6] = -result[..., :6] + offsets
    return result.reshape(values.shape)


def mirror_ur5e_qd(qd: np.ndarray) -> np.ndarray:
    """Reflect reference/initial velocities with stationary closure coordinates."""
    values = np.asarray(qd, dtype=np.float32)
    if values.shape[-1] != 40:
        raise ValueError("UR5e + 2F85 reflection requires its 40-DOF layout")
    result = values.reshape(*values.shape[:-1], 2, 20)[..., ::-1, :].copy()
    if not np.allclose(result[..., 14:], 0, atol=1e-6):
        raise ValueError("reflection requires stationary closure coordinates")
    result[..., :6] *= -1
    return result.reshape(values.shape)


def retarget_mirror_trajectory(
    original: Path,
    source: Path,
    output: Path,
    *,
    kinematics,
    runtime,
) -> dict:
    """Publish a reflected target using the same source-event position gates.

    `original` is the original full derived target (including initial velocity).
    Cloth and source event annotations come from the already mirrored input.
    All published robot states come from the mapped actions' cold replay.
    """
    started = perf_counter()
    reader = TrajectoryReader(original)
    source_reader = TrajectoryReader(source)
    if (
        reader.manifest.scene.robot_id != UR5E_2F85
        or source_reader.manifest.scene.robot_id != "piper"
    ):
        raise ValueError("numeric retarget reflection requires UR5e target and Piper source")
    if not reader.manifest.metadata["retarget"]["validation"]["training_eligible"]:
        raise ValueError("original retarget must pass before mirror reuse")
    if source_reader.manifest.metadata.get("augmentation", {}).get("kind") != "bilateral_mirror":
        raise ValueError("mirror reuse requires a recorded bilateral mirror source")
    if reader.manifest.state_count != source_reader.manifest.state_count:
        raise ValueError("original and mirrored episode ranges differ")
    if (
        reader.manifest.control_fps != source_reader.manifest.control_fps
        or reader.manifest.sim_substeps != source_reader.manifest.sim_substeps
    ):
        raise ValueError("original and mirror simulation timing differs")
    profile = robot_profile(UR5E_2F85)
    arms = np.asarray(profile.arm_coord_indices)
    dofs = np.asarray(profile.arm_dof_indices)
    source_states = source_reader.state_arrays(
        ("body_q", "grasp_attached", "grasp_previous_open", "grasp_pins", "grasp_pin_local")
    )
    source_poses = piper_attach_poses(source_states["body_q"])
    target = source_poses[..., :3]
    attached = source_states["grasp_attached"]
    render_only = reader.manifest.metadata.get("storage", {}).get("kind") == "render_only_v1"
    states = reader.state_arrays(("joint_q",) if render_only else ("joint_q", "joint_qd", "grasp_attached"))
    actions = reader.action_arrays()
    with np.load(original / "retarget_metrics.npz", allow_pickle=False) as archive:
        previous = {key: archive[key] for key in archive.files}
    if not np.allclose(
        target, previous["source_attach_position"][:, ::-1] * [1, -1, 1], atol=5e-6, rtol=0
    ):
        raise ValueError("source frames do not match the original reflected centers")
    if not render_only and not np.array_equal(attached, states["grasp_attached"][:, ::-1]):
        raise ValueError("mirror source attachment event timeline differs")
    offsets = mirror_arm_offsets(
        [
            states["joint_q"][:, arms],
            previous["reference_joint_q"][:, arms],
            actions["joint_target"],
        ],
        runtime.model.joint_limit_lower.numpy()[dofs],
        runtime.model.joint_limit_upper.numpy()[dofs],
    )
    reference_q = mirror_ur5e_q(previous["reference_joint_q"], offsets)
    reference_qd = mirror_ur5e_qd(previous["reference_joint_qd"])
    initial_q = mirror_ur5e_q(states["joint_q"][0], offsets)
    # Render-only retargets deliberately omit velocity. The dynamics retargeter
    # was initialized from the preserved reference velocity, so it is the exact
    # initial condition needed to replay the mirrored action sequence.
    source_initial_qd = (
        previous["reference_joint_qd"][0] if render_only else states["joint_qd"][0]
    )
    initial_qd = mirror_ur5e_qd(source_initial_qd)
    # Loop-closure coordinates are passive. Existing retarget starts them at
    # identity/zero; do not guess a general reflected quaternion velocity map.
    for side in range(2):
        if not np.allclose(
            states["joint_q"][0, side * 22 + 14 : side * 22 + 22], [0, 0, 0, 1] * 2, atol=1e-7
        ):
            raise ValueError("mirror replay requires identity initial closure coordinates")
        if not np.allclose(source_initial_qd[side * 20 + 14 : side * 20 + 20], 0, atol=1e-7):
            raise ValueError("mirror replay requires zero initial closure velocity")
    arm_target = (
        (-actions["joint_target"].reshape(-1, 2, 6)[:, ::-1] + offsets)
        .reshape(-1, 12)
        .astype(np.float32)
    )
    gripper_command = actions["gripper_open_command"][:, ::-1].copy()
    reference_poses = kinematics.attach_poses_batch(reference_q)
    geometry_errors = np.linalg.norm(reference_poses[..., :3] - target, axis=-1)
    geometry = _event_position_validation(geometry_errors, attached, tolerance_m=0.005)
    runtime.restore_rigid_state(initial_q, initial_qd)
    replay = {
        name: values[:, 0]
        for name, values in runtime.rollout_rigid(
            arm_target[:, None],
            gripper_command[:, None],
        ).items()
    }
    poses = _attach_poses_from_body(
        replay["body_q"],
        profile.attach_frames,
        tuple(profile.body_names.index(frame.parent_body) for frame in profile.attach_frames),
    )
    errors = np.linalg.norm(poses[..., :3] - target, axis=-1)
    dynamics = _event_position_validation(errors, attached, tolerance_m=0.01)
    fingers = replay["joint_q"][:, profile.gripper_coord_indices].reshape(-1, 2, 2)
    finger_open = (fingers - np.asarray(profile.gripper_closed_positions)) / (
        np.asarray(profile.gripper_open_positions) - np.asarray(profile.gripper_closed_positions)
    )
    gripper_open = np.clip(finger_open.mean(axis=-1), 0, 1).astype(np.float32)
    event_diagnostic = _gripper_event_validation(finger_open, attached != 0)
    event_diagnostic["required_for_acceptance"] = False
    dynamic_validation = {
        "dynamics_valid": dynamics["valid"],
        "event_position_validation": dynamics,
        "acceptance_scope": "source_attach_release_positions_v2",
        "actuator_effort_scale": 1.0,
        "source_timeline_preserved": True,
        "robot_dynamics_rollout": True,
        "style3d_solver_used": False,
        "collision_checked": False,
        "optimizer_evaluations": 0,
        "trial_evaluations": 0,
        "max_attach_position_error_m": float(errors.max()),
        "gripper_event_validation": event_diagnostic,
        "limits": {"attach_position_error_m": 0.01},
        "mirror_original_replay_max_center_difference_m": float(
            np.linalg.norm(
                poses[..., :3] - previous["dynamics_attach_position"][:, ::-1] * [1, -1, 1], axis=-1
            ).max()
        ),
    }
    validation = {
        "acceptance_scope": "source_attach_release_positions_v2",
        "kinematic_valid": geometry["valid"],
        "dynamics_valid": dynamics["valid"],
        "training_eligible": bool(geometry["valid"] and dynamics["valid"]),
        "event_position_validation": geometry,
        "robot_dynamics": dynamic_validation,
        "source_timeline_preserved": True,
        "position_tolerance_m": 0.005,
        "orientation_constraint_scope": "none",
        "orientation_preference": "reflected_original_reference",
        "collision_checked": False,
        "max_attach_position_error_m": float(geometry_errors.max()),
        "source_state_count": len(reference_q),
        "output_state_count": len(reference_q),
    }
    manifest = copy.deepcopy(reader.manifest.to_json_dict())
    metadata = manifest["metadata"]
    metadata["source_metadata"] = copy.deepcopy(source_reader.manifest.metadata)
    metadata["action_semantics"] = "runtime_target"
    metadata["storage"] = {
        "kind": "derived_overlay",
        "base_trajectory": str(source),
        "base_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
    }
    metadata["retarget"].update(
        source_trajectory=str(source),
        validation=validation,
        mirror_reuse={
            "original_trajectory": str(original),
            "original_manifest_sha256": hashlib.sha256(
                (original / "manifest.json").read_bytes()
            ).hexdigest(),
            "arm_signs": [-1] * 6,
            "arm_offsets_radians": offsets.tolist(),
            "gripper_hands_swapped": True,
            "finger_order_preserved": True,
            "ik_optimization_used": False,
            "shooting_used": False,
            "action_only_replay_used": True,
        },
    )
    metrics = {
        "source_attach_position": target,
        "source_attach_rotation": source_poses[..., 3:],
        "reference_joint_q": reference_q,
        "reference_joint_qd": reference_qd,
        "kinematic_attach_position": reference_poses[..., :3],
        "kinematic_attach_position_error": geometry_errors,
        "dynamics_attach_position": poses[..., :3],
        "dynamics_attach_position_error": errors,
        "pregrasp_mask": np.zeros_like(attached, dtype=bool),
        "approach_validation_mask": np.zeros_like(attached, dtype=bool),
    }
    direction = previous["target_approach_direction"][:, ::-1] * [1, -1, 1]
    metrics["target_approach_direction"] = direction.astype(np.float32)
    for name, value in [("kinematic", reference_poses), ("dynamics", poses)]:
        axis = _quat_rotate(value[..., 3:], np.asarray([0.0, 0.0, 1.0]))
        metrics[name + "_approach_direction"] = axis.astype(np.float32)
        metrics[name + "_approach_error"] = _direction_angle(axis, direction).astype(np.float32)
    if "target_up_direction" in previous:
        metrics["target_up_direction"] = (
            previous["target_up_direction"][:, ::-1] * [1, -1, 1]
        ).astype(np.float32)
    state_arrays = {
        "sample_index": np.arange(len(reference_q), dtype=np.int64),
        **{key: replay[key] for key in ["joint_q", "joint_qd", "body_q", "body_qd"]},
        "gripper_open_state": gripper_open,
        "grasp_attached": attached,
        "grasp_previous_open": source_states["grasp_previous_open"],
        "grasp_pin_count": np.zeros_like(attached, dtype=np.int32),
        "grasp_pins": np.full_like(source_states["grasp_pins"], -1),
        "grasp_pin_local": np.zeros_like(source_states["grasp_pin_local"]),
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "states.npz", **state_arrays)
    np.savez_compressed(
        output / "actions.npz",
        sample_index=np.arange(len(arm_target), dtype=np.int64),
        joint_target=arm_target,
        gripper_open_command=gripper_command,
    )
    np.savez_compressed(output / "retarget_metrics.npz", **metrics)
    metadata["retarget"]["mirror_reuse"]["seconds"] = perf_counter() - started
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
