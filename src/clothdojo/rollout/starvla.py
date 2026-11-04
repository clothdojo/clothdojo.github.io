"""MR0 StarVLA rollout bridge: pickle-TCP client and the EEF command episode."""

from __future__ import annotations

import pickle
import socket
import struct

import numpy as np

STARVLA_ENDPOINT_SCHEME = "starvla"
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
ACTION_DIM = 20
LEFT_XYZ = slice(0, 3)
LEFT_ROT6D = slice(3, 9)
LEFT_GRIP = 9
RIGHT_XYZ = slice(10, 13)
RIGHT_ROT6D = slice(13, 19)
RIGHT_GRIP = 19


def is_starvla_endpoint(endpoint: object) -> bool:
    """True for policy endpoints served by the MR0 StarVLA pickle protocol."""

    return (
        isinstance(endpoint, str)
        and endpoint.startswith(STARVLA_ENDPOINT_SCHEME + "://")
    )


def starvla_endpoint_host_port(endpoint: str) -> tuple[str, int]:
    """Accept both ``host:port`` and ``starvla://host:port`` endpoint forms."""

    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("policy endpoint must be a non-empty string")
    value = endpoint.strip()
    prefix = STARVLA_ENDPOINT_SCHEME + "://"
    if value.startswith(prefix):
        value = value[len(prefix):]
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise ValueError(f"invalid StarVLA policy endpoint: {endpoint!r}")
    try:
        number = int(port)
    except ValueError as exc:
        raise ValueError(f"invalid StarVLA policy endpoint port: {endpoint!r}") from exc
    if not 1 <= number <= 65535:
        raise ValueError(f"StarVLA policy endpoint port out of range: {endpoint!r}")
    return host, number


def _fixed_axis_rpy(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def _action_frame_fk(q6):
    """4x4 pose of the action-frame EE in the arm base frame.

    The fixed origins come from tools/piper_kinematics/piper_arm_kinematics.urdf
    (joint1..joint6); this FK matches the training-label FK bit for bit and is
    only used for the episode-begin frame consistency check.
    """

    origins = (
        ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.5708, -0.1359, -3.1416)),
        ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.7939)),
        ((-0.021984, -0.25075, 0.0), (1.5708, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),
        ((8.8259e-05, -0.091, 0.0), (1.5708, 0.0, 0.0)),
    )
    matrix = np.eye(4)
    for angle, (xyz, rpy) in zip(q6, origins, strict=True):
        origin = np.eye(4)
        origin[:3, :3] = _fixed_axis_rpy(*rpy)
        origin[:3, 3] = xyz
        rotation = np.eye(4)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation[:3, :3] = ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        matrix = matrix @ origin @ rotation
    return matrix


def _quat_xyzw_to_matrix(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def _pose_matrix(position, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def _plus90_z_pose():
    """The constant action-frame to Newton-link6 rotation Rz(+90 degrees)."""

    matrix = np.eye(4)
    matrix[:3, :3] = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    return matrix


def _rot6d_to_matrix(rot6d):
    """Robust Gram-Schmidt 6-D rotation to a proper 3x3 matrix."""

    values = np.asarray(rot6d, dtype=np.float64).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError(f"rot6d must be finite with shape (6,), got {values.shape}")
    first = values[:3]
    second = values[3:]
    norm = float(np.linalg.norm(first))
    axis_x = first / norm if norm > 1e-6 else np.asarray((1.0, 0.0, 0.0))
    residual = second - float(axis_x @ second) * axis_x
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm > 1e-6:
        axis_y = residual / residual_norm
    else:
        fallback = np.eye(3)[int(np.argmin(np.abs(axis_x)))]
        fallback = fallback - float(axis_x @ fallback) * axis_x
        axis_y = fallback / float(np.linalg.norm(fallback))
    axis_z = np.cross(axis_x, axis_y)
    return np.column_stack((axis_x, axis_y, axis_z))


def _rotation_angle(rotation):
    cosine = np.clip((float(np.trace(rotation)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


class StarVLAPolicyAdapter:
    """Rollout policy backed by the py3.11 MR0 StarVLA policy server.

    The worker sends the canonical 14-D joint state plus the three policy
    cameras with every request; the server owns the EEF state FK, state
    normalization and action unnormalization, and returns one absolute 20-D
    action chunk per request. The connection persists for the lifetime of the
    adapter, so one multi-worker server can isolate per-connection RNG state.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        expected_checkpoint_sha256: str | None = None,
        action_horizon: int | None = None,
    ) -> None:
        if expected_checkpoint_sha256 is not None:
            _require_sha256(expected_checkpoint_sha256, "expected StarVLA checkpoint")
        invalid_horizon = (
            isinstance(action_horizon, bool)
            or not isinstance(action_horizon, int)
            or action_horizon < 1
        )
        if action_horizon is not None and invalid_horizon:
            raise ValueError("action_horizon must be a positive integer")
        self._address = starvla_endpoint_host_port(endpoint)
        self._socket: socket.socket | None = None
        identity = self._request({"identity": True})["identity"]
        space = str(identity.get("action_space"))
        if space != "eef_abs_20d_action_frame":
            raise ValueError(f"unexpected action space: {space}")
        if int(identity.get("action_dim", 0)) != ACTION_DIM:
            raise ValueError("the StarVLA server must serve 20-D actions")
        self.action_horizon = int(identity["action_horizon"])
        if action_horizon is not None and self.action_horizon != action_horizon:
            raise ValueError(
                "StarVLA action horizon differs from the scenario: "
                f"{self.action_horizon} != {action_horizon}"
            )
        if expected_checkpoint_sha256 is not None:
            reported = identity.get("checkpoint_sha256")
            if reported != expected_checkpoint_sha256:
                raise ValueError(
                    "StarVLA checkpoint sha256 mismatch: "
                    f"{reported!r} != {expected_checkpoint_sha256!r}"
                )
        self.identity = identity

    def _connect(self) -> socket.socket:
        if self._socket is None:
            self._socket = socket.create_connection(self._address, timeout=3600)
        return self._socket

    @staticmethod
    def _recv_exact(sock: socket.socket, count: int) -> bytes:
        chunks = []
        remaining = count
        while remaining:
            chunk = sock.recv(min(remaining, 1 << 20))
            if not chunk:
                raise ConnectionError("starvla policy server closed the connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _request(self, payload: dict) -> dict:
        data = pickle.dumps(payload, protocol=5)
        sock = self._connect()
        sock.sendall(struct.pack("<Q", len(data)) + data)
        (length,) = struct.unpack("<Q", self._recv_exact(sock, 8))
        response = pickle.loads(self._recv_exact(sock, length))
        if "error" in response:
            raise RuntimeError("starvla policy server: " + str(response["error"]))
        return response

    def reset(self, seed: int, *, mode: str, exec_horizon: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("policy seed must be a non-negative integer")
        if mode != "chunk":
            raise ValueError("the StarVLA rollout executes chunk mode")
        if not 1 <= exec_horizon <= self.action_horizon:
            raise ValueError("exec_horizon must be in [1, action_horizon]")
        self._request({"reset": int(seed)})

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
        *,
        inference_delay_steps: int = 0,
    ) -> np.ndarray:
        if inference_delay_steps:
            raise ValueError("the StarVLA rollout does not use RTC delays")
        state = np.asarray(observation["observation.state"])
        if state.ndim != 2 or state.shape[1] != 14:
            raise ValueError(f"expected [batch, 14] observation.state, got {state.shape}")
        payload: dict = {"task": prompt}
        payload["observation.state"] = np.ascontiguousarray(state, dtype=np.float64)
        for key in CAMERA_KEYS:
            images = np.asarray(observation[key])
            if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(
                    f"expected uint8 [batch, H, W, 3] images for {key}, got {images.shape}"
                )
            payload[key] = np.ascontiguousarray(images)
        actions = np.asarray(self._request(payload)["actions"], dtype=np.float64)
        expected = (state.shape[0], self.action_horizon, ACTION_DIM)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"unexpected StarVLA actions: {actions.shape}")
        return actions


class StarVLAEFCommandEpisode:
    """Episode wrapper converting 20-D absolute EEF targets into 14-D commands.

    Action semantics follow the MR0 eef6d training data: per-arm absolute
    [xyz, rot6d, gripper-openness] targets in the fixed per-arm action frame
    (the piper_arm_kinematics link6 frame). The Newton link6 body frame equals
    the action frame composed with a constant +90 degree z rotation, so
    targets convert to world-frame IK poses through the static arm-base
    transforms; the relation is verified against the live state at begin().
    """

    action_dim = ACTION_DIM

    def __init__(self, episode, runtime) -> None:
        from clothdojo.teleop.ik import DualArmIKController

        self.episode = episode
        self._world_count = int(episode.world_count)
        self.ik = DualArmIKController(runtime)
        names = list(runtime.scene.robot_profile.body_names)
        self._base_index = (names.index("left/base_link"), names.index("right/base_link"))
        self._plus90 = _plus90_z_pose()
        self._base_poses: np.ndarray | None = None
        self._last_targets: np.ndarray | None = None
        self.diagnostics: list[dict] = []
        self.frame_check: list[dict] | None = None
        self._step_index = 0
        self._ik_fallbacks = 0

    @property
    def world_count(self) -> int:
        return self.episode.world_count

    def observe(self) -> dict[str, np.ndarray]:
        return self.episode.observe()

    def synchronize(self) -> None:
        self.episode.synchronize()

    def _measured_ee_poses(self) -> np.ndarray:
        measured = self.ik.measured_targets_batch()
        poses = np.empty((self._world_count, 2, 4, 4), dtype=np.float64)
        for world in range(self._world_count):
            for side in range(2):
                poses[world, side] = _pose_matrix(
                    measured[world, side, :3],
                    _quat_xyzw_to_matrix(measured[world, side, 3:]),
                )
        return poses

    def begin(self) -> None:
        """Record arm-base transforms and verify the action-frame relation."""

        self.ik.reset_from_state()
        bodies = np.asarray(self.ik.runtime.state_0.body_q.numpy(), dtype=np.float64)
        bodies_per_world = len(bodies) // self._world_count
        base_poses = np.empty((self._world_count, 2, 4, 4), dtype=np.float64)
        for world in range(self._world_count):
            for side, local_index in enumerate(self._base_index):
                body = bodies[local_index + world * bodies_per_world]
                base_poses[world, side] = _pose_matrix(
                    body[:3], _quat_xyzw_to_matrix(body[3:])
                )
        self._base_poses = base_poses
        observation = self.episode.observe()
        joints = np.asarray(observation["observation.state"], dtype=np.float64)
        measured = self._measured_ee_poses()
        checks = []
        for world in range(self._world_count):
            for side, arm_slice, label in (
                (0, slice(0, 6), "left"), (1, slice(7, 13), "right")
            ):
                fk_pose = _action_frame_fk(joints[world, arm_slice])
                predicted = base_poses[world, side] @ fk_pose @ self._plus90
                position_error = float(
                    np.linalg.norm(predicted[:3, 3] - measured[world, side, :3, 3])
                )
                angle_error = _rotation_angle(
                    predicted[:3, :3].T @ measured[world, side, :3, :3]
                )
                check = {
                    "world": world,
                    "side": label,
                    "position_error_mm": position_error * 1000.0,
                    "rotation_error_deg": float(np.degrees(angle_error)),
                }
                checks.append(check)
                if position_error > 0.002 or np.degrees(angle_error) > 0.5:
                    raise ValueError(f"action-frame consistency check failed: {check}")
        self.frame_check = checks
        self._last_targets = None
        self.diagnostics = []
        self._step_index = 0
        self._ik_fallbacks = 0

    def step(self, eef_actions: np.ndarray) -> None:
        actions = np.asarray(eef_actions, dtype=np.float64)
        if actions.shape != (self._world_count, ACTION_DIM):
            raise ValueError(
                f"EEF actions must have shape {(self._world_count, ACTION_DIM)}"
            )
        if self._base_poses is None:
            raise RuntimeError("call begin() before the first step")
        measured = self._measured_ee_poses()
        targets = np.empty((self._world_count, 2, 4, 4), dtype=np.float64)
        record: dict = {"step": self._step_index}
        errors = {
            "left_pos_err": [], "left_rot_err": [],
            "right_pos_err": [], "right_rot_err": [],
        }
        for world in range(self._world_count):
            action = actions[world]
            for side, xyz_slice, rot_slice, label in (
                (0, LEFT_XYZ, LEFT_ROT6D, "left"),
                (1, RIGHT_XYZ, RIGHT_ROT6D, "right"),
            ):
                rotation = _rot6d_to_matrix(action[rot_slice])
                targets[world, side] = _pose_matrix(action[xyz_slice], rotation)
                if self._last_targets is not None:
                    achieved = (
                        np.linalg.inv(self._base_poses[world, side])
                        @ measured[world, side]
                        @ np.linalg.inv(self._plus90)
                    )
                    previous = self._last_targets[world, side]
                    errors[f"{label}_pos_err"].append(
                        float(np.linalg.norm(achieved[:3, 3] - previous[:3, 3]))
                    )
                    errors[f"{label}_rot_err"].append(
                        _rotation_angle(previous[:3, :3].T @ achieved[:3, :3])
                    )
        if self._last_targets is not None:
            record.update({name: float(np.max(values)) for name, values in errors.items()})
            self.diagnostics.append(record)
        self._last_targets = targets.copy()
        ik_targets = np.empty_like(targets)
        for world in range(self._world_count):
            for side in range(2):
                ik_targets[world, side] = (
                    self._base_poses[world, side]
                    @ targets[world, side]
                    @ self._plus90
                )
        self.ik.set_targets_batch(ik_targets)
        arm = np.asarray(self.ik.solve().numpy(), dtype=np.float32).reshape(
            self._world_count, 12
        )
        if self.ik.solution_status() != 0:
            self._ik_fallbacks += 1
        gripper = np.clip(actions[:, [LEFT_GRIP, RIGHT_GRIP]], 0.0, 1.0).astype(
            np.float32
        )
        command = np.concatenate(
            (arm[:, :6], gripper[:, :1], arm[:, 6:], gripper[:, 1:]), axis=1
        ).astype(np.float32)
        if not np.isfinite(command).all():
            raise FloatingPointError("non-finite bridged joint command")
        self.episode.step(command)
        self._step_index += 1
