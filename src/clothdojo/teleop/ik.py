"""Newton 1.5 dual-arm IK used by the XR control loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp


@wp.kernel
def _select_finite_solution(
    candidate: wp.array(dtype=float),
    last_safe: wp.array(dtype=float),
    command: wp.array(dtype=float),
    status: wp.array(dtype=int),
):
    index = wp.tid()
    value = candidate[index]
    if not wp.isfinite(value):
        command[index] = last_safe[index]
        wp.atomic_max(status, 0, 2)
    else:
        last_safe[index] = value
        command[index] = value


@wp.kernel
def _gather_arm_command(
    source: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    target: wp.array2d(dtype=float),
):
    index = wp.tid()
    world = index // 12
    joint = index - world * 12
    target[world, joint] = source[indices[index]]


@wp.kernel
def _override_home_command(
    home: wp.array2d(dtype=float),
    left: int,
    right: int,
    command: wp.array2d(dtype=float),
):
    index = wp.tid()
    world = index // 12
    joint = index - world * 12
    side = joint // 6
    if (side == 0 and left != 0) or (side == 1 and right != 0):
        command[world, joint] = home[world, joint]


@wp.kernel
def _clip_arm_command(
    lower: wp.array2d(dtype=float),
    upper: wp.array2d(dtype=float),
    command: wp.array2d(dtype=float),
):
    world, joint = wp.tid()
    command[world, joint] = wp.clamp(
        command[world, joint], lower[world, joint], upper[world, joint]
    )


@dataclass
class DualArmIKController:
    runtime: object
    iterations: int = 10
    use_cuda_graph: bool = True
    position_weight: float = 100.0
    rotation_weight: float = 1.0
    joint_limit_weight: float = 10.0
    solver: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import newton
        from newton.ik import (
            IKJacobianType,
            IKObjectiveJointLimit,
            IKObjectivePosition,
            IKObjectiveRotation,
            IKSolver,
        )

        if not isinstance(self.use_cuda_graph, bool):
            raise TypeError("use_cuda_graph must be a bool")
        self.simulation_model = self.runtime.model
        self.world_count = int(self.simulation_model.world_count)
        self.model = (
            self.runtime.scene.ik_model
            if self.world_count > 1
            else self.simulation_model
        )
        if self.model is None:
            raise ValueError("multi-world IK requires a single-world kinematic model")
        self.device = self.model.device
        if self.world_count < 1:
            raise ValueError("IK requires at least one world")
        if self.world_count > 1 and int(self.model.world_count) != 1:
            raise ValueError("the batched IK model must contain exactly one world")
        self._solve_graph = None
        left_ee_body, right_ee_body = self.runtime.scene.end_effector_body_indices
        bodies_per_world = int(self.simulation_model.body_count) // self.world_count
        coordinates_per_world = (
            int(self.simulation_model.joint_coord_count) // self.world_count
        )
        self._ee_bodies = tuple(
            (left_ee_body + world * bodies_per_world, right_ee_body + world * bodies_per_world)
            for world in range(self.world_count)
        )
        self.left_ee_body, self.right_ee_body = self._ee_bodies[0]
        self.joint_coord_count = int(self.model.joint_coord_count)
        if coordinates_per_world != self.joint_coord_count:
            raise ValueError("simulation and IK coordinate layouts differ")
        self._joint_q_in = wp.zeros(
            (self.world_count, self.joint_coord_count),
            dtype=wp.float32,
            device=self.device,
        )
        self._joint_q_out = wp.zeros_like(self._joint_q_in)
        flat_coord_count = self.world_count * self.joint_coord_count
        self._joint_q_in_flat = self._joint_q_in.reshape((flat_coord_count,))
        self._joint_q_out_flat = self._joint_q_out.reshape((flat_coord_count,))
        self._joint_q_safe = wp.clone(self.runtime.state_0.joint_q)
        self._joint_q_command = wp.clone(self.runtime.state_0.joint_q)
        self._solution_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        local_arm_indices = tuple(self.runtime.scene.arm_joint_coord_indices)
        self._arm_indices = wp.array(
            [
                index + world * coordinates_per_world
                for world in range(self.world_count)
                for index in local_arm_indices
            ],
            dtype=wp.int32,
            device=self.device,
        )
        self._arm_command = wp.zeros(
            (self.world_count, 12), dtype=wp.float32, device=self.device
        )
        local_arm_indices_np = np.asarray(local_arm_indices, dtype=np.int64)
        limit_lower = np.asarray(self.model.joint_limit_lower.numpy(), dtype=np.float32)
        limit_upper = np.asarray(self.model.joint_limit_upper.numpy(), dtype=np.float32)
        self._arm_lower = wp.array(
            np.tile(limit_lower[local_arm_indices_np], (self.world_count, 1)),
            dtype=wp.float32,
            device=self.device,
        )
        self._arm_upper = wp.array(
            np.tile(limit_upper[local_arm_indices_np], (self.world_count, 1)),
            dtype=wp.float32,
            device=self.device,
        )
        self._home_arm_command = wp.zeros_like(self._arm_command)
        self.sync_from_state()
        # Restored trajectories may already have changed state_0 before IK is created.
        # The authored robot posture, never the current trajectory pose, defines Home.
        home_state = self.model.state()
        home_q_ik = wp.array(
            self.runtime.scene.robot_profile.home_joint_q,
            dtype=wp.float32,
            device=self.device,
        )
        newton.eval_fk(
            self.model,
            home_q_ik,
            wp.zeros_like(self.model.joint_qd),
            home_state,
        )
        home_poses = home_state.body_q.numpy()
        self._home_poses = tuple(
            tuple(
                _pose_matrix(home_poses[index][:3], home_poses[index][3:])
                for index in (left_ee_body, right_ee_body)
            )
            for _ in range(self.world_count)
        )
        home_q_runtime = wp.array(
            np.tile(self.runtime.scene.robot_profile.home_joint_q, self.world_count),
            dtype=wp.float32,
            device=self.device,
        )
        wp.launch(
            _gather_arm_command,
            dim=self.world_count * 12,
            inputs=[home_q_runtime, self._arm_indices],
            outputs=[self._home_arm_command],
            device=self.device,
        )

        measured = self.measured_targets_batch()
        self._target_positions = []
        self._target_rotations = []
        objectives = []
        world_offsets = np.asarray(
            getattr(
                self.runtime.scene,
                "world_offsets",
                ((0.0, 0.0, 0.0),) * self.world_count,
            ),
            dtype=np.float32,
        )
        for side, body in enumerate((left_ee_body, right_ee_body)):
            position = wp.array(
                measured[:, side, :3] - world_offsets,
                dtype=wp.vec3,
                device=self.device,
            )
            rotation = wp.array(
                measured[:, side, 3:], dtype=wp.vec4, device=self.device
            )
            self._target_positions.append(position)
            self._target_rotations.append(rotation)
            objectives.extend(
                [
                    IKObjectivePosition(
                        body, wp.vec3(0.0), position, float(self.position_weight)
                    ),
                    IKObjectiveRotation(
                        body,
                        wp.quat_identity(),
                        rotation,
                        weight=float(self.rotation_weight),
                    ),
                ]
            )
        objectives.append(
            IKObjectiveJointLimit(
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                float(self.joint_limit_weight),
            )
        )
        self._world_offsets = world_offsets
        self.solver = IKSolver(
            model=self.model,
            n_problems=self.world_count,
            objectives=objectives,
            jacobian_mode=IKJacobianType.MIXED,
            lambda_initial=0.1,
        )

    def measured_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.world_count != 1:
            raise ValueError("measured_targets is only defined for one-world teleop")
        left, right = self.measured_targets_batch()[0]
        return left[:3], _normalize_quat(left[3:]), right[:3], _normalize_quat(right[3:])

    def measured_targets_batch(self) -> np.ndarray:
        poses = np.asarray(self.runtime.state_0.body_q.numpy(), dtype=np.float32)
        result = np.empty((self.world_count, 2, 7), dtype=np.float32)
        for world, pair in enumerate(self._ee_bodies):
            for side, index in enumerate(pair):
                pose = poses[index].reshape(7)
                result[world, side, :3] = pose[:3]
                result[world, side, 3:] = _normalize_quat(pose[3:])
        return result

    def measured_pose_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        left_pos, left_rot, right_pos, right_rot = self.measured_targets()
        return _pose_matrix(left_pos, left_rot), _pose_matrix(right_pos, right_rot)

    def home_pose_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        if self.world_count != 1:
            raise ValueError("home_pose_matrices is only defined for one-world teleop")
        return tuple(pose.copy() for pose in self._home_poses[0])

    def set_targets(self, left_pose: np.ndarray, right_pose: np.ndarray) -> None:
        if self.world_count != 1:
            raise ValueError("set_targets is only defined for one-world teleop")
        self.set_targets_batch(np.asarray(((left_pose, right_pose),)))

    def set_targets_batch(self, poses: np.ndarray) -> None:
        values = np.asarray(poses, dtype=np.float64)
        if values.shape != (self.world_count, 2, 4, 4):
            raise ValueError(
                f"batched IK targets must have shape {(self.world_count, 2, 4, 4)}"
            )
        for side in range(2):
            positions = np.stack(
                [values[world, side, :3, 3] for world in range(self.world_count)]
            ).astype(np.float32)
            positions -= self._world_offsets
            rotations = np.stack(
                [
                    _matrix_to_quat(values[world, side, :3, :3])
                    for world in range(self.world_count)
                ]
            ).astype(np.float32)
            self._target_positions[side].assign(positions)
            self._target_rotations[side].assign(rotations)

    def solve(self):
        if self.use_cuda_graph and wp.get_device(self.device).is_cuda:
            if self._solve_graph is None:
                self._capture_solve_graph()
            else:
                wp.capture_launch(self._solve_graph)
            return self._arm_command
        self._submit_solve()
        return self._arm_command

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._solve_graph is not None

    def _submit_solve(self) -> None:
        self.sync_from_state()
        self.solver.step(self._joint_q_in, self._joint_q_out, iterations=int(self.iterations))
        self._solution_status.zero_()
        wp.launch(
            _select_finite_solution,
            dim=self.world_count * self.joint_coord_count,
            inputs=[
                self._joint_q_out_flat,
                self._joint_q_safe,
                self._joint_q_command,
                self._solution_status,
            ],
            device=self.device,
        )
        wp.launch(
            _gather_arm_command,
            dim=self.world_count * 12,
            inputs=[self._joint_q_command, self._arm_indices],
            outputs=[self._arm_command],
            device=self.device,
        )
        wp.launch(
            _clip_arm_command,
            dim=(self.world_count, 12),
            inputs=[self._arm_lower, self._arm_upper],
            outputs=[self._arm_command],
            device=self.device,
        )

    def _capture_solve_graph(self) -> None:
        # Run once to compile Newton's specialized IK kernels. The result is
        # already valid for this call; later solves replay the captured graph.
        self._submit_solve()
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
            self._submit_solve()
        self._solve_graph = capture.graph

    def reset_from_state(self) -> None:
        self.solver.reset()
        self.sync_from_state()
        wp.copy(self._joint_q_safe, self.runtime.state_0.joint_q)
        wp.copy(self._joint_q_command, self.runtime.state_0.joint_q)
        measured = self.measured_targets_batch()
        poses = np.empty((self.world_count, 2, 4, 4), dtype=np.float64)
        for world in range(self.world_count):
            for side in range(2):
                poses[world, side] = _pose_matrix(
                    measured[world, side, :3], measured[world, side, 3:]
                )
        self.set_targets_batch(poses)

    def override_home(self, command, *, left: bool, right: bool) -> None:
        """Replace selected arm targets with the configured Home posture on device."""

        if not left and not right:
            return
        wp.launch(
            _override_home_command,
            dim=self.world_count * 12,
            inputs=[self._home_arm_command, int(left), int(right)],
            outputs=[command],
            device=self.device,
        )

    def solution_status(self) -> int:
        return int(self._solution_status.numpy()[0])

    def sync_from_state(self) -> None:
        wp.copy(self._joint_q_in_flat, self.runtime.state_0.joint_q)


def _normalize_quat(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-8:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    return value / norm


def _pose_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quat(quaternion)
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def _matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        value = (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
            0.25 * scale,
        )
        value = (value[0] / scale, value[1] / scale, value[2] / scale, value[3])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            value = (
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            value = (
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            value = (
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
    return _normalize_quat(np.asarray(value, dtype=np.float32))
