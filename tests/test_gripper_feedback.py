import unittest

import numpy as np

from clothdojo.trajectory.newton import gripper_openness


class GripperFeedbackTests(unittest.TestCase):
    def test_piper_leader_endpoints_apply_to_both_sides(self):
        positions = np.asarray(
            [[0.04, -0.04, 0.04, -0.04], [0, 0, 0.04, -0.04],
             [0.04, -0.04, 0, 0], [0.01, -0.01, 0.03, -0.03]],
            dtype=np.float32,
        )
        limits = np.zeros_like(positions)
        openness = gripper_openness(
            positions, limits, limits,
            open_positions=(0.04, -0.04), closed_positions=(0, 0),
        )
        np.testing.assert_allclose(openness, [[1, 1], [0, 1], [1, 0], [0.25, 0.75]])

    def test_one_coordinate_gripper_with_reversed_endpoints(self):
        positions = np.asarray([[0, 0.4], [0.8, 0]], dtype=np.float32)
        limits = np.zeros_like(positions)
        openness = gripper_openness(
            positions, limits, limits,
            open_positions=(0,), closed_positions=(0.8,),
        )
        np.testing.assert_allclose(openness, [[1, 0.5], [0, 1]])


if __name__ == '__main__':
    unittest.main()
