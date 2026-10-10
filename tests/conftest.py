from pathlib import Path
import numpy as np
import pytest
from clothdojo.trajectory.schema import TrajectoryState, TrajectoryAction

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def root():
    return ROOT


@pytest.fixture
def state_factory():
    def make(index=0):
        body = np.zeros((21, 7), np.float32)
        body[:, 6] = 1
        return TrajectoryState(
            index,
            np.zeros(16),
            np.zeros(16),
            body,
            np.zeros((21, 6)),
            np.zeros((4, 3)),
            np.zeros((4, 3)),
            np.ones(2),
            np.zeros(2),
            np.ones(2),
            np.zeros(2),
            np.full((2, 2), -1),
            np.zeros((2, 2, 3)),
        )

    return make


@pytest.fixture
def action_factory():
    return lambda index=0: TrajectoryAction(
        index, np.arange(12, dtype=np.float32), np.array([0, 1], dtype=np.float32)
    )
