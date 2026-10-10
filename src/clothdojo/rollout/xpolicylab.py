"""ClothDojo's environment boundary for XPolicyLab policy deployments."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from clothdojo.rollout.engine import Episode
from clothdojo.rollout.scenario import RolloutScenario

_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
_ACTION_KEYS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)


class ClothDojoTaskEnv:
    """XPolicyLab task API backed by a synchronized ClothDojo episode batch.

    All worlds share one scenario and finish at its fixed control-step limit.
    The benchmark owns that limit; a policy's action-chunk length does not alter it.
    """

    def __init__(
        self,
        episode: Episode,
        scenario: RolloutScenario,
        *,
        frequency: int = 50,
        readback_interval: int = 10,
    ):
        if episode.action_dim != 14:
            raise ValueError("XPolicyLab joint adapter requires 14-dimensional actions")
        if episode.world_count < 1:
            raise ValueError("episode must contain at least one world")
        if frequency < 1:
            raise ValueError("frequency must be positive")
        if not 1 <= readback_interval <= scenario.action_horizon:
            raise ValueError("readback interval must fit the trajectory buffer")
        self.episode = episode
        self.scenario = scenario
        self.frequency = frequency
        self.readback_interval = readback_interval
        self.steps = 0

    @property
    def world_count(self) -> int:
        return self.episode.world_count

    def is_episode_end(self) -> bool:
        return self.steps >= self.scenario.steps

    def get_running_env_idx_list(self) -> list[int]:
        return [] if self.is_episode_end() else list(range(self.world_count))

    def get_obs(self) -> dict[str, Any]:
        if self.world_count != 1:
            raise ValueError("get_obs is for one world; use get_obs_batch")
        return self.get_obs_batch([0])[0]

    def get_obs_batch(self, env_idx_list: Sequence[int] | None = None) -> list[dict[str, Any]]:
        indices = list(range(self.world_count)) if env_idx_list is None else list(env_idx_list)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("environment indices must be distinct and non-empty")
        if any(type(index) is not int or not 0 <= index < self.world_count for index in indices):
            raise ValueError("environment index out of range")
        batch = self.episode.observe()
        state = np.asarray(batch["observation.state"])
        if state.shape != (self.world_count, 14) or not np.isfinite(state).all():
            raise ValueError("invalid ClothDojo state batch")
        images = {}
        for camera in _CAMERAS:
            image = np.asarray(batch[f"observation.images.{camera}"])
            if (image.ndim != 4 or image.shape[0] != self.world_count or
                    image.shape[-1] != 3 or image.dtype != np.uint8):
                raise ValueError(f"invalid ClothDojo RGB image batch: {camera}")
            images[camera] = image
        result = []
        for index in indices:
            row = state[index]
            result.append({
                "env_idx": index,
                "instruction": self.scenario.prompt,
                "additional_info": {"frequency": self.frequency},
                "state": {
                    "left_arm_joint_state": row[:6].copy(),
                    "left_ee_joint_state": row[6:7].copy(),
                    "right_arm_joint_state": row[7:13].copy(),
                    "right_ee_joint_state": row[13:14].copy(),
                },
                "vision": {camera: {"color": images[camera][index]} for camera in _CAMERAS},
            })
        return result

    def take_action(self, action: Mapping[str, Any]) -> None:
        if self.world_count != 1:
            raise ValueError("take_action is for one world; use take_action_batch")
        self.take_action_batch([action], [0])

    def take_action_batch(
        self,
        actions_list: Sequence[Mapping[str, Any]],
        env_idx_list: Sequence[int] | None = None,
    ) -> None:
        if self.is_episode_end():
            raise RuntimeError("episode has already reached its step limit")
        indices = list(range(self.world_count)) if env_idx_list is None else list(env_idx_list)
        if indices != list(range(self.world_count)) or len(actions_list) != self.world_count:
            raise ValueError("ClothDojo's fixed-horizon batch requires one action per world")
        actions = np.stack([_pack_joint_action(action) for action in actions_list])
        self.episode.step(actions)
        self.steps += 1
        if self.steps % self.readback_interval == 0:
            self.episode.synchronize()

    def synchronize(self) -> None:
        self.episode.synchronize()


def _pack_joint_action(action: Mapping[str, Any]) -> np.ndarray:
    if not isinstance(action, Mapping) or set(action) != {key for key, _ in _ACTION_KEYS}:
        raise ValueError("XPolicyLab action must contain exactly the four Piper joint/gripper keys")
    parts = []
    for key, size in _ACTION_KEYS:
        part = np.asarray(action[key], dtype=np.float32)
        if part.shape != (size,) or not np.isfinite(part).all():
            raise ValueError(f"invalid {key}: expected finite shape ({size},)")
        parts.append(part)
    return np.concatenate(parts)


def run_xpolicylab_deploy(
    task_env: ClothDojoTaskEnv,
    model_client: Any,
    *,
    policy_name: str,
) -> dict[str, int | str]:
    """Run the policy's own XPolicyLab deployment loop against ClothDojo."""

    if not policy_name.isidentifier():
        raise ValueError("policy_name must be an XPolicyLab Python package name")
    deploy = importlib.import_module(f"XPolicyLab.policy.{policy_name}.deploy")
    function = "eval_one_episode" if task_env.world_count == 1 else "eval_one_episode_batch"
    getattr(deploy, function)(TASK_ENV=task_env, model_client=model_client)
    task_env.synchronize()
    if not task_env.is_episode_end():
        raise RuntimeError("policy deployment returned before ClothDojo's step limit")
    return {"policy_name": policy_name, "steps": task_env.steps, "world_count": task_env.world_count}
