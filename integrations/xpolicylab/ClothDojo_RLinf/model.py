"""Serve an unchanged ClothDojo RLinf Pi0.5 bundle through XPolicyLab."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate

_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
_STATE_PARTS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)


def _read_observation(obs: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], str]:
    if not isinstance(obs, Mapping):
        raise ValueError("XPolicyLab observation must be a mapping")
    state_fields = obs.get("state")
    if not isinstance(state_fields, Mapping):
        raise ValueError("XPolicyLab observation has no Piper state")
    state_parts = []
    for key, size in _STATE_PARTS:
        part = np.asarray(state_fields.get(key), dtype=np.float32)
        if part.shape != (size,) or not np.isfinite(part).all():
            raise ValueError(f"invalid Piper state field: {key}")
        state_parts.append(part)
    observation = {"observation.state": np.concatenate(state_parts)[None]}
    vision = obs.get("vision")
    if not isinstance(vision, Mapping):
        raise ValueError("XPolicyLab observation has no camera views")
    for name in _CAMERAS:
        camera = vision.get(name)
        if not isinstance(camera, Mapping):
            raise ValueError(f"missing camera: {name}")
        image = np.asarray(camera.get("color"))
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"invalid RGB camera: {name}")
        observation[f"observation.images.{name}"] = image[None]
    prompt = obs.get("instruction")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("XPolicyLab observation has no instruction")
    return observation, prompt


def _joint_action(row: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(row, dtype=np.float32)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError("RLinf action must be a finite 14-vector")
    return {
        "left_arm_joint_state": action[:6].copy(),
        "left_ee_joint_state": action[6:7].copy(),
        "right_arm_joint_state": action[7:13].copy(),
        "right_ee_joint_state": action[13:14].copy(),
    }


class Model(ModelTemplate):
    """XPolicyLab model API backed by the existing, hash-checked RLinf loader."""

    def __init__(self, model_cfg: dict[str, Any]):
        if model_cfg.get("env_cfg_type") != "piper" or model_cfg.get("action_type") != "joint":
            raise ValueError("ClothDojo_RLinf requires the Piper joint action profile")
        from clothdojo.rollout.policy import Pi05Policy

        self.model = Pi05Policy(
            model_cfg["bundle"],
            device=str(model_cfg.get("device", "cuda:0")),
            num_steps=int(model_cfg.get("num_steps", 10)),
        )
        self.seed = int(model_cfg.get("seed", 42))
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.observation: dict[str, np.ndarray] | None = None
        self.prompt: str | None = None

    def reset(self):
        self.model.reset(self.seed, mode="chunk", exec_horizon=10)
        self.observation = None
        self.prompt = None

    def update_obs(self, obs):
        self.observation, self.prompt = _read_observation(obs)

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("ClothDojo_RLinf supports one world per policy server")
        self.update_obs(obs_list[0])

    def get_action(self):
        if self.observation is None or self.prompt is None:
            raise RuntimeError("update_obs must be called before get_action")
        actions = np.asarray(self.model.infer(self.observation, self.prompt))
        if actions.shape != (1, 50, 14):
            raise ValueError("RLinf Pi0.5 must return one 50-by-14 action chunk")
        return [_joint_action(row) for row in actions[0]]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and list(env_idx_list) != [0]:
            raise ValueError("ClothDojo_RLinf supports one world per policy server")
        return [self.get_action()]
