"""Local EMA Diffusion Policy and adjacent-frame history for the existing rollout loop."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from clothdojo.training.diffusion import make_policy
from clothdojo.training.diffusion_data import IMAGE_KEYS, resize_images


class DiffusionPolicy:
    def __init__(self, checkpoint: str | Path, *, device="cuda:0"):
        from diffusers.training_utils import EMAModel

        path = Path(checkpoint)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("schema") != "clothdojo_diffusion_v1":
            raise ValueError("not a ClothDojo Diffusion Policy checkpoint")
        self.cfg = saved["config"]
        self.device = torch.device(device)
        self.model = make_policy(self.cfg)
        self.model.load_state_dict(saved["model"], strict=True)
        ema = EMAModel(self.model.parameters())
        ema.load_state_dict(saved["ema"])
        ema.copy_to(self.model.parameters())
        self.model.to(self.device).eval()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.identity = {
            "backend": "lerobot_diffusion",
            "checkpoint_sha256": digest,
            "step": saved["step"],
            "weights": "ema",
            "task_conditioning": "none",
            "observation_horizon": 2,
            "prediction_horizon": self.cfg["horizon"],
            "action_horizon": self.cfg["n_action_steps"],
            "inference_steps": self.cfg["num_inference_steps"],
        }
        self.generator = None

    def reset(self, seed, *, mode, exec_horizon):
        if mode != "chunk" or exec_horizon != self.cfg["n_action_steps"]:
            raise ValueError("Diffusion Policy uses its configured receding execution horizon")
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.inference_mode()
    def infer(self, observation, prompt, *, inference_delay_steps=0):
        del prompt  # Deliberately no language or externally supplied phase conditioning.
        if inference_delay_steps or self.generator is None:
            raise ValueError("reset a synchronous diffusion session before inference")
        state = torch.as_tensor(observation["observation.state"], dtype=torch.float32)
        if state.ndim != 3 or state.shape[1:] != (2, 14):
            raise ValueError("Diffusion Policy requires adjacent [B,2,14] observations")
        batch = {"observation.state": state.to(self.device)}
        for key in IMAGE_KEYS:
            images = np.asarray(observation[key])
            if images.dtype != np.uint8 or images.shape[:2] != tuple(state.shape[:2]):
                raise ValueError("expected two uint8 image observations per world")
            images = torch.from_numpy(images.copy()).flatten(0, 1).permute(0, 3, 1, 2).float() / 255
            batch[key] = (
                resize_images(images, self.cfg["image_size"])
                .unflatten(0, state.shape[:2])
                .to(self.device)
            )
        batch = self.model.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[key] for key in IMAGE_KEYS], dim=2)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            conditioning = self.model.diffusion._prepare_global_conditioning(batch)
            actions = self.model.diffusion.conditional_sample(
                state.shape[0],
                global_cond=conditioning,
                generator=self.generator,
            )
        actions = actions[:, 1 : 1 + self.cfg["n_action_steps"]]
        actions = self.model.unnormalize_outputs({"action": actions})["action"]
        if not torch.isfinite(actions).all():
            raise FloatingPointError("nonfinite diffusion actions")
        return actions.float().cpu().numpy()


class AdjacentObservationEpisode:
    """Capture t-1 immediately before the final control step of each execution chunk.

    Caching the previous *inference* observation would incorrectly use t-8 instead.
    Only the two observations needed by the next inference are rendered.
    """

    def __init__(self, episode, exec_horizon):
        self.episode = episode
        self.exec_horizon = exec_horizon
        self.executed = 0
        self.previous = None

    @property
    def world_count(self):
        return self.episode.world_count

    @property
    def action_dim(self):
        return self.episode.action_dim

    def synchronize(self):
        self.episode.synchronize()

    def observe(self):
        current = self.episode.observe()
        previous = current if self.previous is None else self.previous
        return {
            key: np.stack([previous[key], current[key]], axis=1)
            for key in ("observation.state", *IMAGE_KEYS)
        }

    def step(self, actions):
        if (self.executed + 1) % self.exec_horizon == 0:
            self.previous = {key: value.copy() for key, value in self.episode.observe().items()}
        self.episode.step(actions)
        self.executed += 1


class _DiffusionServerPolicy:
    def __init__(self, policy: DiffusionPolicy) -> None:
        self.policy = policy

    def reset(self, seed: int) -> None:
        self.policy.reset(
            seed,
            mode="chunk",
            exec_horizon=int(self.policy.cfg["n_action_steps"]),
        )

    def capture_session_state(self):
        if self.policy.generator is None:
            raise RuntimeError("Diffusion Policy session has no generator")
        return self.policy.generator.get_state().clone()

    def restore_session_state(self, state) -> None:
        if self.policy.generator is None:
            self.policy.generator = torch.Generator(device=self.policy.device)
        self.policy.generator.set_state(state)

    def infer(self, observation, prompt, rtc):
        if rtc is not None:
            raise ValueError("Diffusion Policy does not support RTC")
        return self.policy.infer(observation, prompt), None


def serve_diffusion_policy(
    checkpoint: str | Path,
    norm_stats: str | Path,
    *,
    image_preprocessing: str = "none",
    device: str = "cuda:0",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> dict:
    """Serve the trained EMA policy without changing its embedded normalization."""

    import json

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from clothdojo.rendering import CAMERA_NAMES, POLICY_TRANSPORT_SIZE
    from clothdojo.rollout.newton import POLICY_ACTION_DIM
    from clothdojo.rollout.policy import POLICY_PROTOCOL_VERSION, _PolicySessionRouter
    from clothdojo.runtime_identity import runtime_source_sha256

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if Path(norm_stats).expanduser().resolve() != checkpoint_path:
        raise ValueError("Diffusion Policy normalization is embedded in its checkpoint")
    if image_preprocessing != "none":
        raise ValueError("Diffusion Policy evaluation uses its trained resize preprocessing")
    policy = DiffusionPolicy(checkpoint_path, device=device)
    identity = {
        **policy.identity,
        "protocol_version": POLICY_PROTOCOL_VERSION,
        "clothdojo_runtime_sha256": runtime_source_sha256(),
        "norm_stats_sha256": policy.identity["checkpoint_sha256"],
        "action_dim": POLICY_ACTION_DIM,
        "camera_names": list(CAMERA_NAMES),
        "transport_image_size": list(POLICY_TRANSPORT_SIZE),
        "batched_inference": True,
        "image_preprocessing": "none",
    }
    print(json.dumps({"clothdojo": identity}, sort_keys=True), flush=True)
    WebsocketPolicyServer(
        _PolicySessionRouter(_DiffusionServerPolicy(policy)),
        host=host,
        port=port,
        metadata={"clothdojo": identity},
    ).serve_forever()
    return identity
