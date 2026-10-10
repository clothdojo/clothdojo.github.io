"""Pi0.5 inference using the pinned RLinf/OpenPI transforms."""
from __future__ import annotations
import dataclasses
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import Any
import numpy as np
from clothdojo.rendering import CAMERA_NAMES, POLICY_TRANSPORT_SIZE
from clothdojo.rollout.checkpoint import inspect_rlinf_openpi_checkpoint, rlinf_openpi_weights_sha256
from clothdojo.rollout.guided_rtc import GuidedRTCConfig, sample_guided_openpi_rlinf_actions
from clothdojo.rollout.newton import POLICY_ACTION_DIM
IMAGE_PREPROCESSING_POLICIES = ("none", "training_fixed_crop")
DELTA_ACTION_MASK = np.asarray([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)


class Pi05Policy:
    """A local, single-session policy loaded from a self-contained model bundle."""

    def __init__(self, bundle: str | Path, *, device: str = "cuda:0", num_steps: int = 10):
        import hashlib

        root = Path(bundle).expanduser().resolve()
        metadata = json.loads((root / "policy.json").read_text())
        if metadata.get("schema") != "clothdojo_pi05_policy_v1":
            raise ValueError("unsupported policy bundle schema")
        if metadata.get("task") not in {"shirt_flatten", "shirt_fold"}:
            raise ValueError("unsupported policy task")
        if metadata.get("cloth_visual") not in {"nocs", "white", "rgb"}:
            raise ValueError("unsupported policy observation type")
        if type(num_steps) is not int or num_steps < 1:
            raise ValueError("num_steps must be positive")
        self.metadata = metadata
        self.action_horizon = 50
        if "initial_states_sha256" in metadata:
            actual = hashlib.sha256((root / "initial_states.npz").read_bytes()).hexdigest()
            if actual != metadata["initial_states_sha256"]:
                raise ValueError("policy bundle checksum mismatch: initial_states.npz")
        _validate_image_preprocessing(metadata["image_preprocessing"])
        for relative, key in (
            ("norm_stats.json", "norm_stats_sha256"),
            ("cache/big_vision/paligemma_tokenizer.model", "tokenizer_sha256"),
        ):
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
            if actual != metadata[key]:
                raise ValueError(f"policy bundle checksum mismatch: {relative}")
        os.environ["CLOTHDOJO_OPENPI_CACHE"] = str(root / "cache")
        _configure_openpi_data_home()
        checkpoint = root / "weights"
        inspection = inspect_rlinf_openpi_checkpoint(checkpoint, action_horizon=self.action_horizon)
        if inspection["weights_sha256"] != metadata["weights_sha256"]:
            raise ValueError("policy weights checksum mismatch")
        self._backend = _RLinfSFTPolicy(
            checkpoint, root / "norm_stats.json", openpi_config="pi05_aloha_robotwin",
            action_horizon=self.action_horizon, num_steps=num_steps, autocast=False,
            device=device, rtc_guidance=GuidedRTCConfig(),
            expected_checkpoint_sha256=metadata["weights_sha256"],
            image_preprocessing=metadata["image_preprocessing"],
        )
        self._mode = None
        self._previous = None

    def reset(self, seed: int, *, mode: str, exec_horizon: int) -> None:
        if mode not in {"chunk", "rtc"}:
            raise ValueError("mode must be chunk or rtc")
        if not 1 <= exec_horizon <= self.action_horizon:
            raise ValueError("invalid execution horizon")
        if mode == "rtc" and exec_horizon == self.action_horizon:
            raise ValueError("RTC requires an overlapping action chunk")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self._backend.reset(seed)
        self._mode = mode
        self._exec_horizon = exec_horizon
        self._previous = None

    def infer(self, observation, prompt, *, inference_delay_steps=0):
        if self._mode is None:
            raise RuntimeError("reset the policy before inference")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if type(inference_delay_steps) is not int or not 0 <= inference_delay_steps < self._exec_horizon:
            raise ValueError("invalid inference delay")
        if self._mode == "chunk" and inference_delay_steps:
            raise ValueError("chunk execution does not accept an inference delay")
        prepared = _prepare_remote_observation(observation)
        rtc = None
        if self._mode == "rtc" and self._previous is not None:
            previous_actions, previous_state = self._previous
            rtc = {
                "previous_model_actions": previous_actions,
                "previous_state": previous_state,
                "executed_horizon": self._exec_horizon,
                "inference_delay_steps": inference_delay_steps,
            }
        actions, model_actions = self._backend.infer(prepared, prompt, rtc)
        if actions.shape != (len(prepared["observation.state"]), self.action_horizon, 14):
            raise ValueError("policy returned an unexpected action shape")
        if not np.isfinite(actions).all() or not np.isfinite(model_actions).all():
            raise ValueError("policy returned non-finite actions")
        self._previous = (model_actions.copy(), prepared["observation.state"].copy())
        return actions


def _validate_image_preprocessing(value: Any) -> None:
    if value not in IMAGE_PREPROCESSING_POLICIES:
        raise ValueError(
            f"image_preprocessing must be one of {IMAGE_PREPROCESSING_POLICIES}, got {value!r}"
        )


def _preprocess_inference_images(observation: Any, image_preprocessing: str) -> Any:
    """Replay historical FSDP BF16 image preprocessing without changing model mode."""

    _validate_image_preprocessing(image_preprocessing)
    if image_preprocessing == "none":
        return observation

    import torch
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as model_module

    if any(image.device.type != "cuda" for image in observation.images.values()):
        raise ValueError("training_fixed_crop requires CUDA to reproduce BF16 image operations")
    # FSDP cast_root_forward_inputs rounds images before augmentation. Reuse
    # the pinned pure preprocessing function, including rotate(0) and the wrist
    # range round-trip: both matter at BF16 precision. rng=None is deterministic;
    # this does not enable model training, dropout, or random augmentation.
    prepared = dataclasses.replace(
        observation,
        images={key: image.to(torch.bfloat16) for key, image in observation.images.items()},
    )
    return model_module.preprocess_observation(prepared, train=True, rng=None)


def _install_inference_image_preprocessing(model: Any, image_preprocessing: str) -> None:
    """Wrap the single observation boundary shared by SFT, CFG and guided RTC."""

    _validate_image_preprocessing(image_preprocessing)
    method = model._observation_dict_to_device
    installed = getattr(method, "_clothdojo_image_preprocessing", None)
    if installed is not None:
        if installed != image_preprocessing:
            raise ValueError("inference image preprocessing is already installed differently")
        return
    if image_preprocessing == "none":
        return
    original = method.__func__

    @wraps(original)
    def prepare(instance: Any, processed: dict[str, Any]) -> Any:
        return _preprocess_inference_images(original(instance, processed), image_preprocessing)

    prepare._clothdojo_image_preprocessing = image_preprocessing
    model._observation_dict_to_device = MethodType(prepare, model)


class _RLinfSFTPolicy:
    """RLinf's current ``openpi_rlinf`` model behind the NumPy boundary."""

    def __init__(
        self,
        checkpoint: Path,
        norm_stats: Path,
        *,
        openpi_config: str,
        action_horizon: int,
        num_steps: int,
        autocast: bool,
        device: str,
        rtc_guidance: GuidedRTCConfig,
        expected_checkpoint_sha256: str,
        image_preprocessing: str = "none",
    ) -> None:
        import torch
        from omegaconf import OmegaConf

        _configure_openpi_data_home()
        from rlinf.models import get_model

        config = OmegaConf.create(
            {
                "precision": "bf16",
                "model_path": str(checkpoint),
                "model_type": "openpi_rlinf",
                "pi05": True,
                "is_lora": False,
                "lora_rank": 32,
                "load_to_device": False,
                "num_action_chunks": action_horizon,
                "num_steps": num_steps,
                "action_dim": POLICY_ACTION_DIM,
                "add_value_head": False,
                "openpi": {
                    "task": "eval",
                    "config_name": openpi_config,
                    "model_action_dim": 32,
                    "paligemma_variant": "gemma_2b",
                    "action_expert_variant": "gemma_300m",
                    "max_token_len": 200,
                    "use_rlt": False,
                },
                "openpi_data": {
                    "adapt_to_pi": False,
                    "extra_delta_transform": True,
                    "default_prompt": "flatten the shirt",
                    "norm_stats_path": str(norm_stats),
                },
            }
        )
        model = get_model(config)
        if model is None:
            raise RuntimeError("pinned RLINF did not register openpi_rlinf")
        if rlinf_openpi_weights_sha256(checkpoint) != expected_checkpoint_sha256:
            raise RuntimeError("SFT checkpoint changed while the model was loading")
        self.device = torch.device(device)
        if autocast:
            raise ValueError("openpi_rlinf uses native BF16; autocast is not a mode")
        self.model = model.to(self.device).eval().requires_grad_(False)
        _install_inference_image_preprocessing(self.model, image_preprocessing)
        self.autocast = False
        self.rtc_guidance = rtc_guidance
        self.num_steps = int(num_steps)
        self._delta_action_scale = _load_action_quantile_scale(norm_stats)

    def _guided_infer(
        self,
        env_observation: dict[str, Any],
        previous_actions: Any,
        *,
        inference_delay_steps: int,
    ) -> tuple[Any, Any]:
        processed = self.model.input_transform(
            self.model._repack_env_obs(env_observation),
            transpose=False,
        )
        observation = self.model._observation_dict_to_device(processed)
        model_actions = sample_guided_openpi_rlinf_actions(
            self.model.model,
            observation,
            previous_actions,
            inference_delay=inference_delay_steps,
            num_steps=self.num_steps,
            config=self.rtc_guidance,
        )
        actions = self.model.output_transform(
            {"actions": model_actions, "state": observation.state}
        )["actions"]
        return actions, model_actions

    @staticmethod
    def _model_actions_from_forward(actions: Any, forward: dict[str, Any]) -> Any:
        return forward["forward_inputs"]["model_action"].reshape(
            actions.shape[0], actions.shape[1], -1
        )

    def reset(self, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    def prepare_model_observation(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
    ) -> Any:
        """Apply the exact serving input path and return the model observation."""

        env_observation = self._to_env_observation(observation, prompt)
        processed = self.model.input_transform(
            self.model._repack_env_obs(env_observation),
            transpose=False,
        )
        return self.model._observation_dict_to_device(processed)

    @staticmethod
    def _to_env_observation(
        observation: dict[str, np.ndarray],
        prompt: str,
    ) -> dict[str, Any]:
        import torch

        state = np.asarray(observation["observation.state"], dtype=np.float32)
        main = np.asarray(observation[f"observation.images.{CAMERA_NAMES[0]}"])
        wrists = np.stack(
            [
                observation[f"observation.images.{CAMERA_NAMES[1]}"],
                observation[f"observation.images.{CAMERA_NAMES[2]}"],
            ],
            axis=1,
        )
        return {
            "main_images": torch.from_numpy(np.array(main, copy=True, order="C")),
            "wrist_images": torch.from_numpy(np.array(wrists, copy=True, order="C")),
            "extra_view_images": None,
            "states": torch.from_numpy(np.array(state, copy=True, order="C")),
            "task_descriptions": [prompt] * len(state),
        }

    def infer(
        self,
        observation: dict[str, np.ndarray],
        prompt: str,
        rtc: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        state = np.asarray(observation["observation.state"], dtype=np.float32)
        env_observation = self._to_env_observation(observation, prompt)
        previous_remaining = None
        if rtc is not None:
            inference_delay_steps = rtc["inference_delay_steps"]
            if inference_delay_steps > self.rtc_guidance.guidance_horizon:
                raise ValueError(
                    "RTC inference delay exceeds the policy guidance horizon"
                )
            previous_model_actions = _rebase_delta_model_actions(
                rtc["previous_model_actions"],
                previous_state=rtc["previous_state"],
                current_state=state,
                action_scale=self._delta_action_scale,
            )
            previous_remaining = torch.from_numpy(
                previous_model_actions[:, rtc["executed_horizon"] :]
            ).to(self.device)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.autocast and self.device.type == "cuda",
            ),
        ):
            if previous_remaining is None:
                actions, forward = self.model.predict_action_batch(
                    env_observation,
                    mode="eval",
                    compute_values=False,
                )
                model_actions = self._model_actions_from_forward(actions, forward)
            else:
                actions, model_actions = self._guided_infer(
                    env_observation,
                    previous_remaining,
                    inference_delay_steps=inference_delay_steps,
                )
        return (
            _to_numpy(actions),
            _to_numpy(model_actions[..., :POLICY_ACTION_DIM]),
        )


def _prepare_remote_observation(
    observation: dict[str, np.ndarray],
    *,
    observation_horizon: int = 1,
    transport_width_height: tuple[int, int] = POLICY_TRANSPORT_SIZE,
) -> dict[str, Any]:
    from openpi_client import image_tools

    required = {"observation.state", *(f"observation.images.{name}" for name in CAMERA_NAMES)}
    if set(observation) != required:
        raise ValueError(
            "policy observation fields differ: "
            f"missing={sorted(required - set(observation))}, "
            f"extra={sorted(set(observation) - required)}"
        )
    state = np.asarray(observation["observation.state"], dtype=np.float32)
    if (
        observation_horizon < 1
        or state.ndim != (2 if observation_horizon == 1 else 3)
        or state.shape[0] < 1
        or state.shape[-1] != POLICY_ACTION_DIM
        or (observation_horizon > 1 and state.shape[1] != observation_horizon)
        or not np.isfinite(state).all()
    ):
        raise ValueError(
            "observation.state must be finite float32"
            f"[batch, {POLICY_ACTION_DIM}] or declared history; "
            f"got shape={state.shape}, finite={bool(np.isfinite(state).all())}"
        )
    leading_shape = state.shape[:-1]
    if (
        len(transport_width_height) != 2
        or any(type(value) is not int or value < 1 for value in transport_width_height)
    ):
        raise ValueError("transport_width_height must contain two positive integers")
    target_width, target_height = transport_width_height
    raw_images = {}
    for name in CAMERA_NAMES:
        key = f"observation.images.{name}"
        image = np.asarray(observation[key])
        if image.shape[:-3] != leading_shape or image.ndim != state.ndim + 2 or image.shape[-1] != 3:
            raise ValueError(f"{key} must match the state batch/history dimensions")
        raw_images[name] = image

    def prepare(name: str) -> tuple[str, np.ndarray]:
        key = f"observation.images.{name}"
        image = raw_images[name]
        flattened = image.reshape((-1, *image.shape[-3:]))
        resized = image_tools.resize_with_pad(flattened, target_height, target_width)
        converted = image_tools.convert_to_uint8(resized)
        if converted.dtype != np.uint8:
            raise TypeError(f"{key} could not be converted to uint8")
        converted = converted.reshape((*leading_shape, *converted.shape[-3:]))
        return key, np.ascontiguousarray(converted)

    with ThreadPoolExecutor(max_workers=len(CAMERA_NAMES)) as executor:
        images = dict(executor.map(prepare, CAMERA_NAMES))
    return {"observation.state": np.ascontiguousarray(state), **images}


def _load_action_quantile_scale(norm_stats: Path) -> np.ndarray:
    source = norm_stats / "norm_stats.json" if norm_stats.is_dir() else norm_stats
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        action_stats = payload["norm_stats"]["actions"]
        q01 = np.asarray(action_stats["q01"], dtype=np.float32)
        q99 = np.asarray(action_stats["q99"], dtype=np.float32)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot read action quantiles from {source}: {exc}") from exc
    if q01.shape != (POLICY_ACTION_DIM,) or q99.shape != q01.shape:
        raise ValueError("action normalization quantiles must both have shape (14,)")
    scale = (q99 - q01 + 1.0e-6) / 2.0
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("action normalization quantile scales must be finite and positive")
    return np.ascontiguousarray(scale)


def _configure_openpi_data_home() -> None:
    """Bind offline RLINF serving to the project-owned OpenPI asset cache."""

    configured = os.environ.get("CLOTHDOJO_OPENPI_CACHE")
    if not configured:
        return
    cache = Path(configured).expanduser().resolve()
    tokenizer = cache / "big_vision" / "paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"OpenPI tokenizer does not exist in cache: {tokenizer}")
    current = os.environ.get("OPENPI_DATA_HOME")
    if current is not None and Path(current).expanduser().resolve() != cache:
        raise ValueError("OPENPI_DATA_HOME differs from CLOTHDOJO_OPENPI_CACHE")
    os.environ["OPENPI_DATA_HOME"] = str(cache)


def _rebase_delta_model_actions(
    previous_model_actions: np.ndarray,
    *,
    previous_state: np.ndarray,
    current_state: np.ndarray,
    action_scale: np.ndarray,
) -> np.ndarray:
    """Keep old absolute joint targets fixed when the delta-action base moves."""

    actions = np.asarray(previous_model_actions, dtype=np.float32)
    previous = np.asarray(previous_state, dtype=np.float32)
    current = np.asarray(current_state, dtype=np.float32)
    scale = np.asarray(action_scale, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[0] < 1 or actions.shape[2] != POLICY_ACTION_DIM:
        raise ValueError("previous model actions must have shape (batch, horizon, 14)")
    expected_state = (actions.shape[0], POLICY_ACTION_DIM)
    if previous.shape != expected_state or current.shape != previous.shape:
        raise ValueError("previous and current policy states must have shape (batch, 14)")
    if scale.shape != (POLICY_ACTION_DIM,) or np.any(scale <= 0.0):
        raise ValueError("action scale must be positive with shape (14,)")
    shift = np.zeros_like(previous)
    shift[:, DELTA_ACTION_MASK] = (
        previous[:, DELTA_ACTION_MASK] - current[:, DELTA_ACTION_MASK]
    ) / scale[DELTA_ACTION_MASK]
    result = actions.copy()
    result[..., :POLICY_ACTION_DIM] += shift[:, None, :]
    return np.ascontiguousarray(result)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))
