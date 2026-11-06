"""ClothDojo's environment adapter for the pinned official XPolicyLab protocol.

Transport, framing, RGB decoding and request replay are owned by XPolicyLab.
Actions are absolute Piper joint targets and gripper-open commands, never deltas.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import random
import re
import sys
import uuid

import numpy as np

from clothdojo.rendering import CAMERA_NAMES, POLICY_RENDER_SIZE, POLICY_TRANSPORT_SIZE

ROBOT_DIMS = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
XPL_REVISION = "6baccd57ff958570d0db1588370289ee36c8bb70"


def artifact_sha256(path):
    from clothdojo.rollout.policy import _tree_sha256

    path = Path(path).expanduser().resolve()
    if path.is_dir():
        return _tree_sha256(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observation_batch(request):
    """Map canonical RGB arrays to XPolicyLab observations without channel swaps."""
    from XPolicyLab.utils.process_data import unpack_robot_state

    states = np.asarray(request["observation.state"])
    history = states.ndim == 3
    if not history:
        states = states[:, None]
    result = []
    for index, state in enumerate(states):
        vision = {}
        for name in CAMERA_NAMES:
            frames = np.asarray(request[f"observation.images.{name}"])[index]
            if not history:
                frames = frames[None]
            vision[name] = {"color": frames[-1], "history": frames}
        result.append(
            {
                "env_idx": index,
                "instruction": request["prompt"],
                "state": unpack_robot_state(state[-1], "joint", ROBOT_DIMS, source_type="obs"),
                "state_history": state,
                "vision": vision,
            }
        )
    return result


def action_array(result, batch_size, horizon):
    from XPolicyLab.utils.process_data import pack_robot_state

    if not isinstance(result, list) or len(result) != batch_size:
        raise ValueError("XPolicyLab must return one action chunk per environment")
    actions = np.asarray(
        [
            [
                pack_robot_state({"state": step}, "joint", ROBOT_DIMS, source_type="obs")
                for step in chunk
            ]
            for chunk in result
        ],
        dtype=np.float32,
    )
    if actions.shape != (batch_size, horizon, 14) or not np.isfinite(actions).all():
        raise ValueError(
            f"expected finite actions [{batch_size},{horizon},14], got {actions.shape}"
        )
    return actions


class ArrayModel:
    """XPolicyLab Model interface for adapters returning physical [B,H,14] arrays.

    Subclass load/reset/predict in your own module; the evaluator never imports models.
    """

    action_horizon = 8
    observation_horizon = 1
    transport_image_size = POLICY_RENDER_SIZE

    def __init__(self, config):
        self.config = config
        self.seed = 42
        self.observations = None

    def prepare_case(self, case_meta=None):
        self.seed = int((case_meta or {}).get("seed", 42))

    def reset(self):
        self.observations = None
        random.seed(self.seed)
        # Keep the full benchmark seed for Python/Torch; NumPy's legacy API is uint32.
        np.random.seed(self.seed % (2**32))
        torch = sys.modules.get("torch")
        if torch is not None:
            torch.manual_seed(self.seed)

    def update_obs(self, observation):
        self.update_obs_batch([observation])

    def update_obs_batch(self, observations):
        self.observations = observations

    def get_action(self):
        return self.get_action_batch()[0]

    def get_action_batch(self, env_idx_list=None):
        from XPolicyLab.utils.process_data import unpack_robot_state

        if self.observations is None:
            raise RuntimeError("update_obs_batch must precede get_action_batch")
        indices = [item["env_idx"] for item in self.observations]
        if env_idx_list is not None and env_idx_list != indices:
            raise ValueError("environment indices differ from the observation batch")
        actions = np.asarray(self.predict(self.observations), dtype=np.float32)
        if (
            actions.shape != (len(indices), self.action_horizon, 14)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("predict must return finite absolute [B,H,14] actions")
        return [
            unpack_robot_state(chunk, "joint", ROBOT_DIMS, source_type="obs") for chunk in actions
        ]

    def predict(self, observations):
        raise NotImplementedError


def canonical_observations(observations):
    """Preserve adjacent control-frame history required by our trained DP."""
    return {
        "observation.state": np.stack([obs["state_history"] for obs in observations]),
        **{
            f"observation.images.{name}": np.stack(
                [obs["vision"][name]["history"] for obs in observations]
            )
            for name in CAMERA_NAMES
        },
    }


class DiffusionModel(ArrayModel):
    observation_horizon = 2
    transport_image_size = POLICY_TRANSPORT_SIZE

    def __init__(self, config):
        super().__init__(config)
        from clothdojo.rollout.diffusion import DiffusionPolicy

        self.policy = DiffusionPolicy(config["checkpoint"], device=config["device"])
        self.action_horizon = int(self.policy.cfg["n_action_steps"])

    def reset(self):
        super().reset()
        self.policy.reset(self.seed, mode="chunk", exec_horizon=self.action_horizon)

    def predict(self, observations):
        return self.policy.infer(canonical_observations(observations), "")


def load_model(name, config):
    validate_model_name(name)
    if name == "diffusion":
        return DiffusionModel(config)
    module, separator, symbol = name.partition(":")
    if not separator or not module or not symbol:
        raise ValueError("model must be 'diffusion' or an importable module:Model")
    return getattr(importlib.import_module(module), symbol)(config)


def validate_model_name(name):
    if name != "diffusion" and not re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", name
    ):
        raise ValueError("model must be 'diffusion' or an importable module:Model")


class HostedModel:
    """Expose the standard model plus artifact identity through official RPCs."""

    def __init__(self, model, checkpoint):
        from clothdojo.rollout.policy import POLICY_PROTOCOL_VERSION
        from clothdojo.runtime_identity import runtime_source_sha256

        self.model = model
        for field in ("action_horizon", "observation_horizon"):
            value = getattr(model, field, None)
            if type(value) is not int or value < 1:
                raise ValueError(f"model must declare a positive integer {field}")
        path = Path(inspect.getfile(type(model)))
        digest = artifact_sha256(checkpoint)
        self.identity = {
            "backend": "xpolicylab",
            "protocol_version": POLICY_PROTOCOL_VERSION,
            "checkpoint_sha256": digest,
            "norm_stats_sha256": digest,
            "adapter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "clothdojo_runtime_sha256": runtime_source_sha256(),
            "xpolicylab_revision": XPL_REVISION,
            "action_dim": 14,
            "action_horizon": model.action_horizon,
            "observation_horizon": model.observation_horizon,
            "batched_inference": True,
            "camera_names": list(CAMERA_NAMES),
            "transport_image_size": list(
                getattr(model, "transport_image_size", POLICY_RENDER_SIZE)
            ),
            "image_preprocessing": "none",
        }

    def get_metadata(self):
        return {"clothdojo": self.identity}

    def __getattr__(self, name):
        return getattr(self.model, name)


class XPolicyClient:
    def __init__(self, endpoint):
        from client_server.ws.model_client import WsModelClient

        uid = uuid.uuid4().hex
        self.client = WsModelClient(
            url=endpoint.replace("xpolicy://", "ws://", 1),
            evaluation_id=uid,
            trial_id=uid,
            action_case_id=uid,
            connect_timeout_s=5,
            request_timeout_s=180,
            max_connect_attempts=1,
        )
        self.metadata = self.client.call("get_metadata")

    def get_server_metadata(self):
        return self.metadata

    def infer(self, request):
        if "clothdojo.rtc" in request:
            raise ValueError("XPolicyLab adapters use synchronous chunk execution")
        if "clothdojo.seed" in request:
            self.client.call("prepare_case", {"seed": int(request["clothdojo.seed"])})
            self.client.call("reset")
        observations = observation_batch(request)
        self.client.call("update_obs_batch", observations)
        result = self.client.call("get_action_batch", list(range(len(observations))))
        return {
            "actions": action_array(
                result, len(observations), self.metadata["clothdojo"]["action_horizon"]
            )
        }

    def close(self):
        self.client.close()

    def __del__(self):
        if hasattr(self, "client"):
            self.close()


def serve(checkpoint, *, model="diffusion", device="cuda:0", host="127.0.0.1", port=8080):
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    checkpoint = str(Path(checkpoint).expanduser().resolve())
    adapter = load_model(model, {"checkpoint": checkpoint, "device": device})
    hosted = HostedModel(adapter, checkpoint)
    print(json.dumps(hosted.get_metadata()), flush=True)
    asyncio.run(PolicyServer(hosted, PolicyServerConfig(host=host, port=port)).serve_forever())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="diffusion")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    serve(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
