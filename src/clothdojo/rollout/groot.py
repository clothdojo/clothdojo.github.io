"""GR00T N1.7 bridge for ClothDojo's synchronous rollout policy boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.rendering import CAMERA_NAMES, POLICY_RENDER_SIZE
from clothdojo.rollout.newton import POLICY_ACTION_DIM
from clothdojo.runtime_identity import runtime_source_sha256

_JOINT_SLICES = {
    "left_arm": slice(0, 6),
    "left_gripper": slice(6, 7),
    "right_arm": slice(7, 13),
    "right_gripper": slice(13, 14),
}


class GrootZmqClient:
    """Adapt NVIDIA's ZMQ service to the interface used by ``RemotePolicy``."""

    def __init__(self, host: str, *, timeout_ms: int = 600_000) -> None:
        import zmq

        if not host.startswith("tcp://"):
            raise ValueError("GR00T endpoint must start with tcp://")
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(host)

    def _call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self._socket.send(_packb(request))
            response = _unpackb(self._socket.recv())
        except self._zmq.error.Again as exc:
            raise TimeoutError(f"GR00T service timed out at {endpoint}") from exc
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T service error: {response['error']}")
        return response

    def get_server_metadata(self) -> dict[str, Any]:
        return self._call("get_server_metadata")

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("clothdojo_infer", request)

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


class _GrootBridgePolicy:
    def __init__(self, policy: Any, identity: dict[str, Any]) -> None:
        self.policy = policy
        self.identity = identity
        self._rng: dict[str, tuple[Any, Any]] = {}

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        del options
        return {}

    def get_modality_config(self) -> dict[str, Any]:
        return self.policy.get_modality_config()

    def get_action(self, observation: dict[str, Any], options=None):
        return self.policy.get_action(observation, options)

    def infer(self, **request: Any) -> dict[str, np.ndarray]:
        import torch

        session = request.pop("clothdojo.session_id", None)
        seed = request.pop("clothdojo.seed", None)
        prompt = request.pop("prompt", None)
        if request.keys() != {
            "observation.state",
            *(f"observation.images.{name}" for name in CAMERA_NAMES),
        }:
            raise ValueError("GR00T request observation fields differ")
        if not isinstance(session, str) or not session:
            raise ValueError("GR00T request is missing a session id")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("GR00T seed must be a non-negative integer")
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        elif session in self._rng:
            cpu_state, cuda_state = self._rng[session]
            torch.random.set_rng_state(cpu_state)
            torch.cuda.set_rng_state(cuda_state)
        else:
            raise ValueError("a new GR00T session must provide its seed")

        state = np.asarray(request["observation.state"], dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != POLICY_ACTION_DIM:
            raise ValueError("GR00T state must have shape [batch,14]")
        batch = int(state.shape[0])
        observation = {
            "video": {
                name: np.asarray(
                    request[f"observation.images.{name}"], dtype=np.uint8
                )[:, None]
                for name in CAMERA_NAMES
            },
            "state": {
                name: np.ascontiguousarray(state[:, None, part])
                for name, part in _JOINT_SLICES.items()
            },
            "language": {
                "annotation.human.task_description": [[str(prompt)]] * batch,
            },
        }
        actions, _ = self.policy.get_action(observation)
        result = np.concatenate(
            [np.asarray(actions[name], dtype=np.float32) for name in _JOINT_SLICES],
            axis=-1,
        )
        if result.shape != (batch, 40, POLICY_ACTION_DIM):
            raise ValueError(f"GR00T returned an unexpected action shape: {result.shape}")
        if not np.isfinite(result).all():
            raise FloatingPointError("GR00T returned non-finite actions")
        self._rng[session] = (
            torch.random.get_rng_state(),
            torch.cuda.get_rng_state(),
        )
        return {"actions": np.ascontiguousarray(result)}


def serve_groot_policy(
    checkpoint: str | Path,
    norm_stats: str | Path,
    *,
    groot_root: str | Path,
    checkpoint_sha256: str,
    embodiment_tag: str = "new_embodiment",
    device: str = "cuda:0",
    host: str = "127.0.0.1",
    port: int = 5555,
) -> None:
    """Serve a frozen GR00T checkpoint through its official processor and ZMQ server."""

    import os
    import sys

    root = Path(groot_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # The frozen checkpoint records the official relative model identifier
    # ``nvidia/Cosmos-Reason2-2B``.  Its verified offline symlink lives below
    # this runtime root, matching NVIDIA's training and deployment layout.
    os.chdir(root)
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    checkpoint = Path(checkpoint).expanduser().resolve()
    norm_stats = Path(norm_stats).expanduser().resolve()
    identity = {
        "backend": "gr00t-n1.7",
        "protocol_version": 2,
        "checkpoint_sha256": _require_sha256(checkpoint_sha256),
        "norm_stats_sha256": _file_sha256(norm_stats),
        "action_dim": POLICY_ACTION_DIM,
        "action_horizon": 40,
        "camera_names": list(CAMERA_NAMES),
        # The checkpoint-owned processor was trained from native 4:3 frames and
        # performs its own resize and center crop.  OpenPI's square transport
        # would letterbox the image before that processor and change its visual
        # token layout.
        "transport_image_size": list(POLICY_RENDER_SIZE),
        "image_preprocessing": "none",
        "batched_inference": True,
        "embodiment_tag": embodiment_tag,
        "clothdojo_runtime_sha256": runtime_source_sha256(
            Path(__file__).resolve().parents[1]
        ),
    }
    bridge = _GrootBridgePolicy(
        Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=str(checkpoint),
            device=device,
            strict=True,
        ),
        identity,
    )
    _GrootPolicyServer(bridge, identity, host=host, port=port).run()


class _GrootPolicyServer:
    """Small synchronous REP server; GR00T inference itself stays NVIDIA-native."""

    def __init__(
        self,
        policy: _GrootBridgePolicy,
        identity: dict[str, Any],
        *,
        host: str,
        port: int,
    ) -> None:
        import zmq

        self.policy = policy
        self.identity = identity
        self.endpoint = f"tcp://{host}:{port}"
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self.endpoint)

    def run(self) -> None:
        print(f"GR00T service ready at {self.endpoint}", flush=True)
        try:
            while True:
                request = _unpackb(self._socket.recv())
                try:
                    endpoint = request.get("endpoint")
                    if endpoint == "ping":
                        result: Any = {"status": "ok"}
                    elif endpoint == "get_server_metadata":
                        result = {"clothdojo": self.identity}
                    elif endpoint == "clothdojo_infer":
                        result = self.policy.infer(**request.get("data", {}))
                    else:
                        raise ValueError(f"unknown GR00T endpoint: {endpoint!r}")
                except Exception as error:
                    import traceback

                    traceback.print_exc()
                    result = {"error": f"{type(error).__name__}: {error}"}
                self._socket.send(_packb(result))
        finally:
            self._socket.close(linger=0)
            self._context.term()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_sha256(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest") from exc
    if value != value.lower():
        raise ValueError("checkpoint_sha256 must be lowercase")
    return value


def _packb(value: Any) -> bytes:
    import msgpack

    return msgpack.packb(value, default=_encode_numpy, use_bin_type=True)


def _unpackb(value: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(value, object_hook=_decode_numpy, raw=False)


def _encode_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V"}:
            raise TypeError(f"cannot encode ndarray dtype {value.dtype}")
        return {
            b"nd": True,
            b"type": value.dtype.str,
            b"kind": b"",
            b"shape": value.shape,
            b"data": value.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__}")


def _decode_numpy(value: dict[Any, Any]) -> Any:
    nd = value.get(b"nd", value.get("nd"))
    if nd is not True:
        return value
    kind = value.get(b"kind", value.get("kind"))
    if kind in {b"O", "O"}:
        raise ValueError("refusing object-dtype ndarray payload")
    dtype = value.get(b"type", value.get("type"))
    shape = value.get(b"shape", value.get("shape"))
    data = value.get(b"data", value.get("data"))
    return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=tuple(shape))
