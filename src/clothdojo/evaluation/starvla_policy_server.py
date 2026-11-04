"""MR0 StarVLA policy server for ClothDojo flatten rollouts.

Loads the final QwenMMDiT checkpoint of an MR0 StarVLA run inside the starVLA
training environment (py3.11 on an H100 node), converts the live canonical
14-D joint state into the 20-D absolute action-frame EEF-6D state with the
same FK chain that produced the training labels, and serves per-request
predict_action chunks over a length-prefixed pickle protocol on TCP. The
rollout worker bridges the returned 20-D absolute EEF targets to joint
commands through its own IK; this server owns only model-facing concerns
(state FK, min-max normalization, action unnormalization).

Several rollout workers may hold persistent connections at the same time.
Each connection owns an isolated RNG state that is restored before every
inference and captured afterwards, so interleaved requests from parallel
workers never cross-contaminate their seeded noise streams. Model forward
passes serialize on one inference lock per server process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

from clothdojo.rendering import POLICY_RENDER_SIZE
from clothdojo.runtime_identity import runtime_source_sha256

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
JOINT_DIM = 14
ACTION_DIM = 20

# Fixed origins of tools/piper_kinematics/piper_arm_kinematics.urdf,
# joint1..joint6. The action-frame end effector is gripper_base, which the
# URDF fixes to the link6 frame, so the chain below is the exact FK that
# produced the observation.state.eef.6d / action.eef.6d training columns.
_ORIGINS = (
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, -0.1359, -3.1416)),
    ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.7939)),
    ((-0.021984, -0.25075, 0.0), (1.5708, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),
    ((8.8259e-05, -0.091, 0.0), (1.5708, 0.0, 0.0)),
)


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


def _origin_transform(xyz, rpy):
    matrix = np.eye(4)
    matrix[:3, :3] = _fixed_axis_rpy(*rpy)
    matrix[:3, 3] = xyz
    return matrix


def _z_rotation(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[:3, :3] = ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    return matrix


def action_frame_fk(q6):
    """Return the 4x4 action-frame EE pose in the arm base frame."""
    matrix = np.eye(4)
    for angle, (xyz, rpy) in zip(q6, _ORIGINS, strict=True):
        matrix = matrix @ _origin_transform(xyz, rpy) @ _z_rotation(angle)
    return matrix


def joints_to_eef6d_state(joints):
    """Canonical 14-D state to the 20-D absolute action-frame EEF state."""
    values = np.asarray(joints, dtype=np.float64).reshape(-1)
    if values.shape != (JOINT_DIM,) or not np.isfinite(values).all():
        raise ValueError(f"expected finite canonical joints shaped (14,), got {values.shape}")
    left = action_frame_fk(values[:6])
    right = action_frame_fk(values[7:13])
    return np.concatenate(
        (
            left[:3, 3], left[:3, 0], left[:3, 1], values[6:7],
            right[:3, 3], right[:3, 0], right[:3, 1], values[13:14],
        )
    )


class MinMaxNormalization:
    """Per-dimension min-max scaling identical to the training transform."""

    def __init__(self, run_dir: Path, key: str = "new_embodiment") -> None:
        stats = json.loads((run_dir / "dataset_statistics.json").read_text())
        if key not in stats:
            raise KeyError(f"normalization key missing from stats: {sorted(stats)}")
        blocks = {}
        for name in ("state", "action"):
            block = stats[key][name]
            low = np.asarray(block["min"], dtype=np.float64)
            high = np.asarray(block["max"], dtype=np.float64)
            if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
                raise ValueError(f"{name} statistics must be 20-D, got {low.shape}")
            blocks[name] = (low, high)
        self._state = blocks["state"]
        self._action = blocks["action"]

    def normalize_state(self, values):
        low, high = self._state
        span = high - low
        safe = np.where(span > 1e-12, span, 1.0)
        normalized = 2.0 * (values - low) / safe - 1.0
        return np.where(span > 1e-12, normalized, 0.0)

    def unnormalize_actions(self, values):
        low, high = self._action
        clipped = np.clip(values, -1.0, 1.0)
        return 0.5 * (clipped + 1.0) * (high - low) + low


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy(checkpoint: Path, starvla_root: Path, device: str):
    workspace = starvla_root.resolve()
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from starVLA.model.framework.base_framework import baseframework

    config = checkpoint.parents[1] / "config.full.yaml"
    if not config.is_file():
        config = checkpoint.parents[1] / "config.yaml"
    model = baseframework.from_pretrained(str(checkpoint))
    model = model.to(device).eval()
    normalization = MinMaxNormalization(checkpoint.parents[1])
    identity = {
        "backend": "mr0-starvla",
        "policy_class": type(model).__name__,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "action_space": "eef_abs_20d_action_frame",
        "action_dim": ACTION_DIM,
        "action_horizon": int(model.action_model.action_horizon),
        "state_input": "canonical_joints_14d",
        "normalization": "min_max",
        "camera_names": list(CAMERA_KEYS),
        "transport_image_size": list(POLICY_RENDER_SIZE),
        "batched_inference": True,
        "clothdojo_runtime_sha256": runtime_source_sha256(
            Path(__file__).resolve().parents[1]
        ),
        "device": device,
        "bridge": "worker_ik_position_commands_v1",
    }
    return model, normalization, identity


def _run_inference(model, normalization, action_horizon, request):
    task = str(request["task"])
    states = np.asarray(request["observation.state"], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != JOINT_DIM:
        raise ValueError(f"observation.state must be [batch, 14], got {states.shape}")
    batch = int(states.shape[0])
    images = {}
    for key in CAMERA_KEYS:
        value = np.asarray(request[key])
        if (
            value.dtype != np.uint8
            or value.ndim != 4
            or value.shape[-1] != 3
            or value.shape[0] != batch
        ):
            raise ValueError(f"expected uint8 [batch, H, W, 3] images for {key}, got {value.shape}")
        images[key] = value
    examples = []
    for index in range(batch):
        state20 = joints_to_eef6d_state(states[index])
        examples.append(
            {
                "image": [images[key][index] for key in CAMERA_KEYS],
                "lang": task,
                "state": normalization.normalize_state(state20).astype(np.float32)[None, :],
            }
        )
    output = model.predict_action(examples=examples)
    normalized = np.asarray(output["normalized_actions"], dtype=np.float64)
    if (
        normalized.shape != (batch, action_horizon, ACTION_DIM)
        or not np.isfinite(normalized).all()
    ):
        raise ValueError(f"unexpected normalized actions: {normalized.shape}")
    return normalization.unnormalize_actions(normalized)


def _recv_exact(conn: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = conn.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("client closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send(conn: socket.socket, payload: dict) -> None:
    data = pickle.dumps(payload, protocol=5)
    conn.sendall(struct.pack("<Q", len(data)) + data)


def _capture_rng_state():
    import torch

    cpu = torch.random.get_rng_state().clone()
    cuda = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    return cpu, cuda


def _restore_rng_state(state) -> None:
    import torch

    cpu, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda:
        torch.cuda.set_rng_state_all(list(cuda))


def _serve_connection(
    conn: socket.socket,
    address,
    model,
    normalization,
    action_horizon: int,
    identity: dict,
    infer_lock: threading.Lock,
) -> None:
    import torch

    rng_state = None
    with conn:
        while True:
            try:
                (length,) = struct.unpack("<Q", _recv_exact(conn, 8))
                request = pickle.loads(_recv_exact(conn, length))
            except (ConnectionError, OSError, pickle.UnpicklingError):
                break
            try:
                if "reset" in request:
                    seed = int(request["reset"])
                    if seed < 0:
                        raise ValueError("reset seed must be non-negative")
                    with infer_lock:
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        rng_state = _capture_rng_state()
                    response = {"ok": True}
                elif request.get("identity"):
                    response = {"identity": identity}
                else:
                    started = time.perf_counter()
                    with infer_lock:
                        if rng_state is not None:
                            _restore_rng_state(rng_state)
                        actions = _run_inference(
                            model, normalization, action_horizon, request
                        )
                        rng_state = _capture_rng_state()
                    response = {
                        "actions": actions.tolist(),
                        "infer_seconds": time.perf_counter() - started,
                    }
            except BaseException as error:  # reported back to the worker
                import traceback

                traceback.print_exc()
                response = {"error": f"{type(error).__name__}: {error}"}
            _send(conn, response)


def serve_starvla_policy(
    checkpoint: str | Path,
    *,
    starvla_root: str | Path,
    device: str,
    host: str,
    port: int,
) -> None:
    """Load one MR0 StarVLA checkpoint and serve it on a pickle-TCP socket."""

    resolved = Path(checkpoint).expanduser().resolve()
    model, normalization, identity = _load_policy(
        resolved, Path(starvla_root).expanduser(), device
    )
    action_horizon = int(identity["action_horizon"])
    print(json.dumps({"phase": "server_ready", "identity": identity}), flush=True)

    infer_lock = threading.Lock()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen(64)
    print(json.dumps({"phase": "listening", "host": host, "port": port}), flush=True)
    while True:
        conn, address = server.accept()
        print(json.dumps({"phase": "client_connected", "address": address}), flush=True)
        threading.Thread(
            target=_serve_connection,
            args=(
                conn,
                address,
                model,
                normalization,
                action_horizon,
                identity,
                infer_lock,
            ),
            daemon=True,
        ).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--starvla-root",
        required=True,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8741)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    serve_starvla_policy(
        args.checkpoint,
        starvla_root=args.starvla_root,
        device=args.device,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
