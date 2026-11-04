"""LingBot-VA policy server for the debug rollout (run under the py3.12 runtime).

Loads the LeRobot-fork LingBot-VA checkpoint plus its frozen Wan VAE and the
precomputed UMT5 text embedding, then serves per-step ``select_action`` requests
from the py3.11 rollout worker over a length-prefixed pickle protocol on TCP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import socket
import struct
import sys
from pathlib import Path

import numpy as np

from collections import deque

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _load_policy(args, device: str):
    import torch

    sys.path.insert(0, str(Path(args.lerobot_src).resolve() / "src"))
    from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
    from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    checkpoint = Path(args.checkpoint).resolve()
    config = LingBotVAConfig.from_pretrained(str(checkpoint))
    config.device = "cpu"  # load fp32 weights on CPU, then move the bf16 transformer
    config.wan_pretrained_path = str(Path(args.wan_root).resolve())
    config.precomputed_text_embeddings_path = str(Path(args.text_embeddings).resolve())
    if args.attn_mode:
        config.attn_mode = args.attn_mode
    policy = LingBotVAPolicy.from_pretrained(str(checkpoint), config=config)
    policy.transformer.to(torch.bfloat16).to(device)
    config.device = device
    postprocessor = DataProcessorPipeline.from_pretrained(
        str(checkpoint), config_filename="policy_postprocessor.json"
    )
    identity = {
        "backend": "lingbot_va",
        "policy_class": "LingBotVAPolicy",
        "checkpoint": str(checkpoint),
        "config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        "action_space": "eef_delta_16d_initial_relative",
        "bridge": "ik_position_commands_v1",
        "device": device,
        "attn_mode": config.attn_mode,
    }
    return policy, postprocessor, identity


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/training/lingbot_va_flatten_generated_20260913/serve/pretrained_model",
    )
    parser.add_argument(
        "--wan-root",
        default="outputs/training/lingbot_va_flatten_generated_20260913/serve/wan",
    )
    parser.add_argument(
        "--text-embeddings",
        default="outputs/training/lingbot_va_flatten_generated_20260913/serve/flatten_text_embeddings.pt",
    )
    parser.add_argument(
        "--lerobot-src",
        default="outputs/training/lingbot_va_fold_generated_20260913/runtime/lerobot",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-mode", choices=("torch", "flashattn", "flex"), default=None)
    parser.add_argument("--salvage", action="store_true", help="fresh chunk-1 every N steps, execute action frame 0 (misaligned checkpoints)")
    args = parser.parse_args()

    import torch

    policy, postprocessor, identity = _load_policy(args, args.device)
    if args.salvage:
        policy.config.salvage_frame0 = True
        identity["salvage_frame0"] = True
    print(json.dumps({"phase": "server_ready", "identity": identity}), flush=True)

    salvage_queue = deque()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(json.dumps({"phase": "listening", "host": args.host, "port": args.port}), flush=True)
    while True:
        conn, address = server.accept()
        print(json.dumps({"phase": "client_connected", "address": address}), flush=True)
        with conn:
            while True:
                try:
                    (length,) = struct.unpack("<Q", _recv_exact(conn, 8))
                    request = pickle.loads(_recv_exact(conn, length))
                except (ConnectionError, OSError):
                    break
                try:
                    if "reset" in request:
                        policy.reset()
                        salvage_queue.clear()
                        seed = int(request["reset"])
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        response = {"ok": True}
                    elif request.get("identity"):
                        response = {"identity": identity}
                    else:
                        batch = {}
                        for key in CAMERA_KEYS:
                            images = np.asarray(request[key])
                            if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
                                raise ValueError(f"expected uint8 [batch, H, W, 3] images for {key}")
                            batch[key] = (
                                torch.from_numpy(images.copy())
                                .to(args.device, torch.float32)
                                .permute(0, 3, 1, 2)
                                / 255.0
                            )
                        batch["task"] = str(request["task"])
                        if args.salvage:
                            if not salvage_queue:
                                policy.reset()
                                chunk = policy.predict_action_chunk(batch)
                                flat = postprocessor({"action": chunk.reshape(-1, chunk.shape[-1])})["action"]
                                chunk = flat.reshape(chunk.shape)[0].float().cpu().numpy()
                                salvage_queue.extend(chunk[: policy.config.action_per_frame])
                            action = salvage_queue.popleft()[None]
                        else:
                            action = policy.select_action(batch)
                            processed = postprocessor({"action": action})
                            action = processed["action"].float().cpu().numpy()
                        response = {"action": action.tolist()}
                except BaseException as error:  # noqa: BLE001 - reported back to the worker
                    import traceback

                    traceback.print_exc()
                    response = {"error": f"{type(error).__name__}: {error}"}
                _send(conn, response)


if __name__ == "__main__":
    main()
