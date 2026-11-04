"""Explicit transport settings applied at the RLINF worker boundary."""

from __future__ import annotations

import os
import shlex
from typing import Any

_KEYS = frozenset(
    {
        "CLOTHDOJO_LEROBOT_VIDEO_BACKEND",
        "GLOO_SOCKET_IFNAME",
        "LD_LIBRARY_PATH",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_HCA",
        "RAYON_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    }
)


def worker_environment_from_config(cfg: Any) -> dict[str, str | None]:
    from omegaconf import OmegaConf

    project = cfg.get("clothdojo", None)
    configured = None if project is None else project.get("worker_environment")
    if configured is None:
        return {}
    resolved = OmegaConf.to_container(configured, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("clothdojo.worker_environment must be a mapping")
    unknown = set(resolved) - _KEYS
    if unknown:
        raise ValueError(f"unsupported distributed environment keys: {sorted(unknown)}")
    return {str(name): None if value is None else str(value) for name, value in resolved.items()}


def apply_worker_environment(cfg: Any) -> None:
    """Override container defaults inside the actor, before FSDP starts."""
    for name, value in worker_environment_from_config(cfg).items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def configure_cluster_worker_environment(cluster: Any, cfg: Any) -> None:
    """Publish startup-sensitive variables before Ray creates actor processes."""

    values = worker_environment_from_config(cfg)
    if not values:
        return
    for node_rank in range(int(cfg.cluster.num_nodes)):
        node = cluster.get_node_info(node_rank)
        for name, value in values.items():
            if value is not None:
                node.env_vars[name] = value
        library_path = values.get("LD_LIBRARY_PATH")
        if library_path is not None:
            interpreter = node.python_interpreter_path
            if not interpreter:
                raise ValueError("LD_LIBRARY_PATH requires a worker Python interpreter")
            # Ray's runtime-env process mutates os.environ before exec, but glibc
            # can retain the loader search path from that process. Put the value
            # on the final exec command so the actor Python starts with it.
            node.python_interpreter_path = (
                f"/usr/bin/env LD_LIBRARY_PATH={shlex.quote(library_path)} "
                f"{shlex.quote(interpreter)}"
            )
