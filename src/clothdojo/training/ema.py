"""FP32 master-state checks and EMA updates used by reference SFT."""
from __future__ import annotations
from typing import Any
import torch


def _validate_fp32_training_state(model: Any, optimizer: Any) -> None:
    """Fail before step one if FSDP or AdamW lost FP32 master state."""

    parameter_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if parameter_dtypes != {torch.float32}:
        raise RuntimeError(
            f"OpenPI/RLinf trainable parameters must be FP32 masters, got {parameter_dtypes}"
        )
    state_dtypes = {
        value.dtype
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value) and value.is_floating_point()
    }
    if state_dtypes and state_dtypes != {torch.float32}:
        raise RuntimeError(
            f"OpenPI/RLinf optimizer state must be FP32, got {state_dtypes}"
        )


def _trainable_parameters(model: Any) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.is_floating_point()
    }


@torch.no_grad()
def _initialize_ema_state(model: Any) -> dict[str, torch.Tensor]:
    parameters = _trainable_parameters(model)
    if not parameters:
        raise RuntimeError("OpenPI/RLinf EMA requires trainable floating parameters")
    return {name: parameter.detach().clone() for name, parameter in parameters.items()}


@torch.no_grad()
def _update_ema_state(
    model: Any,
    ema_state: dict[str, torch.Tensor],
    decay: float,
) -> None:
    parameters = _trainable_parameters(model)
    _validate_ema_layout(parameters, ema_state)
    ema_values = list(ema_state.values())
    parameter_values = [parameter.detach() for parameter in parameters.values()]
    torch._foreach_mul_(ema_values, decay)
    torch._foreach_add_(ema_values, parameter_values, alpha=1.0 - decay)


@torch.no_grad()
def _swap_parameter_data_with_ema(
    model: Any,
    ema_state: dict[str, torch.Tensor],
    *,
    chunk_elements: int = 4 * 1024 * 1024,
) -> None:
    """Swap online and EMA values without allocating another model copy."""

    parameters = _trainable_parameters(model)
    _validate_ema_layout(parameters, ema_state)
    for name, parameter in parameters.items():
        online = parameter.view(-1)
        ema = ema_state[name].view(-1)
        for start in range(0, online.numel(), chunk_elements):
            stop = min(start + chunk_elements, online.numel())
            temporary = online[start:stop].clone()
            online[start:stop].copy_(ema[start:stop])
            ema[start:stop].copy_(temporary)


def _validate_ema_layout(
    parameters: dict[str, torch.nn.Parameter],
    ema_state: dict[str, torch.Tensor],
) -> None:
    if parameters.keys() != ema_state.keys():
        raise RuntimeError("OpenPI/RLinf EMA parameter names changed after initialization")
    for name, parameter in parameters.items():
        shadow = ema_state[name]
        if parameter.shape != shadow.shape or parameter.dtype != shadow.dtype:
            raise RuntimeError(
                f"OpenPI/RLinf EMA layout changed for {name}: "
                f"{parameter.shape}@{parameter.dtype} != {shadow.shape}@{shadow.dtype}"
            )
