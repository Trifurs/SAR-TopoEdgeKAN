"""Explicit mixed-precision policy resolution."""

from __future__ import annotations

import torch


def resolve_amp(device: torch.device, enabled: bool, requested: str) -> tuple[bool, torch.dtype, bool]:
    if requested not in {"auto", "float16", "bfloat16"}:
        raise ValueError("training.amp_dtype must be auto, float16, or bfloat16")
    active = bool(enabled) and device.type == "cuda"
    if requested == "bfloat16" or (requested == "auto" and active and torch.cuda.is_bf16_supported()):
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    if active and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 was requested but this CUDA device does not support it")
    return active, dtype, active and dtype == torch.float16
