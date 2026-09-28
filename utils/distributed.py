"""Small torchrun/DDP helpers; single-process remains the default."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from typing import Any, Mapping


def initialize_distributed(device_preference: str = "auto") -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    if device_preference == "cpu":
        device = torch.device("cpu")
    elif device_preference.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else device_preference)
    else:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size, local_rank


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(value: Any, source: int = 0) -> Any:
    """Broadcast a small Python control object, with a single-process no-op."""

    if not dist.is_initialized():
        return value
    payload = [value if dist.get_rank() == source else None]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def reduce_weighted_metrics(
    metrics: Mapping[str, float], sample_count: int, device: torch.device
) -> dict[str, float]:
    """All-reduce metric numerators and their sample denominator."""

    if not dist.is_initialized():
        return dict(metrics)
    names = sorted(metrics)
    numerators = [float(metrics[name]) * sample_count for name in names]
    values = torch.tensor(
        [*numerators, float(sample_count)], device=device, dtype=torch.float64
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    denominator = max(float(values[-1].item()), 1.0)
    return {
        name: float(values[index].item()) / denominator
        for index, name in enumerate(names)
    }


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
