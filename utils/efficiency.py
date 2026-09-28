"""Consistent inference-efficiency and model-size measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch


MEBIBYTE = 1024.0**2


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class InferenceEfficiency:
    """Accumulate synchronized model/estimator latency without timing data I/O."""

    device: torch.device
    enabled: bool = True
    batch_seconds: list[float] = field(default_factory=list)
    samples: int = 0
    output_pixels: int = 0
    warmup_batches: int | None = None
    _warmup_records: list[tuple[float, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.warmup_batches is None:
            self.warmup_batches = 1 if self.device.type == "cuda" else 0
        if self.warmup_batches < 0:
            raise ValueError("warmup_batches must be non-negative")
        if self.enabled and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def start(self) -> float | None:
        if not self.enabled:
            return None
        _synchronize(self.device)
        return time.perf_counter()

    def stop(self, started: float | None, *, samples: int, output_pixels: int) -> None:
        if not self.enabled or started is None:
            return
        _synchronize(self.device)
        elapsed = time.perf_counter() - started
        if elapsed < 0.0 or not math.isfinite(elapsed):
            raise RuntimeError(f"Invalid inference duration: {elapsed}")
        if len(self._warmup_records) < int(self.warmup_batches):
            self._warmup_records.append((elapsed, int(samples), int(output_pixels)))
            return
        self.batch_seconds.append(elapsed)
        self.samples += int(samples)
        self.output_pixels += int(output_pixels)

    def summary(self) -> dict[str, Any]:
        if not self.enabled or not self.batch_seconds:
            if not self.enabled or not self._warmup_records:
                return {}
            # A one-batch diagnostic should still report efficiency; formal full
            # evaluations exclude the CUDA warm-up batch below.
            measured_seconds = [record[0] for record in self._warmup_records]
            measured_samples = sum(record[1] for record in self._warmup_records)
            measured_pixels = sum(record[2] for record in self._warmup_records)
            excluded_warmup_batches = 0
        else:
            measured_seconds = self.batch_seconds
            measured_samples = self.samples
            measured_pixels = self.output_pixels
            excluded_warmup_batches = len(self._warmup_records)
        latencies = np.asarray(measured_seconds, dtype=np.float64)
        total = float(latencies.sum())
        result: dict[str, Any] = {
            "efficiency_timing_scope": "model_forward_only_synchronized",
            "efficiency_device": str(self.device),
            "efficiency_warmup_batches_excluded": excluded_warmup_batches,
            "efficiency_batches": int(latencies.size),
            "efficiency_samples": int(measured_samples),
            "efficiency_output_pixels": int(measured_pixels),
            "efficiency_forward_seconds": total,
            "efficiency_samples_per_second": float(measured_samples / total),
            "efficiency_megapixels_per_second": float(
                measured_pixels / 1_000_000.0 / total
            ),
            "efficiency_mean_batch_latency_ms": float(latencies.mean() * 1_000.0),
            "efficiency_p50_batch_latency_ms": float(
                np.quantile(latencies, 0.50) * 1_000.0
            ),
            "efficiency_p95_batch_latency_ms": float(
                np.quantile(latencies, 0.95) * 1_000.0
            ),
        }
        if self.device.type == "cuda":
            result.update(
                {
                    "efficiency_accelerator": torch.cuda.get_device_name(self.device),
                    "efficiency_peak_gpu_allocated_mib": float(
                        torch.cuda.max_memory_allocated(self.device) / MEBIBYTE
                    ),
                    "efficiency_peak_gpu_reserved_mib": float(
                        torch.cuda.max_memory_reserved(self.device) / MEBIBYTE
                    ),
                }
            )
        return result


def model_size_metrics(model: torch.nn.Module) -> dict[str, Any]:
    """Return parameter counts and in-memory tensor footprint for one model."""

    unwrapped = model.module if hasattr(model, "module") else model
    parameters = list(unwrapped.parameters())
    buffers = list(unwrapped.buffers())
    return {
        "total_parameters": int(sum(item.numel() for item in parameters)),
        "trainable_parameters": int(
            sum(item.numel() for item in parameters if item.requires_grad)
        ),
        "parameter_storage_mib": float(
            sum(item.numel() * item.element_size() for item in parameters) / MEBIBYTE
        ),
        "buffer_storage_mib": float(
            sum(item.numel() * item.element_size() for item in buffers) / MEBIBYTE
        ),
    }


def checkpoint_size_metrics(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve(strict=True)
    size = checkpoint.stat().st_size
    return {
        "checkpoint_bytes": int(size),
        "checkpoint_size_mib": float(size / MEBIBYTE),
    }
