"""Synchronous geometric augmentation for SAR-and-terrain raster samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _map_tensors(value: Any, function: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim >= 2:
        return function(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    return value


@dataclass
class SynchronousAugment:
    """Apply spatially aligned flips and right-angle rotations."""

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotate90_probability: float = 0.0
    feature_dropout_probability: float | None = None
    sensor_missing_simulation_probability: float | None = None
    input_mode: str = "s1_terrain"

    def __post_init__(self) -> None:
        if self.input_mode != "s1_terrain":
            raise ValueError("Only the SAR-and-terrain input mode is supported")
        for name in (
            "horizontal_flip_probability",
            "vertical_flip_probability",
            "rotate90_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if float(self.feature_dropout_probability or 0.0) != 0.0:
            raise ValueError("feature dropout is not supported for the single-sensor model")
        if float(self.sensor_missing_simulation_probability or 0.0) != 0.0:
            raise ValueError("sensor-missing simulation is not supported for the single-sensor model")

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample.setdefault("metadata", {})
        if torch.rand(()) < self.horizontal_flip_probability:
            sample = _map_tensors(sample, lambda tensor: torch.flip(tensor, dims=(-1,)))
        if torch.rand(()) < self.vertical_flip_probability:
            sample = _map_tensors(sample, lambda tensor: torch.flip(tensor, dims=(-2,)))
        if torch.rand(()) < self.rotate90_probability:
            turns = int(torch.randint(1, 4, ()).item())
            sample = _map_tensors(
                sample,
                lambda tensor: torch.rot90(tensor, turns, dims=(-2, -1)),
            )
        sample["metadata"]["augmentation"] = "geometric"
        return sample
