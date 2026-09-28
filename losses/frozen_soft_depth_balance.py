"""Frozen, train-only soft depth balancing for pixel-wise depth regression.

The previous soft balancing helper re-scaled a train-derived curve against every
minibatch.  That makes the weight of one physical target depend on its
neighbours in the batch.  This module deliberately has no such operation:
``FrozenSoftDepthBalance`` is calibrated once from canonical train pixels, then
evaluated as a pure target-to-weight function at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def _validate_bounds(minimum: float, maximum: float) -> None:
    if minimum <= 0.0 or maximum < minimum or not minimum <= 1.0 <= maximum:
        raise ValueError("soft-depth bounds must satisfy 0 < minimum <= 1 <= maximum")


def _validate_knots(values: Sequence[float]) -> tuple[float, ...]:
    knots = tuple(float(value) for value in values)
    if len(knots) < 2:
        raise ValueError("frozen soft-depth balance requires at least two depth knots")
    if not all(torch.isfinite(torch.tensor(value)).item() for value in knots):
        raise ValueError("frozen soft-depth knots must be finite")
    if any(right <= left for left, right in zip(knots[:-1], knots[1:])):
        raise ValueError("frozen soft-depth knots must be strictly increasing")
    return knots


def _interpolate(
    target: torch.Tensor,
    knots: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate a monotone curve, with constant endpoint tails."""

    location = target.clamp(min=float(knots[0]), max=float(knots[-1]))
    upper = torch.bucketize(location, knots, right=False).clamp(1, knots.numel() - 1)
    lower = upper - 1
    left_x, right_x = knots[lower], knots[upper]
    blend = ((location - left_x) / (right_x - left_x).clamp_min(1.0e-12)).clamp(0.0, 1.0)
    return values[lower] + blend * (values[upper] - values[lower])


def _bounded_mean_one_scale(
    raw: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
    iterations: int = 64,
) -> torch.Tensor:
    """Find a *single train-derived* scale whose bounded train mean is one."""

    _validate_bounds(minimum, maximum)
    if raw.numel() == 0:
        raise ValueError("cannot calibrate frozen soft-depth balance without train pixels")
    if not torch.isfinite(raw).all() or torch.any(raw <= 0.0):
        raise ValueError("raw frozen soft-depth weights must be finite and positive")
    low = raw.new_tensor(0.0)
    high = (raw.new_tensor(float(maximum)) / raw.min()).clamp_min(1.0)
    # This loop is normally unnecessary after the analytic upper bound, but it
    # makes the invariant explicit if the implementation changes in the future.
    while bool(torch.mean((high * raw).clamp(minimum, maximum)) < 1.0):
        high = high * 2.0
    for _ in range(int(iterations)):
        midpoint = 0.5 * (low + high)
        mean = torch.mean((midpoint * raw).clamp(minimum, maximum))
        low = torch.where(mean < 1.0, midpoint, low)
        high = torch.where(mean >= 1.0, midpoint, high)
    return 0.5 * (low + high)


@dataclass(frozen=True)
class FrozenSoftDepthBalance:
    """A serializable, bounded target-to-weight curve fixed from train pixels."""

    depth_knots_m: tuple[float, ...]
    raw_weights: tuple[float, ...]
    normalization_constant: float
    final_weights_at_knots: tuple[float, ...]
    bin_counts: tuple[int, ...]
    minimum: float
    maximum: float
    alpha: float
    tau: float
    train_positive_pixels: int
    train_weight_mean: float
    train_weight_min: float
    train_weight_max: float

    @classmethod
    def from_train_depths(
        cls,
        depths: torch.Tensor | Sequence[float],
        depth_knots_m: Sequence[float],
        *,
        minimum: float = 0.5,
        maximum: float = 3.0,
        alpha: float = 0.5,
        tau: float = 10.0,
    ) -> "FrozenSoftDepthBalance":
        """Calibrate once from canonical positive *train* depths.

        ``depth_knots_m`` are bin boundaries.  Counts are measured in those
        bins, inverse-frequency weights are made monotone toward the tail, and
        one global scale is fitted on the complete train distribution.
        """

        _validate_bounds(float(minimum), float(maximum))
        if alpha < 0.0 or tau < 0.0:
            raise ValueError("soft_depth_balance alpha and tau must be nonnegative")
        knots = _validate_knots(depth_knots_m)
        values = torch.as_tensor(depths, dtype=torch.float64, device="cpu").flatten()
        values = values[torch.isfinite(values) & (values > 0.0)]
        if values.numel() == 0:
            raise ValueError("no finite positive train depths are available for calibration")
        tolerance = 1.0e-7
        if float(values.min()) < knots[0] - tolerance or float(values.max()) > knots[-1] + tolerance:
            raise ValueError(
                "frozen soft-depth knots must cover all canonical train depths: "
                f"observed=[{float(values.min())}, {float(values.max())}], knots={knots}"
            )
        knot_tensor = values.new_tensor(knots)
        bin_index = torch.bucketize(values, knot_tensor[1:-1], right=False)
        counts = torch.bincount(bin_index, minlength=len(knots) - 1).to(torch.int64)
        raw_by_bin = (counts.to(values.dtype) + float(tau)).clamp_min(1.0e-12).pow(-float(alpha))
        raw_by_bin = torch.cummax(raw_by_bin, dim=0).values
        # Values at bin boundaries give a continuous curve while preserving the
        # train-bin inverse-frequency construction.  Adjacent bins are averaged
        # at their shared boundary; endpoint tails are constant.
        raw_knots = torch.empty(len(knots), dtype=values.dtype)
        raw_knots[0] = raw_by_bin[0]
        raw_knots[-1] = raw_by_bin[-1]
        if raw_knots.numel() > 2:
            raw_knots[1:-1] = 0.5 * (raw_by_bin[:-1] + raw_by_bin[1:])
        raw_values = _interpolate(values, knot_tensor, raw_knots)
        scale = _bounded_mean_one_scale(
            raw_values, minimum=float(minimum), maximum=float(maximum)
        )
        final_values = (scale * raw_values).clamp(float(minimum), float(maximum))
        final_knots = (scale * raw_knots).clamp(float(minimum), float(maximum))
        return cls(
            depth_knots_m=knots,
            raw_weights=tuple(float(value) for value in raw_knots.tolist()),
            normalization_constant=float(scale),
            final_weights_at_knots=tuple(float(value) for value in final_knots.tolist()),
            bin_counts=tuple(int(value) for value in counts.tolist()),
            minimum=float(minimum),
            maximum=float(maximum),
            alpha=float(alpha),
            tau=float(tau),
            train_positive_pixels=int(values.numel()),
            train_weight_mean=float(final_values.mean()),
            train_weight_min=float(final_values.min()),
            train_weight_max=float(final_values.max()),
        )

    def weights(self, target: torch.Tensor, positive: torch.Tensor | None = None) -> torch.Tensor:
        """Return frozen bounded weights; never inspect batch composition."""

        knots = target.new_tensor(self.depth_knots_m)
        raw = target.new_tensor(self.raw_weights)
        weights = (
            float(self.normalization_constant) * _interpolate(target, knots, raw)
        ).clamp(float(self.minimum), float(self.maximum))
        if positive is None:
            return weights
        selected = torch.broadcast_to(positive, target.shape) > 0.5
        return torch.where(selected, weights, torch.ones_like(weights))

    def _payload_without_sha256(self) -> dict[str, Any]:
        return {
            "schema": "frozen_soft_depth_balance",
            "kind": "frozen_soft_depth_balance",
            "runtime_rule": "weight = clamp(normalization_constant * interpolate(raw_weights, target), minimum, maximum); no minibatch normalization",
            "depth_knots_m": list(self.depth_knots_m),
            "raw_weights": list(self.raw_weights),
            "normalization_constant": self.normalization_constant,
            "final_weights_at_knots": list(self.final_weights_at_knots),
            "bin_counts": list(self.bin_counts),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "alpha": self.alpha,
            "tau": self.tau,
            "train_positive_pixels": self.train_positive_pixels,
            "train_weight_mean": self.train_weight_mean,
            "train_weight_min": self.train_weight_min,
            "train_weight_max": self.train_weight_max,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self._payload_without_sha256(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_sha256(), "sha256": self.sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenSoftDepthBalance":
        expected = payload.get("sha256")
        value = cls(
            depth_knots_m=tuple(float(item) for item in payload["depth_knots_m"]),
            raw_weights=tuple(float(item) for item in payload["raw_weights"]),
            normalization_constant=float(payload["normalization_constant"]),
            final_weights_at_knots=tuple(float(item) for item in payload["final_weights_at_knots"]),
            bin_counts=tuple(int(item) for item in payload["bin_counts"]),
            minimum=float(payload["minimum"]),
            maximum=float(payload["maximum"]),
            alpha=float(payload["alpha"]),
            tau=float(payload["tau"]),
            train_positive_pixels=int(payload["train_positive_pixels"]),
            train_weight_mean=float(payload["train_weight_mean"]),
            train_weight_min=float(payload["train_weight_min"]),
            train_weight_max=float(payload["train_weight_max"]),
        )
        _validate_knots(value.depth_knots_m)
        _validate_bounds(value.minimum, value.maximum)
        if len(value.raw_weights) != len(value.depth_knots_m):
            raise ValueError("frozen raw weights must align with depth knots")
        if len(value.final_weights_at_knots) != len(value.depth_knots_m):
            raise ValueError("frozen final weights must align with depth knots")
        if len(value.bin_counts) != len(value.depth_knots_m) - 1:
            raise ValueError("frozen bin counts must align with depth-knot intervals")
        if expected is not None and str(expected) != value.sha256:
            raise ValueError("frozen soft-depth balance SHA-256 mismatch")
        return value


def load_frozen_soft_depth_balance(path: str | Path) -> FrozenSoftDepthBalance:
    """Load and validate a JSON curve emitted by the train-only calibration."""

    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"frozen soft-depth artifact must be a JSON object: {resolved}")
    return FrozenSoftDepthBalance.from_dict(payload)
