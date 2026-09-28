"""Central, model-output-aware supervision masks for flood-depth tasks.

Continuous depth labels are meaningful only where the selected model can produce a
valid S1/terrain estimate.  Keeping this definition here prevents training,
selection and reporting from silently drifting onto different pixel domains.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


CANONICAL_POSITIVE_MASK = "valid_depth_mask_and_output_valid"


def _boolean_mask(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    return value > 0.5


def canonical_positive_supervision_mask(
    valid_depth_mask: torch.Tensor,
    output_valid: torch.Tensor,
) -> torch.Tensor:
    """Return ``valid_depth_mask & output_valid`` with explicit broadcasting."""

    valid = _boolean_mask(valid_depth_mask, "valid_depth_mask")
    output = _boolean_mask(output_valid, "output_valid")
    try:
        valid, output = torch.broadcast_tensors(valid, output)
    except RuntimeError as exc:
        raise ValueError(
            "valid_depth_mask and output_valid are not broadcast-compatible: "
            f"{tuple(valid.shape)} vs {tuple(output.shape)}"
        ) from exc
    return valid & output


def canonical_positive_mask_from_batch(batch: Mapping[str, Any]) -> torch.Tensor:
    """Resolve the canonical positive-depth supervision mask from a batch."""

    try:
        masks = batch["masks"]
        validity = batch["validity"]
        valid_depth_mask = masks["valid_depth_mask"]
        output_valid = validity["output_valid"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "canonical supervision requires masks.valid_depth_mask and "
            "validity.output_valid"
        ) from exc
    return canonical_positive_supervision_mask(valid_depth_mask, output_valid)


def s1_event_support_mask(
    s1_available: torch.Tensor,
    s1_t2_valid_fraction: torch.Tensor,
    minimum_event_band_fraction: float,
) -> torch.Tensor:
    """Return the selected-event-band S1 support mask.

    ``s1_t2_valid_fraction`` is the fraction of *selected model event bands* with
    valid raster values, so this remains invariant to BandSpec ordering.
    """

    threshold = float(minimum_event_band_fraction)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_event_band_fraction must lie in [0, 1]")
    available = _boolean_mask(s1_available, "s1_available")
    if not isinstance(s1_t2_valid_fraction, torch.Tensor):
        raise TypeError("s1_t2_valid_fraction must be a tensor")
    try:
        available, fraction = torch.broadcast_tensors(
            available, s1_t2_valid_fraction
        )
    except RuntimeError as exc:
        raise ValueError(
            "s1_available and s1_t2_valid_fraction are not "
            "broadcast-compatible"
        ) from exc
    return available & (fraction >= threshold)


def s1_output_valid_mask(
    s1_available: torch.Tensor,
    s1_t2_valid_fraction: torch.Tensor,
    dem_available: torch.Tensor,
    minimum_event_band_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(s1_event_support, output_valid)`` for an S1-only model."""

    event_support = s1_event_support_mask(
        s1_available, s1_t2_valid_fraction, minimum_event_band_fraction
    )
    dem = _boolean_mask(dem_available, "dem_available")
    try:
        event_support, dem = torch.broadcast_tensors(event_support, dem)
    except RuntimeError as exc:
        raise ValueError("s1_event_support and dem_available are incompatible") from exc
    return event_support, event_support & dem


def supervision_mask_counts(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return differentiability-independent count diagnostics for one batch."""

    valid = _boolean_mask(batch["masks"]["valid_depth_mask"], "valid_depth_mask")
    output = _boolean_mask(batch["validity"]["output_valid"], "output_valid")
    valid, output = torch.broadcast_tensors(valid, output)
    positive = valid & output
    excluded = valid & ~output
    dtype = batch["label"].dtype
    valid_count = valid.sum().to(dtype)
    output_count = output.sum().to(dtype)
    positive_count = positive.sum().to(dtype)
    excluded_count = excluded.sum().to(dtype)
    return {
        "valid_depth_mask_pixels": valid_count,
        "output_valid_pixels": output_count,
        "positive_supervision_pixels": positive_count,
        "positive_excluded_by_output_valid_pixels": excluded_count,
        "positive_excluded_by_output_valid_fraction": excluded_count
        / valid_count.clamp_min(1.0),
    }


def validate_supervision_config(config: Mapping[str, Any]) -> str:
    """Validate and return the one supported training/validation mask schema."""

    supervision = config.get("supervision", {})
    if supervision is None:
        supervision = {}
    if not isinstance(supervision, Mapping):
        raise ValueError("supervision must be a mapping")
    positive = str(supervision.get("positive_mask", CANONICAL_POSITIVE_MASK))
    validation = str(supervision.get("validation_mask", CANONICAL_POSITIVE_MASK))
    if positive != CANONICAL_POSITIVE_MASK or validation != CANONICAL_POSITIVE_MASK:
        raise ValueError(
            "supervision.positive_mask and supervision.validation_mask must both be "
            f"{CANONICAL_POSITIVE_MASK!r}"
        )
    return CANONICAL_POSITIVE_MASK
