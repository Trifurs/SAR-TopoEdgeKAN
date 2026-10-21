"""Explicit observation whitelist shared by discussion model adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from models.terrain_features import _masked_mean, ground_like_proxy


OBSERVATION_GROUPS = ("s1_t1", "s1_t2", "s1_change", "terrain")


def shared_observation_channels(band_spec: Any, raw_names: tuple[str, ...]) -> int:
    return (
        sum(band_spec.channels(key) for key in OBSERVATION_GROUPS)
        + band_spec.channels("s1_conditioning")
        + len(raw_names)
        + 3  # S1 validity, event support, DEM validity
        + len(OBSERVATION_GROUPS)  # per-branch valid fractions
        + 1  # supplied known flood range
    )


def observation_columns(band_spec: Any, raw_names: tuple[str, ...]) -> list[str]:
    names = [f"{key}/{name}" for key in OBSERVATION_GROUPS for name in band_spec.names(key)]
    names.extend(f"s1_conditioning/{name}" for name in band_spec.names("s1_conditioning"))
    names.extend(f"terrain_raw/{name}" for name in raw_names)
    names.extend(("s1_valid", "s1_event_support", "dem_valid"))
    names.extend(f"branch_validity/{key}" for key in OBSERVATION_GROUPS)
    names.append("valid_depth_mask")
    return names


def assemble_observations(
    inputs: Mapping[str, Any], raw_names: tuple[str, ...]
) -> torch.Tensor:
    """Expose exactly the observations available to the full production model.

    Raw terrain is expressed in km and fractions of 90 degrees at the model
    adapter, without clipping or estimating any new dataset statistics. Its
    original metre-valued tensor remains available for the shared prior.
    Labels, event IDs, QA/reliability values and other masks are never appended.
    """

    parts = [inputs[key].float() for key in OBSERVATION_GROUPS]
    angles = inputs.get("s1_conditioning")
    if angles is not None:
        parts.append(angles.float())
    scales = {"elevation_m_DSM": 1000.0, "slope_deg": 90.0}
    raw = inputs["terrain_raw"].float()
    if raw.shape[1] != len(raw_names):
        raise ValueError("Raw terrain channels differ from the audited contract")
    raw_scales = raw.new_tensor([scales.get(name, 1.0) for name in raw_names])
    parts.append(raw / raw_scales.view(1, -1, 1, 1))
    parts.extend(inputs[key].float() for key in ("s1_valid", "s1_event_support", "dem_valid"))
    validity = inputs["branch_validity"]
    parts.extend(validity[key].float() for key in OBSERVATION_GROUPS)
    parts.append(inputs["valid_depth_mask"].float())
    return torch.cat(parts, dim=1)


def evaluation_terrain_features(
    inputs: Mapping[str, Any], raw_names: tuple[str, ...], model_config: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Compute common physical diagnostics outside the model timing interval."""

    index = raw_names.index("elevation_m_DSM")
    elevation = inputs["terrain_raw"][:, index:index + 1].float()
    valid = (inputs["dem_valid"] > 0.5).to(elevation.dtype)
    mean = _masked_mean(elevation, valid, 9)
    ground = ground_like_proxy(
        elevation, valid, int(model_config.get("ground_proxy_kernel_size", 9))
    )
    physics = ground if model_config.get("physics_elevation", "z_ground_proxy") == "z_ground_proxy" else mean
    relief = (_masked_mean(elevation.square(), valid, 9) - mean.square()).clamp_min(0).sqrt() * valid
    return {
        "dsm_elevation": elevation,
        "z_hyd": physics,
        "physics_elevation": physics,
        "local_relief": relief,
        "dem_valid": valid,
    }
