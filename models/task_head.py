"""Small task head for the production flood-depth model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.encoders import group_count


class TaskHead(nn.Module):
    """A compact convolutional prediction head for dense outputs."""

    def __init__(
        self,
        channels: int,
        groups: int,
        *,
        input_channels: int | None = None,
    ) -> None:
        super().__init__()
        input_width = int(input_channels if input_channels is not None else channels)
        if input_width <= 0:
            raise ValueError("task-head input channels must be positive")
        self.trunk = nn.Sequential(
            nn.Conv2d(input_width, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class KnownRangeHydrostaticPrior(nn.Module):
    """Construct a parameter-free depth prior from flood range and DSM.

    The repository's established comparison protocol supplies the exact binary
    flood range to every baseline.  Given the same range, a low boundary-
    elevation quantile is a robust water-surface candidate: it avoids isolated
    DSM obstacles while retaining the hydrostatic relation ``depth = WSE - z``.
    The neural head predicts only a bounded log-depth correction to this prior.
    """

    def __init__(
        self,
        *,
        boundary_quantile: float = 0.25,
        minimum_depth_m: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 <= boundary_quantile <= 1.0:
            raise ValueError("hydrostatic boundary quantile must lie in [0, 1]")
        if minimum_depth_m <= 0.0:
            raise ValueError("hydrostatic minimum depth must be positive")
        self.boundary_quantile = float(boundary_quantile)
        self.minimum_depth_m = float(minimum_depth_m)

    @staticmethod
    def _component_labels(support: torch.Tensor) -> torch.Tensor:
        """Approximate exact 8-connected labels with GPU min propagation.

        FloodDepthNet patches are 256 px wide and contain compact flood
        objects. Propagating for the maximum image dimension gives every
        compact component a common minimum index. Extremely winding objects may
        split into local subcomponents, which is conservative for water-level
        estimation and never joins disconnected ranges.
        """

        height, width = support.shape[-2:]
        identifiers = torch.arange(
            1,
            height * width + 1,
            device=support.device,
            dtype=torch.float32,
        ).reshape(1, 1, height, width)
        identifiers = identifiers.expand(support.shape[0], -1, -1, -1)
        background = float(height * width + 1)
        labels = torch.where(
            support,
            identifiers,
            torch.full_like(identifiers, background),
        )
        for _ in range(max(height, width) - 1):
            neighbour_minimum = -F.max_pool2d(
                -labels, kernel_size=3, stride=1, padding=1
            )
            labels = torch.where(
                support, torch.minimum(labels, neighbour_minimum), labels
            )
        return labels

    def _componentwise_prior(
        self,
        elevation: torch.Tensor,
        support: torch.Tensor,
        boundary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = self._component_labels(support)
        prior = torch.full_like(elevation, self.minimum_depth_m)
        water_surface = torch.zeros_like(elevation)
        for batch_index in range(elevation.shape[0]):
            sample_support = support[batch_index, 0]
            component_ids = torch.unique(labels[batch_index, 0][sample_support])
            for component_id in component_ids.tolist():
                component = sample_support & (
                    labels[batch_index, 0] == float(component_id)
                )
                component_boundary = boundary[batch_index, 0] & component
                values = elevation[batch_index, 0][component_boundary]
                if values.numel() == 0:
                    values = elevation[batch_index, 0][component]
                level = torch.quantile(values, self.boundary_quantile)
                surface = level + self.minimum_depth_m
                water_surface[batch_index, 0][component] = surface
                prior[batch_index, 0][component] = (
                    surface - elevation[batch_index, 0][component]
                ).clamp_min(self.minimum_depth_m)
        return prior, water_surface

    def forward(
        self,
        elevation_m: torch.Tensor,
        flood_support: torch.Tensor,
        dem_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = elevation_m[:, :1].shape
        if flood_support.shape != expected or dem_valid.shape != expected:
            raise ValueError(
                "elevation, flood support, and DEM validity must share Bx1xHxW shape"
            )
        # The prior is deterministic input geometry, so sorting the boundary
        # elevations need not build an autograd graph.  Float32 also keeps the
        # quantile stable when the surrounding network runs under AMP.
        with torch.no_grad():
            elevation = elevation_m.detach().float()
            support = (flood_support > 0.5) & (dem_valid > 0.5)
            # Binary erosion with an invalid exterior; subtracting it from the
            # support yields the one-pixel inner boundary used by the audit.
            invalid = (~support).to(elevation.dtype)
            eroded = 1.0 - F.max_pool2d(
                F.pad(invalid, (1, 1, 1, 1), value=1.0),
                kernel_size=3,
                stride=1,
            )
            boundary = support & (eroded < 0.5)
            prior, water_surface = self._componentwise_prior(elevation, support, boundary)
        return prior.to(elevation_m.dtype), water_surface.to(elevation_m.dtype)
