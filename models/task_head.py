"""Small task head for the production flood-depth model."""

from __future__ import annotations

import math

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
        componentwise: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= boundary_quantile <= 1.0:
            raise ValueError("hydrostatic boundary quantile must lie in [0, 1]")
        if minimum_depth_m <= 0.0:
            raise ValueError("hydrostatic minimum depth must be positive")
        self.boundary_quantile = float(boundary_quantile)
        self.minimum_depth_m = float(minimum_depth_m)
        self.componentwise = bool(componentwise)

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
            if self.componentwise:
                prior, water_surface = self._componentwise_prior(
                    elevation, support, boundary
                )
            else:
                flat_boundary = elevation.masked_fill(~boundary, torch.nan).flatten(1)
                flat_support = elevation.masked_fill(~support, torch.nan).flatten(1)
                boundary_level = torch.nanquantile(
                    flat_boundary, self.boundary_quantile, dim=1, keepdim=True
                )
                support_level = torch.nanquantile(
                    flat_support, self.boundary_quantile, dim=1, keepdim=True
                )
                level = torch.where(
                    torch.isfinite(boundary_level), boundary_level, support_level
                )
                level = torch.nan_to_num(level, nan=0.0)
                water_surface = (
                    level[:, :, None, None] + self.minimum_depth_m
                ).expand_as(elevation)
                prior = (water_surface - elevation).clamp_min(
                    self.minimum_depth_m
                )
                prior = torch.where(
                    support, prior, torch.full_like(prior, self.minimum_depth_m)
                )
        return prior.to(elevation_m.dtype), water_surface.to(elevation_m.dtype)


class ContextualDepthRangeCalibration(nn.Module):
    """Small, identity-initialized calibration internal to the depth head.

    Robust pixel losses can compress the upper tail even when decoded spatial
    structure remains informative. This branch predicts bounded, spatially
    varying logit scale and bias residuals from decoded features, the baseline
    depth logit, and mask-agnostic patch context. A fixed smooth tail support
    protects the already accurate shallow regime. It never consumes a flood or
    label-validity mask.
    """

    def __init__(
        self,
        channels: int,
        groups: int,
        *,
        width: int = 16,
        maximum_scale_residual: float = 1.0,
        maximum_bias_residual: float = 2.0,
        tail_threshold_m: float = 0.46,
        tail_temperature_m: float = 0.10,
        depth_epsilon: float = 0.001,
        strength: float = 1.0,
        mode: str = "predicted_tail",
    ) -> None:
        super().__init__()
        if min(channels, width) <= 0:
            raise ValueError("depth-range calibration channels must be positive")
        if maximum_scale_residual <= 0.0 or maximum_bias_residual <= 0.0:
            raise ValueError("depth-range calibration bounds must be positive")
        if tail_threshold_m < 0.0 or tail_temperature_m <= 0.0:
            raise ValueError("depth-range calibration tail parameters are invalid")
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("depth-range calibration strength must be finite and nonnegative")
        self.mode = str(mode)
        if self.mode not in {"predicted_tail", "all", "depth_log_scale"}:
            raise ValueError(
                "depth-range calibration mode must be predicted_tail, all, or "
                "depth_log_scale"
            )
        self.local = nn.Sequential(
            nn.Conv2d(channels + 1, width, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(width, groups), width),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, bias=False),
            nn.GroupNorm(group_count(width, groups), width),
            nn.SiLU(inplace=True),
        )
        self.projection = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.maximum_scale_residual = float(maximum_scale_residual)
        self.maximum_bias_residual = float(maximum_bias_residual)
        self.tail_threshold_m = float(tail_threshold_m)
        self.tail_temperature_m = float(tail_temperature_m)
        self.depth_epsilon = float(depth_epsilon)
        # This is deliberately a non-persistent scalar: sweeping it changes no
        # learned tensor and therefore keeps checkpoints fully compatible.  A
        # value of zero recovers the pre-calibration depth head exactly.
        self.strength = float(strength)
        self.returns_depth = self.mode == "depth_log_scale"

    def forward(
        self, features: torch.Tensor, depth_logit: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if depth_logit.shape != features[:, :1].shape:
            raise ValueError("depth calibration logit must match the feature grid")
        local = self.local(
            torch.cat((features, torch.tanh(depth_logit / 4.0)), dim=1)
        )
        context = self.context(
            torch.cat(
                (
                    F.adaptive_avg_pool2d(local, 1),
                    F.adaptive_max_pool2d(local, 1),
                ),
                dim=1,
            )
        )
        raw_scale, raw_bias = self.projection(local + context).chunk(2, dim=1)
        scale = self.maximum_scale_residual * torch.tanh(
            raw_scale / self.maximum_scale_residual
        )
        bias = self.maximum_bias_residual * torch.tanh(
            raw_bias / self.maximum_bias_residual
        )
        baseline_depth = F.softplus(depth_logit) + self.depth_epsilon
        if self.mode == "predicted_tail":
            support = torch.sigmoid(
                (baseline_depth - self.tail_threshold_m)
                / self.tail_temperature_m
            )
        else:
            # The residual predictor already consumes local and patch context.
            # Applying it everywhere avoids the circular failure mode in which a
            # truly deep pixel receives no calibration gradient merely because
            # the baseline currently predicts it as shallow.
            support = torch.ones_like(baseline_depth)
        effective_scale = self.strength * support * scale
        effective_bias = self.strength * support * bias
        if self.mode == "depth_log_scale":
            # ``effective_scale`` is a residual in log-depth space.  It can
            # express large event-dependent ranges without requiring the
            # shallow-regime softplus logit itself to grow by tens of metres.
            calibrated = (
                F.softplus(depth_logit + effective_bias) + self.depth_epsilon
            ) * torch.exp(effective_scale.clamp(-8.0, 8.0))
        else:
            calibrated = depth_logit * (1.0 + effective_scale) + effective_bias
        return calibrated, effective_scale, effective_bias
