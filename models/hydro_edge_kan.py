"""S1-only topographic Edge-KAN aggregation.

This graph is a static topographic affinity operator.  It is not a hydraulic
solver and does not use a second sensor or a cross-sensor time difference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import residual_path_scale
from models.kan_layers import KANLinear
from models.terrain_features import path_barrier_proxy
from models.graph_utils import DIRECTIONS, _masked_pool, _roll_with_boundary_mask


S1_EDGE_FEATURE_NAMES = (
    "signed_dz",
    "edge_slope",
    "relative_height",
    "path_barrier",
    "local_relief",
    "distance",
)


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(float(value), 1.0e-6)))


class HydroEdgeKAN(nn.Module):
    """Topographic-Affinity Edge-KAN (TAE-KAN) on an eight-neighbour graph."""

    def __init__(
        self,
        channels: int,
        heads: int = 2,
        grid_size: int = 4,
        spline_order: int = 3,
        graph_feature_stride: int = 4,
        terrain_pixel_size_m: float = 20.0,
        feature_centers: Sequence[float] | None = None,
        feature_scales: Sequence[float] | None = None,
        gamma_init_effective: float = 0.02,
        gamma_max: float = 0.25,
        gamma_parameterization: str = "positive",
        latent_compatibility_enabled: bool = True,
        path_barrier_mode: str = "full_resolution",
        drop_path_probability: float = 0.0,
        diagnostic_mode: bool = False,
        diagnostics_enabled: bool | None = None,
    ) -> None:
        super().__init__()
        if heads not in {2, 4} or channels % heads:
            raise ValueError("HydroEdgeKAN heads must be 2 or 4 and divide channels")
        if graph_feature_stride not in {4, 8}:
            raise ValueError("HydroEdgeKAN graph_feature_stride must be 4 or 8")
        gamma_parameterization = str(gamma_parameterization)
        valid_gamma = (
            0 <= gamma_init_effective < gamma_max
            if gamma_parameterization == "positive"
            else abs(gamma_init_effective) < gamma_max
        )
        if (
            terrain_pixel_size_m <= 0
            or gamma_max <= 0
            or gamma_parameterization not in {"positive", "signed"}
            or not valid_gamma
        ):
            raise ValueError("invalid graph scale, pixel size, or gamma bounds")
        centers = list(feature_centers or [0.0] * 6)
        scales = list(feature_scales or [0.10, 0.10, 1.0, 1.0, 1.0, 1.0])
        if len(centers) != 6 or len(scales) != 6 or any(float(value) <= 0 for value in scales):
            raise ValueError("S1 graph feature centers/scales must contain six valid values")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_channels = channels // heads
        self.graph_feature_stride = int(graph_feature_stride)
        self.graph_pixel_size_m = (
            float(self.graph_feature_stride) * float(terrain_pixel_size_m)
        )
        self.gamma_max = float(gamma_max)
        self.gamma_parameterization = gamma_parameterization
        self.latent_compatibility_enabled = bool(latent_compatibility_enabled)
        self.path_barrier_mode = str(path_barrier_mode)
        self.drop_path_probability = float(drop_path_probability)
        if not 0.0 <= self.drop_path_probability < 1.0:
            raise ValueError("TAE-KAN drop-path probability must lie in [0, 1)")
        if self.path_barrier_mode not in {"full_resolution", "graph_scale"}:
            raise ValueError(
                "HydroEdgeKAN path_barrier_mode must be full_resolution or graph_scale"
            )
        self.diagnostic_mode = (
            bool(diagnostic_mode)
            if diagnostics_enabled is None
            else bool(diagnostics_enabled)
        )
        self.edge_feature_names = S1_EDGE_FEATURE_NAMES
        self.register_buffer(
            "feature_centers",
            torch.tensor(centers, dtype=torch.float32).view(1, 1, 6, 1, 1),
        )
        self.register_buffer(
            "feature_scales",
            torch.tensor(scales, dtype=torch.float32).view(1, 1, 6, 1, 1),
        )
        self.edge_kan = nn.ModuleList(
            [
                KANLinear(
                    6,
                    1,
                    grid_size,
                    spline_order,
                    spline_scale_init=1.0,
                    zero_output_init=True,
                )
                for _ in range(heads)
            ]
        )
        # Each head owns a disjoint channel slice. Direct neighbour-current
        # differences implement the diffusion message without dense channel
        # projections around the graph operator.
        self.latent_compatibility = nn.ModuleList(
            [nn.Conv2d(2, 1, 1) for _ in range(heads)]
        )
        for projection in self.latent_compatibility:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        if not self.latent_compatibility_enabled:
            # The LCA is intentionally absent in this variant.  Keeping its
            # tensors in the state dictionary preserves checkpoint schema while
            # excluding them from optimization and distributed gradient checks.
            self.latent_compatibility.requires_grad_(False)
        # A zero-initialized within-head mixer lets the graph learn which
        # latent channels should exchange topographic messages without
        # adding dense channel projections.
        # The bounded residual keeps the initial operator exactly equal to the
        # audited physical diffusion and limits learned amplification.
        self.message_mixer = nn.ModuleList(
            [
                nn.Conv2d(
                    self.head_channels,
                    self.head_channels,
                    1,
                    bias=False,
                )
                for _ in range(heads)
            ]
        )
        for mixer in self.message_mixer:
            nn.init.zeros_(mixer.weight)
        self.prior_raw_scales = nn.Parameter(
            torch.tensor([_inverse_softplus(value) for value in (0.5, 0.25, 0.10, 0.15, 0.10, 0.02)])
        )
        self.prior_bias = nn.Parameter(torch.zeros(heads))
        initial_gamma = (
            _inverse_sigmoid(float(gamma_init_effective) / float(gamma_max))
            if self.gamma_parameterization == "positive"
            else math.atanh(float(gamma_init_effective) / float(gamma_max))
        )
        self.raw_gamma = nn.Parameter(torch.full((heads,), initial_gamma))

    @property
    def gamma(self) -> torch.Tensor:
        if self.gamma_parameterization == "signed":
            return self.gamma_max * torch.tanh(self.raw_gamma)
        # A smooth positive gate cannot strand a head on the zero side of a
        # hard clamp or saturate at a signed boundary.  The small configured
        # budget keeps this operator a topographic diffusion refinement rather
        # than an unconstrained replacement for the bottleneck features.
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    @property
    def prior_scales(self) -> torch.Tensor:
        return F.softplus(self.prior_raw_scales)

    @property
    def neighbour_distances_m(self) -> tuple[float, ...]:
        return tuple(
            self.graph_pixel_size_m * math.sqrt(dx * dx + dy * dy)
            for dy, dx in DIRECTIONS
        )

    def graph_identity(self, feature_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        """Serializable verified graph-scale metadata."""

        return {
            "graph_feature_stride": self.graph_feature_stride,
            "graph_node_spacing_m": self.graph_pixel_size_m,
            "graph_feature_shape": list(feature_shape) if feature_shape else None,
            "orthogonal_neighbour_distance_m": self.graph_pixel_size_m,
            "diagonal_neighbour_distance_m": self.graph_pixel_size_m * math.sqrt(2.0),
        }

    def spline_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = torch.cat(
            [layer.spline_coefficients.reshape(-1) for layer in self.edge_kan]
        )
        # Zero-initialized residual splines must have a finite derivative at the
        # identity point; sqrt(mean(x^2)) without epsilon yields 0/0 there.
        magnitude = (coefficients.square().mean() + 1.0e-12).sqrt()
        smoothness = torch.cat(
            [
                (layer.spline_coefficients[..., 2:] - 2.0 * layer.spline_coefficients[..., 1:-1] + layer.spline_coefficients[..., :-2]).reshape(-1)
                for layer in self.edge_kan
            ]
        ).square().mean()
        return magnitude, smoothness

    def function_regularization(self, points: int = 33) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        grid = torch.linspace(-1.0, 1.0, points, device=self.feature_centers.device)
        monotonicity, smoothness, diagnostics = [], [], {}
        for feature_index, name in enumerate(("signed_dz", "edge_slope")):
            probe = torch.zeros(points, 6, device=grid.device)
            probe[:, feature_index] = grid
            values = torch.stack([layer(probe).squeeze(-1) for layer in self.edge_kan])
            first = values[:, 1:] - values[:, :-1]
            second = values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
            monotonicity.append(F.relu(first).mean())
            smoothness.append(second.square().mean())
            diagnostics[f"{name}_violation_fraction"] = (first > 0).to(values.dtype).mean()
        return torch.stack(monotonicity).mean(), torch.stack(smoothness).mean(), diagnostics

    @staticmethod
    def _stack_roll(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rolled, boundaries = zip(
            *[_roll_with_boundary_mask(value, dy, dx) for dy, dx in DIRECTIONS]
        )
        return torch.stack(rolled, dim=1), torch.stack(boundaries, dim=1)

    def _descriptors(
        self,
        physical: Mapping[str, torch.Tensor],
        size: tuple[int, int],
        sensor_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dem_full = physical["dem_valid"]
        dem_fraction = F.adaptive_avg_pool2d(dem_full, size)
        sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, size)
        node_valid = (dem_fraction > 0.5).to(sensor_fraction.dtype) * (sensor_fraction > 0).to(sensor_fraction.dtype)
        z = _masked_pool(physical["physics_elevation"], dem_full, size)
        relative = _masked_pool(physical["z_relative"], dem_full, size)
        relief = _masked_pool(physical["local_relief"], dem_full, size)
        neighbour_z, boundary = self._stack_roll(z)
        neighbour_relative, _ = self._stack_roll(relative)
        neighbour_relief, _ = self._stack_roll(relief)
        distance_values = list(self.neighbour_distances_m)
        distance = z.new_tensor(distance_values).view(1, len(DIRECTIONS), 1, 1, 1)
        signed_dz = (neighbour_z - z.unsqueeze(1)) / distance.clamp_min(1.0e-6)
        edge_slope = signed_dz.abs()
        relative_height = 0.5 * (relative.unsqueeze(1) + neighbour_relative)
        if self.path_barrier_mode == "full_resolution":
            path_barrier, path_valid = path_barrier_proxy(
                physical["dsm_elevation"], dem_full, self.graph_feature_stride,
                physical["z_ground_proxy"], statistic="max", quantile=0.9,
            )
            path_barrier = torch.stack(
                [
                    F.adaptive_avg_pool2d(path_barrier[:, index], size)
                    for index in range(len(DIRECTIONS))
                ],
                dim=1,
            )
            path_valid = torch.stack(
                [
                    F.adaptive_avg_pool2d(path_valid[:, index], size)
                    for index in range(len(DIRECTIONS))
                ],
                dim=1,
            )
        else:
            # A graph edge joins adjacent stride-sized terrain cells. Pool each
            # cell's DSM crest before constructing the eight-neighbour edge to
            # retain the barrier interpretation without 65 full-resolution
            # shifts per forward pass. This introduces no trainable parameters
            # and preserves the checkpoint schema.
            invalid_fill = torch.finfo(physical["dsm_elevation"].dtype).min
            dsm_crest = F.adaptive_max_pool2d(
                torch.where(
                    dem_full > 0.5,
                    physical["dsm_elevation"],
                    torch.full_like(physical["dsm_elevation"], invalid_fill),
                ),
                size,
            )
            dsm_crest = torch.where(
                dem_fraction > 0.0, dsm_crest, torch.zeros_like(dsm_crest)
            )
            ground = _masked_pool(physical["z_ground_proxy"], dem_full, size)
            neighbour_crest, boundary = self._stack_roll(dsm_crest)
            neighbour_ground, _ = self._stack_roll(ground)
            path_barrier = F.relu(
                torch.maximum(dsm_crest.unsqueeze(1), neighbour_crest)
                - torch.maximum(ground.unsqueeze(1), neighbour_ground)
            )
            pooled_valid = (dem_fraction > 0.5).to(dem_fraction.dtype)
            neighbour_valid, _ = self._stack_roll(pooled_valid)
            path_valid = pooled_valid.unsqueeze(1) * neighbour_valid * boundary
        relief_pair = 0.5 * (relief.unsqueeze(1) + neighbour_relief)
        distance_normalized = (
            distance / max(self.graph_pixel_size_m, 1.0)
        ).expand(z.shape[0], -1, -1, size[0], size[1])
        raw = torch.cat(
            (signed_dz, edge_slope, relative_height, path_barrier, relief_pair, distance_normalized),
            dim=2,
        )
        normalized = (
            (raw - self.feature_centers.to(raw.dtype))
            / self.feature_scales.to(raw.dtype)
        )
        descriptor = torch.tanh(normalized)
        valid_edge = node_valid.unsqueeze(1) * self._stack_roll(node_valid)[0] * boundary * path_valid
        return descriptor, raw, valid_edge, dem_fraction, sensor_fraction

    def forward(
        self,
        features: torch.Tensor,
        physical: Mapping[str, torch.Tensor],
        sensor_valid: torch.Tensor,
        observation_confidence: torch.Tensor | None = None,
        *,
        feature_stride: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if feature_stride is not None and int(feature_stride) != self.graph_feature_stride:
            raise ValueError(
                "HydroEdgeKAN feature-stride mismatch: configured "
                f"{self.graph_feature_stride}, received {feature_stride}"
            )
        size = features.shape[-2:]
        descriptor, raw, valid_edge, dem_fraction, sensor_fraction = self._descriptors(
            physical, size, sensor_valid
        )
        if observation_confidence is None:
            observation_confidence = sensor_valid
        confidence = F.adaptive_avg_pool2d(observation_confidence, size)
        neighbour_confidence, _ = self._stack_roll(confidence)
        neighbour_sensor, _ = self._stack_roll(sensor_fraction)
        observation = confidence.unsqueeze(1) * neighbour_confidence * sensor_fraction.unsqueeze(1) * neighbour_sensor
        head_features = features.reshape(
            features.shape[0], self.heads, self.head_channels, *size
        )
        collect_diagnostics = self.training or self.diagnostic_mode
        messages, affinities, gates = [], [], []
        compatibilities, kan_logits, base_outputs, spline_outputs = [], [], [], []
        prior = -(raw * self.prior_scales.view(1, 1, 6, 1, 1)).sum(dim=2, keepdim=True)
        flat_descriptor = descriptor.permute(0, 1, 3, 4, 2).reshape(-1, 6)
        for head, layer in enumerate(self.edge_kan):
            if collect_diagnostics:
                residual, base, spline = layer.forward_with_contributions(
                    flat_descriptor
                )
            else:
                residual = layer(flat_descriptor)
                base = spline = None
            residual = residual.reshape(features.shape[0], len(DIRECTIONS), 1, *size)
            if collect_diagnostics:
                assert base is not None and spline is not None
                base = base.reshape(features.shape[0], len(DIRECTIONS), 1, *size)
                spline = spline.reshape(features.shape[0], len(DIRECTIONS), 1, *size)
            total = prior + self.prior_bias[head] + residual
            topographic_affinity = torch.sigmoid(total)
            affinity = topographic_affinity * valid_edge
            current = head_features[:, head]
            neighbour, _ = self._stack_roll(current)
            latent_difference = neighbour - current.unsqueeze(1)
            pair = torch.stack(
                (
                    latent_difference.abs().mean(2),
                    (current.unsqueeze(1) * neighbour).mean(2),
                ),
                dim=2,
            )
            compatibility = (
                torch.sigmoid(self.latent_compatibility[head](pair.reshape(-1, 2, *size))).reshape(features.shape[0], len(DIRECTIONS), 1, *size)
                if self.latent_compatibility_enabled
                else torch.ones_like(affinity)
            )
            gate = affinity * observation * compatibility
            denominator = gate.sum(1, keepdim=True).clamp_min(1.0e-6)
            weighted = (gate * latent_difference).sum(1, keepdim=True) / denominator
            weighted = torch.tanh(weighted).squeeze(1)
            weighted = weighted + 0.25 * torch.tanh(
                self.message_mixer[head](weighted)
            )
            confidence_weight = gate.sum(1, keepdim=True) / valid_edge.sum(1, keepdim=True).clamp_min(1.0)
            messages.append(
                weighted
                * confidence_weight.squeeze(1)
                * self.gamma[head]
            )
            if collect_diagnostics:
                affinities.append(topographic_affinity)
                gates.append(gate)
                compatibilities.append(compatibility)
                kan_logits.append(residual)
                base_outputs.append(base)
                spline_outputs.append(spline)
        update = torch.cat(messages, dim=1)
        update = update * residual_path_scale(
            features, self.drop_path_probability, self.training
        )
        output = features + update
        coefficient_magnitude, coefficient_smoothness = self.spline_regularization()
        monotonicity, curve_smoothness, monotonicity_diagnostics = self.function_regularization()
        if not collect_diagnostics:
            diagnostics = {
                "kan_coefficient_magnitude": coefficient_magnitude,
                "kan_coefficient_smoothness": coefficient_smoothness,
                "kan_monotonicity": monotonicity,
                "kan_curve_smoothness": curve_smoothness,
                "gamma_mean": self.gamma.mean(),
                "graph_gamma_mean": self.gamma.mean(),
                **monotonicity_diagnostics,
            }
            return output, diagnostics
        gate_stack = torch.cat(gates, dim=2)
        affinity_stack = torch.cat(affinities, dim=2)
        compatibility_stack = torch.cat(compatibilities, dim=2)
        kan_logit_stack = torch.cat(kan_logits, dim=2)
        base_stack = torch.cat(base_outputs, dim=2)
        spline_stack = torch.cat(spline_outputs, dim=2)
        valid_edge_heads = valid_edge.expand_as(affinity_stack)
        graph_input_rms = features.float().square().mean().sqrt()
        graph_update_rms = update.float().square().mean().sqrt()
        spline_rms = spline_stack.float().square().mean().sqrt()
        base_rms = base_stack.float().square().mean().sqrt()
        diagnostics: dict[str, Any] = {
            "gate_mean": gate_stack.mean(),
            "final_graph_gate_mean": gate_stack.mean(),
            "gate_std": gate_stack.float().std(unbiased=False),
            "valid_edge_fraction": valid_edge.mean(),
            "topographic_kan_logit_mean": kan_logit_stack.mean(),
            "topographic_affinity_mean": (
                (affinity_stack * valid_edge_heads).sum()
                / valid_edge_heads.sum().clamp_min(1.0)
            ),
            "static_topographic_affinity_mean": (
                (affinity_stack * valid_edge_heads).sum()
                / valid_edge_heads.sum().clamp_min(1.0)
            ),
            "observation_confidence_mean": observation.mean(),
            "observation_amplitude_mean": observation.mean(),
            "latent_compatibility_mean": compatibility_stack.mean(),
            "kan_coefficient_magnitude": coefficient_magnitude,
            "kan_coefficient_smoothness": coefficient_smoothness,
            "kan_monotonicity": monotonicity,
            "kan_curve_smoothness": curve_smoothness,
            "gamma_mean": self.gamma.mean(),
            "graph_gamma_mean": self.gamma.mean(),
            "gamma_values": self.gamma,
            "graph_input_rms": graph_input_rms,
            "graph_update_rms": graph_update_rms,
            "graph_update_rms_ratio": graph_update_rms / graph_input_rms.clamp_min(1.0e-6),
            "spline_output_rms": spline_rms,
            "base_output_rms": base_rms,
            "spline_base_rms_ratio": spline_rms / base_rms.clamp_min(1.0e-8),
        }
        diagnostics.update(monotonicity_diagnostics)
        if self.diagnostic_mode:
            diagnostics.update({
                "edge_descriptors": descriptor.detach(),
                "raw_edge_descriptors": raw.detach(),
                "static_topographic_affinity": affinity_stack.detach(),
                "valid_edges": valid_edge.detach(),
                "topographic_kan_logits": kan_logit_stack.detach(),
                "latent_compatibility": compatibility_stack.detach(),
            })
        return output, diagnostics
