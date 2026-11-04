"""Production SAR-TopoEdgeKAN SAR-and-terrain model.

Paper name: SAR-based Topographic Edge Kolmogorov-Arnold Network.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from datasets.band_selection import BandSpec, resolve_band_spec
from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec
from models.hydro_edge_kan import HydroEdgeKAN
from models.s1_hydrology_backbone import (
    HydrologyContext,
    JointSARHydrologyEncoder,
    S1FeatureAlignment,
)
from models.sar_hydro_decoder import SARHydroDecoder
from models.task_head import (
    KnownRangeHydrostaticPrior,
    TaskHead,
)
from models.terrain_features import TerrainFeaturePyramid


PAPER_MODEL_NAME = "SAR-TopoEdgeKAN"
PAPER_MODEL_EXPANSION = "SAR-based Topographic Edge Kolmogorov-Arnold Network"


REQUIRED_INPUTS = {
    "s1_t1",
    "s1_t2",
    "s1_change",
    "terrain",
    "terrain_raw",
    "s1_valid",
    "s1_event_support",
    "dem_valid",
}
FORBIDDEN_INPUTS = {
    "label",
    "masks",
    "flood_mask",
    "split",
    "sample_origin",
}


class SARTopoEdgeKANHeads(nn.Module):
    """Conditional depth or bounded multiplicative hydrostatic-prior residual."""

    def __init__(
        self, channels: int, groups: int, *, epsilon: float,
        depth_initialization_bias: float,
        known_range_hydrostatic_enabled: bool = False,
        hydrostatic_residual_max_log_abs: float = 2.0,
    ) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError("depth epsilon must be positive")
        self.epsilon = float(epsilon)
        self.known_range_hydrostatic_enabled = bool(known_range_hydrostatic_enabled)
        self.hydrostatic_residual_max_log_abs = float(hydrostatic_residual_max_log_abs)
        if self.hydrostatic_residual_max_log_abs <= 0.0:
            raise ValueError("hydrostatic residual log bound must be positive")
        self.depth_output_semantics = "conditional_positive"
        self.depth_head = TaskHead(
            channels, groups,
            input_channels=channels + int(self.known_range_hydrostatic_enabled),
        )
        depth_final = self.depth_head.trunk[-1]
        assert isinstance(depth_final, nn.Conv2d)
        if self.known_range_hydrostatic_enabled:
            nn.init.zeros_(depth_final.weight)
            nn.init.zeros_(depth_final.bias)
        else:
            nn.init.constant_(depth_final.bias, float(depth_initialization_bias))

    def forward(
        self, features: torch.Tensor, hydrostatic_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.known_range_hydrostatic_enabled:
            if hydrostatic_prior is None:
                raise ValueError("known-range hydrostatic head received no depth prior")
            if hydrostatic_prior.shape != features[:, :1].shape:
                raise ValueError("hydrostatic prior must match the depth prediction grid")
            depth_features = torch.cat(
                (features, torch.log1p(hydrostatic_prior.clamp_min(0.0))), dim=1
            )
        else:
            depth_features = features
        depth_logit = self.depth_head(depth_features)
        hydrostatic_log_correction = None
        if self.known_range_hydrostatic_enabled:
            bound = self.hydrostatic_residual_max_log_abs
            hydrostatic_log_correction = bound * torch.tanh(depth_logit / bound)
            depth = hydrostatic_prior * torch.exp(hydrostatic_log_correction)
        else:
            depth = F.softplus(depth_logit) + self.epsilon
        return {
            "conditional_depth": depth, "depth": depth,
            "hydrostatic_prior_depth": hydrostatic_prior,
            "hydrostatic_log_correction": hydrostatic_log_correction,
        }


class SARTopoEdgeKAN(nn.Module):
    """SAR-TopoEdgeKAN: a prior-aware SAR/topography depth estimator.

    The named paper modules are TCSE (temporal-change SAR encoder), TPP
    (topographic-prior pyramid), TAE-KAN (topographic-affinity Edge-KAN), DGD
    (dual-gated decoder), and CDH (conditional-depth head).
    """

    def __init__(
        self,
        model_config: Mapping[str, Any],
        band_spec: BandSpec,
        raw_terrain_names: tuple[str, ...],
        input_spec: ModelInputSpec,
    ) -> None:
        super().__init__()
        if not input_spec.is_s1_only:
            raise ValueError("SARTopoEdgeKAN requires the S1-only input contract")
        self.input_spec = input_spec
        self.band_spec = band_spec
        self.known_flood_range_enabled = bool(
            model_config.get("known_flood_range_enabled", False)
        )
        flood_support = str(model_config.get("flood_support", "none"))
        if self.known_flood_range_enabled and flood_support != "valid_depth_mask":
            raise ValueError(
                "known-range SAR-TopoEdgeKAN requires flood_support=valid_depth_mask"
            )
        self.known_range_hydrostatic_enabled = bool(
            model_config.get("known_range_hydrostatic_enabled", False)
        )
        if (
            self.known_range_hydrostatic_enabled
            and not self.known_flood_range_enabled
        ):
            raise ValueError(
                "known-range hydrostatic prior requires known_flood_range_enabled"
            )
        self.topographic_affinity_enabled = bool(
            model_config.get("topographic_affinity_enabled", True)
        )
        # LCA is an internal TAE-KAN mechanism, so its runtime state cannot be
        # enabled independently when the parent graph path is disabled.  This
        # keeps component metadata faithful to the executed computation graph
        # while preserving the state-dictionary schema for every ablation.
        self.latent_compatibility_enabled = self.topographic_affinity_enabled
        channels = [int(value) for value in model_config["channels"]]
        if len(channels) != 4 or any(value <= 0 for value in channels):
            raise ValueError("SARTopoEdgeKAN requires four positive encoder scales")
        dropout = float(model_config["dropout"])
        groups = int(model_config["group_norm_groups"])
        block_kind = str(model_config["residual_block"])
        sar_block_kind = str(model_config.get("sar_residual_block", block_kind))
        terrain_block_kind = str(
            model_config.get("terrain_residual_block", block_kind)
        )
        decoder_block_kind = str(
            model_config.get("decoder_residual_block", block_kind)
        )
        encoder_options: dict[str, Any] = {
            "dropout": dropout,
            "groups": groups,
            "block_kind": sar_block_kind,
            "known_flood_range_enabled": self.known_flood_range_enabled,
        }
        self.sar_encoder = JointSARHydrologyEncoder(
            band_spec.channels("s1_t1"),
            band_spec.channels("s1_change"),
            channels,
            **encoder_options,
        )
        self.terrain = TerrainFeaturePyramid(
            band_spec.channels("terrain"),
            channels,
            dropout,
            groups,
            float(model_config["terrain_pixel_size_m"]),
            raw_terrain_names,
            int(model_config["ground_proxy_kernel_size"]),
            str(model_config["physics_elevation"]),
            terrain_block_kind,
            model_config.get("topographic_context_scales_m", ()),
        )
        self.fusion = S1FeatureAlignment(channels, groups=groups)
        self.context = HydrologyContext(
            channels[-1],
            groups,
            dropout=0.05,
            residual_init=float(model_config.get("context_residual_init", 0.15)),
            residual_max=float(model_config.get("context_residual_max", 0.50)),
        )
        graph_stride = int(model_config["graph_feature_stride"])
        if graph_stride != 8:
            raise ValueError("SARTopoEdgeKAN applies graph reasoning at encoder stride 8")
        self.graph = None
        if self.topographic_affinity_enabled:
            # Edge-module initialization cannot shift the shared backbone RNG.
            with torch.random.fork_rng(devices=[]):
                self.graph = HydroEdgeKAN(
                    channels[-1],
                    heads=int(model_config["graph_heads"]),
                    grid_size=int(model_config["kan_grid_size"]),
                    spline_order=int(model_config["kan_spline_order"]),
                    graph_feature_stride=graph_stride,
                    terrain_pixel_size_m=float(model_config["terrain_pixel_size_m"]),
                    feature_centers=model_config["graph_feature_centers"],
                    feature_scales=model_config["graph_feature_scales"],
                    gamma_init_effective=float(model_config["kan_gamma_init_effective"]),
                    gamma_max=float(model_config["kan_gamma_max"]),
                    diagnostics_enabled=bool(model_config.get("diagnostics_enabled", False)),
                )
        widths = [int(value) for value in model_config["decoder_widths"]]
        self.decoder = SARHydroDecoder(
            channels,
            dropout,
            groups,
            decoder_block_kind,
            widths,
        )
        self.heads = SARTopoEdgeKANHeads(
            widths[-1], groups,
            epsilon=float(model_config["depth_epsilon"]),
            depth_initialization_bias=float(model_config["depth_initialization_bias"]),
            known_range_hydrostatic_enabled=self.known_range_hydrostatic_enabled,
            hydrostatic_residual_max_log_abs=float(model_config["hydrostatic_residual_max_log_abs"]),
        )
        self.hydrostatic_prior = (
            KnownRangeHydrostaticPrior(
                boundary_quantile=float(
                    model_config.get("hydrostatic_boundary_quantile", 0.25)
                ),
                minimum_depth_m=float(
                    model_config.get("hydrostatic_minimum_depth_m", 0.10)
                ),
            )
            if self.known_range_hydrostatic_enabled
            else None
        )
        self._last_graph_feature_shape: tuple[int, int] | None = None

    def component_flags(self) -> dict[str, bool]:
        """Return the named-paper component status for logs and ablation reports."""

        return {
            "topographic_affinity_enabled": self.topographic_affinity_enabled,
            "latent_compatibility_enabled": self.latent_compatibility_enabled,
            "known_flood_range_enabled": self.known_flood_range_enabled,
            "known_range_hydrostatic_enabled": self.known_range_hydrostatic_enabled,
            "joint_temporal_change_encoder": True,
        }

    def graph_identity(self) -> dict[str, Any] | None:
        if not self.topographic_affinity_enabled:
            return None
        return self.graph.graph_identity(self._last_graph_feature_shape)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        forbidden = FORBIDDEN_INPUTS.intersection(inputs)
        if not self.known_flood_range_enabled and "valid_depth_mask" in inputs:
            forbidden.add("valid_depth_mask")
        if forbidden:
            raise ValueError(f"Forbidden non-S1 inputs: {sorted(forbidden)}")
        missing = REQUIRED_INPUTS.difference(inputs)
        if self.known_flood_range_enabled and "valid_depth_mask" not in inputs:
            missing.add("valid_depth_mask")
        if missing:
            raise KeyError(f"Missing flood-depth inputs: {sorted(missing)}")
        branch_validity = dict(inputs.get("branch_validity", {}))
        if self.known_flood_range_enabled:
            flood_range = inputs["valid_depth_mask"].to(inputs["s1_t1"].dtype)
            if flood_range.ndim != 4 or flood_range.shape[1] != 1:
                raise ValueError(
                    "valid_depth_mask must have shape [batch, 1, height, width]"
                )
        else:
            flood_range = None
        sar, sar_diagnostics = self.sar_encoder(
            inputs["s1_t1"],
            inputs["s1_t2"],
            inputs["s1_change"],
            inputs["s1_valid"],
            flood_range,
            branch_validity,
        )
        terrain, physical = self.terrain(
            inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"]
        )
        fused, fusion_diagnostics = self.fusion(
            sar, inputs["s1_event_support"]
        )
        bottleneck = self.context(fused[-1])
        if self.topographic_affinity_enabled:
            self._last_graph_feature_shape = tuple(
                int(value) for value in bottleneck.shape[-2:]
            )
            bottleneck, graph_diagnostics = self.graph(
                bottleneck,
                physical,
                inputs["s1_event_support"],
                sar_diagnostics["quality_gates"][-1],
                feature_stride=8,
            )
        else:
            self._last_graph_feature_shape = None
            graph_diagnostics = {}
        decoded, decoder_gates = self.decoder(
            bottleneck,
            fused,
            terrain,
            physical["dem_valid_fractions"],
            inputs["s1_event_support"],
            sar_diagnostics["change_evidence"],
        )
        hydrostatic_prior = None
        hydrostatic_water_surface = None
        if self.hydrostatic_prior is not None:
            assert flood_range is not None
            hydrostatic_prior, hydrostatic_water_surface = self.hydrostatic_prior(
                physical["dsm_elevation"], flood_range, inputs["dem_valid"]
            )
        outputs = self.heads(
            decoded,
            hydrostatic_prior=hydrostatic_prior,
        )
        if flood_range is not None:
            range_mask = (flood_range > 0.5).to(outputs["conditional_depth"].dtype)
            range_constrained_depth = outputs["conditional_depth"] * range_mask
            outputs["depth"] = range_constrained_depth
        outputs.update(
            {
                "decoder_gates": decoder_gates,
                "fusion_diagnostics": fusion_diagnostics,
                "sar_diagnostics": sar_diagnostics,
                "graph_diagnostics": graph_diagnostics,
                "physical_features": physical,
                "component_flags": self.component_flags(),
                "hydrostatic_water_surface": hydrostatic_water_surface,
            }
        )
        return outputs


def build_sar_topo_edge_kan(config: Mapping[str, Any]) -> SARTopoEdgeKAN:
    """Build SAR-TopoEdgeKAN from a resolved configuration."""

    if "model" not in config:
        raise ValueError("A resolved model configuration is required")
    input_spec = ModelInputSpec.from_config(config)
    if not input_spec.is_s1_only:
        raise ValueError("The production model requires S1-only inputs")
    contract = DatasetContract.load(config["dataset"]["contract"])
    band_spec = resolve_band_spec(config, contract)
    raw_names = tuple(str(value) for value in contract.group("terrain")["band_descriptions"])
    return SARTopoEdgeKAN(config["model"], band_spec, raw_names, input_spec)
