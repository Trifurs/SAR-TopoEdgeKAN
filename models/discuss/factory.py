"""Optional model controls that reuse the common production training loop."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from compare.common.comparison_factory import BUILDERS, build_comparison_model
from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec
from models.discuss.edge_controls import ControlledEdgeGraph
from models.discuss.inputs import assemble_observations, observation_columns, shared_observation_channels
from models.sar_topo_edge_kan import SARTopoEdgeKAN, SARTopoEdgeKANHeads
from models.task_head import KnownRangeHydrostaticPrior


def make_prior(config: Mapping[str, Any]) -> KnownRangeHydrostaticPrior:
    return KnownRangeHydrostaticPrior(
        boundary_quantile=float(config.get("hydrostatic_boundary_quantile", 0.25)),
        minimum_depth_m=float(config.get("hydrostatic_minimum_depth_m", 0.10)),
    )


class FeatureOnlyHead(nn.Module):
    def forward(self, features, flood_range):
        return features


def expand_input_convolution(network: nn.Module, channels: int) -> None:
    """Preserve native ten-channel weights and zero-initialize added observations."""

    matches = [
        (name, layer) for name, layer in network.named_modules()
        if isinstance(layer, nn.Conv2d) and layer.in_channels == 10
    ]
    if not matches:
        raise ValueError("No input convolution found in discussion comparator")
    for name, layer in matches:
        # A residual stem also has an input-channel skip projection.
        if layer.groups != 1:
            raise ValueError("Discussion comparators require the native ten-channel stem")
        with torch.random.fork_rng(devices=[]):
            expanded = nn.Conv2d(
                channels, layer.out_channels, layer.kernel_size, layer.stride,
                layer.padding, layer.dilation, layer.groups, layer.bias is not None,
                layer.padding_mode,
            )
        with torch.no_grad():
            expanded.weight.zero_()
            # Normalized SAR/terrain are first; known range is always last.
            expanded.weight[:, :9].copy_(layer.weight[:, :9])
            expanded.weight[:, -1:].copy_(layer.weight[:, 9:10])
            if layer.bias is not None:
                expanded.bias.copy_(layer.bias)
        parent_name, _, child_name = name.rpartition(".")
        parent = network.get_submodule(parent_name) if parent_name else network
        setattr(parent, child_name, expanded)


class MatchedComparator(nn.Module):
    """Give an existing comparator all observations consumed by the full model."""

    known_flood_range_enabled = True
    depth_output_semantics = "conditional_positive"

    def __init__(self, config: Mapping[str, Any], band_spec, raw_names) -> None:
        super().__init__()
        self.discussion_model_config = deepcopy(dict(config["model"]))
        self.discussion_raw_names = tuple(raw_names)
        self.band_spec = band_spec
        discussion = self.discussion_model_config["discuss"]
        self.head_mode = str(discussion.get("head_mode", "native"))
        channels = shared_observation_channels(band_spec, self.discussion_raw_names)
        if channels != int(discussion["shared_input_channels"]):
            raise ValueError(f"Configured observation count differs from audited inputs: {channels}")
        self.input_columns = observation_columns(band_spec, self.discussion_raw_names)
        native = deepcopy(dict(config))
        native["model"]["name"] = str(discussion["architecture"])
        native["model"]["input_schema"] = "s1_terrain_range"
        native["model"]["input_channels"] = 10
        self.network = build_comparison_model(native)
        expand_input_convolution(self.network, channels)
        self.heads = None
        self.hydrostatic_prior = None
        if self.head_mode == "native":
            nn.init.constant_(
                self.network.head.depth.bias,
                float(self.discussion_model_config.get("depth_initialization_bias", 0.0)),
            )
        elif self.head_mode in {"shared_direct", "shared_prior"}:
            width = self.network.head.depth.in_channels
            self.network.head = FeatureOnlyHead()
            prior_enabled = self.head_mode == "shared_prior"
            with torch.random.fork_rng(devices=[]):
                self.heads = SARTopoEdgeKANHeads(
                    width, int(self.discussion_model_config["group_norm_groups"]),
                    epsilon=float(self.discussion_model_config["depth_epsilon"]),
                    depth_initialization_bias=float(self.discussion_model_config["depth_initialization_bias"]),
                    known_range_hydrostatic_enabled=prior_enabled,
                    hydrostatic_residual_max_log_abs=float(self.discussion_model_config["hydrostatic_residual_max_log_abs"]),
                )
            self.hydrostatic_prior = make_prior(self.discussion_model_config) if prior_enabled else None
        else:
            raise ValueError(f"Unknown discussion head mode: {self.head_mode}")

    def component_flags(self):
        return {
            "known_flood_range_enabled": True,
            "known_range_hydrostatic_enabled": self.hydrostatic_prior is not None,
            "topographic_affinity_enabled": False,
        }

    def graph_identity(self):
        return None

    def forward(self, inputs: Mapping[str, Any]):
        packed = assemble_observations(inputs, self.discussion_raw_names)
        flood_range = inputs["valid_depth_mask"]
        value = self.network(packed, flood_range)
        if self.heads is None:
            outputs = value
        else:
            prior = None
            if self.hydrostatic_prior is not None:
                index = self.discussion_raw_names.index("elevation_m_DSM")
                prior, _ = self.hydrostatic_prior(
                    inputs["terrain_raw"][:, index:index + 1], flood_range, inputs["dem_valid"]
                )
            outputs = self.heads(value, hydrostatic_prior=prior)
            outputs["depth"] = outputs["conditional_depth"] * (flood_range > 0.5)
        outputs.update({"graph_diagnostics": {}, "component_flags": self.component_flags()})
        return outputs


class TerrainOnlyModel(SARTopoEdgeKAN):
    """Remove the SAR encoder and use terrain features as the decoder backbone."""

    def __init__(self, model_config, band_spec, raw_names, input_spec):
        super().__init__(model_config, band_spec, raw_names, input_spec)
        self.sar_encoder = None
        self.fusion = None
        self.decoder.change_projection = None
        self.decoder.change_injection_scale = 0.0

    def forward(self, inputs: Mapping[str, Any]):
        # No SAR, reliability or S1 validity tensor is read here. The
        # common evaluator still uses the same canonical supervision domain.
        terrain, physical = self.terrain(inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"])
        bottleneck = self.context(terrain[-1])
        if self.topographic_affinity_enabled:
            self._last_graph_feature_shape = tuple(bottleneck.shape[-2:])
            bottleneck, graph_diagnostics = self.graph(
                bottleneck, physical, inputs["dem_valid"], inputs["dem_valid"], feature_stride=8
            )
        else:
            graph_diagnostics = {}
        decoded, gates = self.decoder(
            bottleneck, terrain, terrain, physical["dem_valid_fractions"], inputs["dem_valid"], None
        )
        prior = surface = None
        if self.hydrostatic_prior is not None:
            prior, surface = self.hydrostatic_prior(
                physical["dsm_elevation"], inputs["valid_depth_mask"], inputs["dem_valid"]
            )
        outputs = self.heads(decoded, hydrostatic_prior=prior)
        outputs["depth"] = outputs["conditional_depth"] * (inputs["valid_depth_mask"] > 0.5)
        outputs.update({
            "physical_features": physical,
            "graph_diagnostics": graph_diagnostics, "decoder_gates": gates,
            "sar_diagnostics": {}, "fusion_diagnostics": {},
            "component_flags": self.component_flags(), "hydrostatic_water_surface": surface,
        })
        return outputs

    def component_flags(self):
        return {**super().component_flags(), "joint_temporal_change_encoder": False, "sar_encoder_enabled": False}


class HydrostaticPriorOnly(nn.Module):
    """Parameter-free reference with no checkpoint or random-seed training stage."""

    known_flood_range_enabled = True
    depth_output_semantics = "conditional_positive"

    def __init__(self, config, raw_names):
        super().__init__()
        self.discussion_model_config = deepcopy(dict(config["model"]))
        self.discussion_raw_names = tuple(raw_names)
        self.hydrostatic_prior = make_prior(self.discussion_model_config)

    def component_flags(self):
        return {"known_flood_range_enabled": True, "known_range_hydrostatic_enabled": True, "topographic_affinity_enabled": False}

    def graph_identity(self):
        return None

    def forward(self, inputs):
        index = self.discussion_raw_names.index("elevation_m_DSM")
        prior, surface = self.hydrostatic_prior(
            inputs["terrain_raw"][:, index:index + 1], inputs["valid_depth_mask"], inputs["dem_valid"]
        )
        depth = prior * (inputs["valid_depth_mask"] > 0.5)
        return {
            "depth": depth, "conditional_depth": prior,
            "hydrostatic_prior_depth": prior, "hydrostatic_water_surface": surface,
            "component_flags": self.component_flags(),
        }


def build_discussion_model(config: Mapping[str, Any]) -> nn.Module:
    model_config = config["model"]
    discussion = model_config["discuss"]
    architecture = str(discussion["architecture"])
    contract = DatasetContract.load(config["dataset"]["contract"])
    spec = ModelInputSpec.from_config(config)
    bands = resolve_band_spec(config, contract)
    raw_names = tuple(contract.group("terrain")["band_descriptions"])
    if architecture in BUILDERS:
        return MatchedComparator(config, bands, raw_names)
    if architecture == "hydrostatic_prior_only":
        return HydrostaticPriorOnly(config, raw_names)
    if architecture != "sar_topo_edge_kan":
        raise ValueError(f"Unsupported discussion architecture: {architecture}")
    cls = TerrainOnlyModel if discussion.get("observation_mode", "full") == "terrain_only" else SARTopoEdgeKAN
    model = cls(model_config, bands, raw_names, spec)
    mode = str(discussion.get("edge_function", "kan"))
    if not model.topographic_affinity_enabled:
        model.graph = None  # physically remove inactive tensors for size/efficiency
    elif mode != "kan":
        model.graph = ControlledEdgeGraph(model.graph, model_config, mode)
    model.discussion_model_config = deepcopy(dict(model_config))
    model.discussion_raw_names = raw_names
    return model
