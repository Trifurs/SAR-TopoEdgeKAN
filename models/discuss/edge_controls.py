"""Matched edge-function controls on the existing topographic graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from models.hydro_edge_kan import HydroEdgeKAN


class EdgeFunction(nn.Module):
    """Zero-initialized MLP/linear correction or a parameter-free zero map."""

    def __init__(self, mode: str, hidden: int = 7) -> None:
        super().__init__()
        if mode == "mlp":
            self.network = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        elif mode == "linear":
            self.network = nn.Linear(6, 1)
            nn.init.zeros_(self.network.weight)
            nn.init.zeros_(self.network.bias)
        elif mode == "fixed":
            self.network = None
        else:
            raise ValueError(f"Unsupported discussion edge function: {mode}")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.new_zeros((*value.shape[:-1], 1)) if self.network is None else self.network(value)

    def forward_with_contributions(self, value: torch.Tensor):
        output = self(value)
        return output, output, torch.zeros_like(output)


class ControlledEdgeGraph(HydroEdgeKAN):
    """Change only the edge function; keep graph topology and message mechanics."""

    def __init__(self, source: HydroEdgeKAN, config: Mapping[str, Any], mode: str) -> None:
        # Construction is isolated so replacing the graph cannot advance the
        # RNG used by shared model initialization or subsequent training.
        with torch.random.fork_rng(devices=[]):
            super().__init__(
                source.channels,
                heads=source.heads,
                grid_size=int(config["kan_grid_size"]),
                spline_order=int(config["kan_spline_order"]),
                graph_feature_stride=source.graph_feature_stride,
                terrain_pixel_size_m=float(config["terrain_pixel_size_m"]),
                feature_centers=config["graph_feature_centers"],
                feature_scales=config["graph_feature_scales"],
                gamma_init_effective=float(config["kan_gamma_init_effective"]),
                gamma_max=source.gamma_max,
                diagnostics_enabled=source.diagnostic_mode,
            )
            self.load_state_dict(source.state_dict(), strict=True)
            self.edge_kan = nn.ModuleList(
                EdgeFunction(mode, int(config.get("discuss", {}).get("mlp_hidden", 7)))
                for _ in range(source.heads)
            )
        self.edge_function_mode = mode
        if mode == "fixed":
            # The topographic affinity is genuinely fixed. Shared observation
            # gates, latent compatibility, message mixers and gamma still learn.
            for name in ("prior_raw_scales", "prior_bias"):
                value = getattr(self, name).detach().clone()
                delattr(self, name)
                self.register_buffer(name, value)

    def spline_regularization(self):
        zero = self.feature_centers.sum() * 0.0
        return zero, zero

    def graph_identity(self, feature_shape=None):
        return {**super().graph_identity(feature_shape), "edge_function": self.edge_function_mode}
