"""Registry for the production SAR-TopoEdgeKAN model."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from models.sar_topo_edge_kan import (
    PAPER_MODEL_EXPANSION,
    PAPER_MODEL_NAME,
    build_sar_topo_edge_kan,
)


MODEL_NAME = "sar_topo_edge_kan"
MODEL_DISPLAY_NAME = PAPER_MODEL_NAME
MODEL_FORMAL_NAME = PAPER_MODEL_EXPANSION


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build the production SAR-and-terrain SAR-TopoEdgeKAN model."""

    if bool(config["model"].get("discuss", {}).get("enabled", False)):
        from models.discuss.factory import build_discussion_model

        return build_discussion_model(config)
    name = str(config["model"]["name"])
    if name != MODEL_NAME:
        raise KeyError(f"Unsupported model {name!r}; expected {MODEL_NAME!r}")
    return build_sar_topo_edge_kan(config)
