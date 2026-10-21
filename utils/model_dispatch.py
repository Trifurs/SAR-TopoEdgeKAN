"""Resolve a model family from the model-owned XML configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from compare.common.comparison_factory import BUILDERS
from utils.run_paths import RuntimeConfigError


def configured_model_kind(config: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(kind, identifier)`` for SAR-TopoEdgeKAN or a comparison method."""

    model = config.get("model")
    if isinstance(model, Mapping):
        identifier = str(model.get("name", ""))
        if bool(model.get("discuss", {}).get("enabled", False)):
            # Discussion controls share one trainer, optimizer and evaluator,
            # including adapters around the five comparator architectures.
            return "sar_topo_edge_kan", identifier
        if identifier == "sar_topo_edge_kan":
            return "sar_topo_edge_kan", identifier
        if identifier in BUILDERS:
            return "learned_comparator", identifier
        raise RuntimeConfigError(f"unsupported configured learned model: {identifier!r}")
    compare = config.get("compare")
    if isinstance(compare, Mapping):
        identifier = str(compare.get("method", ""))
        if not identifier:
            raise RuntimeConfigError("compare.method must identify a traditional model")
        return "traditional", identifier
    raise RuntimeConfigError("configuration must contain either <model> or <compare>")
