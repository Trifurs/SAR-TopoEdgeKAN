from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import load_config
from utils.experiment_catalog import deep_learning_config_paths
from utils.graph_metadata import resolved_graph_identity
from utils.registry import build_model


CONFIGS = {
    "wo_tae_kan": "sar_topo_edge_kan_wo_tae_kan.xml",
}

FLAG_BY_MODULE = {
    "TAE-KAN": "topographic_affinity_enabled",
}


def _config(variant_id: str) -> dict:
    return load_config(Path("configs/ablation") / CONFIGS[variant_id])


def test_ablation_config_builds_the_declared_intervention() -> None:
    config = _config("wo_tae_kan")
    removed = set(config["ablation"]["removed_modules"])
    assert config["ablation"]["variant_id"] == "wo_tae_kan"
    model = build_model(config)
    flags = model.component_flags()
    for module, flag in FLAG_BY_MODULE.items():
        assert flags[flag] is (module not in removed)
    # LCA is not an independent factor: its effective state follows TAE-KAN.
    assert flags["latent_compatibility_enabled"] is ("TAE-KAN" not in removed)
    assert (resolved_graph_identity(config) is None) is ("TAE-KAN" in removed)


def test_ablation_design_contains_only_edge_kan_removal() -> None:
    observed = {
        frozenset(_config(variant)["ablation"]["removed_modules"])
        for variant in CONFIGS
    }
    assert observed == {frozenset(("TAE-KAN",))}


def test_disabled_top_level_paths_are_frozen() -> None:
    no_tae = build_model(_config("wo_tae_kan"))
    assert not any(parameter.requires_grad for parameter in no_tae.graph.parameters())


def test_production_tae_uses_a_bounded_identity_initialized_residual() -> None:
    config = load_config(Path("configs/sar_topo_edge_kan.xml"))
    model = build_model(config)
    assert config["model"]["kan_gamma_parameterization"] == "signed"
    assert config["model"]["kan_gamma_init_effective"] == 0.0
    assert config["model"]["kan_gamma_max"] == 0.04
    assert model.graph.gamma.tolist() == pytest.approx([0.0, 0.0])


def test_full_model_is_catalogued_once_and_has_no_redundant_ablation_config() -> None:
    paths = deep_learning_config_paths(include_ablations=True)
    assert sum(path.name == "sar_topo_edge_kan.xml" for path in paths) == 1
    assert len(tuple(Path("configs/ablation").glob("*.xml"))) == 1


def test_every_learned_experiment_uses_the_same_known_flood_range() -> None:
    for path in deep_learning_config_paths(include_ablations=True):
        config = load_config(path)
        assert config["model"]["flood_support"] == "valid_depth_mask", path
    production = load_config(Path("configs/sar_topo_edge_kan.xml"))
    assert production["model"]["known_flood_range_enabled"] is True
    assert production["model"]["known_range_hydrostatic_enabled"] is True
