from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from datasets.train_depth_calibration import TrainDepthScan
from models.sar_topo_edge_kan import SARTopoEdgeKANHeads
from models.task_head import KnownRangeHydrostaticPrior
from tools.train_sar_topo_edge_kan import (
    apply_train_only_calibration,
    prepare_train_only_calibration,
)


def test_known_range_hydrostatic_prior_uses_inner_boundary_quantile() -> None:
    module = KnownRangeHydrostaticPrior(
        boundary_quantile=0.25, minimum_depth_m=0.10
    )
    elevation = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    support = torch.zeros_like(elevation)
    support[..., 1:4, 1:4] = 1.0
    prior, water_surface = module(elevation, support, torch.ones_like(support))
    boundary_values = torch.tensor((6, 7, 8, 11, 13, 16, 17, 18), dtype=torch.float32)
    expected_surface = torch.quantile(boundary_values, 0.25) + 0.10
    assert torch.allclose(
        water_surface[support > 0.5],
        expected_surface.expand_as(water_surface)[support > 0.5],
    )
    assert torch.count_nonzero(water_surface[support <= 0.5]) == 0
    expected = (expected_surface - elevation).clamp_min(0.10)
    expected = torch.where(support > 0.5, expected, torch.full_like(expected, 0.10))
    assert torch.allclose(prior, expected)


def test_known_range_hydrostatic_prior_separates_connected_components() -> None:
    module = KnownRangeHydrostaticPrior(
        boundary_quantile=0.25,
        minimum_depth_m=0.10,
    )
    elevation = torch.zeros(1, 1, 7, 7)
    elevation[..., 1:3, 1:3] = torch.tensor(((10.0, 11.0), (12.0, 13.0)))
    elevation[..., 4:6, 4:6] = torch.tensor(((100.0, 101.0), (102.0, 103.0)))
    support = torch.zeros_like(elevation)
    support[..., 1:3, 1:3] = 1.0
    support[..., 4:6, 4:6] = 1.0
    prior, water_surface = module(elevation, support, torch.ones_like(support))
    first_surface = torch.quantile(
        torch.tensor((10.0, 11.0, 12.0, 13.0)), 0.25
    ) + 0.10
    second_surface = torch.quantile(
        torch.tensor((100.0, 101.0, 102.0, 103.0)), 0.25
    ) + 0.10
    assert torch.all(water_surface[..., 1:3, 1:3] == first_surface)
    assert torch.all(water_surface[..., 4:6, 4:6] == second_surface)
    assert torch.all(prior[support > 0.5] >= 0.10)


def test_known_range_head_is_prior_identity_with_trainable_residual() -> None:
    heads = SARTopoEdgeKANHeads(
        8,
        4,
        epsilon=0.001,
        depth_initialization_bias=0.0,
        known_range_hydrostatic_enabled=True,
        hydrostatic_residual_max_log_abs=2.0,
    )
    features = torch.randn(2, 8, 16, 16, requires_grad=True)
    prior = torch.rand(2, 1, 16, 16) + 0.1
    outputs = heads(features, hydrostatic_prior=prior)
    assert torch.equal(outputs["conditional_depth"], prior)
    assert torch.count_nonzero(outputs["hydrostatic_log_correction"]) == 0
    (outputs["conditional_depth"] - 1.0).abs().mean().backward()
    final = heads.depth_head.trunk[-1]
    assert final.weight.grad is not None
    assert torch.count_nonzero(final.weight.grad) > 0


def test_train_depth_initialization_uses_only_train_positive_median(production_config, monkeypatch):
    import tools.train_sar_topo_edge_kan as training

    config = deepcopy(production_config)
    config["model"]["depth_initialization_mode"] = "train_positive_median"
    dataset = SimpleNamespace(contract=object(), band_spec=object(), minimum_event_band_fraction=1.0)
    scan = TrainDepthScan(np.array([0.1, 0.3, 0.5], dtype=np.float32), 1, 3)
    observed = []
    def collect(contract, bands, **kwargs):
        observed.append((contract, bands, kwargs))
        return scan
    monkeypatch.setattr(training, "collect_canonical_train_depths", collect)
    payload = prepare_train_only_calibration(config, dataset)
    assert apply_train_only_calibration(config, payload) is None
    assert len(observed) == 1
    assert observed[0][:2] == (dataset.contract, dataset.band_spec)
    assert observed[0][2] == {"minimum_event_band_fraction": 1.0}
    bias = config["model"]["depth_initialization_bias"]
    assert torch.nn.functional.softplus(torch.tensor(bias)).item() == pytest.approx(0.3)
    assert config["model"]["depth_initialization_source"] == "canonical_train_positive_depth_median"


def test_configured_depth_initialization_does_not_scan_targets(production_config, monkeypatch):
    import tools.train_sar_topo_edge_kan as training

    config = deepcopy(production_config)
    config["model"]["depth_initialization_mode"] = "configured"
    def forbidden(*args, **kwargs):
        raise AssertionError("configured initialization must not scan training data")
    monkeypatch.setattr(training, "collect_canonical_train_depths", forbidden)
    assert prepare_train_only_calibration(config, SimpleNamespace()) == {}
