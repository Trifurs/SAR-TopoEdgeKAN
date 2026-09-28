from __future__ import annotations

from copy import deepcopy

import torch

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from tools.train_sar_topo_edge_kan import apply_train_only_calibration, prepare_train_only_calibration
from models.evidence_calibration import GlobalEvidenceCalibration
from models.task_head import ContextualDepthRangeCalibration, KnownRangeHydrostaticPrior
from models.sar_topo_edge_kan import SARTopoEdgeKANHeads


def test_train_calibration_builds_a_frozen_weight_curve(production_config) -> None:
    config = deepcopy(production_config)
    config["loss"]["soft_depth_balance"] = True
    contract = DatasetContract.load(config["dataset"]["contract"])
    dataset = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        "train",
        band_spec=resolve_band_spec(config, contract),
        input_spec=ModelInputSpec.from_config(config),
        minimum_event_band_fraction=float(
            config["dataset"]["minimum_event_band_fraction"]
        ),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    payload = prepare_train_only_calibration(config, dataset)
    balance = apply_train_only_calibration(config, payload)
    assert balance is not None
    assert balance.train_positive_pixels > 0
    assert config["model"]["depth_initialization_bias"] < 0


def test_grouped_global_evidence_matches_dense_concatenation() -> None:
    module = GlobalEvidenceCalibration(7, hidden_channels=8, dropout=0.0).eval()
    groups = (torch.randn(3, 2, 9, 11), torch.randn(3, 5, 9, 11))
    dense_scale, dense_bias = module(torch.cat(groups, dim=1))
    grouped_scale, grouped_bias = module(groups)
    assert torch.allclose(grouped_scale, dense_scale, atol=1.0e-7, rtol=1.0e-6)
    assert torch.allclose(grouped_bias, dense_bias, atol=1.0e-7, rtol=1.0e-6)


def test_contextual_depth_range_calibration_is_identity_initialized() -> None:
    module = ContextualDepthRangeCalibration(
        8,
        groups=4,
        width=8,
        maximum_scale_residual=1.0,
        maximum_bias_residual=2.0,
    )
    features = torch.randn(2, 8, 16, 16, requires_grad=True)
    depth_logit = torch.randn(2, 1, 16, 16, requires_grad=True)
    calibrated, scale, bias = module(features, depth_logit)
    assert torch.equal(calibrated, depth_logit)
    assert torch.count_nonzero(scale) == 0
    assert torch.count_nonzero(bias) == 0
    calibrated.mean().backward()
    assert module.projection.weight.grad is not None
    assert torch.isfinite(module.projection.weight.grad).all()


def test_contextual_depth_range_calibration_strength_zero_recovers_baseline() -> None:
    module = ContextualDepthRangeCalibration(
        8,
        groups=4,
        width=8,
        maximum_scale_residual=1.0,
        maximum_bias_residual=2.0,
        strength=0.0,
    )
    with torch.no_grad():
        module.projection.weight.fill_(0.1)
        module.projection.bias.copy_(torch.tensor((0.2, -0.3)))
    features = torch.randn(2, 8, 16, 16)
    depth_logit = torch.randn(2, 1, 16, 16)
    calibrated, scale, bias = module(features, depth_logit)
    assert torch.equal(calibrated, depth_logit)
    assert torch.count_nonzero(scale) == 0
    assert torch.count_nonzero(bias) == 0


def test_depth_log_scale_calibration_has_tail_gradient_from_shallow_prediction() -> None:
    module = ContextualDepthRangeCalibration(
        8,
        groups=4,
        width=8,
        maximum_scale_residual=4.0,
        maximum_bias_residual=3.0,
        mode="depth_log_scale",
    )
    features = torch.randn(2, 8, 16, 16, requires_grad=True)
    shallow_logit = torch.full(
        (2, 1, 16, 16), -1.5, requires_grad=True
    )
    depth, scale, bias = module(features, shallow_logit)
    baseline = torch.nn.functional.softplus(shallow_logit) + 0.001
    assert torch.equal(depth, baseline)
    assert torch.count_nonzero(scale) == 0
    assert torch.count_nonzero(bias) == 0

    # A deep target must backpropagate into the calibration projection even
    # though the current baseline prediction is shallow.
    (depth - 20.0).abs().mean().backward()
    assert module.projection.weight.grad is not None
    assert torch.count_nonzero(module.projection.weight.grad) > 0
    assert torch.isfinite(module.projection.weight.grad).all()


def test_disabled_uncertainty_head_has_no_parameters_and_returns_fixed_scale() -> None:
    heads = SARTopoEdgeKANHeads(
        8,
        4,
        epsilon=0.001,
        maximum=5.0,
        depth_initialization_bias=-1.0,
        uncertainty_initial_scale_m=0.35,
        uncertainty_head_enabled=False,
    )
    assert heads.uncertainty_head is None
    outputs = heads(torch.randn(2, 8, 16, 16))
    assert outputs["uncertainty_scale"].shape == (2, 1, 16, 16)
    assert torch.all(outputs["uncertainty_scale"] == 0.35)


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
        water_surface, torch.full_like(water_surface, expected_surface)
    )
    expected = (expected_surface - elevation).clamp_min(0.10)
    expected = torch.where(support > 0.5, expected, torch.full_like(expected, 0.10))
    assert torch.allclose(prior, expected)


def test_known_range_hydrostatic_prior_separates_connected_components() -> None:
    module = KnownRangeHydrostaticPrior(
        boundary_quantile=0.25,
        minimum_depth_m=0.10,
        componentwise=True,
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
        maximum=5.0,
        depth_initialization_bias=0.0,
        uncertainty_initial_scale_m=0.35,
        uncertainty_head_enabled=False,
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
