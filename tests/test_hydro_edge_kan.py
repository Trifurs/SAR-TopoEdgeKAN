from __future__ import annotations

import pytest
import torch

from models.hydro_edge_kan import HydroEdgeKAN
from models.graph_utils import DIRECTIONS
from models.terrain_features import path_barrier_proxy


def _physical(batch_size: int = 2, size: int = 16) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    elevation = torch.rand(batch_size, 1, size, size, generator=generator) * 5.0
    ground = elevation - torch.rand(
        batch_size, 1, size, size, generator=generator
    )
    valid = torch.ones_like(elevation)
    return {
        "dem_valid": valid,
        "dsm_elevation": elevation,
        "z_ground_proxy": ground,
        "physics_elevation": ground,
        "z_relative": elevation - ground,
        "local_relief": torch.ones_like(elevation),
    }


@pytest.mark.parametrize("mode", ["full_resolution", "graph_scale"])
def test_hydro_edge_kan_barrier_modes_preserve_contract(mode: str) -> None:
    graph = HydroEdgeKAN(
        8,
        heads=2,
        graph_feature_stride=8,
        path_barrier_mode=mode,
    )
    features = torch.randn(2, 8, 2, 2)
    valid = torch.ones(2, 1, 16, 16)
    output, diagnostics = graph(features, _physical(), valid, feature_stride=8)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(diagnostics["graph_update_rms_ratio"])


def test_hydro_edge_kan_rejects_unknown_barrier_mode() -> None:
    with pytest.raises(ValueError, match="path_barrier_mode"):
        HydroEdgeKAN(8, heads=2, path_barrier_mode="unknown")


def test_hydro_edge_kan_eval_omits_large_training_diagnostics() -> None:
    graph = HydroEdgeKAN(
        8,
        heads=2,
        graph_feature_stride=8,
        path_barrier_mode="full_resolution",
    ).eval()
    features = torch.randn(2, 8, 2, 2)
    valid = torch.ones(2, 1, 16, 16)
    with torch.inference_mode():
        output, diagnostics = graph(
            features, _physical(), valid, feature_stride=8
        )
    assert output.shape == features.shape
    assert "graph_update_rms_ratio" not in diagnostics
    assert torch.isfinite(diagnostics["kan_coefficient_magnitude"])


def test_hydro_edge_kan_has_smooth_bounded_lightweight_diffusion() -> None:
    graph = HydroEdgeKAN(
        16,
        heads=2,
        graph_feature_stride=8,
        gamma_init_effective=0.04,
        gamma_max=0.08,
        gamma_parameterization="positive",
    )
    features = torch.randn(2, 16, 2, 2)
    valid = torch.ones(2, 1, 16, 16)
    output, _ = graph(features, _physical(), valid, feature_stride=8)

    update = output - features
    assert not torch.equal(output, features)
    assert graph.gamma.tolist() == pytest.approx([0.04, 0.04])
    assert update.abs().max().item() <= 0.100001
    assert sum(parameter.numel() for parameter in graph.parameters()) < 15_000
    output.square().mean().backward()
    assert graph.raw_gamma.grad is not None
    assert torch.isfinite(graph.raw_gamma.grad).all()
    assert torch.count_nonzero(graph.raw_gamma.grad) == graph.heads
    assert all(
        mixer.weight.grad is not None
        and torch.isfinite(mixer.weight.grad).all()
        and torch.count_nonzero(mixer.weight.grad) > 0
        for mixer in graph.message_mixer
    )


def test_signed_zero_gamma_starts_as_identity_and_can_learn() -> None:
    graph = HydroEdgeKAN(
        16,
        heads=2,
        graph_feature_stride=8,
        gamma_init_effective=0.0,
        gamma_max=0.04,
        gamma_parameterization="signed",
    )
    features = torch.randn(2, 16, 2, 2, requires_grad=True)
    valid = torch.ones(2, 1, 16, 16)
    output, _ = graph(features, _physical(), valid, feature_stride=8)

    torch.testing.assert_close(output, features)
    assert graph.gamma.tolist() == pytest.approx([0.0, 0.0])
    output.square().mean().backward()
    assert graph.raw_gamma.grad is not None
    assert torch.isfinite(graph.raw_gamma.grad).all()
    assert torch.count_nonzero(graph.raw_gamma.grad) == graph.heads


def _naive_path_barrier(
    elevation: torch.Tensor,
    valid: torch.Tensor,
    ground: torch.Tensor,
    pixel_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs, valids = [], []
    for dy, dx in DIRECTIONS:
        samples, sample_valid = [], []
        for step in range(pixel_step + 1):
            offset_y, offset_x = dy * step, dx * step
            boundary = torch.ones_like(valid)
            if offset_y > 0:
                boundary[..., :offset_y, :] = 0
            elif offset_y < 0:
                boundary[..., offset_y:, :] = 0
            if offset_x > 0:
                boundary[..., :, :offset_x] = 0
            elif offset_x < 0:
                boundary[..., :, offset_x:] = 0
            samples.append(
                torch.roll(elevation, (offset_y, offset_x), (-2, -1))
            )
            sample_valid.append(
                torch.roll(valid, (offset_y, offset_x), (-2, -1)) * boundary
            )
        values = torch.stack(samples, 1)
        stacked_valid = torch.stack(sample_valid, 1)
        path_valid = stacked_valid.prod(1)
        values = torch.where(
            stacked_valid > 0.5,
            values,
            torch.full_like(values, torch.finfo(values.dtype).min),
        )
        crest = values.max(1).values
        neighbour_ground = torch.roll(
            ground, (dy * pixel_step, dx * pixel_step), (-2, -1)
        )
        endpoint_valid = (
            valid
            * path_valid
            * torch.roll(valid, (dy * pixel_step, dx * pixel_step), (-2, -1))
        )
        outputs.append(
            torch.relu(crest - torch.maximum(ground, neighbour_ground))
            * endpoint_valid
        )
        valids.append(endpoint_valid)
    return torch.stack(outputs, 1), torch.stack(valids, 1)


@pytest.mark.parametrize("pixel_step", [1, 3, 8])
def test_reverse_symmetric_path_barrier_matches_naive_reference(
    pixel_step: int,
) -> None:
    generator = torch.Generator().manual_seed(29)
    elevation = torch.rand(2, 1, 23, 21, generator=generator) * 10.0
    ground = elevation - torch.rand(2, 1, 23, 21, generator=generator)
    valid = (torch.rand(2, 1, 23, 21, generator=generator) > 0.08).float()
    expected = _naive_path_barrier(elevation, valid, ground, pixel_step)
    actual = path_barrier_proxy(elevation, valid, pixel_step, ground)
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[0], expected[0])
