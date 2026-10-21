from __future__ import annotations

import pytest
import torch
from torch.utils.data import default_collate

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from models.discuss.inputs import assemble_observations
from tools.discuss.experiments import CONFIG_ROOT, config_paths, validate_protocol
from utils.config import load_config
from utils.experiment_catalog import deep_learning_config_paths
from utils.misc import move_to_device
from utils.registry import build_model


def configured(name):
    return load_config(CONFIG_ROOT / f"{name}.xml")


@pytest.fixture(scope="module")
def real_batch():
    config = configured("full")
    contract = DatasetContract.load(config["dataset"]["contract"])
    dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], "train",
        band_spec=resolve_band_spec(config, contract), input_spec=ModelInputSpec.from_config(config),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    return default_collate([dataset[0]])


def inputs_for(batch, config, device="cpu"):
    values = prepare_model_inputs(batch, ModelInputSpec.from_config(config), include_valid_depth_mask=True, include_reliability=False)
    if not config["dataset"]["model_bands"]["s1_conditioning"]:
        values.pop("s1_conditioning", None)
    return move_to_device(values, torch.device(device))


def test_discussion_protocol_and_default_catalog_are_separate():
    paths = config_paths(("all",))
    validate_protocol(paths)
    assert len(paths) == 16
    assert len(config_paths(("all",), trainable_only=True)) == 15
    assert len(deep_learning_config_paths()) == 7
    assert not any("discuss" in path.parts for path in deep_learning_config_paths())


def test_protocol_rejects_a_changed_training_condition(tmp_path):
    overlay = tmp_path / "changed.xml"
    overlay.write_text(
        f'<config><include path="{CONFIG_ROOT / "full.xml"}" /><training><batch_size type="int">1</batch_size></training></config>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="training"):
        validate_protocol((overlay,))


def test_shared_observations_include_angles_raw_terrain_and_availability(real_batch):
    config = configured("unet_depth_regression_matched")
    model = build_model(config)
    inputs = inputs_for(real_batch, config)
    packed = assemble_observations(inputs, model.discussion_raw_names)
    assert packed.shape[1] == 21
    assert len(model.input_columns) == 21
    assert any("angle_pre_deg" in name for name in model.input_columns)
    assert any("angle_event_deg" in name for name in model.input_columns)
    assert "terrain_raw/elevation_m_DSM" in model.input_columns
    assert "branch_validity/s1_t1" in model.input_columns
    assert torch.equal(packed[:, -1:], real_batch["masks"]["valid_depth_mask"])
    changed = dict(inputs, s1_conditioning=inputs["s1_conditioning"] + 1)
    assert not torch.equal(packed, assemble_observations(changed, model.discussion_raw_names))
    assert not any("label" in name for name in model.input_columns)


def test_graph_replacements_preserve_shared_initialization_and_rng():
    torch.manual_seed(731)
    reference = build_model(configured("full"))
    random_after_reference = torch.rand(4)
    for name in ("mlp_edges", "linear_edges", "fixed_edges", "no_graph"):
        torch.manual_seed(731)
        model = build_model(configured(name))
        assert torch.equal(torch.rand(4), random_after_reference)
        common = {key: value for key, value in model.state_dict().items() if not key.startswith("graph.")}
        assert all(torch.equal(value, reference.state_dict()[key]) for key, value in common.items())
    mlp = build_model(configured("mlp_edges"))
    kan_edges = sum(p.numel() for p in reference.graph.edge_kan.parameters())
    mlp_edges = sum(p.numel() for p in mlp.graph.edge_kan.parameters())
    assert abs(kan_edges - mlp_edges) == 4
    no_graph = build_model(configured("no_graph"))
    assert no_graph.graph is None
    assert not any(key.startswith("graph.") for key in no_graph.state_dict())
    fixed = build_model(configured("fixed_edges"))
    assert not list(fixed.graph.edge_kan.parameters())
    assert "prior_raw_scales" not in dict(fixed.graph.named_parameters())
    assert "prior_bias" not in dict(fixed.graph.named_parameters())


def test_discussion_reference_preserves_production_weights_and_rng():
    torch.manual_seed(731)
    reference = build_model(load_config(CONFIG_ROOT.parent / "sar_topo_edge_kan.xml"))
    next_random = torch.rand(4)
    torch.manual_seed(731)
    discussion = build_model(configured("full"))
    assert torch.equal(torch.rand(4), next_random)
    assert reference.state_dict().keys() == discussion.state_dict().keys()
    assert all(torch.equal(value, reference.state_dict()[key]) for key, value in discussion.state_dict().items())


@pytest.mark.parametrize("name", [path.stem for path in config_paths(("all",))])
def test_each_control_runs_on_audited_sample(name, real_batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configured(name)
    model = build_model(config).to(device).eval()
    inputs = inputs_for(real_batch, config, device)
    # A real full-size sample exercises component labels, graph stride and range.
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
        result = model(inputs)
    assert result["depth"].shape == real_batch["label"].shape
    assert torch.isfinite(result["conditional_depth"]).all()
    assert torch.count_nonzero(result["depth"] * (inputs["valid_depth_mask"] <= 0.5)) == 0
    if config["model"]["discuss"]["trainable"]:
        assert any(parameter.requires_grad for parameter in model.parameters())
    else:
        assert sum(parameter.numel() for parameter in model.parameters()) == 0


@pytest.mark.parametrize("name", [path.stem for path in config_paths(("all",), trainable_only=True)])
def test_each_trainable_control_accepts_the_shared_loss_and_optimizer(name, real_batch):
    from losses.composite_loss import CompositeFloodDepthLoss
    from utils.optim import build_optimizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configured(name)
    model = build_model(config).to(device).train()
    batch = move_to_device(real_batch, device)
    optimizer = build_optimizer(model, config)
    criterion = CompositeFloodDepthLoss(config["loss"], [0.1, 0.5, 1.0, 2.0, 100.0])
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
        outputs = model(inputs_for(batch, config, device))
        loss, _ = criterion(outputs, batch, 0)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert any(bool(torch.any(value != 0)) for value in gradients)
    optimizer.step()


def test_terrain_only_never_reads_sar_observations(real_batch):
    config = configured("terrain_only")
    model = build_model(config).eval()
    assert model.sar_encoder is None and model.fusion is None
    assert model.decoder.change_projection is None
    assert not any(name.startswith(("sar_encoder.", "fusion.")) for name, _ in model.named_parameters())
    full_inputs = inputs_for(real_batch, config)
    minimal = {name: full_inputs[name] for name in ("terrain", "terrain_raw", "dem_valid", "valid_depth_mask")}
    with torch.no_grad():
        # Activate the learned correction, so identity initialization cannot
        # conceal an accidental dependency on an excluded sensor.
        model.heads.depth_head.trunk[-1].weight.fill_(0.001)
        expected = model(full_inputs)["depth"]
        actual = model(minimal)["depth"]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("name", ["full", "unet_prior"])
def test_prior_residual_heads_start_at_the_exact_same_prior(name, real_batch):
    config = configured(name)
    inputs = inputs_for(real_batch, config)
    with torch.no_grad():
        expected = build_model(configured("prior_only")).eval()(inputs)["depth"]
        actual = build_model(config).eval()(inputs)["depth"]
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("name", ["full", "mlp_edges", "linear_edges"])
def test_edge_function_has_finite_nonzero_training_gradient(name, real_batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configured(name)
    model = build_model(config).to(device).train()
    inputs = inputs_for(real_batch, config, device)
    with torch.no_grad():
        _, physical = model.terrain(inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"])
        model.graph.raw_gamma.fill_(0.3)
    features = torch.randn(1, config["model"]["channels"][-1], 32, 32, device=device, requires_grad=True)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
        output, _ = model.graph(features, physical, inputs["s1_event_support"], feature_stride=8)
        loss = output.float().square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.graph.edge_kan.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    if name == "full":
        spline_gradients = [layer.spline_coefficients.grad for layer in model.graph.edge_kan]
        assert all(value is not None and torch.isfinite(value).all() for value in spline_gradients)
        assert sum(float(value.abs().sum()) for value in spline_gradients) > 0


@pytest.mark.parametrize("name", ["unet_depth_regression_matched", "prior_only"])
def test_common_test_evaluator_accepts_adapters_and_preserves_physical_metrics(name, real_batch):
    from tools.evaluate_sar_topo_edge_kan import evaluate_loader
    from models.discuss.inputs import evaluation_terrain_features

    config = configured(name)
    model = build_model(config)
    summary, samples, _, _ = evaluate_loader(
        model, [real_batch], torch.device("cpu"), [0.1, 0.5, 1.0, 2.0],
        input_spec=ModelInputSpec.from_config(config), measure_efficiency=False,
    )
    assert summary["pixel_micro_pixels"] > 0
    assert summary["depth_output_semantics"] == "conditional_positive"
    assert "positive_region_wse_laplacian" in samples[0]
    assert torch.isfinite(evaluation_terrain_features(inputs_for(real_batch, config), model.discussion_raw_names, config["model"])["z_hyd"]).all()
