from __future__ import annotations

from pathlib import Path

from utils.config import load_config
from utils.experiment_catalog import deep_learning_config_paths


def test_learned_catalog_uses_one_fixed_split_and_training_protocol() -> None:
    paths = deep_learning_config_paths(include_ablations=True)
    assert len(paths) == 7
    configs = [load_config(path) for path in paths]
    reference = configs[0]

    for config in configs:
        for section in ("dataset", "supervision", "training", "optimizer", "scheduler", "loss"):
            assert config[section] == reference[section]
        assert config["seed"] == reference["seed"]
        assert config["model"]["flood_support"] == "valid_depth_mask"

    assert reference["loss"]["objective_mode"] == "task_adaptive"
    assert set(reference["loss"]) == {
        "objective_mode", "depth_huber_beta_m", "log_depth_huber_beta",
        "lambda_depth", "lambda_log", "lambda_metric_pixel_mae",
        "lambda_metric_pixel_rmse", "lambda_metric_event_mae",
        "lambda_metric_event_rmse", "lambda_metric_event_hierarchical_mae",
        "lambda_metric_depth_bin_rmse", "lambda_metric_event_hierarchical_rmse",
        "lambda_depth_bias", "depth_bias_beta_m",
    }
    assert reference["runtime"]["random_seed_validation"]["require_cuda"] is True

    learned_comparators = [
        config for config in configs if config["model"]["name"] != "sar_topo_edge_kan"
    ]
    assert len(learned_comparators) == 5
    assert all(config["model"]["input_schema"] == "s1_terrain_range" for config in learned_comparators)
    assert all(config["model"]["input_channels"] == 10 for config in learned_comparators)


def test_formal_edge_kan_ablation_inherits_main_protocol() -> None:
    main = load_config(Path("configs/sar_topo_edge_kan.xml"))
    ablation = load_config(Path("configs/ablation/sar_topo_edge_kan_wo_tae_kan.xml"))
    assert main["model"]["topographic_affinity_enabled"] is True
    assert ablation["model"]["topographic_affinity_enabled"] is False
    for section in ("dataset", "supervision", "training", "optimizer", "scheduler", "loss"):
        assert ablation[section] == main[section]
