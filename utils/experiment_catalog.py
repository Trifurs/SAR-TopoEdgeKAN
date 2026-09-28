"""Canonical configuration catalog for full-model comparison workflows."""

from __future__ import annotations

from pathlib import Path

from utils.config import load_config
from utils.model_dispatch import configured_model_kind


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
MODEL_DISPLAY_NAMES = {
    "sar_topo_edge_kan": "SAR-TopoEdgeKAN",
    "dlsim_attention_unet": "DLSIM Attention U-Net",
    "dlsim_linknet": "DLSIM LinkNet",
    "resnet18_depth_regression": "ResNet18 Flood-Depth Regression",
    "unet_depth_regression": "U-Net Flood-Depth Regression",
    "unetplusplus_depth_regression": "U-Net++ Flood-Depth Regression",
    "fwdet_v2": "FwDET v2",
    "tsa": "TSA",
    "fldepth": "FlDepth",
}


def deep_learning_config_paths(*, include_ablations: bool = True) -> tuple[Path, ...]:
    """Return every unique trainable experiment in a stable publication order."""

    paths = [CONFIG_ROOT / "sar_topo_edge_kan.xml"]
    paths.extend(sorted((CONFIG_ROOT / "compare" / "deep_learning").glob("*.xml")))
    if include_ablations:
        ablations = list((CONFIG_ROOT / "ablation").glob("*.xml"))
        ablations.sort(
            key=lambda path: (
                int(load_config(path).get("ablation", {}).get("sequence_step", 10_000)),
                path.name,
            )
        )
        paths.extend(ablations)
    return tuple(path.resolve() for path in paths)


def traditional_config_paths() -> tuple[Path, ...]:
    return tuple(
        path.resolve()
        for path in sorted((CONFIG_ROOT / "compare" / "traditional").glob("*.xml"))
    )


def experiment_id(config: dict) -> str:
    """Identify a main/comparator/ablation experiment without duplicating full mode."""

    ablation = config.get("ablation")
    if isinstance(ablation, dict):
        return str(ablation["variant_id"])
    kind, identifier = configured_model_kind(config)
    return str(config.get("run_name", identifier)) if kind != "traditional" else identifier


def experiment_display_name(config: dict) -> str:
    ablation = config.get("ablation")
    if isinstance(ablation, dict):
        return str(ablation.get("display_name", experiment_id(config)))
    model = config.get("model")
    if isinstance(model, dict):
        identifier = str(model.get("name", experiment_id(config)))
        return str(model.get("display_name", MODEL_DISPLAY_NAMES.get(identifier, identifier)))
    comparison = config.get("compare", {})
    identifier = str(comparison.get("method", experiment_id(config)))
    return str(comparison.get("display_name", MODEL_DISPLAY_NAMES.get(identifier, identifier)))
