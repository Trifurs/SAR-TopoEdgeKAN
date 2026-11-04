#!/usr/bin/env python3
"""Create one parameter and protocol inventory for all reported methods."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.logging import write_rows
from utils.misc import atomic_write_json, atomic_write_text
from utils.registry import MODEL_DISPLAY_NAME, build_model
from compare.common.comparison_factory import BUILDERS, build_comparison_model


LEARNED_COMPARISON_CONFIGS = (
    Path("configs/compare/deep_learning/dlsim_attention_unet.xml"),
    Path("configs/compare/deep_learning/dlsim_linknet.xml"),
    Path("configs/compare/deep_learning/unet_depth_regression.xml"),
    Path("configs/compare/deep_learning/resnet18_depth_regression.xml"),
    Path("configs/compare/deep_learning/unetplusplus_depth_regression.xml"),
)


def _count(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _learned_row(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    name = str(config["model"]["name"])
    if name not in BUILDERS:
        raise ValueError(f"Unsupported learned comparison configuration {config_path}: {name!r}")
    total, trainable = _count(build_comparison_model(config))
    schema = str(config["model"]["input_schema"])
    inputs = (
        "S1 change + DSM + valid_depth_mask"
        if schema == "dlsim_change_dsm_range"
        else "S1 T1/T2/change + DSM/slope + valid_depth_mask"
    )
    architecture = config["model"]
    channel_text = "/".join(str(value) for value in architecture.get("channels", ()))
    return {
        "method": name.replace("_", " "),
        "identifier": name,
        "learnable": "yes",
        "total_parameters": total,
        "trainable_parameters": trainable,
        "inputs": inputs,
        "flood_support": "valid_depth_mask",
        "key_configuration": (
            f"schema={schema}; channels={channel_text or 'ResNet18 encoder'}; "
            "shared task-adaptive depth objective"
        ),
        "source_record": "docs/COMPARISON_SOURCES.md",
    }


def inventory(depth_config: Path, comparison_configs: tuple[Path, ...]) -> list[dict[str, Any]]:
    depth = load_config(depth_config)
    depth_total, depth_trainable = _count(build_model(depth))
    return [
        {
            "method": MODEL_DISPLAY_NAME, "identifier": "sar_topo_edge_kan", "learnable": "yes",
            "total_parameters": depth_total, "trainable_parameters": depth_trainable,
            "inputs": "S1 T1/T2/change + DSM/slope + valid_depth_mask",
            "flood_support": "valid_depth_mask (known/oracle range)",
            "key_configuration": (
                "channels=24/48/96/144; identity-initialized signed TAE-KAN; "
                "graph stride=8"
            ),
            "source_record": "docs/MODEL.md",
        },
        {
            "method": "FwDET v2.0", "identifier": "fwdet_v2", "learnable": "no",
            "total_parameters": 0, "trainable_parameters": 0,
            "inputs": "valid_depth_mask + DSM",
            "flood_support": "valid_depth_mask",
            "key_configuration": "nearest valid outer-boundary elevation allocation",
            "source_record": "docs/COMPARISON_PROTOCOL.md",
        },
        {
            "method": "Trend Surface Analysis", "identifier": "tsa", "learnable": "no",
            "total_parameters": 0, "trainable_parameters": 0,
            "inputs": "valid_depth_mask + DSM",
            "flood_support": "valid_depth_mask",
            "key_configuration": "per-component first-degree boundary-elevation WSE fit",
            "source_record": "docs/COMPARISON_PROTOCOL.md",
        },
        {
            "method": "FlDepth", "identifier": "fldepth", "learnable": "no",
            "total_parameters": 0, "trainable_parameters": 0,
            "inputs": "valid_depth_mask + DSM",
            "flood_support": "valid_depth_mask",
            "key_configuration": "medial-ridge cross-section bank-elevation reconstruction",
            "source_record": "docs/COMPARISON_PROTOCOL.md",
        },
    ] + [_learned_row(path) for path in comparison_configs]


def _markdown(rows: list[dict[str, Any]]) -> str:
    fields = ("method", "identifier", "learnable", "total_parameters", "trainable_parameters", "inputs", "flood_support", "key_configuration", "source_record")
    header = "| " + " | ".join(field.replace("_", " ") for field in fields) + " |\n"
    separator = "|" + "|".join("---" for _ in fields) + "|\n"
    body = "".join("| " + " | ".join(str(row[field]) for field in fields) + " |\n" for row in rows)
    return "# Model inventory\n\n" + header + separator + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-config", type=Path, default=Path("configs/sar_topo_edge_kan.xml"))
    parser.add_argument("--comparison-config", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison_configs = (
        tuple(args.comparison_config)
        if args.comparison_config is not None
        else LEARNED_COMPARISON_CONFIGS
    )
    rows = inventory(args.depth_config, comparison_configs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "model_inventory.csv", rows)
    atomic_write_json(output / "model_inventory.json", rows)
    atomic_write_text(output / "model_inventory.md", _markdown(rows))
    print(_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
