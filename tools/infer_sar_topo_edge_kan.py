#!/usr/bin/env python3
"""Run SAR-TopoEdgeKAN inference for one audited sample."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from losses.composite_loss import CompositeFloodDepthLoss
from tools.evaluate_sar_topo_edge_kan import (
    dataset_fingerprint,
    embed_source_fingerprints,
    evaluate_loader,
    frozen_depth_balance_for_config,
)
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.logging import setup_logging, write_rows
from utils.misc import atomic_write_json
from utils.registry import build_model


def locate_sample(config: dict, requested: str) -> tuple[FloodDepthDataset, int]:
    requested_path = Path(requested).expanduser()
    requested_stem = requested_path.stem if requested_path.suffix else requested
    band_spec = resolve_band_spec(
        config, DatasetContract.load(config["dataset"]["contract"])
    )
    input_spec = ModelInputSpec.from_config(config)
    for split in ("train", "val", "test"):
        dataset = FloodDepthDataset(
            config["dataset"]["contract"], config["dataset"]["train_stats"], split,
            band_spec=band_spec,
            input_spec=input_spec,
            minimum_event_band_fraction=float(
                config["dataset"].get("minimum_event_band_fraction", 1.0)
            ),
            s1_qa_names=config["dataset"].get("model_s1_qa_names"),
        )
        for index, row in enumerate(dataset.rows):
            if row["sample_id"] == requested or row["sample_id"] == requested_stem or Path(
                row["filename"]
            ).stem == requested_stem:
                return dataset, index
    raise KeyError(
        f"Input {requested!r} does not match a manifest sample ID or raster basename"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-geotiff", action="store_true")
    parser.add_argument("--save-visualization", action="store_true")
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    config = embed_source_fingerprints(load_config(args.config))
    dataset, index = locate_sample(config, args.input)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    )
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(
        args.checkpoint,
        model,
        expected_fingerprint=dataset_fingerprint(config),
        map_location=device,
    )
    if args.weights == "ema":
        if checkpoint.get("ema_model") is None:
            raise ValueError("--weights ema requested but checkpoint has no EMA state")
        model.load_state_dict(checkpoint["ema_model"], strict=True)
    loader = DataLoader(Subset(dataset, [index]), batch_size=1, shuffle=False)
    normalizer = RobustNormalizer(Path(config["dataset"]["train_stats"]), dataset.contract)
    depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    criterion = CompositeFloodDepthLoss(
        config["loss"],
        depth_bins,
        normalizer.train_depth_bins,
        normalizer.train_depth_bin_counts,
        frozen_depth_balance_for_config(config),
    )
    sample_id = dataset.rows[index]["sample_id"]
    output = args.output or (
        Path(config["runs_root"])
        / "infer"
        / str(config["run_name"])
        / f"{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output = output.resolve()
    summary, samples, events, bins = evaluate_loader(
        model,
        loader,
        device,
        depth_bins,
        primary_depth_bins=normalizer.train_depth_bins,
        criterion=criterion,
        output_dir=output,
        save_predictions=args.save_geotiff or args.save_visualization,
        input_spec=ModelInputSpec.from_config(config),
    )
    atomic_write_json(output / "summary.json", summary)
    write_rows(output / "metrics_by_sample.csv", samples)
    write_rows(output / "metrics_by_event.csv", events)
    write_rows(output / "metrics_by_train_depth_bin.csv", bins)
    print(f"inference output: {output / sample_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
