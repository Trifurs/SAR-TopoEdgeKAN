#!/usr/bin/env python3
"""Inspect or run the optional input, graph and prior discussion experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.discuss.experiments import CONFIG_ROOT, GROUPS, config_paths, validate_protocol
from utils.config import jsonable_config, load_config
from utils.experiment_catalog import experiment_id


def evaluate_prior(output=None, *, max_batches=None, device_name="cuda"):
    import torch
    from torch.utils.data import DataLoader
    from datasets.band_selection import resolve_band_spec
    from datasets.contract import DatasetContract
    from datasets.flooddepth_dataset import FloodDepthDataset
    from datasets.model_input_spec import ModelInputSpec
    from datasets.preprocessing import resolve_depth_stratification_bins
    from losses.composite_loss import CompositeFloodDepthLoss
    from tools.evaluate_sar_topo_edge_kan import dataset_fingerprint, evaluate_loader
    from utils.amp import resolve_amp
    from utils.efficiency import model_size_metrics
    from utils.logging import write_rows
    from utils.misc import atomic_write_json
    from utils.registry import build_model

    config = load_config(CONFIG_ROOT / "prior_only.xml")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the discussion experiment configuration")
    contract = DatasetContract.load(config["dataset"]["contract"])
    contract.verify_fingerprints(include_normalization=True)
    dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], "test",
        band_spec=resolve_band_spec(config, contract), input_spec=ModelInputSpec.from_config(config),
        minimum_event_band_fraction=config["dataset"]["minimum_event_band_fraction"],
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)
    normalizer = dataset.normalizer
    bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    criterion = CompositeFloodDepthLoss(config["loss"], bins, normalizer.train_depth_bins)
    model = build_model(config).to(device).eval()
    amp_enabled, amp_dtype, _ = resolve_amp(device, config["training"]["amp"], config["training"]["amp_dtype"])
    summary, samples, events, depth_bins = evaluate_loader(
        model, loader, device, bins, primary_depth_bins=normalizer.train_depth_bins,
        criterion=criterion, max_batches=max_batches, input_spec=ModelInputSpec.from_config(config),
        amp_enabled=amp_enabled, amp_dtype=amp_dtype, measure_efficiency=True,
        progress=bool(config["logging"]["progress_bar"]), progress_label="Prior-only test",
    )
    summary.update({
        **model_size_metrics(model), "model": experiment_id(config), "split": "test",
        "training_required": False, "weights": "deterministic", "flood_support": "valid_depth_mask",
        "dataset_fingerprint": dataset_fingerprint(config), "diagnostic_max_batches": max_batches,
    })
    destination = Path(output) if output else Path(config["runs_root"]) / "prior_only" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    atomic_write_json(destination / "summary.json", summary)
    atomic_write_json(destination / "resolved_config.json", jsonable_config(config))
    write_rows(destination / "metrics_by_sample.csv", samples)
    write_rows(destination / "metrics_by_event.csv", events)
    write_rows(destination / "metrics_by_train_depth_bin.csv", depth_bins)
    print(f"Prior-only test report: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "check", "train"):
        child = sub.add_parser(command)
        child.add_argument("--groups", nargs="+", choices=(*GROUPS, "all"), default=["all"])
        if command == "check":
            child.add_argument("--build", action="store_true", help="Also instantiate each model and report parameter counts.")
        if command == "train":
            child.add_argument("--seeds", type=int, default=5)
            child.add_argument("--output", type=Path)
            child.add_argument("--resume", type=Path)
    prior = sub.add_parser("prior")
    prior.add_argument("--output", type=Path)
    prior.add_argument("--max-batches", type=int, help="Diagnostic subset only; omit for a formal full test.")
    args = parser.parse_args()
    if args.command == "prior":
        if args.max_batches is not None and args.max_batches < 1:
            parser.error("--max-batches must be positive")
        evaluate_prior(args.output, max_batches=args.max_batches)
        return 0
    paths = config_paths(args.groups)
    reference = validate_protocol(paths)
    if args.command in {"list", "check"}:
        for path in paths:
            config = load_config(path)
            discussion = config["model"]["discuss"]
            text = f"{experiment_id(config):44s} {path.relative_to(PROJECT_ROOT)}"
            if args.command == "check" and args.build:
                from utils.registry import build_model
                model = build_model(config)
                total = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                text += f" | parameters={total:,} trainable={trainable:,}"
            if not discussion["trainable"]:
                text += " | deterministic evaluation only"
            print(text)
        print(f"Common protocol verified for {len(paths)} unique configurations.")
        return 0
    if args.output and args.resume:
        parser.error("--output and --resume are mutually exclusive")
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    from validate_random_seeds import run_random_seed_validation
    trainable = config_paths(args.groups, trainable_only=True)
    destination = args.output
    if destination is None and args.resume is None:
        destination = Path(reference["runs_root"]) / "random_seed_validation" / f"k{args.seeds}" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_random_seed_validation(args.seeds, output=destination, resume=args.resume, model_config_paths=trainable)
    print("Training and paired test evaluation complete. Run the deterministic control separately with the prior subcommand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
