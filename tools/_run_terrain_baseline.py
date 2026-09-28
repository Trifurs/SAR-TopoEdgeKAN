"""Private deterministic runner selected by the unified XML dispatcher."""

from __future__ import annotations

import logging
import os
from itertools import islice
from pathlib import Path
import sys
import time
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from compare.common.traditional_registry import available_methods, estimator_for
from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from datasets.supervision_masks import canonical_positive_mask_from_batch
from metrics.aggregator import EvaluationAggregator
from utils.config import jsonable_config
from utils.efficiency import InferenceEfficiency
from utils.logging import format_duration, setup_logging, write_rows
from utils.misc import atomic_write_json
from utils.raster_io import write_geotiff
from utils.tensorboard import add_metadata, add_scalars, create_summary_writer, flush


FLOOD_SUPPORT = "valid_depth_mask"
LOGGER = logging.getLogger("comparison.traditional")


def _dataset(config: Mapping[str, Any], split: str) -> FloodDepthDataset:
    contract = DatasetContract.load(config["dataset"]["contract"])
    return FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], split,
        band_spec=resolve_band_spec(config, contract),
        input_spec=ModelInputSpec.from_config(config),
        minimum_event_band_fraction=float(config["dataset"].get("minimum_event_band_fraction", 1.0)),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )


def _metadata_item(mapping: Mapping[str, Any], key: str) -> Any:
    value = mapping[key]
    if isinstance(value, (list, tuple)):
        return value[0]
    if isinstance(value, torch.Tensor):
        return value[0].item()
    return value


def _validate_config(config: Mapping[str, Any], method: str) -> None:
    compare = config.get("compare")
    if not isinstance(compare, Mapping):
        raise KeyError("Comparison configuration requires a <compare> section")
    if compare.get("method") != method:
        raise ValueError(
            f"Configuration is for {compare.get('method')!r}, not {method!r}"
        )
    if compare.get("flood_support") != FLOOD_SUPPORT:
        raise ValueError(f"Only {FLOOD_SUPPORT!r} is supported as flood range")
    if method not in available_methods():
        raise ValueError(f"Unknown comparison model {method!r}")


def run_model(
    config: Mapping[str, Any], split: str, method: str, output: Path,
    max_batches: int | None, save_predictions: bool,
) -> dict[str, Any]:
    """Evaluate one named terrain model using ``valid_depth_mask`` directly."""

    _validate_config(config, method)
    output.mkdir(parents=True, exist_ok=True)
    setup_logging(
        output / "evaluate.log",
        show_python_warnings=bool(config["logging"].get("show_python_warnings", True)),
    )
    started = time.perf_counter()
    LOGGER.info("━" * 78)
    LOGGER.info("Deterministic evaluation started | model=%s | split=%s", method, split)
    LOGGER.info("Run directory: %s", output)
    LOGGER.info("━" * 78)
    estimator = estimator_for(method)
    dataset = _dataset(config, split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    normalizer = RobustNormalizer(Path(config["dataset"]["train_stats"]), dataset.contract)
    depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    aggregator = EvaluationAggregator(depth_bins, primary_depth_bins=normalizer.train_depth_bins)
    efficiency = InferenceEfficiency(torch.device("cpu"))
    atomic_write_json(output / "resolved_config.json", jsonable_config(config))
    progress = bool(config["logging"].get("progress_bar", False))
    disable_progress = not progress or os.environ.get("FLOOD_DEPTH_DISABLE_TQDM", "").lower() in {
        "1",
        "true",
        "yes",
    }
    selected_batches = loader if max_batches is None else islice(loader, max_batches)
    progress_total = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch_index, batch in enumerate(
        tqdm(
            selected_batches,
            total=progress_total,
            desc=f"{method} {split}",
            leave=False,
            disable=disable_progress,
        )
    ):
        metadata = batch["metadata"]
        valid = canonical_positive_mask_from_batch(batch)[0, 0].numpy() > 0.5
        terrain = batch["terrain_raw"][0, 0].numpy().astype(np.float64)
        target = batch["label"][0, 0].numpy()
        output_valid = batch["validity"]["output_valid"][0, 0].numpy() > 0.5
        support = batch["masks"]["valid_depth_mask"][0, 0].numpy() > 0.5
        sample_id = str(_metadata_item(metadata, "sample_id"))
        event_id = str(_metadata_item(metadata, "source_event_id"))
        label_path = Path(str(_metadata_item(metadata, "label_path")))
        inference_started = efficiency.start()
        prediction = estimator(support, terrain)
        efficiency.stop(
            inference_started,
            samples=1,
            output_pixels=int(prediction.size),
        )
        aggregator.add(sample_id, event_id, prediction, target, np.ones_like(prediction), valid)
        if save_predictions:
            with rasterio.open(label_path) as reference:
                write_geotiff(
                    output / "predicted_depth_m" / f"{sample_id}.tif", prediction,
                    crs=reference.crs, transform=reference.transform,
                    valid_mask=output_valid, descriptions=("predicted_depth_m",),
                )
    summary, samples, events, bins = aggregator.summarize()
    summary = {
        "method": method,
        "model_family": "traditional",
        "flood_support": FLOOD_SUPPORT,
        "total_parameters": 0,
        "trainable_parameters": 0,
        "parameter_storage_mib": 0.0,
        "buffer_storage_mib": 0.0,
        "efficiency_precision": "float64_numpy",
        **summary,
        **efficiency.summary(),
    }
    summary["efficiency_end_to_end_seconds"] = float(time.perf_counter() - started)
    write_rows(output / "metrics_by_sample.csv", samples)
    write_rows(output / "metrics_by_event.csv", events)
    write_rows(output / "metrics_by_train_depth_bin.csv", bins)
    atomic_write_json(output / "summary.json", summary)
    writer = create_summary_writer(
        output / "tensorboard",
        enabled=bool(config["logging"].get("tensorboard", False)),
        flush_seconds=int(config["logging"].get("tensorboard_flush_seconds", 30)),
        logger=LOGGER,
    )
    add_metadata(
        writer,
        {"model": method, "split": split, "flood_support": FLOOD_SUPPORT},
    )
    add_scalars(writer, summary, step=0, prefix="evaluation")
    flush(writer)
    if writer is not None:
        writer.close()
    LOGGER.info(
        "Evaluation complete | elapsed=%s | pixel_micro_mae=%.5f | event_macro_mae=%.5f",
        format_duration(time.perf_counter() - started),
        float(summary["pixel_micro_mae"]),
        float(summary["event_macro_mae"]),
    )
    return summary


if __name__ == "__main__":
    raise SystemExit("Use `python test.py <training-runs-directory>` from the project root.")
