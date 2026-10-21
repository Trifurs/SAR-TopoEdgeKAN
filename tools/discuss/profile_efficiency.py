#!/usr/bin/env python3
"""Measure discussion model size, synchronized latency, memory and supported FLOPs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import gc
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import default_collate

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from tools.discuss.experiments import config_paths, validate_protocol
from tools.evaluate_sar_topo_edge_kan import dataset_fingerprint
from utils.amp import resolve_amp
from utils.checkpoint import load_checkpoint
from utils.config import jsonable_config, load_config
from utils.efficiency import model_size_metrics
from utils.evaluation_suite import completed_run_directories, load_resolved_run_config
from utils.experiment_catalog import experiment_id
from utils.experiment_inventory import resolved_config_sha256, runtime_source_sha256
from utils.logging import write_rows
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model
from utils.seed import seed_everything


def checkpoint_index(root: Path, *, seed: int):
    choices = {}
    if not root.is_dir():
        return choices
    for run in completed_run_directories(root):
        saved = load_resolved_run_config(run)
        if int(saved.get("seed", -1)) != seed:
            continue
        identifier = experiment_id(saved)
        previous = choices.get(identifier)
        if previous is None or (run / "training_summary.json").stat().st_mtime_ns > (previous / "training_summary.json").stat().st_mtime_ns:
            choices[identifier] = run
    return choices


def sample_inputs(config, batch_size, device):
    contract = DatasetContract.load(config["dataset"]["contract"])
    dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], "test",
        band_spec=resolve_band_spec(config, contract), input_spec=ModelInputSpec.from_config(config),
        minimum_event_band_fraction=config["dataset"]["minimum_event_band_fraction"],
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    indices = np.linspace(0, len(dataset) - 1, batch_size, dtype=int).tolist()
    samples = [dataset[index] for index in indices]
    batch = default_collate(samples)
    inputs = prepare_model_inputs(batch, ModelInputSpec.from_config(config), include_valid_depth_mask=True, include_reliability=False)
    return move_to_device(inputs, device), [sample["metadata"]["sample_id"] for sample in samples]


def stage_modules(model):
    result = {}
    for name in ("sar_encoder", "terrain", "fusion", "context", "graph", "decoder", "hydrostatic_prior", "heads", "network"):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            result[name] = module
    return result


def measure(model, inputs, device, *, precision, warmup, iterations, repetitions, stages=True, flops=True):
    if min(iterations, repetitions) < 1 or warmup < 0:
        raise ValueError("Iteration/repetition counts must be positive and warm-up nonnegative")
    amp_enabled, dtype, _ = resolve_amp(device, precision != "float32", precision if precision != "float32" else "auto")
    model.eval()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def invoke():
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=dtype):
            return model(inputs)

    latency, throughput, peaks, increments = [], [], [], []
    batch_size = int(inputs["valid_depth_mask"].shape[0])
    with torch.inference_mode():
        for _ in range(repetitions):
            for _ in range(warmup):
                output = invoke()
                del output
            sync()
            baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            durations = []
            for _ in range(iterations):
                sync()
                start = time.perf_counter()
                output = invoke()
                sync()
                durations.append(time.perf_counter() - start)
                del output
            latency.extend(durations)
            throughput.append(batch_size * iterations / sum(durations))
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device)
                peaks.append(peak / 1024**2)
                increments.append((peak - baseline) / 1024**2)

        # Hooks run in a separate pass and cannot affect the latency samples.
        executed = set()
        hooks = []
        for module in model.modules():
            def record(module, arguments, output):
                executed.update(id(parameter) for parameter in module.parameters(recurse=False))
            hooks.append(module.register_forward_hook(record))
        output = invoke()
        del output
        sync()
        for hook in hooks:
            hook.remove()
        executed_count = sum(p.numel() for p in model.parameters() if id(p) in executed)

        stage_samples = defaultdict(list)
        if stages:
            starts = {}
            for name, module in stage_modules(model).items():
                def before(module, arguments, name=name):
                    sync()
                    starts[name] = time.perf_counter()
                def after(module, arguments, output, name=name):
                    sync()
                    stage_samples[name].append((time.perf_counter() - starts.pop(name)) * 1000)
                hooks.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
            output = invoke()
            del output
            sync()
            for hook in hooks:
                hook.remove()

        flop_count = None
        flop_operators = []
        uncounted = []
        if flops:
            # CPU-side ATen records include the shapes of CUDA conv/mm calls;
            # this does not require CUPTI or claim coverage of every operation.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, with_flops=True) as trace:
                output = invoke()
                del output
                sync()
            events = list(trace.key_averages())
            flop_count = sum(int(event.flops or 0) for event in events)
            flop_operators = [event.key for event in events if event.flops]
            uncounted = [event.key for event in events if event.key.startswith("aten::") and not event.flops]

    values = np.asarray(latency) * 1000
    return {
        **model_size_metrics(model),
        "executed_module_parameters": executed_count,
        "executed_parameter_scope": "parameters owned directly by modules visited in a separate forward pass",
        "batch_size": batch_size, "precision": precision,
        "warmup_iterations_per_repetition": warmup, "measurement_iterations_per_repetition": iterations,
        "repetitions": repetitions, "latency_samples": len(latency),
        "mean_batch_latency_ms": float(values.mean()),
        "p50_batch_latency_ms": float(np.quantile(values, 0.5)),
        "p95_batch_latency_ms": float(np.quantile(values, 0.95)),
        "mean_amortized_sample_latency_ms": float(values.mean() / batch_size),
        "samples_per_second_mean": statistics.mean(throughput),
        "samples_per_second_std": statistics.stdev(throughput) if len(throughput) > 1 else 0.0,
        "peak_gpu_allocated_mib": max(peaks) if peaks else None,
        "peak_gpu_incremental_mib": max(increments) if increments else None,
        "supported_flops_per_sample": flop_count / batch_size if flop_count is not None else None,
        "flops_scope": "torch.profiler supported operators only; not complete FLOPs (e.g. quantile, pooling and elementwise operations may be uncounted)",
        "counted_flop_operators": flop_operators,
        "operators_without_flop_counts": uncounted,
        "stage_latency_ms": {name: sum(values) for name, values in stage_samples.items()},
        "stage_timing_scope": "separate synchronized diagnostic forward; hooks add overhead; nested stages must not be summed",
        "timing_scope": "synchronized model forward including required priors; excludes data IO and evaluation-only physical diagnostics",
        "raw_batch_latency_ms": values.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/discuss/efficiency.xml")
    parser.add_argument("--runs", type=Path, help="Search completed discussion training runs recursively.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", nargs="+", help="Optional discussion config stems, e.g. full no_graph unet_depth_regression_matched.")
    parser.add_argument("--allow-untrained", action="store_true", help="Explicit architecture-only diagnostic; records that weights are untrained.")
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--skip-flops", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    policy = deepcopy(config["efficiency"])
    if policy["split"] != "test":
        parser.error("Efficiency inputs must use the common test split")
    for name, argument in (("batch_sizes", args.batch_sizes), ("measurement_iterations", args.iterations), ("warmup_iterations", args.warmup), ("repetitions", args.repetitions)):
        if argument is not None:
            policy[name] = argument
    if any(size < 1 for size in policy["batch_sizes"]) or min(policy["measurement_iterations"], policy["repetitions"]) < 1 or policy["warmup_iterations"] < 0:
        parser.error("Invalid batch or iteration counts")
    if not torch.cuda.is_available():
        raise RuntimeError("This discussion efficiency protocol requires CUDA")
    device = torch.device("cuda")
    paths = config_paths(policy["groups"])
    if args.models:
        requested = set(args.models)
        missing = requested.difference(path.stem for path in paths)
        if missing:
            parser.error(f"Unknown discussion config stems: {sorted(missing)}")
        paths = tuple(path for path in paths if path.stem in requested)
    validate_protocol(paths)
    seed_index = int(policy["checkpoint_seed_index"])
    if seed_index < 1:
        parser.error("efficiency.checkpoint_seed_index must be positive")
    stride = int(config["runtime"]["random_seed_validation"]["seed_stride"])
    checkpoint_seed = int(config["seed"]) + (seed_index - 1) * stride
    runs = checkpoint_index(
        (args.runs or Path(config["runs_root"])).expanduser().resolve(), seed=checkpoint_seed
    )
    for path in paths:
        model_config = load_config(path)
        if policy["require_checkpoint"] and not args.allow_untrained and model_config["model"]["discuss"]["trainable"] and experiment_id(model_config) not in runs:
            raise FileNotFoundError(f"No completed checkpoint for {path.stem} at seed {checkpoint_seed}; train it first or explicitly use --allow-untrained for a diagnostic")
    destination = (args.output or Path(policy["output_root"]) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    runtime_hash, _ = runtime_source_sha256(PROJECT_ROOT)
    atomic_write_json(destination / "protocol.json", {
        "efficiency": jsonable_config(policy), "allow_untrained": args.allow_untrained,
        "checkpoint_seed": checkpoint_seed,
        "runtime_source_sha256": runtime_hash, "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "configs": {path.name: resolved_config_sha256(path) for path in paths},
    })
    rows = []
    for path in paths:
        model_config = load_config(path)
        seed_everything(checkpoint_seed, model_config["deterministic"])
        model = build_model(model_config).to(device).eval()
        identifier = experiment_id(model_config)
        source = runs.get(identifier)
        checkpoint = source / "best_raw.pth" if source else None
        trained = model_config["model"]["discuss"]["trainable"]
        if checkpoint is not None:
            load_checkpoint(checkpoint, model, map_location="cpu", expected_fingerprint=dataset_fingerprint(model_config))
        for batch_size in policy["batch_sizes"]:
            inputs, sample_ids = sample_inputs(model_config, batch_size, device)
            if tuple(inputs["terrain"].shape[-2:]) != (policy["patch_size"], policy["patch_size"]):
                raise ValueError("Dataset patch shape differs from the efficiency configuration")
            result = measure(
                model, inputs, device, precision=policy["precision"],
                warmup=policy["warmup_iterations"], iterations=policy["measurement_iterations"],
                repetitions=policy["repetitions"], stages=policy["profile_stages"],
                flops=policy["profile_supported_flops"] and not args.skip_flops,
            )
            result.update({
                "model": identifier, "split": "test", "sample_ids": sample_ids,
                "seed": checkpoint_seed,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "weights_source": "checkpoint" if checkpoint else "untrained_architecture" if trained else "deterministic",
            })
            rows.append(result)
            atomic_write_json(destination / "summary.json", {"status": "running", "results": rows})
            write_rows(destination / "efficiency.csv", [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows])
            print(f"{identifier} | batch={batch_size} | latency={result['mean_batch_latency_ms']:.3f} ms | throughput={result['samples_per_second_mean']:.2f} sample/s", flush=True)
            del inputs
        del model
        gc.collect()
        torch.cuda.empty_cache()
    atomic_write_json(destination / "summary.json", {"status": "complete", "results": rows})
    print(f"Efficiency report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
