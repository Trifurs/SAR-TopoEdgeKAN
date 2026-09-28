"""Configuration-driven paths and runtime options for model-agnostic entry points."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


class RuntimeConfigError(ValueError):
    """Raised when an XML runtime section is incomplete or unsafe."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeConfigError(f"{name} must be a configuration mapping")
    return value


def _path_component(value: Any, name: str) -> str:
    component = str(value or "").strip()
    if not component or Path(component).name != component:
        raise RuntimeConfigError(f"{name} must be one path component, got {component!r}")
    return component


def runtime_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one optional ``runtime`` subsection with a stable empty default."""

    runtime = _mapping(config.get("runtime"), "runtime")
    return _mapping(runtime.get(name), f"runtime.{name}")


def optional_path(value: Any, name: str) -> Path | None:
    """Convert an XML optional path while rejecting ambiguous values."""

    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    if isinstance(value, str):
        return Path(value).expanduser().resolve()
    raise RuntimeConfigError(f"{name} must be a path or none, got {type(value).__name__}")


def optional_nonnegative_int(value: Any, name: str) -> int | None:
    """Validate a nullable batch-limit/count configuration value."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeConfigError(f"{name} must be an integer or none")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"{name} must be an integer or none") from exc
    if result < 0:
        raise RuntimeConfigError(f"{name} must be non-negative")
    return result


def model_identifier(config: Mapping[str, Any]) -> str:
    """Resolve the per-model directory name without relying on a CLI wrapper."""

    value = config.get("run_name")
    if value is None:
        model = _mapping(config.get("model"), "model")
        compare = _mapping(config.get("compare"), "compare")
        value = model.get("name", compare.get("method"))
    return _path_component(value, "model run identifier")


def started_at_run_id(
    config: Mapping[str, Any], started_at: datetime | None = None
) -> str:
    """Create the collision-resistant, start-time-based directory identifier."""

    runtime = _mapping(config.get("runtime"), "runtime")
    time_format = str(runtime.get("run_id_format", "%Y%m%d-%H%M%S-%f"))
    if not time_format.strip():
        raise RuntimeConfigError("runtime.run_id_format must not be empty")
    timestamp = (started_at or datetime.now().astimezone()).strftime(time_format)
    return _path_component(timestamp, "runtime.run_id_format result")


def _runs_root(config: Mapping[str, Any]) -> Path:
    value = config.get("runs_root")
    if value is None:
        raise RuntimeConfigError("configuration has no runs_root")
    return Path(value)


def training_runs_root(config: Mapping[str, Any]) -> Path:
    """Return the collection directory that contains timestamped training runs."""

    ablation = _mapping(config.get("ablation"), "ablation")
    if ablation:
        variant = _path_component(ablation.get("variant_id"), "ablation.variant_id")
        return _runs_root(config) / "ablation" / variant
    return _runs_root(config) / "train" / model_identifier(config)


def train_output_path(config: Mapping[str, Any], run_id: str | None = None) -> Path:
    """Resolve a configured output or a unique start-time-based training path."""

    train = runtime_section(config, "train")
    explicit = optional_path(train.get("output"), "runtime.train.output")
    if explicit is not None:
        return explicit
    identifier = run_id or started_at_run_id(config)
    return training_runs_root(config) / _path_component(identifier, "run identifier")


def evaluation_split(config: Mapping[str, Any]) -> str:
    """Return the one configured validation/test split."""

    evaluation = runtime_section(config, "evaluation")
    split = str(evaluation.get("split", "val"))
    if split not in {"val", "test"}:
        raise RuntimeConfigError("runtime.evaluation.split must be 'val' or 'test'")
    return split


def evaluation_output_path(config: Mapping[str, Any], run_id: str | None = None) -> Path:
    """Resolve a configured output or a start-time/source-run evaluation path."""

    evaluation = runtime_section(config, "evaluation")
    explicit = optional_path(evaluation.get("output"), "runtime.evaluation.output")
    if explicit is not None:
        return explicit
    identifier = run_id or started_at_run_id(config)
    return (
        _runs_root(config)
        / "evaluate"
        / model_identifier(config)
        / evaluation_split(config)
        / _path_component(identifier, "evaluation run identifier")
    )


def _configured_source_run(config: Mapping[str, Any]) -> str | None:
    value = runtime_section(config, "evaluation").get("source_run")
    if value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
    ):
        return None
    return _path_component(value, "runtime.evaluation.source_run")


def _latest_completed_checkpoint(config: Mapping[str, Any]) -> Path:
    root = training_runs_root(config)
    candidates = (
        [
            (child / "best_raw.pth", child / "training_summary.json")
            for child in root.iterdir()
            if child.is_dir()
            and (child / "best_raw.pth").is_file()
            and (child / "training_summary.json").is_file()
        ]
        if root.is_dir()
        else []
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed training run with best_raw.pth exists in {root}. "
            "Run `python train.py <model-config.xml>` first, or set "
            "runtime.evaluation.checkpoint/source_run in the XML."
        )
    checkpoint, _ = max(
        candidates,
        key=lambda item: (item[1].stat().st_mtime_ns, item[0].parent.name),
    )
    return checkpoint


def evaluation_checkpoint_path(config: Mapping[str, Any]) -> Path:
    """Resolve an explicit checkpoint, named source run, or latest completed run."""

    evaluation = runtime_section(config, "evaluation")
    explicit = optional_path(evaluation.get("checkpoint"), "runtime.evaluation.checkpoint")
    if explicit is not None:
        return explicit
    source_run = _configured_source_run(config)
    if source_run is not None:
        return training_runs_root(config) / source_run / "best_raw.pth"
    return _latest_completed_checkpoint(config)


def allow_existing_output(config: Mapping[str, Any], operation: str) -> bool:
    """Read the explicit opt-in that permits replacing an existing result folder."""

    section = runtime_section(config, operation)
    return bool(section.get("allow_existing_output", False))


def ensure_output_is_available(
    path: Path,
    *,
    allow_existing: bool,
    operation: str,
) -> None:
    """Fail closed before a formal run could overwrite prior evidence."""

    if path.exists() and not allow_existing:
        raise FileExistsError(
            f"{operation} output already exists: {path}. "
            "A new run receives a distinct start-time directory automatically; for an "
            "explicit output, choose a different directory or deliberately set "
            "allow_existing_output=true."
        )
