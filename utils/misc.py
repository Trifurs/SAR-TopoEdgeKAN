"""Small dependency-free utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text via a sibling temporary file and atomic replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def move_to_device(value: Any, device: torch.device, non_blocking: bool | None = None) -> Any:
    """Recursively transfer tensors while leaving metadata/provenance untouched."""

    if non_blocking is None:
        non_blocking = device.type == "cuda"
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=bool(non_blocking))
    if isinstance(value, dict):
        return {key: move_to_device(item, device, non_blocking) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in value)
    return value
