"""Stable fingerprints for repeated-seed experiment configurations and code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from utils.config import jsonable_config, load_config
from utils.experiment_catalog import experiment_id


RUNTIME_SOURCE_DIRECTORIES = (
    "compare",
    "datasets",
    "losses",
    "metrics",
    "models",
    "tools",
    "utils",
)
RUNTIME_ROOT_FILES = ("train.py", "evaluate.py", "test.py")


def resolved_config_sha256(path: Path) -> str:
    """Hash the fully merged semantic configuration, including all includes."""

    payload = json.dumps(
        jsonable_config(load_config(path)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_source_sha256(project_root: Path) -> tuple[str, int]:
    """Fingerprint Python sources that can affect a repeated-seed job."""

    root = project_root.expanduser().resolve(strict=True)
    paths = [root / name for name in RUNTIME_ROOT_FILES if (root / name).is_file()]
    for directory_name in RUNTIME_SOURCE_DIRECTORIES:
        directory = root / directory_name
        if directory.is_dir():
            paths.extend(directory.rglob("*.py"))
    selected = sorted(set(path.resolve() for path in paths))
    if not selected:
        raise ValueError(f"No runtime Python sources found below {root}")
    digest = hashlib.sha256()
    for path in selected:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), len(selected)


def config_inventory(paths: Iterable[Path]) -> list[dict[str, str]]:
    """Record resolved config identities for a repeated-seed campaign."""

    rows: list[dict[str, str]] = []
    for path in paths:
        config = load_config(path)
        rows.append(
            {
                "model": experiment_id(config),
                "config": str(path.resolve()),
                "resolved_config_sha256": resolved_config_sha256(path),
            }
        )
    return rows
