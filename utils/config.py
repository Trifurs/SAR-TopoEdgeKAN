"""Type-safe XML configuration loading with recursive includes and overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET


class ConfigError(RuntimeError):
    """Raised for malformed or inconsistent configuration trees."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _optional_text(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    return stripped if stripped else None


def _parse_scalar(text: str | None, type_name: str, source: Path) -> Any:
    value = _optional_text(text)
    lowered = type_name.lower()
    if lowered in {"none", "null", "optional"}:
        return None if value is None or value.lower() in {"none", "null"} else value
    if value is None:
        raise ConfigError(f"Empty {type_name} value in {source}")
    if lowered == "str":
        return value
    if lowered == "int":
        return int(value)
    if lowered == "float":
        return float(value)
    if lowered == "bool":
        if value.lower() in {"true", "1", "yes", "on"}:
            return True
        if value.lower() in {"false", "0", "no", "off"}:
            return False
        raise ConfigError(f"Invalid bool {value!r} in {source}")
    if lowered == "path":
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    raise ConfigError(f"Unsupported config type {type_name!r} in {source}")


def _parse_element(element: ET.Element, source: Path) -> Any:
    children = [child for child in element if child.tag != "include"]
    explicit_type = element.attrib.get("type")
    if explicit_type == "list" or (children and all(child.tag == "item" for child in children)):
        item_type = element.attrib.get("item_type", "str")
        return [
            _parse_element(child, source)
            if list(child) or child.attrib.get("type")
            else _parse_scalar(child.text, item_type, source)
            for child in children
        ]
    if children:
        result: dict[str, Any] = {}
        for child in children:
            if child.tag in result:
                raise ConfigError(f"Duplicate key {child.tag!r} in {source}")
            result[child.tag] = _parse_element(child, source)
        return result
    return _parse_scalar(element.text, explicit_type or "str", source)


def _load_recursive(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved))
        raise ConfigError(f"Configuration include cycle: {cycle}")
    try:
        root = ET.parse(resolved).getroot()
    except ET.ParseError as exc:
        raise ConfigError(f"Invalid XML {resolved}: {exc}") from exc
    if root.tag != "config":
        raise ConfigError(f"Root element must be <config>: {resolved}")
    combined: dict[str, Any] = {}
    for include in root.findall("include"):
        reference = _optional_text(include.attrib.get("path") or include.text)
        if not reference:
            raise ConfigError(f"Empty include in {resolved}")
        include_path = Path(reference).expanduser()
        if not include_path.is_absolute():
            include_path = resolved.parent / include_path
        combined = _merge(combined, _load_recursive(include_path, (*stack, resolved)))
    own: dict[str, Any] = {}
    for child in root:
        if child.tag == "include":
            continue
        if child.tag in own:
            raise ConfigError(f"Duplicate root key {child.tag!r} in {resolved}")
        own[child.tag] = _parse_element(child, resolved)
    return _merge(combined, own)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and fully resolve a configuration XML tree."""

    return _load_recursive(Path(path), ())


def parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides to a resolved configuration."""

    result = deepcopy(config)
    for expression in overrides:
        if "=" not in expression:
            raise ConfigError(f"Override must be key=value: {expression!r}")
        dotted, raw = expression.split("=", 1)
        keys = [key for key in dotted.split(".") if key]
        if not keys:
            raise ConfigError(f"Empty override key: {expression!r}")
        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            existing = cursor.get(key)
            if not isinstance(existing, dict):
                raise ConfigError(f"Override parent is not a mapping: {dotted!r}")
            cursor = existing
        if keys[-1] not in cursor:
            raise ConfigError(f"Override key does not exist: {dotted!r}")
        parsed = parse_override_value(raw)
        current = cursor[keys[-1]]
        if isinstance(current, Path) and isinstance(parsed, str):
            path_value = Path(parsed).expanduser()
            parsed = path_value.resolve() if path_value.is_absolute() else (PROJECT_ROOT / path_value).resolve()
        cursor[keys[-1]] = parsed
    return result


def jsonable_config(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_config(item) for item in value]
    return value
