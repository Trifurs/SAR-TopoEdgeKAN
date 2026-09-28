"""Deterministic spatial test-time augmentation for dense predictions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


_FLIP_DIMS: dict[str, tuple[int, ...]] = {
    "identity": (),
    "horizontal": (-1,),
    "vertical": (-2,),
    "horizontal_vertical": (-2, -1),
}
_AVERAGED_OUTPUTS = (
    "depth",
    "conditional_depth",
    "positive_depth",
    "expected_depth",
    "uncertainty_scale",
)


def resolve_spatial_tta(transforms: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a deterministic transform list and ensure identity is included."""

    values = tuple(str(value) for value in (transforms or ("identity",)))
    if not values:
        values = ("identity",)
    unknown = sorted(set(values).difference(_FLIP_DIMS))
    if unknown:
        raise ValueError(f"unsupported spatial TTA transforms: {unknown}")
    unique = tuple(dict.fromkeys(values))
    return ("identity", *(value for value in unique if value != "identity"))


def _flip_nested(value: Any, dims: tuple[int, ...]) -> Any:
    if not dims:
        return value
    if isinstance(value, torch.Tensor):
        return torch.flip(value, dims) if value.ndim >= 4 else value
    if isinstance(value, Mapping):
        return {key: _flip_nested(item, dims) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_flip_nested(item, dims) for item in value)
    if isinstance(value, list):
        return [_flip_nested(item, dims) for item in value]
    return value


def spatial_tta_forward(
    forward: Callable[[Any], Mapping[str, Any]],
    model_inputs: Any,
    transforms: Sequence[str] | None,
) -> dict[str, Any]:
    """Average inverse-transformed dense outputs while retaining diagnostics.

    Only prediction tensors are averaged. Diagnostics and physical features are
    taken from the identity pass so downstream reporting keeps its established
    schema. Every spatial tensor in the input tree is transformed together,
    preserving SAR/terrain/validity alignment without introducing label data.
    """

    selected = resolve_spatial_tta(transforms)
    outputs_by_transform: list[tuple[tuple[int, ...], Mapping[str, Any]]] = []
    for name in selected:
        dims = _FLIP_DIMS[name]
        outputs_by_transform.append((dims, forward(_flip_nested(model_inputs, dims))))
    identity = dict(outputs_by_transform[0][1])
    for key in _AVERAGED_OUTPUTS:
        values = [
            torch.flip(output[key], dims) if dims else output[key]
            for dims, output in outputs_by_transform
            if isinstance(output.get(key), torch.Tensor)
        ]
        if len(values) == len(outputs_by_transform):
            identity[key] = torch.stack(values, dim=0).mean(dim=0)
    identity["spatial_tta_transforms"] = selected
    return identity
