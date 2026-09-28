"""Factory for individually named learned comparison models."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from torch import nn

from compare.deep_learning.dlsim_attention_unet import build_dlsim_attention_unet
from compare.deep_learning.dlsim_linknet import build_dlsim_linknet
from compare.deep_learning.resnet18_depth_regression import build_resnet18_depth_regression
from compare.deep_learning.unet_depth_regression import build_unet_depth_regression
from compare.deep_learning.unetplusplus_depth_regression import (
    build_unetplusplus_depth_regression,
)


BUILDERS: dict[str, Callable[[Mapping[str, object]], nn.Module]] = {
    "dlsim_attention_unet": build_dlsim_attention_unet,
    "dlsim_linknet": build_dlsim_linknet,
    "unet_depth_regression": build_unet_depth_regression,
    "resnet18_depth_regression": build_resnet18_depth_regression,
    "unetplusplus_depth_regression": build_unetplusplus_depth_regression,
}


def build_comparison_model(config: Mapping[str, object]) -> nn.Module:
    try:
        name = str(config["model"]["name"])
        builder = BUILDERS[name]
    except (KeyError, TypeError) as exc:
        expected = ", ".join(sorted(BUILDERS))
        raise KeyError(f"Unsupported learned comparison model; expected one of {expected}") from exc
    return builder(config)
