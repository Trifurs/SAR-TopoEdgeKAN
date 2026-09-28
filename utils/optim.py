"""Configuration-consuming optimizer and scheduler builders."""

from __future__ import annotations

import math
import torch


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    name = str(optimizer_config["name"]).lower()
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer.name {name!r}")
    base_lr = float(optimizer_config["learning_rate"])
    base_decay = float(optimizer_config["weight_decay"])
    betas = (
        float(optimizer_config.get("beta1", 0.9)),
        float(optimizer_config.get("beta2", 0.999)),
    )
    epsilon = float(optimizer_config.get("epsilon", 1.0e-8))
    amsgrad = bool(optimizer_config.get("amsgrad", False))
    kan_lr = float(optimizer_config.get("kan_lr_multiplier", 1.0))
    kan_decay = float(optimizer_config.get("kan_weight_decay", base_decay))
    head_lr = float(optimizer_config.get("head_lr_multiplier", 1.0))
    adapter_lr = float(optimizer_config.get("adapter_lr_multiplier", 1.0))
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized = parameter_name.lower()
        no_decay = parameter.ndim == 1 or normalized.endswith("bias") or "norm" in normalized or "gamma" in normalized
        is_adapter = any(
            marker in normalized
            for marker in (
                "context_adapter",
                "skip_merge",
                "global_context",
                "hydrostatic_adapter",
                "global_depth_calibration_adapter",
                "depth_range_calibration",
            )
        )
        lr_multiplier = (
            kan_lr
            if "spline_coefficients" in normalized
            else adapter_lr
            if is_adapter
            else head_lr
            if "heads" in normalized
            else 1.0
        )
        decay = 0.0 if no_decay else kan_decay if "spline_coefficients" in normalized else base_decay
        groups.setdefault((lr_multiplier, decay), []).append(parameter)
    return torch.optim.AdamW([
        {"params": parameters, "lr": base_lr * multiplier, "weight_decay": decay,
         "lr_multiplier": multiplier}
        for (multiplier, decay), parameters in groups.items()
    ], lr=base_lr, betas=betas, eps=epsilon, amsgrad=amsgrad)


def build_scheduler(optimizer, config: dict, total_steps: int, warmup_steps: int):
    scheduler_config = config["scheduler"]
    name = str(scheduler_config["name"]).lower()
    minimum_ratio = float(scheduler_config.get("minimum_learning_rate", 0.0)) / float(config["optimizer"]["learning_rate"])
    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        if name == "constant_with_warmup":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    if name in {"cosine_with_warmup", "constant_with_warmup"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    raise ValueError(f"Unsupported scheduler.name {name!r}")
