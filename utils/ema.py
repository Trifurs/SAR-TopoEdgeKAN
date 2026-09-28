"""Checkpointable exponential moving average updated after successful steps."""

from __future__ import annotations

from collections import OrderedDict
import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, warmup_steps: int = 0) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay, self.warmup_steps, self.updates = float(decay), int(warmup_steps), 0
        unwrapped = model.module if hasattr(model, "module") else model
        self.shadow = OrderedDict((k, v.detach().clone()) for k, v in unwrapped.state_dict().items())

    def current_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return min(self.decay, self.decay * self.updates / self.warmup_steps)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        unwrapped = model.module if hasattr(model, "module") else model
        self.updates += 1
        decay = self.current_decay()
        for name, value in unwrapped.state_dict().items():
            target = self.shadow[name]
            if torch.is_floating_point(target):
                target.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                target.copy_(value)

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay,
                "warmup_steps": self.warmup_steps, "updates": self.updates}

    def load_state_dict(self, state) -> None:
        self.decay = float(state["decay"])
        self.warmup_steps = int(state.get("warmup_steps", 0))
        self.updates = int(state.get("updates", 0))
        self.shadow = OrderedDict((k, v.detach().clone()) for k, v in state["shadow"].items())

    @torch.no_grad()
    def reset_from_model(self, model: torch.nn.Module) -> None:
        """Make EMA exactly match already-loaded raw model weights.

        This is required when a checkpoint has no EMA payload: keeping the shadow
        initialized before checkpoint loading would evaluate unrelated parameters.
        """

        unwrapped = model.module if hasattr(model, "module") else model
        self.shadow = OrderedDict(
            (name, value.detach().clone())
            for name, value in unwrapped.state_dict().items()
        )
        self.updates = 0

    def model_state_dict(self):
        return self.shadow

    def copy_to(self, model: torch.nn.Module) -> None:
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.load_state_dict(self.shadow, strict=True)

    def swap_in(self, model: torch.nn.Module):
        return _EMASwap(self, model)


def restore_ema_after_checkpoint_load(
    ema: ModelEMA | None,
    checkpoint: dict,
    model: torch.nn.Module,
) -> bool:
    """Restore saved EMA, or safely initialize it from loaded raw weights.

    Returns ``True`` when an EMA payload was restored and ``False`` otherwise.
    """

    if ema is None:
        return False
    state = checkpoint.get("ema")
    if state is not None:
        ema.load_state_dict(state)
        return True
    ema.reset_from_model(model)
    return False


class _EMASwap:
    def __init__(self, ema: ModelEMA, model: torch.nn.Module) -> None:
        self.ema, self.model, self.original = ema, model, None

    def __enter__(self):
        unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        self.original = OrderedDict((k, v.detach().clone()) for k, v in unwrapped.state_dict().items())
        self.ema.copy_to(self.model)
        return self.model

    def __exit__(self, *_):
        unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        unwrapped.load_state_dict(self.original, strict=True)
