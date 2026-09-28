"""Mask-agnostic patch-level depth calibration for SAR-TopoEdgeKAN."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class GlobalEvidenceCalibration(nn.Module):
    """Predict bounded patch-level depth-logit scale and bias residuals.

    Mean, standard deviation, minimum, and maximum pooling capture event-level
    acquisition and terrain context. The zero-initialized output projection
    preserves a transferred checkpoint's prediction at initialization.
    """

    def __init__(
        self,
        input_channels: int,
        *,
        hidden_channels: int = 96,
        dropout: float = 0.05,
        maximum_scale_residual: float = 0.5,
        maximum_bias_residual: float = 4.0,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("global calibration channel counts must be positive")
        if maximum_scale_residual <= 0.0 or maximum_bias_residual <= 0.0:
            raise ValueError("global calibration residual bounds must be positive")
        pooled_channels = 4 * int(input_channels)
        self.network = nn.Sequential(
            nn.LayerNorm(pooled_channels),
            nn.Linear(pooled_channels, int(hidden_channels)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_channels), 2),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.maximum_scale_residual = float(maximum_scale_residual)
        self.maximum_bias_residual = float(maximum_bias_residual)

    @staticmethod
    def _moments(
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = evidence.mean(dim=(-2, -1))
        variance = evidence.square().mean(dim=(-2, -1)) - mean.square()
        return (
            mean,
            variance.clamp_min(1.0e-8).sqrt(),
            evidence.amin(dim=(-2, -1)),
            evidence.amax(dim=(-2, -1)),
        )

    def forward(
        self, evidence: torch.Tensor | Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(evidence, torch.Tensor):
            groups = (self._moments(evidence),)
        else:
            if not evidence:
                raise ValueError("global calibration evidence groups cannot be empty")
            groups = tuple(self._moments(value) for value in evidence)
        # Preserve dense-concatenation feature order: all channel means,
        # followed by all standard deviations, minima, and maxima.
        pooled = torch.cat(
            tuple(
                torch.cat(tuple(group[index] for group in groups), dim=1)
                for index in range(4)
            ),
            dim=1,
        )
        raw_scale, raw_bias = self.network(pooled).chunk(2, dim=1)
        scale_bound = self.maximum_scale_residual
        bias_bound = self.maximum_bias_residual
        scale = scale_bound * torch.tanh(raw_scale / scale_bound)
        bias = bias_bound * torch.tanh(raw_bias / bias_bound)
        return scale[:, :, None, None], bias[:, :, None, None]
