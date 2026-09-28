"""Fixed-grid cubic B-spline layer used by terrain graph affinities."""

from __future__ import annotations

import math

import torch
from torch import nn


class KANLinear(nn.Module):
    """Additive univariate B-spline mapping on already bounded descriptors."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 4,
        spline_order: int = 3,
        *,
        spline_scale_init: float = 1.0,
        zero_output_init: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or grid_size < 2 or spline_order < 1:
            raise ValueError("Invalid KAN dimensions, grid size, or spline order")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        internal = torch.linspace(-1.0, 1.0, self.grid_size + 1)[1:-1]
        knots = torch.cat(
            (
                torch.full((self.spline_order + 1,), -1.0),
                internal,
                torch.full((self.spline_order + 1,), 1.0),
            )
        )
        self.register_buffer("knots", knots)
        self.n_basis = knots.numel() - self.spline_order - 1
        self.base_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        self.spline_coefficients = nn.Parameter(
            torch.empty(out_features, in_features, self.n_basis)
        )
        self.spline_scale = nn.Parameter(
            torch.full((out_features, in_features), float(spline_scale_init))
        )
        if zero_output_init:
            nn.init.zeros_(self.spline_coefficients)
        else:
            nn.init.normal_(self.spline_coefficients, mean=0.0, std=0.02)

    @property
    def effective_spline_coefficients(self) -> torch.Tensor:
        return self.spline_coefficients * self.spline_scale.unsqueeze(-1)

    def b_spline_basis(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the open-uniform B-spline basis in float precision."""

        values = inputs.clamp(-1.0, 1.0 - torch.finfo(inputs.dtype).eps)
        knots = self.knots.to(dtype=values.dtype, device=values.device)
        basis = (
            (values.unsqueeze(-1) >= knots[:-1])
            & (values.unsqueeze(-1) < knots[1:])
        ).to(values.dtype)
        for degree in range(1, self.spline_order + 1):
            count = knots.numel() - degree - 1
            left_denominator = knots[degree : degree + count] - knots[:count]
            right_denominator = (
                knots[degree + 1 : degree + 1 + count] - knots[1 : 1 + count]
            )
            left = torch.where(
                left_denominator > 0,
                (values.unsqueeze(-1) - knots[:count])
                / left_denominator.clamp_min(1.0e-12),
                torch.zeros_like(values.unsqueeze(-1)),
            )
            right = torch.where(
                right_denominator > 0,
                (knots[degree + 1 : degree + 1 + count] - values.unsqueeze(-1))
                / right_denominator.clamp_min(1.0e-12),
                torch.zeros_like(values.unsqueeze(-1)),
            )
            basis = left * basis[..., :count] + right * basis[..., 1 : count + 1]
        return basis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_contributions(inputs)[0]

    def forward_with_contributions(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"KANLinear expected {self.in_features} features, got {inputs.shape[-1]}"
            )
        base = torch.einsum("...i,oi->...o", inputs, self.base_weight)
        base = base + self.base_bias.to(base.dtype)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            basis = self.b_spline_basis(inputs.float())
            spline_terms = torch.einsum(
                "...ik,oik->...io", basis, self.spline_coefficients.float()
            )
            spline = (
                spline_terms * self.spline_scale.float().transpose(0, 1)
            ).sum(dim=-2)
        spline = spline.to(base.dtype)
        return base + spline, base, spline
