from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyReluSquaredFF(nn.Module):
    """Dense MLP with the LeakyReLU(0.5)^2 activation from tinymodel.md."""

    def __init__(self, d_model: int, dff: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, dff, bias=False)
        self.proj = nn.Linear(dff, d_model, bias=False)

    def activated(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.fc(x), negative_slope=0.5).square()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.activated(x))
