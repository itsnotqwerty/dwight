"""Position-wise feed-forward network layer."""

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """Two-layer FFN with GELU activation."""

    def __init__(self, d_model: int, dff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Linear(dff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
