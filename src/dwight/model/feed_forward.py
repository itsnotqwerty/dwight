"""Position-wise feed-forward network with SwiGLU activation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """SwiGLU FFN: three projections, SiLU gate (no bias).

    Hidden dimension is set to ``2 * dff // 3`` so that the total parameter
    count is equivalent to a standard two-matrix FFN of width *dff*.
    """

    def __init__(self, d_model: int, dff: int) -> None:
        super().__init__()
        hidden = 2 * dff // 3
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj   = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
