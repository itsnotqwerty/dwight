"""Transformer block: pre-LayerNorm, causal attention, FFN, residuals."""

import torch
import torch.nn as nn

from .attention import MultiHeadCausalAttention
from .feed_forward import FeedForwardNetwork


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (attention + FFN with residuals)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = MultiHeadCausalAttention(d_model, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.drop1(self.attn(self.norm1(x)))
        # Pre-norm + FFN + residual
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x
