"""Transformer block: pre-RMSNorm, RoPE causal attention, SwiGLU FFN, residuals."""

import torch
import torch.nn as nn

from .attention import MultiHeadCausalAttention
from .feed_forward import FeedForwardNetwork


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no bias)."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


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
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.drop1(self.attn(self.norm1(x), freqs))
        # Pre-norm + FFN + residual
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x
