from __future__ import annotations

import torch
import torch.nn as nn

from ..transformer_block import RMSNorm
from .attention import GroupedQueryAttention
from .feed_forward import LeakyReluSquaredFF


class TinyTransformerBlock(nn.Module):
    """Tiny architecture block with grouped-query attention and LeakyReLU^2 MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dff: int,
        rope_dims: int,
        max_seq_len: int,
        layer_index: int,
        dropout: float = 0.0,
        ln_scale: bool = True,
    ) -> None:
        super().__init__()
        self.layer_scale = (layer_index + 1) ** -0.5 if ln_scale else 1.0
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_dims=rope_dims,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.ffn = LeakyReluSquaredFF(d_model=d_model, dff=dff)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_source: tuple[torch.Tensor, torch.Tensor] | None = None,
        vocab_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_in = self.norm1(x) * self.layer_scale
        attn_out, kv = self.attn(attn_in, kv_source=kv_source)
        x = x + self.drop1(attn_out)
        if vocab_residual is not None:
            x = x + vocab_residual
        ffn_in = self.norm2(x) * self.layer_scale
        x = x + self.drop2(self.ffn(ffn_in))
        return x, kv
