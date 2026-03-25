"""Multi-head causal self-attention layer."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalAttention(nn.Module):
    """Scaled dot-product attention with a causal (autoregressive) mask."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads: (B, T, d_model) -> (B, H, T, D)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Attention scores (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Additive causal mask: upper triangle = -1e9
        causal_bias = (
            torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.float32), diagonal=1
            )
            * -1e9
        )
        scores = scores + causal_bias  # broadcast over batch and heads

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        # Merge heads: (B, H, T, D) -> (B, T, d_model)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.out_proj(out)
