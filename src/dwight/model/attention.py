"""Multi-head causal self-attention with RoPE and Flash-Attention (SDPA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class MultiHeadCausalAttention(nn.Module):
    """Causal self-attention using RoPE positional encoding and SDPA."""

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
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Args:
            x:     (B, T, d_model)
            freqs: complex tensor (T, head_dim // 2) from precompute_freqs
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Apply rotary embeddings to Q and K
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Flash-Attention-compatible causal SDPA (uses FlashAttention on CUDA)
        attn_dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=attn_dropout, is_causal=True
        )  # (B, H, T, D)

        # Merge heads: (B, H, T, D) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.out_proj(out)
