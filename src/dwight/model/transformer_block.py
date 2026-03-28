"""Transformer block: pre-RMSNorm, RoPE causal attention, SwiGLU FFN, residuals.

Supports optional MLA (Multi-Head Latent Attention) and MoE (Mixture-of-Experts)
via keyword arguments.  The default (all flags False) is identical to the
original dense MHA + SwiGLU architecture and is fully backward compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadCausalAttention
from .feed_forward import FeedForwardNetwork
from .mla import MultiHeadLatentAttention
from .moe import MoEFeedForward


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
    """Pre-norm transformer block (attention + FFN with residuals).

    Args:
        d_model, num_heads, dff, dropout: standard hyperparameters.
        use_mla:           Replace MHA with Multi-Head Latent Attention.
        kv_latent_dim:     KV compression bottleneck (MLA only).
        q_latent_dim:      Q compression bottleneck (MLA only).
        qk_rope_dim:       Per-head RoPE dimension appended to Q and K (MLA only).
        use_moe:           Replace dense FFN with Mixture-of-Experts.
        num_experts:       Total routed experts (MoE only).
        num_active_experts: Top-K active experts per token (MoE only).
        num_shared_experts: Always-active shared experts (MoE only).
        expert_hidden_dim: SwiGLU hidden dim inside each expert (MoE only).

    Returns from forward:
        (x, aux_loss) — aux_loss is a scalar tensor (zero when use_moe=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float = 0.1,
        *,
        use_mla: bool = False,
        kv_latent_dim: int = 512,
        q_latent_dim: int = 1536,
        qk_rope_dim: int = 64,
        use_moe: bool = False,
        num_experts: int = 8,
        num_active_experts: int = 2,
        num_shared_experts: int = 1,
        expert_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.use_moe = use_moe

        self.norm1 = RMSNorm(d_model)
        if use_mla:
            self.attn: nn.Module = MultiHeadLatentAttention(
                d_model, num_heads, kv_latent_dim, q_latent_dim, qk_rope_dim, dropout
            )
        else:
            self.attn = MultiHeadCausalAttention(d_model, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        if use_moe:
            self.ffn: nn.Module = MoEFeedForward(
                d_model,
                num_experts,
                num_active_experts,
                num_shared_experts,
                expert_hidden_dim,
            )
        else:
            self.ffn = FeedForwardNetwork(d_model, dff)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, freqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns:
        (x, aux_loss) — aux_loss is zero when use_moe=False.
        """
        # Pre-norm + attention + residual
        # MLA returns (out, kv_cache); standard attention returns just out.
        attn_out = self.attn(self.norm1(x), freqs)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]  # discard kv_cache during training
        x = x + self.drop1(attn_out)

        # Pre-norm + FFN + residual
        ffn_out = self.ffn(self.norm2(x))
        if self.use_moe:
            ffn_tensor, aux_loss = ffn_out
        else:
            ffn_tensor = ffn_out
            aux_loss = x.new_zeros(())  # scalar zero on same device/dtype
        x = x + self.drop2(ffn_tensor)

        return x, aux_loss
