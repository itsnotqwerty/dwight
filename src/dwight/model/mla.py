"""Multi-Head Latent Attention (MLA) — DeepSeek-style compressed KV with decoupled RoPE.

Architecture summary
--------------------
Standard attention caches (K, V) per token — O(n_heads * head_dim) per position.
MLA instead caches (c_KV, K_rope): a small latent of size ``kv_latent_dim`` plus
the absolute rope keys of size ``qk_rope_dim``.  Full K and V are recovered at
inference by up-projecting c_KV on the fly, eliminating the large KV cache.

During attention the query and key each have two concatenated parts:
  * A *content* part  (head_dim dims)  — derived from the latent projection
  * A *rope* part     (qk_rope_dim dims) — RoPE-rotated, decoupled from the latent

The effective scale is 1/√(head_dim + qk_rope_dim), which PyTorch SDPA computes
automatically from the last dimension of Q/K.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class MultiHeadLatentAttention(nn.Module):
    """Causal MLA with decoupled RoPE and compressed KV latent."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kv_latent_dim: int,
        q_latent_dim: int,
        qk_rope_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_latent_dim = kv_latent_dim
        self.q_latent_dim = q_latent_dim
        self.qk_rope_dim = qk_rope_dim
        self.dropout = dropout

        # ── KV compression path ───────────────────────────────────────────────
        # c_KV = W_DKV(x)  — the latent that is cached at inference
        self.W_DKV = nn.Linear(d_model, kv_latent_dim, bias=False)
        self.W_UK = nn.Linear(kv_latent_dim, num_heads * self.head_dim, bias=False)
        self.W_UV = nn.Linear(kv_latent_dim, num_heads * self.head_dim, bias=False)
        # Decoupled RoPE keys — shared across all heads
        self.W_KR = nn.Linear(d_model, qk_rope_dim, bias=False)

        # ── Q compression path ────────────────────────────────────────────────
        self.W_DQ = nn.Linear(d_model, q_latent_dim, bias=False)
        self.W_UQ = nn.Linear(q_latent_dim, num_heads * self.head_dim, bias=False)
        # Per-head rope queries
        self.W_QR = nn.Linear(q_latent_dim, num_heads * qk_rope_dim, bias=False)

        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x:     (B, T, d_model)
            freqs: complex tensor (T, qk_rope_dim // 2) from precompute_freqs

        Returns:
            (output, kv_cache) where:
              output:   (B, T, d_model)
              kv_cache: (c_KV, K_rope_single) — c_KV is (B, T, kv_latent_dim),
                        K_rope_single is (B, T, qk_rope_dim).  Cached at inference
                        to avoid recomputing the full KV on every step.
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # ── KV path ───────────────────────────────────────────────────────────
        c_KV = self.W_DKV(x)  # (B, T, kv_latent_dim) — the cached latent
        K_content = self.W_UK(c_KV).view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        V = self.W_UV(c_KV).view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)

        # Decoupled RoPE for keys — one set of rotated dims shared across heads
        K_rope_raw = self.W_KR(x).unsqueeze(1)  # (B, 1, T, qk_rope_dim)
        K_rope = apply_rope(K_rope_raw, freqs)  # (B, 1, T, qk_rope_dim)
        K_rope = K_rope.expand(-1, H, -1, -1)  # (B, H, T, qk_rope_dim)

        # ── Q path ────────────────────────────────────────────────────────────
        c_Q = self.W_DQ(x)  # (B,T,q_latent)
        Q_content = self.W_UQ(c_Q).view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        Q_rope = (
            self.W_QR(c_Q).view(B, T, H, self.qk_rope_dim).transpose(1, 2)
        )  # (B, H, T, qk_rope_dim)
        Q_rope = apply_rope(Q_rope, freqs)  # (B, H, T, qk_rope_dim)

        # ── Attention ─────────────────────────────────────────────────────────
        # Concatenate content + rope dims for Q and K.
        # V stays at head_dim; SDPA automatically scales by 1/√(D + qk_rope_dim).
        Q = torch.cat([Q_content, Q_rope], dim=-1)  # (B, H, T, D+qk_rope_dim)
        K = torch.cat([K_content, K_rope], dim=-1)  # (B, H, T, D+qk_rope_dim)

        attn_dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=attn_dropout, is_causal=True
        )  # (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)

        # KV cache stores the small latent + rope keys rather than full K/V
        kv_cache = (c_KV, K_rope[:, 0, :, :])  # (B,T,kv_latent_dim), (B,T,qk_rope_dim)
        return self.out_proj(out), kv_cache
