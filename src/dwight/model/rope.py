"""Rotary Position Embedding (RoPE) helpers."""

from __future__ import annotations

import torch


def precompute_freqs(dim: int, max_seq_len: int) -> torch.Tensor:
    """Return complex frequency tensor of shape (max_seq_len, dim // 2).

    Each position gets a vector of complex numbers ``e^{i * m * theta_k}``
    where ``theta_k = 10000^{-2k/dim}`` following the original RoPE paper.

    *dim* is the number of dimensions to rotate (e.g. ``head_dim`` for standard
    RoPE, or ``qk_rope_dim`` for MLA's decoupled RoPE).
    """
    assert dim % 2 == 0, "dim must be even for RoPE"
    half = dim // 2
    # theta_k for k in [0, half)
    theta = 1.0 / (10_000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, theta)  # (max_seq_len, half)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a query or key tensor.

    Args:
        x:     Float tensor of shape (B, H, T, head_dim).
        freqs: Complex tensor of shape (T, head_dim // 2) — pre-sliced to T.

    Returns:
        Rotated tensor of the same shape and dtype as *x*.
    """
    # View last dim as pairs → complex numbers: (B, H, T, head_dim//2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs: (T, head_dim//2) → broadcast over (B, H, T, head_dim//2)
    rotated = x_complex * freqs.unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)
