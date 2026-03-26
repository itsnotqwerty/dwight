from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..rope import precompute_freqs


def apply_partial_rope(
    x: torch.Tensor, freqs: torch.Tensor, rope_dims: int
) -> torch.Tensor:
    rotated = x[..., :rope_dims]
    passthrough = x[..., rope_dims:]
    rotated_complex = torch.view_as_complex(
        rotated.float().reshape(*rotated.shape[:-1], -1, 2)
    )
    rope_freqs = freqs[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    rotated = torch.view_as_real(rotated_complex * rope_freqs).flatten(-2).to(x.dtype)
    return torch.cat((rotated, passthrough), dim=-1)


class GroupedQueryAttention(nn.Module):
    """Grouped-query causal attention with optional shared KV input for XSA."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        rope_dims: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.rope_dims = rope_dims
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "freqs", precompute_freqs(rope_dims, max_seq_len), persistent=False
        )

    def _expand_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        repeat_factor = self.num_heads // self.num_kv_heads
        return tensor.repeat_interleave(repeat_factor, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        kv_source: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = apply_partial_rope(q, self.freqs, self.rope_dims)

        if kv_source is None:
            k = (
                self.k_proj(x)
                .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(x)
                .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = apply_partial_rope(k, self.freqs, self.rope_dims)
            kv = (k, v)
        else:
            kv = kv_source

        k, v = kv
        expanded_k = self._expand_kv(k)
        expanded_v = self._expand_kv(v)
        attn_dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            expanded_k,
            expanded_v,
            attn_mask=None,
            dropout_p=attn_dropout,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out), kv
