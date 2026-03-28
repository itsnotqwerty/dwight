"""Mixture-of-Experts feed-forward layer (DeepSeek-V2 style).

Architecture
------------
* A learned *router* selects the top-K experts for each token.
* ``num_experts`` routed experts (only K active per token).
* ``num_shared_experts`` always-active experts added to every token's output.
* Each expert is a standard SwiGLU FFN with a smaller hidden dimension
  (``expert_hidden_dim``) than the dense equivalent, keeping active FLOPs
  lower than a single large FFN while increasing total parameter capacity.
* A load-balancing auxiliary loss (Switch Transformer / DeepSeek-V2 style)
  is returned alongside the output so the training loop can penalise uneven
  expert utilisation.

Auxiliary loss formula
----------------------
  L_aux = N · Σᵢ fᵢ · Pᵢ

where N = num_experts, fᵢ = fraction of tokens routed to expert i (no
gradient), and Pᵢ = mean router softmax probability for expert i (gradient
flows through Pᵢ so the router learns to balance load).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SwiGLUExpert(nn.Module):
    """Small SwiGLU FFN used as a single MoE expert."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFeedForward(nn.Module):
    """Mixture-of-Experts FFN with load-balance auxiliary loss.

    Args:
        d_model:           Residual stream width.
        num_experts:       Total number of routed experts.
        num_active:        Top-K experts selected per token.
        num_shared:        Always-active shared experts (not gated).
        expert_hidden_dim: SwiGLU hidden dimension inside each expert.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_active: int,
        num_shared: int,
        expert_hidden_dim: int,
    ) -> None:
        super().__init__()
        assert num_active <= num_experts, "num_active cannot exceed num_experts"
        self.num_experts = num_experts
        self.num_active = num_active

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_SwiGLUExpert(d_model, expert_hidden_dim) for _ in range(num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [_SwiGLUExpert(d_model, expert_hidden_dim) for _ in range(num_shared)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: (B, T, d_model)

        Returns:
            (output, aux_loss) — output has the same shape as x; aux_loss is a
            scalar tensor that the caller should scale by ``moe_aux_loss_coeff``
            and add to the cross-entropy loss.
        """
        B, T, D = x.shape
        flat = x.view(-1, D)  # (N, D) where N = B*T
        N = flat.size(0)

        # ── Router ────────────────────────────────────────────────────────────
        logits = self.router(flat)  # (N, num_experts)
        topk_weights, topk_idx = torch.topk(logits, self.num_active, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)  # (N, num_active)

        # ── Load-balance auxiliary loss ───────────────────────────────────────
        # f_i: fraction of top-K selections going to each expert (no gradient)
        with torch.no_grad():
            token_count = torch.zeros(
                self.num_experts, dtype=flat.dtype, device=flat.device
            )
            ones = torch.ones(N * self.num_active, dtype=flat.dtype, device=flat.device)
            token_count.scatter_add_(0, topk_idx.flatten(), ones)
            f = token_count / (N * self.num_active)  # fraction per expert

        # P_i: mean router softmax prob per expert (gradient flows here)
        P = F.softmax(logits, dim=-1).mean(0)  # (num_experts,)
        aux_loss = (f * P).sum() * self.num_experts

        # ── Expert dispatch ───────────────────────────────────────────────────
        out = torch.zeros_like(flat)
        for e_idx, expert in enumerate(self.experts):
            # Which tokens have this expert in their top-K?
            expert_mask = topk_idx == e_idx  # (N, num_active) bool
            token_mask = expert_mask.any(dim=-1)  # (N,)
            if not token_mask.any():
                continue
            selected = flat[token_mask]  # (k, D)
            expert_out = expert(selected)  # (k, D)
            # The routing weight for this expert at each selected token
            weights = (
                expert_mask[token_mask].to(topk_weights.dtype)
                * topk_weights[token_mask]
            ).sum(
                dim=-1, keepdim=True
            )  # (k, 1)
            out[token_mask] = out[token_mask] + expert_out * weights

        # ── Shared experts (always active) ────────────────────────────────────
        for shared_expert in self.shared_experts:
            out = out + shared_expert(flat)

        return out.view(B, T, D), aux_loss
