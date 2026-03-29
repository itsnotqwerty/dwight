"""Full GPT-style causal language model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt
from typing import cast

from ..config import ModelConfig
from .rope import precompute_freqs
from .transformer_block import RMSNorm, TransformerBlock


class GPTModel(nn.Module):
    """GPT-style transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self._gradient_checkpointing = False

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.dff,
                    config.dropout,
                    use_mla=config.use_mla,
                    kv_latent_dim=config.kv_latent_dim,
                    q_latent_dim=config.q_latent_dim,
                    qk_rope_dim=config.qk_rope_dim,
                    use_moe=config.use_moe,
                    num_experts=config.num_experts,
                    num_active_experts=config.num_active_experts,
                    num_shared_experts=config.num_shared_experts,
                    expert_hidden_dim=config.expert_hidden_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        head_dim = config.d_model // config.num_heads
        # For MLA use qk_rope_dim as the rope dimension; for standard MHA use head_dim.
        rope_dim = config.qk_rope_dim if config.use_mla else head_dim
        # Precomputed RoPE frequencies — registered as a non-parameter buffer so
        # they travel with the model to GPU and are excluded from optimizer state.
        self.register_buffer(
            "freqs",
            precompute_freqs(rope_dim, config.max_seq_len),
            persistent=False,
        )

        self.apply(self._init_weights)
        # Tie input and output embeddings so they share parameters.
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style weight initialisation with residual-path depth scaling."""
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            std = 0.02
            # Residual projections — scale down by 1/√(2·num_layers) to keep
            # the residual stream variance constant with depth (GPT-2 §2.3).
            if module is self.lm_head or any(
                module is getattr(blk.attn, "out_proj", None)
                or module is getattr(blk.ffn, "down_proj", None)
                or any(
                    module is expert.down_proj
                    for expert_list in (
                        getattr(blk.ffn, "experts", []),
                        getattr(blk.ffn, "shared_experts", []),
                    )
                    for expert in expert_list
                )
                for blk in self.blocks
            ):
                std /= (2 * self.config.num_layers) ** 0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: int64 tensor of shape (batch, seq_len).

        Returns:
            (logits, aux_loss) where logits is (batch, seq_len, vocab_size) and
            aux_loss is a scalar MoE load-balance loss (zero when use_moe=False).
        """
        B, T = x.shape

        h = self.emb_drop(self.token_embedding(x))  # (B, T, d_model)

        freqs = cast(torch.Tensor, self.freqs)[:T]  # slice to current sequence length
        total_aux = h.new_zeros(())  # scalar accumulator for MoE aux losses
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                h, aux = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    _ckpt(block, h, freqs, use_reentrant=False),
                )
            else:
                h, aux = block(h, freqs)
            total_aux = total_aux + aux

        h = self.ln_f(h)
        return self.lm_head(h), total_aux  # (B, T, vocab_size), scalar

    def enable_gradient_checkpointing(self) -> None:
        """Recompute activations during backward to reduce peak VRAM usage."""
        self._gradient_checkpointing = True

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        """Yield token IDs one at a time (autoregressive)."""
        ids = list(prompt_ids)
        device = next(self.parameters()).device

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = ids[-self.config.max_seq_len :]
                ctx = torch.tensor([context], dtype=torch.long, device=device)  # (1, T)
                logits, _ = self(ctx)  # (1, T, vocab_size) — discard aux_loss
                last_logits = logits[0, -1].float().cpu().numpy().astype(np.float64)

                if temperature <= 0.0:
                    token_id = int(np.argmax(last_logits))
                else:
                    last_logits = last_logits / temperature
                    token_id = _sample_top_p(last_logits, top_p)

                ids.append(token_id)
                yield token_id


def _sample_top_p(logits: np.ndarray, top_p: float) -> int:
    """Nucleus (top-p) sampling from a logits array."""
    # Stable softmax
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    # Sort descending
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)

    # Zero out tokens beyond the nucleus
    cutoff = (cumsum - sorted_probs) > top_p
    sorted_probs[cutoff] = 0.0
    sorted_probs /= sorted_probs.sum()

    sampled = np.random.choice(len(sorted_probs), p=sorted_probs)
    return int(sorted_idx[sampled])
