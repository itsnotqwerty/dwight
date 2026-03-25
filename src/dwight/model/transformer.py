"""Full GPT-style causal language model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import ModelConfig
from .transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """GPT-style transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model, config.num_heads, config.dff, config.dropout
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model, eps=1e-5)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: int64 tensor of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device)

        tok_emb = self.token_embedding(x)  # (B, T, d_model)
        pos_emb = self.pos_embedding(positions)  # (T, d_model) – broadcast

        h = self.emb_drop(tok_emb + pos_emb)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        return self.lm_head(h)  # (B, T, vocab_size)

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
                logits = self(ctx)  # (1, T, vocab_size)
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
