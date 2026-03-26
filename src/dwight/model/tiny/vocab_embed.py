from __future__ import annotations

import torch
import torch.nn as nn


class FactoredVocabEmbed(nn.Module):
    """Low-rank vocabulary embedding used as an additional residual feature."""

    def __init__(self, vocab_size: int, ve_dim: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, ve_dim)
        self.proj = nn.Linear(ve_dim, d_model, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(tokens))
