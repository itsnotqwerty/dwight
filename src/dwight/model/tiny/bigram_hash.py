from __future__ import annotations

import torch
import torch.nn as nn


class BigramHashEmbedding(nn.Module):
    """Fixed hash over adjacent token pairs projected through an embedding table."""

    def __init__(self, vocab_size: int, bigram_vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.bigram_vocab_size = bigram_vocab_size
        self.embedding = nn.Embedding(bigram_vocab_size, d_model)
        self.register_buffer("_prime_a", torch.tensor(1_315_423_911, dtype=torch.long), persistent=False)
        self.register_buffer("_prime_b", torch.tensor(2_654_435_761, dtype=torch.long), persistent=False)

    def hashed_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        previous = torch.zeros_like(tokens)
        previous[:, 1:] = tokens[:, :-1]
        hashed = ((previous * self._prime_a) ^ (tokens * self._prime_b)) % self.bigram_vocab_size
        return hashed.long()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.hashed_ids(tokens))
