from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import TinyModelConfig


def _chunked(tokens: torch.Tensor, chunk_size: int) -> list[torch.Tensor]:
    return [
        tokens[i : i + chunk_size]
        for i in range(0, tokens.numel(), chunk_size)
        if tokens[i : i + chunk_size].numel() > 1
    ]


def _score_chunk(model: torch.nn.Module, chunk: torch.Tensor, vocab_size: int) -> float:
    with torch.inference_mode():
        inp = chunk[:-1].unsqueeze(0)
        tgt = chunk[1:].unsqueeze(0)
        logits = model(inp)
        loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1))
    return float(loss.item() / math.log(2))


def test_time_train(
    model: torch.nn.Module,
    val_tokens: list[int] | torch.Tensor,
    config: TinyModelConfig,
) -> list[float]:
    """Score-first test-time training over sequential token chunks."""
    device = next(model.parameters()).device
    tensor = torch.as_tensor(val_tokens, dtype=torch.long, device=device)
    chunks = _chunked(tensor, config.ttt_chunk_tokens)
    if not chunks:
        return []

    results: list[float] = []
    model_was_training = model.training
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.ttt_lr,
        momentum=config.ttt_momentum,
    )
    model.eval()
    for index, chunk in enumerate(chunks):
        results.append(_score_chunk(model, chunk, config.vocab_size))
        if index == len(chunks) - 1:
            continue
        model.train()
        for _ in range(config.ttt_epochs):
            inp = chunk[:-1].unsqueeze(0)
            tgt = chunk[1:].unsqueeze(0)
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), tgt.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.ttt_grad_clip)
            optimizer.step()
        model.eval()
    model.train(model_was_training)
    return results


test_time_train.__test__ = False
