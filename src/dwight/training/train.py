"""Training loop with Adam optimiser + linear warmup / cosine-decay LR."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..config import ModelConfig
from ..model.transformer import GPTModel
from ..tokenizer import TiktokenWrapper
from .dataset import DEFAULT_ARCHIVE, chan_dataloader


def _cosine_decay_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float = 1e-5,
) -> float:
    """Linear warmup then cosine decay to *min_lr*."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(
    epochs: int = 3,
    batch_size: int = 8,
    max_lr: float = 3e-4,
    warmup_steps: int = 100,
    checkpoint_dir: str = "checkpoints",
    max_steps: int | None = None,
    data: str = DEFAULT_ARCHIVE,
) -> GPTModel:
    """Train a GPT model on the 4chan /pol/ archive and return the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = ModelConfig()
    tokenizer = TiktokenWrapper()

    print(f"Streaming dataset from {data} …")
    loader = chan_dataloader(
        data,
        tokenizer,
        seq_len=config.max_seq_len,
        batch_size=batch_size,
    )
    # The archive is too large to count upfront; use max_steps when known.
    total_steps = max_steps if max_steps is not None else warmup_steps * 1000

    model = GPTModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.95))

    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

        for inp, tgt in pbar:
            if max_steps is not None and global_step >= max_steps:
                break

            inp = inp.to(device)
            tgt = tgt.to(device)

            lr = _cosine_decay_lr(global_step, warmup_steps, total_steps, max_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            logits = model(inp)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                tgt.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1
            avg_loss = running_loss / n_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

        model.eval()
        avg_loss = running_loss / max(n_batches, 1)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        print(f"Epoch {epoch} complete — loss: {avg_loss:.4f}")

        if max_steps is not None and global_step >= max_steps:
            break

    return model
