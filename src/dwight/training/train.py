"""Training loop with Adam optimiser + linear warmup / cosine-decay LR."""

from __future__ import annotations

import math
import os
import time

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
    resume: bool = False,
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
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    _CHECKPOINT_INTERVAL_SECS = 30 * 60  # 30 minutes
    _CHECKPOINT_INTERVAL_STEPS = 10_000

    def _save_checkpoint(avg_loss: float, step: int, completed_epochs: int) -> None:
        model.eval()
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": step,
                "completed_epochs": completed_epochs,
            },
            checkpoint_path,
        )
        model.train()
        print(f"\nCheckpoint saved at step {step} — loss: {avg_loss:.4f}")

    global_step = 0
    last_ckpt_step = 0
    last_ckpt_time = time.monotonic()
    start_epoch = 1

    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            global_step = ckpt.get("global_step", 0)
            last_ckpt_step = global_step
            last_ckpt_time = time.monotonic()
            start_epoch = ckpt.get("completed_epochs", 0) + 1
            print(
                f"Resumed from checkpoint at step {global_step} "
                f"(epoch {start_epoch}/{epochs})"
            )
        else:
            # Old-format checkpoint: bare state dict — load weights only, reset progress.
            model.load_state_dict(ckpt)
            print("Resumed model weights from legacy checkpoint (step counter reset).")
    elif resume:
        print(
            "Warning: --resume requested but no checkpoint found — starting from scratch."
        )

    for epoch in range(start_epoch, epochs + 1):
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

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(inp)  # (B, T, vocab_size) — bf16, halving peak VRAM
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

            steps_since_ckpt = global_step - last_ckpt_step
            time_since_ckpt = time.monotonic() - last_ckpt_time
            if (
                steps_since_ckpt >= _CHECKPOINT_INTERVAL_STEPS
                or time_since_ckpt >= _CHECKPOINT_INTERVAL_SECS
            ):
                _save_checkpoint(avg_loss, global_step, completed_epochs=epoch - 1)
                last_ckpt_step = global_step
                last_ckpt_time = time.monotonic()

        avg_loss = running_loss / max(n_batches, 1)
        _save_checkpoint(avg_loss, global_step, completed_epochs=epoch)
        last_ckpt_step = global_step
        last_ckpt_time = time.monotonic()
        print(f"Epoch {epoch} complete — loss: {avg_loss:.4f}")

        if max_steps is not None and global_step >= max_steps:
            break

    return model
