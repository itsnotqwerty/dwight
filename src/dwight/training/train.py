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


def _autocast_kwargs(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"enabled": False}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"enabled": True, "dtype": dtype}


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
    warmup_steps: int = 1000,
    checkpoint_dir: str = "checkpoints",
    max_steps: int | None = None,
    data: str = DEFAULT_ARCHIVE,
    resume: bool = False,
    grad_accum_steps: int = 1,
) -> GPTModel:
    """Train a GPT model on the 4chan /pol/ archive and return the trained model."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        amp_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        print(f"AMP enabled on CUDA ({amp_dtype})")
    config = ModelConfig()
    tokenizer = TiktokenWrapper()
    autocast_kwargs = _autocast_kwargs(device)

    print(f"Streaming dataset from {data} …")
    loader = chan_dataloader(
        data,
        tokenizer,
        seq_len=config.max_seq_len,
        batch_size=batch_size,
    )
    # The archive is too large to count upfront; use max_steps when known.
    total_steps = max_steps if max_steps is not None else warmup_steps * 100

    model = GPTModel(config).to(device)
    if device.type == "cuda":
        model.enable_gradient_checkpointing()
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
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        try:
            model.load_state_dict(state_dict, strict=True)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
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
                print(
                    "Resumed model weights from legacy checkpoint (step counter reset)."
                )
        except RuntimeError as exc:
            print(
                f"Warning: checkpoint is incompatible with the current model architecture "
                f"({exc}). Starting from scratch with fresh weights."
            )
    elif resume:
        print(
            "Warning: --resume requested but no checkpoint found — starting from scratch."
        )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        accum_loss = 0.0
        micro_step = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

        for inp, tgt in pbar:
            if max_steps is not None and global_step >= max_steps:
                break

            try:
                inp = inp.to(device)
                tgt = tgt.to(device)

                with torch.autocast(device_type=device.type, **autocast_kwargs):
                    logits = model(inp)
                    loss = F.cross_entropy(
                        logits.view(-1, config.vocab_size),
                        tgt.view(-1),
                    )
            except RuntimeError as exc:
                if device.type == "cuda" and "cuda out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM during training. Try --batch-size 1 and increase "
                        "--grad-accum-steps, or reduce max_seq_len in ModelConfig."
                    ) from exc
                raise

            (loss / grad_accum_steps).backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % grad_accum_steps != 0:
                continue

            lr = _cosine_decay_lr(global_step, warmup_steps, total_steps, max_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += accum_loss / grad_accum_steps
            n_batches += 1
            global_step += 1
            accum_loss = 0.0
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

        # Flush leftover accumulated gradients if the epoch didn't end on a full accumulation window.
        if accum_loss > 0.0:
            remainder = micro_step % grad_accum_steps or grad_accum_steps
            lr = _cosine_decay_lr(global_step, warmup_steps, total_steps, max_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += accum_loss / remainder
            n_batches += 1
            global_step += 1

        avg_loss = running_loss / max(n_batches, 1)
        _save_checkpoint(avg_loss, global_step, completed_epochs=epoch)
        last_ckpt_step = global_step
        last_ckpt_time = time.monotonic()
        print(f"Epoch {epoch} complete — loss: {avg_loss:.4f}")

        if max_steps is not None and global_step >= max_steps:
            break

    return model
