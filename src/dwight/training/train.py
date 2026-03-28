"""Training loop with Adam optimiser + linear warmup / cosine-decay LR."""

from __future__ import annotations

import math
import os
import time
from typing import cast

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..config import ModelConfig
from ..model.registry import get_model_entry
from ..model.tiny import TinyModel, TinyModelConfig
from ..model.tiny.muon import ParallelMuon
from ..model.tiny.quantize import save_artifact
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


def _next_step_checkpoint(step: int, interval: int) -> int:
    if interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    return ((step // interval) + 1) * interval


def _apply_scheduled_lr(
    optimizer: torch.optim.Optimizer | ParallelMuon,
    scheduled_lr: float,
    max_lr: float,
) -> None:
    if max_lr <= 0.0:
        scale = 0.0
    else:
        scale = scheduled_lr / max_lr
    for param_group in optimizer.param_groups:
        base_lr = float(param_group.setdefault("initial_lr", param_group["lr"]))
        param_group["lr"] = base_lr * scale


def _maybe_update_ema(model: torch.nn.Module) -> None:
    update_ema = getattr(model, "update_ema", None)
    if callable(update_ema):
        update_ema()


def _maybe_record_swa_snapshot(model: torch.nn.Module) -> None:
    record_snapshot = getattr(model, "record_swa_snapshot", None)
    if callable(record_snapshot):
        record_snapshot()


def _maybe_offload_auxiliary_state(model: torch.nn.Module) -> None:
    offload_auxiliary_state = getattr(model, "offload_auxiliary_state_to_cpu", None)
    if callable(offload_auxiliary_state):
        offload_auxiliary_state()


def _cleanup_after_cuda_oom(
    optimizer: torch.optim.Optimizer | ParallelMuon,
    model: torch.nn.Module,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    _maybe_offload_auxiliary_state(model)
    torch.cuda.empty_cache()


def _training_seq_len(config: object) -> int:
    return int(getattr(config, "train_seq_len", getattr(config, "max_seq_len")))


def _training_batch_size(config: object, batch_size: int | None) -> int:
    if batch_size is not None:
        return batch_size
    return int(getattr(config, "train_batch_size", 8))


def _training_grad_accum_steps(config: object, grad_accum_steps: int | None) -> int:
    if grad_accum_steps is not None:
        return grad_accum_steps
    return int(getattr(config, "train_grad_accum_steps", 1))


def _min_training_seq_len(config: object) -> int:
    return int(
        getattr(config, "min_train_seq_len", min(128, _training_seq_len(config)))
    )


def _training_uses_gradient_checkpointing(config: object, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return bool(getattr(config, "train_gradient_checkpointing", True))


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "cuda out of memory" in str(exc).lower()


def train(
    epochs: int = 3,
    batch_size: int | None = None,
    max_lr: float = 3e-4,
    warmup_steps: int = 1000,
    checkpoint_dir: str = "checkpoints",
    max_steps: int | None = None,
    data: str = DEFAULT_ARCHIVE,
    resume: bool = False,
    grad_accum_steps: int | None = None,
    model_id: str = "dwight",
) -> torch.nn.Module:
    """Train a GPT model on the 4chan /pol/ archive and return the trained model."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        amp_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        print(f"AMP enabled on CUDA ({amp_dtype})")
    tokenizer = TiktokenWrapper()
    autocast_kwargs = _autocast_kwargs(device)
    entry = get_model_entry(model_id)
    config = entry.config_cls()
    current_batch_size = _training_batch_size(config, batch_size)
    current_seq_len = _training_seq_len(config)
    min_seq_len = _min_training_seq_len(config)
    grad_accum_steps = _training_grad_accum_steps(config, grad_accum_steps)
    uses_gradient_checkpointing = _training_uses_gradient_checkpointing(config, device)

    print(f"Streaming dataset from {data} …")
    print(
        f"Training runtime: seq_len={current_seq_len}, batch_size={current_batch_size}, "
        f"grad_accum_steps={grad_accum_steps}, gradient_checkpointing={uses_gradient_checkpointing}"
    )
    # The archive is too large to count upfront; use max_steps when known.
    total_steps = max_steps if max_steps is not None else warmup_steps * 100

    model = cast(GPTModel | TinyModel, entry.model_cls(config).to(device))
    if uses_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    if model_id == "tiny":
        assert isinstance(config, TinyModelConfig)
        optimizer = ParallelMuon(
            model,
            matrix_lr=config.matrix_lr,
            scalar_lr=config.scalar_lr,
            tied_embed_lr=config.tied_embed_lr,
            momentum=config.muon_momentum,
            matrix_weight_decay=config.muon_wd,
            scalar_weight_decay=config.adam_wd,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.95))

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, os.path.basename(entry.checkpoint_path)
    )
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
    next_step_ckpt = _next_step_checkpoint(0, _CHECKPOINT_INTERVAL_STEPS)
    last_timed_ckpt = time.monotonic()
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
                next_step_ckpt = _next_step_checkpoint(
                    global_step, _CHECKPOINT_INTERVAL_STEPS
                )
                last_timed_ckpt = time.monotonic()
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
        while True:
            print(
                f"Epoch {epoch}/{epochs}: loading seq_len={current_seq_len}, batch_size={current_batch_size}"
            )
            loader = chan_dataloader(
                data,
                tokenizer,
                seq_len=current_seq_len,
                batch_size=current_batch_size,
            )
            model.train()
            running_loss = 0.0
            n_batches = 0
            accum_loss = 0.0
            micro_step = 0
            retry_epoch = False
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

            for inp, tgt in pbar:
                if max_steps is not None and global_step >= max_steps:
                    break

                try:
                    inp = inp.to(device)
                    tgt = tgt.to(device)

                    with torch.autocast(device_type=device.type, **autocast_kwargs):
                        output = model(inp)
                        if isinstance(output, tuple):
                            logits, aux_loss = output
                        else:
                            logits, aux_loss = output, 0.0
                        ce_loss = F.cross_entropy(
                            logits.view(-1, config.vocab_size),
                            tgt.view(-1),
                        )
                        moe_coeff = float(getattr(config, "moe_aux_loss_coeff", 0.0))
                        loss = ce_loss + moe_coeff * aux_loss
                    (loss / grad_accum_steps).backward()
                    accum_loss += ce_loss.item()
                    micro_step += 1

                    if micro_step % grad_accum_steps != 0:
                        continue

                    lr = _cosine_decay_lr(
                        global_step, warmup_steps, total_steps, max_lr
                    )
                    _apply_scheduled_lr(optimizer, lr, max_lr)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    _maybe_update_ema(model)
                    if isinstance(config, TinyModelConfig):
                        if global_step > 0 and global_step % config.swa_every == 0:
                            _maybe_record_swa_snapshot(model)
                    optimizer.zero_grad(set_to_none=True)

                    running_loss += accum_loss / grad_accum_steps
                    n_batches += 1
                    global_step += 1
                    accum_loss = 0.0
                    avg_loss = running_loss / n_batches
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                    now = time.monotonic()
                    if (
                        global_step >= next_step_ckpt
                        or now - last_timed_ckpt >= _CHECKPOINT_INTERVAL_SECS
                    ):
                        _save_checkpoint(
                            avg_loss, global_step, completed_epochs=epoch - 1
                        )
                        if global_step >= next_step_ckpt:
                            next_step_ckpt = _next_step_checkpoint(
                                global_step, _CHECKPOINT_INTERVAL_STEPS
                            )
                        last_timed_ckpt = now
                except RuntimeError as exc:
                    if device.type != "cuda" or not _is_cuda_oom(exc):
                        raise
                    _cleanup_after_cuda_oom(optimizer, model)
                    if current_batch_size > 1:
                        new_batch_size = max(1, current_batch_size // 2)
                        print(
                            f"CUDA OOM during training; reducing batch_size {current_batch_size} -> {new_batch_size} and retrying epoch {epoch}."
                        )
                        current_batch_size = new_batch_size
                        retry_epoch = True
                        break
                    if current_seq_len > min_seq_len:
                        new_seq_len = max(min_seq_len, current_seq_len // 2)
                        print(
                            f"CUDA OOM during training at batch_size=1; reducing seq_len {current_seq_len} -> {new_seq_len} and retrying epoch {epoch}."
                        )
                        current_seq_len = new_seq_len
                        retry_epoch = True
                        break
                    raise RuntimeError(
                        "CUDA OOM during training even after reducing batch_size to 1 "
                        f"and seq_len to {current_seq_len}."
                    ) from exc

            if retry_epoch:
                continue

            if accum_loss > 0.0:
                try:
                    remainder = micro_step % grad_accum_steps or grad_accum_steps
                    lr = _cosine_decay_lr(
                        global_step, warmup_steps, total_steps, max_lr
                    )
                    _apply_scheduled_lr(optimizer, lr, max_lr)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    _maybe_update_ema(model)
                    optimizer.zero_grad(set_to_none=True)
                    running_loss += accum_loss / remainder
                    n_batches += 1
                    global_step += 1
                except RuntimeError as exc:
                    if device.type != "cuda" or not _is_cuda_oom(exc):
                        raise
                    _cleanup_after_cuda_oom(optimizer, model)
                    if current_seq_len > min_seq_len:
                        current_seq_len = max(min_seq_len, current_seq_len // 2)
                        print(
                            f"CUDA OOM during optimizer step; reducing seq_len to {current_seq_len} and retrying epoch {epoch}."
                        )
                        retry_epoch = True
                    else:
                        raise RuntimeError(
                            "CUDA OOM during optimizer step even at minimum tiny training sequence length."
                        ) from exc

            if retry_epoch:
                continue

            break

        avg_loss = running_loss / max(n_batches, 1)
        _save_checkpoint(avg_loss, global_step, completed_epochs=epoch)
        next_step_ckpt = _next_step_checkpoint(global_step, _CHECKPOINT_INTERVAL_STEPS)
        last_timed_ckpt = time.monotonic()
        print(f"Epoch {epoch} complete — loss: {avg_loss:.4f}")

        if max_steps is not None and global_step >= max_steps:
            break

    # Export compressed artifact when the model registry defines one
    entry = get_model_entry(model_id)
    if entry.artifact_path is not None:
        save_artifact(model, entry.artifact_path)
        print(f"Artifact saved \u2192 {entry.artifact_path}")

    return model
