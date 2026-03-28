"""Fine-tuning helpers: supervised fine-tuning (SFT) and RLHF policy updates.

SFT runs in a background thread so the live server model can be updated
in-place without a restart.  RLHF uses REINFORCE with a mean-reward baseline
to update the policy from human thumbs-up / thumbs-down ratings.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from ..config import ModelConfig
from ..model.transformer import GPTModel
from ..tokenizer import TiktokenWrapper
from .dataset import DEFAULT_CORPUS, corpus_dataloader

_CHECKPOINT_PATH = os.path.join("checkpoints", "model.pt")


def _autocast_kwargs(device: torch.device) -> dict:
    """Return autocast settings suitable for the current device."""
    if device.type != "cuda":
        return {"enabled": False}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"enabled": True, "dtype": dtype}


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda out of memory" in msg


# ---------------------------------------------------------------------------
# Supervised fine-tuning (SFT)
# ---------------------------------------------------------------------------


def sft_finetune(
    model: torch.nn.Module,
    tokenizer: TiktokenWrapper,
    config: ModelConfig,
    *,
    corpus_path: str = DEFAULT_CORPUS,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_steps: int | None = None,
    stop_event: threading.Event | None = None,
    log_fn: Callable[[str], None] | None = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_name: str = "model.pt",
) -> None:
    """Fine-tune *model* on a plain-text corpus file in the calling thread.

    Parameters
    ----------
    model:
        The live ``GPTModel`` to update in-place.
    tokenizer:
        Tokenizer used by *model*.
    config:
        Model configuration (needed for ``vocab_size``).
    corpus_path:
        Path to the plain-text / Markdown corpus file.
    epochs:
        Number of passes over the corpus.
    batch_size:
        Micro-batch size (keep small for in-process use: default 4).
    lr:
        Learning rate for the AdamW optimizer.
    max_steps:
        Stop after this many gradient steps (``None`` = run to completion).
    stop_event:
        A ``threading.Event``; when set the loop exits at the next batch check.
    log_fn:
        Callable receiving a log-line string; called after every gradient step
        and at the end.  Defaults to ``print``.
    checkpoint_dir:
        Directory where ``model.pt`` is saved after each epoch.
    """
    if log_fn is None:
        log_fn = print

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = next(model.parameters()).device
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = Path(checkpoint_dir) / checkpoint_name
    autocast_kwargs = _autocast_kwargs(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    global_step = 0
    interrupted = False
    current_batch_size = batch_size
    current_seq_len = config.max_seq_len

    for epoch in range(1, epochs + 1):
        while True:
            loader = corpus_dataloader(
                corpus_path,
                tokenizer,
                seq_len=current_seq_len,
                batch_size=current_batch_size,
            )
            model.train()
            running_loss = 0.0
            n_batches = 0
            t0 = time.monotonic()
            retry_epoch = False

            for inp, tgt in loader:
                if stop_event is not None and stop_event.is_set():
                    interrupted = True
                    log_fn("[SFT] Stop requested - exiting early.")
                    break
                if max_steps is not None and global_step >= max_steps:
                    interrupted = True
                    log_fn(f"[SFT] Reached max_steps={max_steps}.")
                    break

                try:
                    inp = inp.to(device)
                    tgt = tgt.to(device)

                    with torch.autocast(device_type=device.type, **autocast_kwargs):
                        output = model(inp)
                        logits = output[0] if isinstance(output, tuple) else output
                        loss = F.cross_entropy(
                            logits.view(-1, config.vocab_size),
                            tgt.view(-1),
                        )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                except RuntimeError as exc:
                    if device.type != "cuda" or not _is_cuda_oom(exc):
                        raise

                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                    if current_batch_size > 1:
                        new_batch_size = max(1, current_batch_size // 2)
                        log_fn(
                            "[SFT] CUDA OOM; reducing batch_size "
                            f"{current_batch_size} -> {new_batch_size} and retrying epoch {epoch}."
                        )
                        current_batch_size = new_batch_size
                        retry_epoch = True
                        break

                    if current_seq_len > 128:
                        new_seq_len = max(128, current_seq_len // 2)
                        log_fn(
                            "[SFT] CUDA OOM at batch_size=1; reducing seq_len "
                            f"{current_seq_len} -> {new_seq_len} and retrying epoch {epoch}."
                        )
                        current_seq_len = new_seq_len
                        retry_epoch = True
                        break

                    raise RuntimeError(
                        "[SFT] CUDA OOM even at batch_size=1 and seq_len=128. "
                        "Close other GPU-heavy processes or run on CPU."
                    ) from exc

                running_loss += loss.item()
                n_batches += 1
                global_step += 1

                elapsed = time.monotonic() - t0
                avg = running_loss / n_batches
                log_fn(
                    f"[SFT] epoch {epoch}/{epochs} step {global_step}"
                    f"  loss={avg:.4f}  elapsed={elapsed:.0f}s"
                    f"  bs={current_batch_size} seq={current_seq_len}"
                )

            if retry_epoch:
                continue
            break

        if interrupted:
            break

        avg_loss = running_loss / max(n_batches, 1)
        log_fn(f"[SFT] Epoch {epoch} complete — avg loss: {avg_loss:.4f}")
        _save_checkpoint(
            model, optimizer, ckpt_path, epoch, global_step, avg_loss, log_fn
        )

    model.eval()
    log_fn("[SFT] Fine-tuning finished.")


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    epoch: int,
    step: int,
    loss: float,
    log_fn: Callable[[str], None],
) -> None:
    model.eval()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": step,
            "completed_epochs": epoch,
        },
        path,
    )
    model.train()
    log_fn(f"[SFT] Checkpoint saved → {path}  step={step}  loss={loss:.4f}")


# ---------------------------------------------------------------------------
# RLHF — REINFORCE with baseline
# ---------------------------------------------------------------------------


def rlhf_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: TiktokenWrapper,
    config: ModelConfig,
    *,
    prompt: str,
    completions: list[str],
    rewards: list[float],
) -> float:
    """One REINFORCE-with-baseline policy-gradient update.

    Parameters
    ----------
    model:
        The live policy model (should be in ``eval`` mode between calls;
        this function sets it to ``train`` mode internally).
    optimizer:
        An optimizer already attached to *model* (e.g. ``Adam`` at lr=1e-5).
    tokenizer:
        Tokenizer used to encode the prompt and completions.
    config:
        Model configuration (for ``vocab_size`` and ``max_seq_len``).
    prompt:
        The prompt text that was fed to the model to produce *completions*.
    completions:
        List of completion strings (one per sample; typically 3).
    rewards:
        Per-completion scalar reward (``+1.0`` for thumbs-up, ``-1.0`` for
        thumbs-down).  Must have the same length as *completions*.

    Returns
    -------
    float
        The scalar policy loss value (before the update step).
    """
    assert len(completions) == len(rewards), "completions and rewards must match"

    device = next(model.parameters()).device
    baseline = sum(rewards) / len(rewards)

    prompt_ids = tokenizer.encode(prompt)
    model.train()

    total_loss = torch.tensor(0.0, device=device, requires_grad=False)
    optimizer.zero_grad(set_to_none=True)

    for completion, reward in zip(completions, rewards):
        advantage = reward - baseline
        if advantage == 0.0:
            continue  # no gradient signal for this sample

        completion_ids = tokenizer.encode(completion)
        if not completion_ids:
            continue

        # Build a single sequence: [prompt_ids | completion_ids], truncated to
        # max_seq_len so we never exceed the model's context window.
        full_ids = (prompt_ids + completion_ids)[-(config.max_seq_len + 1) :]
        prompt_len = max(0, len(full_ids) - len(completion_ids))

        inp = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        tgt = torch.tensor(full_ids[1:], dtype=torch.long, device=device)

        with torch.autocast(device_type=device.type, **_autocast_kwargs(device)):
            output = model(inp)  # (1, T, vocab_size) or ((1, T, vocab_size), aux)
            logits = output[0] if isinstance(output, tuple) else output

        # Only measure log-prob over the completion tokens (not the prompt)
        comp_logits = logits[0, prompt_len - 1 :].float()  # (comp_len, vocab_size)
        comp_tgt = tgt[prompt_len - 1 :]  # (comp_len,)

        # Token-level cross-entropy gives us negative log-prob
        nll = F.cross_entropy(
            comp_logits.view(-1, config.vocab_size),
            comp_tgt.view(-1),
            reduction="mean",
        )
        # REINFORCE: L = -advantage * log_prob  (advantage = reward - baseline)
        loss_term = -torch.tensor(advantage, device=device, dtype=torch.float32) * (
            -nll
        )
        total_loss = total_loss + loss_term.detach()
        loss_term.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    model.eval()

    n = len(completions)
    return float(total_loss.item()) / max(n, 1)
