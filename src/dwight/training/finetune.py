"""Fine-tuning helpers: supervised fine-tuning (SFT) and RLHF policy updates.

SFT runs in a background thread so the live server model can be updated
in-place without a restart.  RLHF uses REINFORCE with a mean-reward baseline
to update the policy from human thumbs-up / thumbs-down ratings.
"""

from __future__ import annotations

import copy
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from ..config import ModelConfig
from ..model.transformer import GPTModel
from ..tokenizer import TiktokenWrapper
from .dataset import (
    DEFAULT_CORPUS,
    DEFAULT_DPO,
    DEFAULT_PROMPTS,
    corpus_dataloader,
    dpo_dataloader,
    prompt_dataloader,
)

_CHECKPOINT_PATH = os.path.join("checkpoints", "model.pt")
_STRUCTURAL_FILLER_PHRASES = (
    "worth noting",
    "in conclusion",
    "on the other hand",
    "it is important to",
    "furthermore",
    "moreover",
)
_STRUCTURAL_HEDGE_OPENERS = (
    "i think",
    "well,",
    "it is important",
    "in my opinion",
    "personally",
    "to be fair",
)
_STRUCTURAL_HEDGE_WORDS = (
    "maybe",
    "perhaps",
    "probably",
    "sort of",
    "kind of",
    "i think",
    "i guess",
)


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


def tuned_checkpoint_name(checkpoint_name: str) -> str:
    """Return a sibling tuned checkpoint name for *checkpoint_name*."""
    path = Path(checkpoint_name)
    suffix = path.suffix or ".pt"
    return f"{path.stem}_tuned{suffix}"


def dpo_checkpoint_name(checkpoint_name: str) -> str:
    """Return a sibling DPO checkpoint name for *checkpoint_name*."""
    path = Path(checkpoint_name)
    suffix = path.suffix or ".pt"
    return f"{path.stem}_dpo{suffix}"


def _clamp_score(score: float) -> float:
    return max(-1.0, min(1.0, score))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def structural_reward(text: str) -> float:
    """Return a heuristic structural quality score in the range [-1, 1]."""
    normalized = " ".join(text.split())
    if not normalized:
        return -1.0

    lowered = normalized.lower()
    words = normalized.split()
    word_count = len(words)
    score = 0.0

    if 30 <= word_count <= 80:
        score += 0.45
    elif 20 <= word_count <= 110:
        score += 0.2
    elif word_count < 10 or word_count > 150:
        score -= 0.45
    else:
        score -= 0.15

    if any(lowered.startswith(marker) for marker in _STRUCTURAL_HEDGE_OPENERS):
        score -= 0.3

    filler_hits = sum(1 for phrase in _STRUCTURAL_FILLER_PHRASES if phrase in lowered)
    score -= min(0.5, filler_hits * 0.2)

    sentences = _split_sentences(normalized)
    if sentences:
        first_sentence = sentences[0].lower()
        last_sentence = sentences[-1].lower()
        last_word_count = len(sentences[-1].split())

        if any(
            first_sentence.startswith(marker) for marker in _STRUCTURAL_HEDGE_OPENERS
        ):
            score -= 0.2

        if 2 <= last_word_count <= 12:
            score += 0.25
        elif last_word_count > 24:
            score -= 0.15

        if any(marker in last_sentence for marker in _STRUCTURAL_HEDGE_WORDS):
            score -= 0.15
        else:
            score += 0.15

    return _clamp_score(score)


def auto_rate_completion(text: str) -> float:
    """Expose the structural heuristic as an RLHF-ready score."""
    return structural_reward(text)


# ---------------------------------------------------------------------------
# Supervised fine-tuning (SFT)
# ---------------------------------------------------------------------------


def sft_finetune(
    model: torch.nn.Module,
    tokenizer: TiktokenWrapper,
    config: ModelConfig,
    *,
    corpus_path: str = DEFAULT_CORPUS,
    prompts_path: str | None = None,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    label_smoothing: float = 0.1,
    max_steps: int | None = None,
    stop_event: threading.Event | None = None,
    log_fn: Callable[[str], None] | None = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_name: str = "model_tuned.pt",
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
    prompts_path:
        Optional path to a structured prompt corpus using ``[SYSTEM]``,
        ``[USER]``, and ``[ASSISTANT]`` blocks. If omitted, prompt files are
        auto-detected from ``corpus_path`` when they contain those markers.
    epochs:
        Number of passes over the corpus.
    batch_size:
        Micro-batch size (keep small for in-process use: default 4).
    lr:
        Learning rate for the AdamW optimizer.
    label_smoothing:
        Smoothing applied to the target distribution during SFT to reduce
        overconfident memorization of exact token sequences.
    max_steps:
        Stop after this many gradient steps (``None`` = run to completion).
    stop_event:
        A ``threading.Event``; when set the loop exits at the next batch check.
    log_fn:
        Callable receiving a log-line string; called after every gradient step
        and at the end.  Defaults to ``print``.
    checkpoint_dir:
        Directory where the tuned checkpoint is saved after each epoch.
    checkpoint_name:
        Output filename for the tuned weights. Defaults to ``model_tuned.pt``
        so SFT does not overwrite the source checkpoint.
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
    active_prompt_path = Path(prompts_path) if prompts_path is not None else None

    def _looks_like_prompt_corpus(path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(4096)
        except OSError:
            return False
        return all(marker in head for marker in ("[SYSTEM]", "[USER]", "[ASSISTANT]"))

    for epoch in range(1, epochs + 1):
        while True:
            dataset_path = active_prompt_path or Path(corpus_path)
            if prompts_path is not None or _looks_like_prompt_corpus(dataset_path):
                loader = prompt_dataloader(
                    dataset_path,
                    tokenizer,
                    seq_len=current_seq_len,
                    batch_size=current_batch_size,
                )
            else:
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
                            label_smoothing=label_smoothing,
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
    *,
    prefix: str = "[SFT]",
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
    log_fn(f"{prefix} Checkpoint saved → {path}  step={step}  loss={loss:.4f}")


def _sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Return summed token log-probs over non-masked targets for each example."""
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if target_ids.ndim == 1:
        target_ids = target_ids.unsqueeze(0)

    output = model(input_ids)
    logits = output[0] if isinstance(output, tuple) else output
    log_probs = F.log_softmax(logits.float(), dim=-1)
    target_mask = target_ids != -100
    safe_targets = target_ids.masked_fill(~target_mask, 0)
    token_log_probs = log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1))
    token_log_probs = token_log_probs.squeeze(-1).masked_fill(~target_mask, 0.0)
    return token_log_probs.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Return the standard DPO objective over a batch."""
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    return -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()


def dpo_finetune(
    model: torch.nn.Module,
    tokenizer: TiktokenWrapper,
    config: ModelConfig,
    *,
    dpo_path: str = DEFAULT_DPO,
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 1e-5,
    beta: float = 0.1,
    max_steps: int | None = None,
    stop_event: threading.Event | None = None,
    log_fn: Callable[[str], None] | None = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_name: str = "model_dpo.pt",
) -> None:
    """Fine-tune *model* using Direct Preference Optimization."""
    if log_fn is None:
        log_fn = print

    os.makedirs(checkpoint_dir, exist_ok=True)
    device = next(model.parameters()).device
    ckpt_path = Path(checkpoint_dir) / checkpoint_name
    autocast_kwargs = _autocast_kwargs(device)

    ref_model = copy.deepcopy(model)
    ref_model.to(device)
    ref_model.requires_grad_(False)
    ref_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    global_step = 0
    interrupted = False

    for epoch in range(1, epochs + 1):
        loader = dpo_dataloader(
            dpo_path,
            tokenizer,
            seq_len=config.max_seq_len,
            batch_size=batch_size,
        )
        model.train()
        running_loss = 0.0
        n_batches = 0
        t0 = time.monotonic()

        for chosen_inp, chosen_tgt, rejected_inp, rejected_tgt in loader:
            if stop_event is not None and stop_event.is_set():
                interrupted = True
                log_fn("[DPO] Stop requested - exiting early.")
                break
            if max_steps is not None and global_step >= max_steps:
                interrupted = True
                log_fn(f"[DPO] Reached max_steps={max_steps}.")
                break

            chosen_inp = chosen_inp.to(device)
            chosen_tgt = chosen_tgt.to(device)
            rejected_inp = rejected_inp.to(device)
            rejected_tgt = rejected_tgt.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, **autocast_kwargs):
                policy_chosen_logps = _sequence_log_probs(model, chosen_inp, chosen_tgt)
                policy_rejected_logps = _sequence_log_probs(
                    model, rejected_inp, rejected_tgt
                )
                with torch.no_grad():
                    ref_chosen_logps = _sequence_log_probs(
                        ref_model, chosen_inp, chosen_tgt
                    )
                    ref_rejected_logps = _sequence_log_probs(
                        ref_model, rejected_inp, rejected_tgt
                    )
                loss = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=beta,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1

            elapsed = time.monotonic() - t0
            avg = running_loss / n_batches
            log_fn(
                f"[DPO] epoch {epoch}/{epochs} step {global_step}"
                f"  loss={avg:.4f}  elapsed={elapsed:.0f}s"
                f"  beta={beta:.3f}"
            )

        avg_loss = running_loss / max(n_batches, 1)
        log_fn(f"[DPO] Epoch {epoch} complete — avg loss: {avg_loss:.4f}")
        _save_checkpoint(
            model,
            optimizer,
            ckpt_path,
            epoch,
            global_step,
            avg_loss,
            log_fn,
            prefix="[DPO]",
        )

        if interrupted:
            break

    model.eval()
    log_fn("[DPO] Fine-tuning finished.")


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
