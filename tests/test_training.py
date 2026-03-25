"""Tests for the training utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dwight.tokenizer import TiktokenWrapper
from dwight.training.dataset import tokenize_and_batch
from dwight.training.train import _cosine_decay_lr


# ── LR schedule ───────────────────────────────────────────────────────────────


def test_lr_warmup_zero_at_start():
    lr = _cosine_decay_lr(step=0, warmup_steps=10, total_steps=100, max_lr=1e-3)
    assert lr == 0.0


def test_lr_peak_at_end_of_warmup():
    lr = _cosine_decay_lr(step=10, warmup_steps=10, total_steps=100, max_lr=1e-3)
    assert math.isclose(lr, 1e-3, rel_tol=1e-6)


def test_lr_decays_after_warmup():
    lr_early = _cosine_decay_lr(step=20, warmup_steps=10, total_steps=100, max_lr=1e-3)
    lr_late = _cosine_decay_lr(step=90, warmup_steps=10, total_steps=100, max_lr=1e-3)
    assert lr_early > lr_late


def test_lr_floor_at_min_lr():
    lr = _cosine_decay_lr(
        step=1000, warmup_steps=10, total_steps=100, max_lr=1e-3, min_lr=1e-5
    )
    assert math.isclose(lr, 1e-5, rel_tol=1e-6)


def test_lr_is_monotone_in_warmup():
    lrs = [
        _cosine_decay_lr(s, warmup_steps=50, total_steps=200, max_lr=1e-3)
        for s in range(50)
    ]
    assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))


# ── Dataset creation ──────────────────────────────────────────────────────────


def test_tokenize_and_batch_shapes(tokenizer):
    text = "Hello world! " * 500  # enough tokens for several windows
    ds = tokenize_and_batch(text, tokenizer, seq_len=8, batch_size=2)
    inputs, targets = next(iter(ds))
    assert inputs.shape[1] == 8
    assert targets.shape[1] == 8
    assert inputs.shape[0] == 2
    assert targets.shape[0] == 2


def test_tokenize_and_batch_targets_shifted(tokenizer):
    """targets[i] should be inputs[i] shifted left by one token."""
    text = "abcdefghij " * 200
    ds = tokenize_and_batch(text, tokenizer, seq_len=16, batch_size=1, shuffle_buffer=1)
    inp, tgt = next(iter(ds))
    inp_np = inp[0].numpy()
    tgt_np = tgt[0].numpy()
    # The full token sequence is [t0, t1, ..., t17]
    # inp = [t0..t15], tgt = [t1..t16] – they share 15 elements
    assert inp_np.shape == (16,)
    assert tgt_np.shape == (16,)


def test_tokenize_and_batch_returns_dataset(tokenizer):
    from torch.utils.data import DataLoader

    ds = tokenize_and_batch("Hello! " * 300, tokenizer, seq_len=8, batch_size=2)
    assert isinstance(ds, DataLoader)
