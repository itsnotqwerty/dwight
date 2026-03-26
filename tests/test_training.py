"""Tests for the training utilities."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from dwight.tokenizer import TiktokenWrapper
from dwight.training.dataset import chan_dataloader
from dwight.training.train import _cosine_decay_lr, _min_training_seq_len, _training_seq_len

# Force num_workers=0 so mocks work in-process
_real_dataloader = __import__("torch.utils.data", fromlist=["DataLoader"]).DataLoader


def _inline_dataloader(*args, **kwargs):
    kwargs["num_workers"] = 0
    return _real_dataloader(*args, **kwargs)


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


def test_training_seq_len_uses_tiny_runtime_default():
    from dwight.model.tiny import TinyModelConfig

    cfg = TinyModelConfig()
    assert _training_seq_len(cfg) == 2048
    assert _min_training_seq_len(cfg) == 256


def test_training_seq_len_falls_back_to_max_seq_len():
    from dwight.config import ModelConfig

    cfg = ModelConfig(max_seq_len=1024)
    assert _training_seq_len(cfg) == 1024
    assert _min_training_seq_len(cfg) == 128


# ── Dataset creation ──────────────────────────────────────────────────────────

_FAKE_POSTS = ["Hello world! " * 500]


def test_chan_dataloader_shapes(tokenizer):
    with (
        patch(
            "dwight.training.dataset._iter_post_texts", return_value=iter(_FAKE_POSTS)
        ),
        patch("dwight.training.dataset.DataLoader", _inline_dataloader),
    ):
        ds = chan_dataloader("fake.tar.zst", tokenizer, seq_len=8, batch_size=2)
        inputs, targets = next(iter(ds))
    assert inputs.shape[1] == 8
    assert targets.shape[1] == 8
    assert inputs.shape[0] == 2
    assert targets.shape[0] == 2


def test_chan_dataloader_targets_shifted(tokenizer):
    """targets[i] should be inputs[i] shifted left by one token."""
    fake_posts = ["abcdefghij " * 200]
    with (
        patch(
            "dwight.training.dataset._iter_post_texts", return_value=iter(fake_posts)
        ),
        patch("dwight.training.dataset.DataLoader", _inline_dataloader),
    ):
        ds = chan_dataloader("fake.tar.zst", tokenizer, seq_len=16, batch_size=1)
        inp, tgt = next(iter(ds))
    inp_np = inp[0].numpy()
    tgt_np = tgt[0].numpy()
    assert inp_np.shape == (16,)
    assert tgt_np.shape == (16,)


def test_chan_dataloader_returns_dataloader(tokenizer):
    from torch.utils.data import DataLoader

    with (
        patch(
            "dwight.training.dataset._iter_post_texts", return_value=iter(_FAKE_POSTS)
        ),
        patch("dwight.training.dataset.DataLoader", _inline_dataloader),
    ):
        ds = chan_dataloader("fake.tar.zst", tokenizer, seq_len=8, batch_size=2)
    assert isinstance(ds, DataLoader)
