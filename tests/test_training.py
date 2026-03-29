"""Tests for the training utilities."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

from dwight.tokenizer import TiktokenWrapper
from dwight.training.dataset import chan_dataloader
from dwight.training.train import (
    LossAutoStopConfig,
    LossAutoStopper,
    _apply_scheduled_lr,
    _cleanup_after_cuda_oom,
    _cosine_decay_lr,
    _loss_is_finite,
    _min_training_seq_len,
    _maybe_offload_auxiliary_state,
    _next_step_checkpoint,
    _restore_scheduler_from_checkpoint,
    _training_batch_size,
    _training_grad_accum_steps,
    _training_uses_gradient_checkpointing,
    _training_seq_len,
)

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


def test_next_step_checkpoint_uses_next_interval_boundary():
    assert _next_step_checkpoint(0, 10_000) == 10_000
    assert _next_step_checkpoint(9_999, 10_000) == 10_000
    assert _next_step_checkpoint(10_000, 10_000) == 20_000


def test_apply_scheduled_lr_scales_each_group_from_its_base_lr():
    layer = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(
        [
            {"params": [layer.weight], "lr": 0.4},
            {"params": [layer.bias], "lr": 0.1},
        ]
    )

    _apply_scheduled_lr(optimizer, scheduled_lr=0.2, max_lr=0.4)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.05)

    _apply_scheduled_lr(optimizer, scheduled_lr=0.1, max_lr=0.4)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.025)


def test_loss_is_finite_accepts_normal_loss():
    assert _loss_is_finite(torch.tensor(4.25)) is True


def test_loss_is_finite_rejects_nan_and_inf():
    assert _loss_is_finite(torch.tensor(float("nan"))) is False
    assert _loss_is_finite(torch.tensor(float("inf"))) is False


def test_loss_auto_stopper_tracks_new_best_window():
    stopper = LossAutoStopper(
        LossAutoStopConfig(window=3, ratio=1.5, patience=2, min_steps=0, min_delta=0.5)
    )

    assert stopper.update(1, 6.0) is None
    assert stopper.update(2, 5.0) is None
    assert stopper.update(3, 4.0) is None

    assert stopper.best_window_avg == pytest.approx(5.0)
    assert stopper.consecutive_violations == 0


def test_loss_auto_stopper_requires_consecutive_violations():
    stopper = LossAutoStopper(
        LossAutoStopConfig(window=2, ratio=1.5, patience=2, min_steps=0, min_delta=0.5)
    )

    for step, loss in enumerate((4.0, 4.0, 8.0), start=1):
        assert stopper.update(step, loss) is None

    reason = stopper.update(4, 8.0)

    assert reason is not None
    assert "Auto-stop triggered at step 4" in reason


def test_loss_auto_stopper_resets_after_recovery():
    stopper = LossAutoStopper(
        LossAutoStopConfig(window=2, ratio=1.4, patience=3, min_steps=0, min_delta=0.25)
    )

    for step, loss in enumerate((4.0, 4.0, 8.0), start=1):
        assert stopper.update(step, loss) is None

    assert stopper.consecutive_violations == 1
    assert stopper.update(4, 4.1) is None
    assert stopper.consecutive_violations == 2
    assert stopper.update(5, 4.0) is None
    assert stopper.consecutive_violations == 0


def test_loss_auto_stopper_ignores_steps_before_min_steps():
    stopper = LossAutoStopper(
        LossAutoStopConfig(window=2, ratio=1.2, patience=1, min_steps=5, min_delta=0.1)
    )

    for step, loss in enumerate((4.0, 4.0, 9.0, 9.0), start=1):
        assert stopper.update(step, loss) is None


def test_loss_auto_stopper_can_be_disabled():
    stopper = LossAutoStopper(
        LossAutoStopConfig(enabled=False, window=2, ratio=1.2, patience=1, min_steps=0)
    )

    for step, loss in enumerate((4.0, 4.0, 9.0, 9.0), start=1):
        assert stopper.update(step, loss) is None


def test_loss_auto_stopper_state_dict_round_trip():
    stopper = LossAutoStopper(
        LossAutoStopConfig(window=3, ratio=1.3, patience=2, min_steps=0, min_delta=0.2)
    )

    for step, loss in enumerate((5.0, 4.0, 3.0, 3.1, 3.2), start=1):
        stopper.update(step, loss)

    restored = LossAutoStopper(
        LossAutoStopConfig(window=3, ratio=1.3, patience=2, min_steps=0, min_delta=0.2)
    )
    restored.load_state_dict(stopper.state_dict())

    assert list(restored._window) == pytest.approx(list(stopper._window))
    assert restored.best_window_avg == pytest.approx(stopper.best_window_avg)
    assert restored.consecutive_violations == stopper.consecutive_violations


def test_loss_auto_stopper_resume_catches_spike_with_restored_baseline():
    stopper = LossAutoStopper(
        LossAutoStopConfig(
            window=3,
            ratio=1.2,
            patience=2,
            min_steps=0,
            min_delta=0.1,
            post_resume_steps=0,
        )
    )
    stopper.load_state_dict(
        {
            "window": [2.0, 2.0, 2.0],
            "best_window_avg": 2.0,
            "consecutive_violations": 0,
        }
    )

    assert stopper.update(1, 2.4) is None
    assert stopper.update(2, 2.4) is None
    assert stopper.update(3, 2.4) is None
    reason = stopper.update(4, 2.4)

    assert reason is not None
    assert "Auto-stop triggered at step 4" in reason


def test_loss_auto_stopper_seed_from_loss_initializes_baseline():
    stopper = LossAutoStopper(
        LossAutoStopConfig(window=4, ratio=1.2, patience=1, min_steps=0, min_delta=0.1)
    )

    stopper.seed_from_loss(4.0)

    assert list(stopper._window) == pytest.approx([4.0, 4.0, 4.0, 4.0])
    assert stopper.best_window_avg == pytest.approx(4.0)
    assert stopper.consecutive_violations == 0


def test_loss_auto_stopper_post_resume_grace_suppresses_violations():
    stopper = LossAutoStopper(
        LossAutoStopConfig(
            window=2,
            ratio=1.2,
            patience=1,
            min_steps=0,
            min_delta=0.1,
            post_resume_steps=2,
        )
    )
    stopper.load_state_dict(
        {
            "window": [2.0, 2.0],
            "best_window_avg": 2.0,
            "consecutive_violations": 0,
        }
    )

    assert stopper.update(1, 2.6) is None
    assert stopper.update(2, 2.6) is None
    reason = stopper.update(3, 2.6)

    assert reason is not None
    assert "Auto-stop triggered at step 3" in reason


def _make_adam_optimizer(lr: float) -> torch.optim.Adam:
    layer = torch.nn.Linear(4, 4)
    return torch.optim.Adam(layer.parameters(), lr=lr)


def test_restore_scheduler_sets_initial_lr_from_saved_state():
    opt = _make_adam_optimizer(lr=3e-4)
    opt.param_groups[0]["lr"] = 1e-5

    sched_state = {
        "total_steps": 200_000,
        "warmup_steps": 1_000,
        "max_lr": 3e-4,
        "initial_lr_per_group": [3e-4],
    }
    total_steps = _restore_scheduler_from_checkpoint(
        opt,
        sched_state,
        global_step=50_000,
        max_steps_override=None,
        warmup_steps=1_000,
        max_lr=3e-4,
    )

    assert total_steps == 200_000
    assert opt.param_groups[0]["initial_lr"] == pytest.approx(3e-4)


def test_restore_scheduler_respects_max_steps_override():
    opt = _make_adam_optimizer(lr=3e-4)
    sched_state = {"total_steps": 100_000, "initial_lr_per_group": [3e-4]}

    total_steps = _restore_scheduler_from_checkpoint(
        opt,
        sched_state,
        global_step=10_000,
        max_steps_override=50_000,
        warmup_steps=1_000,
        max_lr=3e-4,
    )

    assert total_steps == 50_000


def test_restore_scheduler_legacy_checkpoint_falls_back_to_max_lr():
    opt = _make_adam_optimizer(lr=3e-4)
    opt.param_groups[0]["lr"] = 1e-5

    total_steps = _restore_scheduler_from_checkpoint(
        opt,
        sched_state={},
        global_step=5_000,
        max_steps_override=None,
        warmup_steps=1_000,
        max_lr=3e-4,
    )

    assert opt.param_groups[0].get("initial_lr") == pytest.approx(3e-4)
    assert total_steps == 100_000


def test_restore_scheduler_yields_expected_mid_schedule_lr():
    opt = _make_adam_optimizer(lr=3e-4)
    opt.param_groups[0]["lr"] = 1e-5

    sched_state = {
        "total_steps": 200_000,
        "warmup_steps": 1_000,
        "max_lr": 3e-4,
        "initial_lr_per_group": [3e-4],
    }
    total_steps = _restore_scheduler_from_checkpoint(
        opt,
        sched_state,
        global_step=100_000,
        max_steps_override=None,
        warmup_steps=1_000,
        max_lr=3e-4,
    )

    scheduled_lr = _cosine_decay_lr(100_000, 1_000, total_steps, 3e-4)
    _apply_scheduled_lr(opt, scheduled_lr, max_lr=3e-4)

    assert opt.param_groups[0]["lr"] > 1e-4


def test_training_seq_len_uses_tiny_runtime_default():
    from dwight.model.tiny import TinyModelConfig

    cfg = TinyModelConfig()
    assert _training_seq_len(cfg) == 2048
    assert _training_batch_size(cfg, None) == 1
    assert _training_grad_accum_steps(cfg, None) == 8
    assert _min_training_seq_len(cfg) == 256


def test_training_defaults_respect_explicit_overrides():
    from dwight.model.tiny import TinyModelConfig

    cfg = TinyModelConfig()
    assert _training_batch_size(cfg, 3) == 3
    assert _training_grad_accum_steps(cfg, 5) == 5


def test_tiny_training_disables_gradient_checkpointing_by_default():
    from dwight.model.tiny import TinyModelConfig

    cfg = TinyModelConfig()
    assert _training_uses_gradient_checkpointing(cfg, torch.device("cuda")) is False


def test_training_seq_len_falls_back_to_max_seq_len():
    from dwight.config import ModelConfig

    cfg = ModelConfig(max_seq_len=1024)
    assert _training_seq_len(cfg) == 1024
    assert _training_batch_size(cfg, None) == 8
    assert _training_grad_accum_steps(cfg, None) == 1
    assert _min_training_seq_len(cfg) == 128
    assert _training_uses_gradient_checkpointing(cfg, torch.device("cuda")) is True


# ── Dataset creation ──────────────────────────────────────────────────────────


def test_chan_dataloader_shapes(tokenizer):
    fake_toks = tokenizer.encode("Hello world! " * 500)
    fake_seqs = [(fake_toks, [False] * len(fake_toks))]
    with (
        patch(
            "dwight.training.dataset._iter_thread_token_sequences",
            side_effect=lambda *_: iter([fake_seqs]),
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
    fake_toks = tokenizer.encode("abcdefghij " * 200)
    fake_seqs = [(fake_toks, [False] * len(fake_toks))]
    with (
        patch(
            "dwight.training.dataset._iter_thread_token_sequences",
            side_effect=lambda *_: iter([fake_seqs]),
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

    fake_toks = tokenizer.encode("Hello world! " * 500)
    fake_seqs = [(fake_toks, [False] * len(fake_toks))]
    with (
        patch(
            "dwight.training.dataset._iter_thread_token_sequences",
            side_effect=lambda *_: iter([fake_seqs]),
        ),
        patch("dwight.training.dataset.DataLoader", _inline_dataloader),
    ):
        ds = chan_dataloader("fake.tar.zst", tokenizer, seq_len=8, batch_size=2)
    assert isinstance(ds, DataLoader)


def test_chan_dataloader_parent_tokens_masked(tokenizer):
    """Parent-prefix positions in the target must be masked to -100."""
    n_parent, n_reply = 4, 20
    # Token IDs chosen to be far from EOT (100257) so the EOT mask doesn't interfere.
    parent_toks = [100] * n_parent
    reply_toks = [200] * n_reply
    toks = parent_toks + reply_toks
    mask = [True] * n_parent + [False] * n_reply
    fake_seqs = [(toks, mask)]
    with (
        patch(
            "dwight.training.dataset._iter_thread_token_sequences",
            side_effect=lambda *_: iter([fake_seqs]),
        ),
        patch("dwight.training.dataset.DataLoader", _inline_dataloader),
    ):
        ds = chan_dataloader("fake.tar.zst", tokenizer, seq_len=8, batch_size=1)
        inp, tgt = next(iter(ds))
    tgt_flat = tgt[0]
    # First chunk: seq[0..8] = [100,100,100,100, 200,200,200,200, 200]
    # tgt = seq[1..8]        = [100,100,100, 200,200,200,200,200]
    # is_parent = mask[1..8] = [T,  T,  T,   F,  F,  F,  F,  F ]
    assert (tgt_flat[:3] == -100).all(), "parent-prefix positions must be masked"
    assert (tgt_flat[3:] == 200).all(), "reply positions must carry real token IDs"


def test_chan_dataloader_disables_multiprocessing_workers(tokenizer):
    fake_toks = tokenizer.encode("Hello world! " * 500)
    fake_seqs = [(fake_toks, [False] * len(fake_toks))]
    with patch(
        "dwight.training.dataset._iter_thread_token_sequences",
        side_effect=lambda *_: iter([fake_seqs]),
    ):
        ds = chan_dataloader("fake.tar.zst", tokenizer, seq_len=8, batch_size=2)

    assert ds.num_workers == 0


def test_chan_dataloader_does_not_cross_thread_boundaries(tokenizer):
    """Tokens from a thread that produced no complete windows must not bleed into the next thread."""
    # Thread A: 4 tokens — with seq_len=8 the buffer never reaches chunk=9, so
    # everything is discarded at the thread boundary.
    thread_a = [([111] * 4, [False] * 4)]
    # Thread B: 20 tokens — enough for multiple complete windows.
    thread_b = [([222] * 20, [False] * 20)]
    with (
        patch(
            "dwight.training.dataset._iter_thread_token_sequences",
            side_effect=lambda *_: iter([thread_a, thread_b]),
        ),
        patch("dwight.training.dataset.DataLoader", _inline_dataloader),
    ):
        ds = chan_dataloader("fake.tar.zst", tokenizer, seq_len=8, batch_size=1)
        all_inp_toks = [t for inp, _tgt in ds for t in inp.view(-1).tolist()]
    assert 111 not in all_inp_toks, "thread A tokens must not appear in any window"
    assert 222 in all_inp_toks, "thread B tokens must appear in windows"


def test_chan_dataloader_short_thread_yields_no_windows(tokenizer):
    """A thread whose total tokens are fewer than seq_len+1 must produce zero windows."""
    # 5 tokens + 1 EOT separator = 6 total in buf; chunk = seq_len+1 = 9.
    thread = [([300] * 5, [False] * 5)]
    with (
        patch(
            "dwight.training.dataset._iter_thread_token_sequences",
            side_effect=lambda *_: iter([thread]),
        ),
        patch("dwight.training.dataset.DataLoader", _inline_dataloader),
    ):
        ds = chan_dataloader("fake.tar.zst", tokenizer, seq_len=8, batch_size=1)
        windows = list(ds)
    assert windows == [], "short thread must yield no windows"


def test_chan_dataloader_all_masked_chunks_are_dropped(tokenizer):
    """Chunks where every target position is masked must not be yielded (avoids NaN loss)."""
    # seq_len=8, chunk=9.
    # parent_toks has 12 tokens (all masked), no reply tokens.
    # First chunk [0:9] = 9 parent tokens → all masked → must be dropped.
    # Second chunk [9:18] has only 3 parent + 1 EOT(masked) = 4 tokens → buffer
    # never fills a second chunk.  So the dataloader should yield zero windows.
    parent_toks = [100] * 12
    fake_seqs = [(parent_toks, [True] * 12)]
    with (
        patch(
            "dwight.training.dataset._iter_thread_token_sequences",
            side_effect=lambda *_: iter([fake_seqs]),
        ),
        patch("dwight.training.dataset.DataLoader", _inline_dataloader),
    ):
        ds = chan_dataloader("fake.tar.zst", tokenizer, seq_len=8, batch_size=1)
        windows = list(ds)
    assert windows == [], "all-masked chunks must be filtered out to prevent NaN loss"


def test_maybe_offload_auxiliary_state_calls_model_hook():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def offload_auxiliary_state_to_cpu(self) -> None:
            self.called = True

    model = DummyModel()
    _maybe_offload_auxiliary_state(model)
    assert model.called is True


def test_cleanup_after_cuda_oom_zeroes_grads_and_offloads_aux_state(monkeypatch):
    class DummyOptimizer:
        def __init__(self):
            self.zero_grad_called = False

        def zero_grad(self, *, set_to_none: bool = True) -> None:
            self.zero_grad_called = set_to_none

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offloaded = False

        def offload_auxiliary_state_to_cpu(self) -> None:
            self.offloaded = True

    emptied = False

    def fake_empty_cache() -> None:
        nonlocal emptied
        emptied = True

    optimizer = DummyOptimizer()
    model = DummyModel()
    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

    _cleanup_after_cuda_oom(optimizer, model)

    assert optimizer.zero_grad_called is True
    assert model.offloaded is True
    assert emptied is True
