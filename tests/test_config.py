"""Tests for ModelConfig."""

import pytest

from dwight.config import ModelConfig


def test_default_values():
    cfg = ModelConfig()
    assert cfg.num_layers == 10
    assert cfg.d_model == 768
    assert cfg.num_heads == 12
    assert cfg.dff == 3072
    assert cfg.vocab_size == 100_277
    assert cfg.max_seq_len == 1024
    assert cfg.dropout == 0.1


def test_custom_values():
    cfg = ModelConfig(num_layers=2, d_model=64, num_heads=4, dff=128)
    assert cfg.num_layers == 2
    assert cfg.d_model == 64
    assert cfg.num_heads == 4


def test_d_model_divisible_by_num_heads():
    # Valid
    cfg = ModelConfig(d_model=64, num_heads=8)
    assert cfg.d_model // cfg.num_heads == 8


def test_raises_when_not_divisible():
    with pytest.raises(AssertionError):
        ModelConfig(d_model=65, num_heads=8)
