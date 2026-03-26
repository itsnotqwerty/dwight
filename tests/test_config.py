"""Tests for ModelConfig."""

import pytest

from dwight.config import ModelConfig
from dwight.model.tiny import TinyModelConfig


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


def test_tiny_model_config_defaults():
    cfg = TinyModelConfig()
    assert cfg.num_layers == 11
    assert cfg.d_model == 512
    assert cfg.num_heads == 8
    assert cfg.num_kv_heads == 4
    assert cfg.bigram_vocab_size == 1536
    assert cfg.rope_dims == 16
    assert cfg.train_seq_len == 2048
    assert cfg.min_train_seq_len == 256
    assert cfg.train_batch_size == 1
    assert cfg.train_grad_accum_steps == 8


def test_tiny_model_config_validation():
    with pytest.raises(AssertionError):
        TinyModelConfig(d_model=32, num_heads=4, num_kv_heads=3)

    with pytest.raises(AssertionError):
        TinyModelConfig(train_seq_len=4096, max_seq_len=2048)
