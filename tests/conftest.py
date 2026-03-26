"""Shared test fixtures and configuration."""

from __future__ import annotations

import numpy as np
import pytest

from dwight.config import ModelConfig
from dwight.model.tiny import TinyModel, TinyModelConfig
from dwight.model.transformer import GPTModel
from dwight.tokenizer import TiktokenWrapper

# A tiny config so model tests run quickly
TINY = ModelConfig(
    num_layers=1,
    d_model=32,
    num_heads=2,
    dff=64,
    vocab_size=100_277,
    max_seq_len=16,
    dropout=0.0,
)

TINY_ARCH = TinyModelConfig(
    num_layers=4,
    d_model=32,
    num_heads=4,
    num_kv_heads=2,
    dff=96,
    vocab_size=100_277,
    bigram_vocab_size=64,
    max_seq_len=16,
    train_seq_len=16,
    min_train_seq_len=8,
    dropout=0.0,
    rope_dims=4,
    xsa_last_n=2,
    ve_enabled=True,
    ve_dim=8,
    ve_layers=(2, 3),
    swa_every=2,
)


@pytest.fixture(scope="session")
def tiny_config() -> ModelConfig:
    return TINY


@pytest.fixture(scope="session")
def tokenizer() -> TiktokenWrapper:
    return TiktokenWrapper()


@pytest.fixture(scope="session")
def tiny_model(tiny_config: ModelConfig) -> GPTModel:
    model = GPTModel(tiny_config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_arch_config() -> TinyModelConfig:
    return TINY_ARCH


@pytest.fixture(scope="session")
def tiny_arch_model(tiny_arch_config: TinyModelConfig) -> TinyModel:
    model = TinyModel(tiny_arch_config)
    model.eval()
    return model
