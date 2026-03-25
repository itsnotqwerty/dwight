"""Shared test fixtures and configuration."""

from __future__ import annotations

import numpy as np
import pytest

from dwight.config import ModelConfig
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
