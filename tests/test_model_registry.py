from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from dwight.config import ModelConfig
from dwight.model.registry import ModelEntry, get_model_entry, load_model
from dwight.model.tiny import TinyModel, TinyModelConfig
from dwight.model.transformer import GPTModel


def test_registry_contains_dwight_and_tiny():
    from dwight.model.registry import MODEL_REGISTRY

    assert "dwight" in MODEL_REGISTRY
    assert "tiny" in MODEL_REGISTRY


def test_load_model_returns_expected_types(monkeypatch, tiny_arch_config):
    import dwight.model.registry as registry

    class SmallDwightConfig(ModelConfig):
        def __init__(self):
            super().__init__(
                num_layers=1,
                d_model=32,
                num_heads=2,
                dff=64,
                max_seq_len=16,
                dropout=0.0,
            )

    class SmallTinyConfig(TinyModelConfig):
        def __init__(self):
            super().__init__(**replace(tiny_arch_config).__dict__)

    monkeypatch.setattr(
        registry,
        "MODEL_REGISTRY",
        {
            "dwight": ModelEntry(GPTModel, SmallDwightConfig, "missing-dwight.pt"),
            "tiny": ModelEntry(TinyModel, SmallTinyConfig, "missing-tiny.pt"),
        },
    )

    dwight_model, dwight_config, _ = load_model("dwight", torch.device("cpu"))
    tiny_model, tiny_config, _ = load_model("tiny", torch.device("cpu"))

    assert isinstance(dwight_model, GPTModel)
    assert isinstance(dwight_config, SmallDwightConfig)
    assert isinstance(tiny_model, TinyModel)
    assert isinstance(tiny_config, SmallTinyConfig)


def test_get_model_entry_unknown_key_raises():
    with pytest.raises(KeyError):
        get_model_entry("nope")
