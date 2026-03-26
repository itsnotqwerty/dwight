from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from ..config import ModelConfig
from .tiny import TinyModel, TinyModelConfig
from .transformer import GPTModel


@dataclass(frozen=True)
class ModelEntry:
    model_cls: type[torch.nn.Module]
    config_cls: type
    checkpoint_path: str


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "dwight": ModelEntry(
        GPTModel, ModelConfig, os.path.join("checkpoints", "model.pt")
    ),
    "tiny": ModelEntry(
        TinyModel, TinyModelConfig, os.path.join("checkpoints", "tiny.pt")
    ),
}


def get_model_entry(model_id: str) -> ModelEntry:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model_id: {model_id}") from exc


def load_model(
    model_id: str, device: torch.device
) -> tuple[torch.nn.Module, object, str]:
    entry = get_model_entry(model_id)
    config = entry.config_cls()
    model = entry.model_cls(config)
    checkpoint_path = entry.checkpoint_path
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {checkpoint_path} (device: {device})")
        except RuntimeError as exc:
            print(
                f"Warning: checkpoint {checkpoint_path} is incompatible with the current model architecture ({exc}). Starting with random weights."
            )
    else:
        print(
            f"No checkpoint found for {model_id} – starting with random weights (device: {device})."
        )
    model.to(device)
    model.eval()
    return model, config, checkpoint_path
