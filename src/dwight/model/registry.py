from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from ..config import ModelConfig
from .tiny import TinyModel, TinyModelConfig
from .tiny.quantize import load_artifact
from .transformer import GPTModel


@dataclass(frozen=True)
class ModelEntry:
    model_cls: type[torch.nn.Module]
    config_cls: type
    checkpoint_path: str
    artifact_path: Optional[str] = None


MODEL_REGISTRY: dict[str, ModelEntry] = {
    # Default architecture: MLA + MoE (use_mla=True, use_moe=True by default)
    "dwight": ModelEntry(
        GPTModel, ModelConfig, os.path.join("checkpoints", "model.pt")
    ),
    # Individual feature variants
    "dwight-moe": ModelEntry(
        GPTModel,
        lambda: ModelConfig(use_mla=False, use_moe=True),  # type: ignore
        os.path.join("checkpoints", "dwight_moe.pt"),
    ),
    "dwight-mla": ModelEntry(
        GPTModel,
        lambda: ModelConfig(use_mla=True, use_moe=False),  # type: ignore
        os.path.join("checkpoints", "dwight_mla.pt"),
    ),
    # Original dense MHA + SwiGLU FFN (no MLA, no MoE)
    "dwight-dense": ModelEntry(
        GPTModel,
        lambda: ModelConfig(use_mla=False, use_moe=False),  # type: ignore
        os.path.join("checkpoints", "dwight_dense.pt"),
    ),
    "tiny": ModelEntry(
        TinyModel,
        TinyModelConfig,
        os.path.join("checkpoints", "tiny.pt"),
        artifact_path=os.path.join("checkpoints", "tiny_artifact.lzma"),
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

    artifact_path = entry.artifact_path
    checkpoint_path = entry.checkpoint_path

    artifact_exists = artifact_path is not None and os.path.exists(artifact_path)
    checkpoint_exists = os.path.exists(checkpoint_path)

    # Prefer the artifact only when it is at least as recent as the checkpoint.
    # If the checkpoint is newer (e.g. training ran but the artifact was not yet
    # re-exported), fall through to the checkpoint so stale artifacts are ignored.
    prefer_artifact = artifact_exists and (
        not checkpoint_exists
        or os.path.getmtime(artifact_path) >= os.path.getmtime(checkpoint_path)
    )

    if prefer_artifact:
        try:
            load_artifact(model, artifact_path)
            print(f"Loaded artifact from {artifact_path} (device: {device})")
            model.to(device)
            model.eval()
            return model, config, checkpoint_path
        except Exception as exc:
            print(
                f"Warning: artifact {artifact_path} could not be loaded ({exc}); "
                "falling back to checkpoint."
            )
    if checkpoint_exists:
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
