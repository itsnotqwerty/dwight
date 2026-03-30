"""Shared model-loading and variant-discovery helpers.

Used by both the OpenAI-compatible API routes and the Web UI routes so that
checkpoint management logic lives in one place.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from ..model.registry import MODEL_REGISTRY, get_model_entry, load_model
from ..training.finetune import dpo_checkpoint_name, tuned_checkpoint_name

# Recognised variant suffixes and their checkpoint-name generators.
_VARIANT_BUILDERS: dict[str, callable] = {  # type: ignore
    "tuned": tuned_checkpoint_name,
    "dpo": dpo_checkpoint_name,
}


def resolve_variant_checkpoint(base_checkpoint_path: str | Path, variant: str) -> Path:
    """Return the sibling checkpoint path for *variant* of *base_checkpoint_path*.

    Example::

        resolve_variant_checkpoint("checkpoints/model.pt", "dpo")
        # → Path("checkpoints/model_dpo.pt")
    """
    builder = _VARIANT_BUILDERS[variant]
    base = Path(base_checkpoint_path)
    return base.with_name(builder(base.name))


def list_model_variants(model_id: str, base_checkpoint_path: str | Path) -> list[str]:
    """Return all addressable model IDs for *model_id*, including existing variants.

    The base ``model_id`` is always included first.  ``<model_id>:tuned`` and
    ``<model_id>:dpo`` are appended only when the corresponding checkpoint file
    exists on disk.
    """
    ids = [model_id]
    for variant in _VARIANT_BUILDERS:
        path = resolve_variant_checkpoint(base_checkpoint_path, variant)
        if path.exists():
            ids.append(f"{model_id}:{variant}")
    return ids


def parse_model_id(model_str: str) -> tuple[str, str | None]:
    """Split ``"base_id:variant"`` → ``(base_id, variant | None)``.

    Returns ``(model_str, None)`` unchanged when *model_str* contains no colon,
    so arbitrary model names (e.g. from generic OpenAI clients) are accepted and
    simply use the currently-loaded model.

    Raises ``ValueError`` only when a colon is present but the base ID or
    variant is unrecognised.
    """
    if ":" not in model_str:
        return model_str, None

    base_id, variant = model_str.split(":", 1)

    if base_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {base_id!r}")
    if variant not in _VARIANT_BUILDERS:
        raise ValueError(
            f"Unknown variant: {variant!r}. Choose from: {list(_VARIANT_BUILDERS)}"
        )
    return base_id, variant


def release_current_model(app) -> None:
    """Move the active model to CPU, delete it, and free GPU memory."""
    current_model = getattr(app.state, "model", None)
    if current_model is None:
        return
    current_model.to("cpu")
    del current_model
    app.state.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_checkpoint(
    model_id: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, object, str]:
    """Load *model_id* architecture with weights from *checkpoint_path*."""
    entry = get_model_entry(model_id)
    config = entry.config_cls()
    model = entry.model_cls(config)

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, config, str(checkpoint_path)


def swap_model_if_needed(
    app,
    base_id: str,
    variant: str | None,
) -> None:
    """Ensure *app.state.model* matches *base_id* + *variant*.

    If the currently-loaded model already matches, this is a no-op.  Otherwise
    the old model is released and the new checkpoint is loaded in-place.

    Must be called while holding ``app.state.model_lock``.
    """
    entry = MODEL_REGISTRY[base_id]
    if variant is None:
        target_path_str = str(entry.checkpoint_path)
    else:
        target_path_str = str(
            resolve_variant_checkpoint(entry.checkpoint_path, variant)
        )

    active_id = getattr(app.state, "active_model_id", None)
    active_path = str(getattr(app.state, "active_checkpoint_path", ""))

    if active_id == base_id and active_path == target_path_str:
        return  # already loaded

    device = getattr(
        app.state,
        "device",
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    release_current_model(app)

    if variant is None:
        model, config, ckpt_path = load_model(base_id, device)
    else:
        model, config, ckpt_path = load_checkpoint(
            base_id, Path(target_path_str), device
        )

    app.state.model = model
    app.state.model_config = config
    app.state.active_model_id = base_id
    app.state.active_checkpoint_path = ckpt_path
    # Reset RLHF state after any model swap.
    app.state.rlhf_optimizer = None
    app.state.rlhf_pending = None
