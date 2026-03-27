from __future__ import annotations

import io
import lzma

import torch

# ---------------------------------------------------------------------------
# Int4 nibble packing helpers
# ---------------------------------------------------------------------------


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack a 1-D uint8 tensor of values in [0, 15] into half as many bytes.

    Even-indexed values occupy the low nibble; odd-indexed values the high
    nibble.  When numel is odd, the final high nibble is zero-padded.
    """
    flat = values.view(-1).to(torch.uint8)
    # Pad to even length
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, flat.new_zeros(1)])
    evens = flat[0::2]
    odds = flat[1::2]
    return (evens & 0xF) | ((odds & 0xF) << 4)


def unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack a byte tensor produced by :func:`pack_int4` into *numel* values."""
    packed = packed.view(-1).to(torch.uint8)
    out = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    out[0::2] = packed & 0xF
    out[1::2] = (packed >> 4) & 0xF
    return out[:numel]


# ---------------------------------------------------------------------------
# Group-wise int4 quantization / dequantization
# ---------------------------------------------------------------------------


def quantize_int4(tensor: torch.Tensor, group_size: int = 512) -> dict:
    """Group-wise 4-bit affine quantization with 2-values-per-byte packing."""
    flat = tensor.detach().float().view(-1)
    groups: list[torch.Tensor] = []
    scales: list[float] = []
    zeros: list[float] = []
    for start in range(0, flat.numel(), group_size):
        chunk = flat[start : start + group_size]
        min_val = float(chunk.min())
        max_val = float(chunk.max())
        scale = (max_val - min_val) / 15.0 if max_val != min_val else 1.0
        quantized = torch.clamp(torch.round((chunk - min_val) / scale), 0, 15).to(
            torch.uint8
        )
        groups.append(quantized)
        scales.append(scale)
        zeros.append(min_val)
    unpacked = torch.cat(groups)
    return {
        "shape": tuple(tensor.shape),
        "group_size": group_size,
        "numel": flat.numel(),
        "packed_values": pack_int4(unpacked),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "zeros": torch.tensor(zeros, dtype=torch.float32),
    }


def dequantize_int4(payload: dict) -> torch.Tensor:
    """Reconstruct a float32 tensor from a :func:`quantize_int4` payload."""
    unpacked = unpack_int4(payload["packed_values"], payload["numel"]).float()
    group_size = payload["group_size"]
    scales = payload["scales"]
    zeros = payload["zeros"]
    out = torch.empty_like(unpacked)
    for i, (scale, zero) in enumerate(zip(scales.tolist(), zeros.tolist())):
        start = i * group_size
        end = min(start + group_size, unpacked.numel())
        out[start:end] = unpacked[start:end] * scale + zero
    return out.reshape(payload["shape"])


# ---------------------------------------------------------------------------
# Artifact save / load
# ---------------------------------------------------------------------------

# Names to skip during export — restored by the model's __init__ weight-tying
# (lm_head.weight == token_embedding.weight) so storing it would be redundant.
_TIED_WEIGHT_NAMES: frozenset[str] = frozenset({"lm_head.weight"})


def save_artifact(
    model: torch.nn.Module,
    path: str,
    group_size: int = 512,
) -> None:
    """Export the model as a 16MB-class LZMA-compressed int4 artifact.

    Prefers EMA shadow weights when available (``model.ema_shadow`` is
    non-empty).  Tied weights (``lm_head.weight``) are omitted and restored
    automatically by the model's ``__init__`` on load.

    If the resulting artifact is still above 16 MB you can reduce size by
    increasing *group_size* to 1024 (saves ~0.7 MB in scale/zero metadata)
    or by accepting the slightly lower quality.
    """
    ema_shadow: dict = getattr(model, "ema_shadow", {})
    if ema_shadow:
        source: dict[str, torch.Tensor] = {
            name: shadow.to(dtype=torch.float32) for name, shadow in ema_shadow.items()
        }
    else:
        source = {
            name: tensor.detach().float()
            for name, tensor in model.state_dict().items()
            if torch.is_tensor(tensor)
        }

    payload = {
        name: quantize_int4(tensor, group_size)
        for name, tensor in source.items()
        if name not in _TIED_WEIGHT_NAMES
    }

    raw = io.BytesIO()
    torch.save(payload, raw)
    with lzma.open(path, "wb", preset=9) as handle:
        handle.write(raw.getvalue())


def _dequantize_int6_legacy(payload: dict) -> torch.Tensor:
    """Reconstruct a float32 tensor from the old per-byte uint8 format.

    The original ``save_compressed`` stored each quantized value as a
    separate ``uint8`` byte (range 0–63) under the key ``"values"``.
    """
    values = payload["values"].float()
    group_size = payload["group_size"]
    scales = payload["scales"]
    zeros = payload["zeros"]
    out = torch.empty_like(values)
    for i, (scale, zero) in enumerate(zip(scales.tolist(), zeros.tolist())):
        start = i * group_size
        end = min(start + group_size, values.numel())
        out[start:end] = values[start:end] * scale + zero
    return out.reshape(payload["shape"])


def load_artifact(model: torch.nn.Module, path: str) -> None:
    """Load weights from a :func:`save_artifact` file into *model* in-place.

    Supports both the current int4-packed format (``"packed_values"`` key)
    and the legacy per-byte int6 format (``"values"`` key) produced by the
    original ``save_compressed``.
    """
    with lzma.open(path, "rb") as handle:
        raw = io.BytesIO(handle.read())
    payload = torch.load(raw, weights_only=True)
    state_dict = {}
    for name, p in payload.items():
        if "packed_values" in p:
            state_dict[name] = dequantize_int4(p)
        else:
            state_dict[name] = _dequantize_int6_legacy(p)
    model.load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Legacy alias (kept for any code that still imports the old name)
# ---------------------------------------------------------------------------


def quantize_int6(tensor: torch.Tensor, group_size: int = 128) -> dict:
    """Deprecated: use :func:`quantize_int4` instead."""
    return quantize_int4(tensor, group_size=group_size)


def save_compressed(model: torch.nn.Module, path: str) -> None:
    """Deprecated: use :func:`save_artifact` instead."""
    save_artifact(model, path)
