from __future__ import annotations

import io
import lzma

import torch


def quantize_int6(tensor: torch.Tensor, group_size: int = 128) -> dict:
    """Group-wise 6-bit affine quantization for checkpoint export."""
    flat = tensor.detach().float().view(-1)
    groups: list[torch.Tensor] = []
    scales: list[float] = []
    zeros: list[float] = []
    for start in range(0, flat.numel(), group_size):
        chunk = flat[start : start + group_size]
        min_val = float(chunk.min())
        max_val = float(chunk.max())
        scale = (max_val - min_val) / 63.0 if max_val != min_val else 1.0
        quantized = torch.clamp(torch.round((chunk - min_val) / scale), 0, 63).to(torch.uint8)
        groups.append(quantized)
        scales.append(scale)
        zeros.append(min_val)
    return {
        "shape": tuple(tensor.shape),
        "group_size": group_size,
        "values": torch.cat(groups),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "zeros": torch.tensor(zeros, dtype=torch.float32),
    }


def save_compressed(model: torch.nn.Module, path: str) -> None:
    payload = {name: quantize_int6(param) for name, param in model.state_dict().items() if torch.is_tensor(param)}
    raw = io.BytesIO()
    torch.save(payload, raw)
    with lzma.open(path, "wb") as handle:
        handle.write(raw.getvalue())
