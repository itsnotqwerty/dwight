from __future__ import annotations

from dataclasses import dataclass

import torch


def newton_schulz5(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-6) -> torch.Tensor:
    """Approximate inverse square-root using a short Newton-Schulz iteration."""
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype).expand(matrix.shape[0], -1, -1)
    norm = matrix.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    y = matrix / norm
    z = identity.clone()
    for _ in range(steps):
        t = 0.5 * (3.0 * identity - torch.bmm(z, y))
        y = torch.bmm(y, t)
        z = torch.bmm(t, z)
    return z / torch.sqrt(norm)


@dataclass
class ParameterBank:
    """Minimal bank wrapper that groups matrix parameters for batched operations."""

    parameters: list[torch.nn.Parameter]

    @classmethod
    def from_module(cls, module: torch.nn.Module) -> "ParameterBank":
        matrices = [param for param in module.parameters() if param.ndim >= 2]
        return cls(parameters=matrices)

    def orthogonalize_(self) -> None:
        for param in self.parameters:
            rows = param.shape[0]
            flat = param.data.view(rows, -1)
            gram = flat @ flat.transpose(0, 1)
            inv_sqrt = newton_schulz5(gram.float()).squeeze(0).to(flat.dtype)
            param.data.copy_((inv_sqrt @ flat).view_as(param.data))


class ParallelMuon:
    """Composite optimizer: SGD-style matrix updates plus AdamW for the rest."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        matrix_lr: float,
        scalar_lr: float,
        tied_embed_lr: float,
        momentum: float,
        matrix_weight_decay: float,
        scalar_weight_decay: float,
    ) -> None:
        named_params = list(model.named_parameters())
        matrix_params = [param for _, param in named_params if param.ndim >= 2 and param.requires_grad]
        embed_params = [param for name, param in named_params if "token_embedding" in name and param.requires_grad]
        embed_ids = {id(param) for param in embed_params}
        scalar_params = [param for _, param in named_params if param.requires_grad and id(param) not in embed_ids and param.ndim < 2]

        matrix_only = [param for param in matrix_params if id(param) not in embed_ids]
        self.matrix_optimizer = torch.optim.SGD(
            [{"params": matrix_only, "lr": matrix_lr}],
            lr=matrix_lr,
            momentum=momentum,
            weight_decay=matrix_weight_decay,
        )
        self.scalar_optimizer = torch.optim.AdamW(
            [
                {"params": scalar_params, "lr": scalar_lr, "weight_decay": scalar_weight_decay},
                {"params": embed_params, "lr": tied_embed_lr, "weight_decay": scalar_weight_decay},
            ] if scalar_params or embed_params else [],
            lr=scalar_lr,
            betas=(0.9, 0.95),
        )
        self.param_groups = self.matrix_optimizer.param_groups + self.scalar_optimizer.param_groups
        self.bank = ParameterBank(parameters=matrix_only)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.matrix_optimizer.zero_grad(set_to_none=set_to_none)
        self.scalar_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.matrix_optimizer.step()
        self.scalar_optimizer.step()

    def state_dict(self) -> dict:
        return {
            "matrix": self.matrix_optimizer.state_dict(),
            "scalar": self.scalar_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.matrix_optimizer.load_state_dict(state_dict.get("matrix", {}))
        self.scalar_optimizer.load_state_dict(state_dict.get("scalar", {}))
