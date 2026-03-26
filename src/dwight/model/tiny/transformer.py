from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt

from ..transformer_block import RMSNorm
from .bigram_hash import BigramHashEmbedding
from .config import TinyModelConfig
from .transformer_block import TinyTransformerBlock
from .vocab_embed import FactoredVocabEmbed


class TinyModel(nn.Module):
    """Separate tiny model with grouped-query attention and bigram features."""

    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.config = config
        self._gradient_checkpointing = False
        self.quantization_enabled = False
        self._swa_snapshots: list[dict[str, torch.Tensor]] = []

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.bigram_embedding = BigramHashEmbedding(
            vocab_size=config.vocab_size,
            bigram_vocab_size=config.bigram_vocab_size,
            d_model=config.d_model,
        )
        self.emb_drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TinyTransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    dff=config.dff,
                    rope_dims=config.rope_dims,
                    max_seq_len=config.max_seq_len,
                    layer_index=index,
                    dropout=config.dropout,
                    ln_scale=config.ln_scale,
                )
                for index in range(config.num_layers)
            ]
        )
        self.vocab_residual = (
            FactoredVocabEmbed(config.vocab_size, config.ve_dim, config.d_model)
            if config.ve_enabled
            else None
        )
        self.ve_layers = set(config.ve_layers)
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight
        self.ema_shadow: dict[str, torch.Tensor] = {}
        if config.ema_enabled:
            self.reset_ema()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_ema(self) -> None:
        self.ema_shadow = {
            name: parameter.detach().clone()
            for name, parameter in self.named_parameters()
        }

    def _sync_ema_shadow(self) -> None:
        synced: dict[str, torch.Tensor] = {}
        for name, parameter in self.named_parameters():
            shadow = self.ema_shadow.get(name)
            if shadow is None or shadow.device != parameter.device or shadow.dtype != parameter.dtype:
                shadow = parameter.detach().clone()
            synced[name] = shadow
        self.ema_shadow = synced

    def update_ema(self) -> None:
        if not self.config.ema_enabled:
            return
        if not self.ema_shadow:
            self.reset_ema()
        self._sync_ema_shadow()
        for name, parameter in self.named_parameters():
            shadow = self.ema_shadow[name]
            shadow.mul_(self.config.ema_decay)
            shadow.add_(parameter.detach(), alpha=1.0 - self.config.ema_decay)

    def record_swa_snapshot(self) -> None:
        if not self.config.swa_enabled:
            return
        self._swa_snapshots.append({
            name: tensor.detach().clone() for name, tensor in self.state_dict().items()
        })

    def set_training_progress(self, progress: float) -> None:
        self.quantization_enabled = bool(self.config.late_qat and progress >= self.config.late_qat_threshold)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        x = self.token_embedding(tokens) + self.bigram_embedding(tokens)
        x = self.emb_drop(x)
        vocab_residual = self.vocab_residual(tokens) if self.vocab_residual is not None else None
        shared_index = self.config.num_layers - self.config.xsa_last_n if self.config.xsa_last_n else None
        shared_kv = None
        for index, block in enumerate(self.blocks):
            block_vocab_residual = vocab_residual if vocab_residual is not None and index in self.ve_layers else None
            use_shared = shared_index is not None and shared_kv is not None and index > shared_index
            kv_source = shared_kv if use_shared else None

            if self._gradient_checkpointing and self.training:
                def run_block(
                    hidden: torch.Tensor,
                    layer_block: TinyTransformerBlock = block,
                    layer_kv_source: tuple[torch.Tensor, torch.Tensor] | None = kv_source,
                    layer_vocab_residual: torch.Tensor | None = block_vocab_residual,
                ):
                    return layer_block(
                        hidden,
                        kv_source=layer_kv_source,
                        vocab_residual=layer_vocab_residual,
                    )

                x, kv = _ckpt(run_block, x, use_reentrant=False)
            else:
                x, kv = block(
                    x,
                    kv_source=kv_source,
                    vocab_residual=block_vocab_residual,
                )
            if shared_index is not None and index == shared_index:
                shared_kv = kv
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits[:, :seq_len]

    def enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = True

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        ids = list(prompt_ids)
        device = next(self.parameters()).device
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = ids[-self.config.max_seq_len :]
                logits = self(torch.tensor([context], dtype=torch.long, device=device))
                last_logits = logits[0, -1].float().cpu().numpy().astype(np.float64)
                if temperature <= 0.0:
                    token_id = int(np.argmax(last_logits))
                else:
                    last_logits = last_logits / temperature
                    token_id = _sample_top_p(last_logits, top_p)
                ids.append(token_id)
                yield token_id


def _sample_top_p(logits: np.ndarray, top_p: float) -> int:
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = (cumsum - sorted_probs) > top_p
    sorted_probs[cutoff] = 0.0
    sorted_probs /= sorted_probs.sum()
    sampled = np.random.choice(len(sorted_probs), p=sorted_probs)
    return int(sorted_idx[sampled])
