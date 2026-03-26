from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TinyModelConfig:
    """Configuration for the separate tiny architecture."""

    num_layers: int = 11
    d_model: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    dff: int = 1536
    vocab_size: int = 100_277
    bigram_vocab_size: int = 1536
    max_seq_len: int = 32_768
    train_seq_len: int = 2_048
    min_train_seq_len: int = 256
    train_batch_size: int = 1
    train_grad_accum_steps: int = 8
    train_gradient_checkpointing: bool = False
    dropout: float = 0.0
    rope_dims: int = 16
    xsa_last_n: int = 4
    ln_scale: bool = True
    ve_enabled: bool = True
    ve_dim: int = 128
    ve_layers: tuple[int, ...] = (9, 10)
    ema_enabled: bool = True
    ema_decay: float = 0.997
    swa_enabled: bool = True
    swa_every: int = 50
    late_qat: bool = True
    late_qat_threshold: float = 0.15
    matrix_lr: float = 0.025
    scalar_lr: float = 0.025
    tied_embed_lr: float = 0.035
    muon_momentum: float = 0.99
    muon_wd: float = 0.04
    adam_wd: float = 0.04
    ttt_lr: float = 0.002
    ttt_epochs: int = 3
    ttt_chunk_tokens: int = 32_768
    ttt_momentum: float = 0.9
    ttt_grad_clip: float = 1.0

    def __post_init__(self) -> None:
        head_dim = self.d_model // self.num_heads
        assert self.d_model % self.num_heads == 0, "d_model must divide num_heads"
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), "num_heads must divide num_kv_heads"
        assert self.rope_dims % 2 == 0, "rope_dims must be even"
        assert self.rope_dims <= head_dim, "rope_dims must fit within head_dim"
        assert (
            0 <= self.xsa_last_n <= self.num_layers
        ), "xsa_last_n must be within layer count"
        assert (
            1 <= self.train_seq_len <= self.max_seq_len
        ), "train_seq_len must be within max_seq_len"
        assert (
            1 <= self.min_train_seq_len <= self.train_seq_len
        ), "min_train_seq_len must be within train_seq_len"
        assert self.train_batch_size >= 1, "train_batch_size must be positive"
        assert (
            self.train_grad_accum_steps >= 1
        ), "train_grad_accum_steps must be positive"
