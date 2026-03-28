"""Tests for Multi-Head Latent Attention (MLA)."""

from __future__ import annotations

import torch
import pytest

from dwight.config import ModelConfig
from dwight.model.mla import MultiHeadLatentAttention
from dwight.model.rope import precompute_freqs
from dwight.model.transformer import GPTModel
from dwight.model.transformer_block import TransformerBlock


# ── Helpers ───────────────────────────────────────────────────────────────────

BATCH, SEQ = 2, 8
D_MODEL, HEADS = 32, 2
HEAD_DIM = D_MODEL // HEADS  # 16
KV_LATENT = 12
Q_LATENT = 24
QK_ROPE = 8  # must be even


def _freqs(seq: int = SEQ) -> torch.Tensor:
    return precompute_freqs(QK_ROPE, max(seq, SEQ))[:seq]


def _mla(dropout: float = 0.0) -> MultiHeadLatentAttention:
    return MultiHeadLatentAttention(
        d_model=D_MODEL,
        num_heads=HEADS,
        kv_latent_dim=KV_LATENT,
        q_latent_dim=Q_LATENT,
        qk_rope_dim=QK_ROPE,
        dropout=dropout,
    )


# ── MultiHeadLatentAttention ──────────────────────────────────────────────────


def test_mla_output_shape():
    layer = _mla()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out, (c_KV, K_rope) = layer(x, _freqs())
    assert out.shape == (BATCH, SEQ, D_MODEL)


def test_mla_kv_cache_shapes():
    """The KV cache should be (c_KV, K_rope) with expected compressed sizes."""
    layer = _mla()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    _, (c_KV, K_rope) = layer(x, _freqs())
    # c_KV is the KV latent — stored instead of full K and V
    assert c_KV.shape == (BATCH, SEQ, KV_LATENT)
    # K_rope is the single-head decoupled rope key
    assert K_rope.shape == (BATCH, SEQ, QK_ROPE)


def test_mla_cache_smaller_than_standard_kv():
    """Cached (c_KV, K_rope) must consume fewer floats than standard (K, V)."""
    layer = _mla()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    _, (c_KV, K_rope) = layer(x, _freqs())
    cache_floats = c_KV.numel() + K_rope.numel()
    # Standard KV cache would be 2 × B × H × T × head_dim
    standard_kv_floats = 2 * BATCH * HEADS * SEQ * HEAD_DIM
    assert (
        cache_floats < standard_kv_floats
    ), f"MLA cache ({cache_floats}) should be smaller than standard KV ({standard_kv_floats})"


def test_mla_output_changes_with_input():
    """Different inputs must produce different outputs (non-trivial mapping)."""
    layer = _mla()
    x1 = torch.randn(BATCH, SEQ, D_MODEL)
    x2 = torch.randn(BATCH, SEQ, D_MODEL)
    out1, _ = layer(x1, _freqs())
    out2, _ = layer(x2, _freqs())
    assert not torch.allclose(out1, out2)


def test_mla_different_seq_lengths():
    layer = _mla()
    for seq in (1, 4, SEQ):
        x = torch.randn(BATCH, seq, D_MODEL)
        out, _ = layer(x, _freqs(seq))
        assert out.shape == (BATCH, seq, D_MODEL)


def test_mla_causal_masking():
    """MLA must be causal: changing future tokens must not affect past outputs."""
    layer = _mla()
    layer.eval()
    x = torch.randn(1, 5, D_MODEL)
    x_mod = x.clone()
    x_mod[0, 3:] = torch.randn(2, D_MODEL)  # alter positions 3 and 4

    with torch.no_grad():
        out_orig, _ = layer(x, _freqs(5))
        out_mod, _ = layer(x_mod, _freqs(5))

    # Positions 0-2 should be identical (future tokens don't affect past)
    assert torch.allclose(out_orig[0, :3], out_mod[0, :3], atol=1e-5)


# ── TransformerBlock with MLA ─────────────────────────────────────────────────


def test_block_with_mla_output_shape():
    block = TransformerBlock(
        d_model=D_MODEL,
        num_heads=HEADS,
        dff=64,
        use_mla=True,
        kv_latent_dim=KV_LATENT,
        q_latent_dim=Q_LATENT,
        qk_rope_dim=QK_ROPE,
    )
    x = torch.randn(BATCH, SEQ, D_MODEL)
    # TransformerBlock with MLA needs freqs of size (T, qk_rope_dim//2)
    freqs = _freqs()
    out, aux = block(x, freqs)
    assert out.shape == (BATCH, SEQ, D_MODEL)
    assert aux.item() == 0.0  # MLA alone has no aux loss


# ── GPTModel with MLA ─────────────────────────────────────────────────────────


@pytest.fixture
def mla_config() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        d_model=32,
        num_heads=2,
        dff=64,
        vocab_size=100_277,
        max_seq_len=16,
        dropout=0.0,
        use_mla=True,
        kv_latent_dim=KV_LATENT,
        q_latent_dim=Q_LATENT,
        qk_rope_dim=QK_ROPE,
    )


@pytest.fixture
def mla_model(mla_config: ModelConfig) -> GPTModel:
    model = GPTModel(mla_config)
    model.eval()
    return model


def test_gptmodel_mla_output_shape(mla_model, mla_config):
    x = torch.randint(0, mla_config.vocab_size, (2, 8))
    with torch.no_grad():
        logits, aux = mla_model(x)
    assert logits.shape == (2, 8, mla_config.vocab_size)
    assert aux.item() == 0.0


def test_gptmodel_mla_freqs_buffer_size(mla_model, mla_config):
    """Freqs buffer should reflect qk_rope_dim, not head_dim."""
    freqs = mla_model.freqs
    assert freqs.shape == (mla_config.max_seq_len, mla_config.qk_rope_dim // 2)


def test_gptmodel_mla_generate(mla_model):
    tokens = list(mla_model.generate([1, 2, 3], max_new_tokens=5))
    assert len(tokens) == 5
    assert all(isinstance(t, int) for t in tokens)
