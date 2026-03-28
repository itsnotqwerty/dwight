"""Tests for Mixture-of-Experts feed-forward layer (MoEFeedForward)."""

from __future__ import annotations

import torch
import pytest

from dwight.config import ModelConfig
from dwight.model.moe import MoEFeedForward
from dwight.model.transformer import GPTModel
from dwight.model.transformer_block import TransformerBlock


# ── Helpers ───────────────────────────────────────────────────────────────────

BATCH, SEQ = 2, 8
D_MODEL = 32
NUM_EXPERTS = 4
NUM_ACTIVE = 2
NUM_SHARED = 1
EXPERT_HIDDEN = 16


def _moe() -> MoEFeedForward:
    return MoEFeedForward(
        d_model=D_MODEL,
        num_experts=NUM_EXPERTS,
        num_active=NUM_ACTIVE,
        num_shared=NUM_SHARED,
        expert_hidden_dim=EXPERT_HIDDEN,
    )


# ── MoEFeedForward ────────────────────────────────────────────────────────────


def test_moe_output_shape():
    moe = _moe()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out, aux = moe(x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


def test_moe_aux_loss_is_scalar():
    moe = _moe()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    _, aux = moe(x)
    assert aux.shape == ()  # 0-d scalar


def test_moe_aux_loss_is_positive():
    """Aux loss should be non-negative (product of non-negative quantities)."""
    moe = _moe()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    _, aux = moe(x)
    assert aux.item() >= 0.0


def test_moe_aux_loss_has_gradient():
    """Aux loss must produce gradients through the router weights."""
    moe = _moe()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out, aux = moe(x)
    aux.backward()
    # Router weight should have a gradient
    assert moe.router.weight.grad is not None
    assert not torch.all(moe.router.weight.grad == 0)


def test_moe_output_has_gradient():
    """Output must be differentiable w.r.t. input."""
    moe = _moe()
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
    out, _ = moe(x)
    out.sum().backward()
    assert x.grad is not None


def test_moe_all_tokens_receive_output():
    """Every token position must get a non-zero output (shared experts guarantee this)."""
    moe = _moe()
    moe.eval()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    with torch.no_grad():
        out, _ = moe(x)
    # With at least one shared expert, no output position should be all-zeros
    zero_rows = (out.view(-1, D_MODEL).abs().sum(-1) == 0).sum().item()
    assert zero_rows == 0


def test_moe_output_changes_with_input():
    moe = _moe()
    x1 = torch.randn(BATCH, SEQ, D_MODEL)
    x2 = torch.randn(BATCH, SEQ, D_MODEL)
    out1, _ = moe(x1)
    out2, _ = moe(x2)
    assert not torch.allclose(out1.detach(), out2.detach())


def test_moe_num_active_constraint():
    with pytest.raises(AssertionError):
        MoEFeedForward(
            d_model=D_MODEL,
            num_experts=2,
            num_active=5,  # more active than total → should fail
            num_shared=0,
            expert_hidden_dim=EXPERT_HIDDEN,
        )


# ── TransformerBlock with MoE ─────────────────────────────────────────────────


def test_block_with_moe_output_shape():
    from dwight.model.rope import precompute_freqs

    block = TransformerBlock(
        d_model=D_MODEL,
        num_heads=2,
        dff=64,
        use_moe=True,
        num_experts=NUM_EXPERTS,
        num_active_experts=NUM_ACTIVE,
        num_shared_experts=NUM_SHARED,
        expert_hidden_dim=EXPERT_HIDDEN,
    )
    freqs = precompute_freqs(D_MODEL // 2, SEQ)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out, aux = block(x, freqs)
    assert out.shape == (BATCH, SEQ, D_MODEL)
    assert aux.item() >= 0.0


def test_block_with_moe_aux_differentiable():
    from dwight.model.rope import precompute_freqs

    block = TransformerBlock(
        d_model=D_MODEL,
        num_heads=2,
        dff=64,
        use_moe=True,
        num_experts=NUM_EXPERTS,
        num_active_experts=NUM_ACTIVE,
        num_shared_experts=NUM_SHARED,
        expert_hidden_dim=EXPERT_HIDDEN,
    )
    freqs = precompute_freqs(D_MODEL // 2, SEQ)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out, aux = block(x, freqs)
    (out.sum() + aux).backward()
    # At least one router grad should be non-zero
    assert block.ffn.router.weight.grad is not None


# ── GPTModel with MoE ─────────────────────────────────────────────────────────


@pytest.fixture
def moe_config() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        d_model=32,
        num_heads=2,
        dff=64,
        vocab_size=100_277,
        max_seq_len=16,
        dropout=0.0,
        use_moe=True,
        num_experts=NUM_EXPERTS,
        num_active_experts=NUM_ACTIVE,
        num_shared_experts=NUM_SHARED,
        expert_hidden_dim=EXPERT_HIDDEN,
        moe_aux_loss_coeff=0.01,
    )


@pytest.fixture
def moe_model(moe_config: ModelConfig) -> GPTModel:
    model = GPTModel(moe_config)
    model.eval()
    return model


def test_gptmodel_moe_output_shape(moe_model, moe_config):
    x = torch.randint(0, moe_config.vocab_size, (2, 8))
    with torch.no_grad():
        logits, aux = moe_model(x)
    assert logits.shape == (2, 8, moe_config.vocab_size)


def test_gptmodel_moe_aux_loss_positive(moe_model):
    x = torch.randint(0, 100_277, (2, 8))
    with torch.no_grad():
        _, aux = moe_model(x)
    # aux should be positive (sum over all MoE blocks)
    assert aux.item() >= 0.0


def test_gptmodel_moe_loss_differentiable(moe_config):
    import torch.nn.functional as F

    model = GPTModel(moe_config)
    model.train()
    x = torch.randint(0, moe_config.vocab_size, (2, 8))
    y = torch.randint(0, moe_config.vocab_size, (2, 8))
    logits, aux = model(x)
    loss = F.cross_entropy(logits.view(-1, moe_config.vocab_size), y.view(-1))
    total = loss + moe_config.moe_aux_loss_coeff * aux
    total.backward()
    # Router weights in first MoE block should have gradients
    router = model.blocks[0].ffn.router
    assert router.weight.grad is not None


def test_gptmodel_moe_generate(moe_model):
    tokens = list(moe_model.generate([1, 2, 3], max_new_tokens=5))
    assert len(tokens) == 5
    assert all(isinstance(t, int) for t in tokens)


# ── MoE + MLA combined ────────────────────────────────────────────────────────


@pytest.fixture
def moe_mla_config() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        d_model=32,
        num_heads=2,
        dff=64,
        vocab_size=100_277,
        max_seq_len=16,
        dropout=0.0,
        use_mla=True,
        kv_latent_dim=12,
        q_latent_dim=24,
        qk_rope_dim=8,
        use_moe=True,
        num_experts=NUM_EXPERTS,
        num_active_experts=NUM_ACTIVE,
        num_shared_experts=NUM_SHARED,
        expert_hidden_dim=EXPERT_HIDDEN,
        moe_aux_loss_coeff=0.01,
    )


def test_gptmodel_moe_mla_combined(moe_mla_config):
    import torch.nn.functional as F

    model = GPTModel(moe_mla_config)
    model.eval()
    x = torch.randint(0, moe_mla_config.vocab_size, (2, 8))
    with torch.no_grad():
        logits, aux = model(x)
    assert logits.shape == (2, 8, moe_mla_config.vocab_size)
    assert aux.item() >= 0.0


def test_gptmodel_moe_mla_generate(moe_mla_config):
    model = GPTModel(moe_mla_config)
    model.eval()
    tokens = list(model.generate([1, 2, 3], max_new_tokens=4))
    assert len(tokens) == 4
