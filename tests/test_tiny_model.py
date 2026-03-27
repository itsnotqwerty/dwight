from __future__ import annotations

from dataclasses import replace

import torch

from dwight.model.tiny import TinyModel
from dwight.model.tiny.attention import GroupedQueryAttention
from dwight.model.tiny.bigram_hash import BigramHashEmbedding
from dwight.model.tiny.feed_forward import LeakyReluSquaredFF
from dwight.model.tiny.muon import ParameterBank
from dwight.model.tiny.muon import ParallelMuon
from dwight.model.tiny.quantize import (
    quantize_int4,
    quantize_int6,
    dequantize_int4,
    save_artifact,
    load_artifact,
)
from dwight.model.tiny.ttt import test_time_train
from dwight.model.tiny.vocab_embed import FactoredVocabEmbed


def test_bigram_hash_embedding_shape(tiny_arch_config):
    layer = BigramHashEmbedding(
        vocab_size=tiny_arch_config.vocab_size,
        bigram_vocab_size=tiny_arch_config.bigram_vocab_size,
        d_model=tiny_arch_config.d_model,
    )
    tokens = torch.randint(0, tiny_arch_config.vocab_size, (2, 8))
    out = layer(tokens)
    assert out.shape == (2, 8, tiny_arch_config.d_model)
    assert not any(name.startswith("_prime") for name, _ in layer.named_parameters())


def test_grouped_query_attention_output_and_xsa(tiny_arch_config):
    attn = GroupedQueryAttention(
        d_model=tiny_arch_config.d_model,
        num_heads=tiny_arch_config.num_heads,
        num_kv_heads=tiny_arch_config.num_kv_heads,
        rope_dims=tiny_arch_config.rope_dims,
        max_seq_len=tiny_arch_config.max_seq_len,
        dropout=0.0,
    )
    x = torch.randn(2, 8, tiny_arch_config.d_model)
    out, kv = attn(x)
    x2 = torch.randn(2, 8, tiny_arch_config.d_model)
    out_xsa, kv_xsa = attn(x2, kv_source=kv)
    assert out.shape == (2, 8, tiny_arch_config.d_model)
    assert out_xsa.shape == (2, 8, tiny_arch_config.d_model)
    assert torch.equal(kv[0], kv_xsa[0])
    assert torch.equal(kv[1], kv_xsa[1])


def test_leaky_relu_squared_activation_non_negative(tiny_arch_config):
    layer = LeakyReluSquaredFF(tiny_arch_config.d_model, tiny_arch_config.dff)
    x = torch.randn(2, 4, tiny_arch_config.d_model)
    activated = layer.activated(x)
    out = layer(x)
    assert torch.all(activated >= 0)
    assert out.shape == (2, 4, tiny_arch_config.d_model)


def test_factored_vocab_embed_shape(tiny_arch_config):
    embed = FactoredVocabEmbed(
        tiny_arch_config.vocab_size,
        tiny_arch_config.ve_dim,
        tiny_arch_config.d_model,
    )
    tokens = torch.randint(0, tiny_arch_config.vocab_size, (2, 6))
    assert embed(tokens).shape == (2, 6, tiny_arch_config.d_model)


def test_tiny_model_forward_shape(tiny_arch_model, tiny_arch_config):
    tokens = torch.randint(0, tiny_arch_config.vocab_size, (2, 8))
    with torch.no_grad():
        logits = tiny_arch_model(tokens)
    assert logits.shape == (2, 8, tiny_arch_config.vocab_size)


def test_tiny_model_generate_outputs_ids(tiny_arch_model, tiny_arch_config):
    generated = list(
        tiny_arch_model.generate([1, 2, 3], max_new_tokens=4, temperature=0.0)
    )
    assert len(generated) == 4
    assert all(0 <= token < tiny_arch_config.vocab_size for token in generated)


def test_parallel_muon_step_runs_bank_orthogonalization(tiny_arch_config, monkeypatch):
    model = TinyModel(tiny_arch_config)
    optimizer = ParallelMuon(
        model,
        matrix_lr=tiny_arch_config.matrix_lr,
        scalar_lr=tiny_arch_config.scalar_lr,
        tied_embed_lr=tiny_arch_config.tied_embed_lr,
        momentum=tiny_arch_config.muon_momentum,
        matrix_weight_decay=tiny_arch_config.muon_wd,
        scalar_weight_decay=tiny_arch_config.adam_wd,
    )
    called = False

    def orthogonalize() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(optimizer.bank, "orthogonalize_", orthogonalize)
    loss = model(torch.randint(0, tiny_arch_config.vocab_size, (1, 8))).float().mean()
    loss.backward()

    optimizer.step()

    assert called is True


def test_parameter_bank_orthogonalize_runs_via_cpu_and_preserves_device():
    linear = torch.nn.Linear(8, 4, bias=False)
    original = linear.weight.detach().clone()
    bank = ParameterBank(parameters=[linear.weight])

    bank.orthogonalize_()

    assert linear.weight.device == original.device
    assert linear.weight.dtype == original.dtype
    assert not torch.equal(linear.weight.detach(), original)


def test_parameter_bank_orthogonalize_uses_smaller_gram_for_tall_matrix(monkeypatch):
    parameter = torch.nn.Parameter(torch.randn(128, 8))
    bank = ParameterBank(parameters=[parameter])
    seen_shapes: list[tuple[int, ...]] = []
    original_newton_schulz5 = __import__(
        "dwight.model.tiny.muon", fromlist=["newton_schulz5"]
    ).newton_schulz5

    def recording_newton_schulz5(
        matrix: torch.Tensor, steps: int = 5, eps: float = 1e-6
    ) -> torch.Tensor:
        seen_shapes.append(tuple(matrix.shape))
        return original_newton_schulz5(matrix, steps=steps, eps=eps)

    monkeypatch.setattr(
        "dwight.model.tiny.muon.newton_schulz5", recording_newton_schulz5
    )

    bank.orthogonalize_()

    assert seen_shapes == [(8, 8)]


def test_tiny_model_gradient_checkpointing_forward_path(tiny_arch_config):
    from dwight.model.tiny import TinyModel

    model = TinyModel(tiny_arch_config)
    model.enable_gradient_checkpointing()
    model.train()
    tokens = torch.randint(0, tiny_arch_config.vocab_size, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, tiny_arch_config.vocab_size)


def test_tiny_model_gradient_checkpointing_backward_path(tiny_arch_config):
    from dwight.model.tiny import TinyModel

    model = TinyModel(tiny_arch_config)
    model.enable_gradient_checkpointing()
    model.train()
    tokens = torch.randint(0, tiny_arch_config.vocab_size, (2, 8))
    logits = model(tokens)
    loss = logits.float().mean()
    loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_tiny_model_ema_updates(tiny_arch_model):
    model = tiny_arch_model
    model.reset_ema()
    before = model.ema_shadow["token_embedding.weight"].clone()
    with torch.no_grad():
        model.token_embedding.weight.add_(0.1)
    model.update_ema()
    after = model.ema_shadow["token_embedding.weight"]
    assert not torch.equal(before, after)
    assert after.device.type == "cpu"


def test_tiny_model_ema_resyncs_shadow_dtype_on_cpu(tiny_arch_model):
    model = tiny_arch_model
    model.reset_ema()
    model.ema_shadow["token_embedding.weight"] = model.ema_shadow[
        "token_embedding.weight"
    ].double()
    model.update_ema()
    shadow = model.ema_shadow["token_embedding.weight"]
    weight = model.token_embedding.weight
    assert shadow.dtype == weight.dtype
    assert shadow.device.type == "cpu"


def test_tiny_model_swa_snapshots_stay_on_cpu(tiny_arch_model):
    tiny_arch_model.record_swa_snapshot()
    snapshot = tiny_arch_model._swa_snapshots[0]
    assert snapshot["token_embedding.weight"].device.type == "cpu"


def test_tiny_model_offloads_auxiliary_state_to_cpu(tiny_arch_model):
    tiny_arch_model.reset_ema()
    tiny_arch_model.record_swa_snapshot()
    tiny_arch_model.offload_auxiliary_state_to_cpu()
    assert all(
        tensor.device.type == "cpu" for tensor in tiny_arch_model.ema_shadow.values()
    )
    assert all(
        tensor.device.type == "cpu"
        for snapshot in tiny_arch_model._swa_snapshots
        for tensor in snapshot.values()
    )


def test_quantize_int6_returns_expected_payload():
    tensor = torch.randn(33)
    payload = quantize_int6(tensor, group_size=8)
    assert payload["shape"] == (33,)
    assert payload["group_size"] == 8
    # Legacy alias delegates to int4 — packed_values is uint8, numel present
    assert payload["packed_values"].dtype == torch.uint8
    assert payload["numel"] == 33


def test_quantize_int4_returns_expected_payload():
    tensor = torch.randn(33)
    payload = quantize_int4(tensor, group_size=8)
    assert payload["shape"] == (33,)
    assert payload["group_size"] == 8
    assert payload["numel"] == 33
    assert payload["packed_values"].dtype == torch.uint8
    # Half as many bytes as elements (rounded up)
    assert payload["packed_values"].numel() == (33 + 1) // 2
    assert payload["scales"].dtype == torch.float32
    assert payload["zeros"].dtype == torch.float32


def test_quantize_int4_roundtrip_within_tolerance():
    torch.manual_seed(0)
    tensor = torch.randn(512)
    payload = quantize_int4(tensor, group_size=64)
    recovered = dequantize_int4(payload)
    assert recovered.shape == tensor.shape
    # Max error within int4 range should be < half a step ≈ range/30 per group
    assert torch.allclose(tensor, recovered, atol=0.4)


def test_save_load_artifact_roundtrip(tiny_arch_config, tmp_path):
    model_a = TinyModel(tiny_arch_config)
    model_a.reset_ema()
    artifact_path = str(tmp_path / "test_artifact.lzma")
    save_artifact(model_a, artifact_path, group_size=64)

    model_b = TinyModel(tiny_arch_config)
    load_artifact(model_b, artifact_path)

    # All loaded parameters should be finite
    for name, param in model_b.named_parameters():
        assert torch.isfinite(param).all(), f"{name} contains non-finite values"

    # Embedding weights should be close (within int4 quantization error)
    orig = model_a.token_embedding.weight.detach().float()
    loaded = model_b.token_embedding.weight.detach().float()
    assert torch.allclose(orig, loaded, atol=0.5), "token_embedding drift too large"


def test_save_artifact_uses_ema_shadow(tiny_arch_config, tmp_path):
    model = TinyModel(tiny_arch_config)
    model.reset_ema()
    # Corrupt live weights so they differ from EMA
    with torch.no_grad():
        model.token_embedding.weight.fill_(999.0)

    artifact_path = str(tmp_path / "ema_artifact.lzma")
    save_artifact(model, artifact_path, group_size=64)

    model2 = TinyModel(tiny_arch_config)
    load_artifact(model2, artifact_path)
    # The loaded embedding should reflect EMA (near-zero init), not the 999-fill
    assert model2.token_embedding.weight.abs().mean() < 1.0


def test_ttt_returns_score_per_chunk(tiny_arch_model, tiny_arch_config):
    tokens = torch.randint(0, tiny_arch_config.vocab_size, (40,)).tolist()
    cfg = replace(tiny_arch_config)
    cfg.ttt_chunk_tokens = 16
    scores = test_time_train(tiny_arch_model, tokens, cfg)
    assert len(scores) == 3
    assert all(isinstance(score, float) for score in scores)
