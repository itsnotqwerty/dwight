from __future__ import annotations

from dataclasses import replace

import torch

from dwight.model.tiny.attention import GroupedQueryAttention
from dwight.model.tiny.bigram_hash import BigramHashEmbedding
from dwight.model.tiny.feed_forward import LeakyReluSquaredFF
from dwight.model.tiny.quantize import quantize_int6
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
    generated = list(tiny_arch_model.generate([1, 2, 3], max_new_tokens=4, temperature=0.0))
    assert len(generated) == 4
    assert all(0 <= token < tiny_arch_config.vocab_size for token in generated)


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


def test_tiny_model_ema_resyncs_shadow_dtype_and_device_metadata(tiny_arch_model):
    model = tiny_arch_model
    model.reset_ema()
    model.ema_shadow["token_embedding.weight"] = model.ema_shadow[
        "token_embedding.weight"
    ].double()
    model.update_ema()
    shadow = model.ema_shadow["token_embedding.weight"]
    weight = model.token_embedding.weight
    assert shadow.dtype == weight.dtype
    assert shadow.device == weight.device


def test_quantize_int6_returns_expected_payload():
    tensor = torch.randn(33)
    payload = quantize_int6(tensor, group_size=8)
    assert payload["shape"] == (33,)
    assert payload["group_size"] == 8
    assert payload["values"].dtype == torch.uint8


def test_ttt_returns_score_per_chunk(tiny_arch_model, tiny_arch_config):
    tokens = torch.randint(0, tiny_arch_config.vocab_size, (40,)).tolist()
    cfg = replace(tiny_arch_config)
    cfg.ttt_chunk_tokens = 16
    scores = test_time_train(tiny_arch_model, tokens, cfg)
    assert len(scores) == 3
    assert all(isinstance(score, float) for score in scores)
