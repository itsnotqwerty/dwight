"""Tests for corpus fine-tuning and RLHF utilities."""

from __future__ import annotations

import threading

import torch
from fastapi import FastAPI
from starlette.testclient import TestClient

from dwight.model.transformer import GPTModel
from dwight.server.ui_routes import ui_router
from dwight.training.dataset import CorpusDataset, corpus_dataloader
from dwight.training.finetune import rlhf_step, sft_finetune


def _make_tune_client(model, tokenizer) -> TestClient:
    app = FastAPI()
    app.include_router(ui_router)
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.training_process = None
    app.state.training_log_lines = []
    app.state.finetune_thread = None
    app.state.finetune_log_lines = []
    app.state.finetune_status = "idle"
    app.state.finetune_stop_event = threading.Event()
    app.state.rlhf_optimizer = None
    app.state.rlhf_pending = None
    return TestClient(app)


def test_corpus_dataset_shapes_and_shift(tmp_path, tokenizer):
    corpus = tmp_path / "corpus.md"
    corpus.write_text("A small corpus for testing. " * 200, encoding="utf-8")

    ds = CorpusDataset(corpus, tokenizer, seq_len=8)
    inp, tgt = next(iter(ds))

    assert inp.shape == (8,)
    assert tgt.shape == (8,)


def test_corpus_dataloader_returns_batches(tmp_path, tokenizer):
    corpus = tmp_path / "corpus.md"
    corpus.write_text("Batchable data. " * 300, encoding="utf-8")

    loader = corpus_dataloader(corpus, tokenizer, seq_len=8, batch_size=2)
    inp, tgt = next(iter(loader))

    assert inp.shape == (2, 8)
    assert tgt.shape == (2, 8)


def test_sft_finetune_saves_checkpoint(tiny_config, tokenizer, tmp_path):
    model = GPTModel(tiny_config)
    model.eval()

    corpus = tmp_path / "corpus.md"
    # Very short corpus is enough for this smoke test.
    corpus.write_text("hello world", encoding="utf-8")

    logs: list[str] = []
    ckpt_dir = tmp_path / "ckpt"

    sft_finetune(
        model,
        tokenizer,
        tiny_config,
        corpus_path=str(corpus),
        epochs=1,
        batch_size=1,
        lr=1e-4,
        max_steps=None,
        stop_event=threading.Event(),
        log_fn=logs.append,
        checkpoint_dir=str(ckpt_dir),
    )

    assert (ckpt_dir / "model.pt").exists()
    assert any("[SFT] Fine-tuning finished." in line for line in logs)
    assert model.training is False


def test_rlhf_step_runs_and_updates_params(tiny_config, tokenizer):
    model = GPTModel(tiny_config)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    before = model.token_embedding.weight.detach().clone()

    loss = rlhf_step(
        model,
        optimizer,
        tokenizer,
        tiny_config,
        prompt="Hello",
        completions=["This is better.", "This is worse.", "Neutral answer."],
        rewards=[1.0, -1.0, 0.0],
    )

    after = model.token_embedding.weight.detach()
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, after)


def test_tune_page_renders(tiny_config, tokenizer):
    model = GPTModel(tiny_config)
    model.eval()
    client = _make_tune_client(model, tokenizer)

    resp = client.get("/tune")
    assert resp.status_code == 200
    assert "Fine-tuning" in resp.text


def test_tune_rlhf_generate_and_rate(tiny_config, tokenizer):
    model = GPTModel(tiny_config)
    model.eval()
    client = _make_tune_client(model, tokenizer)

    generate_resp = client.post(
        "/ui/tune/rlhf/generate",
        json={
            "prompt": "Write a short sentence about testing.",
            "max_tokens": 2,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 3,
        },
    )
    assert generate_resp.status_code == 200
    body = generate_resp.json()
    assert body["ok"] is True
    assert len(body["completions"]) == 3

    rate_resp = client.post("/ui/tune/rlhf/rate", json={"ratings": [1, -1, 0]})
    assert rate_resp.status_code == 200
    rate_body = rate_resp.json()
    assert rate_body["ok"] is True
    assert isinstance(rate_body["loss"], float)


def test_tune_rlhf_rate_requires_pending_round(tiny_config, tokenizer):
    model = GPTModel(tiny_config)
    model.eval()
    client = _make_tune_client(model, tokenizer)

    resp = client.post("/ui/tune/rlhf/rate", json={"ratings": [1, -1, 0]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "Generate completions first" in body["error"]
