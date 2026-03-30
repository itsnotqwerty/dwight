"""Tests for corpus fine-tuning and RLHF utilities."""

from __future__ import annotations

import threading
from click.testing import CliRunner
from typing import cast
from unittest.mock import patch

import torch
from fastapi import FastAPI
from starlette.testclient import TestClient

import dwight.__main__ as main_module
from dwight.model.transformer import GPTModel
from dwight.server.auth import require_auth
from dwight.server.ui_routes import ui_router
from dwight.training.dataset import (
    CorpusDataset,
    DEFAULT_DPO,
    DPODataset,
    PromptDataset,
    _parse_dpo_pairs,
    _parse_prompt_pairs,
    corpus_dataloader,
)
from dwight.training.finetune import (
    auto_rate_completion,
    dpo_finetune,
    dpo_loss,
    rlhf_step,
    sft_finetune,
)
from dwight.training.generate_dpo_prompts import (
    generate_dpo_examples,
    write_dpo_examples,
)
from dwight.training.generate_prompts import (
    CORPORATE_REGISTER_PHRASES,
    generate_prompt_examples,
    write_prompt_examples,
)


def _make_tune_client(model, tokenizer) -> TestClient:
    app = FastAPI()
    app.include_router(ui_router)
    app.dependency_overrides[require_auth] = lambda: None
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


class _AliveThread:
    def __init__(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive


class _ImmediateThread:
    def __init__(self, target, *args, **kwargs) -> None:
        self._target = target

    def start(self) -> None:
        self._target()

    def is_alive(self) -> bool:
        return False


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


def test_parse_prompt_pairs_returns_triples(tmp_path):
    prompts = tmp_path / "prompts.md"
    prompts.write_text(
        "[SYSTEM]\nSystem voice\n\n[USER]\nQuestion\n\n[ASSISTANT]\nAnswer\n\n---\n\n"
        "[SYSTEM]\nSystem 2\n\n[USER]\nQuestion 2\n\n[ASSISTANT]\nAnswer 2\n",
        encoding="utf-8",
    )

    pairs = _parse_prompt_pairs(prompts)

    assert pairs == [
        ("System voice", "Question", "Answer"),
        ("System 2", "Question 2", "Answer 2"),
    ]


def test_prompt_dataset_masks_prompt_tokens(tokenizer):
    with (
        patch(
            "dwight.training.dataset._parse_prompt_pairs",
            return_value=[("System", "User", "Response")],
        ),
        patch.object(
            tokenizer,
            "encode",
            side_effect=lambda text: (
                [10, 11, 12, 13] if text.startswith("[SYSTEM]") else [20] * 10
            ),
        ),
    ):
        ds = PromptDataset("ignored.md", tokenizer, seq_len=8)
        inp, tgt = next(iter(ds))

    assert inp.shape == (8,)
    assert tgt.shape == (8,)
    assert (tgt[:3] == -100).all()
    assert (tgt[3:8] == 20).all()


def test_prompt_dataset_does_not_cross_example_boundaries(tokenizer):
    pairs = [
        ("System 1", "User 1", "First"),
        ("System 2", "User 2", "Second"),
    ]

    def fake_encode(text: str) -> list[int]:
        if text.startswith("[SYSTEM]") and "System 1" in text:
            return [10, 11]
        if text.startswith("[SYSTEM]") and "System 2" in text:
            return [30, 31]
        if text == "First":
            return [20]
        if text == "Second":
            return [40]
        raise AssertionError(f"Unexpected text: {text!r}")

    with (
        patch("dwight.training.dataset._parse_prompt_pairs", return_value=pairs),
        patch.object(tokenizer, "encode", side_effect=fake_encode),
    ):
        ds = PromptDataset("ignored.md", tokenizer, seq_len=4)

        assert list(ds) == []


def test_generate_prompt_examples_count_and_domains():
    examples = generate_prompt_examples(count=70, seed=123)
    domains = {example.domain for example in examples}

    assert len(examples) == 70
    assert domains == {
        "politics",
        "news",
        "conspiracy",
        "memes",
        "self_expression",
        "adversarial",
        "greentext",
    }


def test_generate_prompt_examples_have_reply_variance():
    examples = generate_prompt_examples(count=140, seed=9)
    assistants = {example.assistant for example in examples}

    assert len(assistants) > 100


def test_generate_prompt_examples_mix_short_and_long_non_greentext_replies():
    examples = generate_prompt_examples(count=220, seed=19)
    word_counts = [
        len(example.assistant.split())
        for example in examples
        if example.domain != "greentext"
    ]

    assert any(count <= 18 for count in word_counts)
    assert any(count >= 35 for count in word_counts)


def test_generate_prompt_examples_avoid_corporate_register_phrases():
    examples = generate_prompt_examples(count=160, seed=23)
    bad_phrases = tuple(CORPORATE_REGISTER_PHRASES)

    assert all(
        phrase not in example.assistant.lower()
        for example in examples
        for phrase in bad_phrases
    )


def test_write_prompt_examples_round_trips(tmp_path):
    prompts = tmp_path / "prompts.md"
    examples = generate_prompt_examples(count=5, seed=7)

    write_prompt_examples(examples, prompts)
    parsed = _parse_prompt_pairs(prompts)

    assert len(parsed) == 5
    assert all(system and user and assistant for system, user, assistant in parsed)


def test_generate_dpo_examples_count_and_domains():
    examples = generate_dpo_examples(count=70, seed=123)
    domains = {example.domain for example in examples}

    assert len(examples) == 70
    assert domains == {
        "politics",
        "news",
        "conspiracy",
        "memes",
        "self_expression",
        "adversarial",
        "greentext",
    }


def test_generate_dpo_examples_emit_distinct_preferences():
    examples = generate_dpo_examples(count=80, seed=17)

    assert all(example.chosen and example.rejected for example in examples)
    assert all(example.chosen != example.rejected for example in examples)
    assert any(
        ">" in example.chosen for example in examples if example.domain == "greentext"
    )


def test_write_dpo_examples_round_trips(tmp_path):
    dpo_path = tmp_path / "dpo.md"
    examples = generate_dpo_examples(count=5, seed=11)

    write_dpo_examples(examples, dpo_path)
    parsed = _parse_dpo_pairs(dpo_path)

    assert len(parsed) == 5
    assert all(
        system and user and chosen and rejected
        for system, user, chosen, rejected in parsed
    )
    assert all(chosen != rejected for _system, _user, chosen, rejected in parsed)


def test_sft_finetune_saves_checkpoint(tiny_config, tokenizer, tmp_path):
    model = GPTModel(tiny_config)
    model.eval()

    corpus = tmp_path / "corpus.md"
    corpus.write_text("training text. " * 60, encoding="utf-8")

    logs: list[str] = []
    ckpt_dir = tmp_path / "ckpt"
    before = model.token_embedding.weight.detach().clone()

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

    assert (ckpt_dir / "model_tuned.pt").exists()
    assert any("[SFT] Fine-tuning finished." in line for line in logs)
    assert not torch.equal(before, model.token_embedding.weight.detach())
    assert model.training is False


def test_parse_dpo_pairs_returns_quadruples(tmp_path):
    prompts = tmp_path / "dpo.md"
    prompts.write_text(
        "[SYSTEM]\nSystem voice\n\n[USER]\nQuestion\n\n[CHOSEN]\nGood answer\n\n"
        "[REJECTED]\nBad answer\n\n---\n\n"
        "[SYSTEM]\nSystem 2\n\n[USER]\nQuestion 2\n\n[CHOSEN]\nGood 2\n\n"
        "[REJECTED]\nBad 2\n",
        encoding="utf-8",
    )

    pairs = _parse_dpo_pairs(prompts)

    assert pairs == [
        ("System voice", "Question", "Good answer", "Bad answer"),
        ("System 2", "Question 2", "Good 2", "Bad 2"),
    ]


def test_dpo_dataset_yields_shapes_and_masking(tokenizer):
    with (
        patch(
            "dwight.training.dataset._parse_dpo_pairs",
            return_value=[("System", "User", "Chosen", "Rejected")],
        ),
        patch.object(
            tokenizer,
            "encode",
            side_effect=lambda text: (
                [10, 11, 12]
                if text.startswith("[SYSTEM]")
                else [20] * 6 if text == "Chosen" else [30] * 6
            ),
        ),
    ):
        ds = DPODataset("ignored.md", tokenizer, seq_len=8)
        chosen_inp, chosen_tgt, rejected_inp, rejected_tgt = next(iter(ds))

    assert chosen_inp.shape == (8,)
    assert chosen_tgt.shape == (8,)
    assert rejected_inp.shape == (8,)
    assert rejected_tgt.shape == (8,)
    assert chosen_tgt[0].item() == -100
    assert rejected_tgt[0].item() == -100
    assert (chosen_tgt[1:7] == 20).all()
    assert (rejected_tgt[1:7] == 30).all()


def test_dpo_loss_direction():
    better = dpo_loss(
        torch.tensor([3.0]),
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([2.0]),
        beta=0.1,
    )
    worse = dpo_loss(
        torch.tensor([1.0]),
        torch.tensor([3.0]),
        torch.tensor([2.0]),
        torch.tensor([2.0]),
        beta=0.1,
    )

    assert better.item() < worse.item()


def test_dpo_finetune_saves_checkpoint_and_updates_params(
    tiny_config, tokenizer, tmp_path
):
    model = GPTModel(tiny_config)
    model.eval()

    dpo_path = tmp_path / "dpo.md"
    dpo_path.write_text(
        "[SYSTEM]\nYou are Dwight.\n\n[USER]\nGive me a blunt answer.\n\n"
        "[CHOSEN]\nThis is the sharper preferred response. "
        "This is the sharper preferred response. This is the sharper preferred response.\n\n"
        "[REJECTED]\nThis is the weaker rejected response. "
        "This is the weaker rejected response. This is the weaker rejected response.\n",
        encoding="utf-8",
    )

    logs: list[str] = []
    ckpt_dir = tmp_path / "ckpt_dpo"
    before = model.token_embedding.weight.detach().clone()

    dpo_finetune(
        model,
        tokenizer,
        tiny_config,
        dpo_path=str(dpo_path),
        epochs=1,
        batch_size=1,
        lr=1e-5,
        max_steps=1,
        stop_event=threading.Event(),
        log_fn=logs.append,
        checkpoint_dir=str(ckpt_dir),
    )

    assert (ckpt_dir / "model_dpo.pt").exists()
    assert any("[DPO] Fine-tuning finished." in line for line in logs)
    assert not torch.equal(before, model.token_embedding.weight.detach())
    assert model.training is False


def test_dpo_finetune_uses_default_dpo_path(
    tiny_config, tokenizer, tmp_path, monkeypatch
):
    model = GPTModel(tiny_config)
    model.eval()

    captured: dict[str, object] = {}

    def fake_dpo_dataloader(prompt_path, tokenizer, seq_len, batch_size):
        captured["prompt_path"] = prompt_path
        return []

    monkeypatch.setattr("dwight.training.finetune.dpo_dataloader", fake_dpo_dataloader)

    dpo_finetune(
        model,
        tokenizer,
        tiny_config,
        epochs=1,
        batch_size=1,
        checkpoint_dir=str(tmp_path / "ckpt_default_dpo"),
        log_fn=lambda _line: None,
    )

    assert captured["prompt_path"] == DEFAULT_DPO


def test_sft_finetune_uses_prompt_pairs_file(
    tiny_config, tokenizer, tmp_path, monkeypatch
):
    model = GPTModel(tiny_config)
    model.eval()

    prompts = tmp_path / "prompts.md"
    prompts.write_text(
        "[SYSTEM]\nYou are Dwight.\n\n[USER]\nSay something blunt.\n\n"
        "[ASSISTANT]\nThis is a long enough assistant answer to produce at least one training window. "
        "This is a long enough assistant answer to produce at least one training window.\n",
        encoding="utf-8",
    )

    called = False

    def fake_prompt_dataloader(*args, **kwargs):
        nonlocal called
        called = True
        inp = torch.tensor([[1] * 8], dtype=torch.long)
        tgt = torch.tensor([[2] * 8], dtype=torch.long)
        return [(inp, tgt)]

    def fail_corpus_dataloader(*args, **kwargs):
        raise AssertionError("corpus_dataloader should not be used for prompt corpora")

    monkeypatch.setattr(
        "dwight.training.finetune.prompt_dataloader", fake_prompt_dataloader
    )
    monkeypatch.setattr(
        "dwight.training.finetune.corpus_dataloader", fail_corpus_dataloader
    )

    sft_finetune(
        model,
        tokenizer,
        tiny_config,
        corpus_path=str(prompts),
        epochs=1,
        batch_size=1,
        max_steps=1,
        stop_event=threading.Event(),
        log_fn=lambda _line: None,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )

    assert called is True


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


def test_auto_rate_completion_prefers_punchy_structure():
    strong = (
        "The polished line is fake. Everyone involved is managing incentives, and the whole "
        "thing reads like theater dressed up as policy. Once that clicks, the story gets a lot "
        "less mysterious."
    )
    weak = (
        "In my opinion, it is important to note that, on the other hand, there are several "
        "perspectives worth noting here and perhaps the matter is more nuanced than it first appears."
    )

    assert auto_rate_completion(strong) > auto_rate_completion(weak)


def test_tune_page_renders(tiny_config, tokenizer):
    model = GPTModel(tiny_config)
    model.eval()
    client = _make_tune_client(model, tokenizer)

    resp = client.get("/tune")
    assert resp.status_code == 200
    assert "Fine-tuning" in resp.text
    assert "Direct Preference Optimization" in resp.text
    assert "Start DPO" in resp.text


def test_tune_dpo_start_rejects_missing_file(tiny_config, tokenizer, tmp_path):
    model = GPTModel(tiny_config)
    model.eval()
    client = _make_tune_client(model, tokenizer)

    resp = client.post(
        "/ui/tune/dpo/start",
        json={
            "dpo_path": str(tmp_path / "missing-dpo.md"),
            "epochs": 1,
            "batch_size": 1,
            "lr": 1e-5,
            "beta": 0.1,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "DPO file not found" in body["error"]


def test_tune_dpo_start_runs_finetune(tiny_config, tokenizer, tmp_path, monkeypatch):
    model = GPTModel(tiny_config)
    model.eval()
    client = _make_tune_client(model, tokenizer)
    dpo_path = tmp_path / "dpo.md"
    dpo_path.write_text(
        "[SYSTEM]\nSystem\n\n[USER]\nUser\n\n[CHOSEN]\nChosen\n\n[REJECTED]\nRejected\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_dpo_finetune(model, tokenizer, config, **kwargs):
        captured["dpo_path"] = kwargs["dpo_path"]
        captured["beta"] = kwargs["beta"]
        kwargs["log_fn"]("[DPO] synthetic log line")

    monkeypatch.setattr("dwight.training.finetune.dpo_finetune", fake_dpo_finetune)
    monkeypatch.setattr("dwight.server.ui_routes.threading.Thread", _ImmediateThread)

    resp = client.post(
        "/ui/tune/dpo/start",
        json={
            "dpo_path": str(dpo_path),
            "epochs": 2,
            "batch_size": 1,
            "lr": 2e-5,
            "beta": 0.2,
            "max_steps": 3,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert captured["dpo_path"] == str(dpo_path)
    assert captured["beta"] == 0.2
    app = cast(FastAPI, client.app)
    assert app.state.finetune_status == "idle"
    assert any("synthetic log line" in line for line in app.state.finetune_log_lines)


def test_tune_dpo_stop_sets_stop_event(tiny_config, tokenizer):
    model = GPTModel(tiny_config)
    model.eval()
    client = _make_tune_client(model, tokenizer)
    app = cast(FastAPI, client.app)
    app.state.finetune_thread = _AliveThread()

    resp = client.post("/ui/tune/dpo/stop")

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert app.state.finetune_stop_event.is_set() is True


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


def test_tune_rlhf_generate_rejects_while_dpo_running(tiny_config, tokenizer):
    model = GPTModel(tiny_config)
    model.eval()
    client = _make_tune_client(model, tokenizer)
    app = cast(FastAPI, client.app)
    app.state.finetune_thread = _AliveThread()
    app.state.finetune_status = "dpo"

    resp = client.post(
        "/ui/tune/rlhf/generate",
        json={
            "prompt": "Write a short sentence about testing.",
            "max_tokens": 2,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 3,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "Fine-tuning is running" in body["error"]


def test_tune_rlhf_rate_blends_human_and_auto_rewards(
    tiny_config, tokenizer, monkeypatch
):
    model = GPTModel(tiny_config)
    model.eval()
    client = _make_tune_client(model, tokenizer)
    client.app.state.rlhf_pending = {  # type: ignore
        "prompt": "Prompt",
        "completions": ["first", "second", "third"],
    }

    captured: dict[str, object] = {}

    def fake_auto_rate_completion(text: str) -> float:
        return {"first": 0.5, "second": -0.5, "third": 1.0}[text]

    def fake_rlhf_step(*args, **kwargs):
        captured["rewards"] = kwargs["rewards"]
        return 0.25

    monkeypatch.setattr(
        "dwight.server.ui_routes.auto_rate_completion", fake_auto_rate_completion
    )
    monkeypatch.setattr("dwight.training.finetune.rlhf_step", fake_rlhf_step)

    resp = client.post("/ui/tune/rlhf/rate", json={"ratings": [1, -1, 0]})

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert captured["rewards"] == [0.85, -0.85, 0.3]


def test_tune_rlhf_rate_requires_pending_round(tiny_config, tokenizer):
    model = GPTModel(tiny_config)
    model.eval()
    client = _make_tune_client(model, tokenizer)

    resp = client.post("/ui/tune/rlhf/rate", json={"ratings": [1, -1, 0]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "Generate completions first" in body["error"]


def test_cli_dpo_command_invokes_dpo_finetune(tiny_config, tmp_path, monkeypatch):
    model = GPTModel(tiny_config)
    model.eval()
    dpo_path = tmp_path / "dpo.md"
    dpo_path.write_text(
        "[SYSTEM]\nSystem\n\n[USER]\nUser\n\n[CHOSEN]\nChosen\n\n[REJECTED]\nRejected\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_load_model(model_id, device):
        captured["model_id"] = model_id
        return model, tiny_config, "checkpoints/model.pt"

    def fake_dpo_finetune(model, tokenizer, config, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(main_module, "load_model", fake_load_model)
    monkeypatch.setattr("dwight.training.finetune.dpo_finetune", fake_dpo_finetune)

    runner = CliRunner()
    result = runner.invoke(
        main_module.cli,
        [
            "dpo",
            "--dpo-path",
            str(dpo_path),
            "--epochs",
            "2",
            "--batch-size",
            "3",
            "--lr",
            "0.0002",
            "--beta",
            "0.3",
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--max-steps",
            "4",
            "--model",
            "dwight",
        ],
    )

    assert result.exit_code == 0
    assert captured["model_id"] == "dwight"
    assert captured["dpo_path"] == str(dpo_path)
    assert captured["epochs"] == 2
    assert captured["batch_size"] == 3
    assert captured["lr"] == 0.0002
    assert captured["beta"] == 0.3
    assert captured["max_steps"] == 4
