from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from fastapi import FastAPI
from starlette.testclient import TestClient

from dwight.config import ModelConfig
from dwight.model.tiny import TinyModel
from dwight.model.transformer import GPTModel
from dwight.server.auth import require_auth
from dwight.server.ui_routes import ui_router


def _make_ui_client(
    model, tokenizer, config, active_checkpoint_path: str = "checkpoints/model.pt"
) -> TestClient:
    app = FastAPI()
    app.include_router(ui_router)
    app.dependency_overrides[require_auth] = lambda: None
    app.state.model = model
    app.state.model_config = config
    app.state.tokenizer = tokenizer
    app.state.device = "cpu"
    app.state.active_model_id = "dwight"
    app.state.active_checkpoint_path = active_checkpoint_path
    app.state.available_models = ["dwight", "tiny"]
    app.state.training_process = None
    app.state.training_log_lines = []
    app.state.finetune_thread = None
    app.state.finetune_log_lines = []
    app.state.finetune_status = "idle"
    app.state.finetune_stop_event = threading.Event()
    app.state.rlhf_optimizer = None
    app.state.rlhf_pending = None
    return TestClient(app)


def test_train_page_shows_active_model_config(tiny_config, tokenizer):
    client = _make_ui_client(GPTModel(tiny_config).eval(), tokenizer, tiny_config)
    response = client.get("/train")
    assert response.status_code == 200
    assert "num_layers" in response.text
    assert "num_kv_heads" not in response.text
    assert 'name="batch_size"' in response.text
    assert 'value="8"' in response.text


def test_model_select_switches_to_tiny(
    monkeypatch, tokenizer, tiny_config, tiny_arch_config
):
    import dwight.server.ui_routes as ui_routes_module

    client = _make_ui_client(GPTModel(tiny_config).eval(), tokenizer, tiny_config)

    def fake_load_model(model_id, device):
        model = TinyModel(tiny_arch_config).eval()
        return model, tiny_arch_config, "checkpoints/tiny.pt"

    monkeypatch.setattr(ui_routes_module, "load_model", fake_load_model)

    response = client.post("/ui/model/select", json={"model_id": "tiny"})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "model_id": "tiny"}

    train_page = client.get("/train")
    assert "num_kv_heads" in train_page.text
    assert "bigram_vocab_size" in train_page.text
    assert 'name="batch_size"' in train_page.text
    assert 'value="1"' in train_page.text
    assert 'name="grad_accum_steps"' in train_page.text
    assert 'value="8"' in train_page.text


def test_model_select_rejects_while_training_running(tokenizer, tiny_config):
    client = _make_ui_client(GPTModel(tiny_config).eval(), tokenizer, tiny_config)
    app = cast(FastAPI, client.app)
    app.state.training_process = SimpleNamespace(returncode=None)

    response = client.post("/ui/model/select", json={"model_id": "tiny"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "training" in body["error"].lower()


def test_inference_page_renders_model_selector(tiny_config, tokenizer):
    client = _make_ui_client(GPTModel(tiny_config).eval(), tokenizer, tiny_config)
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="model-select"' in response.text
    assert "tiny" in response.text


def test_inference_page_shows_tuned_toggle_when_checkpoint_exists(
    tiny_config, tokenizer, tmp_path
):
    base_ckpt = tmp_path / "model.pt"
    tuned_ckpt = tmp_path / "model_tuned.pt"
    base_ckpt.write_bytes(b"base")
    tuned_ckpt.write_bytes(b"tuned")

    client = _make_ui_client(
        GPTModel(tiny_config).eval(),
        tokenizer,
        tiny_config,
        active_checkpoint_path=str(base_ckpt),
    )

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="tuned-toggle"' in response.text
    assert "Use fine-tuned checkpoint" in response.text
    assert "model_tuned.pt" in response.text
    assert 'id="tuned-toggle" checked' not in response.text
    assert 'id="tuned-toggle" disabled' not in response.text


def test_inference_checkpoint_toggle_switches_to_tuned(
    monkeypatch, tiny_config, tokenizer, tmp_path
):
    import dwight.server.ui_routes as ui_routes_module

    base_ckpt = tmp_path / "model.pt"
    tuned_ckpt = tmp_path / "model_tuned.pt"
    base_ckpt.write_bytes(b"base")
    tuned_ckpt.write_bytes(b"tuned")

    client = _make_ui_client(
        GPTModel(tiny_config).eval(),
        tokenizer,
        tiny_config,
        active_checkpoint_path=str(base_ckpt),
    )

    def fake_load_model_from_checkpoint(model_id, checkpoint_path, device):
        assert model_id == "dwight"
        assert Path(checkpoint_path) == tuned_ckpt
        return GPTModel(tiny_config).eval(), tiny_config, str(checkpoint_path)

    monkeypatch.setattr(
        ui_routes_module,
        "load_checkpoint",
        fake_load_model_from_checkpoint,
    )

    response = client.post("/ui/inference/checkpoint/select", json={"use_tuned": True})

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "model_id": "dwight",
        "use_tuned": True,
        "use_dpo": False,
        "checkpoint_path": str(tuned_ckpt),
    }
    app = cast(FastAPI, client.app)
    assert app.state.active_checkpoint_path == str(tuned_ckpt)


def test_inference_checkpoint_toggle_rejects_missing_tuned(
    tiny_config, tokenizer, tmp_path
):
    base_ckpt = tmp_path / "model.pt"
    base_ckpt.write_bytes(b"base")

    client = _make_ui_client(
        GPTModel(tiny_config).eval(),
        tokenizer,
        tiny_config,
        active_checkpoint_path=str(base_ckpt),
    )

    response = client.post("/ui/inference/checkpoint/select", json={"use_tuned": True})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "Fine-tuned checkpoint not found" in body["error"]
