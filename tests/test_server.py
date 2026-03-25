"""Tests for the server: schemas, generation helpers, and API routes."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from dwight.server.generation import format_chat_prompt, generate_tokens
from dwight.server.routes import router
from dwight.server.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_client(tiny_model, tokenizer) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.model = tiny_model
    app.state.tokenizer = tokenizer
    return TestClient(app)


# ── Pydantic schemas ──────────────────────────────────────────────────────────


def test_chat_message_valid_roles():
    for role in ("system", "user", "assistant", "developer"):
        msg = ChatMessage(role=role, content="hi")
        assert msg.role == role


def test_chat_completion_request_defaults():
    req = ChatCompletionRequest(
        model="aiml",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    assert req.temperature == 1.0
    assert req.top_p == 1.0
    assert req.stream is False
    assert req.max_tokens == 256


def test_chat_completion_response_structure():
    from dwight.server.schemas import Choice, ChoiceMessage, UsageInfo

    resp = ChatCompletionResponse(
        model="aiml",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content="Hi!"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    assert resp.object == "chat.completion"
    assert resp.choices[0].message.content == "Hi!"
    assert resp.usage.total_tokens == 8


def test_completion_chunk_structure():
    from dwight.server.schemas import DeltaMessage, StreamingChoice

    chunk = ChatCompletionChunk(
        id="chatcmpl-abc",
        created=1700000000,
        model="aiml",
        choices=[
            StreamingChoice(
                index=0,
                delta=DeltaMessage(content="Hi"),
                finish_reason=None,
            )
        ],
    )
    assert chunk.object == "chat.completion.chunk"
    data = json.loads(chunk.model_dump_json())
    assert data["choices"][0]["delta"]["content"] == "Hi"


# ── format_chat_prompt ────────────────────────────────────────────────────────


def test_format_user_only():
    msgs = [ChatMessage(role="user", content="Hello")]
    prompt = format_chat_prompt(msgs)
    assert "User: Hello" in prompt
    assert prompt.endswith("Assistant:")


def test_format_system_and_user():
    msgs = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hi"),
    ]
    prompt = format_chat_prompt(msgs)
    assert "System: You are helpful." in prompt
    assert "User: Hi" in prompt
    assert prompt.endswith("Assistant:")


def test_format_multi_turn():
    msgs = [
        ChatMessage(role="user", content="Ping"),
        ChatMessage(role="assistant", content="Pong"),
        ChatMessage(role="user", content="Again"),
    ]
    prompt = format_chat_prompt(msgs)
    assert "User: Ping" in prompt
    assert "Assistant: Pong" in prompt
    assert "User: Again" in prompt
    assert prompt.endswith("Assistant:")


# ── generate_tokens ───────────────────────────────────────────────────────────


def test_generate_tokens_yields_strings(tiny_model, tokenizer):
    pieces = list(generate_tokens(tiny_model, tokenizer, "Hello!", max_tokens=5))
    assert len(pieces) == 5
    assert all(isinstance(p, str) for p in pieces)


def test_generate_tokens_greedy_deterministic(tiny_model, tokenizer):
    a = list(
        generate_tokens(tiny_model, tokenizer, "Hi", max_tokens=4, temperature=0.0)
    )
    b = list(
        generate_tokens(tiny_model, tokenizer, "Hi", max_tokens=4, temperature=0.0)
    )
    assert a == b


# ── API: GET /v1/models ───────────────────────────────────────────────────────


def test_list_models(tiny_model, tokenizer):
    client = _make_client(tiny_model, tokenizer)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) >= 1
    assert body["data"][0]["id"] == "aiml"


# ── API: POST /v1/chat/completions (non-streaming) ───────────────────────────


def test_non_streaming_response_structure(tiny_model, tokenizer):
    client = _make_client(tiny_model, tokenizer)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "aiml",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert "id" in body
    assert "created" in body
    assert len(body["choices"]) == 1
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "content" in body["choices"][0]["message"]
    assert "usage" in body


def test_non_streaming_usage_counts(tiny_model, tokenizer):
    client = _make_client(tiny_model, tokenizer)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "aiml",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4,
        },
    )
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] >= 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_invalid_role_returns_422(tiny_model, tokenizer):
    client = _make_client(tiny_model, tokenizer)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "aiml",
            "messages": [{"role": "badactor", "content": "hack"}],
        },
    )
    assert resp.status_code == 422


def test_missing_messages_returns_422(tiny_model, tokenizer):
    client = _make_client(tiny_model, tokenizer)
    resp = client.post("/v1/chat/completions", json={"model": "aiml"})
    assert resp.status_code == 422


# ── API: POST /v1/chat/completions (streaming) ───────────────────────────────


def test_streaming_response_content_type(tiny_model, tokenizer):
    client = _make_client(tiny_model, tokenizer)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "aiml",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 3,
        },
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


def test_streaming_response_ends_with_done(tiny_model, tokenizer):
    client = _make_client(tiny_model, tokenizer)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "aiml",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 3,
        },
    ) as resp:
        lines = [line for line in resp.iter_lines() if line]
    assert any("[DONE]" in line for line in lines)


def test_streaming_chunks_are_valid_json(tiny_model, tokenizer):
    client = _make_client(tiny_model, tokenizer)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "aiml",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 3,
        },
    ) as resp:
        lines = [line for line in resp.iter_lines() if line and line != "data: [DONE]"]

    for line in lines:
        assert line.startswith("data: ")
        payload = json.loads(line[6:])
        assert "choices" in payload
        assert payload["object"] == "chat.completion.chunk"


def test_streaming_first_chunk_has_role(tiny_model, tokenizer):
    client = _make_client(tiny_model, tokenizer)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "aiml",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 3,
        },
    ) as resp:
        lines = [
            line
            for line in resp.iter_lines()
            if line and "data:" in line and "[DONE]" not in line
        ]

    first = json.loads(lines[0][6:])
    assert first["choices"][0]["delta"].get("role") == "assistant"
