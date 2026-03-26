"""FastAPI route handlers for the OpenAI-compatible API."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..model.registry import MODEL_REGISTRY
from .generation import format_chat_prompt, generate_tokens
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    StreamingChoice,
    UsageInfo,
)

router = APIRouter()


@router.get("/v1/models")
async def list_models(req: Request):
    """Return the list of available models."""
    available_models = getattr(req.app.state, "available_models", list(MODEL_REGISTRY))
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1_700_000_000,
                "owned_by": "user",
            }
            for model_id in available_models
        ],
    }


@router.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, req: Request):
    """Create a chat completion (streaming or non-streaming)."""
    model = req.app.state.model
    tokenizer = req.app.state.tokenizer

    prompt = format_chat_prompt(list(body.messages))
    prompt_tokens = len(tokenizer.encode(prompt))

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    max_tok = body.max_tokens or 256

    if body.stream:
        return StreamingResponse(
            _sse_stream(
                model, tokenizer, body, prompt, completion_id, created, max_tok
            ),
            media_type="text/event-stream",
        )

    # ── Non-streaming ────────────────────────────────────────────────────────
    pieces = list(
        generate_tokens(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tok,
            temperature=body.temperature,
            top_p=body.top_p,
        )
    )
    full_text = "".join(pieces)
    completion_tokens = len(tokenizer.encode(full_text))

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=body.model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=full_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _sse_stream(model, tokenizer, body, prompt, cid, created, max_tok):
    """Async generator that yields SSE lines."""
    # Opening chunk carries the role
    first = ChatCompletionChunk(
        id=cid,
        created=created,
        model=body.model,
        choices=[
            StreamingChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first.model_dump_json()}\n\n"

    for piece in generate_tokens(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tok,
        temperature=body.temperature,
        top_p=body.top_p,
    ):
        chunk = ChatCompletionChunk(
            id=cid,
            created=created,
            model=body.model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=DeltaMessage(content=piece),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Terminal chunk
    last = ChatCompletionChunk(
        id=cid,
        created=created,
        model=body.model,
        choices=[
            StreamingChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {last.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
