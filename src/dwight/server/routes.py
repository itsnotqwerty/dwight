"""FastAPI route handlers for the OpenAI-compatible API."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..model.registry import MODEL_REGISTRY
from .generation import format_chat_prompt, generate_tokens
from .model_manager import list_model_variants, parse_model_id, swap_model_if_needed
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
    """Return all addressable model IDs, including existing tuned/DPO variants."""
    ids: list[str] = []
    for model_id, entry in MODEL_REGISTRY.items():
        ids.extend(list_model_variants(model_id, entry.checkpoint_path))
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": 1_700_000_000,
                "owned_by": "user",
            }
            for mid in ids
        ],
    }


@router.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, req: Request):
    """Create a chat completion (streaming or non-streaming)."""
    try:
        base_id, variant = parse_model_id(body.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Only do a checkpoint swap when a specific variant was requested
    # (colon syntax). Plain model names just use whatever is loaded.
    needs_swap = variant is not None

    tokenizer = req.app.state.tokenizer
    lock = getattr(req.app.state, "model_lock", None) if needs_swap else None

    prompt = format_chat_prompt(list(body.messages))
    prompt_tokens = len(tokenizer.encode(prompt))

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    max_tok = body.max_tokens or 256

    if body.stream:
        return StreamingResponse(
            _sse_stream(
                req.app,
                lock,
                base_id,
                variant,
                tokenizer,
                body,
                prompt,
                completion_id,
                created,
                max_tok,
            ),
            media_type="text/event-stream",
        )

    # ── Non-streaming ────────────────────────────────────────────────────────
    if lock is not None:
        async with lock:
            swap_model_if_needed(req.app, base_id, variant)
            model = req.app.state.model
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
    else:
        model = req.app.state.model
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


async def _sse_stream(
    app, lock, base_id, variant, tokenizer, body, prompt, cid, created, max_tok
):
    """Async generator that yields SSE lines, holding the model lock for its duration."""

    async def _generate():
        if variant is not None:
            swap_model_if_needed(app, base_id, variant)
        model = app.state.model
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

    if lock is not None:
        async with lock:
            async for chunk in _generate():
                yield chunk
    else:
        async for chunk in _generate():
            yield chunk
