"""Web UI routes – only mounted when the server is started with ``--web-ui``."""

from __future__ import annotations

import asyncio
import datetime
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Form
from fastapi.requests import Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .auth import (
    check_password,
    delete_session_cookie,
    is_authenticated,
    login_redirect_response,
    require_auth,
    set_session_cookie,
)
from .generation import generate_tokens

ui_router = APIRouter()

_templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


# ---------------------------------------------------------------------------
# Inference UI
# ---------------------------------------------------------------------------


@ui_router.get("/")
async def inference_page(request: Request, _: None = Depends(require_auth)):
    return _templates.TemplateResponse(request, "inference.html")


@ui_router.get("/ui/generate")
async def generate_sse(
    request: Request,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    _: None = Depends(require_auth),
):
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer

    tokens: list[str] = []
    done = [False]
    interrupted = [False]

    def _run() -> None:
        for t in generate_tokens(
            model, tokenizer, prompt, max_tokens, temperature, top_p
        ):
            if interrupted[0]:
                break
            tokens.append(t)
        done[0] = True

    async def event_stream():
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(None, _run)
        offset = 0
        try:
            while True:
                # Drain any buffered tokens first
                while offset < len(tokens):
                    yield f"data: {json.dumps({'token': tokens[offset]})}\n\n"
                    offset += 1
                if done[0]:
                    break
                if await request.is_disconnected():
                    interrupted[0] = True
                    break
                await asyncio.sleep(0.05)
        finally:
            interrupted[0] = True
            try:
                await task
            except Exception:
                pass
        # Drain any final tokens emitted after done[0] was set
        while offset < len(tokens):
            yield f"data: {json.dumps({'token': tokens[offset]})}\n\n"
            offset += 1
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


# ---------------------------------------------------------------------------
# Training UI
# ---------------------------------------------------------------------------


class TrainStartRequest(BaseModel):
    epochs: int = 3
    batch_size: int = 8
    max_lr: float = 3e-4
    warmup_steps: int = 100
    checkpoint_dir: str = "checkpoints"
    max_steps: Optional[int] = None


@ui_router.get("/train")
async def train_page(request: Request, _: None = Depends(require_auth)):
    from ..config import ModelConfig

    config = ModelConfig()
    config_fields = {f.name: getattr(config, f.name) for f in fields(config)}

    checkpoint_path = Path("checkpoints") / "model.pt"
    checkpoint_info: dict = {"exists": checkpoint_path.exists()}
    if checkpoint_info["exists"]:
        stat = checkpoint_path.stat()
        checkpoint_info["size_mb"] = round(stat.st_size / 1_048_576, 1)
        checkpoint_info["mtime"] = datetime.datetime.fromtimestamp(
            stat.st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

    process = request.app.state.training_process
    is_training = process is not None and process.returncode is None

    return _templates.TemplateResponse(
        request,
        "train.html",
        context={
            "config": config_fields,
            "checkpoint": checkpoint_info,
            "is_training": is_training,
        },
    )


@ui_router.post("/ui/train/start")
async def train_start(body: TrainStartRequest, request: Request, _: None = Depends(require_auth)):
    app = request.app
    process = app.state.training_process
    if process is not None and process.returncode is None:
        return {"ok": False, "error": "Training is already in progress."}

    cmd = [
        sys.executable,
        "-u",  # unbuffered stdout so log lines appear in real time
        "-m",
        "dwight",
        "train",
        "--epochs",
        str(body.epochs),
        "--batch-size",
        str(body.batch_size),
        "--max-lr",
        str(body.max_lr),
        "--warmup-steps",
        str(body.warmup_steps),
        "--checkpoint-dir",
        body.checkpoint_dir,
    ]
    if body.max_steps is not None:
        cmd += ["--max-steps", str(body.max_steps)]

    new_process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    app.state.training_process = new_process
    app.state.training_log_lines.clear()
    asyncio.create_task(_stream_training_output(app))
    return {"ok": True}


@ui_router.post("/ui/train/stop")
async def train_stop(request: Request, _: None = Depends(require_auth)):
    process = request.app.state.training_process
    if process is None or process.returncode is not None:
        return {"ok": False, "error": "No training process is running."}
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    return {"ok": True}


async def _stream_training_output(app) -> None:
    """Background task: read subprocess stdout and append to log_lines.

    tqdm uses ``\\r`` (not ``\\n``) to overwrite progress lines, so we read
    raw chunks and split on both ``\\r`` and ``\\n`` so that every progress
    update is forwarded to the UI.
    """
    process = app.state.training_process
    buf = b""
    try:
        while True:
            chunk = await process.stdout.read(256)
            if not chunk:
                break
            buf += chunk
            # Normalise CR-only and CRLF to LF, then split into lines.
            parts = buf.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
            buf = parts[-1]  # last element may be an incomplete line
            for part in parts[:-1]:
                decoded = part.decode(errors="replace")
                if decoded:
                    app.state.training_log_lines.append(decoded)
    finally:
        # Flush any data left in the buffer after the pipe closes.
        if buf:
            decoded = buf.decode(errors="replace")
            if decoded:
                app.state.training_log_lines.append(decoded)
        await process.wait()


@ui_router.get("/ui/train/logs")
async def train_logs_sse(request: Request, _: None = Depends(require_auth)):
    async def event_stream():
        last_status: Optional[str] = None

        # Start from the tail of existing history (at most 50 lines).
        lines: list[str] = request.app.state.training_log_lines
        offset = max(0, len(lines) - 50)

        while True:
            if await request.is_disconnected():
                break

            # Send any new log lines
            lines = request.app.state.training_log_lines
            while offset < len(lines):
                yield f"data: {json.dumps({'line': lines[offset].rstrip()})}\n\n"
                offset += 1

            # Send a status event only when it changes
            process = request.app.state.training_process
            is_running = process is not None and process.returncode is None
            status = "training" if is_running else "idle"
            if status != last_status:
                yield f"data: {json.dumps({'type': 'status', 'value': status})}\n\n"
                last_status = status

            # SSE keepalive comment so proxies don't close the connection
            yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


# ---------------------------------------------------------------------------
# Auth – login / logout
# ---------------------------------------------------------------------------


@ui_router.get("/login")
async def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse("/", status_code=302)
    return _templates.TemplateResponse(request, "login.html", context={"error": False})


@ui_router.post("/login")
async def login_submit(
    request: Request,
    password: str = Form(...),
):
    if check_password(password):
        import os

        response = RedirectResponse("/", status_code=302)
        set_session_cookie(response, os.environ.get("DWIGHT_PASSWORD", ""))
        return response
    return _templates.TemplateResponse(
        request, "login.html", context={"error": True}, status_code=401
    )


@ui_router.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=302)
    delete_session_cookie(response)
    return response