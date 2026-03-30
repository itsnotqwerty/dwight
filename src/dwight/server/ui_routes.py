"""Web UI routes – only mounted when the server is started with ``--web-ui``."""

from __future__ import annotations

import asyncio
import datetime
import gc
import json
import sys
import threading
from dataclasses import fields
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import APIRouter, Depends, Form
from fastapi.requests import Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .auth import (
    check_password,
    delete_session_cookie,
    is_authenticated,
    login_redirect_response,
    require_auth,
    set_session_cookie,
)
from .generation import generate_tokens
from ..model.registry import MODEL_REGISTRY, get_model_entry, load_model
from ..training.dataset import DEFAULT_CORPUS, DEFAULT_DPO
from ..training.finetune import auto_rate_completion, tuned_checkpoint_name

ui_router = APIRouter()

_templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _active_model_id(app) -> str:
    return getattr(app.state, "active_model_id", "dwight")


def _available_models(app) -> list[str]:
    return list(getattr(app.state, "available_models", list(MODEL_REGISTRY)))


def _active_config(app):
    config = getattr(app.state, "model_config", None)
    if config is not None:
        return config
    return get_model_entry(_active_model_id(app)).config_cls()


def _active_checkpoint_path(app) -> Path:
    checkpoint_path = getattr(app.state, "active_checkpoint_path", None)
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    return Path(get_model_entry(_active_model_id(app)).checkpoint_path)


def _base_checkpoint_path(app) -> Path:
    checkpoint_path = _active_checkpoint_path(app)
    suffix = checkpoint_path.suffix or ".pt"
    for variant in (f"_tuned{suffix}", f"_dpo{suffix}"):
        if checkpoint_path.name.endswith(variant):
            base_name = checkpoint_path.name[: -len(variant)] + suffix
            return checkpoint_path.with_name(base_name)
    return checkpoint_path


def _tuned_checkpoint_path(app) -> Path:
    base_path = _base_checkpoint_path(app)
    return base_path.with_name(tuned_checkpoint_name(base_path.name))


def _dpo_checkpoint_path(app) -> Path:
    base_path = _base_checkpoint_path(app)
    suffix = base_path.suffix or ".pt"
    return base_path.with_name(f"{base_path.stem}_dpo{suffix}")


def _inference_checkpoint_info(app) -> dict:
    base_path = _base_checkpoint_path(app)
    tuned_path = _tuned_checkpoint_path(app)
    dpo_path = _dpo_checkpoint_path(app)
    active_path = _active_checkpoint_path(app)
    return {
        "active_path": str(active_path),
        "base_path": str(base_path),
        "base_name": base_path.name,
        "tuned_path": str(tuned_path),
        "tuned_name": tuned_path.name,
        "tuned_exists": tuned_path.exists(),
        "using_tuned": active_path == tuned_path,
        "dpo_path": str(dpo_path),
        "dpo_name": dpo_path.name,
        "dpo_exists": dpo_path.exists(),
        "using_dpo": active_path == dpo_path,
    }


def _checkpoint_info(app) -> dict:
    checkpoint_path = _active_checkpoint_path(app)
    checkpoint_info: dict = {
        "exists": checkpoint_path.exists(),
        "path": str(checkpoint_path),
        "name": checkpoint_path.name,
    }
    if checkpoint_info["exists"]:
        stat = checkpoint_path.stat()
        checkpoint_info["size_mb"] = round(stat.st_size / 1_048_576, 1)
        checkpoint_info["mtime"] = datetime.datetime.fromtimestamp(
            stat.st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")
    return checkpoint_info


def _artifact_info(app) -> dict | None:
    entry = get_model_entry(_active_model_id(app))
    if entry.artifact_path is None:
        return None
    artifact_path = Path(entry.artifact_path)
    info: dict = {
        "exists": artifact_path.exists(),
        "path": str(artifact_path),
        "name": artifact_path.name,
    }
    if info["exists"]:
        stat = artifact_path.stat()
        info["size_mb"] = round(stat.st_size / 1_048_576, 1)
        info["mtime"] = datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    return info


def _training_defaults(config) -> dict[str, int | float | str]:
    return {
        "epochs": 3,
        "batch_size": getattr(config, "train_batch_size", 8),
        "max_lr": 3e-4,
        "warmup_steps": 1000,
        "checkpoint_dir": "checkpoints",
        "grad_accum_steps": getattr(config, "train_grad_accum_steps", 1),
    }


def _release_current_model(app) -> None:
    current_model = getattr(app.state, "model", None)
    if current_model is None:
        return
    current_model.to("cpu")
    del current_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model_from_checkpoint(
    model_id: str, checkpoint_path: Path, device
) -> tuple[torch.nn.Module, object, str]:
    entry = get_model_entry(model_id)
    config = entry.config_cls()
    model = entry.model_cls(config)

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, config, str(checkpoint_path)


def _load_selected_model(
    app, model_id: str, checkpoint_path: Path | None = None
) -> tuple:
    device = getattr(
        app.state,
        "device",
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    _release_current_model(app)
    if checkpoint_path is None:
        return load_model(model_id, device)
    return _load_model_from_checkpoint(model_id, checkpoint_path, device)


# ---------------------------------------------------------------------------
# Inference UI
# ---------------------------------------------------------------------------


@ui_router.get("/")
async def inference_page(request: Request, _: None = Depends(require_auth)):
    return _templates.TemplateResponse(
        request,
        "inference.html",
        context={
            "available_models": _available_models(request.app),
            "active_model": _active_model_id(request.app),
            "inference_checkpoint": _inference_checkpoint_info(request.app),
        },
    )


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
    model_id: str = "dwight"
    epochs: int = 3
    batch_size: int = 8
    max_lr: float = 3e-4
    warmup_steps: int = 100
    checkpoint_dir: str = "checkpoints"
    max_steps: Optional[int] = None
    resume: bool = False
    grad_accum_steps: int = 1


class ModelSelectRequest(BaseModel):
    model_id: str


class InferenceCheckpointSelectRequest(BaseModel):
    use_tuned: bool = False
    use_dpo: bool = False


@ui_router.get("/train")
async def train_page(request: Request, _: None = Depends(require_auth)):
    app = request.app
    config = _active_config(app)
    config_fields = {f.name: getattr(config, f.name) for f in fields(config)}

    checkpoint_info = _checkpoint_info(app)
    process = app.state.training_process
    is_training = process is not None and process.returncode is None

    return _templates.TemplateResponse(
        request,
        "train.html",
        context={
            "config": config_fields,
            "checkpoint": checkpoint_info,
            "is_training": is_training,
            "available_models": _available_models(app),
            "active_model": _active_model_id(app),
            "training_defaults": _training_defaults(config),
            "artifact": _artifact_info(app),
        },
    )


@ui_router.post("/ui/model/select")
async def select_model(
    body: ModelSelectRequest, request: Request, _: None = Depends(require_auth)
):
    app = request.app
    if body.model_id not in MODEL_REGISTRY:
        return {"ok": False, "error": f"Unknown model: {body.model_id}"}

    process = getattr(app.state, "training_process", None)
    if process is not None and process.returncode is None:
        return {"ok": False, "error": "Cannot switch models while training is running."}

    thread = getattr(app.state, "finetune_thread", None)
    if thread is not None and thread.is_alive():
        return {
            "ok": False,
            "error": "Cannot switch models while fine-tuning is running.",
        }

    model, config, checkpoint_path = _load_selected_model(app, body.model_id)
    app.state.model = model
    app.state.model_config = config
    app.state.active_model_id = body.model_id
    app.state.active_checkpoint_path = checkpoint_path
    app.state.rlhf_optimizer = None
    app.state.rlhf_pending = None
    return {"ok": True, "model_id": body.model_id}


@ui_router.post("/ui/inference/checkpoint/select")
async def select_inference_checkpoint(
    body: InferenceCheckpointSelectRequest,
    request: Request,
    _: None = Depends(require_auth),
):
    app = request.app

    process = getattr(app.state, "training_process", None)
    if process is not None and process.returncode is None:
        return {
            "ok": False,
            "error": "Cannot switch checkpoints while training is running.",
        }

    thread = getattr(app.state, "finetune_thread", None)
    if thread is not None and thread.is_alive():
        return {
            "ok": False,
            "error": "Cannot switch checkpoints while fine-tuning is running.",
        }

    if body.use_tuned and body.use_dpo:
        return {"ok": False, "error": "Cannot select both tuned and DPO checkpoints."}

    model_id = _active_model_id(app)
    checkpoint_path: Path | None = None
    if body.use_tuned:
        checkpoint_path = _tuned_checkpoint_path(app)
        if not checkpoint_path.exists():
            return {
                "ok": False,
                "error": f"Fine-tuned checkpoint not found: {checkpoint_path}",
            }
    elif body.use_dpo:
        checkpoint_path = _dpo_checkpoint_path(app)
        if not checkpoint_path.exists():
            return {
                "ok": False,
                "error": f"DPO checkpoint not found: {checkpoint_path}",
            }

    model, config, active_checkpoint_path = _load_selected_model(
        app, model_id, checkpoint_path
    )
    app.state.model = model
    app.state.model_config = config
    app.state.active_checkpoint_path = active_checkpoint_path
    app.state.rlhf_optimizer = None
    app.state.rlhf_pending = None
    return {
        "ok": True,
        "model_id": model_id,
        "use_tuned": body.use_tuned,
        "use_dpo": body.use_dpo,
        "checkpoint_path": active_checkpoint_path,
    }


@ui_router.post("/ui/train/start")
async def train_start(
    body: TrainStartRequest, request: Request, _: None = Depends(require_auth)
):
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
        "--model",
        body.model_id,
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
    if body.resume:
        cmd.append("--resume")
    if body.grad_accum_steps > 1:
        cmd += ["--grad-accum-steps", str(body.grad_accum_steps)]

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
        # Reload the model from the freshly-written checkpoint so that
        # subsequent inference uses the trained weights, not the stale
        # in-memory weights from server startup.
        if process.returncode == 0:
            try:
                device = getattr(
                    app.state,
                    "device",
                    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                )
                model_id = getattr(app.state, "active_model_id", "dwight")
                model, config, checkpoint_path = load_model(model_id, device)
                app.state.model = model
                app.state.model_config = config
                app.state.active_checkpoint_path = checkpoint_path
                app.state.training_log_lines.append(
                    f"[server] Reloaded model weights from {checkpoint_path}."
                )
            except Exception as exc:
                app.state.training_log_lines.append(
                    f"[server] Warning: could not reload model after training: {exc}"
                )


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
# Fine-tuning UI  (/tune)
# ---------------------------------------------------------------------------


class TuneStartRequest(BaseModel):
    corpus_path: str = DEFAULT_CORPUS
    epochs: int = 1
    batch_size: int = 1
    lr: float = 1e-4
    max_steps: Optional[int] = None


class DpoStartRequest(BaseModel):
    dpo_path: str = DEFAULT_DPO
    epochs: int = 1
    batch_size: int = 1
    lr: float = 1e-5
    beta: float = 0.1
    max_steps: Optional[int] = None


class RlhfRoundRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    n: int = Field(default=3, ge=1, le=5)


class RlhfRateRequest(BaseModel):
    ratings: List[int]  # +1 (thumbs-up) or -1 (thumbs-down) per completion


@ui_router.get("/tune")
async def tune_page(request: Request, _: None = Depends(require_auth)):
    app = request.app
    config = _active_config(app)
    thread = app.state.finetune_thread
    is_finetune_running = thread is not None and thread.is_alive()

    return _templates.TemplateResponse(
        request,
        "tune.html",
        context={
            "checkpoint": _checkpoint_info(app),
            "config": {f.name: getattr(config, f.name) for f in fields(config)},
            "is_finetune_running": is_finetune_running,
            "finetune_status": app.state.finetune_status,
            "default_corpus": DEFAULT_CORPUS,
            "default_dpo": DEFAULT_DPO,
            "available_models": _available_models(app),
            "active_model": _active_model_id(app),
            "artifact": _artifact_info(app),
        },
    )


@ui_router.post("/ui/tune/sft/start")
async def tune_sft_start(
    body: TuneStartRequest, request: Request, _: None = Depends(require_auth)
):
    app = request.app
    thread = app.state.finetune_thread
    if thread is not None and thread.is_alive():
        return {"ok": False, "error": "Fine-tuning is already in progress."}

    corpus = Path(body.corpus_path)
    if not corpus.exists():
        return {"ok": False, "error": f"Corpus file not found: {body.corpus_path}"}

    # Fresh stop event for this run
    stop_event = threading.Event()
    app.state.finetune_stop_event = stop_event
    app.state.finetune_log_lines.clear()
    app.state.finetune_status = "sft"

    model = app.state.model
    tokenizer = app.state.tokenizer

    from ..training.finetune import sft_finetune, tuned_checkpoint_name

    config = _active_config(app)
    checkpoint_name = tuned_checkpoint_name(_active_checkpoint_path(app).name)

    def run():
        try:
            sft_finetune(
                model,
                tokenizer,
                config,
                corpus_path=body.corpus_path,
                epochs=body.epochs,
                batch_size=body.batch_size,
                lr=body.lr,
                max_steps=body.max_steps,
                stop_event=stop_event,
                log_fn=lambda line: app.state.finetune_log_lines.append(line),
                checkpoint_name=checkpoint_name,
            )
        except Exception as exc:
            app.state.finetune_log_lines.append(f"[SFT] Error: {exc}")
        finally:
            app.state.finetune_status = "idle"

    t = threading.Thread(target=run, daemon=True, name="sft-finetune")
    app.state.finetune_thread = t
    t.start()
    return {"ok": True}


@ui_router.post("/ui/tune/dpo/start")
async def tune_dpo_start(
    body: DpoStartRequest, request: Request, _: None = Depends(require_auth)
):
    app = request.app
    thread = app.state.finetune_thread
    if thread is not None and thread.is_alive():
        return {"ok": False, "error": "Fine-tuning is already in progress."}

    dpo_path = Path(body.dpo_path)
    if not dpo_path.exists():
        return {"ok": False, "error": f"DPO file not found: {body.dpo_path}"}

    stop_event = threading.Event()
    app.state.finetune_stop_event = stop_event
    app.state.finetune_log_lines.clear()
    app.state.finetune_status = "dpo"

    model = app.state.model
    tokenizer = app.state.tokenizer

    from ..training.finetune import dpo_finetune

    config = _active_config(app)

    def run():
        try:
            dpo_finetune(
                model,
                tokenizer,
                config,
                dpo_path=body.dpo_path,
                epochs=body.epochs,
                batch_size=body.batch_size,
                lr=body.lr,
                beta=body.beta,
                max_steps=body.max_steps,
                stop_event=stop_event,
                log_fn=lambda line: app.state.finetune_log_lines.append(line),
            )
        except Exception as exc:
            app.state.finetune_log_lines.append(f"[DPO] Error: {exc}")
        finally:
            app.state.finetune_status = "idle"

    t = threading.Thread(target=run, daemon=True, name="dpo-finetune")
    app.state.finetune_thread = t
    t.start()
    return {"ok": True}


@ui_router.post("/ui/tune/sft/stop")
async def tune_sft_stop(request: Request, _: None = Depends(require_auth)):
    app = request.app
    thread = app.state.finetune_thread
    if thread is None or not thread.is_alive():
        return {"ok": False, "error": "No fine-tuning is running."}
    app.state.finetune_stop_event.set()
    return {"ok": True}


@ui_router.post("/ui/tune/dpo/stop")
async def tune_dpo_stop(request: Request, _: None = Depends(require_auth)):
    app = request.app
    thread = app.state.finetune_thread
    if thread is None or not thread.is_alive():
        return {"ok": False, "error": "No fine-tuning is running."}
    app.state.finetune_stop_event.set()
    return {"ok": True}


@ui_router.get("/ui/tune/logs")
async def tune_logs_sse(request: Request, _: None = Depends(require_auth)):
    async def event_stream():
        last_status: Optional[str] = None
        lines: list[str] = request.app.state.finetune_log_lines
        offset = max(0, len(lines) - 50)

        while True:
            if await request.is_disconnected():
                break

            lines = request.app.state.finetune_log_lines
            while offset < len(lines):
                yield f"data: {json.dumps({'line': lines[offset].rstrip()})}\n\n"
                offset += 1

            status = request.app.state.finetune_status
            if status != last_status:
                yield f"data: {json.dumps({'type': 'status', 'value': status})}\n\n"
                last_status = status

            yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@ui_router.post("/ui/tune/rlhf/generate")
async def tune_rlhf_generate(
    body: RlhfRoundRequest, request: Request, _: None = Depends(require_auth)
):
    app = request.app
    thread = app.state.finetune_thread
    if thread is not None and thread.is_alive():
        return {
            "ok": False,
            "error": "Fine-tuning is running — wait for it to finish before starting an RLHF round.",
        }

    model = app.state.model
    tokenizer = app.state.tokenizer

    loop = asyncio.get_running_loop()

    def _generate_one() -> str:
        return "".join(
            generate_tokens(
                model,
                tokenizer,
                body.prompt,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
            )
        )

    completions: list[str] = []
    for _ in range(body.n):  # type: ignore
        text = await loop.run_in_executor(None, _generate_one)
        completions.append(text)

    app.state.rlhf_pending = {
        "prompt": body.prompt,
        "completions": completions,
    }

    return {"ok": True, "prompt": body.prompt, "completions": completions}


@ui_router.post("/ui/tune/rlhf/rate")
async def tune_rlhf_rate(
    body: RlhfRateRequest, request: Request, _: None = Depends(require_auth)
):
    app = request.app
    pending = app.state.rlhf_pending
    if pending is None:
        return {
            "ok": False,
            "error": "No pending RLHF round. Generate completions first.",
        }

    completions = pending["completions"]
    prompt = pending["prompt"]

    if len(body.ratings) != len(completions):
        return {
            "ok": False,
            "error": f"Expected {len(completions)} ratings, got {len(body.ratings)}.",
        }

    for r in body.ratings:
        if r not in (-1, 0, 1):
            return {"ok": False, "error": "Each rating must be -1, 0, or +1."}

    model = app.state.model
    tokenizer = app.state.tokenizer

    # Lazy-initialise RLHF optimizer (tiny lr to avoid catastrophic forgetting)
    if app.state.rlhf_optimizer is None:
        app.state.rlhf_optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-5, betas=(0.9, 0.95)
        )

    optimizer = app.state.rlhf_optimizer

    from ..training.finetune import rlhf_step

    config = _active_config(app)

    loop = asyncio.get_running_loop()

    def _run_step():
        blended_rewards = [
            (0.7 * float(rating)) + (0.3 * auto_rate_completion(completion))
            for rating, completion in zip(body.ratings, completions)
        ]
        return rlhf_step(
            model,
            optimizer,
            tokenizer,
            config,
            prompt=prompt,
            completions=completions,
            rewards=blended_rewards,
        )

    loss = await loop.run_in_executor(None, _run_step)
    app.state.rlhf_pending = None  # round consumed
    return {"ok": True, "loss": round(loss, 6)}


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
