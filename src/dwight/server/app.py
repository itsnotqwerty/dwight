"""FastAPI application factory with lifespan model loading."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..model.registry import MODEL_REGISTRY, load_model
from ..tokenizer import TiktokenWrapper
from .routes import router


@asynccontextmanager
async def _lifespan(app: FastAPI):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TiktokenWrapper()
    model, config, checkpoint_path = load_model("dwight", device)
    app.state.model = model
    app.state.model_config = config
    app.state.active_model_id = "dwight"
    app.state.active_checkpoint_path = checkpoint_path
    app.state.available_models = list(MODEL_REGISTRY)
    app.state.model_lock = asyncio.Lock()
    app.state.device = device
    app.state.tokenizer = tokenizer
    app.state.training_process = None
    app.state.training_log_lines: list[str] = []  # type: ignore
    # Fine-tuning (in-process, background thread)
    app.state.finetune_thread: threading.Thread | None = None  # type: ignore
    app.state.finetune_log_lines: list[str] = []  # type: ignore
    app.state.finetune_status: str = "idle"  # type: ignore
    app.state.finetune_stop_event: threading.Event = threading.Event()  # type: ignore
    app.state.rlhf_optimizer = None  # lazy Adam; created on first RLHF step
    app.state.rlhf_pending: dict | None = None  # holds current round prompt+completions
    yield
    # Terminate any running training subprocess on shutdown
    if app.state.training_process is not None:
        try:
            app.state.training_process.terminate()
        except ProcessLookupError:
            pass
    # Signal any running fine-tune thread to stop
    app.state.finetune_stop_event.set()
    if app.state.finetune_thread is not None and app.state.finetune_thread.is_alive():
        app.state.finetune_thread.join(timeout=5)


def create_app() -> FastAPI:
    """Application factory (used by uvicorn with ``factory=True``)."""
    app = FastAPI(
        title="Dwight – OpenAI-compatible LLM API",
        lifespan=_lifespan,
    )
    app.include_router(router)
    if os.environ.get("DWIGHT_WEB_UI") == "1":
        from .auth import _LoginRedirect, login_redirect_response
        from .ui_routes import ui_router

        app.add_exception_handler(
            _LoginRedirect, lambda req, exc: login_redirect_response()
        )
        app.include_router(ui_router)
        app.mount(
            "/static",
            StaticFiles(directory=Path(__file__).parent / "static"),
            name="static",
        )
    return app
