"""FastAPI application factory with lifespan model loading."""

from __future__ import annotations

import os

import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI

from ..config import ModelConfig
from ..model.transformer import GPTModel
from ..tokenizer import TiktokenWrapper
from .routes import router

_CHECKPOINT = os.path.join("checkpoints", "model.pt")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()
    tokenizer = TiktokenWrapper()
    model = GPTModel(config)

    if os.path.exists(_CHECKPOINT):
        ckpt = torch.load(_CHECKPOINT, weights_only=False, map_location=device)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        try:
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {_CHECKPOINT} (device: {device})")
        except RuntimeError as exc:
            print(
                f"Warning: checkpoint {_CHECKPOINT} is incompatible with the current "
                f"model architecture ({exc}). Starting with random weights."
            )
    else:
        print(f"No checkpoint found – starting with random weights (device: {device}).")

    model.to(device)
    model.eval()
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.training_process = None
    app.state.training_log_lines: list[str] = []  # type: ignore
    yield
    # Terminate any running training subprocess on shutdown
    if app.state.training_process is not None:
        try:
            app.state.training_process.terminate()
        except ProcessLookupError:
            pass


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
    return app
