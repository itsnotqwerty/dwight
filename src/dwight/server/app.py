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
        model.load_state_dict(
            torch.load(_CHECKPOINT, weights_only=True, map_location=device)
        )
        print(f"Loaded weights from {_CHECKPOINT} (device: {device})")
    else:
        print(f"No checkpoint found – starting with random weights (device: {device}).")

    model.to(device)
    model.eval()
    app.state.model = model
    app.state.tokenizer = tokenizer
    yield
    # Nothing to clean up


def create_app() -> FastAPI:
    """Application factory (used by uvicorn with ``factory=True``)."""
    app = FastAPI(
        title="AIML – OpenAI-compatible LLM API",
        lifespan=_lifespan,
    )
    app.include_router(router)
    return app
