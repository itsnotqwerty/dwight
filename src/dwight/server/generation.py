"""Text generation helpers used by the API routes."""

from __future__ import annotations

from collections.abc import Iterator

from ..model.transformer import GPTModel
from ..tokenizer import TiktokenWrapper
from .schemas import ChatMessage


def format_chat_prompt(messages: list[ChatMessage]) -> str:
    """Format a list of chat messages into a plain-text prompt string."""
    parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            separator = "\n" if parts else ""
            parts.append(f"{separator}User: {msg.content}")
        elif msg.role in ("assistant", "developer"):
            parts.append(f"\nAssistant: {msg.content}")
    parts.append("\nAssistant:")
    return "".join(parts)


def generate_tokens(
    model: GPTModel,
    tokenizer: TiktokenWrapper,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Iterator[str]:
    """Yield decoded token strings one at a time."""
    prompt_ids = tokenizer.encode(prompt)
    for token_id in model.generate(
        prompt_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ):
        yield tokenizer.decode([token_id])
