"""Streaming dataset from a 4chan /pol/ NDJSON archive (tar.zst)."""

from __future__ import annotations

import html
import json
import re
import tarfile
from pathlib import Path
from typing import Iterator

import torch
import zstandard as zstd
from torch.utils.data import DataLoader, IterableDataset

from ..tokenizer import TiktokenWrapper

DEFAULT_ARCHIVE = "data/4chan-pol.tar.zst"

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_REPLY_RE = re.compile(r">>\d+")
_BLANK_RE = re.compile(r"\n{3,}")
_MIN_CHARS = 20
_MAX_DIGIT_RATIO = 0.60


def _strip_html(text: str) -> str:
    """Convert HTML post content to plain text."""
    text = _BR_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    return html.unescape(text).strip()


def _clean_text(text: str) -> str | None:
    """Remove 4chan artifacts and return None for low-quality posts.

    Strips reply references (``>>NUMBER``), collapses excess blank lines,
    then rejects posts that are too short or mostly numeric.
    """
    text = _REPLY_RE.sub("", text)
    text = _BLANK_RE.sub("\n\n", text).strip()
    if len(text) < _MIN_CHARS:
        return None
    non_ws = [c for c in text if not c.isspace()]
    if non_ws and sum(c.isdigit() for c in non_ws) / len(non_ws) > _MAX_DIGIT_RATIO:
        return None
    return text


def _iter_post_texts(archive_path: str | Path) -> Iterator[str]:
    """Yield plain-text comment strings from every post in the archive."""
    dctx = zstd.ZstdDecompressor()
    with open(archive_path, "rb") as fh:
        with dctx.stream_reader(fh) as decomp:
            with tarfile.open(fileobj=decomp, mode="r|") as tf:
                member = next(iter(tf))
                f = tf.extractfile(member)
                if f is None:
                    return
                for raw in f:
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    for post in obj.get("posts", []):
                        com = post.get("com")
                        if com:
                            text = _clean_text(_strip_html(com))
                            if text is not None:
                                yield text


class ChanDataset(IterableDataset):
    """Streaming ``IterableDataset`` over a 4chan NDJSON ``tar.zst`` archive.

    Post texts are read sequentially, tokenized into a rolling buffer, and
    yielded as ``(input_ids, target_ids)`` tensor pairs of length *seq_len*.
    An EOT token is inserted between posts so the model learns boundaries.
    """

    def __init__(
        self,
        archive_path: str | Path,
        tokenizer: TiktokenWrapper,
        seq_len: int = 512,
    ) -> None:
        self.archive_path = Path(archive_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        chunk = self.seq_len + 1
        buf: list[int] = []
        sep = [self.tokenizer.eot_token]

        eot = self.tokenizer.eot_token
        for text in _iter_post_texts(self.archive_path):
            buf.extend(self.tokenizer.encode(text))
            buf.extend(sep)
            while len(buf) >= chunk:
                seq = buf[:chunk]
                buf = buf[chunk:]
                inp = torch.tensor(seq[:-1], dtype=torch.long)
                tgt = torch.tensor(seq[1:], dtype=torch.long)
                # Don't train the model to predict EOT — mask it out so
                # cross_entropy (ignore_index=-100) skips those positions.
                tgt[tgt == eot] = -100
                yield inp, tgt


def chan_dataloader(
    archive_path: str | Path,
    tokenizer: TiktokenWrapper,
    seq_len: int = 512,
    batch_size: int = 8,
) -> DataLoader:
    """Return a ``DataLoader`` streaming from a 4chan ``tar.zst`` archive."""
    dataset = ChanDataset(archive_path, tokenizer, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
    )
