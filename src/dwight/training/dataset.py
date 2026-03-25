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


def _strip_html(text: str) -> str:
    """Convert HTML post content to plain text."""
    text = _BR_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    return html.unescape(text).strip()


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
                            yield _strip_html(com)


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

        for text in _iter_post_texts(self.archive_path):
            buf.extend(self.tokenizer.encode(text))
            buf.extend(sep)
            while len(buf) >= chunk:
                seq = buf[:chunk]
                buf = buf[chunk:]
                yield (
                    torch.tensor(seq[:-1], dtype=torch.long),
                    torch.tensor(seq[1:], dtype=torch.long),
                )


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
