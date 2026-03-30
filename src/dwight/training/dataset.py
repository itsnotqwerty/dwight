"""Streaming dataset from a 4chan /pol/ NDJSON archive (tar.zst)."""

from __future__ import annotations

import html
import json
import re
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import torch
import zstandard as zstd
from torch.utils.data import DataLoader, IterableDataset

from ..tokenizer import TiktokenWrapper

DEFAULT_ARCHIVE = "data/4chan-pol.tar.zst"
DEFAULT_DPO = "data/dpo.md"

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_REPLY_RE = re.compile(r">>\d+")
_REPLY_REF_RE = re.compile(r">>(\d+)")
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


def _iter_thread_posts(
    archive_path: str | Path,
) -> Iterator[list[tuple[int, str, list[int]]]]:
    """Yield one list of ``(post_no, clean_text, ref_nos)`` per thread.

    *ref_nos* preserves the order in which ``>>NUMBER`` references appear in
    the post, extracted before ``_clean_text`` strips them.
    """
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
                    thread: list[tuple[int, str, list[int]]] = []
                    for post in obj.get("posts", []):
                        no = post.get("no")
                        com = post.get("com")
                        if no is None or com is None:
                            continue
                        stripped = _strip_html(com)
                        refs = [int(m) for m in _REPLY_REF_RE.findall(stripped)]
                        clean = _clean_text(stripped)
                        if clean is None:
                            continue
                        thread.append((int(no), clean, refs))
                    if thread:
                        yield thread


def _iter_thread_token_sequences(
    archive_path: str | Path,
    tokenizer: TiktokenWrapper,
) -> Iterator[list[tuple[list[int], list[bool]]]]:
    """Yield one list of ``(token_ids, is_parent_mask)`` per thread.

    Each list element corresponds to one post in the thread, with parent-prefix
    tokens prepended and marked ``True`` in the mask.  Grouping by thread lets
    :class:`ChanDataset` reset its rolling buffer at thread boundaries so that
    training windows never span two unrelated threads.
    """
    eot = tokenizer.eot_token
    for thread in _iter_thread_posts(archive_path):
        text_by_no: dict[int, str] = {}
        thread_seqs: list[tuple[list[int], list[bool]]] = []
        for no, text, refs in thread:
            valid_parents = [text_by_no[r] for r in refs if r in text_by_no]
            parent_toks: list[int] = []
            for pt in valid_parents:
                parent_toks.extend(tokenizer.encode(pt))
                parent_toks.append(eot)
            reply_toks = tokenizer.encode(text)
            toks = parent_toks + reply_toks
            mask = [True] * len(parent_toks) + [False] * len(reply_toks)
            thread_seqs.append((toks, mask))
            # Make this post available as a parent for subsequent posts in the thread.
            text_by_no[no] = text
        yield thread_seqs


def _iter_token_sequences(
    archive_path: str | Path,
    tokenizer: TiktokenWrapper,
) -> Iterator[tuple[list[int], list[bool]]]:
    """Yield ``(token_ids, is_parent_mask)`` for each post in the archive.

    Flat wrapper around :func:`_iter_thread_token_sequences` for callers that
    do not need thread-level grouping.

    Posts with no resolvable parents yield a mask that is all ``False``.
    Cross-thread ``>>`` references that do not appear in ``text_by_no`` are
    silently skipped.
    """
    for thread_seqs in _iter_thread_token_sequences(archive_path, tokenizer):
        yield from thread_seqs


class ChanDataset(IterableDataset):
    """Streaming ``IterableDataset`` over a 4chan NDJSON ``tar.zst`` archive.

    Each post is emitted as ``(input_ids, target_ids)`` tensor pairs of length
    *seq_len*.  When a post contains reply references (``>>NUMBER``) that
    resolve to an earlier post in the same thread, the parent texts are
    prepended as a conditioning prefix (each separated by EOT).  Loss is
    computed **only on the reply tokens**; parent-prefix positions in the
    target are masked to ``-100`` so ``cross_entropy`` ignores them.

    Posts with no resolvable parents behave identically to the original flat
    streaming approach, so the dataset degrades gracefully for OP posts and
    posts whose references point outside the current thread.
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
        eot = self.tokenizer.eot_token

        for thread_seqs in _iter_thread_token_sequences(
            self.archive_path, self.tokenizer
        ):
            # Fresh buffer per thread — windows never span thread boundaries.
            buf: list[int] = []
            mask_buf: list[bool] = []

            for toks, mask in thread_seqs:
                buf.extend(toks)
                mask_buf.extend(mask)
                # Inter-post separator within the thread; masked so no loss is computed on it.
                buf.append(eot)
                mask_buf.append(True)

                while len(buf) >= chunk:
                    seq = buf[:chunk]
                    seq_mask = mask_buf[:chunk]
                    buf = buf[chunk:]
                    mask_buf = mask_buf[chunk:]
                    inp = torch.tensor(seq[:-1], dtype=torch.long)
                    tgt = torch.tensor(seq[1:], dtype=torch.long)
                    # Mask parent-prefix positions — gradient on reply tokens only.
                    is_parent = torch.tensor(seq_mask[1:], dtype=torch.bool)
                    tgt[is_parent] = -100
                    # Belt-and-suspenders: also mask any EOT that wasn't already.
                    tgt[tgt == eot] = -100
                    # Skip chunks where every target position is masked.
                    # cross_entropy with all targets == ignore_index returns NaN
                    # (0/0), which corrupts model weights permanently.
                    if not (tgt != -100).any():
                        continue
                    yield inp, tgt
            # Any remaining tokens in buf belong to this thread alone but don't
            # fill a complete window.  They are discarded here rather than
            # carried forward, which would contaminate the next thread.


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
        # The archive is a single compressed stream, so extra workers only
        # duplicate the read/decompression work and create multiprocessing
        # semaphores that can be left behind on interrupted retries.
        num_workers=0,
        pin_memory=_pin_memory_enabled(),
    )


# ---------------------------------------------------------------------------
# Plain-text corpus dataset (for fine-tuning on corpus.md or similar files)
# ---------------------------------------------------------------------------

DEFAULT_CORPUS = "data/corpus.md"
DEFAULT_PROMPTS = "data/prompts.md"


@lru_cache(maxsize=1)
def _pin_memory_enabled() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.empty(1).pin_memory()
    except Exception:
        return False
    return True


def _parse_tagged_blocks(
    prompt_path: str | Path,
    section_names: dict[str, str],
) -> list[tuple[str, ...]]:
    """Parse ``---``-delimited blocks with named tagged sections."""
    text = Path(prompt_path).read_text(encoding="utf-8", errors="replace")
    blocks = [
        block.strip() for block in re.split(r"(?m)^---\s*$", text) if block.strip()
    ]
    parsed: list[tuple[str, ...]] = []
    ordered_sections = tuple(dict.fromkeys(section_names.values()))

    for idx, block in enumerate(blocks, start=1):
        sections: dict[str, list[str]] = {name: [] for name in ordered_sections}
        current: str | None = None
        for raw_line in block.splitlines():
            stripped = raw_line.strip()
            if stripped in section_names:
                current = section_names[stripped]
                continue
            if current is None:
                if stripped:
                    raise ValueError(
                        f"Prompt block {idx} has text before a section header"
                    )
                continue
            sections[current].append(raw_line)

        values = tuple("\n".join(sections[name]).strip() for name in ordered_sections)
        if any(not value for value in values):
            expected = "/".join(name.strip("[]") for name in section_names)
            raise ValueError(f"Prompt block {idx} is missing one of {expected}")
        parsed.append(values)

    return parsed


def _parse_prompt_pairs(prompt_path: str | Path) -> list[tuple[str, str, str]]:
    """Parse ``[SYSTEM]/[USER]/[ASSISTANT]`` blocks from a prompt corpus."""
    parsed = _parse_tagged_blocks(
        prompt_path,
        {
            "[SYSTEM]": "system",
            "[USER]": "user",
            "[ASSISTANT]": "assistant",
        },
    )
    return [(system, user, assistant) for system, user, assistant in parsed]


def _parse_dpo_pairs(prompt_path: str | Path) -> list[tuple[str, str, str, str]]:
    """Parse ``[SYSTEM]/[USER]/[CHOSEN]/[REJECTED]`` blocks."""
    parsed = _parse_tagged_blocks(
        prompt_path,
        {
            "[SYSTEM]": "system",
            "[USER]": "user",
            "[CHOSEN]": "chosen",
            "[REJECTED]": "rejected",
        },
    )
    return [
        (system, user, chosen, rejected) for system, user, chosen, rejected in parsed
    ]


def _build_preference_pair(
    prompt_toks: list[int],
    response_toks: list[int],
    *,
    eot: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    chunk = seq_len + 1
    buf = prompt_toks + response_toks + [eot]
    mask_buf = [True] * len(prompt_toks) + [False] * len(response_toks) + [True]
    if len(buf) < chunk:
        pad_len = chunk - len(buf)
        buf = buf + [eot] * pad_len
        mask_buf = mask_buf + [True] * pad_len

    seq = buf[-chunk:]
    seq_mask = mask_buf[-chunk:]
    inp = torch.tensor(seq[:-1], dtype=torch.long)
    tgt = torch.tensor(seq[1:], dtype=torch.long)
    is_prompt = torch.tensor(seq_mask[1:], dtype=torch.bool)
    tgt[is_prompt] = -100
    tgt[tgt == eot] = -100
    if not (tgt != -100).any():
        return None
    return inp, tgt


class CorpusDataset(IterableDataset):
    """``IterableDataset`` over a plain-text or Markdown file.

    The file is read once, tokenized into a flat sequence, and then sliced
    into ``(input_ids, target_ids)`` pairs of length *seq_len* using the same
    rolling-window approach as ``ChanDataset``.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        tokenizer: TiktokenWrapper,
        seq_len: int = 512,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        text = self.corpus_path.read_text(encoding="utf-8", errors="replace")
        tokens = self.tokenizer.encode(text)
        chunk = self.seq_len + 1
        for start in range(0, len(tokens) - chunk + 1, self.seq_len):
            seq = tokens[start : start + chunk]
            inp = torch.tensor(seq[:-1], dtype=torch.long)
            tgt = torch.tensor(seq[1:], dtype=torch.long)
            yield inp, tgt


def corpus_dataloader(
    corpus_path: str | Path,
    tokenizer: TiktokenWrapper,
    seq_len: int = 512,
    batch_size: int = 4,
) -> DataLoader:
    """Return a ``DataLoader`` over a plain-text corpus file."""
    dataset = CorpusDataset(corpus_path, tokenizer, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # single-file read; no benefit from multiple workers
        pin_memory=_pin_memory_enabled(),
    )


class PromptDataset(IterableDataset):
    """Iterable SFT dataset over structured prompt-response examples.

    Each example is encoded as a prompt prefix made of ``[SYSTEM]``, ``[USER]``,
    and ``[ASSISTANT]`` tags followed by the assistant response and a trailing
    EOT. Prompt tokens are masked to ``-100`` so loss is computed only on the
    response. Example boundaries are respected.
    """

    def __init__(
        self,
        prompt_path: str | Path,
        tokenizer: TiktokenWrapper,
        seq_len: int = 512,
    ) -> None:
        self.prompt_path = Path(prompt_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        chunk = self.seq_len + 1
        eot = self.tokenizer.eot_token

        for system, user, assistant in _parse_prompt_pairs(self.prompt_path):
            prompt_text = (
                "[SYSTEM]\n" f"{system}\n\n" "[USER]\n" f"{user}\n\n" "[ASSISTANT]\n"
            )
            prompt_toks = self.tokenizer.encode(prompt_text)
            response_toks = self.tokenizer.encode(assistant)
            if not response_toks:
                continue

            buf = prompt_toks + response_toks + [eot]
            mask_buf = [True] * len(prompt_toks) + [False] * len(response_toks) + [True]

            while len(buf) >= chunk:
                seq = buf[:chunk]
                seq_mask = mask_buf[:chunk]
                buf = buf[chunk:]
                mask_buf = mask_buf[chunk:]
                inp = torch.tensor(seq[:-1], dtype=torch.long)
                tgt = torch.tensor(seq[1:], dtype=torch.long)
                is_prompt = torch.tensor(seq_mask[1:], dtype=torch.bool)
                tgt[is_prompt] = -100
                tgt[tgt == eot] = -100
                if not (tgt != -100).any():
                    continue
                yield inp, tgt


def prompt_dataloader(
    prompt_path: str | Path,
    tokenizer: TiktokenWrapper,
    seq_len: int = 512,
    batch_size: int = 4,
) -> DataLoader:
    """Return a ``DataLoader`` over structured prompt-response examples."""
    dataset = PromptDataset(prompt_path, tokenizer, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=_pin_memory_enabled(),
    )


class DPODataset(IterableDataset):
    """Iterable preference dataset over chosen/rejected response pairs."""

    def __init__(
        self,
        prompt_path: str | Path,
        tokenizer: TiktokenWrapper,
        seq_len: int = 512,
    ) -> None:
        self.prompt_path = Path(prompt_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        eot = self.tokenizer.eot_token

        for system, user, chosen, rejected in _parse_dpo_pairs(self.prompt_path):
            prompt_text = (
                "[SYSTEM]\n" f"{system}\n\n" "[USER]\n" f"{user}\n\n" "[ASSISTANT]\n"
            )
            prompt_toks = self.tokenizer.encode(prompt_text)
            chosen_toks = self.tokenizer.encode(chosen)
            rejected_toks = self.tokenizer.encode(rejected)
            if not chosen_toks or not rejected_toks:
                continue

            chosen_pair = _build_preference_pair(
                prompt_toks,
                chosen_toks,
                eot=eot,
                seq_len=self.seq_len,
            )
            rejected_pair = _build_preference_pair(
                prompt_toks,
                rejected_toks,
                eot=eot,
                seq_len=self.seq_len,
            )
            if chosen_pair is None or rejected_pair is None:
                continue

            chosen_inp, chosen_tgt = chosen_pair
            rejected_inp, rejected_tgt = rejected_pair
            yield chosen_inp, chosen_tgt, rejected_inp, rejected_tgt


def dpo_dataloader(
    prompt_path: str | Path,
    tokenizer: TiktokenWrapper,
    seq_len: int = 512,
    batch_size: int = 1,
) -> DataLoader:
    """Return a ``DataLoader`` over chosen/rejected preference examples."""
    dataset = DPODataset(prompt_path, tokenizer, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=_pin_memory_enabled(),
    )
