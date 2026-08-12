"""Packed-block data streams for continued pretraining.

Documents are tokenized, joined with EOS separators, and packed into dense
``[batch_size, seq_len]`` blocks — no padding, which keeps the 4D prior mask
valid for every position.

Sources: a streaming Hugging Face dataset (default: fineweb-edu sample) or a
local UTF-8 text file (blank-line separated documents; cycled forever), which
also makes the trainer runnable offline and testable.
"""

from __future__ import annotations

import itertools
from typing import Iterator

import torch


def _texts_from_file(path: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as fd:
        docs = [d.strip() for d in fd.read().split("\n\n") if d.strip()]
    if not docs:
        raise ValueError(f"No documents found in {path}")
    yield from itertools.cycle(docs)


def _texts_from_hub(
    dataset: str,
    config: str | None,
    split: str,
    text_column: str,
    seed: int,
) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        text = row.get(text_column)
        if text:
            yield text


def packed_batches(
    tokenizer,
    *,
    seq_len: int,
    batch_size: int,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str | None = "sample-10BT",
    split: str = "train",
    text_column: str = "text",
    text_file: str | None = None,
    seed: int = 0,
) -> Iterator[torch.Tensor]:
    """Yield ``[batch_size, seq_len]`` int64 blocks forever (or until the
    source is exhausted)."""
    if text_file is not None:
        texts = _texts_from_file(text_file)
    else:
        texts = _texts_from_hub(dataset, dataset_config, split, text_column, seed)

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define an EOS token for packing")

    block = seq_len * batch_size
    buffer: list[int] = []
    for text in texts:
        buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        buffer.append(eos)
        while len(buffer) >= block:
            chunk = torch.tensor(buffer[:block], dtype=torch.long)
            buffer = buffer[block:]
            yield chunk.reshape(batch_size, seq_len)
