"""HotpotQA cases for attribution evaluation.

Candidates are the individual context *sentences*; the gold labels are
HotpotQA's supporting-facts annotations. Sentence positions are recorded as
character spans while the context string is built, and supporting facts are
resolved directly from their (title, sentence_id) pairs — no text matching, so
duplicate or oddly-whitespaced sentences can't crash or mislabel a case.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)

INSTRUCTION = (
    "Do not explain your reasoning. Simply provide the answer or say 'n/a' "
    "if the question cannot be answered."
)


@dataclass
class EvalCase:
    question: str
    answer: str
    context: str
    sentences: list[str]
    sentence_char_spans: list[tuple[int, int]]  # spans within ``context``
    supporting: set[int] = field(default_factory=set)  # indices into ``sentences``


def format_row(row: dict) -> EvalCase | None:
    """Build one EvalCase from a raw HotpotQA row (None if unusable)."""
    titles = row["context"]["title"]
    all_sentences = row["context"]["sentences"]

    parts: list[str] = ['Documents:\n"""\n']
    cursor = len(parts[0])
    sentences: list[str] = []
    spans: list[tuple[int, int]] = []
    # (title, sent_id) -> global sentence index, resolved structurally.
    index_of: dict[tuple[str, int], int] = {}

    for title, doc_sentences in zip(titles, all_sentences):
        header = f"{title}:\n"
        parts.append(header)
        cursor += len(header)
        for sent_id, sentence in enumerate(doc_sentences):
            text = sentence.strip()
            if not text:
                continue
            if sentences:
                parts.append(" ")
                cursor += 1
            index_of[(title, sent_id)] = len(sentences)
            sentences.append(text)
            spans.append((cursor, cursor + len(text)))
            parts.append(text)
            cursor += len(text)
        parts.append("\n\n")
        cursor += 2

    parts.append('"""\n\n')
    question = row["question"].strip()
    parts.append(question)
    parts.append("\n" + INSTRUCTION)
    context = "".join(parts)

    supporting: set[int] = set()
    for i, title in enumerate(row["supporting_facts"]["title"]):
        sent_id = row["supporting_facts"]["sent_id"][i]
        idx = index_of.get((title, sent_id))
        if idx is None:
            # A handful of HotpotQA rows have out-of-range annotations.
            logger.warning(
                "Supporting fact (%r, %d) not found in context; skipping fact.",
                title,
                sent_id,
            )
            continue
        supporting.add(idx)

    if not supporting:
        logger.warning("Case %r has no resolvable supporting facts; skipping case.",
                       question[:60])
        return None

    answer = row["answer"].strip()
    if not answer:
        return None

    return EvalCase(
        question=question,
        answer=answer,
        context=context,
        sentences=sentences,
        sentence_char_spans=spans,
        supporting=supporting,
    )


def load_cases(
    *,
    split: str = "validation",
    config: str = "fullwiki",
    trust_remote_code: bool = False,
) -> Iterator[EvalCase]:
    """Stream formatted HotpotQA cases (rows that fail formatting are skipped)."""
    from datasets import load_dataset

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        config,
        split=split,
        trust_remote_code=trust_remote_code,
    )
    for row in ds:
        case = format_row(row)
        if case is not None:
            yield case
