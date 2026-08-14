"""Mapping document sentences and the answer to token spans.

The original Attributor located sentences by tokenizing each one independently
and searching for the token subsequence in the prompt. That is brittle:
tokenizers add special tokens, and a sentence tokenizes differently mid-text
(leading-space merges) than in isolation, so candidates silently drop out of
the ranking.

Here spans are tracked in *character* space — sentence char offsets are
recorded when the context string is built, the chat-formatted text embeds the
context verbatim, and a fast tokenizer's offset mapping converts char spans to
token spans exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenSpan:
    """Half-open token index range [start, end)."""

    start: int
    end: int

    def __len__(self):
        return max(0, self.end - self.start)


@dataclass
class PreparedCase:
    """A teacher-forced sequence with everything attribution needs."""

    input_ids: torch.Tensor  # [n]
    answer_span: TokenSpan  # tokens of the gold answer (the "output")
    sentence_spans: list[TokenSpan | None]  # per candidate sentence; None if unmapped


def char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> TokenSpan | None:
    """Tokens whose character ranges overlap [char_start, char_end)."""
    start = end = None
    for t, (a, b) in enumerate(offsets):
        if a == b:  # zero-width (special) tokens never match
            continue
        if a < char_end and b > char_start:
            if start is None:
                start = t
            end = t + 1
    if start is None:
        return None
    return TokenSpan(start, end)


def prepare_case(
    tokenizer,
    *,
    context: str,
    answer: str,
    sentence_char_spans: list[tuple[int, int]],
) -> PreparedCase:
    """Build the teacher-forced token sequence for (context -> gold answer) and
    map the answer and every candidate sentence to token spans.

    Requires a fast tokenizer (offset mappings).
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "unsquash.spans requires a fast tokenizer (offset mappings); "
            f"{type(tokenizer).__name__} is not fast."
        )

    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": context},
            {"role": "assistant", "content": answer},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Base-LM checkpoints (e.g. from-scratch pretraining) have no chat
        # template — and chat markup would be out-of-distribution for them
        # anyway. Plain concatenation keeps the context verbatim at offset 0.
        full_text = f"{context}\nAnswer: {answer}"

    # The chat template embeds the user content verbatim; locate it once and
    # shift the sentence spans recorded at context-construction time.
    context_offset = full_text.find(context)
    if context_offset < 0:
        raise ValueError("Chat template did not embed the context verbatim")

    # The assistant turn comes after the user turn; take the last occurrence
    # so answers that also appear inside the context resolve to the answer turn.
    answer_offset = full_text.rfind(answer)
    if answer_offset < context_offset + len(context):
        raise ValueError("Could not locate the answer text in the formatted chat")

    enc = tokenizer(
        full_text,
        add_special_tokens=False,  # the template already includes them
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]

    answer_span = char_span_to_token_span(
        offsets, answer_offset, answer_offset + len(answer)
    )
    if answer_span is None:
        raise ValueError("Answer text mapped to no tokens")

    sentence_spans: list[TokenSpan | None] = []
    for a, b in sentence_char_spans:
        sentence_spans.append(
            char_span_to_token_span(offsets, context_offset + a, context_offset + b)
        )

    return PreparedCase(
        input_ids=torch.tensor(enc["input_ids"], dtype=torch.long),
        answer_span=answer_span,
        sentence_spans=sentence_spans,
    )


def score_spans(
    attributions: torch.Tensor,
    answer_span: TokenSpan,
    spans: list[TokenSpan | None],
    *,
    reduction: str = "mean",
) -> list[float]:
    """Score each candidate span by the attribution its tokens received while
    generating the answer tokens.

    ``reduction="mean"`` averages over the span's tokens (length-unbiased,
    the default); ``"sum"`` reproduces the legacy total-mass scoring, which
    favors long documents.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Unknown reduction {reduction!r}")
    rows = attributions[answer_span.start : answer_span.end].mean(dim=0)
    scores = []
    for span in spans:
        if span is None or len(span) == 0:
            scores.append(float("-inf"))
            continue
        mass = rows[span.start : span.end]
        scores.append(float(mass.mean() if reduction == "mean" else mass.sum()))
    return scores
