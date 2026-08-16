"""Synthetic long-context tasks, built at the token level.

Every builder returns a :class:`LadderCase`: a full token sequence plus the
index where the gold continuation starts. Scoring is teacher-forced — the
model's log-probabilities of the gold tokens given everything before them —
so no instruction-following or sampling is required (these are base models),
and no KV-cache decoding is needed (the bias implementations are
full-sequence only).

Tiers:

- ``passkey``: a 5-digit key planted at a controlled depth inside filler
  text, retrieved by an induction-style cue ("The pass key is"). The
  classic length-extrapolation retrieval probe.
- ``kv``: key-value retrieval (Lost-in-the-Middle style) — N random
  key/value records, query one planted at a controlled depth. Retrieval
  under distractor interference: every record looks like the needle.
- ``copy``: a marked word list at the *start* of the context, re-cued after
  the filler; gold continues the list. Early-token copying at distance is
  the capability over-squashing predicts causal transformers lose (Barbero
  et al. 2024, "Transformers need glasses!"), and the capability the
  unsquash prior — built to cancel exactly that pileup — should preserve.
  ``depth`` is fixed at 0.0 by construction.

Length accounting is in tokens of the provided tokenizer; builders hit the
requested total length to within one filler unit. All randomness comes from
an explicit seed, so cases are reproducible across runs and models — every
model scores the *same* sequences.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Deterministic filler in the passkey-test tradition: semantically inert,
# low-entropy prose that any LM can predict but that carries no signal about
# the needle.
FILLER_SENTENCES = (
    "The grass is green. ",
    "The sky is blue. ",
    "The sun is yellow. ",
    "Here we go. ",
    "There and back again. ",
)

# Word pool for the copy tier: common, single-token-ish words so list items
# survive tokenization intact enough to score.
COPY_WORDS = (
    "time year people way day man thing woman life child world school "
    "state family student group country problem hand part place case week "
    "company system program question work government number night point "
    "home water room mother area money story fact month lot right study "
    "book eye job word business issue side kind head house service friend "
    "father power hour game line end member law car city community name"
).split()


@dataclass
class LadderCase:
    tier: str
    length: int  # requested context length (tokens)
    depth: float  # needle position as a fraction of the context
    idx: int  # case index within the (tier, length, depth) cell
    input_ids: list[int] = field(repr=False)  # full sequence incl. gold
    gold_start: int  # input_ids[gold_start:] is the gold continuation
    meta: dict = field(default_factory=dict)

    @property
    def gold_ids(self) -> list[int]:
        return self.input_ids[self.gold_start :]

    @property
    def key(self) -> str:
        return f"{self.tier}:{self.length}:{self.depth}:{self.idx}"


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _filler_stream(tokenizer, rng: random.Random, n_tokens: int) -> list[int]:
    """~n_tokens of deterministic filler (may overshoot by one sentence)."""
    units = [_encode(tokenizer, s) for s in FILLER_SENTENCES]
    out: list[int] = []
    while len(out) < n_tokens:
        out.extend(units[rng.randrange(len(units))])
    return out[:n_tokens] if n_tokens >= 0 else []


def _assemble(
    tokenizer,
    rng: random.Random,
    *,
    prefix_ids: list[int],
    needle_ids: list[int],
    query_ids: list[int],
    gold_ids: list[int],
    length: int,
    depth: float,
) -> tuple[list[int], int]:
    """prefix + filler + needle + filler + query + gold, with the needle's
    start at ``depth`` (fraction) of the total and the total context (before
    gold) equal to ``length`` tokens."""
    fixed = len(prefix_ids) + len(needle_ids) + len(query_ids)
    budget = max(0, length - fixed)
    before = int(round(budget * depth))
    after = budget - before
    ids = (
        prefix_ids
        + _filler_stream(tokenizer, rng, before)
        + needle_ids
        + _filler_stream(tokenizer, rng, after)
        + query_ids
    )
    gold_start = len(ids)
    return ids + gold_ids, gold_start


def passkey_case(
    tokenizer, *, length: int, depth: float, idx: int, seed: int = 0
) -> LadderCase:
    # String seeds: random.Random(str) seeds via sha512, so cases are
    # bit-identical across processes (tuple hashes are salted per process).
    rng = random.Random(f"{seed}:passkey:{length}:{depth}:{idx}")
    key = rng.randint(10000, 99999)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prefix = bos + _encode(
        tokenizer,
        "There is a pass key hidden inside a lot of irrelevant text. "
        "Find it and memorize it.\n",
    )
    needle = _encode(
        tokenizer, f" The pass key is {key}. Remember it. {key} is the pass key. "
    )
    query = _encode(tokenizer, "\nThe pass key is")
    gold = _encode(tokenizer, f" {key}")
    ids, gold_start = _assemble(
        tokenizer, rng,
        prefix_ids=prefix, needle_ids=needle, query_ids=query, gold_ids=gold,
        length=length, depth=depth,
    )
    return LadderCase(
        tier="passkey", length=length, depth=depth, idx=idx,
        input_ids=ids, gold_start=gold_start, meta={"passkey": key},
    )


def kv_case(
    tokenizer,
    *,
    length: int,
    depth: float,
    idx: int,
    seed: int = 0,
    pair_tokens: int = 16,
) -> LadderCase:
    """Fill the context with ``key-<hex>: <hex>`` records; the queried record
    sits at ``depth``. Unlike passkey filler, every line is needle-shaped."""
    rng = random.Random(f"{seed}:kv:{length}:{depth}:{idx}")

    def record() -> tuple[str, str]:
        return (
            "".join(rng.choice("0123456789abcdef") for _ in range(8)),
            "".join(rng.choice("0123456789abcdef") for _ in range(8)),
        )

    target_key, target_val = record()
    needle = _encode(tokenizer, f"key-{target_key}: {target_val}\n")
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prefix = bos + _encode(tokenizer, "A list of key-value records.\n")
    query = _encode(tokenizer, f"key-{target_key}:")
    gold = _encode(tokenizer, f" {target_val}")

    fixed = len(prefix) + len(needle) + len(query)
    budget = max(0, length - fixed)
    n_before = int(round(budget * depth))
    ids = list(prefix)
    while len(ids) < len(prefix) + n_before:
        k, v = record()
        ids.extend(_encode(tokenizer, f"key-{k}: {v}\n"))
    ids.extend(needle)
    while len(ids) < length - len(query):
        k, v = record()
        ids.extend(_encode(tokenizer, f"key-{k}: {v}\n"))
    ids.extend(query)
    gold_start = len(ids)
    return LadderCase(
        tier="kv", length=length, depth=depth, idx=idx,
        input_ids=ids + gold, gold_start=gold_start,
        meta={"target_key": target_key, "target_value": target_val,
              "pair_tokens": pair_tokens},
    )


def copy_case(
    tokenizer,
    *,
    length: int,
    idx: int,
    seed: int = 0,
    list_words: int = 24,
    cue_words: int = 8,
    gold_words: int = 6,
) -> LadderCase:
    """A word list at the very start; after the filler, the list restarts and
    the gold continues it — copying *early* tokens across the full context."""
    rng = random.Random(f"{seed}:copy:{length}:{idx}")
    words = rng.sample(COPY_WORDS, list_words)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    listing = " ".join(words)
    prefix = bos + _encode(tokenizer, f"Word list: {listing}.\n")
    query = _encode(tokenizer, "\nWord list: " + " ".join(words[:cue_words]))
    gold = _encode(
        tokenizer, " " + " ".join(words[cue_words : cue_words + gold_words])
    )
    fixed = len(prefix) + len(query)
    filler = _filler_stream(tokenizer, rng, max(0, length - fixed))
    ids = prefix + filler + query
    gold_start = len(ids)
    return LadderCase(
        tier="copy", length=length, depth=0.0, idx=idx,
        input_ids=ids + gold, gold_start=gold_start,
        meta={"words": words, "cue_words": cue_words},
    )


BUILDERS = {
    "passkey": passkey_case,
    "kv": kv_case,
    "copy": copy_case,
}
