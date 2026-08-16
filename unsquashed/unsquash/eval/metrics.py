"""Retrieval metrics for ranked attribution lists.

Definitions (``ranked`` is the candidate ids sorted by attribution score,
best first; ``relevant`` is the set of gold supporting ids):

- precision@k  = |top-k ∩ relevant| / k
- recall@k     = |top-k ∩ relevant| / |relevant|
- r_precision  = precision@|relevant|  (== recall@|relevant|)
- mrr          = 1 / rank of the first relevant item
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    hits = len(set(ranked[:k]) & relevant)
    return hits / k


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        raise ValueError("relevant set is empty")
    hits = len(set(ranked[:k]) & relevant)
    return hits / len(relevant)


def r_precision(ranked: Sequence[int], relevant: set[int]) -> float:
    return recall_at_k(ranked, relevant, len(relevant))


def mrr(ranked: Sequence[int], relevant: set[int]) -> float:
    for i, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


class MeanMetrics:
    """Streaming means of the metrics above, per k."""

    def __init__(self, ks: Iterable[int] = (1, 2, 5, 10)):
        self.ks = tuple(ks)
        self.n = 0
        self._sums: dict[str, float] = defaultdict(float)

    def update(self, ranked: Sequence[int], relevant: set[int]) -> dict[str, float]:
        values: dict[str, float] = {}
        for k in self.ks:
            values[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
            values[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        values["r_precision"] = r_precision(ranked, relevant)
        values["mrr"] = mrr(ranked, relevant)
        self.n += 1
        for name, v in values.items():
            self._sums[name] += v
        return values

    def means(self) -> dict[str, float]:
        if self.n == 0:
            return {}
        return {name: s / self.n for name, s in sorted(self._sums.items())}
