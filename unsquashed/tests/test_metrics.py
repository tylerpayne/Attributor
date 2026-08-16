import pytest

from unsquash.eval.metrics import (
    MeanMetrics,
    mrr,
    precision_at_k,
    r_precision,
    recall_at_k,
)

RANKED = [3, 0, 7, 1, 4]
RELEVANT = {0, 1}


def test_precision_at_k():
    assert precision_at_k(RANKED, RELEVANT, 1) == 0.0
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.5
    assert precision_at_k(RANKED, RELEVANT, 4) == 0.5
    assert precision_at_k(RANKED, RELEVANT, 5) == 2 / 5


def test_recall_at_k():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 2) == 0.5
    assert recall_at_k(RANKED, RELEVANT, 4) == 1.0


def test_r_precision():
    # |relevant| = 2, hits in top-2 = 1
    assert r_precision(RANKED, RELEVANT) == 0.5


def test_mrr():
    assert mrr(RANKED, RELEVANT) == 0.5  # first relevant at rank 2
    assert mrr(RANKED, {3}) == 1.0
    assert mrr(RANKED, {99}) == 0.0


def test_precision_is_not_recall():
    # The legacy metric divided all-attributed hits by |supporting|,
    # silently turning precision@None into recall. Guard the distinction.
    ranked = [0, 1, 2, 3]
    relevant = {0, 1, 2, 3}
    assert precision_at_k(ranked, relevant, 2) == 1.0
    assert recall_at_k(ranked, relevant, 2) == 0.5


def test_mean_metrics_streaming():
    mm = MeanMetrics(ks=(1, 2))
    mm.update([0, 1], {0})   # p@1=1, r@1=1
    mm.update([1, 0], {0})   # p@1=0, r@1=0
    means = mm.means()
    assert means["precision@1"] == 0.5
    assert means["recall@1"] == 0.5
    assert mm.n == 2


def test_validation():
    with pytest.raises(ValueError):
        precision_at_k(RANKED, RELEVANT, 0)
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 1)
