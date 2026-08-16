"""End-to-end evaluation smoke test on the offline tiny model."""

import json
import os

from unsquash.eval.hotpot import format_row
from unsquash.eval.runner import EvalSettings, evaluate, print_summary
from unsquash.rollout import RolloutAttributor
from tests.test_spans import FAKE_ROW

FAKE_ROW_2 = {
    "question": "Where is the powerhouse of the cell?",
    "answer": "the mitochondria",
    "context": {
        "title": ["Biology", "Distractor"],
        "sentences": [
            ["The mitochondria is the powerhouse of the cell.",
             "Cells contain organelles."],
            ["The quick brown fox jumps over the lazy dog.",
             "Napoleon Bonaparte was a French military officer."],
        ],
    },
    "supporting_facts": {"title": ["Biology"], "sent_id": [0]},
}


def test_evaluate_end_to_end(tiny_model, tiny_tokenizer, tmp_path):
    attributor = RolloutAttributor(tiny_model, tiny_tokenizer)
    cases = [format_row(FAKE_ROW), format_row(FAKE_ROW_2)]
    assert all(c is not None for c in cases)

    settings = EvalSettings(
        methods=("attention_sum", "rollout", "unsquashed"),
        ks=(1, 2),
        max_context_tokens=2000,
        out_dir=str(tmp_path / "results"),
    )
    summary = evaluate(attributor, cases, settings)
    print_summary(summary)

    assert summary["evaluated_cases"] == 2
    for method in settings.methods:
        means = summary["metrics"][method]
        assert set(means) == {"precision@1", "precision@2", "recall@1",
                              "recall@2", "r_precision", "mrr"}
        for value in means.values():
            assert 0.0 <= value <= 1.0

    # One JSONL record per case, each with rankings for every method.
    with open(os.path.join(settings.out_dir, "results.jsonl")) as fd:
        records = [json.loads(line) for line in fd]
    assert len(records) == 2
    assert all(set(r["methods"]) == set(settings.methods) for r in records)
    assert os.path.exists(os.path.join(settings.out_dir, "summary.json"))

    # Resume: a second run must skip both completed cases.
    summary2 = evaluate(attributor, cases, settings)
    with open(os.path.join(settings.out_dir, "results.jsonl")) as fd:
        assert len(fd.readlines()) == 2
    assert summary2["evaluated_cases"] == 0
