"""Ladder harness: chunked-attention parity (the load-bearing invariant for
every long-context number), task-builder contracts, and an offline
end-to-end run over the tiny fixtures."""

import json
import math
import os

import pytest
import torch

from unsquash.ladder.runner import LadderSettings, run_ladder_with, summarize
from unsquash.ladder.tasks import copy_case, kv_case, passkey_case
from unsquash.pretrain.model import PriorLlama, ModelSpec
from tests.conftest import CORPUS

TINY = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    max_seq_len=64,
    rope_theta=10000.0,
)


def tiny_ids(n=64, seed=1):
    torch.manual_seed(seed)
    return torch.randint(0, TINY["vocab_size"], (1, n))


@pytest.mark.parametrize(
    "kind", ["prior", "alibi", "none"]
)
@pytest.mark.parametrize("q_chunk", [5, 64])
def test_chunked_hidden_matches_forward(kind, q_chunk):
    torch.manual_seed(0)
    spec = ModelSpec(
        unsquashed_k=2.0 if kind == "prior" else None,
        alibi=kind == "alibi",
        **TINY,
    )
    model = PriorLlama(spec).eval()
    ids = tiny_ids()
    with torch.no_grad():
        full_logits = model(ids)
        hidden = model.chunked_hidden(ids, q_chunk=q_chunk)
        chunked_logits = model.lm_head(hidden)
    torch.testing.assert_close(chunked_logits, full_logits, atol=1e-4, rtol=1e-4)


def test_chunked_hidden_rejects_overlong_input():
    torch.manual_seed(0)
    model = PriorLlama(ModelSpec(unsquashed_k=None, **TINY)).eval()
    with pytest.raises(ValueError, match="max_seq_len"):
        model.chunked_hidden(tiny_ids(n=TINY["max_seq_len"] + 1))


def test_passkey_case_contract(tiny_tokenizer):
    positions = []
    for depth in (0.0, 0.5, 1.0):
        case = passkey_case(tiny_tokenizer, length=256, depth=depth, idx=0, seed=0)
        assert abs(len(case.input_ids) - case.gold_start) < 16  # short gold
        assert case.gold_start <= 256 + 16
        text = tiny_tokenizer.decode(case.input_ids)
        key = str(case.meta["passkey"])
        assert text.count(key) == 3  # twice in the needle, once as gold
        gold_text = tiny_tokenizer.decode(case.gold_ids)
        assert key in gold_text
        positions.append(text.find(key))
    # Depth moves the needle monotonically deeper into the context (the
    # fixed prefix/query offsets make absolute fractions scale-dependent).
    assert positions[0] < positions[1] < positions[2]


def test_passkey_case_deterministic(tiny_tokenizer):
    a = passkey_case(tiny_tokenizer, length=128, depth=0.5, idx=3, seed=7)
    b = passkey_case(tiny_tokenizer, length=128, depth=0.5, idx=3, seed=7)
    assert a.input_ids == b.input_ids and a.meta == b.meta
    c = passkey_case(tiny_tokenizer, length=128, depth=0.5, idx=4, seed=7)
    assert c.meta["passkey"] != a.meta["passkey"] or c.input_ids != a.input_ids


def test_kv_case_contract(tiny_tokenizer):
    case = kv_case(tiny_tokenizer, length=256, depth=0.5, idx=0, seed=0)
    text = tiny_tokenizer.decode(case.input_ids)
    assert f"key-{case.meta['target_key']}" in text
    assert tiny_tokenizer.decode(case.gold_ids).strip() == case.meta["target_value"]
    # Distractor records exist on both sides of the needle.
    assert text.count("key-") > 4


def test_copy_case_contract(tiny_tokenizer):
    case = copy_case(tiny_tokenizer, length=192, idx=0, seed=0)
    words = case.meta["words"]
    cue = case.meta["cue_words"]
    gold_text = tiny_tokenizer.decode(case.gold_ids)
    # Gold continues the list right after the cue prefix.
    assert gold_text.split() == words[cue : cue + 6]
    assert case.depth == 0.0


def test_ladder_end_to_end(tiny_tokenizer, tmp_path):
    torch.manual_seed(0)
    spec = ModelSpec(
        vocab_size=len(tiny_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=512,
        rope_theta=10000.0,
        unsquashed_k=2.0,
    )
    model = PriorLlama(spec).eval()

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("\n\n".join(CORPUS * 40))

    settings = LadderSettings(
        model="tiny",
        out_dir=str(tmp_path / "ladder"),
        tiers=("ppl", "passkey", "kv", "copy"),
        lengths=(64, 128),
        depths=(0.0, 1.0),
        extreme_lengths=(256,),
        cases_per_cell=1,
        extreme_cases_per_cell=1,
        ppl_blocks=1,
        text_file=str(text_file),
        device="cpu",
    )
    summary = run_ladder_with(model, tiny_tokenizer, settings)

    results_path = os.path.join(settings.out_dir, "results.jsonl")
    with open(results_path) as fd:
        rows = [json.loads(line) for line in fd]

    # ppl at each length + (passkey + kv) x 2 lengths x 2 depths + copy x 2
    # lengths + extreme passkey at 3 depths.
    assert len(rows) == 2 + 8 + 2 + 3
    ppl_rows = [r for r in rows if r["tier"] == "ppl"]
    assert all(math.isfinite(r["loss"]) for r in ppl_rows)
    task_rows = [r for r in rows if r["tier"] != "ppl"]
    assert all(math.isfinite(r["gold_logprob"]) for r in task_rows)
    assert all(isinstance(r["exact"], bool) for r in task_rows)
    extreme = [r for r in rows if r["tier"] == "passkey" and r["length"] == 256]
    assert len(extreme) == 3

    assert summary["rows"] == len(rows)
    assert "passkey" in summary["effective_context_length"]
    assert os.path.exists(os.path.join(settings.out_dir, "summary.json"))

    # Resumability: a second run adds nothing.
    summary2 = run_ladder_with(model, tiny_tokenizer, settings)
    with open(results_path) as fd:
        assert len(fd.readlines()) == len(rows)
    assert summary2["rows"] == len(rows)


def test_summarize_effective_context(tmp_path):
    results = tmp_path / "results.jsonl"
    rows = [
        {"tier": "passkey", "length": 64, "depth": 0.0, "idx": 0,
         "exact": True, "gold_logprob": -0.1},
        {"tier": "passkey", "length": 128, "depth": 0.0, "idx": 0,
         "exact": False, "gold_logprob": -5.0},
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    settings = LadderSettings(model="x", out_dir=str(tmp_path))
    summary = summarize(str(results), settings)
    assert summary["effective_context_length"]["passkey"] == 64
    cell = summary["cells"]["passkey"]["64"]["0.0"]
    assert cell["exact_rate"] == 1.0
