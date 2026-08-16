"""End-to-end continued-pretraining smoke test on the offline tiny model."""

import json
import math
import os

import torch

from unsquash.prior import PriorConfig
from unsquash.rollout import RolloutAttributor
from unsquash.train.runner import TrainSettings, train
from tests.conftest import CORPUS


def test_train_and_reload(tiny_model, tiny_tokenizer, tmp_path):
    base = tmp_path / "base"
    tiny_model.save_pretrained(base)
    tiny_tokenizer.save_pretrained(base)

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("\n\n".join(CORPUS * 30))

    out_dir = tmp_path / "run"
    settings = TrainSettings(
        model=str(base),
        out_dir=str(out_dir),
        text_file=str(text_file),
        seq_len=64,
        batch_size=2,
        grad_accum=1,
        steps=4,
        lr=1e-4,
        lr_warmup_steps=1,
        prior_warmup_steps=2,
        log_every=1,
        eval_every=2,
        eval_batches=1,
        save_every=100,
        device="cpu",
    )
    final = train(settings)

    # Checkpoint contents: model, tokenizer, and the recorded prior.
    assert os.path.exists(os.path.join(final, "model.safetensors"))
    prior = PriorConfig.load(final)
    assert prior is not None
    assert prior.k == tiny_model.config.num_hidden_layers
    assert prior.lam == 1.0  # past warmup by the end

    # Training log: finite losses, lambda annealed 0 -> 1, sink monitored.
    with open(os.path.join(str(out_dir), "train_log.jsonl")) as fd:
        entries = [json.loads(line) for line in fd]
    losses = [e["loss"] for e in entries if "loss" in e]
    assert losses and all(math.isfinite(x) for x in losses)
    lambdas = [e["lambda"] for e in entries if "loss" in e]
    assert lambdas[0] < lambdas[-1] == 1.0
    evals = [e for e in entries if "eval_loss" in e]
    assert evals and all(0.0 <= e["sink_mass"] <= 1.0 for e in evals)

    # The evaluator's auto-prior path: reload the checkpoint and attribute.
    from unsquash.eval.runner import build_attributor

    attributor = build_attributor(model_name_or_path=final, device_map=None)
    assert attributor.prior == prior

    ids = tiny_tokenizer("Paris is the capital of France.",
                         return_tensors="pt")["input_ids"][0]
    result = attributor(ids, methods=("rollout", "unsquashed"))
    n = ids.shape[0]
    for Y in result.values():
        assert Y.shape == (n, n)
        assert torch.isfinite(Y).all()

    # The prior must actually change the captured attentions.
    bare = RolloutAttributor(attributor.model, attributor.tokenizer, prior=None)
    with_prior = attributor.capture(ids)
    without = bare.capture(ids)
    assert not torch.allclose(with_prior[0], without[0])


def test_packed_batches_identical_across_batching_and_prefetch(tiny_tokenizer, tmp_path):
    """encode_docs/prefetch are throughput knobs only: the token stream must
    be bit-identical to the naive one-doc-at-a-time path."""
    import itertools

    from unsquash.train.data import packed_batches

    path = tmp_path / "docs.txt"
    path.write_text("\n\n".join(f"document number {i} with some text" for i in range(40)))

    def blocks(**kw):
        gen = packed_batches(
            tiny_tokenizer, seq_len=8, batch_size=2, text_file=str(path), **kw
        )
        return list(itertools.islice(gen, 12))

    naive = blocks(encode_docs=1, prefetch=0)
    fast = blocks(encode_docs=64, prefetch=4)
    for a, b in zip(naive, fast):
        assert (a == b).all()
