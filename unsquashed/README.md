# Unsquashed

De-biased attention attribution for decoder-only Transformers: the **unsquash**
correction, a HotpotQA evaluation harness for A/B-testing attribution methods,
and a continued-pretraining recipe that retrofits the correction into a model
as an attention prior.

## The idea

Attention rollout composes per-layer attention matrices to attribute output
tokens to input tokens. But composition itself fabricates a bias: uniform
causal attention acts as a discrete integrator (its unnormalized pattern is
the lower-triangular ones matrix `L`), so a stack of `k` layers integrates `k`
times and piles attribution mass onto early tokens like `m^(k-1)` (lag
`m = i - j`). In a null model with 8 perfectly uniform layers over 64 tokens,
**87% of rollout attribution lands on token 0** — pure artifact.

The fix: multiply each layer's attention by

```
c_m = Γ(m + 1/k) / (Γ(m + 1) · Γ(1/k))
```

— the Taylor coefficients of `(1 - x)^(-1/k)`. The Toeplitz matrix of these
coefficients is the **k-th matrix root of `L`** (exactly, at any finite size),
so each corrected layer becomes a *fractional integrator of order 1/k* and the
whole stack integrates exactly once: the null model's attribution comes out
uniform. Structural bias goes to zero by construction; attention that survives
the correction is attention the model spent logits to earn.

The same correction has an in-model form: adding `λ · ln(c_m)` to attention
logits (a logarithmic cousin of ALiBi's linear distance penalty). A model
continued-pretrained with that prior has a uniform rollout null natively, so
plain rollout attribution is de-biased with no post-hoc step.

## Install

```sh
pip install -e .
```

## 1. Post-hoc evaluation (frozen model)

A/B-test attribution methods on HotpotQA supporting-facts retrieval. All
methods are computed from the *same* captured forward pass per case:

- `attention_sum` — raw attention summed over layers/heads (baseline)
- `rollout` — classic attention rollout (Abnar & Zuidema, 2020)
- `unsquashed` — rollout with the per-layer unsquash correction

```sh
python -m unsquash.eval --model HuggingFaceTB/SmolLM-135M-Instruct \
    --dtype float32 --device_map cuda --max_cases 500
```

Evaluation is teacher-forced (the gold answer is the assistant turn), sentences
are located by exact character offsets (fast-tokenizer offset mappings, no
brittle token matching), heads are mixed by the Frobenius norm of each head's
**column block** of `W_O`, and document scores are per-token means
(length-unbiased; `--reduction sum` reproduces legacy total-mass scoring).
Results stream to `results.jsonl` + `summary.json` and runs resume
automatically. Reported metrics: precision@k, recall@k, R-precision, MRR.

## 2. Continued-pretraining retrofit

Ramp the prior in over `--prior_warmup_steps` (λ: 0 → 1), then hold it while
the model adapts. Held-out loss and attention-sink mass (attention on
position 0 — the early signal of adaptation vs. degradation) are logged
throughout.

```sh
python -m unsquash.train --model HuggingFaceTB/SmolLM-135M-Instruct \
    --steps 5000 --prior_warmup_steps 1000 --seq_len 1024 \
    --batch_size 8 --grad_accum 4 --autocast_bf16 \
    --out_dir runs/smollm-unsquashed
```

Data defaults to streaming `HuggingFaceFW/fineweb-edu` (`sample-10BT`);
`--text_file corpus.txt` trains from a local file instead. Checkpoints record
their prior in `unsquash_prior.json`, and the evaluator applies it
automatically:

```sh
python -m unsquash.eval --model runs/smollm-unsquashed/final
```

The A/B that matters: `rollout` on the retrofitted checkpoint (prior applied at
capture) vs. `unsquashed` on the frozen base model vs. plain `rollout` on the
frozen base model.

## Running on Modal GPUs

[`modal_app.py`](modal_app.py) wraps both experiments for [Modal](https://modal.com):

```sh
pip install modal
modal setup                      # one-time auth

# Experiment 1: frozen-model A/B eval (L4 by default)
modal run modal_app.py::evaluate --max-cases 500

# Experiment 2: retrofit (A10G by default; detach for long runs)
modal run --detach modal_app.py::retrofit --steps 5000

# Evaluate the retrofitted checkpoint (its recorded prior is auto-applied)
modal run modal_app.py::evaluate --model /results/train/HuggingFaceTB__SmolLM-135M-Instruct__unsquashed/final

# Everything: base eval + retrofit in parallel, then checkpoint eval
modal run --detach modal_app.py::pipeline
```

Artifacts persist in the `unsquash-results` volume
(`modal volume get unsquash-results eval/<name>/summary.json .`); the HF cache
lives in `unsquash-hf-cache` so models/datasets download once. Eval runs
resume automatically under the same `--out-name`. Override GPU types with
`UNSQUASH_EVAL_GPU` / `UNSQUASH_TRAIN_GPU` (e.g.
`UNSQUASH_TRAIN_GPU=A100 modal run --detach modal_app.py::retrofit`).

## Notes and limitations

- The correction is derived for the attention-only composition; with the
  residual term in the rollout the null changes and the same coefficients
  overcorrect. Method defaults reflect this (`rollout`: residual on,
  `unsquashed`: residual off); `--residual` overrides.
- The prior mask is square and unpadded (teacher-forced eval, packed training
  blocks). It is not wired for incremental decoding with a KV cache.
- Attribution capture requires `attn_implementation="eager"`
  (`output_attentions=True`), which the entry points set for you.
- `k` defaults to the model's layer count everywhere; override with
  `--unsquashed_k`. ("Prior" is the generic term — you choose the unsquashed
  prior or the ALiBi prior; unsquash-specific knobs say "unsquashed".)

## Tests

Fully offline (a tiny random Llama and a from-scratch tokenizer are built in
the fixtures):

```sh
pytest tests/
```
