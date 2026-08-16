# Handoff: ALiBi control + long-context ladder

**State: implemented and tested (78/78 offline tests pass), not yet run on
GPUs.** Everything below launches from your machine with Modal set up
(`pip install modal && modal setup`), from the `unsquashed/` directory.
Nothing here has touched the existing checkpoints or results in the
`unsquash-results` volume.

## What this adds

Two pieces, designed in the session that produced this branch:

1. **An ALiBi pretraining control** — the third arm of the run matrix.
   Same SmolLM2-135M shape, same 2.7B fineweb-edu tokens, same seed and
   data order as the existing prior/control runs; the only change is the
   attention bias: per-head linear `-m_h * (i-j)` (canonical ALiBi slopes;
   for 9 heads: 1/2 … 1/256 plus 2^-1/2) instead of the prior's shared
   `λ·ln c_(i-j)`. **RoPE stays on**, matching the prior run, so the bias
   *shape* (linear vs logarithmic) is the isolated treatment.

2. **A long-context benchmark ladder** — one harness run identically over
   all three checkpoints, teacher-forced (no generation, works on base
   models), swept over length × needle-depth:

   | tier | task | default lengths |
   |---|---|---|
   | `ppl` | held-out LM loss on packed fineweb-edu | 2k, 4k, 8k, 16k, 32k |
   | `passkey` | 5-digit needle at depth, induction cue | same |
   | `kv` | key-value retrieval among distractor records | same |
   | `copy` | word list at position 0, re-cued at the end (Barbero-style early-token copying) | same |
   | extreme | passkey only, 3 depths | 64k, 128k, 256k |

   All scoring goes through a query-chunked attention path
   (`PriorLlama.chunked_hidden`) that builds bias slices on the fly — the
   `[n, n]` mask (17 GB fp32 at 64k) and full-vocab logits (~100 GB at
   512k) are never materialized, so 2k and 256k+ run through the same code.

## Commands to run (in order)

### 1. The ALiBi pretraining run (~same cost as the prior run, 1–2 H100-hr)

```bash
modal run --detach modal_app.py::pretrain_from_scratch --prior alibi
```

Checkpoint lands in
`/results/pretrain/HuggingFaceTB__SmolLM2-135M__scratch_alibi/final` with an
`unsquash_prior.json` sidecar (`kind: "alibi"`); every "auto" consumer
(eval, ladder) reads it. Watch `train_log.jsonl` for the two numbers that
discriminate hypotheses early: **eval loss** (is the ~0.04-nat tax
shape-specific?) and **sink_mass** (ALiBi penalizes position 0 hardest, yet
MPT-7B still formed sinks — if this run forms one too, sink prevention is
specifically the prior's flat composed null, not "any decaying bias").

### 2. The ladder over all three arms (one detached H100 job per model)

```bash
modal run --detach modal_app.py::ladder --models \
  /results/pretrain/HuggingFaceTB__SmolLM2-135M__scratch_unsquashed/final,\
/results/pretrain/HuggingFaceTB__SmolLM2-135M__scratch_control/final,\
/results/pretrain/HuggingFaceTB__SmolLM2-135M__scratch_alibi/final
```

(Adjust the first two paths if your existing runs used different
`--out-name`s — `modal volume ls unsquash-results pretrain/` to check.)

Results: `ladder/<slug>/results.jsonl` (one row per case, resumable — rerun
the same command to continue after a preemption) and `summary.json`
(per-cell exact rate + gold logprob, plus a RULER-style
`effective_context_length` per tier = longest length with depth-averaged
exact rate ≥ 0.5).

Optional overlays / scaling knobs:

```bash
# NTK-style RoPE rescale at eval (separates "bias failed" from "RoPE failed"):
modal run --detach modal_app.py::ladder --models <ckpt> \
  --rope-theta-scale 8 --out-suffix _ntk8

# Push the extreme tier further (mind the O(n^2) eval time):
modal run --detach modal_app.py::ladder --models <ckpt> \
  --extreme-lengths 65536,131072,262144,524288
```

Local equivalent (no Modal): `python -m unsquash.ladder --model <ckpt-dir>`.

### 3. After the ladder: the attribution leg (existing harness, unchanged)

The alibi checkpoint drops into the existing eval:

```bash
modal run modal_app.py::evaluate --model /results/pretrain/...__scratch_alibi/final
```

`RolloutAttributor` now dispatches on the sidecar kind and applies the
ALiBi bias at capture, so attention_sum/rollout/unsquashed numbers are
directly comparable with the §V table. The HotpotQA QA fine-tune used for
the both-correct protocol is whatever produced the checkpoints referenced
in the report — that harness isn't in this repo, so bring the alibi model
through the same fine-tune before comparing attribution MRR.

## Predictions on record (from the design discussion)

| measurement | prior model | ALiBi arm | control |
|---|---|---|---|
| sink mass | ~0.02 (known) | **open — the discriminating cell** | ~0.14 (known) |
| ppl at 4x length | flat (known) | flat (by construction) | collapses (known) |
| passkey/kv beyond ~2x | should hold (power-law tail keeps far tokens visible) | should fail once distance ≫ widest-head window (~256 tokens at slope 1/256) | fails |
| copy (early tokens at distance) | should hold if the prior treats over-squashing | open | should degrade with length |

If the prior retrieves where ALiBi can't while both hold ppl flat, that's
the "extrapolation without amnesia" headline. If ALiBi matches the prior on
retrieval, the log-shape story loses its main selling point — equally worth
knowing before scaling anything up.

## What changed in the code

- `unsquash/alibi.py` — slopes (paper recipe incl. non-power-of-2) and the
  `[1, H, n, n]` bias; same 4D-mask contract as `unsquashed_attention_bias`.
- `unsquash/prior.py` — `PriorConfig` gained `kind` ("unsquashed"/"alibi"),
  `num_heads`, and `attention_bias()` dispatch. Old sidecar files load
  unchanged (kind defaults to "unsquashed").
- `unsquash/rollout.py` — capture applies the recorded bias via that
  dispatch.
- `unsquash/pretrain/model.py` — `ModelSpec.alibi`; `_bias`/`_sdpa_bias`
  produce the per-head mask; `chunked_hidden` + `lm_head` (long-context
  path); `from_pretrained` (checkpoint → this implementation, bias
  auto-applied, `max_seq_len`/`rope_theta` overridable at load).
- `unsquash/pretrain/runner.py` — `use_prior: bool` became
  `prior: "unsquashed" | "alibi" | "none"`; sidecar written for both biased
  arms. (Breaking rename: old scripts passing `use_prior` need
  `prior="unsquashed"/"none"`.)
- Naming convention: "prior" is the generic concept (`PriorConfig`,
  `PriorLlama`, `prior="auto"`); everything specific to the log-distance
  bias says "unsquashed" (`unsquashed_k`, `unsquashed_lam`,
  `unsquashed_attention_bias`, `--unsquashed-k`). You choose the alibi
  prior or the unsquashed prior.
- `unsquash/ladder/` — tasks (`passkey`, `kv`, `copy`; string-seeded so
  every model scores bit-identical cases), runner (resumable results.jsonl,
  summary with effective context length), CLI.
- `modal_app.py` — `pretrain_from_scratch --prior`, new
  `run_ladder`/`ladder` entrypoints.
- Tests: `tests/test_alibi.py`, `tests/test_ladder.py` (chunked-vs-full
  parity is the load-bearing one: every long-context number flows through
  `chunked_hidden`).

## Known limits / deliberate scope cuts

- **Extreme-tier wall-clock is O(n²)**: at 256k expect minutes/case on an
  H100 fp32; the defaults (3 depths × 2 cases) keep it to a modest bill.
  524k works but budget accordingly.
- **bf16 pretraining plateaus** (unchanged from the prior run): the SDPA
  mask is cast to bf16 during training, quantizing adjacent-lag steps past
  lag ~40 for the prior. ALiBi's linear bias quantizes too (relative steps
  are coarser at long lags); evaluation always re-applies exact fp32.
- **No KV-cache decoding** anywhere in the bias paths — the ladder is
  deliberately teacher-forced; don't point generation code at these
  models without building the incremental-mask path first.
- **HotpotQA-at-length (tier 4 of the design) not implemented** — it needs
  the QA fine-tune harness; the synthetic tiers stand alone.
- **RULER-style multi-hop (variable tracking) not implemented** — `copy`
  covers the over-squashing capability probe; add VT later if the copy
  results are interesting.
- The pure no-RoPE ALiBi variant (literature-faithful 4th arm) is not
  wired; `ModelSpec` would need a `rope: bool` switch. Only worth it if
  reviewers demand it.
