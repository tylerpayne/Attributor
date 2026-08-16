"""CLI: python -m unsquash.ladder --model <checkpoint-dir> [options]"""

from __future__ import annotations

import argparse
import json
import logging

from unsquash.ladder.runner import (
    DEFAULT_EXTREME_LENGTHS,
    DEFAULT_LENGTHS,
    DEFAULT_DEPTHS,
    LadderSettings,
    run_ladder,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Long-context ladder: ppl-vs-length, passkey, kv "
        "retrieval, early-token copy, and the extreme length-extension test."
    )
    parser.add_argument("--model", "-m", required=True,
                        help="Checkpoint directory (HF export; the bias in "
                             "unsquash_prior.json is applied automatically)")
    parser.add_argument("--tiers", nargs="+",
                        default=["ppl", "passkey", "kv", "copy"],
                        choices=["ppl", "passkey", "kv", "copy"])
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=list(DEFAULT_LENGTHS))
    parser.add_argument("--depths", nargs="+", type=float,
                        default=list(DEFAULT_DEPTHS))
    parser.add_argument("--extreme_lengths", nargs="+", type=int,
                        default=list(DEFAULT_EXTREME_LENGTHS),
                        help="Passkey-only stress lengths; pass 0 to disable")
    parser.add_argument("--cases_per_cell", type=int, default=5)
    parser.add_argument("--extreme_cases_per_cell", type=int, default=2)
    parser.add_argument("--ppl_blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rope_theta_scale", type=float, default=1.0,
                        help="NTK-style theta rescale at load (1.0 = native)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--text_file", default=None,
                        help="Offline text source for the ppl tier")
    parser.add_argument("--out_dir", default=None,
                        help="Default: ladder_results/<model-slug>")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    slug = args.model.strip("/").replace("/", "__")
    extreme = tuple(n for n in args.extreme_lengths if n > 0)
    settings = LadderSettings(
        model=args.model,
        out_dir=args.out_dir or f"ladder_results/{slug}",
        tiers=tuple(args.tiers),
        lengths=tuple(args.lengths),
        depths=tuple(args.depths),
        extreme_lengths=extreme,
        cases_per_cell=args.cases_per_cell,
        extreme_cases_per_cell=args.extreme_cases_per_cell,
        ppl_blocks=args.ppl_blocks,
        seed=args.seed,
        rope_theta_scale=args.rope_theta_scale,
        device=args.device,
        text_file=args.text_file,
    )
    summary = run_ladder(settings)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
