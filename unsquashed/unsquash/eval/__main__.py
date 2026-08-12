"""CLI: python -m unsquash.eval --model <hf-model-or-checkpoint> [options]"""

from __future__ import annotations

import argparse
import logging

from unsquash.eval import hotpot
from unsquash.eval.runner import EvalSettings, build_attributor, evaluate, print_summary
from unsquash.prior import PriorConfig
from unsquash.rollout import METHODS


def get_args():
    parser = argparse.ArgumentParser(
        description="A/B attribution evaluation on HotpotQA supporting facts."
    )
    parser.add_argument("--model", "-m", required=True,
                        help="HF model id or local checkpoint directory")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        choices=list(METHODS))
    parser.add_argument("--head_weighting", default="o_proj_norm",
                        choices=["o_proj_norm", "uniform"])
    parser.add_argument("--reduction", default="mean", choices=["mean", "sum"],
                        help="Span scoring: per-token mean (default) or legacy sum")
    parser.add_argument("--residual", default=None,
                        choices=["true", "false"],
                        help="Override the per-method residual default")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--hotpot_config", default="fullwiki")
    parser.add_argument("--max_context_tokens", type=int, default=2000)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--out_dir", default=None,
                        help="Default: unsquash_eval_results/<model>/<split>")
    parser.add_argument("--prior_k", type=float, default=None,
                        help="Apply the log-distance prior at capture time "
                             "(defaults to the checkpoint's recorded prior, if any)")
    parser.add_argument("--prior_lambda", type=float, default=1.0)
    parser.add_argument("--no_prior", action="store_true",
                        help="Ignore any prior recorded next to the checkpoint")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    if args.no_prior:
        prior = None
    elif args.prior_k is not None:
        prior = PriorConfig(k=args.prior_k, lam=args.prior_lambda)
    else:
        prior = "auto"

    attributor = build_attributor(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        head_weighting=args.head_weighting,
        prior=prior,
    )

    out_dir = args.out_dir or f"unsquash_eval_results/{args.model.replace('/', '__')}/{args.split}"
    settings = EvalSettings(
        methods=args.methods,
        ks=args.ks,
        reduction=args.reduction,
        max_context_tokens=args.max_context_tokens,
        max_cases=args.max_cases,
        residual=None if args.residual is None else args.residual == "true",
        out_dir=out_dir,
        extra={"model": args.model, "split": args.split,
               "head_weighting": args.head_weighting},
    )

    cases = hotpot.load_cases(
        split=args.split,
        config=args.hotpot_config,
        trust_remote_code=args.trust_remote_code,
    )
    summary = evaluate(attributor, cases, settings)
    print_summary(summary)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
