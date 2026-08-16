"""CLI: python -m unsquash.train --model <hf-model> [options]"""

from __future__ import annotations

import argparse
import dataclasses
import logging

from unsquash.train.runner import TrainSettings, train


def get_args():
    defaults = TrainSettings(model="")
    parser = argparse.ArgumentParser(
        description="Continued pretraining with the annealed unsquash prior."
    )
    parser.add_argument("--model", "-m", required=True)
    for f in dataclasses.fields(TrainSettings):
        if f.name == "model":
            continue
        default = getattr(defaults, f.name)
        if f.type == "bool" or isinstance(default, bool):
            parser.add_argument(f"--{f.name}", action="store_true")
        else:
            arg_type = type(default) if default is not None else str
            if f.name in ("unsquashed_k",):
                arg_type = float
            parser.add_argument(f"--{f.name}", type=arg_type, default=default)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    settings = TrainSettings(**vars(args))
    path = train(settings)
    print(f"\nFinal checkpoint: {path}")
    print("Evaluate it with:")
    print(f"  python -m unsquash.eval --model {path}")
    print("(the recorded unsquash_prior.json is applied automatically)")


if __name__ == "__main__":
    main()
