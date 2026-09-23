"""Command-line entry points; formal runs never start during preparation."""

import argparse
import logging
import re
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .fusion import METHODS
from .prepare import prepare
from .training import train_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--name", default="joint_v1")
    parser.add_argument("--rank-mode", choices=["positive_margin", "original"])
    parser.add_argument("--rank-weight", type=float)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch selection+refit; seed42/fold0; separate run namespace",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume only matching configuration/data/code"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if args.device:
        config = replace(config, device=args.device)
    if args.rank_mode is not None:
        config = replace(config, rank_mode=args.rank_mode)
    if args.rank_weight is not None:
        config = replace(config, rank_weight=args.rank_weight)
    config.validate()
    methods = args.methods.split(",")
    seeds = list(map(int, args.seeds.split(",")))
    folds = list(map(int, args.folds.split(",")))
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(m not in METHODS for m in methods)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(s < 0 for s in seeds)
        or not folds
        or len(set(folds)) != len(folds)
        or any(f not in range(5) for f in folds)
    ):
        parser.error("Invalid methods, seeds or folds")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("--name must contain only letters, digits, _ or -")
    cache = prepare(config)
    if args.command == "prepare":
        return
    if args.smoke:
        seeds = [42]
        folds = [0]
    name = args.name + ("_smoke" if args.smoke else "")
    train_runs(config, cache, methods, seeds, folds, args.smoke, args.resume, name)


if __name__ == "__main__":
    main()
