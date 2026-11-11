"""Build a frozen WBF bank or write a CSV from bank-aligned cached head logits."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import pipeline, three_head_runtime


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fuse = commands.add_parser("fuse", help="fuse seven original-coordinate COCO prediction JSON files")
    fuse.add_argument("--config", type=Path, default=root / "configs/postprocess.yaml")
    fuse.add_argument("--index", type=Path, required=True)
    fuse.add_argument("--prediction", action="append", required=True, metavar="NAME=PATH")
    fuse.add_argument("--output-dir", type=Path, required=True)
    fuse.set_defaults(func=pipeline.fuse_command)

    submit = commands.add_parser("submit", help="apply frozen parameters to aligned cached logits")
    submit.add_argument("--bank", type=Path, required=True)
    for name in three_head_runtime.HEADS:
        submit.add_argument(f"--{name}", type=Path, required=True, help="head logit-array manifest")
    submit.add_argument("--fusion", type=Path, default=root / "configs/fusion.json")
    submit.add_argument("--submission-config", type=Path, default=root / "configs/submission.json")
    submit.add_argument("--cache-bindings", type=Path, help="verified identity bindings for legacy caches only")
    submit.add_argument("--images", type=Path, help="Source image directory for current cache identity verification")
    submit.add_argument("--checkpoint-manifest", type=Path, help="Independently trusted catalog used to produce the caches")
    submit.add_argument("--expected-head", action="append", metavar="NAME=SHA256", help="Trusted producer receipt for a head manifest; repeat for all three heads")
    submit.add_argument("--legacy", action="store_true", help="Explicitly use independently bound historical caches and the old policy-global-top-100 order")
    submit.add_argument("--policy", choices=("specialist", "global"), default="specialist")
    submit.add_argument("--index", type=Path, required=True)
    submit.add_argument("--output-dir", type=Path, required=True)
    submit.set_defaults(func=three_head_runtime.run)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
