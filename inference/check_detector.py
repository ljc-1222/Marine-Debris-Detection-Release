"""Verify and load one trained detector, then run a single-image forward check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.checkpoints import DEFAULT_MANIFEST, destination, load_manifest
from training.final_detectors import DETECTORS, forward_image, load_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=DETECTORS, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the catalog path under --directory")
    parser.add_argument("--directory", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    target = torch.device(args.device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.model == "codetr" and target.type != "cuda":
        raise ValueError("Co-DETR's compiled operators require a CUDA compute allocation")
    artifact = load_manifest(args.manifest)["artifacts"][args.model]
    checkpoint = args.checkpoint or destination(args.directory, artifact)
    loaded = load_detector(args.model, checkpoint, manifest=args.manifest, device=target)
    details = forward_image(loaded, args.image)
    print(json.dumps({
        "model": args.model, "checkpoint_sha256": artifact["sha256"],
        "checkpoint_bytes": artifact["bytes"], "device": str(target),
        "scope": "catalog identity, strict trained-state loading, one-image forward shapes and finiteness",
        "does_not_prove": "Full detector/TTA/classifier/CSV replay, training convergence, accuracy, or an official competition score",
        **details,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
