"""Collect one fixed detector's original or flip predictions as COCO JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import torch

from training.checkpoints import DEFAULT_MANIFEST, destination, load_manifest
from training.final_detectors import DETECTORS, class_names, load_detector
from inference import detector_backends as backends
from inference.detector_geometry import ENCODED_DECODER, encoded_image_info


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(path: Path, images: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    categories = document.get("categories", [])
    if [(row["id"], row["name"]) for row in categories] != list(enumerate(class_names(), 1)):
        raise ValueError("Inference index categories must match the ordered 34 project classes")
    if document.get("annotations"):
        raise ValueError("Inference indexes must not contain labels")
    rows = document.get("images")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Inference index must contain images")
    if len({row["id"] for row in rows}) != len(rows) or len({row["file_name"] for row in rows}) != len(rows):
        raise ValueError("Inference image IDs and filenames must be unique")
    for row in rows:
        if type(row["id"]) is not int:
            raise ValueError("Inference image IDs must be integers")
        name = PurePosixPath(row["file_name"])
        if not name.parts or name.is_absolute() or ".." in name.parts or "\\" in row["file_name"]:
            raise ValueError("Image filenames must be relative to --images")
        path = images / row["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        width, height, _ = encoded_image_info(path)
        if (width, height) != (int(row["width"]), int(row["height"])):
            raise ValueError(f"Encoded image dimensions disagree with the inference index: {row['file_name']}")
        if "sha256" in row and sha256(path) != row["sha256"]:
            raise ValueError(f"Image identity mismatch: {row['file_name']}")
    return document


def write_json(path: Path, value, *, compact=False):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=not compact, separators=(",", ":") if compact else None, indent=None if compact else 2)
        if not compact:
            handle.write("\n")
    try:
        os.link(partial, path)
    finally:
        partial.unlink()


def collect(args):
    index_path = args.index.expanduser().resolve()
    images = args.images.expanduser().resolve()
    output = args.output.expanduser().absolute()
    metadata_path = output.with_suffix(".manifest.json")
    if output == metadata_path:
        raise ValueError("Prediction output must have a distinct filename from its manifest")
    for path in (output, metadata_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if args.batch < 1 or not 0 <= args.workers <= 28:
        raise ValueError("Batch must be positive and workers must be in 0..28")
    if args.model == "codetr" and args.batch != 1:
        raise ValueError("Co-DETR requires --batch 1 with its fixed image preprocessing")
    target = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    document = load_index(index_path, images)
    index_hash = sha256(index_path)
    artifact = load_manifest(args.manifest)["artifacts"][args.model]
    checkpoint = destination(args.checkpoint_root.expanduser().resolve(), artifact)
    loaded = load_detector(args.model, checkpoint, manifest=args.manifest, device=target)
    if args.model == "codetr":
        predictions = backends.collect_codetr(loaded, document, images, index_path, batch=args.batch, workers=args.workers, flip=args.flip)
        precision, resolution = "native_mmcv", [2048, 1280]
    else:
        collector = {"tinyformer": backends.collect_tinyformer, "rfdetr": backends.collect_rfdetr, "yolo26": backends.collect_yolo26}[args.model]
        predictions = collector(loaded, document, images, batch=args.batch, flip=args.flip)
        precision = "bfloat16_autocast" if args.model == "tinyformer" and target.type == "cuda" else "float32"
        resolution = loaded.config["eval_spatial_size"] if args.model == "tinyformer" else int(loaded.config["resolution"]) if args.model == "rfdetr" else 960
    valid_images = {int(row["id"]) for row in document["images"]}
    if any(int(row["image_id"]) not in valid_images or not 1 <= int(row["category_id"]) <= 34 for row in predictions):
        raise ValueError("Detector output contains unknown image or category IDs")
    write_json(output, predictions, compact=True)
    metadata = {
        "schema_version": 1, "model": args.model, "view": "flip" if args.flip else "original",
        "checkpoint_sha256": artifact["sha256"], "checkpoint_bytes": artifact["bytes"],
        "checkpoint_catalog_sha256": sha256(args.manifest),
        "index_sha256": index_hash, "prediction_sha256": sha256(output),
        "images": len(document["images"]), "detections": len(predictions),
        "category_id_space": "coco_1_based", "coordinate_system": "encoded_image_pixels",
        "image_decoder": "native_mmcv_display_then_inverse_exif_v2" if args.model == "codetr" else ENCODED_DECODER,
        "settings": {"resolution": resolution, "threshold": backends.THRESHOLD, "candidates": backends.MAX_DETECTIONS, "batch": args.batch, "precision": precision},
        "scope": "Fresh detector inference only; no labels, fitting or score evaluation",
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=DETECTORS, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1, help="Images per batch; Co-DETR requires 1")
    parser.add_argument("--workers", type=int, default=0)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
