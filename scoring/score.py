"""Score a submission CSV with the COCO API at IoU 0.5.

The competition identifies COCO API mAP@0.5 as its accuracy metric. This
module computes a local score on caller-supplied annotations; it does not
authenticate ground-truth provenance or contact the competition platform.
See the Quick start in README.md for the input format and scoring procedure.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import hashlib
from importlib.metadata import version
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from postprocessing.submit import HEADER


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        raise ValueError(f"{name} must be {'a positive integer' if positive else 'an integer'}")
    return value


def load_annotations(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("COCO ground truth must be a JSON object")
    for field in ("images", "categories", "annotations"):
        if not isinstance(document.get(field), list):
            raise ValueError(f"COCO ground truth requires a {field} array")
    categories = [integer(row["id"], "category ID") for row in document["categories"]]
    if len(categories) != 34 or set(categories) != set(range(1, 35)):
        raise ValueError("ground truth must declare the fixed COCO category IDs 1..34")
    images: dict[int, Mapping[str, Any]] = {}
    filenames = set()
    for image in document["images"]:
        image_id = integer(image["id"], "image ID")
        filename = image["file_name"]
        if image_id in images or filename in filenames:
            raise ValueError("ground truth has duplicate image IDs or filenames")
        if not isinstance(filename, str) or not filename:
            raise ValueError("ground truth image filename must be nonempty text")
        for field in ("width", "height"):
            integer(image[field], f"image {field}", positive=True)
        images[image_id] = image
        filenames.add(filename)
    if not images:
        raise ValueError("ground truth has no images")
    annotation_ids = set()
    for annotation in document["annotations"]:
        annotation_id = integer(annotation["id"], "annotation ID", positive=True)
        if annotation_id in annotation_ids:
            raise ValueError("ground truth has duplicate annotation IDs")
        annotation_ids.add(annotation_id)
        if integer(annotation["image_id"], "annotation image ID") not in images:
            raise ValueError("ground truth annotation references an unknown image")
        if integer(annotation["category_id"], "annotation category ID") not in categories:
            raise ValueError("ground truth annotation references an unknown category")
        box = list(map(float, annotation["bbox"]))
        if (len(box) != 4 or not all(math.isfinite(value) for value in box)
                or box[2] <= 0 or box[3] <= 0):
            raise ValueError("ground truth bbox must be finite pixel XYWH with positive size")
        area = float(annotation["area"])
        if not math.isfinite(area) or area < 0:
            raise ValueError("ground truth area must be finite and nonnegative")
        if annotation.get("iscrowd") not in (0, 1):
            raise ValueError("COCO ground truth requires iscrowd 0 or 1")
    return document


def load_submission(path: Path, annotations: Mapping[str, Any]) -> list[dict[str, Any]]:
    images = {row["file_name"]: row["id"] for row in annotations["images"]}
    predictions = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise ValueError(f"submission columns must be {','.join(HEADER)}")
        for line_number, row in enumerate(reader, 2):
            try:
                if set(row) != set(HEADER) or row["image_filename"] not in images:
                    raise ValueError("unknown filename or malformed CSV row")
                label = int(row["label_id"])
                box = [float(row[field]) for field in ("x", "y", "w", "h")]
                score = float(row["confidence"])
                if not 0 <= label < 34:
                    raise ValueError("label_id must be in 0..33")
                if (not all(math.isfinite(value) for value in (*box, score))
                        or box[2] <= 0 or box[3] <= 0 or not 0 <= score <= 1):
                    raise ValueError("bbox/score must be finite, with positive size and score in 0..1")
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid submission row {line_number}: {error}") from error
            predictions.append({"image_id": images[row["image_filename"]],
                                "category_id": label + 1, "bbox": box, "score": score})
    return predictions


def score_submission(
    annotations_path: Path,
    submission_path: Path,
    *,
    split: str,
    ground_truth_source: str,
) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not split.strip() or not ground_truth_source.strip():
        raise ValueError("declare the split and ground-truth source")
    annotations = load_annotations(annotations_path)
    predictions = load_submission(submission_path, annotations)
    # Keep the CLI output to the final score/provenance, not API progress logs.
    with redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = {**annotations, "info": annotations.get("info", {})}
        truth.createIndex()
        if predictions:
            detected = truth.loadRes([dict(row) for row in predictions])
        else:
            detected = COCO()
            detected.dataset = {"images": annotations["images"],
                                "categories": annotations["categories"], "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.params.imgIds = sorted(truth.getImgIds())
        evaluator.params.catIds = list(range(1, 35))
        evaluator.params.iouThrs = np.asarray([0.5], dtype=np.float64)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
    # precision is [IoU, recall, category, area, maxDet]. COCO omits -1 entries
    # for categories without evaluable GT rather than assigning them zero AP.
    precision = evaluator.eval["precision"][0, :, :, 0, 2]
    valid = precision > -1
    if not valid.any():
        raise ValueError("mAP is undefined: supplied annotations have no evaluable ground truth")
    return {
        "schema_version": 1,
        "scope": "local_coco_api_score_on_user_supplied_annotations",
        "official_leaderboard_score": False,
        "metric": "COCO bbox mAP@0.5",
        "mAP50": float(precision[valid].mean()),
        "split": split,
        "ground_truth_source": ground_truth_source,
        "ground_truth_source_status": "caller_supplied_not_independently_authenticated",
        "annotations": {"filename": annotations_path.name, "sha256": sha256(annotations_path)},
        "submission": {"filename": submission_path.name, "sha256": sha256(submission_path)},
        "images": len(annotations["images"]),
        "annotation_rows": len(annotations["annotations"]),
        "prediction_rows": len(predictions),
        "declared_categories": 34,
        "categories_with_evaluable_ground_truth": int(valid.any(axis=0).sum()),
        "parameters": {
            "iou_thresholds": [0.5], "recall_thresholds": 101,
            "area": "all", "max_detections_per_image_per_category": 100,
            "category_ids": "1..34", "submission_label_ids": "0..33",
        },
        "software": {"pycocotools": version("pycocotools"), "numpy": np.__version__,
                     "scorer_sha256": sha256(Path(__file__))},
        "what_this_does_not_prove": (
            "Evaluation results apply to the provided annotation and prediction files."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="authorized COCO GT with IDs 1..34")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--split", required=True, help="actual annotation split, recorded without relabeling")
    parser.add_argument("--ground-truth-source", required=True, help="source URL/release description supplied by the caller")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = score_submission(args.annotations, args.submission, split=args.split,
                              ground_truth_source=args.ground_truth_source)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    print(encoded, end="")
