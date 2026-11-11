"""Build the frozen candidate bank from seven detector prediction JSON files.

Reuses the fuse branch of the project's frozen release scripts/pipeline.py.
Flipped predictions must already be mapped back to original image coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from . import submit, wbf


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=not compact,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
        )
        if not compact:
            handle.write("\n")
    partial.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported pipeline config")
    if int(config.get("classes", 0)) != 34:
        raise ValueError("pipeline must contain exactly 34 classes")
    return config


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        if name in output:
            raise ValueError(f"duplicate input name: {name}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        output[name] = path
    return output


def dataset_ids(index: Mapping[str, Any]) -> tuple[set[int], set[int]]:
    image_ids = {int(row["id"]) for row in index.get("images", [])}
    category_ids = {int(row["id"]) for row in index.get("categories", [])}
    if not image_ids:
        raise ValueError("dataset index contains no images")
    if category_ids != set(range(1, 35)):
        raise ValueError("dataset index must contain category IDs 1..34")
    return image_ids, category_ids


def submission_rows(
    predictions: Sequence[Mapping[str, Any]], index: Mapping[str, Any]
) -> list[dict[str, Any]]:
    images = {int(row["id"]): row for row in index["images"]}
    rows = []
    for prediction in predictions:
        image = images[int(prediction["image_id"])]
        x, y, width, height = map(float, prediction["bbox"])
        row = submit.submission_row(
            str(image["file_name"]),
            int(prediction["category_id"]) - 1,
            (x, y, x + width, y + height),
            float(prediction["score"]),
            int(image["width"]),
            int(image["height"]),
        )
        if row is not None:
            rows.append(row)
    return rows


def write_submission(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    submit.write_submission(path, rows)


def fuse_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    index = load_json(args.index)
    image_ids, category_ids = dataset_ids(index)
    paths = parse_named_paths(args.prediction)
    raw = {
        name: wbf.index_predictions(
            load_json(path), image_ids=image_ids, category_ids=category_ids
        )
        for name, path in paths.items()
    }

    tta_sources: dict[str, wbf.Indexed] = {}
    for model, settings in config.get("tta", {}).items():
        original_name, flip_name = f"{model}_original", f"{model}_flip"
        missing = [name for name in (original_name, flip_name) if name not in raw]
        if missing:
            raise ValueError(f"missing TTA sources: {missing}")
        predictions = wbf.fuse(
            (raw[original_name], raw[flip_name]),
            settings["weights"],
            iou_threshold=float(settings["iou_threshold"]),
            score_threshold=float(settings["score_threshold"]),
            max_detections=int(settings["max_detections"]),
        )
        tta_sources[f"{model}_tta"] = wbf.index_predictions(predictions)

    settings = config["wbf"]
    sources = []
    for name in settings["sources"]:
        if name in tta_sources:
            sources.append(tta_sources[name])
        elif name in raw:
            sources.append(raw[name])
        else:
            raise ValueError(f"missing WBF source: {name}")
    candidates = wbf.fuse(
        sources,
        settings["weights"],
        iou_threshold=float(settings["iou_threshold"]),
        score_threshold=float(settings["score_threshold"]),
        max_detections=int(config["postprocess"]["max_detections"]),
        score_scales=settings["score_scales"],
        support_factors=settings["support_factors"],
        category_iou_thresholds={
            int(category): float(threshold)
            for category, threshold in settings.get(
                "category_iou_thresholds", {}
            ).items()
        },
        finalize=False,
    )
    for ordinal, row in enumerate(candidates):
        row["ordinal"] = ordinal

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "candidates.json"
    baseline_path = output / "wbf.predictions.json"
    submission_path = output / "wbf.submission.csv"
    write_json(candidates_path, candidates, compact=True)
    baseline = wbf.top_per_image(
        candidates, int(config["postprocess"]["max_detections"])
    )
    write_json(baseline_path, baseline, compact=True)
    write_submission(submission_path, submission_rows(baseline, index))
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "config_sha256": sha256(args.config),
            "index_sha256": sha256(args.index),
            "candidates": len(candidates),
            "candidate_sha256": sha256(candidates_path),
            "baseline_sha256": sha256(baseline_path),
            "submission_sha256": sha256(submission_path),
            "images": len(image_ids),
            "sources": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in sorted(paths.items())
            },
        },
    )
    print(f"Wrote {len(candidates)} candidates to {candidates_path}")
