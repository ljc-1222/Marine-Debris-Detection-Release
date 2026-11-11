#!/usr/bin/env python3
"""Apply fixed fusion parameters with per-class retention before specialist merge.

The bank/head readers and numerical functions are reused from the project's
release scripts/three_head_runtime.py. The current path retains each class's
top-100 until selection; historical policy-global truncation requires --legacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import cache_contract, submit, wbf


HEADS = ("hplus", "v7", "siglip")
EPS = 1e-8
CHUNK = 32768


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact_path(manifest: Path, row: Mapping[str, Any]) -> Path:
    path = Path(str(row["path"])).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = row.get("sha256")
    if expected is not None and sha256(path) != str(expected):
        raise ValueError(f"artifact SHA256 mismatch: {path}")
    if "bytes" in row and path.stat().st_size != row["bytes"]:
        raise ValueError(f"artifact byte count mismatch: {path}")
    return path.resolve()


def load_bank(path: Path) -> dict[str, np.ndarray]:
    document = load_json(path)
    if isinstance(document, list):
        rows = document
        return {
            "image_ids": np.asarray([row["image_id"] for row in rows], dtype=np.int64),
            "category_ids": np.asarray(
                [row["category_id"] for row in rows], dtype=np.int16
            ),
            "boxes_xywh": np.asarray([row["bbox"] for row in rows], dtype=np.float64).reshape(-1, 4),
            "scores": np.asarray([row["score"] for row in rows], dtype=np.float64),
            "ordinals": np.asarray(
                [row.get("ordinal", index) for index, row in enumerate(rows)],
                dtype=np.int64,
            ),
        }
    arrays = document.get("arrays") if isinstance(document, dict) else None
    if not isinstance(arrays, dict):
        raise ValueError("bank must be a candidate JSON array or array manifest")
    required = ("image_ids", "category_ids", "boxes_xywh", "scores")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"bank manifest is missing arrays: {missing}")
    bank = {
        name: np.load(artifact_path(path, arrays[name]), mmap_mode="r")
        for name in required
    }
    bank["ordinals"] = (
        np.load(artifact_path(path, arrays["ordinals"]), mmap_mode="r")
        if "ordinals" in arrays
        else np.arange(len(bank["image_ids"]), dtype=np.int64)
    )
    lengths = {len(value) for value in bank.values()}
    if (len(lengths) != 1 or bank["boxes_xywh"].shape != (len(bank["scores"]), 4)
            or any(bank[name].ndim != 1 for name in ("image_ids", "category_ids", "scores", "ordinals"))):
        raise ValueError("bank arrays are not row-aligned")
    if any(not np.issubdtype(bank[name].dtype, np.integer) for name in ("image_ids", "category_ids", "ordinals")):
        raise ValueError("bank ID and ordinal arrays must have integer dtypes")
    if any(not (np.issubdtype(bank[name].dtype, np.integer) or np.issubdtype(bank[name].dtype, np.floating))
           for name in ("boxes_xywh", "scores")):
        raise ValueError("bank boxes and scores must have real numeric dtypes")
    if int(document.get("rows", len(bank["scores"]))) != len(bank["scores"]):
        raise ValueError("bank row count changed")
    return bank


def load_head(manifest_path: Path, rows: int) -> dict[str, np.ndarray]:
    manifest = load_json(manifest_path)
    logits = np.load(
        artifact_path(manifest_path, manifest["class_logits"]), mmap_mode="r"
    )
    objectness = np.load(
        artifact_path(manifest_path, manifest["objectness_logits"]), mmap_mode="r"
    )
    if logits.shape != (rows, 34) or objectness.shape != (rows,):
        raise ValueError(f"head arrays are not aligned: {manifest_path}")
    if logits.dtype != np.float32 or objectness.dtype != np.float32:
        raise ValueError(f"head arrays must be float32: {manifest_path}")
    valid = np.ones(rows, dtype=bool)
    if "valid_crops" in manifest:
        valid = np.load(artifact_path(manifest_path, manifest["valid_crops"]), mmap_mode="r", allow_pickle=False)
        if valid.shape != (rows,) or valid.dtype != np.bool_:
            raise ValueError(f"valid-crop mask is not aligned: {manifest_path}")
    return {"logits": logits, "objectness": objectness, "valid": valid}


def check_alignment(bank_path: Path, head_path: Path, binding: Mapping[str, Any]) -> None:
    bank = load_json(bank_path)
    head = load_json(head_path)
    if head.get("candidate_sha256") is not None:
        if head["candidate_sha256"] != sha256(bank_path):
            raise ValueError(f"candidate identity mismatch: {head_path}")
        return
    candidate = bank.get("arrays", {}).get("candidate_ids") if isinstance(bank, dict) else None
    if candidate is not None and "candidate_ids" in head:
        bank_ids = artifact_path(bank_path, candidate)
        head_ids = artifact_path(head_path, head["candidate_ids"])
        if sha256(bank_ids) != sha256(head_ids):
            raise ValueError(f"candidate order mismatch: {head_path}")
        return
    # Old manifests without row identity require an independently verified binding.
    if binding.get("bank_sha256") != sha256(bank_path):
        raise ValueError(f"unbound legacy cache; supply --cache-bindings: {head_path}")
    for field in ("class_logits", "objectness_logits"):
        if binding.get(field) != sha256(artifact_path(head_path, head[field])):
            raise ValueError(f"legacy cache binding mismatch: {head_path}: {field}")


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64)
    shifted -= shifted.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, EPS, 1.0 - EPS)
    return np.log(clipped) - np.log1p(-clipped)


def effective_weights(
    weights: Sequence[float], valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (3,) or np.any(values < 0.0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("head weights must be non-negative and sum to one")
    per_row = np.broadcast_to(values, (len(valid), 3)).copy()
    per_row[~valid, 0] = 0.0
    return per_row, per_row.sum(axis=1)


def mix_columns(
    columns: Sequence[np.ndarray],
    weights: Sequence[float],
    valid: np.ndarray,
    family: str,
    *,
    neutral: float,
) -> np.ndarray:
    per_row, denominator = effective_weights(weights, valid)
    columns = [np.asarray(column, dtype=np.float64) for column in columns]
    output = np.full(len(valid), neutral, dtype=np.float64)
    active = denominator > 0.0
    if family == "geometric":
        numerator = sum(
            per_row[:, index] * np.log(np.clip(column, EPS, 1.0))
            for index, column in enumerate(columns)
        )
        output[active] = np.exp(numerator[active] / denominator[active])
    elif family == "odds":
        numerator = sum(
            per_row[:, index] * logit(column)
            for index, column in enumerate(columns)
        )
        output[active] = sigmoid(numerator[active] / denominator[active])
    elif family == "arithmetic":
        numerator = sum(
            per_row[:, index] * column for index, column in enumerate(columns)
        )
        output[active] = numerator[active] / denominator[active]
    else:
        raise ValueError(f"unsupported fusion family: {family}")
    return output


def build_features(
    heads: Mapping[str, Mapping[str, np.ndarray]], policy: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    if set(policy["fusion"]["weights"]) != set(HEADS):
        raise ValueError(f"policy must contain exactly {HEADS}")
    weights = tuple(float(policy["fusion"]["weights"][name]) for name in HEADS)
    family = str(policy["fusion"]["family"])
    rows = len(heads[HEADS[0]]["logits"])
    valid = np.asarray(heads["hplus"]["valid"], dtype=bool)
    probabilities = {
        name: np.empty((rows, 34), dtype=np.float32) for name in HEADS
    }
    objectness = {name: np.empty(rows, dtype=np.float32) for name in HEADS}
    for name in HEADS:
        for start in range(0, rows, CHUNK):
            stop = min(rows, start + CHUNK)
            probabilities[name][start:stop] = softmax(
                heads[name]["logits"][start:stop]
            )
            objectness[name][start:stop] = sigmoid(
                heads[name]["objectness"][start:stop]
            )

    fused = np.empty((rows, 34), dtype=np.float32)
    for start in range(0, rows, CHUNK):
        stop = min(rows, start + CHUNK)
        local_valid = valid[start:stop]
        for category in range(34):
            fused[start:stop, category] = mix_columns(
                [probabilities[name][start:stop, category] for name in HEADS],
                weights,
                local_valid,
                family,
                neutral=1.0,
            )
    fused_objectness = mix_columns(
        [objectness[name] for name in HEADS],
        weights,
        valid,
        "odds",
        neutral=1.0,
    ).astype(np.float32)
    model_top1 = np.column_stack(
        [np.argmax(probabilities[name], axis=1) for name in HEADS]
    ).astype(np.int16)
    model_top1[~valid, 0] = -1
    return {
        "q_all": fused,
        "objectness": fused_objectness,
        "top1": np.argmax(fused, axis=1).astype(np.int16),
        "model_top1": model_top1,
    }


def ranking_scores(
    bank: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
) -> np.ndarray:
    rows = np.arange(len(bank["scores"]), dtype=np.int64)
    classes = np.asarray(bank["category_ids"], dtype=np.int64) - 1
    q = np.asarray(features["q_all"])[rows, classes].astype(np.float64)
    scores = np.asarray(bank["scores"], dtype=np.float64) * np.power(
        np.clip(q, EPS, 1.0), float(policy["beta"])
    )
    scores *= np.power(
        np.clip(np.asarray(features["objectness"], dtype=np.float64), EPS, 1.0),
        float(policy.get("ranking", {}).get("objectness", 0.0)),
    )
    return scores


def rescue(
    bank: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    scores: np.ndarray,
    policy: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    settings = policy.get("rescue", {"mode": "off"})
    if settings.get("mode", "off") == "off":
        return {name: np.asarray(value) for name, value in bank.items()}, scores
    rows = np.arange(len(scores), dtype=np.int64)
    original = np.asarray(bank["category_ids"], dtype=np.int64) - 1
    alternative = np.asarray(features["top1"], dtype=np.int64)
    q_all = np.asarray(features["q_all"])
    original_q = np.asarray(q_all[rows, original], dtype=np.float64)
    alternative_q = np.asarray(q_all[rows, alternative], dtype=np.float64)
    votes = np.sum(
        np.asarray(features["model_top1"]) == alternative[:, None], axis=1
    )
    selected = (
        (alternative != original)
        & (votes >= int(settings["consensus"]))
        & (alternative_q >= float(settings["qmin"]))
        & (
            alternative_q / np.maximum(original_q, EPS)
            >= float(settings["ratio"])
        )
    )
    alternative_scores = np.asarray(bank["scores"], dtype=np.float64) * np.power(
        np.clip(alternative_q, EPS, 1.0), float(policy["beta"])
    )
    alternative_scores *= np.power(
        np.clip(np.asarray(features["objectness"], dtype=np.float64), EPS, 1.0),
        float(policy.get("ranking", {}).get("objectness", 0.0)),
    )
    alternative_scores *= float(settings.get("gamma", 1.0))
    mode = str(settings["mode"])
    if mode == "swap":
        changed = {name: np.asarray(value).copy() for name, value in bank.items()}
        changed["category_ids"][selected] = alternative[selected] + 1
        output_scores = scores.copy()
        output_scores[selected] = alternative_scores[selected]
        return changed, output_scores
    if mode != "add":
        raise ValueError(f"unsupported rescue mode: {mode}")
    selected_rows = np.flatnonzero(selected)
    combined = {
        "image_ids": np.concatenate(
            (np.asarray(bank["image_ids"]), np.asarray(bank["image_ids"])[selected_rows])
        ),
        "category_ids": np.concatenate(
            (
                np.asarray(bank["category_ids"]),
                alternative[selected_rows].astype(np.int16) + 1,
            )
        ),
        "boxes_xywh": np.concatenate(
            (
                np.asarray(bank["boxes_xywh"]),
                np.asarray(bank["boxes_xywh"])[selected_rows],
            )
        ),
        "ordinals": np.concatenate(
            (
                np.asarray(bank["ordinals"], dtype=np.int64) * 2,
                np.asarray(bank["ordinals"], dtype=np.int64)[selected_rows] * 2 + 1,
            )
        ),
    }
    return combined, np.concatenate((scores, alternative_scores[selected_rows]))


def iou(box: np.ndarray, other: np.ndarray) -> float:
    return wbf.iou(
        (box[0], box[1], box[0] + box[2], box[1] + box[3]),
        (other[0], other[1], other[0] + other[2], other[1] + other[3]),
    )


def finalize(
    bank: Mapping[str, np.ndarray], scores: np.ndarray, policy: Mapping[str, Any],
    *, per_class: bool = False,
) -> list[dict[str, Any]]:
    settings = policy["finalizer"]
    if str(settings["mode"]) != "soft_gaussian":
        raise ValueError("release runtime supports the frozen soft_gaussian finalizer")
    threshold = float(settings["threshold"])
    sigma = float(settings["sigma"])
    if sigma <= 0.0 or threshold < 0.0 or int(settings["topk"]) <= 0:
        raise ValueError("invalid finalizer settings")
    image_ids = np.asarray(bank["image_ids"])
    category_ids = np.asarray(bank["category_ids"])
    boxes = np.asarray(bank["boxes_xywh"])
    ordinals = np.asarray(bank["ordinals"], dtype=np.int64)
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in np.flatnonzero(adjusted >= threshold):
        groups[int(image_ids[index]), int(category_ids[index])].append(int(index))
    kept: list[int] = []
    for key in sorted(groups):
        remaining = groups[key]
        class_kept = 0
        while remaining and class_kept < int(settings["topk"]):
            remaining.sort(key=lambda row: (-adjusted[row], int(ordinals[row])))
            selected = remaining.pop(0)
            if adjusted[selected] < threshold:
                break
            kept.append(selected)
            class_kept += 1
            survivors = []
            for index in remaining:
                overlap = iou(boxes[selected], boxes[index])
                adjusted[index] *= math.exp(-(overlap * overlap) / sigma)
                if adjusted[index] >= threshold:
                    survivors.append(index)
            remaining = survivors
    by_image: dict[int, list[int]] = defaultdict(list)
    for index in kept:
        by_image[int(image_ids[index])].append(index)
    output = []
    for image_id in sorted(by_image):
        indices = np.asarray(by_image[image_id], dtype=np.int64)
        order = np.lexsort((ordinals[indices], -adjusted[indices]))
        selected_indices = indices[order] if per_class else indices[order][: int(settings["topk"])]
        for index in selected_indices:
            output.append(
                {
                    "image_id": image_id,
                    "category_id": int(category_ids[index]),
                    "bbox": [float(value) for value in boxes[index]],
                    "score": float(min(adjusted[index], 1.0)),
                }
            )
    return output


def apply_policy(
    bank: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
    *, per_class: bool = False, legacy_global_top100: bool = False,
) -> list[dict[str, Any]]:
    scores = ranking_scores(bank, features, policy)
    extended, extended_scores = rescue(bank, features, scores, policy)
    if legacy_global_top100:
        return finalize(extended, extended_scores, policy)
    predictions = finalize(extended, extended_scores, policy, per_class=True)
    if per_class:
        return predictions
    return merge_classes({"global": predictions}, {str(category): "global" for category in range(34)})


def merge_classes(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]], selected: Mapping[str, str]
) -> list[dict[str, Any]]:
    rows = [
        row
        for class_id, key in sorted(
            ((int(class_id), key) for class_id, key in selected.items())
        )
        for row in predictions[key]
        if int(row["category_id"]) == class_id + 1
    ]
    grouped: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for ordinal, row in enumerate(rows):
        grouped[int(row["image_id"])].append((ordinal, row))
    return [
        {**row, "bbox": np.asarray(row["bbox"], dtype=np.float32).astype(float).tolist()}
        for image_id in sorted(grouped)
        for _ordinal, row in sorted(
            grouped[image_id], key=lambda item: (-float(item[1]["score"]), item[0])
        )[:100]
    ]


def write_predictions(path: Path, predictions: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(list(predictions), handle, separators=(",", ":"))


def write_csv(
    path: Path, predictions: Sequence[Mapping[str, Any]], index: Mapping[str, Any]
) -> int:
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
    submit.write_submission(path, rows)
    return len(rows)


def load_policy(path: Path) -> dict[str, Any]:
    """Validate the fixed numerical contract before using the flat parameters."""
    config = load_json(path)
    if config.get("schema_version") != 3 or tuple(config.get("head_order", ())) != HEADS:
        raise ValueError(f"fusion config must have schema 3 and head order {HEADS}")
    if config.get("class_id_space") != "classifier_0_based":
        raise ValueError("class rescue overrides must use classifier IDs 0..33")
    numerics = {
        "epsilon": EPS,
        "class_logits_normalization": "softmax",
        "objectness_logits_normalization": "sigmoid",
        "objectness_logit_clip": [-50.0, 50.0],
        "probability_cache_dtype": "float32",
        "fusion_computation_dtype": "float64",
        "fused_class_probability_dtype": "float32",
        "fused_objectness_dtype": "float32",
        "normalize_fused_class_probabilities": False,
        "objectness_fusion": "weighted_log_odds",
        "invalid_hplus_weight": "zero_then_renormalize",
        "invalid_hplus_consensus_vote": -1,
    }
    if config.get("numerics") != numerics:
        raise ValueError("unsupported fusion numerical contract")
    finalizer = config["finalizer"]
    if (finalizer.get("mode") != "soft_gaussian"
            or finalizer.get("gaussian_decay_scope") != "all_overlaps"
            or finalizer.get("tie_break") != "input_ordinal"):
        raise ValueError("unsupported finalizer contract")
    merge = {
        "policy_evaluation_scope": "full_candidate_bank",
        "selection_key": "emitted_classifier_label",
        "finalize_policy_before_class_selection": True,
        "policy_topk_scope": "per_image_per_class",
        "policy_max_detections_per_class": 100,
        "score_threshold": 0,
        "max_detections": 100,
        "box_dtype": "float32",
        "score_dtype": "float64",
        "class_order": "ascending",
        "tie_break": "class_concatenation_ordinal",
    }
    if config.get("class_merge") != merge or int(finalizer["topk"]) != 100:
        raise ValueError("unsupported per-class-policy/final-global-top-100 contract")
    if config["fusion"].get("family") != "geometric":
        raise ValueError("the frozen policy requires geometric class fusion")
    if set(config.get("ranking", {})) != {"objectness"}:
        raise ValueError("only objectness ranking is supported")
    for class_id in config.get("class_rescue_overrides", {}):
        if str(int(class_id)) != class_id or not 0 <= int(class_id) < 34:
            raise ValueError("class rescue override is outside classifier IDs 0..33")
    json.dumps(config, allow_nan=False)
    return config


def base_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in ("beta", "finalizer", "fusion", "ranking", "rescue")
    }


def specialist_predictions(
    bank: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
    class_rescue_overrides: Mapping[str, Mapping[str, Any]],
    *, legacy_policy_global_top100: bool = False,
) -> list[dict[str, Any]]:
    """Keep each policy's class top-100 until specialist selection is complete."""
    policies = {}
    selected = {}
    for class_id in range(34):
        local = {
            **policy,
            "rescue": class_rescue_overrides.get(str(class_id), policy["rescue"]),
        }
        key = policy_hash(local)
        policies[key] = local
        selected[str(class_id)] = key
    predictions = {
        key: apply_policy(bank, features, local, per_class=not legacy_policy_global_top100,
                          legacy_global_top100=legacy_policy_global_top100)
        for key, local in policies.items()
    }
    return merge_classes(predictions, selected)


def load_submission_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    expected = {
        "schema_version": 1,
        "columns": submit.HEADER,
        "label_id_range": [0, 33],
        "box_format": "xywh_pixels",
        "coordinate_system": "encoded_image_pixels",
        "clip_to_image_bounds": True,
        "decimal_places": 6,
        "max_detections_per_image": 100,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported submission CSV contract")
    return config


def run(args: argparse.Namespace) -> dict[str, Any]:
    from .pipeline import dataset_ids

    config = load_policy(args.fusion)
    submission_config = load_submission_config(args.submission_config)
    legacy = bool(getattr(args, "legacy", False))
    if args.cache_bindings and not legacy:
        raise ValueError("Legacy cache bindings require explicit --legacy")
    index = load_json(args.index)
    image_ids, _ = dataset_ids(index)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    bank_path = args.bank.expanduser().resolve()
    bank = load_bank(bank_path)
    rows = len(bank["scores"])
    if not all(np.isfinite(value).all() for value in bank.values()):
        raise ValueError("non-finite candidate bank")
    if np.any((bank["category_ids"] < 1) | (bank["category_ids"] > 34)):
        raise ValueError("candidate category IDs must be in 1..34")
    if not set(map(int, bank["image_ids"])).issubset(image_ids):
        raise ValueError("candidate bank contains an unknown image ID")
    if np.any(bank["boxes_xywh"][:, 2:] <= 0) or np.any((bank["scores"] < 0) | (bank["scores"] > 1)):
        raise ValueError("candidate boxes/scores are invalid")
    bindings = load_json(args.cache_bindings) if args.cache_bindings else {}
    head_paths = {
        name: getattr(args, name).expanduser().resolve() for name in HEADS
    }
    legacy_cache = legacy and bool(args.cache_bindings)
    if legacy_cache:
        for name, path in head_paths.items():
            binding = dict(bindings.get(name, {}))
            if binding.get("bank_sha256") != sha256(bank_path):
                raise ValueError("Legacy caches require an independently verified bank binding: " + name)
            head = load_json(path)
            if binding.get("manifest_sha256") is not None:
                if binding["manifest_sha256"] != sha256(path):
                    raise ValueError("Legacy head manifest binding mismatch: " + name)
                for field in ("class_logits", "objectness_logits", "valid_crops"):
                    if field in head and not head[field].get("sha256"):
                        raise ValueError("Sealed legacy manifests must identify every array by SHA256")
                    if field in head:
                        artifact_path(path, head[field])
                        # The trusted manifest supplies array bindings for caches without row IDs.
                        binding[field] = head[field]["sha256"]
            else:
                for field in ("class_logits", "objectness_logits", "valid_crops"):
                    if field in head and binding.get(field) != sha256(artifact_path(path, head[field])):
                        raise ValueError("Legacy cache array binding mismatch: " + name + ": " + field)
            check_alignment(bank_path, path, binding)
    else:
        if not getattr(args, "images", None) or not getattr(args, "checkpoint_manifest", None):
            raise ValueError("Current caches require --images and --checkpoint-manifest")
        expected = {}
        for value in getattr(args, "expected_head", ()) or ():
            name, separator, checksum = value.partition("=")
            if (not separator or name not in HEADS or name in expected or len(checksum) != 64
                    or any(value not in "0123456789abcdef" for value in checksum)):
                raise ValueError("Expected three distinct --expected-head NAME=SHA256 seals")
            expected[name] = checksum
        if set(expected) != set(HEADS):
            raise ValueError("Supply an independently trusted --expected-head seal for every classifier")
        if [row.get("name") for row in sorted(index["categories"], key=lambda row: int(row["id"]))] != cache_contract.class_order():
            raise ValueError("Index class order differs from the fixed foreground classes")
        images = cache_contract.image_identities(index, args.images.expanduser().resolve(), set(map(int, bank["image_ids"])))
        for name, path in head_paths.items():
            cache_contract.verify_cache(path, name, bank_path, bank, args.index, images,
                                        args.checkpoint_manifest, expected[name])
    heads = {name: load_head(path, rows) for name, path in head_paths.items()}
    for name, head in heads.items():
        if (not np.isfinite(head["logits"]).all()
                or not np.isfinite(head["objectness"]).all()):
            raise ValueError(f"non-finite logits in {name}")
    policy = base_policy(config)
    features = build_features(heads, policy)
    mode = getattr(args, "policy", "specialist")
    predictions = (apply_policy(bank, features, policy, legacy_global_top100=legacy) if mode == "global" else specialist_predictions(
        bank, features, policy, config.get("class_rescue_overrides", {}), legacy_policy_global_top100=legacy
    ))
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.json"
    submission_path = output / "submission.csv"
    write_predictions(predictions_path, predictions)
    csv_rows = write_csv(submission_path, predictions, index)
    result: dict[str, Any] = {
        "schema_version": 2,
        "policy": ("legacy_class_specialist_policy_global_top100" if legacy else "class_specialist_per_class_top100") if mode == "specialist" else "global",
        "cache_verification_scope": "legacy_independent_bindings" if legacy_cache else "sealed_semantic_inputs_and_row_ids_v2",
        "bank_rows": rows,
        "bank_sha256": sha256(bank_path),
        "index_sha256": sha256(args.index),
        "fusion_sha256": sha256(args.fusion),
        "submission_config_sha256": sha256(args.submission_config),
        "head_manifest_sha256": {
            name: sha256(path) for name, path in head_paths.items()
        },
        "prediction_rows": len(predictions),
        "rows": csv_rows,
        "predictions_sha256": sha256(predictions_path),
        "submission_sha256": sha256(submission_path),
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result
