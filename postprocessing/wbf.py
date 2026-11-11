"""Deterministic class-aware weighted box fusion."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


Indexed = dict[
    tuple[int, int], list[tuple[float, float, float, float, float]]
]


def iou(first: Iterable[float], second: Iterable[float]) -> float:
    first = tuple(first)
    second = tuple(second)
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _summarize(
    cluster: list[tuple[float, float, int, float, float, float, float]],
    total_weight: float,
    support_factors: tuple[float, float, float],
) -> tuple[float, float, float, float, float]:
    coordinate_weight = sum(row[0] for row in cluster)
    coordinate_index = 0
    if coordinate_weight <= 0.0:
        coordinate_weight = sum(row[1] for row in cluster)
        coordinate_index = 1
    coordinates = tuple(
        sum(row[coordinate_index] * row[3 + axis] for row in cluster)
        / coordinate_weight
        for axis in range(4)
    )
    support = len(cluster)
    confidence = (
        sum(row[0] for row in cluster)
        / total_weight
        * support_factors[min(support, 3) - 1]
    )
    return (*coordinates, min(confidence, 1.0))


def _fuse_group(
    model_boxes: tuple[list[tuple[float, float, float, float, float]], ...],
    weights: tuple[float, ...],
    score_scales: tuple[float, ...],
    support_factors: tuple[float, float, float],
    iou_threshold: float,
    score_threshold: float,
) -> list[tuple[float, float, float, float, float]]:
    candidates = sorted(
        (
            (
                score * score_scales[source] * weights[source],
                weights[source],
                source,
                x,
                y,
                x + width,
                y + height,
            )
            for source, boxes in enumerate(model_boxes)
            if weights[source] > 0.0
            for x, y, width, height, score in boxes
            if score >= score_threshold
        ),
        key=lambda row: row[0],
        reverse=True,
    )
    clusters: list[list[tuple[float, float, int, float, float, float, float]]] = []
    fused: list[tuple[float, float, float, float, float]] = []
    for candidate in candidates:
        best_index, best_overlap = -1, iou_threshold
        for index, (cluster, combined) in enumerate(zip(clusters, fused, strict=True)):
            if any(row[2] == candidate[2] for row in cluster):
                continue
            overlap = iou(candidate[3:7], combined[:4])
            if overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index < 0:
            clusters.append([candidate])
            fused.append(_summarize(clusters[-1], sum(weights), support_factors))
        else:
            clusters[best_index].append(candidate)
            fused[best_index] = _summarize(
                clusters[best_index], sum(weights), support_factors
            )
    return fused


def index_predictions(
    predictions: Sequence[dict[str, Any]],
    *,
    image_ids: set[int] | None = None,
    category_ids: set[int] | None = None,
) -> Indexed:
    indexed: dict[
        tuple[int, int], list[tuple[float, float, float, float, float]]
    ] = defaultdict(list)
    for row in predictions:
        image_id = int(row["image_id"])
        category_id = int(row["category_id"])
        box = tuple(map(float, row["bbox"]))
        score = float(row["score"])
        if image_ids is not None and image_id not in image_ids:
            raise ValueError(f"unknown image ID: {image_id}")
        if category_ids is not None and category_id not in category_ids:
            raise ValueError(f"unknown category ID: {category_id}")
        if len(box) != 4 or box[2] <= 0.0 or box[3] <= 0.0:
            raise ValueError(f"invalid COCO bbox: {box}")
        if not all(math.isfinite(value) for value in (*box, score)):
            raise ValueError("prediction contains a non-finite value")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid score: {score}")
        indexed[image_id, category_id].append((*box, score))
    return dict(indexed)


def top_per_image(
    predictions: Sequence[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[int(row["image_id"])].append(row)
    return [
        row
        for image_id in sorted(grouped)
        for row in sorted(
            grouped[image_id], key=lambda item: float(item["score"]), reverse=True
        )[:limit]
    ]


def fuse(
    sources: Sequence[Indexed],
    weights: Sequence[float],
    *,
    iou_threshold: float,
    score_threshold: float,
    max_detections: int,
    score_scales: Sequence[float] | None = None,
    support_factors: Sequence[float] = (1.0, 1.0, 1.0),
    category_iou_thresholds: Mapping[int, float] | None = None,
    finalize: bool = True,
) -> list[dict[str, Any]]:
    sources = tuple(sources)
    weights = tuple(map(float, weights))
    score_scales = tuple(map(float, score_scales or (1.0,) * len(sources)))
    support_factors = tuple(map(float, support_factors))
    category_iou_thresholds = {
        int(category): float(threshold)
        for category, threshold in (category_iou_thresholds or {}).items()
    }
    if len(sources) != len(weights) or len(sources) < 2:
        raise ValueError("sources and weights must have equal length >= 2")
    if len(score_scales) != len(sources):
        raise ValueError("score_scales must match sources")
    if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
        raise ValueError("weights must be non-negative with a positive sum")
    if any(value <= 0.0 for value in score_scales):
        raise ValueError("score scales must be positive")
    if len(support_factors) != 3 or not (
        0.0 < support_factors[0] <= support_factors[1] <= support_factors[2]
    ):
        raise ValueError("support_factors must contain three positive ordered values")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if any(not 0.0 < value <= 1.0 for value in category_iou_thresholds.values()):
        raise ValueError("category IoU thresholds must be in (0, 1]")
    if not 0.0 <= score_threshold <= 1.0 or max_detections < 1:
        raise ValueError("invalid score_threshold or max_detections")

    predictions: list[dict[str, Any]] = []
    keys = set().union(*(source.keys() for source in sources))
    for image_id, category_id in sorted(keys):
        fused = _fuse_group(
            tuple(source.get((image_id, category_id), []) for source in sources),
            weights,
            score_scales,
            support_factors,
            category_iou_thresholds.get(category_id, iou_threshold),
            score_threshold,
        )
        for x1, y1, x2, y2, score in fused:
            if score >= score_threshold and x2 > x1 and y2 > y1:
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
    return top_per_image(predictions, max_detections) if finalize else predictions
