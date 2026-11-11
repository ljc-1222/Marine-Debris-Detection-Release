#!/usr/bin/env python3
"""CSV helpers reused from the project's frozen release scripts/submit.py."""

from __future__ import annotations

import csv
import math
from pathlib import Path

HEADER = ["image_filename", "label_id", "x", "y", "w", "h", "confidence"]
CLASS_COUNT = 34


def clip_xyxy(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


def submission_row(
    image_filename: str,
    label_id: int,
    box: tuple[float, float, float, float],
    confidence: float,
    image_width: int,
    image_height: int,
) -> dict | None:
    """Convert one valid detector output into a submission row."""

    if not all(math.isfinite(value) for value in (*box, confidence)):
        raise ValueError("Submission coordinates and confidence must be finite")
    if not 0 <= label_id < CLASS_COUNT:
        raise ValueError(f"Submission labels must be in 0..{CLASS_COUNT - 1}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Submission confidence must be in 0..1")
    x1, y1, x2, y2 = clip_xyxy(box, image_width, image_height)
    width, height = x2 - x1, y2 - y1
    if width <= 0.0 or height <= 0.0:
        return None
    return {
        "image_filename": image_filename,
        "label_id": label_id,
        "x": x1,
        "y": y1,
        "w": width,
        "h": height,
        "confidence": confidence,
    }


def write_submission(path: Path, rows: list[dict]) -> None:
    """Sort, format, and exclusively create a submission CSV."""

    if any(not 0 <= int(row["label_id"]) < CLASS_COUNT for row in rows):
        raise ValueError(f"Submission labels must be in 0..{CLASS_COUNT - 1}")
    numeric_fields = ("x", "y", "w", "h", "confidence")
    if any(
        not all(math.isfinite(float(row[field])) for field in numeric_fields)
        for row in rows
    ):
        raise ValueError("Submission numeric values must be finite")
    if any(
        float(row["x"]) < 0
        or float(row["y"]) < 0
        or float(row["w"]) <= 0
        or float(row["h"]) <= 0
        or not 0 <= float(row["confidence"]) <= 1
        for row in rows
    ):
        raise ValueError("Submission boxes and confidence are invalid")
    ordered = sorted(
        rows,
        key=lambda row: (
            row["image_filename"],
            -float(row["confidence"]),
            int(row["label_id"]),
            float(row["x"]),
            float(row["y"]),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    **row,
                    **{
                        field: f"{float(row[field]):.6f}"
                        for field in numeric_fields
                    },
                }
            )
