"""Encoded-image geometry reused from the frozen detector collection runtime."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

EXIF_ORIENTATION_TAG = 274
SWAPPED_EXIF_ORIENTATIONS = {5, 6, 7, 8}
ENCODED_DECODER = "encoded_rgb_ignore_exif_v1"


def _encoded_info(image):
    orientation = int(image.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
    orientation = orientation if orientation in range(1, 9) else 1
    # Pillow 12.3 reports the displayed TIFF size before its implicit transpose.
    size = (int(image.tag_v2[256]), int(image.tag_v2[257])) if image.format == "TIFF" else image.size
    return int(size[0]), int(size[1]), orientation


def encoded_image_info(path: Path):
    """Return stored width/height and EXIF orientation without loading pixels."""
    with Image.open(path) as image:
        return _encoded_info(image)


def decode_encoded_rgb(path: Path):
    """Decode stored pixels, ignoring EXIF, for the public encoded-coordinate contract."""
    with Image.open(path) as image:
        width, height, orientation = _encoded_info(image)
        is_tiff = image.format == "TIFF"
        rgb = image.convert("RGB")
    if is_tiff and orientation != 1:
        # Unlike JPEG/PNG, pinned Pillow implicitly orients TIFF while loading.
        inverse = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                   4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                   6: Image.Transpose.ROTATE_90, 7: Image.Transpose.TRANSVERSE,
                   8: Image.Transpose.ROTATE_270}
        rgb = rgb.transpose(inverse[orientation])
    if rgb.size != (width, height):
        raise ValueError("Decoded pixels disagree with the stored image dimensions")
    rgb.info.pop("exif", None)
    rgb.getexif().pop(EXIF_ORIENTATION_TAG, None)
    return rgb


def inverse_horizontal_boxes(boxes, width: float):
    return [(width - float(x2), float(y1), width - float(x1), float(y2)) for x1, y1, x2, y2 in boxes]


def display_size(width: float, height: float, orientation: int) -> tuple[float, float]:
    return (height, width) if orientation in SWAPPED_EXIF_ORIENTATIONS else (width, height)


def display_boxes_to_encoded(boxes_xyxy, width: float, height: float, orientation: int):
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    x1, y1, x2, y2 = boxes.T
    orientation = orientation if orientation in range(1, 9) else 1
    if orientation == 1:
        output = boxes.copy()
    elif orientation == 2:
        output = np.stack((width - x2, y1, width - x1, y2), axis=1)
    elif orientation == 3:
        output = np.stack((width - x2, height - y2, width - x1, height - y1), axis=1)
    elif orientation == 4:
        output = np.stack((x1, height - y2, x2, height - y1), axis=1)
    elif orientation == 5:
        output = np.stack((y1, x1, y2, x2), axis=1)
    elif orientation == 6:
        output = np.stack((y1, height - x2, y2, height - x1), axis=1)
    elif orientation == 7:
        output = np.stack((width - y2, height - x2, width - y1, height - x1), axis=1)
    else:
        output = np.stack((width - y2, x1, width - y1, x2), axis=1)
    output[:, [0, 2]] = np.clip(output[:, [0, 2]], 0.0, width)
    output[:, [1, 3]] = np.clip(output[:, [1, 3]], 0.0, height)
    return output.astype(np.float32, copy=False)


def codetr_geometry(document: dict[str, Any], images: Path):
    output = {}
    for row in document["images"]:
        encoded_width, encoded_height, orientation = encoded_image_info(images / row["file_name"])
        encoded_size = encoded_width, encoded_height
        width, height = float(row["width"]), float(row["height"])
        if encoded_size != (int(width), int(height)):
            raise ValueError("Encoded image dimensions disagree with the inference index")
        output[int(row["id"])] = width, height, display_size(width, height, orientation)[0], orientation
    return output


def restore_codetr_predictions(predictions, geometry, *, horizontal_flip: bool):
    grouped = defaultdict(list)
    for row in predictions:
        grouped[int(row["image_id"])].append(row)
    output = []
    for image_id, rows in grouped.items():
        width, height, display_width, orientation = geometry[image_id]
        boxes = np.asarray([[float(row["bbox"][0]), float(row["bbox"][1]), float(row["bbox"][0]) + float(row["bbox"][2]), float(row["bbox"][1]) + float(row["bbox"][3])] for row in rows], dtype=np.float32)
        if horizontal_flip:
            # A single augmentation uses simple_test, which rescales but never
            # inverse-flips. Undo its display-space flip once before EXIF.
            boxes = np.asarray(inverse_horizontal_boxes(boxes, display_width), dtype=np.float32)
        restored = display_boxes_to_encoded(boxes, width, height, orientation)
        for row, box in zip(rows, restored, strict=True):
            x1, y1, x2, y2 = map(float, box)
            box_width = width - x1 if x2 >= width else x2 - x1
            box_height = height - y1 if y2 >= height else y2 - y1
            if box_width > 0.0 and box_height > 0.0:
                output.append({**row, "bbox": [x1, y1, box_width, box_height]})
    return output
