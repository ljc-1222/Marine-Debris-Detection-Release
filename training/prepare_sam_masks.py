"""Remove machine-specific mask metadata while preserving every training field."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re

from training.prepare_data import ROOT, read_coco, sha256

FIXED_MASKS_SHA256 = "8ed4294a3219a161fe32edee6c2143c5e756f6e1f2eb5590d5587a33c17787fc"
FIXED_MASKS_BYTES = 50677726
# Pinned by full-field comparison with the fixed source, excluding one path field.
PORTABLE_MASKS_SHA256: str | None = "06490fafd9745effe5833659783057f468582e02cdbaa3594669b92fe0578aa7"
PORTABLE_MASKS_BYTES: int | None = 50677587
TRANSFORMATION = "drop_info_source_annotations_v1"
REMOVED_FIELDS = ["info.source_annotations"]
PRIVATE_PATH = re.compile(r"/(?:home|work|mnt|root|workspace|tmp)/")


def manifest_path(masks: Path) -> Path:
    return masks.with_suffix(".manifest.json")


def _reject_private_paths(value) -> None:
    if isinstance(value, str):
        if PRIVATE_PATH.search(value):
            raise ValueError("SAM mask metadata still contains a private machine path")
    elif isinstance(value, dict):
        for key, child in value.items():
            _reject_private_paths(key)
            _reject_private_paths(child)
    elif isinstance(value, list):
        for child in value:
            _reject_private_paths(child)


def portable_document(document: dict) -> dict:
    """Delete only the known metadata field; never rewrite image names or masks."""
    result = deepcopy(document)
    info = result.get("info")
    if not isinstance(info, dict) or "source_annotations" not in info:
        raise ValueError("SAM mask source lacks info.source_annotations metadata")
    del info["source_annotations"]
    _reject_private_paths(result)
    for name in ("images", "annotations", "categories"):
        if not isinstance(result.get(name), list):
            raise ValueError(f"SAM masks require a {name} list")
    return result


def verify_mask_annotations(
    masks: Path,
    *,
    expected_source_sha256: str = FIXED_MASKS_SHA256,
    expected_source_bytes: int = FIXED_MASKS_BYTES,
    expected_portable_sha256: str | None = None,
    expected_portable_bytes: int | None = None,
    expected_masks_sha256: str | None = None,
    manifest: Path | None = None,
    annotations: Path | None = None,
    image_root: Path | None = None,
) -> dict:
    """Accept the fixed original, or a pinned portable file with its sidecar."""
    actual_bytes = masks.stat().st_size
    actual_sha = sha256(masks)
    if expected_masks_sha256 is not None and actual_sha != expected_masks_sha256:
        raise ValueError("SAM masks SHA256 does not match the supplied input contract")
    if actual_sha == expected_source_sha256 and actual_bytes == expected_source_bytes:
        return {"origin": "original_masks", "source_sha256": actual_sha, "source_bytes": actual_bytes,
                "masks_sha256": actual_sha, "masks_bytes": actual_bytes}
    if expected_masks_sha256 is not None:
        sidecar = manifest_path(masks) if manifest is None else manifest
        identity = json.loads(sidecar.read_text(encoding="utf-8"))
        if identity.get("origin") == "user_prepared_masks":
            if image_root is None:
                raise ValueError("User-prepared masks require the local image root")
            expected = user_mask_binding(json.loads(masks.read_text()), annotations, image_root)
            if (identity.get("schema_version") != 2 or identity.get("masks_sha256") != actual_sha
                    or identity.get("masks_bytes") != actual_bytes
                    or any(identity.get(key) != value for key, value in expected.items())):
                raise ValueError("User SAM masks do not match their manifest/GT/image binding")
            return identity
    portable_sha = PORTABLE_MASKS_SHA256 if expected_portable_sha256 is None else expected_portable_sha256
    portable_bytes = PORTABLE_MASKS_BYTES if expected_portable_bytes is None else expected_portable_bytes
    if portable_sha is None or portable_bytes is None or (actual_sha, actual_bytes) != (portable_sha, portable_bytes):
        raise ValueError("SAM mask artifact SHA256 or bytes do not match its contract")
    identity = json.loads(manifest_path(masks).read_text(encoding="utf-8"))
    origin = "fixed_mask_metadata_migration" if expected_source_sha256 == FIXED_MASKS_SHA256 else "mask_metadata_migration"
    expected = {"schema_version": 1, "origin": origin,
                "transformation": TRANSFORMATION, "removed_metadata_fields": REMOVED_FIELDS,
                "source_sha256": expected_source_sha256, "source_bytes": expected_source_bytes,
                "masks_sha256": actual_sha, "masks_bytes": actual_bytes}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("SAM masks do not match their preparation manifest")
    document = json.loads(masks.read_text(encoding="utf-8"))
    _reject_private_paths(document)
    if "source_annotations" in document.get("info", {}):
        raise ValueError("Portable SAM masks retain info.source_annotations")
    if any(identity.get(name) != len(document[name]) for name in ("images", "annotations", "categories")):
        raise ValueError("SAM mask counts do not match their preparation manifest")
    return identity


def _publish(content: bytes, output: Path, identity: dict) -> dict:
    sidecar = manifest_path(output)
    partial = output.with_name(output.name + ".partial")
    sidecar_partial = sidecar.with_name(sidecar.name + ".partial")
    if any(path.exists() for path in (output, sidecar, partial, sidecar_partial)):
        raise FileExistsError("SAM mask output, manifest or staging file already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    published = []
    temporary = []
    try:
        with partial.open("xb") as handle:
            temporary.append(partial)
            handle.write(content)
        identity = {**identity, "masks_sha256": sha256(partial), "masks_bytes": partial.stat().st_size}
        with sidecar_partial.open("x", encoding="utf-8") as handle:
            temporary.append(sidecar_partial)
            handle.write(json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        for source, destination in ((partial, output), (sidecar_partial, sidecar)):
            os.link(source, destination)
            published.append(destination)
        return identity
    except BaseException:
        for path in published:
            path.unlink()
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def migrate_masks(
    source: Path, output: Path, *, expected_source_sha256: str = FIXED_MASKS_SHA256
) -> dict:
    source_sha = sha256(source)
    if source_sha != expected_source_sha256:
        raise ValueError("source SAM masks SHA256 does not match the supplied contract")
    document = portable_document(json.loads(source.read_text(encoding="utf-8")))
    content = (json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    identity = {"schema_version": 1,
                "origin": "fixed_mask_metadata_migration" if source_sha == FIXED_MASKS_SHA256 else "mask_metadata_migration",
                "transformation": TRANSFORMATION, "removed_metadata_fields": REMOVED_FIELDS,
                "source_sha256": source_sha, "source_bytes": source.stat().st_size,
                **{name: len(document[name]) for name in ("images", "annotations", "categories")}}
    return _publish(content, output, identity)


def prepare_masks(source: Path, output: Path, **contract) -> dict:
    identity = verify_mask_annotations(source, **contract)
    if identity["origin"] == "original_masks":
        return migrate_masks(source, output)
    return _publish(source.read_bytes(), output, identity)


def decode_rle(segmentation: dict, height: int, width: int):
    """Normalize compressed or uncompressed COCO RLE without running SAM."""
    import numpy as np
    from pycocotools import mask as coco_mask
    if not isinstance(segmentation, dict) or segmentation.get("size") != [height, width]:
        raise ValueError("SAM masks require COCO RLE with the stored image height/width")
    encoded = dict(segmentation)
    if isinstance(encoded.get("counts"), list):
        encoded = coco_mask.frPyObjects(encoded, height, width)
    elif isinstance(encoded.get("counts"), str):
        encoded["counts"] = encoded["counts"].encode("ascii")
    else:
        raise ValueError("COCO RLE counts must be a compressed string or an integer list")
    mask = np.asarray(coco_mask.decode(encoded), dtype=np.uint8)
    if mask.shape != (height, width) or not mask.any():
        raise ValueError("SAM RLE must decode to a nonempty stored-size mask")
    encoded["counts"] = encoded["counts"].decode("ascii")
    return mask, encoded


def user_mask_binding(document: dict, annotations: Path | None, image_root: Path) -> dict:
    """Check user mask IDs/classes/crops against GT and bind donor image bytes."""
    import math
    import numpy as np
    from PIL import Image
    expected_categories = json.loads((ROOT / "configs/categories.json").read_text())["categories"]
    if [(row["id"], row["name"]) for row in document["categories"]] != [(row["id"], row["name"]) for row in expected_categories]:
        raise ValueError("User SAM masks require the ordered 34-class COCO taxonomy")
    canonical = None if annotations is None else read_coco(annotations, image_root)[0]
    gt_images = None if canonical is None else {row["id"]: row for row in canonical["images"]}
    gt = None if canonical is None else {row["id"]: row for row in canonical["annotations"]}
    images = {row["id"]: row for row in document["images"]}
    if len(images) != len(document["images"]) or any(type(key) is not int or key < 1 for key in images):
        raise ValueError("User SAM image IDs repeat")
    bindings, seen = {}, set()
    for row in document["annotations"]:
        if (type(row["id"]) is not int or row["id"] < 1 or type(row["image_id"]) is not int
                or type(row["category_id"]) is not int or row["id"] in seen
                or row["image_id"] not in images or row["category_id"] not in range(1, 35)):
            raise ValueError("User SAM annotation IDs/images/classes are invalid")
        seen.add(row["id"])
        image = images[row["image_id"]]
        relative = Path(image["file_name"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("User SAM image filenames must be relative")
        if gt_images is not None and any(image[key] != gt_images.get(image["id"], {}).get(key) for key in ("file_name", "width", "height")):
            raise ValueError("User SAM image IDs/dimensions differ from training GT")
        if gt is not None:
            target = gt.get(row["id"])
            if target is None or any(row[key] != target[key] for key in ("image_id", "category_id")):
                raise ValueError("User SAM annotation IDs/classes differ from training GT")
            source_box = row.get("source_bbox", target["bbox"])
            if len(source_box) != 4 or any(not math.isfinite(float(a)) or abs(float(a)-float(b)) > 1e-6 for a, b in zip(source_box, target["bbox"])):
                raise ValueError("User SAM source crop differs from its GT bbox")
        mask, _ = decode_rle(row["segmentation"], int(image["height"]), int(image["width"]))
        yy, xx = np.nonzero(mask)
        tight = [int(xx.min()), int(yy.min()), int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)]
        if "bbox" in row and (len(row["bbox"]) != 4 or any(not math.isfinite(float(a)) or abs(float(a)-b) > 1e-6 for a, b in zip(row["bbox"], tight))):
            raise ValueError("User SAM bbox does not match its decoded RLE")
        if "area" in row and float(row["area"]) != int(mask.sum()):
            raise ValueError("User SAM area does not match its decoded RLE")
        key = str(image["id"])
        if key not in bindings:
            path = image_root / relative
            with Image.open(path) as pixels:
                if pixels.size != (image["width"], image["height"]):
                    raise ValueError("User SAM image bytes have different stored dimensions")
            bindings[key] = {"file_name": relative.as_posix(), "width": image["width"], "height": image["height"], "sha256": sha256(path)}
    if not seen:
        raise ValueError("User SAM mask annotations cannot be empty")
    result = {"image_bindings": bindings, "images": len(images), "annotations": len(seen), "categories": len(document["categories"])}
    if annotations is not None:
        result["annotations_sha256"] = sha256(annotations)
    return result


def prepare_user_masks(source: Path, annotations: Path, image_root: Path, output: Path, *, expected_sha256: str) -> dict:
    """Seal user-supplied RLE/GT/image relationships in a portable sidecar."""
    import numpy as np
    if sha256(source) != expected_sha256:
        raise ValueError("User mask source SHA256 mismatch")
    document = json.loads(source.read_text())
    binding = user_mask_binding(document, annotations, image_root)
    targets = {row["id"]: row for row in read_coco(annotations, image_root)[0]["annotations"]}
    images = {row["id"]: row for row in document["images"]}
    for row in document["annotations"]:
        image = images[row["image_id"]]
        mask, row["segmentation"] = decode_rle(row["segmentation"], image["height"], image["width"])
        yy, xx = np.nonzero(mask)
        row.setdefault("bbox", [int(xx.min()), int(yy.min()), int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)])
        row.setdefault("area", int(mask.sum()))
        row.setdefault("iscrowd", 0)
        row.setdefault("source_bbox", targets[row["id"]]["bbox"])
    document.get("info", {}).pop("source_annotations", None)
    _reject_private_paths(document)
    content = (json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return _publish(content, output, {"schema_version": 2, "origin": "user_prepared_masks", "source_sha256": expected_sha256, **binding})
