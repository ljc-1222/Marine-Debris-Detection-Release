"""Package the fixed SAM donor bytes with portable metadata and source credits."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

from PIL import Image

from training.prepare_data import NAMR_SOURCE_SHA256, ROOT, sha256, write_json
from training.prepare_sam import FIXED_BANK_SHA256, read_training_annotations

CONTRACT = ROOT / "configs/sam_bank_manifest.json"
FORMAT = "marine-debris-fixed-sam-bank"
USER_FORMAT = "marine-debris-user-sam-bank"
NAMR_URL = "https://mdimagenet.namr.gov.tw/namrOceanTrash/dataset/detail/JQdYLLKk9Qelnk6xAkqeDQ%3D%3D"
MASK_SOURCE_SHA256 = "8ed4294a3219a161fe32edee6c2143c5e756f6e1f2eb5590d5587a33c17787fc"
ROW_FIELDS = (
    "annotation_id", "image_id", "class_id", "source_path", "source_bbox",
    "mask_bbox", "mask_area", "rgba_path", "cached_size", "cache_bytes",
)
PACKAGE_README = f"""# Fixed SAM3 donor inputs

This package contains the fixed RGBA donor images used by Co-DETR and TinyFormer
Copy-Paste. It contains cropped source-image pixels. No original source-image
files, SAM weights, pretrained initializers, experiments or crop-mining outputs
are included.

Source dataset: [National Academy of Marine Research (NAMR)]({NAMR_URL}).
Retain NAMR and the image providers' credits in `ATTRIBUTION.json`, the dataset
source link, and the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
The provider fields are copied from the official training metadata; empty source
values remain empty. No capture GPS, device metadata or inferred author names
are added. Source-image rights and applicable third-party rights remain in force.

Modifications: existing SAM3 binary masks, boundary-clamped two-pixel crop padding,
RGBA WebP encoding, and portable metadata cleanup. The fixed WebP bytes are
preserved. The original source pipeline and this package have distinct SHA256
identities. Donor membership, class order and per-class row order are preserved.

SAM3 generated the masks upstream; it is not run during packaging or importing.
Meta's [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
governs its model/software materials and does not replace the source-image terms.
The project code's root license does not relicense these source-image derivatives.

Keep `pipeline.json`, every listed file, and `ATTRIBUTION.json` together. Verify
the pipeline against the separately supplied SHA256 before importing. Remote
repository/revision fields remain null until an actual publication is pinned.
"""


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_path(root: Path, value: str, *, existing: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("SAM package paths must stay relative to their bank")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError("SAM package paths must stay relative to their bank")
    path = root / relative
    if existing and (not path.is_file() or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError("SAM package file is missing or escapes its bank")
    return path


def release_contract() -> dict:
    contract = load_json(CONTRACT)
    if contract.get("schema_version") != 1 or contract.get("historical_pipeline_sha256") != FIXED_BANK_SHA256:
        raise ValueError("Unsupported fixed SAM release contract")
    return contract


def file_identity(path: Path) -> dict:
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def donor_order(rows_by_class: dict) -> str:
    import hashlib
    ordered = [[int(class_id), int(row["annotation_id"]), int(row["image_id"]), row["rgba_path"]]
               for class_id, rows in rows_by_class.items() for row in rows]
    return hashlib.sha256(json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def new_staging(output: Path) -> Path:
    staging = output.with_name(output.name + ".preparing")
    if output.exists() or staging.exists():
        raise FileExistsError("SAM package output and staging must be new")
    staging.mkdir(parents=True)
    return staging


def clean_row(row: dict, class_id: int, annotations: dict, images: dict) -> dict:
    if any(key not in row for key in ROW_FIELDS):
        raise ValueError("Fixed donor metadata lacks a required field")
    result = {key: deepcopy(row[key]) for key in ROW_FIELDS}
    annotation_id, image_id = int(result["annotation_id"]), int(result["image_id"])
    target = annotations.get(annotation_id)
    image = images.get(image_id)
    if target is None or image is None or int(target["image_id"]) != image_id or int(target["category_id"]) != class_id or int(result["class_id"]) != class_id:
        raise ValueError("SAM donor IDs/classes do not match the training annotations")
    source_name = Path(result["source_path"]).name
    if source_name != Path(image["file_name"]).name:
        raise ValueError("SAM donor source filename differs from its training image")
    result["source_path"] = source_name
    box = result["source_bbox"]
    if len(box) != 4 or len(target["bbox"]) != 4 or any(not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(float(a) - float(b)) > 1e-6 for a, b in zip(box, target["bbox"])):
        raise ValueError("SAM donor source bbox differs from its training annotation")
    safe_path(Path("."), result["rgba_path"])
    return result


def package_fixed_bank(
    source: Path, annotations: Path, image_metadata: Path, source_hashes: Path, output: Path,
    *, expected_source_sha256: str = FIXED_BANK_SHA256,
    expected_metadata_sha256: str = NAMR_SOURCE_SHA256["metadata_image_train.csv"],
) -> dict:
    """Copy pinned WebP bytes; remove run metadata and retain official credits."""
    if sha256(source / "pipeline.json") != expected_source_sha256:
        raise ValueError("Fixed source pipeline SHA256 mismatch")
    if sha256(image_metadata) != expected_metadata_sha256:
        raise ValueError("Official image metadata SHA256 mismatch")
    pipeline = load_json(source / "pipeline.json")
    document = read_training_annotations(annotations)
    targets = {int(row["id"]): row for row in document["annotations"]}
    images = {int(row["id"]): row for row in document["images"]}
    hashes = load_json(source_hashes)
    with image_metadata.open(encoding="utf-8-sig", newline="") as handle:
        providers = {}
        for row in csv.DictReader(handle):
            name = row["image_file_name"].strip()
            if name in providers:
                raise ValueError("Official metadata repeats an image filename")
            providers[name] = {field: row[field] for field in ("provider_account", "provider_account_zh")}
    staging = new_staging(output)
    try:
        files, manifests, class_counts, rows_by_class, credits, seen, used_paths = {}, {}, {}, {}, [], set(), set()
        for class_id, filename in pipeline["object_bank"]["files"].items():
            source_manifest = safe_path(source, filename, existing=True)
            if sha256(source_manifest) != pipeline.get("artifact_sha256", {}).get(filename):
                raise ValueError("Fixed class-manifest SHA256 mismatch")
            rows = []
            for original in load_json(source_manifest):
                row = clean_row(original, int(class_id), targets, images)
                annotation_id = int(row["annotation_id"])
                if annotation_id in seen or row["rgba_path"] in used_paths:
                    raise ValueError("SAM donor annotation or cache path is duplicated")
                seen.add(annotation_id)
                used_paths.add(row["rgba_path"])
                cached = safe_path(source, row["rgba_path"], existing=True)
                identity = file_identity(cached)
                if identity["sha256"] != hashes.get(row["rgba_path"]) or identity["bytes"] != int(row["cache_bytes"]):
                    raise ValueError("Fixed donor bytes do not match their source hash receipt")
                with Image.open(cached) as image:
                    if image.format != "WEBP" or image.size != tuple(row["cached_size"]) or any(key in image.info for key in ("exif", "xmp")):
                        raise ValueError("Donor format/shape differs or contains unexpected embedded metadata")
                destination = safe_path(staging, row["rgba_path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached, destination)
                files[row["rgba_path"]] = identity
                provider = providers.get(row["source_path"])
                if provider is None:
                    raise ValueError("SAM donor lacks its official image-provider credit")
                credits.append({"annotation_id": annotation_id, "image_id": int(row["image_id"]),
                                "file_name": row["source_path"], **provider})
                rows.append(row)
            destination = safe_path(staging, filename)
            write_json(destination, rows)
            identity = file_identity(destination)
            files[filename] = identity
            manifests[class_id] = filename
            class_counts[class_id] = len(rows)
            rows_by_class[class_id] = rows
        if not seen or set(hashes) != used_paths:
            raise ValueError("Source cache-hash receipt does not cover exactly the fixed donor set")
        expected_count = pipeline["object_bank"].get("objects", pipeline["object_bank"].get("cache", {}).get("count"))
        if expected_count is not None and len(seen) != int(expected_count):
            raise ValueError("Fixed donor count mismatch")
        if expected_source_sha256 == FIXED_BANK_SHA256 and len(seen) != 10647:
            raise ValueError("The release requires all 10647 fixed donors")
        expected_bytes = pipeline["object_bank"].get("cache", {}).get("bytes")
        if expected_bytes is not None and sum(files[path]["bytes"] for path in used_paths) != expected_bytes:
            raise ValueError("Fixed donor total byte count mismatch")
        write_json(staging / "ATTRIBUTION.json", {"source_url": NAMR_URL,
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "metadata_sha256": expected_metadata_sha256, "donors": credits})
        (staging / "README.md").write_text(PACKAGE_README, encoding="utf-8")
        for filename in ("ATTRIBUTION.json", "README.md"):
            files[filename] = file_identity(staging / filename)
        result = {"schema_version": 3, "format": FORMAT,
            "source": {"historical_pipeline_sha256": expected_source_sha256,
                "annotations_sha256": sha256(annotations), "mask_source_sha256": MASK_SOURCE_SHA256,
                "official_image_metadata_sha256": expected_metadata_sha256,
                "source_cache_hash_receipt_sha256": sha256(source_hashes), "dataset_url": NAMR_URL},
            "payload_identity": "original_webp_bytes_preserved",
            "metadata_transformation": "portable_donor_fields_and_official_provider_credits_v1",
            "object_bank": {"mode": "sam3_rgba_only", "files": manifests, "class_counts": class_counts,
                "objects": len(seen), "donor_order_sha256": donor_order(rows_by_class),
                "cache": deepcopy(pipeline["object_bank"].get("cache", {}))},
            "artifact_sha256": {filename: files[filename]["sha256"] for filename in manifests.values()},
            "files": files, "remote": {"repo_id": None, "revision": None}}
        write_json(staging / "pipeline.json", result)
        identity = file_identity(staging / "pipeline.json")
        verify_portable_bank(staging, expected_pipeline_sha256=identity["sha256"], expected_source_sha256=expected_source_sha256)
        staging.rename(output)
        return {"pipeline_sha256": identity["sha256"], "pipeline_bytes": identity["bytes"], "objects": len(seen)}
    except BaseException:
        shutil.rmtree(staging)
        raise


def verify_portable_bank(
    bank: Path, *, expected_pipeline_sha256: str | None = None,
    expected_source_sha256: str = FIXED_BANK_SHA256, manifest_name: str = "pipeline.json",
    annotations: Path | None = None, image_root: Path | None = None,
) -> dict:
    """Verify the externally pinned manifest and every donor/metadata file."""
    if expected_pipeline_sha256 is None:
        contract = release_contract()
        expected_pipeline_sha256 = contract.get("package_pipeline_sha256")
    if not expected_pipeline_sha256 or sha256(bank / manifest_name) != expected_pipeline_sha256:
        raise ValueError("Portable SAM pipeline SHA256 does not match its release contract")
    pipeline = load_json(bank / manifest_name)
    if pipeline.get("schema_version") == 4:
        return verify_user_bank(bank, pipeline, annotations=annotations, image_root=image_root)
    if pipeline.get("schema_version") != 3 or pipeline.get("format") != FORMAT or pipeline.get("payload_identity") != "original_webp_bytes_preserved":
        raise ValueError("Unsupported portable SAM bank")
    if pipeline.get("source", {}).get("historical_pipeline_sha256") != expected_source_sha256:
        raise ValueError("Portable SAM source identity mismatch")
    files = pipeline.get("files", {})
    for filename, expected in files.items():
        if file_identity(safe_path(bank, filename, existing=True)) != expected:
            raise ValueError("Portable SAM file SHA256 or bytes mismatch: " + filename)
    object_bank = pipeline.get("object_bank", {})
    if object_bank.get("mode") != "sam3_rgba_only":
        raise ValueError("Portable SAM bank requires RGBA-only donors")
    rows_by_class, ids, paths = {}, set(), set()
    required = {"ATTRIBUTION.json", "README.md"}
    for class_id, filename in object_bank.get("files", {}).items():
        if filename not in files or pipeline.get("artifact_sha256", {}).get(filename) != files[filename]["sha256"]:
            raise ValueError("Portable class manifest is not bound to its pipeline")
        rows = load_json(safe_path(bank, filename, existing=True))
        if len(rows) != object_bank.get("class_counts", {}).get(class_id):
            raise ValueError("Portable class donor count mismatch")
        rows_by_class[class_id] = rows
        required.add(filename)
        for row in rows:
            if tuple(row) != ROW_FIELDS or int(row["class_id"]) != int(class_id) or Path(row["source_path"]).name != row["source_path"]:
                raise ValueError("Portable donor fields/class/path mismatch")
            if row["annotation_id"] in ids or row["rgba_path"] in paths:
                raise ValueError("Portable donor ID/cache path repeats")
            ids.add(row["annotation_id"])
            paths.add(row["rgba_path"])
            required.add(row["rgba_path"])
            if row["rgba_path"] not in files or files[row["rgba_path"]]["bytes"] != int(row["cache_bytes"]):
                raise ValueError("Portable donor cache is not bound to its manifest")
    if required != set(files) or len(ids) != object_bank.get("objects") or donor_order(rows_by_class) != object_bank.get("donor_order_sha256"):
        raise ValueError("Portable SAM file set/count/order differs from its contract")
    if expected_source_sha256 == FIXED_BANK_SHA256 and len(ids) != 10647:
        raise ValueError("The release requires all 10647 fixed donors")
    credits = load_json(bank / "ATTRIBUTION.json")
    expected_credits = [(row["annotation_id"], row["image_id"], row["source_path"])
                        for rows in rows_by_class.values() for row in rows]
    if [(row["annotation_id"], row["image_id"], row["file_name"]) for row in credits.get("donors", [])] != expected_credits:
        raise ValueError("Portable image-provider credits do not follow the donor order")
    if any(set(row) != {"annotation_id", "image_id", "file_name", "provider_account", "provider_account_zh"} for row in credits["donors"]):
        raise ValueError("Portable credits include missing or unexpected metadata fields")
    return pipeline


def import_portable_bank(source: Path, annotations: Path, output: Path, *, expected_pipeline_sha256: str,
                         expected_source_sha256: str = FIXED_BANK_SHA256, image_root: Path | None = None) -> dict:
    pipeline = verify_portable_bank(source, expected_pipeline_sha256=expected_pipeline_sha256,
                                    expected_source_sha256=expected_source_sha256,
                                    annotations=annotations, image_root=image_root)
    document = read_training_annotations(annotations)
    if sha256(annotations) != pipeline["source"]["annotations_sha256"]:
        raise ValueError("Portable SAM bank does not match the canonical training annotations")
    targets = {int(row["id"]): row for row in document["annotations"]}
    images = {int(row["id"]): row for row in document["images"]}
    for class_id, filename in pipeline["object_bank"]["files"].items():
        for row in load_json(source / filename):
            if clean_row(row, int(class_id), targets, images) != row:
                raise ValueError("Portable SAM donor metadata differs from the training annotations")
    staging = new_staging(output)
    try:
        for filename in pipeline["files"]:
            destination = safe_path(staging, filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / filename, destination)
        shutil.copyfile(source / "pipeline.json", staging / "portable-package.json")
        user = pipeline["schema_version"] == 4
        local = {"schema_version": pipeline["schema_version"],
            "source": {"annotations": str(annotations.resolve()), "annotations_sha256": sha256(annotations),
                "source_sha256": expected_pipeline_sha256 if user else pipeline["source"]["historical_pipeline_sha256"],
                "origin": "user_prepared_bank" if user else "portable_fixed_bank", "package_pipeline_sha256": expected_pipeline_sha256},
            "portable_manifest": "portable-package.json", "object_bank": deepcopy(pipeline["object_bank"]),
            "artifact_sha256": deepcopy(pipeline["artifact_sha256"])}
        write_json(staging / "pipeline.json", local)
        staging.rename(output)
        return local
    except BaseException:
        shutil.rmtree(staging)
        raise


def verify_local_bank(bank: Path, local: dict, annotations: Path, *, expected_pipeline_sha256: str | None = None,
                      image_root: Path | None = None) -> None:
    if local.get("schema_version") == 4:
        if not expected_pipeline_sha256 or local.get("source", {}).get("package_pipeline_sha256") != expected_pipeline_sha256 or local.get("portable_manifest") != "portable-package.json":
            raise ValueError("User SAM bank requires its own pinned package SHA256")
        if image_root is None:
            raise ValueError("User SAM bank requires the local training image root")
        pipeline = verify_portable_bank(bank, expected_pipeline_sha256=expected_pipeline_sha256,
            manifest_name="portable-package.json", annotations=annotations, image_root=image_root)
        if local.get("object_bank") != pipeline["object_bank"] or local.get("artifact_sha256") != pipeline["artifact_sha256"]:
            raise ValueError("Local user SAM bank changed the donor order or mapping")
        if local["source"].get("annotations_sha256") != pipeline["source"]["annotations_sha256"]:
            raise ValueError("Local user SAM annotation binding differs from its package")
        return
    contract = release_contract()
    expected = contract.get("package_pipeline_sha256")
    if not expected or local.get("portable_manifest") != "portable-package.json" or local.get("source", {}).get("package_pipeline_sha256") != expected:
        raise ValueError("Local SAM bank lacks its pinned portable package identity")
    if (type(contract.get("package_pipeline_bytes")) is not int
            or (bank / "portable-package.json").stat().st_size != contract["package_pipeline_bytes"]):
        raise ValueError("Local SAM package manifest bytes differ from its release contract")
    pipeline = verify_portable_bank(bank, expected_pipeline_sha256=expected,
        expected_source_sha256=contract["historical_pipeline_sha256"], manifest_name="portable-package.json")
    if local.get("object_bank") != pipeline["object_bank"] or local.get("artifact_sha256") != pipeline["artifact_sha256"]:
        raise ValueError("Local SAM bank changed the pinned donor order or mapping")
    if sha256(annotations) != pipeline["source"]["annotations_sha256"] or local["source"].get("annotations_sha256") != pipeline["source"]["annotations_sha256"]:
        raise ValueError("Local SAM bank annotation content differs from its package")


def bank_settings(recipe: dict, bank: Path, *, expected_sha256: str | None, image_root: Path) -> dict:
    """Resolve input identity/cardinality separately from augmentation parameters."""
    result = {**recipe, "artifact_root": str(bank)}
    if not (bank / "pipeline.json").is_file():
        if expected_sha256:
            raise FileNotFoundError(bank / "pipeline.json")
        return result
    if load_json(bank / "pipeline.json").get("schema_version") == 4:
        if not expected_sha256:
            raise ValueError("Set tinyformer.sam_bank_sha256 to your user bank's package SHA256")
        result.update(bank_pipeline_sha256=expected_sha256, bank_objects=None, image_root=str(image_root))
    return result


def verify_user_bank(bank: Path, pipeline: dict, *, annotations: Path | None, image_root: Path | None) -> dict:
    """Validate a user's RGBA bank with its own manifest, not historical hashes."""
    if pipeline.get("format") != USER_FORMAT or pipeline.get("object_bank", {}).get("mode") != "rgba_only":
        raise ValueError("Unsupported user SAM bank format")
    files = pipeline.get("files", {})
    for filename, expected in files.items():
        if file_identity(safe_path(bank, filename, existing=True)) != expected:
            raise ValueError("User SAM bank file SHA256 or bytes mismatch")
    targets = images = None
    if annotations is not None:
        document = read_training_annotations(annotations)
        if sha256(annotations) != pipeline["source"].get("annotations_sha256"):
            raise ValueError("User SAM bank does not match the supplied training annotations")
        targets = {row["id"]: row for row in document["annotations"]}
        images = {row["id"]: row for row in document["images"]}
    rows_by_class, seen, required = {}, set(), set()
    object_bank = pipeline["object_bank"]
    for class_id, filename in object_bank["files"].items():
        if int(class_id) not in range(34) or filename not in files or pipeline["artifact_sha256"].get(filename) != files[filename]["sha256"]:
            raise ValueError("User SAM class manifest is not bound to its package")
        rows = load_json(bank / filename)
        if len(rows) != object_bank["class_counts"].get(class_id):
            raise ValueError("User SAM class count mismatch")
        rows_by_class[class_id] = rows
        required.add(filename)
        for row in rows:
            if (type(row["annotation_id"]) is not int or row["annotation_id"] < 1
                    or type(row["image_id"]) is not int or row["image_id"] < 1
                    or type(row["class_id"]) is not int or row["annotation_id"] in seen or row["class_id"] != int(class_id)):
                raise ValueError("User SAM donor IDs/classes are inconsistent")
            seen.add(row["annotation_id"])
            if targets is not None and clean_row(row, int(class_id), targets, images) != row:
                raise ValueError("User SAM donor crop/ID differs from training GT")
            filename_rgba = row["rgba_path"]
            required.add(filename_rgba)
            if filename_rgba not in files or files[filename_rgba]["bytes"] != row["cache_bytes"]:
                raise ValueError("User SAM RGBA is not bound to its class manifest")
            with Image.open(safe_path(bank, filename_rgba, existing=True)) as rgba:
                if rgba.mode != "RGBA" or rgba.size != tuple(row["cached_size"]) or rgba.getchannel("A").getbbox() is None:
                    raise ValueError("User SAM donor requires a nonempty RGBA image with the declared size")
                alpha = rgba.getchannel("A")
                binding = pipeline["source"]["image_bindings"].get(str(row["image_id"]))
                box = row["mask_bbox"]
                if binding is None or len(box) != 4 or any(not math.isfinite(float(value)) or int(value) != value for value in box) or min(box[2:]) <= 0:
                    raise ValueError("User SAM mask bbox/image binding is invalid")
                x, y, width, height = map(int, box)
                padding = int(object_bank.get("cache", {}).get("transparent_padding", 2))
                left, top = max(0, x-padding), max(0, y-padding)
                right, bottom = min(binding["width"], x+width+padding), min(binding["height"], y+height+padding)
                histogram = alpha.histogram()
                if (rgba.size != (right-left, bottom-top) or alpha.getbbox() != (x-left, y-top, x+width-left, y+height-top)
                        or sum(histogram[1:255]) or histogram[255] != row["mask_area"]):
                    raise ValueError("User SAM donor alpha/crop differs from its mask geometry")
    if (not seen or required != set(files) or len(seen) != object_bank["objects"]
            or donor_order(rows_by_class) != object_bank["donor_order_sha256"]):
        raise ValueError("User SAM donor set/count/order differs from its manifest")
    referenced = {str(row["image_id"]) for rows in rows_by_class.values() for row in rows}
    bindings = pipeline["source"]["image_bindings"]
    if set(bindings) != referenced:
        raise ValueError("User SAM source-image bindings do not cover its donor set")
    if image_root is not None:
        for image_id, binding in bindings.items():
            if images is not None and any(binding[key] != images.get(int(image_id), {}).get(key) for key in ("file_name", "width", "height")):
                raise ValueError("User SAM source image ID/dimensions differ from training GT")
            path = safe_path(image_root, binding["file_name"])
            if not path.is_file() or sha256(path) != binding["sha256"]:
                raise ValueError("User SAM source image bytes differ from its manifest")
    return pipeline


def build_user_bank(masks: Path, annotations: Path, image_root: Path, output: Path, *, expected_masks_sha256: str,
                    mask_manifest: Path | None = None) -> dict:
    """Build lossless RGBA crops from user-supplied RLE; no SAM inference."""
    import numpy as np
    from training.prepare_sam_masks import decode_rle, verify_mask_annotations
    verify_mask_annotations(masks, expected_masks_sha256=expected_masks_sha256, manifest=mask_manifest, image_root=image_root)
    document = read_training_annotations(annotations)
    targets = {row["id"]: row for row in document["annotations"]}
    images = {row["id"]: row for row in document["images"]}
    masks_document = load_json(masks)
    mask_images = {row["id"]: row for row in masks_document["images"]}
    rows_by_class = {str(index): [] for index in range(34)}
    bindings, files, manifests = {}, {}, {}
    staging = new_staging(output)
    try:
        for mask_row in masks_document["annotations"]:
            if mask_row.get("iscrowd", 0):
                continue
            aid, image_id, class_id = int(mask_row["id"]), int(mask_row["image_id"]), int(mask_row["category_id"]) - 1
            target = targets.get(aid)
            info = images.get(image_id)
            if target is None or info is None or target["image_id"] != image_id or target["category_id"] != class_id:
                raise ValueError("User bank mask IDs/classes differ from zero-based training GT")
            if any(mask_images[image_id][key] != info[key] for key in ("file_name", "width", "height")):
                raise ValueError("User bank mask image metadata differs from training GT")
            if "source_bbox" in mask_row and (len(mask_row["source_bbox"]) != 4 or any(not math.isfinite(float(a)) or abs(float(a)-float(b)) > 1e-6 for a, b in zip(mask_row["source_bbox"], target["bbox"]))):
                raise ValueError("User bank source crop differs from GT")
            source_path = safe_path(image_root, info["file_name"])
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            with Image.open(source_path) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            if rgb.shape[:2] != (info["height"], info["width"]):
                raise ValueError("User bank source-image stored dimensions differ")
            mask, _ = decode_rle(mask_row["segmentation"], info["height"], info["width"])
            yy, xx = np.nonzero(mask)
            x, y, width, height = int(xx.min()), int(yy.min()), int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)
            left, top, right, bottom = max(0, x-2), max(0, y-2), min(info["width"], x+width+2), min(info["height"], y+height+2)
            relative = f"rgba_cache/class_{class_id:02d}/annotation_{aid:08d}.png"
            destination = safe_path(staging, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.dstack((rgb[top:bottom, left:right], mask[top:bottom, left:right] * 255)), "RGBA").save(destination, format="PNG")
            files[relative] = file_identity(destination)
            rows_by_class[str(class_id)].append({"annotation_id": aid, "image_id": image_id, "class_id": class_id,
                "source_path": Path(info["file_name"]).name, "source_bbox": target["bbox"], "mask_bbox": [x,y,width,height],
                "mask_area": int(mask.sum()), "rgba_path": relative, "cached_size": [right-left,bottom-top],
                "cache_bytes": files[relative]["bytes"]})
            bindings[str(image_id)] = {key: info[key] for key in ("file_name", "width", "height")}
            bindings[str(image_id)]["sha256"] = sha256(source_path)
        for class_id, rows in rows_by_class.items():
            filename = f"object_bank/class_{int(class_id):02d}.json"
            write_json(staging / filename, rows)
            files[filename] = file_identity(staging / filename)
            manifests[class_id] = filename
        pipeline = {"schema_version": 4, "format": USER_FORMAT,
            "source": {"origin": "user_prepared_bank", "annotations_sha256": sha256(annotations),
                "masks_sha256": expected_masks_sha256, "mask_manifest_sha256": sha256(mask_manifest or masks.with_suffix(".manifest.json")),
                "image_bindings": bindings},
            "object_bank": {"mode": "rgba_only", "files": manifests,
                "class_counts": {key: len(rows) for key, rows in rows_by_class.items()},
                "objects": sum(map(len, rows_by_class.values())), "donor_order_sha256": donor_order(rows_by_class),
                "cache": {"format": "png_rgba", "transparent_padding": 2}},
            "artifact_sha256": {name: files[name]["sha256"] for name in manifests.values()}, "files": files}
        write_json(staging / "pipeline.json", pipeline)
        result = file_identity(staging / "pipeline.json")
        verify_portable_bank(staging, expected_pipeline_sha256=result["sha256"], annotations=annotations, image_root=image_root)
        staging.rename(output)
        return {"pipeline_sha256": result["sha256"], "pipeline_bytes": result["bytes"], "objects": pipeline["object_bank"]["objects"]}
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    package = commands.add_parser("package")
    package.add_argument("--bank", type=Path, required=True)
    package.add_argument("--annotations", type=Path, required=True)
    package.add_argument("--image-metadata", type=Path, required=True)
    package.add_argument("--source-hashes", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--bank", type=Path, required=True)
    verify.add_argument("--pipeline-sha256")
    verify.add_argument("--annotations", type=Path, help="zero-based COCO GT for user-bank binding checks")
    verify.add_argument("--images", type=Path, help="local source images for user-bank binding checks")
    args = parser.parse_args()
    if args.mode == "package":
        result = package_fixed_bank(args.bank, args.annotations, args.image_metadata, args.source_hashes, args.output)
    else:
        pipeline = verify_portable_bank(args.bank, expected_pipeline_sha256=args.pipeline_sha256,
                                       annotations=args.annotations, image_root=args.images)
        result = {"objects": pipeline["object_bank"]["objects"],
                  "source_sha256": pipeline["source"].get("historical_pipeline_sha256", pipeline["source"].get("masks_sha256"))}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
