"""Prepare every labeled NAMR image for the fixed full-data training entrypoints."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMR_SOURCE_SHA256 = {
    "metadata_image_train.csv": "95681716068132edaaf604049834d9cb9609089817f3e82d4a7b71955f16c3a0",
    "metadata_annotation_train.csv": "fab161731e0aa94c0fcb7a08a9935db40246bdb59a9c7832fb73c309ba239a07",
    "data_NAMR33.yaml": "1ba38c049d40bff3af05ee6fdbe75c2172902717b4547dadea480e8788a1e6c6",
}
NAMR_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
NAMR_IMAGE_ORDER = ROOT / "configs/training_image_order.json"
NAMR_IMAGE_ORDER_SHA256 = "4fa948e9c3f9fa2e8cf02162b721cbbeae8f51e7c4a2a12b31a059058b353373"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_coco(annotations: Path, images: Path):
    document = json.loads(annotations.read_text())
    expected = json.loads((ROOT / "configs/categories.json").read_text())["categories"]
    categories = sorted(document["categories"], key=lambda row: int(row["id"]))
    if [(row["id"], row["name"]) for row in categories] != [(row["id"], row["name"]) for row in expected]:
        raise ValueError("COCO category IDs and names must match configs/categories.json")
    image_ids, annotation_ids, files = set(), set(), {}
    for image in document["images"]:
        image_id = int(image["id"])
        if "source_group" in image and (type(image["source_group"]) is not int or image["source_group"] not in (0, 1)):
            raise ValueError("COCO source_group must identify training group 0 or 1")
        relative = Path(image["file_name"])
        path = (images / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("COCO image file_name must remain inside the image root")
        if image_id in image_ids or relative.name in files or not path.is_file():
            raise ValueError("COCO images require unique IDs, filenames, and existing files")
        if int(image["width"]) <= 0 or int(image["height"]) <= 0:
            raise ValueError("COCO image dimensions must be positive")
        image_ids.add(image_id)
        files[relative.name] = path
    for row in document["annotations"]:
        box = row["bbox"]
        if int(row["id"]) in annotation_ids or int(row["image_id"]) not in image_ids:
            raise ValueError("COCO annotation IDs must be unique and reference known images")
        if int(row["category_id"]) not in range(1, 35) or len(box) != 4:
            raise ValueError("COCO annotations require categories 1..34 and XYWH boxes")
        if not all(math.isfinite(float(value)) for value in box) or min(map(float, box[2:])) <= 0:
            raise ValueError("COCO boxes require finite coordinates and positive dimensions")
        annotation_ids.add(int(row["id"]))
    return document, files


def training_image_order(path: Path, hashes: dict, count: int, *, check_release: bool = True):
    """Read the complete ID permutation and the two full-training source groups."""
    if check_release and (NAMR_IMAGE_ORDER_SHA256 is None or sha256(path) != NAMR_IMAGE_ORDER_SHA256):
        raise ValueError("NAMR image-order protocol does not match its fixed identity")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if type(protocol.get("schema_version")) is not int or protocol["schema_version"] != 1 or protocol.get("source_metadata_sha256") != hashes:
        raise ValueError("NAMR image-order protocol is not bound to these source metadata files")
    order = protocol.get("image_row_order")
    if (not isinstance(order, list) or len(order) != count
            or any(type(index) is not int for index in order) or set(order) != set(range(count))):
        raise ValueError("NAMR image-order protocol must contain every metadata row exactly once")
    groups = protocol.get("training_group_sizes")
    if (not isinstance(groups, list) or len(groups) != 2
            or any(type(size) is not int or size < 1 for size in groups) or sum(groups) != count):
        raise ValueError("NAMR training groups must be nonempty and cover the complete dataset")
    return order, groups


def read_namr(root: Path, *, check_release: bool = True, id_protocol: Path | None = None):
    """Convert all official training images using the fixed annotation-ID protocol."""
    root = root.resolve()
    hashes = {name: sha256(root / name) for name in NAMR_SOURCE_SHA256}
    if check_release and hashes != NAMR_SOURCE_SHA256:
        raise ValueError("NAMR metadata does not match the fixed v1.0 source files; see docs/training.md")
    categories = json.loads((ROOT / "configs/categories.json").read_text())["categories"]
    expected_names = {int(row["id"]) - 1: row["name"] for row in categories}
    taxonomy = yaml.safe_load((root / "data_NAMR33.yaml").read_text())
    raw_names = taxonomy.get("names")
    if isinstance(raw_names, list):
        names = dict(enumerate(raw_names))
    elif isinstance(raw_names, dict):
        names = {int(key): value for key, value in raw_names.items()}
    else:
        raise ValueError("NAMR taxonomy requires a names list or mapping")
    if int(taxonomy.get("nc", -1)) != 34 or names != expected_names:
        raise ValueError("NAMR class IDs/names must match configs/categories.json")

    images, files, stems = [], {}, {}
    image_root = root / "train/images"
    with (root / "metadata_image_train.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"image_file_name", "image_width", "image_height"} <= set(reader.fieldnames or ()):
            raise ValueError("NAMR image metadata is missing required columns")
        for row in reader:
            name = row["image_file_name"].strip()
            path = Path(name)
            if path.name != name or path.suffix.lower() not in NAMR_IMAGE_EXTENSIONS or name in files or path.stem in stems:
                raise ValueError("NAMR images require unique, plain image filenames")
            width, height = int(row["image_width"]), int(row["image_height"])
            if width <= 0 or height <= 0 or not (image_root / name).is_file():
                raise ValueError("NAMR images require positive dimensions and existing files")
            images.append({"file_name": name, "width": width, "height": height, "license": 1})
            files[name] = image_root / name
            stems[path.stem] = name
    disk_names = {path.name for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in NAMR_IMAGE_EXTENSIONS}
    if disk_names != set(files):
        raise ValueError("NAMR image files differ from the image metadata")

    grouped, seen = defaultdict(list), set()
    with (root / "metadata_annotation_train.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"bounding_box_name", "bbox1", "bbox2", "bbox3", "bbox4", "namr33_class_id", "namr33_class_name"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("NAMR annotation metadata is missing required columns")
        for row in reader:
            key = row["bounding_box_name"].strip()
            parts = key.rsplit("_", 2)
            if len(parts) != 3 or key in seen:
                raise ValueError("NAMR bounding-box names must contain unique image/total/sequence identifiers")
            stem, total, sequence = parts[0], int(parts[1]), int(parts[2])
            if stem not in stems or not 1 <= sequence <= total:
                raise ValueError("NAMR bounding-box name does not identify a valid image/sequence")
            seen.add(key)
            values = [float(row[f"bbox{index}"]) for index in range(1, 5)]
            cx, cy, width, height = values
            if not all(map(math.isfinite, values)) or min(width, height) <= 0 or not (0 <= cx - width / 2 <= cx + width / 2 <= 1 and 0 <= cy - height / 2 <= cy + height / 2 <= 1):
                raise ValueError("NAMR normalized boxes must be finite, positive and inside the image")
            label = int(row["namr33_class_id"])
            if label not in names or row["namr33_class_name"].strip() != names[label]:
                raise ValueError("NAMR annotation class ID/name differs from the taxonomy")
            grouped[stems[stem]].append((sequence, label, values))
    if set(grouped) != set(files):
        raise ValueError("Every NAMR training image must have annotation metadata")
    for rows in grouped.values():
        rows.sort(key=lambda row: row[0])
        if len({row[0] for row in rows}) != len(rows):
            raise ValueError("NAMR annotation sequences must be unique within an image")
    if check_release and (len(images), len(seen)) != (16618, 34189):
        raise ValueError("NAMR metadata counts differ from the fixed training release")
    protocol_path = NAMR_IMAGE_ORDER if id_protocol is None else Path(id_protocol)
    order, group_sizes = training_image_order(protocol_path, hashes, len(images), check_release=check_release)
    ordered = [images[index] for index in order]
    union = {"info": {"description": "Marine Debris Image Dataset v1.0 NAMR33 complete labeled training corpus", "version": "1.0"},
             "licenses": [{"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}],
             "images": [], "annotations": [],
             "categories": [{**row, "supercategory": "marine_debris"} for row in categories]}
    start = 0
    for group, group_size in enumerate(group_sizes):
        source_annotation_id = 0
        selected = ordered[start:start + group_size]
        start += group_size
        for image_id, image in enumerate(selected, 1):
            global_image_id = len(union["images"]) + 1
            union["images"].append({"id": global_image_id, **image, "source_group": group, "source_image_id": image_id})
            for _sequence, label, (cx, cy, width, height) in grouped[image["file_name"]]:
                width_px, height_px = width * image["width"], height * image["height"]
                box = [cx * image["width"] - width_px / 2, cy * image["height"] - height_px / 2, width_px, height_px]
                source_annotation_id += 1
                union["annotations"].append({"id": len(union["annotations"]) + 1, "image_id": global_image_id,
                                             "category_id": label + 1, "bbox": box, "area": box[2] * box[3], "iscrowd": 0,
                                             "source_group": group, "source_annotation_id": source_annotation_id})
    encoded = json.dumps(union, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    union = json.loads(encoded)
    sources = {"union": hashlib.sha256(encoded.encode()).hexdigest(), "image_order": sha256(protocol_path), **hashes}
    return union, files, sources


def zero_based(document: dict) -> dict:
    result = copy.deepcopy(document)
    for row in result["categories"]:
        row["id"] = int(row["id"]) - 1
    for row in result["annotations"]:
        row["category_id"] = int(row["category_id"]) - 1
    return result


def yolo_rows(document: dict) -> dict[str, str]:
    images = {int(row["id"]): row for row in document["images"]}
    rows = defaultdict(list)
    for annotation in sorted(document["annotations"], key=lambda row: int(row["id"])):
        if annotation.get("ignore") or annotation.get("iscrowd"):
            continue
        image = images[int(annotation["image_id"])]
        x, y, width, height = map(float, annotation["bbox"])
        normalized = ((x + width / 2) / image["width"], (y + height / 2) / image["height"], width / image["width"], height / image["height"])
        if not all(0 <= value <= 1 for value in normalized):
            raise ValueError("YOLO boxes must have normalized center and size in [0,1]")
        rows[int(image["id"])].append(str(int(annotation["category_id"]) - 1) + " " + " ".join(f"{value:.9f}" for value in normalized))
    result = {}
    for image_id, image in images.items():
        filename = Path(image["file_name"]).stem + ".txt"
        if filename in result:
            raise ValueError("YOLO images require unique filename stems")
        result[filename] = "\n".join(rows[image_id]) + ("\n" if rows[image_id] else "")
    return result


def prepare_views(document: dict, image_files: dict[str, Path], output: Path, source_hashes: dict) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError("data output must use a new directory")
    staging = output.with_name(output.name + ".preparing")
    if staging.exists():
        raise FileExistsError("data staging directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        union = copy.deepcopy(document)
        union["categories"] = sorted(union["categories"], key=lambda row: int(row["id"]))
        for row in union["images"]:
            row["file_name"] = Path(row["file_name"]).name
        if len({row["file_name"] for row in union["images"]}) != len(union["images"]):
            raise ValueError("training union filenames must be unique")
        images = staging / "images/train_all"
        images.mkdir(parents=True)
        for row in union["images"]:
            destination = output / "images/train_all" / row["file_name"]
            (images / row["file_name"]).symlink_to(os.path.relpath(image_files[row["file_name"]].resolve(), destination.parent))
        one_based = staging / "annotations/instances_train_all.json"
        zero = staging / "annotations/instances_train_all_0based.json"
        write_json(one_based, union)
        write_json(zero, zero_based(union))
        write_json(staging / "training.index.json", {"images": union["images"], "categories": union["categories"], "annotations": []})
        # RF-DETR requires both storage names; each contains the complete corpus.
        for storage_name in ("train", "valid"):
            native = staging / "rfdetr_train_all" / storage_name
            native.mkdir(parents=True)
            write_json(native / "_annotations.coco.json", union)
            for row in union["images"]:
                (native / row["file_name"]).symlink_to(Path("../../images/train_all") / row["file_name"])
        yolo = staging / "yolo_train_all"
        (yolo / "images/train").mkdir(parents=True)
        for row in union["images"]:
            (yolo / "images/train" / row["file_name"]).symlink_to(
                Path("../../../images/train_all") / row["file_name"]
            )
        labels = yolo / "labels/train"
        labels.mkdir(parents=True)
        for filename, contents in yolo_rows(union).items():
            (labels / filename).write_text(contents)
        names = [row["name"] for row in union["categories"]]
        (staging / "yolo_train_all.yaml").write_text(yaml.safe_dump({"path": "yolo_train_all", "train": "images/train", "val": "images/train", "names": dict(enumerate(names))}, sort_keys=False))
        config = yaml.safe_load((ROOT / "training/data.example.yaml").read_text())
        config["coco"] = {"images": "images/train_all", "annotations_1based": "annotations/instances_train_all.json", "annotations_0based": "annotations/instances_train_all_0based.json"}
        config["tinyformer"]["sam_object_bank"] = "augmentation/sam_object_bank"
        config["rfdetr"] = {"dataset_dir": "rfdetr_train_all", "sam_mask_annotations": "augmentation/instances_train_sam.json", "sam_image_root": "images/train_all"}
        config["yolo26"]["dataset_yaml"] = "yolo_train_all.yaml"
        config["dcr"] = {"records": "dcr/train_all.records.jsonl", "manifest": "dcr/train_all.records.manifest.json"}
        (staging / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        manifest = {"schema_version": 1, "source_annotations": source_hashes, "images": len(union["images"]), "objects": len(union["annotations"]), "holdout_images": 0, "annotations_1based_sha256": sha256(one_based), "annotations_0based_sha256": sha256(zero)}
        write_json(staging / "preparation.json", manifest)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namr-root", type=Path, required=True, help="extracted official NAMR v1.0 directory containing the full labeled training data")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    union, image_files, sources = read_namr(args.namr_root)
    print(json.dumps(prepare_views(union, image_files, args.output_root, sources), sort_keys=True))


if __name__ == "__main__":
    main()
