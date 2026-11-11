"""Prepare user RLE masks/RGBA banks, or import verified derived training inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from training.prepare_data import sha256, write_json
from training.prepare_sam_masks import (
    FIXED_MASKS_SHA256,
    manifest_path as mask_manifest_path,
    prepare_masks,
    verify_mask_annotations,
)

FIXED_BANK_SHA256 = "507c1869df8e314730b76c1737edcbcd679a9ba0b59eceaa996fa661d1a58858"


def read_training_annotations(path: Path) -> dict:
    document = json.loads(path.read_text())
    categories = sorted(document["categories"], key=lambda row: int(row["id"]))
    if [int(row["id"]) for row in categories] != list(range(34)):
        raise ValueError("SAM bank requires the zero-based training COCO view")
    expected = json.loads((Path(__file__).resolve().parents[1] / "configs/categories.json").read_text())["categories"]
    if [row["name"] for row in categories] != [row["name"] for row in expected]:
        raise ValueError("SAM bank class order differs from the fixed categories")
    if len({int(row["id"]) for row in document["annotations"]}) != len(document["annotations"]) or len({int(row["id"]) for row in document["images"]}) != len(document["images"]):
        raise ValueError("SAM training image and annotation IDs must be unique")
    return document


def bank_pipeline(annotations: Path, source_sha: str, origin: str, files: dict, objects: int) -> dict:
    return {"schema_version": 2, "source": {"annotations": str(annotations.resolve()), "annotations_sha256": sha256(annotations), "source_sha256": source_sha, "origin": origin}, "object_bank": {"mode": "sam3_rgba_only", "files": files, "objects": objects}}


def new_staging(output: Path) -> Path:
    if output.exists():
        raise FileExistsError("SAM bank output must use a new directory")
    staging = output.with_name(output.name + ".preparing")
    if staging.exists():
        raise FileExistsError("SAM staging directory already exists")
    staging.mkdir(parents=True)
    return staging


def import_bank(source: Path, annotations: Path, output: Path, *, expected_pipeline_sha256: str = FIXED_BANK_SHA256,
                image_root: Path | None = None) -> dict:
    source_sha = sha256(source / "pipeline.json")
    if source_sha != expected_pipeline_sha256:
        from training.portable_sam_bank import release_contract
        known_portable = release_contract().get("package_pipeline_sha256")
        if expected_pipeline_sha256 != FIXED_BANK_SHA256 or source_sha != known_portable:
            raise ValueError("SAM bank pipeline SHA256 does not match its contract")
    pipeline = json.loads((source / "pipeline.json").read_text())
    if pipeline.get("schema_version") in {3, 4}:
        from training.portable_sam_bank import import_portable_bank
        if pipeline["schema_version"] == 4 and image_root is None:
            raise ValueError("User SAM bank import requires --images for its source-image binding")
        return import_portable_bank(source, annotations, output, expected_pipeline_sha256=source_sha, image_root=image_root)
    if pipeline.get("object_bank", {}).get("mode") != "sam3_rgba_only":
        raise ValueError("SAM donor bank must contain SAM3 RGBA objects")
    artifact_hashes = pipeline.get("artifact_sha256", {})
    fixed_source = source_sha == FIXED_BANK_SHA256
    if not artifact_hashes:
        raise ValueError("SAM bank must bind its class manifests by SHA256")
    target = read_training_annotations(annotations)
    targets = {int(row["id"]): row for row in target["annotations"]}
    images = {int(row["id"]): row for row in target["images"]}
    staging = new_staging(output)
    try:
        files, objects, cache_bytes, seen = {}, 0, 0, set()
        for class_id, filename in pipeline["object_bank"]["files"].items():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("SAM class manifest escapes its bank")
            manifest_path = source / relative
            expected_manifest_sha = artifact_hashes.get(relative.as_posix())
            if sha256(manifest_path) != expected_manifest_sha:
                raise ValueError("SAM class manifest SHA256 does not match its pipeline")
            rows = json.loads(manifest_path.read_text())
            expected_count = pipeline["object_bank"].get("class_counts", {}).get(str(class_id))
            if expected_count is not None and len(rows) != int(expected_count):
                raise ValueError("SAM class donor count differs from its pipeline")
            for row in rows:
                annotation_id = int(row["annotation_id"])
                original = targets.get(annotation_id)
                if annotation_id in seen or original is None or int(original["category_id"]) != int(class_id) or int(row["class_id"]) != int(class_id):
                    raise ValueError("SAM donor annotation IDs/classes differ from the training union")
                seen.add(annotation_id)
                image_id = int(row["image_id"])
                if image_id not in images or image_id != int(original["image_id"]):
                    raise ValueError("SAM donor image IDs differ from the training union")
                if Path(row["source_path"]).name != Path(images[image_id]["file_name"]).name:
                    raise ValueError("SAM donor source filenames differ from the training union")
                source_box, target_box = row["source_bbox"], original["bbox"]
                if len(source_box) != 4 or len(target_box) != 4 or any(not math.isfinite(float(first)) or not math.isfinite(float(second)) or abs(float(first) - float(second)) > 1e-6 for first, second in zip(source_box, target_box)):
                    raise ValueError("SAM donor source boxes differ from the training union")
                cache = Path(row["rgba_path"])
                if cache.is_absolute() or ".." in cache.parts or not (source / cache).is_file():
                    raise ValueError("SAM RGBA donor is missing or escapes its bank")
                size = (source / cache).stat().st_size
                if size != int(row["cache_bytes"]):
                    raise ValueError("SAM donor cache bytes differ from its class manifest")
                with Image.open(source / cache) as cached:
                    if cached.format != "WEBP" or cached.size != tuple(row["cached_size"]):
                        raise ValueError("SAM donor cache dimensions differ from its class manifest")
                cache_bytes += size
                destination = staging / cache
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to((source / cache).resolve())
            (staging / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(manifest_path, staging / relative)
            files[class_id] = relative.as_posix()
            objects += len(rows)
        if not objects:
            raise ValueError("SAM bank contains no donor objects")
        expected_objects = pipeline["object_bank"].get("objects", pipeline["object_bank"].get("cache", {}).get("count"))
        if expected_objects is not None and objects != int(expected_objects):
            raise ValueError("SAM bank donor count differs from its pipeline")
        expected_bytes = pipeline["object_bank"].get("cache", {}).get("bytes")
        if expected_bytes is not None and cache_bytes != int(expected_bytes):
            raise ValueError("SAM bank cache bytes differ from its pipeline")
        result = bank_pipeline(annotations, source_sha, "fixed_bank" if fixed_source else "supplied_bank", files, objects)
        if artifact_hashes:
            result["artifact_sha256"] = {filename: artifact_hashes[filename] for filename in files.values()}
        for key in ("class_counts", "cache"):
            if key in pipeline["object_bank"]:
                result["object_bank"][key] = pipeline["object_bank"][key]
        result["source"]["source_annotations_sha256"] = pipeline.get("source", {}).get("annotations_sha256")
        write_json(staging / "pipeline.json", result)
        staging.rename(output)
        return result
    except BaseException:
        shutil.rmtree(staging)
        raise


def read_data_config(path: Path) -> dict:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(not isinstance(data.get(key), dict) for key in ("tinyformer", "rfdetr")):
        raise ValueError("Use the training data configuration generated by training.prepare_data")
    return data


def configure_training_inputs(path: Path, data: dict, bank: Path, masks: Path,
                              bank_identity: dict, mask_identity: dict, images: Path | None) -> None:
    import yaml

    path = path.resolve()

    def relative(value: Path) -> str:
        return Path(os.path.relpath(value.resolve(), path.parent)).as_posix()

    data["tinyformer"].update(
        sam_object_bank=relative(bank),
        sam_bank_sha256=bank_identity["source"].get("package_pipeline_sha256", bank_identity["source"]["source_sha256"]),
    )
    data["rfdetr"].update(
        sam_mask_annotations=relative(masks), sam_mask_manifest=relative(mask_manifest_path(masks)),
        sam_mask_sha256=mask_identity["masks_sha256"],
    )
    if images is not None:
        data["rfdetr"]["sam_image_root"] = relative(images)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("import", "migrate-masks", "prepare-masks", "build-bank"))
    parser.add_argument("--masks", type=Path, required=True, help="user COCO RLE mask JSON, or an explicitly obtained historical artifact")
    parser.add_argument("--mask-output", type=Path)
    parser.add_argument("--masks-sha256", help="SHA256 of the user-supplied mask JSON")
    parser.add_argument("--mask-manifest", type=Path, help="user mask sidecar; defaults to MASKS.with_suffix('.manifest.json')")
    parser.add_argument("--images", type=Path, help="local stored-pixel training images for user-derived input bindings")
    parser.add_argument("--annotations", type=Path, help="COCO GT: category IDs 1..34 for prepare-masks; 0..33 for build-bank/import")
    parser.add_argument("--bank", type=Path, help="user-prepared bank directory with pipeline.json, or a historical compatible bank")
    parser.add_argument("--bank-sha256", help="expected bank pipeline SHA256 supplied by its producer")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--data-config", type=Path, help="Update the prepared training YAML with imported masks and donors")
    args = parser.parse_args()
    if args.data_config is not None and args.mode != "import":
        parser.error("--data-config is only used when importing the training inputs")
    if args.mode == "prepare-masks":
        if any(value is None for value in (args.annotations, args.images, args.mask_output)):
            parser.error("prepare-masks requires --annotations (COCO 1..34), --images and --mask-output")
        from training.prepare_sam_masks import prepare_user_masks
        prepare_user_masks(args.masks, args.annotations, args.images, args.mask_output,
                           expected_sha256=args.masks_sha256 or sha256(args.masks))
        print(json.dumps({"masks": str(args.mask_output), "status": "ready"}, sort_keys=True))
        return
    if args.mode == "build-bank":
        if any(value is None for value in (args.annotations, args.images, args.output_root)):
            parser.error("build-bank requires --annotations (COCO 0..33), --images and --output-root")
        from training.portable_sam_bank import build_user_bank
        identity = build_user_bank(args.masks, args.annotations, args.images, args.output_root,
            expected_masks_sha256=args.masks_sha256 or sha256(args.masks), mask_manifest=args.mask_manifest)
        print(json.dumps({"bank": str(args.output_root), "objects": identity["objects"], "status": "ready"}, sort_keys=True))
        return
    if args.mask_output is None:
        parser.error("import and migrate-masks require --mask-output")
    if args.mode == "migrate-masks":
        if any((args.annotations, args.bank, args.bank_sha256, args.output_root)):
            parser.error("migrate-masks uses --masks and --mask-output only")
        print(json.dumps(prepare_masks(args.masks, args.mask_output), sort_keys=True))
        return
    if any(value is None for value in (args.annotations, args.bank, args.output_root)):
        parser.error("import requires --annotations, --bank and --output-root")
    data = read_data_config(args.data_config) if args.data_config is not None else None
    if args.mask_output.exists() or mask_manifest_path(args.mask_output).exists():
        raise FileExistsError("SAM mask output already exists")
    mask_contract = dict(expected_masks_sha256=args.masks_sha256 or sha256(args.masks), manifest=args.mask_manifest, image_root=args.images)
    verify_mask_annotations(args.masks, **mask_contract)
    result = import_bank(args.bank, args.annotations, args.output_root,
                         expected_pipeline_sha256=args.bank_sha256 or sha256(args.bank / "pipeline.json"), image_root=args.images)
    masks = prepare_masks(args.masks, args.mask_output, **mask_contract)
    if data is not None:
        configure_training_inputs(args.data_config, data, args.output_root, args.mask_output, result, masks, args.images)
    summary = {"objects": result["object_bank"]["objects"], "bank": str(args.output_root),
               "masks": str(args.mask_output), "status": "ready"}
    if args.data_config is not None:
        summary["data_config"] = str(args.data_config)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
