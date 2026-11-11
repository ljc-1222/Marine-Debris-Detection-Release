"""Import portable fixed crop records, migrate originals, or prepare new records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from training.prepare_data import read_coco, sha256, write_json

FIXED_RECORDS_SHA256 = "8cf7b21f3956e0df12de8e2b875eedfca7960cde510d8916b52428be081bad92"
FOREGROUND_SOURCES = {"gt_original", "gt_jitter", "gt_jitter_fallback", "proposal_matched", "proposal_wrong_class"}
BACKGROUND_SOURCES = {"background_hard", "background_regular"}


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def manifest_path(records: Path) -> Path:
    return records.with_suffix(".manifest.json")


def valid_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def verify_records_manifest(records: Path, *, allow_new: bool = False, manifest: Path | None = None,
                             expected_records_sha256: str | None = None) -> dict:
    identity = json.loads((manifest_path(records) if manifest is None else manifest).read_text())
    if identity.get("schema_version") != 1 or sha256(records) != identity.get("records_sha256"):
        raise ValueError("DCR records do not match their preparation manifest")
    if expected_records_sha256 is not None and identity["records_sha256"] != expected_records_sha256:
        raise ValueError("DCR records SHA256 does not match the supplied input contract")
    source_hash = identity.get("source_sha256")
    if not valid_sha256(source_hash):
        raise ValueError("DCR preparation manifest lacks a source SHA256")
    fixed = identity.get("origin") == "fixed_record_migration" and identity.get("source_sha256") == FIXED_RECORDS_SHA256
    if not fixed and not allow_new:
        raise ValueError("these records have a new source identity; use --allow-new-records explicitly")
    if identity.get("origin") not in {"fixed_record_migration", "record_migration", "generated_from_proposals"}:
        raise ValueError("unsupported DCR records origin")
    return identity


def portable_row(row: dict, image_root: Path, image_hashes: dict[str, str]) -> dict:
    source = str(row["source_type"])
    label = int(row["label"])
    box = list(map(float, row["box_xyxy"]))
    if len(box) != 4 or not all(map(math.isfinite, box)) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("DCR records require finite, nondegenerate XYXY boxes")
    if not 0 <= label <= 34 or (source not in FOREGROUND_SOURCES if label < 34 else source not in BACKGROUND_SOURCES):
        raise ValueError("DCR record label and source type disagree")
    if label < 34 and row.get("gt_instance_id") is None:
        raise ValueError("foreground crop lacks GT instance identity")
    if label == 34 and bool(row.get("hard", source == "background_hard")) != (source == "background_hard"):
        raise ValueError("DCR background source type disagrees with its hard flag")
    relative = Path(str(row["image_path"]))
    if relative.is_absolute():
        relative = Path(relative.name)
    if ".." in relative.parts:
        raise ValueError("DCR image paths cannot escape the image root")
    image = (image_root / relative).resolve()
    # A prepared image root may contain symlinks to the supplied image corpus.
    if not image.is_file():
        raise FileNotFoundError("DCR source image is missing from the prepared training union")
    key = relative.as_posix()
    if key not in image_hashes:
        image_hashes[key] = sha256(image)
    expected_image = row.get("image_sha256")
    if expected_image is not None and image_hashes[key] != expected_image:
        raise ValueError("DCR image bytes do not match the source record")
    keep = ("record_id", "image_id", "image_width", "image_height", "gt_instance_id", "hard")
    return {**{name: row[name] for name in keep if name in row}, "image_path": key, "image_sha256": image_hashes[key], "box_xyxy": box, "label": label, "source_type": source}


def write_records(rows, output: Path, identity: dict) -> dict:
    if output.exists() or manifest_path(output).exists():
        raise FileExistsError("DCR output records or manifest already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError("DCR output staging file already exists")
    completed = False
    try:
        count = 0
        with partial.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical(row) + "\n")
                count += 1
        if not count:
            raise ValueError("DCR records cannot be empty")
        identity = {"schema_version": 1, **identity, "rows": count, "records_sha256": sha256(partial)}
        partial.replace(output)
        completed = True
        write_json(manifest_path(output), identity)
        return identity
    except BaseException:
        partial.unlink(missing_ok=True)
        if completed:
            output.unlink(missing_ok=True)
            manifest_path(output).unlink(missing_ok=True)
        raise


def migrate_records(source: Path, image_root: Path, output: Path, *, expected_source_sha256: str = FIXED_RECORDS_SHA256) -> dict:
    source_sha = sha256(source)
    if source_sha != expected_source_sha256:
        raise ValueError("source DCR records SHA256 does not match the supplied contract")
    image_hashes = {}
    def rows():
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield portable_row(json.loads(line), image_root, image_hashes)
    identity = {"origin": "fixed_record_migration" if source_sha == FIXED_RECORDS_SHA256 else "record_migration", "source_sha256": source_sha, "metadata_fields_removed": ["split"]}
    return write_records(rows(), output, identity)


def import_records(
    source: Path,
    image_root: Path,
    output: Path,
    *,
    expected_records_sha256: str,
    source_manifest: Path | None = None,
    allow_new: bool = False,
    annotations: Path | None = None,
) -> dict:
    """Bind released relative records to local images without changing sampler data."""
    if not valid_sha256(expected_records_sha256):
        raise ValueError("supply the producer's expected portable records SHA256")
    sidecar = manifest_path(source) if source_manifest is None else source_manifest
    original = verify_records_manifest(source, manifest=sidecar, allow_new=allow_new)
    if original["records_sha256"] != expected_records_sha256:
        raise ValueError("portable records SHA256 does not match the supplied release identity")
    expected_rows = original.get("rows")
    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError("portable records manifest requires a positive row count")
    identity = {
        "origin": original["origin"],
        "source_sha256": original["source_sha256"],
        "portable_source_sha256": original["records_sha256"],
        "portable_manifest_sha256": sha256(sidecar),
        "metadata_fields_removed": ["split"],
    }
    if original.get("annotations_sha256"):
        identity["annotations_sha256"] = original["annotations_sha256"]
    gt_images = gt = None
    if annotations is not None:
        document = read_coco(annotations, image_root)[0]
        if original.get("annotations_sha256") and original["annotations_sha256"] != sha256(annotations):
            raise ValueError("DCR record manifest does not match the supplied GT annotations")
        identity["annotations_sha256"] = sha256(annotations)
        gt_images = {row["id"]: row for row in document["images"]}
        gt = {row["id"]: row for row in document["annotations"]}
    image_hashes = {}

    def rows():
        count = 0
        seen = set()
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                relative = Path(str(row["image_path"]))
                if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                    raise ValueError("portable records require relative image paths inside --images")
                if not valid_sha256(row.get("image_sha256")):
                    raise ValueError("portable records require an image SHA256 for every row")
                # Validate with the existing crop contract, but preserve the original
                # row values/types instead of reconstructing them from converted floats.
                portable_row(row, image_root, image_hashes)
                if allow_new:
                    if not row.get("record_id") or row["record_id"] in seen:
                        raise ValueError("User DCR records require unique record_id values")
                    seen.add(row["record_id"])
                if gt is not None:
                    image = gt_images.get(row.get("image_id"))
                    if image is None or image["file_name"] != row["image_path"] or any(
                            key in row and row[key] != image[dimension]
                            for key, dimension in (("image_width", "width"), ("image_height", "height"))):
                        raise ValueError("DCR record image ID/path/dimensions differ from GT")
                    if int(row["label"]) < 34:
                        target = gt.get(row.get("gt_instance_id"))
                        if target is None or target["image_id"] != row["image_id"] or target["category_id"] - 1 != int(row["label"]):
                            raise ValueError("DCR foreground instance/class differs from GT")
                        if row["source_type"] == "gt_original" and any(abs(float(a)-b) > 1e-6 for a, b in zip(row["box_xyxy"], xyxy(target["bbox"]))):
                            raise ValueError("DCR original GT crop differs from its instance bbox")
                    elif row.get("gt_instance_id") is not None:
                        raise ValueError("DCR background must not claim a GT instance")
                row.pop("split", None)
                count += 1
                yield row
        if count != expected_rows:
            raise ValueError("portable record count does not match its manifest")
        # write_records consumes the stream before copying this identity.
        identity["images_verified"] = len(image_hashes)

    return write_records(rows(), output, identity)


def xyxy(box):
    x, y, width, height = map(float, box)
    return x, y, x + width, y + height


def area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(first, second) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(0.0, min(first[3], second[3]) - max(first[1], second[1]))


def iou(first, second) -> float:
    overlap = intersection(first, second)
    union = area(first) + area(second) - overlap
    return overlap / union if union > 0 else 0.0


def expand(box, scale=1.3):
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_width, half_height = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    return center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height


def clean_background(box, protected) -> bool:
    if area(box) <= 0:
        return False
    for candidate in (box, expand(box)):
        for other in protected:
            overlap = intersection(candidate, other)
            if max(iou(candidate, other), overlap / max(area(candidate), 1e-12), overlap / max(area(other), 1e-12)) > 0.1 + 1e-12:
                return False
    return True


def ambiguous(ious, labels) -> bool:
    eligible = sorted(((overlap, label) for overlap, label in zip(ious, labels) if overlap >= 0.5), reverse=True)
    if len(eligible) < 2:
        return False
    best, label = eligible[0]
    return any(best - overlap < 0.1 for overlap, other in eligible[1:] if other != label)


def jitter_box(box, boxes, labels, gt_index, seed, width, height):
    rng = random.Random(seed)
    x1, y1, x2, y2 = box
    size_x, size_y = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    for _ in range(16):
        moved_x = center_x + rng.uniform(-0.1, 0.1) * size_x
        moved_y = center_y + rng.uniform(-0.1, 0.1) * size_y
        new_x, new_y = size_x * rng.uniform(0.9, 1.1), size_y * rng.uniform(0.9, 1.1)
        candidate = (max(0.0, min(width, moved_x - new_x / 2)), max(0.0, min(height, moved_y - new_y / 2)), max(0.0, min(width, moved_x + new_x / 2)), max(0.0, min(height, moved_y + new_y / 2)))
        overlaps = [iou(candidate, other) for other in boxes]
        if area(candidate) > 0 and overlaps[gt_index] >= 0.5 and max(range(len(overlaps)), key=overlaps.__getitem__) == gt_index and not ambiguous(overlaps, labels):
            return candidate, False
    return box, True


def distinct(box, selected) -> bool:
    return all(iou(box, row["box"]) < 0.9 for row in selected)


def select_positive(rows):
    ordered = sorted(rows, key=lambda row: (-int(row["wrong"]), -row["score"] if row["wrong"] else 0.0, row["iou"], -row["score"], row["ordinal"]))
    selected = []
    for row in ordered:
        if len(selected) == 4:
            break
        if distinct(row["box"], selected):
            selected.append(row)
    return selected


def select_background(rows):
    hard = sorted((row for row in rows if row["hard"]), key=lambda row: (-row["score"], row["ordinal"]))
    regular = sorted((row for row in rows if not row["hard"]), key=lambda row: (-row["score"], row["ordinal"]))
    selected = []
    for pool, quota in ((hard, 6), (regular, 2), (hard + regular, 8)):
        added = 0
        for row in pool:
            if len(selected) >= 8 or added >= quota:
                break
            if row not in selected and distinct(row["box"], selected):
                selected.append(row)
                added += 1
    return selected


def generate_records(annotations: Path, images: Path, proposals: Path, output: Path, *, seed: int = 42) -> dict:
    document, _ = read_coco(annotations, images)
    categories = sorted(document["categories"], key=lambda row: int(row["id"]))
    mapping_hash = hashlib.sha256(canonical([row["name"] for row in categories]).encode()).hexdigest()
    gt_by_image, ignored_by_image, proposal_by_image = defaultdict(list), defaultdict(list), defaultdict(list)
    image_ids = {int(row["id"]) for row in document["images"]}
    for annotation in document["annotations"]:
        target = ignored_by_image if annotation.get("iscrowd") or annotation.get("ignore") else gt_by_image
        target[int(annotation["image_id"])].append(annotation)
    for ordinal, proposal in enumerate(json.loads(proposals.read_text())):
        score = float(proposal["score"])
        if int(proposal["image_id"]) not in image_ids or int(proposal["category_id"]) not in range(1, 35) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("proposals must reference labeled training images, classes 1..34, and finite scores")
        proposal_by_image[int(proposal["image_id"])].append((ordinal, proposal))
    records, backgrounds, hashes = [], [], {}
    def record(image, box, label, gt_id, source, ordinal, **extra):
        relative = Path(image["file_name"])
        key = relative.as_posix()
        if key not in hashes:
            hashes[key] = sha256(images / relative)
        payload = {"image_sha256": hashes[key], "box": [round(value, 8) for value in box], "label": label, "gt_id": gt_id, "source_type": source, "mapping_hash": mapping_hash, "ordinal": ordinal}
        return {"record_id": hashlib.sha256(canonical(payload).encode()).hexdigest()[:24], "image_id": int(image["id"]), "image_path": key, "image_sha256": hashes[key], "image_width": int(image["width"]), "image_height": int(image["height"]), "box_xyxy": list(box), "label": label, "gt_instance_id": gt_id, "source_type": source, "hard": source == "background_hard", **extra}
    for image in sorted(document["images"], key=lambda row: int(row["id"])):
        image_id = int(image["id"])
        gt_rows = sorted(gt_by_image[image_id], key=lambda row: int(row["id"]))
        boxes = [xyxy(row["bbox"]) for row in gt_rows]
        labels = [int(row["category_id"]) - 1 for row in gt_rows]
        ignored = [xyxy(row["bbox"]) for row in ignored_by_image[image_id]]
        for index, (gt, box, label) in enumerate(zip(gt_rows, boxes, labels)):
            records.append(record(image, box, label, int(gt["id"]), "gt_original", 0))
            jitter_seed = int(hashlib.sha256(f"{seed}:{gt['id']}:jitter".encode()).hexdigest(), 16)
            jitter, fallback = jitter_box(box, boxes, labels, index, jitter_seed, int(image["width"]), int(image["height"]))
            records.append(record(image, jitter, label, int(gt["id"]), "gt_jitter_fallback" if fallback else "gt_jitter", 1))
        positive, negative = defaultdict(list), []
        for ordinal, proposal in proposal_by_image[image_id]:
            raw = xyxy(proposal["bbox"])
            if not all(map(math.isfinite, raw)):
                raise ValueError("proposal boxes must be finite")
            box = (max(0.0, min(image["width"], raw[0])), max(0.0, min(image["height"], raw[1])), max(0.0, min(image["width"], raw[2])), max(0.0, min(image["height"], raw[3])))
            if area(box) <= 0 or any(intersection(box, other) > 0 for other in ignored):
                continue
            overlaps = [iou(box, other) for other in boxes]
            best = max(overlaps, default=0.0)
            score, predicted = float(proposal["score"]), int(proposal["category_id"]) - 1
            row = {"box": box, "score": score, "ordinal": ordinal}
            if best >= 0.5 and not ambiguous(overlaps, labels):
                index = max(range(len(overlaps)), key=overlaps.__getitem__)
                positive[index].append({**row, "iou": best, "wrong": predicted != labels[index]})
            elif best <= 0.1 and clean_background(box, boxes + ignored):
                negative.append({**row, "hard": score >= 0.3})
        for index, rows in positive.items():
            for ordinal, row in enumerate(select_positive(rows), 2):
                records.append(record(image, row["box"], labels[index], int(gt_rows[index]["id"]), "proposal_wrong_class" if row["wrong"] else "proposal_matched", ordinal))
        for ordinal, row in enumerate(select_background(negative)):
            backgrounds.append(record(image, row["box"], 34, None, "background_hard" if row["hard"] else "background_regular", ordinal, proposal_score=row["score"]))
    gt_count = sum(map(len, gt_by_image.values()))
    hard = sorted((row for row in backgrounds if row["hard"]), key=lambda row: (-row["proposal_score"], row["record_id"]))
    regular = sorted((row for row in backgrounds if not row["hard"]), key=lambda row: (-row["proposal_score"], row["record_id"]))
    quota = math.ceil(gt_count * 0.75)
    selected = hard[:quota] + regular[:gt_count - quota]
    used = {row["record_id"] for row in selected}
    selected.extend([row for row in hard + regular if row["record_id"] not in used][:gt_count - len(selected)])
    if len(selected) < gt_count or not hard or not regular:
        raise ValueError("fixed proposals lack the required clean background pool; adjust proposals to supply enough clean hard and regular backgrounds, or import your own records")
    records.extend(selected)
    records.sort(key=lambda row: (row["image_id"], row["gt_instance_id"] or 0, row["source_type"], row["record_id"]))
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("generated crop record IDs are not unique")
    return write_records(records, output, {"origin": "generated_from_proposals", "source_sha256": sha256(proposals), "annotations_sha256": sha256(annotations), "seed": seed, "fixed_source_sha256": FIXED_RECORDS_SHA256})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("import", "migrate", "generate"))
    parser.add_argument("--records", type=Path)
    parser.add_argument("--manifest", type=Path, help="portable sidecar; defaults to RECORDS.with_suffix('.manifest.json')")
    parser.add_argument("--records-sha256", help="producer's expected prepared-record digest; required by import")
    parser.add_argument("--allow-new-records", action="store_true", help="accept your own prepared record manifest instead of the historical source")
    parser.add_argument("--source-sha256", default=FIXED_RECORDS_SHA256)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "import":
        if args.records is None or args.records_sha256 is None or args.proposals is not None:
            parser.error("import requires --records and --records-sha256, with no proposals")
        identity = import_records(args.records, args.images, args.output,
                                  expected_records_sha256=args.records_sha256,
                                  source_manifest=args.manifest, allow_new=args.allow_new_records,
                                  annotations=args.annotations)
    elif args.mode == "migrate":
        if args.records is None or args.annotations is not None or args.proposals is not None or args.manifest is not None or args.records_sha256 is not None:
            parser.error("migrate requires --records only")
        identity = migrate_records(args.records, args.images, args.output, expected_source_sha256=args.source_sha256)
    else:
        if args.annotations is None or args.proposals is None or args.records is not None or args.manifest is not None or args.records_sha256 is not None:
            parser.error("generate requires --annotations and --proposals only")
        identity = generate_records(args.annotations, args.images, args.proposals, args.output)
    print(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()
