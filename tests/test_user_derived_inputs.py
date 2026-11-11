from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import yaml

from training import portable_sam_bank as banks, prepare_sam
from training import prepare_sam_masks as masks, prepare_dcr_records as records
from training.prepare_data import sha256, write_json, zero_based
from training.rfdetr_copy_paste import ClassAwareCopyPaste, CopyPasteSettings
from training.sam_copy_paste import SAMCopyPastePlan

ROOT = Path(__file__).resolve().parents[1]


class UserDerivedInputsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-user-derived-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.categories = json.loads((ROOT / "configs/categories.json").read_text())["categories"]
        self.gt = {"categories": self.categories, "images": [], "annotations": []}
        self.mask_document = {"categories": self.categories, "images": [], "annotations": []}
        self.binary = {}
        for index, (image_id, annotation_id, category_id) in enumerate(((101, 701, 24), (202, 902, 30))):
            name = f"source-{image_id}.png"
            pixels = Image.new("RGB", (64, 48), (10, 20, 30))
            pixels.paste((230, 40 + index * 80, 60), (3, 4, 9, 9))
            pixels.save(self.root / name)
            # Prepared data normally uses source-image symlinks.
            (self.images / name).symlink_to(self.root / name)
            image = {"id": image_id, "file_name": name, "width": 64, "height": 48}
            self.gt["images"].append(image)
            self.gt["annotations"].append({"id": annotation_id, "image_id": image_id,
                "category_id": category_id, "bbox": [1.0, 2.0, 10.0, 9.0], "area": 90.0, "iscrowd": 0})
            binary = np.zeros((48, 64), dtype=np.uint8)
            binary[4:9, 3:9] = 1
            self.binary[annotation_id] = binary
            rle = coco_mask.encode(np.asfortranarray(binary))
            rle["counts"] = rle["counts"].decode("ascii")
            if index == 1:
                counts, value, length = [], 0, 0
                for pixel in binary.flatten(order="F"):
                    if pixel == value:
                        length += 1
                    else:
                        counts.append(length)
                        value, length = int(pixel), 1
                counts.append(length)
                rle["counts"] = counts
            self.mask_document["images"].append(image)
            self.mask_document["annotations"].append({"id": annotation_id, "image_id": image_id,
                "category_id": category_id, "segmentation": rle})
        self.gt1, self.gt0 = self.root / "gt-one.json", self.root / "gt-zero.json"
        write_json(self.gt1, self.gt)
        write_json(self.gt0, zero_based(self.gt))
        self.raw_masks = self.root / "supplied-rle.json"
        write_json(self.raw_masks, self.mask_document)
        self.prepared_masks = self.root / "prepared-masks.json"

    def prepare_masks(self):
        return masks.prepare_user_masks(self.raw_masks, self.gt1, self.images, self.prepared_masks,
                                        expected_sha256=sha256(self.raw_masks))

    def build_bank(self):
        identity = self.prepare_masks()
        package = self.root / "user-bank"
        bank_identity = banks.build_user_bank(self.prepared_masks, self.gt0, self.images, package,
                                               expected_masks_sha256=identity["masks_sha256"])
        return package, bank_identity

    def test_cli_prepares_and_configures_training_inputs_without_manual_checksums(self):
        config = self.root / "config" / "data.yaml"
        config.parent.mkdir()
        original = {
            "schema_version": 1, "class_names": [row["name"] for row in self.categories],
            "coco": {"images": "../images"},
            "tinyformer": {"sam_object_bank": "old-bank", "keep": 3},
            "rfdetr": {"dataset_dir": "unchanged-dataset"},
        }
        config.write_text(yaml.safe_dump(original, sort_keys=False))
        package, local = self.root / "user-bank", self.root / "local-bank"
        imported = self.root / "imported-masks.json"
        commands = [
            ["prepare-masks", "--masks", str(self.raw_masks), "--annotations", str(self.gt1),
             "--images", str(self.images), "--mask-output", str(self.prepared_masks)],
            ["build-bank", "--masks", str(self.prepared_masks), "--annotations", str(self.gt0),
             "--images", str(self.images), "--output-root", str(package)],
            ["import", "--bank", str(package), "--masks", str(self.prepared_masks),
             "--mask-output", str(imported), "--annotations", str(self.gt0),
             "--images", str(self.images), "--output-root", str(local), "--data-config", str(config)],
        ]
        for command in commands:
            with patch("sys.argv", ["prepare_sam", *command]), patch("sys.stdout", new_callable=io.StringIO) as output:
                prepare_sam.main()
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["status"], "ready")
            self.assertNotIn("sha256", output.getvalue())
        updated = yaml.safe_load(config.read_text())
        self.assertEqual(updated["class_names"], original["class_names"])
        self.assertEqual(updated["coco"], original["coco"])
        self.assertEqual(updated["tinyformer"]["keep"], 3)
        self.assertEqual(updated["rfdetr"]["dataset_dir"], "unchanged-dataset")
        self.assertEqual((config.parent / updated["tinyformer"]["sam_object_bank"]).resolve(), local)
        self.assertEqual((config.parent / updated["rfdetr"]["sam_mask_annotations"]).resolve(), imported)
        self.assertEqual((config.parent / updated["rfdetr"]["sam_mask_manifest"]).resolve(), masks.manifest_path(imported))
        recipe = yaml.safe_load((ROOT / "training/final_recipes.yaml").read_text())["detectors"]["tinyformer"]["sam_copy_paste"]
        settings = banks.bank_settings({**recipe, "base_images": 2}, local,
            expected_sha256=updated["tinyformer"]["sam_bank_sha256"], image_root=self.images)
        plan = SAMCopyPastePlan(settings, self.gt0, [101, 202])
        self.assertEqual(plan.stats["bank_objects"], 2)
        identity = masks.verify_mask_annotations(imported, annotations=self.gt1, image_root=self.images,
            manifest=config.parent / updated["rfdetr"]["sam_mask_manifest"],
            expected_masks_sha256=updated["rfdetr"]["sam_mask_sha256"])
        self.assertEqual(identity["origin"], "user_prepared_masks")

    def test_bad_training_config_is_rejected_before_import_outputs_are_created(self):
        config = self.root / "bad-data.yaml"
        config.write_text("tinyformer: []\nrfdetr: {}\n")
        local, imported = self.root / "local-bank", self.root / "imported-masks.json"
        command = ["prepare_sam", "import", "--bank", str(self.root / "bank"),
                   "--masks", str(self.raw_masks), "--annotations", str(self.gt0),
                   "--images", str(self.images), "--output-root", str(local),
                   "--mask-output", str(imported), "--data-config", str(config)]
        with patch("sys.argv", command), self.assertRaisesRegex(ValueError, "training data configuration"):
            prepare_sam.main()
        self.assertFalse(local.exists())
        self.assertFalse(imported.exists())

    def test_two_valid_user_mask_hashes_are_accepted_and_bound_to_gt_and_images(self):
        first = self.prepare_masks()
        self.assertNotEqual(first["masks_sha256"], masks.FIXED_MASKS_SHA256)
        verified = masks.verify_mask_annotations(self.prepared_masks, expected_masks_sha256=first["masks_sha256"],
                                                 annotations=self.gt1, image_root=self.images)
        self.assertEqual(verified, first)
        self.assertEqual(first["image_bindings"]["101"]["sha256"], sha256(self.images / "source-101.png"))
        document = json.loads(self.prepared_masks.read_text())
        self.assertEqual(document["annotations"][0]["bbox"], [3, 4, 6, 5])
        self.assertEqual(document["annotations"][0]["source_bbox"], [1.0, 2.0, 10.0, 9.0])
        np.testing.assert_array_equal(coco_mask.decode(document["annotations"][0]["segmentation"]), self.binary[701])
        self.assertIsInstance(document["annotations"][1]["segmentation"]["counts"], str)
        np.testing.assert_array_equal(coco_mask.decode(document["annotations"][1]["segmentation"]), self.binary[902])
        # A different, still valid mask may have its own digest and donor area.
        other = deepcopy(self.mask_document)
        binary = self.binary[701].copy()
        binary[5, 5] = 0
        encoded = coco_mask.encode(np.asfortranarray(binary))
        encoded["counts"] = encoded["counts"].decode("ascii")
        other["annotations"][0]["segmentation"] = encoded
        source = self.root / "other-rle.json"
        output = self.root / "other-prepared.json"
        write_json(source, other)
        second = masks.prepare_user_masks(source, self.gt1, self.images, output, expected_sha256=sha256(source))
        self.assertNotEqual(first["masks_sha256"], second["masks_sha256"])
        self.assertEqual(masks.verify_mask_annotations(output, expected_masks_sha256=second["masks_sha256"],
                         annotations=self.gt1, image_root=self.images)["origin"], "user_prepared_masks")

    def test_user_mask_image_id_class_crop_rle_and_sidecar_mismatches_are_rejected(self):
        for field, value in (("id", 999), ("category_id", 1), ("source_bbox", [9, 2, 10, 9]),
                             ("bbox", [2, 4, 6, 5]), ("segmentation", {"size": [64, 48], "counts": "bad"})):
            with self.subTest(field=field):
                changed = deepcopy(self.mask_document)
                changed["annotations"][0][field] = value
                source = self.root / f"bad-{field}.json"
                write_json(source, changed)
                with self.assertRaises((ValueError, TypeError)):
                    masks.prepare_user_masks(source, self.gt1, self.images, self.root / f"bad-out-{field}.json",
                                              expected_sha256=sha256(source))
        identity = self.prepare_masks()
        sidecar = masks.manifest_path(self.prepared_masks)
        altered = json.loads(sidecar.read_text())
        altered["image_bindings"]["101"]["sha256"] = "0" * 64
        write_json(sidecar, altered)
        with self.assertRaisesRegex(ValueError, "binding"):
            masks.verify_mask_annotations(self.prepared_masks, expected_masks_sha256=identity["masks_sha256"], image_root=self.images)
        write_json(sidecar, identity)
        Image.new("RGB", (64, 48), "blue").save(self.root / "source-101.png")
        with self.assertRaisesRegex(ValueError, "binding"):
            masks.verify_mask_annotations(self.prepared_masks, expected_masks_sha256=identity["masks_sha256"], image_root=self.images)

    def test_bank_cli_uses_own_count_and_preserves_shared_paste_math_and_rng(self):
        with patch("sys.argv", ["prepare_sam", "prepare-masks", "--masks", str(self.raw_masks),
                "--masks-sha256", sha256(self.raw_masks), "--annotations", str(self.gt1),
                "--images", str(self.images), "--mask-output", str(self.prepared_masks)]), patch("sys.stdout", new_callable=io.StringIO):
            prepare_sam.main()
        package = self.root / "user-bank"
        with patch("sys.argv", ["prepare_sam", "build-bank", "--masks", str(self.prepared_masks),
                "--masks-sha256", sha256(self.prepared_masks), "--annotations", str(self.gt0),
                "--images", str(self.images), "--output-root", str(package)]), patch("sys.stdout", new_callable=io.StringIO):
            prepare_sam.main()
        bank_sha = sha256(package / "pipeline.json")
        local = self.root / "local-bank"
        with patch("sys.argv", ["prepare_sam", "import", "--bank", str(package), "--bank-sha256", bank_sha,
                "--masks", str(self.prepared_masks), "--masks-sha256", sha256(self.prepared_masks),
                "--mask-output", str(self.root / "imported-masks.json"), "--annotations", str(self.gt0),
                "--images", str(self.images), "--output-root", str(local)]), patch("sys.stdout", new_callable=io.StringIO):
            prepare_sam.main()
        recipe = yaml.safe_load((ROOT / "training/final_recipes.yaml").read_text())["detectors"]["tinyformer"]["sam_copy_paste"]
        config = banks.bank_settings({**recipe, "base_images": 2}, local, expected_sha256=bank_sha, image_root=self.images)
        self.assertEqual({key: config[key] for key in ("schedule", "seed", "class_weights", "scale_range", "max_gt_iou")},
                         {key: recipe[key] for key in ("schedule", "seed", "class_weights", "scale_range", "max_gt_iou")})
        self.assertIsNone(config["bank_objects"])
        plan = SAMCopyPastePlan(config, self.gt0, [101, 202])
        self.assertEqual(plan.stats["bank_objects"], 2)
        with patch("sys.argv", ["portable_sam_bank", "verify", "--bank", str(package),
                "--pipeline-sha256", bank_sha, "--annotations", str(self.gt0), "--images", str(self.images)]), patch("sys.stdout", new_callable=io.StringIO):
            banks.main()
        pipeline = json.loads((package / "pipeline.json").read_text())
        entry = next(row for filename in pipeline["object_bank"]["files"].values()
                     for row in json.loads((package / filename).read_text()))
        with Image.open(package / entry["rgba_path"]) as rgba:
            np.testing.assert_array_equal(np.asarray(rgba.getchannel("A")), self.binary[entry["annotation_id"]][2:11, 1:11] * 255)
        legacy = self.root / "legacy-bank"
        legacy.mkdir()
        for filename in pipeline["files"]:
            target = legacy / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(package / filename)
        write_json(legacy / "pipeline.json", {"schema_version": 2, "source": {"annotations": str(self.gt0)},
            "object_bank": {**pipeline["object_bank"], "mode": "sam3_rgba_only"}})
        original = SAMCopyPastePlan({**recipe, "artifact_root": str(legacy), "bank_pipeline_sha256": None,
                                    "bank_objects": 2, "base_images": 2}, self.gt0, [101, 202])
        plan.set_epoch(2)
        original.set_epoch(2)
        self.assertEqual(plan.stats, original.stats)
        self.assertEqual({key: [row["annotation_id"] for row in rows] for key, rows in plan._paste_plan.items()},
                         {key: [row["annotation_id"] for row in rows] for key, rows in original._paste_plan.items()})
        canvas = Image.new("RGB", (128, 96))
        left, left_targets = plan._paste_objects(canvas, [], plan._natural_pool, random.Random(7))
        right, right_targets = original._paste_objects(canvas, [], original._natural_pool, random.Random(7))
        self.assertEqual(left.tobytes(), right.tobytes())
        self.assertEqual(left_targets, right_targets)
        self.assertTrue(left_targets)

    def test_bank_rejects_wrong_pin_gt_source_image_and_rgba_bytes(self):
        package, identity = self.build_bank()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            banks.verify_portable_bank(package, expected_pipeline_sha256="0" * 64, annotations=self.gt0, image_root=self.images)
        changed_gt = self.root / "changed-gt.json"
        document = zero_based(deepcopy(self.gt))
        document["annotations"][0]["bbox"][0] += 1
        write_json(changed_gt, document)
        with self.assertRaisesRegex(ValueError, "training annotations"):
            banks.verify_portable_bank(package, expected_pipeline_sha256=identity["pipeline_sha256"], annotations=changed_gt, image_root=self.images)
        original_image = (self.root / "source-101.png").read_bytes()
        Image.new("RGB", (64, 48), "green").save(self.root / "source-101.png")
        with self.assertRaisesRegex(ValueError, "image bytes"):
            banks.verify_portable_bank(package, expected_pipeline_sha256=identity["pipeline_sha256"], annotations=self.gt0, image_root=self.images)
        (self.root / "source-101.png").write_bytes(original_image)
        pipeline = json.loads((package / "pipeline.json").read_text())
        class_name = pipeline["object_bank"]["files"]["23"]
        original_class, original_pipeline = (package / class_name).read_bytes(), (package / "pipeline.json").read_bytes()
        rows = json.loads(original_class)
        # An interior x translation preserves padding-relative alpha geometry;
        # changing the tight width makes the declared crop disagree with RGBA.
        rows[0]["mask_bbox"][2] += 1
        write_json(package / class_name, rows)
        pipeline["files"][class_name] = banks.file_identity(package / class_name)
        pipeline["artifact_sha256"][class_name] = pipeline["files"][class_name]["sha256"]
        write_json(package / "pipeline.json", pipeline)
        with self.assertRaisesRegex(ValueError, "alpha/crop"):
            banks.verify_portable_bank(package, expected_pipeline_sha256=sha256(package / "pipeline.json"), annotations=self.gt0, image_root=self.images)
        (package / class_name).write_bytes(original_class)
        (package / "pipeline.json").write_bytes(original_pipeline)
        path = package / next(name for name in pipeline["files"] if name.endswith(".png"))
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 or bytes"):
            banks.verify_portable_bank(package, expected_pipeline_sha256=identity["pipeline_sha256"], annotations=self.gt0, image_root=self.images)

    def test_rf_uses_actual_nonempty_subset_and_retains_explicit_target_exclusions(self):
        self.prepare_masks()
        recipe = yaml.safe_load((ROOT / "training/final_recipes.yaml").read_text())["detectors"]["rfdetr"]["sam_copy_paste"]
        settings = {**recipe, "mask_annotations": self.prepared_masks, "image_root": self.images}
        transform = ClassAwareCopyPaste(CopyPasteSettings.from_mapping(settings))
        self.assertEqual(transform.summary["donor_count"], 2)
        self.assertEqual(transform.summary["sampling"], recipe["sampling"])
        with self.assertRaisesRegex(ValueError, "no accepted masks"):
            ClassAwareCopyPaste(CopyPasteSettings.from_mapping({**settings, "target_classes": ["cigarette_pack", "plastic_bottle"], "class_weights": {}}))
        excluded = deepcopy(self.mask_document)
        for row in excluded["annotations"]:
            row["category_id"] = 20
        path = self.root / "excluded-only.json"
        write_json(path, excluded)
        with self.assertRaisesRegex(ValueError, "pool is empty"):
            ClassAwareCopyPaste(CopyPasteSettings.from_mapping({**settings, "mask_annotations": path}))

    def test_generated_dcr_records_use_own_manifest_and_reject_gt_binding_mismatches(self):
        proposals = self.root / "proposals.json"
        write_json(proposals, [
            {"image_id": image_id, "category_id": 1, "bbox": box, "score": score}
            for image_id in (101, 202) for box, score in (([40, 30, 6, 4], .9), ([30, 30, 5, 4], .2), ([1, 2, 10, 9], .8))])
        source = self.root / "own-records.jsonl"
        identity = records.generate_records(self.gt1, self.images, proposals, source)
        self.assertEqual(identity["origin"], "generated_from_proposals")
        self.assertNotEqual(identity["source_sha256"], records.FIXED_RECORDS_SHA256)
        with self.assertRaisesRegex(ValueError, "new source identity"):
            records.verify_records_manifest(source)
        target = self.root / "imported-records.jsonl"
        imported = records.import_records(source, self.images, target, expected_records_sha256=identity["records_sha256"],
                                           allow_new=True, annotations=self.gt1)
        self.assertEqual(source.read_bytes(), target.read_bytes())
        self.assertEqual(records.verify_records_manifest(target, allow_new=True, expected_records_sha256=imported["records_sha256"]), imported)
        with self.assertRaisesRegex(ValueError, "input contract"):
            records.verify_records_manifest(target, allow_new=True, expected_records_sha256="0" * 64)
        original_rows = [json.loads(line) for line in source.read_text().splitlines()]
        for key, value in (("image_id", 999), ("label", 0), ("box_xyxy", [2, 2, 11, 11])):
            with self.subTest(field=key):
                changed = deepcopy(original_rows)
                row = next(row for row in changed if row["source_type"] == "gt_original")
                row[key] = value
                bad = self.root / f"bad-records-{key}.jsonl"
                bad.write_text("".join(records.canonical(row) + "\n" for row in changed))
                write_json(records.manifest_path(bad), {**identity, "records_sha256": sha256(bad)})
                with self.assertRaisesRegex(ValueError, "GT|bbox"):
                    records.import_records(bad, self.images, self.root / f"bad-import-{key}.jsonl",
                        expected_records_sha256=sha256(bad), allow_new=True, annotations=self.gt1)


if __name__ == "__main__":
    unittest.main()
