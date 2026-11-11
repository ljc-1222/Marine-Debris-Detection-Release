from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from training import portable_sam_bank as portable
from training.prepare_data import sha256, write_json
from training.prepare_sam import import_bank
from training.sam_copy_paste import SAMCopyPastePlan

ROOT = Path(__file__).resolve().parents[1]


class PortableSAMBankTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-portable-sam-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "original"
        self.source.mkdir()
        self.annotations = self.root / "annotations.json"
        categories = json.loads((ROOT / "configs/categories.json").read_text())["categories"]
        self.document = {"images": [{"id": 9, "file_name": "one.jpg", "width": 32, "height": 24},
                                    {"id": 11, "file_name": "two.jpg", "width": 32, "height": 24}],
            "categories": [{"id": index, "name": row["name"]} for index, row in enumerate(categories)],
            "annotations": [{"id": aid, "image_id": image, "category_id": category,
                             "bbox": [1.25, 2.5, 4.0, 3.0], "area": 12.0, "iscrowd": 0}
                            for aid, image, category in ((3, 11, 23), (1, 9, 23), (2, 11, 0))]}
        write_json(self.annotations, self.document)
        self.metadata = self.root / "metadata_image_train.csv"
        with self.metadata.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_file_name", "provider_account", "provider_account_zh", "capture_gps", "device_model"])
            writer.writeheader()
            writer.writerows([
                {"image_file_name": "one.jpg", "provider_account": "provider_one", "provider_account_zh": "影像提供者甲", "capture_gps": "UNNEEDED-GPS", "device_model": "UNNEEDED-DEVICE"},
                {"image_file_name": "two.jpg", "provider_account": "provider_two", "provider_account_zh": "影像提供者乙", "capture_gps": "UNNEEDED-GPS", "device_model": "UNNEEDED-DEVICE"},
            ])
        rows_by_class = {"23": [], "0": []}
        self.hashes = {}
        for annotation in self.document["annotations"]:
            aid, category = annotation["id"], annotation["category_id"]
            relative = f"rgba_cache/class_{category:02d}/annotation_{aid:08d}.webp"
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGBA", (7, 6), (13, 41, 79, 0))
            image.paste((20 * aid, 180, 45, 255), (1, 1, 6, 5))
            image.save(path, format="WEBP", lossless=True, method=4)
            self.hashes[relative] = sha256(path)
            rows_by_class[str(category)].append({
                "annotation_id": aid, "image_id": annotation["image_id"], "class_id": category,
                "source_path": "/legacy-fixture/private/" + ("one.jpg" if annotation["image_id"] == 9 else "two.jpg"),
                "source_bbox": annotation["bbox"], "mask_bbox": [2.0, 3.0, 5.0, 4.0],
                "mask_area": 20, "rgba_path": relative, "cached_size": [7, 6],
                "cache_bytes": path.stat().st_size, "sam_score": 0.95, "sam_method": "historical-run"})
        files, artifact_hashes = {}, {}
        for category, rows in rows_by_class.items():
            filename = f"object_bank/class_{int(category):02d}.json"
            write_json(self.source / filename, rows)
            files[category] = filename
            artifact_hashes[filename] = sha256(self.source / filename)
        write_json(self.source / "pipeline.json", {"schema_version": 2,
            "source": {"annotations": "/legacy-fixture/annotations.json"},
            "object_bank": {"mode": "sam3_rgba_only", "files": files, "objects": 3,
                            "class_counts": {key: len(rows) for key, rows in rows_by_class.items()},
                            "cache": {"count": 3, "rgb_quality": 95, "transparent_padding": 2}},
            "artifact_sha256": artifact_hashes})
        self.source_sha = sha256(self.source / "pipeline.json")
        self.hash_path = self.root / "cache-hashes.json"
        write_json(self.hash_path, self.hashes)
        self.package = self.root / "package"

    def package_bank(self, output=None):
        return portable.package_fixed_bank(self.source, self.annotations, self.metadata, self.hash_path,
            output or self.package, expected_source_sha256=self.source_sha,
            expected_metadata_sha256=sha256(self.metadata))

    def verify(self, identity, bank=None):
        return portable.verify_portable_bank(bank or self.package,
            expected_pipeline_sha256=identity["pipeline_sha256"], expected_source_sha256=self.source_sha)

    def contract(self, identity):
        return {"schema_version": 1, "historical_pipeline_sha256": self.source_sha,
                "package_pipeline_sha256": identity["pipeline_sha256"], "package_pipeline_bytes": identity["pipeline_bytes"]}

    def test_package_preserves_exact_webp_and_donor_order_with_only_official_credit_fields(self):
        identity = self.package_bank()
        pipeline = self.verify(identity)
        self.assertEqual(list(pipeline["object_bank"]["files"]), ["23", "0"])
        ids = []
        for filename in pipeline["object_bank"]["files"].values():
            for row in portable.load_json(self.package / filename):
                self.assertEqual(tuple(row), portable.ROW_FIELDS)
                self.assertEqual((self.package / row["rgba_path"]).read_bytes(), (self.source / row["rgba_path"]).read_bytes())
                ids.append(row["annotation_id"])
        self.assertEqual(ids, [3, 1, 2])
        credits = portable.load_json(self.package / "ATTRIBUTION.json")["donors"]
        self.assertEqual([row["annotation_id"] for row in credits], ids)
        self.assertEqual([row["provider_account"] for row in credits], ["provider_two", "provider_one", "provider_two"])
        self.assertEqual(credits[1]["provider_account_zh"], "影像提供者甲")
        for path in self.package.rglob("*.json"):
            text = path.read_text()
            for unwanted in ("/home/", str(self.root), "UNNEEDED-GPS", "UNNEEDED-DEVICE", "sam_score", "historical-run"):
                self.assertNotIn(unwanted, text)
        self.assertEqual(pipeline["remote"], {"repo_id": None, "revision": None})
        repeated = self.root / "repeated-package"
        self.assertEqual(self.package_bank(repeated), identity)
        self.assertEqual((repeated / "pipeline.json").read_bytes(), (self.package / "pipeline.json").read_bytes())

    def test_package_requires_source_pipeline_metadata_and_exact_donor_hash_receipt(self):
        with self.assertRaisesRegex(ValueError, "source pipeline"):
            portable.package_fixed_bank(self.source, self.annotations, self.metadata, self.hash_path, self.package)
        self.assertFalse(self.package.exists())
        hashes = {**self.hashes, "unexpected.webp": "0" * 64}
        write_json(self.hash_path, hashes)
        with self.assertRaisesRegex(ValueError, "exactly the fixed donor set"):
            self.package_bank()
        self.assertFalse(self.package.exists())
        self.assertFalse(self.package.with_name(self.package.name + ".preparing").exists())

    def test_verifier_rejects_unpinned_self_signed_and_changed_payloads(self):
        identity = self.package_bank()
        with self.assertRaisesRegex(ValueError, "release contract"):
            portable.verify_portable_bank(self.package)
        relative = next(iter(self.hashes))
        path = self.package / relative
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 or bytes"):
            self.verify(identity)
        path.write_bytes(original)
        pipeline = portable.load_json(self.package / "pipeline.json")
        pipeline["remote"]["repo_id"] = "unpublished-fixture"
        write_json(self.package / "pipeline.json", pipeline)
        with self.assertRaisesRegex(ValueError, "pipeline SHA256"):
            self.verify(identity)

    def test_verifier_checks_donor_order_and_path_containment_even_with_supplied_manifest_hash(self):
        identity = self.package_bank()
        pipeline = portable.load_json(self.package / "pipeline.json")
        filename = pipeline["object_bank"]["files"]["23"]
        rows = portable.load_json(self.package / filename)
        write_json(self.package / filename, list(reversed(rows)))
        pipeline["files"][filename] = portable.file_identity(self.package / filename)
        pipeline["artifact_sha256"][filename] = pipeline["files"][filename]["sha256"]
        write_json(self.package / "pipeline.json", pipeline)
        with self.assertRaisesRegex(ValueError, "order"):
            self.verify({**identity, "pipeline_sha256": sha256(self.package / "pipeline.json")})
        pipeline["files"]["../outside"] = {"sha256": "0" * 64, "bytes": 0}
        write_json(self.package / "pipeline.json", pipeline)
        with self.assertRaisesRegex(ValueError, "relative"):
            self.verify({**identity, "pipeline_sha256": sha256(self.package / "pipeline.json")})

    def test_import_rebinds_annotations_and_shared_plan_preserves_paste_pixels_and_rng(self):
        identity = self.package_bank()
        local = self.root / "local"
        imported = portable.import_portable_bank(self.package, self.annotations, local,
            expected_pipeline_sha256=identity["pipeline_sha256"], expected_source_sha256=self.source_sha)
        self.assertEqual(imported["source"]["source_sha256"], self.source_sha)
        self.assertEqual(imported["source"]["package_pipeline_sha256"], identity["pipeline_sha256"])
        self.assertEqual((local / "portable-package.json").read_bytes(), (self.package / "pipeline.json").read_bytes())
        reference = self.root / "reference"
        import_bank(self.source, self.annotations, reference, expected_pipeline_sha256=self.source_sha)
        recipe = yaml.safe_load((ROOT / "training/final_recipes.yaml").read_text())["detectors"]["tinyformer"]["sam_copy_paste"]
        config = {**recipe, "bank_pipeline_sha256": self.source_sha, "base_images": 4,
                  "bank_objects": 3, "excluded_classes": [], "scale_range": [1.0, 1.0]}
        with patch.object(portable, "release_contract", return_value=self.contract(identity)):
            original = SAMCopyPastePlan({**config, "artifact_root": str(reference)}, self.annotations, [9, 11, 9, 11])
            moved = SAMCopyPastePlan({**config, "artifact_root": str(local)}, self.annotations, [9, 11, 9, 11])
        self.assertEqual([row["annotation_id"] for row in original._natural_pool], [3, 1, 2])
        self.assertEqual([row["annotation_id"] for row in moved._natural_pool], [3, 1, 2])
        original.set_epoch(2)
        moved.set_epoch(2)
        self.assertEqual({index: [row["annotation_id"] for row in rows] for index, rows in original._paste_plan.items()},
                         {index: [row["annotation_id"] for row in rows] for index, rows in moved._paste_plan.items()})
        canvas, gt = Image.new("RGB", (96, 64)), [{"bbox": [0.0, 0.0, 4.0, 4.0], "category_id": 0}]
        expected_image, expected_gt = original._paste_objects(canvas, gt, original._natural_pool, random.Random(42))
        actual_image, actual_gt = moved._paste_objects(canvas, gt, moved._natural_pool, random.Random(42))
        self.assertEqual(actual_image.tobytes(), expected_image.tobytes())
        self.assertEqual(actual_gt, expected_gt)
        self.assertGreater(len(actual_gt), len(gt))

    def test_loader_rejects_changed_payload_before_pil_and_import_rejects_wrong_annotation_content(self):
        identity = self.package_bank()
        local = self.root / "local"
        imported = portable.import_portable_bank(self.package, self.annotations, local,
            expected_pipeline_sha256=identity["pipeline_sha256"], expected_source_sha256=self.source_sha)
        bad = local / next(iter(self.hashes))
        bad.write_bytes(bad.read_bytes() + b"altered")
        with patch.object(portable, "release_contract", return_value=self.contract(identity)), patch("training.sam_copy_paste.Image.open") as decoder:
            with self.assertRaisesRegex(ValueError, "SHA256 or bytes"):
                SAMCopyPastePlan({"artifact_root": str(local)}, self.annotations, [9, 11])
            decoder.assert_not_called()
        wrong = self.root / "wrong-annotations.json"
        changed = json.loads(self.annotations.read_text())
        changed["annotations"][0]["bbox"][0] += 0.5
        write_json(wrong, changed)
        with self.assertRaisesRegex(ValueError, "canonical training annotations"):
            portable.import_portable_bank(self.package, wrong, self.root / "wrong-import",
                expected_pipeline_sha256=identity["pipeline_sha256"], expected_source_sha256=self.source_sha)
        self.assertFalse((self.root / "wrong-import").exists())


if __name__ == "__main__":
    unittest.main()
