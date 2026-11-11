from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from training import prepare_dcr_records as dcr


class PortableRecordsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-records-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "relocated-images"
        self.images.mkdir()
        for name, color in (("one.png", "red"), ("two.png", "blue")):
            Image.new("RGB", (20, 20), color).save(self.images / name)
        self.rows = [
            {"record_id": "second", "image_path": "two.png", "image_sha256": dcr.sha256(self.images / "two.png"),
             "box_xyxy": [-0.0, 1, 12.125, 18], "label": 34, "gt_instance_id": None,
             "source_type": "background_hard", "hard": True},
            {"record_id": "first", "image_path": "one.png", "image_sha256": dcr.sha256(self.images / "one.png"),
             "box_xyxy": [1, 2.5, 12, 18.75], "label": 0, "gt_instance_id": 7,
             "source_type": "gt_original", "hard": False},
        ]
        self.source = self.root / "download.records.jsonl"
        self.output = self.root / "imported/train_all.records.jsonl"

    def write_source(self, *, identity_changes=None):
        self.source.write_text("".join(dcr.canonical(row) + "\n" for row in self.rows))
        identity = {"schema_version": 1, "origin": "fixed_record_migration",
                    "source_sha256": dcr.FIXED_RECORDS_SHA256, "rows": len(self.rows),
                    "records_sha256": dcr.sha256(self.source), **(identity_changes or {})}
        dcr.manifest_path(self.source).write_text(json.dumps(identity))
        return identity

    def run_import(self, **kwargs):
        return dcr.import_records(self.source, self.images, self.output,
                                  expected_records_sha256=dcr.sha256(self.source), **kwargs)

    def assert_no_output(self):
        self.assertFalse(self.output.exists())
        self.assertFalse(dcr.manifest_path(self.output).exists())
        self.assertFalse(self.output.with_suffix(self.output.suffix + ".partial").exists())

    def test_clean_portable_import_preserves_bytes_order_values_and_source_binding(self):
        source_identity = self.write_source()
        identity = self.run_import()
        self.assertEqual(self.output.read_bytes(), self.source.read_bytes())
        self.assertEqual(identity["source_sha256"], dcr.FIXED_RECORDS_SHA256)
        self.assertEqual(identity["portable_source_sha256"], source_identity["records_sha256"])
        self.assertEqual(identity["records_sha256"], source_identity["records_sha256"])
        self.assertEqual(identity["images_verified"], 2)
        self.assertEqual(dcr.verify_records_manifest(self.output), identity)
        self.assertNotIn(str(self.images), dcr.manifest_path(self.output).read_text())
        self.assertTrue(all("split" not in row for row in map(json.loads, self.output.read_text().splitlines())))
        with self.assertRaises(FileExistsError):
            self.run_import()

    def test_legacy_portable_import_only_removes_unused_split_metadata(self):
        for row in self.rows:
            row["split"] = "unused-source-group"
        self.write_source()
        self.run_import()
        actual = [json.loads(line) for line in self.output.read_text().splitlines()]
        expected = [{key: value for key, value in row.items() if key != "split"} for row in self.rows]
        self.assertEqual(actual, expected)
        for left, right in zip(expected, actual):
            self.assertEqual([type(value) for value in left["box_xyxy"]],
                             [type(value) for value in right["box_xyxy"]])
        self.assertEqual([row["record_id"] for row in actual], ["second", "first"])

    def test_import_rehashes_every_referenced_image_and_cleans_failed_outputs(self):
        self.write_source()
        Image.new("RGB", (20, 20), "green").save(self.images / "one.png")
        with self.assertRaisesRegex(ValueError, "image bytes"):
            self.run_import()
        self.assert_no_output()

    def test_release_digest_sidecar_origin_and_declared_rows_are_checked(self):
        self.write_source()
        with self.assertRaisesRegex(ValueError, "release identity"):
            dcr.import_records(self.source, self.images, self.output, expected_records_sha256="0" * 64)
        self.assert_no_output()
        self.write_source(identity_changes={"source_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "new source identity"):
            self.run_import()
        self.assert_no_output()
        self.write_source(identity_changes={"rows": 3})
        with self.assertRaisesRegex(ValueError, "count"):
            self.run_import()
        self.assert_no_output()
        self.write_source()
        self.source.write_text(self.source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "preparation manifest"):
            self.run_import()

    def test_import_requires_portable_paths_and_explicit_image_hashes(self):
        for path in ("../one.png", str(self.images / "one.png"), "folder\\one.png"):
            with self.subTest(path=path):
                self.rows[0]["image_path"] = path
                self.write_source()
                with self.assertRaisesRegex(ValueError, "relative image paths"):
                    self.run_import()
                self.assert_no_output()
        self.rows[0]["image_path"] = "two.png"
        self.rows[0].pop("image_sha256")
        self.write_source()
        with self.assertRaisesRegex(ValueError, "image SHA256"):
            self.run_import()
        self.assert_no_output()

    def test_import_cli_accepts_a_separately_named_sidecar(self):
        self.write_source()
        sidecar = self.root / "download-sidecar.json"
        dcr.manifest_path(self.source).rename(sidecar)
        args = ["import", "--records", str(self.source), "--manifest", str(sidecar),
                "--records-sha256", dcr.sha256(self.source), "--images", str(self.images),
                "--output", str(self.output)]
        dcr.main(args)
        self.assertEqual(dcr.verify_records_manifest(self.output)["rows"], 2)
        with self.assertRaises(SystemExit):
            dcr.main(["import", "--records", str(self.source), "--images", str(self.images), "--output", str(self.output)])


if __name__ == "__main__":
    unittest.main()
