from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training import prepare_sam
from training import prepare_sam_masks as masks
from training.prepare_data import sha256


class SAMMaskMigrationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-sam-masks-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.json"
        self.output = self.root / "portable.json"
        self.document = {
            "licenses": [{"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}],
            "images": [{"id": 9, "file_name": "second.jpg", "width": 20, "height": 10, "license": 1},
                       {"id": 4, "file_name": "first.jpg", "width": 12, "height": 8, "license": 1}],
            "categories": [{"id": 2, "name": "second"}, {"id": 1, "name": "first"}],
            "info": {"description": "Masks", "contributor": "NAMR", "version": "1.0",
                     "sam_pseudo_masks": {"model": "sam2_sam3_ensemble", "accepted_only": True},
                     "source_annotations": "/workspace/training-source/annotations.json"},
            "annotations": [{"id": 18, "image_id": 9, "category_id": 2, "bbox": [1.0, 2.5, 3.0, 4.0],
                             "area": 7, "iscrowd": 0, "source_bbox": [1.25, 2.25, 4.5, 4.25],
                             "segmentation": {"size": [10, 20], "counts": "a1RLE0"},
                             "sam": {"model": "sam3", "score": .875},
                             "ensemble": {"selected_model": "sam3", "candidates": {"sam2": {"score": .75}}}},
                            {"id": 7, "image_id": 4, "category_id": 1, "bbox": [1, 1, 3, 3],
                             "area": 9, "iscrowd": 0, "segmentation": {"size": [8, 12], "counts": "b2RLE1"}}],
        }
        self.write_source()

    def write_source(self):
        self.source.write_text(json.dumps(self.document, separators=(",", ":")), encoding="utf-8")

    def migrate(self):
        return masks.migrate_masks(self.source, self.output, expected_source_sha256=sha256(self.source))

    def verify(self, identity):
        return masks.verify_mask_annotations(
            self.output, expected_source_sha256=sha256(self.source), expected_source_bytes=self.source.stat().st_size,
            expected_portable_sha256=identity["masks_sha256"], expected_portable_bytes=identity["masks_bytes"],
        )

    def test_migration_preserves_all_fields_and_order_except_source_metadata(self):
        with self.assertRaisesRegex(ValueError, "source SAM masks SHA256"):
            masks.migrate_masks(self.source, self.output)
        self.assertFalse(self.output.exists())
        identity = self.migrate()
        expected = deepcopy(self.document)
        del expected["info"]["source_annotations"]
        portable = json.loads(self.output.read_text())
        self.assertEqual(portable, expected)
        self.assertEqual(list(portable), list(expected))
        self.assertEqual(list(portable["info"]), list(expected["info"]))
        self.assertEqual([row["id"] for row in portable["images"]], [9, 4])
        self.assertEqual([row["id"] for row in portable["annotations"]], [18, 7])
        self.assertEqual([row["id"] for row in portable["categories"]], [2, 1])
        self.assertEqual(identity["removed_metadata_fields"], ["info.source_annotations"])
        self.assertEqual(self.verify(identity), identity)
        with self.assertRaises(FileExistsError):
            self.migrate()

    def test_contract_accepts_original_and_rejects_self_signed_portable(self):
        identity = masks.verify_mask_annotations(
            self.source, expected_source_sha256=sha256(self.source), expected_source_bytes=self.source.stat().st_size,
        )
        self.assertEqual(identity["origin"], "original_masks")
        self.migrate()
        with self.assertRaisesRegex(ValueError, "contract"):
            masks.verify_mask_annotations(self.output)

    def test_portable_contract_rejects_changed_source_binding_and_content(self):
        identity = self.migrate()
        sidecar = masks.manifest_path(self.output)
        changed = {**identity, "source_sha256": "0" * 64}
        sidecar.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "preparation manifest"):
            self.verify(identity)
        sidecar.write_text(json.dumps(identity))
        document = json.loads(self.output.read_text())
        document["annotations"][0]["area"] += 1
        self.output.write_text(json.dumps(document))
        sidecar.write_text(json.dumps({**identity, "masks_sha256": sha256(self.output), "masks_bytes": self.output.stat().st_size}))
        with self.assertRaisesRegex(ValueError, "contract"):
            self.verify(identity)

    def test_unexpected_private_paths_are_rejected_without_rewriting_fields(self):
        self.document["images"][0]["debug_path"] = "/work/private/image.jpg"
        self.write_source()
        with self.assertRaisesRegex(ValueError, "private machine path"):
            self.migrate()
        self.assertFalse(self.output.exists())
        self.assertFalse(masks.manifest_path(self.output).exists())
        self.assertIn("source_annotations", self.document["info"])

    def test_migrate_masks_cli_does_not_require_or_import_a_bank(self):
        arguments = ["prepare_sam", "migrate-masks", "--masks", str(self.source), "--mask-output", str(self.output)]
        with patch("sys.argv", arguments), patch.object(prepare_sam, "prepare_masks", return_value={"images": 2}) as prepare, patch.object(prepare_sam, "import_bank") as bank, patch("sys.stdout", new_callable=io.StringIO):
            prepare_sam.main()
        prepare.assert_called_once_with(self.source, self.output)
        bank.assert_not_called()


if __name__ == "__main__":
    unittest.main()
