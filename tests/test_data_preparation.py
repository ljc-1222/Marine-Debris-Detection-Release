from __future__ import annotations

import json
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from training import prepare_data as data
from training import prepare_dcr_records as dcr
from training import prepare_sam as sam


class DataPreparationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-data-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.image = self.images / "one.png"
        Image.new("RGB", (100, 100), (30, 80, 150)).save(self.image)
        self.categories = json.loads(Path("configs/categories.json").read_text())["categories"]
        self.document = {"images": [{"id": 7, "file_name": "one.png", "width": 100, "height": 100, "source_group": 0}], "annotations": [{"id": 9, "image_id": 7, "category_id": 1, "bbox": [20, 20, 10, 10], "area": 100, "iscrowd": 0}], "categories": self.categories}
        self.annotations = self.root / "annotations.json"
        data.write_json(self.annotations, self.document)

    def test_views_preserve_gt_ids_and_share_the_training_union(self):
        output = self.root / "data"
        preparation = data.prepare_views(self.document, {"one.png": self.image}, output, {"union": data.sha256(self.annotations)})
        coco = json.loads((output / "annotations/instances_train_all.json").read_text())
        zero = json.loads((output / "annotations/instances_train_all_0based.json").read_text())
        self.assertEqual(coco["annotations"][0]["id"], 9)
        self.assertEqual(zero["annotations"][0]["category_id"], 0)
        self.assertEqual(zero["annotations"][0]["id"], 9)
        self.assertEqual(preparation["holdout_images"], 0)
        index = json.loads((output / "training.index.json").read_text())
        self.assertEqual(index, {"images": coco["images"], "categories": coco["categories"], "annotations": []})
        for split in ("train", "valid"):
            native = output / "rfdetr_train_all" / split
            self.assertEqual(json.loads((native / "_annotations.coco.json").read_text()), coco)
            self.assertTrue((native / "one.png").is_file())
        label = (output / "yolo_train_all/labels/train/one.txt").read_text().split()
        self.assertEqual(label[0], "0")
        self.assertEqual(list(map(float, label[1:])), [0.25, 0.25, 0.1, 0.1])
        self.assertTrue((output / "yolo_train_all/images/train/one.png").is_file())
        yolo = yaml.safe_load((output / "yolo_train_all.yaml").read_text())
        self.assertFalse(Path(yolo["path"]).is_absolute())
        self.assertEqual((output / yolo["path"] / yolo["train"]).resolve(), output / "yolo_train_all/images/train")

    def test_raw_and_prepared_project_can_move_without_original_absolute_paths(self):
        project = self.root / "project"
        raw = project / "raw"
        raw.mkdir(parents=True)
        image = raw / "one.png"
        Image.new("RGB", (100, 100), (30, 80, 150)).save(image)
        original_hash = data.sha256(image)
        data.prepare_views(self.document, {"one.png": image}, project / "data", {})
        moved = self.root / "moved-project"
        project.rename(moved)
        linked = moved / "data/images/train_all/one.png"
        self.assertFalse(linked.readlink().is_absolute())
        self.assertEqual(data.sha256(linked), original_hash)
        for storage in ("rfdetr_train_all/train", "rfdetr_train_all/valid", "yolo_train_all/images/train"):
            self.assertEqual(data.sha256(moved / "data" / storage / "one.png"), original_hash)
        native = yaml.safe_load((moved / "data/yolo_train_all.yaml").read_text())
        self.assertTrue((moved / "data" / native["path"] / native["train"] / "one.png").is_file())
        config = yaml.safe_load((moved / "data/data.yaml").read_text())
        self.assertEqual(config["coco"]["annotations_0based"], "annotations/instances_train_all_0based.json")
        with self.assertRaises(FileExistsError):
            data.prepare_views(self.document, {"one.png": self.image}, moved / "data", {})

    def _namr_fixture(self):
        root = self.root / "namr"
        images = root / "train/images"
        images.mkdir(parents=True)
        names = ["z_beach.png", "a_beach.png", "m_beach.png"]
        for name in names:
            Image.new("RGB", (100, 80), (30, 80, 150)).save(images / name)
        taxonomy = {int(row["id"]) - 1: row["name"] for row in self.categories}
        (root / "data_NAMR33.yaml").write_text(yaml.safe_dump({"nc": 34, "names": taxonomy}))
        with (root / "metadata_image_train.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_file_name", "image_width", "image_height"])
            writer.writeheader()
            writer.writerows({"image_file_name": name, "image_width": 100, "image_height": 80} for name in names)
        rows = [{"bounding_box_name": "z_beach_2_2", "bbox1": .75, "bbox2": .5, "bbox3": .1, "bbox4": .2, "namr33_class_id": 1, "namr33_class_name": taxonomy[1]},
                {"bounding_box_name": "m_beach_1_1", "bbox1": .5, "bbox2": .5, "bbox3": .2, "bbox4": .4, "namr33_class_id": 0, "namr33_class_name": taxonomy[0]},
                {"bounding_box_name": "z_beach_2_1", "bbox1": .5, "bbox2": .5, "bbox3": .2, "bbox4": .4, "namr33_class_id": 0, "namr33_class_name": taxonomy[0]},
                {"bounding_box_name": "a_beach_1_1", "bbox1": .5, "bbox2": .5, "bbox3": .2, "bbox4": .4, "namr33_class_id": 2, "namr33_class_name": taxonomy[2]}]
        with (root / "metadata_annotation_train.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        data.write_json(root / "image_order.json", {
            "schema_version": 1,
            "source_metadata_sha256": {name: data.sha256(root / name) for name in data.NAMR_SOURCE_SHA256},
            "image_row_order": [2, 0, 1], "training_group_sizes": [2, 1],
        })
        return root

    def test_namr_raw_metadata_preserves_all_images_fixed_ids_sequence_and_pixel_boxes(self):
        root = self._namr_fixture()
        union, files, sources = data.read_namr(root, check_release=False, id_protocol=root / "image_order.json")
        self.assertEqual([row["id"] for row in union["images"]], [1, 2, 3])
        self.assertEqual([row["file_name"] for row in union["images"]], ["m_beach.png", "z_beach.png", "a_beach.png"])
        self.assertEqual([row["source_group"] for row in union["images"]], [0, 0, 1])
        self.assertEqual([row["source_image_id"] for row in union["images"]], [1, 2, 1])
        self.assertEqual([row["source_annotation_id"] for row in union["annotations"]], [1, 2, 3, 1])
        self.assertEqual([row["id"] for row in union["annotations"]], [1, 2, 3, 4])
        self.assertTrue(all("source_split" not in row for key in ("images", "annotations") for row in union[key]))
        self.assertEqual({Path(row["file_name"]).name for row in union["images"]}, set(files))
        image = next(row for row in union["images"] if Path(row["file_name"]).name == "z_beach.png")
        boxes = [row for row in union["annotations"] if row["image_id"] == image["id"]]
        self.assertEqual([row["category_id"] for row in boxes], [1, 2])
        self.assertEqual(boxes[0]["bbox"], [40.0, 24.0, 20.0, 32.0])
        self.assertEqual(boxes[0]["area"], 640.0)
        self.assertEqual(len(sources["union"]), 64)
        output = self.root / "namr-prepared"
        data.prepare_views(union, files, output, sources)
        prepared = json.loads((output / "annotations/instances_train_all.json").read_text())
        self.assertEqual([row["id"] for row in prepared["annotations"]], [row["id"] for row in union["annotations"]])
        self.assertTrue(all(Path(row["file_name"]).name == row["file_name"] for row in prepared["images"]))

    def test_training_id_protocol_requires_complete_unique_rows_and_full_groups(self):
        root = self._namr_fixture()
        path = root / "image_order.json"
        original = json.loads(path.read_text())
        hashes = original["source_metadata_sha256"]
        for order in ([0, 0, 2], [0, 1], [0, 1, 3], [True, 1, 2]):
            with self.subTest(order=order):
                data.write_json(path, {**original, "image_row_order": order})
                with self.assertRaisesRegex(ValueError, "every metadata row exactly once"):
                    data.training_image_order(path, hashes, 3, check_release=False)
        for groups in ([2, 0], [1, 1], [3], [True, 2]):
            with self.subTest(groups=groups):
                data.write_json(path, {**original, "training_group_sizes": groups})
                with self.assertRaisesRegex(ValueError, "cover the complete dataset"):
                    data.training_image_order(path, hashes, 3, check_release=False)
        data.write_json(path, {**original, "source_metadata_sha256": {}})
        with self.assertRaisesRegex(ValueError, "bound to these source metadata"):
            data.training_image_order(path, hashes, 3, check_release=False)

    def test_fixed_training_id_protocol_rejects_a_changed_valid_permutation(self):
        root = self._namr_fixture()
        path = root / "image_order.json"
        protocol = json.loads(path.read_text())
        with patch.object(data, "NAMR_IMAGE_ORDER_SHA256", data.sha256(path)):
            data.training_image_order(path, protocol["source_metadata_sha256"], 3)
            data.write_json(path, {**protocol, "image_row_order": [0, 1, 2]})
            with self.assertRaisesRegex(ValueError, "fixed identity"):
                data.training_image_order(path, protocol["source_metadata_sha256"], 3)

    def test_full_dataset_preparation_rejects_a_missing_source_image(self):
        root = self._namr_fixture()
        (root / "train/images/a_beach.png").unlink()
        with self.assertRaisesRegex(ValueError, "existing files"):
            data.read_namr(root, check_release=False, id_protocol=root / "image_order.json")

    def test_namr_raw_metadata_rejects_changed_release_and_out_of_bounds_boxes(self):
        root = self._namr_fixture()
        with self.assertRaisesRegex(ValueError, "fixed v1.0 source"):
            data.read_namr(root)
        annotation = root / "metadata_annotation_train.csv"
        annotation.write_text(annotation.read_text().replace("0.75,0.5,0.1,0.2", "0.99,0.5,0.1,0.2"))
        with self.assertRaisesRegex(ValueError, "inside the image"):
            data.read_namr(root, check_release=False, id_protocol=root / "image_order.json")

    def test_records_migration_binds_source_and_preserves_crop_content(self):
        source = self.root / "source.jsonl"
        rows = [{"record_id": "first", "image_path": "/source/one.png", "image_sha256": data.sha256(self.image), "box_xyxy": [20, 20, 30, 30], "label": 0, "source_type": "gt_original", "gt_instance_id": 9, "split": "train"}, {"record_id": "second", "image_path": "/source/one.png", "image_sha256": data.sha256(self.image), "box_xyxy": [70, 70, 80, 80], "label": 34, "source_type": "background_hard", "gt_instance_id": None, "split": "train"}]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        output = self.root / "portable.jsonl"
        with self.assertRaisesRegex(ValueError, "source DCR records SHA256"):
            dcr.migrate_records(source, self.images, output)
        self.assertFalse(output.exists())
        identity = dcr.migrate_records(source, self.images, output, expected_source_sha256=data.sha256(source))
        migrated = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([row["record_id"] for row in migrated], ["first", "second"])
        for original, portable in zip(rows, migrated):
            for field in ("box_xyxy", "label", "source_type", "gt_instance_id"):
                self.assertEqual(original[field], portable[field])
            self.assertEqual(portable["image_path"], "one.png")
            self.assertNotIn("split", portable)
        self.assertEqual(identity["metadata_fields_removed"], ["split"])
        with self.assertRaisesRegex(ValueError, "new source identity"):
            dcr.verify_records_manifest(output)
        self.assertEqual(dcr.verify_records_manifest(output, allow_new=True), identity)
        output.write_text(output.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "preparation manifest"):
            dcr.verify_records_manifest(output, allow_new=True)

    def test_record_migration_rejects_changed_images_and_escaping_paths(self):
        row = {"image_path": "one.png", "image_sha256": "0" * 64, "box_xyxy": [0, 0, 1, 1], "label": 0, "source_type": "gt_original", "gt_instance_id": 1, "split": "train"}
        with self.assertRaisesRegex(ValueError, "image bytes"):
            dcr.portable_row(row, self.images, {})
        row["image_path"] = "../one.png"
        with self.assertRaisesRegex(ValueError, "escape"):
            dcr.portable_row(row, self.images, {})

    def test_generated_records_use_gt_labels_and_conservative_backgrounds(self):
        document = {**self.document, "annotations": self.document["annotations"] + [{"id": 10, "image_id": 7, "category_id": 2, "bbox": [40, 20, 10, 10], "area": 100, "iscrowd": 0}]}
        data.write_json(self.annotations, document)
        proposals = self.root / "proposals.json"
        data.write_json(proposals, [{"image_id": 7, "category_id": 2, "bbox": [20, 20, 10, 10], "score": 0.9}, {"image_id": 7, "category_id": 2, "bbox": [40, 20, 10, 10], "score": 0.8}, {"image_id": 7, "category_id": 3, "bbox": [70, 70, 10, 10], "score": 0.7}, {"image_id": 7, "category_id": 3, "bbox": [85, 70, 10, 10], "score": 0.2}])
        output = self.root / "generated.jsonl"
        identity = dcr.generate_records(self.annotations, self.images, proposals, output)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        wrong = next(row for row in rows if row["source_type"] == "proposal_wrong_class")
        self.assertEqual(wrong["label"], 0)
        self.assertEqual(wrong["gt_instance_id"], 9)
        self.assertEqual({row["source_type"] for row in rows}, {"gt_original", "gt_jitter", "proposal_matched", "proposal_wrong_class", "background_hard", "background_regular"})
        self.assertTrue(all("split" not in row for row in rows))
        self.assertEqual(identity["origin"], "generated_from_proposals")
        with self.assertRaisesRegex(ValueError, "new source identity"):
            dcr.verify_records_manifest(output)
        self.assertFalse(dcr.clean_background([30, 20, 40, 30], [[20, 20, 30, 30]]))

    def _sam_import_fixture(self):
        zero_path = self.root / "zero.json"
        document = data.zero_based(self.document)
        document["annotations"].append({**document["annotations"][0], "id": 10, "bbox": [40, 20, 10, 10]})
        data.write_json(zero_path, document)
        source = self.root / "bank"
        rows = []
        for annotation in reversed(document["annotations"]):
            cache = Path("rgba_cache/class_00") / f"annotation_{annotation['id']:08d}.webp"
            (source / cache).parent.mkdir(parents=True, exist_ok=True)
            rgba = Image.new("RGBA", (12, 11), (30, 80, 150, 255))
            rgba.putpixel((0, 0), (0, 0, 0, 0))
            rgba.save(source / cache, format="WEBP", quality=95)
            rows.append({"annotation_id": annotation["id"], "image_id": 7, "class_id": 0, "source_path": "one.png", "source_bbox": annotation["bbox"], "rgba_path": cache.as_posix(), "cached_size": [12, 11], "cache_bytes": (source / cache).stat().st_size})
        manifest = Path("object_bank/class_00.json")
        data.write_json(source / manifest, rows)
        pipeline = {"schema_version": 2, "source": {"annotations_sha256": data.sha256(zero_path)}, "object_bank": {"mode": "sam3_rgba_only", "files": {"0": manifest.as_posix()}, "class_counts": {"0": 2}, "cache": {"count": 2, "bytes": sum(row["cache_bytes"] for row in rows)}}, "artifact_sha256": {manifest.as_posix(): data.sha256(source / manifest)}}
        data.write_json(source / "pipeline.json", pipeline)
        return source, zero_path, manifest, rows

    def test_sam_import_preserves_donor_order_and_original_webp_bytes(self):
        source, zero_path, manifest, rows = self._sam_import_fixture()
        moved = self.root / "imported-bank"
        imported = sam.import_bank(source, zero_path, moved, expected_pipeline_sha256=data.sha256(source / "pipeline.json"))
        self.assertEqual(imported["object_bank"]["objects"], 2)
        self.assertEqual((moved / manifest).read_bytes(), (source / manifest).read_bytes())
        self.assertEqual([row["annotation_id"] for row in json.loads((moved / manifest).read_text())], [10, 9])
        for row in rows:
            linked = moved / row["rgba_path"]
            self.assertTrue(linked.is_symlink())
            self.assertEqual(linked.resolve(), (source / row["rgba_path"]).resolve())
            self.assertEqual(linked.read_bytes(), (source / row["rgba_path"]).read_bytes())

    def test_sam_import_rejects_changed_class_manifest(self):
        source, zero_path, manifest, rows = self._sam_import_fixture()
        data.write_json(source / manifest, list(reversed(rows)))
        with self.assertRaisesRegex(ValueError, "class manifest SHA256"):
            sam.import_bank(source, zero_path, self.root / "wrong-bank", expected_pipeline_sha256=data.sha256(source / "pipeline.json"))

    def test_sam_import_rejects_mismatched_gt_source(self):
        source, zero_path, _manifest, _rows = self._sam_import_fixture()
        original = json.loads(zero_path.read_text())
        for failure in ("image IDs", "source filenames", "source boxes"):
            with self.subTest(failure=failure):
                document = json.loads(json.dumps(original))
                if failure == "image IDs":
                    document["annotations"][0]["image_id"] = 8
                elif failure == "source filenames":
                    document["images"][0]["file_name"] = "wrong.png"
                else:
                    document["annotations"][0]["bbox"][0] += 1
                data.write_json(zero_path, document)
                output = self.root / "wrong-bank"
                with self.assertRaisesRegex(ValueError, failure):
                    sam.import_bank(source, zero_path, output, expected_pipeline_sha256=data.sha256(source / "pipeline.json"))
                self.assertFalse(output.exists())
                self.assertFalse(output.with_name(output.name + ".preparing").exists())

    def test_sam_import_rejects_changed_cache_bytes(self):
        source, zero_path, _manifest, rows = self._sam_import_fixture()
        with (source / rows[0]["rgba_path"]).open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "cache bytes"):
            sam.import_bank(source, zero_path, self.root / "wrong-bank", expected_pipeline_sha256=data.sha256(source / "pipeline.json"))

    def test_sam_import_rejects_changed_cache_dimensions(self):
        source, zero_path, manifest, rows = self._sam_import_fixture()
        rows[0]["cached_size"] = [99, 99]
        data.write_json(source / manifest, rows)
        pipeline = json.loads((source / "pipeline.json").read_text())
        pipeline["artifact_sha256"][manifest.as_posix()] = data.sha256(source / manifest)
        data.write_json(source / "pipeline.json", pipeline)
        with self.assertRaisesRegex(ValueError, "cache dimensions"):
            sam.import_bank(source, zero_path, self.root / "wrong-bank", expected_pipeline_sha256=data.sha256(source / "pipeline.json"))


if __name__ == "__main__":
    unittest.main()
