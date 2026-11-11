from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image
import torch
import yaml

from training import models, train_dcr, train_detector
from training.sam_copy_paste import SAMCopyPastePlan


class TrainingContractTest(unittest.TestCase):
    def test_sampling_keeps_fixed_foreground_and_background_counts(self):
        records = [train_dcr.Record(Path("image.png"), (0, 0, 5, 5), label, "gt_original", label) for label in range(34)]
        records += [train_dcr.Record(Path("image.png"), (0, 0, 5, 5), 34, source, None) for source in ("background_hard", "background_regular")]
        pools = train_dcr.sampling_pools(records)
        selected = train_dcr.schedule(records, pools, 192, 192, 20260919, 101)
        labels = [records[index].label for index, _ in selected]
        self.assertEqual(sum(label < 34 for label in labels), 128)
        self.assertEqual(labels.count(34), 64)
        self.assertEqual(set(labels), set(range(35)))
        self.assertEqual(sum(records[index].source == "background_hard" for index, _ in selected), 48)

    def test_checked_in_class_contract_can_be_loaded(self):
        data = train_detector.load_data(Path("training/data.example.yaml"))
        self.assertEqual(len(data["class_names"]), 34)

    def test_record_paths_cannot_escape_or_be_absolute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (10, 10)).save(root / "outside.png")
            Image.new("RGB", (10, 10)).save(images / "inside.png")
            records = root / "records.jsonl"
            row = {"image_path": "../outside.png", "box_xyxy": [0, 0, 5, 5], "label": 0, "source_type": "gt_original", "gt_instance_id": "one"}
            for path in ("../outside.png", str(images / "inside.png")):
                row["image_path"] = path
                records.write_text(json.dumps(row) + "\n")
                with self.assertRaises(ValueError):
                    train_dcr.load_records(records, images)

    def test_training_crop_uses_box_width_and_height(self):
        recipe = train_dcr.load_recipe(Path("training/final_recipes.yaml"), "v7")
        recipe["input_size"] = 32
        recipe["context_scale"] = 1.0
        recipe["mean"] = [0.0, 0.0, 0.0]
        recipe["std"] = [1.0, 1.0, 1.0]
        recipe["augmentation"] = {
            "translation_fraction": 0.0, "scale_range": [1.0, 1.0],
            "horizontal_flip_probability": 0.0, "rotation_degrees": 0.0,
            "brightness_range": [1.0, 1.0], "contrast_range": [1.0, 1.0],
            "saturation_range": [1.0, 1.0], "blur_probability": 0.0,
            "jpeg_probability": 0.0,
        }
        image = Image.new("RGB", (80, 80), "blue")
        image.paste("red", (20, 20, 40, 40))
        pixels, valid = train_dcr.render(image, [20, 20, 40, 40], recipe, random.Random(1))
        self.assertGreater(float(pixels[0].mean()), 0.98)
        self.assertLess(float(pixels[2].mean()), 0.02)
        self.assertTrue(bool(valid.all()))

    def test_lora_merge_preserves_native_masked_linear(self):
        class MaskedLinear(torch.nn.Linear):
            def forward(self, values):
                return torch.nn.functional.linear(values, self.weight, self.bias * 0.5)

        torch.manual_seed(7)
        wrapped = models.LoRALinear(MaskedLinear(4, 6), rank=2, alpha=2.0, dropout=0.0).eval()
        with torch.no_grad():
            wrapped.lora_B.normal_(0.0, 0.02)
        values = torch.randn(3, 4)
        expected = wrapped(values)
        merged = wrapped.merge_into_base()
        torch.testing.assert_close(merged(values), expected)

    def test_lora_updates_do_not_train_the_base(self):
        wrapped = models.LoRALinear(torch.nn.Linear(4, 6), rank=2, alpha=2.0, dropout=0.0)
        wrapped(torch.ones(3, 4)).square().mean().backward()
        self.assertIsNone(wrapped.base.weight.grad)
        self.assertIsNone(wrapped.base.bias.grad)
        self.assertGreater(float(wrapped.lora_B.grad.abs().sum()), 0.0)

    def test_hplus_rejects_a_crop_with_only_padding(self):
        recipe = train_dcr.load_recipe(Path("training/final_recipes.yaml"), "hplus")
        recipe["input_size"] = 32
        with self.assertRaisesRegex(ValueError, "no valid image pixels"):
            train_dcr.render(Image.new("RGB", (10, 10)), [100, 100, 102, 102], recipe, random.Random(1), augment=False)

    def test_shared_sam_plan_preserves_existing_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.json"
            annotations.write_text("{}")
            Image.new("RGBA", (8, 8), (0, 255, 0, 255)).save(root / "object.png")
            entry = {"annotation_id": 1, "class_id": 23, "rgba_path": "object.png"}
            (root / "class.json").write_text(json.dumps([entry]))
            (root / "pipeline.json").write_text(json.dumps({"schema_version": 1, "source": {"annotations": str(annotations)}, "object_bank": {"mode": "sam3_rgba_only", "files": {"23": "class.json"}}}))
            recipe = yaml.safe_load(Path("training/final_recipes.yaml").read_text())["detectors"]["tinyformer"]["sam_copy_paste"]
            config = {**recipe, "artifact_root": str(root), "bank_pipeline_sha256": None, "base_images": 2, "bank_objects": 1}
            plan = SAMCopyPastePlan(config, annotations, [1, 2])
            original = [{"bbox": [0.0, 0.0, 5.0, 5.0], "category_id": 0}]
            image, rows = plan._paste_objects(Image.new("RGB", (64, 64)), original, [entry], random.Random(2))
            self.assertEqual(rows[0], original[0])
            self.assertEqual(len(original), 1)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["category_id"], 23)
            self.assertGreater(rows[1]["area"], 0.0)
            plan.set_epoch(16)
            self.assertEqual(plan._paste_plan, {})


if __name__ == "__main__":
    unittest.main()
