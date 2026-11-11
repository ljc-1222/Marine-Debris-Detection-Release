from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from inference import check_classifier as check


class CropClassifier(nn.Module):
    def forward(self, pixels, valid=None):
        self.seen_pixels = pixels.detach().cpu()
        self.seen_valid = valid.detach().cpu() if valid is not None else None
        logits = pixels.new_zeros((len(pixels), 34))
        logits[:, 2] = pixels[:, 0].mean(dim=(1, 2))
        return logits, pixels.mean(dim=(1, 2, 3))


class ClassifierCheckTest(unittest.TestCase):
    def test_xywh_geometry_uses_actual_image_and_keeps_subpixel_box_size(self):
        image = Image.new("RGB", (80, 80), "blue")
        image.paste("red", (20, 20, 40, 40))
        config = {**check.load_model_config("v7"), "input_size": 32, "context_scale": 1.0, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        pixels, valid = check.render_input(image, (20, 20, 20, 20), config)
        self.assertGreater(float(pixels[0].mean()), 0.98)
        self.assertLess(float(pixels[2].mean()), 0.02)
        self.assertTrue(bool(valid.all()))
        pixels, _ = check.render_input(image, (20.8, 20.8, 0.1, 0.1), config)
        self.assertEqual(tuple(pixels.shape), (3, 32, 32))
        self.assertEqual(check.crop_rectangle((20.8, 20.8, 0.1, 0.1), 1.0), (20, 20, 21, 21))

    def test_padding_only_and_invalid_boxes_are_rejected_for_all_heads(self):
        image = Image.new("RGB", (10, 10), "blue")
        for name in ("hplus", "v7", "siglip"):
            for box in ((100, 100, 2, 2), (0, 0, 0, 2), (0, 0, 2, -1), (math.nan, 0, 2, 2)):
                with self.subTest(model=name, box=box), self.assertRaises(ValueError):
                    check.render_input(image, box, check.load_model_config(name))

    def test_real_crop_reaches_model_and_produces_class_and_objectness_json(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-classifier-check-") as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (48, 32), "red").save(image_path)
            for name, size in (("hplus", 448), ("v7", 384), ("siglip", 384)):
                with self.subTest(model=name):
                    model = CropClassifier().eval()
                    with patch.object(check, "load_classifier", return_value=model):
                        result = check.check(name, Path(directory) / "model.pt", image_path, (3, 4, 20, 16), device="cpu")
                    self.assertEqual(result["input_shape"], [1, 3, size, size])
                    self.assertEqual(result["class_logits_shape"], [1, 34])
                    self.assertEqual(result["objectness_logits_shape"], [1])
                    self.assertEqual(len(result["foreground_probabilities"]), 34)
                    self.assertAlmostEqual(sum(result["foreground_probabilities"]), 1.0, places=6)
                    self.assertTrue(0.0 < result["objectness_probability"] < 1.0)
                    self.assertTrue(bool(model.seen_pixels.any()))
                    self.assertEqual(model.seen_valid is None, name == "siglip")

    def test_wrong_shape_and_non_finite_model_outputs_are_rejected(self):
        class BadClassifier(nn.Module):
            def __init__(self, wrong_shape):
                super().__init__()
                self.wrong_shape = wrong_shape

            def forward(self, pixels):
                count = 33 if self.wrong_shape else 34
                return torch.full((1, count), 0.0 if self.wrong_shape else math.nan), torch.zeros(1)

        for wrong_shape in (False, True):
            with self.subTest(wrong_shape=wrong_shape), self.assertRaises(ValueError):
                check.predict_crop(BadClassifier(wrong_shape), torch.ones(3, 32, 32), None, "cpu")


if __name__ == "__main__":
    unittest.main()
