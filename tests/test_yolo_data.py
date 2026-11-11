from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

from training import yolo_data
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import DATASET_CACHE_VERSION
from ultralytics.data.utils import get_hash, save_dataset_cache_file


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class YOLOStoredPixelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-yolo-data-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "data/images/train"
        self.labels = self.root / "data/labels/train"
        self.sources = self.root / "sources"
        for path in (self.images, self.labels, self.sources):
            path.mkdir(parents=True)
        self.data = {"names": {index: f"class_{index}" for index in range(34)}, "channels": 3}

    def fixture(self, orientation, *, name=None):
        name = name or f"orientation-{orientation}"
        source = self.sources / f"{name}.jpg"
        image = Image.new("RGB", (80, 48), (20, 30, 40))
        image.paste((240, 20, 20), (8, 12, 28, 32))
        image.paste((20, 220, 40), (64, 2, 76, 8))
        exif = Image.Exif()
        exif[274] = orientation
        image.save(source, quality=100, subsampling=0, exif=exif)
        image_path = self.images / source.name
        image_path.symlink_to(source)
        label = self.labels / f"{name}.txt"
        label.write_text("0 0.225000000 0.458333333 0.250000000 0.416666667\n", encoding="utf-8")
        return image_path, label, source

    def dataset(self, *, cache_root=None, **overrides):
        arguments = dict(
            img_path=str(self.images), imgsz=80, batch_size=2, augment=False,
            hyp=get_cfg(), rect=False, cache=False, stride=32, pad=0.0,
            task="detect", data=self.data,
            cache_root=cache_root or self.root / "output/dataset-cache/stored-pixels/train",
        )
        arguments.update(overrides)
        return yolo_data.StoredYOLODataset(**arguments)

    def test_all_exif_orientations_keep_stored_pixels_and_labels_through_letterbox(self):
        fixtures = [self.fixture(orientation) for orientation in range(1, 9)]
        source_hashes = {source: digest(source) for _image, _label, source in fixtures}
        dataset = self.dataset()
        baseline_pixels = baseline_tensor = None
        for index, label in enumerate(dataset.labels):
            with self.subTest(image=Path(label["im_file"]).name):
                self.assertEqual(label["shape"], (48, 80))
                self.assertEqual(len(label["cls"]), 1)
                pixels, original_shape, resized_shape = dataset.load_image(index)
                self.assertEqual(original_shape, (48, 80))
                self.assertEqual(resized_shape, (48, 80))
                self.assertGreater(int(pixels[20, 18, 2]), 230)
                self.assertLess(int(pixels[20, 18, 0]), 30)
                sample = dataset[index]
                self.assertEqual(tuple(sample["img"].shape), (3, 80, 80))
                torch.testing.assert_close(
                    sample["bboxes"], torch.tensor([[0.225, 0.475, 0.25, 0.25]]),
                    atol=1e-6, rtol=0,
                )
                # The stored red object and its transformed box occupy the same
                # noncentral region; rotations/mirroring cannot pass this check.
                self.assertGreater(int(sample["img"][0, 36, 18]), 230)
                self.assertLess(int(sample["img"][2, 36, 18]), 30)
                if baseline_pixels is None:
                    baseline_pixels, baseline_tensor = pixels, sample["img"]
                else:
                    np.testing.assert_array_equal(pixels, baseline_pixels)
                    torch.testing.assert_close(sample["img"], baseline_tensor, rtol=0, atol=0)
        self.assertTrue(dataset.cv2_flag & cv2.IMREAD_IGNORE_ORIENTATION)
        self.assertEqual({path: digest(path) for path in source_hashes}, source_hashes)

    def test_adjacent_npy_and_label_caches_are_ignored_and_label_content_is_rescanned(self):
        image, label, source = self.fixture(6)
        adjacent_npy = image.with_suffix(".npy")
        np.save(adjacent_npy, np.full((80, 48, 3), 191, dtype=np.uint8))
        adjacent_cache = self.labels.with_suffix(".cache")
        old_cache = {
            "labels": [{"im_file": str(image), "shape": (80, 48),
                        "cls": np.array([[0]], dtype=np.float32),
                        "bboxes": np.array([[0.8, 0.2, 0.1, 0.1]], dtype=np.float32),
                        "segments": [], "keypoints": None, "normalized": True, "bbox_format": "xywh"}],
            "hash": get_hash([str(label), str(image)]),
            "results": (1, 0, 0, 0, 1), "msgs": [],
        }
        save_dataset_cache_file("", adjacent_cache, old_cache, DATASET_CACHE_VERSION)
        unchanged = {path: digest(path) for path in (source, adjacent_npy, adjacent_cache)}
        with patch("ultralytics.data.dataset.load_dataset_cache_file", side_effect=AssertionError("cache reuse is forbidden")):
            first = self.dataset()
            pixels, shape, _resized = first.load_image(0)
            self.assertEqual(shape, (48, 80))
            self.assertGreater(int(pixels[20, 18, 2]), 230)
            self.assertAlmostEqual(float(first.labels[0]["bboxes"][0, 0]), 0.225)
            self.assertTrue(first.label_cache_path.is_relative_to(first.cache_root))
            self.assertTrue(first.label_cache_path.is_file())
            # Same length and path give the native size-only hash no new identity.
            old_hash = first.get_cache_hash()
            label.write_text(label.read_text().replace("0.225000000", "0.325000000"), encoding="utf-8")
            self.assertEqual(first.get_cache_hash(), old_hash)
            second = self.dataset(cache_root=first.cache_root)
            self.assertAlmostEqual(float(second.labels[0]["bboxes"][0, 0]), 0.325)
        self.assertEqual({path: digest(path) for path in unchanged}, unchanged)
        output_npy = second.npy_files[0]
        self.assertTrue(output_npy.is_relative_to(second.cache_root))
        output_npy.parent.mkdir(parents=True)
        np.save(output_npy, np.zeros((48, 80, 3), dtype=np.uint8))
        npy_hash = digest(output_npy)
        with self.assertRaisesRegex(ValueError, "unverified YOLO image cache"):
            second.load_image(0)
        self.assertEqual(digest(output_npy), npy_hash)

    def test_train_and_validation_builders_use_local_dataset_and_stored_rectangle_shape(self):
        self.fixture(8)
        trainer = object.__new__(yolo_data.FinalYOLOTrainer)
        trainer.args = get_cfg(overrides={"task": "detect", "imgsz": 80, "cache": False, "rect": False})
        trainer.data = self.data
        trainer.model = types.SimpleNamespace(stride=torch.tensor([32]))
        trainer.save_dir = self.root / "training-output"
        training = trainer.build_dataset(str(self.images), "train", batch=2)
        validation = trainer.build_dataset(str(self.images), "val", batch=2)
        for mode, dataset in (("train", training), ("val", validation)):
            with self.subTest(mode=mode):
                self.assertIsInstance(dataset, yolo_data.StoredYOLODataset)
                self.assertEqual(dataset.cache_root, trainer.save_dir / "dataset-cache" / yolo_data.IMAGE_POLICY / mode)
                self.assertTrue(dataset.cv2_flag & cv2.IMREAD_IGNORE_ORIENTATION)
                self.assertEqual(dataset.load_image(0)[1], (48, 80))
        self.assertTrue(training.augment)
        self.assertFalse(training.rect)
        self.assertFalse(validation.augment)
        self.assertTrue(validation.rect)
        self.assertEqual(validation.batch_shapes.tolist(), [[64, 96]])

    def test_verification_keeps_native_duplicate_class_checks_and_never_repairs_source(self):
        image, label, source = self.fixture(6)
        source_hash = digest(source)
        label.write_text(label.read_text() * 2, encoding="utf-8")
        arguments = (str(image), str(label), "", False, 34, 0, 0, False)
        with patch.object(yolo_data, "imread", side_effect=AssertionError("ordinary JPEG scanning must remain lightweight")):
            result = yolo_data.verify_stored_image_label(arguments)
        self.assertEqual(result[2], (48, 80))
        self.assertEqual(result[1].shape, (1, 5))
        self.assertIn("duplicate", result[-1])
        self.assertEqual(digest(source), source_hash)
        label.write_text("34 0.2 0.3 0.1 0.1\n", encoding="utf-8")
        invalid_label = yolo_data.verify_stored_image_label(arguments)
        self.assertIsNone(invalid_label[0])
        self.assertEqual(invalid_label[8], 1)
        self.assertEqual(digest(source), source_hash)

    def test_jpeg_endings_follow_native_decode_boundary_without_source_repairs(self):
        image, label, source = self.fixture(8)
        original_bytes = source.read_bytes()
        arguments = (str(image), str(label), "", False, 34, 0, 0, False)
        flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        baseline = cv2.imdecode(np.frombuffer(original_bytes, dtype=np.uint8), flags)
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.shape, (48, 80, 3))
        with patch("ultralytics.data.utils.check_image", side_effect=AssertionError("native repair must never run")) as repair:
            for case, content in (("valid", original_bytes),
                                  ("trailing-metadata", original_bytes + b"\nfixture trailing metadata\n"),
                                  ("missing-eoi", original_bytes[:-2])):
                with self.subTest(case=case):
                    source.write_bytes(content)
                    expected_sha = digest(source)
                    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), flags)
                    readable = decoded is not None and decoded.shape[:2] == (48, 80)
                    if case != "missing-eoi":
                        self.assertTrue(readable)
                        np.testing.assert_array_equal(decoded, baseline)
                    # A missing EOI is not universally decodable. Use the pinned
                    # native decoder as the oracle, without accepting a repair.
                    if readable:
                        result = yolo_data.verify_stored_image_label(arguments)
                        self.assertEqual(result[2], (48, 80))
                        self.assertEqual(result[1].shape, (1, 5))
                        dataset = self.dataset()
                        pixels, original_shape, _ = dataset.load_image(0)
                        self.assertEqual(original_shape, (48, 80))
                        np.testing.assert_array_equal(pixels, decoded)
                    else:
                        with self.assertRaisesRegex(ValueError, "cannot be decoded"):
                            yolo_data.verify_stored_image_label(arguments)
                        with self.assertRaisesRegex(ValueError, "cannot be decoded"):
                            self.dataset()
                    self.assertEqual(digest(source), expected_sha)
                    self.assertEqual(image.resolve(), source)
            repair.assert_not_called()

    def test_undecodable_image_fails_without_repair_or_silent_dataset_omission(self):
        image, label, source = self.fixture(6)
        arguments = (str(image), str(label), "", False, 34, 0, 0, False)
        # Keep the JPEG header recognizable but remove its compressed scan.
        original = source.read_bytes()
        source.write_bytes(original[:original.index(b"\xff\xda")])
        source_sha = digest(source)
        with patch("ultralytics.data.utils.check_image", side_effect=AssertionError("native repair must never run")) as repair, \
             patch("ultralytics.utils.checks.check_requirements", side_effect=AssertionError("image verification must not install codecs")) as install:
            with self.assertRaises((OSError, ValueError)):
                yolo_data.verify_stored_image_label(arguments)
            with self.assertRaises((OSError, ValueError)):
                self.dataset()
            repair.assert_not_called()
            install.assert_not_called()
        self.assertEqual(digest(source), source_sha)


if __name__ == "__main__":
    unittest.main()
