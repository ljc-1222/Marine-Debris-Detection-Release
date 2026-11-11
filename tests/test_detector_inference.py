from __future__ import annotations

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

from inference import detector_backends as backends, detector_geometry as geometry, detectors
from inference import run as workflow
from training.final_detectors import LoadedDetector, class_names


class DetectorInferenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-detector-inference-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "image.png"
        Image.fromarray(np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)).save(self.image)
        self.row = {"id": 4, "file_name": self.image.name, "width": 13, "height": 9}

    def test_all_eight_exif_orientations_restore_encoded_coordinates(self):
        expected = {1: [1, 2, 4, 5], 2: [6, 2, 9, 5], 3: [6, 3, 9, 6], 4: [1, 3, 4, 6], 5: [2, 1, 5, 4], 6: [2, 4, 5, 7], 7: [5, 4, 8, 7], 8: [5, 1, 8, 4]}
        for orientation, box in expected.items():
            with self.subTest(orientation=orientation):
                actual = geometry.display_boxes_to_encoded([[1, 2, 4, 5]], 10, 8, orientation)
                np.testing.assert_array_equal(actual, np.asarray([box], dtype=np.float32))
        boxes = [[1.0, 2.0, 4.0, 5.0]]
        self.assertEqual(geometry.inverse_horizontal_boxes(geometry.inverse_horizontal_boxes(boxes, 10), 10), [tuple(boxes[0])])

    def test_inverse_flip_background_removal_and_native_rf_threshold(self):
        image = {"id": 5, "width": 20, "height": 10}
        rows = backends.prediction_rows([[1, 2, 5, 6]] * 3, [0, 34, 1], [0.9, 0.8, 0.001], image, flip=True, rfdetr=True)
        self.assertEqual(rows, [{"image_id": 5, "category_id": 1, "bbox": [15.0, 2.0, 4.0, 4.0], "score": 0.9}])

    def test_candidate_limit_preserves_input_order_for_score_ties(self):
        image = {"id": 5, "width": 1000, "height": 10}
        rows = backends.prediction_rows([[index, 0, index + 1, 1] for index in range(301)], [0] * 301, [0.5] * 301, image)
        self.assertEqual(len(rows), 300)
        self.assertEqual([row["bbox"][0] for row in rows], list(range(300)))

    def test_tinyformer_batch_uses_width_height_sizes_and_upstream_flip_pixels(self):
        from torchvision.transforms import v2

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        config = {"val_dataloader": {"dataset": {"transforms": {"ops": [{"type": "Resize", "size": [12, 16]}, {"type": "ConvertPILImage"}, {"type": "Normalize", "mean": mean, "std": std}]}}}}
        actual, sizes = backends.tinyformer_inputs([self.image], config, flip=True)
        transform = v2.Compose([v2.Resize((12, 16)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)])
        with Image.open(self.image) as image:
            expected = torch.stack([transform(ImageOps.mirror(image.convert("RGB")))])
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        self.assertEqual(sizes.tolist(), [[13, 9]])
        self.assertTrue(actual.is_contiguous())

    def test_rf_reuses_antialias_free_tensor_resize_and_height_width_postprocess(self):
        from torchvision.transforms import functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))

            def forward(self, images):
                self.inputs = images.clone()
                self.inference_mode = torch.is_inference_mode_enabled()
                return {}

        class Postprocessor(nn.Module):
            def forward(self, predictions, target_sizes):
                self.sizes = target_sizes.tolist()
                return [{"boxes": torch.tensor([[1.0, 2.0, 5.0, 6.0]]), "scores": torch.tensor([0.5]), "labels": torch.tensor([0])}]

        model, post = Model(), Postprocessor()
        loaded = LoadedDetector("rfdetr", model, class_names(), {"num_select": 300, "resolution": 56}, post)
        rows = backends.collect_rfdetr(loaded, {"images": [self.row]}, self.root, batch=1, flip=False)
        with Image.open(self.image) as image:
            pixels = F.to_tensor(image.convert("RGB"))
        expected = F.normalize(torch.stack([F.resize(pixels, [56, 56], antialias=False)]), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        torch.testing.assert_close(model.inputs, expected, atol=0.0, rtol=0.0)
        self.assertEqual(post.sizes, [[9, 13]])
        self.assertTrue(model.inference_mode)
        self.assertEqual(rows[0]["category_id"], 1)

    def test_index_rejects_labels_changed_images_and_unsafe_relative_paths(self):
        path = self.root / "index.json"
        document = {"categories": [{"id": index, "name": name} for index, name in enumerate(class_names(), 1)], "images": [self.row], "annotations": []}
        path.write_text(json.dumps(document))
        self.assertEqual(detectors.load_index(path, self.root)["images"], [self.row])
        for bad_row in ({**self.row, "file_name": "../image.png"}, {**self.row, "width": 12}, {**self.row, "sha256": "0" * 64}):
            path.write_text(json.dumps({**document, "images": [bad_row]}))
            with self.assertRaises(ValueError):
                detectors.load_index(path, self.root)
        path.write_text(json.dumps({**document, "annotations": [{"category_id": 1}]}))
        with self.assertRaisesRegex(ValueError, "must not contain labels"):
            detectors.load_index(path, self.root)

    def test_codetr_batch_two_is_rejected_before_model_loading(self):
        args = types.SimpleNamespace(model="codetr", batch=2, workers=0, device="cpu",
            index=self.root / "not-loaded-index.json", images=self.root, output=self.root / "predictions.json",
            manifest=self.root / "not-loaded-catalog.json", checkpoint_root=self.root, flip=False)
        with patch.object(detectors, "load_detector") as load, self.assertRaisesRegex(ValueError, "Co-DETR requires --batch 1"):
            detectors.collect(args)
        load.assert_not_called()

    def test_workflow_batch_two_keeps_both_codetr_views_at_one(self):
        second = self.root / "portrait.png"
        Image.new("RGB", (9, 13)).save(second)
        index = self.root / "index.json"
        index.write_text(json.dumps({"images": [self.row, {"id": 5, "file_name": second.name, "width": 9, "height": 13}],
            "categories": [{"id": index, "name": name} for index, name in enumerate(class_names(), 1)]}))
        # Only process orchestration is exercised; the subprocesses never load these fixture files as models.
        catalog = {"schema_version": 2, "artifacts": {}}
        weights = self.root / "weights"
        for name in sorted({name for name, _flip in workflow.VIEWS} | set(workflow.HEADS)):
            relative = "checkpoints/" + name + ".pt"
            path = weights / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            catalog["artifacts"][name] = {"path": relative, "sha256": workflow.sha256(path), "bytes": path.stat().st_size}
        manifest = self.root / "catalog.json"
        manifest.write_text(json.dumps(catalog))
        args = types.SimpleNamespace(index=index, images=self.root, checkpoint_root=weights, manifest=manifest,
            output_dir=self.root / "workflow", detector_batch=2, classifier_batch=2, workers=0, device="cpu",
            codetr_command=None)
        commands = []

        def execute(command, **_kwargs):
            commands.append(command)
            if "inference.classifiers" in command or "submit" in command:
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir(parents=True)
                (output / "manifest.json").write_text(json.dumps({"fixture_scope": "Subprocess model inference is mocked"}))
            return types.SimpleNamespace(returncode=0)

        with patch.object(workflow, "configure_caches"), patch.object(workflow.subprocess, "run", side_effect=execute):
            result = workflow.run(args)
        self.assertEqual(result["status"], "complete")
        detector_commands = [command for command in commands if "inference.detectors" in command]
        actual = [(command[command.index("--model") + 1], "--flip" in command, int(command[command.index("--batch") + 1]))
                  for command in detector_commands]
        self.assertEqual(actual, [(name, flip, 1 if name == "codetr" else 2) for name, flip in workflow.VIEWS])


if __name__ == "__main__":
    unittest.main()
