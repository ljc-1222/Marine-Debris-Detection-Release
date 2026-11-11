from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from training import final_detectors as detectors


class FinalDetectorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-final-detectors-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "checkpoints.json"
        self.names = detectors.class_names()

    def write_checkpoint(self, name="codetr", *, state=None, overrides=None):
        expected = nn.Linear(2, 34)
        weights = expected.state_dict() if state is None else state
        if name == "codetr":
            document = {"schema_version": 1, "model_name": name, "state_dict": weights, "meta": {"CLASSES": self.names}}
        elif name == "tinyformer":
            document = {"backend": name, "num_classes": 34, "class_names": self.names, "tinyformer_config": {}, "model": weights}
        elif name == "rfdetr":
            document = {"model_name": "RFDETRLargeDeprecated", "model_config": {}, "args": {"num_classes": 34, "class_names": self.names}, "model": weights}
        else:
            document = {"schema_version": 1, "model_name": name, "class_names": self.names, "model_config": {"nc": 34}, "inference": {"imgsz": 960, "task": "detect"}, "model": weights}
        document.update(overrides or {})
        path = self.root / f"{name}.pt"
        torch.save(document, path)
        self.manifest.write_text(json.dumps({"schema_version": 2, "groups": {"final": [name]}, "artifacts": {name: {
            "path": f"checkpoints/{name}.pt", "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }}}))
        return path, expected

    def load(self, name, path):
        return detectors.load_detector(name, path, manifest=self.manifest)

    def test_every_detector_verifies_size_and_digest_before_deserialization(self):
        for name in detectors.DETECTORS:
            for field in ("bytes", "sha256"):
                with self.subTest(model=name, field=field):
                    path, _ = self.write_checkpoint(name)
                    catalog = json.loads(self.manifest.read_text())
                    catalog["artifacts"][name][field] = path.stat().st_size + 1 if field == "bytes" else "0" * 64
                    self.manifest.write_text(json.dumps(catalog))
                    with patch.object(detectors.torch, "load") as deserialize, self.assertRaises(ValueError):
                        self.load(name, path)
                    deserialize.assert_not_called()

    def test_restricted_load_restores_all_tensors_and_sets_inference_mode(self):
        for name in detectors.DETECTORS:
            with self.subTest(model=name):
                path, expected = self.write_checkpoint(name)
                def factory(_document, _path):
                    return detectors.LoadedDetector(name, nn.Linear(2, 34), self.names, {})
                with patch.object(detectors, f"_build_{name}", side_effect=factory), patch.object(detectors.torch, "load", wraps=torch.load) as deserialize:
                    restored = self.load(name, path)
                self.assertTrue(deserialize.call_args.kwargs["weights_only"])
                self.assertFalse(restored.model.training)
                self.assertTrue(all(parameter.requires_grad == (name == "tinyformer") for parameter in restored.model.parameters()))
                for key, value in expected.state_dict().items():
                    torch.testing.assert_close(restored.model.state_dict()[key], value)

    def test_tinyformer_preserves_factory_trainable_frozen_and_integer_parameter_flags(self):
        model = nn.Module()
        model.trainable = nn.Parameter(torch.ones(2))
        model.fixed = nn.Parameter(torch.ones(2), requires_grad=False)
        model.integer = nn.Parameter(torch.tensor([1, 2]), requires_grad=False)
        flags = {name: value.requires_grad for name, value in model.named_parameters()}
        path, _expected = self.write_checkpoint("tinyformer", state=model.state_dict())
        loaded = detectors.LoadedDetector("tinyformer", model, self.names, {})
        with patch.object(detectors, "_build_tinyformer", return_value=loaded):
            restored = self.load("tinyformer", path)
        self.assertEqual({name: value.requires_grad for name, value in restored.model.named_parameters()}, flags)
        self.assertFalse(restored.model.training)

    def test_tinyformer_real_image_matches_upstream_stack_pixels_and_contiguous_layout(self):
        from torchvision.transforms import v2

        image = Image.fromarray(np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3))
        path = self.root / "layout.png"
        image.save(path)
        size, mean, std = [12, 16], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = v2.Compose([v2.Resize(tuple(size)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)])
        single = transform(image)
        expected = torch.stack([single])
        self.assertFalse(single.unsqueeze(0).is_contiguous())
        config = {"val_dataloader": {"dataset": {"transforms": {"ops": [
            {"type": "Resize", "size": size}, {"type": "ConvertPILImage", "dtype": "float32", "scale": True}, {"type": "Normalize", "mean": mean, "std": std},
        ]}}}}
        loaded = detectors.LoadedDetector("tinyformer", nn.Identity(), self.names, config)
        actual = detectors.prepare_image(loaded, path)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(actual.stride(), expected.stride())

    def test_deployment_precision_context_uses_cuda_bfloat16_only_for_tinyformer(self):
        for name in detectors.DETECTORS:
            with self.subTest(model=name):
                with patch.object(detectors.torch, "autocast") as autocast:
                    with detectors.inference_autocast(name, torch.device("cuda:0")):
                        pass
                if name == "tinyformer":
                    autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)
                    self.assertEqual(detectors.inference_precision(name, "cuda:0"), "bfloat16_autocast")
                else:
                    autocast.assert_not_called()
                    self.assertEqual(detectors.inference_precision(name, "cuda:0"), "native_mmcv" if name == "codetr" else "float32")
                with patch.object(detectors.torch, "autocast") as autocast:
                    with detectors.inference_autocast(name, torch.device("cpu")):
                        pass
                autocast.assert_not_called()

    def test_image_forward_runs_in_inference_mode_and_reports_cpu_precision(self):
        class DetectorFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))

            def forward(self, images):
                self.was_inference_mode = torch.is_inference_mode_enabled()
                self.grad_was_enabled = torch.is_grad_enabled()
                return {"pred_logits": torch.zeros(1, 2, 34), "pred_boxes": torch.zeros(1, 2, 4)}

        path = self.root / "image.png"
        Image.new("RGB", (4, 4)).save(path)
        model = DetectorFixture()
        loaded = detectors.LoadedDetector("tinyformer", model, self.names, {})
        with patch.object(detectors, "prepare_image", return_value=torch.zeros(1, 3, 4, 4)):
            result = detectors.forward_image(loaded, path)
        self.assertTrue(model.was_inference_mode)
        self.assertFalse(model.grad_was_enabled)
        self.assertEqual(result["precision"], "float32")

    def test_missing_and_unexpected_state_keys_fail_strict_loading(self):
        for unexpected in (False, True):
            with self.subTest(unexpected=unexpected):
                state = nn.Linear(2, 34).state_dict()
                if unexpected:
                    state["unexpected.weight"] = torch.zeros(1)
                else:
                    del state["bias"]
                path, _ = self.write_checkpoint(state=state)
                result = detectors.LoadedDetector("codetr", nn.Linear(2, 34), self.names, {})
                with patch.object(detectors, "_build_codetr", return_value=result), self.assertRaises(RuntimeError):
                    self.load("codetr", path)

    def test_wrong_class_order_and_model_identity_fail_before_construction(self):
        for name in detectors.DETECTORS:
            with self.subTest(model=name):
                reversed_names = self.names[::-1]
                overrides = {"meta": {"CLASSES": reversed_names}} if name == "codetr" else {"args": {"num_classes": 34, "class_names": reversed_names}} if name == "rfdetr" else {"class_names": reversed_names}
                path, _ = self.write_checkpoint(name, overrides=overrides)
                with patch.object(detectors, f"_build_{name}") as factory, self.assertRaises(ValueError):
                    self.load(name, path)
                factory.assert_not_called()
        for overrides in ({"schema_version": True}, {"model_name": "rfdetr"}, {"state_dict": {"weight": "not a tensor"}}):
            with self.subTest(overrides=overrides):
                path, _ = self.write_checkpoint(overrides=overrides)
                with patch.object(detectors, "_build_codetr") as factory, self.assertRaises(ValueError):
                    self.load("codetr", path)
                factory.assert_not_called()

    def test_arbitrary_serialized_module_is_not_loaded(self):
        path, _ = self.write_checkpoint("yolo26", overrides={"model": nn.Linear(2, 34)})
        with patch.object(detectors, "_build_yolo26") as factory, self.assertRaises(pickle.UnpicklingError):
            self.load("yolo26", path)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
