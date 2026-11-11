from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from training import export_yolo26 as exporter
from training.checkpoints import destination

BASELINE_ARCHIVE_FILENAME = "yolo26x_detector_full_finaltarget_e07_ema.preparing.pt"


class NativeDetectorFixture(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.yaml = {"nc": 34, "scale": "x", "weights_path": "/example/initial.pth", "pretrained": True, "yaml_file": "/example/native.yaml"}


class YOLO26ExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-yolo-export-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "release"

    def fixture(self, use_ema=True):
        native = {"model": NativeDetectorFixture(-2.0), "ema": NativeDetectorFixture(3.0) if use_ema else None, "optimizer": {"unused": True}}
        selected = native["ema"] if use_ema else native["model"]
        source = self.root / "native.pt"
        torch.save(native, source)
        expected = {
            "schema_version": 1, "model_name": "yolo26",
            "model": dict(selected.state_dict()),
            "model_config": {"nc": 34, "scale": "x", "weights_path": None, "pretrained": False, "yaml_file": "yolo26x.yaml"},
            "class_names": exporter.class_names(), "inference": {"imgsz": 960, "task": "detect"},
        }
        reference = self.root / "reference" / BASELINE_ARCHIVE_FILENAME
        reference.parent.mkdir(exist_ok=True)
        torch.save(expected, reference)
        checksum = hashlib.sha256(reference.read_bytes()).hexdigest()
        artifact = {"path": "checkpoints/detectors/yolo26x.pt", "bytes": reference.stat().st_size, "sha256": checksum,
                    "source_checkpoint": {"bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
        return source, reference, artifact, expected

    def export(self, source, artifact):
        with patch.object(exporter, "load_manifest", return_value={"artifacts": {"yolo26": artifact}}), patch.object(exporter, "_pin_ultralytics"):
            return exporter.export_yolo26(source, self.output)

    def test_ema_priority_model_fallback_and_fixed_archive_basename_match_reference_bytes(self):
        for use_ema in (True, False):
            with self.subTest(use_ema=use_ema):
                self.output = self.root / ("release-ema" if use_ema else "release-model")
                source, reference, artifact, expected = self.fixture(use_ema)
                result = self.export(source, artifact)
                target = destination(self.output, artifact)
                self.assertEqual(result["status"], "exported")
                self.assertEqual(target.read_bytes(), reference.read_bytes())
                restored = torch.load(target, map_location="cpu", weights_only=True)
                self.assertEqual(set(restored), set(expected))
                self.assertEqual(restored["model_config"], expected["model_config"])
                torch.testing.assert_close(restored["model"]["weight"], expected["model"]["weight"])
                self.assertFalse(target.with_name(BASELINE_ARCHIVE_FILENAME).exists())

    def test_source_identity_is_checked_before_native_deserialization(self):
        source, _reference, artifact, _expected = self.fixture()
        artifact["source_checkpoint"]["sha256"] = "0" * 64
        with patch.object(exporter.torch, "load") as deserialize, self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.export(source, artifact)
        deserialize.assert_not_called()

    def test_existing_valid_release_is_reused_and_changed_release_is_preserved(self):
        source, reference, artifact, _expected = self.fixture()
        target = destination(self.output, artifact)
        target.parent.mkdir(parents=True)
        target.write_bytes(reference.read_bytes())
        with patch.object(exporter.torch, "load") as deserialize:
            self.assertEqual(self.export(source, artifact)["status"], "existing_verified")
        deserialize.assert_not_called()
        target.write_bytes(b"unrelated existing file")
        with patch.object(exporter.torch, "load") as deserialize, self.assertRaises(ValueError):
            self.export(source, artifact)
        deserialize.assert_not_called()
        self.assertEqual(target.read_bytes(), b"unrelated existing file")

    def test_previous_release_filename_is_reused_without_rebuilding(self):
        source, reference, artifact, _expected = self.fixture()
        target = destination(self.output, artifact, canonical=True)
        previous = target.with_name(f"previous_model_{artifact['sha256'][:8]}.pt")
        target.parent.mkdir(parents=True)
        previous.write_bytes(reference.read_bytes())
        with patch.object(exporter.torch, "load") as deserialize:
            result = self.export(source, artifact)
        self.assertEqual(result["status"], "existing_verified")
        self.assertEqual(result["path"], previous.relative_to(self.output).as_posix())
        self.assertFalse(target.exists())
        deserialize.assert_not_called()

    def test_changed_export_identity_never_publishes_a_canonical_file(self):
        source, _reference, artifact, _expected = self.fixture()
        artifact["sha256"] = "0" * 64
        artifact["path"] = "checkpoints/detectors/yolo26x.pt"
        with self.assertRaisesRegex(ValueError, "No canonical checkpoint was published"):
            self.export(source, artifact)
        target = destination(self.output, artifact)
        self.assertFalse(target.exists())
        self.assertFalse(target.with_name(BASELINE_ARCHIVE_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
