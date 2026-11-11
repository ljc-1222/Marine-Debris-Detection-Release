"""Check local guards without contacting analytics or creating a real YOLO model."""
import importlib
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

from training import local_ultralytics as local
from training.final_detectors import LoadedDetector
from inference.detector_backends import collect_yolo26


class LocalUltralyticsTest(unittest.TestCase):
    def modules(self):
        package = types.ModuleType("ultralytics")
        package.settings = dict.fromkeys(local.LOCAL_FLAGS, True)
        utils = types.ModuleType("ultralytics.utils")
        package.utils = utils
        callbacks = types.ModuleType("ultralytics.utils.callbacks")
        callbacks.add_integration_callbacks = Mock(side_effect=AssertionError("External callback injection"))
        utils.callbacks = callbacks
        events = types.ModuleType("ultralytics.utils.events")
        events.events = types.SimpleNamespace(enabled=True, events=[{"synthetic": True}])
        sentry = types.ModuleType("sentry_sdk")
        sentry.init = Mock(side_effect=AssertionError("Sentry initialization"))
        utils.set_sentry = sentry.init
        return {"ultralytics": package, "ultralytics.utils": utils,
                "ultralytics.utils.callbacks": callbacks, "ultralytics.utils.events": events,
                "sentry_sdk": sentry}

    def test_cached_events_sentry_and_constructor_integrations_are_disabled_without_key_changes(self):
        modules = self.modules()
        original_loader = modules["ultralytics.utils.callbacks"].add_integration_callbacks
        real_import = importlib.import_module

        def inspect_import(name):
            if name == "ultralytics":
                self.assertIsNone(sys.modules["sentry_sdk"])
            if name == "ultralytics.utils.events":
                random.random()  # Upstream singleton initialization consumes this.
            return real_import(name)

        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"ULTRALYTICS_API_KEY": "fixture-key"}), patch.object(local, "configure_caches"), patch.object(local.importlib, "import_module", side_effect=inspect_import):
            random_state = random.getstate()
            with local.local_ultralytics() as package:
                self.assertTrue(all(package.settings[name] is False for name in local.LOCAL_FLAGS))
                self.assertFalse(modules["ultralytics.utils.events"].events.enabled)
                self.assertEqual(modules["ultralytics.utils.events"].events.events, [])
                modules["ultralytics.utils.callbacks"].add_integration_callbacks(object())
                modules["ultralytics.utils"].set_sentry()
                self.assertEqual(random.getstate(), random_state)
                self.assertEqual(os.environ["ULTRALYTICS_API_KEY"], "fixture-key")
            self.assertIs(sys.modules["sentry_sdk"], modules["sentry_sdk"])
            self.assertIs(modules["ultralytics.utils.callbacks"].add_integration_callbacks, original_loader)
        original_loader.assert_not_called()
        modules["sentry_sdk"].init.assert_not_called()

    def test_api_and_lazily_created_predictor_are_both_guarded(self):
        modules = self.modules()
        callbacks = modules["ultralytics.utils.callbacks"]
        created = []

        class FakeYOLO:
            def __init__(self, *args, **kwargs):
                callbacks.add_integration_callbacks(self)
                created.append("api")
                self.model = types.SimpleNamespace(args={})
                self.overrides = {}

            def predict(self, *, source, **kwargs):
                callbacks.add_integration_callbacks(self)
                created.append("predictor")
                boxes = types.SimpleNamespace(xyxy=torch.empty(0, 4), cls=torch.empty(0), conf=torch.empty(0))
                return [types.SimpleNamespace(boxes=boxes) for item in source]

        modules["ultralytics"].YOLO = FakeYOLO
        loaded = LoadedDetector("yolo26", torch.nn.Linear(1, 1), [str(index) for index in range(34)], {})
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-yolo-guard-") as directory:
            images = Path(directory)
            Image.new("RGB", (13, 9), "blue").save(images / "image.png")
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {"ULTRALYTICS_API_KEY": "fixture-key"}), patch.object(local, "configure_caches"):
                result = collect_yolo26(loaded, {"images": [{"id": 1, "file_name": "image.png", "width": 13, "height": 9}]}, images, batch=1, flip=False)
        self.assertEqual(result, [])
        self.assertEqual(created, ["api", "predictor"])
        callbacks.add_integration_callbacks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
