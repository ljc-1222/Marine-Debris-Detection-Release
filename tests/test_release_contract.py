"""Keep trained verification downloads separate from upstream retraining inputs."""
import json
from pathlib import Path
import unittest

import yaml

from training.checkpoints import DEFAULT_MANIFEST, load_manifest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTest(unittest.TestCase):
    def test_trained_release_is_standalone_seven_model_contract(self):
        catalog = load_manifest(DEFAULT_MANIFEST)
        trained = {"codetr", "tinyformer", "rfdetr", "yolo26", "hplus", "v7", "siglip"}
        self.assertEqual(set(catalog["artifacts"]), trained)
        self.assertEqual(set(catalog["groups"]["final"]), trained)
        self.assertEqual(set(catalog["groups"]["all"]), trained)
        for key, artifact in catalog["artifacts"].items():
            with self.subTest(artifact=key):
                self.assertFalse(artifact.get("requires"))
                self.assertTrue(artifact["display_name"].strip())
                path = Path(artifact["path"])
                directory = "classifiers" if key in ("hplus", "v7", "siglip") else "detectors"
                self.assertEqual(path.parent, Path("checkpoints") / directory)
                self.assertRegex(path.stem, r"^[a-z0-9-]+$")
                self.assertNotRegex(path.stem, r"-[0-9a-f]{8}$")
        for key in ("hplus", "v7", "siglip"):
            self.assertEqual(catalog["artifacts"][key]["role"], "classifier_final_full_model")

    def test_upstream_initialization_references_match_fixed_training(self):
        upstream = json.loads((ROOT / "configs/upstream_initialization.json").read_text())
        self.assertIs(upstream["provided_in_project_release"], False)
        artifacts = upstream["artifacts"]
        self.assertEqual(set(artifacts), {"codetr_initial", "tinyformer_initial", "rfdetr_initial", "yolo26_initial", "siglip_initial", "hplus_base", "v7_base"})
        recipes = yaml.safe_load((ROOT / "training/final_recipes.yaml").read_text())
        self.assertEqual(artifacts["codetr_initial"]["sha256"], recipes["detectors"]["codetr"]["initialization"]["sha256"])
        for name in ("tinyformer", "rfdetr", "yolo26"):
            self.assertEqual(artifacts[name + "_initial"]["sha256"], recipes["detectors"][name]["initialization_sha256"])
        for name in ("hplus", "v7"):
            self.assertEqual(artifacts[name + "_base"]["sha256"], recipes["dcr"][name]["initialization_sha256"])
        for key, artifact in artifacts.items():
            with self.subTest(artifact=key):
                self.assertNotIn("hf", artifact)
                self.assertTrue(artifact["official_page"].startswith("https://"))
                self.assertNotIn("MarineDebrisImageRecognition/", artifact["official_page"])


if __name__ == "__main__":
    unittest.main()
