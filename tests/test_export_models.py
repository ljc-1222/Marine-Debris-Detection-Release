from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
import yaml

from inference import run as inference_run
from training import checkpoints, export_models as exporter, final_detectors
from training.hplus_lora import LoRALinear


class SmallRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_embed = nn.Linear(2, 35)


class SmallHPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        layers = []
        for index in range(160):
            base = nn.Linear(1, 1)
            if index < 32:
                base.register_buffer("bias_mask", torch.ones(1))
            layers.append(LoRALinear(base, rank=1, alpha=1, dropout=0))
        self.backbone.layers = nn.ModuleList(layers)
        self.head = nn.Linear(2, 34)


class NativeYOLO(nn.Module):
    def __init__(self, value, config, names):
        super().__init__()
        self.detector = nn.Linear(2, 34)
        with torch.no_grad():
            self.detector.weight.fill_(value)
        self.yaml, self.names = config, dict(enumerate(names))


class ExportModelsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-model-export-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.names = final_detectors.class_names()
        self.output = self.root / "release"
        self.sources_file = self.root / "sources.json"
        self.config = yaml.safe_load((exporter.ROOT / "third_party/ultralytics/ultralytics/cfg/models/26/yolo26.yaml").read_text())
        self.config.update(nc=34, scale="x", channels=3, yaml_file="yolo26x.yaml")
        self.source_paths = {}
        self.expected = {}
        for name in exporter.MODELS:
            model = self.prototype(name)
            state = model.state_dict()
            document = {"model": state, "optimizer": {"private_training_state": "do_not_export"},
                        "requested_max_steps": 2, "full_recipe_completed": False}
            if name == "codetr":
                document = {"state_dict": {**state, "ema_unused": torch.zeros(1)}, "meta": {"CLASSES": self.names},
                            "requested_max_steps": 2, "full_recipe_completed": False}
            elif name == "tinyformer":
                document.update(backend=name, num_classes=34, class_names=self.names,
                                source_config=str(final_detectors.TINYFORMER_CONFIG), tinyformer_config={
                                    "model": "DEIM", "postprocessor": "PostProcessor", "num_classes": 34,
                                    "remap_mscoco_category": False, "eval_spatial_size": [960, 960],
                                    "DEIM": {"backbone": "DINOv3SSAs_4Scale", "encoder": "HybridEncoder_4Scale", "decoder": "DEIMTransformer"},
                                    "DINOv3SSAs_4Scale": {"name": "dinov3_vitb16", "weights_path": "/private/source/base.pth"},
                                    "HybridEncoder_4Scale": {}, "DEIMTransformer": {},
                                    "output_dir": "/private/source/training", "secret_marker": "do_not_export",
                                    "val_dataloader": {"dataset": {"transforms": {"ops": [
                                        {"type": "Resize", "size": [960, 960]}, {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                                        {"type": "Normalize", "mean": [.5] * 3, "std": [.5] * 3},
                                    ]}}},
                                })
            elif name == "rfdetr":
                document.update(model_name="RFDETRLargeDeprecated", rfdetr_version="1.9.0",
                                model_config={"encoder": "dinov2_windowed_base", "num_classes": 34, "pretrain_weights": "/private/source/initial.pth"},
                                args={"class_names": self.names})
            elif name == "yolo26":
                document.update(model=NativeYOLO(-1, self.config, self.names), ema=NativeYOLO(3, self.config, self.names))
                state = document["ema"].state_dict()
            self.expected[name] = {key: value.clone() for key, value in state.items()}
            path = self.root / f"{name}.pt"
            torch.save(document, path)
            self.source_paths[name] = {"path": path.name, **exporter.identity(path)}
        self.sources_file.write_text(json.dumps(self.source_paths))

    def prototype(self, name):
        if name == "hplus":
            return SmallHPlus()
        if name == "rfdetr":
            return SmallRF()
        if name == "yolo26":
            model = nn.Module()
            model.detector = nn.Linear(2, 34)
            return model
        return nn.Linear(2, 34)

    def fake_factories(self):
        stack = ExitStack()
        for name in final_detectors.DETECTORS:
            def factory(document, _path, model_name=name):
                config = {"encoder": "dinov2_windowed_base", "num_classes": 34, "device": "cpu", "pretrain_weights": None}
                return final_detectors.LoadedDetector(model_name, self.prototype(model_name), self.names, config)
            stack.enter_context(patch.object(final_detectors, "_build_" + name, side_effect=factory))
        stack.enter_context(patch.object(exporter, "classifier_prototype", side_effect=self.prototype))
        stack.enter_context(patch.object(exporter, "_pin_ultralytics"))
        def check_codetr(path, _command):
            document = torch.load(path, map_location="cpu", weights_only=True)
            self.prototype("codetr").load_state_dict(document["state_dict"], strict=True)
        stack.enter_context(patch.object(exporter, "check_codetr", side_effect=check_codetr))
        return stack

    def export(self, **options):
        with self.fake_factories():
            return exporter.export_models(self.sources_file, self.output, trust_native=True, allow_partial=True, **options)

    def test_seven_sources_export_separate_safe_catalog_and_preserve_all_selected_tensors(self):
        catalog = self.export()
        self.assertEqual(set(catalog["artifacts"]), set(exporter.MODELS))
        self.assertEqual(catalog["model_origin"], "local_retraining")
        self.assertTrue(catalog["allow_partial"])
        self.assertFalse(catalog["training_completion_verified"])
        self.assertEqual(catalog["artifacts"]["v7"]["display_name"], "DINOv3 ViT-L/16")
        self.assertEqual(catalog["artifacts"]["v7"]["path"], "checkpoints/classifiers/dinov3-vit-l16-retrained.pt")
        checkpoints.load_manifest(self.output / "checkpoint-manifest.json")
        for name, artifact in catalog["artifacts"].items():
            with self.subTest(model=name):
                self.assertIsNone(artifact["hf"])
                path = checkpoints.destination(self.output, artifact)
                checkpoints.verify_file(path, artifact)
                document = torch.load(path, map_location="cpu", weights_only=True)
                field = "state_dict" if name == "codetr" else "model"
                self.assertEqual(set(document[field]), set(self.expected[name]))
                for key, value in self.expected[name].items():
                    torch.testing.assert_close(document[field][key], value, rtol=0, atol=0)
                self.assertNotIn("optimizer", document)
                self.assertNotIn("ema_unused", document[field])
                if name == "tinyformer":
                    self.assertNotIn("secret_marker", document["tinyformer_config"])
                    self.assertNotIn("output_dir", document["tinyformer_config"])
                    self.assertIsNone(document["tinyformer_config"]["DINOv3SSAs_4Scale"]["weights_path"])
        self.assertNotIn(str(self.root), json.dumps(catalog))
        with self.assertRaises(FileExistsError):
            self.export()

    def test_source_hash_mismatch_is_rejected_before_deserialization(self):
        self.source_paths["v7"]["sha256"] = "0" * 64
        self.sources_file.write_text(json.dumps(self.source_paths))
        with patch.object(exporter.torch, "load") as load, self.assertRaisesRegex(ValueError, "Source identity mismatch"):
            exporter.export_models(self.sources_file, self.output)
        load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_selected_tinyformer_exports_without_other_sources_or_classifier_construction(self):
        self.sources_file.write_text(json.dumps({"tinyformer": self.source_paths["tinyformer"]}))
        for name in exporter.MODELS:
            if name != "tinyformer":
                (self.root / self.source_paths[name]["path"]).unlink()
        with self.fake_factories(), patch.object(
            exporter, "classifier_prototype", side_effect=AssertionError("Detector export must not build a classifier")
        ) as classifier, patch.object(
            exporter, "check_codetr", side_effect=AssertionError("TinyFormer export must not build Co-DETR")
        ):
            with patch.object(exporter, "configure_caches"), patch.object(exporter.sys, "argv", [
                "export_models", "--sources", str(self.sources_file), "--directory", str(self.output),
                "--model", "tinyformer", "--model", "tinyformer", "--allow-partial",
            ]), patch("builtins.print"):
                exporter.main()
            manifest = self.output / "checkpoint-manifest.json"
            catalog = checkpoints.load_manifest(manifest)
            self.assertEqual(set(catalog["artifacts"]), {"tinyformer"})
            self.assertEqual(catalog["groups"], {"final": ["tinyformer"], "all": ["tinyformer"]})
            artifact = catalog["artifacts"]["tinyformer"]
            self.assertEqual(artifact["path"], "checkpoints/detectors/tinyformer-xl-pbm-retrained.pt")
            loaded = final_detectors.load_detector("tinyformer", self.output / artifact["path"], manifest=manifest)
            for key, value in self.expected["tinyformer"].items():
                torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
            classifier.assert_not_called()
        self.assertFalse((self.output / "checkpoints/classifiers").exists())
        report = json.loads((self.output / "export-report.json").read_text())
        self.assertEqual(set(report["artifacts"]), {"tinyformer"})

    def test_selected_sources_ignore_unselected_checkpoints_and_require_each_selected_model(self):
        self.sources_file.write_text(json.dumps({
            "tinyformer": self.source_paths["tinyformer"],
            "v7": "not-created.pt", "hplus_base": "not-created-base.pt",
        }))
        sources = exporter.read_sources(self.sources_file, models=["tinyformer", "tinyformer"])
        self.assertEqual(set(sources), {"tinyformer"})
        with self.assertRaisesRegex(ValueError, "each selected canonical model ID"):
            exporter.read_sources(self.sources_file)
        with self.assertRaisesRegex(ValueError, "each selected canonical model ID"):
            exporter.read_sources(self.sources_file, models=["rfdetr"])
        for models in ([], ["unknown"]):
            with self.subTest(models=models), self.assertRaisesRegex(ValueError, "canonical model ID"):
                exporter.read_sources(self.sources_file, models=models)

    def test_native_module_requires_explicit_trust_and_uses_restricted_load_first(self):
        source = self.root / "yolo26.pt"
        with patch.object(exporter, "_pin_ultralytics") as guard, patch.object(exporter.torch, "load", wraps=torch.load) as load:
            with self.assertRaisesRegex(ValueError, "--trust-native"):
                exporter.load_native(source, "yolo26", trust_native=False)
            self.assertEqual([call.kwargs["weights_only"] for call in load.call_args_list], [True])
            guard.assert_not_called()
        with patch.object(exporter, "_pin_ultralytics") as guard, patch.object(exporter.torch, "load", wraps=torch.load) as load:
            document, unsafe = exporter.load_native(source, "yolo26", trust_native=True)
            self.assertTrue(unsafe)
            self.assertIsInstance(document["ema"], NativeYOLO)
            self.assertEqual([call.kwargs["weights_only"] for call in load.call_args_list], [True, False])
            guard.assert_called_once()

    def test_bounded_training_and_wrong_class_shape_cannot_claim_a_complete_catalog(self):
        with self.fake_factories(), self.assertRaisesRegex(ValueError, "--allow-partial"):
            exporter.export_models(self.sources_file, self.output, trust_native=True)
        self.assertFalse((self.output / "checkpoint-manifest.json").exists())
        native = {"model": nn.Linear(2, 33).state_dict()}
        document, _prototype = exporter.pack_model("v7", native, {})
        with self.fake_factories(), self.assertRaises(RuntimeError):
            exporter.validate_model("v7", document, self.root / "v7.pt")

    def test_hplus_assembles_local_base_adapters_head_and_bias_masks_without_a_hub_call(self):
        template = SmallHPlus()
        adapters = {key: torch.full_like(value, .25) for key, value in template.backbone.state_dict().items() if key.endswith(("lora_A", "lora_B"))}
        base_state = {key.replace(".base.", "."): value.clone() for key, value in template.backbone.state_dict().items() if key not in adapters}
        base = self.root / "official-base.pt"
        torch.save(base_state, base)
        binding = exporter.identity(base)
        (self.root / "configs").mkdir()
        (self.root / "configs/upstream_initialization.json").write_text(json.dumps({"artifacts": {"hplus_base": binding}}))
        native = {"trainable_state": {"adapter": adapters, "head": template.head.state_dict()}}
        with patch.object(exporter, "ROOT", self.root), patch.object(exporter, "classifier_prototype", side_effect=lambda _name: SmallHPlus()), patch.object(torch.hub, "load_state_dict_from_url", side_effect=AssertionError("No initialization download during export")):
            state, model = exporter.hplus_state(native, {"hplus_base": (base, binding)})
            self.assertEqual(sum(key.endswith(("lora_A", "lora_B")) for key in state), 320)
            self.assertEqual(sum(key.endswith(".base.bias_mask") for key in state), 32)
            for key, value in model.backbone.state_dict().items():
                expected = adapters[key] if key in adapters else base_state[key.replace(".base.", ".")]
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            broken = {"trainable_state": {"adapter": dict(adapters), "head": template.head.state_dict()}}
            broken["trainable_state"]["adapter"].pop(next(iter(adapters)))
            with self.assertRaisesRegex(ValueError, "adapter keys"):
                exporter.hplus_state(broken, {"hplus_base": (base, binding)})

    def test_custom_catalog_is_forwarded_to_every_detector_and_classifier_process(self):
        self.export()
        manifest = self.output / "checkpoint-manifest.json"
        images = self.root / "images"
        images.mkdir()
        (images / "image.png").write_bytes(b"file-existence-only fixture; subprocess inference is mocked")
        index = self.root / "index.json"
        index.write_text(json.dumps({"images": [{"id": 1, "file_name": "image.png", "width": 8, "height": 8}],
                                     "categories": [{"id": index, "name": name} for index, name in enumerate(self.names, 1)]}))
        args = types.SimpleNamespace(index=index, images=images, checkpoint_root=self.output, manifest=manifest,
                                     output_dir=self.root / "inference", detector_batch=1, classifier_batch=2,
                                     workers=0, device="cpu", codetr_command=None)
        commands = []
        def execute(command, **_kwargs):
            commands.append(command)
            if "inference.classifiers" in command:
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir(parents=True)
                model = command[command.index("--model") + 1]
                (output / "manifest.json").write_text(json.dumps({"schema_version": 2, "model": model,
                                                                 "fixture_scope": "Producer receipt only; subprocess model inference is mocked"}))
            if "submit" in command:
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir(parents=True)
                (output / "manifest.json").write_text(json.dumps({"csv_rows": 0}))
            return types.SimpleNamespace(returncode=0)
        with patch.object(inference_run, "configure_caches"), patch.object(inference_run.subprocess, "run", side_effect=execute):
            result = inference_run.run(args)
        calls = [command for command in commands if any(name in command for name in ("inference.detectors", "inference.classifiers"))]
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(command[command.index("--manifest") + 1] == str(manifest) for command in calls))
        self.assertEqual(result["checkpoint_catalog_sha256"], exporter.sha256(manifest))
        self.assertEqual(result["model_origin"], "local_retraining")
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["pipeline_semantics"], "specialist_per_class_top100")
        self.assertEqual(set(result["sealed_heads"]), set(inference_run.HEADS))
        submission = next(command for command in commands if "submit" in command)
        self.assertEqual(submission[submission.index("--images") + 1], str(images))
        self.assertEqual(submission[submission.index("--checkpoint-manifest") + 1], str(manifest))
        seals = [submission[index + 1] for index, value in enumerate(submission) if value == "--expected-head"]
        self.assertEqual(seals, [name + "=" + result["sealed_heads"][name]["sha256"] for name in inference_run.HEADS])
        for name, binding in result["sealed_heads"].items():
            self.assertEqual(binding["sha256"], exporter.sha256(args.output_dir / binding["path"]))


if __name__ == "__main__":
    unittest.main()
