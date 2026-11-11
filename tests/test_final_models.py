from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import yaml

from training import final_models, models


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 34)


class TinyDINOBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.qkv = nn.Linear(8, 24)
        block.attn.proj = nn.Linear(8, 8)
        block.mlp = nn.Module()
        block.mlp.w1 = nn.Linear(8, 16)
        block.mlp.w2 = nn.Linear(8, 16)
        block.mlp.w3 = nn.Linear(16, 8)
        self.blocks = nn.ModuleList([block])


class FinalClassifierTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-final-models-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "postprocess.yaml"
        self.config.write_text(final_models.DEFAULT_CONFIG.read_text())
        self.manifest = self.root / "checkpoints.json"

    def write_checkpoint(self, name="hplus", *, overrides=None, state=None):
        expected = TinyClassifier()
        document = {"schema_version": 1, "model_name": name, "model": expected.state_dict() if state is None else state}
        document.update(overrides or {})
        path = self.root / f"{name}.pt"
        torch.save(document, path)
        catalog = {"schema_version": 2, "groups": {"final": [name]}, "artifacts": {name: {
            "path": f"checkpoints/{name}.pt", "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }}}
        self.manifest.write_text(json.dumps(catalog))
        return path, expected

    def load(self, name, path):
        return final_models.load_classifier(name, path, manifest=self.manifest, config=self.config)

    def test_full_states_use_no_pretraining_download_and_are_loaded_strictly(self):
        for name, factory_name, argument in (("hplus", "HPlusDCR", "pretrained_weights"), ("v7", "DINOAttentionDCR", "pretrained_weights"), ("siglip", "ImageTowerDCR", "pretrained_model")):
            with self.subTest(model=name):
                path, expected = self.write_checkpoint(name)
                with patch.object(final_models, factory_name, side_effect=lambda *args, **kwargs: TinyClassifier()) as factory, patch("training.checkpoints.download_file", side_effect=AssertionError("Inference must not download weights")), patch.object(final_models.torch, "load", wraps=torch.load) as deserialize, patch.object(final_models, "merge_lora_", return_value=160) as merge:
                    restored = self.load(name, path)
                self.assertEqual(factory.call_args.kwargs, {argument: None})
                self.assertTrue(deserialize.call_args.kwargs["weights_only"])
                self.assertFalse(restored.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in restored.parameters()))
                if name == "hplus":
                    merge.assert_called_once_with(restored.backbone)
                else:
                    merge.assert_not_called()
                for key, value in expected.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], value)

    def test_unmerged_state_can_be_requested_without_changing_the_checkpoint(self):
        path, expected = self.write_checkpoint()
        before = path.read_bytes()
        with patch.object(final_models, "HPlusDCR", side_effect=lambda *args, **kwargs: TinyClassifier()), patch.object(final_models, "merge_lora_") as merge:
            restored = final_models.load_classifier("hplus", path, manifest=self.manifest, config=self.config, deployment=False)
        merge.assert_not_called()
        self.assertEqual(before, path.read_bytes())
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)

    def test_hplus_deployment_rejects_an_incomplete_merge(self):
        path, _ = self.write_checkpoint()
        with patch.object(final_models, "HPlusDCR", side_effect=lambda *args, **kwargs: TinyClassifier()), patch.object(final_models, "merge_lora_", return_value=159), self.assertRaisesRegex(RuntimeError, "160"):
            self.load("hplus", path)

    def test_size_and_sha_are_verified_before_deserialization(self):
        for field in ("bytes", "sha256"):
            with self.subTest(field=field):
                path, _ = self.write_checkpoint()
                catalog = json.loads(self.manifest.read_text())
                catalog["artifacts"]["hplus"][field] = path.stat().st_size + 1 if field == "bytes" else "0" * 64
                self.manifest.write_text(json.dumps(catalog))
                with patch.object(final_models.torch, "load") as deserialize, self.assertRaises(ValueError):
                    self.load("hplus", path)
                deserialize.assert_not_called()

    def test_wrong_schema_model_name_and_non_tensor_state_are_rejected(self):
        for overrides in ({"schema_version": 2}, {"schema_version": True}, {"model_name": "v7"}, {"model": {}}, {"model": {"weight": "not a tensor"}}):
            with self.subTest(overrides=overrides):
                path, _ = self.write_checkpoint(overrides=overrides)
                with patch.object(final_models, "HPlusDCR") as factory, self.assertRaises(ValueError):
                    self.load("hplus", path)
                factory.assert_not_called()

    def test_missing_or_unexpected_model_keys_are_rejected(self):
        for unexpected in (False, True):
            with self.subTest(unexpected=unexpected):
                state = TinyClassifier().state_dict()
                if unexpected:
                    state["unexpected.weight"] = torch.zeros(1)
                else:
                    del state["head.bias"]
                path, _ = self.write_checkpoint(state=state)
                with patch.object(final_models, "HPlusDCR", side_effect=lambda *args, **kwargs: TinyClassifier()), self.assertRaises(RuntimeError):
                    self.load("hplus", path)

    def test_wrong_class_count_and_remote_architecture_are_rejected(self):
        for remote_architecture in (False, True):
            with self.subTest(remote_architecture=remote_architecture):
                path, _ = self.write_checkpoint("siglip")
                document = yaml.safe_load(final_models.DEFAULT_CONFIG.read_text())
                if remote_architecture:
                    next(row for row in document["dcr"]["models"] if row["name"] == "siglip")["architecture"] = "hf_hub:unrequested/model"
                else:
                    document["classes"] = 33
                self.config.write_text(yaml.safe_dump(document))
                with patch.object(final_models, "ImageTowerDCR") as factory, self.assertRaises(ValueError):
                    self.load("siglip", path)
                factory.assert_not_called()

    def test_hplus_empty_architecture_keeps_separate_base_and_lora_state(self):
        config = {"attention_heads": 2, "projection_dimensions": 8, "adapter": {
            "rank": 2, "alpha": 2.0, "dropout": 0.0, "verify_hplus_contract": False,
        }}
        with patch.object(models, "load_dino_backbone", return_value=TinyDINOBackbone()) as factory:
            classifier = models.HPlusDCR(config)
        factory.assert_called_once_with(config, None)
        self.assertIsInstance(classifier.backbone.blocks[0].attn.qkv, models.LoRALinear)
        state = classifier.state_dict()
        self.assertIn("backbone.blocks.0.attn.qkv.base.weight", state)
        self.assertIn("backbone.blocks.0.attn.qkv.lora_A", state)
        self.assertIn("backbone.blocks.0.attn.qkv.lora_B", state)
        self.assertEqual(classifier.head.classifier.out_features, 34)


if __name__ == "__main__":
    unittest.main()
