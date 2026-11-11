from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.checkpoints import destination, download_file, load_manifest, main, verify_artifact, verify_file


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-checkpoints-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.content = b"not a pickle: checkpoint bytes are never executed"
        self.artifact = {"path": "checkpoints/example.pt", "bytes": len(self.content),
                         "sha256": hashlib.sha256(self.content).hexdigest(),
                         "hf": {"repo_id": "example/model", "revision": "a" * 40,
                                "filename": "remote.pt"}}

    def model_paths(self):
        prefix = self.artifact["sha256"][:8]
        artifact = {**self.artifact, "path": f"checkpoints/detectors/example-{prefix}.pt"}
        target = self.root / artifact["path"]
        return artifact, target, target.with_name(f"previous_model_{prefix}.pt")

    def test_verify_detects_missing_truncated_and_same_length_modified_files(self):
        path = destination(self.root, self.artifact)
        with self.assertRaises(FileNotFoundError):
            verify_file(path, self.artifact)
        path.parent.mkdir()
        path.write_bytes(self.content[:-1])
        with self.assertRaises(ValueError):
            verify_file(path, self.artifact)
        path.write_bytes(b"x" * len(self.content))
        with self.assertRaises(ValueError):
            verify_file(path, self.artifact)
        path.write_bytes(self.content)
        self.assertEqual(verify_file(path, self.artifact), {
            "path": self.artifact["path"], "bytes": len(self.content),
            "sha256": self.artifact["sha256"], "status": "verified",
        })

    def test_named_verification_errors_identify_the_model(self):
        artifact = {**self.artifact, "display_name": "DINOv3 ViT-H+/16 (LoRA)"}
        path = destination(self.root, artifact)
        path.parent.mkdir()
        for content in (None, self.content[:-1], b"x" * len(self.content)):
            with self.subTest(content=content):
                if content is not None:
                    path.write_bytes(content)
                with self.assertRaises((FileNotFoundError, ValueError)) as failure:
                    verify_file(path, artifact)
                self.assertIn(artifact["display_name"], str(failure.exception))

    def test_cli_reports_model_names_with_pinned_downloads(self):
        names = {"hplus": "DINOv3 ViT-H+/16 (LoRA)", "v7": "DINOv3 ViT-L/16"}
        artifacts = {key: {**self.artifact, "display_name": name, "path": f"checkpoints/{key}.pt"}
                     for key, name in names.items()}
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 2, "groups": {"final": list(names)},
                                        "artifacts": artifacts}))
        cached = [self.root / f"cached-{key}.pt" for key in names]
        for path in cached:
            path.write_bytes(self.content)
        with patch("huggingface_hub.hf_hub_download", side_effect=[str(path) for path in cached]) as fetch:
            for action in ("list", "download", "verify"):
                with self.subTest(action=action), redirect_stdout(io.StringIO()) as output, patch("sys.argv", [
                    "checkpoints", action, "--manifest", str(manifest), "--directory", str(self.root),
                    "--local-files-only",
                ]):
                    self.assertEqual(main(), 0)
                    results = json.loads(output.getvalue())["artifacts"]
                    self.assertEqual(set(results), set(names))
                    for key, name in names.items():
                        self.assertEqual(results[key]["display_name"], name)
                        self.assertEqual(results[key]["path"], artifacts[key]["path"])
            self.assertEqual(fetch.call_count, len(names))
            for fetch_call in fetch.call_args_list:
                self.assertEqual(fetch_call.kwargs["revision"], "a" * 40)
                self.assertTrue(fetch_call.kwargs["local_files_only"])

    def test_private_download_failure_names_model_and_hides_remote_error_details(self):
        artifact = {**self.artifact, "display_name": "DINOv3 ViT-L/16"}
        with patch("huggingface_hub.hf_hub_download", side_effect=PermissionError("denied: token=private-secret")):
            with self.assertRaises(RuntimeError) as failure:
                download_file(self.root, "v7", artifact)
        message = str(failure.exception)
        self.assertIn(artifact["display_name"], message)
        self.assertIn("hf auth login", message)
        self.assertIn("obtain repository access", message)
        self.assertNotIn("private-secret", message)

    def test_download_checks_before_publication_and_does_not_replace_existing_data(self):
        cached = self.root / "cached.pt"
        cached.write_bytes(self.content[:-1])
        artifact, target, _previous = self.model_paths()
        with patch("huggingface_hub.hf_hub_download", return_value=str(cached)) as fetch:
            with self.assertRaises(ValueError):
                download_file(self.root, "example", artifact)
            self.assertFalse(target.exists())
            cached.write_bytes(self.content)
            download_file(self.root, "example", artifact)
            self.assertEqual(target.read_bytes(), self.content)
            fetch.assert_called_with(repo_id="example/model", filename="remote.pt", revision="a" * 40,
                                     repo_type="model", local_dir=self.root / ".cache/checkpoint-downloads/example",
                                     local_files_only=False)
            target.write_bytes(self.content[:-1])
            calls = fetch.call_count
            with self.assertRaises(ValueError):
                download_file(self.root, "example", artifact)
            self.assertEqual(fetch.call_count, calls)

    def test_cli_reads_verified_earlier_filename_without_downloading_or_renaming_it(self):
        artifact, target, previous = self.model_paths()
        previous.parent.mkdir(parents=True)
        previous.write_bytes(self.content)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 2, "groups": {"final": ["example"]},
                                        "artifacts": {"example": {**artifact, "hf": None}}}))
        with patch("huggingface_hub.hf_hub_download") as fetch:
            for action in ("verify", "download"):
                with self.subTest(action=action), redirect_stdout(io.StringIO()) as output, patch("sys.argv", [
                    "checkpoints", action, "--manifest", str(manifest), "--directory", str(self.root),
                ]):
                    self.assertEqual(main(), 0)
                    self.assertEqual(json.loads(output.getvalue())["artifacts"]["example"]["path"], previous.relative_to(self.root).as_posix())
            previous.write_bytes(self.content[:-1])
            with self.assertRaisesRegex(ValueError, "expected .* bytes"):
                download_file(self.root, "example", artifact)
            fetch.assert_not_called()
        self.assertTrue(previous.is_file())
        self.assertFalse(target.exists())

    def test_catalog_path_precedes_aliases_and_ambiguous_aliases_are_rejected(self):
        artifact, target, previous = self.model_paths()
        previous.parent.mkdir(parents=True)
        previous.write_bytes(self.content)
        self.assertEqual(destination(self.root, artifact, canonical=True), target)
        previous.with_name("another_" + previous.name).write_bytes(self.content)
        with self.assertRaisesRegex(ValueError, "Multiple earlier filenames match"):
            destination(self.root, artifact)
        target.write_bytes(self.content[:-1])
        self.assertEqual(destination(self.root, artifact), target)
        with self.assertRaisesRegex(ValueError, "expected .* bytes"):
            download_file(self.root, "example", artifact)

    def test_earlier_filename_cannot_escape_the_checkpoint_root(self):
        artifact, _target, previous = self.model_paths()
        previous.parent.mkdir(parents=True)
        previous.symlink_to(self.root.parent / previous.name)
        with self.assertRaisesRegex(ValueError, "escapes the output directory"):
            destination(self.root, artifact)

    def test_plain_filename_download_does_not_calculate_or_report_checksums(self):
        artifact = {**self.artifact, "display_name": "YOLO26x", "path": "checkpoints/detectors/yolo26x.pt"}
        cached = self.root / "cached.pt"
        cached.write_bytes(self.content)
        with patch("huggingface_hub.hf_hub_download", return_value=str(cached)), patch(
            "training.checkpoints.hashlib.sha256", side_effect=AssertionError("Downloads must not hash model files")
        ):
            result = download_file(self.root, "yolo26", artifact)
            self.assertEqual(result["display_name"], "YOLO26x")
            self.assertEqual(result["path"], artifact["path"])
            self.assertNotIn("sha256", result)
            self.assertEqual(destination(self.root, artifact).read_bytes(), self.content)

    def test_plain_filename_reuses_previous_catalog_filename(self):
        artifact = {**self.artifact, "path": "checkpoints/detectors/example.pt"}
        previous = self.root / f"checkpoints/detectors/example-{artifact['sha256'][:8]}.pt"
        previous.parent.mkdir(parents=True)
        previous.write_bytes(self.content)
        with patch("huggingface_hub.hf_hub_download") as fetch:
            result = download_file(self.root, "example", artifact)
        self.assertEqual(result["path"], previous.relative_to(self.root).as_posix())
        fetch.assert_not_called()

    def test_manifest_rejects_paths_and_mutable_revisions(self):
        path = self.root / "manifest.json"
        def write(artifact):
            path.write_text(json.dumps({"schema_version": 2, "groups": {"final": ["example"]},
                                        "artifacts": {"example": artifact}}))
        for invalid in (".", "checkpoints", "../escape.pt", "/escape.pt", "checkpoints/../../escape.pt", "checkpoints\\..\\escape.pt"):
            write({**self.artifact, "path": invalid})
            with self.assertRaises(ValueError):
                load_manifest(path)
        write({**self.artifact, "hf": {**self.artifact["hf"], "revision": "main"}})
        with self.assertRaises(ValueError):
            load_manifest(path)
        write(self.artifact)
        self.assertEqual(load_manifest(path)["artifacts"]["example"], self.artifact)
        (self.root / "checkpoints").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            destination(self.root, self.artifact)

    def test_unpublished_trained_release_does_not_fetch_upstream_weights(self):
        artifact = {**self.artifact, "display_name": "DINOv3 ViT-L/16", "hf": None}
        with patch("huggingface_hub.hf_hub_download") as fetch:
            with self.assertRaisesRegex(ValueError, "trained release has no published HF mapping") as failure:
                download_file(self.root, "example", artifact)
            self.assertIn(artifact["display_name"], str(failure.exception))
            fetch.assert_not_called()

    def test_contract_inspection_follows_identity_and_precedes_download_publication(self):
        cached = self.root / "cached.pt"
        cached.write_bytes(b"x" * len(self.content))
        with patch("training.checkpoint_contracts.inspect_checkpoint") as inspect:
            with self.assertRaises(ValueError):
                verify_artifact(cached, "v7", self.artifact, contract=True)
            inspect.assert_not_called()
            cached.write_bytes(self.content)
            verify_artifact(cached, "v7", self.artifact)
            inspect.assert_not_called()
            inspect.side_effect = ValueError("wrong checkpoint schema")
            artifact = {**self.artifact, "display_name": "DINOv3 ViT-L/16"}
            with patch("huggingface_hub.hf_hub_download", return_value=str(cached)):
                with self.assertRaisesRegex(ValueError, "wrong checkpoint schema") as failure:
                    download_file(self.root, "v7", artifact, contract=True)
                self.assertIn(artifact["display_name"], str(failure.exception))
            self.assertFalse(destination(self.root, self.artifact).exists())
            self.assertTrue(cached.exists())


class CheckpointContractTest(unittest.TestCase):
    @staticmethod
    def fixture(name):
        import torch

        names = [row["name"] for row in json.loads((Path(__file__).resolve().parents[1] / "configs/categories.json").read_text())["categories"]]
        document = {"schema_version": 1, "model_name": name, "model": {}}
        state = document["model"]

        def tensor(key, shape):
            # Shape-only fixtures allocate no model or backing weight storage.
            state[key] = torch.empty(shape, device="meta")

        if name == "codetr":
            document["state_dict"] = document.pop("model")
            document["meta"] = {"CLASSES": names}
            for index in range(7):
                tensor(f"query_head.cls_branches.{index}.weight", (34, 256))
                tensor(f"query_head.cls_branches.{index}.bias", (34,))
            for key, shape in {
                "roi_head.0.bbox_head.fc_cls.weight": (35, 1024),
                "bbox_head.0.atss_cls.weight": (34, 256, 3, 3),
                "query_head.label_embedding.weight": (34, 256),
                "backbone.patch_embed.proj.weight": (1024, 3, 16, 16),
            }.items():
                tensor(key, shape)
        elif name == "tinyformer":
            document.update(backend=name, num_classes=34, class_names=names,
                source_config="configs/tinyformer/object365/tinyformer_dinov3_xl_obj2coco_pbm_24e.yml",
                tinyformer_config={"num_classes": 34, "remap_mscoco_category": False, "model": "DEIM",
                    "DEIM": {"backbone": "DINOv3SSAs_4Scale", "encoder": "HybridEncoder_4Scale", "decoder": "DEIMTransformer"},
                    "DINOv3SSAs_4Scale": {"name": "dinov3_vitb16"}})
            tensor("decoder.enc_score_head.weight", (34, 256))
            tensor("decoder.denoising_class_embed.weight", (35, 256))
            tensor("backbone.dinov3.patch_embed.proj.weight", (768, 3, 16, 16))
            for index in range(6):
                tensor(f"decoder.dec_score_head.{index}.weight", (34, 256))
        elif name == "rfdetr":
            document.update(model_name="RFDETRLargeDeprecated", rfdetr_version="1.9.0",
                args={"num_classes": 34, "class_names": names},
                model_config={"encoder": "dinov2_windowed_base", "hidden_dim": 384, "num_classes": 34})
            tensor("class_embed.weight", (35, 384))
            tensor("class_embed.bias", (35,))
        elif name == "yolo26":
            document.update(class_names=names, model_config={"nc": 34, "scale": "x", "channels": 3, "reg_max": 1, "end2end": True},
                            inference={"imgsz": 960, "task": "detect"})
            for branch in ("cv3", "one2one_cv3"):
                for index in range(3):
                    tensor(f"model.23.{branch}.{index}.2.weight", (34, 128, 1, 1))
                    tensor(f"model.23.{branch}.{index}.2.bias", (34,))
        elif name in {"hplus", "v7"}:
            for key, shape in {"head.classifier.weight": (34, 1024), "head.classifier.bias": (34,),
                               "head.objectness.weight": (1, 1024), "head.objectness.bias": (1,)}.items():
                tensor(key, shape)
            tensor("backbone.patch_embed.proj.weight", (1280 if name == "hplus" else 1024, 3, 16, 16))
            if name == "hplus":
                for index in range(32):
                    for target, (inputs, outputs) in {"attn.qkv": (1280, 3840), "attn.proj": (1280, 1280),
                            "mlp.w1": (1280, 5120), "mlp.w2": (1280, 5120), "mlp.w3": (5120, 1280)}.items():
                        prefix = f"backbone.blocks.{index}.{target}"
                        tensor(prefix + ".base.weight", (outputs, inputs))
                        tensor(prefix + ".lora_A", (32, inputs))
                        tensor(prefix + ".lora_B", (outputs, 32))
                    tensor(f"backbone.blocks.{index}.attn.qkv.base.bias_mask", (3840,))
        else:
            for key, shape in {"foreground.weight": (34, 1024), "foreground.bias": (34,),
                               "objectness.weight": (1, 1024), "objectness.bias": (1,),
                               "encoder.patch_embed.proj.weight": (1024, 3, 16, 16)}.items():
                tensor(key, shape)
        return document

    def test_seven_required_shape_contracts_without_model_construction(self):
        from training.checkpoint_contracts import check_document, MODEL_IDS

        for name in MODEL_IDS:
            with self.subTest(name=name):
                result = check_document(self.fixture(name), name)
                self.assertEqual(result["status"], "contract_verified")
                self.assertIn("No tensor-value scan", result["scope"])

    def test_rejects_wrong_schema_classes_architecture_and_training_state(self):
        from training.checkpoint_contracts import check_document

        for key, value in (("schema_version", True), ("model_name", "hplus"), ("optimizer", {})):
            document = self.fixture("v7")
            document[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_document(document, "v7")
        document = self.fixture("rfdetr")
        document["args"]["class_names"].reverse()
        with self.assertRaisesRegex(ValueError, "class names/order"):
            check_document(document, "rfdetr")
        document = self.fixture("yolo26")
        document["model_config"]["scale"] = "n"
        with self.assertRaisesRegex(ValueError, "architecture"):
            check_document(document, "yolo26")

    def test_hplus_requires_unmerged_pairs_base_weights_and_mask_buffers(self):
        import torch
        from training.checkpoint_contracts import check_document

        for key in ("backbone.blocks.31.mlp.w3.lora_B", "backbone.blocks.0.attn.qkv.base.weight",
                    "backbone.blocks.0.attn.qkv.base.bias_mask"):
            document = self.fixture("hplus")
            del document["model"][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "tensor shape mismatch"):
                check_document(document, "hplus")
        document = self.fixture("hplus")
        document["model"]["backbone.blocks.0.attn.qkv.lora_A"] = torch.empty((16, 1280), device="meta")
        with self.assertRaisesRegex(ValueError, "tensor shape mismatch"):
            check_document(document, "hplus")
        document = self.fixture("hplus")
        document["model"]["backbone.extra.lora_A"] = torch.empty((32, 1280), device="meta")
        with self.assertRaisesRegex(ValueError, "exactly 160 paired"):
            check_document(document, "hplus")

    def test_restricted_mmap_load_has_no_unsafe_fallback(self):
        import pickle
        from training.checkpoint_contracts import inspect_checkpoint

        path = Path("fixture.pt")
        with patch("training.checkpoint_contracts.torch.load", return_value=self.fixture("v7")) as load:
            inspect_checkpoint(path, "v7")
            load.assert_called_once_with(path, map_location="cpu", mmap=True, weights_only=True)
        with patch("training.checkpoint_contracts.torch.load", side_effect=pickle.UnpicklingError("native module")) as load:
            with self.assertRaisesRegex(ValueError, "Restricted checkpoint contract read failed"):
                inspect_checkpoint(path, "v7")
            self.assertEqual(load.call_count, 1)
