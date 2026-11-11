from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from inference import classifiers, detector_backends, detectors, run as workflow
from postprocessing.__main__ import main as postprocessing_main
from postprocessing import three_head_runtime as postprocessing
from postprocessing import cache_contract


class SmallHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, pixels, valid=None):
        self.batch_sizes.append(len(pixels))
        if valid is not None:
            if valid.shape != (len(pixels), (pixels.shape[-1] // 16) ** 2):
                raise AssertionError("patch masks were not supplied with the input batch")
        values = pixels[:, 0, pixels.shape[-2] // 2, pixels.shape[-1] // 2].float()
        logits = torch.arange(34, device=pixels.device).float()[None] + values[:, None]
        return logits, values


class ClassifierInferenceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-classifier-inference-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        Image.new("RGB", (20, 12), (32, 64, 96)).save(self.images / "image.png")
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({"images": [{"id": 3, "file_name": "image.png", "width": 20, "height": 12}],
                                         "categories": [{"id": index, "name": name} for index, name in enumerate(cache_contract.class_order(), 1)],
                                         "annotations": []}))
        self.candidates = [
            {"image_id": 3, "category_id": 1, "bbox": [0, 0, 6, 4], "score": .9},
            {"image_id": 3, "category_id": 2, "bbox": [0, 0, 6, 4], "score": .8},
            {"image_id": 3, "category_id": 1, "bbox": [100, 100, 4, 4], "score": .7},
            {"image_id": 3, "category_id": 1, "bbox": [1, 1, 5, 5], "score": .6},
        ]
        self.bank = self.root / "candidates.json"
        self.bank.write_text(json.dumps(self.candidates))
        self.catalog = self.root / "checkpoints.json"
        artifacts = {}
        for name in ("hplus", "v7", "siglip"):
            path = self.root / "checkpoints" / f"{name}.pt"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(name.encode())
            artifacts[name] = {"path": f"checkpoints/{name}.pt", **classifiers.fingerprint(path)}
        self.catalog.write_text(json.dumps({"schema_version": 2, "groups": {"final": list(artifacts)}, "artifacts": artifacts}))

    def arguments(self, name):
        return argparse.Namespace(
            model=name, bank=self.bank, index=self.index, images=self.images,
            checkpoint_root=self.root, manifest=self.catalog, output_dir=self.root / f"cache-{name}", batch=2, workers=0, device="cpu",
        )

    def test_unique_crops_deduplicate_integer_context_and_preserve_first_seen_order(self):
        rows = [
            {"image_id": 3, "bbox": [1, 2, 4, 6], "category_id": 1},
            {"image_id": 3, "bbox": [1.05, 2.05, 3.9, 5.9], "category_id": 2},
            {"image_id": 7, "bbox": [1, 2, 4, 6], "category_id": 1},
            {"image_id": 3, "bbox": [4, 5, 2, 2], "category_id": 1},
        ]
        inverse, images, rectangles = classifiers.unique_crops(rows, 1.2)
        np.testing.assert_array_equal(inverse, [0, 0, 1, 2])
        np.testing.assert_array_equal(images, [3, 7, 3])
        np.testing.assert_array_equal(rectangles[:2], [[0, 1, 6, 9], [0, 1, 6, 9]])

    def test_renderer_keeps_mean_fill_patch_threshold_and_invalid_fallback(self):
        image = Image.new("RGB", (8, 8), (32, 64, 96))
        options = {"size": 32, "mean": [.5, .5, .5], "std": [.5, .5, .5], "return_valid_patches": True}
        pixels, patches, renderable = classifiers.render_crop(image, [-2, 0, 6, 4], **options)
        self.assertTrue(renderable)
        torch.testing.assert_close(patches, torch.tensor([False, True, False, True]))
        torch.testing.assert_close(pixels[:, 16, 16], (torch.tensor([32, 64, 96]).float() / 255 - .5) / .5, rtol=0, atol=0)
        expected_fill = (128 / 255 - .5) / .5
        self.assertAlmostEqual(float(pixels[0, 0, 0]), expected_fill, places=6)
        _, hplus_mask, _ = classifiers.render_crop(image, [-2, 0, 6, 4], minimum_valid_patch_fraction=0.0, **options)
        self.assertTrue(hplus_mask.all())
        filled, fallback, renderable = classifiers.render_crop(image, [50, 50, 54, 54], minimum_valid_patch_fraction=0.0, **options)
        self.assertFalse(renderable)
        torch.testing.assert_close(fallback, torch.tensor([False, False, True, False]))
        self.assertAlmostEqual(float(filled.mean()), expected_fill, places=6)

    def test_three_heads_write_bound_float32_caches_and_skip_only_invalid_hplus(self):
        for name in ("hplus", "v7", "siglip"):
            with self.subTest(model=name):
                args = self.arguments(name)
                model = SmallHead()
                with patch.object(classifiers, "DEFAULT_MANIFEST", self.catalog), patch.object(classifiers, "load_classifier", return_value=model) as load:
                    result = classifiers.run(args)
                    with self.assertRaises(FileExistsError):
                        classifiers.run(args)
                load.assert_called_once()
                self.assertEqual(load.call_args.args[0], name)
                self.assertEqual(load.call_args.args[1], self.root / "checkpoints" / f"{name}.pt")
                self.assertEqual(load.call_args.kwargs["manifest"], self.catalog)
                self.assertNotIn("deployment", load.call_args.kwargs)
                self.assertEqual(result["rows"], 4)
                self.assertEqual(result["schema_version"], 2)
                self.assertEqual(result["class_order"], cache_contract.class_order())
                self.assertEqual(result["unique_crops"], 3)
                manifest = args.output_dir / "manifest.json"
                postprocessing.check_alignment(self.bank, manifest, {})
                source_images = cache_contract.image_identities(json.loads(self.index.read_text()), self.images, {3})
                cache_contract.verify_cache(manifest, name, self.bank, postprocessing.load_bank(self.bank),
                                            self.index, source_images, self.catalog, classifiers.sha256(manifest))
                identities = np.load(args.output_dir / "row_ids.npy", allow_pickle=False)
                self.assertEqual(identities.shape, (4, 32))
                self.assertEqual(identities.dtype, np.uint8)
                self.assertFalse(np.array_equal(identities[0], identities[1]))
                head = postprocessing.load_head(manifest, 4)
                self.assertEqual(head["logits"].dtype, np.float32)
                self.assertEqual(head["objectness"].dtype, np.float32)
                np.testing.assert_array_equal(head["logits"][0], head["logits"][1])
                self.assertEqual(sum(model.batch_sizes), 2 if name == "hplus" else 3)
                if name == "hplus":
                    np.testing.assert_array_equal(head["valid"], [True, True, False, True])
                    np.testing.assert_array_equal(head["logits"][2], np.zeros(34, dtype=np.float32))
                    self.assertEqual(float(head["objectness"][2]), 0.0)
                    self.assertEqual(result["deployment"], "merged_lora_160")
                else:
                    self.assertTrue(head["valid"].all())
                    self.assertNotIn("valid_crops", result)
                    self.assertFalse(np.all(head["logits"][2] == 0))
                for field in ("class_logits", "objectness_logits"):
                    self.assertEqual(result[field]["path"], f"{field}.npy")
                    self.assertEqual(result[field]["bytes"], (args.output_dir / result[field]["path"]).stat().st_size)
        self.bank.write_text(self.bank.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "candidate identity mismatch"):
            postprocessing.check_alignment(self.bank, self.root / "cache-hplus/manifest.json", {})

    def test_bad_outputs_do_not_publish_a_completed_cache(self):
        class BadHead(SmallHead):
            def __init__(self, mode):
                super().__init__()
                self.mode = mode

            def forward(self, pixels, valid=None):
                logits, objectness = super().forward(pixels, valid)
                if self.mode == "finite":
                    logits[:, 0] = float("nan")
                else:
                    logits = logits[:, :-1]
                return logits, objectness

        for mode, message in (("finite", "non-finite"), ("shape", "output shapes")):
            with self.subTest(mode=mode):
                args = self.arguments("v7")
                with patch.object(classifiers, "DEFAULT_MANIFEST", self.catalog), patch.object(classifiers, "load_classifier", return_value=BadHead(mode)), self.assertRaisesRegex(ValueError, message):
                    classifiers.run(args)
                self.assertFalse((args.output_dir / "manifest.json").exists())

    def test_array_bank_keeps_values_and_rejects_changed_array_bytes(self):
        arrays = {
            "image_ids": np.asarray([row["image_id"] for row in self.candidates], dtype=np.int64),
            "category_ids": np.asarray([row["category_id"] for row in self.candidates], dtype=np.int16),
            "boxes_xywh": np.asarray([row["bbox"] for row in self.candidates], dtype=np.float64),
            "scores": np.asarray([row["score"] for row in self.candidates], dtype=np.float64),
        }
        manifest = self.root / "array-bank.json"
        document = {"rows": 4, "arrays": {name: classifiers.save_npy(self.root / f"{name}.npy", values) for name, values in arrays.items()}}
        manifest.write_text(json.dumps(document))
        loaded = classifiers.load_candidates(manifest)
        self.assertEqual(loaded, [{key: row[key] for key in ("image_id", "category_id", "bbox")} for row in self.candidates])
        score_sha = document["arrays"]["scores"].pop("sha256")
        manifest.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "bind each input array"):
            classifiers.load_candidates(manifest)
        document["arrays"]["scores"]["sha256"] = score_sha
        manifest.write_text(json.dumps(document))
        (self.root / "boxes_xywh.npy").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            classifiers.load_candidates(manifest)

    def test_unknown_image_is_rejected_before_loading_a_classifier(self):
        self.candidates[0]["image_id"] = 999
        self.bank.write_text(json.dumps(self.candidates))
        with patch.object(classifiers, "load_classifier") as load, self.assertRaisesRegex(ValueError, "unknown image"):
            classifiers.run(self.arguments("hplus"))
        load.assert_not_called()

    def test_empty_seven_view_workflow_writes_sealed_caches_and_header_only_csv_without_classifier_load(self):
        catalog = json.loads(self.catalog.read_text())
        for name in ("codetr", "tinyformer", "rfdetr", "yolo26"):
            path = self.root / "checkpoints" / f"{name}.pt"
            path.write_bytes(name.encode())
            catalog["artifacts"][name] = {"path": f"checkpoints/{name}.pt", **classifiers.fingerprint(path)}
        catalog["groups"]["final"] = list(catalog["artifacts"])
        self.catalog.write_text(json.dumps(catalog))

        def execute(command, **kwargs):
            # Run the real CLI stages in-process. Only detector model loading and
            # collection are stubbed; no classifier model or forward is allowed.
            module_index = command.index("-m")
            module, argv = command[module_index + 1], command[module_index + 2:]
            with redirect_stdout(kwargs["stdout"]), redirect_stderr(kwargs["stderr"]):
                if module == "inference.detectors":
                    with patch("sys.argv", [module, *argv]):
                        detectors.main()
                elif module == "inference.classifiers":
                    classifiers.main(argv)
                elif module == "postprocessing":
                    postprocessing_main(argv)
                else:
                    self.fail("Unexpected workflow stage: " + module)
            return argparse.Namespace(returncode=0)

        loaded = argparse.Namespace(config={"eval_spatial_size": [1280, 1280], "resolution": 560})
        for case, predictions in (("empty_views", []), ("all_below_wbf_threshold", [
                {"image_id": 3, "category_id": 1, "bbox": [0, 0, 6, 4], "score": .002}])):
            with self.subTest(case=case):
                output = self.root / case
                args = argparse.Namespace(index=self.index, images=self.images, checkpoint_root=self.root,
                    manifest=self.catalog, output_dir=output, device="cpu", detector_batch=1,
                    classifier_batch=2, workers=0, codetr_command=None, legacy=False)
                with patch.object(workflow.subprocess, "run", side_effect=execute), \
                     patch.object(detectors, "load_detector", return_value=loaded), \
                     patch.object(detector_backends, "collect_codetr", return_value=predictions), \
                     patch.object(detector_backends, "collect_tinyformer", return_value=predictions), \
                     patch.object(detector_backends, "collect_rfdetr", return_value=predictions), \
                     patch.object(detector_backends, "collect_yolo26", return_value=predictions), \
                     patch.object(classifiers, "load_classifier", side_effect=AssertionError("Empty caches must not load a classifier")) as load, \
                     patch.object(classifiers, "extract_logits", side_effect=AssertionError("Empty caches must not claim forward execution")) as forward:
                    result = workflow.run(args)
                load.assert_not_called()
                forward.assert_not_called()
                self.assertEqual(result["status"], "complete")
                self.assertEqual(len(result["steps"]), 12)
                self.assertTrue(all(row["returncode"] == 0 for row in result["steps"]))
                bank = postprocessing.load_bank(output / "bank/candidates.json")
                self.assertEqual(bank["boxes_xywh"].shape, (0, 4))
                self.assertEqual(bank["scores"].shape, (0,))
                for name in workflow.HEADS:
                    manifest = output / "heads" / name / "manifest.json"
                    head = json.loads(manifest.read_text())
                    self.assertEqual(head["rows"], 0)
                    self.assertEqual(head["unique_crops"], 0)
                    self.assertEqual(head["execution"]["status"], "skipped_no_candidates")
                    self.assertFalse(head["execution"]["classifier_loaded"])
                    self.assertFalse(head["execution"]["deployment_performed"])
                    self.assertEqual(head["settings"]["precision"], "not_executed")
                    self.assertEqual(set(head["images"]), {"3"})
                    self.assertEqual(head["images"]["3"]["sha256"], classifiers.sha256(self.images / "image.png"))
                    self.assertEqual(result["sealed_heads"][name]["sha256"], classifiers.sha256(manifest))
                    values = postprocessing.load_head(manifest, 0)
                    self.assertEqual(values["logits"].shape, (0, 34))
                    self.assertEqual(values["logits"].dtype, np.float32)
                    self.assertEqual(values["objectness"].shape, (0,))
                    self.assertEqual(values["valid"].shape, (0,))
                    self.assertEqual(np.load(manifest.parent / "row_ids.npy", allow_pickle=False).shape, (0, 32))
                self.assertEqual(result["result"]["rows"], 0)
                self.assertEqual(result["result"]["cache_verification_scope"], "sealed_semantic_inputs_and_row_ids_v2")
                self.assertEqual(json.loads((output / "final/predictions.json").read_text()), [])
                with (output / "final/submission.csv").open(newline="") as handle:
                    self.assertEqual(list(csv.reader(handle)), [postprocessing.submit.HEADER])

        # Even with zero crops, sealed caches bind the real image bytes.
        Image.new("RGB", (20, 12), (99, 88, 77)).save(self.images / "image.png")
        manifests = {name: output / "heads" / name / "manifest.json" for name in workflow.HEADS}
        with self.assertRaisesRegex(ValueError, "source image identity"):
            postprocessing.run(argparse.Namespace(bank=output / "bank/candidates.json", index=self.index,
                images=self.images, checkpoint_manifest=self.catalog, output_dir=self.root / "changed-image-submit",
                fusion=classifiers.ROOT / "configs/fusion.json",
                submission_config=classifiers.ROOT / "configs/submission.json",
                hplus=manifests["hplus"], v7=manifests["v7"], siglip=manifests["siglip"],
                expected_head=[name + "=" + classifiers.sha256(path) for name, path in manifests.items()],
                cache_bindings=None, legacy=False, policy="specialist"))

    def test_empty_bank_does_not_accept_bad_index_images_catalog_or_checkpoint(self):
        self.bank.write_text("[]")
        original = json.loads(self.index.read_text())
        bad_indexes = [
            {**original, "images": []},
            {**original, "categories": []},
            {**original, "images": [{**original["images"][0], "width": 19}]},
            {**original, "images": [{**original["images"][0], "file_name": "../image.png"}]},
            {**original, "images": [{**original["images"][0], "file_name": "missing.png"}]},
        ]
        with patch.object(classifiers, "load_classifier") as load:
            for document in bad_indexes:
                with self.subTest(index=document), self.assertRaises((ValueError, FileNotFoundError)):
                    self.index.write_text(json.dumps(document))
                    classifiers.run(self.arguments("hplus"))
            self.index.write_text(json.dumps(original))
            catalog = json.loads(self.catalog.read_text())
            catalog["artifacts"]["hplus"]["sha256"] = "invalid"
            self.catalog.write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                classifiers.run(self.arguments("hplus"))
            catalog["artifacts"]["hplus"]["sha256"] = classifiers.sha256(self.root / "checkpoints/hplus.pt")
            self.catalog.write_text(json.dumps(catalog))
            (self.root / "checkpoints/hplus.pt").write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                classifiers.run(self.arguments("hplus"))
        load.assert_not_called()
        self.assertFalse((self.arguments("hplus").output_dir / "manifest.json").exists())

    def test_empty_array_bank_keeps_shape_and_rejects_malformed_shapes_or_dtypes(self):
        base = {"image_ids": np.empty(0, dtype=np.int64), "category_ids": np.empty(0, dtype=np.int16),
                "boxes_xywh": np.empty((0, 4), dtype=np.float64), "scores": np.empty(0, dtype=np.float64)}
        for case, changes, error in (("valid", {}, None),
                ("bad_id_dtype", {"image_ids": np.empty(0, dtype=np.float32)}, "integer dtypes"),
                ("bad_score_dtype", {"scores": np.empty(0, dtype=np.complex64)}, "real numeric dtypes"),
                ("bad_box_shape", {"boxes_xywh": np.empty(0, dtype=np.float64)}, "not row-aligned")):
            with self.subTest(case=case):
                directory = self.root / case
                directory.mkdir()
                document = {"rows": 0, "arrays": {name: classifiers.save_npy(directory / f"{name}.npy", values)
                                                  for name, values in {**base, **changes}.items()}}
                manifest = directory / "bank.json"
                manifest.write_text(json.dumps(document))
                if error is None:
                    self.assertEqual(classifiers.load_candidates(manifest), [])
                else:
                    with self.assertRaisesRegex(ValueError, error):
                        classifiers.load_candidates(manifest)
        for value in (None, {}, "", {"arrays": {}}):
            self.bank.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                classifiers.load_candidates(self.bank)


if __name__ == "__main__":
    unittest.main()
