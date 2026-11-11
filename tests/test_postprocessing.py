from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from postprocessing import cache_contract, submit, three_head_runtime as fusion, wbf
from postprocessing.__main__ import main
from inference.detector_geometry import ENCODED_DECODER


ROOT = Path(__file__).resolve().parents[1]


class PostprocessingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = fusion.load_policy(ROOT / "configs/fusion.json")
        self.policy = fusion.base_policy(self.config)
        self.images = self.root / "images"
        self.images.mkdir()
        Image.new("RGB", (10, 10), (32, 64, 96)).save(self.images / "test.png")
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({"images": [{"id": 1, "file_name": "test.png", "width": 10, "height": 10}],
                                         "categories": [{"id": category, "name": name} for category, name in enumerate(cache_contract.class_order(), 1)]}))
        self.catalog = self.root / "checkpoint-manifest.json"
        self.catalog.write_text(json.dumps({"schema_version": 2, "artifacts": {
            name: {"path": f"checkpoints/{name}.pt", "sha256": str(index + 1) * 64, "bytes": 100 + index}
            for index, name in enumerate(fusion.HEADS)}}))

    def bank(self, classes, scores):
        rows = len(classes)
        return {
            "image_ids": np.ones(rows, dtype=np.int64),
            "category_ids": np.asarray(classes, dtype=np.int16),
            "boxes_xywh": np.asarray([[index * 2, 0, 1, 1] for index in range(rows)], dtype=np.float64),
            "scores": np.asarray(scores, dtype=np.float64),
            "ordinals": np.arange(rows, dtype=np.int64),
        }

    def write_head(self, name, bank_path, rows=1, *, strict=False, logits=None, objectness=None):
        directory = self.root / name
        directory.mkdir()
        if logits is None:
            logits = np.zeros((rows, 34), dtype=np.float32)
            logits[:, 0] = 10 + np.arange(rows)
        np.save(directory / "class_logits.npy", logits)
        np.save(directory / "objectness_logits.npy", np.ones(rows, dtype=np.float32) if objectness is None else objectness)
        document = {"candidate_sha256": fusion.sha256(bank_path)}
        for key in ("class_logits", "objectness_logits"):
            path = directory / f"{key}.npy"
            document[key] = {"path": path.name, **cache_contract.fingerprint(path)}
        manifest = directory / "manifest.json"
        if strict:
            bank = fusion.load_bank(bank_path)
            config_path = ROOT / "configs/postprocess.yaml"
            config = next(row for row in yaml.safe_load(config_path.read_text())["dcr"]["models"] if row["name"] == name)
            images = cache_contract.image_identities(fusion.load_json(self.index), self.images, set(map(int, bank["image_ids"])))
            artifact = fusion.load_json(self.catalog)["artifacts"][name]
            inputs = {"bank": cache_contract.fingerprint(bank_path), "index": cache_contract.fingerprint(self.index),
                      "config": cache_contract.fingerprint(config_path), "checkpoint_catalog": cache_contract.fingerprint(self.catalog)}
            document.update(schema_version=2, model=name, rows=rows, model_config=config,
                            class_order=cache_contract.class_order(), class_id_space="foreground_0_based",
                            coordinate_system="encoded_image_pixels", image_decode_policy=ENCODED_DECODER, inputs=inputs,
                            index_sha256=inputs["index"]["sha256"], config_sha256=inputs["config"]["sha256"],
                            checkpoint_catalog_sha256=inputs["checkpoint_catalog"]["sha256"],
                            checkpoint_sha256=artifact["sha256"], checkpoint_bytes=artifact["bytes"], images=images,
                            deployment="merged_lora_160" if name == "hplus" else "full_model",
                            row_identity_algorithm=cache_contract.ROW_ID_ALGORITHM,
                            code={name: cache_contract.fingerprint(ROOT / name) for name in cache_contract.CODE_FILES})
            np.save(directory / "row_ids.npy", cache_contract.row_ids(bank, name, config, images, artifact["sha256"]))
            fields = ["row_ids"]
            if name == "hplus":
                np.save(directory / "valid_crops.npy", np.ones(rows, dtype=bool))
                fields.append("valid_crops")
            for field in fields:
                path = directory / f"{field}.npy"
                document[field] = {"path": path.name, **cache_contract.fingerprint(path)}
        manifest.write_text(json.dumps(document))
        return manifest

    def test_wbf_never_counts_one_source_twice_in_a_cluster(self):
        first = wbf.index_predictions([
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": score}
            for score in (0.9, 0.8)
        ])
        second = wbf.index_predictions([
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.7}
        ])
        rows = wbf.fuse([first, second], [1, 1], iou_threshold=0.5,
                        score_threshold=0.0, max_detections=100, finalize=False)
        self.assertEqual(len(rows), 2)
        np.testing.assert_allclose([row["score"] for row in rows], [0.8, 0.4])

    def test_zero_weight_siglip_still_votes_and_invalid_hplus_is_removed(self):
        heads = {}
        for name, category in (("hplus", 0), ("v7", 0), ("siglip", 1)):
            logits = np.zeros((2, 34), dtype=np.float32)
            logits[:, category] = 8
            heads[name] = {"logits": logits, "objectness": np.asarray([2, -2], dtype=np.float32),
                           "valid": np.asarray([True, name != "hplus"])}
        features = fusion.build_features(heads, self.policy)
        np.testing.assert_array_equal(features["model_top1"], [[0, 0, 1], [-1, 0, 1]])
        expected_v7 = fusion.softmax(heads["v7"]["logits"][1:2]).astype(np.float32)[0]
        np.testing.assert_allclose(features["q_all"][1], expected_v7, rtol=1e-6)
        self.assertAlmostEqual(float(features["objectness"][1]), 1 / (1 + math.exp(2)), places=6)
        heads["siglip"]["logits"][:] = heads["v7"]["logits"]
        other = fusion.build_features(heads, self.policy)
        np.testing.assert_array_equal(features["q_all"], other["q_all"])
        self.assertEqual(int(np.sum(other["model_top1"][0] == 0)), 3)
        self.assertEqual(int(np.sum(features["model_top1"][0] == 0)), 2)
        bank = self.bank([2, 2], [.8, .8])
        unchanged, _ = fusion.rescue(bank, features, fusion.ranking_scores(bank, features, self.policy), self.policy)
        rescued, _ = fusion.rescue(bank, other, fusion.ranking_scores(bank, other, self.policy), self.policy)
        np.testing.assert_array_equal(unchanged["category_ids"], [2, 2])
        np.testing.assert_array_equal(rescued["category_ids"], [2, 2, 1])

    def test_geometric_classes_are_not_renormalized_and_objectness_uses_log_odds(self):
        heads = {}
        for name, sign in (("hplus", 1), ("v7", -1), ("siglip", 1)):
            logits = np.full((1, 34), -100, dtype=np.float32)
            logits[0, :2] = [sign * math.log(4), 0]
            heads[name] = {"logits": logits, "objectness": np.asarray([sign * math.log(4)], dtype=np.float32),
                           "valid": np.ones(1, dtype=bool)}
        features = fusion.build_features(heads, self.policy)
        self.assertAlmostEqual(float(features["q_all"][0, 0]), .8**.875 * .2**.125, places=7)
        self.assertLess(float(features["q_all"].sum()), 1)
        self.assertAlmostEqual(float(features["objectness"][0]), 1 / (1 + math.exp(-.75 * math.log(4))), places=7)
        self.assertNotAlmostEqual(float(features["objectness"][0]), .875 * .8 + .125 * .2, places=3)

    def test_gaussian_decay_applies_below_the_iou_field(self):
        bank = self.bank([1, 1], [0.9, 0.8])
        bank["boxes_xywh"] = np.asarray([[0, 0, 10, 10], [6, 0, 10, 10]], dtype=np.float64)
        rows = fusion.finalize(bank, bank["scores"], self.policy)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[1]["score"], 0.8 * math.exp(-(0.25 ** 2) / 0.25))

    def test_specialist_keeps_class_candidate_below_other_class_global_top100(self):
        bank = self.bank([3] * 101 + [1], [0.9] * 101 + [0.1])
        q = np.full((102, 34), 0.001, dtype=np.float32)
        q[:, 1] = 0.9
        features = {"q_all": q, "objectness": np.ones(102, dtype=np.float32),
                    "top1": np.ones(102, dtype=np.int16),
                    "model_top1": np.ones((102, 3), dtype=np.int16)}
        policy = {**self.policy, "beta": 0.0, "ranking": {"objectness": 0.0}, "rescue": {"mode": "off"}}
        overrides = {"2": {"mode": "swap", "consensus": 3, "ratio": 1.1, "qmin": 0.2, "gamma": 1.0}}
        # Under the selected class-2 policy, all its rows become emitted class 1.
        # The class-0 specialist must retain its own low row until class selection.
        self.assertEqual(len(fusion.apply_policy(bank, features, policy)), 100)
        self.assertEqual(len(fusion.apply_policy(bank, features, policy, per_class=True)), 101)
        rows = fusion.specialist_predictions(bank, features, policy, overrides)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["category_id"], 1)
        self.assertEqual(rows[0]["bbox"], bank["boxes_xywh"][-1].tolist())

    def test_merge_uses_class_order_for_ties_and_float32_boxes(self):
        box = [0.123456789, 0, 1, 1]
        predictions = {
            "b": [{"image_id": 1, "category_id": 2, "bbox": box, "score": 0.5}],
            "a": [{"image_id": 1, "category_id": 1, "bbox": box, "score": 0.5}],
        }
        rows = fusion.merge_classes(predictions, {"1": "b", "0": "a"})
        self.assertEqual([row["category_id"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["bbox"][0], float(np.float32(box[0])))

    def test_cache_alignment_and_array_hash_reject_changed_inputs(self):
        bank = self.root / "candidates.json"
        bank.write_text("[]")
        head = self.write_head("head", bank)
        fusion.check_alignment(bank, head, {})
        self.assertEqual(fusion.load_head(head, 1)["logits"].shape, (1, 34))
        bank.write_text("[{}]")
        with self.assertRaisesRegex(ValueError, "candidate identity mismatch"):
            fusion.check_alignment(bank, head, {})
        (head.parent / "class_logits.npy").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            fusion.load_head(head, 1)

    def test_csv_clips_sorts_quotes_and_formats_six_decimals(self):
        rows = [submit.submission_row("b.jpg", 2, (0, 0, 1, 1), 0.2, 10, 10),
                submit.submission_row("a,jpg", 0, (-1, 1, 5, 12), 0.9, 10, 10)]
        path = self.root / "submission.csv"
        submit.write_submission(path, rows)
        expected = ("image_filename,label_id,x,y,w,h,confidence\r\n"
                    '"a,jpg",0,0.000000,1.000000,5.000000,9.000000,0.900000\r\n'
                    "b.jpg,2,0.000000,0.000000,1.000000,1.000000,0.200000\r\n")
        self.assertEqual(path.read_bytes(), expected.encode())
        with self.assertRaises(FileExistsError):
            submit.write_submission(path, rows)

    def test_raw_json_to_bound_cache_to_csv_cli(self):
        index = self.index
        raw = self.root / "raw.json"
        raw.write_text(json.dumps([{"image_id": 1, "category_id": 1, "bbox": [-1, -1, 5, 5], "score": 0.8}]))
        bank_dir = self.root / "bank"
        args = ["fuse", "--index", str(index), "--output-dir", str(bank_dir)]
        for name in ("codetr_original", "codetr_flip", "tinyformer_original", "rfdetr_original", "rfdetr_flip", "yolo26_original", "yolo26_flip"):
            args.extend(["--prediction", f"{name}={raw}"])
        main(args)
        bank = bank_dir / "candidates.json"
        self.assertEqual(len(fusion.load_bank(bank)["scores"]), 1)
        args = ["submit", "--bank", str(bank), "--index", str(index), "--images", str(self.images),
                "--checkpoint-manifest", str(self.catalog)]
        for name in fusion.HEADS:
            head = self.write_head(name, bank, strict=True)
            args.extend([f"--{name}", str(head), "--expected-head", name + "=" + fusion.sha256(head)])
        output = self.root / "output"
        main([*args, "--output-dir", str(output)])
        with (output / "submission.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label_id"], "0")
        self.assertEqual(rows[0]["x"], "0.000000")
        self.assertEqual(rows[0]["w"], "4.000000")
        manifest = fusion.load_json(output / "manifest.json")
        self.assertEqual(manifest["submission_sha256"], fusion.sha256(output / "submission.csv"))
        with self.assertRaises(FileExistsError):
            main([*args, "--output-dir", str(output)])

    def test_global_and_specialist_share_softnms_threshold_ties_and_final_limit(self):
        bank = self.bank([1] * 102 + [2] * 102, [0.5] * 204)
        policy = {**self.policy, "beta": 0.0, "ranking": {"objectness": 0.0}, "rescue": {"mode": "off"}}
        q = np.full((204, 34), 1 / 34, dtype=np.float32)
        features = {"q_all": q, "objectness": np.ones(204, dtype=np.float32),
                    "top1": np.zeros(204, dtype=np.int16), "model_top1": np.zeros((204, 3), dtype=np.int16)}
        for _ in range(3):
            global_rows = fusion.apply_policy(bank, features, policy)
            specialist = fusion.specialist_predictions(bank, features, policy, {})
            self.assertEqual(global_rows, specialist)
            self.assertEqual(len(global_rows), 100)
            self.assertEqual([row["bbox"][0] for row in global_rows], list(map(float, range(0, 200, 2))))
        bank["scores"][:] = 0.0009
        self.assertEqual(fusion.apply_policy(bank, features, policy), [])
        self.assertEqual(fusion.specialist_predictions(bank, features, policy, {}), [])
        bank = self.bank([2, 1], [.5, .5])
        features = {key: value[:2] for key, value in features.items()}
        self.assertEqual(fusion.apply_policy(bank, features, policy),
                         fusion.specialist_predictions(bank, features, policy, {}))
        self.assertEqual([row["category_id"] for row in fusion.apply_policy(bank, features, policy)], [1, 2])

    def strict_bundle(self):
        Image.new("RGB", (10, 10), (100, 80, 60)).save(self.images / "second.png")
        index = fusion.load_json(self.index)
        index["images"].append({"id": 2, "file_name": "second.png", "width": 10, "height": 10})
        self.index.write_text(json.dumps(index))
        bank = self.root / "candidates.json"
        bank.write_text(json.dumps([
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4], "score": .8, "ordinal": 0},
            {"image_id": 2, "category_id": 2, "bbox": [2, 0, 4, 4], "score": .7, "ordinal": 1},
        ]))
        heads = {name: self.write_head(name, bank, rows=2, strict=True) for name in fusion.HEADS}
        seals = {name: fusion.sha256(path) for name, path in heads.items()}
        return bank, heads, seals

    def verify_cache(self, bank, head, model, seal):
        arrays = fusion.load_bank(bank)
        images = cache_contract.image_identities(fusion.load_json(self.index), self.images, set(map(int, arrays["image_ids"])))
        return cache_contract.verify_cache(head, model, bank, arrays, self.index, images, self.catalog, seal)

    def test_cache_row_permutation_is_rejected_even_with_updated_array_and_manifest_hashes(self):
        bank, heads, seals = self.strict_bundle()
        self.verify_cache(bank, heads["hplus"], "hplus", seals["hplus"])
        head = heads["hplus"]
        document = fusion.load_json(head)
        for field in ("class_logits", "objectness_logits", "row_ids", "valid_crops"):
            path = head.parent / document[field]["path"]
            values = np.load(path, allow_pickle=False)
            np.save(path, values[::-1])
            document[field].update(cache_contract.fingerprint(path))
        head.write_text(json.dumps(document))
        # Even granting a new self-consistent seal, bank-derived row identities disagree.
        with self.assertRaisesRegex(ValueError, "candidate/crop row order"):
            self.verify_cache(bank, head, "hplus", fusion.sha256(head))

    def test_cache_logits_only_permutation_cannot_replace_the_independent_producer_seal(self):
        bank, heads, seals = self.strict_bundle()
        head = heads["v7"]
        document = fusion.load_json(head)
        path = head.parent / document["class_logits"]["path"]
        values = np.load(path, allow_pickle=False)
        np.save(path, values[::-1])
        document["class_logits"].update(cache_contract.fingerprint(path))
        head.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "Sealed head manifest"):
            self.verify_cache(bank, head, "v7", seals["v7"])

    def test_cache_bank_row_permutation_is_rejected_after_rebinding_its_json_sha(self):
        bank, heads, _ = self.strict_bundle()
        rows = fusion.load_json(bank)
        bank.write_text(json.dumps(rows[::-1]))
        head = heads["v7"]
        document = fusion.load_json(head)
        document["candidate_sha256"] = fusion.sha256(bank)
        document["inputs"]["bank"] = cache_contract.fingerprint(bank)
        head.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "candidate/crop row order"):
            self.verify_cache(bank, head, "v7", fusion.sha256(head))

    def test_cache_rejects_changed_source_image_bytes_with_unchanged_dimensions(self):
        bank, heads, seals = self.strict_bundle()
        Image.new("RGB", (10, 10), (99, 88, 77)).save(self.images / "second.png")
        with self.assertRaisesRegex(ValueError, "source image identity"):
            self.verify_cache(bank, heads["siglip"], "siglip", seals["siglip"])

    def test_cache_rejects_changed_catalog_even_when_head_checkpoint_entry_is_unchanged(self):
        bank, heads, seals = self.strict_bundle()
        document = fusion.load_json(self.catalog)
        document["artifacts"]["siglip"]["sha256"] = "f" * 64
        self.catalog.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "checkpoint catalog identity"):
            self.verify_cache(bank, heads["hplus"], "hplus", seals["hplus"])

    def test_cache_rejects_resealed_wrong_head_class_order_checkpoint_and_crop_configuration(self):
        bank, heads, _ = self.strict_bundle()
        head = heads["hplus"]
        original = fusion.load_json(head)
        variants = [({**original, "model": "v7"}, "model hplus"),
                    ({**original, "class_order": original["class_order"][::-1]}, "class order"),
                    ({**original, "checkpoint_sha256": "f" * 64}, "checkpoint identity"),
                    ({**original, "model_config": {**original["model_config"], "context_scale": 1.1}}, "crop/model configuration")]
        for document, message in variants:
            with self.subTest(message=message):
                head.write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, message):
                    self.verify_cache(bank, head, "hplus", fusion.sha256(head))

    def test_legacy_policy_is_explicit_and_submission_has_no_recorded_results(self):
        bank = self.bank([3] * 101 + [1], [.9] * 101 + [.1])
        q = np.full((102, 34), .001, dtype=np.float32)
        q[:, 1] = .9
        features = {"q_all": q, "objectness": np.ones(102), "top1": np.ones(102, dtype=np.int16),
                    "model_top1": np.ones((102, 3), dtype=np.int16)}
        policy = {**self.policy, "beta": 0., "ranking": {"objectness": 0.}, "rescue": {"mode": "off"}}
        overrides = {"2": {"mode": "swap", "consensus": 3, "ratio": 1.1, "qmin": .2, "gamma": 1.}}
        self.assertEqual(fusion.specialist_predictions(bank, features, policy, overrides, legacy_policy_global_top100=True), [])
        submission = fusion.load_submission_config(ROOT / "configs/submission.json")
        self.assertEqual(set(submission), {
            "schema_version", "columns", "label_id_range", "box_format",
            "coordinate_system", "clip_to_image_bounds", "decimal_places",
            "max_detections_per_image",
        })

    def legacy_cli_bundle(self, bank):
        heads, bindings = {}, {}
        for name in fusion.HEADS:
            head = self.write_head(name, bank)
            document = fusion.load_json(head)
            document.pop("candidate_sha256")
            document.update(schema_version=1, classes=34, rows=1)
            head.write_text(json.dumps(document))
            heads[name] = head
            # These receipts are sealed by the fixture's producer before any replacement.
            bindings[name] = {"bank_sha256": fusion.sha256(bank), "manifest_sha256": fusion.sha256(head)}
        receipt = self.root / "trusted-producer-bindings.json"
        receipt.write_text(json.dumps(bindings))
        arguments = ["submit", "--legacy", "--bank", str(bank), "--index", str(self.index),
                     "--cache-bindings", str(receipt)]
        for name, head in heads.items():
            arguments += ["--" + name, str(head)]
        return arguments, heads, receipt

    def test_minimal_legacy_manifest_binding_accepts_v7_without_row_ids_and_rejects_replaced_array(self):
        bank = self.root / "legacy-bank.json"
        bank.write_text(json.dumps([{"image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4], "score": .8}]))
        arguments, heads, receipt = self.legacy_cli_bundle(bank)
        original_receipt = receipt.read_bytes()
        self.assertFalse({"candidate_sha256", "candidate_ids", "row_ids"} & set(fusion.load_json(heads["v7"])))
        output = self.root / "legacy-good"
        main([*arguments, "--output-dir", str(output)])
        result = fusion.load_json(output / "manifest.json")
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["policy"], "legacy_class_specialist_policy_global_top100")
        self.assertEqual(result["cache_verification_scope"], "legacy_independent_bindings")
        # Same shape/dtype; consumer keeps the original independent receipt and sealed descriptor.
        np.save(heads["v7"].parent / "class_logits.npy", np.zeros((1, 34), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "artifact SHA256 mismatch"):
            main([*arguments, "--output-dir", str(self.root / "legacy-replaced")])
        self.assertEqual(receipt.read_bytes(), original_receipt)
        self.assertFalse((self.root / "legacy-replaced").exists())

    def test_sealed_legacy_manifest_does_not_override_mismatched_candidate_ids(self):
        arrays = {"image_ids": np.asarray([1], dtype=np.int64), "category_ids": np.asarray([1], dtype=np.int16),
                  "boxes_xywh": np.asarray([[0, 0, 4, 4]], dtype=np.float64), "scores": np.asarray([.8]),
                  "candidate_ids": np.asarray([[1, 1, 0]], dtype=np.int64)}
        descriptors = {}
        for name, values in arrays.items():
            path = self.root / f"bank-{name}.npy"
            np.save(path, values)
            descriptors[name] = {"path": path.name, **cache_contract.fingerprint(path)}
        bank = self.root / "legacy-bank.json"
        bank.write_text(json.dumps({"rows": 1, "arrays": descriptors}))
        arguments, heads, receipt = self.legacy_cli_bundle(bank)
        head = heads["v7"]
        ids = head.parent / "candidate_ids.npy"
        np.save(ids, np.asarray([[1, 1, 1]], dtype=np.int64))
        document = fusion.load_json(head)
        document["candidate_ids"] = {"path": ids.name, **cache_contract.fingerprint(ids)}
        head.write_text(json.dumps(document))
        # Even an explicitly trusted manifest cannot waive bank/head candidate-ID equality.
        bindings = fusion.load_json(receipt)
        bindings["v7"]["manifest_sha256"] = fusion.sha256(head)
        receipt.write_text(json.dumps(bindings))
        with self.assertRaisesRegex(ValueError, "candidate order mismatch"):
            main([*arguments, "--output-dir", str(self.root / "legacy-wrong-ids")])

    def test_rescue_swap_requires_all_gates_and_applies_gamma_without_retaining_original(self):
        bank = self.bank([1] * 5, [.8] * 5)
        q = np.full((5, 34), .001, dtype=np.float32)
        q[:, 0], q[:, 1] = .2, .8
        q[1, 1], q[2, 0] = .3, .6
        votes = np.ones((5, 3), dtype=np.int16)
        votes[3, 2] = 0
        features = {"q_all": q, "objectness": np.full(5, .5, dtype=np.float32),
                    "top1": np.asarray([1, 1, 1, 1, 0]), "model_top1": votes}
        policy = {**self.policy, "rescue": {"mode": "swap", "consensus": 3, "ratio": 2, "qmin": .4, "gamma": .5}}
        scores = fusion.ranking_scores(bank, features, policy)
        changed, output = fusion.rescue(bank, features, scores, policy)
        np.testing.assert_array_equal(changed["category_ids"], [2, 1, 1, 1, 1])
        self.assertEqual(len(output), 5)
        self.assertAlmostEqual(output[0], .8 * float(q[0, 1])**.375 * .5**.125 * .5)
        np.testing.assert_array_equal(output[1:], scores[1:])

    def test_full_seven_view_wbf_dcr_rescue_softnms_csv_golden(self):
        Image.new("RGB", (40, 30), (32, 64, 96)).save(self.images / "test.png")
        index = fusion.load_json(self.index)
        index["images"][0].update(width=40, height=30)
        self.index.write_text(json.dumps(index))
        arguments = ["fuse", "--index", str(self.index), "--output-dir", str(self.root / "golden-bank")]
        # Unequal view scores have a known weighted mean under .5/.5,
        # .65/.35 and .35/.65; reversing TTA weights changes the golden bank.
        for model, scale, shift, views in (("codetr", 1.25, 0, (("original", .05), ("flip", -.05))),
                                          ("tinyformer", 1.325, 2, (("original", 0),)),
                                          ("rfdetr", 1, 0, (("original", .035), ("flip", -.065))),
                                          ("yolo26", 1.05, 2, (("original", .065), ("flip", -.035)))):
            for view, offset in views:
                raw = self.root / f"{model}-{view}.json"
                raw.write_text(json.dumps([{"image_id": 1, "category_id": 1, "bbox": [x + shift, 0, 10, 10], "score": (score + offset) / scale}
                                           for x, score in ((0, 0.5), (6, 0.4))]))
                arguments += ["--prediction", f"{model}_{view}={raw}"]
        main(arguments)
        bank_path = self.root / "golden-bank/candidates.json"
        bank = fusion.load_bank(bank_path)
        np.testing.assert_allclose(bank["scores"], [.5125, .41], rtol=0, atol=1e-14)
        np.testing.assert_allclose(bank["boxes_xywh"], [[.7918748261457332, 0, 10, 10], [6.791874826145733, 0, 10, 10]], rtol=0, atol=1e-14)
        logits = np.full((2, 34), -100, dtype=np.float32)
        logits[:, 0], logits[:, 1] = 0, np.float32(math.log(3))
        arguments = ["submit", "--bank", str(bank_path), "--index", str(self.index), "--images", str(self.images),
                     "--checkpoint-manifest", str(self.catalog)]
        for name in fusion.HEADS:
            head = self.write_head(name, bank_path, rows=2, strict=True, logits=logits, objectness=np.zeros(2, dtype=np.float32))
            arguments += [f"--{name}", str(head), "--expected-head", name + "=" + fusion.sha256(head)]
        expected = []
        # Hand-calculated specification: q=(.25,.75), objectness=.5;
        # class-1 rescue gamma=.75; IoU=.25 => second-box decay exp(-.25).
        for label, confidence in ((1, .5125 * .75**.375 * .5**.125 * .75),
                                  (0, .5125 * .25**.375 * .5**.125),
                                  (1, .41 * .75**.375 * .5**.125 * .75 * math.exp(-.25)),
                                  (0, .41 * .25**.375 * .5**.125 * math.exp(-.25))):
            x = ".791875" if len(expected) < 2 else "6.791875"
            expected.append(f"test.png,{label},{float(x):.6f},0.000000,10.000000,10.000000,{confidence:.6f}\r\n")
        expected_csv = (",".join(submit.HEADER) + "\r\n" + "".join(expected)).encode()
        for suffix in ("a", "b"):
            output = self.root / ("golden-" + suffix)
            main([*arguments, "--output-dir", str(output)])
            self.assertEqual((output / "submission.csv").read_bytes(), expected_csv)

    def test_wbf_iou_overrides_use_one_based_categories(self):
        rows = [{"image_id": 1, "category_id": category, "bbox": [x, 0, 10, 10], "score": .8}
                for category in (1, 2) for x in (0,)]
        first = wbf.index_predictions(rows)
        second = wbf.index_predictions([{**row, "bbox": [4, 0, 10, 10]} for row in rows])
        result = wbf.fuse([first, second], [1, 1], iou_threshold=.5, category_iou_thresholds={2: .325},
                          score_threshold=0, max_detections=100, finalize=False)
        self.assertEqual([row["category_id"] for row in result], [1, 1, 2])

    def test_unsupported_numerical_contract_is_rejected(self):
        self.config["numerics"]["normalize_fused_class_probabilities"] = True
        path = self.root / "fusion.json"
        path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "numerical contract"):
            fusion.load_policy(path)


if __name__ == "__main__":
    unittest.main()
