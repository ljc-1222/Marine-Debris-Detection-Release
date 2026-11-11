from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scoring.score import load_annotations, load_submission, main, score_submission, sha256
from postprocessing.submit import HEADER


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotations = self.root / "synthetic.json"
        self.submission = self.root / "submission.csv"
        self.document = {
            "images": [{"id": 1, "file_name": "one.jpg", "width": 100, "height": 100}],
            "categories": [{"id": category, "name": str(category - 1)} for category in range(1, 35)],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                             "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}],
        }

    def write(self, rows):
        self.annotations.write_text(json.dumps(self.document))
        with self.submission.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerows(rows)

    def score(self):
        return score_submission(self.annotations, self.submission, split="synthetic",
                                ground_truth_source="test fixture; not competition labels")

    def test_perfect_match_uses_coco_absent_class_semantics_and_provenance(self):
        self.write([["one.jpg", 0, 0, 0, 10, 10, 0.9]])
        result = self.score()
        self.assertAlmostEqual(result["mAP50"], 1.0)
        self.assertEqual(result["declared_categories"], 34)
        self.assertEqual(result["categories_with_evaluable_ground_truth"], 1)
        self.assertFalse(result["official_leaderboard_score"])
        self.assertEqual(result["annotations"]["sha256"], sha256(self.annotations))
        self.assertEqual(result["split"], "synthetic")

    def test_false_positive_before_true_positive_changes_precision(self):
        self.write([["one.jpg", 0, 50, 50, 10, 10, 0.95],
                    ["one.jpg", 0, 0, 0, 10, 10, 0.9]])
        self.assertAlmostEqual(self.score()["mAP50"], 0.5)

    def test_duplicate_does_not_match_the_same_non_crowd_object_twice(self):
        self.document["annotations"].append({"id": 2, "image_id": 1, "category_id": 1,
                                              "bbox": [20, 20, 10, 10], "area": 100, "iscrowd": 0})
        self.write([["one.jpg", 0, 0, 0, 10, 10, 0.9],
                    ["one.jpg", 0, 0, 0, 10, 10, 0.85],
                    ["one.jpg", 0, 20, 20, 10, 10, 0.8]])
        self.assertAlmostEqual(self.score()["mAP50"], (51 + 50 * 2 / 3) / 101)

    def test_coco_accepts_exact_iou_half_and_rejects_less(self):
        self.write([["one.jpg", 0, 0, 0, 5, 10, 0.9]])
        self.assertAlmostEqual(self.score()["mAP50"], 1.0)
        self.write([["one.jpg", 0, 0, 0, 4.9, 10, 0.9]])
        self.assertEqual(self.score()["mAP50"], 0.0)

    def test_all_gt_images_are_scored_even_when_csv_omits_an_image(self):
        self.document["images"].append({"id": 2, "file_name": "two.jpg", "width": 100, "height": 100})
        self.document["annotations"].append({"id": 2, "image_id": 2, "category_id": 1,
                                              "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0})
        self.write([["one.jpg", 0, 0, 0, 10, 10, 0.9]])
        result = self.score()
        self.assertAlmostEqual(result["mAP50"], 51 / 101)
        self.assertEqual(result["images"], 2)

    def test_empty_submission_is_zero_but_no_evaluable_gt_is_undefined(self):
        self.write([])
        self.assertEqual(self.score()["mAP50"], 0.0)
        self.document["annotations"] = []
        self.write([])
        with self.assertRaisesRegex(ValueError, "no evaluable ground truth"):
            self.score()

    def test_scoring_does_not_clip_or_silently_drop_provided_predictions(self):
        self.write([["one.jpg", 0, -20, 0, 30, 10, 0.9]])
        self.assertEqual(self.score()["mAP50"], 0.0)
        rows = load_submission(self.submission, self.document)
        self.assertEqual(rows[0]["bbox"], [-20, 0, 30, 10])

    def test_invalid_csv_identity_category_and_numbers_are_rejected(self):
        for row in (["missing.jpg", 0, 0, 0, 10, 10, 0.9],
                    ["one.jpg", 34, 0, 0, 10, 10, 0.9],
                    ["one.jpg", 0, 0, 0, 0, 10, 0.9],
                    ["one.jpg", 0, 0, 0, 10, 10, "nan"]):
            with self.subTest(row=row):
                self.write([row])
                with self.assertRaises(ValueError):
                    load_submission(self.submission, self.document)
        self.submission.write_text("image_filename,label_id\none.jpg,0\n")
        with self.assertRaisesRegex(ValueError, "columns"):
            load_submission(self.submission, self.document)

    def test_ambiguous_ground_truth_ids_are_rejected(self):
        self.write([])
        self.document["images"].append(dict(self.document["images"][0]))
        self.annotations.write_text(json.dumps(self.document))
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            load_annotations(self.annotations)
        self.document["images"].pop()
        self.document["annotations"][0]["id"] = 0
        self.annotations.write_text(json.dumps(self.document))
        with self.assertRaisesRegex(ValueError, "positive"):
            load_annotations(self.annotations)

    def test_cli_writes_only_final_report_and_refuses_overwrite(self):
        self.write([["one.jpg", 0, 0, 0, 10, 10, 0.9]])
        output = self.root / "score.json"
        args = ["--annotations", str(self.annotations), "--submission", str(self.submission),
                "--split", "synthetic", "--ground-truth-source", "test fixture", "--output", str(output)]
        main(args)
        result = json.loads(output.read_text())
        self.assertEqual(result["metric"], "COCO bbox mAP@0.5")
        self.assertNotIn("per_class", result)
        self.assertNotIn("precision", result)
        with self.assertRaises(FileExistsError):
            main(args)


if __name__ == "__main__":
    unittest.main()
