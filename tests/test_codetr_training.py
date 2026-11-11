from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
from PIL import Image
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
HAS_MMCV = importlib.util.find_spec("mmcv") is not None
if HAS_MMCV:
    sys.path.insert(0, str(ROOT / "third_party/Co-DETR"))
    from mmcv import Config
    from mmdet.datasets import build_dataset
    from training.codetr_runtime import CompetitionSAMCopyPasteEpochHook, CompetitionTerminalExportHook, _OptimizerBoundedLoader
    from training.train_codetr import configure


@unittest.skipUnless(HAS_MMCV, "Run this integration check with python scripts/codetr_python.py")
class CoDETRTrainingTest(unittest.TestCase):
    def test_configured_training_loader_keeps_exif_1_to_8_pixels_and_gt_in_stored_frame(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-codetr-exif-") as directory:
            root = Path(directory)
            categories = json.loads((ROOT / "configs/categories.json").read_text())["categories"]
            recipe = yaml.safe_load((ROOT / "training/final_recipes.yaml").read_text())["detectors"]["codetr"]
            document = {"images": [], "annotations": [], "categories": categories}
            expected_pixels = {}
            for group in (0, 1):
                for orientation in range(1, 9):
                    global_id = len(document["images"]) + 1
                    path = root / f"group-{group}-exif-{orientation}.jpg"
                    image = Image.new("RGB", (160, 96), (8, 16, 24))
                    image.paste((0, 255, 0), (35, 22, 64, 45))
                    image.paste((255, 0, 0), (120, 70, 144, 88))
                    exif = Image.Exif()
                    exif[274] = orientation
                    image.save(path, quality=100, subsampling=0, exif=exif)
                    with Image.open(path) as encoded:
                        self.assertEqual(encoded.getexif()[274], orientation)
                        expected_pixels[global_id] = np.asarray(encoded.convert("RGB")).copy()
                    document["images"].append({
                        "id": global_id, "source_image_id": orientation, "source_group": group,
                        "file_name": path.name, "width": 160, "height": 96,
                    })
                    document["annotations"].append({
                        "id": global_id, "source_annotation_id": orientation, "image_id": global_id,
                        "category_id": 1, "bbox": [35, 22, 29, 23], "area": 667, "iscrowd": 0,
                    })
            one_based = root / "union.json"
            zero_based = root / "union-zero.json"
            one_based.write_text(json.dumps(document))
            zero_based.write_text(json.dumps({
                **document,
                "categories": [{**row, "id": row["id"] - 1} for row in categories],
                "annotations": [{**row, "category_id": row["category_id"] - 1} for row in document["annotations"]],
            }))
            bank = root / "bank"
            bank.mkdir()
            data = {
                "class_names": [row["name"] for row in categories],
                "coco": {"images": str(root), "annotations_1based": str(one_based), "annotations_0based": str(zero_based)},
                "tinyformer": {"sam_object_bank": str(bank)},
            }
            native = Config.fromfile(str(ROOT / recipe["source_config"]))
            original_pipeline = copy.deepcopy(native.data.train.pipeline)
            original_inference = copy.deepcopy({key: native.data[key] for key in ("val", "test")})
            configure(native, recipe, data, root / "data.yaml", root / "unused-initial.pth", root / "output")
            expected_pipeline = copy.deepcopy(original_pipeline)
            expected_pipeline[0]["color_type"] = "color_ignore_orientation"
            self.assertEqual(native.data.train.pipeline[:2] + native.data.train.pipeline[3:], expected_pipeline)
            self.assertEqual(native.data.train.pipeline[2]["type"], "CompetitionSAMCopyPaste")
            self.assertEqual({key: native.data[key] for key in ("val", "test")}, original_inference)
            dataset = build_dataset(native.data.train)
            alignment = recipe["batch_per_device"] * recipe["gradient_accumulation"] * recipe["devices"]
            self.assertEqual([len(child) for child in dataset.datasets], [alignment, alignment])
            for group, child in enumerate(dataset.datasets):
                for index in range(8):
                    with self.subTest(group=group, orientation=index + 1):
                        info = child.data_infos[index]
                        global_id = group * 8 + index + 1
                        self.assertEqual((info["id"], info["sam_image_id"]), (index + 1, global_id))
                        self.assertEqual((info["width"], info["height"]), (160, 96))
                        results = dict(img_info=info, ann_info=child.get_ann_info(index))
                        child.pre_pipeline(results)
                        for transform in child.pipeline.transforms[:2]:
                            results = transform(results)
                        self.assertEqual(results["img_shape"], (96, 160, 3))
                        self.assertEqual(results["ori_shape"], (96, 160, 3))
                        np.testing.assert_allclose(results["img"][..., ::-1], expected_pixels[global_id], atol=3, rtol=0)
                        np.testing.assert_array_equal(results["gt_bboxes"], [[35, 22, 64, 45]])
                        np.testing.assert_array_equal(results["gt_labels"], [0])
                        self.assertGreater(results["img"][25:42, 38:61, 1].min(), 245)

    def test_padded_concat_slots_share_one_plan_and_keep_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-codetr-") as directory:
            root = Path(directory)
            categories = json.loads((ROOT / "configs/categories.json").read_text())["categories"]
            zero_annotations = root / "union-zero.json"
            union = {"images": [], "annotations": [], "categories": [{**row, "id": row["id"] - 1} for row in categories]}
            source_files = []
            for group, source_ids in ((0, [42, 43, 44]), (1, [42, 43])):
                images, annotations = [], []
                for local_index, source_id in enumerate(source_ids, 1):
                    union_id = len(union["images"]) + 1
                    image = {"id": source_id, "sam_image_id": union_id, "source_group": group, "file_name": f"{group}-{source_id}.jpg", "width": 32, "height": 32}
                    annotation = {"id": local_index, "image_id": source_id, "category_id": 1, "bbox": [1, 1, 3, 3], "area": 9, "iscrowd": 0}
                    images.append(image)
                    annotations.append(annotation)
                    union["images"].append({**image, "id": union_id})
                    union["annotations"].append({**annotation, "id": union_id, "image_id": union_id, "category_id": 0})
                path = root / ("group-" + str(group) + ".json")
                path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
                source_files.append(str(path))
            zero_annotations.write_text(json.dumps(union))
            Image.new("RGBA", (8, 8), (0, 255, 0, 255)).save(root / "object.png")
            (root / "class.json").write_text(json.dumps([{"annotation_id": 1, "image_id": 1, "class_id": 0, "rgba_path": "object.png"}]))
            (root / "pipeline.json").write_text(json.dumps({"schema_version": 2, "source": {"annotations": str(zero_annotations)}, "object_bank": {"mode": "sam3_rgba_only", "files": {"0": "class.json"}}}))
            recipe = yaml.safe_load((ROOT / "training/final_recipes.yaml").read_text())["detectors"]["codetr"]["sam_copy_paste"]
            config = {**recipe, "artifact_root": str(root), "bank_pipeline_sha256": None, "bank_objects": 1, "base_images": 8, "class_weights": {0: 1.0}}
            dataset = build_dataset(dict(
                type="CompetitionTrainCocoDataset", ann_file=source_files,
                img_prefix=[str(root)] * 2, classes=tuple(row["name"] for row in categories),
                filter_empty_gt=False, accumulation_alignment=4,
                pipeline=[dict(type="CompetitionSAMCopyPaste", annotation_file=str(zero_annotations), **config)],
            ))
            first = dataset.datasets[0]
            self.assertIs(first.data_infos[0], first.data_infos[3])
            flags = dataset.flag.copy()
            model = torch.nn.Linear(2, 2)
            runner = types.SimpleNamespace(data_loader=torch.utils.data.DataLoader(dataset, batch_size=1), epoch=10, meta=None, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), iter=0, work_dir=str(root))
            hook = CompetitionSAMCopyPasteEpochHook()
            hook.before_train_epoch(runner)
            self.assertEqual(hook.plan.ids, [1, 2, 3, 1, 4, 5, 4, 5])
            self.assertEqual(hook.plan.stats["base_images"], 8)
            self.assertIsNot(first.data_infos[0], first.data_infos[3])
            self.assertEqual([row["sam_sample_index"] for child in dataset.datasets for row in child.data_infos], list(range(8)))
            self.assertTrue(all(child.pipeline.transforms[0].plan is hook.plan for child in dataset.datasets))
            np.testing.assert_array_equal(dataset.flag, flags)
            calls = []
            def record_paste(image, annotations, entries, rng):
                calls.append(entries)
                return image, annotations
            transform = first.pipeline.transforms[0]
            transform.plan = types.SimpleNamespace(_paste_plan={0: ["original"], 3: ["repeat"]}, _paste_objects=record_paste, config={"seed": 42}, epoch=10)
            for index in (0, 3):
                transform({"img_info": first.data_infos[index], "img": np.zeros((32, 32, 3), dtype=np.uint8), "gt_bboxes": np.empty((0, 4), dtype=np.float32), "gt_labels": np.empty(0, dtype=np.int64)})
            self.assertEqual(calls, [["original"], ["repeat"]])
            transform.plan = hook.plan
            runner.epoch = 19
            hook.before_train_epoch(runner)
            self.assertEqual(hook.plan.stats["scheduled_objects"], 0)
            self.assertEqual(runner.meta["sam_copy_paste"], hook.plan.stats)
            runner.epoch = 20
            CompetitionTerminalExportHook([row["name"] for row in categories]).after_run(runner)
            checkpoint = torch.load(root / "checkpoint_terminal_ema.pth", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["meta"]["sam_copy_paste"], hook.plan.stats)
            self.assertEqual(set(checkpoint["state_dict"]), set(model.state_dict()))

    def test_bounded_loader_keeps_full_dataset_and_stops_at_successful_update(self):
        consumed = []

        class Loader:
            dataset = object()
            sampler = object()

            def __len__(self):
                return 24

            def __iter__(self):
                for index in range(24):
                    consumed.append(index)
                    yield index

        source = Loader()
        runner = types.SimpleNamespace(meta={"codetr_optimizer_steps": 0})
        bounded = _OptimizerBoundedLoader(source, runner, 2)
        self.assertIs(bounded.dataset, source.dataset)
        self.assertIs(bounded.sampler, source.sampler)
        self.assertEqual(len(bounded), 24)
        seen = []
        for batch in bounded:
            seen.append(batch)
            # The first complete accumulation window is skipped, then two succeed.
            if batch in (7, 11):
                runner.meta["codetr_optimizer_steps"] += 1
        self.assertEqual(seen, list(range(12)))
        self.assertEqual(consumed, seen)


if __name__ == "__main__":
    unittest.main()
