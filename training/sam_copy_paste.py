"""Online SAM object-bank Copy-Paste for the fixed detector recipe."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image, ImageStat


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _xywh_iou(first: list[float], second: list[float]) -> float:
    ax, ay, aw, ah = map(float, first)
    bx, by, bw, bh = map(float, second)
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / max(union, 1e-12)


class SAMCopyPastePlan:
    """Plan deterministic object-bank pastes for one training epoch."""

    def __init__(
        self,
        config: Mapping[str, Any],
        annotation_file: Path,
        image_ids: list[int],
    ) -> None:
        self.config = dict(config)
        self.ids = list(map(int, image_ids))

        self._artifact_root = Path(self.config["artifact_root"]).resolve()
        manifest = _load_json(self._artifact_root / "pipeline.json")
        if manifest.get("schema_version") not in {1, 2, 3, 4}:
            raise ValueError("Unsupported SAM object-bank pipeline schema")
        if manifest.get("schema_version") in {3, 4}:
            from training.portable_sam_bank import verify_local_bank
            verify_local_bank(self._artifact_root, manifest, annotation_file,
                expected_pipeline_sha256=self.config.get("bank_pipeline_sha256"),
                image_root=Path(self.config["image_root"]) if self.config.get("image_root") else None)
        expected_source = self.config.get("bank_pipeline_sha256")
        if expected_source and manifest.get("source", {}).get("source_sha256") != expected_source:
            raise ValueError("SAM object bank does not match the fixed source pipeline")
        source_annotations = Path(manifest["source"]["annotations"]).resolve()
        if source_annotations != annotation_file.resolve():
            raise ValueError("SAM object bank does not match training annotations")
        object_bank = manifest.get("object_bank", {})
        if object_bank.get("mode") not in {"sam3_rgba_only", "rgba_only"}:
            raise ValueError("SAM Copy-Paste requires a sam3_rgba_only object bank")

        excluded = {int(value) for value in self.config.get("excluded_classes", ())}
        self._objects_by_class: dict[int, list[dict[str, Any]]] = {}
        for class_id_raw, relative in object_bank.get("files", {}).items():
            class_id = int(class_id_raw)
            if class_id in excluded:
                continue
            entries = list(_load_json(self._artifact_root / str(relative)))
            if entries:
                self._objects_by_class[class_id] = entries
        if not self._objects_by_class:
            raise ValueError("SAM object bank contains no eligible objects")

        missing_cache = next(
            (
                self._artifact_root / str(entry["rgba_path"])
                for entries in self._objects_by_class.values()
                for entry in entries
                if not (self._artifact_root / str(entry["rgba_path"])).is_file()
            ),
            None,
        )
        if missing_cache is not None:
            raise FileNotFoundError(f"Missing SAM RGBA cache: {missing_cache}")

        self._natural_pool = [
            entry
            for entries in self._objects_by_class.values()
            for entry in entries
        ]
        self._class_weights = {
            int(class_id): float(weight)
            for class_id, weight in self.config.get("class_weights", {}).items()
            if int(class_id) in self._objects_by_class
        }
        if (
            float(self.config.get("target_sample_probability", 0.0)) > 0.0
            and not self._class_weights
        ):
            raise ValueError("No weighted target classes exist in the SAM bank")

        self._epoch = 0
        self._paste_plan: dict[int, list[dict[str, Any]]] = {}
        self.stats: dict[str, Any] = {
            "base_images": len(self.ids),
            "bank_objects": len(self._natural_pool),
            "hard_negative": 0,
        }
        for key in ("base_images", "bank_objects"):
            if self.config.get(key) is not None and self.stats[key] != int(self.config[key]):
                raise ValueError(f"SAM {key} does not match the final recipe")
        self.set_epoch(0)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def epoch(self) -> int:
        return self._epoch

    def _stage_probability(self) -> float:
        for stage in self.config["schedule"]:
            start, end = stage["epochs"]
            if int(start) <= self._epoch <= int(end):
                return float(stage["copy_paste_probability"])
        raise RuntimeError(f"No SAM Copy-Paste schedule for epoch {self._epoch}")

    def _choose_entry(
        self,
        rng: random.Random,
        reuse: Counter[int],
    ) -> tuple[dict[str, Any], bool] | None:
        target_probability = float(
            self.config.get("target_sample_probability", 0.0)
        )
        force_target = bool(self._class_weights) and rng.random() < target_probability
        if force_target:
            classes = list(self._class_weights)
            weights = [self._class_weights[class_id] for class_id in classes]
            class_id = rng.choices(classes, weights=weights, k=1)[0]
            pool = self._objects_by_class[class_id]
        else:
            pool = self._natural_pool

        maximum_reuse = int(self.config["max_source_reuse_per_epoch"])
        for _ in range(min(len(pool) * 2, 200)):
            entry = pool[rng.randrange(len(pool))]
            annotation_id = int(entry["annotation_id"])
            if reuse[annotation_id] < maximum_reuse:
                reuse[annotation_id] += 1
                return entry, force_target
        return None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        probability = self._stage_probability()
        rng = random.Random(int(self.config["seed"]) + self._epoch * 1_000_003)
        reuse: Counter[int] = Counter()
        planned_by_class: Counter[int] = Counter()
        target_draws = 0
        self._paste_plan = {}
        for index in range(len(self)):
            if rng.random() >= probability:
                continue
            requested = rng.randint(
                int(self.config["min_objects"]),
                int(self.config["max_objects"]),
            )
            selected: list[dict[str, Any]] = []
            for _ in range(requested):
                choice = self._choose_entry(rng, reuse)
                if choice is None:
                    continue
                entry, forced = choice
                selected.append(entry)
                planned_by_class[int(entry["class_id"])] += 1
                target_draws += int(forced)
            if selected:
                self._paste_plan[index] = selected
        self.stats.update(
            {
                "epoch": self._epoch,
                "scheduled_images": len(self._paste_plan),
                "scheduled_objects": sum(planned_by_class.values()),
                "forced_target_objects": target_draws,
                "scheduled_by_class": dict(sorted(planned_by_class.items())),
            }
        )

    def _paste_objects(
        self,
        image: Image.Image,
        annotations: list[dict[str, Any]],
        entries: list[dict[str, Any]],
        rng: random.Random,
    ) -> tuple[Image.Image, list[dict[str, Any]]]:
        image = image.copy().convert("RGB")
        annotations = [dict(row) for row in annotations]
        scale_min, scale_max = map(float, self.config["scale_range"])
        maximum_iou = float(self.config["max_gt_iou"])

        for entry in entries:
            cache_path = self._artifact_root / str(entry["rgba_path"])
            with Image.open(cache_path) as cached:
                rgba = cached.convert("RGBA").copy()
            scale = rng.uniform(scale_min, scale_max)
            rgba = rgba.resize(
                (
                    max(2, round(rgba.width * scale)),
                    max(2, round(rgba.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
            mask = rgba.getchannel("A")
            mask_bbox = mask.getbbox()
            if mask_bbox is None:
                continue
            rgba = rgba.crop(mask_bbox)
            mask = rgba.getchannel("A")
            patch = rgba.convert("RGB")
            width, height = patch.size
            if width >= image.width or height >= image.height:
                continue

            destination: tuple[int, int, list[float]] | None = None
            for _ in range(30):
                left = rng.randint(0, image.width - width)
                top = rng.randint(0, image.height - height)
                candidate = [float(left), float(top), float(width), float(height)]
                if all(
                    _xywh_iou(candidate, annotation["bbox"]) <= maximum_iou
                    for annotation in annotations
                ):
                    destination = (left, top, candidate)
                    break
            if destination is None:
                continue

            left, top, candidate = destination
            image.paste(patch, (left, top), mask)
            mask_area = float(ImageStat.Stat(mask).sum[0]) / 255.0
            annotations.append(
                {
                    "id": -(int(entry["annotation_id"]) * 100 + len(annotations) + 1),
                    "image_id": -1,
                    "category_id": int(entry["class_id"]),
                    "bbox": candidate,
                    "area": mask_area,
                    "iscrowd": 0,
                }
            )
        return image, annotations

class SAMCopyPasteTinyFormerDataset(SAMCopyPastePlan, torch.utils.data.Dataset[Any]):
    """Wrap the native TinyFormer COCO dataset with the shared paste plan."""

    def __init__(self, dataset, config, convert_to_tv_tensor) -> None:
        required = ("ids", "coco", "prepare", "_transforms", "_load_image", "_load_target", "set_epoch", "ann_file")
        if missing := [name for name in required if not hasattr(dataset, name)]:
            raise TypeError("SAM Copy-Paste requires CocoDetection attributes: " + ", ".join(missing))
        if getattr(dataset, "return_masks", False):
            raise ValueError("SAM Copy-Paste supports bounding-box detection only")
        self.dataset = dataset
        self._convert_to_tv_tensor = convert_to_tv_tensor
        self.coco = dataset.coco
        self.prepare = dataset.prepare
        self._transforms = dataset._transforms
        self.root = getattr(dataset, "root", None)
        self.img_folder = getattr(dataset, "img_folder", None)
        self.ann_file = dataset.ann_file
        self.return_masks = False
        self.remap_mscoco_category = getattr(dataset, "remap_mscoco_category", False)
        super().__init__(config, Path(self.ann_file), list(dataset.ids))

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        super().set_epoch(epoch)

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return index

    def load_item(self, index: int) -> tuple[Any, dict[str, Any]]:
        index = self._normalize_index(index)
        image_id = int(self.ids[index])
        source = self.dataset._load_image(image_id)
        try:
            image = source.copy()
        finally:
            source.close()
        annotations = list(self.dataset._load_target(image_id))
        entries = self._paste_plan.get(index, ())
        if entries:
            rng = random.Random(
                int(self.config["seed"]) + self._epoch * 10_000_019 + index
            )
            image, annotations = self._paste_objects(
                image,
                annotations,
                list(entries),
                rng,
            )

        image, target = self.prepare(
            image,
            {"image_id": image_id, "annotations": annotations},
        )
        target["idx"] = torch.tensor([index])
        if "boxes" in target:
            target["boxes"] = self._convert_to_tv_tensor(
                target["boxes"],
                key="boxes",
                box_format="xyxy",
                spatial_size=image.size[::-1],
            )
        return image, target

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        image, target = self.load_item(index)
        if self._transforms is None:
            return image, target
        transformed = self._transforms(image, target, self)
        if not isinstance(transformed, (tuple, list)) or len(transformed) != 3:
            raise TypeError("TinyFormer transforms must return image, target, dataset")
        return transformed[0], transformed[1]


def rebuild_tinyformer_train_loader(loader: Any, dataset: Any) -> Any:
    """Rebuild a native loader so its sampler references the wrapped dataset."""

    shuffle = bool(getattr(loader, "shuffle", False))
    expected_sampler = (
        torch.utils.data.RandomSampler
        if shuffle
        else torch.utils.data.SequentialSampler
    )
    if not isinstance(loader.sampler, expected_sampler):
        raise TypeError("SAM Copy-Paste requires a standard TinyFormer sampler")
    kwargs: dict[str, Any] = {
        "batch_size": loader.batch_size,
        "shuffle": shuffle,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": loader.generator,
        "persistent_workers": loader.persistent_workers,
    }
    if loader.num_workers > 0:
        kwargs["prefetch_factor"] = loader.prefetch_factor
    for name in ("pin_memory_device", "in_order"):
        if hasattr(loader, name):
            kwargs[name] = getattr(loader, name)
    rebuilt = type(loader)(dataset, **kwargs)
    rebuilt.shuffle = shuffle
    return rebuilt
