"""Class-aware instance Copy-Paste for RF-DETR detection training."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask


@dataclass(frozen=True)
class CopyPasteSettings:
    """Validated settings used by :class:`ClassAwareCopyPaste`."""

    mask_annotations: Path
    image_root: Path
    target_classes: tuple[str, ...]
    excluded_classes: tuple[str, ...]
    target_sample_probability: float
    class_weights: Mapping[str, float]
    schedule: tuple[tuple[int, int, float], ...]
    probability: float
    max_pastes: int
    scale_min: float
    scale_max: float
    max_overlap_fraction: float
    placement_attempts: int
    sampling: str
    start_epoch: int
    end_epoch: int | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CopyPasteSettings:
        """Build settings from a validated training config."""

        if value.get("enabled") is not True:
            raise ValueError("CopyPasteSettings requires copy_paste.enabled=true")
        return cls(
            mask_annotations=Path(value["mask_annotations"]).resolve(),
            image_root=Path(value["image_root"]).resolve(),
            target_classes=tuple(map(str, value["target_classes"])),
            excluded_classes=tuple(map(str, value.get("excluded_classes", ()))),
            target_sample_probability=float(
                value.get("target_sample_probability", 0.0)
            ),
            class_weights={
                str(name): float(weight)
                for name, weight in value.get("class_weights", {}).items()
            },
            schedule=tuple(
                (
                    int(stage["epochs"][0]),
                    int(stage["epochs"][1]),
                    float(stage["copy_paste_probability"]),
                )
                for stage in value.get("schedule", ())
            ),
            probability=float(value["probability"]),
            max_pastes=int(value["max_pastes"]),
            scale_min=float(value["scale_min"]),
            scale_max=float(value["scale_max"]),
            max_overlap_fraction=float(value["max_overlap_fraction"]),
            placement_attempts=int(value["placement_attempts"]),
            sampling=str(value["sampling"]),
            start_epoch=int(value["start_epoch"]),
            end_epoch=(
                None if value["end_epoch"] is None else int(value["end_epoch"])
            ),
        )

    def probability_for_epoch(self, epoch: int) -> float:
        for start, end, probability in self.schedule:
            if start <= epoch <= end:
                return probability
        if self.schedule:
            raise RuntimeError(f"No RF-DETR Copy-Paste schedule for epoch {epoch}")
        return self.probability


@dataclass(frozen=True)
class _Donor:
    annotation_id: int
    image_id: int
    category_id: int
    label: int
    image_path: Path
    image_width: int
    image_height: int
    segmentation: Mapping[str, Any]


class ClassAwareCopyPaste:
    """Paste configured SAM-segmented instances into a detection image.

    The transform expects a PIL image after RF-DETR's resize/crop transform and
    before tensor conversion/normalization. Donor objects retain their natural
    apparent scale (relative to the source image's short side), with only the
    configured scale jitter applied.
    """

    def __init__(self, settings: CopyPasteSettings) -> None:
        self.settings = settings
        self._donors, self._categories, self._class_counts = self._load_donors()
        if settings.sampling not in {"inverse_sqrt", "uniform"}:
            raise ValueError(f"Unsupported Copy-Paste sampling: {settings.sampling!r}")
        self._weights = (
            torch.ones(len(self._donors), dtype=torch.float64)
            if settings.sampling == "uniform"
            else torch.as_tensor(
                [
                    1.0 / math.sqrt(self._class_counts[donor.category_id])
                    for donor in self._donors
                ],
                dtype=torch.float64,
            )
        )
        name_to_id = {
            name: category_id for category_id, name in self._categories.items()
        }
        self._target_category_ids = torch.as_tensor(
            [name_to_id[name] for name in settings.class_weights], dtype=torch.int64
        )
        self._target_category_weights = torch.as_tensor(
            list(settings.class_weights.values()), dtype=torch.float64
        )
        self._donor_category_ids = torch.as_tensor(
            [donor.category_id for donor in self._donors], dtype=torch.int64
        )
        # RF-DETR enables persistent DataLoader workers by default. Shared memory
        # lets the Lightning callback change the epoch gate without rebuilding
        # those workers at every epoch.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    @property
    def summary(self) -> dict[str, Any]:
        """Return JSON-safe donor provenance for the run manifest."""

        return {
            "mask_annotations": str(self.settings.mask_annotations),
            "image_root": str(self.settings.image_root),
            "sampling": self.settings.sampling,
            "target_sample_probability": self.settings.target_sample_probability,
            "class_weights": dict(self.settings.class_weights),
            "excluded_classes": list(self.settings.excluded_classes),
            "schedule": [
                {
                    "epochs": [start, end],
                    "copy_paste_probability": probability,
                }
                for start, end, probability in self.settings.schedule
            ],
            "active_epoch_range": [
                self.settings.start_epoch,
                self.settings.end_epoch,
            ],
            "donor_count": len(self._donors),
            "donors_per_class": {
                self._categories[category_id]: count
                for category_id, count in sorted(self._class_counts.items())
            },
        }

    @property
    def epoch(self) -> int:
        """Return the zero-based training epoch visible to worker processes."""

        return int(self._epoch.item())

    def set_epoch(self, epoch: int) -> None:
        """Update the shared zero-based epoch used by the augmentation gate."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Copy-Paste epoch must be a non-negative integer")
        self._epoch.fill_(epoch)

    def is_active(self) -> bool:
        """Return whether Copy-Paste is enabled for the current epoch."""

        epoch = self.epoch
        return (
            self.settings.probability_for_epoch(epoch) > 0.0
            and epoch >= self.settings.start_epoch
            and (
                self.settings.end_epoch is None
                or epoch < self.settings.end_epoch
            )
        )

    def validate_dataset_categories(
        self,
        categories: Mapping[int, Mapping[str, Any]],
        cat2label: Mapping[int, int] | None,
    ) -> None:
        """Ensure donor labels exactly match the RF-DETR training dataset."""

        if cat2label is None:
            raise ValueError("Class-aware Copy-Paste requires RF-DETR category remapping")
        donor_ids = set(self._categories)
        dataset_ids = {int(category_id) for category_id in categories}
        if donor_ids != dataset_ids:
            raise ValueError("Copy-Paste and training category IDs do not match")
        donor_cat2label = {
            category_id: label
            for label, category_id in enumerate(sorted(self._categories))
        }
        for category_id, donor_name in self._categories.items():
            dataset_category = categories.get(category_id)
            dataset_name = None if dataset_category is None else dataset_category.get("name")
            if dataset_name != donor_name:
                raise ValueError(
                    f"Copy-Paste category {category_id} is {donor_name!r}, "
                    f"but training data uses {dataset_name!r}"
                )
            expected_label = int(cat2label[category_id])
            donor_label = donor_cat2label[category_id]
            if donor_label != expected_label:
                raise ValueError(
                    f"Copy-Paste label mapping differs for category_id={category_id}: "
                    f"{donor_label} != {expected_label}"
                )

    def __call__(
        self,
        image: Image.Image,
        target: dict[str, Any] | None,
    ) -> tuple[Image.Image, dict[str, Any] | None]:
        probability = self.settings.probability_for_epoch(self.epoch)
        if (
            target is None
            or probability <= 0.0
            or not self.is_active()
            or torch.rand(()).item() >= probability
        ):
            return image, target
        if not isinstance(image, Image.Image):
            raise TypeError("Class-aware Copy-Paste must run before RF-DETR ToImage")
        self._validate_target(target)

        canvas = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        output = target.copy()
        occupied_boxes = output["boxes"].detach().to(dtype=torch.float32, device="cpu")
        current_image_id = _scalar_image_id(output.get("image_id"))
        used_annotations: set[int] = set()
        paste_count = int(torch.randint(1, self.settings.max_pastes + 1, ()).item())

        for _ in range(paste_count):
            donor = self._sample_donor(current_image_id, used_annotations)
            if donor is None:
                break
            prepared = self._prepare_donor(donor, canvas.shape[1], canvas.shape[0])
            if prepared is None:
                used_annotations.add(donor.annotation_id)
                continue
            patch, mask = prepared
            placement = self._find_placement(
                mask,
                canvas.shape[1],
                canvas.shape[0],
                occupied_boxes,
            )
            used_annotations.add(donor.annotation_id)
            if placement is None:
                continue

            left, top, box = placement
            region = canvas[top : top + patch.shape[0], left : left + patch.shape[1]]
            region[mask] = patch[mask]
            output = self._append_instance(output, donor.label, box, int(mask.sum()))
            occupied_boxes = torch.cat((occupied_boxes, box.reshape(1, 4)), dim=0)

        return Image.fromarray(canvas), output

    def _load_donors(
        self,
    ) -> tuple[list[_Donor], dict[int, str], Counter[int]]:
        with self.settings.mask_annotations.open(encoding="utf-8") as handle:
            document = json.load(handle)
        categories_raw = document.get("categories")
        images_raw = document.get("images")
        annotations_raw = document.get("annotations")
        if not isinstance(categories_raw, list):
            raise ValueError("Copy-Paste mask annotations require a categories list")
        if not isinstance(images_raw, list) or not isinstance(annotations_raw, list):
            raise ValueError("Copy-Paste mask annotations require images and annotations lists")

        categories = {int(row["id"]): str(row["name"]) for row in categories_raw}
        if len(categories) != len(categories_raw):
            raise ValueError("Copy-Paste category IDs must be unique")
        name_to_id = {name: category_id for category_id, name in categories.items()}
        if len(name_to_id) != len(categories):
            raise ValueError("Copy-Paste category names must be unique")
        missing = set(self.settings.target_classes) - set(name_to_id)
        if missing:
            raise ValueError(f"Copy-Paste target classes not found: {sorted(missing)}")
        excluded = set(self.settings.excluded_classes)
        if missing := excluded - set(name_to_id):
            raise ValueError(f"Copy-Paste excluded classes not found: {sorted(missing)}")
        target_names = set(self.settings.target_classes) or set(name_to_id)
        target_names -= excluded
        if missing := set(self.settings.class_weights) - target_names:
            raise ValueError(
                f"Copy-Paste weighted classes are not eligible: {sorted(missing)}"
            )
        target_ids = {name_to_id[name] for name in target_names}
        cat2label = {
            category_id: label for label, category_id in enumerate(sorted(categories))
        }
        images = {int(row["id"]): row for row in images_raw}
        if len(images) != len(images_raw):
            raise ValueError("Copy-Paste image IDs must be unique")

        donors: list[_Donor] = []
        seen_annotation_ids: set[int] = set()
        root = self.settings.image_root.resolve()
        for annotation in annotations_raw:
            category_id = int(annotation["category_id"])
            if category_id not in target_ids or int(annotation.get("iscrowd", 0)) != 0:
                continue
            annotation_id = int(annotation["id"])
            if annotation_id in seen_annotation_ids:
                raise ValueError(f"Duplicate Copy-Paste annotation id: {annotation_id}")
            seen_annotation_ids.add(annotation_id)
            image_id = int(annotation["image_id"])
            image_row = images.get(image_id)
            if image_row is None:
                raise ValueError(f"Copy-Paste annotation references unknown image_id={image_id}")
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, dict):
                raise ValueError(
                    f"Copy-Paste annotation {annotation_id} requires COCO RLE segmentation"
                )
            image_width = int(image_row["width"])
            image_height = int(image_row["height"])
            if list(segmentation.get("size", [])) != [image_height, image_width]:
                raise ValueError(
                    f"Copy-Paste annotation {annotation_id} RLE size does not match its image"
                )
            relative_path = Path(str(image_row["file_name"]))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Copy-Paste image file_name must stay below image_root: {relative_path}"
                )
            # Keep the lexical path so prepared datasets may use per-image
            # symlinks to the immutable source image tree.
            image_path = root / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Copy-Paste donor image not found: {image_path}")
            donors.append(
                _Donor(
                    annotation_id=annotation_id,
                    image_id=image_id,
                    category_id=category_id,
                    label=cat2label[category_id],
                    image_path=image_path,
                    image_width=image_width,
                    image_height=image_height,
                    segmentation=segmentation,
                )
            )

        if not donors:
            raise ValueError("Copy-Paste donor pool is empty")
        counts = Counter(donor.category_id for donor in donors)
        missing_donors = target_ids - set(counts) if self.settings.target_classes else set()
        if missing_donors:
            missing_names = [categories[category_id] for category_id in sorted(missing_donors)]
            raise ValueError(f"Copy-Paste target classes have no accepted masks: {missing_names}")
        return donors, categories, counts

    def _sample_donor(
        self,
        current_image_id: int | None,
        used_annotations: set[int],
    ) -> _Donor | None:
        eligible = torch.as_tensor(
            [
                donor.annotation_id not in used_annotations
                and (current_image_id is None or donor.image_id != current_image_id)
                for donor in self._donors
            ],
            dtype=torch.bool,
        )
        if not eligible.any():
            return None
        if (
            self._target_category_ids.numel()
            and torch.rand(()).item() < self.settings.target_sample_probability
        ):
            category_index = int(
                torch.multinomial(self._target_category_weights, 1).item()
            )
            target_eligible = eligible & self._donor_category_ids.eq(
                self._target_category_ids[category_index]
            )
            if target_eligible.any():
                eligible = target_eligible
        weights = self._weights * eligible.to(dtype=self._weights.dtype)
        index = int(torch.multinomial(weights, 1).item())
        return self._donors[index]

    def _prepare_donor(
        self,
        donor: _Donor,
        canvas_width: int,
        canvas_height: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        with Image.open(donor.image_path) as source_handle:
            source = np.asarray(source_handle.convert("RGB"), dtype=np.uint8)
        if source.shape[:2] != (donor.image_height, donor.image_width):
            raise ValueError(
                f"Copy-Paste donor dimensions do not match COCO metadata: {donor.image_path}"
            )
        mask = coco_mask.decode(dict(donor.segmentation))
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        mask = np.asarray(mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        patch = source[top:bottom, left:right]
        mask = mask[top:bottom, left:right]

        jitter = self.settings.scale_min + (
            self.settings.scale_max - self.settings.scale_min
        ) * torch.rand(()).item()
        source_short_side = min(donor.image_width, donor.image_height)
        canvas_short_side = min(canvas_width, canvas_height)
        scale = canvas_short_side / source_short_side * jitter
        width = max(1, int(round(patch.shape[1] * scale)))
        height = max(1, int(round(patch.shape[0] * scale)))
        if width > canvas_width or height > canvas_height:
            return None

        patch_image = Image.fromarray(patch).resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        )
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        )
        resized_mask = np.asarray(mask_image, dtype=np.uint8) > 0
        if not resized_mask.any():
            return None
        return np.asarray(patch_image, dtype=np.uint8), resized_mask

    def _find_placement(
        self,
        mask: np.ndarray,
        canvas_width: int,
        canvas_height: int,
        occupied_boxes: torch.Tensor,
    ) -> tuple[int, int, torch.Tensor] | None:
        height, width = mask.shape
        ys, xs = np.nonzero(mask)
        mask_left, mask_right = int(xs.min()), int(xs.max()) + 1
        mask_top, mask_bottom = int(ys.min()), int(ys.max()) + 1
        for _ in range(self.settings.placement_attempts):
            left = int(torch.randint(0, canvas_width - width + 1, ()).item())
            top = int(torch.randint(0, canvas_height - height + 1, ()).item())
            box = torch.tensor(
                [
                    left + mask_left,
                    top + mask_top,
                    left + mask_right,
                    top + mask_bottom,
                ],
                dtype=torch.float32,
            )
            if _maximum_overlap_fraction(box, occupied_boxes) <= self.settings.max_overlap_fraction:
                return left, top, box
        return None

    @staticmethod
    def _append_instance(
        target: dict[str, Any],
        label: int,
        box: torch.Tensor,
        mask_area: int,
    ) -> dict[str, Any]:
        output = target.copy()
        output["boxes"] = torch.cat(
            (target["boxes"], box.to(device=target["boxes"].device).reshape(1, 4)),
            dim=0,
        )
        output["labels"] = torch.cat(
            (target["labels"], target["labels"].new_tensor([label])),
            dim=0,
        )
        output["area"] = torch.cat(
            (target["area"], target["area"].new_tensor([mask_area])),
            dim=0,
        )
        output["iscrowd"] = torch.cat(
            (target["iscrowd"], target["iscrowd"].new_zeros(1)),
            dim=0,
        )
        return output

    @staticmethod
    def _validate_target(target: Mapping[str, Any]) -> None:
        required = {"boxes", "labels", "area", "iscrowd"}
        missing = required - set(target)
        if missing:
            raise ValueError(f"RF-DETR Copy-Paste target is missing fields: {sorted(missing)}")
        instance_count = int(target["boxes"].shape[0])
        for key in required:
            value = target[key]
            if not torch.is_tensor(value) or value.shape[0] != instance_count:
                raise ValueError(f"RF-DETR Copy-Paste target field {key!r} is misaligned")
        if "keypoints" in target:
            raise ValueError("Class-aware Copy-Paste does not support keypoint targets")
        if "masks" in target:
            raise ValueError(
                "Detection Copy-Paste must be used with RF-DETR segmentation_head=false"
            )


def _scalar_image_id(value: Any) -> int | None:
    if torch.is_tensor(value):
        return int(value.item()) if value.numel() == 1 else None
    return int(value) if isinstance(value, int) else None


def _maximum_overlap_fraction(box: torch.Tensor, occupied: torch.Tensor) -> float:
    """Return maximum intersection divided by the smaller of two box areas."""

    if occupied.numel() == 0:
        return 0.0
    occupied = occupied.reshape(-1, 4).to(dtype=torch.float32, device="cpu")
    box = box.to(dtype=torch.float32, device="cpu")
    top_left = torch.maximum(occupied[:, :2], box[:2])
    bottom_right = torch.minimum(occupied[:, 2:], box[2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    occupied_area = (occupied[:, 2] - occupied[:, 0]) * (
        occupied[:, 3] - occupied[:, 1]
    )
    denominator = torch.minimum(box_area.expand_as(occupied_area), occupied_area).clamp(min=1)
    return float((intersection / denominator).max().item())


__all__ = ["ClassAwareCopyPaste", "CopyPasteSettings"]
