"""Fixed detector collection reused from the portable competition runtime."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import os
from pathlib import Path

from PIL import ImageOps
import torch

from training.final_detectors import LoadedDetector, ROOT, inference_autocast
from training.local_ultralytics import local_ultralytics
from inference.detector_geometry import codetr_geometry, decode_encoded_rgb, inverse_horizontal_boxes, restore_codetr_predictions

THRESHOLD = 0.001
MAX_DETECTIONS = 300


def prediction_rows(boxes, labels, scores, image, *, flip=False, rfdetr=False):
    """Preserve native selection order, inverse flip, then clip encoded pixels."""
    if flip:
        boxes = inverse_horizontal_boxes(boxes, float(image["width"]))
    detections = []
    for box, label, score in zip(boxes, labels, scores, strict=True):
        label, score = int(label), float(score)
        if rfdetr and label == 34:
            continue
        if not 0 <= label < 34:
            raise ValueError(f"Invalid detector class ID: {label}")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Invalid detector confidence")
        if score < THRESHOLD or (rfdetr and score == THRESHOLD):
            continue
        box = tuple(map(float, box))
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("Invalid detector xyxy coordinates")
        detections.append((box, label, score))
    detections.sort(key=lambda row: row[2], reverse=True)
    rows = []
    width, height = float(image["width"]), float(image["height"])
    for (x1, y1, x2, y2), label, score in detections[:MAX_DETECTIONS]:
        x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
        y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
        if x2 > x1 and y2 > y1:
            rows.append({"image_id": int(image["id"]), "category_id": label + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    return rows


def tinyformer_inputs(paths, config, *, flip=False):
    from torchvision.transforms import v2

    operations = config["val_dataloader"]["dataset"]["transforms"]["ops"]
    if [operation["type"] for operation in operations] != ["Resize", "ConvertPILImage", "Normalize"]:
        raise ValueError("Unexpected TinyFormer inference transforms")
    transform = v2.Compose([v2.Resize(tuple(operations[0]["size"])), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=operations[2]["mean"], std=operations[2]["std"])])
    tensors, sizes = [], []
    for path in paths:
        rgb = decode_encoded_rgb(path)
        sizes.append(rgb.size)
        tensors.append(transform(ImageOps.mirror(rgb) if flip else rgb))
    return torch.stack(tensors), torch.tensor(sizes, dtype=torch.int64)


def collect_tinyformer(loaded: LoadedDetector, document, images, *, batch, flip):
    target = next(loaded.model.parameters()).device
    output = []
    for start in range(0, len(document["images"]), batch):
        rows = document["images"][start:start + batch]
        samples, sizes = tinyformer_inputs([images / row["file_name"] for row in rows], loaded.config, flip=flip)
        with torch.inference_mode(), inference_autocast("tinyformer", target):
            predictions = loaded.model(samples.to(target))
            results = loaded.postprocessor(predictions, sizes.to(target))
        if len(results) != len(rows):
            raise RuntimeError("TinyFormer returned an unexpected image count")
        for image, result in zip(rows, results, strict=True):
            output.extend(prediction_rows(result["boxes"].detach().cpu(), result["labels"].detach().cpu(), result["scores"].detach().cpu(), image, flip=flip))
    return output


def collect_rfdetr(loaded: LoadedDetector, document, images, *, batch, flip):
    """Use RF-DETR 1.9.0 predict's resize/normalization and native PostProcess."""
    from torchvision.transforms import functional as F

    target = next(loaded.model.parameters()).device
    if int(loaded.config["num_select"]) != MAX_DETECTIONS:
        raise ValueError("The fixed RF-DETR export must select 300 native candidates")
    resolution = int(loaded.config["resolution"])
    output = []
    for start in range(0, len(document["images"]), batch):
        rows = document["images"][start:start + batch]
        tensors, original_sizes = [], []
        for row in rows:
            rgb = decode_encoded_rgb(images / row["file_name"])
            if flip:
                rgb = ImageOps.mirror(rgb)
            tensor = F.to_tensor(rgb)
            original_sizes.append(tuple(tensor.shape[1:]))
            tensors.append(tensor.to(target))
        samples = torch.stack([F.resize(tensor, [resolution, resolution], antialias=False) for tensor in tensors])
        samples = F.normalize(samples, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        with torch.inference_mode():
            predictions = loaded.model(samples)
            results = loaded.postprocessor(predictions, target_sizes=torch.tensor(original_sizes, device=target))
        if len(results) != len(rows):
            raise RuntimeError("RF-DETR returned an unexpected image count")
        for image, result in zip(rows, results, strict=True):
            # Native predict applies a strict score threshold before its Detections wrapper.
            keep = result["scores"] > THRESHOLD
            output.extend(prediction_rows(result["boxes"][keep].float().detach().cpu(), result["labels"][keep].detach().cpu(), result["scores"][keep].float().detach().cpu(), image, flip=flip, rfdetr=True))
    return output


@local_ultralytics()
def yolo_api(loaded: LoadedDetector):
    """Attach a safely loaded DetectionModel to the pinned local YOLO API."""
    from ultralytics import YOLO

    config = ROOT / "third_party/ultralytics/ultralytics/cfg/models/26/yolo26.yaml"
    wrapper = YOLO(str(config), task="detect", verbose=False)
    arguments = dict(wrapper.model.args)
    arguments.update(task="detect", imgsz=960)
    loaded.model.args = arguments
    loaded.model.task = "detect"
    wrapper.model = loaded.model
    wrapper.task = "detect"
    wrapper.overrides.update(task="detect", imgsz=960)
    wrapper.ckpt = {}
    return wrapper


@local_ultralytics()
def collect_yolo26(loaded: LoadedDetector, document, images, *, batch, flip):
    target = next(loaded.model.parameters()).device
    model = yolo_api(loaded)
    output = []
    for start in range(0, len(document["images"]), batch):
        rows = document["images"][start:start + batch]
        paths = [images / row["file_name"] for row in rows]
        # String sources invoke Ultralytics' exif_transpose, while PIL sources
        # do not. Both views must receive the same stored-pixel RGB decoder.
        inputs = [decode_encoded_rgb(path) for path in paths]
        if flip:
            inputs = [ImageOps.mirror(image) for image in inputs]
        results = model.predict(source=inputs, imgsz=960, batch=len(paths), device=str(target), conf=THRESHOLD, max_det=MAX_DETECTIONS, verbose=False, save=False)
        if len(results) != len(rows):
            raise RuntimeError("YOLO26 returned an unexpected image count")
        for image, result in zip(rows, results, strict=True):
            boxes = result.boxes
            output.extend(prediction_rows(boxes.xyxy.detach().cpu(), boxes.cls.detach().cpu(), boxes.conf.detach().cpu(), image, flip=flip))
    return output


def collect_codetr(loaded: LoadedDetector, document, images, index_path, *, batch, workers, flip):
    """Use the frozen MMDetection dataset pipeline and encoded-EXIF restoration."""
    from mmcv.parallel import MMDataParallel
    from mmdet.apis import single_gpu_test
    from mmdet.datasets import DATASETS, CocoDataset, build_dataloader, build_dataset
    from mmdet.datasets.pipelines import Compose

    if DATASETS.get("CompetitionCocoDataset") is None:
        @DATASETS.register_module()
        class CompetitionCocoDataset(CocoDataset):
            def __init__(self, prediction_score_threshold=THRESHOLD, **kwargs):
                self.prediction_score_threshold = float(prediction_score_threshold)
                super().__init__(**kwargs)

            def _det2json(self, results):
                return [row for row in super()._det2json(results) if float(row["score"]) >= self.prediction_score_threshold]

    target = next(loaded.model.parameters()).device
    if target.type != "cuda":
        raise ValueError("Co-DETR collection requires its CUDA/MMCV environment")
    geometry = codetr_geometry(document, images)
    config = loaded.config
    config.data.test = deepcopy(config.data.test)
    config.data.test.update(type="CompetitionCocoDataset", ann_file=str(index_path), img_prefix=str(images) + os.sep, classes=tuple(loaded.class_names), test_mode=True, prediction_score_threshold=THRESHOLD)
    if flip:
        augmentation = config.data.test.pipeline[1]

        class HorizontalFlipAug:
            def __init__(self):
                self.transforms = Compose(augmentation.transforms)
                self.scale = tuple(augmentation.img_scale)

            def __call__(self, results):
                transformed = results.copy()
                transformed.update(scale=self.scale, flip=True, flip_direction="horizontal")
                result = self.transforms(transformed)
                return {key: [value] for key, value in result.items()}

        config.data.test.pipeline[1] = HorizontalFlipAug()
    dataset = build_dataset(config.data.test)
    if list(map(str, dataset.CLASSES)) != loaded.class_names:
        raise ValueError("Co-DETR dataset class order changed")
    loader = build_dataloader(dataset, samples_per_gpu=batch, workers_per_gpu=workers, dist=False, shuffle=False, persistent_workers=workers > 0, pin_memory=True)
    torch.cuda.set_device(target)
    with torch.inference_mode():
        results = single_gpu_test(MMDataParallel(loaded.model, device_ids=[target.index or 0]), loader)
    if len(results) != len(dataset):
        raise RuntimeError("Co-DETR returned an unexpected image count")
    predictions = dataset._det2json(results)
    restored = restore_codetr_predictions(predictions, geometry, horizontal_flip=flip)
    grouped = defaultdict(list)
    for row in restored:
        grouped[int(row["image_id"])].append(row)
    return [row for image_id in sorted(grouped) for row in sorted(grouped[image_id], key=lambda row: float(row["score"]), reverse=True)[:MAX_DETECTIONS]]
