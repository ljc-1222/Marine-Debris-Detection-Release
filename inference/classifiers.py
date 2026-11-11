"""Write row-aligned classifier logits for a fused candidate bank."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from training import checkpoints
from training.final_models import DEFAULT_CONFIG, load_classifier, load_model_config
from training.prepare_data import sha256
from inference.detector_geometry import ENCODED_DECODER, decode_encoded_rgb
from postprocessing import cache_contract
from postprocessing.cache_contract import crop_rectangle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = checkpoints.DEFAULT_MANIFEST
CLASS_COUNT = 34
PATCH_SIZE = 16


def render_crop(
    image: Image.Image,
    rectangle: Sequence[int],
    *,
    size: int,
    mean: Sequence[float],
    std: Sequence[float],
    return_valid_patches: bool,
    minimum_valid_patch_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    left, top, right, bottom = map(int, rectangle)
    fill = tuple(int(round(255.0 * float(value))) for value in mean)
    crop = Image.new("RGB", (right - left, bottom - top), fill)
    valid = Image.new("L", crop.size, 0)
    source = (
        max(left, 0),
        max(top, 0),
        min(right, image.width),
        min(bottom, image.height),
    )
    if source[2] > source[0] and source[3] > source[1]:
        offset = source[0] - left, source[1] - top
        source_crop = image.crop(source)
        crop.paste(source_crop, offset)
        valid.paste(
            255,
            (*offset, offset[0] + source_crop.width, offset[1] + source_crop.height),
        )
    scale = size / max(crop.size)
    resized = (
        max(1, min(size, int(round(crop.width * scale)))),
        max(1, min(size, int(round(crop.height * scale)))),
    )
    crop = crop.resize(resized, Image.Resampling.BICUBIC)
    valid = valid.resize(resized, Image.Resampling.NEAREST)
    offset = (size - resized[0]) // 2, (size - resized[1]) // 2
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(crop, offset)
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32)[:, None, None]
    std_tensor = torch.as_tensor(std, dtype=torch.float32)[:, None, None]
    tensor = (tensor - mean_tensor) / std_tensor
    if not return_valid_patches:
        return tensor, None, bool(np.asarray(valid).any())
    mask = Image.new("L", (size, size), 0)
    mask.paste(valid, offset)
    valid_pixels = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)
    patch_fraction = F.avg_pool2d(
        valid_pixels[None, None], PATCH_SIZE, PATCH_SIZE
    ).flatten()
    patches = (
        patch_fraction > 0.0
        if minimum_valid_patch_fraction == 0.0
        else patch_fraction >= minimum_valid_patch_fraction
    )
    renderable = bool((valid_pixels > 0.0).any())
    if not patches.any():
        patches[patches.numel() // 2] = True
    return tensor, patches, renderable


class ImageCache:
    def __init__(self, maximum: int = 32) -> None:
        self.maximum = maximum
        self.images: OrderedDict[Path, Image.Image] = OrderedDict()

    def get(self, path: Path) -> Image.Image:
        if path not in self.images:
            self.images[path] = decode_encoded_rgb(path)
            if len(self.images) > self.maximum:
                self.images.popitem(last=False)
        self.images.move_to_end(path)
        return self.images[path]


class CropDataset(Dataset):
    def __init__(
        self,
        image_paths: Mapping[int, Path],
        image_ids: np.ndarray,
        rectangles: np.ndarray,
        model: Mapping[str, Any],
    ) -> None:
        self.image_paths = image_paths
        self.image_ids = image_ids
        self.rectangles = rectangles
        self.model = model
        self.cache: ImageCache | None = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        if self.cache is None:
            self.cache = ImageCache()
        image_id = int(self.image_ids[index])
        image = self.cache.get(self.image_paths[image_id])
        pixels, valid, renderable = render_crop(
            image,
            self.rectangles[index],
            size=int(self.model["input_size"]),
            mean=self.model["mean"],
            std=self.model["std"],
            return_valid_patches=self.model["kind"]
            in {"dinov3_attention", "dinov3_hplus_lora"},
            minimum_valid_patch_fraction=float(
                self.model.get("minimum_valid_patch_fraction", 0.5)
            ),
        )
        if valid is None:
            return pixels, index
        if self.model["kind"] == "dinov3_hplus_lora":
            return pixels, valid, index, renderable
        return pixels, valid, index


def unique_crops(
    candidates: Sequence[Mapping[str, Any]], context_scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup: dict[tuple[int, tuple[int, int, int, int]], int] = {}
    row_to_unique = np.empty(len(candidates), dtype=np.int64)
    image_ids: list[int] = []
    rectangles: list[tuple[int, int, int, int]] = []
    for row_index, row in enumerate(candidates):
        image_id = int(row["image_id"])
        rectangle = crop_rectangle(row["bbox"], context_scale)
        key = image_id, rectangle
        unique_index = lookup.get(key)
        if unique_index is None:
            unique_index = len(image_ids)
            lookup[key] = unique_index
            image_ids.append(image_id)
            rectangles.append(rectangle)
        row_to_unique[row_index] = unique_index
    return (
        row_to_unique,
        np.asarray(image_ids, dtype=np.int32),
        np.asarray(rectangles, dtype=np.int32).reshape(-1, 4),
    )


def load_candidates(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict):
        from postprocessing.three_head_runtime import load_bank

        arrays = document.get("arrays")
        if not isinstance(arrays, dict):
            raise ValueError("candidate array manifests require an arrays mapping")
        for name in ("image_ids", "category_ids", "boxes_xywh", "scores", "ordinals"):
            if name not in arrays:
                continue
            expected = arrays[name].get("sha256", "") if isinstance(arrays[name], dict) else ""
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(value not in "0123456789abcdef" for value in expected)):
                raise ValueError("candidate array manifests must bind each input array by SHA256")
        bank = load_bank(path)
        document = [
            {"image_id": int(image), "category_id": int(category), "bbox": box.tolist()}
            for image, category, box in zip(
                bank["image_ids"], bank["category_ids"], bank["boxes_xywh"], strict=True
            )
        ]
    if not isinstance(document, list):
        raise ValueError("candidate bank must contain a JSON array or array manifest")
    for row in document:
        box = row["bbox"]
        if (len(box) != 4 or not all(math.isfinite(float(value)) for value in box)
                or float(box[2]) <= 0 or float(box[3]) <= 0):
            raise ValueError("candidate boxes require finite XYWH with positive width and height")
        if not 1 <= int(row["category_id"]) <= CLASS_COUNT:
            raise ValueError("candidate category IDs must be in 1..34")
    return document


def fingerprint(path: Path) -> dict[str, Any]:
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def save_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
    return {"path": path.name, **fingerprint(path)}


def extract_logits(
    model: torch.nn.Module, dataset: CropDataset, model_config: Mapping[str, Any],
    device: torch.device, *, batch_size: int, workers: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    channels_last = bool(model_config.get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    unique_logits = np.empty((len(dataset), CLASS_COUNT), dtype=np.float32)
    unique_objectness = np.empty(len(dataset), dtype=np.float32)
    unique_valid = np.empty(len(dataset), dtype=np.bool_) if model_config["kind"] == "dinov3_hplus_lora" else None
    with torch.inference_mode():
        for batch in loader:
            if model_config["kind"] == "dinov3_hplus_lora":
                pixels, valid, indices, renderable = batch
                selected = renderable.bool()
                class_values = torch.zeros((len(indices), CLASS_COUNT))
                object_values = torch.zeros(len(indices))
                if selected.any():
                    selected_pixels = pixels[selected].to(device, non_blocking=True)
                    selected_valid = valid[selected].to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
                    ):
                        logits, objectness = model(selected_pixels, selected_valid)
                    if logits.shape != (int(selected.sum()), CLASS_COUNT) or objectness.shape != (int(selected.sum()),):
                        raise ValueError("classifier output shapes do not match the crop batch")
                    class_values[selected] = logits.float().cpu()
                    object_values[selected] = objectness.float().cpu()
                logits, objectness = class_values, object_values
            elif model_config["kind"] == "dinov3_attention":
                pixels, valid, indices = batch
                inputs = (pixels.to(device, non_blocking=True), valid.to(device, non_blocking=True))
            else:
                pixels, indices = batch
                pixels = pixels.to(device, non_blocking=True)
                if channels_last:
                    pixels = pixels.contiguous(memory_format=torch.channels_last)
                inputs = (pixels,)
            if model_config["kind"] != "dinov3_hplus_lora":
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
                ):
                    logits, objectness = model(*inputs)
            if logits.shape != (len(indices), CLASS_COUNT) or objectness.shape != (len(indices),):
                raise ValueError("classifier output shapes do not match the crop batch")
            if not torch.isfinite(logits).all() or not torch.isfinite(objectness).all():
                raise ValueError("classifier produced non-finite outputs")
            batch_indices = np.asarray(indices, dtype=np.int64)
            unique_logits[batch_indices] = logits.float().cpu().numpy()
            unique_objectness[batch_indices] = objectness.float().cpu().numpy()
            if unique_valid is not None:
                unique_valid[batch_indices] = np.asarray(renderable, dtype=bool)
    result = {"class_logits": unique_logits, "objectness_logits": unique_objectness}
    if unique_valid is not None:
        result["valid_crops"] = unique_valid
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch < 1 or args.workers < 0:
        raise ValueError("batch must be positive and workers non-negative")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    target = torch.device(args.device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("classifier inference supports CPU or BF16-capable CUDA")
    bank_path = args.bank.expanduser().resolve()
    index_path = args.index.expanduser().resolve()
    manifest_path = Path(getattr(args, "manifest", DEFAULT_MANIFEST)).expanduser().resolve()
    inputs = {"bank": fingerprint(bank_path), "index": fingerprint(index_path),
              "config": fingerprint(DEFAULT_CONFIG), "checkpoint_catalog": fingerprint(manifest_path)}
    candidates = load_candidates(bank_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    images = args.images.expanduser().resolve()
    if not candidates:
        from inference.detectors import load_index

        index = load_index(index_path, images)
    elif target.type == "cuda":
        with torch.cuda.device(target):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("classifier inference requires BF16-capable CUDA")
    image_paths = {}
    for row in index["images"]:
        image_id = int(row["id"])
        relative = Path(str(row["file_name"]))
        if image_id in image_paths or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("index image IDs must be unique and filenames relative to --images")
        image_paths[image_id] = images / relative
    referenced = {int(row["image_id"]) for row in candidates}
    if referenced - set(image_paths):
        raise ValueError("candidate bank references unknown image IDs")
    for image_id in referenced:
        if not image_paths[image_id].is_file():
            raise FileNotFoundError(image_paths[image_id])
    model_config = load_model_config(args.model, DEFAULT_CONFIG)
    catalog = checkpoints.load_manifest(manifest_path)
    artifact = catalog["artifacts"][args.model]
    source_images = cache_contract.image_identities(index, images, referenced)
    from postprocessing.three_head_runtime import load_bank
    identities = cache_contract.row_ids(load_bank(bank_path), args.model, model_config,
                                        source_images, artifact["sha256"])
    code = {name: fingerprint(ROOT / name) for name in cache_contract.CODE_FILES}
    checkpoint = checkpoints.destination(args.checkpoint_root.expanduser().resolve(), artifact)
    row_to_unique, unique_image_ids, rectangles = unique_crops(candidates, float(model_config["context_scale"]))
    if candidates:
        model = load_classifier(args.model, checkpoint, manifest=manifest_path, config=DEFAULT_CONFIG, device=target)
        dataset = CropDataset(image_paths, unique_image_ids, rectangles, model_config)
        unique_arrays = extract_logits(model, dataset, model_config, target, batch_size=args.batch, workers=args.workers)
    else:
        checkpoints.verify_file(checkpoint, artifact)
        unique_arrays = {"class_logits": np.empty((0, CLASS_COUNT), dtype=np.float32),
                         "objectness_logits": np.empty(0, dtype=np.float32)}
        if model_config["kind"] == "dinov3_hplus_lora":
            unique_arrays["valid_crops"] = np.empty(0, dtype=np.bool_)
    current_inputs = {"bank": fingerprint(bank_path), "index": fingerprint(index_path),
                      "config": fingerprint(DEFAULT_CONFIG), "checkpoint_catalog": fingerprint(manifest_path)}
    if current_inputs != inputs or cache_contract.image_identities(index, images, referenced) != source_images:
        raise ValueError("Classifier source inputs changed during cache generation")
    output.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 2, "model": args.model, "rows": len(candidates), "unique_crops": len(unique_image_ids),
        "class_order": cache_contract.class_order(), "class_id_space": "foreground_0_based",
        "model_config": model_config, "row_identity_algorithm": cache_contract.ROW_ID_ALGORITHM,
        "candidate_sha256": inputs["bank"]["sha256"], "index_sha256": inputs["index"]["sha256"],
        "config_sha256": inputs["config"]["sha256"], "checkpoint_catalog_sha256": inputs["checkpoint_catalog"]["sha256"],
        "checkpoint_sha256": artifact["sha256"], "checkpoint_bytes": artifact["bytes"],
        "coordinate_system": "encoded_image_pixels",
        "image_decode_policy": ENCODED_DECODER,
        "deployment": "merged_lora_160" if args.model == "hplus" else "full_model",
        "settings": {"batch": args.batch, "workers": args.workers, "device": str(target),
                     "precision": ("bfloat16_autocast" if target.type == "cuda" else "float32") if candidates else "not_executed",
                     "context_scale": float(model_config["context_scale"]), "input_size": int(model_config["input_size"])},
        "inputs": inputs, "code": code, "images": source_images,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "cuda_build": torch.version.cuda},
    }
    if not candidates:
        document["execution"] = dict(cache_contract.EMPTY_EXECUTION)
    for name, values in unique_arrays.items():
        document[name] = save_npy(output / f"{name}.npy", values[row_to_unique])
    document["row_ids"] = save_npy(output / "row_ids.npy", identities)
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("hplus", "v7", "siglip"), required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    document = run(parse_args(argv))
    print(json.dumps({"model": document["model"], "rows": document["rows"], "unique_crops": document["unique_crops"]}, sort_keys=True))


if __name__ == "__main__":
    main()
