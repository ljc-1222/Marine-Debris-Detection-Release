"""Check one trained crop classifier on an encoded-image XYWH crop."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn

from training.checkpoints import DEFAULT_MANIFEST
from training.final_models import DEFAULT_CONFIG, MODEL_ARCHITECTURES, load_classifier, load_model_config
from training.models import CLASS_COUNT, crop_rectangle


def render_input(
    image: Image.Image, box_xywh: Sequence[float], model: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use the inference crop's context, letterbox, normalization and patch mask."""
    if len(box_xywh) != 4 or not all(math.isfinite(float(value)) for value in box_xywh):
        raise ValueError("--box requires four finite XYWH values")
    x, y, width, height = map(float, box_xywh)
    if width <= 0 or height <= 0:
        raise ValueError("Crop width and height must be positive")
    try:
        left, top, right, bottom = crop_rectangle(box_xywh, float(model["context_scale"]))
    except (OverflowError, ValueError):
        raise ValueError("Crop rectangle must be finite") from None
    source = max(left, 0), max(top, 0), min(right, image.width), min(bottom, image.height)
    if source[2] <= source[0] or source[3] <= source[1]:
        raise ValueError("Crop contains no valid image pixels")
    mean = model["mean"]
    fill = tuple(int(round(255.0 * float(value))) for value in mean)
    crop = Image.new("RGB", (right - left, bottom - top), fill)
    valid = Image.new("L", crop.size, 0)
    offset = source[0] - left, source[1] - top
    source_crop = image.crop(source)
    crop.paste(source_crop, offset)
    valid.paste(255, (*offset, offset[0] + source_crop.width, offset[1] + source_crop.height))
    size = int(model["input_size"])
    scale = size / max(crop.size)
    resized = tuple(max(1, min(size, int(round(value * scale)))) for value in crop.size)
    crop = crop.resize(resized, Image.Resampling.BICUBIC)
    valid = valid.resize(resized, Image.Resampling.NEAREST)
    if not np.asarray(valid).any():
        raise ValueError("Crop contains no valid image pixels after resizing")
    offset = (size - resized[0]) // 2, (size - resized[1]) // 2
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(crop, offset)
    pixels = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
    pixels = (pixels - torch.as_tensor(mean, dtype=torch.float32)[:, None, None]) / torch.as_tensor(model["std"], dtype=torch.float32)[:, None, None]
    if model["kind"] == "image_tower":
        return pixels, None
    mask = Image.new("L", (size, size), 0)
    mask.paste(valid, offset)
    valid_pixels = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)
    fractions = F.avg_pool2d(valid_pixels[None, None], 16, 16).flatten()
    minimum = float(model.get("minimum_valid_patch_fraction", 0.5))
    patches = fractions > 0.0 if minimum == 0.0 else fractions >= minimum
    if not patches.any():
        patches[patches.numel() // 2] = True
    return pixels, patches


def predict_crop(
    model: nn.Module, pixels: torch.Tensor, valid: torch.Tensor | None, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.device(device)
    inputs = (pixels.unsqueeze(0).to(target),)
    if valid is not None:
        inputs += (valid.unsqueeze(0).to(target),)
    with torch.inference_mode(), torch.autocast(device_type=target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
        logits, objectness = model(*inputs)
    if logits.shape != (1, CLASS_COUNT) or objectness.shape != (1,):
        raise ValueError(f"Unexpected classifier output shapes: {tuple(logits.shape)}, {tuple(objectness.shape)}")
    if not torch.isfinite(logits).all() or not torch.isfinite(objectness).all():
        raise ValueError("Classifier produced non-finite outputs")
    return logits.float().cpu(), objectness.float().cpu()


def check(
    model_name: str,
    checkpoint: Path,
    image_path: Path,
    box_xywh: Sequence[float],
    *,
    device: str = "cuda:0",
    manifest: Path = DEFAULT_MANIFEST,
    config: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("Classifier check supports CPU or BF16-capable CUDA")
    if target.type == "cuda":
        with torch.cuda.device(target):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Classifier check requires BF16-capable CUDA")
    model_config = load_model_config(model_name, config)
    from inference.detector_geometry import decode_encoded_rgb
    image = decode_encoded_rgb(image_path)
    pixels, valid = render_input(image, box_xywh, model_config)
    model = load_classifier(model_name, checkpoint, manifest=manifest, config=config, device=target)
    logits, objectness = predict_crop(model, pixels, valid, target)
    probabilities = torch.softmax(logits.double(), dim=1).float()[0]
    foreground = torch.sigmoid(objectness.double().clamp(-50.0, 50.0)).float()[0]
    return {
        "model_name": model_name,
        "checkpoint": str(Path(checkpoint).resolve()),
        "image": str(Path(image_path).resolve()),
        "image_size": list(image.size),
        "box_xywh": list(map(float, box_xywh)),
        "context_rectangle_xyxy": list(crop_rectangle(box_xywh, float(model_config["context_scale"]))),
        "input_shape": [1, *pixels.shape],
        "valid_patches": int(valid.sum()) if valid is not None else None,
        "precision": "bfloat16_autocast" if target.type == "cuda" else "float32",
        "deployment": "merged_lora" if model_name == "hplus" else "full_model",
        "catalog_identity_verified": True,
        "state_dict_loaded_strictly": True,
        "renderable": True,
        "outputs_finite": True,
        "class_logits_shape": list(logits.shape),
        "objectness_logits_shape": list(objectness.shape),
        "class_logits": logits[0].tolist(),
        "foreground_probabilities": probabilities.tolist(),
        "objectness_logit": float(objectness[0]),
        "objectness_probability": float(foreground),
        "top_class_id": int(probabilities.argmax()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_ARCHITECTURES))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--box", nargs=4, type=float, required=True, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        result = check(args.model, args.checkpoint, args.image, args.box, device=args.device, manifest=args.manifest)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Classifier check failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
