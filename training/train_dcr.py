#!/usr/bin/env python3
"""Train one frozen final-fit DCR recipe without validation or selection."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.environment import configure_caches

if __name__ == "__main__":
    configure_caches()

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from timm.optim import param_groups_layer_decay

from training.models import (
    DINOAttentionDCR,
    HPlusDCR,
    ImageTowerDCR,
    crop_rectangle,
)
from training.prepare_dcr_records import verify_records_manifest


CLASS_COUNT = 34
BACKGROUND = 34


@dataclass(frozen=True)
class Record:
    image: Path
    box: tuple[float, float, float, float]
    label: int
    source: str
    gt_id: int | None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_recipe(path: Path, name: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    common = document["dcr"]["common"]
    model = document["dcr"][name]
    return {**common, **model, "name": name}


def load_records(path: Path, image_root: Path) -> list[Record]:
    records = []
    root = image_root.resolve()
    image_hashes = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        relative = Path(str(row["image_path"]))
        if relative.is_absolute():
            raise ValueError(f"record {line_number} requires a relative image_path")
        if ".." in relative.parts:
            raise ValueError(f"record {line_number} escapes --image-root")
        image = (root / relative).resolve()
        label = int(row["label"])
        box = tuple(map(float, row["box_xyxy"]))
        if not image.is_file() or len(box) != 4 or not 0 <= label <= BACKGROUND or not all(map(math.isfinite, box)) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"invalid record {line_number}")
        if row.get("image_sha256"):
            if image not in image_hashes:
                image_hashes[image] = sha256(image)
            if image_hashes[image] != row["image_sha256"]:
                raise ValueError(f"image bytes changed for record {line_number}")
        records.append(
            Record(
                image=image,
                box=box,
                label=label,
                source=str(row["source_type"]),
                gt_id=(
                    int(row["gt_instance_id"])
                    if row.get("gt_instance_id") is not None
                    else None
                ),
            )
        )
    if not records:
        raise ValueError("record file is empty")
    return records


def filter_hplus_records(
    records: Sequence[Record], recipe: Mapping[str, Any]
) -> list[Record]:
    """Require at least 2x2 expected valid pixels at the fallback resolution."""
    audit_size = min(int(recipe["input_size"]), 384)
    dimensions = {}
    renderable = []
    for record in records:
        if record.image not in dimensions:
            with Image.open(record.image) as image:
                dimensions[record.image] = image.size
        width, height = dimensions[record.image]
        x1, y1, x2, y2 = record.box
        left, top, right, bottom = crop_rectangle(
            (x1, y1, x2 - x1, y2 - y1), float(recipe["context_scale"])
        )
        valid_width = max(0, min(right, width) - max(left, 0))
        valid_height = max(0, min(bottom, height) - max(top, 0))
        crop_width, crop_height = right - left, bottom - top
        scale = audit_size / max(crop_width, crop_height)
        resized_width = max(1, min(audit_size, int(round(crop_width * scale))))
        resized_height = max(1, min(audit_size, int(round(crop_height * scale))))
        if (
            valid_width * resized_width / crop_width >= 2.0
            and valid_height * resized_height / crop_height >= 2.0
        ):
            renderable.append(record)
        elif record.source == "gt_original" and record.label < BACKGROUND:
            raise RuntimeError(f"H+ contains invalid original GT crops: {record.image} {record.box}")
    return renderable


def sampling_pools(records: Sequence[Record]) -> dict[str, Any]:
    sources: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    classes: dict[int, set[int]] = defaultdict(set)
    hard, regular = [], []
    for index, record in enumerate(records):
        if record.label < CLASS_COUNT:
            if record.gt_id is None:
                raise ValueError("foreground records require gt_instance_id")
            source = (
                "gt_jitter"
                if record.source.startswith("gt_jitter")
                else record.source
            )
            sources[record.gt_id][source].append(index)
            classes[record.label].add(record.gt_id)
        elif record.source == "background_hard":
            hard.append(index)
        elif record.source == "background_regular":
            regular.append(index)
    if set(classes) != set(range(CLASS_COUNT)) or not hard or not regular:
        raise ValueError("records do not satisfy the 34-class foreground/background contract")
    return {
        "sources": sources,
        "classes": {key: sorted(value) for key, value in classes.items()},
        "hard": hard,
        "regular": regular,
    }


def choose_gt(choices: Sequence[int], used: set[int], rng: random.Random) -> int:
    start = rng.randrange(len(choices))
    for offset in range(len(choices)):
        value = choices[(start + offset) % len(choices)]
        if value not in used:
            return value
    return choices[start]


def schedule(
    records: Sequence[Record],
    pools: Mapping[str, Any],
    draws: int,
    effective_batch: int,
    seed: int,
    epoch: int,
) -> list[tuple[int, int]]:
    if effective_batch % 3 or draws % effective_batch:
        raise ValueError("draws and effective batch must be divisible by three")
    foreground_slots = 2 * effective_batch // 3
    background_slots = effective_batch - foreground_slots
    rng = random.Random(f"{seed}:{epoch}:v7")
    output = []
    label_cursor = rng.randrange(CLASS_COUNT)
    for _ in range(draws // effective_batch):
        used: set[int] = set()
        labels = [
            (label_cursor + offset) % CLASS_COUNT
            for offset in range(foreground_slots)
        ]
        label_cursor = (label_cursor + foreground_slots) % CLASS_COUNT
        rng.shuffle(labels)
        batch = []
        for label in labels:
            gt_id = choose_gt(pools["classes"][label], used, rng)
            used.add(gt_id)
            available = [
                name
                for name in (
                    "gt_original",
                    "gt_jitter",
                    "proposal_matched",
                    "proposal_wrong_class",
                )
                if pools["sources"][gt_id].get(name)
            ]
            if not available:
                raise ValueError(f"GT {gt_id} has no supported foreground source")
            index = rng.choice(pools["sources"][gt_id][rng.choice(available)])
            batch.append((index, rng.getrandbits(63)))
        for slot in range(background_slots):
            pool = pools["hard"] if slot < math.ceil(0.75 * background_slots) else pools["regular"]
            batch.append((rng.choice(pool), rng.getrandbits(63)))
        rng.shuffle(batch)
        output.extend(batch)
    return output


def transformed_box(
    box: Sequence[float], rng: random.Random, augmentation: Mapping[str, Any]
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, box)
    width, height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    shift = float(augmentation["translation_fraction"])
    center_x += rng.uniform(-shift, shift) * width
    center_y += rng.uniform(-shift, shift) * height
    scale = rng.uniform(*map(float, augmentation["scale_range"]))
    width, height = max(1.0, width * scale), max(1.0, height * scale)
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def render(
    image: Image.Image,
    box: Sequence[float],
    recipe: Mapping[str, Any],
    rng: random.Random,
    *,
    augment: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    augmentation = recipe["augmentation"]
    if not augment:
        augmentation = {"translation_fraction": 0.0, "scale_range": [1.0, 1.0], "horizontal_flip_probability": 0.0, "rotation_degrees": 0.0, "brightness_range": [1.0, 1.0], "contrast_range": [1.0, 1.0], "saturation_range": [1.0, 1.0], "blur_probability": 0.0, "jpeg_probability": 0.0}
    x1, y1, x2, y2 = transformed_box(box, rng, augmentation)
    rectangle = crop_rectangle(
        (x1, y1, x2 - x1, y2 - y1), float(recipe["context_scale"])
    )
    left, top, right, bottom = rectangle
    mean = recipe["mean"]
    fill = tuple(int(round(255 * float(value))) for value in mean)
    crop = Image.new("RGB", (right - left, bottom - top), fill)
    valid = Image.new("L", crop.size, 0)
    source = max(left, 0), max(top, 0), min(right, image.width), min(bottom, image.height)
    if source[2] > source[0] and source[3] > source[1]:
        offset = source[0] - left, source[1] - top
        pixels = image.crop(source)
        crop.paste(pixels, offset)
        valid.paste(255, (*offset, offset[0] + pixels.width, offset[1] + pixels.height))
    if rng.random() < float(augmentation["horizontal_flip_probability"]):
        crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        valid = valid.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    angle = rng.uniform(
        -float(augmentation["rotation_degrees"]),
        float(augmentation["rotation_degrees"]),
    )
    crop = crop.rotate(angle, Image.Resampling.BICUBIC, fillcolor=fill)
    valid = valid.rotate(angle, Image.Resampling.NEAREST, fillcolor=0)
    crop = ImageEnhance.Brightness(crop).enhance(
        rng.uniform(*map(float, augmentation["brightness_range"]))
    )
    crop = ImageEnhance.Contrast(crop).enhance(
        rng.uniform(*map(float, augmentation["contrast_range"]))
    )
    crop = ImageEnhance.Color(crop).enhance(
        rng.uniform(*map(float, augmentation["saturation_range"]))
    )
    if rng.random() < float(augmentation["blur_probability"]):
        crop = crop.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 1.0)))
    if rng.random() < float(augmentation["jpeg_probability"]):
        encoded = io.BytesIO()
        crop.save(encoded, format="JPEG", quality=rng.randint(75, 95))
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            crop = decoded.convert("RGB")
    crop.paste(fill, mask=ImageOps.invert(valid))
    size = int(recipe["input_size"])
    scale = size / max(crop.size)
    resized = tuple(max(1, min(size, int(round(value * scale)))) for value in crop.size)
    crop = crop.resize(resized, Image.Resampling.BICUBIC)
    valid = valid.resize(resized, Image.Resampling.NEAREST)
    offset = (size - resized[0]) // 2, (size - resized[1]) // 2
    canvas = Image.new("RGB", (size, size), fill)
    mask = Image.new("L", (size, size), 0)
    canvas.paste(crop, offset)
    mask.paste(valid, offset)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    tensor = (
        tensor - torch.as_tensor(mean, dtype=torch.float32)[:, None, None]
    ) / torch.as_tensor(recipe["std"], dtype=torch.float32)[:, None, None]
    if recipe["kind"] not in {"dinov3_attention", "dinov3_hplus_lora"}:
        return tensor, None
    valid_pixels = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)
    fractions = F.avg_pool2d(valid_pixels[None, None], 16, 16).flatten()
    patches = fractions > 0.0 if recipe["kind"] == "dinov3_hplus_lora" else fractions >= 0.5
    if not patches.any():
        if recipe["kind"] == "dinov3_hplus_lora":
            raise ValueError("H+ crop contains no valid image pixels")
        patches[patches.numel() // 2] = True
    return tensor, patches


class CropDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Record],
        indices: Sequence[tuple[int, int]],
        recipe: Mapping[str, Any],
    ) -> None:
        self.records = records
        self.indices = indices
        self.recipe = recipe

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        record_index, seed = self.indices[index]
        record = self.records[record_index]
        with Image.open(record.image) as opened:
            image = opened.convert("RGB")
            try:
                pixels, valid = render(image, record.box, self.recipe, random.Random(seed))
            except ValueError:
                if self.recipe["kind"] != "dinov3_hplus_lora":
                    raise
                pixels, valid = render(image, record.box, self.recipe, random.Random(seed), augment=False)
        return (pixels, valid, record.label) if valid is not None else (pixels, record.label)


def worker_init(_worker: int) -> None:
    torch.set_num_threads(1)


def schedule_multiplier(
    update: int,
    total_updates: int,
    *,
    warmup_fraction: float,
    minimum_ratio: float,
) -> float:
    warmup = math.ceil(total_updates * warmup_fraction)
    if warmup and update < warmup:
        return (update + 1) / warmup
    denominator = max(1, total_updates - warmup - 1)
    progress = min(1.0, max(0.0, (update - warmup) / denominator))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def dino_parameter_groups(
    model: torch.nn.Module, recipe: Mapping[str, Any]
) -> list[dict[str, Any]]:
    depth = len(model.backbone.blocks)
    maximum_layer = depth + 1
    grouped: dict[tuple[float, float], list[torch.nn.Parameter]] = defaultdict(list)
    backbone_lr = float(recipe["trunk_lr"])
    layer_decay = float(recipe["layer_decay"])
    weight_decay = float(recipe["weight_decay"])
    for name, parameter in model.backbone.named_parameters():
        layer = 0
        if name.startswith("blocks."):
            token = name.split(".", 2)[1]
            if token.isdigit():
                layer = int(token) + 1
        elif name.startswith(("norm.", "cls_norm.")):
            layer = maximum_layer
        learning_rate = backbone_lr * layer_decay ** (maximum_layer - layer)
        no_decay = parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()
        grouped[learning_rate, 0.0 if no_decay else weight_decay].append(parameter)
    for name, parameter in model.head.named_parameters():
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        grouped[float(recipe["head_lr"]), 0.0 if no_decay else weight_decay].append(
            parameter
        )
    return [
        {
            "params": parameters,
            "lr": learning_rate,
            "peak_lr": learning_rate,
            "weight_decay": decay,
        }
        for (learning_rate, decay), parameters in sorted(grouped.items())
    ]


def optimizer_groups(
    model: torch.nn.Module, recipe: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if recipe["kind"] == "dinov3_hplus_lora":
        adapter = [parameter for name, parameter in model.backbone.named_parameters() if name.endswith(("lora_A", "lora_B"))]
        if {id(parameter) for parameter in adapter} != {id(parameter) for parameter in model.backbone.parameters() if parameter.requires_grad}:
            raise ValueError("H+ backbone may train only its LoRA adapters")
        groups = [{"params": adapter, "lr": float(recipe["adapter_lr"]), "peak_lr": float(recipe["adapter_lr"]), "weight_decay": float(recipe["weight_decay"]), "kind": "adapter"}]
        for no_decay in (False, True):
            parameters = [parameter for name, parameter in model.head.named_parameters() if (parameter.ndim <= 1 or "norm" in name.lower() or name.endswith("bias")) == no_decay]
            groups.append({"params": parameters, "lr": float(recipe["head_lr"]), "peak_lr": float(recipe["head_lr"]), "weight_decay": 0.0 if no_decay else float(recipe["weight_decay"]), "kind": "head"})
        return groups
    if recipe["kind"] == "dinov3_attention":
        return dino_parameter_groups(model, recipe)
    groups = param_groups_layer_decay(
        model.encoder,
        weight_decay=float(recipe["weight_decay"]),
        layer_decay=float(recipe["layer_decay"]),
    )
    for group in groups:
        peak = float(recipe["trunk_lr"]) * float(group.pop("lr_scale"))
        group["lr"] = group["peak_lr"] = peak
    head_decay, head_no_decay = [], []
    for module in (model.foreground, model.objectness):
        for parameter in module.parameters():
            (head_no_decay if parameter.ndim <= 1 else head_decay).append(parameter)
    groups.extend(
        [
            {
                "params": head_decay,
                "lr": float(recipe["head_lr"]),
                "peak_lr": float(recipe["head_lr"]),
                "weight_decay": float(recipe["weight_decay"]),
            },
            {
                "params": head_no_decay,
                "lr": float(recipe["head_lr"]),
                "peak_lr": float(recipe["head_lr"]),
                "weight_decay": 0.0,
            },
        ]
    )
    return groups


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimizer(model: torch.nn.Module, recipe: Mapping[str, Any]):
    groups = optimizer_groups(model, recipe)
    if recipe["optimizer_backend"] == "native":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if recipe["optimizer_backend"] != "fused_or_foreach":
        raise ValueError("unsupported DCR optimizer backend")
    try:
        return torch.optim.AdamW(groups, betas=(0.9, 0.999), fused=True)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(groups, betas=(0.9, 0.999), foreach=True)


def initialize_training(factory, recipe: Mapping[str, Any], device: torch.device):
    if recipe["seed_before_initialization"]:
        seed_everything(int(recipe["seed"]))
    model = factory().to(device)
    if recipe.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    optimizer = create_optimizer(model, recipe)
    if not recipe["seed_before_initialization"]:
        # ViT-L's fixed source seeds training after constructing its readout.
        seed_everything(int(recipe["seed"]))
    return model, optimizer


def training_runtime(model, recipe: Mapping[str, Any], cache_root: Path):
    mode = recipe.get("compile_mode")
    if mode is None:
        return model
    for variable, directory in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")):
        path = (cache_root / directory).resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(path))
    threads = min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(threads))
    return torch.compile(model, mode=str(mode))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("v7", "siglip", "hplus"), required=True)
    parser.add_argument("--recipes", type=Path, default=Path("training/final_recipes.yaml"))
    parser.add_argument("--records", type=Path)
    parser.add_argument("--records-manifest", type=Path, help="own record sidecar; defaults beside --records")
    parser.add_argument("--records-sha256", help="optional explicit SHA256 pin for the prepared records")
    parser.add_argument("--allow-new-records", action="store_true", help="accept a prepared record source with a different identity")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--initial-backbone", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, help="Data-loader workers; defaults to the fixed recipe")
    parser.add_argument("--max-steps", type=int, help="Stop after this many real optimizer updates and write a partial recipe checkpoint")
    args = parser.parse_args()
    configure_caches()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.workers is not None and not 0 <= args.workers <= 28:
        raise ValueError("--workers must be in 0..28")
    if args.output.exists():
        raise FileExistsError(args.output)
    recipe = load_recipe(args.recipes, args.model)
    workers = int(recipe["workers"]) if args.workers is None else args.workers
    records_path = (args.records or Path(recipe["records"])).expanduser().resolve()
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    records_identity = verify_records_manifest(records_path, allow_new=args.allow_new_records,
        manifest=args.records_manifest, expected_records_sha256=args.records_sha256)
    records_sha = records_identity["records_sha256"]
    print(f"records_origin={records_identity['origin']} source_sha256={records_identity['source_sha256']} records_sha256={records_sha}", flush=True)
    records = load_records(records_path, args.image_root)
    if len(records) != records_identity.get("rows"):
        raise ValueError("DCR record count differs from its preparation manifest")
    if recipe["kind"] == "dinov3_hplus_lora":
        records = filter_hplus_records(records, recipe)
    pools = sampling_pools(records)
    model_config = {
        "name": args.model,
        "kind": recipe["kind"],
        "architecture": recipe["architecture"],
        "input_size": recipe["input_size"],
        "context_scale": recipe["context_scale"],
        "mean": recipe["mean"],
        "std": recipe["std"],
        "attention_heads": recipe.get("attention_heads"),
        "projection_dimensions": recipe.get("projection_dimensions"),
        "adapter": recipe.get("adapter"),
    }
    if recipe["kind"] in {"dinov3_attention", "dinov3_hplus_lora"}:
        if args.initial_backbone is None:
            raise ValueError("DINOv3 training requires --initial-backbone")
        initial = args.initial_backbone.expanduser().resolve()
        if not initial.is_file() or sha256(initial) != recipe["initialization_sha256"]:
            raise ValueError("DINOv3 initialization checkpoint is missing or has the wrong SHA256")
        model_class = HPlusDCR if recipe["kind"] == "dinov3_hplus_lora" else DINOAttentionDCR
        factory = lambda: model_class(model_config, pretrained_weights=initial)
    else:
        if args.initial_backbone is not None:
            raise ValueError("--initial-backbone is only valid for DINOv3 models")
        factory = lambda: ImageTowerDCR(
            model_config,
            pretrained_model=(
                f"hf_hub:{recipe['pretrained_repo']}@{recipe['pretrained_revision']}"
            ),
        )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("final-fit DCR training requires a CUDA device")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = bool(recipe.get("cudnn_benchmark", False))
    model, optimizer = initialize_training(factory, recipe, device)
    runtime = training_runtime(model, recipe, args.output.parent / "compiler_cache")
    trunk = model.backbone if hasattr(model, "backbone") else model.encoder
    effective = int(recipe["effective_batch"])
    physical = int(recipe["physical_batch"])
    if effective % physical:
        raise ValueError("effective batch must be divisible by physical batch")
    accumulation = effective // physical
    updates_per_epoch = int(recipe["draws_per_epoch"]) // effective
    total_updates = int(recipe["epochs"]) * updates_per_epoch
    if args.max_steps is not None and args.max_steps > total_updates:
        raise ValueError("--max-steps exceeds the complete recipe's optimizer updates")
    update = 0
    completed_epochs = 0
    for epoch in range(1, int(recipe["epochs"]) + 1):
        train_trunk = epoch > int(recipe["warmup_epochs"])
        if recipe["kind"] == "dinov3_hplus_lora":
            model.set_head_only(not train_trunk)
        else:
            for parameter in trunk.parameters():
                parameter.requires_grad_(train_trunk)
        indices = schedule(
            records,
            pools,
            int(recipe["draws_per_epoch"]),
            effective,
            int(recipe["seed"]),
            epoch + int(recipe.get("epoch_offset", 0)),
        )
        if args.max_steps is not None:
            indices = indices[:max(0, args.max_steps - update) * effective]
        loader_options: dict[str, Any] = {}
        if workers > 0:
            loader_options.update(
                persistent_workers=True,
                worker_init_fn=worker_init,
                prefetch_factor=int(recipe.get("prefetch_factor", 2)),
            )
        loader = DataLoader(
            CropDataset(records, indices, recipe),
            batch_size=physical,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
            **loader_options,
        )
        model.train()
        trunk.eval() if not train_trunk else trunk.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            pixels = batch[0].to(device, non_blocking=True)
            if recipe.get("channels_last", False):
                pixels = pixels.contiguous(memory_format=torch.channels_last)
            if recipe["kind"] in {"dinov3_attention", "dinov3_hplus_lora"}:
                valid = batch[1].to(device, non_blocking=True)
                labels = batch[2].to(device, non_blocking=True)
                inputs = pixels, valid
            else:
                labels = batch[1].to(device, non_blocking=True)
                inputs = (pixels,)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, objectness = runtime(*inputs)
                foreground = labels < BACKGROUND
                if foreground.any():
                    class_loss = F.cross_entropy(
                        logits[foreground],
                        labels[foreground],
                        label_smoothing=float(recipe["label_smoothing"]),
                        reduction="sum",
                    ) * accumulation / (2 * effective // 3)
                else:
                    class_loss = logits.sum() * 0.0
                object_loss = F.binary_cross_entropy_with_logits(
                    objectness, foreground.to(objectness.dtype)
                )
                loss = (
                    class_loss + float(recipe["objectness_weight"]) * object_loss
                ) / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite DCR training loss")
            loss.backward()
            if step % accumulation:
                continue
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"]))
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite DCR gradient norm")
            scale = schedule_multiplier(
                update,
                total_updates,
                warmup_fraction=float(recipe["warmup_fraction"]),
                minimum_ratio=float(recipe["minimum_lr_ratio"]),
            )
            for group in optimizer.param_groups:
                group_scale = scale
                if group.get("kind") == "adapter":
                    warmup_updates = int(recipe["warmup_epochs"]) * updates_per_epoch
                    group_scale = 0.0 if not train_trunk else schedule_multiplier(
                        max(0, update - warmup_updates), max(1, total_updates - warmup_updates),
                        warmup_fraction=float(recipe["warmup_fraction"]), minimum_ratio=float(recipe["minimum_lr_ratio"]),
                    )
                group["lr"] = float(group["peak_lr"]) * group_scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            if args.max_steps is not None and update >= args.max_steps:
                break
        if update >= epoch * updates_per_epoch:
            completed_epochs = epoch
        print(f"epoch={epoch}/{recipe['epochs']} updates={update}", flush=True)
        if args.max_steps is not None and update >= args.max_steps:
            break
    if args.max_steps is not None and update != args.max_steps:
        raise RuntimeError("DCR training did not complete the requested optimizer updates")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    state = {"model": model.state_dict()}
    if recipe["kind"] == "dinov3_hplus_lora":
        state = {"trainable_state": {"adapter": {name: value.detach().cpu() for name, value in model.backbone.state_dict().items() if name.endswith(("lora_A", "lora_B"))}, "head": {name: value.detach().cpu() for name, value in model.head.state_dict().items()}}}
    torch.save(
        {
            **state,
            "completed_epoch": completed_epochs,
            "optimizer_updates": update,
            "requested_max_steps": args.max_steps,
            "full_recipe_completed": update == total_updates,
            "recipe_sha256": sha256(args.recipes),
            "records_sha256": records_sha,
            "records_source_sha256": records_identity["source_sha256"],
            "records_origin": records_identity["origin"],
        },
        partial,
    )
    partial.replace(args.output)


if __name__ == "__main__":
    main()
