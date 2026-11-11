#!/usr/bin/env python3
"""Run one portable fixed detector recipe without metric-based selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from scripts.environment import configure_caches
from training.local_ultralytics import local_ultralytics
from training.rfdetr_copy_paste import ClassAwareCopyPaste, CopyPasteSettings
from training.prepare_sam_masks import verify_mask_annotations
from training.sam_copy_paste import (
    SAMCopyPasteTinyFormerDataset,
    rebuild_tinyformer_train_loader,
)


ROOT = Path(__file__).resolve().parents[1]
CLASS_COUNT = 34
TINYFORMER_CONFIG = Path(
    "third_party/TinyFormer/configs/tinyformer/object365/"
    "tinyformer_dinov3_xl_obj2coco_pbm_24e.yml"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def recipes(path: Path, model: str) -> dict[str, Any]:
    document = load_yaml(path)
    try:
        value = document["detectors"][model]
    except KeyError as error:
        raise ValueError(f"recipe does not define detector {model!r}") from error
    if not isinstance(value, dict):
        raise ValueError(f"detector recipe {model!r} is not a mapping")
    return value


def load_data(path: Path) -> dict[str, Any]:
    document = load_yaml(path)
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("unsupported training data contract")
    expected = json.loads(
        (ROOT / "configs/categories.json").read_text(encoding="utf-8")
    )["categories"]
    names = [str(row["name"]) for row in expected]
    if document.get("class_names") != names:
        raise ValueError("training class_names differ from configs/categories.json")
    return document


def data_path(data_file: Path, data: Mapping[str, Any], *keys: str) -> Path:
    value: Any = data
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"training data contract is missing {'.'.join(keys)}")
        value = value[key]
    path = resolve(data_file.parent, str(value))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def verify_initial(
    path: Path, recipe: Mapping[str, Any], supplied_sha256: str | None
) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = recipe.get("initialization_sha256") or supplied_sha256
    if not expected:
        raise ValueError(
            "this recipe has no redistributable initialization identity; "
            "provide --initial-sha256"
        )
    actual = sha256(path)
    if actual != str(expected):
        raise ValueError(
            f"initial checkpoint SHA256 mismatch: expected {expected}, found {actual}"
        )
    expected_bytes = recipe.get("initialization_bytes")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise ValueError("initial checkpoint byte size does not match the recipe")
    return actual


def verify_file_identity(
    path: Path, expected_sha256: str, expected_bytes: int, label: str
) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, found {actual}"
        )
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte size does not match the recipe")
    return actual


def prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def install_vendor(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"missing pinned submodule {path}; run git submodule update --init --recursive"
        )
    for name in tuple(sys.modules):
        if name == "engine" or name.startswith("engine."):
            del sys.modules[name]
    sys.path.insert(0, str(path))


def set_tinyformer_resolution(config: dict[str, Any], resolution: int) -> None:
    size = [resolution, resolution]
    config["eval_spatial_size"] = size
    config["train_dataloader"]["collate_fn"]["base_size"] = resolution
    for loader_name in ("train_dataloader", "val_dataloader"):
        operations = config[loader_name]["dataset"]["transforms"]["ops"]
        resize = [row for row in operations if row.get("type") == "Resize"]
        if len(resize) != 1:
            raise ValueError("TinyFormer requires exactly one Resize operation")
        resize[0]["size"] = size.copy()
    mosaic = [
        row
        for row in config["train_dataloader"]["dataset"]["transforms"]["ops"]
        if row.get("type") == "Mosaic"
    ]
    if len(mosaic) != 1:
        raise ValueError("TinyFormer requires exactly one Mosaic operation")
    mosaic[0]["output_size"] = resolution // 2


def cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def run_tinyformer(
    recipe: Mapping[str, Any],
    data: Mapping[str, Any],
    data_file: Path,
    initial: Path,
    output: Path,
    device: str,
    workers: int,
    max_steps: int | None = None,
) -> Path:
    vendor = ROOT / "third_party/TinyFormer"
    install_vendor(vendor)
    from engine.core import YAMLConfig
    from engine.data._misc import convert_to_tv_tensor
    from engine.optim.lr_scheduler import flat_cosine_schedule
    from pytorch_lightning import Trainer, seed_everything
    from training.tinyformer_training import TinyFormerEMA, TinyFormerModule, load_pretrained

    images = data_path(data_file, data, "coco", "images")
    annotations = data_path(data_file, data, "coco", "annotations_0based")
    class_names = list(map(str, data["class_names"]))
    loader = {
        "total_batch_size": int(recipe["batch"]),
        "num_workers": workers,
    }
    overrides = {
        "num_classes": CLASS_COUNT,
        "remap_mscoco_category": False,
        "epoches": int(recipe["epochs"]),
        "device": device,
        "output_dir": str(output),
        "use_amp": False,
        "use_ema": True,
        "lr_gamma": float(recipe["lr_gamma"]),
        "flat_epoch": int(recipe["flat_epoch"]),
        "no_aug_epoch": int(recipe["no_aug_epoch"]),
        "warmup_iter": int(recipe["warmup_iter"]),
        "optimizer": {"lr": float(recipe["learning_rate"])},
        "ema": {"decay": float(recipe["ema_decay"]), "warmups": int(recipe["ema_warmups"])},
        "train_dataloader": {
            **loader,
            "dataset": {
                "img_folder": str(images),
                "ann_file": str(annotations),
                "transforms": {
                    "policy": {"epoch": list(recipe["augmentation_policy_epochs"])}
                },
            },
            "collate_fn": {
                "mixup_epochs": list(recipe["mixup_epochs"]),
                "copyblend_epochs": list(recipe["copyblend_epochs"]),
                "stop_epoch": int(recipe["multi_scale_stop_epoch"]),
                "copyblend_prob": 0.0,
            },
        },
        # Keep the native configuration complete; the trainer disables validation.
        "val_dataloader": {
            **loader,
            "dataset": {"img_folder": str(images), "ann_file": str(annotations)},
        },
    }
    categories = json.loads(annotations.read_text(encoding="utf-8"))["categories"]
    if {int(row["id"]): row["name"] for row in categories} != dict(enumerate(class_names)):
        raise ValueError("TinyFormer requires category IDs 0..33 in the fixed class order")
    seed_everything(int(recipe["seed"]), workers=True)
    torch.set_float32_matmul_precision("high")
    native = YAMLConfig(str(ROOT / TINYFORMER_CONFIG), **overrides)
    set_tinyformer_resolution(native.yaml_cfg, int(recipe["resolution"]))
    model = native.model
    initialization = load_pretrained(model, initial)
    criterion, optimizer = native.criterion, native.optimizer
    from training.portable_sam_bank import bank_settings
    sam = bank_settings(dict(recipe["sam_copy_paste"]), data_path(data_file, data, "tinyformer", "sam_object_bank"),
        expected_sha256=data["tinyformer"].get("sam_bank_sha256"), image_root=images)
    wrapped = SAMCopyPasteTinyFormerDataset(native.train_dataloader.dataset, sam, convert_to_tv_tensor)
    loader = rebuild_tinyformer_train_loader(native.train_dataloader, wrapped)
    ema = native.ema
    if ema is None:
        raise ValueError("TinyFormer requires EMA")
    module = TinyFormerModule(
        model, criterion, optimizer, loader, native.yaml_cfg, flat_cosine_schedule,
    )
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("TinyFormer requires a BF16-capable CUDA device")
    trainer = Trainer(
        accelerator="gpu", devices=[target.index or 0], max_epochs=int(recipe["epochs"]),
        precision=str(recipe["precision"]), accumulate_grad_batches=int(recipe["gradient_accumulation"]),
        gradient_clip_val=float(recipe["gradient_clip"]), callbacks=[TinyFormerEMA(ema)],
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, limit_val_batches=0, deterministic=False,
        default_root_dir=str(output),
        max_steps=max_steps if max_steps is not None else -1,
    )
    trainer.fit(module, train_dataloaders=loader)
    if max_steps is not None and int(trainer.global_step) != max_steps:
        raise RuntimeError("TinyFormer did not complete the requested optimizer updates")
    artifact = output / "checkpoint_terminal_ema.pth"
    torch.save({
        "schema_version": 1, "backend": "tinyformer", "model": cpu_state(ema.module),
        "best_total_source": "terminal_ema", "num_classes": CLASS_COUNT,
        "class_names": class_names, "tinyformer_config": native.yaml_cfg,
        "source_commit": recipe["source_revision"],
        "source_config": str(TINYFORMER_CONFIG.relative_to("third_party/TinyFormer")),
        "pretrained": {"sha256": sha256(initial), **initialization},
        "training": {
            "seed": int(recipe["seed"]), "precision": str(trainer.precision),
            "physical_batch": int(loader.batch_size), "gradient_accumulation": int(recipe["gradient_accumulation"]),
            "effective_batch": int(loader.batch_size) * int(recipe["gradient_accumulation"]),
            "gradient_clip": float(recipe["gradient_clip"]), "loss": "native_detection_criterion_fp32",
            "completed_epochs": int(trainer.current_epoch), "optimizer_updates": int(trainer.global_step),
            "ema_updates": int(ema.updates),
            "requested_max_steps": max_steps,
            "full_recipe_completed": max_steps is None,
        },
    }, artifact)
    return artifact


def rfdetr_configs(recipe, data, data_file, initial, output, device, workers):
    from rfdetr.config import RFDETRLargeDeprecatedConfig, TrainConfig
    model_config = RFDETRLargeDeprecatedConfig(
        pretrain_weights=str(initial), num_classes=CLASS_COUNT, resolution=int(recipe["resolution"]),
        model_name="RFDETRLargeDeprecated", device=device, amp=True,
    )
    train_config = TrainConfig(
        dataset_dir=str(data_path(data_file, data, "rfdetr", "dataset_dir")),
        output_dir=str(output), epochs=int(recipe["epochs"]), batch_size=int(recipe["batch"]),
        grad_accum_steps=int(recipe["gradient_accumulation"]), seed=int(recipe["seed"]),
        lr=float(recipe["learning_rate"]), lr_encoder=float(recipe["encoder_learning_rate"]),
        weight_decay=float(recipe["weight_decay"]), drop_path=float(recipe["drop_path"]),
        lr_scheduler="step", lr_scheduler_kwargs={"lr_drop": int(recipe["lr_drop"])},
        warmup_epochs=float(recipe["warmup_epochs"]), use_ema=True, checkpoint_interval=1,
        num_workers=workers, augmentation_backend="torchvision", amp_dtype="bf16",
        clip_max_norm=float(recipe["gradient_clip"]), accelerator="gpu", devices=1,
        class_names=list(data["class_names"]), early_stopping=False, run_test=False,
        eval_interval=int(recipe["epochs"]) + 1, tensorboard=False, wandb=False, mlflow=False,
        clearml=False, save_dataset_grids=False,
    )
    return model_config, train_config


def run_rfdetr(
    recipe: Mapping[str, Any],
    data: Mapping[str, Any],
    data_file: Path,
    initial: Path,
    output: Path,
    device: str,
    workers: int,
    max_steps: int | None = None,
) -> Path:
    from importlib.metadata import version
    from rfdetr.datasets._torchvision import Compose, RandomHorizontalFlip
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    if version("rfdetr") != "1.9.0":
        raise RuntimeError("RF-DETR training requires rfdetr==1.9.0")

    settings = dict(recipe["sam_copy_paste"])
    mask_annotations = data_path(data_file, data, "rfdetr", "sam_mask_annotations")
    mask_identity = verify_mask_annotations(
        mask_annotations,
        expected_source_sha256=str(settings["mask_annotations_sha256"]),
        expected_source_bytes=int(settings["mask_annotations_bytes"]),
        expected_masks_sha256=data["rfdetr"].get("sam_mask_sha256"),
        manifest=data_path(data_file, data, "rfdetr", "sam_mask_manifest") if data["rfdetr"].get("sam_mask_manifest") else None,
        annotations=data_path(data_file, data, "coco", "annotations_1based") if data["rfdetr"].get("sam_mask_sha256") else None,
        image_root=data_path(data_file, data, "rfdetr", "sam_image_root"),
    )
    settings["mask_annotations"] = str(mask_annotations)
    settings["image_root"] = str(
        data_path(data_file, data, "rfdetr", "sam_image_root")
    )
    copy_paste = ClassAwareCopyPaste(CopyPasteSettings.from_mapping(settings))
    if mask_identity.get("origin") != "user_prepared_masks" and copy_paste.summary["donor_count"] != int(settings["donor_count"]):
        raise ValueError("RF-DETR Copy-Paste donor count does not match the recipe")

    from pytorch_lightning import Callback

    class FinalDataModule(RFDETRDataModule):
        def setup(self, stage: str) -> None:
            super().setup(stage)
            if stage != "fit" or getattr(self, "_final_copy_paste", False):
                return
            dataset = getattr(self, "_dataset_train", None)
            transforms = getattr(dataset, "_transforms", None)
            if not isinstance(transforms, Compose):
                raise RuntimeError("RF-DETR final recipe requires torchvision transforms")
            flips = [
                index
                for index, transform in enumerate(transforms.transforms)
                if isinstance(transform, RandomHorizontalFlip)
            ]
            if len(flips) != 1 or flips[0] == 0:
                raise RuntimeError("RF-DETR expected one post-resize horizontal flip")
            coco = getattr(dataset, "coco", None)
            copy_paste.validate_dataset_categories(
                getattr(coco, "cats", {}), getattr(dataset, "cat2label", None)
            )
            index = flips[0]
            dataset._transforms = Compose(
                [
                    *transforms.transforms[:index],
                    copy_paste,
                    *transforms.transforms[index:],
                ]
            )
            self._final_copy_paste = True

    class FinalCopyPasteEpochCallback(Callback):
        def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
            del pl_module
            copy_paste.set_epoch(int(trainer.current_epoch))

        def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
            del pl_module
            copy_paste.set_epoch(int(trainer.current_epoch))

    from pytorch_lightning import seed_everything
    seed_everything(int(recipe["seed"]), workers=True)
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("RF-DETR requires a BF16-capable CUDA device")
    model_config, train_config = rfdetr_configs(recipe, data, data_file, initial, output, device, workers)
    module = RFDETRModelModule(model_config, train_config)
    datamodule = FinalDataModule(model_config, train_config)
    trainer = build_trainer(train_config, model_config, accelerator="gpu", devices=[target.index or 0], limit_val_batches=0, num_sanity_val_steps=0, logger=False, max_steps=max_steps if max_steps is not None else -1)
    trainer.callbacks.append(FinalCopyPasteEpochCallback())
    trainer.fit(module, datamodule)
    if max_steps is not None and int(trainer.global_step) != max_steps:
        raise RuntimeError("RF-DETR did not complete the requested optimizer updates")
    (output / "training_runtime.json").write_text(json.dumps({
        "schema_version": 1, "seed": int(train_config.seed), "precision": str(trainer.precision),
        "physical_batch": int(train_config.batch_size), "gradient_accumulation": int(train_config.grad_accum_steps),
        "effective_batch": int(train_config.batch_size) * int(train_config.grad_accum_steps),
        "gradient_clip": float(train_config.clip_max_norm), "loss": "native_rfdetr_detection_criterion",
        "num_classes": int(model_config.num_classes), "resolution": int(model_config.resolution),
        "completed_epochs": int(trainer.current_epoch), "optimizer_updates": int(trainer.global_step),
        "sam_mask_identity": mask_identity,
        "requested_max_steps": max_steps, "full_recipe_completed": max_steps is None,
    }, indent=2) + "\n", encoding="utf-8")
    source = output / "last_ema.pth"
    if not source.is_file():
        raise FileNotFoundError(f"RF-DETR did not write terminal EMA checkpoint: {source}")
    artifact = output / "checkpoint_terminal_ema.pth"
    shutil.copy2(source, artifact)
    return artifact


def yolo_stage_options(recipe, data_yaml, output, device, workers):
    return {
        "data": str(data_yaml), "project": str(output.parent), "name": output.name,
        "exist_ok": True, "device": device, "workers": workers,
        "epochs": int(recipe["epochs"]), "batch": int(recipe["batch"]),
        "nbs": int(recipe["nominal_batch"]), "imgsz": int(recipe["resolution"]),
        "optimizer": str(recipe["optimizer"]), "lr0": float(recipe["learning_rate"]),
        "lrf": float(recipe["final_lr_ratio"]), "momentum": float(recipe["momentum"]),
        "weight_decay": float(recipe["weight_decay"]), "warmup_epochs": float(recipe["warmup_epochs"]),
        "freeze": int(recipe["freeze"]), "box": float(recipe["box"]),
        "cls": float(recipe["cls"]), "cls_pw": float(recipe["cls_pw"]), "dfl": float(recipe["dfl"]),
        "close_mosaic": int(recipe["epochs"]) - int(recipe["augmentation"]["final_stage_start_epoch"]) + 1,
        "amp": True, "seed": int(recipe["seed"]), "deterministic": False, "val": False,
        "plots": False, "save": True, "save_period": -1, "patience": 0,
        "cls_remap": True, "mixup": 0.0, "copy_paste": 0.0, "cutmix": 0.0,
        "multi_scale": 0.0, "cos_lr": False, "cache": False,
        **{key: float(value) for key, value in recipe["augmentation"]["initial"].items()},
    }


def run_yolo_stage(YOLO, recipe, dataset_yaml, initial, output, device, workers, max_steps=None):
    from training.yolo_data import FinalYOLOTrainer, IMAGE_POLICY

    fixed_epoch = int(recipe["fixed_export_epoch"])
    if not 1 <= fixed_epoch <= int(recipe["epochs"]):
        raise ValueError("YOLO26 export epoch must be inside the fixed training stage")
    artifact = output / (f"checkpoint_after_steps_{max_steps}_ema.pt" if max_steps is not None else f"checkpoint_fixed_epoch_{fixed_epoch}.pt")
    batch_steps = 0
    optimizer_updates = 0
    step_hook = None

    def start_training(trainer):
        nonlocal step_hook
        # Upstream also evaluates at the final epoch when args.val is False.
        # This recipe trains on every labeled image and exports a fixed EMA.
        trainer.validate = lambda: ({}, 0.0)
        trainer.final_eval = lambda: None

        def count_update(optimizer, arguments, keywords):
            nonlocal optimizer_updates
            optimizer_updates += 1

        step_hook = trainer.optimizer.register_step_post_hook(count_update)

    def final_augmentation(trainer):
        start = int(trainer.epochs) - int(trainer.args.close_mosaic)
        if int(trainer.epoch) != start:
            return
        for name in recipe["augmentation"]["final_disabled"]:
            if name == "albumentations":
                trainer.args.augmentations = []
            elif hasattr(trainer.args, name):
                setattr(trainer.args, name, 0.0)
        for name, value in recipe["augmentation"]["final"].items():
            setattr(trainer.args, name, float(value))

    def count_batch(trainer):
        nonlocal batch_steps
        batch_steps += 1
        if max_steps is not None and optimizer_updates >= max_steps:
            trainer.stop = True

    def fixed_snapshot(trainer):
        if max_steps is None and int(trainer.epoch) + 1 != fixed_epoch:
            return
        if max_steps is not None and optimizer_updates < max_steps:
            return
        if trainer.ema is None:
            raise RuntimeError("YOLO26 fixed export requires EMA")
        trainer.save_model()
        shutil.copy2(Path(trainer.last), artifact)
        (output / "stage_manifest.json").write_text(json.dumps({
            "schema_version": 1, "initial_sha256": sha256(initial),
            "artifact": artifact.name, "artifact_sha256": sha256(artifact),
            "export_epoch": int(trainer.epoch) + 1, "batch_steps": batch_steps,
            "optimizer_updates": optimizer_updates if step_hook is not None else int(trainer.ema.updates), "variant": "ema",
            "ema_updates": int(trainer.ema.updates), "requested_max_steps": max_steps,
            "full_recipe_completed": max_steps is None,
            "seed": int(recipe["seed"]), "amp": bool(trainer.amp),
            "precision": "fp16-autocast" if bool(trainer.amp) else "float32",
            "physical_batch": int(trainer.args.batch), "nominal_batch": int(recipe["nominal_batch"]),
            "resolution": int(recipe["resolution"]),
            "training_image_policy": IMAGE_POLICY,
            "label_cache_reuse": False, "adjacent_npy_reuse": False,
        }, indent=2) + "\n", encoding="utf-8")

    model = YOLO(str(initial))
    architecture = getattr(model.model, "yaml", {})
    if "yolo26x" not in str(architecture.get("yaml_file", "")).lower() and not (
        "yolo26" in str(architecture.get("yaml_file", "")).lower() and architecture.get("scale") == "x"
    ):
        raise ValueError("YOLO26 fixed stages require the YOLO26x architecture")
    model.add_callback("on_train_epoch_start", final_augmentation)
    model.add_callback("on_train_start", start_training)
    model.add_callback("on_train_batch_end", count_batch)
    model.add_callback("on_fit_epoch_end", fixed_snapshot)
    model.train(trainer=FinalYOLOTrainer, **yolo_stage_options(recipe, dataset_yaml, output, device, workers))
    if step_hook is not None:
        step_hook.remove()
    if max_steps is not None and optimizer_updates != max_steps:
        raise RuntimeError("YOLO26 did not complete the requested optimizer updates")
    if not artifact.is_file():
        raise FileNotFoundError(f"YOLO26 fixed epoch checkpoint was not written: {artifact}")
    return artifact


def run_yolo26(recipe, data, data_file, initial, output, device, workers, max_steps=None):
    settings_dir = output / "ultralytics_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    vendor = ROOT / "third_party/ultralytics"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    weights_dir = output / "pretrained_cache"
    weights_dir.mkdir(exist_ok=True)
    native_yaml = data_path(data_file, data, "yolo26", "dataset_yaml")
    native_data = load_yaml(native_yaml)
    native_names = native_data.get("names")
    if isinstance(native_names, dict):
        native_names = [native_names[index] for index in range(CLASS_COUNT)]
    if native_names != list(data["class_names"]):
        raise ValueError("YOLO26 dataset class order differs from the fixed categories")
    dataset_root = resolve(native_yaml.parent, str(native_data.get("path", ".")))
    native_data["path"] = str(dataset_root)
    dataset_yaml = output / "dataset.yolo.yaml"
    dataset_yaml.write_text(yaml.safe_dump(native_data, sort_keys=False), encoding="utf-8")
    artifact = initial
    stages = []
    with local_ultralytics() as ultralytics:
        if not Path(ultralytics.__file__).resolve().is_relative_to(vendor.resolve()):
            raise ImportError("YOLO26 training imported an unpinned ultralytics package")
        if ultralytics.__version__ != "8.4.138":
            raise RuntimeError(f"unexpected ultralytics version {ultralytics.__version__}")
        ultralytics.settings.update({"weights_dir": str(weights_dir)})
        from ultralytics import YOLO
        import ultralytics.utils
        ultralytics.utils.WEIGHTS_DIR = weights_dir
        for name in ("stage1", "stage2"):
            stage_recipe = {**recipe, **recipe["stages"][name]}
            stage_output = prepare_output(output / name)
            artifact = run_yolo_stage(YOLO, stage_recipe, dataset_yaml, artifact, stage_output, device, workers, max_steps=max_steps)
            stages.append({"stage": name, "artifact": str(artifact.relative_to(output)), "sha256": sha256(artifact)})
    (output / "stages.json").write_text(json.dumps({"schema_version": 1, "stages": stages}, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("tinyformer", "rfdetr", "yolo26"),
        required=True,
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--initial-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--recipes", type=Path, default=Path("training/final_recipes.yaml")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-steps", type=int, help="Stop after this many real optimizer updates per stage; writes a partial recipe artifact")
    args = parser.parse_args()
    configure_caches()

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("fixed detector training requires CUDA")
    if args.workers is not None and not 0 <= args.workers <= 28:
        raise ValueError("--workers must be in 0..28")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    recipes_path = args.recipes.expanduser().resolve()
    data_file = args.data.expanduser().resolve()
    recipe = recipes(recipes_path, args.model)
    data = load_data(data_file)
    initial = args.initial.expanduser().resolve()
    initial_sha = verify_initial(initial, recipe, args.initial_sha256)
    output = prepare_output(args.output_dir)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    runners = {
        "tinyformer": run_tinyformer,
        "rfdetr": run_rfdetr,
        "yolo26": run_yolo26,
    }
    artifact = runners[args.model](
        recipe,
        data,
        data_file,
        initial,
        output,
        args.device,
        args.workers if args.workers is not None else int(recipe["workers"]),
        max_steps=args.max_steps,
    )
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "recipe_sha256": sha256(recipes_path),
        "data_contract_sha256": sha256(data_file),
        "initial_sha256": initial_sha,
        "artifact": str(artifact.relative_to(output)),
        "artifact_sha256": sha256(artifact),
        "requested_max_steps": args.max_steps,
        "full_recipe_completed": args.max_steps is None,
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(artifact, flush=True)


if __name__ == "__main__":
    main()
