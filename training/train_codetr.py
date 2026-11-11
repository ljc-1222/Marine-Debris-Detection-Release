"""Train the fixed Co-DETR recipe through the pinned upstream runner."""

from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import yaml
from scripts.environment import configure_caches

ROOT = Path(__file__).resolve().parents[1]


def prepare_output(path):
    """Approve one fresh output directory once for the entire torchrun launch."""
    output = path.expanduser().resolve()

    def create():
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("output directory is not empty: " + str(output))
        output.mkdir(parents=True, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        create()
        return output
    launch_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if not launch_id or launch_id == "none":
        raise RuntimeError("launch Co-DETR with torchrun --standalone or a unique --rdzv-id")
    import torch.distributed as distributed

    identity = launch_id + ":" + os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    token = hashlib.sha256(identity.encode()).hexdigest()[:24]
    output.parent.mkdir(parents=True, exist_ok=True)
    store = distributed.FileStore(str(output.parent / ("." + output.name + "-start-" + token + ".store")), world_size)
    store.set_timeout(timedelta(seconds=120))
    rank = int(os.environ["RANK"])
    if rank == 0:
        status = {}
        try:
            create()
        except OSError as error:
            status = {"error": type(error).__name__, "message": str(error)}
        store.set("output-check", json.dumps(status))
    status = json.loads(store.get("output-check").decode())
    store.set("accepted-" + str(rank), "1")
    store.wait(["accepted-" + str(index) for index in range(world_size)])
    if status:
        if status["error"] == "FileExistsError":
            raise FileExistsError(status["message"])
        raise RuntimeError("output setup failed: " + status["message"])
    return output


def group_training_annotations(document):
    """Keep the fixed source groups as independently padded training datasets."""
    result = {
        group: {**document, "images": [], "annotations": []}
        for group in (0, 1)
    }
    image_lookup = {}
    for image in document["images"]:
        group = image.get("source_group")
        if type(group) is not int or group not in result:
            raise ValueError("Co-DETR training images require source_group integer 0 or 1")
        global_id = int(image["id"])
        if global_id in image_lookup:
            raise ValueError("Co-DETR union image IDs must be unique across the labeled sources")
        source_id = int(image.get("source_image_id", global_id))
        image_lookup[global_id] = group, source_id
        result[group]["images"].append({**image, "id": source_id, "sam_image_id": global_id})
    for annotation in document["annotations"]:
        group, image_id = image_lookup[int(annotation["image_id"])]
        result[group]["annotations"].append({
            **annotation, "image_id": image_id,
            "id": int(annotation.get("source_annotation_id", annotation["id"])),
        })
    for group, subset in result.items():
        if not subset["images"]:
            raise ValueError("Co-DETR training is missing source group " + str(group))
        for key in ("images", "annotations"):
            if len({int(row["id"]) for row in subset[key]}) != len(subset[key]):
                raise ValueError("Duplicate Co-DETR source IDs in group " + str(group) + " " + key)
    return result


def configure(config, recipe, data, data_file, initial, output, max_steps=None):
    names = tuple(data["class_names"])
    expected = json.loads((ROOT / "configs/categories.json").read_text())["categories"]
    if list(names) != [row["name"] for row in expected]:
        raise ValueError("training class order differs from configs/categories.json")

    def data_path(*keys):
        value = data
        for key in keys:
            value = value[key]
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else data_file.parent / path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    images = data_path("coco", "images")
    annotations = data_path("coco", "annotations_1based")
    document = json.loads(annotations.read_text())
    categories = document["categories"]
    if [(row["id"], row["name"]) for row in categories] != [(row["id"], row["name"]) for row in expected]:
        raise ValueError("Co-DETR requires ordered COCO categories 1..34")
    for head in (config["model"]["query_head"], config["model"]["roi_head"][0]["bbox_head"], config["model"]["bbox_head"][0]):
        head["num_classes"] = len(names)
    config["model"]["backbone"].update(xattn=True, use_act_checkpoint=False)
    config["data"]["samples_per_gpu"] = int(recipe["batch_per_device"])
    config["data"]["workers_per_gpu"] = 0
    config["data"]["train_dataloader"] = dict(persistent_workers=False, pin_memory=True)
    from training.portable_sam_bank import bank_settings
    sam = bank_settings(dict(recipe["sam_copy_paste"]), data_path("tinyformer", "sam_object_bank"),
        expected_sha256=data["tinyformer"].get("sam_bank_sha256"), image_root=images)
    sam["annotation_file"] = str(data_path("coco", "annotations_0based"))
    sam["type"] = "CompetitionSAMCopyPaste"
    pipeline = copy.deepcopy(config["data"]["train"]["pipeline"])
    # Training GT and the fixed SAM bank use stored-raster coordinates.
    pipeline[0]["color_type"] = "color_ignore_orientation"
    pipeline.insert(2, sam)
    subsets = group_training_annotations(document)
    output.mkdir(parents=True, exist_ok=True)
    annotation_files = []
    for group, subset in subsets.items():
        path = output / ("training-group-" + str(group) + "-rank" + os.environ.get("RANK", "0") + ".json")
        path.write_text(json.dumps(subset, ensure_ascii=False) + "\n", encoding="utf-8")
        annotation_files.append(str(path))
    config["data"]["train"].update(
        type="CompetitionTrainCocoDataset", ann_file=annotation_files,
        img_prefix=[str(images) + os.sep] * len(annotation_files), classes=names, filter_empty_gt=False,
        accumulation_alignment=int(recipe["batch_per_device"]) * int(recipe["gradient_accumulation"]) * int(recipe["devices"]),
        pipeline=pipeline,
    )
    config["runner"]["max_epochs"] = int(recipe["epochs"])
    config["optimizer"]["lr"] = float(recipe["learning_rate"])
    config["optimizer"]["paramwise_cfg"]["layer_decay_rate"] = float(recipe["layer_decay"])
    config["lr_config"]["step"] = list(recipe["lr_step_epochs"])
    accumulation = int(recipe["gradient_accumulation"])
    config["optimizer_config"] = dict(type="CompetitionGradientCumulativeBf16OptimizerHook", cumulative_iters=accumulation, grad_clip=dict(max_norm=float(recipe["gradient_clip"]), norm_type=2))
    config["fp16"] = None
    config["load_from"] = "codetr-objects365://" + str(initial)
    config["resume_from"] = None
    config["checkpoint_config"] = dict(interval=int(recipe["epochs"]), max_keep_ckpts=1, save_last=True, create_symlink=False)
    config["log_config"] = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
    hooks = []
    for hook in config.get("custom_hooks", []):
        if hook["type"] == "ExpMomentumEMAHook":
            hook = dict(hook, type="CompetitionExpMomentumEMAHook", interval=accumulation, total_iter=int(hook.get("total_iter", 2000)) * accumulation)
            hooks.append(hook)
    if len(hooks) != 1:
        raise ValueError("Co-DETR base config requires exactly one EMA hook")
    config["custom_hooks"] = hooks + [dict(type="CompetitionSAMCopyPasteEpochHook"), dict(type="CompetitionTerminalExportHook", class_names=list(names))]
    if max_steps is not None:
        config["runner"]["max_epochs"] = 1
        config["custom_hooks"].append(dict(type="CompetitionLimitStepsHook", max_steps=max_steps))
    config["custom_imports"] = dict(imports=["training.codetr_runtime"], allow_failed_imports=False)
    config["work_dir"] = str(output)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, default=Path("training/final_recipes.yaml"))
    parser.add_argument("--max-steps", type=int, help="Bound a training smoke run by successful optimizer updates")
    args = parser.parse_args()
    configure_caches()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    recipe = yaml.safe_load(args.recipes.read_text())["detectors"]["codetr"]
    if int(os.environ.get("WORLD_SIZE", "1")) != int(recipe["devices"]):
        raise RuntimeError("launch Co-DETR with torchrun --nproc-per-node=" + str(recipe["devices"]))
    args.output_dir = prepare_output(args.output_dir)
    initial = args.initial.resolve()
    digest = hashlib.sha256()
    with initial.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != recipe["initialization"]["sha256"]:
        raise ValueError("Co-DETR initialization SHA256 mismatch")
    vendor = ROOT / "third_party/Co-DETR"
    sys.path.insert(0, str(vendor))
    from mmcv import Config

    native = Config.fromfile(str(ROOT / recipe["source_config"]))
    data_file = args.data.resolve()
    data = yaml.safe_load(data_file.read_text())
    configure(native, recipe, data, data_file, initial, args.output_dir.resolve(), max_steps=args.max_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_file = args.output_dir.resolve() / ("fixed-config-rank" + os.environ.get("RANK", "0") + ".py")
    native.dump(str(config_file))
    launcher = "pytorch" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "none"
    sys.argv = [str(vendor / "tools/train.py"), str(config_file), "--no-validate", "--seed", "42", "--deterministic", "--launcher", launcher]
    runpy.run_path(str(vendor / "tools/train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
