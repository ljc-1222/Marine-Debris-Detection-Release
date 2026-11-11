"""Export locally retrained checkpoints and a separate inference catalog."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch
import yaml

from training import checkpoints, final_detectors, final_models
from training.export_yolo26 import _pin_ultralytics, safe_config
from training.prepare_data import sha256
from scripts.environment import configure_caches

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("codetr", "tinyformer", "rfdetr", "yolo26", "hplus", "v7", "siglip")
# Keep torch.save's internal ZIP basenames stable across local exports.
_ARCHIVE_STEMS = {
    "codetr": "codetr_vitl16_5scale_retrained",
    "tinyformer": "tinyformer_xl_pbm_retrained",
    "rfdetr": "rfdetr_large_legacy_retrained",
    "yolo26": "yolo26x_retrained",
    "hplus": "dinov3_vith16plus_dcr_retrained_full_unmerged",
    "v7": "dinov3_vitl16_dcr_retrained_full",
    "siglip": "siglip2_vitl16_384_dcr_retrained_full",
}


def identity(path: Path) -> dict[str, Any]:
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def tensor_state(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value or any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor) for name, tensor in value.items()
    ):
        raise ValueError("Expected a nonempty tensor-only model state_dict")
    return {name: tensor.detach().cpu() for name, tensor in value.items()}


def read_sources(path: Path, models: Sequence[str] | None = None) -> dict[str, tuple[Path, dict[str, Any]]]:
    selected = tuple(dict.fromkeys(MODELS if models is None else models))
    if not selected or set(selected) - set(MODELS):
        raise ValueError("Select at least one canonical model ID")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) - set(MODELS) - {"hplus_base"} or set(selected) - set(document):
        raise ValueError("Sources must map each selected canonical model ID to a local checkpoint; hplus_base is optional")
    sources = {}
    required = [name for name in MODELS if name in selected]
    if "hplus" in selected and "hplus_base" in document:
        required.append("hplus_base")
    for name in required:
        value = document[name]
        row = {"path": value} if isinstance(value, str) else value
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or set(row) - {"path", "sha256", "bytes"}:
            raise ValueError("Each source must be a local path or a path/SHA256/bytes mapping")
        source = Path(row["path"]).expanduser()
        source = (source if source.is_absolute() else path.parent / source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        binding = identity(source)
        if any(field in row and row[field] != binding[field] for field in ("sha256", "bytes")):
            raise ValueError(f"Source identity mismatch for {name}")
        sources[name] = source, binding
    return sources


def load_native(path: Path, model_name: str, *, trust_native: bool) -> tuple[Mapping[str, Any], bool]:
    try:
        document = torch.load(path, map_location="cpu", weights_only=True)
        unsafe = False
    except pickle.UnpicklingError as error:
        if not trust_native:
            raise ValueError("Restricted checkpoint loading failed; use --trust-native only for native files you trust") from error
        with _pin_ultralytics() if model_name == "yolo26" else nullcontext():
            document = torch.load(path, map_location="cpu", weights_only=False)
        unsafe = True
    if not isinstance(document, Mapping):
        raise ValueError(f"Expected a checkpoint mapping for {model_name}")
    return document, unsafe


def check_names(names: Any) -> None:
    if isinstance(names, Mapping):
        if set(map(int, names)) != set(range(34)):
            raise ValueError("Checkpoint class IDs must be 0..33")
        names = [names[index] if index in names else names[str(index)] for index in range(34)]
    if not isinstance(names, (list, tuple)) or list(names) != final_detectors.class_names():
        raise ValueError("Checkpoint class names/order do not match the fixed 34 classes")


def tinyformer_config(native: Mapping[str, Any]) -> dict[str, Any]:
    config = native.get("tinyformer_config")
    if (native.get("backend") != "tinyformer" or native.get("num_classes") != 34
            or not isinstance(config, dict) or config.get("num_classes") != 34
            or config.get("remap_mscoco_category") is not False):
        raise ValueError("Expected the 34-class TinyFormer native export")
    if Path(native.get("source_config", "")) != final_detectors.TINYFORMER_CONFIG:
        raise ValueError("Unexpected TinyFormer architecture source")
    if config.get("model") != "DEIM" or config.get("DEIM") != {
        "backbone": "DINOv3SSAs_4Scale", "encoder": "HybridEncoder_4Scale", "decoder": "DEIMTransformer",
    }:
        raise ValueError("Expected the fixed TinyFormer-XL-PBM architecture")
    if config.get("DINOv3SSAs_4Scale", {}).get("name") != "dinov3_vitb16":
        raise ValueError("Expected the TinyFormer DINOv3 ViT-B/16 backbone")
    keep = ("model", "postprocessor", "num_classes", "remap_mscoco_category", "eval_spatial_size",
            "use_focal_loss", "num_top_queries", "DEIM", "DINOv3SSAs_4Scale", "HybridEncoder_4Scale", "DEIMTransformer")
    clean = safe_config({name: config[name] for name in keep if name in config})
    operations = config["val_dataloader"]["dataset"]["transforms"]["ops"]
    if [row["type"] for row in operations] != ["Resize", "ConvertPILImage", "Normalize"]:
        raise ValueError("Unexpected TinyFormer inference transform")
    fields = (("type", "size"), ("type", "dtype", "scale"), ("type", "mean", "std"))
    clean["val_dataloader"] = {"dataset": {"transforms": {"ops": [
        {name: row[name] for name in names if name in row} for row, names in zip(operations, fields, strict=True)
    ]}}}
    clean["DINOv3SSAs_4Scale"]["weights_path"] = None
    clean["device"] = "cpu"
    return clean


def yolo_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("nc") != 34:
        raise ValueError("YOLO26 requires its native 34-class architecture mapping")
    path = ROOT / "third_party/ultralytics/ultralytics/cfg/models/26/yolo26.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config.update(nc=34, scale="x", yaml_file="yolo26x.yaml", channels=3)
    for name in ("activation", "depth_multiple", "width_multiple", "kpt_shape"):
        if name in raw and name not in config:
            raise ValueError("YOLO26 native configuration changes the fixed architecture")
    if any(safe_config(raw.get(name)) != value for name, value in config.items() if name != "yaml_file"):
        raise ValueError("Expected the pinned YOLO26x detector architecture")
    return config


def classifier_prototype(name: str) -> torch.nn.Module:
    config = final_models.load_model_config(name)
    if name == "hplus":
        return final_models.HPlusDCR(config, pretrained_weights=None)
    if name == "v7":
        return final_models.DINOAttentionDCR(config, pretrained_weights=None)
    return final_models.ImageTowerDCR(config, pretrained_model=None)


def hplus_state(native: Mapping[str, Any], sources) -> tuple[dict[str, torch.Tensor], torch.nn.Module]:
    if native.get("model_name", "hplus") != "hplus":
        raise ValueError("Classifier model_name does not match hplus")
    model = classifier_prototype("hplus")
    expected = {name for name in model.backbone.state_dict() if name.endswith(("lora_A", "lora_B"))}
    if len(expected) != 320:
        raise ValueError("H+ requires exactly 320 unmerged LoRA tensors")
    if isinstance(native.get("model"), Mapping):
        full = tensor_state(native["model"])
        target_full = model.state_dict()
        if any(name in target_full and value.dtype != target_full[name].dtype for name, value in full.items()):
            raise ValueError("H+ source dtype differs from the fixed training factory")
        model.load_state_dict(full, strict=True)
    else:
        if "hplus_base" not in sources:
            raise ValueError("An H+ adapter checkpoint requires hplus_base in the source mapping")
        base, binding = sources["hplus_base"]
        expected_base = json.loads((ROOT / "configs/upstream_initialization.json").read_text())["artifacts"]["hplus_base"]
        if any(binding[field] != expected_base[field] for field in ("sha256", "bytes")):
            raise ValueError("H+ base does not match the fixed official initializer")
        checkpoints.verify_file(base, {**expected_base, "path": base.name})
        base_state = tensor_state(torch.load(base, map_location="cpu", weights_only=True))
        target = model.backbone.state_dict()
        converted = {name: name.replace(".base.", ".") for name in target if name not in expected}
        if set(converted.values()) != set(base_state):
            raise ValueError("Official H+ base keys do not match the native backbone")
        incompatible = model.backbone.load_state_dict({name: base_state[source] for name, source in converted.items()}, strict=False)
        if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
            raise ValueError("H+ base restore changed non-adapter state")
        trained = native.get("trainable_state")
        if not isinstance(trained, Mapping) or set(trained) != {"adapter", "head"}:
            raise ValueError("H+ checkpoint requires adapter and head state")
        adapters = tensor_state(trained["adapter"])
        if set(adapters) != expected:
            raise ValueError("H+ adapter keys do not match its 160 LoRA modules")
        if any(adapters[name].dtype != target[name].dtype for name in adapters):
            raise ValueError("H+ adapter dtype differs from the fixed training factory")
        incompatible = model.backbone.load_state_dict(adapters, strict=False)
        if incompatible.unexpected_keys or set(incompatible.missing_keys) != set(target) - expected:
            raise ValueError("H+ adapter restore changed native backbone state")
        head = tensor_state(trained["head"])
        target_head = model.head.state_dict()
        if any(name in target_head and value.dtype != target_head[name].dtype for name, value in head.items()):
            raise ValueError("H+ head dtype differs from the fixed training factory")
        model.head.load_state_dict(head, strict=True)
    state = tensor_state(model.state_dict())
    if sum(name.endswith(".base.bias_mask") for name in state) != 32:
        raise ValueError("H+ full state must preserve all 32 masked-bias buffers")
    return state, model


def pack_model(name: str, native: Mapping[str, Any], sources):
    document = {"schema_version": 1, "model_name": name}
    prototype = None
    if name == "codetr":
        check_names(native.get("meta", {}).get("CLASSES"))
        state = {key: value for key, value in tensor_state(native.get("state_dict")).items() if not key.startswith("ema_")}
        document.update(state_dict=state, meta={"CLASSES": final_detectors.class_names()})
    elif name == "tinyformer":
        check_names(native.get("class_names"))
        state = native.get("ema", {}).get("module") if isinstance(native.get("ema"), Mapping) else None
        document.update(backend="tinyformer", num_classes=34, class_names=final_detectors.class_names(),
                        source_config=str(final_detectors.TINYFORMER_CONFIG), tinyformer_config=tinyformer_config(native),
                        model=tensor_state(state if state is not None else native.get("model")))
    elif name == "rfdetr":
        args = native.get("args")
        args = args if isinstance(args, Mapping) else vars(args) if hasattr(args, "__dict__") else {}
        config = native.get("model_config")
        if (not isinstance(config, dict) or config.get("encoder") != "dinov2_windowed_base"
                or config.get("num_classes") != 34 or args.get("num_classes", 34) != 34
                or native.get("model_name", "RFDETRLargeDeprecated") != "RFDETRLargeDeprecated"
                or config.get("model_name") not in (None, "RFDETRLargeDeprecated")
                or native.get("rfdetr_version") != "1.9.0"):
            raise ValueError("Expected an RFDETRLargeDeprecated 1.9.0 native checkpoint")
        check_names(args.get("class_names"))
        state = tensor_state(native.get("model"))
        if state.get("class_embed.weight") is None or state["class_embed.weight"].ndim != 2 or state["class_embed.weight"].shape[0] != 35:
            raise ValueError("RF-DETR requires 34 foreground classes plus background")
        document.update(model_name="RFDETRLargeDeprecated", model=state,
                        model_config=safe_config(config), args={"num_classes": 34, "class_names": final_detectors.class_names()},
                        rfdetr_version="1.9.0")
    elif name == "yolo26":
        selected = native.get("ema")
        if not isinstance(selected, (torch.nn.Module, Mapping)):
            selected = native.get("model")
        if isinstance(selected, torch.nn.Module):
            check_names(selected.names)
            state, config = selected.state_dict(), selected.yaml
        else:
            check_names(native.get("class_names"))
            state, config = selected, native.get("model_config")
        document.update(model=tensor_state(state), model_config=yolo_config(config),
                        class_names=final_detectors.class_names(), inference={"imgsz": 960, "task": "detect"})
    elif name == "hplus":
        state, prototype = hplus_state(native, sources)
        document["model"] = state
    else:
        if native.get("model_name", name) != name:
            raise ValueError("Classifier model_name does not match its source slot")
        document["model"] = tensor_state(native.get("model"))
    return document, prototype


def validate_model(name: str, document: dict, checkpoint: Path, prototype=None) -> torch.nn.Module:
    if prototype is None:
        if name in final_detectors.DETECTORS:
            loaded = getattr(final_detectors, "_build_" + name)(document, checkpoint)
            prototype = loaded.model
            if name == "rfdetr":
                document["model_config"] = safe_config(loaded.config)
        else:
            prototype = classifier_prototype(name)
    prototype.load_state_dict(document["state_dict" if name == "codetr" else "model"], strict=True)
    return prototype


def training_metadata(native: Mapping[str, Any], path: Path) -> dict[str, Any]:
    keep = ("requested_max_steps", "full_recipe_completed", "completed_epoch", "completed_epochs", "optimizer_updates", "global_step")
    result = {}
    values = [native, native.get("training", {}), native.get("meta", {})]
    for filename in ("training_runtime.json", "stage_manifest.json"):
        sidecar = path.parent / filename
        if sidecar.is_file():
            values.append(json.loads(sidecar.read_text(encoding="utf-8")))
    for value in values:
        if isinstance(value, Mapping):
            result.update({key: value[key] for key in keep if key in value and (value[key] is None or type(value[key]) in (int, bool))})
    return result


def check_codetr(checkpoint: Path, command: str | None) -> None:
    prefix = shlex.split(command) if command else [sys.executable, str(ROOT / "scripts/codetr_python.py")]
    subprocess.run([*prefix, "-m", "training.export_models", "--check-codetr", str(checkpoint)], cwd=ROOT, check=True)


def export_models(sources_file: Path, directory: Path, *, trust_native: bool = False, allow_partial: bool = False,
                  codetr_command: str | None = None, models: Sequence[str] | None = None) -> dict:
    sources_file = sources_file.expanduser().resolve()
    directory = directory.expanduser().resolve()
    if directory.exists():
        raise FileExistsError("Use a new output directory for a local model export")
    sources = read_sources(sources_file, models)
    selected = [name for name in MODELS if name in sources]
    release_artifacts = checkpoints.load_manifest(checkpoints.DEFAULT_MANIFEST)["artifacts"]
    directory.mkdir(parents=True)
    catalog = {"schema_version": 2, "default_group": "final", "model_origin": "local_retraining",
               "allow_partial": allow_partial, "training_completion_verified": False,
               "source_configuration_sha256": sha256(sources_file), "classes": final_detectors.class_names(),
               "configuration_sha256": sha256(final_models.DEFAULT_CONFIG),
               "groups": {"final": selected, "all": selected}, "artifacts": {}}
    report = {"schema_version": 1, "model_origin": "local_retraining", "allow_partial": allow_partial, "artifacts": {},
              "scope": "Strict architecture/state checks and safe standalone serialization; no full-training, reference-model or score equivalence claim"}
    for name in selected:
        release = release_artifacts[name]
        source, binding = sources[name]
        checkpoints.verify_file(source, {**binding, "path": source.name})
        native, unsafe = load_native(source, name, trust_native=trust_native)
        training = training_metadata(native, source)
        partial = training.get("requested_max_steps") is not None or training.get("full_recipe_completed") is False
        if partial and not allow_partial:
            raise ValueError("A bounded checkpoint requires --allow-partial; it is not a completed full training run")
        document, prototype = pack_model(name, native, sources)
        folder = directory / "checkpoints" / ("detectors" if name in final_detectors.DETECTORS else "classifiers")
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / (_ARCHIVE_STEMS[name] + ".preparing.pt")
        temporary.open("xb").close()
        try:
            # RF-DETR receives an existing safe path to suppress upstream initialization.
            torch.save(document, temporary)
            if name == "codetr":
                check_codetr(temporary, codetr_command)
            else:
                prototype = validate_model(name, document, temporary, prototype)
            torch.save(document, temporary)
            restored = torch.load(temporary, map_location="cpu", weights_only=True)
            state_field = "state_dict" if name == "codetr" else "model"
            state = document[state_field]
            if set(restored[state_field]) != set(state) or any(
                restored[state_field][key].shape != value.shape or restored[state_field][key].dtype != value.dtype
                or not torch.equal(restored[state_field][key], value) for key, value in state.items()
            ):
                raise ValueError("Safe checkpoint roundtrip changed trained tensors")
            exported = identity(temporary)
            release_stem = Path(release["path"]).stem.removesuffix(f"-{release['sha256'][:8]}")
            target = folder / f"{release_stem}-retrained.pt"
            os.link(temporary, target)
        finally:
            temporary.unlink()
        artifact = {"display_name": release["display_name"],
                    "path": target.relative_to(directory).as_posix(), **exported, "hf": None,
                    "model_origin": "local_retraining", "source_checkpoint": binding,
                    "architecture": release["architecture"], "class_names": final_detectors.class_names(),
                    "serialization": "version_1_tensor_and_primitive_state", "declared_partial_run": partial,
                    "training": training}
        if name == "hplus" and "hplus_base" in sources:
            artifact["included_backbone"] = sources["hplus_base"][1]
        catalog["artifacts"][name] = artifact
        report["artifacts"][name] = {**artifact, "tensor_count": len(state), "native_pickle_opt_in_used": unsafe,
                                     "strict_architecture_state_load": True, "restricted_roundtrip_equal": True}
        del native, document, restored, state, prototype
        gc.collect()
    manifest = directory / "checkpoint-manifest.json"
    with manifest.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(catalog, indent=2) + "\n")
    checkpoints.load_manifest(manifest)
    with (directory / "export-report.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    return catalog


def main() -> None:
    configure_caches()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, help="JSON mapping selected model IDs and optional hplus_base to local paths")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--model", choices=MODELS, action="append", help="Export only this model; repeat to select several (default: all seven)")
    parser.add_argument("--trust-native", action="store_true", help="Allow native pickle loading only for the explicitly selected source files you trust")
    parser.add_argument("--allow-partial", action="store_true", help="Label and permit checkpoints produced by bounded training runs")
    parser.add_argument("--codetr-command", help="Co-DETR Python command; default: the current Python with scripts/codetr_python.py")
    parser.add_argument("--check-codetr", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_codetr is not None:
        if args.sources is not None or args.directory is not None or args.model is not None:
            parser.error("the Co-DETR state check does not accept export inputs")
        document = torch.load(args.check_codetr, map_location="cpu", weights_only=True)
        if document.get("schema_version") != 1 or document.get("model_name") != "codetr":
            raise ValueError("Expected a safe Co-DETR export")
        check_names(document.get("meta", {}).get("CLASSES"))
        validate_model("codetr", document, args.check_codetr)
        return
    if args.sources is None or args.directory is None:
        parser.error("export requires --sources and --directory")
    catalog = export_models(args.sources, args.directory, trust_native=args.trust_native, allow_partial=args.allow_partial,
                            codetr_command=args.codetr_command, models=args.model)
    print(json.dumps({"model_origin": catalog["model_origin"], "models": list(catalog["artifacts"]), "allow_partial": catalog["allow_partial"]}, sort_keys=True))


if __name__ == "__main__":
    main()
