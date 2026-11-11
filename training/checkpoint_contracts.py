"""Inspect trained checkpoint metadata and required shapes without building models."""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = ("codetr", "tinyformer", "rfdetr", "yolo26", "hplus", "v7", "siglip")
SCOPE = "Restricted CPU mmap read; schema, class order, architecture metadata and required tensor shapes only. No tensor-value scan, complete architecture/state equivalence, model construction, forward, accuracy or legal-clearance claim."


def check_document(document, model_name):
    """Check the small inference-container contract, not every model state key."""
    if model_name not in MODEL_IDS:
        raise ValueError("Unknown checkpoint contract: " + str(model_name))
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("Expected checkpoint schema_version integer 1")
    expected_name = "RFDETRLargeDeprecated" if model_name == "rfdetr" else model_name
    if document.get("model_name") != expected_name:
        raise ValueError("Checkpoint model_name does not match " + model_name)
    state_key = "state_dict" if model_name == "codetr" else "model"
    allowed = {"schema_version", "model_name", state_key}
    required = set(allowed)
    extra = {
        "codetr": {"meta"},
        "tinyformer": {"backend", "num_classes", "class_names", "source_config", "tinyformer_config"},
        "rfdetr": {"model_config", "args", "rfdetr_version"},
        "yolo26": {"model_config", "class_names", "inference"},
    }.get(model_name, set())
    allowed |= extra
    required |= extra
    if model_name == "tinyformer":
        allowed.add("source_commit")
    if not required <= set(document) or set(document) - allowed:
        raise ValueError("Unexpected or missing top-level inference-container keys")
    state = document[state_key]
    if not isinstance(state, dict) or not state or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("Expected a nonempty tensor-only state dictionary")

    def primitive(value):
        if value is None or type(value) in (bool, int, float, str):
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                primitive(item)
        elif isinstance(value, dict) and all(isinstance(key, str) for key in value):
            for item in value.values():
                primitive(item)
        else:
            raise ValueError("Expected primitive architecture metadata outside the tensor state")

    primitive({key: value for key, value in document.items() if key != state_key})
    expected_names = [row["name"] for row in json.loads((ROOT / "configs/categories.json").read_text())["categories"]]
    checked = []

    def names(value):
        if not isinstance(value, (list, tuple)) or list(value) != expected_names:
            raise ValueError("Checkpoint class names/order must match the fixed 34 categories")

    def shape(key, expected):
        value = state.get(key)
        if value is None or tuple(value.shape) != tuple(expected):
            raise ValueError("Required tensor shape mismatch: " + key)
        checked.append(key)

    def mapping(key):
        value = document.get(key)
        if not isinstance(value, dict):
            raise ValueError("Expected checkpoint mapping: " + key)
        return value

    if model_name == "codetr":
        names(mapping("meta").get("CLASSES"))
        for index in range(7):
            shape(f"query_head.cls_branches.{index}.weight", (34, 256))
            shape(f"query_head.cls_branches.{index}.bias", (34,))
        shape("roi_head.0.bbox_head.fc_cls.weight", (35, 1024))
        shape("bbox_head.0.atss_cls.weight", (34, 256, 3, 3))
        shape("query_head.label_embedding.weight", (34, 256))
        shape("backbone.patch_embed.proj.weight", (1024, 3, 16, 16))
    elif model_name == "tinyformer":
        names(document["class_names"])
        config = mapping("tinyformer_config")
        backbone = config.get("DINOv3SSAs_4Scale")
        if (document["backend"] != "tinyformer" or document["num_classes"] != 34
                or config.get("num_classes") != 34 or config.get("remap_mscoco_category") is not False
                or config.get("model") != "DEIM" or config.get("DEIM") != {
                    "backbone": "DINOv3SSAs_4Scale", "encoder": "HybridEncoder_4Scale", "decoder": "DEIMTransformer"}
                or not isinstance(backbone, dict) or backbone.get("name") != "dinov3_vitb16"
                or document["source_config"] != "configs/tinyformer/object365/tinyformer_dinov3_xl_obj2coco_pbm_24e.yml"):
            raise ValueError("Expected the fixed 34-class TinyFormer-XL-PBM architecture")
        shape("decoder.enc_score_head.weight", (34, 256))
        shape("decoder.denoising_class_embed.weight", (35, 256))
        for index in range(6):
            shape(f"decoder.dec_score_head.{index}.weight", (34, 256))
        if not any(key.startswith("backbone.") and key.endswith("patch_embed.proj.weight") and tuple(value.shape) == (768, 3, 16, 16) for key, value in state.items()):
            raise ValueError("TinyFormer is missing its DINOv3 ViT-B/16 patch embedding")
    elif model_name == "rfdetr":
        names(mapping("args").get("class_names"))
        config = mapping("model_config")
        if (document["rfdetr_version"] != "1.9.0" or document["args"].get("num_classes") != 34
                or config.get("num_classes") != 34 or config.get("encoder") != "dinov2_windowed_base"
                or config.get("hidden_dim") != 384):
            raise ValueError("Expected RFDETRLargeDeprecated 1.9.0 with 34 foreground classes")
        shape("class_embed.weight", (35, 384))
        shape("class_embed.bias", (35,))
    elif model_name == "yolo26":
        names(document["class_names"])
        config = mapping("model_config")
        if (config.get("nc") != 34 or config.get("scale") != "x" or config.get("channels") != 3
                or config.get("end2end") is not True or config.get("reg_max") != 1
                or document["inference"] != {"imgsz": 960, "task": "detect"}):
            raise ValueError("Expected the 34-class end-to-end YOLO26x architecture and 960 input")
        for branch in ("cv3", "one2one_cv3"):
            for index in range(3):
                key = f"model.23.{branch}.{index}.2.weight"
                value = state.get(key)
                if value is None or value.ndim != 4 or value.shape[0] != 34 or tuple(value.shape[2:]) != (1, 1):
                    raise ValueError("YOLO26 class-output convolution shape mismatch: " + key)
                checked.append(key)
                shape(f"model.23.{branch}.{index}.2.bias", (34,))
    else:
        if model_name in {"hplus", "v7"}:
            shape("head.classifier.weight", (34, 1024))
            shape("head.classifier.bias", (34,))
            shape("head.objectness.weight", (1, 1024))
            shape("head.objectness.bias", (1,))
            dimensions = 1280 if model_name == "hplus" else 1024
            shape("backbone.patch_embed.proj.weight", (dimensions, 3, 16, 16))
        else:
            shape("foreground.weight", (34, 1024))
            shape("foreground.bias", (34,))
            shape("objectness.weight", (1, 1024))
            shape("objectness.bias", (1,))
            shape("encoder.patch_embed.proj.weight", (1024, 3, 16, 16))
        adapters = {key for key in state if key.endswith((".lora_A", ".lora_B"))}
        if model_name == "hplus":
            expected_adapters = set()
            for index in range(32):
                for target, (inputs, outputs) in {
                    "attn.qkv": (1280, 3840), "attn.proj": (1280, 1280),
                    "mlp.w1": (1280, 5120), "mlp.w2": (1280, 5120), "mlp.w3": (5120, 1280),
                }.items():
                    prefix = f"backbone.blocks.{index}.{target}"
                    shape(prefix + ".base.weight", (outputs, inputs))
                    shape(prefix + ".lora_A", (32, inputs))
                    shape(prefix + ".lora_B", (outputs, 32))
                    expected_adapters.update({prefix + ".lora_A", prefix + ".lora_B"})
                shape(f"backbone.blocks.{index}.attn.qkv.base.bias_mask", (3840,))
            if adapters != expected_adapters:
                raise ValueError("H+ requires exactly 160 paired unmerged LoRA modules")
        elif adapters:
            raise ValueError("Unexpected LoRA tensors in this classifier architecture")
    return {"status": "contract_verified", "model_name": model_name, "schema_version": 1,
            "state_field": state_key, "tensor_keys": len(state), "required_shapes_checked": len(checked),
            "unmerged_lora_modules": 160 if model_name == "hplus" else 0,
            "scope": SCOPE}


def inspect_checkpoint(path: Path, model_name: str):
    """Call only after byte verification; never fall back to unrestricted pickle."""
    try:
        document = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except (OSError, ValueError, RuntimeError, TypeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError("Restricted checkpoint contract read failed: " + type(error).__name__) from error
    return check_document(document, model_name)
