"""Load complete trained DCR classifiers without upstream initialization downloads."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml

from training.checkpoints import DEFAULT_MANIFEST, load_manifest, verify_file
from training.hplus_lora import merge_lora_
from training.models import CLASS_COUNT, DINOAttentionDCR, HPlusDCR, ImageTowerDCR

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/postprocess.yaml"
MODEL_ARCHITECTURES = {
    "hplus": ("dinov3_hplus_lora", "dinov3_vith16plus"),
    "v7": ("dinov3_attention", "dinov3_vitl16"),
    "siglip": ("image_tower", "vit_large_patch16_siglip_384.v2_webli"),
}


def load_model_config(model_name: str, config: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    if model_name not in MODEL_ARCHITECTURES:
        raise ValueError(f"Unknown final classifier: {model_name}")
    document = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or type(document.get("classes")) is not int or document["classes"] != CLASS_COUNT:
        raise ValueError(f"Final classifiers require exactly {CLASS_COUNT} classes")
    rows = document.get("dcr", {}).get("models", [])
    matches = [row for row in rows if isinstance(row, dict) and row.get("name") == model_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one configuration for {model_name}")
    model = matches[0]
    if (model.get("kind"), model.get("architecture")) != MODEL_ARCHITECTURES[model_name]:
        raise ValueError(f"Unexpected local architecture for {model_name}")
    return model


def load_classifier(
    model_name: str,
    checkpoint: Path,
    *,
    manifest: Path = DEFAULT_MANIFEST,
    config: Path = DEFAULT_CONFIG,
    device: str | torch.device = "cpu",
    deployment: bool = True,
) -> nn.Module:
    """Strictly load trained state; deployment merges H+ LoRA in memory only."""
    if model_name not in MODEL_ARCHITECTURES:
        raise ValueError(f"Unknown final classifier: {model_name}")
    catalog = load_manifest(Path(manifest))
    if model_name not in catalog["artifacts"]:
        raise ValueError(f"Classifier is missing from the checkpoint catalog: {model_name}")
    path = Path(checkpoint).expanduser().resolve()
    verify_file(path, catalog["artifacts"][model_name])
    checkpoint_data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_data, dict) or type(checkpoint_data.get("schema_version")) is not int or checkpoint_data["schema_version"] != 1:
        raise ValueError("Expected a version-1 complete classifier checkpoint")
    if checkpoint_data.get("model_name") != model_name:
        raise ValueError(f"Checkpoint model_name must be {model_name}")
    state = checkpoint_data.get("model")
    if not isinstance(state, dict) or not state or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("Checkpoint model must be a complete tensor state_dict")
    model_config = load_model_config(model_name, Path(config))
    if model_name == "hplus":
        model = HPlusDCR(model_config, pretrained_weights=None)
    elif model_name == "v7":
        model = DINOAttentionDCR(model_config, pretrained_weights=None)
    else:
        model = ImageTowerDCR(model_config, pretrained_model=None)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model = model.to(device).eval()
    if model_name == "hplus" and deployment:
        settings = model_config.get("deployment", {})
        if settings.get("merge_lora") is not True or settings.get("expected_lora_modules") != 160:
            raise ValueError("H+ deployment requires the frozen 160-module LoRA merge contract")
        if merge_lora_(model.backbone) != 160:
            raise RuntimeError("H+ deployment must merge exactly 160 LoRA modules")
    return model
