"""DCR crop geometry and classifier architectures used by training."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from training.hplus_lora import LoRALinear, inject_lora, merge_lora_, set_adapter_training

ROOT = Path(__file__).resolve().parents[1]
CLASS_COUNT = 34


def crop_rectangle(
    box_xywh: Sequence[float], context_scale: float
) -> tuple[int, int, int, int]:
    x, y, width, height = map(float, box_xywh)
    right, bottom = x + width, y + height
    center_x, center_y = (x + right) / 2.0, (y + bottom) / 2.0
    half_width = (right - x) * context_scale / 2.0
    half_height = (bottom - y) * context_scale / 2.0
    left = math.floor(center_x - half_width)
    top = math.floor(center_y - half_height)
    right = math.ceil(center_x + half_width)
    bottom = math.ceil(center_y + half_height)
    return left, top, max(right, left + 1), max(bottom, top + 1)


class MaskedAttentionPool(nn.Module):
    def __init__(self, dimensions: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dimensions))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(dimensions)
        self.attention = nn.MultiheadAttention(dimensions, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dimensions)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid.clone()
        valid[~valid.any(dim=1), 0] = True
        query = self.query.expand(len(tokens), -1, -1)
        pooled, _ = self.attention(
            query,
            self.norm(tokens),
            self.norm(tokens),
            key_padding_mask=~valid,
            need_weights=False,
        )
        return self.output_norm(query + pooled).squeeze(1)


class DINOAttentionHead(nn.Module):
    def __init__(self, dimensions: int, model: Mapping[str, Any]) -> None:
        super().__init__()
        self.roi_pool = MaskedAttentionPool(dimensions, int(model["attention_heads"]))
        self.cross = nn.ModuleList()
        projected = int(model["projection_dimensions"])
        self.projection = nn.Sequential(
            nn.LayerNorm(2 * dimensions),
            nn.Linear(2 * dimensions, projected),
            nn.GELU(),
            nn.LayerNorm(projected),
        )
        self.classifier = nn.Linear(projected, CLASS_COUNT)
        self.objectness = nn.Linear(projected, 1)

    def forward(
        self, cls: torch.Tensor, patches: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.roi_pool(patches, valid)
        feature = self.projection(torch.cat((cls, pooled), dim=1))
        return self.classifier(feature), self.objectness(feature).squeeze(1)


def load_dino_backbone(model: Mapping[str, Any], pretrained_weights: Path | None):
    vendor = ROOT / "third_party/dinov3"
    if not (vendor / "dinov3/hub/backbones.py").is_file():
        raise FileNotFoundError("DINOv3 submodule is missing; run git submodule update --init --recursive")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from dinov3.hub import backbones

    factory = getattr(backbones, str(model["architecture"]), None)
    if factory is None:
        raise ValueError(f"unsupported DINOv3 architecture: {model['architecture']}")
    return factory(
        pretrained=pretrained_weights is not None,
        **({"weights": str(pretrained_weights), "check_hash": True} if pretrained_weights is not None else {}),
    )


class DINOAttentionDCR(nn.Module):
    def __init__(self, model: Mapping[str, Any], pretrained_weights: Path | None = None) -> None:
        super().__init__()
        self.backbone = load_dino_backbone(model, pretrained_weights)
        self.head = DINOAttentionHead(int(self.backbone.embed_dim), model)
        self.model_id = "R0"

    def forward(
        self, images: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.backbone.forward_features(images)
        cls = output["x_norm_clstoken"].float()
        patches = output["x_norm_patchtokens"].float()
        return self.head(cls, patches, valid)


class HPlusDCR(DINOAttentionDCR):
    def __init__(self, model: Mapping[str, Any], pretrained_weights: Path | None = None) -> None:
        nn.Module.__init__(self)
        self.backbone = load_dino_backbone(model, pretrained_weights)
        self.lora_contract = inject_lora(self.backbone, **model["adapter"])
        self.head = DINOAttentionHead(int(self.backbone.embed_dim), model)
        self.model_id = "R0"

    def set_head_only(self, enabled: bool) -> None:
        set_adapter_training(self.backbone, not enabled)
        self.head.requires_grad_(True)
        self.backbone.eval() if enabled else self.backbone.train()


class ImageTowerDCR(nn.Module):
    def __init__(
        self, model: Mapping[str, Any], pretrained_model: str | None = None
    ) -> None:
        super().__init__()
        import timm

        if str(model["name"]) == "siglip":
            timm.layers.set_fused_attn(True)
        self.encoder = timm.create_model(
            pretrained_model or str(model["architecture"]),
            pretrained=pretrained_model is not None,
            num_classes=0,
        )
        dimensions = int(self.encoder.num_features)
        self.foreground = nn.Linear(dimensions, CLASS_COUNT)
        self.objectness = nn.Linear(dimensions, 1)
        nn.init.trunc_normal_(self.foreground.weight, std=0.02)
        nn.init.zeros_(self.foreground.bias)
        nn.init.trunc_normal_(self.objectness.weight, std=0.02)
        nn.init.zeros_(self.objectness.bias)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        if features.ndim != 2:
            raise RuntimeError("image tower did not return pooled features")
        return self.foreground(features.float()), self.objectness(features.float()).squeeze(1)
