"""Native DINOv3 LoRA helpers preserving masked bias and native forward paths."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

TARGETS = ("attn.qkv", "attn.proj", "mlp.w1", "mlp.w2", "mlp.w3")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 32,
                 alpha: float = 32.0, dropout: float = 0.05) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected Linear or its subclass, got {type(base)}")
        if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid rank/alpha/dropout")
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(
            rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(
            base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def forward(self, x: Tensor) -> Tensor:
        residual = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + residual * self.scaling

    @torch.no_grad()
    def merge_into_base(self) -> nn.Linear:
        if self.training:
            raise RuntimeError("Call eval() before inference-only merge")
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        return self.base


def inject_lora(backbone: nn.Module, rank: int = 32, alpha: float = 32.0,
                dropout: float = 0.05, verify_hplus_contract: bool = True
                ) -> dict[str, Any]:
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError("Native DINOv3 backbone.blocks is required")
    if any(isinstance(m, LoRALinear) for m in backbone.modules()):
        raise ValueError("Backbone already has LoRA: refusing duplicate injection")
    if verify_hplus_contract:
        if len(blocks) != 32 or int(getattr(backbone, "embed_dim", -1)) != 1280:
            raise ValueError("Expected DINOv3 ViT-H+/16: depth32, width1280")
    replacements = []
    for i, block in enumerate(blocks):
        for relative in TARGETS:
            parent_name, leaf = relative.rsplit(".", 1)
            parent = block.get_submodule(parent_name)
            original = getattr(parent, leaf)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"blocks.{i}.{relative}: {type(original)}")
            if verify_hplus_contract:
                expected = {
                    "attn.qkv": (1280, 3840), "attn.proj": (1280, 1280),
                    "mlp.w1": (1280, 5120), "mlp.w2": (1280, 5120),
                    "mlp.w3": (5120, 1280),
                }[relative]
                if (original.in_features, original.out_features) != expected:
                    raise ValueError(f"Unexpected shape at blocks.{i}.{relative}")
            replacements.append((i, relative, parent, leaf, original))
    backbone.requires_grad_(False)
    names = []
    for i, relative, parent, leaf, original in replacements:
        setattr(parent, leaf, LoRALinear(original, rank, alpha, dropout))
        names.append(f"blocks.{i}.{relative}")
    n = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    if verify_hplus_contract and rank == 32 and n != 27_525_120:
        raise AssertionError(f"Expected 27,525,120 adapter parameters, got {n}")
    return {"rank": rank, "alpha": alpha, "dropout": dropout,
            "target_modules": names, "adapter_parameters": n,
            "backbone_total_with_adapters": sum(p.numel() for p in backbone.parameters())}


def set_adapter_training(backbone: nn.Module, enabled: bool) -> None:
    for module in backbone.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(enabled)
            module.lora_B.requires_grad_(enabled)
            module.base.requires_grad_(False)


@torch.no_grad()
def merge_lora_(model: nn.Module) -> int:
    """Permanently merge into a deployment copy; never resume training this copy."""
    if model.training:
        raise RuntimeError("Call eval() before merge_lora_")
    wrapped = [(name, module) for name, module in model.named_modules()
               if isinstance(module, LoRALinear)]
    for name, module in wrapped:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, module.merge_into_base())
    return len(wrapped)
