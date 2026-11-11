"""TinyFormer initialization, accumulated BF16 training, and EMA updates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from pytorch_lightning import Callback, LightningModule


def classification_keys(state: Mapping[str, Any]) -> set[str]:
    return {
        name for name in state
        if name == "decoder.denoising_class_embed.weight"
        or name.startswith("decoder.enc_score_head.")
        or name.startswith("decoder.dec_score_head.")
    }


def load_pretrained(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    """Transfer non-class tensors; initialize taxonomy heads and spatial buffers."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    ema = checkpoint.get("ema")
    source = ema.get("module") if isinstance(ema, Mapping) else None
    if not isinstance(source, Mapping):
        source = checkpoint.get("model")
    if not isinstance(source, Mapping):
        raise ValueError("TinyFormer initializer has neither EMA nor model weights")
    current = model.state_dict()
    class_keys = classification_keys(current)
    spatial_keys = set(current) & {"decoder.anchors", "decoder.valid_mask"}
    reset = class_keys | spatial_keys
    incompatible = (set(current) ^ set(source)) | {
        name for name in set(current) & set(source)
        if current[name].shape != source[name].shape
    }
    if forbidden := incompatible - reset:
        raise ValueError("TinyFormer initializer mismatch outside taxonomy/spatial tensors: " + str(sorted(forbidden)))
    compatible = {name: value for name, value in source.items() if name in current and name not in reset}
    result = model.load_state_dict(compatible, strict=False)
    if set(result.missing_keys) != reset or result.unexpected_keys:
        raise ValueError("TinyFormer initializer did not load exactly the non-class tensors")
    return {
        "source_variant": "ema" if isinstance(ema, Mapping) else "model",
        "loaded_tensors": len(compatible),
        "reinitialized_classification_tensors": sorted(class_keys),
        "reinitialized_resolution_tensors": sorted(spatial_keys),
    }


class TinyFormerModule(LightningModule):
    def __init__(self, model, criterion, optimizer, train_loader, schedule, flat_cosine_schedule):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self._optimizer = optimizer
        self._train_loader = train_loader
        self._schedule = dict(schedule)
        self._flat_cosine_schedule = flat_cosine_schedule
        self._base_lrs = []

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.model(samples, targets=targets)
        with torch.autocast(device_type=self.device.type, enabled=False):
            losses = self.criterion(
                outputs, targets, epoch=int(self.current_epoch), step=int(batch_idx),
                global_step=int(self.global_step), epoch_step=len(self._train_loader),
            )
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite TinyFormer loss")
        return loss

    def configure_optimizers(self):
        for group in self._optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        return self._optimizer

    def on_fit_start(self):
        optimizer = self.optimizers(use_pl_optimizer=False)
        self._base_lrs = [float(group["initial_lr"]) for group in optimizer.param_groups]
        self._set_lr(int(self.global_step), optimizer)

    def on_train_epoch_start(self):
        self._train_loader.set_epoch(int(self.current_epoch))

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self._set_lr(int(self.trainer.global_step) + 1, optimizer)

    def _set_lr(self, step, optimizer):
        per_epoch = math.ceil(len(self._train_loader) / int(self.trainer.accumulate_grad_batches))
        for group, base_lr in zip(optimizer.param_groups, self._base_lrs, strict=True):
            group["lr"] = self._flat_cosine_schedule(
                per_epoch * int(self.trainer.max_epochs), int(self._schedule["warmup_iter"]),
                per_epoch * int(self._schedule["flat_epoch"]),
                per_epoch * int(self._schedule["no_aug_epoch"]), int(step),
                base_lr, base_lr * float(self._schedule["lr_gamma"]),
            )


class TinyFormerEMA(Callback):
    def __init__(self, ema):
        self.ema = ema
        self._step_handle = None

    def on_fit_start(self, trainer, module):
        self.ema.to(module.device)
        self._remove_step_hook()
        if len(trainer.optimizers) != 1:
            raise ValueError("The fixed TinyFormer recipe requires one optimizer")

        def after_optimizer_step(optimizer, arguments, keywords):
            self.ema.update(module.model)

        # This runs after the real step, including the final accumulation window.
        self._step_handle = trainer.optimizers[0].register_step_post_hook(after_optimizer_step)

    def on_fit_end(self, trainer, module):
        self._remove_step_hook()

    def on_exception(self, trainer, module, exception):
        self._remove_step_hook()

    def _remove_step_hook(self):
        if self._step_handle is not None:
            self._step_handle.remove()
            self._step_handle = None
