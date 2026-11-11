"""Co-DETR precision, fixed training, and augmentation registrations."""

from __future__ import annotations

import importlib
import math
import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from mmcv.runner import CheckpointLoader, HOOKS, Hook, fp16_utils, get_dist_info
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmcv.runner.hooks.optimizer import GradientCumulativeOptimizerHook
from mmdet.core.hook.ema import ExpMomentumEMAHook
from mmdet.datasets import DATASETS, PIPELINES, CocoDataset
from mmdet.models.backbones import vit as vit_backbone

from training.sam_copy_paste import SAMCopyPastePlan

CLASSIFIER_SHAPES = {
    **{"query_head.cls_branches.{}.weight".format(index): (365, 256) for index in range(7)},
    **{"query_head.cls_branches.{}.bias".format(index): (365,) for index in range(7)},
    "roi_head.0.bbox_head.fc_cls.weight": (366, 1024),
    "roi_head.0.bbox_head.fc_cls.bias": (366,),
    "bbox_head.0.atss_cls.weight": (365, 256, 3, 3),
    "bbox_head.0.atss_cls.bias": (365,),
    "query_head.label_embedding.weight": (365, 256),
}


_ORIGINAL_CAST = fp16_utils.cast_tensor_type


def _bf16_cast(inputs, source_type, destination_type):
    if source_type == torch.float and destination_type == torch.half:
        return _ORIGINAL_CAST(inputs, torch.float, torch.bfloat16)
    if source_type == torch.half and destination_type == torch.float:
        inputs = _ORIGINAL_CAST(inputs, torch.bfloat16, torch.float)
    return _ORIGINAL_CAST(inputs, source_type, destination_type)


PRECISION = os.environ.get("CODETR_PRECISION", "bf16-mixed").lower()
if PRECISION.startswith("bf16"):
    fp16_utils.cast_tensor_type = _bf16_cast
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 9:
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        torch.use_deterministic_algorithms(True)

ATTENTION_BACKEND = os.environ.get("CODETR_ATTENTION_BACKEND", "flash").lower()
if ATTENTION_BACKEND not in {"auto", "flash"}:
    raise ValueError("CODETR_ATTENTION_BACKEND must be 'auto' or 'flash'")


_BICUBIC_WEIGHTS: dict[tuple[object, ...], torch.Tensor] = {}


def _bicubic_weight(source, target, device, dtype):
    key = (source, target, str(device), dtype)
    if key not in _BICUBIC_WEIGHTS:
        basis = torch.eye(source, dtype=torch.float32).reshape(source, 1, source, 1)
        weight = torch.nn.functional.interpolate(
            basis, size=(target, 1), mode="bicubic", align_corners=False
        )[:, 0, :, 0].transpose(0, 1)
        _BICUBIC_WEIGHTS[key] = weight.to(device=device, dtype=dtype)
    return _BICUBIC_WEIGHTS[key]


def _deterministic_get_abs_pos(abs_pos, has_cls_token, hw):
    height, width = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    positions = abs_pos.shape[1]
    size = int(math.sqrt(positions))
    if size * size != positions:
        raise RuntimeError("ViT absolute position count must be square")
    if size == height and size == width:
        return abs_pos.reshape(1, height, width, -1)
    values = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
    height_weight = _bicubic_weight(size, height, values.device, values.dtype)
    width_weight = _bicubic_weight(size, width, values.device, values.dtype)
    values = torch.matmul(values.transpose(-2, -1), height_weight.transpose(0, 1))
    values = torch.matmul(values.transpose(-2, -1), width_weight.transpose(0, 1))
    return values.permute(0, 2, 3, 1)


vit_backbone.get_abs_pos = _deterministic_get_abs_pos


class _TorchSDPA:
    @staticmethod
    def memory_efficient_attention(query, key, value):
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        if (
            ATTENTION_BACKEND == "flash"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 9
            and query.dtype in (torch.float16, torch.bfloat16)
        ):
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    return torch.nn.functional.scaled_dot_product_attention(query, key, value).transpose(1, 2)
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    raise
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value
        ).transpose(1, 2)


vit_backbone.xops = _TorchSDPA


try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

    _ORIGINAL_DEFORM_FORWARD = MultiScaleDeformableAttention.forward

    def _fp32_deform_forward(self, query, *args, **kwargs):
        with torch.cuda.amp.autocast(enabled=False):
            query = query.float()
            args = [
                value.float()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
                else value
                for value in args
            ]
            kwargs = {
                name: (
                    value.float()
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    else value
                )
                for name, value in kwargs.items()
            }
            return _ORIGINAL_DEFORM_FORWARD(self, query, *args, **kwargs)

    MultiScaleDeformableAttention.forward = _fp32_deform_forward
except ImportError:
    pass


try:
    _nms_ops = importlib.import_module("mmcv.ops.nms")
    _ORIGINAL_NMS_FORWARD = _nms_ops.NMSop.forward
    _ORIGINAL_SOFT_NMS_FORWARD = _nms_ops.SoftNMSop.forward

    def _fp32_nms_forward(
        context, boxes, scores, iou_threshold, offset, score_threshold, max_num
    ):
        with torch.cuda.amp.autocast(enabled=False):
            return _ORIGINAL_NMS_FORWARD(
                context,
                boxes.float(),
                scores.float(),
                iou_threshold,
                offset,
                score_threshold,
                max_num,
            )

    def _fp32_soft_nms_forward(
        context, boxes, scores, iou_threshold, sigma, min_score, method, offset
    ):
        with torch.cuda.amp.autocast(enabled=False):
            return _ORIGINAL_SOFT_NMS_FORWARD(
                context,
                boxes.float(),
                scores.float(),
                iou_threshold,
                sigma,
                min_score,
                method,
                offset,
            )

    _nms_ops.NMSop.forward = staticmethod(_fp32_nms_forward)
    _nms_ops.SoftNMSop.forward = staticmethod(_fp32_soft_nms_forward)
except (ImportError, AttributeError):
    pass


if (
    torch.cuda.is_available()
    and torch.version.cuda == "11.3"
    and torch.cuda.get_device_capability()[0] >= 9
):
    torch.backends.cudnn.enabled = False


@CheckpointLoader.register_scheme(prefixes="codetr-objects365://", force=True)
def load_objects365_checkpoint(filename, map_location=None):
    """Load the pinned detector checkpoint while discarding only taxonomy heads."""

    path = filename[len("codetr-objects365://") :]
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or len(state) != 892:
        raise RuntimeError("Invalid Co-DETR Objects365 checkpoint structure")
    shapes = {key: tuple(state[key].shape) for key in CLASSIFIER_SHAPES if key in state}
    if shapes != CLASSIFIER_SHAPES:
        raise RuntimeError("Co-DETR Objects365 classification heads do not match")
    checkpoint["state_dict"] = type(state)(
        (key, value) for key, value in state.items() if key not in CLASSIFIER_SHAPES
    )
    checkpoint.setdefault("meta", {})["objects365_load_summary"] = {
        "loaded_tensors": len(state) - len(CLASSIFIER_SHAPES),
        "reinitialized_classification_tensors": sorted(CLASSIFIER_SHAPES),
    }
    print(
        "Objects365 initialization: loaded {} detector tensors; reinitialized "
        "{} classification tensors".format(
            len(state) - len(CLASSIFIER_SHAPES), len(CLASSIFIER_SHAPES)
        )
    )
    return checkpoint


@DATASETS.register_module()
class CompetitionTrainCocoDataset(CocoDataset):
    """Pad aspect groups so every epoch ends on an optimizer step."""

    def __init__(self, accumulation_alignment=1, **kwargs):
        self.accumulation_alignment = int(accumulation_alignment)
        if self.accumulation_alignment < 1:
            raise ValueError("accumulation_alignment must be positive")
        super().__init__(**kwargs)

    def _set_group_flag(self):
        super()._set_group_flag()
        original_flag = self.flag.copy()
        extra = []
        for group in np.unique(original_flag):
            indices = np.flatnonzero(original_flag == group)
            missing = (-len(indices)) % self.accumulation_alignment
            if missing:
                extra.extend(np.resize(indices, missing).tolist())
        if not extra:
            return
        self.data_infos.extend(self.data_infos[index] for index in extra)
        self.img_ids.extend(self.img_ids[index] for index in extra)
        if self.proposals is not None:
            self.proposals.extend(self.proposals[index] for index in extra)
        self.flag = np.concatenate((original_flag, original_flag[extra]))


def _optimizer_state_step(optimizer):
    steps = []
    for state in getattr(optimizer, "state", {}).values():
        step = state.get("step")
        if step is not None:
            steps.append(int(step.item() if torch.is_tensor(step) else step))
    return max(steps) if steps else None


@HOOKS.register_module()
class CompetitionGradientCumulativeBf16OptimizerHook(
    GradientCumulativeOptimizerHook
):
    """Accumulate BF16-autocast gradients without an FP16 GradScaler."""

    def before_run(self, runner):
        wrap_fp16_model(runner.model)
        runner._codetr_optimizer_step_succeeded = False

    def _init(self, runner):
        resumed_iter = int(runner.iter)
        super()._init(runner)
        self.divisible_iters += resumed_iter

    def after_train_iter(self, runner):
        runner._codetr_optimizer_step_succeeded = False
        if not self.initialized:
            self._init(runner)
        loss_factor = (
            self.cumulative_iters
            if runner.iter < self.divisible_iters
            else self.remainder_iters
        )
        (runner.outputs["loss"] / loss_factor).backward()
        update_due = self.every_n_iters(runner, self.cumulative_iters) or self.is_last_iter(
            runner
        )
        if not update_due:
            return
        if runner.meta is None:
            runner.meta = {}
        before = int(
            runner.meta.get(
                "codetr_optimizer_steps",
                _optimizer_state_step(runner.optimizer) or 0,
            )
        )
        grad_norm = self.clip_grads(runner.model.parameters())
        if grad_norm is None:
            raise RuntimeError("Co-DETR optimizer step has no gradients")
        runner.log_buffer.update(
            {"grad_norm": float(grad_norm)}, runner.outputs["num_samples"]
        )
        if not bool(torch.isfinite(grad_norm).item()):
            skipped = int(runner.meta.get("codetr_skipped_optimizer_steps", 0)) + 1
            runner.meta["codetr_optimizer_steps"] = before
            runner.meta["codetr_skipped_optimizer_steps"] = skipped
            self._record_nonfinite(runner, skipped)
            runner.optimizer.zero_grad()
            return
        runner.optimizer.step()
        after = _optimizer_state_step(runner.optimizer) or 0
        if after != before + 1:
            raise RuntimeError("AdamW optimizer step counter did not advance exactly once")
        runner.meta["codetr_optimizer_steps"] = after
        runner.meta.setdefault("codetr_skipped_optimizer_steps", 0)
        runner._codetr_optimizer_step_succeeded = True
        runner.optimizer.zero_grad()

    @staticmethod
    def _record_nonfinite(runner, skipped):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        bad_parameters = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all().item()
            ):
                bad_parameters.append(name)
                if len(bad_parameters) == 16:
                    break
        row = {
            "epoch": int(runner.epoch) + 1,
            "iter": int(runner.iter) + 1,
            "optimizer_steps": int(runner.meta["codetr_optimizer_steps"]),
            "skipped_optimizer_steps": skipped,
            "loss": float(runner.outputs["loss"].detach().float().cpu()),
            "parameters": bad_parameters,
        }
        rank, _ = get_dist_info()
        if not rank:
            path = Path(runner.work_dir).parent / "nonfinite_gradients.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        runner.logger.warning(
            "Skipped non-finite BF16 optimizer window at epoch %d iter %d "
            "(total skipped: %d; parameters: %s)",
            row["epoch"],
            row["iter"],
            skipped,
            ", ".join(bad_parameters) or "unknown",
        )


@HOOKS.register_module()
class CompetitionExpMomentumEMAHook(ExpMomentumEMAHook):
    """Register EMA buffers first, then resume with an exact state-dict match."""

    def before_run(self, runner):
        checkpoint = self.checkpoint
        self.checkpoint = None
        super().before_run(runner)
        self.checkpoint = checkpoint
        if checkpoint is None:
            return
        load_checkpoint = runner.load_checkpoint

        def strict_load(filename, *args, **kwargs):
            kwargs["strict"] = True
            checkpoint_data = load_checkpoint(filename, *args, **kwargs)
            model = (
                runner.model.module
                if hasattr(runner.model, "module")
                else runner.model
            )
            expected = model.state_dict()
            loaded = checkpoint_data["state_dict"]
            for name, tensor in expected.items():
                saved = loaded[name]
                if saved.shape != tensor.shape or saved.dtype != tensor.dtype:
                    raise RuntimeError(
                        "Checkpoint tensor schema mismatch for {}".format(name)
                    )
            return checkpoint_data

        runner.load_checkpoint = strict_load
        try:
            runner.resume(checkpoint)
        finally:
            runner.load_checkpoint = load_checkpoint

    def after_train_iter(self, runner):
        update_due = (int(runner.iter) + 1) % self.interval == 0
        if update_due and not getattr(
            runner, "_codetr_optimizer_step_succeeded", True
        ):
            return
        super().after_train_iter(runner)



@PIPELINES.register_module()
class CompetitionSAMCopyPaste:
    def __init__(self, annotation_file, **config):
        self.annotations = Path(annotation_file)
        self.config = dict(config)
        self.plan = None

    def __call__(self, results):
        if self.plan is None:
            raise RuntimeError('SAM plan must be bound before the training iterator starts')
        image_info = results['img_info']
        index = int(image_info['sam_sample_index'])
        entries = self.plan._paste_plan.get(index, ())
        if not entries:
            return results
        image = Image.fromarray(results['img'][..., ::-1].astype(np.uint8))
        annotations = [
            {'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
             'category_id': int(label)}
            for (x1, y1, x2, y2), label in zip(results['gt_bboxes'], results['gt_labels'])
        ]
        rng = random.Random(int(self.plan.config['seed']) + self.plan.epoch * 10_000_019 + index)
        pasted, annotations = self.plan._paste_objects(image, annotations, list(entries), rng)
        results['img'] = np.asarray(pasted)[..., ::-1].copy()
        results['gt_bboxes'] = np.asarray([
            [x, y, x + width, y + height] for x, y, width, height in
            (row['bbox'] for row in annotations)
        ], dtype=np.float32).reshape(-1, 4)
        results['gt_labels'] = np.asarray([row['category_id'] for row in annotations], dtype=np.int64)
        return results


@HOOKS.register_module()
class CompetitionSAMCopyPasteEpochHook(Hook):
    def before_train_epoch(self, runner):
        if not hasattr(self, 'plan'):
            self._bind(runner.data_loader.dataset)
        self.plan.set_epoch(int(runner.epoch))
        if runner.meta is None:
            runner.meta = {}
        runner.meta['sam_copy_paste'] = dict(self.plan.stats)

    def _leaves(self, dataset):
        if hasattr(dataset, 'datasets'):
            for child in dataset.datasets:
                yield from self._leaves(child)
        else:
            yield dataset

    def _bind(self, dataset):
        image_ids, transforms = [], []
        for child in self._leaves(dataset):
            candidates = [item for item in child.pipeline.transforms if isinstance(item, CompetitionSAMCopyPaste)]
            if len(candidates) != 1:
                raise ValueError('Each Co-DETR training dataset requires one SAM transform')
            transform = candidates[0]
            if transforms and (transform.annotations != transforms[0].annotations or transform.config != transforms[0].config):
                raise ValueError('Co-DETR datasets must share one SAM recipe and annotation view')
            # Padding reuses data-info dicts. Copy every slot before stamping it.
            offset = len(image_ids)
            child.data_infos = [dict(info, sam_sample_index=offset + index) for index, info in enumerate(child.data_infos)]
            image_ids.extend(int(info.get('sam_image_id', info['id'])) for info in child.data_infos)
            transforms.append(transform)
        if not transforms:
            raise ValueError('Co-DETR requires a nonempty training dataset')
        self.plan = SAMCopyPastePlan(transforms[0].config, transforms[0].annotations, image_ids)
        for transform in transforms:
            transform.plan = self.plan


class _OptimizerBoundedLoader:
    """Stop between batches while keeping the complete native dataset and length."""

    def __init__(self, loader, runner, max_steps):
        self.loader, self.runner, self.max_steps = loader, runner, max_steps

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        batches = iter(self.loader)
        while int((self.runner.meta or {}).get('codetr_optimizer_steps', 0)) < self.max_steps:
            try:
                batch = next(batches)
            except StopIteration:
                return
            yield batch


@HOOKS.register_module()
class CompetitionLimitStepsHook(Hook):
    """Bound a smoke run by successful updates without shrinking the SAM domain."""

    def __init__(self, max_steps):
        if int(max_steps) < 1:
            raise ValueError('max_steps must be positive')
        self.max_steps = int(max_steps)

    def before_train_epoch(self, runner):
        if runner.meta is None:
            runner.meta = {}
        runner.meta['requested_max_steps'] = self.max_steps
        runner.meta['full_recipe_completed'] = False
        if not isinstance(runner.data_loader, _OptimizerBoundedLoader):
            runner.data_loader = _OptimizerBoundedLoader(runner.data_loader, runner, self.max_steps)


@HOOKS.register_module()
class CompetitionTerminalExportHook(Hook):
    def __init__(self, class_names):
        self.class_names = class_names

    def after_run(self, runner):
        rank, world_size = get_dist_info()
        if rank:
            return
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        metadata = runner.meta or {}
        optimizer_updates = int(metadata.get('codetr_optimizer_steps', _optimizer_state_step(runner.optimizer) or 0))
        requested = metadata.get('requested_max_steps')
        if requested is not None and optimizer_updates != int(requested):
            raise RuntimeError('Co-DETR did not complete the requested successful optimizer updates')
        torch.save({
            'state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items() if not name.startswith('ema_')},
            'meta': {
                'CLASSES': tuple(self.class_names), 'completed_epoch': int(runner.epoch) if requested is None else 0, 'variant': 'terminal_ema',
                'precision': PRECISION, 'world_size': int(world_size), 'seed': int(metadata.get('seed', 42)),
                'optimizer_updates': optimizer_updates,
                'requested_max_steps': requested,
                'full_recipe_completed': bool(metadata.get('full_recipe_completed', True)),
                'skipped_optimizer_windows': int(metadata.get('codetr_skipped_optimizer_steps', 0)),
                'completed_microbatches': int(runner.iter),
                'rank_microbatches_per_epoch': len(runner.data_loader),
                'sam_copy_paste': dict(metadata.get('sam_copy_paste', {})),
            },
        }, Path(runner.work_dir) / 'checkpoint_terminal_ema.pth')
