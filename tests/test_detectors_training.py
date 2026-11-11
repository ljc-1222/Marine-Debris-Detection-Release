from __future__ import annotations

import json
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from pytorch_lightning import Trainer

from training import train_detector
from training.train_codetr import group_training_annotations
from training.tinyformer_training import TinyFormerEMA, TinyFormerModule, load_pretrained


ROOT = Path(__file__).resolve().parents[1]


def recipe(name):
    return train_detector.recipes(ROOT / "training/final_recipes.yaml", name)


class TransferModel(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.decoder = nn.Module()
        self.decoder.denoising_class_embed = nn.Embedding(classes, 2)
        self.decoder.enc_score_head = nn.Linear(2, classes)
        self.decoder.dec_score_head = nn.ModuleList([nn.Linear(2, classes)])
        self.decoder.register_buffer("anchors", torch.rand(3, 4))
        self.decoder.register_buffer("valid_mask", torch.ones(3, 1, dtype=torch.bool))


class DetectorTrainingTest(unittest.TestCase):
    def test_tinyformer_ema_matches_manual_post_step_updates_and_final_saved_state(self):
        class Model(nn.Linear):
            def forward(self, samples, targets=None):
                return super().forward(samples)

        class Criterion(nn.Module):
            def forward(self, outputs, targets, **metadata):
                return {"loss": 0.5 * (outputs - targets).square().mean()}

        class EpochLoader(DataLoader):
            def set_epoch(self, epoch):
                pass

        class EMA:
            def __init__(self, model):
                self.module = deepcopy(model)
                self.updates = 0
                self.observed_steps = []

            def to(self, device):
                self.module.to(device)

            @torch.no_grad()
            def update(self, model):
                self.observed_steps.append(model.weight.detach().clone())
                self.module.weight.mul_(0.5).add_(model.weight, alpha=0.5)
                self.updates += 1

        for microbatches in (4, 5, 6):
            with self.subTest(microbatches=microbatches):
                model = Model(1, 1, bias=False)
                model.weight.data.fill_(1.0)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                ema = EMA(model)
                loader = EpochLoader(TensorDataset(torch.ones(microbatches, 1), torch.zeros(microbatches, 1)), batch_size=1)
                module = TinyFormerModule(model, Criterion(), optimizer, loader,
                    {"warmup_iter": 0, "flat_epoch": 1, "no_aug_epoch": 0, "lr_gamma": 0.5},
                    lambda total, warmup, flat, no_aug, step, base, minimum: base)
                expected_weight, expected_ema = torch.ones(1, 1), torch.ones(1, 1)
                expected_steps = []
                for start in range(0, microbatches, 2):
                    # Lightning scales each microbatch by the configured accumulation=2.
                    gradient = expected_weight * (min(2, microbatches - start) / 2)
                    expected_weight = expected_weight - 0.1 * gradient
                    expected_ema = 0.5 * expected_ema + 0.5 * expected_weight
                    expected_steps.append(expected_weight.clone())
                with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                    trainer = Trainer(accelerator="cpu", devices=1, precision="32-true", max_epochs=1,
                        accumulate_grad_batches=2, callbacks=[TinyFormerEMA(ema)], logger=False,
                        enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0,
                        num_sanity_val_steps=0, default_root_dir=directory)
                    trainer.fit(module, train_dataloaders=loader)
                    saved = io.BytesIO()
                    torch.save({"model": ema.module.state_dict()}, saved)
                    saved.seek(0)
                    exported = torch.load(saved, weights_only=True)["model"]["weight"]
                self.assertEqual(ema.updates, len(expected_steps))
                self.assertEqual(trainer.global_step, len(expected_steps))
                for actual, expected in zip(ema.observed_steps, expected_steps, strict=True):
                    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
                torch.testing.assert_close(model.weight, expected_weight, atol=1e-7, rtol=0)
                torch.testing.assert_close(exported, expected_ema, atol=1e-7, rtol=0)
                # A completed fit must not keep a callback attached to the optimizer.
                model.weight.grad = torch.ones_like(model.weight)
                optimizer.step()
                self.assertEqual(ema.updates, len(expected_steps))

    def test_codetr_source_groups_keep_local_ids_and_global_sam_ids(self):
        source = {
            "images": [
                {"id": 1, "source_group": 0, "source_image_id": 42},
                {"id": 2, "source_group": 1, "source_image_id": 42},
            ],
            "annotations": [
                {"id": 9, "source_annotation_id": 7, "image_id": 1},
                {"id": 10, "source_annotation_id": 7, "image_id": 2},
            ], "categories": [],
        }
        groups = group_training_annotations(source)
        self.assertEqual(list(groups), [0, 1])
        self.assertEqual([groups[group]["images"][0]["id"] for group in groups], [42, 42])
        self.assertEqual([groups[group]["images"][0]["sam_image_id"] for group in groups], [1, 2])
        self.assertEqual([groups[group]["annotations"][0]["id"] for group in groups], [7, 7])
        self.assertEqual([groups[group]["annotations"][0]["image_id"] for group in groups], [42, 42])
        self.assertEqual(source["images"][0]["id"], 1)
        source["images"][0]["source_group"] = False
        with self.assertRaisesRegex(ValueError, "source_group integer"):
            group_training_annotations(source)

    def test_tinyformer_transfer_prefers_ema_and_preserves_new_classes(self):
        source, target = TransferModel(365), TransferModel(34)
        original_head = target.decoder.enc_score_head.weight.detach().clone()
        original_anchors = target.decoder.anchors.clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "initial.pth"
            torch.save({"ema": {"module": source.state_dict()}, "model": {}}, checkpoint)
            report = load_pretrained(target, checkpoint)
        self.assertEqual(report["source_variant"], "ema")
        torch.testing.assert_close(target.backbone.weight, source.backbone.weight)
        torch.testing.assert_close(target.decoder.enc_score_head.weight, original_head)
        torch.testing.assert_close(target.decoder.anchors, original_anchors)

    def test_tinyformer_transfer_rejects_an_unrelated_missing_tensor(self):
        target = TransferModel(34)
        source = TransferModel(365).state_dict()
        source.pop("backbone.weight")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "initial.pth"
            torch.save({"model": source}, checkpoint)
            with self.assertRaisesRegex(ValueError, "backbone.weight"):
                load_pretrained(target, checkpoint)

    def test_tinyformer_accumulation_lr_and_ema_follow_optimizer_steps(self):
        class EpochLoader(DataLoader):
            def set_epoch(self, epoch):
                epochs.append(epoch)

        class Model(nn.Linear):
            def forward(self, samples, targets=None):
                return super().forward(samples)

        class Criterion(nn.Module):
            def forward(self, outputs, targets, **metadata):
                metas.append(metadata)
                return {"loss": (outputs - targets).square().mean()}

        class EMA:
            def __init__(self):
                self.updates = 0

            def to(self, device):
                return self

            def update(self, model):
                self.updates += 1

        epochs, metas, lr_calls = [], [], []

        def scheduler(total, warmup, flat, no_aug, step, base, minimum):
            lr_calls.append((total, step))
            return base

        loader = EpochLoader(TensorDataset(torch.ones(10, 1), torch.zeros(10, 1)), batch_size=2)
        model = Model(1, 1)
        ema = EMA()
        module = TinyFormerModule(
            model, Criterion(), torch.optim.AdamW(model.parameters(), lr=0.01), loader,
            {"warmup_iter": 0, "flat_epoch": 1, "no_aug_epoch": 1, "lr_gamma": 0.5}, scheduler,
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                accelerator="cpu", devices=1, precision="32-true", max_epochs=2,
                accumulate_grad_batches=2, callbacks=[TinyFormerEMA(ema)], gradient_clip_val=0.1,
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                limit_val_batches=0, num_sanity_val_steps=0, default_root_dir=directory,
            )
            trainer.fit(module, train_dataloaders=loader)
        self.assertEqual(epochs, [0, 1])
        self.assertEqual(trainer.global_step, 6)
        self.assertEqual(ema.updates, 6)
        self.assertEqual(lr_calls, [(6, step) for step in range(7)])
        self.assertEqual(len(metas), 10)
        self.assertTrue(all(row["epoch_step"] == 5 for row in metas))

    def test_rf_detr_native_config_pins_seed_precision_and_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = train_detector.load_data(ROOT / "training/data.example.yaml")["class_names"]
            model, training = train_detector.rfdetr_configs(
                recipe("rfdetr"), {"class_names": names, "rfdetr": {"dataset_dir": "."}},
                root / "data.yaml", root / "initial.pth", root / "output", "cuda:0", 8,
            )
        self.assertEqual(model.resolution, 952)
        self.assertEqual(model.num_classes, 34)
        self.assertEqual(model.model_name, "RFDETRLargeDeprecated")
        self.assertEqual(training.seed, 42)
        self.assertEqual(training.amp_dtype, "bf16")
        self.assertEqual(training.clip_max_norm, 0.1)
        self.assertEqual(training.grad_accum_steps, 16)
        self.assertEqual(training.lr_scheduler_kwargs, {"lr_drop": 10})
        self.assertEqual(training.warmup_epochs, 0.0)
        self.assertEqual(training.class_names, names)
        self.assertFalse(training.wandb)
        self.assertFalse(training.mlflow)
        self.assertFalse(training.clearml)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA allocation required")
    def test_tinyformer_bf16_model_and_fp32_criterion_context(self):
        dtypes, criterion_autocast = [], []

        class Model(nn.Linear):
            def forward(self, samples, targets=None):
                output = super().forward(samples)
                dtypes.append(output.dtype)
                return output

        class Criterion(nn.Module):
            def forward(self, outputs, targets, **metadata):
                criterion_autocast.append(torch.is_autocast_enabled("cuda"))
                return {"loss": (outputs.float() - targets.float()).square().mean()}

        class EpochLoader(DataLoader):
            def set_epoch(self, epoch):
                pass

        loader = EpochLoader(TensorDataset(torch.ones(4, 1), torch.zeros(4, 1)), batch_size=2)
        model = Model(1, 1)
        module = TinyFormerModule(
            model, Criterion(), torch.optim.AdamW(model.parameters(), lr=0.01), loader,
            {"warmup_iter": 0, "flat_epoch": 0, "no_aug_epoch": 0, "lr_gamma": 0.5},
            lambda total, warmup, flat, no_aug, step, base, minimum: base,
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                accelerator="gpu", devices=1, precision=recipe("tinyformer")["precision"],
                max_epochs=1, accumulate_grad_batches=2, gradient_clip_val=0.1,
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                limit_val_batches=0, num_sanity_val_steps=0, default_root_dir=directory,
            )
            trainer.fit(module, train_dataloaders=loader)
        self.assertEqual(dtypes, [torch.bfloat16, torch.bfloat16])
        self.assertEqual(criterion_autocast, [False, False])
        self.assertEqual(trainer.global_step, 1)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))

    def test_yolo_stages_chain_fixed_snapshots_and_record_actual_steps(self):
        instances = []

        class FakeYOLO:
            def __init__(self, initial):
                self.initial = Path(initial)
                self.callbacks = {}
                self.model = types.SimpleNamespace(yaml={"yaml_file": "yolo26x.yaml"})
                instances.append(self)

            def add_callback(self, event, callback):
                self.callbacks[event] = callback

            def train(self, **options):
                self.options = options
                vendor.utils.callbacks.add_integration_callbacks(self)
                output = Path(options["project"]) / options["name"]
                last = output / "last.pt"
                trainer = types.SimpleNamespace(
                    epoch=0, epochs=options["epochs"], args=types.SimpleNamespace(**options),
                    last=last, ema=types.SimpleNamespace(updates=0), amp=True,
                    optimizer=torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.1), stop=False,
                )

                def save_model():
                    last.write_text(f"ema:{output.name}:epoch:{trainer.epoch + 1}")

                trainer.save_model = save_model
                self.callbacks["on_train_start"](trainer)
                for epoch in range(options["epochs"]):
                    trainer.epoch = epoch
                    self.callbacks["on_train_epoch_start"](trainer)
                    for _ in range(3):
                        trainer.optimizer.step()
                        trainer.ema.updates += 1
                        self.callbacks["on_train_batch_end"](trainer)
                        if trainer.stop:
                            break
                    if trainer.validate() != ({}, 0.0):
                        raise AssertionError("training must not score its complete input dataset")
                    self.callbacks["on_fit_epoch_end"](trainer)
                    if trainer.stop:
                        break
                trainer.final_eval()
                self.final_augmentation = vars(trainer.args)

        vendor = types.ModuleType("ultralytics")
        vendor.__file__ = str(ROOT / "third_party/ultralytics/ultralytics/__init__.py")
        vendor.__version__ = "8.4.138"
        vendor.YOLO = FakeYOLO
        vendor.settings = {}
        vendor.utils = types.ModuleType("ultralytics.utils")
        def refuse_external_logger(instance):
            raise AssertionError("An external integration was registered")
        vendor.utils.callbacks = types.ModuleType("ultralytics.utils.callbacks")
        vendor.utils.callbacks.add_integration_callbacks = refuse_external_logger
        events = types.ModuleType("ultralytics.utils.events")
        events.events = types.SimpleNamespace(enabled=True, events=[])
        adapter = types.ModuleType("training.yolo_data")
        adapter.FinalYOLOTrainer = type("FinalYOLOTrainerFixture", (), {})
        adapter.IMAGE_POLICY = "stored-opencv-ignore-exif-v1"
        fake_modules = {"ultralytics": vendor, "ultralytics.utils": vendor.utils,
                        "ultralytics.utils.callbacks": vendor.utils.callbacks,
                        "ultralytics.utils.events": events, "training.yolo_data": adapter}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "yolo26x.pt"
            initial.write_text("official-initializer-fixture")
            names = train_detector.load_data(ROOT / "training/data.example.yaml")["class_names"]
            dataset = root / "dataset.yaml"
            dataset.write_text(yaml.safe_dump({"path": ".", "train": "images", "val": "images", "names": names}))
            output = train_detector.prepare_output(root / "output")
            with patch.dict(sys.modules, fake_modules), patch.object(sys, "path", sys.path.copy()), patch.dict(train_detector.os.environ, {"ULTRALYTICS_API_KEY": "disabled-fixture-key"}):
                artifact = train_detector.run_yolo26(
                    recipe("yolo26"), {"class_names": names, "yolo26": {"dataset_yaml": "dataset.yaml"}},
                    root / "data.yaml", initial, output, "cuda:0", 8,
                )
            first = output / "stage1/checkpoint_fixed_epoch_40.pt"
            self.assertEqual(instances[0].initial, initial)
            self.assertEqual(instances[1].initial, first)
            self.assertEqual(first.read_text(), "ema:stage1:epoch:40")
            self.assertEqual(artifact.read_text(), "ema:stage2:epoch:7")
            first_manifest = json.loads((first.parent / "stage_manifest.json").read_text())
            second_manifest = json.loads((artifact.parent / "stage_manifest.json").read_text())
            self.assertEqual(first_manifest["batch_steps"], 120)
            self.assertEqual(second_manifest["batch_steps"], 21)
            self.assertEqual(second_manifest["initial_sha256"], train_detector.sha256(first))
            self.assertEqual(second_manifest["variant"], "ema")
            self.assertTrue(all(instance.options["trainer"] is adapter.FinalYOLOTrainer for instance in instances))
            self.assertEqual(second_manifest["training_image_policy"], adapter.IMAGE_POLICY)
            self.assertFalse(second_manifest["label_cache_reuse"])
            self.assertFalse(second_manifest["adjacent_npy_reuse"])
            self.assertEqual(instances[0].options["lr0"], 0.001)
            self.assertEqual(instances[1].options["lr0"], 0.0001)
            self.assertEqual(instances[1].options["freeze"], 10)
            self.assertEqual(instances[1].options["seed"], 42)
            self.assertEqual(instances[1].final_augmentation["translate"], 0.0)
            self.assertEqual(instances[1].final_augmentation["fliplr"], 0.5)
            self.assertIs(vendor.utils.callbacks.add_integration_callbacks, refuse_external_logger)
            self.assertTrue(all(vendor.settings[name] is False for name in ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb")))
            bounded = train_detector.prepare_output(root / "bounded")
            with patch.dict(sys.modules, fake_modules), patch.object(sys, "path", sys.path.copy()), patch.dict(train_detector.os.environ, {"ULTRALYTICS_API_KEY": "disabled-fixture-key"}):
                partial = train_detector.run_yolo26(
                    recipe("yolo26"), {"class_names": names, "yolo26": {"dataset_yaml": "dataset.yaml"}},
                    root / "data.yaml", initial, bounded, "cuda:0", 0, max_steps=2,
                )
            for stage in ("stage1", "stage2"):
                record = json.loads((bounded / stage / "stage_manifest.json").read_text())
                self.assertEqual(record["optimizer_updates"], 2)
                self.assertEqual(record["batch_steps"], 2)
                self.assertEqual(record["requested_max_steps"], 2)
                self.assertFalse(record["full_recipe_completed"])
                self.assertEqual(record["export_epoch"], 1)
            self.assertEqual(partial.name, "checkpoint_after_steps_2_ema.pt")
            self.assertEqual(instances[3].initial, bounded / "stage1/checkpoint_after_steps_2_ema.pt")
            self.assertTrue(all(instance.options["trainer"] is adapter.FinalYOLOTrainer for instance in instances))
            self.assertIs(vendor.utils.callbacks.add_integration_callbacks, refuse_external_logger)


if __name__ == "__main__":
    unittest.main()
