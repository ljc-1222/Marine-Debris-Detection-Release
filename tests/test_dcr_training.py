from __future__ import annotations

import json
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch._inductor import config as compiler_config

from training import models, train_dcr


class DCRInitializationTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "compiled training check requires CUDA")
    def test_compiled_training_preserves_gradients_and_raw_checkpoint_schema(self):
        from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers

        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-dcr-") as directory:
            root = Path(directory)
            def factory():
                return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 2)).cuda()
            model = factory()
            initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            pixels = torch.randn(3, 4, device="cuda")
            expected = model(pixels).detach()
            recipe = train_dcr.load_recipe(Path("training/final_recipes.yaml"), "siglip")
            with patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"), "TRITON_CACHE_DIR": str(root / "triton"), "TORCHINDUCTOR_COMPILE_THREADS": "2"}), compiler_config.patch(compile_threads=2):
                try:
                    runtime = train_dcr.training_runtime(model, recipe, root)
                    output = runtime(pixels)
                    torch.testing.assert_close(output, expected)
                    output.square().mean().backward()
                    self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
                    optimizer.step()
                    self.assertTrue(any(not torch.equal(value, initial[key]) for key, value in model.state_dict().items()))
                    saved = io.BytesIO()
                    torch.save(model.state_dict(), saved)
                    saved.seek(0)
                    state = torch.load(saved, weights_only=True)
                    self.assertEqual(set(state), set(initial))
                    restored = factory()
                    restored.load_state_dict(state, strict=True)
                    torch.testing.assert_close(restored(pixels), model(pixels))
                finally:
                    # Cache writers must stop before TemporaryDirectory removes their files.
                    try:
                        torch.cuda.synchronize()
                    finally:
                        try:
                            shutdown_compile_workers()
                        finally:
                            if AsyncCompile.pool.cache_info().currsize:
                                AsyncCompile.pool().shutdown(wait=True)
                                AsyncCompile.pool.cache_clear()

    def test_model_specific_initialization_and_optimizer_order(self):
        for name, expected in (("v7", ["model", "optimizer", "seed"]), ("hplus", ["seed", "model", "optimizer"]), ("siglip", ["seed", "model", "optimizer"])):
            recipe = train_dcr.load_recipe(Path("training/final_recipes.yaml"), name)
            events = []
            def factory():
                events.append("model")
                return torch.nn.Linear(2, 2)
            def optimizer(_model, _recipe):
                events.append("optimizer")
                return object()
            with patch.object(train_dcr, "seed_everything", side_effect=lambda seed: events.append("seed")), patch.object(train_dcr, "create_optimizer", side_effect=optimizer):
                train_dcr.initialize_training(factory, recipe, torch.device("cpu"))
            self.assertEqual(events, expected, name)

    def test_recipe_seed_controls_real_readout_initialization_at_its_declared_boundary(self):
        # Exercise the production readouts without loading pretrained backbones.
        class Encoder(torch.nn.Module):
            num_features = 1024

        for name, dimensions in (("v7", 1024), ("hplus", 1280), ("siglip", 1024)):
            with self.subTest(model=name), torch.random.fork_rng(devices=[]):
                recipe = train_dcr.load_recipe(Path("training/final_recipes.yaml"), name)

                def factory():
                    if name == "siglip":
                        with patch("timm.create_model", side_effect=lambda *args, **kwargs: Encoder()):
                            return models.ImageTowerDCR({**recipe, "name": name})
                    return models.DINOAttentionHead(dimensions, recipe)

                def initialize(external_seed):
                    torch.manual_seed(external_seed)
                    with patch.object(train_dcr, "create_optimizer", return_value=object()):
                        model, _ = train_dcr.initialize_training(factory, recipe, torch.device("cpu"))
                    return {key: value.detach().clone() for key, value in model.state_dict().items()}, torch.get_rng_state()

                first, first_rng = initialize(17)
                second, second_rng = initialize(29)
                equal = all(torch.equal(first[key], second[key]) for key in first)
                self.assertEqual(equal, bool(recipe["seed_before_initialization"]))
                torch.testing.assert_close(first_rng, second_rng, atol=0, rtol=0)
                # Explicitly seeding before construction also reproduces the v7 readout.
                repeated, _ = initialize(17)
                for key in first:
                    torch.testing.assert_close(first[key], repeated[key], atol=0, rtol=0)

    def test_v7_keeps_native_adamw_and_other_heads_prefer_fused(self):
        model = torch.nn.Linear(2, 2)
        for name in ("v7", "hplus", "siglip"):
            recipe = train_dcr.load_recipe(Path("training/final_recipes.yaml"), name)
            with patch.object(train_dcr, "optimizer_groups", return_value=[{"params": list(model.parameters())}]), patch.object(torch.optim, "AdamW") as adamw:
                train_dcr.create_optimizer(model, recipe)
            self.assertEqual(adamw.call_args.kwargs, {"betas": (0.9, 0.999)} if name == "v7" else {"betas": (0.9, 0.999), "fused": True})

    def test_supplied_image_identity_is_checked_even_through_a_symlink(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-dcr-") as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (4, 4), "red").save(source)
            images = root / "images"
            images.mkdir()
            (images / "image.png").symlink_to(source)
            records = root / "records.jsonl"
            row = {"image_path": "image.png", "box_xyxy": [0, 0, 4, 4], "label": 0, "source_type": "gt_original", "gt_instance_id": 1, "image_sha256": train_dcr.sha256(source)}
            records.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(train_dcr.load_records(records, images)), 1)
            Image.new("RGB", (4, 4), "blue").save(source)
            with self.assertRaisesRegex(ValueError, "image bytes changed"):
                train_dcr.load_records(records, images)

    def test_hplus_filters_subpixel_crops_at_fallback_size_and_rejects_invalid_gt(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-dcr-") as directory:
            image = Path(directory) / "image.png"
            Image.new("RGB", (100, 100), "red").save(image)
            recipe = train_dcr.load_recipe(Path("training/final_recipes.yaml"), "hplus")
            foreground = train_dcr.Record(image, (0, 0, 100, 100), 0, "gt_original", 1)
            background = train_dcr.Record(image, (0, 0, 100, 100), 34, "background_regular", None)
            # This ROI intersects the image and retains >2 valid pixels at 448,
            # but fewer than 2 at the fixed 384-pixel audit resolution.
            sliver = train_dcr.Record(image, (0, 0, 18000, 18000), 34, "background_hard", None)
            self.assertEqual(
                train_dcr.filter_hplus_records([foreground, sliver, background], recipe),
                [foreground, background],
            )
            invalid_gt = train_dcr.Record(image, sliver.box, 0, "gt_original", 2)
            with self.assertRaisesRegex(RuntimeError, "invalid original GT crops"):
                train_dcr.filter_hplus_records([invalid_gt], recipe)


if __name__ == "__main__":
    unittest.main()
