"""Synthetic geometry contracts; detector models are deliberately not loaded."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

from inference import classifiers, detector_backends as backends, detector_geometry as geometry
from postprocessing.three_head_runtime import write_csv
from training.final_detectors import LoadedDetector, class_names

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 160, 96
BOX = (32, 24, 64, 48)


def marker_box(pixels, *, bgr=False):
    values = np.asarray(pixels)
    red, green = values[..., 2 if bgr else 0], values[..., 1]
    yy, xx = np.nonzero((green > 127) & (red < 100))
    if not len(xx):
        raise AssertionError("The known green object was lost")
    return np.asarray([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1], dtype=np.float32)


def tensor_results(pixels, sizes, *, sizes_are_wh):
    results = []
    for sample, size in zip(pixels, sizes, strict=True):
        values = sample.detach().cpu().permute(1, 2, 0).numpy()
        box = marker_box(values * 255)
        width, height = map(float, size if sizes_are_wh else size[::-1])
        box *= np.asarray([width / values.shape[1], height / values.shape[0]] * 2)
        results.append({"boxes": torch.from_numpy(box[None]), "labels": torch.tensor([0]), "scores": torch.tensor([.9])})
    return results


class PixelModel(nn.Module):
    def __init__(self, *, rf=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.rf = rf

    def forward(self, pixels):
        self.inputs = pixels.detach().clone()
        if self.rf:
            mean = pixels.new_tensor([.485, .456, .406])[None, :, None, None]
            std = pixels.new_tensor([.229, .224, .225])[None, :, None, None]
            pixels = pixels * std + mean
        return pixels


class ExifGeometryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-exif-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.pixels[..., 2] = 48
        self.pixels[BOX[1]:BOX[3], BOX[0]:BOX[2]] = [0, 255, 0]
        self.pixels[4:12, 112:144] = [255, 0, 0]

    def save(self, orientation, extension="png"):
        path = self.root / f"orientation-{orientation}.{extension}"
        exif = Image.Exif()
        exif[274] = orientation
        Image.fromarray(self.pixels).save(path, exif=exif, **({"quality": 100, "subsampling": 0} if extension == "jpg" else {}))
        return path

    def test_decoder_retains_stored_pixels_for_all_orientations_and_supported_exif_formats(self):
        for extension in ("png", "jpg", "tiff"):
            for orientation in range(1, 9):
                with self.subTest(extension=extension, orientation=orientation):
                    path = self.save(orientation, extension)
                    self.assertEqual(geometry.encoded_image_info(path), (WIDTH, HEIGHT, orientation))
                    decoded = geometry.decode_encoded_rgb(path)
                    self.assertEqual(decoded.size, (WIDTH, HEIGHT))
                    self.assertNotIn(274, decoded.getexif())
                    if extension == "jpg":
                        # JPEG does not implicitly rotate in Pillow; comparison
                        # uses its actual compressed pixels, not lossless input.
                        with Image.open(path) as image:
                            expected = np.asarray(image.convert("RGB"))
                    else:
                        expected = self.pixels
                    np.testing.assert_array_equal(np.asarray(decoded), expected)

    def check_crop_and_csv(self, path, row, predictions, suffix):
        self.assertEqual(len(predictions), 1)
        prediction = predictions[0]
        np.testing.assert_allclose(prediction["bbox"], [32, 24, 32, 24], atol=1e-4, rtol=0)
        self.assertEqual(prediction["category_id"], 1)
        cache = classifiers.ImageCache()
        decoded = cache.get(path)
        np.testing.assert_array_equal(np.asarray(decoded), self.pixels)
        rectangle = classifiers.crop_rectangle(prediction["bbox"], 1.2)
        expected_rectangle = classifiers.crop_rectangle([32, 24, 32, 24], 1.2)
        self.assertEqual(rectangle, expected_rectangle)
        options = dict(size=64, mean=[0., 0., 0.], std=[1., 1., 1.], return_valid_patches=False)
        crop, _, valid = classifiers.render_crop(decoded, rectangle, **options)
        expected, _, _ = classifiers.render_crop(Image.fromarray(self.pixels), expected_rectangle, **options)
        self.assertTrue(valid)
        torch.testing.assert_close(crop, expected, atol=0, rtol=0)
        output = self.root / f"{path.stem}-{suffix}.csv"
        self.assertEqual(write_csv(output, predictions, {"images": [row]}), 1)
        with output.open(newline="") as handle:
            result = list(csv.DictReader(handle))[0]
        self.assertEqual(result["image_filename"], path.name)
        self.assertEqual(int(result["label_id"]), 0)
        self.assertEqual([float(result[key]) for key in ("x", "y", "w", "h")], [32., 24., 32., 24.])

    def test_three_encoded_collectors_flip_restore_crop_and_csv_for_all_orientations(self):
        tiny_config = {"val_dataloader": {"dataset": {"transforms": {"ops": [
            {"type": "Resize", "size": [HEIGHT, WIDTH]}, {"type": "ConvertPILImage"},
            {"type": "Normalize", "mean": [0., 0., 0.], "std": [1., 1., 1.]}]}}}}

        class FakeYolo:
            def predict(self, *, source, **kwargs):
                del kwargs
                # Exercise the pinned vendor's real source-type EXIF dispatch.
                from ultralytics.data.loaders import autocast_list
                outputs = []
                for image in autocast_list(source):
                    self_outer.assertIsInstance(image, Image.Image)
                    self_outer.assertEqual(image.size, (WIDTH, HEIGHT))
                    box = marker_box(np.asarray(image))
                    outputs.append(types.SimpleNamespace(boxes=types.SimpleNamespace(
                        xyxy=torch.from_numpy(box[None]), cls=torch.tensor([0.]), conf=torch.tensor([.9]))))
                return outputs

        self_outer = self
        tiny = LoadedDetector("tinyformer", PixelModel(), class_names(), tiny_config,
            lambda pixels, sizes: tensor_results(pixels, sizes.tolist(), sizes_are_wh=True))
        rf = LoadedDetector("rfdetr", PixelModel(rf=True), class_names(), {"num_select": 300, "resolution": 160},
            lambda pixels, target_sizes: tensor_results(pixels, target_sizes.tolist(), sizes_are_wh=False))
        yolo = LoadedDetector("yolo26", PixelModel(), class_names(), {})
        for orientation in range(1, 9):
            path = self.save(orientation)
            row = {"id": orientation, "file_name": path.name, "width": WIDTH, "height": HEIGHT}
            for flipped in (False, True):
                for name, loaded, collect in (("tiny", tiny, backends.collect_tinyformer), ("rf", rf, backends.collect_rfdetr), ("yolo", yolo, backends.collect_yolo26)):
                    with self.subTest(orientation=orientation, flipped=flipped, model=name):
                        with patch.object(backends, "yolo_api", return_value=FakeYolo()):
                            predictions = collect(loaded, {"images": [row]}, self.root, batch=1, flip=flipped)
                        self.check_crop_and_csv(path, row, predictions, f"{name}-{flipped}")

    def test_codetr_exif_and_flip_restore_follow_independent_pixel_mask(self):
        for orientation in range(1, 9):
            path = self.save(orientation)
            row = {"id": orientation, "file_name": path.name, "width": WIDTH, "height": HEIGHT}
            with Image.open(path) as image:
                display = ImageOps.exif_transpose(image).convert("RGB")
            geometry_map = geometry.codetr_geometry({"images": [row]}, self.root)
            for flipped in (False, True):
                with self.subTest(orientation=orientation, flipped=flipped):
                    view = ImageOps.mirror(display) if flipped else display
                    x1, y1, x2, y2 = map(float, marker_box(view))
                    raw = [{"image_id": orientation, "category_id": 1, "bbox": [x1, y1, x2-x1, y2-y1], "score": .9}]
                    restored = geometry.restore_codetr_predictions(raw, geometry_map, horizontal_flip=flipped)
                    self.check_crop_and_csv(path, row, restored, f"codetr-{flipped}")

    def test_boundary_boxes_follow_pixel_masks_and_csv_clipping(self):
        from postprocessing.submit import submission_row
        for orientation in range(1, 9):
            decoded = classifiers.ImageCache().get(self.save(orientation))
            for original in ((0, 0, 32, 24), (128, 72, 160, 96), (0, 0, 160, 96)):
                for flipped in (False, True):
                    with self.subTest(orientation=orientation, box=original, flipped=flipped):
                        mask = Image.new('L', (WIDTH, HEIGHT), 0)
                        mask.paste(255, original)
                        mask.getexif()[274] = orientation
                        displayed = ImageOps.exif_transpose(mask)
                        view = ImageOps.mirror(displayed) if flipped else displayed
                        yy, xx = np.nonzero(np.asarray(view))
                        box = np.asarray([[xx.min(), yy.min(), xx.max()+1, yy.max()+1]], dtype=np.float32)
                        if flipped:
                            box = geometry.inverse_horizontal_boxes(box, displayed.width)
                        restored = geometry.display_boxes_to_encoded(box, WIDTH, HEIGHT, orientation)[0]
                        np.testing.assert_array_equal(restored, original)
                        row = submission_row('image.png', 0, tuple(restored), .9, WIDTH, HEIGHT)
                        self.assertEqual([row[key] for key in ('x', 'y', 'w', 'h')],
                            [original[0], original[1], original[2]-original[0], original[3]-original[1]])
                        rectangle = classifiers.crop_rectangle([row[key] for key in ('x', 'y', 'w', 'h')], 1.2)
                        options = dict(size=64, mean=[.5]*3, std=[.5]*3, return_valid_patches=True)
                        pixels, valid, renderable = classifiers.render_crop(decoded, rectangle, **options)
                        expected, expected_valid, _ = classifiers.render_crop(Image.fromarray(self.pixels), rectangle, **options)
                        self.assertTrue(renderable)
                        torch.testing.assert_close(pixels, expected, atol=0, rtol=0)
                        torch.testing.assert_close(valid, expected_valid)
        clipped = submission_row('image.png', 0, (-3., -2., WIDTH+9., HEIGHT+7.), .9, WIDTH, HEIGHT)
        self.assertEqual([clipped[key] for key in ('x', 'y', 'w', 'h')], [0., 0., WIDTH, HEIGHT])

    def test_index_and_collection_validate_stored_tiff_dimensions(self):
        from inference.index import build_index
        from inference.detectors import load_index
        for orientation in range(1, 9):
            self.save(orientation, 'tiff')
        path = self.root/'index.json'
        document = build_index(self.root, path)
        self.assertEqual(len(document['images']), 8)
        self.assertTrue(all((row['width'], row['height']) == (WIDTH, HEIGHT) for row in document['images']))
        self.assertEqual(load_index(path, self.root), document)

    def test_single_detector_helpers_share_production_tiff_decoder(self):
        from training.final_detectors import prepare_image
        from training.local_ultralytics import local_ultralytics
        config = {"val_dataloader": {"dataset": {"transforms": {"ops": [
            {"type": "Resize", "size": [HEIGHT, WIDTH]}, {"type": "ConvertPILImage"},
            {"type": "Normalize", "mean": [0., 0., 0.], "std": [1., 1., 1.]}]}}}}
        baseline = self.save(1, 'png')
        for orientation in (6, 8):
            path = self.save(orientation, 'tiff')
            tiny = LoadedDetector('tinyformer', PixelModel(), class_names(), config)
            expected, _ = backends.tinyformer_inputs([path], config)
            torch.testing.assert_close(prepare_image(tiny, path), expected, atol=0, rtol=0)
            rf = LoadedDetector('rfdetr', PixelModel(rf=True), class_names(), {'num_select': 300, 'resolution': 160},
                lambda pixels, target_sizes: tensor_results(pixels, target_sizes.tolist(), sizes_are_wh=False))
            row = {'id': 1, 'file_name': path.name, 'width': WIDTH, 'height': HEIGHT}
            backends.collect_rfdetr(rf, {'images': [row]}, self.root, batch=1, flip=False)
            torch.testing.assert_close(prepare_image(rf, path), rf.model.inputs, atol=0, rtol=0)
            yolo = LoadedDetector('yolo26', PixelModel(), class_names(), {})
            with local_ultralytics():
                torch.testing.assert_close(prepare_image(yolo, path), prepare_image(yolo, baseline), atol=0, rtol=0)

    def test_single_classifier_helper_uses_production_encoded_tiff_crop(self):
        from inference import check_classifier as check

        class CropHead(nn.Module):
            def forward(self, pixels, valid=None):
                self.pixels, self.valid = pixels.clone(), valid.clone()
                return pixels.new_zeros((len(pixels), 34)), pixels.new_zeros(len(pixels))

        config = {**check.load_model_config('v7'), 'input_size': 64}
        box = [32., 24., 32., 24.]
        rectangle = classifiers.crop_rectangle(box, float(config['context_scale']))
        expected, expected_valid, _ = classifiers.render_crop(Image.fromarray(self.pixels), rectangle,
            size=64, mean=config['mean'], std=config['std'], return_valid_patches=True)
        for orientation in (6, 8):
            model = CropHead()
            with patch.object(check, 'load_model_config', return_value=config), patch.object(check, 'load_classifier', return_value=model):
                result = check.check('v7', self.root/'unused-model.pt', self.save(orientation, 'tiff'), box, device='cpu')
            self.assertEqual(result['image_size'], [WIDTH, HEIGHT])
            torch.testing.assert_close(model.pixels[0], expected, atol=0, rtol=0)
            torch.testing.assert_close(model.valid[0], expected_valid)


@unittest.skipUnless(importlib.util.find_spec("mmcv"), "Run with scripts/codetr_python.py for pinned native CPU call graph")
class NativeCoDETRExifTest(unittest.TestCase):
    setUp = ExifGeometryTest.setUp
    save = ExifGeometryTest.save

    def test_native_single_augmentation_head_rescales_without_inverse_flip(self):
        sys.path.insert(0, str(ROOT / "third_party/Co-DETR"))
        from mmdet.datasets import CocoDataset
        from mmdet.datasets.pipelines import LoadImageFromFile, Pad, RandomFlip, Resize
        from mmdet.models.detectors.base import BaseDetector
        from projects.models.co_deformable_detr_head import CoDeformDETRHead

        class OneImage:
            img_ids = [1]
            cat_ids = list(range(1, 35))
            def __len__(self):
                return 1
            def xyxy2xywh(self, box):
                return CocoDataset.xyxy2xywh(self, box)

        for extension, orientation in ((ext, value) for ext in ('jpg', 'png', 'tiff') for value in range(1, 9)):
            path = self.save(orientation, extension)
            row = {"id": 1, "file_name": path.name, "width": WIDTH, "height": HEIGHT}
            with Image.open(path) as image:
                expected_display = ImageOps.exif_transpose(image).convert("RGB")
            display_width, display_height = expected_display.size
            for flipped in (False, True):
                with self.subTest(extension=extension, orientation=orientation, flipped=flipped):
                    result = LoadImageFromFile()(dict(img_prefix=str(self.root), img_info={"filename": path.name}))
                    np.testing.assert_array_equal(marker_box(result['img'], bgr=True), marker_box(expected_display))
                    result = Resize(img_scale=(display_width * 2, display_height * 2), keep_ratio=True)(result)
                    result = RandomFlip(flip_ratio=float(flipped))(result)
                    result = Pad(size_divisor=32)(result)
                    box = marker_box(result['img'], bgr=True)
                    height, width = result['img_shape'][:2]
                    normalized = torch.tensor([[(box[0]+box[2])/(2*width), (box[1]+box[3])/(2*height),
                                                (box[2]-box[0])/width, (box[3]-box[1])/height]])
                    head = types.SimpleNamespace(test_cfg={'max_per_img': 1, 'score_thr': 0.},
                        loss_cls=types.SimpleNamespace(use_sigmoid=True), num_query=1, num_classes=34)
                    cls_score = torch.full((1, 34), -20.)
                    cls_score[0, 0] = 20.
                    calls = []
                    def simple_test(image, metas, rescale):
                        calls.append((metas[0]['flip'], rescale))
                        return CoDeformDETRHead._get_bboxes_single(head, cls_score, normalized,
                            metas[0]['img_shape'], metas[0]['scale_factor'], rescale=rescale)
                    stub = types.SimpleNamespace(simple_test=simple_test)
                    native_boxes, labels = BaseDetector.forward_test(stub,
                        [torch.zeros((1, 3, *result['pad_shape'][:2]))], [[result]], rescale=True)
                    self.assertEqual(calls, [(flipped, True)])
                    view = ImageOps.mirror(expected_display) if flipped else expected_display
                    np.testing.assert_allclose(native_boxes[0, :4].numpy(), marker_box(view), atol=1e-4, rtol=0)
                    by_class = [np.empty((0, 5), dtype=np.float32) for _ in range(34)]
                    by_class[int(labels[0])] = native_boxes.numpy()
                    raw = CocoDataset._det2json(OneImage(), [by_class])
                    restored = geometry.restore_codetr_predictions(raw,
                        geometry.codetr_geometry({'images': [row]}, self.root), horizontal_flip=flipped)
                    np.testing.assert_allclose(restored[0]['bbox'], [32, 24, 32, 24], atol=1e-4, rtol=0)


if __name__ == '__main__':
    unittest.main()
