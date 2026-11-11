# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Dataset/trainer integration adapted from Ultralytics dad7bb4534c95021bc14969ab25d77b77c4efdc3.
"""YOLO training adapters for the project's stored-pixel annotation contract."""
from __future__ import annotations

from copy import copy
from pathlib import Path
import sys
from types import FunctionType

import cv2

from training.local_ultralytics import local_ultralytics

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/ultralytics"
IMAGE_POLICY = "stored-opencv-ignore-exif-v1"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

with local_ultralytics() as upstream:
    if upstream.__version__ != "8.4.138" or not Path(upstream.__file__).resolve().is_relative_to(VENDOR.resolve()):
        raise ImportError("YOLO training requires the pinned local Ultralytics source")
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import IMG_FORMATS, get_split_fraction, verify_image_label
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils import colorstr
    from ultralytics.utils.patches import _image_open as open_stored_image, imread
    from ultralytics.utils.torch_utils import unwrap_model


def check_stored_image(image_path):
    """Check stored size and decode unusual JPEG endings without any repairs."""
    # Use the pinned source's original Pillow decoder. Its global HEIF fallback
    # tries to install an optional codec even when an ordinary JPEG is corrupt.
    with open_stored_image(image_path) as image:
        shape = (image.height, image.width)
        image_format = image.format
        image.verify()
    if min(shape) <= 9 or image_format.lower() not in IMG_FORMATS:
        raise ValueError(f"Invalid YOLO image size or format: {image_path}")
    if image_format in {"JPEG", "JPG"}:
        with open(image_path, "rb") as stream:
            stream.seek(-2, 2)
            ending = stream.read()
        # Trailing metadata and absent EOI can still be decoded. Unlike native
        # check_image, accept that read-only result rather than rewriting JPEGs.
        if ending != b"\xff\xd9":
            pixels = imread(image_path, flags=cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if pixels is None or tuple(pixels.shape[:2]) != shape:
                raise ValueError(f"Stored-pixel image cannot be decoded with its declared size: {image_path}")
    return "", shape


def verify_stored_image_label(arguments):
    """Raise image errors before scanning; retain native label errors/counters."""
    checked = check_stored_image(arguments[0])
    # Rebind only the image check in a private function copy. Patching vendor
    # globals would be unsafe across cache_labels' verification threads.
    label_parser = FunctionType(
        verify_image_label.__code__,
        {**verify_image_label.__globals__, "check_image": lambda _path: checked},
        verify_image_label.__name__, verify_image_label.__defaults__, verify_image_label.__closure__,
    )
    return label_parser(arguments)


class StoredYOLODataset(YOLODataset):
    """Preserve native augmentation while excluding unknown image/label caches."""

    def __init__(self, *args, cache_root: Path, **kwargs):
        if kwargs.get("cache") not in (False, None):
            raise ValueError("The fixed stored-pixel YOLO recipe requires cache=False")
        self.cache_root = Path(cache_root).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.label_cache_path = self.cache_root / "labels.cache"
        super().__init__(*args, **kwargs)
        # BaseDataset reads adjacent .npy files even with cache=False. These new
        # paths are output-owned, never populated by the fixed no-image-cache recipe.
        self.npy_files = [self.cache_root / "images" / (Path(name).name + ".npy") for name in self.im_files]

    def get_labels(self):
        # BaseDataset sets cv2_flag immediately before calling this hook.
        self.cv2_flag |= cv2.IMREAD_IGNORE_ORIENTATION
        return super().get_labels()

    def verify_args(self):
        _native_verifier, arguments = super().verify_args()
        return verify_stored_image_label, arguments

    def _load_or_scan_cache(self, cache_path, cache_hash):
        # Always rescan real labels; old caches have neither frame nor content binding.
        return self.cache_labels(self.label_cache_path), False

    def load_image(self, i, rect_mode=True, resize_short=False):
        if self.npy_files[i].exists():
            raise ValueError("Refusing an unverified YOLO image cache: " + str(self.npy_files[i]))
        return super().load_image(i, rect_mode=rect_mode, resize_short=resize_short)


class FinalYOLOTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        with local_ultralytics():
            super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        if self.args.task != "detect" or mode not in {"train", "val"}:
            raise ValueError("The fixed YOLO trainer supports detection train/val datasets")
        # Match pinned build_yolo_dataset's detection arguments; only the dataset
        # class and its output-owned cache policy differ.
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        fraction = 1.0 if self.data.get("complete") else get_split_fraction(self.args.fraction, mode)
        return StoredYOLODataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
            augment=mode == "train", hyp=copy(self.args), rect=self.args.rect or mode == "val",
            cache=self.args.cache or None, single_cls=self.args.single_cls or False,
            stride=stride, pad=0.0 if mode == "train" else 0.5, prefix=colorstr(f"{mode}: "),
            task=self.args.task, classes=self.args.classes, data=self.data, fraction=fraction,
            cache_root=Path(self.save_dir) / "dataset-cache" / IMAGE_POLICY / mode,
        )
