"""Load the four complete trained detector exports without upstream weights."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from training.checkpoints import DEFAULT_MANIFEST, load_manifest, verify_file
from training.local_ultralytics import local_ultralytics

ROOT = Path(__file__).resolve().parents[1]
DETECTORS = ("codetr", "tinyformer", "rfdetr", "yolo26")
CLASS_COUNT = 34
TINYFORMER_CONFIG = Path("configs/tinyformer/object365/tinyformer_dinov3_xl_obj2coco_pbm_24e.yml")
CODETR_CONFIG = ROOT / "third_party/Co-DETR/projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py"


@dataclass
class LoadedDetector:
    name: str
    model: nn.Module
    class_names: list[str]
    config: Any
    postprocessor: nn.Module | None = None


def class_names() -> list[str]:
    categories = json.loads((ROOT / "configs/categories.json").read_text(encoding="utf-8"))["categories"]
    if [row["id"] for row in categories] != list(range(1, CLASS_COUNT + 1)):
        raise ValueError("Expected ordered COCO categories 1..34")
    return [row["name"] for row in categories]


def _vendor(path: Path, package_name: str) -> None:
    if not (path / package_name / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing pinned {package_name} submodule; initialize the repository submodules")
    existing = sys.modules.get(package_name)
    source = getattr(existing, "__file__", None)
    if source is not None and not Path(source).resolve().is_relative_to(path.resolve()):
        raise ImportError(f"An unpinned {package_name} package is already imported")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _tensor_state(document: Mapping[str, Any], key: str) -> Mapping[str, torch.Tensor]:
    state = document.get(key)
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError(f"Expected a complete tensor state_dict under {key}")
    return state


def _build_tinyformer(document: Mapping[str, Any], _checkpoint: Path) -> LoadedDetector:
    vendor = ROOT / "third_party/TinyFormer"
    _vendor(vendor, "engine")
    from engine.core import YAMLConfig

    config = deepcopy(document["tinyformer_config"])
    source = Path(document.get("source_config", str(TINYFORMER_CONFIG)))
    if source != TINYFORMER_CONFIG:
        raise ValueError("Unexpected TinyFormer source configuration")
    if config.get("num_classes") != CLASS_COUNT or config.get("remap_mscoco_category") is not False:
        raise ValueError("TinyFormer requires the 34-class no-remap configuration")
    config.setdefault("DINOv3SSAs_4Scale", {})["weights_path"] = None
    native = YAMLConfig(str(vendor / source), **config)
    return LoadedDetector("tinyformer", native.model, class_names(), config, native.postprocessor)


def _build_rfdetr(document: Mapping[str, Any], checkpoint: Path) -> LoadedDetector:
    if importlib.metadata.version("rfdetr") != "1.9.0":
        raise RuntimeError("The trained legacy detector requires rfdetr==1.9.0")
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import RFDETRLargeDeprecatedConfig, TrainConfig
    from rfdetr.models import PostProcess, build_model

    config = deepcopy(document["model_config"])
    config.update(num_classes=CLASS_COUNT, pretrain_weights=None, device="cpu")
    model_config = RFDETRLargeDeprecatedConfig(**config)
    if model_config.encoder != "dinov2_windowed_base":
        raise ValueError("Expected the legacy RF-DETR Large backbone")
    args = _namespace_from_configs(model_config, TrainConfig(dataset_dir=None, output_dir="output"))
    args.force_no_pretrain = True
    args.pretrained_encoder = None
    # The pinned builder uses this non-None path solely to disable DINOv2 Hub initialization.
    args.pretrain_weights = str(checkpoint)
    model = build_model(args)
    postprocessor = PostProcess(num_select=int(model_config.num_select))
    return LoadedDetector("rfdetr", model, class_names(), model_config.model_dump(), postprocessor)


def _build_yolo26(document: Mapping[str, Any], _checkpoint: Path) -> LoadedDetector:
    _vendor(ROOT / "third_party/ultralytics", "ultralytics")
    with local_ultralytics() as ultralytics:
        from ultralytics.nn.tasks import DetectionModel
        if ultralytics.__version__ != "8.4.138":
            raise RuntimeError("The trained YOLO26 detector requires the pinned Ultralytics 8.4.138 source")
        config = deepcopy(document["model_config"])
        if not isinstance(config, dict) or config.get("nc") != CLASS_COUNT:
            raise ValueError("YOLO26 requires its complete 34-class architecture mapping")
        model = DetectionModel(cfg=config, ch=3, nc=CLASS_COUNT, verbose=False)
        model.names = dict(enumerate(class_names()))
    return LoadedDetector("yolo26", model, class_names(), dict(document["inference"]))


def _build_codetr(_document: Mapping[str, Any], _checkpoint: Path) -> LoadedDetector:
    _vendor(ROOT / "third_party/Co-DETR", "mmdet")
    from mmcv import Config
    from mmcv.runner.fp16_utils import wrap_fp16_model
    import projects  # noqa: F401
    from training import codetr_runtime  # noqa: F401
    from mmdet.models import build_detector

    config = Config.fromfile(str(CODETR_CONFIG))
    config.model.backbone.update(xattn=True, use_act_checkpoint=False, init_cfg=None)
    config.model.pretrained = None
    config.model.query_head.num_classes = CLASS_COUNT
    config.model.roi_head[0].bbox_head.num_classes = CLASS_COUNT
    config.model.bbox_head[0].num_classes = CLASS_COUNT
    config.model.test_cfg[0].max_per_img = 300
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    wrap_fp16_model(model)
    model.CLASSES = tuple(class_names())
    model.cfg = config
    return LoadedDetector("codetr", model, class_names(), config)


def load_detector(
    model_name: str,
    checkpoint: Path,
    *,
    manifest: Path = DEFAULT_MANIFEST,
    device: str | torch.device = "cpu",
) -> LoadedDetector:
    """Check catalog bytes/SHA256 before restricted deserialization and strict load."""
    if model_name not in DETECTORS:
        raise ValueError(f"Unknown final detector: {model_name}")
    catalog = load_manifest(Path(manifest))
    if model_name not in catalog["artifacts"]:
        raise ValueError(f"Detector is missing from the checkpoint catalog: {model_name}")
    path = Path(checkpoint).expanduser().resolve()
    verify_file(path, catalog["artifacts"][model_name])
    document = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(document, dict):
        raise ValueError("Expected a trained detector checkpoint mapping")
    expected_names = class_names()
    if model_name in {"codetr", "yolo26"}:
        if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
            raise ValueError("Expected a version-1 trained detector export")
        if document.get("model_name") != model_name:
            raise ValueError(f"Expected detector model_name={model_name}")
    if model_name == "codetr":
        names = document.get("meta", {}).get("CLASSES")
    elif model_name == "tinyformer":
        if document.get("backend") != "tinyformer" or document.get("num_classes") != CLASS_COUNT:
            raise ValueError("Expected a 34-class TinyFormer inference export")
        if not isinstance(document.get("tinyformer_config"), dict):
            raise ValueError("TinyFormer is missing its resolved architecture configuration")
        names = document.get("class_names")
    elif model_name == "rfdetr":
        if document.get("model_name") != "RFDETRLargeDeprecated" or not isinstance(document.get("model_config"), dict):
            raise ValueError("Expected the legacy RF-DETR Large export")
        if not isinstance(document.get("args"), dict) or document["args"].get("num_classes") != CLASS_COUNT:
            raise ValueError("RF-DETR must declare 34 foreground classes")
        names = document["args"].get("class_names")
    else:
        names = document.get("class_names")
        if not isinstance(document.get("inference"), dict) or document["inference"] != {"imgsz": 960, "task": "detect"}:
            raise ValueError("YOLO26 requires its fixed 960-pixel detection contract")
    if not isinstance(names, (list, tuple)) or list(names) != expected_names:
        raise ValueError("Detector class names differ from the project's ordered 34 categories")
    state = _tensor_state(document, "state_dict" if model_name == "codetr" else "model")
    factories = {"codetr": _build_codetr, "tinyformer": _build_tinyformer, "rfdetr": _build_rfdetr, "yolo26": _build_yolo26}
    loaded = factories[model_name](document, path)
    loaded.model.load_state_dict(state, strict=True)
    # TinyFormer's upstream parameter flags select its pinned encoder execution path.
    if model_name != "tinyformer":
        loaded.model.requires_grad_(False)
    loaded.model.to(device).eval()
    if loaded.postprocessor is not None:
        loaded.postprocessor.to(device).eval()
    return loaded


def prepare_image(loaded: LoadedDetector, image_path: Path) -> torch.Tensor:
    """Prepare one RGB image with the detector's fixed input transform."""
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if loaded.name == "codetr":
        raise ValueError("Co-DETR uses the native MMDetection test pipeline")
    from inference.detector_geometry import decode_encoded_rgb
    image = decode_encoded_rgb(path)
    if loaded.name == "tinyformer":
        from torchvision.transforms import v2

        operations = loaded.config["val_dataloader"]["dataset"]["transforms"]["ops"]
        if [row["type"] for row in operations] != ["Resize", "ConvertPILImage", "Normalize"]:
            raise ValueError("Unexpected TinyFormer inference transforms")
        transform = v2.Compose([v2.Resize(tuple(operations[0]["size"])), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=operations[2]["mean"], std=operations[2]["std"])])
        tensor = transform(image)
    elif loaded.name == "rfdetr":
        from torchvision.transforms import functional as F

        tensor = F.pil_to_tensor(image).float().div(255)
        tensor = F.resize(tensor, [int(loaded.config["resolution"])] * 2, antialias=False)
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        from ultralytics.data.augment import LetterBox

        pixels = LetterBox(new_shape=(960, 960), auto=False)(image=np.asarray(image))
        tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).float().div(255)
    return tensor.unsqueeze(0).contiguous()


def inference_precision(model_name: str, device: str | torch.device) -> str:
    if model_name == "codetr":
        return "native_mmcv"
    return "bfloat16_autocast" if model_name == "tinyformer" and torch.device(device).type == "cuda" else "float32"


def inference_autocast(model_name: str, device: str | torch.device):
    if inference_precision(model_name, device) == "bfloat16_autocast":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def forward_image(loaded: LoadedDetector, image_path: Path) -> dict[str, Any]:
    """Run one image and check output shape/finiteness, without evaluating a score."""
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if loaded.name == "codetr":
        from mmdet.apis import inference_detector

        result = inference_detector(loaded.model, str(path))
        if not isinstance(result, list) or len(result) != CLASS_COUNT:
            raise ValueError("Co-DETR must return one detection array per foreground class")
        arrays = [np.asarray(values) for values in result]
        if any(values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all() for values in arrays):
            raise ValueError("Co-DETR returned malformed or non-finite detections")
        return {"output_kind": "per_class_xyxy_score", "classes": CLASS_COUNT, "detections": sum(len(values) for values in arrays), "precision": "native_mmcv"}
    device = next(loaded.model.parameters()).device
    tensor = prepare_image(loaded, path).to(device)
    with inference_autocast(loaded.name, device):
        output = loaded.model(tensor)
    if loaded.name == "yolo26":
        predictions = output[0] if isinstance(output, tuple) else output
        if not isinstance(predictions, torch.Tensor) or predictions.ndim != 3 or predictions.shape[0] != 1 or predictions.shape[-1] != 6:
            raise ValueError("YOLO26 must return batch/query/[xyxy, score, class] predictions")
        if not torch.isfinite(predictions).all():
            raise ValueError("YOLO26 returned non-finite predictions")
        return {"output_kind": "end_to_end_xyxy_score_class", "prediction_shape": list(predictions.shape), "input_shape": list(tensor.shape), "precision": inference_precision(loaded.name, device)}
    if not isinstance(output, dict):
        raise ValueError("Expected a DETR output mapping")
    logits, boxes = output.get("pred_logits"), output.get("pred_boxes")
    expected_classes = CLASS_COUNT + (loaded.name == "rfdetr")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != expected_classes:
        raise ValueError("Unexpected detector class-logit shape")
    if not isinstance(boxes, torch.Tensor) or boxes.shape != (*logits.shape[:2], 4):
        raise ValueError("Detector boxes and logits are not aligned")
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
        raise ValueError("Detector returned non-finite logits or boxes")
    return {"output_kind": "raw_detr_logits_boxes", "logit_shape": list(logits.shape), "box_shape": list(boxes.shape), "input_shape": list(tensor.shape), "precision": inference_precision(loaded.name, device)}
