"""Rebuild the fixed YOLO26 trained release from its verified native source."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any

import torch

from training.checkpoints import DEFAULT_MANIFEST, destination, load_manifest, verify_file
from training.final_detectors import ROOT, _vendor, class_names
from training.local_ultralytics import local_ultralytics

# torch.save uses this basename inside the ZIP. Keep it for exact byte rebuilds.
_ARCHIVE_FILENAME = "yolo26x_detector_full_finaltarget_e07_ema.preparing.pt"


def safe_config(value: Any, key: str = "") -> Any:
    """Preserve architecture order while removing initialization and host paths."""
    if key in ("weights_path", "pretrain_weights", "resume", "tuning"):
        return None
    if key == "pretrained":
        return False
    if isinstance(value, dict):
        return {str(name): safe_config(item, str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_config(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return "data/" + Path(value).name
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Nonprimitive inference configuration: {key}: {type(value).__name__}")


@contextmanager
def _pin_ultralytics():
    _vendor(ROOT / "third_party/ultralytics", "ultralytics")
    with local_ultralytics() as ultralytics:
        if ultralytics.__version__ != "8.4.138":
            raise RuntimeError("YOLO26 export requires the pinned Ultralytics 8.4.138 source")
        yield ultralytics


def export_yolo26(checkpoint: Path, directory: Path) -> dict[str, Any]:
    catalog = load_manifest(DEFAULT_MANIFEST)
    artifact = catalog["artifacts"]["yolo26"]
    source = artifact.get("source_checkpoint")
    if not isinstance(source, dict) or not {"bytes", "sha256"} <= source.keys():
        raise ValueError("The catalog must identify the original native YOLO26 checkpoint")
    checkpoint = Path(checkpoint).expanduser().resolve()
    verify_file(checkpoint, {**source, "path": checkpoint.name})
    directory = Path(directory).expanduser().resolve()
    target = destination(directory, artifact, canonical=True)
    if target.name != "yolo26x.pt":
        raise ValueError("Unexpected fixed YOLO26 release filename")
    existing = destination(directory, artifact)
    if existing.exists() or existing.is_symlink():
        return {**verify_file(existing, artifact), "path": existing.relative_to(directory).as_posix(),
                "status": "existing_verified"}
    temporary = target.with_name(_ARCHIVE_FILENAME)
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing export staging file: {temporary}")
    # Only the catalog-matched owner source may use native module deserialization.
    with _pin_ultralytics():
        native = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(native, dict):
        raise ValueError("Expected the fixed native YOLO26 checkpoint mapping")
    selected = native.get("ema")
    if not isinstance(selected, torch.nn.Module):
        selected = native.get("model")
    if not isinstance(selected, torch.nn.Module):
        raise ValueError("The native checkpoint contains no detector or EMA module")
    state = dict(selected.state_dict())
    if not state or any(not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in state.items()):
        raise ValueError("Expected the native detector's complete tensor state_dict")
    config = safe_config(selected.yaml)
    if not isinstance(config, dict):
        raise ValueError("Expected the native YOLO26 architecture mapping")
    config["yaml_file"] = "yolo26x.yaml"
    export = {"schema_version": 1, "model_name": "yolo26"}
    export.update(model=state, model_config=config, class_names=list(class_names()), inference={"imgsz": 960, "task": "detect"})
    target.parent.mkdir(parents=True, exist_ok=True)
    # Reserve exclusively, then save by pathname: the ZIP basename is part of its identity.
    temporary.open("xb").close()
    try:
        torch.save(export, temporary)
        try:
            result = verify_file(temporary, artifact)
        except ValueError as error:
            raise ValueError("Exported bytes differ from the fixed release catalog; use the pinned PyTorch/environment. No canonical checkpoint was published") from error
        restored = torch.load(temporary, map_location="cpu", mmap=True, weights_only=True)
        if set(restored["model"]) != set(state) or any(
            restored["model"][key].dtype != value.dtype
            or restored["model"][key].shape != value.shape
            or not torch.equal(restored["model"][key], value)
            for key, value in state.items()
        ):
            raise ValueError("Exported model tensors differ from the selected native state")
        try:
            os.link(temporary, target)
        except FileExistsError:
            return {**verify_file(target, artifact), "status": "existing_verified"}
        return {**result, "status": "exported", "source_sha256": source["sha256"], "tensor_count": len(state)}
    finally:
        temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="The catalog-matched original trained native checkpoint")
    parser.add_argument("--directory", type=Path, required=True, help="Root under which to create the canonical checkpoints/ path")
    args = parser.parse_args()
    print(json.dumps(export_yolo26(args.checkpoint, args.directory), sort_keys=True))


if __name__ == "__main__":
    main()
