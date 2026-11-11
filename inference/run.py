"""Generate a submission from images and the seven complete trained models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

from training.checkpoints import DEFAULT_MANIFEST, destination, load_manifest, verify_file
from scripts.environment import configure_caches

ROOT = Path(__file__).resolve().parents[1]
VIEWS = (("codetr", False), ("codetr", True), ("tinyformer", False),
         ("rfdetr", False), ("rfdetr", True), ("yolo26", False), ("yolo26", True))
HEADS = ("hplus", "v7", "siglip")


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def run(args: argparse.Namespace) -> dict:
    configure_caches()
    legacy = bool(getattr(args, "legacy", False))
    index = args.index.expanduser().resolve()
    images = args.images.expanduser().resolve()
    weights = args.checkpoint_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not index.is_file() or not images.is_dir():
        raise FileNotFoundError("Supply an image index and the corresponding image directory")
    document = json.loads(index.read_text(encoding="utf-8"))
    if not document.get("images") or sorted(int(row["id"]) for row in document.get("categories", [])) != list(range(1, 35)):
        raise ValueError("The index requires images and one-based categories 1..34")
    if len({int(row["id"]) for row in document["images"]}) != len(document["images"]):
        raise ValueError("Image IDs must be unique")
    expected_categories = json.loads((ROOT / "configs/categories.json").read_text(encoding="utf-8"))["categories"]
    if [(int(row["id"]), row["name"]) for row in sorted(document["categories"], key=lambda row: int(row["id"]))] != [(row["id"], row["name"]) for row in expected_categories]:
        raise ValueError("Index category names must match the fixed 34-class mapping")
    for row in document["images"]:
        name = PurePosixPath(str(row["file_name"]))
        if name.is_absolute() or ".." in name.parts or not (images / name).is_file():
            raise ValueError("Every indexed image must have a safe relative filename and exist")
        if int(row["width"]) < 1 or int(row["height"]) < 1:
            raise ValueError("Image dimensions must be positive")
    if min(args.detector_batch, args.classifier_batch) < 1 or not 0 <= args.workers <= 16:
        raise ValueError("Batch sizes must be positive and workers must be in 0..16")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    catalog_path = Path(getattr(args, "manifest", DEFAULT_MANIFEST)).expanduser().resolve()
    catalog = load_manifest(catalog_path)
    if set(catalog["artifacts"]) != {name for name, _flip in VIEWS} | set(HEADS):
        raise ValueError("The inference catalog must contain exactly the four detectors and three classifiers")
    identities = {name: verify_file(destination(weights, artifact), artifact)
                  for name, artifact in catalog["artifacts"].items()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    manifest = {
        "schema_version": 2, "status": "running", "index_sha256": sha256(index),
        "pipeline_semantics": "legacy_policy_global_top100" if legacy else "specialist_per_class_top100",
        "images": {str(row["file_name"]): sha256(images / row["file_name"]) for row in document["images"]},
        "checkpoints": identities,
        "checkpoint_catalog_sha256": sha256(catalog_path),
        "checkpoint_catalog_bytes": catalog_path.stat().st_size,
        "model_origin": catalog.get("model_origin", "project_release"),
        "allow_partial": bool(catalog.get("allow_partial", False)),
        "configuration": {name: sha256(ROOT / "configs" / name) for name in ("postprocess.yaml", "fusion.json", "submission.json")},
        "code": {path.relative_to(ROOT).as_posix(): sha256(path)
                 for directory in ("inference", "postprocessing") for path in sorted((ROOT / directory).glob("*.py"))},
        "device": args.device, "detector_batch": args.detector_batch,
        "classifier_batch": args.classifier_batch, "steps": [], "sealed_heads": {},
    }

    def save():
        temporary = output / "manifest.json.partial"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "manifest.json")

    def execute(name, command):
        stdout = output / "logs" / (name + ".out")
        stderr = output / "logs" / (name + ".err")
        with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
            result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=False)
        manifest["steps"].append({"name": name, "command": list(map(str, command)),
                                  "returncode": result.returncode,
                                  "stdout": stdout.relative_to(output).as_posix(),
                                  "stderr": stderr.relative_to(output).as_posix()})
        save()
        if result.returncode:
            manifest["status"] = "failed"
            save()
            raise RuntimeError(f"{name} failed; inspect {stderr}")

    save()
    codetr_command = shlex.split(args.codetr_command) if args.codetr_command else [sys.executable, str(ROOT / "scripts/codetr_python.py")]
    prediction_args = []
    for name, flip in VIEWS:
        view = name + ("_flip" if flip else "_original")
        prediction = output / "predictions" / (view + ".json")
        prefix = codetr_command if name == "codetr" else [sys.executable]
        command = [*prefix, "-m", "inference.detectors", "--model", name,
                   "--index", str(index), "--images", str(images), "--checkpoint-root", str(weights),
                   "--manifest", str(catalog_path),
                   "--output", str(prediction), "--device", args.device,
                   "--batch", str(1 if name == "codetr" else args.detector_batch), "--workers", str(args.workers)]
        if flip:
            command.append("--flip")
        execute(view, command)
        prediction_args += ["--prediction", view + "=" + str(prediction)]
    execute("fuse", [sys.executable, "-m", "postprocessing", "fuse", "--index", str(index),
                     *prediction_args, "--output-dir", str(output / "bank")])
    for name in HEADS:
        execute(name, [sys.executable, "-m", "inference.classifiers", "--model", name,
                       "--bank", str(output / "bank/candidates.json"), "--index", str(index),
                       "--images", str(images), "--checkpoint-root", str(weights),
                       "--manifest", str(catalog_path),
                       "--output-dir", str(output / "heads" / name), "--batch", str(args.classifier_batch),
                       "--workers", str(args.workers), "--device", args.device])
        head_path = output / "heads" / name / "manifest.json"
        manifest["sealed_heads"][name] = {"path": head_path.relative_to(output).as_posix(),
                                           "sha256": sha256(head_path)}
        save()
    if sha256(index) != manifest["index_sha256"] or sha256(catalog_path) != manifest["checkpoint_catalog_sha256"]:
        raise ValueError("Index or checkpoint catalog changed during inference")
    if any(sha256(images / name) != expected for name, expected in manifest["images"].items()):
        raise ValueError("Source image bytes changed during inference")
    command = [sys.executable, "-m", "postprocessing", "submit", "--bank", str(output / "bank/candidates.json"),
               "--index", str(index), "--images", str(images),
               "--checkpoint-manifest", str(catalog_path), "--output-dir", str(output / "final")]
    for name in HEADS:
        command += ["--" + name, str(output / "heads" / name / "manifest.json")]
        command += ["--expected-head", name + "=" + manifest["sealed_heads"][name]["sha256"]]
    if legacy:
        command.append("--legacy")
    execute("submit", command)
    final = json.loads((output / "final/manifest.json").read_text(encoding="utf-8"))
    manifest.update(status="complete", result=final)
    save()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector-batch", type=int, default=1, help="Images per batch for TinyFormer, RF-DETR and YOLO26; Co-DETR always uses 1")
    parser.add_argument("--classifier-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--codetr-command", help="Co-DETR Python command; default: the current Python with scripts/codetr_python.py")
    parser.add_argument("--legacy", action="store_true", help="Explicitly use the historical policy-global top-100 finalization order")
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Inference failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "result": result["result"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
