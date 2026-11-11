"""Download trained models from Hugging Face."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "configs/four_detector_three_head_checkpoints.json"


def relative_path(value: str) -> Path:
    if not isinstance(value, str):
        raise ValueError("Checkpoint filenames must be strings")
    path = PurePosixPath(value)
    if not value or not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Checkpoint filenames must be safe relative paths")
    return Path(*path.parts)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or not manifest.get("artifacts"):
        raise ValueError("Expected a version-2 checkpoint manifest")
    paths = set()
    for key, artifact in manifest["artifacts"].items():
        name = artifact.get("display_name", key)
        if not re.fullmatch(r"[a-z0-9_]+", key):
            raise ValueError("Invalid checkpoint identifier")
        relative = relative_path(artifact["path"])
        if len(relative.parts) < 2 or relative.parts[0] != "checkpoints" or relative in paths:
            raise ValueError("Checkpoint paths must be distinct and inside checkpoints/")
        paths.add(relative)
        if not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
            raise ValueError(f"{name}: a complete SHA256 is required")
        if type(artifact["bytes"]) is not int or artifact["bytes"] <= 0:
            raise ValueError(f"{name}: a positive byte count is required")
        if hf := artifact.get("hf"):
            if not re.fullmatch(r"[0-9a-f]{40}", hf["revision"]):
                raise ValueError(f"{name}: HF revision must be a full commit, not a branch")
            relative_path(hf["filename"])
    for group, keys in manifest.get("groups", {}).items():
        if not keys or any(key not in manifest["artifacts"] for key in keys):
            raise ValueError(f"{group}: unknown or empty checkpoint group")
    return manifest


def file_information(path: Path, artifact: dict) -> dict:
    name = artifact.get("display_name", path.name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {name} ({path})")
    size = path.stat().st_size
    if size != artifact["bytes"]:
        raise ValueError(f"{name}: expected {artifact['bytes']} bytes, found {size}")
    result = {"path": artifact["path"], "bytes": size, "status": "ready"}
    if "display_name" in artifact:
        result["display_name"] = artifact["display_name"]
    return result


def verify_file(path: Path, artifact: dict) -> dict:
    result = file_information(path, artifact)
    name = artifact.get("display_name", path.name)
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            checksum.update(block)
    actual = checksum.hexdigest()
    if actual != artifact["sha256"]:
        raise ValueError(f"{name}: SHA256 mismatch")
    return {**result, "sha256": actual, "status": "verified"}


def destination(root: Path, artifact: dict, *, canonical: bool = False) -> Path:
    """Find a local checkpoint; writers can request the exact catalog path."""
    relative = relative_path(artifact["path"])
    name = artifact.get("display_name", str(relative))
    path = root / relative
    resolved_root = root.resolve()

    def contained(candidate: Path) -> Path:
        if not candidate.resolve().is_relative_to(resolved_root):
            raise ValueError(f"{name}: Checkpoint path escapes the output directory through a symlink")
        return candidate

    contained(path)
    if (canonical or path.exists() or path.is_symlink()
            or relative.parts[:-1] not in (("checkpoints", "detectors"), ("checkpoints", "classifiers"))
            or path.suffix not in (".pt", ".pth")):
        return path
    digest = artifact.get("sha256", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        return path
    # Earlier packages used an underscore before the same digest prefix.
    # Callers still verify the complete digest and byte count before use.
    stem = path.stem.removesuffix(f"-{digest[:8]}")
    previous = path.with_name(f"{stem}-{digest[:8]}{path.suffix}")
    if previous != path and (previous.exists() or previous.is_symlink()):
        return contained(previous)
    earlier = tuple(path.parent.glob(f"*_{digest[:8]}{path.suffix}"))
    if len(earlier) > 1:
        raise ValueError(f"{name}: Multiple earlier filenames match {relative}; put the verified checkpoint at the catalog path")
    return contained(earlier[0]) if earlier else path


def verify_artifact(path: Path, key: str, artifact: dict, *, contract: bool = False, checksum: bool = True) -> dict:
    result = verify_file(path, artifact) if checksum else file_information(path, artifact)
    if contract:
        name = artifact.get("display_name", key)
        try:
            from training.checkpoint_contracts import inspect_checkpoint
        except ImportError as error:
            raise ValueError(f"{name}: --contract requires the project's PyTorch dependencies") from error
        try:
            result["contract"] = inspect_checkpoint(path, key)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
    return result


def download_file(root: Path, key: str, artifact: dict, *, local_files_only: bool = False, contract: bool = False) -> dict:
    name = artifact.get("display_name", key)
    existing = destination(root, artifact)
    if existing.exists() or existing.is_symlink():
        # Never replace unrelated/corrupt data, and report the file that was verified.
        return {**verify_artifact(existing, key, artifact, contract=contract, checksum=False), "path": existing.relative_to(root).as_posix()}
    target = destination(root, artifact, canonical=True)
    hf = artifact.get("hf")
    if not hf:
        raise ValueError(f"{name}: the trained release has no published HF mapping yet; see README.md for release availability. Upstream pretraining weights cannot substitute for this trained checkpoint")
    cache = root / ".cache/checkpoint-downloads" / key
    if not cache.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{name}: Download staging path escapes the output directory")
    from huggingface_hub import hf_hub_download
    try:
        fetched = Path(hf_hub_download(repo_id=hf["repo_id"], filename=hf["filename"],
                                      revision=hf["revision"], repo_type=hf.get("repo_type", "model"),
                                      local_dir=cache, local_files_only=local_files_only))
    except Exception as exc:
        raise RuntimeError(f"{name}: HF download failed ({type(exc).__name__}). Use hf auth login and obtain repository access; review THIRD_PARTY_NOTICES.md.") from None
    result = verify_artifact(fetched, key, artifact, contract=contract, checksum=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(fetched, target)  # Exclusive publication of verified bytes on the same filesystem.
    except FileExistsError:
        result = verify_artifact(target, key, artifact, contract=contract, checksum=False)
    fetched.unlink()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "download", "verify"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--directory", type=Path, default=Path("."), help="Root containing the manifest's checkpoints/ paths")
    parser.add_argument("--group", default="final", help="Manifest group, or all (default: final)")
    parser.add_argument("--artifact", action="append", help="Select one artifact ID; repeat to select several, overriding --group")
    parser.add_argument("--local-files-only", action="store_true", help="Do not contact HF; use an existing download staging directory")
    parser.add_argument("--contract", action="store_true", help="After byte verification, inspect restricted CPU checkpoint metadata and required shapes; imports PyTorch, never builds models")
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
        selected = args.artifact or (list(manifest["artifacts"]) if args.group == "all" else manifest["groups"].get(args.group))
        if not selected or any(key not in manifest["artifacts"] for key in selected):
            raise ValueError("Unknown checkpoint group or artifact; use list to inspect the manifest")
        selected = list(dict.fromkeys(selected))
        if args.action == "download":
            unavailable = [
                manifest["artifacts"][key].get("display_name", key) for key in selected
                if not manifest["artifacts"][key].get("hf")
                and not destination(args.directory, manifest["artifacts"][key]).exists()
            ]
            if unavailable:
                raise ValueError("No verified HF mirror for: " + ", ".join(unavailable) + ". See README.md for weight availability.")
        results = {}
        for key in selected:
            artifact = manifest["artifacts"][key]
            if args.action == "list":
                results[key] = {"display_name": artifact.get("display_name", key), "path": artifact["path"]}
            elif args.action == "verify":
                path = destination(args.directory, artifact)
                results[key] = {**verify_artifact(path, key, artifact, contract=args.contract),
                                "path": path.relative_to(args.directory).as_posix()}
            else:
                results[key] = download_file(args.directory, key, artifact, local_files_only=args.local_files_only, contract=args.contract)
        print(json.dumps({"action": args.action, "artifacts": results}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"Checkpoint check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
