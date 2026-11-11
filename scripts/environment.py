"""Shared standard-library setup and Co-DETR process environment helpers."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISIONS = {
    "third_party/Co-DETR": "a3a9d20b7b2e19ad6e140590a6e3d8f2d3c88e9b",
    "third_party/TinyFormer": "9075d9fdf5ce989ca3c7e830a780dfc7fbbe4ec1",
    "third_party/dinov3": "6876159a11b4df116f30f667f8c9888617df0751",
    "third_party/ultralytics": "dad7bb4534c95021bc14969ab25d77b77c4efdc3",
}
CACHE_SUBDIRECTORIES = {
    "TMPDIR": "tmp", "PIP_CACHE_DIR": "pip", "HF_HUB_CACHE": "huggingface/hub",
    "HF_XET_CACHE": "huggingface/xet", "TORCH_HOME": "torch",
    "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "XDG_CACHE_HOME": "xdg", "MPLCONFIGDIR": "matplotlib", "YOLO_CONFIG_DIR": "ultralytics",
}


def caches(directory: Path) -> dict[str, str]:
    directory = Path(directory).expanduser().resolve()
    env = dict(os.environ)
    for key, relative in CACHE_SUBDIRECTORIES.items():
        path = directory / relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
    env.update(PIP_DISABLE_PIP_VERSION_CHECK="1", HF_HUB_DISABLE_TELEMETRY="1")
    return env


def configure_caches(directory: Path = ROOT / ".cache") -> None:
    """Give direct Python entry points project-local defaults, preserving caller paths."""
    defaults = caches(directory)
    for key in (*CACHE_SUBDIRECTORIES, "PIP_DISABLE_PIP_VERSION_CHECK", "TORCHINDUCTOR_COMPILE_THREADS"):
        os.environ.setdefault(key, defaults[key])
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def source_revisions(*, initialize=False) -> dict[str, dict[str, str]]:
    missing = [name for name in SOURCE_REVISIONS if not (ROOT / name).is_dir() or not any((ROOT / name).iterdir())]
    if missing and initialize:
        subprocess.run(["git", "-C", str(ROOT), "submodule", "update", "--init", "--recursive"], check=True)
    result = {}
    for name, expected in SOURCE_REVISIONS.items():
        path = ROOT / name
        if not path.is_dir() or not any(path.iterdir()):
            raise FileNotFoundError(f"Missing pinned upstream source: {name}; initialize Git submodules or extract the complete source archive")
        if (path / ".git").exists():
            actual = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
            if actual != expected:
                raise ValueError(f"{name}: expected upstream revision {expected}, found {actual}")
            result[name] = {"revision": actual, "verification": "git_revision"}
        else:
            result[name] = {"revision": expected, "verification": "archive_source_present_revision_not_independently_checked"}
    return result


def cuda_toolkit(cuda_home: str | None = None) -> tuple[Path, str]:
    root = cuda_home or os.environ.get("CUDA_HOME")
    nvcc = Path(root) / "bin/nvcc" if root else Path(shutil.which("nvcc") or "")
    if not nvcc.is_file():
        raise RuntimeError("CUDA toolkit nvcc was not found; set CUDA_HOME or pass --cuda-home for Co-DETR compilation")
    output = subprocess.run([str(nvcc), "--version"], check=True, capture_output=True, text=True).stdout
    version = re.search(r"release (\d+\.\d+)", output)
    if version is None:
        raise RuntimeError("Could not determine the CUDA toolkit version from nvcc")
    return Path(root).expanduser().resolve() if root else nvcc.resolve().parent.parent, version.group(1)


def cuda_wheel_variant(value: str, cuda_home: str | None = None) -> str:
    if value != "auto":
        if value not in {"cu126", "cu128", "cu130"}:
            raise ValueError("CUDA wheel variant must be auto, cu126, cu128 or cu130")
        return value
    _root, version = cuda_toolkit(cuda_home)
    variants = {"12.6": "cu126", "12.8": "cu128", "13.0": "cu130"}
    if version not in variants:
        raise RuntimeError(f"No default wheel mapping for CUDA toolkit {version}; select --cuda explicitly")
    return variants[version]


def codetr_environment(runtime: Path, *, cuda_home: str | None = None) -> dict[str, str]:
    configure_caches()
    runtime = Path(runtime).expanduser().resolve()
    overlay, mmcv = runtime / "site-packages", runtime / "mmcv"
    if not overlay.is_dir() or not (mmcv / "mmcv/__init__.py").is_file():
        raise FileNotFoundError("Co-DETR runtime is missing; run python scripts/setup_codetr.py first")
    cuda_root, _version = cuda_toolkit(cuda_home)
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_HOME=str(cuda_root), TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    env["PYTHONPATH"] = os.pathsep.join([str(mmcv), str(overlay), str(ROOT / "third_party/Co-DETR"), str(ROOT), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    env["PATH"] = str(cuda_root / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(cuda_root / "lib64"), str(cuda_root / "lib"), env.get("LD_LIBRARY_PATH", "")])
    return env
