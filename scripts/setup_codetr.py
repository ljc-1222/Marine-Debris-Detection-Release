"""Build the pinned Co-DETR MMCV overlay with the active PyTorch CUDA toolkit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request

if __package__:
    from .environment import ROOT, caches, cuda_toolkit, source_revisions
else:
    from environment import ROOT, caches, cuda_toolkit, source_revisions

MMCV_URL = "https://files.pythonhosted.org/packages/source/m/mmcv-full/mmcv-full-1.5.0.tar.gz"
MMCV_SHA256 = "2de4b86914840afc9ec80f7a988bb94647a63b585b36790d0f424a2f4f7c6749"


def build_environment(env: dict[str, str], cuda_root: Path, overlay: Path, max_jobs: int) -> dict[str, str]:
    if not 1 <= max_jobs <= 16:
        raise ValueError("--max-jobs must be in 1..16")
    env = dict(env)
    env.update(CUDA_HOME=str(cuda_root), MAX_JOBS=str(max_jobs), PYTHONPATH=str(overlay))
    env["PATH"] = os.pathsep.join([str(cuda_root / "bin"), sysconfig.get_path("scripts"), env.get("PATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(cuda_root / "lib64"), str(cuda_root / "lib"), env.get("LD_LIBRARY_PATH", "")])
    return env


def require_ninja(env: dict[str, str]) -> None:
    try:
        version = importlib.metadata.version("ninja")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("The active environment needs ninja==1.13.2; rerun scripts/setup_environment.py") from error
    if version != "1.13.2":
        raise RuntimeError("The MMCV build requires ninja==1.13.2")
    executable = shutil.which("ninja", path=env["PATH"])
    if executable is None:
        raise RuntimeError("The active environment's Ninja executable is missing from PATH")
    subprocess.run([executable, "--version"], env=env, check=True)
    # Torch uses this same PATH check before selecting BuildExtension's backend.
    subprocess.run([sys.executable, "-c", "from torch.utils.cpp_extension import verify_ninja_availability; verify_ninja_availability()"], env=env, check=True)


def patch_source(source: Path) -> None:
    setup = source / "setup.py"
    setup.write_text(setup.read_text().replace("std=c++14", "std=c++17"))
    path = source / "mmcv/parallel/_functions.py"
    text = path.read_text()
    old = "streams = [_get_stream(device) for device in target_gpus]"
    new = "streams = [_get_stream(torch.device('cuda', device) if isinstance(device, int) else device) for device in target_gpus]"
    if text.count(old) != 1:
        raise ValueError("MMCV stream compatibility patch did not match exactly once")
    path.write_text(text.replace(old, new))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / ".venv-codetr")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--cuda-home")
    parser.add_argument("--max-jobs", type=int, default=4)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use the project's Python 3.11 interpreter")
    if not 1 <= args.max_jobs <= 16:
        raise ValueError("--max-jobs must be in 1..16")
    source_revisions(initialize=True)
    import torch

    cuda_root, cuda_version = cuda_toolkit(args.cuda_home)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU allocation is required to build and check MMCV")
    if torch.version.cuda != cuda_version:
        raise RuntimeError(f"PyTorch CUDA {torch.version.cuda} does not match nvcc {cuda_version}; select matching --cuda wheels during setup")
    runtime = args.runtime.expanduser().resolve()
    overlay, mmcv = runtime / "site-packages", runtime / "mmcv"
    overlay.mkdir(parents=True, exist_ok=True)
    env = caches(args.cache_dir)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--target", str(overlay), "--no-deps", "--requirement", str(ROOT / "requirements-codetr.txt")], cwd=ROOT, env=env, check=True)
    source_dir = args.cache_dir.expanduser().resolve() / "sources"
    build_dir = args.cache_dir.expanduser().resolve() / "build"
    source_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    archive = source_dir / "mmcv-full-1.5.0.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(archive.suffix + ".partial")
        with urllib.request.urlopen(MMCV_URL) as response, temporary.open("xb") as target:
            shutil.copyfileobj(response, target)
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != MMCV_SHA256:
        raise ValueError("MMCV source archive SHA256 mismatch")
    if not list((mmcv / "mmcv").glob("_ext*.so")):
        if mmcv.exists():
            raise FileExistsError("An incomplete MMCV runtime already exists; choose a new --runtime directory")
        work = Path(tempfile.mkdtemp(prefix="mmcv-", dir=build_dir))
        try:
            with tarfile.open(archive) as package:
                package.extractall(work, filter="data")
            source = work / "mmcv-full-1.5.0"
            patch_source(source)
            include = sysconfig.get_path("include")
            env.update(MMCV_WITH_OPS="1", FORCE_CUDA="1", TORCH_CUDA_ARCH_LIST="%s.%s" % torch.cuda.get_device_capability())
            env = build_environment(env, cuda_root, overlay, args.max_jobs)
            require_ninja(env)
            print(f"MMCV build backend: Ninja; MAX_JOBS={args.max_jobs}", flush=True)
            for key in ("CPATH", "CPLUS_INCLUDE_PATH"):
                env[key] = include + os.pathsep + env.get(key, "")
            subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=source, env=env, check=True)
            source.rename(mmcv)
        finally:
            shutil.rmtree(work)
    command = [sys.executable, str(ROOT / "scripts/codetr_python.py"), "--runtime", str(runtime), "--cuda-home", str(cuda_root), str(ROOT / "scripts/check_environment.py"), "--mode", "codetr"]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
