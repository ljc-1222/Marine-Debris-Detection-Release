"""Create the pinned project environment using the selected CUDA wheel build."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import venv

if __package__:
    from .environment import ROOT, caches, cuda_wheel_variant, source_revisions
else:
    from environment import ROOT, caches, cuda_wheel_variant, source_revisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--cuda", choices=("auto", "cu126", "cu128", "cu130"), default="auto")
    parser.add_argument("--cuda-home")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--codetr", action="store_true", help="Also build the isolated Co-DETR overlay")
    parser.add_argument("--codetr-runtime", type=Path, default=ROOT / ".venv-codetr")
    parser.add_argument("--max-jobs", type=int, default=4)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Run setup with Python 3.11")
    if not 1 <= args.max_jobs <= 16:
        raise ValueError("--max-jobs must be in 1..16")
    variant = cuda_wheel_variant(args.cuda, args.cuda_home)
    env = caches(args.cache_dir)
    import os
    os.environ.update(env)
    source_revisions(initialize=True)
    target = args.venv.expanduser().resolve()
    if target.exists() and not (target / "pyvenv.cfg").is_file() and any(target.iterdir()):
        raise FileExistsError("--venv must be a new directory or an existing virtual environment")
    venv.EnvBuilder(with_pip=True).create(target)
    python = target / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    commands = [
        [str(python), "-m", "pip", "install", "pip==26.2.1", "wheel==0.48.0", "setuptools==84.0.0"],
        [str(python), "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/" + variant, "torch==2.13.0", "torchvision==0.28.0"],
        [str(python), "-m", "pip", "install", "--requirement", str(ROOT / "requirements.txt")],
        [str(python), "-m", "pip", "install", "--no-deps", "--editable", str(ROOT / "third_party/ultralytics")],
        [str(python), "-m", "pip", "check"],
        [str(python), "-c", "import torch; expected='" + variant[2:-1] + "." + variant[-1] + "'; assert torch.version.cuda == expected, (torch.__version__, torch.version.cuda, expected); print('Pinned CUDA wheel:', torch.__version__, torch.version.cuda)"],
    ]
    for command in commands:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.codetr:
        command = [str(python), str(ROOT / "scripts/setup_codetr.py"), "--runtime", str(args.codetr_runtime), "--cache-dir", str(args.cache_dir), "--max-jobs", str(args.max_jobs)]
        if args.cuda_home:
            command += ["--cuda-home", args.cuda_home]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    print(f"Environment ready. Use {python} for the project commands.", flush=True)


if __name__ == "__main__":
    main()
