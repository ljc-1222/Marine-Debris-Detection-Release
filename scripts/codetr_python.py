"""Run Python in the Co-DETR overlay without a shell or scheduler wrapper."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

if __package__:
    from .environment import ROOT, codetr_environment
else:
    from environment import ROOT, codetr_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--runtime", type=Path, default=ROOT / ".venv-codetr")
    parser.add_argument("--cuda-home")
    parser.add_argument("--python", default=sys.executable)
    options, arguments = parser.parse_known_args()
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not arguments:
        raise SystemExit("Usage: python scripts/codetr_python.py [--runtime DIR] -m MODULE [arguments]")
    env = codetr_environment(options.runtime, cuda_home=options.cuda_home)
    os.execve(options.python, [options.python, *arguments], env)


if __name__ == "__main__":
    main()
