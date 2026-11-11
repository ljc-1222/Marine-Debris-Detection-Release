#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
exec python3.11 "$script_dir/setup_environment.py" --venv "$project_root/.venv" --codetr --max-jobs "${MAX_JOBS:-4}" "$@"
