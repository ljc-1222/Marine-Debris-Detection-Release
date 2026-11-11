#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
python_exec="$project_root/.venv/bin/python"
if [[ ! -x "$python_exec" ]]; then
    python_exec=$(command -v python3.11)
fi
exec "$python_exec" "$script_dir/codetr_python.py" "$@"
