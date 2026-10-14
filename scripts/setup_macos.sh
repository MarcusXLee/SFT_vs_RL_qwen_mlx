#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This setup targets Apple silicon macOS.' >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv .venv
  else
    shopt -s nullglob
    local_pythons=(.python/cpython-3.11-*/bin/python3.11)
    if (( ${#local_pythons[@]} == 0 )); then
      echo 'Install Python 3.11 before running this setup.' >&2
      exit 1
    fi
    "${local_pythons[0]}" -m venv .venv
  fi
fi
.venv/bin/python -m pip install -r requirements-macos.lock.txt
.venv/bin/python -m pip install -e gym
.venv/bin/python - <<'PY'
from pathlib import Path
import sysconfig
root = Path.cwd()
(Path(sysconfig.get_paths()['purelib']) / 'sftvsrl-source.pth').write_text(str(root)+'\n'+str(root/'sft/src')+'\n')
PY
.venv/bin/python scripts/check_environment.py
