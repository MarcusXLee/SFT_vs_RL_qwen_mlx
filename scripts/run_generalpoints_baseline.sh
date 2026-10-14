#!/usr/bin/env bash
set -euo pipefail
baseline_root="$(cd "$(dirname "$0")/.." && pwd)"
baseline_output="${1:-$baseline_root/output/qwen-generalpoints-seed42}"
baseline_count="${2:-50}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$baseline_root/.cache"
"$baseline_root/.venv/bin/python" "$baseline_root/scripts/baseline/data.py" prepare --output "$baseline_output" --count "$baseline_count"
"$baseline_root/.venv-qwen/bin/python" "$baseline_root/scripts/baseline/infer.py" --output "$baseline_output"
"$baseline_root/.venv/bin/python" "$baseline_root/scripts/baseline/data.py" score --output "$baseline_output"
echo "Results: $baseline_output/summary.json"
