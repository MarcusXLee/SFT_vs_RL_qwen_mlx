#!/usr/bin/env bash
set -euo pipefail
baseline_root="$(cd "$(dirname "$0")/.." && pwd)"
baseline_source="$baseline_root/output/qwen-generalpoints-seed42"
baseline_output="$baseline_root/output/qwen-generalpoints-seed42-v2"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$baseline_root/.cache"
"$baseline_root/.venv/bin/python" "$baseline_root/scripts/baseline/data.py" revise --source "$baseline_source" --output "$baseline_output"
"$baseline_root/.venv-qwen/bin/python" "$baseline_root/scripts/baseline/infer.py" --output "$baseline_output"
"$baseline_root/.venv/bin/python" "$baseline_root/scripts/baseline/data.py" score --output "$baseline_output"
echo "Revised baseline results: $baseline_output/summary.json"
