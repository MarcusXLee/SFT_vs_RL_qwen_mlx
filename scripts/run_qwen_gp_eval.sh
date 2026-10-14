#!/usr/bin/env bash
# Evaluate the local Qwen3-VL checkpoint on GeneralPoints using the paper's own
# evaluation loop (verify_iter retries + verifier feedback, per-trajectory success).
#   usage: bash scripts/run_qwen_gp_eval.sh <config.yaml> [--num_traj=N ...]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
config="${1:?usage: run_qwen_gp_eval.sh <config.yaml> [overrides...]}"
shift
export PYTHONPATH="$root:$root/gym${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="$root/.cache"
cd "$root"
exec ./.venv-qwen/bin/python -m evaluation.launcher -f "$config" "$@"
