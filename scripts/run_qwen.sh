#!/usr/bin/env bash
set -euo pipefail
qwen_project="$(cd "$(dirname "$0")/.." && pwd)"
qwen_python="$qwen_project/.venv-qwen/bin/python"
qwen_model="$qwen_project/models/Qwen3-VL-8B-Instruct"
if [[ ! -x "$qwen_python" ]]; then
  echo 'The .venv-qwen Python environment is missing.' >&2
  exit 1
fi
if [[ ! -f "$qwen_model/model.safetensors.index.json" ]]; then
  echo "The model download is missing: $qwen_model" >&2
  exit 1
fi
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="$qwen_project/.cache"
if [[ $# -eq 0 ]]; then
  set -- --prompt 'Briefly introduce yourself in English.'
fi
exec "$qwen_python" -m mlx_vlm.generate \
  --model "$qwen_model" --max-tokens 128 --temperature 0.0 "$@"
