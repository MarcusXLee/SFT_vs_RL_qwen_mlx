#!/usr/bin/env bash
# GeneralPoints whitespace ablation for a Qwen3-VL checkpoint.
#
# Runs the SAME config twice -- identical seed, prompts, num_traj, model and
# adapter -- changing exactly one thing:
#
#   strict   env_config.strip_formula_whitespace = False  (verifier as shipped:
#            a space inside the formula is an illegal character, reward -3)
#   tolerant env_config.strip_formula_whitespace = True   (whitespace removed
#            before validation, so the arithmetic is judged on its merits)
#
# "tolerant" is the headline condition for this reproduction; "strict" is kept
# and reported so the formatting tax is visible rather than silently absorbed.
# Both arms are driven by the paper's own evaluation loop (multi-turn retries
# with verifier feedback, success scored per trajectory), so only the flag differs.
#
#   usage: run_qwen_gp_ablation.sh <config.yaml> [num_traj] [label] [-- extra overrides]
#
#   # base model
#   bash scripts/run_qwen_gp_ablation.sh evaluation/configs/qwen_gp_language.yaml 50
#   # an SFT checkpoint, results kept under a separate label
#   bash scripts/run_qwen_gp_ablation.sh evaluation/configs/qwen_gp_language.yaml 50 sft \
#        -- --adapter_path=train_ckpt/sft_gp_language
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
config="${1:?usage: run_qwen_gp_ablation.sh <config.yaml> [num_traj] [label] [-- overrides]}"
num_traj="${2:-50}"
label="${3:-init}"
shift $(( $# < 3 ? $# : 3 ))
[ "${1:-}" = "--" ] && shift
extra=( "$@" )

# NOTE: BaseEvaluator timestamps the output path with output_dir.replace('.', ...),
# which substitutes EVERY dot, not just the extension. A '.' in this name would
# be rewritten into the directory. Keep the separator dot-free.
name="$(basename "$config" .yaml)__$label"
out="$root/logs/ablation/$name"

if [ -e "$out" ]; then
  echo "Refusing to overwrite an existing ablation at logs/ablation/$name" >&2
  echo "Move or delete it first, or pass a different label, so old and new results are never mixed." >&2
  exit 1
fi

echo "### $name | $num_traj trajectories | strict vs tolerant ###"
[ ${#extra[@]} -gt 0 ] && echo "### overrides: ${extra[*]} ###"
for arm in strict tolerant; do
  [ "$arm" = tolerant ] && strip=True || strip=False
  echo "--- $arm (strip_formula_whitespace=$strip) ---"
  bash "$root/scripts/run_qwen_gp_eval.sh" "$config" \
    --num_traj="$num_traj" \
    --env_config.strip_formula_whitespace=$strip \
    --output_dir="logs/ablation/$name/$arm/log.jsonl" \
    "${extra[@]+"${extra[@]}"}" 2>&1 \
    | grep -v 'processor_kwargs' | tail -5
done

echo
"$root/.venv-qwen/bin/python" "$root/scripts/analyze_gp_log.py" compare \
  "$(ls -t "$out"/strict/*.jsonl | head -1)" \
  "$(ls -t "$out"/tolerant/*.jsonl | head -1)" 2>&1 | grep -v 'processor_kwargs'
