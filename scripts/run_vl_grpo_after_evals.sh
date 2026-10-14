#!/usr/bin/env bash
# Launch the GP-VL multi-turn GRPO run once the GP-VL evaluations are done,
# and only if every precondition holds. Otherwise it stops and says which one
# failed, without launching anything.
#
#   1. the evaluation queue has exited, and both suites wrote their
#      "finished" marker;
#   2. all six GP-VL evaluation logs (Init and SFT x in-dist, rule-OOD,
#      visual-OOD) hold exactly one log with 200 complete trajectories;
#   3. the GPU acceptance checks passed (logs/diagnostics/vl_grpo_verify_mb1.log);
#   4. nothing else holds the GPU. A GP-VL job must run alone
#      (memory: gpu-exclusive-during-vl-jobs).
#
# The run is GP-L v5's recipe with --vision, the GP-VL SFT adapter and
# microbatch 1 (peak 67.6 GB measured). See QWEN_EVAL.md "GP-VL GRPO".
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
OUT=train_ckpt/mt_grpo_vl_v1
A=logs/ablation

echo "[$(date +%H:%M)] waiting for the GP-VL evaluation queue to finish"
while pgrep -f run_gp_vl_eval_suite.sh >/dev/null || pgrep -f evaluation.launcher >/dev/null; do
    sleep 60
done
echo "[$(date +%H:%M)] evaluation queue has exited; checking preconditions"

fail() { echo "NOT LAUNCHED: $*"; exit 1; }
grep -q "VL suite finished: init=" $A/gp_vl_init_suite2.log 2>/dev/null || fail "Init OOD suite did not finish"
grep -q "VL suite finished: sft=train_ckpt/sft_gp_vl" $A/gp_vl_sft_suite.log 2>/dev/null || fail "SFT suite did not finish"
for d in qwen_gp_vl__init qwen_gp_vl_ruleood__init qwen_gp_vl_visualood__init \
         qwen_gp_vl__sft qwen_gp_vl_ruleood__sft qwen_gp_vl_visualood__sft; do
    files=( $A/${d}-n200/tolerant/*.jsonl )
    [ ${#files[@]} -eq 1 ] && [ -f "${files[0]}" ] || fail "$d: expected exactly one log, found ${#files[@]}"
    n=$(grep -c '"Split"' "${files[0]}")
    [ "$n" -eq 200 ] || fail "$d: $n/200 trajectories"
    echo "  ok  $d: 200/200"
done
grep -q "RESULT: PASS" logs/diagnostics/vl_grpo_verify_mb1.log || fail "GPU acceptance checks did not pass"
echo "  ok  acceptance checks passed"
if pgrep -f "train_sft_qwen|train_grpo_qwen|verify_vl_grpo|evaluation.launcher" >/dev/null; then
    fail "another GPU job is running"
fi
[ -e "$OUT/adapters.safetensors" ] && fail "$OUT already holds a finished run"

echo "[$(date +%H:%M)] launching GP-VL GRPO -> $OUT"
mkdir -p "$OUT"
caffeinate -i -m ./.venv-qwen/bin/python -u scripts/train_grpo_qwen.py \
    --vision --multi-turn --adapter train_ckpt/sft_gp_vl --out "$OUT" \
    --updates 60 --group-size 8 --prompts-per-update 4 --temperature 1.0 --max-tokens 128 \
    --lr 1e-5 --kl-coef 0.01 --clip-eps 0.2 --microbatch 1 --batch-rollout \
    --lora-rank 8 --target 24 --strip-formula-whitespace --reward-mode final --max-turns 5 \
    --entropy-coef 0.002 --dynamic-sampling --dynamic-max-draws 0 --save-every 10 \
    --seed 0 --eval-seed 42 --eval-traj 200 2>&1 \
  | grep --line-buffered -v -E "PyTorch was not found|processor_kwargs" | tee "$OUT/train.log"
echo "[$(date +%H:%M)] GP-VL GRPO exited with status ${PIPESTATUS[0]}"
