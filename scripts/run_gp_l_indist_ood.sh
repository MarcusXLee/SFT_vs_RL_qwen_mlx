#!/usr/bin/env bash
# GP-L evaluation suite: Init, SFT and multi-turn GRPO v5, in-distribution and
# rule-OOD, n=200 each. Tolerant verifier (the headline condition), greedy,
# seed 42, verify_iter 5 -- identical to every earlier n=50 evaluation except n.
#
#   in-distribution  face cards J/Q/K count as 10 (the training rule).
#                    n=200 with seed 42 is exactly the stream whose hands SFT and
#                    RL training withheld (eval_hands(42, 200): 184 distinct),
#                    so every hand is unseen. Its first 50 are the earlier n=50
#                    set, which makes the old results a consistency check.
#   rule-OOD         as scripts/gp_evaluation/language_ood_eval.sh upstream:
#                    ood=True (every hand holds a J, Q or K) and
#                    treat_face_cards_as_10=False (they count as 11, 12, 13; the
#                    prompt says so). No model was trained under this rule.
#
#   usage: bash scripts/run_gp_l_indist_ood.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
N=200
declare -a MODELS=("mt-v5=train_ckpt/mt_grpo_v5" "sft=train_ckpt/sft_gp_language" "init=")

run() {  # run <label> <adapter> <suite> <extra overrides...>
    local label=$1 adapter=$2 suite=$3; shift 3
    local out="logs/ablation/${suite}__${label}-n${N}/tolerant/log.jsonl"
    echo "########## ${suite} / ${label} (n=${N}) -> ${out}"
    bash scripts/run_qwen_gp_eval.sh evaluation/configs/qwen_gp_language.yaml \
        --num_traj=${N} --env_config.strip_formula_whitespace=True \
        --adapter_path="${adapter}" --output_dir="${out}" "$@"
}

for m in "${MODELS[@]}"; do
    run "${m%%=*}" "${m#*=}" qwen_gp_language \
        --env_config.ood=False --env_config.treat_face_cards_as_10=True
done
for m in "${MODELS[@]}"; do
    run "${m%%=*}" "${m#*=}" qwen_gp_language_ood \
        --env_config.ood=True --env_config.treat_face_cards_as_10=False
done
echo "########## all six evaluations finished"
