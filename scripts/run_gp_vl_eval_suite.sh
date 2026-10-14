#!/usr/bin/env bash
# GP-VL evaluation suite, n=200 per setting, for any list of checkpoints.
# Settings follow upstream scripts/gp_evaluation/vl_*_eval.sh exactly:
#
#   in-distribution  ood=False  J/Q/K = 10        black face cards (training)
#   rule-OOD         ood=True   J/Q/K = 11/12/13  black
#   visual-OOD       ood=False  J/Q/K = 10        red face cards
#
# Tolerant verifier, greedy, seed 42, resolution 1200, verify_iter 5 -- as the
# GP-VL Init n=50 run, whose hands are the first 50 of the in-distribution
# stream here. The in-distribution and visual-OOD streams hold the same 184
# hands (card values do not depend on colour), all withheld from SFT/RL data.
#
#   usage: bash scripts/run_gp_vl_eval_suite.sh LABEL=ADAPTER [LABEL=ADAPTER ...]
#          (ADAPTER empty for the base model, e.g. init=)
#   SETTINGS="ruleood visualood" bash ... runs a subset (default: all three).
#
# GP-VL jobs must run ALONE on the GPU: a VL evaluation next to VL SFT ran the
# 128 GB machine out of Metal memory and killed both (BUGS.md #29).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
N=200
[ $# -ge 1 ] || { echo "usage: $0 LABEL=ADAPTER ..."; exit 2; }

run() {  # run <label> <adapter> <suite> <overrides...>
    local label=$1 adapter=$2 suite=$3; shift 3
    local out="logs/ablation/${suite}__${label}-n${N}/tolerant/log.jsonl"
    echo "########## ${suite} / ${label} (n=${N}) -> ${out}"
    bash scripts/run_qwen_gp_eval.sh evaluation/configs/qwen_gp_vl.yaml \
        --num_traj=${N} --env_config.strip_formula_whitespace=True \
        --adapter_path="${adapter}" --output_dir="${out}" "$@"
}

SETTINGS="${SETTINGS:-indist ruleood visualood}"
for m in "$@"; do
    label="${m%%=*}"; adapter="${m#*=}"
    for s in $SETTINGS; do
        case "$s" in
          indist)    run "$label" "$adapter" qwen_gp_vl \
                         --env_config.ood=False --env_config.treat_face_cards_as_10=True --env_config.face_cards_color=black ;;
          ruleood)   run "$label" "$adapter" qwen_gp_vl_ruleood \
                         --env_config.ood=True --env_config.treat_face_cards_as_10=False --env_config.face_cards_color=black ;;
          visualood) run "$label" "$adapter" qwen_gp_vl_visualood \
                         --env_config.ood=False --env_config.treat_face_cards_as_10=True --env_config.face_cards_color=red ;;
          *) echo "unknown setting: $s"; exit 2 ;;
        esac
    done
done
echo "########## VL suite finished: $*"
