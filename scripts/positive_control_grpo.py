"""Can the GRPO loop learn ANYTHING over multiple updates?

Six training runs have produced no gains, and I have been inferring "the task is
hard" from that without ever testing the alternative: that the loop does not
learn. A single-step test (log-probs move the right way) is not the same claim.

This swaps in a trivially learnable objective -- reward short completions -- and
checks whether the behaviour actually changes over updates. Length is directly
controllable by the policy and independent of arithmetic ability, so a failure
here is a loop failure, not a task failure.

    .venv-qwen/bin/python scripts/positive_control_grpo.py [--updates 10]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.trainer.utils import apply_lora_layers

from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG
from rl.trainer.grpo_trainer_mlx import GRPOConfig, GRPOTrainer
from scripts.make_sft_data import eval_hands


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adapter', default='train_ckpt/sft_gp_language')
    ap.add_argument('--updates', type=int, default=10)
    ap.add_argument('--target-tokens', type=int, default=45,
                    help='reporting threshold only; the reward itself is continuous')
    ap.add_argument('--group-size', type=int, default=8)
    ap.add_argument('--prompts-per-update', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--out', type=Path, default=ROOT / 'logs/positive_control.json')
    args = ap.parse_args()

    model, processor = load(str(ROOT / 'models/Qwen3-VL-8B-Instruct'))
    model = apply_lora_layers(model, args.adapter)
    tok = processor.tokenizer

    env = GeneralPointEnv_oneline(language_only=True, resolution=64,
                                  strip_formula_whitespace=True)
    env.reset()

    seen = {'lengths': []}

    def reward_fn(text, info):
        """Nothing to do with solving the puzzle -- purely output length.

        Continuous, not a threshold. A threshold at 40 tokens produced zero
        reward variance (natural output is ~47.5 tokens, so every completion
        scored the same), every group was discarded, and no gradient step was
        ever taken -- the control tested nothing. A continuous reward has
        variance whenever lengths differ at all, which they always do.
        """
        n = len(tok.encode(text))
        seen['lengths'].append(n)
        return float(-n)          # shorter is better; advantages are in tokens

    cfg = GRPOConfig(group_size=args.group_size, prompts_per_update=args.prompts_per_update,
                     microbatch=4, temperature=1.0, max_tokens=256, learning_rate=args.lr,
                     kl_coef=0.01, entropy_coef=0.0, seed=0,
                     dynamic_sampling=True, batch_rollout=True)
    trainer = GRPOTrainer(
        model=model, processor=processor, env=env,
        prompt_template=Q_GeneralPoint_EQN_L_EG,
        oracle_arguments={'target_number': '24',
                          'face_card_msg': "'J', 'Q', and 'K' count as '10'."},
        reward_fn=reward_fn, config=cfg,
        holdout=eval_hands(42, 200, target_points=24, treat_face_cards_as_10=True))

    print(f'objective: reward = -(token count); shorter is better '
          f'(reporting threshold {args.target_tokens})')
    print(f'{"step":>4} {"mean len":>9} {"< target":>9} {"reward":>8} {"kept":>5} {"|g|":>7}')
    history = []
    for step in range(args.updates):
        seen['lengths'] = []
        prompts, completions, rewards, _ = trainer.collect()
        lens = list(seen['lengths'])
        metrics = trainer.update(prompts, completions, rewards)
        short = float(np.mean([l < args.target_tokens for l in lens])) if lens else float('nan')
        row = {'step': step, 'mean_len': float(np.mean(lens)) if lens else None,
               'frac_short': short, 'reward_mean': metrics.get('reward_mean'),
               # groups_used comes from update(); group_rewards is added later by
               # train(), which this script bypasses. Reading the latter here
               # silently gave 0 on every update.
               'kept': metrics.get('groups_used', 0),
               'grad_norm': metrics.get('grad_norm')}
        history.append(row)
        print(f'{step:>4} {row["mean_len"]:>9.1f} {short:>8.1%} '
              f'{(row["reward_mean"] if row["reward_mean"] is not None else float("nan")):>+8.2f} '
              f'{row["kept"]:>5} {(row["grad_norm"] or 0):>7.3f}', flush=True)

    first = [h['mean_len'] for h in history[:3] if h['mean_len']]
    last = [h['mean_len'] for h in history[-3:] if h['mean_len']]
    fs = [h['frac_short'] for h in history[:3]]
    ls = [h['frac_short'] for h in history[-3:]]
    print(f'\nmean completion length : {np.mean(first):.1f} -> {np.mean(last):.1f} '
          f'({np.mean(last) - np.mean(first):+.1f} tokens)')
    print(f'fraction under target  : {np.mean(fs):.1%} -> {np.mean(ls):.1%} '
          f'({np.mean(ls) - np.mean(fs):+.1%})')
    # continuous reward: judge on mean length, which is what is optimised
    learned = np.mean(last) < np.mean(first) - 1.0
    steps_with_grad = sum(1 for h in history if (h['grad_norm'] or 0) > 0)
    print(f'updates that actually took a gradient step: {steps_with_grad}/{len(history)}')
    if steps_with_grad == 0:
        print('\nINVALID: no gradient step was taken, so this tests nothing about the loop')
    else:
        print(f'\n{"PASS" if learned else "FAIL"}: the loop '
              f'{"moved behaviour toward the objective" if learned else "did NOT move behaviour"}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'settings': {k: str(v) for k, v in vars(args).items()},
                                    'history': history}, indent=2) + '\n')
    print(f'wrote {args.out}')
    return 0 if learned else 1


if __name__ == '__main__':
    raise SystemExit(main())
