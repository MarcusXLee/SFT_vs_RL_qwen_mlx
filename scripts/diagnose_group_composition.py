"""Why do GRPO's active groups disappear? Compare checkpoints on identical hands.

A falling active-group count has at least three causes needing different fixes:

  all-correct rises      -> the sampled hands became too easy for the policy
  all-wrong rises        -> capability degraded (or the hands are too hard)
  distinct completions   -> exploration collapsed
  falls

Group categories (per hand, G completions each):
  all-correct       every completion solved it            (no spread, dropped)
  all-wrong         every completion failed, same reward  (no spread, dropped)
  mixed             at least one correct AND one wrong    (the useful signal)
  varied-failure    no correct answer, but rewards differ (spread, but only
                    ranks degrees of wrongness)

Every checkpoint sees the SAME hands in the SAME order with the SAME sampling
settings, so differences cannot be hand difficulty.

    .venv-qwen/bin/python scripts/diagnose_group_composition.py [--hands N] [--group-size G]
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import mlx.core as mx
import numpy as np
from mlx_vlm import batch_generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.sample_utils import make_sampler
from mlx_vlm.trainer.utils import apply_lora_layers

from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG

CHECKPOINTS = [
    ('SFT',     'train_ckpt/sft_gp_language'),
    ('v1@60',   'train_ckpt/grpo_from_sft'),
    ('v2@20',   'train_ckpt/grpo_v2_from_sft/adapters-00020.safetensors'),
    ('v2@40',   'train_ckpt/grpo_v2_from_sft/adapters-00040.safetensors'),
    ('v2@60',   'train_ckpt/grpo_v2_from_sft'),
]


def fixed_hands(n, seed, target=24):
    """The same hands for every checkpoint, drawn once."""
    random.seed(seed)
    env = GeneralPointEnv_oneline(language_only=True, resolution=64, target_points=target)
    out = []
    for _ in range(n):
        _, info = env.reset()
        out.append({'cards': list(info['Plain Cards']), 'numbers': list(info['Numbers']),
                    'solution': list(info['Solution'])})
    return out


def load_checkpoint(path):
    """Accept a directory, or a specific .safetensors inside one."""
    import shutil, tempfile
    p = Path(path)
    if p.is_file():
        tmp = Path(tempfile.mkdtemp())
        shutil.copy2(p.parent / 'adapter_config.json', tmp / 'adapter_config.json')
        shutil.copy2(p, tmp / 'adapters.safetensors')
        return str(tmp)
    return str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hands', type=int, default=20)
    ap.add_argument('--group-size', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--out', type=Path, default=ROOT / 'logs/group_composition.json')
    ap.add_argument('--checkpoints', nargs='*', default=None,
                    help='LABEL=PATH pairs; defaults to the built-in list')
    args = ap.parse_args()

    checkpoints = CHECKPOINTS
    if args.checkpoints:
        checkpoints = [tuple(c.split('=', 1)) for c in args.checkpoints]

    hands = fixed_hands(args.hands, args.seed)
    print(f'{args.hands} fixed hands, G={args.group_size}, temp={args.temperature}, '
          f'seed={args.seed}\n')

    env = GeneralPointEnv_oneline(language_only=True, resolution=64,
                                  strip_formula_whitespace=True)
    env.reset()
    results = {}

    for label, path in checkpoints:
        if not Path(path).exists():
            print(f'{label}: missing ({path}) - skipped')
            continue
        model, processor = load(str(ROOT / 'models/Qwen3-VL-8B-Instruct'))
        model = apply_lora_layers(model, load_checkpoint(path))
        model.eval()

        cats, distinct, winrate = Counter(), [], []
        # Per-hand records, so checkpoints can be compared with a PAIRED test.
        # Saving only counts forced an unpaired Fisher on the A/B comparison,
        # which is weaker than McNemar on the same hands.
        per_hand = []
        for hand in hands:
            q = Q_GeneralPoint_EQN_L_EG.format(
                cards=hand['cards'], target_number='24',
                face_card_msg="'J', 'Q', and 'K' count as '10'.")
            text = apply_chat_template(
                processor, model.config,
                [{"role": "user", "content": [{"type": "text", "text": q}]}], num_images=0)
            # every checkpoint samples from the same RNG state for this hand
            mx.random.seed(args.seed)
            outs = list(batch_generate(
                model, processor, images=None, prompts=[text] * args.group_size,
                max_tokens=args.max_tokens,
                sampler=make_sampler(temp=args.temperature), verbose=False).texts)

            env.cards_num = list(hand['numbers'])
            env.cards_without_suit = list(hand['cards'])
            env.solution = list(hand['solution'])
            rewards = []
            for t in outs:
                env.remaining_step, env.verify_info = env.verify_iter, None
                rewards.append(env.step(t)[1])

            n_correct = sum(1 for r in rewards if r == 5)
            if n_correct == len(rewards):
                cats['all-correct'] += 1
            elif n_correct == 0:
                cats['varied-failure' if len(set(rewards)) > 1 else 'all-wrong'] += 1
            else:
                cats['mixed'] += 1
            distinct.append(len(set(outs)))
            winrate.append(n_correct / len(rewards))
            per_hand.append({'cards': hand['cards'],
                             'category': ('all-correct' if n_correct == len(rewards)
                                          else 'mixed' if n_correct
                                          else 'varied-failure' if len(set(rewards)) > 1
                                          else 'all-wrong'),
                             'n_correct': n_correct,
                             'distinct': len(set(outs)),
                             'rewards': [float(r) for r in rewards]})
            mx.clear_cache()

        results[label] = {'categories': dict(cats),
                          'mean_distinct': float(np.mean(distinct)),
                          'win_rate': float(np.mean(winrate)),
                          'per_hand': per_hand}
        del model
        mx.clear_cache()

        c = cats
        print(f'{label:<7} all-correct {c["all-correct"]:>2}  all-wrong {c["all-wrong"]:>2}  '
              f'varied-failure {c["varied-failure"]:>2}  mixed {c["mixed"]:>2}   |  '
              f'distinct {np.mean(distinct):.1f}/{args.group_size}  win {np.mean(winrate):.1%}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {'settings': vars(args) | {'out': str(args.out)},
         'hands': [h['cards'] for h in hands], 'results': results}, indent=2, default=str) + '\n')
    print(f'\nwrote {args.out}')

    print('\ninterpretation:')
    print('  all-correct up   -> hands too easy for the policy')
    print('  all-wrong up     -> capability degraded')
    print('  distinct down    -> exploration collapsed')
    print('  mixed is the only category GRPO learns a real success signal from')


if __name__ == '__main__':
    main()
