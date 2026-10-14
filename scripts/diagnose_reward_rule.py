"""Reward-design diagnostic: score the SAME trajectories under `final` and `sum`.

Measures signal availability at one checkpoint. It does NOT predict how either
rule behaves once training moves the policy.

  final : 1 if the trajectory ever succeeded, else 0
  sum   : the sum of the actual environment rewards across its attempts

Trajectories are real multi-turn rollouts -- up to `--max-turns` attempts with
verifier feedback, stopping immediately on success -- so the two rules are
compared on identical data. Per-puzzle and per-trajectory records are saved.

    .venv-qwen/bin/python scripts/diagnose_reward_rule.py [--puzzles N]
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
from rl.trainer.grpo_multiturn_mlx import MultiTurnConfig, MultiTurnGRPOTrainer
from rl.trainer.grpo_trainer_mlx import group_advantages
from scripts.make_sft_data import eval_hands

SUCCESS = 5.0
EPS = 1e-6


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adapter', default='train_ckpt/sft_gp_language')
    ap.add_argument('--puzzles', type=int, default=12)
    ap.add_argument('--group-size', type=int, default=8)
    ap.add_argument('--max-turns', type=int, default=5)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=ROOT / 'logs/reward_rule_diagnostic.json')
    args = ap.parse_args()

    model, processor = load(str(ROOT / 'models/Qwen3-VL-8B-Instruct'))
    model = apply_lora_layers(model, args.adapter)
    model.eval()

    env = GeneralPointEnv_oneline(language_only=True, resolution=64,
                                  verify_iter=args.max_turns,
                                  strip_formula_whitespace=True)
    env.reset()
    holdout = eval_hands(42, 200, target_points=24, treat_face_cards_as_10=True)
    print(f'withholding {len(holdout)} evaluation hands; puzzles are training-eligible')

    cfg = MultiTurnConfig(group_size=args.group_size, prompts_per_update=1,
                          temperature=args.temperature, max_tokens=args.max_tokens,
                          max_turns=args.max_turns, seed=args.seed,
                          dynamic_sampling=False)
    trainer = MultiTurnGRPOTrainer(
        model=model, processor=processor, env=env,
        prompt_template=Q_GeneralPoint_EQN_L_EG,
        oracle_arguments={'target_number': '24',
                          'face_card_msg': "'J', 'Q', and 'K' count as '10'."},
        reward_fn=lambda t, i: 0.0, config=cfg, holdout=holdout)

    puzzles = []
    for p in range(args.puzzles):
        info = trainer._reset_to_training_hand()
        group = trainer._rollout_group(info)
        trajs = []
        for t in group:
            solved = any(r == SUCCESS for r in t.rewards)
            trajs.append({
                'rewards': t.rewards,
                'attempts': t.turns,
                'solved': bool(solved),
                'solved_on_attempt': (t.rewards.index(SUCCESS) + 1) if solved else None,
                'sum': float(sum(t.rewards)),
                'final': 1.0 if solved else 0.0,
                'answer': t.last_text[:300],
            })
        puzzles.append({'cards': info['Plain Cards'], 'trajectories': trajs})
        s = sum(x['solved'] for x in trajs)
        print(f"  puzzle {p:>2} {str(info['Plain Cards']):<22} solved {s}/{args.group_size}  "
              f"sum {[round(x['sum'],1) for x in trajs]}", flush=True)
        mx.clear_cache()

    report(puzzles, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'settings': {k: str(v) for k, v in vars(args).items()},
                                    'puzzles': puzzles}, indent=2) + '\n')
    print(f'\nwrote {args.out}')


def report(puzzles, args):
    G = args.group_size
    all_tr = [t for p in puzzles for t in p['trajectories']]
    n = len(all_tr)

    def active(vals):
        return float(np.std(vals)) > EPS

    act_sum = [active([t['sum'] for t in p['trajectories']]) for p in puzzles]
    act_fin = [active([t['final'] for t in p['trajectories']]) for p in puzzles]

    print(f'\n{"="*72}\n1. ACTIVE-GROUP RATE ({len(puzzles)} puzzles, G={G})')
    print(f'   sum   : {sum(act_sum)}/{len(puzzles)} = {sum(act_sum)/len(puzzles):.0%}')
    print(f'   final : {sum(act_fin)}/{len(puzzles)} = {sum(act_fin)/len(puzzles):.0%}')

    print('\n2. GROUP OUTCOME COUNTS')
    allsucc = sum(1 for p in puzzles if all(t['solved'] for t in p['trajectories']))
    allfail = sum(1 for p in puzzles if not any(t['solved'] for t in p['trajectories']))
    mixed = len(puzzles) - allsucc - allfail
    print(f'   all succeed {allsucc}   all fail {allfail}   mixed {mixed}')
    print('   (final is active exactly on the mixed groups, by construction)')

    print('\n3. GROUPS ACTIVE UNDER sum BUT NOT final')
    only_sum = [i for i, (s, f) in enumerate(zip(act_sum, act_fin)) if s and not f]
    print(f'   {len(only_sum)} of {len(puzzles)}')
    for i in only_sum:
        tr = puzzles[i]['trajectories']
        solved = [t for t in tr if t['solved']]
        failed = [t for t in tr if not t['solved']]
        src = []
        if solved and len({t['sum'] for t in solved}) > 1:
            src.append(f'successes differing in attempts {sorted({t["attempts"] for t in solved})}')
        if failed and len({t['sum'] for t in failed}) > 1:
            src.append('failures differing in intermediate penalties')
        print(f'     puzzle {i} ({puzzles[i]["cards"]}): all-{"succeed" if solved else "fail"}; '
              f'variation from {" and ".join(src) if src else "none"}')

    print('\n4. DOES sum EVER RANK A FAILURE >= A SUCCESS? (within a group)')
    viol = 0
    for i, p in enumerate(puzzles):
        tr = p['trajectories']
        sm = [t for t in tr if t['solved']]
        fm = [t for t in tr if not t['solved']]
        for f in fm:
            for s in sm:
                if f['sum'] >= s['sum']:
                    viol += 1
                    if viol <= 5:
                        print(f'     puzzle {i}: failure sum {f["sum"]:+.1f} >= success sum '
                              f'{s["sum"]:+.1f} (success took {s["attempts"]} attempts)')
    print(f'   {viol} violating pairs' + ('  -> sum does NOT guarantee success outranks failure'
                                          if viol else '  -> success always outranked failure here'))

    print('\n5. TRAJECTORY STATISTICS (all trajectories, before any filtering)')
    first = sum(1 for t in all_tr if t['solved_on_attempt'] == 1)
    later = sum(1 for t in all_tr if t['solved'] and t['solved_on_attempt'] != 1)
    print(f'   first-attempt success     : {first}/{n} = {first/n:.1%}')
    print(f'   success after feedback    : {later}/{n} = {later/n:.1%}')
    print(f'   final success             : {first+later}/{n} = {(first+later)/n:.1%}')
    print(f'   mean attempts used        : {np.mean([t["attempts"] for t in all_tr]):.2f}'
          f' of {args.max_turns}')

    print('\n6. ADVANTAGE SCALE (current default: normalize_advantage_std=False)')
    for name in ('sum', 'final'):
        rows = np.array([[t[name] for t in p['trajectories']] for p in puzzles], np.float32)
        adv = group_advantages(rows)
        act = np.array([active(r) for r in rows])
        live = np.abs(adv[act]).max(axis=1) if act.any() else np.array([0.0])
        print(f'   {name:<6} max|adv| over active groups: mean {live.mean():.2f}  '
              f'range [{live.min():.2f}, {live.max():.2f}]')
    print('='*72)


if __name__ == '__main__':
    main()
