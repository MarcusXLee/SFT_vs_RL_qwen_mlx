"""Summarise and compare GeneralPoints trajectory logs from evaluation/launcher.py.

    analyze_gp_log.py report  <log>                 one log, full breakdown
    analyze_gp_log.py compare <strict> <tolerant>   the whitespace ablation, side by side

Reports the paper's headline metrics (per-trajectory success, per-step reward
breakdown, card-recognition accuracy) plus a diagnostic split of *why* steps
failed: rejected on formatting vs. genuinely wrong arithmetic.

`compare` refuses to print a table unless both runs saw the identical puzzle
stream, so a difference between them can only come from the variable under test.
"""
import argparse
import ast
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]

from utils_general import re_match  # noqa: E402
from utils_rl import REWARD_FN  # noqa: E402

REWARD_NAME = {v: k for k, v in REWARD_FN.items()}


def arithmetic_ok(formula, target):
    """True if the formula, ignoring whitespace, evaluates to the target.

    Deliberately more permissive than the environment's verifier: that is the
    point. The gap between this and the env's verdict is the formatting tax.
    """
    if not isinstance(formula, str):
        return False
    expr = ''.join(formula.split('=')[0].split())
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return False
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.USub,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Constant)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return False
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return False
    try:
        return abs(eval(compile(tree, '<f>', 'eval')) - target) < 1e-6
    except Exception:
        return False


def load_steps(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    steps = [r for r in rows if isinstance(r, dict) and 'veri_step' in r]
    if not steps:
        raise SystemExit(f'No trajectory steps found in {path}')
    return steps


def puzzle_stream(steps):
    """The ordered list of card hands the run was given, one per trajectory."""
    first = {}
    for s in steps:
        first.setdefault(s['sample_id'], tuple(s['info']['Plain Cards']))
    return [first[k] for k in sorted(first)]


def analyse(steps, target):
    trajectories, rewards = {}, Counter()
    causes = Counter()
    for step in steps:
        traj = trajectories.setdefault(step['sample_id'], {'success': False, 'steps': 0})
        traj['steps'] += 1
        reward = step['reward']
        rewards[REWARD_NAME.get(reward, f'OTHER({reward})')] += 1
        if reward == REWARD_FN['CORRECT_SOLUTION']:
            traj['success'] = True
            continue
        if reward == REWARD_FN['INCORRECT_VISION']:
            causes['card misrecognition'] += 1
            continue
        formula = re_match(step['output'], 'formula')
        if not isinstance(formula, str) or formula == 'None':
            causes['no formula parsed'] += 1
        elif arithmetic_ok(formula, target):
            causes['formatting only (maths OK)'] += 1
        else:
            causes['wrong arithmetic'] += 1

    n_traj, n_step = len(trajectories), len(steps)
    n_success = sum(t['success'] for t in trajectories.values())
    first_try = sum(1 for s in steps
                    if s['veri_step'] == 0 and s['reward'] == REWARD_FN['CORRECT_SOLUTION'])
    # The env charges INCORRECT_VISION before it looks at the formula at all.
    vision_errors = rewards.get('INCORRECT_VISION', 0)
    return {
        'trajectories': n_traj, 'steps': n_step,
        'success_rate': n_success / n_traj, 'successes': n_success,
        'first_try_rate': first_try / n_traj,
        'steps_per_traj': n_step / n_traj,
        'card_recognition': 1 - vision_errors / n_step,
        'rewards': rewards, 'causes': causes,
    }


def print_report(path, r):
    print(f'log                      : {path}')
    print(f'trajectories             : {r["trajectories"]}')
    print(f'steps                    : {r["steps"]}  (mean {r["steps_per_traj"]:.2f} per trajectory)')
    print(f'SUCCESS RATE (per traj)  : {r["success_rate"]:.1%}   [{r["successes"]}/{r["trajectories"]}]')
    print(f'  ... solved on 1st try  : {r["first_try_rate"]:.1%}')
    print(f'card recognition acc     : {r["card_recognition"]:.1%}')
    print('\nper-step reward breakdown:')
    for name, count in r['rewards'].most_common():
        print(f'  {name:<20} {count:>5}  {count / r["steps"]:6.1%}')
    failed = sum(r['causes'].values())
    print('\nfailed steps, by cause:')
    for label in ('formatting only (maths OK)', 'wrong arithmetic',
                  'no formula parsed', 'card misrecognition'):
        count = r['causes'].get(label, 0)
        print(f'  {label:<28} {count:>5}  {count / failed if failed else 0:6.1%} of failures')


def print_comparison(strict_path, tolerant_path, strict, tolerant):
    print('GeneralPoints whitespace ablation')
    print('  strict   = verifier as shipped; a space in the formula is an illegal character')
    print('  tolerant = whitespace stripped before validation; arithmetic judged on its merits')
    print(f'  strict   log: {strict_path}')
    print(f'  tolerant log: {tolerant_path}')
    print()
    rows = [
        ('SUCCESS RATE (per traj)', f'{strict["success_rate"]:.1%}', f'{tolerant["success_rate"]:.1%}'),
        ('  solved on 1st try', f'{strict["first_try_rate"]:.1%}', f'{tolerant["first_try_rate"]:.1%}'),
        ('card recognition acc', f'{strict["card_recognition"]:.1%}', f'{tolerant["card_recognition"]:.1%}'),
        ('steps used per traj', f'{strict["steps_per_traj"]:.2f}', f'{tolerant["steps_per_traj"]:.2f}'),
        ('ILLEGAL_FORMULA (steps)',
         f'{strict["rewards"].get("ILLEGAL_FORMULA", 0) / strict["steps"]:.1%}',
         f'{tolerant["rewards"].get("ILLEGAL_FORMULA", 0) / tolerant["steps"]:.1%}'),
    ]
    width = max(len(r[0]) for r in rows)
    print(f'{"":<{width}}   {"strict":>10} {"tolerant":>10}')
    for label, a, b in rows:
        print(f'{label:<{width}}   {a:>10} {b:>10}')
    delta = tolerant['success_rate'] - strict['success_rate']
    print(f'\nformatting tax on success rate: {delta:+.1%} '
          f'({strict["successes"]} -> {tolerant["successes"]} of {strict["trajectories"]} trajectories)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='action', required=True)
    rep = sub.add_parser('report', help='full breakdown of one log')
    rep.add_argument('log', type=Path)
    rep.add_argument('--target', type=int, default=24)
    cmp_ = sub.add_parser('compare', help='side-by-side whitespace ablation')
    cmp_.add_argument('strict', type=Path)
    cmp_.add_argument('tolerant', type=Path)
    cmp_.add_argument('--target', type=int, default=24)
    args = ap.parse_args()

    if args.action == 'report':
        steps = load_steps(args.log)
        print_report(args.log, analyse(steps, args.target))
        return

    s_steps, t_steps = load_steps(args.strict), load_steps(args.tolerant)
    s_puzzles, t_puzzles = puzzle_stream(s_steps), puzzle_stream(t_steps)
    if s_puzzles != t_puzzles:
        n = next((i for i, (x, y) in enumerate(zip(s_puzzles, t_puzzles)) if x != y), None)
        raise SystemExit(
            'Refusing to compare: the two runs did not see the same puzzles.\n'
            f'  strict   has {len(s_puzzles)} trajectories, tolerant has {len(t_puzzles)}\n'
            + (f'  first divergence at trajectory {n}: {s_puzzles[n]} vs {t_puzzles[n]}\n' if n is not None else '')
            + '  Re-run both conditions from the same seed and num_traj.')
    print(f'[ok] both runs saw the identical {len(s_puzzles)}-puzzle stream\n')
    print_comparison(args.strict, args.tolerant,
                     analyse(s_steps, args.target), analyse(t_steps, args.target))


if __name__ == '__main__':
    main()
