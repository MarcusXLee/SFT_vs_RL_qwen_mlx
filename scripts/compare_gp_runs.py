"""Paired comparison of GeneralPoints evaluation logs on the same hands.

    compare_gp_runs.py LABEL=LOG [LABEL=LOG ...] [--pairs A:B ...]

Each LOG is a launcher .jsonl, or a directory holding exactly one. All logs
must cover the identical hand stream (asserted -- a paired test on different
hands would be meaningless). Reports, per model:

  1st try    hands solved on the first attempt
  solved     hands solved within verify_iter attempts (the paper's metric)
  recovered  solved later, among hands failed on the first attempt
  ILLEGAL    share of steps rejected for formatting

and for each requested pair, exact two-sided McNemar tests on final and
first-attempt success, plus recovery on the hands BOTH failed first time.
Pairs default to every model against the first one.
"""
import argparse, glob, json, sys
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
from utils_rl import REWARD_FN  # noqa: E402

CS = REWARD_FN['CORRECT_SOLUTION']


def load(path):
    p = Path(path)
    if p.is_dir():
        files = sorted(glob.glob(str(p / '*.jsonl')))
        if len(files) != 1:
            raise SystemExit(f'{p}: expected exactly one .jsonl, found {len(files)}')
        p = Path(files[0])
    hands, first, solved, when = {}, {}, {}, {}
    steps = illegal = 0
    for line in open(p):
        r = json.loads(line)
        if not (isinstance(r, dict) and 'veri_step' in r):
            continue
        steps += 1
        illegal += r['reward'] == REWARD_FN['ILLEGAL_FORMULA']
        i = r['sample_id']
        hands[i] = tuple(sorted(r['info']['Plain Cards']))
        if r['reward'] == CS and not solved.get(i):
            when[i] = r['veri_step'] + 1
        solved[i] = solved.get(i, False) or r['reward'] == CS
        if r['veri_step'] == 0:
            first[i] = r['reward'] == CS
    return dict(hands=hands, first=first, solved=solved, when=when,
                steps=steps, illegal=illegal / max(steps, 1), path=str(p))


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans: (only-a, only-b, p)."""
    ids = sorted(a)
    x = sum(1 for i in ids if a[i] and not b[i])
    y = sum(1 for i in ids if b[i] and not a[i])
    n = x + y
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return x, y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='LABEL=LOG')
    ap.add_argument('--pairs', nargs='*', default=None, help='A:B (default: each vs first)')
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        label, path = spec.split('=', 1)
        runs[label] = load(path)
    labels = list(runs)
    ref = runs[labels[0]]['hands']
    for lab in labels[1:]:
        if runs[lab]['hands'] != ref:
            raise SystemExit(f'{lab} saw a different hand stream than {labels[0]}; '
                             f'a paired comparison would be meaningless')
    n = len(ref)
    print(f'[ok] identical stream: {n} trajectories, {len(set(ref.values()))} distinct hands\n')

    print(f"{'model':<16} {'1st try':>8} {'solved':>8} {'rate':>6} {'recovered':>12} {'steps':>6} {'ILLEGAL':>8}")
    for lab in labels:
        r = runs[lab]
        failed = [i for i in r['first'] if not r['first'][i]]
        rec = sum(1 for i in failed if r['solved'][i])
        print(f"{lab:<16} {sum(r['first'].values()):>8} {sum(r['solved'].values()):>8} "
              f"{sum(r['solved'].values()) / n:>6.1%} {rec:>5}/{len(failed):<6} "
              f"{r['steps']:>6} {r['illegal']:>8.1%}")

    pairs = ([tuple(p.split(':')) for p in args.pairs] if args.pairs
             else [(labels[0], lab) for lab in labels[1:]])
    for a, b in pairs:
        A, B = runs[a], runs[b]
        print(f'\n{a} vs {b}')
        x, y, p = mcnemar(A['solved'], B['solved'])
        print(f'  final success      only {a} {x:>3}   only {b} {y:>3}   exact McNemar p = {p:.4f}')
        x, y, p = mcnemar(A['first'], B['first'])
        print(f'  first attempt      only {a} {x:>3}   only {b} {y:>3}   exact McNemar p = {p:.4f}')
        both = [i for i in A['first'] if not A['first'][i] and not B['first'][i]]
        ra = {i: A['solved'][i] for i in both}; rb = {i: B['solved'][i] for i in both}
        x, y, p = mcnemar(ra, rb)
        print(f'  recovery, n={len(both):<3}   {a} {sum(ra.values())}/{len(both)}   '
              f'{b} {sum(rb.values())}/{len(both)}   (only {x} vs {y}, p = {p:.4f})')

    print('\nattempt on which solved hands were solved:')
    for lab in labels:
        print(f"  {lab:<16} {dict(sorted(Counter(runs[lab]['when'].values()).items()))}")


if __name__ == '__main__':
    main()
