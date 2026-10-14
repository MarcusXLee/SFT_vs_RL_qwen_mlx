"""Generate GeneralPoints SFT data from the environment's own solver.

Each example pairs the evaluation prompt with a gold answer in the exact format
the verifier accepts -- valid JSON, correct card ranks, and a formula written
**without spaces**, which is what the shipped verifier requires.

Train/eval separation is enforced, not assumed: any hand whose sorted card
multiset also appears in the evaluation stream is dropped. In a paper about
memorisation, an overlapping train set would invalidate the whole comparison.

Messages use the evaluator's exact structure -- content lists, image marker
first -- because the chat template strips the prompt's surrounding newlines on
that path but not for a plain-string content. Data built with plain strings
(the original data/sft_gp_language) trained on a prompt one newline longer at
each end than the one it was evaluated on (BUGS.md #28).

--vision renders each hand exactly as the GP-VL evaluation does (resolution
1200, black face cards; evaluation/configs/qwen_gp_vl.yaml) and stores the PNG
beside the data, referenced relative to the repository root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline  # noqa: E402
from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG, Q_GeneralPoint_EQN_VL_EG  # noqa: E402
from utils_rl import step_rewards  # noqa: E402


def hand_key(plain_cards):
    """Order-independent identity of a hand, so '5,2' and '2,5' collide."""
    return tuple(sorted(plain_cards))


def eval_hands(seed, n, **env_kwargs):
    """Replay the evaluation stream to learn exactly which hands are held out.

    Card values do not depend on suit colour or rendering: replaying with
    'mixed' text-only, 'black' text-only and 'black' rendered at 1200 all
    reproduce the GP-VL evaluation log's hands 50/50, and the 'red' (visual-OOD)
    stream equals the 'black' one over 200. So one holdout covers GP-L, GP-VL
    in-distribution and GP-VL visual-OOD alike. Rule-OOD (ood=True, faces
    11-13) is a different stream with different arithmetic, and is not held out
    -- as on the language side.
    """
    random.seed(seed)
    env = GeneralPointEnv_oneline(language_only=True, resolution=64, **env_kwargs)
    held = set()
    for _ in range(n):
        _, info = env.reset(seed=seed)
        held.add(hand_key(info['Plain Cards']))
    return held


def build(args):
    holdout = eval_hands(args.eval_seed, args.eval_traj,
                         target_points=args.target, treat_face_cards_as_10=True)
    print(f'held out {len(holdout)} distinct hands from the eval stream (seed {args.eval_seed})')

    random.seed(args.seed)
    resolution = args.resolution or (1200 if args.vision else 64)
    env_kwargs = dict(language_only=not args.vision, resolution=resolution,
                      target_points=args.target, treat_face_cards_as_10=True)
    if args.vision:
        env_kwargs['face_cards_color'] = args.face_cards_color
        image_dir = args.out / 'images'
        image_dir.mkdir(parents=True, exist_ok=True)
    env = GeneralPointEnv_oneline(**env_kwargs)
    face_msg = "'J', 'Q', and 'K' count as '10'."
    prompt_tpl = Q_GeneralPoint_EQN_VL_EG if args.vision else Q_GeneralPoint_EQN_L_EG

    rows, seen, overlap, attempts, unusable = [], set(), 0, 0, 0
    while len(rows) < args.n and attempts < args.n * 200:
        attempts += 1
        obs, info = env.reset(seed=args.seed)
        key = hand_key(info['Plain Cards'])
        if key in holdout:
            overlap += 1
            continue
        if not args.allow_duplicate_hands and key in seen:
            continue
        seen.add(key)

        # Never emit a target the verifier would reject. The env's solver uses
        # exact Fractions but step_rewards compares with float `eval`, so a few
        # gold solutions (~0.3%, all involving division) miss by ~4e-15:
        # eval("(2/5)*6*10") == 24.000000000000004. Keep only what scores 5.
        # Cheap pre-filter on the float compare the verifier actually performs,
        # then confirm the single chosen candidate through step_rewards itself.
        candidates = [s for s in info['Solution'] if eval(s) == args.target]
        random.shuffle(candidates)
        solution = None
        for cand in candidates:
            reward, _ = step_rewards(
                card_nums=info['Numbers'], current_formula=cand,
                solutions=info['Solution'], target_points=args.target,
                recognized_cards=info['Plain Cards'], translated_number=info['Numbers'],
                gt_cards=info['Plain Cards'], language_only=not args.vision)
            if reward == 5:
                solution = cand
                break
        if solution is None:
            unusable += 1
            seen.discard(key)
            continue
        answer = {'cards': info['Plain Cards'],
                  'number': info['Numbers'],
                  'formula': f'{solution}={args.target}'}

        kwargs = dict(target_number=str(args.target), face_card_msg=face_msg)
        if not args.vision:
            kwargs['cards'] = info['Plain Cards']
        # The evaluator's structure exactly (QwenEvaluator.formulate_payload /
        # append_intermidiate_res), so training and evaluation see one prompt.
        user = [{'type': 'text', 'text': prompt_tpl.format(**kwargs)}]
        row = {}
        if args.vision:
            user.insert(0, {'type': 'image'})
            image = image_dir / f'{len(rows):05d}.png'
            (Image.fromarray(obs) if isinstance(obs, np.ndarray) else obs).save(image)
            row['images'] = [str(image.resolve().relative_to(ROOT))]
        row['messages'] = [
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps(answer)}]},
        ]
        rows.append(row)

    if len(rows) < args.n:
        print(f'warning: only {len(rows)} unique hands available (asked {args.n})')

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    splits = {'valid': rows[:n_val], 'train': rows[n_val:]}

    args.out.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        payload = ''.join(json.dumps(r) + '\n' for r in split)
        (args.out / f'{name}.jsonl').write_text(payload)
        print(f'{name:<6} {len(split):>5} examples  sha256={hashlib.sha256(payload.encode()).hexdigest()[:12]}')

    meta = {'seed': args.seed, 'eval_seed': args.eval_seed, 'eval_traj': args.eval_traj,
            'target': args.target, 'vision': args.vision, 'resolution': resolution,
            'face_cards_color': args.face_cards_color if args.vision else None,
            'message_format': 'evaluator content lists',
            'held_out_hands': len(holdout), 'overlaps_dropped': overlap,
            'unique_hands': len(seen), 'hands_without_usable_solution': unusable, 'n_train': len(splits['train']), 'n_valid': len(splits['valid'])}
    (args.out / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'dropped {overlap} draws that collided with the eval stream')
    print(f'dropped {unusable} hands whose every gold solution fails the verifier')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'data/sft_gp_language')
    ap.add_argument('-n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1234, help='must differ from --eval-seed')
    ap.add_argument('--eval-seed', type=int, default=42)
    ap.add_argument('--eval-traj', type=int, default=200,
                    help='how much of the eval stream to hold out (>= any eval num_traj)')
    ap.add_argument('--target', type=int, default=24)
    ap.add_argument('--resolution', type=int, default=None,
                    help='render size; default 1200 with --vision (as the GP-VL eval), else unused')
    ap.add_argument('--face-cards-color', default='black', choices=['black', 'red', 'mixed'],
                    help='with --vision: suit colour of the rendered cards (eval in-dist: black)')
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--vision', action='store_true')
    ap.add_argument('--allow-duplicate-hands', action='store_true')
    a = ap.parse_args()
    if a.seed == a.eval_seed:
        ap.error('--seed must differ from --eval-seed or train and test overlap')
    build(a)
