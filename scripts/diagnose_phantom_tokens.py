"""Does re-tokenising sampled text reproduce the tokens that were sampled?

The multi-turn path uses batch_generate, which returns only text, and rebuilds
token ids with encode(text). If the model samples a NON-canonical split -- e.g.
'"' then '}' as two tokens -- the text is '"}', and encode() returns the single
merged id 9207. Training then scores a token the model never produced ("a
phantom"), typically at near-zero probability, and a failed trajectory pushes
it lower still.

The earlier 80/80 check covered ~3,800 turn-1 tokens: too few to see a
~1-in-7,000 event, and never beyond turn 1. This uses generate(), which does
return the sampled ids, over full multi-turn conversations.

    .venv-qwen/bin/python scripts/diagnose_phantom_tokens.py [--hands N]
"""
import argparse, json, os, random, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer.utils import apply_lora_layers
from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hands', type=int, default=40)
    ap.add_argument('--max-turns', type=int, default=5)
    ap.add_argument('--out', type=Path, default=ROOT / 'logs/phantom_tokens.json')
    a = ap.parse_args()
    model, proc = load(str(ROOT / 'models/Qwen3-VL-8B-Instruct'))
    model = apply_lora_layers(model, str(ROOT / 'train_ckpt/sft_gp_language')); model.eval()
    tok = proc.tokenizer; EOS = tok.eos_token_id
    random.seed(7); mx.random.seed(7)
    env = GeneralPointEnv_oneline(language_only=True, resolution=64, verify_iter=a.max_turns,
                                  strip_formula_whitespace=True)
    per_turn = {t: {'tokens': 0, 'completions': 0, 'mismatched': 0, 'phantoms': 0} for t in range(1, a.max_turns+1)}
    examples, phantom_ids = [], Counter()
    for h in range(a.hands):
        _, info = env.reset()
        q = Q_GeneralPoint_EQN_L_EG.format(cards=info['Plain Cards'], target_number='24',
                                           face_card_msg="'J', 'Q', and 'K' count as '10'.")
        payload = [{"role": "user", "content": [{"type": "text", "text": q}]}]
        for turn in range(1, a.max_turns + 1):
            prompt = apply_chat_template(proc, model.config, payload, num_images=0)
            r = generate(model, proc, prompt, image=None, max_tokens=128, temperature=1.0, verbose=False)
            sampled = list(r.token_ids or [])
            if sampled and sampled[-1] == EOS:
                sampled = sampled[:-1]                  # encode(text) never contains it
            reenc = tok.encode(r.text)
            s = per_turn[turn]; s['completions'] += 1; s['tokens'] += len(sampled)
            if reenc != sampled:
                s['mismatched'] += 1
                extra = [t for t in reenc if t not in sampled]   # ids we'd score but never sampled
                s['phantoms'] += len(extra)
                phantom_ids.update(extra)
                if len(examples) < 8:
                    examples.append({'turn': turn, 'text_tail': r.text[-40:],
                                     'sampled_tail': sampled[-6:], 'reencoded_tail': reenc[-6:],
                                     'phantom_tokens': [tok.decode([t]) for t in extra]})
            env.remaining_step, env.verify_info = env.verify_iter, None
            env.cards_num, env.solution = info['Numbers'], info['Solution']
            env.cards_without_suit = info['Plain Cards']
            _, reward, done, trunc, sinfo = env.step(r.text)
            if done or trunc:
                break
            payload += [{"role": "assistant", "content": [{"type": "text", "text": r.text}]},
                        {"role": "user", "content": [{"type": "text",
                         "text": f"You failed this trial because {env.verify_info}"}]}]
            mx.clear_cache()
        print(f'  hand {h+1}/{a.hands}', flush=True)

    tot_tok = sum(s['tokens'] for s in per_turn.values())
    tot_ph = sum(s['phantoms'] for s in per_turn.values())
    tot_c = sum(s['completions'] for s in per_turn.values())
    tot_m = sum(s['mismatched'] for s in per_turn.values())
    print(f'\n{"turn":>4} {"completions":>11} {"tokens":>7} {"mismatched":>11} {"phantom tokens":>15}')
    for t, s in per_turn.items():
        if s['completions']:
            print(f"{t:>4} {s['completions']:>11} {s['tokens']:>7} {s['mismatched']:>11} {s['phantoms']:>15}")
    print(f"\nTOTAL: {tot_c} completions, {tot_tok} tokens")
    print(f"  completions with re-tokenisation mismatch: {tot_m} ({tot_m/max(tot_c,1):.1%})")
    print(f"  phantom tokens (scored, never sampled):    {tot_ph}  "
          f"= 1 per {tot_tok/tot_ph:.0f} tokens" if tot_ph else "  phantom tokens: 0")
    if phantom_ids:
        print('  most common phantom tokens:',
              [(tok.decode([t]), n) for t, n in phantom_ids.most_common(8)])
    for e in examples[:5]:
        print(f"\n  turn {e['turn']}: ...{e['text_tail']!r}")
        print(f"    sampled   {e['sampled_tail']}\n    re-encoded {e['reencoded_tail']}  phantoms {e['phantom_tokens']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'per_turn': per_turn, 'total_tokens': tot_tok, 'phantoms': tot_ph,
                                 'phantom_ids': {tok.decode([k]): v for k, v in phantom_ids.items()},
                                 'examples': examples}, indent=2) + '\n')

if __name__ == '__main__':
    main()
