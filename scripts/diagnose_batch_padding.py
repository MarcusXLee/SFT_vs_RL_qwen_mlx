"""Does batch_generate sample from the model's distribution when prompt lengths differ?

diagnose_sampler_scorer.py found that tokens batch_generate actually sampled
(capture verified: decode(ids) == text on 50/50) score a mean of -1.21 nats
under a clean forward pass, with 44/2440 below -15. A sampler drawing from the
distribution the scorer computes averages ~ -entropy (~ -0.14 here) and can
essentially never draw a -15 token. That batch mixed prompt lengths.

Single-turn GRPO never mixes lengths (a group is G copies of one prompt);
multi-turn does from turn 2 on, because trajectories diverge. So the test holds
the prompts FIXED and changes only the batching:

  SAME   each prompt batched with copies of itself    (no padding; single-turn regime)
  SOLO   each prompt alone, batch of 1                 (no batching at all)
  MIXED  prompts of different lengths batched together (multi-turn regime)

Every sampled token is scored by the same clean batch-of-1 forward pass. If
MIXED is anomalous and SAME/SOLO are not, the rollout sampler -- not the
scorer, not the re-tokenisation -- is what breaks in multi-turn. Within MIXED,
rows are also split by how much left padding they received.
"""
import os, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import mlx.core as mx, numpy as np
from mlx_vlm import load, generate, batch_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.sample_utils import make_sampler
from mlx_vlm.trainer.utils import apply_lora_layers
from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG
from rl.trainer.grpo_trainer_mlx import pad_sequences, sequence_logprobs

model, proc = load(str(ROOT / 'models/Qwen3-VL-8B-Instruct'))
model = apply_lora_layers(model, str(ROOT / 'train_ckpt/sft_gp_language')); model.eval()
tok = proc.tokenizer

# ---- record the ids batch_generate actually samples (verified below) --------
det = proc.detokenizer
_reset, _add = det.reset, det.add_token
CAPTURED = []
def _rec_reset(*a, **k):
    CAPTURED.append([]); return _reset(*a, **k)
def _rec_add(t, *a, **k):
    CAPTURED[-1].append(int(t)); return _add(t, *a, **k)
det.reset, det.add_token = _rec_reset, _rec_add

def sample(prompts):
    """batch_generate one batch; return (texts, sampled ids), alignment asserted."""
    out = batch_generate(model, proc, images=None, prompts=prompts, max_tokens=128,
                         sampler=make_sampler(temp=1.0), verbose=False)
    got = CAPTURED[-len(prompts):]   # one extra reset precedes the per-row ones
    texts = list(out.texts)
    assert all(tok.decode(g) == t for g, t in zip(got, texts)), 'capture misaligned'
    mx.clear_cache()
    return texts, got

def clean_lp(prompt, sid):
    """Log-probs of the sampled ids under a clean batch-of-1 forward pass."""
    pid = tok.encode(prompt)
    i, m = pad_sequences([pid + sid], [[0.0] * len(pid) + [1.0] * len(sid)], 0)
    lp, mm = sequence_logprobs(model, i, m, 1.0); mx.eval(lp)
    v = np.array(lp)[0][np.array(mm)[0] > 0]
    mx.clear_cache()
    return v

# ---- real multi-turn conversations of varying length -------------------------
random.seed(3); mx.random.seed(3)
env = GeneralPointEnv_oneline(language_only=True, resolution=64, verify_iter=5,
                              strip_formula_whitespace=True)
convs = []
for h in range(4):
    _, info = env.reset()
    q = Q_GeneralPoint_EQN_L_EG.format(cards=info['Plain Cards'], target_number='24',
                                       face_card_msg="'J', 'Q', and 'K' count as '10'.")
    pl = [{"role": "user", "content": [{"type": "text", "text": q}]}]
    convs.append(list(pl))
    for turn in range(2, 6):
        pr = apply_chat_template(proc, model.config, pl, num_images=0)
        r = generate(model, proc, pr, image=None, max_tokens=128, temperature=1.0, verbose=False)
        env.remaining_step, env.verify_info = env.verify_iter, None
        env.cards_num, env.solution, env.cards_without_suit = info['Numbers'], info['Solution'], info['Plain Cards']
        _, rew, done, trunc, _ = env.step(r.text.strip())
        if done: break
        pl = pl + [{"role": "assistant", "content": [{"type": "text", "text": r.text.strip()}]},
                   {"role": "user", "content": [{"type": "text", "text": f"You failed this trial because {env.verify_info}"}]}]
        convs.append(list(pl))
    mx.clear_cache()
prompts = [apply_chat_template(proc, model.config, c, num_images=0) for c in convs]
plen = [len(tok.encode(p)) for p in prompts]
order = np.argsort(plen)
P = [prompts[i] for i in order]                 # every prompt, shortest first
L = [plen[i] for i in order]
print(f'{len(P)} prompts, lengths {min(L)}..{max(L)} tokens\n')

records = {'SAME': [], 'SOLO': [], 'MIXED': []}   # (lp array, left_pad, prompt_len)
REPS = 4
mx.random.seed(11)
for k, p in enumerate(P):
    _, ids = sample([p] * REPS)                 # SAME: equal lengths, no padding
    for sid in ids:
        records['SAME'].append((clean_lp(p, sid), 0, L[k]))
    _, ids = sample([p])                        # SOLO
    records['SOLO'].append((clean_lp(p, ids[0]), 0, L[k]))
for rep in range(REPS):                         # MIXED: batches of 8 spanning lengths
    perm = np.random.RandomState(rep).permutation(len(P))
    for s in range(0, len(P), 8):
        idx = list(perm[s:s + 8])
        _, ids = sample([P[i] for i in idx])
        width = max(L[i] for i in idx)
        for i, sid in zip(idx, ids):
            records['MIXED'].append((clean_lp(P[i], sid), width - L[i], L[i]))

def summary(name, recs):
    if not recs:
        return
    v = np.concatenate([r[0] for r in recs])
    print(f'  {name:<22} rows {len(recs):>3}  tokens {v.size:>5}  mean {v.mean():+.3f}  '
          f'min {v.min():+7.2f}  < -10: {(v < -10).sum():>3}  < -15: {(v < -15).sum():>3}')

print('sampled tokens, scored by a clean batch-of-1 forward pass')
print('(a sampler that matches the model should average ~ -entropy and have ~0 tokens < -15)\n')
for name in ('SOLO', 'SAME', 'MIXED'):
    summary(name, records[name])
print('\nMIXED split by the left padding each row received:')
summary('MIXED, pad == 0', [r for r in records['MIXED'] if r[1] == 0])
summary('MIXED, 0 < pad <= 100', [r for r in records['MIXED'] if 0 < r[1] <= 100])
summary('MIXED, pad > 100', [r for r in records['MIXED'] if r[1] > 100])
