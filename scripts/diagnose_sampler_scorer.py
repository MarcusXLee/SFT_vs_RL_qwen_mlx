"""Why does a token the policy JUST SAMPLED score log-prob -31.6 under that policy?

A genuinely sampled token cannot have probability ~2e-14. So somewhere the
thing that SAMPLED the token and the thing that SCORES it disagree. Three
candidates, each isolated here on tokens that batch_generate actually produced:

  (A) re-tokenisation  - encode(text) differs from the ids actually sampled
  (C1) true model      - log-prob under a clean batch-of-1 forward, no padding
  (C2) training scorer - log-prob under sequence_logprobs on a padded batch,
                         exactly as update() computes old_lp / cur_lp

Prompts of DIFFERENT conversation lengths are batched together, reproducing the
heavy padding that later multi-turn turns create.

The sampled ids are recorded by wrapping the detokenizer that _generate_batch
feeds them through; the wrapper only observes.
"""
import os, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import mlx.core as mx, numpy as np
import mlx_vlm
from mlx_vlm import load, generate, batch_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.sample_utils import make_sampler
from mlx_vlm.trainer.utils import apply_lora_layers
from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG
from rl.trainer.grpo_trainer_mlx import pad_sequences, sequence_logprobs

model, proc = load(str(ROOT / 'models/Qwen3-VL-8B-Instruct'))
model = apply_lora_layers(model, str(ROOT / 'train_ckpt/sft_gp_language')); model.eval()
tok = proc.tokenizer; EOS = tok.eos_token_id

# ---- record the ids batch_generate actually samples -------------------------
det = proc.detokenizer
_reset, _add = det.reset, det.add_token
CAPTURED = []
def _rec_reset(*a, **k):
    CAPTURED.append([]); return _reset(*a, **k)
def _rec_add(t, *a, **k):
    CAPTURED[-1].append(int(t)); return _add(t, *a, **k)
det.reset, det.add_token = _rec_reset, _rec_add

# ---- build real multi-turn conversations of varying length ------------------
random.seed(3); mx.random.seed(3)
env = GeneralPointEnv_oneline(language_only=True, resolution=64, verify_iter=5,
                              strip_formula_whitespace=True)
convs = []
for h in range(10):
    _, info = env.reset()
    q = Q_GeneralPoint_EQN_L_EG.format(cards=info['Plain Cards'], target_number='24',
                                       face_card_msg="'J', 'Q', and 'K' count as '10'.")
    pl = [{"role": "user", "content": [{"type": "text", "text": q}]}]
    convs.append(list(pl))                      # a turn-1 prompt
    for turn in range(2, 6):                    # extend with real failed attempts
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
print(f'{len(prompts)} prompts, lengths {min(plen)}..{max(plen)} tokens (mixed => heavy padding)\n')

# ---- sample them TOGETHER with batch_generate, as the rollout does ----------
CAPTURED.clear()
BATCH = 8
texts, ids = [], []
for s in range(0, len(prompts), BATCH):
    chunk = prompts[s:s+BATCH]
    out = batch_generate(model, proc, images=None, prompts=chunk, max_tokens=128,
                         sampler=make_sampler(temp=1.0), verbose=False)
    # batch_generate resets the detokenizer ONCE up front (ar.py:3230) and then
    # once per completion inside _generate_batch. Taking the first N lists after
    # the call was off by one: ids[i] held completion i-1, so every token was
    # scored in the wrong context. The per-completion lists are the LAST N.
    got = CAPTURED[-len(chunk):]
    texts += list(out.texts); ids += got
    mx.clear_cache()
assert len(ids) == len(texts) == len(prompts), (len(ids), len(texts), len(prompts))
# Hard check: the captured ids ARE what the text was built from, so decoding
# them must reproduce it exactly. If not, the capture is misaligned and every
# number below is meaningless.
bad = [i for i in range(len(ids)) if tok.decode(ids[i]) != texts[i]]
print(f'capture alignment: decode(ids) == text on {len(ids)-len(bad)}/{len(ids)} completions')
assert not bad, f'capture misaligned at {bad[:5]}'

# ---- (A) re-tokenisation on the batch path ----------------------------------
mism = [i for i in range(len(texts)) if tok.encode(texts[i]) != ids[i]]
print(f'(A) re-tokenisation mismatches on the BATCH path: {len(mism)}/{len(texts)} completions')

# ---- (C1) clean batch-of-1 vs (C2) padded training scorer, on SAMPLED ids ----
seqs, masks = [], []
for p, sid in zip(prompts, ids):
    pid = tok.encode(p); seqs.append(pid + sid)
    masks.append([0.0]*len(pid) + [1.0]*len(sid))
clean = []
for sq, mk in zip(seqs, masks):
    i, m = pad_sequences([sq], [mk], 0)
    lp, mm = sequence_logprobs(model, i, m, 1.0); mx.eval(lp)
    clean.append(np.array(lp)[0][np.array(mm)[0] > 0])
    mx.clear_cache()
pad_lp = []
for s in range(0, len(seqs), 4):
    i, m = pad_sequences(seqs[s:s+4], masks[s:s+4], 0)
    lp, mm = sequence_logprobs(model, i, m, 1.0); mx.eval(lp)
    lp, mm = np.array(lp), np.array(mm)
    for r in range(lp.shape[0]):
        pad_lp.append(lp[r][mm[r] > 0])
    mx.clear_cache()

c1 = np.concatenate(clean); c2 = np.concatenate(pad_lp)
print(f'\nsampled tokens scored: {c1.size}')
for name, v in (('(C1) clean batch-of-1', c1), ('(C2) padded training scorer', c2)):
    print(f'  {name:<30} mean {v.mean():+.3f}  min {v.min():+.2f}  '
          f'< -10: {(v < -10).sum()}   < -15: {(v < -15).sum()}')
diff = np.abs(c1 - c2)
print(f'\n|C1 - C2| per token: mean {diff.mean():.2e}  max {diff.max():.3f}  '
      f'tokens differing by > 1 nat: {(diff > 1).sum()}')
worst = np.argsort(-diff)[:6]
print('largest disagreements between true model and training scorer:')
flat_ids = [t for sid in ids for t in sid]
for w in worst:
    print(f'  token {tok.decode([flat_ids[w]])!r:<10} clean {c1[w]:+8.3f}   padded {c2[w]:+8.3f}   gap {diff[w]:.3f}')
