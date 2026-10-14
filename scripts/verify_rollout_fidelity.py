"""Does the GRPO rollout sample from the distribution the update scores?

Acceptance test for the rollout fix (BUGS.md #26). Drives the trainer's OWN
rollout (`_rollout_group` -> `_batch_generate`), built as train_grpo_qwen.py
builds it, and checks:

  1. guard   the old call (a templated string) is refused; the fixed call passes
  2. RoPE    every batch_generate call's positions run 0, 1, 2, ... in lockstep
             -- the mixed-length bug decoded at 17, -2, -1, 0. A negative
             control shows the check flags that bug on a raw mixed batch.
  3. fidelity  the ids batch_generate ACTUALLY sampled (captured at the
             detokenizer, alignment asserted), scored by the trainer's own
             scorer on the trajectories it assembled. A sampler that matches
             the scorer averages ~ -entropy per token, and essentially never
             draws a token below -15. Reported per turn, since the bug lived in
             turns >= 2. Also: re-tokenisation mismatches, and the appended EOS
             where the strip removed trailing whitespace.
  4. D1 size  how off-policy the double-templated context made the earlier
             single-turn runs: tokens sampled in the double context, scored in
             both. mean(lp_double - lp_single) estimates KL(double || single).

Exits non-zero if 1 or 2 fail, or if 3 finds a sampled token below -15.
"""
import copy, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import mlx.core as mx, numpy as np
from mlx_vlm import load, batch_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.sample_utils import make_sampler
from mlx_vlm.trainer.utils import apply_lora_layers
from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG
from rl.trainer.grpo_trainer_mlx import sequence_logprobs_chunked, pad_sequences, sequence_logprobs
from rl.trainer.grpo_multiturn_mlx import MultiTurnConfig, MultiTurnGRPOTrainer
from scripts.train_grpo_qwen import make_reward_fn

GROUPS = int(os.environ.get('GROUPS', 6))
OUT = ROOT / 'logs/diagnostics/rollout_fidelity'; OUT.mkdir(parents=True, exist_ok=True)
failures = []

model, proc = load(str(ROOT / 'models/Qwen3-VL-8B-Instruct'))
adapter = ROOT / 'train_ckpt/sft_gp_language'
model = apply_lora_layers(model, str(adapter))
model.config.lora = json.loads((adapter / 'adapter_config.json').read_text())
tok = proc.tokenizer
env = GeneralPointEnv_oneline(language_only=True, resolution=64, target_points=24,
                              treat_face_cards_as_10=True, verify_iter=5,
                              strip_formula_whitespace=True)
oracle = {'target_number': '24', 'face_card_msg': "'J', 'Q', and 'K' count as '10'."}
trainer = MultiTurnGRPOTrainer(
    model=model, processor=proc, env=env, prompt_template=Q_GeneralPoint_EQN_L_EG,
    oracle_arguments=oracle, reward_fn=make_reward_fn(env),
    config=MultiTurnConfig(group_size=8, prompts_per_update=4, temperature=1.0,
                           max_tokens=128, microbatch=4, batch_rollout=True,
                           max_turns=5, reward_mode='final', seed=0))

# ---- instrumentation: observe only ------------------------------------------
det = proc.detokenizer
_reset, _add = det.reset, det.add_token
CAPTURED = []
def _rec_reset(*a, **k):
    CAPTURED.append([]); return _reset(*a, **k)
def _rec_add(t, *a, **k):
    CAPTURED[-1].append(int(t)); return _add(t, *a, **k)
det.reset, det.add_token = _rec_reset, _rec_add

inner = model.language_model.model
_orig_call = type(inner).__call__
POS = []                                   # (seq_len, positions [B, L]) per forward
def _rec_call(self, inputs, *a, **k):
    pid = k.get('position_ids')
    if pid is not None:
        p = np.array(pid); p = p[0] if p.ndim == 3 else p
        POS.append((inputs.shape[1], p))
    return _orig_call(self, inputs, *a, **k)
type(inner).__call__ = _rec_call

CALLS = []                                 # per _batch_generate: lens, texts, sampled ids, positions
_orig_bg = trainer._batch_generate
def _rec_bg(conversations, prompt_ids):
    n_pos = len(POS)
    texts = _orig_bg(conversations, prompt_ids)
    got = CAPTURED[-len(conversations):]   # batch_generate resets once more up front
    assert all(tok.decode(g) == t for g, t in zip(got, texts)), 'capture misaligned'
    CALLS.append({'lens': [len(p) for p in prompt_ids], 'texts': texts, 'ids': got,
                  'pos': POS[n_pos:]})
    return texts
trainer._batch_generate = _rec_bg

# ---- 1. guard ----------------------------------------------------------------
first = [{"role": "user", "content": [{"type": "text", "text": "Say the number 24."}]}]
templated = apply_chat_template(proc, model.config, first, num_images=0)
ids = tok.encode(templated)
try:
    _orig_bg([templated, templated], [ids, ids])
    failures.append('guard: the double-templating call was NOT refused')
    print('1. guard     FAIL: templated string accepted')
except RuntimeError as e:
    print(f'1. guard     refuses the old call: {str(e)[:90]}...')
_orig_bg([copy.deepcopy(first)] * 2, [ids, ids])
print('             accepts the message-list call (token count matches)')
try:
    _orig_bg([first, first + first], [ids, ids + ids])
    failures.append('guard: a mixed-length call was NOT refused')
except ValueError:
    print('             refuses a mixed-length call')
mx.clear_cache()

# ---- rollout through the real code path --------------------------------------
CALLS.clear(); POS.clear()
trajs, t0 = [], time.time()
for g in range(GROUPS):
    info = trainer._reset_to_training_hand()
    trajs += trainer._rollout_group(info)
    mx.clear_cache()
secs = time.time() - t0

# ---- 2. RoPE positions -------------------------------------------------------
# Rows are NOT tracked by index: batch_generate drops finished rows from the
# batch mid-generation, so row r of one forward is not row r of the next (two
# earlier versions of this check tripped on that, and one also expected the
# first single-token step at L, where batch_generate actually runs the LAST
# prompt token at L-1).
#
# Invariant for an equal-length call: the prefill forwards cover exactly
# 0..P-1 on every row, and the k-th single-token forward is at P+k on EVERY row
# still active -- equal-length rows advance in lockstep.
def lockstep_ok(call):
    B = len(call['lens'])
    pre = [np.broadcast_to(p, (B, p.shape[1])) for n, p in call['pos'] if n > 1]
    steps = [p for n, p in call['pos'] if n == 1]
    P = sum(q.shape[1] for q in pre)
    if not pre or P > call['lens'][0]:
        return False
    if not all(np.array_equal(row, np.arange(P)) for row in np.concatenate(pre, axis=1)):
        return False
    return all(np.all(q == P + k) for k, q in enumerate(steps))

bad_calls = [i for i, c in enumerate(CALLS) if not lockstep_ok(c)]
n_steps = sum(1 for c in CALLS for n, _ in c['pos'] if n == 1)
print(f'\n2. RoPE      {len(CALLS)} batch_generate calls, {n_steps} single-token steps; '
      f'calls not in lockstep from 0: {len(bad_calls)}')
if bad_calls:
    failures.append(f'RoPE: {len(bad_calls)} calls with wrong positions')
sizes = [len(c['lens']) for c in CALLS]
print(f'            call sizes: {np.bincount(sizes)[1:].tolist()} (count of calls with 1..8 rows)')

# Negative control: the check's core must flag the bug when it is present.
# Mixed lengths, raw batch_generate; per row, consecutive single-token steps
# must advance by exactly +1 (the bug jumps 17 -> -2). Two rows, 4 tokens, so
# no row finishes and indices are stable -- verified, not assumed.
POS.clear()
short = [{"role": "user", "content": [{"type": "text", "text": "Hi."}]}]
long_ = [{"role": "user", "content": [{"type": "text", "text": "Please write one sentence "
          "about the number twenty four and why mathematicians find it interesting."}]}]
batch_generate(model, proc, images=None, prompts=[short, long_], max_tokens=4,
               sampler=make_sampler(temp=0.0), verbose=False)
steps = [q for n, q in POS if n == 1]
if all(q.shape[0] == 2 for q in steps) and len(steps) >= 3:
    jumps = [np.diff(np.array([q[r, 0] for q in steps])) for r in range(2)]
    flagged = [bool(np.any(j != 1)) for j in jumps]
    print(f'            negative control (mixed lengths, raw batch_generate): padded row '
          f'steps {[int(q[0, 0]) for q in steps]} flagged {flagged[0]}; '
          f'unpadded row flagged {flagged[1]}')
    if flagged != [True, False]:
        failures.append(f'RoPE negative control did not behave: {flagged}')
else:
    failures.append('RoPE negative control inconclusive (rows finished early)')
mx.clear_cache()

# ---- 3. fidelity on the assembled trajectories ------------------------------
retok = sum(1 for c in CALLS for t, i in zip(c['texts'], c['ids'])
            if tok.encode(t.strip()) != i and tok.encode(t) != i)
stripped = sum(1 for c in CALLS for t in c['texts'] if t != t.strip())
n_comp = sum(len(c['texts']) for c in CALLS)

seqs = [t.ids for t in trajs]; masks = [t.mask for t in trajs]
input_ids, comp_mask = trainer._build_batch(seqs, masks)
lp, m = sequence_logprobs_chunked(model, input_ids, comp_mask, 4, 1.0)
lp, m = np.array(lp), np.array(m)
per_turn = {}
ids_np = np.array(input_ids)[:, 1:]
eos_lp = []
for r in range(lp.shape[0]):
    turn, prev = 0, 0.0
    for j in range(lp.shape[1]):
        if m[r, j] > 0 and prev == 0:
            turn += 1
        if m[r, j] > 0:
            per_turn.setdefault(turn, []).append(lp[r, j])
            if ids_np[r, j] == trainer.eos_id:
                eos_lp.append(lp[r, j])
        prev = m[r, j]
allv = np.concatenate([np.array(v) for v in per_turn.values()])
print(f'\n3. fidelity  {len(trajs)} trajectories, {n_comp} completions, {allv.size} credited tokens, '
      f'{secs / GROUPS:.0f} s per group')
print(f'            (expected: mean ~ -entropy, ~0 tokens < -15; before the fix padded rows '
      f'gave mean -1.2..-1.3 and 72/3040 < -15)')
for turn in sorted(per_turn):
    v = np.array(per_turn[turn])
    print(f'   turn {turn}   tokens {v.size:>5}  mean {v.mean():+.3f}  min {v.min():+7.2f}  '
          f'< -10: {(v < -10).sum():>3}  < -15: {(v < -15).sum():>3}')
print(f'   all      tokens {allv.size:>5}  mean {allv.mean():+.3f}  min {allv.min():+7.2f}  '
      f'< -10: {(allv < -10).sum():>3}  < -15: {(allv < -15).sum():>3}')
print(f'   re-tokenisation mismatches: {retok}/{n_comp} completions')
print(f'   completions with surrounding whitespace stripped: {stripped}/{n_comp}')
e = np.array(eos_lp)
print(f'   appended EOS tokens: {e.size}  mean lp {e.mean():+.3f}  min {e.min():+.2f}  < -5: {(e < -5).sum()}')
if (allv < -15).any():
    failures.append(f'fidelity: {(allv < -15).sum()} sampled tokens below -15')

# ---- 4. size of D1 in the single-turn regime ---------------------------------
diffs, lp_d_all = [], []
for g in range(8):
    info = trainer._reset_to_training_hand()
    msgs = trainer._messages_for(info)
    single = apply_chat_template(proc, model.config, msgs, num_images=0)
    double = apply_chat_template(proc, model.config, single, num_images=0)  # what batch_generate built
    out = batch_generate(model, proc, images=None, prompts=[single] * 8, max_tokens=128,
                         sampler=make_sampler(temp=1.0), verbose=False)
    got = CAPTURED[-8:]
    for sid in got:
        sid = sid + [trainer.eos_id] if len(sid) < 128 else sid
        vals = []
        for ctx in (double, single):
            pid = tok.encode(ctx)
            i_, m_ = pad_sequences([pid + sid], [[0.0] * len(pid) + [1.0] * len(sid)], 0)
            l_, mm_ = sequence_logprobs(model, i_, m_, 1.0); mx.eval(l_)
            vals.append(np.array(l_)[0][np.array(mm_)[0] > 0])
        lp_d_all.append(vals[0]); diffs.append(vals[0] - vals[1])
    mx.clear_cache()
d = np.concatenate(diffs); ld = np.concatenate(lp_d_all)
print(f'\n4. D1 size   {d.size} tokens sampled in the double-templated context (64 completions)')
print(f'            lp under the context that sampled them: mean {ld.mean():+.3f}  < -15: {(ld < -15).sum()}')
print(f'            lp_double - lp_single: mean {d.mean():+.4f} nats/token (~KL)  '
      f'|diff| > 1: {(np.abs(d) > 1).sum()}  max |diff| {np.abs(d).max():.2f}')

print('\nRESULT:', 'PASS' if not failures else 'FAIL -- ' + '; '.join(failures))
sys.exit(1 if failures else 0)
