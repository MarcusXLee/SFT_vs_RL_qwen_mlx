"""Acceptance checks for GP-VL multi-turn GRPO, before any VL training run.

GPU: loads the model. Run ONLY when no other GP-VL job holds the GPU
(memory: gpu-exclusive-during-vl-jobs). Model-free checks are in
scripts/test_rl_vl.py.

Every check encodes a lesson from the language run (BUGS.md #26-#28):

  1 guard      the old call shape (templated string) is refused, now with images
  2 positions  every forward of every batch_generate call sits at the exact
               3-D mRoPE positions the scorer computes for those tokens -- the
               RoPE bug lived in rope_deltas, which images make non-zero
  3 fidelity   tokens batch_generate actually sampled score ~ -entropy under the
               VL scorer, on every turn; re-tokenisation, truncation, EOS
  4 scorer     precomputed-input logits == the full model forward's
  5 step-0 KL  ref == old exactly; cur differs only by the known bf16 noise
  6 entropy    pg vs entropy gradient ratio, to set entropy_coef the way GP-L did
  7 memory     peak memory and seconds of one real update() at MB rows/chunk

    MB=1 GROUPS=3 python scripts/verify_vl_grpo.py      (repeat for MB=2, 4)

Writes logs/diagnostics/vl_grpo_verify/mb{MB}.json; exits non-zero on failure.
"""
import copy, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import mlx.core as mx, mlx.nn as nn, numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer.utils import apply_lora_layers
from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from prompt_lib.prompts import Q_GeneralPoint_EQN_VL_EG
from rl.trainer.grpo_trainer_mlx import (grpo_objective, group_advantages, active_groups,
                                         sequence_logprobs, sequence_logprobs_chunked)
from rl.trainer.policy_mlx import token_logprobs
from rl.trainer.grpo_multiturn_mlx import MultiTurnConfig
from rl.trainer.grpo_vl_mlx import MultiTurnVLGRPOTrainer
from scripts.make_sft_data import eval_hands
from scripts.train_grpo_qwen import make_reward_fn

MB = int(os.environ.get('MB', 1))
GROUPS = int(os.environ.get('GROUPS', 3))
ADAPTER = Path(os.environ.get('ADAPTER', ROOT / 'train_ckpt/sft_gp_vl'))
OUT = ROOT / 'logs/diagnostics/vl_grpo_verify'; OUT.mkdir(parents=True, exist_ok=True)
report, failures = {'microbatch': MB, 'groups': GROUPS, 'adapter': str(ADAPTER)}, []

def check(name, ok, detail):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}', flush=True)
    report[name] = {'ok': bool(ok), 'detail': detail}
    if not ok:
        failures.append(name)

# ---- setup: exactly the launcher's GP-VL construction ------------------------
model, proc = load(str(ROOT / 'models/Qwen3-VL-8B-Instruct'))
model = apply_lora_layers(model, str(ADAPTER))
model.config.lora = json.loads((ADAPTER / 'adapter_config.json').read_text())
tok = proc.tokenizer
env = GeneralPointEnv_oneline(language_only=False, resolution=1200, face_cards_color='black',
                              target_points=24, treat_face_cards_as_10=True, verify_iter=5,
                              strip_formula_whitespace=True)
oracle = {'target_number': '24', 'face_card_msg': "'J', 'Q', and 'K' count as '10'."}
cfg = MultiTurnConfig(group_size=8, prompts_per_update=GROUPS, temperature=1.0, max_tokens=128,
                      microbatch=MB, batch_rollout=True, max_turns=5, reward_mode='final',
                      kl_coef=0.01, learning_rate=1e-5, entropy_coef=0.002, seed=0)
trainer = MultiTurnVLGRPOTrainer(
    model=model, processor=proc, env=env, prompt_template=Q_GeneralPoint_EQN_VL_EG,
    oracle_arguments=oracle, reward_fn=make_reward_fn(env), config=cfg,
    holdout=eval_hands(42, 200, target_points=24, treat_face_cards_as_10=True),
    image_dir=OUT / 'rollout_images')
lm = model.language_model

# ---- instrumentation (observe only) ------------------------------------------
det = proc.detokenizer
_reset, _add = det.reset, det.add_token
CAPTURED = []
def _rec_reset(*a, **k):
    CAPTURED.append([]); return _reset(*a, **k)
def _rec_add(t, *a, **k):
    CAPTURED[-1].append(int(t)); return _add(t, *a, **k)
det.reset, det.add_token = _rec_reset, _rec_add

inner = lm.model
_orig_inner = type(inner).__call__
POS = []
def _rec_inner(self, inputs, *a, **k):
    pid = k.get('position_ids')
    if pid is not None:
        p = np.array(pid)
        p = p if p.ndim == 3 else np.broadcast_to(p[None], (3,) + p.shape)   # text: same on all axes
        POS.append((inputs.shape[1], p))
    return _orig_inner(self, inputs, *a, **k)
type(inner).__call__ = _rec_inner

CALLS = []
_orig_bg = trainer._batch_generate
def _rec_bg(conversations, prompt_ids, images=None):
    n0 = len(POS)
    texts = _orig_bg(conversations, prompt_ids, images=images)
    got = CAPTURED[-len(conversations):]
    assert all(tok.decode(g) == t for g, t in zip(got, texts)), 'capture misaligned'
    CALLS.append({'prompt_ids': [list(p) for p in prompt_ids], 'texts': texts, 'ids': got,
                  'image': images[0] if images else None, 'pos': POS[n0:]})
    return texts
trainer._batch_generate = _rec_bg

print(f'== GP-VL GRPO acceptance, microbatch {MB}, {GROUPS} groups of 8\n', flush=True)

# ---- 1. guard ------------------------------------------------------------------
info = trainer._reset_to_training_hand()
q = Q_GeneralPoint_EQN_VL_EG.format(**oracle)
first = trainer._first_payload(q)
ids0, text0 = trainer._templated_ids(first)
img0 = trainer._current_image
try:
    _orig_bg([text0, text0], [ids0, ids0], images=[img0, img0])
    check('1 guard refuses templated string', False, 'accepted')
except RuntimeError as e:
    check('1 guard refuses templated string', True, str(e)[:80])
_orig_bg([copy.deepcopy(first)] * 2, [ids0, ids0], images=[img0, img0])
check('1 guard accepts message list', True,
      f'{len(ids0)} prompt tokens, {ids0.count(trainer.image_token_id)} image tokens')
mx.clear_cache()

# ---- rollout through the real code path -----------------------------------------
CALLS.clear(); POS.clear(); trainer.truncated = 0
trajs, t0 = [], time.time()
for g in range(GROUPS):
    trajs += trainer._rollout_group(trainer._reset_to_training_hand())
    mx.clear_cache()
rollout_s = (time.time() - t0) / GROUPS
report['rollout_seconds_per_group'] = rollout_s
print(f'\nrollout: {len(trajs)} trajectories, {sum(len(c["texts"]) for c in CALLS)} completions, '
      f'{rollout_s:.0f} s per group\n', flush=True)

# ---- 2. positions ----------------------------------------------------------------
filler = tok.encode('a')[0]
bad, n_fwd = [], 0
for ci, c in enumerate(CALLS):
    thw = trainer._image_inputs[c['image']][1]
    total = sum(p.shape[2] for _, p in c['pos'])
    prompt = c['prompt_ids'][0]
    full = mx.array([prompt + [filler] * max(0, total - len(prompt))])
    expected, _ = lm.get_rope_index(full, image_grid_thw=thw)
    expected = np.array(expected)[:, 0, :total]                  # (3, total)
    col = 0
    for n, p in c['pos']:
        n_fwd += 1
        want = expected[:, col:col + n]
        for r in range(p.shape[1]):
            if not np.array_equal(p[:, r, :], want):
                bad.append((ci, col, r)); break
        col += n
check('2 RoPE positions match the scorer', not bad,
      f'{n_fwd} forwards over {len(CALLS)} calls, mismatching forwards: {len(bad)}'
      + (f' (first: call {bad[0][0]}, column {bad[0][1]})' if bad else ''))

# ---- 3. fidelity -----------------------------------------------------------------
trainer.batch_images = [t.image for t in trajs]
ids, cm = trainer._build_batch([t.ids for t in trajs], [t.mask for t in trajs])
for_slice = trainer._prepare_forward(ids)
old, mask = sequence_logprobs_chunked(model, ids, cm, MB, 1.0, for_slice)
old_np, m_np = np.array(old), np.array(mask) > 0
ids_np = np.array(ids)[:, 1:]
per_turn, eos_lp = {}, []
for r in range(old_np.shape[0]):
    turn, prev = 0, False
    for j in range(old_np.shape[1]):
        if m_np[r, j] and not prev:
            turn += 1
        if m_np[r, j]:
            per_turn.setdefault(turn, []).append(old_np[r, j])
            if ids_np[r, j] == trainer.eos_id:
                eos_lp.append(old_np[r, j])
        prev = m_np[r, j]
allv = old_np[m_np]
for turn in sorted(per_turn):
    v = np.array(per_turn[turn])
    print(f'  turn {turn}  tokens {v.size:>5}  mean {v.mean():+.3f}  min {v.min():+7.2f}  '
          f'< -10: {(v < -10).sum():>3}  < -15: {(v < -15).sum():>3}')
retok = sum(1 for c in CALLS for t, i in zip(c['texts'], c['ids'])
            if tok.encode(t.strip()) != i and tok.encode(t) != i)
n_comp = sum(len(c['texts']) for c in CALLS)
report['fidelity'] = {'mean_lp': float(allv.mean()), 'below_10': int((allv < -10).sum()),
                      'below_15': int((allv < -15).sum()), 'tokens': int(allv.size),
                      'retokenisation_mismatch': retok, 'completions': n_comp,
                      'truncated': trainer.truncated, 'eos_mean_lp': float(np.mean(eos_lp))}
check('3 sampled tokens fit the scorer', (allv < -15).sum() == 0,
      f'{allv.size} tokens, mean {allv.mean():+.3f}, < -10: {(allv < -10).sum()}, '
      f'< -15: {(allv < -15).sum()}; retokenisation mismatches {retok}/{n_comp}; '
      f'truncated {trainer.truncated}; EOS mean lp {np.mean(eos_lp):+.3f}')

# ---- 4. scorer == full model forward -----------------------------------------------
gaps = []
for s in range(0, ids.shape[0], MB):
    sl = slice(s, min(s + MB, ids.shape[0]))
    paths = trainer._rows_images[sl]
    pv = mx.concatenate([trainer._image_inputs[p][0] for p in paths], axis=0)
    thw = mx.concatenate([trainer._image_inputs[p][1] for p in paths], axis=0)
    logits = model(ids[sl, :-1], pv, image_grid_thw=thw).logits.astype(mx.float32)
    full = np.array(token_logprobs(logits, ids[sl, 1:]))
    gaps.append(np.abs(full - old_np[sl])[m_np[sl]])
    mx.clear_cache()
gaps = np.concatenate(gaps)
check('4 precomputed scorer == full forward', gaps.max() < 0.05,
      f'|delta lp| mean {gaps.mean():.2e} max {gaps.max():.3f} over {gaps.size} credited tokens')

# ---- 5. step-0 KL ------------------------------------------------------------------
with trainer.reference(model):
    ref, _ = sequence_logprobs_chunked(model, ids, cm, MB, 1.0, for_slice)
ref_np = np.array(ref)
cur_parts = []
for s in range(0, ids.shape[0], MB):
    sl = slice(s, min(s + MB, ids.shape[0]))
    box = {}
    def loss_fn(mdl):
        out = sequence_logprobs(mdl, ids[sl], cm[sl], 1.0, want_entropy=True,
                                logits_fn=for_slice(sl))
        box['lp'] = out[0]
        return (out[0] * out[1]).sum()
    _, g = nn.value_and_grad(model, loss_fn)(model); mx.eval(g, box['lp'])
    cur_parts.append(np.array(mx.stop_gradient(box['lp']))); mx.clear_cache()
cur_np = np.concatenate(cur_parts, axis=0)
d_ref, d_cur = (ref_np - old_np)[m_np], (cur_np - old_np)[m_np]
k3 = np.exp(-d_cur) + d_cur - 1
check('5 step-0: ref == old', np.abs(d_ref).max() == 0.0, f'max |ref-old| {np.abs(d_ref).max():.3g}')
report['step0_cur_vs_old'] = {'mean_abs': float(np.abs(d_cur).mean()),
                              'max_abs': float(np.abs(d_cur).max()),
                              'k3_per_token': float(k3.mean())}
print(f'  [info] step-0 cur vs old: mean |d| {np.abs(d_cur).mean():.2e}, max {np.abs(d_cur).max():.3f}, '
      f'k3/token {k3.mean():.2e} (GP-L: 3.9e-3 mean, 1.46e-4 k3/token)', flush=True)

# ---- 6. entropy gradient ratio -----------------------------------------------------
rewards = np.array([trainer._traj_reward(t) for t in trajs], np.float32).reshape(GROUPS, 8)
keep = active_groups(rewards)
report['group_rewards'] = rewards.tolist()
if keep.any():
    # As in GP-L: only groups with reward spread enter the update, so only their
    # rows count. Inactive rows are masked out inside each precomputed chunk.
    active = np.repeat(keep, 8).astype(np.float32)
    adv_all = group_advantages(rewards).reshape(-1)
    denom = float(active.sum())

    def grad_norm(entropy_only):
        total = None
        for s in range(0, ids.shape[0], MB):
            sl = slice(s, min(s + MB, ids.shape[0]))
            act = mx.array(active[sl])
            if not float(act.sum().item()):
                continue
            adv = mx.zeros(sl.stop - sl.start) if entropy_only else mx.array(adv_all[sl]) * act
            cm_sl = cm[sl] * act[:, None]
            def lf(mdl):
                out = sequence_logprobs(mdl, ids[sl], cm_sl, 1.0, want_entropy=True,
                                        logits_fn=for_slice(sl))
                return grpo_objective(out[0], old[sl], ref[sl], adv, out[1], denom,
                                      kl_coef=0.0, entropy=out[2],
                                      entropy_coef=0.01 if entropy_only else 0.0,
                                      normalization='trajectory')[0]
            _, g = nn.value_and_grad(model, lf)(model); mx.eval(g)
            total = g if total is None else tree_map(lambda a, b: a + b, total, g)
            mx.eval(total); mx.clear_cache()
        return float(np.sqrt(sum(float((v.astype(mx.float32) ** 2).sum().item())
                                 for _, v in tree_flatten(total))))

    g_pg, g_ent = grad_norm(False), grad_norm(True)
    ratio = g_ent / g_pg if g_pg else float('nan')
    coef = 0.01 * 0.0006 / ratio if ratio else float('nan')
    report['entropy'] = {'grad_pg': g_pg, 'grad_entropy_at_0.01': g_ent, 'ratio': ratio,
                         'coef_matching_validated_ratio': coef, 'gp_l_coef': 0.002,
                         'active_rows': int(denom)}
    print(f'  [info] |grad| pg {g_pg:.3f}, entropy@0.01 {g_ent:.5f}, ratio {ratio:.4f} '
          f'(GP-L multi-turn 0.0033) -> coef reproducing the validated 0.0006: {coef:.4f} '
          f'({int(denom)} active rows)', flush=True)
else:
    print('  [info] no group with reward spread; entropy ratio not measured', flush=True)

# ---- 7. memory and time of one real update ---------------------------------------
trainer.batch_images = [t.image for t in trajs]
mx.reset_peak_memory(); t0 = time.time()
metrics = trainer.update([t.ids for t in trajs], [t.mask for t in trajs], rewards)
upd_s, peak = time.time() - t0, mx.get_peak_memory() / 1e9
report['update'] = {'seconds': upd_s, 'peak_gb': peak,
                    'skipped': metrics.get('skipped'), 'kl_per_token': metrics.get('kl_per_token'),
                    'grad_norm': metrics.get('grad_norm')}
check('7 one update fits in memory', not metrics.get('skipped') and peak < 100,
      f'peak {peak:.1f} GB (limit ~107.5), {upd_s:.0f} s, groups used '
      f'{metrics.get("groups_used")}, kl/tok {metrics.get("kl_per_token")}')

(OUT / f'mb{MB}.json').write_text(json.dumps(report, indent=2, default=float) + '\n')
print(f'\nRESULT: {"PASS" if not failures else "FAIL -- " + ", ".join(failures)}')
sys.exit(1 if failures else 0)
