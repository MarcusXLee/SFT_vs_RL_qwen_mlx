"""Why is KL non-zero at step 0, when policy and reference weights are identical?

Pilot after the rollout fix (BUGS.md #26), update 0: ref_lp - old_lp was
exactly 0.0 on every credited token (d_max = d_min = 0), yet the reported kl
sum was 1.13 (1.6e-4 per token). The penalty's d is ref - CUR, and cur is
computed inside nn.value_and_grad, with entropy, while old/ref come from the
chunked no-grad pass. So the gap is between two forward paths on the same
weights. This localises it, on a real multi-turn batch from the fixed rollout:

  A   sequence_logprobs_chunked            (how old_lp / ref_lp are computed)
  A2  the same again                       (is the no-grad path deterministic?)
  C   plain forward, want_entropy=True     (does computing entropy change lp?)
  D   inside value_and_grad, entropy on    (how cur_lp is computed in update())
  E   inside value_and_grad, entropy off

Numerical noise predicts |A - D| of order 1e-2 nats (bf16; C1 vs C2 in
diagnose_sampler_scorer measured 1.45e-2) with no structure; a real defect
would show large, structured differences.
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import mlx.core as mx, mlx.nn as nn, numpy as np
from mlx_vlm import load
from mlx_vlm.trainer.utils import apply_lora_layers
from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG
from rl.trainer.grpo_trainer_mlx import sequence_logprobs, sequence_logprobs_chunked
from rl.trainer.grpo_multiturn_mlx import MultiTurnConfig, MultiTurnGRPOTrainer
from scripts.train_grpo_qwen import make_reward_fn

model, proc = load(str(ROOT / 'models/Qwen3-VL-8B-Instruct'))
adapter = ROOT / 'train_ckpt/sft_gp_language'
model = apply_lora_layers(model, str(adapter))
model.config.lora = json.loads((adapter / 'adapter_config.json').read_text())
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
MB = trainer.cfg.microbatch

trajs = []
for _ in range(2):
    trajs += trainer._rollout_group(trainer._reset_to_training_hand())
    mx.clear_cache()
ids, cmask = trainer._build_batch([t.ids for t in trajs], [t.mask for t in trajs])
n = ids.shape[0]

A, mask = sequence_logprobs_chunked(model, ids, cmask, MB, 1.0)
A2, _ = sequence_logprobs_chunked(model, ids, cmask, MB, 1.0)
with trainer.reference(model):
    R, _ = sequence_logprobs_chunked(model, ids, cmask, MB, 1.0)
mx.eval(A, A2, R)

def per_chunk(fn):
    parts = []
    for s in range(0, n, MB):
        sl = slice(s, min(s + MB, n))
        parts.append(fn(sl)); mx.clear_cache()
    return mx.concatenate(parts, axis=0)

def plain(sl):
    lp = sequence_logprobs(model, ids[sl], cmask[sl], 1.0, want_entropy=True)[0]
    mx.eval(lp); return lp

def in_grad(entropy):
    def run(sl):
        box = {}
        def loss_fn(m):
            out = sequence_logprobs(m, ids[sl], cmask[sl], 1.0, want_entropy=entropy)
            box['lp'] = out[0]
            return (out[0] * out[1]).sum()
        loss, g = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, g, box['lp'])
        return mx.stop_gradient(box['lp'])
    return run

C = per_chunk(plain)
D = per_chunk(in_grad(True))
E = per_chunk(in_grad(False))

m = np.array(mask) > 0
a = np.array(A)[m]
print(f'{len(trajs)} trajectories, {m.sum()} credited tokens\n')
print(f'ref - old (A) on identical weights: max |d| {np.abs(np.array(R)[m] - a).max():.3g}\n')
for name, X in (('A2 no-grad again        ', A2), ('C  plain, entropy on    ', C),
                ('D  value_and_grad, ent  ', D), ('E  value_and_grad, no ent', E)):
    d = np.array(X)[m] - a
    k3 = np.exp(-d) - (-d) - 1          # k3 with d_pen = ref - cur, ref == A
    print(f'{name}  |X - A| mean {np.abs(d).mean():.2e}  max {np.abs(d).max():.3f}  '
          f'> 0.1: {(np.abs(d) > 0.1).sum():>4}  > 1: {(np.abs(d) > 1).sum():>2}   '
          f'k3 sum {k3.sum():.3f} ({k3.sum() / m.sum():.2e}/tok)')
dD = np.array(D)[m] - a
worst = np.argsort(-np.abs(dD))[:5]
print('\nlargest A-vs-D gaps (lp under A, gap):')
for w in worst:
    print(f'  lp {a[w]:+8.3f}   gap {dD[w]:+.3f}')
print(f'\ncorrelation of |gap| with -lp (tail tokens noisier?): '
      f'{np.corrcoef(np.abs(dD), -a)[0, 1]:+.2f}')
