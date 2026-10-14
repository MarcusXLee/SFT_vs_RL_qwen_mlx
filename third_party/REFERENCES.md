# RL reference implementations

Cloned for reference while porting the PPO stack to MLX. **Not dependencies** —
the clones are gitignored; this file pins what was read.

| repo | pinned SHA | dated | license | why |
| --- | --- | --- | --- | --- |
| [andrew-silva/mlx-rlhf](https://github.com/andrew-silva/mlx-rlhf) | `95ff066` | 2024-06-15 | MIT | the only MLX **PPO** with a critic; a port of TRL's `PPOTrainer` |
| [Cargo-Cult-Research/mlx-rl](https://github.com/Cargo-Cult-Research/mlx-rl) | `a553616` | 2026-09-18 | MIT | current-MLX GRPO with verifiable rewards + LoRA; numerically careful |

Re-clone with:

```bash
git clone https://github.com/andrew-silva/mlx-rlhf.git third_party/mlx-rlhf && \
  git -C third_party/mlx-rlhf checkout 95ff06600b8937909b135befd28c24da4a2802cc
git clone https://github.com/Cargo-Cult-Research/mlx-rl.git third_party/mlx-rl && \
  git -C third_party/mlx-rl checkout a553616e859d53ae20bbbd70bf9b36a63de7f122
```

## Why neither is adopted wholesale

Both are **text-only** (built on `mlx-lm`). Neither handles a VLM, so neither can
be installed and pointed at Qwen3-VL.

More fundamentally, both implement a *different algorithm* from the one this
paper uses, despite all three being called "PPO/GRPO with GAE":

| | mlx-rlhf / mlx-rl | this repo (`rl/trainer/storage_tz.py`) |
| --- | --- | --- |
| episode | one prompt -> one response (bandit) | multi-step trajectory in `gym_cards` |
| GAE runs over | generated **tokens** within a response | **environment steps** |
| value head predicts | per-token value | one scalar per env step |
| `gamma` / `lambda` | discount across tokens | discount across env steps (0.9 / 0.95) |
| episode boundaries | n/a | `masks`, plus `bad_masks` for time limits |

Verified directly: `mlx_ppo_trainer.py:836` sets `gen_len = rewards.shape[-1]`
and loops GAE over that axis — response tokens. `storage_tz.py:compute_returns`
loops over `self.rewards.size(0)` = `num_steps` env steps and multiplies by
`self.masks[step+1]`. `mlx-rl` is documented as single-turn only.

`storage_tz.py` is the `a2c_ppo_acktr` (Kostrikov) `RolloutStorage`, i.e. classic
trajectory RL — the RL4VLM lineage this repo descends from, not RLHF.

## What is taken from each

- **mlx-rl `grpo.py`** — MLX idioms and numerical hazards that are expensive to
  rediscover: masking *inside* the ratio exponent (padded logprobs otherwise
  overflow and `inf * 0 = nan`), the k3 KL estimator with a clamped penalty but
  unclamped metric, a global token denominator so microbatch losses sum
  correctly under gradient accumulation, and dropping zero-variance groups
  (their fp16 kernel noise is rescaled by Adam into a full-size random step).
- **mlx-rl `models.py:124-135`** — a context manager that zeroes every adapter's
  multiplicative `scale`, making the forward pass the frozen base model. This is
  the KL reference policy **for free**, with no second copy of the weights.
  `mlx_vlm`'s `LoRaLayer` has the same `scale` attribute, so it ports directly.
- **mlx-rlhf `mlx_ppo_trainer.py`** — the shape of a critic-bearing PPO update in
  MLX (whitening, clipped value loss, microbatching). Its `mx.*` surface was
  checked against the installed MLX: 12 attributes used, **0 missing**, so
  despite its age the array API still holds.

## Plan

Port `storage_tz.py` and `algo/ppo.py` from this repo (torch -> MLX), keeping
their trajectory structure, and borrow the numerical treatment above. GRPO first
as a correctness smoke test, then the critic for PPO.
