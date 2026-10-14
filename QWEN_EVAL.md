# Evaluating Qwen3-VL on GeneralPoints

This repository ships a Llama-3.2-Vision pipeline that expects CUDA and DeepSpeed.
This document covers the Qwen3-VL path added for a local Apple-silicon machine:
what it changes, how to run it, and how to read the results.

Written for step 1 of the reproduction (baseline evaluation). SFT and RL are not
covered here; see "Not done yet" at the bottom.

---

## 1. Why the numbers moved

The environment's verifier tokenises the answer and rejects it when
`tokens_str != current_formula`. A space is an illegal character, so

```json
{"cards": ["K","5","2","4"], "number": [10,5,2,4], "formula": "10 + 5 * 2 + 4 = 24"}
```

scores **−3 (ILLEGAL_FORMULA)** even though `10 + 5*2 + 4` is exactly 24.

Upstream never hit this: its checkpoint was SFT'd into emitting no spaces.
Qwen3-VL-8B-Instruct spaces its formulas by default, so on the shipped verifier
it scores 0% — a formatting result being read as a reasoning result.

Two further consequences, both visible in the logs:

- The feedback string is `"you failed this trial because illegal characters."`
  It never says *remove the spaces*, so the model re-emits the same answer until
  the retry budget is gone (6.00 of 6 steps used per trajectory, every time).
- Reward is constant across rollouts, so there is no signal for RL to climb.

## 2. The ablation

Every GeneralPoints evaluation is run twice, changing one flag and nothing else:

| arm | `strip_formula_whitespace` | meaning |
| --- | --- | --- |
| `strict` | `False` (default) | verifier exactly as shipped; a space is fatal |
| `tolerant` | `True` | whitespace removed before validation; arithmetic judged on its merits |

**`tolerant` is the headline condition** for this reproduction, because it
measures reasoning rather than punctuation, and it lands in the same band as the
paper's Init checkpoint. **`strict` is reported alongside it**, not discarded, so
the formatting tax is a stated finding rather than a silent correction.

`strip_formula_whitespace=False` is a no-op branch, so the `strict` arm is
byte-identical to upstream behaviour. This is pinned by
`test_default_is_upstream_behaviour`.

The flag never removes whitespace **between two digits**: `"1 0"` must not become
`"10"`, or a malformed answer would be upgraded into a correct one. Pinned by
`test_tolerant_does_not_merge_split_digits`. (Checked against every formula
produced so far: 0 of 233 contained a digit-space-digit, so this refinement
changed no existing result.)

## 3. Protocol

Both arms run the repository's own evaluation loop — `QwenEvaluator` subclasses
`BaseEvaluator` and overrides only model loading and generation, so the protocol
is the published one:

- `verify_iter: 5` — up to 5 retries per puzzle
- the verifier's message is appended to the same conversation after each failure
- success is scored **per trajectory**, not per attempt
- greedy decoding (`temperature: 0.0`), `max_new_tokens: 256`, seed 42

This matters: `scripts/baseline/` measures something different — one attempt, no
feedback. On the same puzzles, first-try success is 10% while
success-within-5-tries is 35%. A single-shot harness cannot reproduce the
paper's numbers regardless of the whitespace question.

Comparison rigor: both arms draw the identical puzzle stream (the environment
draws cards from Python's `random`, seeded once, and the model's answers never
consume it). `analyze_gp_log.py compare` **refuses to print a table** unless the
two logs' puzzle sequences match element-for-element.

## 4. Running it

```bash
# both arms + comparison table (this is the normal entry point)
bash scripts/run_qwen_gp_ablation.sh evaluation/configs/qwen_gp_language.yaml 50
bash scripts/run_qwen_gp_ablation.sh evaluation/configs/qwen_gp_vl.yaml 50

# one arm only
bash scripts/run_qwen_gp_eval.sh evaluation/configs/qwen_gp_language.yaml \
     --num_traj=20 --env_config.strip_formula_whitespace=True

# read a log
.venv-qwen/bin/python scripts/analyze_gp_log.py report  <log.jsonl>
.venv-qwen/bin/python scripts/analyze_gp_log.py compare <strict.jsonl> <tolerant.jsonl>

# tests
.venv-qwen/bin/python -m unittest discover -s scripts -p 'test_qwen_*.py'
```

The ablation runner refuses to write into an existing results directory, so old
and new runs are never mixed.

## 5. Environments

Two deliberately separate virtualenvs; neither can hold both stacks.

| venv | holds | used for |
| --- | --- | --- |
| `.venv` | torch, transformers, gym, peft | the original Llama/CUDA code paths |
| `.venv-qwen` | mlx, mlx-vlm, gymnasium, box | everything in this document |

`utils_general.py`, `base_evaluator.py` and `launcher.py` now guard their
`torch` / `accelerate` / `wandb` imports so the repository loads under
`.venv-qwen`. On a CUDA machine these guards are inert.

The evaluator registry imports each backend defensively and reports which
dependency failed rather than crashing at import time, since no single machine
is expected to have both backends installed.

## 6. Files

| path | status | purpose |
| --- | --- | --- |
| `evaluation/evaluator/qwen_evaluator.py` | new | MLX-backed evaluator; inherits the paper's trajectory loop |
| `evaluation/configs/qwen_gp_language.yaml` | new | GP-L config |
| `evaluation/configs/qwen_gp_vl.yaml` | new | GP-VL config |
| `scripts/run_qwen_gp_ablation.sh` | new | both arms + comparison (normal entry point) |
| `scripts/run_qwen_gp_eval.sh` | new | single arm |
| `scripts/analyze_gp_log.py` | new | report / compare |
| `scripts/test_qwen_eval.py` | new | regression tests |
| `scripts/make_sft_data.py` | new | GP SFT data from the env solver, eval hands held out |
| `scripts/train_sft_qwen.py` | new | LoRA SFT via `mlx_vlm.trainer` |
| `scripts/train_grpo_qwen.py` | new | GRPO launcher |
| `rl/trainer/storage_mlx.py` | new | rollout buffer + GAE over env steps (MLX port) |
| `rl/trainer/policy_mlx.py` | new | LoRA policy, reference-policy trick, value head |
| `rl/trainer/grpo_trainer_mlx.py` | new | single-turn GRPO |
| `rl/trainer/__init__.py` | rewritten | defensive backend registry |
| `scripts/test_rl_storage.py` | new | differential test vs the torch storage |
| `scripts/test_rl_grpo.py` | new | GRPO objective unit tests |
| `scripts/test_rl_policy.py` | new | reference-policy test against the real model |
| `gym/.../general_points_oneline.py` | +10 lines | opt-in `strip_formula_whitespace`, default off |
| `utils_rl.py` | +`exact_eval` | exact rational arithmetic (`BUGS.md` #2) |
| `evaluation/evaluator/__init__.py` | rewritten | defensive backend registry |
| `evaluation/evaluator/base_evaluator.py` | guarded imports | load without torch/accelerate |
| `evaluation/launcher.py` | guarded imports | dropped unused `wandb` |
| `utils_general.py` | guarded import | `init_seed` degrades without torch |

Model weights are read from `models/Qwen3-VL-8B-Instruct` (local, offline).

## 7. Results

Qwen3-VL-8B-Instruct, no training of any kind. 50 trajectories per arm, seed 42,
in-distribution, `verify_iter: 5`. Both arms of each pair saw the identical
puzzle stream (asserted by the comparison tool).

|                          | GP-L strict | GP-L tolerant | GP-VL strict | GP-VL tolerant |
| ------------------------ | ----------: | ------------: | -----------: | -------------: |
| **Success (per traj)**   |    **4.0%** |     **34.0%** |     **6.0%** |      **38.0%** |
| solved on 1st try        |        2.0% |          8.0% |         6.0% |          18.0% |
| card recognition acc     |      100.0% |        100.0% |       100.0% |         100.0% |
| steps used per traj      |        5.84 |          4.84 |         5.70 |           4.48 |
| `ILLEGAL_FORMULA` steps  |       74.3% |          0.0% |        78.2% |           0.0% |

The GP-VL columns are a **re-run after `BUGS.md` #9**; the earlier figures
(0% / 30%, recognition ~69%) were an artefact of valid JSON being scored as a
card misread. The superseded run is kept under
`logs/ablation/INVALID-bug9__qwen_gp_vl__init-n20/`.

Reading these:

- **The formatting tax is the whole baseline.** GP-L 4.0% → 34.0%, GP-VL
  6.0% → 38.0%, with model, prompts, puzzles and seed held fixed.
- **Card recognition is perfect**, 100% in every arm. Qwen3-VL reads the
  rendered cards without error, so the vision channel costs nothing here. The
  earlier "vision costs ~31 points of recognition" claim was the bug, and is
  withdrawn.
- **Vision is not harder than language on this task.** 19/50 vs 17/50 is well
  inside noise, so GP-L and GP-VL are statistically indistinguishable. Since
  recognition is perfect, both modes are measuring the same arithmetic ability.
  Do not report GP-VL as either better or worse than GP-L from this data.
- **`strict` rarely terminates early** (5.84 and 5.70 of 6 steps): the model is
  never told spaces are the problem, so it repeats itself until the budget is
  gone. `tolerant` averages 4.84 / 4.48.
- **Retries matter as much as formatting.** GP-L goes 8% → 34% between the
  first attempt and the fifth, GP-VL 18% → 38%.

Against the paper's Init checkpoint (GP-L ≈ 23–28%, GP-VL ≈ 23%, card
recognition ≈ 70–80%), the **tolerant** column lands in band on success and the
**strict** column lands nowhere near it — the evidence for treating tolerant as
the headline condition. Card recognition is well above the paper's ≈70–80%,
which is a property of Qwen3-VL rather than of the protocol.

### Step 1 result: SFT (GP-L), n = 50

LoRA SFT (r=8, 250 iters, 900 examples with the evaluation hands held out;
`scripts/train_sft_qwen.py`), evaluated on 50 puzzles. Both arms of each pair
saw the identical puzzle stream.

|                          | Init strict | Init tolerant | SFT strict | SFT tolerant |
| ------------------------ | ----------: | ------------: | ---------: | -----------: |
| **Success (per traj)**   |    **4.0%** |     **34.0%** |  **22.0%** |    **22.0%** |
| solved on 1st try        |        2.0% |          8.0% |      20.0% |        20.0% |
| `ILLEGAL_FORMULA` steps  |       74.3% |          0.0% |       0.0% |         0.0% |
| steps used per traj      |        5.84 |          4.84 |       4.94 |         4.94 |
| card recognition         |      100.0% |        100.0% |     100.0% |       100.0% |

These are consistent with the n=20 pilot (0/35/20/20), so the pilot was not a
fluke of sample size.

**What is established:**

1. **SFT solves formatting completely.** `ILLEGAL_FORMULA` 74.3% -> 0%, and the
   strict and tolerant runs are identical (same 247 steps, same rates). The
   model no longer emits a space, so the ablation flag has nothing to act on.
   Strict success rises 4.0% -> 22.0% with no information about the verifier,
   purely from imitating space-free targets.

2. **SFT destroys the ability to revise.** This is the strongest signal in the
   data. Counting only trajectories not solved on the first attempt:

   | | still open after try 1 | recovered by try 5 | recovery rate |
   | --- | ---: | ---: | ---: |
   | Init (tolerant) | 46 | 13 | **28.3%** |
   | SFT (tolerant) | 40 | 1 | **2.5%** |

   Fisher exact, two-sided: **p = 0.0011**. The Init model uses the verifier's
   feedback; the SFT model effectively cannot. Its 20% first-try rate is almost
   its entire final 22%.

**What is NOT established:** the apparent drop in raw accuracy, tolerant 34% ->
22%, is **not** statistically significant at this sample size (z = 1.34,
p = 0.18). Do not claim SFT reduces problem-solving ability from this data.

> **Update at n = 200** (see "GP-L at n = 200, in-distribution and rule-OOD"
> below): the drop is now established. SFT scores 18.0% against Init's 33.0%;
> on the same hands Init alone solves 46 and SFT alone solves 16, McNemar
> p = 0.0002. SFT still wins on the first attempt (21 vs 9, p = 0.043). It loses
> on revision: of 161 hands both failed at the first try, Init recovered 39 and
> SFT 6. The
claim the data does support is narrower and more interesting: SFT buys format
compliance and first-attempt accuracy at the cost of *revision*.

That is a concrete instance of "SFT memorises" — the model learned to emit one
well-formed answer rather than to improve one — and revision under verifier
feedback is exactly what RL optimises.

### Step 2 result: multi-turn GRPO from SFT (GP-L), n = 50

This is the first complete and correct RL run: `train_ckpt/mt_grpo_v5`, 60/60
updates, each taking a gradient step. It starts from the SFT checkpoint and uses
`reward=final`, G=8, 4 hands per update, lr 1e-5, KL 0.01, entropy 0.002, seed
0, and about 8.6 h wall-clock. It ran after the rollout fix (BUGS.md #26), so
runs v1–v4 are superseded. Only the pre-planned **final** checkpoint was
evaluated; there was no checkpoint selection. All four logs saw the identical
50 hands (asserted).

| | 1st try | solved | rate | recovered after a 1st-try failure |
| --- | ---: | ---: | ---: | ---: |
| Init | 4 | 17 | 34% | 13/46 |
| SFT | 10 | 11 | 22% | 1/40 |
| single-turn GRPO, best | 5 | 7 | 14% | 2/45 |
| **multi-turn GRPO v5** | **11** | **25** | **50%** | **14/39** |

Paired exact McNemar on final success (discordant hands):

- **v5 vs SFT: 16 vs 2, p = 0.0013**
- v5 vs single-turn best: 19 vs 1, p < 0.0001
- v5 vs Init: 11 vs 3, p = 0.057 (not significant at 0.05)

> **Corrected at n = 200** (section below). Two statements here were limits of
> n = 50:
> - v5 **is** significantly better than Init (p = 0.0011).
> - The gain is **mostly, not only**, revision: first-try success also rises
>   against SFT (44 vs 30, p = 0.044).

**Where the gain comes from: revision, not first attempts.**
- First-try success is unchanged against SFT: 11 vs 10, discordant 8 vs 7,
  p = 1.0.
- On the 32 hands that both SFT and v5 failed at the first try, v5 recovered 9
  and SFT recovered 0 (paired p = 0.0039).
- v5's solves spread across attempts 1–5 (11/6/3/3/2); SFT's are 10 on the
  first try and 1 on the third.
- SFT had cut recovery from Init's 28% to 2.5%. Multi-turn GRPO restores it to
  36% (14/39), which is at or above Init.

**What this does not establish.**
- One run, one seed, n = 50.
- v5 is not significantly better than Init.
- These are in-distribution hands: unseen during training, but under the same
  rule. The paper's central claim, "RL generalises", needs the OOD evaluations,
  which are the next stage.
- The eval is greedy (T = 0), while training sampled at T = 1.

### GP-L at n = 200, in-distribution and rule-OOD

`scripts/run_gp_l_indist_ood.sh` runs the evaluation and
`scripts/compare_gp_runs.py` the analysis. Settings: tolerant, greedy, seed 42,
`verify_iter` 5, n = 200 per arm. Within each setting all three models saw the
identical stream (asserted).

- **In-distribution** uses the training rule, J/Q/K = 10. It is exactly the
  stream whose 184 distinct hands SFT and RL training withheld, so every hand
  is unseen. Its first 50 trajectories are the earlier n = 50 set, and all
  three models reproduce their n = 50 outcomes on them exactly (50/50 each).
- **Rule-OOD** follows upstream `scripts/gp_evaluation/language_ood_eval.sh`:
  `ood=True` (every hand holds a J, Q or K) and
  `treat_face_cards_as_10=False` (they count as 11, 12 and 13, and the prompt
  says so). No model was ever trained under this rule. The verifier accepted
  the environment's own solutions on 40/40 OOD hands before this run.

| | in-dist solved | in-dist 1st try | in-dist recovered | **OOD solved** | OOD 1st try | OOD recovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Init | 66 (33.0%) | 18 | 48/182 | **40 (20.0%)** | 11 | 29/189 |
| SFT | 36 (18.0%) | 30 | 6/170 | **22 (11.0%)** | 15 | 7/185 |
| **multi-turn GRPO v5** | **92 (46.0%)** | 44 | 48/156 | **64 (32.0%)** | 26 | 38/174 |

Paired exact McNemar on final success (hands only the first model solved, vs
only the second):

| pair | in-distribution | rule-OOD |
| --- | --- | --- |
| v5 vs SFT | 62 vs 6, **p < 0.0001** | 50 vs 8, **p < 0.0001** |
| v5 vs Init | 43 vs 17, **p = 0.0011** | 39 vs 15, **p = 0.0015** |
| SFT vs Init | 16 vs 46, **p = 0.0002** (SFT worse) | 13 vs 31, **p = 0.0096** (SFT worse) |

Relative to Init:

| | in-distribution | rule-OOD |
| --- | ---: | ---: |
| SFT | −15.0 points | −9.0 points |
| RL (v5) | +13.0 points | +12.0 points |

**Reading these:**

- **The paper's GP-L pattern reproduces.** Under a rule it was never trained
  on, RL improves on Init and SFT degrades it, both significantly.
- **Recovery after a first-try failure** is where SFT collapses and RL rebuilds.
  On hands both models failed first time, v5 recovers 37/142 against SFT's
  3/142 in-distribution, and 35/165 against 5/165 OOD.
- **v5 also gains first-try accuracy.** Against Init: 38 vs 12 in-distribution
  (p = 0.0003) and 19 vs 4 OOD (p = 0.0026). So the RL policy is better at the
  arithmetic, not only at using feedback.

**Prompt-format caveat (BUGS.md #28).** The language SFT data used a prompt one
newline longer at each end than the evaluation prompt. Re-evaluated with its
own training prompt on the first 176 in-distribution hands, SFT solves 34
against 30 (p = 0.39, not significant). v5 still beats it 47 vs 5
(p = 1.3e-9), and Init still beats it 35 vs 15 (p = 0.0066). No conclusion
depends on the mismatch.

**Deviation from the paper.** Here SFT is below Init even in-distribution. In
the paper, SFT improves in-distribution performance. This is a property of our
SFT stage (Qwen3-VL-Instruct, LoRA r=8, 250 iterations on 900 examples), not of
the evaluation, and should be reported as such.

**Not established.**
- One RL seed and one SFT run.
- Language only; GP-VL and its visual OOD are pending.
- Tolerant verifier; greedy decoding. Of the 200 rule-OOD trajectories, 171 are
  distinct hands.

### GP-VL at n = 200: Init vs SFT, in-distribution, rule-OOD and visual-OOD

`scripts/run_gp_vl_eval_suite.sh` runs the evaluation and
`scripts/compare_gp_runs.py` the analysis. Settings: tolerant, greedy, seed
42, 1200 px. The GP-VL SFT was trained as the GP-L SFT, on the same hands.
Settings follow upstream `vl_*_eval.sh`:
- in-distribution: black face cards, J/Q/K = 10;
- rule-OOD: J/Q/K = 11/12/13;
- visual-OOD: red face cards.

All arms completed 200/200 trajectories. There were no card-recognition
errors, and `ILLEGAL_FORMULA` was ~0% for SFT.

| | Init | SFT | final success, only Init vs only SFT | first try (Init / SFT) | recovery on hands both failed first |
| --- | ---: | ---: | --- | --- | --- |
| in-distribution | 36.5% | 27.5% | 37 vs 19, **p = 0.022** | 25 / 32 (n.s.) | 35 vs 16, **p = 0.003** |
| rule-OOD | 19.5% | 19.0% | 28 vs 27, p = 1.0 | 11 / 27, **p = 0.009** | 21 vs 10, p = 0.06 |
| visual-OOD | 39.0% | 24.0% | 43 vs 13, **p = 0.0001** | 26 / 31 (n.s.) | 36 vs 12, **p = 0.0002** |

**Reading these.**
- The GP-L pattern repeats: SFT raises first-attempt accuracy and loses
  revision. Its solves sit on attempt 1 (32 of 55 in-distribution), while
  Init's spread across attempts 1-5.
- SFT is below Init in-distribution and on visual-OOD. It breaks even on
  rule-OOD, where a large first-try gain cancels the lost recovery.
- Visual-OOD costs SFT 3.5 points relative to in-distribution
  (27.5 → 24.0). Init does not drop (36.5 → 39.0).

The deviation from the paper noted for GP-L applies here too: our SFT does not
improve in-distribution performance.

### GP-VL GRPO result (n = 200)

**The run.** `train_ckpt/mt_grpo_vl_v1`:
- recipe: GP-L v5's, with `--vision`, initialised from the GP-VL SFT,
  microbatch 1;
- 60/60 updates took a gradient step, with no gradient clipping and no
  skipped updates; 22.2 h;
- KL per token levelled off at 0.045-0.06, as v5's did;
- max |d| was 8.9, so the ±10 clamp never acted. 38 updates had a few tokens
  beyond ±5, which the old ±5 clamp would have stripped of their KL brake;
- entropy fell 0.138 → ~0.065;
- 12 of the run's completions hit the 128-token limit.

The table compares the final checkpoint on the same 200 hands per setting as
Init and SFT, with paired exact McNemar tests on final success:

| setting | Init | SFT | **RL** | RL vs SFT | RL vs Init |
| --- | ---: | ---: | ---: | --- | --- |
| in-distribution | 36.5% | 27.5% | **46.5%** | 52 vs 14, p < 0.0001 | 51 vs 31, **p = 0.035** |
| rule-OOD (J/Q/K = 11/12/13) | 19.5% | 19.0% | **28.0%** | 35 vs 17, p = 0.018 | 34 vs 17, **p = 0.024** |
| visual-OOD (red face cards) | 39.0% | 24.0% | **45.0%** | 53 vs 11, p < 0.0001 | 45 vs 33, p = 0.21 (n.s.) |

Change relative to Init:

| | in-distribution | rule-OOD | visual-OOD |
| --- | ---: | ---: | ---: |
| SFT | −9.0 | −0.5 | −15.0 |
| RL | +10.0 | +8.5 | +6.0 (n.s.) |

**Reading these.**
- The paper's GP-VL claims reproduce in direction:
  - RL improves on Init in-distribution and under the unseen rule;
  - SFT is hurt most by the visual shift (−15 points), while RL is not hurt
    by it.
- RL vs Init on visual-OOD is a positive but non-significant +6.

**A difference from GP-L: the RL policy is worse on the first attempt.**
- In-distribution, GP-VL RL solves 14 hands on the first try, against 25 for
  Init and 32 for SFT (RL vs SFT 7 vs 25, p = 0.002). On visual-OOD it is
  12 vs 26 for Init (p = 0.016).
- Its solves peak at attempt 3 (14/19/32/15/13 over attempts 1-5). The whole
  gain is recovery: on hands both failed first time, RL recovers 62/161
  against SFT's 21/161.
- GP-L v5 instead raised first-try accuracy as well (44 vs 30 against SFT).
- A plausible cause, not verified: the `final` reward pays +5 whether the
  hand is solved on attempt 1 or attempt 5, so nothing in the objective
  favours the first attempt. `sum` (summed step rewards) would. This matters
  for any claim about *how* RL helps, not for the success-rate comparison.

### GP-VL GRPO: what is shared with GP-L, what differs, and how each was checked

`rl/trainer/grpo_vl_mlx.py` subclasses the multi-turn trainer. Acceptance
checks are in `scripts/verify_vl_grpo.py` (GPU) and `scripts/test_rl_vl.py`
(model-free). Logs are in `logs/diagnostics/vl_grpo_verify*`.

**Shared with GP-L, unchanged.** The whole RL algorithm and every
hyperparameter carry over from GP-L v5:
- objective and loss: clipped surrogate (0.2), k3 KL at 0.01 with the ±10
  clamp, trajectory normalisation, no std scaling of advantages;
- rollout and reward: reward `final`, G = 8, 4 hands per update, dynamic
  sampling, 5 attempts, temperature 1;
- optimisation: lr 1e-5, 60 updates, seed 0;
- data: the same 184 held-out hands;
- the rollout guards from BUGS.md #26.

Upstream does the same: `rl/configs/llama_gp_vl.yaml` differs from the
language config only in rendering, colour and prompt.

**Differs for GP-VL.** Each change was verified:

| difference | verification |
| --- | --- |
| cards rendered as the GP-VL evaluation renders them (1200 px, black) | the hand stream equals the evaluation's (50/50); visual-OOD shares the holdout |
| image only in the first user turn, passed as a PNG path (the evaluator's structure) | evaluation and rollout preprocessing give identical ids and pixels |
| prompt ids from the processor (1,444 image tokens) | multi-turn prefix invariant holds with the image in history |
| image features, deepstack and mRoPE positions computed once per update, outside the gradient | logits bit-identical to the full forward; trainer refuses if anything outside the language layers trains |
| batch_generate with images | 0/3,682 position mismatches; sampled tokens mean −0.134, none below −10 on any turn |
| microbatch 1 (GP-L used 4) | peak 67.6 GB of ~107.5; the loss is normalised over the whole batch, so this does not change it |

**Parameters checked rather than assumed.**
- `max_tokens` 128: 0 of 113 VL completions truncated.
- `entropy_coef` stays at 0.002, a judgement call. The measured ratio of
  entropy to policy-gradient gradient is 0.0115 at coef 0.01, against GP-L's
  0.0033. The literal GP-L rule would give 0.0005.
  - The rule was introduced for a structural change: single-turn token
    normalisation to multi-turn trajectory normalisation, which rescaled the
    terms 5.5×.
  - GP-L to GP-VL changes nothing in the objective.
  - The gap is almost all in the policy-gradient norm (0.214 vs 0.547), which
    depends on each batch's reward mix. The entropy gradient is similar
    (0.00246 vs 0.00179).
  - Both are single-batch measurements.
  - Keeping GP-L's value matches upstream, which uses the same coefficient
    for both modalities.

**Not verifiable beforehand.**
- Training dynamics over 60 updates. There is no separate pilot: the first
  updates serve as one, with kl/tok and the `d` range logged every update.
- Run time, estimated at roughly 30–40 h alone: about 6 min per 8-trajectory
  group plus about 7–9 min per update, measured with the GPU shared.
- Whether RL improves GP-VL.

## 8. RL pipeline

Built and verified end to end. Runs and results: single-turn GRPO (no configuration beat SFT) and multi-turn GRPO v5 (Step 2 result above).

```bash
# 1. SFT data (holds out the evaluation hands)
.venv-qwen/bin/python scripts/make_sft_data.py -n 1000

# 2. SFT
.venv-qwen/bin/python scripts/train_sft_qwen.py --iters 250

# 3. GRPO smoke test -- does the loop learn at all?
.venv-qwen/bin/python scripts/train_grpo_qwen.py \
    --adapter train_ckpt/sft_gp_language --updates 50

# 4. evaluate any checkpoint through the same ablation
bash scripts/run_qwen_gp_ablation.sh evaluation/configs/qwen_gp_language.yaml 50 grpo \
    -- --adapter_path=train_ckpt/grpo_gp_language
```

| piece | file | state |
| --- | --- | --- |
| rollout buffer + GAE over env steps | `rl/trainer/storage_mlx.py` | ported, differential-tested against the torch original |
| LoRA policy, KL reference, value head | `rl/trainer/policy_mlx.py` | tested against the real model |
| single-turn GRPO | `rl/trainer/grpo_trainer_mlx.py` | runs end to end (3 updates) |
| launcher | `scripts/train_grpo_qwen.py` | runs |
| single-turn GRPO | `rl/trainer/grpo_trainer_mlx.py` | runs; validated; no configuration beats SFT |
| **multi-turn GRPO** | `rl/trainer/grpo_multiturn_mlx.py` | runs v1–v4 **invalid**: rollout sampled at corrupted positions from turn 2 (BUGS.md #26); fixed and verified; 8-update pilot: no spike (kl/tok ≤ 1.4e-2, \|d\| ≤ 2.7); full run next |
| trajectory PPO with the critic | — | not written; may not be attempted |

### Why multi-turn GRPO before PPO

Single-turn GRPO scores one attempt per hand, so it **cannot train revision at
all**. That is a problem, because revision is the deficit this reproduction has
measured most confidently: after SFT, recovery from a first-attempt failure
falls to 1/38 against the base model's 11/38 (paired McNemar p = 0.0020). The
evaluation protocol allows five attempts with verifier feedback, so a policy is
being graded partly on an ability the training objective never touches.

Multi-turn GRPO closes that gap with less machinery than PPO: roll out full
trajectories with verifier feedback, take the trajectory return as the reward,
and group G trajectories on the same hand. No critic, no GAE, no value head —
every validated component (log-probs, completion masking, gradient
accumulation, policy/old/reference separation, the temperature fix) carries over
unchanged.

PPO adds a critic and GAE over environment steps: more machinery, more failure
modes, and a learned value baseline cannot manufacture correct answers the
policy never explores. It remains the fidelity-to-the-paper step, to be
attempted only if multi-turn GRPO shows signal.

Cost: roughly 5x the generation per trajectory, since each rollout runs to the
retry limit instead of stopping at one attempt.

GRPO is deliberately single-turn: one hand, G sampled answers, one reward each,
no verifier-feedback retries and no GAE. It is a correctness check that
generation -> reward -> advantage -> gradient -> improvement is wired up, before
a critic is added. `storage_mlx.py` exists for the PPO stage and is unused by
GRPO.

**Training hands exclude the evaluation hands**, matching what
`make_sft_data.py` does, so SFT and RL are trained on the same terms and the
comparison between them is meaningful. `--no-holdout` opts out.

### Rollout speed

The rollout is ~90% of a GRPO update (46s of 51s at `group_size=4,
prompts_per_update=2`); the three forward passes of the update itself are the
rest. Effective decode was 8.3 tok/s, about 4x below the batch-1
memory-bandwidth ceiling for an 8B bf16 model on this machine (~34 tok/s).

`--batch-rollout` samples a whole group in one `batch_generate` call instead of
G sequential ones. Decoding at batch 1 is bound by reading the weights, so a
group of G costs little more than one sequence:

| | tok/s | wall-clock |
| --- | ---: | ---: |
| sequential (G=4) | 5.0 | 38.5s |
| batched (G=4) | 19.4 | 9.7s |
| | | **3.97x** |

Both measured under the same background load, so the ratio is fair.

It is opt-in; the sequential path is untouched and remains the reference.
Batching is **within a hand only** — the environment must still hold that hand
when `reward_fn` calls `env.step` — and **only across prompts of equal length**:
mlx_vlm 0.7.2 decodes left-padded rows at wrong RoPE positions (BUGS.md #26).
Prompts go in as message lists, never templated text, which `batch_generate`
would template a second time. Every call checks that it processed exactly the
prompt tokens the scorer conditions on.

#### Token reconstruction, validated

> **Scope, learned the hard way (BUGS.md #26):** this validated the *ids* on the
> `generate()` path. It did not check the *context* `batch_generate` conditioned
> on, and never sampled `batch_generate` past turn 1 — where both rollout
> defects lived. `scripts/verify_rollout_fidelity.py` now checks the sampler
> against the scorer directly, on the trainer's own rollout.

`batch_generate` returns only text, not ids, so the batched path rebuilds them
with `_completion_ids`. A wrong rebuild would score a different sequence than
the one that earned the reward — corrupting log-probs and importance ratios
silently, with no crash and no visible symptom beyond "RL does not learn". So it
was checked against the sequential path's ground-truth `token_ids` rather than
assumed:

| condition | samples | exact | `finish_reason` | terminal token |
| --- | ---: | ---: | --- | --- |
| normal (`max_tokens=128`) | 40 | **40/40** | all `stop` | 151645 `<\|im_end\|>`, 40/40 |
| forced truncation (`=12`) | 20 | **20/20** | all `length` | ordinary content (`497` = `",`) |
| near-boundary (`=40`) | 20 | **20/20** | all `length` | ordinary content |
| | **80** | **80/80** | | |

Mechanism: mlx_vlm drops the token whose `finish_reason` is `"stop"` before
detokenising (`ar.py:_generate_batch`), so the text never contains it.
`_completion_ids` appends it back — **except** when the completion reached the
budget, where generation was cut and no stop token was ever emitted; appending
one there would teach the model to stop at an arbitrary truncation point. Both
branches are exact above. `decode(encode(t)) == t` also held on 494 real outputs.

**Known residual risk.** The model declares *two* stop ids — 151645
`<\|im_end\|>` and 151643 `<\|endoftext\|>`. Only the former was ever observed
terminating a completion (40/40). A completion ending on the latter would get
one wrong token out of ~48. The batched API exposes no way to tell them apart,
so this is documented rather than fixed; `stop_id_ambiguous` records that the
model has an ambiguous stop set, and `truncated_completions` is reported per
update so a rising truncation rate is visible.

Three runtime guards: the tokenizer's `eos_token_id` must be among the stop ids
the model declares (else the rebuild would append a token generation never
emits); the batch must return G completions; and the first group must not be
all-identical (which would mean sampling shared one RNG draw, every group would
have zero reward spread, and no update would ever occur).

Pinned by `scripts/test_rl_tokens.py` against the real model, and by
`CompletionIdsTest` for the branch logic.

At `group_size=8, prompts_per_update=4`, sequential is ~2.6 h for 50 updates;
batched should be roughly 1 h.

## 9. Known issues

Bugs found and their measured blast radius are logged in `BUGS.md`. Two affect
how these numbers should be read:

- the whitespace behaviour (`BUGS.md` #1), which is why every run is an ablation;
- a solver/verifier arithmetic mismatch (`BUGS.md` #2), **fixed**; it did not
  occur in any completed evaluation, so no baseline here is affected by it.

## 10. Not done yet

- **SFT on GP-VL.** Done for GP-L (see above); the vision arm is not yet run.
- **RL.** `rl/trainer/` is torch + DeepSpeed + CUDA and cannot run on this
  machine. A Qwen port also needs a value head sized from `config.hidden_size`
  (currently hardcoded to 4096) and removal of the Mllama `cross_attention_mask`
  plumbing, which Qwen has no equivalent of.
- **V-IRL.** Untouched; needs its route dataset and Google Maps credentials.
- **OOD splits.** Only the in-distribution setting has been run. Rule-OOD
  (`treat_face_cards_as_10: false`) and visual-OOD (`face_cards_color`) are
  configured but not yet measured — and those are where the paper's central
  claim actually lives.
