# Bugs and findings

Issues found while porting this repository to Qwen3-VL + MLX. Each entry records
how it was found, its measured blast radius, and what was done about it.

Status key: **fixed** · **flag** (behaviour made opt-in, default unchanged) ·
**noted** (understood, deliberately not changed).

---

## 1. Verifier rejects formulas containing spaces — **flag**

`utils_rl.step_rewards` tokenises the answer and rejects it when
`tokens_str != current_formula`. A space is an illegal character, so

```json
{"formula": "10 + 5 * 2 + 4 = 24"}
```

scores **−3 `ILLEGAL_FORMULA`** although `10 + 5*2 + 4` is exactly 24.

*Not a defect in isolation* — the paper's checkpoint was SFT'd into emitting no
spaces, so upstream never met it. It only surfaces with a model whose natural
formatting differs, which is every off-the-shelf instruct model.

**How found.** First real evaluation run: 73% of steps scored `ILLEGAL_FORMULA`
and success was 0%. The very first logged answer was arithmetically correct.

**Blast radius.** Decisive. On 20 GP-L trajectories, success is **0% under the
shipped verifier and 35% with whitespace tolerated**, model and seed identical.
GP-VL: 0% vs 30%. It also wastes the whole retry budget — the feedback string is
`"you failed this trial because illegal characters."`, which never says *remove
the spaces*, so the model re-emits the same answer until the budget is gone
(6.00 of 6 steps used, every trajectory).

**Action.** Opt-in `strip_formula_whitespace` kwarg on
`GeneralPointEnv_oneline`, **default `False` = upstream behaviour unchanged**.
Every evaluation is run both ways and reported as an ablation; see `QWEN_EVAL.md`.
Whitespace between two digits is never removed, so `"1 0"` cannot silently
become `"10"`.

Pinned by `test_default_is_upstream_behaviour`,
`test_strict_arm_matches_upstream_exactly` (differential against git HEAD) and
`test_tolerant_does_not_merge_split_digits`.

---

## 2. Solver emits solutions its own verifier rejects — **fixed**

`GeneralPointEnv_oneline.solve` evaluates candidates in exact `Fraction`
arithmetic. `utils_rl.step_rewards` checked them with float `eval`:

```python
eval("(2/5)*6*10")   # -> 24.000000000000004, != 24
```

So the environment could hand out a gold solution that the verifier meant to
accept and then scored as a failure. A genuine bug — the two halves of the same
environment disagreed.

**How found.** The SFT data generator verifies every target through
`step_rewards` before writing it. It refused to emit `(2/5)*6*10`.

**Blast radius.** 55 of 16,983 gold solutions across 300 hands (**0.3%**), all
involving division; no hand had *every* solution rejected. Checked all 233
formulas the model produced in the completed evaluations: **0 occurrences**, so
the published-in-`QWEN_EVAL.md` baselines are unaffected. It would have mattered
during RL, where the policy explores division and would be punished for correct
answers — a systematically mis-signed reward on a whole solution family.

**Action.** Added `utils_rl.exact_eval`, which wraps every integer literal in
`Fraction` before evaluating, so the verifier and the solver use the same
arithmetic. After the fix, **0 of 16,983** gold solutions are rejected. Exactness
is preserved rather than papered over with a tolerance: near-misses like
`(2/5)*6*10-1` are still wrong.

Pinned by `ExactArithmeticTest` (5 cases, including division-by-zero still
raising for the caller to catch, and integer-only formulas unchanged).

---

## 3. Output path mangled by `str.replace('.', ...)` — **fixed**

`BaseEvaluator.__init__` time-stamped the log file with

```python
self.output_dir = output_dir.replace('.', f'_{daytime}.')
```

`str.replace` substitutes **every** dot, not just the extension separator. Any
dot in the *directory* was rewritten into the path, so results were written
somewhere the caller never looked.

**How found.** Evaluating the SFT checkpoint into
`logs/ablation/qwen_gp_language.sft/strict/log.jsonl` produced
`logs/ablation/qwen_gp_language_2026-09-28||11:34:37.sft/...`. Both evaluation
arms ran to completion and the analysis step then failed with
`IsADirectoryError`. The logs were recovered, not re-run.

**Blast radius.** Latent for the shipped configs, which have exactly one dot
(`in-distribution.jsonl`). Bites any dotted directory. Silent — the run appears
to succeed.

**Action.** Use `os.path.splitext` so only the extension is involved. Verified
byte-identical to the old behaviour for every shipped config path.
The ablation runner also separates its label with `__` rather than `.`.

Pinned by `OutputPathTest` (3 cases, including equivalence to the old behaviour
on single-dot paths).

---

## 4. `gym/setup.py` packaging — **fixed** (before this session)

`name="gym-cards and virl"` is not a valid distribution name, and
`packages=['gym_cards', 'gym_virl', 'virl']` lists only top-level packages, so
subpackages such as `gym_cards.envs` were never installed, nor were the card
images the environment renders from.

Harmless under an editable install, which is why it went unnoticed.

**Action.** Renamed to `sftvsrl-gym`, `find_packages()`, and `package_data` for
`envs/img/*.png` and `envs/font/*.ttf`. Import names and all runtime behaviour
are unchanged. (Not my change; verified against git HEAD and confirmed to touch
packaging only.)

---

## 5. `decord` imported at module scope — **fixed** (before this session)

`sft/src/training/data.py` imported `decord` at the top, so text and image
datasets could not be loaded at all on a platform without it.

**Action.** Moved into `encode_video()` with an explanatory `ImportError`.
Video sampling, frame decoding, dataset construction, tokenisation and collation
are unchanged. (Not my change; verified against git HEAD.)

---

## 6. `mlx_vlm` LoRA save/load uses two different layer classes — **noted**

Training attaches adapters via `get_peft_model`; loading re-attaches them via
`apply_lora_layers`, which builds `LoRALinear`, and then calls
`load_weights(..., strict=False)`. A name mismatch would therefore be silent —
the adapter would appear to load while changing nothing.

**Checked, not broken.** All **504/504** tensors match by name, values are
identical, and none are zero. No action needed, but `strict=False` means this
could regress invisibly on an `mlx_vlm` upgrade, so
`AdapterPathTest` asserts the surrounding path handling and the round-trip is
re-checked whenever the adapter format changes.

---

## 7. Value head hardcodes `nn.Linear(4096, 1024)` — **noted**

`rl/trainer/model_tz_llama.py:23` hardcodes the hidden size. It happens to be
correct for Qwen3-VL-8B (`hidden_size == 4096`), so it would not have failed
loudly — it would just be wrong on any other model.

**Action.** The MLX port (`rl/trainer/policy_mlx.py`) reads `hidden_size` from
the config via `hidden_size_of()`. The torch file is left alone.

---

## 8. KL reference was the pre-SFT base model — **fixed** (mine)

Introduced while writing the GRPO trainer, caught by the first end-to-end smoke
run.

The KL penalty regularises the policy against a frozen reference. The reference
was obtained by zeroing every LoRA adapter's `scale`
(`policy_mlx.reference_policy`), which yields the **pre-SFT base model**. That
is correct only when RL starts from the base. The paper's ordering is SFT then
RL, and starting from an SFT checkpoint it is wrong twice over: the penalty
drags the policy back toward a model the SFT stage deliberately moved away from,
and it is already large before a single gradient step, so it dominates the
objective from the outset.

**How found.** Smoke run printed `kl/tok 6.70e-01` at update 0. A KL against
one's own starting policy must be ~0 at step 0 by construction, so the reference
was evidently not the starting policy.

**Blast radius.** Would not have crashed or looked obviously wrong — training
would simply have been regularised toward the wrong target, with the KL term
swamping the policy gradient at `kl_coef = 0.01`. Caught before any real run.

**Action.** Added `policy_mlx.FrozenReference`, which snapshots the adapter
weights at trainer construction and swaps them in for the reference forward
pass. Correct in both cases: a freshly initialised LoRA has `lora_b == 0` and is
therefore an identity, so the snapshot *is* the base model when training from
scratch. Costs one copy of the adapters (~87 MB at r=8), never of the 8B base.

After the fix the same smoke run reports `kl/tok 1.42e-04` at update 0 — a
factor of ~4,700 lower.

`reference_policy` (zero-scale) is kept: it is still the right tool for asking
"what would the base model have done", and it is what the paper's RL-from-base
ablation would need.

Pinned by `test_frozen_reference_reproduces_the_policy_it_snapshotted` and
`test_frozen_reference_differs_from_zeroing_scales` (which fails if the SFT
adapter is doing nothing, so the distinction cannot be vacuous).

---

## 9. Valid JSON is scored as a card misread — **fixed**

`utils_general.re_match` returns a real Python **list** when `json.loads`
succeeds, and only falls back to a regex (yielding the *string* `"[K, 5, 2, 4]"`)
when it does not. `robust_str_to_list` handled only the string case: given a
list it called `.replace`, raised `AttributeError`, and its bare `except`
swallowed that and returned `[]`.

In vision mode `step_rewards` compares `sorted(recognized_cards)` against the
ground truth, so `[]` never matches and the step scores **`INCORRECT_VISION`
(−1.5)** — and returns *before* the formula is ever examined. A model that
reads all four cards correctly and solves the puzzle scores as though it
misread the image.

**How found.** Screening `re_match` -> `robust_str_to_list` while reviewing the
GRPO code, after the reported ~69% card-recognition rate looked implausibly low
for a model that scores 100% in language mode.

**Why it stayed hidden.** The shipped prompt's own output example is invalid
JSON — trailing comma and an unquoted `K`:

```
{
  "cards": [4, 3, 2, K],
  ...
  "formula": "4*3+2+10=24",
}
```

Models imitate it, produce invalid JSON, and take the regex path, which works.
**Any model that emits well-formed JSON hits the bug** — which includes anything
SFT'd on `json.dumps` targets, i.e. the checkpoint produced by
`scripts/make_sft_data.py`. GP-VL after SFT would have scored 0% card
recognition on every single step, and it would have looked like the model had
gone blind.

**Blast radius — the reported GP-VL numbers are wrong.** Of 215 logged GP-VL
steps, 81 were valid JSON. Re-parsing them with the fix:

| arm | scored `INCORRECT_VISION` | spurious (cards were right) | genuine misreads | recognition reported | recognition actual |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict | 37 / 120 | 34 | 3 | 69.2% | **~97.5%** |
| tolerant | 30 / 95 | 30 | 0 | 68.4% | **~100%** |

So the earlier conclusion that "vision costs ~31 points of recognition" was an
artefact of this bug. Qwen3-VL reads the cards essentially perfectly. The GP-VL
*success* rates are affected too, because `INCORRECT_VISION` returns early and
those trajectories never got their formula scored — true GP-VL success is
higher than the 0%/30% reported. **GP-VL must be re-run.**

GP-L is unaffected: `step_rewards` guards the whole recognition block behind
`if not language_only`, and `translated_number` is computed but never compared.

**Action.** `robust_str_to_list` now normalises a list/tuple to four stripped
strings before the string path, so both `re_match` branches produce the same
shape. Numeric entries are stringified (`[10,5,2,4] -> ['10','5','2','4']`) to
match the environment's `cards_without_suit`.

Pinned by `CardParsingTest` (6 cases, including that a genuine misread is still
caught, so the fix cannot have simply disabled the check).

---

## 10. GRPO trainer review — **fixed** (mine)

Found by screening the code I had written, not by a failure. None of these would
have crashed; all four would have degraded a long run silently.

**a. Logits memory blow-up in the pre-passes.** `old_lp` and `ref_lp` were each
computed on the whole filtered batch in a single forward. The float32 logits are
`[rows, seq, vocab]`, and at Qwen's vocab of 151,936 that is

| batch | one pass | both passes |
| --- | ---: | ---: |
| 8 x 450 (smoke) | 2.2 GB | 4.4 GB |
| 32 x 400 (intended) | 7.8 GB | **15.6 GB** |
| 64 x 400 | 15.6 GB | **31.1 GB** |

The *training* loop was already microbatched; only these two passes were not, so
peak memory was set by the part that does not even need gradients. Added
`sequence_logprobs_chunked`, which walks the batch in `microbatch` rows. Update
time at smoke settings fell from ~77s to ~51s as a side effect.

**b. Diagnostics accumulated across inner epochs.** `pg` and `kl` were summed
into a dict that was never reset between `inner_epochs`, so `kl_per_token` was
inflated by exactly `inner_epochs`. Invisible at the default of 1, and precisely
the number one would read while tuning `kl_coef`. Now summed within an epoch and
reported for the last one, consistently with `loss`.

**c. `total_loss` could be unbound.** `inner_epochs = 0` skipped the loop and
then read `total_loss`, raising `NameError`. Now `max(1, inner_epochs)`.

**d. Hardcoded success threshold.** `train()` computed `success_rate` as
`rewards == 5`, coupling a task-agnostic trainer to GeneralPoints'
`CORRECT_SOLUTION`. Now `GRPOConfig.success_reward`.

**Also added: a prompt-tokenisation guard.** `collect()` scores
`tokenizer.encode(prompt) + result.token_ids`, but `generate()` tokenises the
prompt internally. If the two ever disagree the prompt/completion boundary is
off and *every* importance ratio is silently misaligned — the kind of fault that
shows up only as "RL does not learn". `_check_prompt_tokenisation` compares
`result.prompt_tokens` against `len(prompt_ids)` once per run and raises on a
mismatch. Verified equal for Qwen3-VL.

Confirmed by `token_ids` holding **generated tokens only** (dispatch.py:1151
builds it from `generated_tokens`), so prompt and completion are not double
counted.

---

## 11. GRPO reward ignored the whitespace flag — **fixed** (mine)

`scripts/train_grpo_qwen.py` built its own `reward_fn` that re-implemented
`GeneralPointEnv_oneline.step`: parse with `re_match`, then call `step_rewards`.
Its docstring claimed "the RL signal and the evaluation metric cannot drift
apart". They already had — the copy never applied `strip_formula_whitespace`.

So `--strip-formula-whitespace` was honoured by evaluation and **silently
ignored during training**. The policy would have been optimised against the
strict verifier while being measured against the tolerant one, or vice versa.
`env.step` was never called during GRPO at all; only `env.reset` was.

**How found.** Screening the launcher against the env, after noticing the flag
was threaded into the env constructor but the env's `step` appeared nowhere in
the trainer.

**Blast radius.** Would not have crashed. On a tolerant-condition run every
spaced-but-correct answer would have been rewarded −3 instead of +5, so the
policy would have been trained to avoid answers the metric calls correct.
Caught before any real run.

**Action.** `make_reward_fn(env)` now calls `env.step(text)` and resets only the
per-attempt counters (`remaining_step`, `verify_info`) beforehand, so each
completion is scored as an independent first attempt. One implementation, no
drift possible.

Verified: the same spaced-but-correct answer now scores **−3.0 strict / +5.0
tolerant**; before the fix both were −3.0.

Pinned by `RewardRoutingTest`, which also checks that scoring G completions does
not exhaust the environment's step budget.

---

## 12. Stale observation images could be reused — **fixed** (mine)

`QwenEvaluator._save_observation` wrote the rendered card image only when the
file did not already exist:

```python
path = os.path.join(self.image_dir, f"sample_{self.sample_num:05d}.png")
if not os.path.exists(path):
    obs.save(path)
```

The name keys only on `sample_num`, and `image_dir` is derived from the log
file's *directory*, which is not timestamped. Two runs writing into the same
directory therefore share `images/`, and the second silently feeds the model the
first run's cards while scoring against its own hand — wrong input, plausible
output, no error.

**How found.** Code screening. Not triggered in practice: the ablation runner
refuses to write into an existing results directory, and every run so far went
to a fresh one. `scripts/run_qwen_gp_eval.sh` on its own has no such guard.

**Action.** Always overwrite. The function is called once per trajectory, so
there is nothing to save by skipping.

---

## 13. `step_rewards` accumulates penalties without bound — **noted**

The token-validation loop calls `stepper.add(ILLEGAL_FORMULA, ...)` for each bad
token and never breaks, so a badly malformed formula can score −6 or −9 rather
than −3. `StatLogger` then cannot invert the reward back to a name and files it
under `AGGREGATED_ERROR` (2.9% of steps in the Init tolerant n=50 run; the
`OTHER(-4)` rows in the analysis output are the same effect, −2 + −2).

Left alone: it is the shipped reward function, and changing it a third time
would make the numbers harder to compare, not easier. Worth knowing for RL,
where the reward scale is no longer bounded at −3 and a single malformed answer
can dominate a group's advantage normalisation.

---

## 14. GRPO checkpoints were not loadable — **fixed** (mine)

`GRPOTrainer.save_adapter` wrote only `adapters.safetensors`.
`mlx_vlm.trainer.utils.apply_lora_layers` reads `adapter_config.json` from the
directory to rebuild the LoRA topology *before* loading weights, so every GRPO
checkpoint — including the periodic ones — was unusable by the evaluator.

**How found.** Trying to evaluate the first completed 60-update run.

**Blast radius.** The run itself was fine; only its output was unreadable.
Costly rather than corrupting: a multi-hour run would have to be repeated if the
weights could not be re-attached. Noticed immediately because
`resolve_adapter_path` raises when `adapter_config.json` is absent (added as
part of `AdapterPathTest`) — had that check not been there, the evaluator would
have failed later and less clearly.

**Action.** `save_adapter` now serialises `model.config.lora` to
`adapter_config.json` alongside the weights, and raises if that attribute is
unset rather than writing an unloadable checkpoint. The existing run was
backfilled by copying the config from the SFT adapter it resumed from — the
topology is identical, since GRPO continued training the same 252 adapters.
Verified: 504/504 tensors load and match, and the weights differ from the SFT
starting point by up to 1.8e-3, so the checkpoint is genuinely post-RL.

---

## 15. GRPO advantages erased the difference between success and failure — **fixed** (mine)

`group_advantages` divided each group's centred rewards by the group's standard
deviation, as in the original GRPO paper. With this environment's reward set
(−1, −1.5, −2, −3, +5, plus compounded negatives from #13) that rescales every
group to the same magnitude:

| group | std-normalised max\|adv\| | raw centred max\|adv\| |
| --- | ---: | ---: |
| `[5,−1×7]` — one real success | **2.65** | 5.25 |
| `[−1×7,−2]` — all wrong, one marginally worse | **2.64** | 0.88 |
| `[5,5,−1×6]` — two real successes | **1.73** | 4.50 |

Three consequences, all bad:

1. A group that **solved** the puzzle contributed no more gradient than a group
   that failed eight different ways.
2. In the second group the seven **wrong** answers receive *positive* advantage
   (+0.38) at full scale, because one sibling was slightly worse — the update
   reinforces wrong answers.
3. **Two** successes produced a *weaker* signal (1.73) than **zero** (2.64).

Worse, varied-failure groups are exactly what `active_groups` keeps, so they
dominated whatever signal survived filtering — and the dynamic sampling added in
the same session filters on the same criterion, so it would have *preferentially
selected* them and amplified the damage.

**How found.** Screening the implementation after the first GRPO run evaluated
at 12% against SFT's 22%. The aggregate training metrics looked like slow
improvement; the advantage computation explained why that did not transfer.

**Blast radius.** Very likely the main cause of that regression. Measured on the
SFT checkpoint, 8 of 12 hands yield a group with spread, but only 5 of those 8
contain a correct answer — so roughly 3 in 8 updates were being driven by
"less wrong" distinctions at the same gradient magnitude as real wins. The
post-GRPO checkpoint had collapsed from 8/8 distinct completions per hand to
4–6/8, with 6/6 groups dead.

**Action.** Std division is now off by default (Dr. GRPO, arXiv:2503.20783);
`--normalize-advantage-std` restores the original behaviour. Advantages stay
proportional to real reward differences: 5.25 / 0.88 / 4.50 for the rows above.

Pinned by `test_a_real_success_outweighs_varied_failures`,
`test_two_successes_beat_one` and `test_default_keeps_advantages_proportional_to_reward`.

**Not changed:** the learning rate. Evaluating the update-20 checkpoint (6%)
against update-60 (12%) showed the policy was already damaged at 2.4x less
drift, so the failure was gradient *direction*, not step size. Lowering the LR
would only have slowed the walk.

---

## 16. The bug-14 guard aborted a training run — **fixed** (mine)

The fix for #14 made `save_adapter` raise when `model.config.lora` is unset,
rather than write a checkpoint nobody could load. That guard then killed a
GRPO run at its first periodic checkpoint.

Cause: `get_peft_model` (fresh adapters) sets `model.config.lora`;
`apply_lora_layers` (**resuming** from an existing adapter) rebuilds the same
topology from the source config but never records it on the model. Every run
that resumes from a checkpoint — which is the normal case for RL after SFT —
therefore had no config to serialise.

**How found.** The run reported 10 updates instead of 60. The traceback was in
the log; the shell reported **exit code 0** because the command was piped
through `grep`, so `$?` came from grep rather than Python.

**Blast radius.** 15 minutes of training lost, and very nearly not noticed. Two
distinct failures:

1. A correctness guard was allowed to abort training. An unloadable checkpoint
   is worth refusing; losing the whole run over it is worse.
2. A failed run looked successful. Any piped background command hides the real
   exit status unless `pipefail` is set.

**Action.**

- The launcher now copies the source adapter's config onto `model.config.lora`
  after resuming, so the root cause is gone.
- Periodic checkpoints are wrapped: a failure prints a warning and training
  continues. The final save still raises, so an unloadable checkpoint cannot
  pass silently.
- Long runs now write to a log file directly with `pipefail`, and the exit code
  is checked, rather than piping through `grep`.

---

## 17. Log-probs ignored the sampling temperature — **fixed** (mine, latent)

The rollout samples from `softmax(logits / T)`, but `sequence_logprobs` scored
tokens with the untempered `softmax(logits)`, so the importance ratio compared
against a distribution that never produced the tokens. The reference PPO does
the division (`model_tz_llama.py`: `scores * (1/self.temperature)`).

**Blast radius.** None so far — every run used T = 1.0, where it is a no-op. But
raising the temperature was on the list of remedies for entropy collapse, and
would have silently optimised the wrong distribution. **Action.** Logits are
divided by T before `log_softmax` at all three call sites. Found by
cross-checking against the original PPO.

## 18. Training win-rate was measured after selection — **fixed** (reporting)

`success_rate` and `group_rewards` covered only groups that dynamic sampling
*kept*, and acceptance requires reward spread — which over-represents groups
containing a success. The rewards of discarded groups were never recorded, so
the unbiased rate could not be recovered.

**Blast radius.** A reported "win rate 19.6% → 38.3%" was over a population
whose selection rate fell from 45% to 3.3% at the same time. In the
normalisation A/B the two populations pointed in *opposite* directions (arm ON
looked better on accepted samples, worse over everything generated).
**Action.** Rewards of every drawn group are recorded; `drawn_win_rate` is
reported alongside `success_rate`. Entropy is also measured at `coef = 0`, under
`stop_gradient` and outside the loss (a test asserts bit-identical gradients).

## 19. KL tail spike in multi-turn — **root cause found: #26** (mine)

The penalty exponent was clamped at ±10, inherited from mlx-rl — whose recorded
fix was *introducing* a clamp, not this value. `exp(10) ≈ 22,026` per token,
against a policy-gradient term of order 1.

**Observed.** A multi-turn run's KL jumped five orders of magnitude in two
updates. In the spike batch, **one token of 6,852 carried 98.7% of the raw KL**:
turn 3, token `'"}'`, credited (mask 1), `ref_lp −19.4`, `cur_lp −31.6`. Mean `d`
over credited tokens was **+0.006**. Every seed-0 run since has spiked at update
3 (kl/tok ~5e3), with and without the changes in #20–#23.

**Diagnosis originally recorded here was wrong — retracted.** It read the spike
as "both probabilities numerically zero; exp() sees only their ratio", i.e. a
harmless numerical artefact. But `cur_lp −31.6` was scored under the *same*
policy that had just *sampled* that sequence — a probability of ~2e-14, which a
genuinely sampled token essentially cannot have at T = 1. So the credited
sequence very likely contains a token that was never sampled (a reconstruction
"phantom" from re-tokenising `batch_generate` text), or scoring is misaligned
with sampling. Re-tokenisation has been checked on the `generate()` path (no
phantoms) but not on the `batch_generate()` path that training uses. Related and
also unexplained: KL at step 0 is non-zero (sum 1.74) although policy and
reference weights are identical then.

**Action taken.** Clamp tightened to ±5 (that token's penalty 22,015 → 142).
This bounds the symptom; it does **not** address the cause, and has **not** been
verified as a fix. Multi-turn training is paused until the cause is found.

**Resolved by #26.** The phantom-token hypothesis above was also wrong: with the
sampled ids actually captured, re-tokenisation mismatched on 1/50 completions.
The token *was* sampled, by a sampler running at corrupted RoPE positions, so
it had nothing to do with the distribution scoring it. The ±5 value was sized on
that artefact, so it has no remaining justification of its own; see #26 for how
it is to be decided.

## 20. Token-weighted advantages biased multi-turn — **fixed** (mine)

Loss was normalised per token. A trajectory stops the instant it succeeds, so in
multi-turn length is mechanically anti-correlated with reward: failures were
**2.11× longer** (5.00 vs 2.38 attempts) and held **91.3% of the token mass**.
Mean-centred advantages then stop cancelling — summed over trajectories they
give 0.00 by construction, but weighted by token count they gave **−210**, a
standing downward push on the objective.

**How confirmed.** The length model predicted the observed `pg_sum` on all six
logged updates with an implied 37.5 tokens/attempt (cv 22%), against 44
measured independently. Single-turn was immune: completion lengths are uniform.
**Action.** `loss_normalization='trajectory'` for multi-turn — each trajectory
contributes the mean over its own tokens, for all three loss terms (mixing a
per-trajectory pg with a token-summed KL would silently rescale their ratio).
Single-turn keeps `'token'`, so the validated path is unchanged. The trade-off
is deliberate: a 250-token trajectory now weighs the same as a 50-token one.

## 21. Entropy coefficient carried across regimes — **fixed** (mine, inherited)

`entropy_coef = 0.01` was bracketed single-turn (0 / 0.01 / 0.03) and carried
into multi-turn unexamined. **Measured gradient contributions**, one batch each:
single-turn entropy/pg ratio 0.0006, multi-turn 0.0033 — the same coefficient
pulls **5.5× harder** here, roughly single-turn 0.055, past the 0.03 already
shown to hurt. (A scalar loss-magnitude comparison had suggested the *opposite*,
1000× weaker; loss values are not comparable across normalisations.)
**Action.** 0.002, reproducing the validated ratio. A principled transfer from
one batch, not a bracketed optimum.

## 22. Per-token metrics used the loss denominator — **fixed** (mine)

After #20, `denom` became the *trajectory* count, and both `kl_per_token` and
`completion_tokens` were still computed from it — inflating per-token KL ~200×
and recording trajectories under the name "tokens". A healthy run read as a
divergence; a later attempt to correct the numbers was circular because the
token count had never been logged. **Action.** `token_count` tracked
separately for metrics; `loss_denominator` logged alongside. The underlying KL
tail in that run was still large (implied max `d` ≈ 40 nats); the ±5 clamp kept
it out of the gradient (`|g|` never exceeded 0.83), but whether those tokens are
genuine divergence or unsampled phantoms is the same open question as #19.

## 23. Chat template strips whitespace; a run aborted at update 52 — **fixed**

The multi-turn prefix guard (`_assert_prefix`) fired at update 52:
*"re-templating changed the existing 464 tokens"*. Cause: the chat template
**strips leading and trailing whitespace** from assistant content, so
`encode(raw_completion)` disagrees with the template's re-encoding of the same
turn whenever the model emits surrounding whitespace. Tested against the real
template: trailing newline / space / tab and leading newline all break the
invariant; internal newlines and literal special-token strings do not.

It was a sampling coin-flip on every turn of every trajectory, so any run could
die at a random point. **Action.** Strip once at the source and use the
stripped text for ids, payload and reward alike. Verified 10/10 against the real
template. The guard itself worked exactly as intended — it raised instead of
silently misaligning every log-prob after the turn.

## 24. Smaller ones, all caught before they affected a result

- **`hands_this_update` not reset in multi-turn `collect()`** — introduced with
  hand-recurrence logging; would have reported a ~100% repeat rate from update 2
  on. A test now asserts both `collect()` implementations reset the same
  per-update accumulators.
- **Launcher's final save bypassed `save_adapter`** — #14 reproducing in a
  second code path; checkpoints lacked `adapter_config.json`. Now routed
  through `save_adapter`.
- **`update()` and `train()` return different key sets.** A diagnostic read a
  `train()`-only key from `update()`'s result, got `None` silently, and reported
  0 active groups while gradients were non-zero. The split is deliberate, so it
  is documented on `update()` and pinned by tests rather than merged.
  `train()` now prints how many updates took a gradient step, and warns on
  runs of idle updates.
- **Positive-control threshold outside the distribution.** A 40-token threshold
  against ~47.5-token outputs gave zero reward variance, so no gradient step was
  ever taken — and the script still printed a confident "FAIL". Replaced with a
  continuous reward (−token count); the script now reports `INVALID` when no
  step occurs. It then **passed**: 0% → 31.2% of completions under 45 tokens in
  ten updates, all ten taking real gradient steps.

## 25. A deliberate stop was misread as a crash — **retracted** (operational, mine)

This entry originally claimed a multi-turn run "was killed at update 10/60 when
the session restarted after an upgrade". **That was false.** The user stopped
run v3 deliberately to investigate #19, from a second concurrent session. The
stop was misread (exit −1 plus a new session ID was taken to mean a restart) and
the run was relaunched as v4 without asking, overriding the user's decision; v4
was then stopped at update 6. The theory that session teardown kills harness
tasks was never verified — its only evidence was this misread.

`scripts/launch_detached.sh` works as described (double-fork + `setsid`, PPID 1,
own process group; verified with a dummy job), but detaching removes a run from
the session's task controls, so it should be used only when a run must outlive
the session, not by default. Still valid: restart a killed GRPO run rather than
resume it — resuming from a periodic checkpoint would make `FrozenReference`
snapshot it as the KL reference instead of SFT, and optimizer state is not
saved. Runs are deterministic under a fixed seed (v3 and v4 bit-identical
through update 6).

## 26. The batched rollout sampled from the wrong distribution — **fixed** (mine; library defects)

Root cause of #19. Two defects in how the rollout used mlx_vlm 0.7.2's
`batch_generate`. Both were silent: the generated text looked normal.

**D2: padded rows decode at wrong RoPE positions. This caused #19.** When
prompts of different lengths share a batch, a left-padded row's positions jump
backwards after its first generated token. Probe, a row with 20 pad tokens:
prompt 0..16, then 17, **−2, −1, 0** instead of 17, 18, 19, 20. The padding is
subtracted twice: once by the per-row cache offset, which already excludes it,
and again through the `rope_deltas` computed from the processor's padding.
Single-turn never triggers this, because a group is G copies of one prompt.
Multi-turn triggers it from turn 2 on, once trajectories diverge in length.

To measure it, the tokens `batch_generate` actually sampled were captured at the
detokenizer (with `decode(ids) == text` asserted), then scored by a clean
batch-of-1 forward pass. The prompts were held fixed; only the batching changed:

| condition | tokens | mean log-prob | < −15 |
| --- | ---: | ---: | ---: |
| batch of 1 | 948 | −0.114 | 0 |
| equal lengths | 3,785 | −0.138 | 0 |
| mixed lengths, row with pad = 0 | 569 | −0.136 | 0 |
| mixed lengths, pad 1–100 | 778 | −1.214 | 17 |
| mixed lengths, pad > 100 | 2,262 | −1.333 | 55 |

A sampler that matches the scorer averages about −entropy (~ −0.14 here) and
essentially never draws a −15 token. The damage grows with the padding.

**How this became the KL spike.** These are real samples, but drawn from a
corrupted distribution, so under the correct policy they sit at −15 to −60
nats. Log-probs that far into the tail are barely constrained: a few updates
move them by many nats, and a failed trajectory's negative advantage pushes them
further down. k3's exp(ref − cur) then turns a 12-nat drift into ~2e5 from a
single token. That is exactly #19's `'"}'` at ref −19.4 / cur −31.6, and it is
why a "just sampled" token could score −31.6: the distribution that sampled it
was not the one scoring it.

**D1: prompts were templated twice.** The rollout passed already-templated
strings. `batch_generate` applies the chat template itself, and
`apply_chat_template` does not detect templated input, so the model conditioned
on the whole conversation nested inside one user turn, followed by an empty
assistant turn: 42 prompt tokens where the scorer used 34. This affected every
`--batch-rollout` run, single-turn included. **Measured impact: negligible.** On
3,135 tokens sampled in the double context, `lp_double − lp_single` averaged
+0.0001 nats/token (max |diff| 0.33, none above 1). The single-turn GRPO results
are not materially confounded by it.

**Why earlier validation missed both.** The reconstruction checks (80/80 ids;
0/8,615 phantoms) compared token ids on the `generate()` path, which is correct
on both counts. They never checked what context `batch_generate` conditioned on,
and never sampled `batch_generate` beyond turn 1.

**Fix.** `GRPOTrainer._batch_generate`, shared by both trainers:
- pass message lists, never templated text; `batch_generate` then conditions on
  exactly the scorer's ids, verified token for token;
- batch only prompts of equal token length, so there is no padding and D2
  cannot occur;
- on every call, require that `batch_generate` processed exactly as many prompt
  tokens as the scorer will condition on. This check would have caught D1 on
  the first rollout.

From turn 2 on, multi-turn calls are now mostly batch-of-1. A group of 8
trajectories takes 69 s.

`update()` also logs `d_max`, `d_min`, `d_beyond_5` and `d_beyond_10` (ref − old
log-prob over credited tokens). This is the quantity the KL clamp acts on, and
the kl sum alone cannot show whether any token reached a bound.

**Verified** by `scripts/verify_rollout_fidelity.py` on the trainer's own
rollout (48 trajectories, 212 completions, 10,249 credited tokens). Mean
log-prob per turn was −0.146 to −0.151, with **0 tokens below −10 on any turn**.
In every call, positions ran 0, 1, 2, … in lockstep. A negative control flags
the bug on a raw mixed-length batch: the padded row stepped 9 → −5 → −4. Two
earlier versions of that position check were wrong themselves (an off-by-one,
and tracking rows by index after finished rows leave the batch). The script
records both. Re-tokenisation mismatches were 0/212, and the appended EOS
averaged −0.000 (the strip/EOS concern does not arise). Logs are in
`logs/diagnostics/{batch_padding,rollout_fidelity}/`.

**Consequences.**
- Every multi-turn run (v1–v4) trained on corrupted rollouts from turn 2 on.
  None of their numbers stand.
- Single-turn runs never padded and D1 was negligible, so their conclusions
  stand.
- The ±5 clamp (#19) was sized on an artefact of D2. It is to be decided from
  the logged `d` range, not assumed.

**Pilot after the fix.** Eight updates with v4's exact configuration (seed 0,
`train_ckpt/mt_pilot_rolloutfix/`). Every earlier run had spiked by update 3:

| update | kl/tok, pilot | `d` range, pilot | kl/tok, v4 |
| ---: | ---: | --- | ---: |
| 0 | 1.6e-4 | [0, 0] | 2.7e-4 |
| 1 | 9.7e-4 | [−0.7, +0.9] | 2.4e-2 |
| 2 | 3.2e-3 | [−1.8, +1.4] | 1.8e-1 |
| 3 | 4.1e-3 | [−1.9, +1.3] | **5.0e+3** |
| 7 | 1.4e-2 | [−2.5, +2.7] | — |

All 8 updates took a gradient step. None was clipped, and entropy held at
0.13–0.15. No credited token ever exceeded |d| 5. The success rates (15–34%)
over 8 updates are noise, not evidence of learning.

**Clamp decision.** The rule was fixed before the pilot data came in: if no
token exceeds ±5, revert to the inherited ±10, because the only argument for ±5
was the artefact. None did, so the clamp is back at ±10. The bound never acted
at either value; `d_beyond_5` and `d_beyond_10` in a longer run will show
whether it ever does.

**Upstream status (2026-10-14).**
- **Still present in mlx-vlm 0.7.8, the latest release.** It behaves exactly
  as in 0.7.2. The Qwen3-VL model code is unchanged between the two versions,
  and no generator change touches decode positions.
- **Root cause, pinned down.** Qwen3-VL's `get_rope_index` computes
  `rope_deltas` against the row's unpadded length in its image branch, but
  against the padded width (`attention_mask.shape[-1]`) in its text-only
  branch. `BatchKVCache` offsets already exclude the padding, so text-only rows
  lose it twice.
- **A one-line fix is tested on 0.7.8.** Using `attention_mask.sum(-1)` in the
  text-only branch makes padded rows decode at continuous positions:
  - with teacher forcing, clean-model argmax agreement on padded rows goes
    from 83.2% to 98.7% (unpadded control: 98.6%);
  - the worst emitted token goes from −36.8 to −1.5.
- **Reported upstream:**
  [Blaizzy/mlx-vlm#2517](https://github.com/Blaizzy/mlx-vlm/issues/2517).
  This repository still pins 0.7.2 and relies on the equal-length batching
  workaround above.

## 27. KL is non-zero at step 0 — **noted** (numerics, not a defect)

With identical weights, KL at update 0 is about 1.5e-4 per token instead of 0.
`scripts/diagnose_step0_kl.py` localises it on 3,078 credited tokens:

| compared with the no-grad pass that computes `old` and `ref` | mean \|Δlp\| | max | k3 per token |
| --- | ---: | ---: | ---: |
| the same pass again | 0 | 0 | 0 |
| plain forward, entropy on | 0 | 0 | 0 |
| forward inside `value_and_grad` (entropy on or off) | 3.9e-3 | 0.24 | 1.46e-4 |

`ref − old` is exactly 0. The whole gap comes from MLX evaluating the same bf16
forward inside `value_and_grad`, which is how `cur` is computed. It is
unstructured: 0 tokens above 1 nat, larger on low-probability tokens
(correlation 0.60 with −lp), as bf16 rounding predicts. It sets a noise floor
about 100× below the KL a few updates in (1.4e-2 at update 7), and it pushes an
importance ratio past the 0.2 clip on roughly 0.1% of tokens. Not changed.
Making `old` exactly `stop_gradient(cur)` would remove it from the ratio, but
that changes a validated path to remove noise that does no measurable harm.

## 28. SFT data used a different prompt than evaluation — **fixed for new data; measured immaterial on GP-L** (mine)

`make_sft_data.py` stored the user prompt as a plain string. The evaluator and
the GRPO rollout send it as a content list, `[{"type": "text", ...}]`. The chat
template strips the text's surrounding newlines on the list path but not the
string path, so the language SFT model was trained on

    <|im_start|>user\n\n[Task Description] ... output the json.\n<|im_end|>

and evaluated on

    <|im_start|>user\n[Task Description] ... output the json.<|im_end|>

GRPO was never affected: its rollout uses the evaluator's structure, verified
token for token in #26.

**How found.** While building the GP-VL SFT data, I compared the SFT prompt to
the evaluator's prompt token for token, instead of assuming they matched. The
same check on the language data showed the same mismatch.

**Why it matters.** The language SFT checkpoint is evaluated off its training
prompt. RL starts from it and trains on the evaluation prompt, so part of RL's
measured gain over SFT could be adapting to the prompt rather than learning.

**Measurement.** `scripts/eval_sft_train_prompt.py` re-evaluated the language
SFT checkpoint with its training prompt. The run hit its 3-hour background time
limit after 176 of the 200 in-distribution hands. The 176 are a prefix of the
fixed stream, cut by order rather than outcome, so they are a valid paired
sample:

| SFT checkpoint, same 176 hands | solved | first try | recovered |
| --- | ---: | ---: | ---: |
| evaluation prompt (as reported) | 30 | 25 | 5/151 |
| its own training prompt | 34 | 30 | 4/146 |

The difference is not significant: solved 4 vs 8 discordant (p = 0.39), first
try 3 vs 8 (p = 0.23). The direction slightly favours the training prompt.
Recovery collapses under both prompts, so "SFT destroys revision" is not a
prompt artefact.

Every conclusion holds even when SFT is given its own prompt, on these 176
hands:
- v5 beats SFT: 47 vs 5, p = 1.3e-9.
- SFT is still below Init: 15 vs 35, p = 0.0066.

**Decision: no retraining.** The language SFT and v5 stand as reported, with
this caveat recorded. Retraining SFT would also have forced retraining RL from
it, to keep the SFT→RL pairing.

**Fix.** `make_sft_data.py` now writes messages in the evaluator's exact
structure (content lists, image marker first), and that is verified: on GP-VL
the SFT prompt equals the evaluator's token for token on 12/12 sampled rows,
with identical pixels. The existing `data/sft_gp_language` was not regenerated
(see the decision above).

**Found alongside, while building GP-VL SFT.** `train_sft_qwen.py` defaulted to
`max_seq_length 1024`. mlx_vlm's trainer *skips* any example whose image tokens
do not fit, with only a log warning. A GP-VL example is about 1,735 tokens, of
which 1,444 are the image, so every example would have been dropped. The
default is now 2048, and the script refuses to train if any example exceeds
it. Language examples (about 290 tokens) were never affected.

## 29. GP-VL SFT ran out of Metal memory — **fixed: batch 1 × accumulation 4, jobs run alone** (mine)

To save time I started VL SFT while the base model's VL evaluation suite was
still running. That crashed both, with `[METAL] Command buffer execution
failed: Insufficient Memory`. The evaluation died 31 hands into rule-OOD, and
SFT died on its first training step. The usable GPU working set is about
107.5 GB of the 128 GB, and a GP-VL example carries 1,444 image tokens.

Nothing invalid was recorded:
- The partial rule-OOD log is quarantined under
  `logs/ablation/INVALID-oom__qwen_gp_vl_ruleood__init-n200/`.
- The completed in-distribution run (200/200) is kept. Its first 50 hands
  reproduce the earlier n = 50 GP-VL Init run exactly, 50/50.
- The failed SFT attempt's log is `train_ckpt/sft_gp_vl/train_oom_attempt1.log`.

**My first diagnosis here, "two jobs at once", was incomplete.** Retried
completely alone, SFT at batch 4 failed at the same point: the first training
step, inside the vision embedding. Forward-only validation at batch 4 ran
fine. Measured with real training steps, alone:

| batch | peak memory | time per step |
| ---: | ---: | ---: |
| 1 | 97.3 GB, stable over 4 steps | 8.7 s |
| 2 | out of memory | — |

97 GB for one ~1,745-token example is far more than the language stack
needs. A plausible cause, **not verified**, is that the frozen vision tower's
activations are retained under the gradient transform.

**Fix.** VL SFT runs at `--batch-size 1 --grad-accum 4 --iters 1000`. mlx_vlm
counts `iters` in micro-batches and steps the optimiser every `grad_accum` of
them (checked in `sft_trainer.py`). That is the same 250 updates of 4 examples
as the language SFT. Two differences:
- The loss is averaged per example and then over the 4, rather than token-
  weighted over the batch. Answers are all about 41 tokens, so this is
  negligible.
- The trainer's element-wise gradient clip at 1.0 now applies per example.
  LoRA gradients do not approach it.

This leaves about 10 GB of headroom, so **GP-VL jobs run alone, one after
another**. That is stated in `scripts/run_gp_vl_eval_suite.sh`, which also
gained a `SETTINGS=` filter so completed settings are not rerun. The batch-4
logs are kept as `train_oom_attempt1.log` (next to an evaluation) and
`train_oom_attempt2_batch4_alone.log`.

## 30. SFT data order was not seeded — **fixed for future runs; reproducibility only** (mine)

`train_sft_qwen.py` called `mx.random.seed(--seed)`, but mlx_vlm's
`iterate_batches` shuffles batches with `np.random.permutation`, which that call
does not seed. So runs with the same `--seed` trained on different example
orders, and `run.json`'s `"seed": 0` overstated reproducibility.

**How found.** A GP-VL SFT restart was expected to reproduce the cut-off run
exactly. Its iteration-40 training loss was 0.672 against 0.625.

**Impact.** Reproducibility only, not validity. The recipe and data are
identical, and the two runs track each other: validation loss was 0.146 at
iteration 200 in both. This affects `train_ckpt/sft_gp_language` and
`train_ckpt/sft_gp_vl`, which are not bit-reproducible.

**Fix.** The script now seeds `np.random` and `random` as well. The GP-VL SFT
run in progress when this was found was not restarted, since the only gain
would be reproducibility, and the language SFT already ran unseeded.
