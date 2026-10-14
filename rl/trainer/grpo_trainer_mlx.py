"""Single-turn GRPO on GeneralPoints, in MLX.

This is step 3 of the plan: a correctness smoke test for the RL loop *before*
the critic is added. GRPO drops the value network -- the baseline is the mean
reward of a group of completions for the same prompt -- which removes the most
common way a PPO port looks like it runs while learning nothing (a value head
that silently fails to train, so advantages are noise).

Deliberately single-turn: one hand, G sampled answers, one reward each, no
verifier-feedback retries and no GAE. If reward climbs here, generation ->
reward -> advantage -> gradient -> improvement is wired up correctly, and the
trajectory machinery in storage_mlx.py can be layered on for PPO with
confidence.

Numerical treatment (masking inside the ratio exponent, k3 KL with a clamped
penalty, a global token denominator, dropping zero-variance groups) follows
Cargo-Cult-Research/mlx-rl; see third_party/REFERENCES.md.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from rl.trainer.policy_mlx import FrozenReference, token_logprobs


@dataclass
class GRPOConfig:
    group_size: int = 8            # completions sampled per hand
    prompts_per_update: int = 4    # hands per update
    temperature: float = 1.0       # must be > 0 or the group has no variance
    max_tokens: int = 128
    clip_eps: float = 0.2
    kl_coef: float = 0.01
    learning_rate: float = 1e-6
    max_grad_norm: float = 1.0
    inner_epochs: int = 1
    microbatch: int = 4
    batch_rollout: bool = False  # sample a group in one batched pass
    dynamic_sampling: bool = False  # keep only groups with reward spread (DAPO 3.2)
    dynamic_max_draws: int = 0      # 0 -> prompts_per_update * 8
    normalize_advantage_std: bool = False  # see group_advantages; GRPO=True, Dr. GRPO=False
    entropy_coef: float = 0.0    # bonus on policy entropy; 0.0 is an exact no-op
    loss_normalization: str = 'token'   # 'token' or 'trajectory'; see grpo_objective
    log_entropy: bool = True     # measure entropy even when it is not in the loss
    success_reward: float = 5.0  # REWARD_FN['CORRECT_SOLUTION']
    save_every: int = 0          # 0 disables periodic checkpointing
    seed: int = 0


def group_advantages(rewards: np.ndarray, normalize_std: bool = False,
                     eps: float = 1e-4) -> np.ndarray:
    """rewards [P, G] -> advantages [P, G], using the group mean as baseline.

    `normalize_std` divides by the group's standard deviation, as in the
    original GRPO. That is actively harmful with this reward function, which
    emits many distinct negative values (-1, -1.5, -2, -3, and compounded
    values -- see BUGS.md #13). Dividing by std rescales every group to the same
    magnitude, so:

        [5,-1,-1,-1,-1,-1,-1,-1]  (one real success)  -> max|adv| 2.65
        [-1,-1,-1,-1,-1,-1,-1,-2] (all wrong, one     -> max|adv| 2.64
                                   marginally worse)
        [5,5,-1,-1,-1,-1,-1,-1]   (two real successes)-> max|adv| 1.73

    A group that solved the puzzle contributes no more than a group that failed
    eight different ways, seven wrong answers in the second group receive
    *positive* advantage at full scale, and two successes score lower than none.
    Worse, varied-failure groups are precisely the ones `active_groups` keeps,
    so they dominate whatever signal survives filtering.

    Leaving the advantages uncentred-by-std (Dr. GRPO, arXiv:2503.20783) keeps
    them proportional to real reward differences: 5.25 / 0.88 / 4.50 for the
    groups above. Default is therefore off.
    """
    adv = rewards - rewards.mean(axis=1, keepdims=True)
    if normalize_std:
        adv = adv / (rewards.std(axis=1, keepdims=True) + eps)
    return adv


def active_groups(rewards: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Groups with any reward spread.

    A zero-variance group carries no signal, but its kernel noise still yields a
    nonzero 'gradient' that Adam rescales into a full-size step -- a destructive
    random walk. Drop them.
    """
    return rewards.std(axis=1) > eps


def pad_batch(prompt_ids, completion_ids, pad_id):
    """Stack prompt+completion rows to equal length.

    Returns (input_ids [B, L], completion_mask [B, L]) where the mask is 1 on
    tokens the policy generated -- the only ones the objective may credit.
    """
    rows = [list(p) + list(c) for p, c in zip(prompt_ids, completion_ids)]
    width = max(len(r) for r in rows)
    ids = np.full((len(rows), width), pad_id, dtype=np.int32)
    mask = np.zeros((len(rows), width), dtype=np.float32)
    for i, (row, prompt) in enumerate(zip(rows, prompt_ids)):
        ids[i, :len(row)] = row
        mask[i, len(prompt):len(row)] = 1.0
    return mx.array(ids), mx.array(mask)


def token_entropy(logits: mx.array) -> mx.array:
    """Shannon entropy of the policy at each position. logits [B,L,V] -> [B,L].

    Computed from the same (temperature-scaled) logits the log-probs use, so it
    is the entropy of the distribution actually being sampled.
    """
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -(mx.exp(logp) * logp).sum(axis=-1)


def pad_sequences(sequences, masks, pad_id):
    """Pad already-assembled token sequences and their completion masks.

    Multi-turn trajectories interleave assistant turns with template scaffold
    and verifier feedback, so their mask cannot be derived from a prompt length
    the way pad_batch does it -- it is built during the rollout.
    """
    if len(sequences) != len(masks):
        raise ValueError(f'{len(sequences)} sequences but {len(masks)} masks')
    width = max(len(s) for s in sequences)
    ids = np.full((len(sequences), width), pad_id, dtype=np.int32)
    out = np.zeros((len(sequences), width), dtype=np.float32)
    for i, (seq, m) in enumerate(zip(sequences, masks)):
        if len(seq) != len(m):
            raise ValueError(f'row {i}: {len(seq)} tokens but {len(m)} mask entries')
        ids[i, :len(seq)] = seq
        out[i, :len(m)] = m
    return mx.array(ids), mx.array(out)


def sequence_logprobs(model, input_ids: mx.array, completion_mask: mx.array,
                      temperature: float = 1.0, want_entropy: bool = False,
                      logits_fn=None):
    """Per-token log-probs, shifted so entry t scores input_ids[t+1].

    Logits are divided by `temperature` first, because the rollout was sampled
    from softmax(logits / T). The importance ratio is only meaningful against the
    distribution that actually produced the tokens, so the policy being
    optimised must be the tempered one. The reference implementation does the
    same (`model_tz_llama.py`: `scores = scores * (1/self.temperature)`).

    At T == 1.0 this is exactly a no-op, which is why every run so far was
    unaffected -- but raising the temperature without this would have silently
    optimised a different distribution than the one being sampled.

    `logits_fn(model)`, if given, must return the logits for exactly these rows'
    `input_ids[:, :-1]`. It exists for GP-VL, whose forward needs the image (see
    grpo_vl_mlx.py). When None the text path below runs unchanged.
    """
    if logits_fn is None:
        logits = model.language_model(input_ids[:, :-1]).logits
    else:
        logits = logits_fn(model)
    logits = logits.astype(mx.float32)
    if temperature != 1.0:
        logits = logits / temperature
    lp = token_logprobs(logits, input_ids[:, 1:])
    mask = completion_mask[:, 1:]
    if not want_entropy:
        return lp, mask
    return lp, mask, token_entropy(logits)


def sequence_logprobs_chunked(model, input_ids, completion_mask, microbatch,
                              temperature: float = 1.0, logits_fn_for=None):
    """sequence_logprobs over row chunks, materialising one chunk at a time.

    The float32 logits are [rows, seq, vocab]; at vocab 151936 a 32-row batch of
    400-token sequences is 7.8 GB, and the old/reference passes need two of
    them. Chunking keeps the peak at one microbatch, which is what the training
    loop already uses.
    """
    parts = []
    for start in range(0, input_ids.shape[0], microbatch):
        sl = slice(start, min(start + microbatch, input_ids.shape[0]))
        lp, _ = sequence_logprobs(model, input_ids[sl], completion_mask[sl], temperature,
                                  logits_fn=logits_fn_for(sl) if logits_fn_for else None)
        mx.eval(lp)
        parts.append(mx.stop_gradient(lp))
    return mx.concatenate(parts, axis=0), completion_mask[:, 1:]


def grpo_objective(cur_lp, old_lp, ref_lp, advantages, mask, denom,
                   clip_eps=0.2, kl_coef=0.01, entropy=None, entropy_coef=0.0,
                   normalization='token'):
    """Clipped surrogate + k3 KL penalty (+ optional entropy bonus).

    Returns (loss, pg_sum, kl_sum, entropy_sum). With entropy_coef == 0 the
    entropy term is not computed at all, so the objective is bit-identical to
    the version without it.
    """
    old_lp = mx.stop_gradient(old_lp)
    ref_lp = mx.stop_gradient(ref_lp)

    # Mask inside the exponent: logprobs at padded positions are garbage and
    # would overflow to inf, poisoning the sum as inf * 0 = nan.
    ratio = mx.exp((cur_lp - old_lp) * mask)
    adv = advantages[:, None]
    surrogate = mx.minimum(ratio * adv,
                           mx.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    pg_sum = (surrogate * mask).sum()

    d = (ref_lp - cur_lp) * mask
    kl_sum = ((mx.exp(d) - d - 1.0) * mask).sum()   # k3, reported unclamped

    # k3's gradient is 1-exp(d), so a few far-tail tokens would otherwise
    # dominate the step direction. +-10 is inherited from mlx-rl, whose recorded
    # fix was introducing a clamp at all, not this value.
    #
    # It was briefly +-5, sized on a single multi-turn token at d = +12. That
    # token turned out to be an artefact of the rollout sampling at corrupted
    # positions (BUGS.md #26), so the reason for +-5 went with it. After the
    # rollout fix, an 8-update pilot never had a credited token beyond |d| 2.7,
    # so the bound did not act at either value; update() logs d_max, d_min and
    # d_beyond_5/_10 so a longer run shows whether it ever does.
    d_pen = mx.clip(d, -10.0, 10.0)
    kl_pen_tok = mx.exp(d_pen) - d_pen - 1.0
    kl_pen = (kl_pen_tok * mask).sum()

    # Subtracted, because the bonus rewards *higher* entropy. When entropy_coef
    # is 0 the entropy is still reported if supplied, but under stop_gradient
    # and never added to the loss -- so measuring it cannot perturb training.
    ent_tok = None
    ent_sum = mx.array(0.0)
    if entropy is not None:
        ent_tok = (entropy if entropy_coef else mx.stop_gradient(entropy)) * mask
        ent_sum = ent_tok.sum()

    if normalization == 'trajectory':
        # Each trajectory contributes the MEAN over its own credited tokens, so
        # every trajectory counts once regardless of length.
        #
        # Token normalisation is wrong whenever length correlates with reward.
        # In multi-turn it always does: a trajectory stops the instant it
        # succeeds, so failures run to the retry limit and successes do not.
        # Measured: failures were 2.11x longer (5.00 vs 2.38 attempts) and held
        # 91.3% of the token mass. Mean-centred advantages then no longer
        # cancel -- summing over trajectories gives 0.00 by construction, while
        # summing weighted by token count gave -210, a standing downward push
        # on the objective. All three terms are normalised the same way; mixing
        # a per-trajectory pg with a token-summed KL would silently rescale
        # their ratio.
        #
        # The trade-off is deliberate, not neutral: a 250-token trajectory and
        # a 50-token one now contribute equal gradient mass.
        rows = mx.maximum(mask.sum(axis=1), 1.0)        # guard empty completions
        pg_term = ((surrogate * mask).sum(axis=1) / rows).sum()
        kl_term = ((kl_pen_tok * mask).sum(axis=1) / rows).sum()
        ent_term = (((ent_tok).sum(axis=1) / rows).sum()
                    if ent_tok is not None else mx.array(0.0))
    else:
        pg_term, kl_term, ent_term = pg_sum, kl_pen, ent_sum

    loss = -pg_term + kl_coef * kl_term
    if entropy_coef:
        loss = loss - entropy_coef * ent_term
    return loss / denom, pg_sum, kl_sum, ent_sum


class GRPOTrainer:
    """Owns the env, the policy and the update loop."""

    def __init__(self, model, processor, env, prompt_template, oracle_arguments,
                 reward_fn, config: GRPOConfig, log_path=None, holdout=None):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.env = env
        self.prompt_template = prompt_template
        self.oracle_arguments = oracle_arguments
        self.reward_fn = reward_fn
        self.cfg = config
        self.optimizer = optim.Adam(learning_rate=config.learning_rate)
        # Hands reserved for evaluation. The SFT data generator excludes these,
        # so RL must too, or the SFT-vs-RL comparison is run on different terms.
        self.holdout = set(holdout or ())
        self.holdout_skips = 0
        self._tokenisation_checked = False
        self._diversity_checked = False
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # KL reference is the policy as it stands right now -- the SFT checkpoint
        # when resuming from one, the base model when starting fresh (a new LoRA
        # has lora_b == 0, so it is an identity). See FrozenReference.
        self.reference = FrozenReference(model)
        pad = getattr(self.tokenizer, "pad_token_id", None)
        self.pad_id = pad if pad is not None else self.tokenizer.eos_token_id
        self.eos_id = getattr(self.tokenizer, 'eos_token_id', None)
        self.stop_ids = _declared_stop_ids(model, self.eos_id)
        if self.eos_id is not None and self.eos_id not in self.stop_ids:
            raise ValueError(
                f'tokenizer eos_token_id {self.eos_id} is not among the stop ids '
                f'the model declares ({sorted(self.stop_ids)}); the batched '
                f'rollout would append a token generation never emits')
        self.truncated = 0
        self.dead_groups = 0
        self.hand_counts = {}
        self.hands_this_update = []
        self.draws_last = 0
        self.kept_last = 0
        self.drawn_rewards_last = []
        mx.random.seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

    # ---------------------------------------------------------------- rollout

    def _messages_for(self, info):
        question = self.prompt_template.format(cards=info['Plain Cards'],
                                               **self.oracle_arguments)
        return [{"role": "user", "content": [{"type": "text", "text": question}]}]

    def _prompt_for(self, info):
        from mlx_vlm.prompt_utils import apply_chat_template
        return apply_chat_template(self.processor, self.model.config,
                                   self._messages_for(info), num_images=0)

    def _batch_generate(self, conversations, prompt_ids, images=None):
        """batch_generate over message lists, guarded against two rollout defects.

        Both made the rollout sample from a different distribution than the one
        update() scores, silently -- the text still looked like normal output.

        1. Pass MESSAGES, never a templated string. batch_generate applies the
           chat template itself, and apply_chat_template does not detect input
           that is already templated: the string was wrapped a second time,
           nesting the whole conversation inside one user turn followed by an
           empty assistant turn (measured: 42 prompt tokens where the scorer
           conditions on 34). Every batched GRPO run before this fix, single-
           and multi-turn, sampled in that context. Given the message list,
           batch_generate conditions on exactly the scorer's ids.

        2. All prompts in one call must have the same length. mlx_vlm 0.7.2
           gives left-padded rows wrong RoPE positions from the second
           generated token on -- a row with 20 pad tokens decoded at positions
           17, -2, -1, 0 instead of 17, 18, 19, 20, because the padding is
           subtracted twice (per-row cache offset, and again via rope_deltas).
           Tokens sampled in padded rows scored a mean of -1.2 to -1.3 nats
           under the model, 72 of 3040 below -15, against -0.14 and none with
           no padding. Equal lengths mean no padding, so it cannot trigger.
           Single-turn groups are always equal; multi-turn trajectories diverge
           from turn 2, which is where every KL spike came from.

        The token-count check is what would have caught (1) on the first
        rollout: batch_generate must process exactly the tokens the scorer
        will condition on.
        """
        from mlx_vlm import batch_generate
        from mlx_vlm.sample_utils import make_sampler
        lengths = {len(p) for p in prompt_ids}
        if len(lengths) != 1:
            raise ValueError(f'prompts of different lengths {sorted(lengths)} in one '
                             f'batch_generate call; padded rows get wrong positions')
        response = batch_generate(
            self.model, self.processor, images=images, prompts=conversations,
            max_tokens=self.cfg.max_tokens,
            sampler=make_sampler(temp=self.cfg.temperature), verbose=False)
        seen = getattr(getattr(response, 'stats', None), 'prompt_tokens', None)
        expected = sum(len(p) for p in prompt_ids)
        if seen is None or int(seen) != expected:
            raise RuntimeError(
                f'batch_generate processed {seen} prompt tokens, the scorer '
                f'conditions on {expected}: the rollout is not sampling in the '
                f'context it will be scored in')
        texts = list(response.texts)
        if len(texts) != len(conversations):
            raise RuntimeError(f'batch_generate returned {len(texts)} completions '
                               f'for {len(conversations)} prompts')
        return texts

    def _check_prompt_tokenisation(self, result, prompt_ids):
        if self._tokenisation_checked:
            return
        self._tokenisation_checked = True
        reported = getattr(result, 'prompt_tokens', None)
        if reported is not None and int(reported) != len(prompt_ids):
            raise RuntimeError(
                f'prompt tokenisation disagrees: generate() saw {reported} tokens, '
                f'tokenizer.encode() produced {len(prompt_ids)}. The completion '
                f'boundary used for scoring would not match the one sampled, so '
                f'every log-prob ratio would be misaligned.')

    def _reset_to_training_hand(self, max_draws=1000):
        """Draw a hand that is not reserved for evaluation.

        Records which hands training actually sees. Dynamic sampling redraws
        until a group has reward spread, which preferentially returns hands of
        intermediate difficulty -- so the same hands can recur many times. In a
        reproduction *about memorisation* that is worth measuring rather than
        assuming away.
        """
        for _ in range(max_draws):
            obs, info = self.env.reset()
            key = tuple(sorted(info['Plain Cards']))
            if key not in self.holdout:
                self.hand_counts[key] = self.hand_counts.get(key, 0) + 1
                self.hands_this_update.append(key)
                self._last_obs = obs        # the rendered cards, for GP-VL
                return info
            self.holdout_skips += 1
        raise RuntimeError(
            f'could not draw a non-holdout hand in {max_draws} attempts; '
            f'the holdout set ({len(self.holdout)} hands) may be too large')

    def collect(self):
        """Sample hands x `group_size` completions until the batch is full.

        With `dynamic_sampling` (DAPO 3.2) a hand is discarded unless its group
        shows some reward spread. GeneralPoints reward is effectively binary per
        hand, and the policy is near-deterministic per hand, so most groups come
        back all-correct or all-wrong; those have zero advantage by construction
        and contribute nothing. Measured on the first GRPO run: 19 of 60 updates
        took no gradient step at all, and by the end only 0.73 of 4 groups per
        update carried any signal.

        Note this trains on a *biased* slice -- hands the policy is currently
        uncertain about -- which is the intent, but it does mean the training
        distribution differs from the evaluation distribution.
        """
        cfg = self.cfg
        prompts, completions, rewards, texts = [], [], [], []
        kept = 0
        self.drawn_rewards_last = []
        self.hands_this_update = []
        max_draws = cfg.dynamic_max_draws or (cfg.prompts_per_update * 8)
        draws = 0
        while kept < cfg.prompts_per_update:
            if cfg.dynamic_sampling and draws >= max_draws:
                break
            draws += 1
            # The environment must still be holding this hand when reward_fn
            # calls env.step, so a group is always sampled and scored before
            # the next reset. That is also why batching is within a hand only.
            info = self._reset_to_training_hand()
            text = self._prompt_for(info)
            prompt_ids = self.tokenizer.encode(text)
            group = (self._sample_group_batched(self._messages_for(info), prompt_ids)
                     if cfg.batch_rollout
                     else self._sample_group_sequential(text, prompt_ids))
            group_rewards = [self.reward_fn(out_text, info) for out_text, _ in group]
            mx.clear_cache()

            self.drawn_rewards_last.append(list(group_rewards))
            if cfg.dynamic_sampling and float(np.std(group_rewards)) <= 1e-6:
                self.dead_groups += 1
                continue

            for (out_text, ids), reward in zip(group, group_rewards):
                prompts.append(prompt_ids)
                completions.append(ids)
                texts.append(out_text)
                rewards.append(reward)
            kept += 1

        self.draws_last = draws
        self.kept_last = kept
        if kept == 0:
            return [], [], np.zeros((0, cfg.group_size), np.float32), []
        return (prompts, completions,
                np.array(rewards, np.float32).reshape(kept, cfg.group_size), texts)

    def _sample_group_sequential(self, text, prompt_ids):
        """One generate() call per completion. The reference implementation."""
        from mlx_vlm import generate
        cfg = self.cfg
        out = []
        for _ in range(cfg.group_size):
            result = generate(self.model, self.processor, text, image=None,
                              max_tokens=cfg.max_tokens,
                              temperature=cfg.temperature, verbose=False)
            # The ratio is only meaningful if the sequence we score is the
            # sequence that was sampled. generate() tokenises the prompt
            # internally; if that disagrees with tokenizer.encode here, the
            # prompt/completion boundary is off and every log-prob is
            # silently misaligned. Check once.
            self._check_prompt_tokenisation(result, prompt_ids)
            ids = list(result.token_ids) if getattr(result, "token_ids", None) \
                else self.tokenizer.encode(result.text)
            out.append((result.text, ids))
        return out

    def _sample_group_batched(self, messages, prompt_ids):
        """Whole group in one batched pass. Opt-in via GRPOConfig.batch_rollout.

        Decoding at batch 1 is memory-bandwidth-bound on the weights, so a group
        of G costs little more than one sequence and the rollout -- ~90% of an
        update -- speeds up accordingly. Measured 3.97x wall-clock at G=4.

        Takes the message list, not the templated prompt: see _batch_generate.

        batch_generate returns only text, not token ids, so completions are
        re-tokenised. Against the sequential path's ground-truth ids,
        `encode(text)` matched on every sample except for one trailing stop
        token, and `decode(encode(t)) == t` held on 494 real outputs.
        `_completion_ids` restores the stop token.
        """
        cfg = self.cfg
        out_texts = self._batch_generate([messages] * cfg.group_size,
                                         [prompt_ids] * cfg.group_size)
        self._check_group_diversity(out_texts)
        return [(t, self._completion_ids(t)) for t in out_texts]

    def _completion_ids(self, out_text):
        """Token ids for a batched completion, matching what generate() returns.

        generate() reports the sampled ids including the stop token that ended
        the completion; tokenizer.encode of the decoded text does not contain
        it, since it is not part of the text (mlx_vlm drops the token whose
        finish_reason is "stop" before detokenising). Append it back -- unless
        the completion filled the whole budget, in which case generation was cut
        by max_tokens and no stop token was ever emitted, so adding one would
        teach the model to stop at an arbitrary truncation point.

        Validated against the sequential path's ground-truth ids: 80/80 exact
        across normal termination (max_tokens=128), forced truncation
        (max_tokens=12) and the boundary (max_tokens=40). All 40 normal
        terminations ended on 151645 <|im_end|>.

        Caveat: this model declares two stop ids, 151645 <|im_end|> and 151643
        <|endoftext|>. Only the former was ever observed. A completion ending on
        the latter would get the wrong final token -- one token in ~48, on a
        sample type not seen in validation. `stop_id_ambiguous` records that the
        risk exists; `truncated` tracks how often the budget was hit.
        """
        ids = self.tokenizer.encode(out_text)
        if len(ids) >= self.cfg.max_tokens:
            self.truncated += 1
            return ids
        if self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    @property
    def stop_id_ambiguous(self):
        """True when the model declares more than one stop id (see _completion_ids)."""
        return len(self.stop_ids) > 1

    def _check_group_diversity(self, out_texts):
        """A group of identical completions has zero reward spread and is dropped.

        If batched sampling shared one RNG draw across the batch, every group
        would collapse and training would silently do nothing at all. Checked on
        the first group only; temperature > 0 is enforced by the launcher.
        """
        if self._diversity_checked:
            return
        self._diversity_checked = True
        if len(set(out_texts)) == 1 and len(out_texts) > 1:
            raise RuntimeError(
                'batched rollout produced identical completions across the '
                'group, so every group would have zero reward spread and no '
                'update would ever occur. Use the sequential path '
                '(batch_rollout=False).')

    # ----------------------------------------------------------------- update

    def _prepare_forward(self, input_ids):
        """Hook for a forward pass that needs more than token ids.

        Returns None here: the text path calls model.language_model directly.
        The GP-VL trainer returns `sl -> logits_fn`, built from image features
        computed once per update (grpo_vl_mlx.py).
        """
        return None

    def _build_batch(self, prompts, completions, rows=None):
        """Prompt ids + completion ids -> (padded ids, completion mask).

        Overridden by the multi-turn trainer, which assembles sequences and
        masks during the rollout instead. `rows` are the indices update() kept
        out of the collected batch; only the GP-VL trainer needs them.
        """
        return pad_batch(prompts, completions, self.pad_id)

    def update(self, prompts, completions, rewards):
        """One optimisation step. Returns per-update metrics.

        NOTE the split contract: this returns the optimisation-side keys only --
        loss, pg, kl, kl_per_token, entropy_per_token, grad_norm, grad_clipped,
        reward_mean/max, completion_tokens, groups_used/total, draws,
        dead_groups, truncated_completions, holdout_skips, skipped,
        d_max/d_min/d_beyond_5/d_beyond_10 (ref - old log-prob, credited tokens).

        `train()` then ADDS the rollout-side keys: step, seconds, success_rate,
        group_rewards, drawn_win_rate, drawn_groups, hands_drawn,
        hands_repeat_rate, hands_distinct_total, best_completion.

        A caller that drives update() directly (diagnostics, tests) sees only
        the first set. Reading a train()-only key there returns None rather than
        raising -- which is how a diagnostic once reported 0 active groups on
        every update while gradients were in fact non-zero.
        """
        cfg = self.cfg
        if rewards.shape[0] == 0:
            return {'skipped': True, 'reason': 'no group with reward spread was found',
                    'groups_used': 0, 'groups_total': 0,
                    'dead_groups': self.dead_groups, 'draws': self.draws_last,
                    'holdout_skips': self.holdout_skips}
        keep = active_groups(rewards)
        if not keep.any():
            return {'skipped': True, 'reason': 'all groups had zero reward spread',
                    'reward_mean': float(rewards.mean()), 'reward_max': float(rewards.max()),
                    'groups_used': 0, 'groups_total': len(keep),
                    'holdout_skips': self.holdout_skips}

        advantages = group_advantages(rewards, cfg.normalize_advantage_std)
        flat_keep = np.repeat(keep, cfg.group_size)
        idx = np.flatnonzero(flat_keep)
        prompts = [prompts[i] for i in idx]
        completions = [completions[i] for i in idx]
        adv = mx.array(advantages.reshape(-1)[idx])

        input_ids, comp_mask = self._build_batch(prompts, completions, rows=idx)
        logits_fn_for = self._prepare_forward(input_ids)   # None on the text path

        # Log-probs at rollout time (the "old" policy) and under the frozen
        # reference. Both are chunked: see sequence_logprobs_chunked.
        old_lp, mask = sequence_logprobs_chunked(
            self.model, input_ids, comp_mask, cfg.microbatch, cfg.temperature,
            logits_fn_for)
        with self.reference(self.model):
            ref_lp, _ = sequence_logprobs_chunked(
                self.model, input_ids, comp_mask, cfg.microbatch, cfg.temperature,
                logits_fn_for)
        mx.eval(old_lp, ref_lp)

        # Where the KL clamp acts. The penalty's d is ref - cur, and cur equals
        # old until the first optimiser step, so this is the d every token enters
        # the (first) epoch with. The kl sum alone cannot say whether any token
        # reached the +-bound, and the bound was once chosen from one token.
        m_np = np.array(mask) > 0
        d_np = (np.array(ref_lp) - np.array(old_lp))[m_np]
        d_stats = ({'d_max': float(d_np.max()), 'd_min': float(d_np.min()),
                    'd_beyond_5': int((np.abs(d_np) > 5).sum()),
                    'd_beyond_10': int((np.abs(d_np) > 10).sum())}
                   if d_np.size else {})

        # One denominator for the whole batch, so microbatch losses sum to the
        # batch loss under gradient accumulation (no GRPO length bias).
        #
        # token_count is tracked separately because METRICS must always be
        # per-token: reusing denom made 'kl_per_token' and 'completion_tokens'
        # hold per-trajectory values under trajectory normalisation, inflating
        # them ~200x so an ordinary run read as a divergence.
        token_count = float(mask.sum().item()) or 1.0
        denom = (float(input_ids.shape[0]) if cfg.loss_normalization == 'trajectory'
                 else token_count)
        n = input_ids.shape[0]
        # Diagnostics are summed within an epoch and reported for the LAST one.
        # Accumulating across epochs would scale kl_per_token by inner_epochs
        # and silently mislead any tuning of kl_coef.
        pg_sum = kl_sum = ent_sum = total_loss = 0.0
        grad_norm = 0.0
        for _ in range(max(1, cfg.inner_epochs)):
            accumulated = None
            pg_sum = kl_sum = ent_sum = total_loss = 0.0
            for start in range(0, n, cfg.microbatch):
                sl = slice(start, min(start + cfg.microbatch, n))

                # nn.value_and_grad differentiates a scalar function and has no
                # has_aux, so the diagnostic sums are captured by closure. They
                # are nodes in the same lazy graph and are evaluated alongside
                # the loss below.
                aux = {}

                def loss_fn(model):
                    out = sequence_logprobs(model, input_ids[sl], comp_mask[sl],
                                            cfg.temperature,
                                            want_entropy=bool(cfg.entropy_coef)
                                            or cfg.log_entropy,
                                            logits_fn=logits_fn_for(sl)
                                            if logits_fn_for else None)
                    cur_lp, m = out[0], out[1]
                    ent = out[2] if len(out) > 2 else None
                    loss, pg, kl, ent_sum = grpo_objective(
                        cur_lp, old_lp[sl], ref_lp[sl], adv[sl], m, denom,
                        cfg.clip_eps, cfg.kl_coef, ent, cfg.entropy_coef,
                        cfg.loss_normalization)
                    aux['pg'], aux['kl'], aux['ent'] = pg, kl, ent_sum
                    return loss

                loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
                mx.eval(loss, grads, aux['pg'], aux['kl'], aux['ent'])
                total_loss += float(loss.item())
                accumulated = grads if accumulated is None else _tree_add(accumulated, grads)
                pg_sum += float(aux['pg'].item())
                kl_sum += float(aux['kl'].item())
                ent_sum += float(aux['ent'].item())

            accumulated, grad_norm = _clip_grads(accumulated, cfg.max_grad_norm)
            self.optimizer.update(self.model, accumulated)
            mx.eval(self.model.parameters(), self.optimizer.state)

        mx.clear_cache()
        return {'skipped': False, 'holdout_skips': self.holdout_skips,
                'loss': total_loss, 'pg': pg_sum, 'kl': kl_sum,
                'kl_per_token': kl_sum / token_count,
                'grad_norm': grad_norm, 'grad_clipped': bool(grad_norm > cfg.max_grad_norm),
                'entropy_per_token': ent_sum / token_count,
                'reward_mean': float(rewards.mean()), 'reward_max': float(rewards.max()),
                # These were BOTH `denom` before. Under trajectory normalisation denom
                # is the trajectory count, so 'completion_tokens' held trajectories
                # and the per-token metrics were inflated ~200x -- an ordinary run
                # then reads as a divergence, which is how it was misread.
                'completion_tokens': token_count, 'loss_denominator': denom,
                'groups_used': int(keep.sum()), 'groups_total': len(keep),
                'draws': self.draws_last, 'dead_groups': self.dead_groups,
                'truncated_completions': self.truncated, **d_stats}

    # ------------------------------------------------------------------ train

    def save_adapter(self, path):
        """Write the adapter weights, and the config MLX needs to load them back.

        `apply_lora_layers` reads adapter_config.json from the directory to
        rebuild the LoRA topology before loading weights; without it the
        checkpoint cannot be evaluated at all. The topology is whatever the
        model currently carries, so it is serialised from model.config.lora.
        """
        from mlx.utils import tree_flatten
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path), dict(tree_flatten(self.model.trainable_parameters())))

        config_path = path.parent / 'adapter_config.json'
        if not config_path.exists():
            lora = getattr(self.model.config, 'lora', None)
            if lora is None:
                raise RuntimeError(
                    'model.config.lora is unset, so adapter_config.json cannot be '
                    'written and the checkpoint would not be loadable')
            config_path.write_text(json.dumps(lora, indent=2) + '\n')
        return path

    def train(self, num_updates):
        cfg = self.cfg
        history = []
        stepped = 0          # updates that actually moved the parameters
        idle_run = 0         # consecutive updates that did not
        for step in range(num_updates):
            t0 = time.monotonic()
            prompts, completions, rewards, texts = self.collect()
            metrics = self.update(prompts, completions, rewards)
            # Record what the policy actually produced. The first run degraded
            # from 22% to 12% and the aggregate metrics gave no clue why; the
            # per-group reward vectors show whether a batch carried real
            # successes or only varied failures.
            metrics.update(
                step=step, seconds=time.monotonic() - t0,
                success_rate=(float((rewards == cfg.success_reward).mean())
                              if rewards.size else 0.0),
                group_rewards=rewards.tolist(),
                # Over every completion generated this update, including the
                # groups dynamic sampling threw away. `success_rate` above is
                # over KEPT groups only, which selection-biases it upward.
                drawn_win_rate=_drawn_win_rate(self.drawn_rewards_last, cfg.success_reward),
                drawn_groups=len(self.drawn_rewards_last),
                hands_drawn=[list(h) for h in self.hands_this_update],
                hands_repeat_rate=_repeat_rate(self.hands_this_update, self.hand_counts),
                hands_distinct_total=len(self.hand_counts),
                best_completion=_best_completion(texts, rewards))
            took_step = bool(metrics.get('grad_norm')) and not metrics.get('skipped')
            stepped += took_step
            idle_run = 0 if took_step else idle_run + 1
            if idle_run and idle_run % 5 == 0:
                print(f'  [warn] {idle_run} consecutive updates took no gradient step '
                      f'(all groups had zero reward spread)', flush=True)
            history.append(metrics)
            if self.log_path:
                with self.log_path.open('a') as fh:
                    fh.write(json.dumps(metrics) + '\n')
            # A multi-hour run must not lose everything to one interruption --
            # including a failure to checkpoint. A periodic save that cannot
            # complete is reported and skipped; the caller's final save still
            # raises, so an unloadable checkpoint never passes silently.
            if cfg.save_every and self.log_path and (step + 1) % cfg.save_every == 0:
                try:
                    self.save_adapter(self.log_path.parent / f'adapters-{step + 1:05d}.safetensors')
                except Exception as exc:  # noqa: BLE001
                    print(f'  [warn] checkpoint at update {step + 1} failed: {exc}', flush=True)
            print(f"[{step:3d}] reward {metrics.get('reward_mean', float('nan')):+.3f}  "
                  f"success {metrics['success_rate']:.1%}  "
                  f"kl/tok {metrics.get('kl_per_token', 0):.2e}  "
                  f"d [{metrics.get('d_min', 0):+.1f},{metrics.get('d_max', 0):+.1f}]  "
                  f"groups {metrics.get('groups_used', 0)}/{metrics.get('groups_total', 0)}  "
                  f"{metrics['seconds']:.0f}s", flush=True)
        print(f'\nupdates that took a gradient step: {stepped}/{len(history)}')
        if stepped == 0:
            print('  [warn] the policy was never updated; this run trained nothing')
        return history


def _repeat_rate(hands_this_update, hand_counts):
    """Fraction of this update's hands that training had already seen before."""
    if not hands_this_update:
        return None
    # count > 1 means the hand had been drawn at least once before this update
    return sum(1 for h in hands_this_update if hand_counts.get(h, 0) > 1) / len(hands_this_update)


def _drawn_win_rate(drawn_rewards, success_reward):
    """Win rate over every completion drawn, kept or discarded."""
    flat = [r for group in drawn_rewards for r in group]
    if not flat:
        return None
    return sum(1 for r in flat if r == success_reward) / len(flat)


def _best_completion(texts, rewards, limit=400):
    """The highest-rewarded completion of the batch, truncated, for the log."""
    if not texts or rewards.size == 0:
        return None
    flat = rewards.reshape(-1)
    n = min(len(texts), flat.shape[0])
    if n == 0:
        return None
    return texts[int(np.argmax(flat[:n]))][:limit]


def _declared_stop_ids(model, fallback):
    """Stop ids the model declares, from its generation config where available."""
    ids = set()
    for holder in (getattr(model, 'generation_config', None), getattr(model, 'config', None)):
        value = getattr(holder, 'eos_token_id', None)
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple)):
            ids.update(int(v) for v in value)
    if not ids and fallback is not None:
        ids.add(int(fallback))
    return ids


def _tree_add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


def _clip_grads(grads, max_norm):
    """Clip, and report the PRE-clip global norm.

    If that norm routinely exceeds max_norm the effective step size is
    max_norm/||g||, which decouples it from the configured learning rate --
    worth knowing before reasoning about the learning rate at all.
    """
    if not max_norm:
        from mlx.utils import tree_flatten
        total = mx.sqrt(sum((v ** 2).sum() for _, v in tree_flatten(grads)))
        return grads, float(total.item())
    clipped, total = optim.clip_grad_norm(grads, max_norm)
    return clipped, float(total.item())
