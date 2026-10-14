"""Multi-turn GRPO: groups of full verifier-feedback trajectories.

Single-turn GRPO scores one attempt per hand, so it cannot train *revision* at
all. That matters here: after SFT, recovery from a first-attempt failure falls
to 1/38 against the base model's 11/38 (paired McNemar p = 0.0020), while the
evaluation protocol grants five attempts with verifier feedback. The policy is
graded partly on an ability the single-turn objective never touches.

This rolls out complete trajectories instead. A group is G trajectories on the
same hand; the reward is the trajectory's summed step reward, so solving on the
first attempt outranks solving on the third, which outranks never solving. Every
assistant turn in a trajectory is credited with that trajectory's advantage.

Reuses the single-turn objective unchanged -- advantages, clipped surrogate, k3
KL, entropy bonus, chunked log-probs, gradient accumulation. Only the rollout
and the token/mask construction differ.

Two correctness hazards are guarded rather than assumed:

  * the chat template inserts tokens *between* turns, so the trajectory's token
    sequence is NOT a concatenation of prompt and completions. Each turn's
    re-templated prefix is checked to extend the previous sequence exactly
    (`_assert_prefix`), and a violation raises instead of silently misaligning
    every log-prob after it.
  * only assistant tokens may be credited. Feedback turns and template scaffold
    are masked out by construction and asserted in tests.

A third was found only after runs had failed, and is now guarded too: the
rollout must sample in exactly the context the update scores. See
GRPOTrainer._batch_generate.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import numpy as np

from rl.trainer.grpo_trainer_mlx import GRPOConfig, GRPOTrainer, pad_sequences


@dataclass
class MultiTurnConfig(GRPOConfig):
    max_turns: int = 5              # matches env verify_iter
    reward_mode: str = 'final'      # 'final' success only, or 'sum' of step rewards
    # Token normalisation biases multi-turn: a trajectory stops on success, so
    # failures are systematically longer (measured 2.11x, 91.3% of token mass)
    # and mean-centred advantages stop cancelling. See grpo_objective.
    loss_normalization: str = 'trajectory'
    # entropy_coef is NOT the single-turn 0.01. Measured gradient contributions
    # on one batch each, at coef 0.01:
    #     single-turn, token norm   |grad| pg 0.333, entropy 0.00021 -> ratio 0.0006
    #     multi-turn, trajectory    |grad| pg 0.547, entropy 0.00179 -> ratio 0.0033
    # i.e. the same coefficient exerts 5.5x more relative pull here. Single-turn
    # bracketing found 0.01 helpful and 0.03 harmful, so 0.01 in this regime is
    # equivalent to ~0.055 there -- past the value already shown to hurt.
    # 0.002 reproduces the validated ratio. This is a principled transfer, not a
    # validated optimum: the measurement is one batch, and the useful range here
    # has not been bracketed.
    entropy_coef: float = 0.002


@dataclass
class _Traj:
    """One in-flight trajectory."""
    payload: list
    ids: List[int]
    mask: List[float]
    remaining: int
    verify_info: Optional[str] = None
    total_reward: float = 0.0
    turns: int = 0
    done: bool = False
    last_text: str = ''
    rewards: List[float] = field(default_factory=list)
    image: Optional[str] = None     # GP-VL: the rendered hand, one per group


class MultiTurnGRPOTrainer(GRPOTrainer):
    def _build_batch(self, sequences, masks, rows=None):
        """collect() already assembled full sequences and their masks."""
        return pad_sequences(sequences, masks, self.pad_id)

    def _assert_prefix(self, new_ids, old_ids, turn):
        if list(new_ids[:len(old_ids)]) != list(old_ids):
            raise RuntimeError(
                f'chat template did not extend the trajectory at turn {turn}: '
                f're-templating changed the existing {len(old_ids)} tokens, so the '
                f'completion mask and every log-prob after it would be misaligned')

    # The GP-VL trainer overrides these three; the text path is unchanged.
    _current_image = None

    def _first_payload(self, question):
        return [{"role": "user", "content": [{"type": "text", "text": question}]}]

    def _call_images(self, n):
        return None

    def _templated_ids(self, payload, add_generation_prompt=True):
        from mlx_vlm.prompt_utils import apply_chat_template
        text = apply_chat_template(self.processor, self.model.config, payload,
                                   add_generation_prompt=add_generation_prompt,
                                   num_images=0)
        return self.tokenizer.encode(text), text

    def _rollout_group(self, info):
        """Run `group_size` trajectories on the hand the env currently holds.

        Trajectories advance in lockstep, dropping those that have finished.
        Each turn batches only trajectories of equal token length, so from turn
        2 on most calls are batch-of-1: slower than one batch per turn, but a
        mixed-length batch samples padded rows at wrong positions.
        """
        cfg = self.cfg
        question = self.prompt_template.format(cards=info['Plain Cards'],
                                               **self.oracle_arguments)
        first = self._first_payload(question)
        base_ids, _ = self._templated_ids(first)

        trajs = [_Traj(payload=[dict(m) for m in first], ids=list(base_ids),
                       mask=[0.0] * len(base_ids), remaining=self.env.verify_iter,
                       image=self._current_image)
                 for _ in range(cfg.group_size)]

        for turn in range(cfg.max_turns):
            active = [t for t in trajs if not t.done]
            if not active:
                break
            # t.ids is exactly what the scorer will condition on. Trajectories
            # diverge after turn 1, so they are batched only with others of the
            # same length: mixing lengths pads, and padded rows are sampled at
            # wrong positions (see _batch_generate). Copies go in because
            # batch_generate re-templates them, and the payload must not change.
            by_len = {}
            for t in active:
                by_len.setdefault(len(t.ids), []).append(t)
            raw = {}
            for rows in by_len.values():
                texts = self._batch_generate([copy.deepcopy(t.payload) for t in rows],
                                             [t.ids for t in rows],
                                             images=self._call_images(len(rows)))
                raw.update({id(t): x for t, x in zip(rows, texts)})

            for t in active:
                raw_text = raw[id(t)]
                # The chat template STRIPS leading/trailing whitespace from
                # assistant content, so encode(raw) and the template's
                # re-encoding of the same turn disagree whenever the model emits
                # surrounding whitespace -- which broke the prefix invariant and
                # aborted a run at update 52. Verified: trailing newline/space/
                # tab and leading newline all break it; the stripped form always
                # holds. Strip once and use it for ids, payload and reward alike,
                # so the three cannot disagree. Surrounding whitespace carries no
                # information here: env.step parses JSON out of the text.
                text = raw_text.strip()
                comp = self._completion_ids(text)
                t.ids += list(comp)
                t.mask += [1.0] * len(comp)
                t.last_text = text
                t.turns += 1

                # Score through the environment, restoring this trajectory's own
                # attempt counters (the env is shared across the group).
                self.env.remaining_step = t.remaining
                self.env.verify_info = t.verify_info
                _, reward, done, truncated, step_info = self.env.step(text)
                t.remaining = self.env.remaining_step
                t.verify_info = self.env.verify_info
                t.total_reward += float(reward)
                t.rewards.append(float(reward))

                if done or truncated or turn == cfg.max_turns - 1:
                    t.done = True
                    continue

                t.payload.append({"role": "assistant",
                                  "content": [{"type": "text", "text": text}]})
                t.payload.append({"role": "user",
                                  "content": [{"type": "text",
                                               "text": f"You failed this trial because {t.verify_info}"}]})
                nxt, _ = self._templated_ids(t.payload)
                self._assert_prefix(nxt, t.ids, turn)
                t.mask += [0.0] * (len(nxt) - len(t.ids))
                t.ids = list(nxt)
            mx.clear_cache()

        return trajs

    def _traj_reward(self, t):
        if self.cfg.reward_mode == 'final':
            return float(self.cfg.success_reward if any(
                r == self.cfg.success_reward for r in t.rewards) else -1.0)
        return t.total_reward

    def collect(self):
        """Groups of trajectories, with the same dynamic-sampling filter."""
        cfg = self.cfg
        seqs, masks, rewards, texts = [], [], [], []
        self.batch_images = []          # aligned with seqs; None on the text path
        kept, draws = 0, 0
        max_draws = cfg.dynamic_max_draws or (cfg.prompts_per_update * 8)
        self.drawn_rewards_last = []
        self.hands_this_update = []

        while kept < cfg.prompts_per_update:
            if cfg.dynamic_sampling and draws >= max_draws:
                break
            draws += 1
            info = self._reset_to_training_hand()
            group = self._rollout_group(info)
            group_rewards = [self._traj_reward(t) for t in group]
            self.drawn_rewards_last.append(list(group_rewards))

            if cfg.dynamic_sampling and float(np.std(group_rewards)) <= 1e-6:
                self.dead_groups += 1
                continue

            for t, r in zip(group, group_rewards):
                self.batch_images.append(t.image)
                seqs.append(t.ids)
                masks.append(t.mask)
                texts.append(t.last_text)
                rewards.append(r)
            kept += 1

        self.draws_last, self.kept_last = draws, kept
        if kept == 0:
            return [], [], np.zeros((0, cfg.group_size), np.float32), []
        # prompts/completions are already merged into full sequences; update()
        # is handed the sequence and its mask through the same pad path.
        return (seqs, masks,
                np.array(rewards, np.float32).reshape(kept, cfg.group_size), texts)
