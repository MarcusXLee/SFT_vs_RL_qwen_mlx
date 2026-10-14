"""Trajectory rollout storage for the MLX trainers.

A port of `storage_tz.py` (itself the `a2c_ppo_acktr` RolloutStorage) with the
structure kept intact: GAE runs over **environment steps**, not over generated
tokens, and `masks` / `bad_masks` distinguish a true terminal state from a
time-limit truncation. That distinction is why this cannot be replaced by an
RLHF-style PPO trainer, where an episode is a single prompt->response pair.

The buffer is numpy-backed. Returns and advantages are a scalar recurrence over
at most a few hundred steps, so numpy is both simpler and faster than building
an MLX graph; arrays are converted to `mx` only at the point of loss
computation. `compute_returns` is numerically identical to the torch original
(see scripts/test_rl_storage.py, which diffs the two implementations directly).
"""
from __future__ import annotations

import numpy as np


class RolloutStorage:
    def __init__(self, num_steps: int, num_processes: int = 1):
        self.num_steps = num_steps
        self.num_processes = num_processes
        n, p = num_steps, num_processes

        # obs[i] holds whatever the policy needs to recompute step i (token ids,
        # pixel values, the image path...). Kept as Python objects, not arrays.
        self.obs = [{} for _ in range(n + 1)]

        self.rewards = np.zeros((n, p), dtype=np.float32)
        self.value_preds = np.zeros((n + 1, p), dtype=np.float32)
        self.returns = np.zeros((n + 1, p), dtype=np.float32)
        self.action_log_probs = np.zeros((n, p), dtype=np.float32)
        # 0 where an episode ended at this step, 1 otherwise
        self.masks = np.ones((n + 1, p), dtype=np.float32)
        # 0 where the episode was cut by a time limit rather than truly ending
        self.bad_masks = np.ones((n + 1, p), dtype=np.float32)

        self.step = 0

    def insert(self, obs, action_log_probs, value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1] = obs
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step] = value_preds
        self.rewards[self.step] = rewards
        self.masks[self.step + 1] = masks
        self.bad_masks[self.step + 1] = bad_masks
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0] = self.obs[-1]
        self.masks[0] = self.masks[-1]
        self.bad_masks[0] = self.bad_masks[-1]

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda,
                        use_proper_time_limits=True):
        """Generalised advantage estimation over environment steps."""
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0.0
                for step in reversed(range(self.rewards.shape[0])):
                    delta = (self.rewards[step]
                             + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                             - self.value_preds[step])
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    # A time-limit truncation is not a real terminal state, so it
                    # must not propagate advantage backwards past the cut.
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        (self.returns[step + 1] * gamma * self.masks[step + 1]
                         + self.rewards[step]) * self.bad_masks[step + 1]
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step])
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0.0
                for step in reversed(range(self.rewards.shape[0])):
                    delta = (self.rewards[step]
                             + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                             - self.value_preds[step])
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (self.returns[step + 1] * gamma
                                          * self.masks[step + 1] + self.rewards[step])

    def feed_forward_generator(self, advantages, mini_batch_size, rng=None):
        """Yield shuffled minibatches of transitions for the PPO epochs."""
        rng = rng or np.random.default_rng()
        batch_size = self.num_processes * self.num_steps
        if mini_batch_size > batch_size:
            raise ValueError(
                f'mini_batch_size {mini_batch_size} exceeds rollout size {batch_size}')
        indices = rng.permutation(batch_size)
        # drop_last=True, matching the torch BatchSampler in storage_tz.py
        for start in range(0, batch_size - mini_batch_size + 1, mini_batch_size):
            idx = indices[start:start + mini_batch_size]
            yield {
                'obs': [self.obs[i] for i in idx],
                'value_preds': self.value_preds[:-1].reshape(-1)[idx],
                'returns': self.returns[:-1].reshape(-1)[idx],
                'masks': self.masks[:-1].reshape(-1)[idx],
                'old_action_log_probs': self.action_log_probs.reshape(-1)[idx],
                'advantages': None if advantages is None else advantages.reshape(-1)[idx],
            }
