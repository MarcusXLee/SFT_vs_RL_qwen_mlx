"""Differential test: the MLX/numpy RolloutStorage vs the original torch one.

GAE is easy to port subtly wrong -- an off-by-one in the value bootstrap, or
dropping the bad_masks term that stops advantage propagating across a time-limit
truncation. Rather than eyeball it, this runs both implementations on the same
random rollouts and asserts the returns match to float32 precision.

Skips when torch is absent (it lives in .venv, not .venv-qwen):
    .venv/bin/python -m unittest discover -s scripts -p 'test_rl_*.py'
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]

try:
    import torch
    from rl.trainer.storage_tz import RolloutStorage as TorchStorage
except ImportError as exc:  # pragma: no cover
    torch = None
    _reason = str(exc)

from rl.trainer.storage_mlx import RolloutStorage as NumpyStorage  # noqa: E402


class Discrete:
    """storage_tz branches on action_space.__class__.__name__ being 'Discrete'."""
    shape = (1,)


@unittest.skipIf(torch is None, 'torch not installed in this interpreter')
class GAEEquivalenceTest(unittest.TestCase):
    NUM_STEPS = 64

    def _fill(self, seed, terminal_rate, timelimit_rate):
        """Random rollout shared by both implementations."""
        rng = np.random.default_rng(seed)
        n = self.NUM_STEPS
        return {
            'rewards': rng.normal(size=(n, 1)).astype(np.float32),
            'values': rng.normal(size=(n + 1, 1)).astype(np.float32),
            # masks[t+1]==0 marks an episode boundary at step t
            'masks': (rng.random((n + 1, 1)) > terminal_rate).astype(np.float32),
            'bad_masks': (rng.random((n + 1, 1)) > timelimit_rate).astype(np.float32),
            'next_value': float(rng.normal()),
        }

    def _torch_returns(self, data, **kw):
        s = TorchStorage(self.NUM_STEPS, Discrete(), max_new_tokens=4)
        s.rewards = torch.from_numpy(data['rewards'].copy())
        s.value_preds = torch.from_numpy(data['values'].copy())
        s.masks = torch.from_numpy(data['masks'].copy())
        s.bad_masks = torch.from_numpy(data['bad_masks'].copy())
        s.returns = torch.zeros(self.NUM_STEPS + 1, 1)
        s.compute_returns(torch.tensor(data['next_value']), **kw)
        return s.returns.numpy()

    def _numpy_returns(self, data, **kw):
        s = NumpyStorage(self.NUM_STEPS, num_processes=1)
        s.rewards = data['rewards'].copy()
        s.value_preds = data['values'].copy()
        s.masks = data['masks'].copy()
        s.bad_masks = data['bad_masks'].copy()
        s.compute_returns(data['next_value'], **kw)
        return s.returns

    def test_all_four_branches_match(self):
        """use_gae x use_proper_time_limits, with real episode boundaries."""
        for seed in (0, 1, 2):
            for use_gae in (True, False):
                for proper in (True, False):
                    data = self._fill(seed, terminal_rate=0.2, timelimit_rate=0.1)
                    kw = dict(use_gae=use_gae, gamma=0.9, gae_lambda=0.95,
                              use_proper_time_limits=proper)
                    with self.subTest(seed=seed, use_gae=use_gae, proper=proper):
                        np.testing.assert_allclose(
                            self._numpy_returns(data, **kw),
                            self._torch_returns(data, **kw),
                            rtol=1e-6, atol=1e-6)

    def test_paper_hyperparameters_on_a_dense_boundary_rollout(self):
        """gamma/lambda from rl/configs/llama_gp_language.yaml, many terminals.

        GeneralPoints episodes are short (<=6 steps), so boundaries are dense and
        the masks terms dominate -- exactly where an off-by-one would hide.
        """
        data = self._fill(7, terminal_rate=0.5, timelimit_rate=0.3)
        kw = dict(use_gae=True, gamma=0.9, gae_lambda=0.95, use_proper_time_limits=False)
        np.testing.assert_allclose(self._numpy_returns(data, **kw),
                                   self._torch_returns(data, **kw),
                                   rtol=1e-6, atol=1e-6)

    def test_bad_masks_actually_change_the_result(self):
        """Guard against the port passing because bad_masks are all ones."""
        data = self._fill(3, terminal_rate=0.2, timelimit_rate=0.4)
        kw = dict(use_gae=True, gamma=0.9, gae_lambda=0.95, use_proper_time_limits=True)
        with_bad = self._numpy_returns(data, **kw)
        data_no_bad = {**data, 'bad_masks': np.ones_like(data['bad_masks'])}
        without = self._numpy_returns(data_no_bad, **kw)
        self.assertFalse(np.allclose(with_bad, without),
                         'bad_masks had no effect; the test would not detect dropping them')


class GeneratorTest(unittest.TestCase):
    def test_minibatches_partition_without_repeats(self):
        s = NumpyStorage(8, num_processes=1)
        advantages = np.arange(8, dtype=np.float32).reshape(8, 1)
        seen = []
        for batch in s.feed_forward_generator(advantages, mini_batch_size=2,
                                              rng=np.random.default_rng(0)):
            seen.extend(batch['advantages'].tolist())
        self.assertEqual(sorted(seen), list(range(8)))

    def test_oversized_minibatch_is_rejected(self):
        s = NumpyStorage(4, num_processes=1)
        with self.assertRaises(ValueError):
            list(s.feed_forward_generator(None, mini_batch_size=8))


if __name__ == '__main__':
    unittest.main()
