"""Tests for the multi-turn GRPO rollout that need neither a GPU nor the model.

The correctness risk specific to multi-turn is token/mask alignment: the chat
template inserts scaffold *between* turns, so a trajectory's token sequence is
not a concatenation of prompt and completions. A silent misalignment would shift
every log-prob after the first turn -- the exact fault class that only shows up
as "RL does not learn".

    .venv-qwen/bin/python -m unittest discover -s scripts -p 'test_rl_multiturn*.py'
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]

from rl.trainer.grpo_multiturn_mlx import (  # noqa: E402
    MultiTurnConfig, MultiTurnGRPOTrainer, _Traj)


def bare(cfg=None):
    t = MultiTurnGRPOTrainer.__new__(MultiTurnGRPOTrainer)
    t.cfg = cfg or MultiTurnConfig()
    return t


class PrefixInvariantTest(unittest.TestCase):
    """Re-templating must extend the sequence, never rewrite it."""

    def test_extension_is_accepted(self):
        t = bare()
        t._assert_prefix([1, 2, 3, 4, 5], [1, 2, 3], turn=0)   # must not raise

    def test_identical_is_accepted(self):
        t = bare()
        t._assert_prefix([1, 2, 3], [1, 2, 3], turn=0)

    def test_a_rewritten_prefix_raises(self):
        t = bare()
        with self.assertRaises(RuntimeError) as ctx:
            t._assert_prefix([1, 9, 3, 4], [1, 2, 3], turn=2)
        self.assertIn('turn 2', str(ctx.exception))
        self.assertIn('misaligned', str(ctx.exception))

    def test_a_shortened_sequence_raises(self):
        t = bare()
        with self.assertRaises(RuntimeError):
            t._assert_prefix([1, 2], [1, 2, 3], turn=1)


class TrajectoryRewardTest(unittest.TestCase):
    """'sum' must rank early success above late success above failure."""

    def _traj(self, rewards):
        tr = _Traj(payload=[], ids=[], mask=[], remaining=5)
        tr.rewards = list(rewards)
        tr.total_reward = float(sum(rewards))
        return tr

    def test_sum_ranks_earlier_success_higher(self):
        t = bare(MultiTurnConfig(reward_mode='sum'))
        first = t._traj_reward(self._traj([5.0]))                    # solved on try 1
        third = t._traj_reward(self._traj([-1.0, -1.0, 5.0]))        # solved on try 3
        never = t._traj_reward(self._traj([-1.0] * 5))               # never solved
        self.assertGreater(first, third)
        self.assertGreater(third, never)

    def test_final_mode_ignores_when_the_solution_arrived(self):
        t = bare(MultiTurnConfig(reward_mode='final'))
        first = t._traj_reward(self._traj([5.0]))
        third = t._traj_reward(self._traj([-1.0, -1.0, 5.0]))
        never = t._traj_reward(self._traj([-1.0] * 5))
        self.assertEqual(first, third, 'final mode should score both as solved')
        self.assertGreater(first, never)

    def test_sum_separates_trajectories_final_mode_ties(self):
        """The two modes encode different objectives; the choice is deliberate."""
        s = bare(MultiTurnConfig(reward_mode='sum'))
        f = bare(MultiTurnConfig(reward_mode='final'))
        a, b = self._traj([5.0]), self._traj([-1.0, -1.0, 5.0])
        self.assertNotEqual(s._traj_reward(a), s._traj_reward(b))
        self.assertEqual(f._traj_reward(a), f._traj_reward(b))


class MaskConstructionTest(unittest.TestCase):
    """Only assistant tokens may be credited."""

    def test_mask_covers_completions_and_nothing_else(self):
        # simulate: prompt(4) + comp(3) + scaffold+feedback(5) + comp(2)
        tr = _Traj(payload=[], ids=[1, 2, 3, 4], mask=[0.0] * 4, remaining=5)
        tr.ids += [10, 11, 12]; tr.mask += [1.0] * 3          # assistant turn 1
        nxt = tr.ids + [20, 21, 22, 23, 24]                   # template + feedback
        bare()._assert_prefix(nxt, tr.ids, turn=0)
        tr.mask += [0.0] * (len(nxt) - len(tr.ids)); tr.ids = nxt
        tr.ids += [30, 31]; tr.mask += [1.0] * 2              # assistant turn 2

        self.assertEqual(len(tr.ids), len(tr.mask))
        self.assertEqual(sum(tr.mask), 5, 'exactly the 3+2 generated tokens')
        credited = [i for i, m in zip(tr.ids, tr.mask) if m]
        self.assertEqual(credited, [10, 11, 12, 30, 31])
        for scaffold in (1, 2, 3, 4, 20, 21, 22, 23, 24):
            self.assertNotIn(scaffold, credited, 'scaffold/feedback must not be credited')


class PadSequencesTest(unittest.TestCase):
    """Multi-turn masks are built during rollout, not derived from a prompt length."""

    def test_pads_and_preserves_masks(self):
        import numpy as np
        from rl.trainer.grpo_trainer_mlx import pad_sequences
        seqs = [[1, 2, 3, 4, 5], [6, 7]]
        masks = [[0, 0, 1, 0, 1], [0, 1]]
        ids, m = pad_sequences(seqs, masks, pad_id=99)
        self.assertEqual(ids.shape, (2, 5))
        np.testing.assert_array_equal(np.array(ids), [[1, 2, 3, 4, 5], [6, 7, 99, 99, 99]])
        np.testing.assert_array_equal(np.array(m), [[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]])
        self.assertEqual(float(np.array(m).sum()), 3.0)

    def test_non_contiguous_mask_survives(self):
        """Unlike single-turn, credited tokens are NOT one contiguous block."""
        import numpy as np
        from rl.trainer.grpo_trainer_mlx import pad_sequences
        _, m = pad_sequences([[1, 2, 3, 4, 5, 6]], [[0, 1, 1, 0, 0, 1]], pad_id=0)
        np.testing.assert_array_equal(np.array(m)[0], [0, 1, 1, 0, 0, 1])

    def test_length_mismatch_raises(self):
        from rl.trainer.grpo_trainer_mlx import pad_sequences
        with self.assertRaises(ValueError):
            pad_sequences([[1, 2, 3]], [[0, 1]], pad_id=0)
        with self.assertRaises(ValueError):
            pad_sequences([[1], [2]], [[0]], pad_id=0)

    def test_multiturn_build_batch_uses_pad_sequences(self):
        import numpy as np
        t = bare()
        t.pad_id = 0
        ids, m = t._build_batch([[1, 2, 3], [4, 5]], [[0, 1, 1], [0, 1]])
        self.assertEqual(ids.shape, (2, 3))
        self.assertEqual(float(np.array(m).sum()), 3.0)


class PerUpdateStateResetTest(unittest.TestCase):
    """Both collect() implementations must reset per-update accumulators.

    Multi-turn overrides collect(), so anything the base class resets there has
    to be reset again. Missing one makes the accumulator grow across updates --
    hands_this_update did exactly that, which would have reported a ~100%
    repeat rate from the second update onward.
    """

    PER_UPDATE = ('drawn_rewards_last', 'hands_this_update')

    def test_both_collects_reset_the_same_accumulators(self):
        base = (ROOT / 'rl/trainer/grpo_trainer_mlx.py').read_text()
        multi = (ROOT / 'rl/trainer/grpo_multiturn_mlx.py').read_text()
        base_collect = base.split('def collect(self)')[1].split('\n    def ')[0]
        multi_collect = multi.split('def collect(self)')[1].split('\n    def ')[0]
        for name in self.PER_UPDATE:
            with self.subTest(attr=name):
                self.assertIn(f'self.{name} = []', base_collect,
                              f'single-turn collect does not reset {name}')
                self.assertIn(f'self.{name} = []', multi_collect,
                              f'multi-turn collect does not reset {name}')


class TemplateWhitespaceTest(unittest.TestCase):
    """The chat template strips whitespace from assistant content.

    encode(raw_text) therefore disagrees with the template's re-encoding of the
    same turn whenever the model emits surrounding whitespace, breaking the
    prefix invariant. That aborted a 52-update run. Stripping at the source
    makes the two agree by construction.
    """

    BREAKING = ('x\n', 'x ', '\nx', 'x\n\n', 'x\t', '   ')
    SAFE = ('x', '{\n "a": 1\n}', '')

    def test_rollout_strips_before_using_the_text(self):
        src = (ROOT / 'rl/trainer/grpo_multiturn_mlx.py').read_text()
        self.assertIn('text = raw_text.strip()', src)
        body = src.split('def _rollout_group')[1]
        # everything downstream must use the stripped name, not the raw one
        after = body.split('text = raw_text.strip()')[1]
        self.assertNotIn('raw_text', after,
                         'raw_text leaked past the strip; ids/payload could disagree')

    def test_stripping_neutralises_every_breaking_shape(self):
        for raw in self.BREAKING:
            with self.subTest(raw=repr(raw)):
                self.assertEqual(raw.strip(), raw.strip().strip(),
                                 'strip must be idempotent')
                self.assertNotIn(raw.strip()[:1], ('\n', ' ', '\t'))

    def test_safe_shapes_are_unchanged_by_stripping(self):
        for raw in self.SAFE:
            with self.subTest(raw=repr(raw)):
                self.assertEqual(raw, raw.strip())

    def test_internal_whitespace_is_preserved(self):
        """Only the ends are stripped; JSON formatting inside must survive."""
        raw = '{\n  "formula": "5*2+2+10=24"\n}'
        self.assertEqual(raw.strip(), raw)
        self.assertIn('\n', raw.strip())


if __name__ == '__main__':
    unittest.main()
