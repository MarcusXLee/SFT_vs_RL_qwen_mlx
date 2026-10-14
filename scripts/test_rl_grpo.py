"""Unit tests for the GRPO core math.

These are pure array-in/array-out functions, so the whole objective can be
verified without loading an 8B model. Each test pins a hazard that would
otherwise only show up as "training runs but does not learn".

    .venv-qwen/bin/python -m unittest discover -s scripts -p 'test_rl_grpo*.py'
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]

import mlx.core as mx  # noqa: E402

from rl.trainer.grpo_trainer_mlx import (  # noqa: E402
    active_groups, grpo_objective, group_advantages, pad_batch,
)


class AdvantageTest(unittest.TestCase):
    def test_group_mean_is_the_baseline(self):
        rewards = np.array([[5.0, -1.0, -1.0, -1.0]], dtype=np.float32)
        adv = group_advantages(rewards)
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=5)
        self.assertGreater(adv[0, 0], 0, 'the rewarded completion must be favoured')
        self.assertTrue((adv[0, 1:] < 0).all(), 'the rest must be discouraged')

    def test_zero_variance_group_gives_no_signal(self):
        rewards = np.full((1, 4), -1.0, dtype=np.float32)
        np.testing.assert_allclose(group_advantages(rewards), 0.0, atol=1e-6)

    def test_active_groups_flags_only_groups_with_spread(self):
        rewards = np.array([[5.0, -1.0], [-1.0, -1.0], [5.0, 5.0]], dtype=np.float32)
        np.testing.assert_array_equal(active_groups(rewards), [True, False, False])

    def test_std_normalisation_equalises_unequal_groups(self):
        """Why it is off by default: std division erases the size of a win."""
        rewards = np.array([[100.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        adv = group_advantages(rewards, normalize_std=True)
        self.assertAlmostEqual(float(adv[0, 0]), float(adv[1, 0]), places=3)

    def test_default_keeps_advantages_proportional_to_reward(self):
        rewards = np.array([[100.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        adv = group_advantages(rewards)
        self.assertAlmostEqual(float(adv[0, 0]) / float(adv[1, 0]), 100.0, places=3)

    def test_a_real_success_outweighs_varied_failures(self):
        """The pathology that motivated the default: with std normalisation a
        group containing a correct answer produced the same gradient magnitude
        as a group of eight wrong answers, one marginally worse."""
        success = np.array([[5, -1, -1, -1, -1, -1, -1, -1]], np.float32)
        failures = np.array([[-1, -1, -1, -1, -1, -1, -1, -2]], np.float32)
        norm = (abs(group_advantages(success, normalize_std=True)).max(),
                abs(group_advantages(failures, normalize_std=True)).max())
        self.assertAlmostEqual(float(norm[0]), float(norm[1]), places=1)

        raw = (abs(group_advantages(success)).max(), abs(group_advantages(failures)).max())
        self.assertGreater(float(raw[0]), 4 * float(raw[1]))

    def test_two_successes_beat_one(self):
        """With std normalisation two successes scored *lower* than none."""
        one = np.array([[5, -1, -1, -1, -1, -1, -1, -1]], np.float32)
        two = np.array([[5, 5, -1, -1, -1, -1, -1, -1]], np.float32)
        self.assertGreater(float(group_advantages(two).sum(where=group_advantages(two) > 0)),
                           0.0)
        # both should carry a large positive signal, unlike a varied-failure group
        failures = np.array([[-1, -1, -1, -1, -1, -1, -1, -2]], np.float32)
        for group in (one, two):
            self.assertGreater(float(abs(group_advantages(group)).max()),
                               4 * float(abs(group_advantages(failures)).max()))


class PadBatchTest(unittest.TestCase):
    def test_mask_covers_completion_tokens_only(self):
        ids, mask = pad_batch([[1, 2, 3], [1, 2]], [[7, 8], [9, 10, 11]], pad_id=0)
        self.assertEqual(ids.shape, (2, 5))
        np.testing.assert_array_equal(np.array(mask), [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]])

    def test_padding_fills_with_pad_id(self):
        ids, _ = pad_batch([[1, 2, 3]], [[7]], pad_id=99)
        # row is 4 long inside a width set by itself, so no padding here
        ids2, mask2 = pad_batch([[1, 2, 3], [1]], [[7, 8], [2]], pad_id=99)
        self.assertEqual(int(np.array(ids2)[1, -1]), 99)
        self.assertEqual(float(np.array(mask2)[1, -1]), 0.0, 'padding must never be credited')


class ObjectiveTest(unittest.TestCase):
    def _call(self, cur, old, ref, adv, mask, **kw):
        return grpo_objective(mx.array(cur), mx.array(old), mx.array(ref),
                              mx.array(adv), mx.array(mask),
                              denom=float(np.sum(mask)) or 1.0, **kw)

    def test_unchanged_policy_gives_ratio_one(self):
        lp = np.full((2, 3), -0.5, np.float32)
        mask = np.ones((2, 3), np.float32)
        adv = np.array([1.0, -1.0], np.float32)
        loss, pg, kl, _ = self._call(lp, lp, lp, adv, mask, kl_coef=0.0)
        # ratio == 1, so pg == sum(adv * mask) == (1 + -1) * 3 == 0
        self.assertAlmostEqual(float(pg.item()), 0.0, places=5)
        self.assertAlmostEqual(float(kl.item()), 0.0, places=6)

    def test_positive_advantage_pushes_loss_down(self):
        lp = np.full((1, 4), -0.5, np.float32)
        mask = np.ones((1, 4), np.float32)
        loss_pos, *_ = self._call(lp, lp, lp, np.array([1.0], np.float32), mask, kl_coef=0.0)
        loss_neg, *_ = self._call(lp, lp, lp, np.array([-1.0], np.float32), mask, kl_coef=0.0)
        self.assertLess(float(loss_pos.item()), float(loss_neg.item()))

    def test_kl_is_zero_against_itself_and_positive_otherwise(self):
        cur = np.full((1, 3), -0.5, np.float32)
        mask = np.ones((1, 3), np.float32)
        adv = np.array([0.0], np.float32)
        _, _, kl_same, _ = self._call(cur, cur, cur, adv, mask)
        _, _, kl_diff, _ = self._call(cur, cur, np.full((1, 3), -2.0, np.float32), adv, mask)
        self.assertAlmostEqual(float(kl_same.item()), 0.0, places=6)
        self.assertGreater(float(kl_diff.item()), 0.0, 'k3 estimator must be non-negative')

    def test_garbage_logprobs_at_padded_positions_do_not_produce_nan(self):
        """The hazard the masking-inside-the-exponent trick exists for.

        Padded positions hold meaningless logprobs. Computing exp() first and
        masking after gives inf * 0 = nan, which silently destroys the step.
        """
        cur = np.array([[-0.5, -0.5, -1e4]], np.float32)
        old = np.array([[-0.5, -0.5, +1e4]], np.float32)
        mask = np.array([[1.0, 1.0, 0.0]], np.float32)
        loss, pg, kl, _ = self._call(cur, old, cur, np.array([1.0], np.float32), mask)
        for name, value in (('loss', loss), ('pg', pg), ('kl', kl)):
            with self.subTest(term=name):
                self.assertFalse(np.isnan(float(value.item())), f'{name} is nan')
                self.assertFalse(np.isinf(float(value.item())), f'{name} is inf')

    def test_clipping_bounds_the_surrogate(self):
        """A huge ratio must not translate into an unbounded objective."""
        old = np.full((1, 2), -10.0, np.float32)
        cur = np.full((1, 2), -0.1, np.float32)      # ratio = exp(9.9), enormous
        mask = np.ones((1, 2), np.float32)
        _, pg, _, _ = self._call(cur, old, cur, np.array([1.0], np.float32), mask,
                                 clip_eps=0.2, kl_coef=0.0)
        self.assertLessEqual(float(pg.item()), 1.2 * 2 + 1e-4,
                             'positive advantage must be clipped at 1+eps per token')


class RewardRoutingTest(unittest.TestCase):
    """The RL reward must come from the environment, not a copy of its logic.

    A hand-rolled reward_fn omitted strip_formula_whitespace, so
    --strip-formula-whitespace was honoured by evaluation but ignored by
    training: the policy was optimised against a different objective than the
    one it was measured on.
    """

    SPACED = '{"cards": ["K","5","2","4"], "number": [10,5,2,4], "formula": "10 + 5 * 2 + 4 = 24"}'

    def _reward(self, strip):
        import importlib.util
        from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
        spec = importlib.util.spec_from_file_location(
            'train_grpo_qwen', ROOT / 'scripts/train_grpo_qwen.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        env = GeneralPointEnv_oneline(language_only=True, resolution=64,
                                      strip_formula_whitespace=strip)
        env.reset()
        env.cards_num = [10, 5, 2, 4]
        env.cards_without_suit = ['K', '5', '2', '4']
        env.solution = ['10+5*2+4']
        return mod.make_reward_fn(env)(self.SPACED, None)

    def test_whitespace_flag_reaches_the_training_reward(self):
        self.assertEqual(self._reward(strip=True), 5.0)
        self.assertEqual(self._reward(strip=False), -3.0)

    def test_each_completion_is_scored_as_a_fresh_attempt(self):
        """Scoring G completions must not exhaust the env's step budget."""
        import importlib.util
        from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
        spec = importlib.util.spec_from_file_location(
            'train_grpo_qwen', ROOT / 'scripts/train_grpo_qwen.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        env = GeneralPointEnv_oneline(language_only=True, resolution=64,
                                      verify_iter=2, strip_formula_whitespace=True)
        env.reset()
        env.cards_num = [10, 5, 2, 4]
        env.cards_without_suit = ['K', '5', '2', '4']
        env.solution = ['10+5*2+4']
        fn = mod.make_reward_fn(env)
        # more completions than verify_iter; none may be truncated
        self.assertEqual([fn(self.SPACED, None) for _ in range(6)], [5.0] * 6)


class CompletionIdsTest(unittest.TestCase):
    """The batched path must reproduce the ids generate() would have returned.

    Verified against real generations: token_ids == encode(text) + [eos], the
    extra id being 151645 (<|im_end|>). Without restoring it the batched and
    sequential paths would train on different sequences.
    """

    class _Tok:
        eos_token_id = 151645
        pad_token_id = 0
        def encode(self, text):
            return [1] * len(text)

    def _trainer(self, max_tokens):
        from rl.trainer.grpo_trainer_mlx import GRPOTrainer, GRPOConfig
        t = GRPOTrainer.__new__(GRPOTrainer)
        t.tokenizer = self._Tok()
        t.eos_id = self._Tok.eos_token_id
        t.stop_ids = {151645, 151643}
        t.truncated = 0
        t.cfg = GRPOConfig(max_tokens=max_tokens)
        return t

    def test_stop_token_is_restored(self):
        t = self._trainer(max_tokens=64)
        self.assertEqual(t._completion_ids('abc')[-1], 151645)
        self.assertEqual(len(t._completion_ids('abc')), 4)

    def test_truncated_completion_gets_no_stop_token(self):
        """Filling the budget means generation was cut, not that it stopped."""
        t = self._trainer(max_tokens=3)
        self.assertNotIn(151645, t._completion_ids('abc'))
        self.assertEqual(len(t._completion_ids('abc')), 3)

    def test_truncation_is_counted(self):
        t = self._trainer(max_tokens=3)
        t._completion_ids('abc')      # exactly at budget -> truncated
        t._completion_ids('ab')       # under budget -> terminated
        self.assertEqual(t.truncated, 1)

    def test_over_budget_is_also_treated_as_truncated(self):
        """Re-tokenisation can land at or past the budget; never append then."""
        t = self._trainer(max_tokens=3)
        self.assertNotIn(151645, t._completion_ids('abcd'))

    def test_ambiguous_stop_set_is_reported(self):
        t = self._trainer(max_tokens=64)
        self.assertTrue(t.stop_id_ambiguous)
        t.stop_ids = {151645}
        self.assertFalse(t.stop_id_ambiguous)

    def test_declared_stop_ids_reads_a_list(self):
        from rl.trainer.grpo_trainer_mlx import _declared_stop_ids

        class Cfg:
            eos_token_id = [151645, 151643]

        class Model:
            generation_config = Cfg()
            config = Cfg()

        self.assertEqual(_declared_stop_ids(Model(), None), {151645, 151643})

    def test_declared_stop_ids_falls_back_to_the_tokenizer(self):
        from rl.trainer.grpo_trainer_mlx import _declared_stop_ids

        class Bare:
            pass

        self.assertEqual(_declared_stop_ids(Bare(), 7), {7})


class DynamicSamplingTest(unittest.TestCase):
    """Groups with no reward spread must be discarded and redrawn.

    GeneralPoints reward is effectively binary per hand and the policy is
    near-deterministic per hand, so most groups come back all-correct or
    all-wrong. Those have zero advantage and contribute nothing: 19 of 60
    updates in the first GRPO run took no gradient step at all.
    """

    def _trainer(self, rewards_per_draw, dynamic, prompts_per_update=2,
                 group_size=4, max_draws=0):
        from rl.trainer.grpo_trainer_mlx import GRPOTrainer, GRPOConfig
        t = GRPOTrainer.__new__(GRPOTrainer)
        t.cfg = GRPOConfig(prompts_per_update=prompts_per_update, group_size=group_size,
                           dynamic_sampling=dynamic, dynamic_max_draws=max_draws,
                           batch_rollout=True)
        t.dead_groups = t.draws_last = t.kept_last = 0
        t.holdout_skips = 0
        seq = list(rewards_per_draw)
        t._reset_to_training_hand = lambda: {'Plain Cards': ['K', '5', '2', '4']}
        t._prompt_for = lambda info: 'PROMPT'
        t.tokenizer = type('T', (), {'encode': staticmethod(lambda x: [1, 2, 3])})()
        t._messages_for = lambda info: [{'role': 'user', 'content': 'PROMPT'}]
        t._sample_group_batched = lambda messages, ids: [(f'c{i}', [9]) for i in range(group_size)]
        def reward_fn(text, info):
            if not seq:
                return -1.0
            batch = seq[0]
            value = batch[reward_fn.i]
            reward_fn.i += 1
            if reward_fn.i == len(batch):
                reward_fn.i = 0
                seq.pop(0)
            return value
        reward_fn.i = 0
        t.reward_fn = reward_fn
        return t

    DEAD = [-1.0, -1.0, -1.0, -1.0]
    LIVE = [5.0, -1.0, -1.0, -1.0]

    def test_disabled_keeps_dead_groups(self):
        t = self._trainer([self.DEAD, self.DEAD], dynamic=False)
        _, _, rewards, _ = t.collect()
        self.assertEqual(rewards.shape, (2, 4))
        self.assertEqual(t.dead_groups, 0)

    def test_enabled_skips_dead_and_redraws(self):
        t = self._trainer([self.DEAD, self.LIVE, self.DEAD, self.LIVE], dynamic=True)
        _, _, rewards, _ = t.collect()
        self.assertEqual(rewards.shape, (2, 4), 'batch must still be filled')
        self.assertEqual(t.dead_groups, 2)
        self.assertEqual(t.draws_last, 4)
        for row in rewards:
            self.assertGreater(float(row.std()), 0, 'a kept group must have spread')

    def test_cap_stops_an_unfillable_batch(self):
        """All-dead checkpoints must not spin forever."""
        t = self._trainer([self.DEAD] * 50, dynamic=True, max_draws=5)
        _, _, rewards, _ = t.collect()
        self.assertEqual(rewards.shape[0], 0)
        self.assertEqual(t.draws_last, 5)

    def test_empty_batch_is_skipped_not_crashed(self):
        import numpy as np
        from rl.trainer.grpo_trainer_mlx import GRPOTrainer
        t = self._trainer([self.DEAD] * 50, dynamic=True, max_draws=3)
        prompts, completions, rewards, _ = t.collect()
        out = GRPOTrainer.update(t, prompts, completions, rewards)
        self.assertTrue(out['skipped'])
        self.assertEqual(out['groups_used'], 0)


class EntropyBonusTest(unittest.TestCase):
    """The bonus must reward higher entropy, and be inert when switched off.

    This is a *test* of whether entropy regularisation helps, not a fix for a
    confirmed cause: exploration collapse is measured (distinct completions
    7.8/8 -> 1.4/8 across checkpoints), but that it caused the evaluation
    outcome remains a hypothesis.
    """

    def test_uniform_logits_give_log_vocab(self):
        from rl.trainer.grpo_trainer_mlx import token_entropy
        import math
        V = 64
        ent = token_entropy(mx.zeros((1, 3, V)))
        self.assertAlmostEqual(float(ent[0, 0].item()), math.log(V), places=4)

    def test_peaked_logits_give_near_zero(self):
        from rl.trainer.grpo_trainer_mlx import token_entropy
        logits = mx.concatenate([mx.full((1, 1, 1), 50.0), mx.zeros((1, 1, 63))], axis=-1)
        self.assertLess(float(token_entropy(logits)[0, 0].item()), 1e-6)

    def test_entropy_is_scaled_by_temperature_via_logits(self):
        """Entropy is read off the same tempered logits as the log-probs."""
        from rl.trainer.grpo_trainer_mlx import token_entropy
        logits = mx.array([[[3.0, 0.0, 0.0, 0.0]]])
        hot = float(token_entropy(logits / 2.0)[0, 0].item())
        cold = float(token_entropy(logits * 2.0)[0, 0].item())
        self.assertGreater(hot, cold)

    def test_coefficient_zero_is_bit_identical(self):
        lp = np.full((2, 3), -0.5, np.float32)
        mask = np.ones((2, 3), np.float32)
        adv = np.array([1.0, -1.0], np.float32)
        args = (mx.array(lp), mx.array(lp), mx.array(lp), mx.array(adv), mx.array(mask), 6.0)
        base = grpo_objective(*args)
        withent = grpo_objective(*args, entropy=mx.ones((2, 3)), entropy_coef=0.0)
        self.assertEqual(float(base[0].item()), float(withent[0].item()))

    def test_higher_entropy_lowers_the_loss(self):
        lp = np.full((1, 3), -0.5, np.float32)
        mask = np.ones((1, 3), np.float32)
        adv = np.array([0.0], np.float32)
        args = (mx.array(lp), mx.array(lp), mx.array(lp), mx.array(adv), mx.array(mask), 3.0)
        low = grpo_objective(*args, entropy=mx.full((1, 3), 0.1), entropy_coef=0.05)[0]
        high = grpo_objective(*args, entropy=mx.full((1, 3), 2.0), entropy_coef=0.05)[0]
        self.assertLess(float(high.item()), float(low.item()),
                        'the bonus must make a higher-entropy policy cheaper')


class KLTailClampTest(unittest.TestCase):
    """The k3 penalty exponent is clamped at +-10 (inherited from mlx-rl).

    It was briefly +-5, sized on one token at d = 12.16 that turned out to be an
    artefact of the rollout sampling at corrupted positions (BUGS.md #26). After
    that fix an 8-update pilot never exceeded |d| 2.7, so these tests pin the
    mechanism, not a claim that +-10 is optimal.
    """
    CLAMP = 10.0

    def _penalty(self, d_value, clamp):
        d = mx.array([[float(d_value)]])
        dp = mx.clip(d, -clamp, clamp)
        return float(((mx.exp(dp) - dp - 1.0)).sum().item())

    def test_penalty_is_capped_at_the_bound(self):
        cap = self._penalty(self.CLAMP, self.CLAMP)
        for d in (12.16, 40.0):
            with self.subTest(d=d):
                self.assertAlmostEqual(self._penalty(d, self.CLAMP), cap, places=2)

    def test_normal_tokens_are_untouched(self):
        """The pilot's whole d range was within +-2.7; nothing there is clamped."""
        for d in (-2.7, -0.97, 0.0, 0.006, 1.0, 2.7, 9.9):
            with self.subTest(d=d):
                self.assertAlmostEqual(self._penalty(d, self.CLAMP), self._penalty(d, 1e9),
                                       places=3)

    def test_penalty_stays_non_negative_and_monotone(self):
        vals = [self._penalty(d, self.CLAMP) for d in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)]
        self.assertTrue(all(v >= 0 for v in vals))
        self.assertTrue(all(b >= a for a, b in zip(vals, vals[1:])))

    def test_reported_kl_is_still_unclamped(self):
        """The metric must keep showing a spike even though the penalty caps."""
        source = (ROOT / 'rl/trainer/grpo_trainer_mlx.py').read_text()
        self.assertIn('kl_sum = ((mx.exp(d) - d - 1.0) * mask).sum()', source)
        self.assertIn('d_pen = mx.clip(d, -10.0, 10.0)', source)


class HandRecurrenceTest(unittest.TestCase):
    """Track which hands training actually sees.

    Dynamic sampling redraws until a group has reward spread, which
    preferentially returns hands of intermediate difficulty -- so hands can
    recur. In a reproduction about memorisation that should be measured.
    """

    def test_first_sighting_is_not_a_repeat(self):
        from rl.trainer.grpo_trainer_mlx import _repeat_rate
        counts = {('A',): 1, ('B',): 1}
        self.assertEqual(_repeat_rate([('A',), ('B',)], counts), 0.0)

    def test_second_sighting_counts_as_a_repeat(self):
        from rl.trainer.grpo_trainer_mlx import _repeat_rate
        counts = {('A',): 2, ('B',): 1}
        self.assertEqual(_repeat_rate([('A',), ('B',)], counts), 0.5)

    def test_all_repeats(self):
        from rl.trainer.grpo_trainer_mlx import _repeat_rate
        self.assertEqual(_repeat_rate([('A',), ('B',)], {('A',): 3, ('B',): 7}), 1.0)

    def test_empty_update_returns_none(self):
        from rl.trainer.grpo_trainer_mlx import _repeat_rate
        self.assertIsNone(_repeat_rate([], {}))

    def test_counts_accumulate_across_updates(self):
        """A hand seen in update 1 must register as a repeat in update 2."""
        from rl.trainer.grpo_trainer_mlx import GRPOTrainer, _repeat_rate
        t = GRPOTrainer.__new__(GRPOTrainer)
        t.holdout, t.holdout_skips = set(), 0
        t.hand_counts, t.hands_this_update = {}, []

        class _Env:
            def __init__(self): self.i = 0
            def reset(self):
                hands = [['5', '2'], ['7', '3'], ['5', '2']]
                h = hands[self.i % len(hands)]; self.i += 1
                return None, {'Plain Cards': h}

        t.env = _Env()
        t.hands_this_update = []
        for _ in range(2):
            t._reset_to_training_hand()
        self.assertEqual(_repeat_rate(t.hands_this_update, t.hand_counts), 0.0)
        t.hands_this_update = []
        t._reset_to_training_hand()           # ['5','2'] again
        self.assertEqual(_repeat_rate(t.hands_this_update, t.hand_counts), 1.0)
        self.assertEqual(len(t.hand_counts), 2)


class MetricContractTest(unittest.TestCase):
    """update() and train() return different key sets; that must stay documented.

    A diagnostic once read a train()-only key from update()'s result, got None
    silently, and reported 0 active groups on every update while gradients were
    non-zero. The split is deliberate, so it is pinned rather than removed.
    """

    UPDATE_ONLY = ('groups_used', 'groups_total', 'grad_norm', 'completion_tokens')
    TRAIN_ADDS = ('group_rewards', 'success_rate', 'drawn_win_rate',
                  'hands_repeat_rate', 'best_completion')

    def setUp(self):
        self.source = (ROOT / 'rl/trainer/grpo_trainer_mlx.py').read_text()
        self.update_body = self.source.split('def update(self, prompts')[1].split('\n    def ')[0]
        self.train_body = self.source.split('def train(self, num_updates')[1].split('\n    def ')[0]

    def test_update_returns_the_optimisation_keys(self):
        for k in self.UPDATE_ONLY:
            with self.subTest(key=k):
                self.assertIn(f"'{k}'", self.update_body)

    def test_train_adds_the_rollout_keys(self):
        for k in self.TRAIN_ADDS:
            with self.subTest(key=k):
                self.assertIn(f'{k}=', self.train_body)

    def test_the_split_is_documented_on_update(self):
        doc = self.update_body.split('"""')[1]
        self.assertIn('split contract', doc)
        for k in ('group_rewards', 'success_rate'):
            self.assertIn(k, doc, f'{k} must be named as a train()-only key')

    def test_train_accounts_for_updates_that_did_nothing(self):
        """A run that never steps must say so, not print a confident summary."""
        self.assertIn('took a gradient step', self.train_body)
        self.assertIn('trained nothing', self.train_body)


class MetricDenominatorTest(unittest.TestCase):
    """Per-token metrics must divide by TOKENS, not by the loss denominator.

    Under trajectory normalisation the loss denominator is the trajectory count.
    Reusing it for kl_per_token divided a token-summed quantity by ~32 instead
    of ~6500, inflating the metric ~200x -- an ordinary run then reads as a
    divergence, which is exactly how it was misread.
    """

    def test_metrics_use_the_token_count(self):
        src = (ROOT / 'rl/trainer/grpo_trainer_mlx.py').read_text()
        self.assertIn("'kl_per_token': kl_sum / token_count", src)
        self.assertIn("'entropy_per_token': ent_sum / token_count", src)

    def test_loss_denominator_is_reported_separately(self):
        """Both quantities are logged so the distinction stays visible."""
        src = (ROOT / 'rl/trainer/grpo_trainer_mlx.py').read_text()
        self.assertIn("'loss_denominator': denom", src)
        self.assertIn("'completion_tokens': token_count", src)
        self.assertIn('token_count = float(mask.sum().item())', src)

    def test_denominator_still_follows_the_normalisation_mode(self):
        src = (ROOT / 'rl/trainer/grpo_trainer_mlx.py').read_text()
        self.assertIn("float(input_ids.shape[0]) if cfg.loss_normalization == 'trajectory'", src)


if __name__ == '__main__':
    unittest.main()
