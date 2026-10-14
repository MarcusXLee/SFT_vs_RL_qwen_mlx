"""Gradient-level audit of the GRPO update, using a toy LM (no GPU, no 8B model).

The earlier controlled-update test used 6 rows of identical 49-token completions,
so it could not exercise variable lengths, padding, partial microbatches, or the
separation of current/old/reference policies. These do, on a small model where
the exact gradient is cheap to compute twice.

    .venv-qwen/bin/python -m unittest discover -s scripts -p 'test_rl_gradients*.py'
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from rl.trainer.grpo_trainer_mlx import (
    grpo_objective, pad_batch, sequence_logprobs, sequence_logprobs_chunked,
    token_entropy)

VOCAB, DIM = 37, 16


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.proj = nn.Linear(DIM, VOCAB)

    def __call__(self, ids):
        return _Out(self.proj(self.emb(ids)))


class ToyModel(nn.Module):
    """Minimal stand-in exposing the attribute sequence_logprobs uses."""

    def __init__(self):
        super().__init__()
        self.language_model = _ToyLM()


def fixture(seed=0):
    """Variable-length prompts and completions, including a 1-token completion."""
    rng = np.random.default_rng(seed)
    prompts = [[int(x) for x in rng.integers(1, VOCAB, n)] for n in (5, 3, 4, 2, 6)]
    comps = [[int(x) for x in rng.integers(1, VOCAB, n)] for n in (3, 1, 5, 2, 4)]
    adv = mx.array(rng.normal(size=len(prompts)).astype(np.float32))
    return prompts, comps, adv


def flat(grads):
    return {k: np.array(v) for k, v in tree_flatten(grads)}


class GradientAccumulationTest(unittest.TestCase):
    """Summing microbatch gradients must equal the whole-batch gradient.

    This holds only because `denom` is the whole batch's completion-token count
    and is shared by every microbatch. A per-microbatch denominator would make
    short rows count more, which is the length bias the global denominator
    exists to avoid.
    """

    def _grads(self, model, input_ids, comp_mask, old_lp, ref_lp, adv, denom,
               microbatch, entropy_coef=0.0):
        total = None
        n = input_ids.shape[0]
        for start in range(0, n, microbatch):
            sl = slice(start, min(start + microbatch, n))

            def loss_fn(m):
                out = sequence_logprobs(m, input_ids[sl], comp_mask[sl], 1.0,
                                        want_entropy=bool(entropy_coef))
                ent = out[2] if entropy_coef else None
                loss, *_ = grpo_objective(out[0], old_lp[sl], ref_lp[sl], adv[sl],
                                          out[1], denom, 0.2, 0.01, ent, entropy_coef)
                return loss

            _, g = nn.value_and_grad(model, loss_fn)(model)
            mx.eval(g)
            total = g if total is None else _add(total, g)
        return total

    def _setup(self, seed=0):
        mx.random.seed(seed)
        model = ToyModel()
        prompts, comps, adv = fixture(seed)
        input_ids, comp_mask = pad_batch(prompts, comps, pad_id=0)
        old_lp, mask = sequence_logprobs_chunked(model, input_ids, comp_mask, 2, 1.0)
        ref_lp = old_lp
        denom = float(mask.sum().item())
        return model, input_ids, comp_mask, old_lp, ref_lp, adv, denom, comps

    def test_microbatched_equals_whole_batch(self):
        model, ids, cm, old, ref, adv, denom, comps = self._setup()
        whole = self._grads(model, ids, cm, old, ref, adv, denom, microbatch=len(comps))
        for mb in (1, 2, 3, 4):   # 3 and 4 leave a partial final microbatch
            chunked = self._grads(model, ids, cm, old, ref, adv, denom, microbatch=mb)
            a, b = flat(whole), flat(chunked)
            with self.subTest(microbatch=mb):
                for k in a:
                    np.testing.assert_allclose(a[k], b[k], rtol=2e-4, atol=2e-6,
                                               err_msg=f'microbatch={mb} param={k}')

    def test_holds_with_the_entropy_term_on(self):
        model, ids, cm, old, ref, adv, denom, comps = self._setup(seed=3)
        whole = self._grads(model, ids, cm, old, ref, adv, denom, len(comps), entropy_coef=0.02)
        chunked = self._grads(model, ids, cm, old, ref, adv, denom, 2, entropy_coef=0.02)
        a, b = flat(whole), flat(chunked)
        for k in a:
            np.testing.assert_allclose(a[k], b[k], rtol=2e-4, atol=2e-6)

    def test_denominator_counts_only_completion_tokens(self):
        _, _, cm, _, _, _, denom, comps = self._setup(seed=1)
        self.assertEqual(denom, float(sum(len(c) for c in comps)))

    def test_padding_contributes_no_gradient(self):
        """Appending pure padding to every row must not change the gradient."""
        model, ids, cm, old, ref, adv, denom, comps = self._setup(seed=2)
        base = self._grads(model, ids, cm, old, ref, adv, denom, 2)
        pad_ids = mx.concatenate([ids, mx.zeros((ids.shape[0], 4), ids.dtype)], axis=1)
        pad_cm = mx.concatenate([cm, mx.zeros((cm.shape[0], 4), cm.dtype)], axis=1)
        old2, mask2 = sequence_logprobs_chunked(model, pad_ids, pad_cm, 2, 1.0)
        self.assertEqual(float(mask2.sum().item()), denom, 'padding changed the denominator')
        padded = self._grads(model, pad_ids, pad_cm, old2, old2, adv, denom, 2)
        a, b = flat(base), flat(padded)
        for k in a:
            np.testing.assert_allclose(a[k], b[k], rtol=2e-4, atol=2e-6,
                                       err_msg=f'padding changed the gradient for {k}')


def _add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


class PolicySeparationTest(unittest.TestCase):
    def setUp(self):
        mx.random.seed(7)
        self.model = ToyModel()
        prompts, comps, self.adv = fixture(7)
        self.ids, self.cm = pad_batch(prompts, comps, pad_id=0)

    def test_ratio_is_one_before_any_update(self):
        old, mask = sequence_logprobs_chunked(self.model, self.ids, self.cm, 2, 1.0)
        cur, _ = sequence_logprobs(self.model, self.ids, self.cm, 1.0)
        ratio = np.array(mx.exp((cur - old) * mask))
        np.testing.assert_allclose(ratio[np.array(mask) > 0], 1.0, rtol=1e-6)

    def test_no_gradient_flows_into_old_or_reference_or_advantages(self):
        """Only the current policy may receive gradient."""
        old, mask = sequence_logprobs_chunked(self.model, self.ids, self.cm, 2, 1.0)
        denom = float(mask.sum().item())

        def loss_of(old_lp, ref_lp, adv):
            out = sequence_logprobs(self.model, self.ids, self.cm, 1.0)
            return grpo_objective(out[0], old_lp, ref_lp, adv, out[1], denom)[0]

        # old_lp and ref_lp are stop_gradient'd inside grpo_objective, so the
        # loss must be flat in them. (Advantages are deliberately NOT detached:
        # the loss depends on them linearly by design. What matters is that they
        # carry no path back to the model -- asserted separately below.)
        for name, idx in (('old_lp', 0), ('ref_lp', 1)):
            args = [old, old + 0.3, self.adv]
            g = mx.grad(lambda *a: loss_of(*a), argnums=idx)(*args)
            mx.eval(g)
            with self.subTest(term=name):
                self.assertLess(float(mx.abs(g).max().item()), 1e-9,
                                f'gradient leaked into {name}')

    def test_advantages_are_a_constant_leaf_in_the_real_update(self):
        """They come from numpy rewards, so no path exists back to the model."""
        source = (ROOT / 'rl/trainer/grpo_trainer_mlx.py').read_text()
        self.assertIn('adv = mx.array(advantages.reshape(-1)[idx])', source)
        self.assertIn('def group_advantages(rewards: np.ndarray', source)

    def test_old_and_reference_are_explicitly_detached(self):
        source = (ROOT / 'rl/trainer/grpo_trainer_mlx.py').read_text()
        self.assertIn('old_lp = mx.stop_gradient(old_lp)', source)
        self.assertIn('ref_lp = mx.stop_gradient(ref_lp)', source)

    def test_reference_snapshot_survives_an_optimizer_step(self):
        import mlx.optimizers as optim
        from rl.trainer.policy_mlx import FrozenReference
        # FrozenReference needs trainable params; the toy model's are all trainable
        ref = FrozenReference(self.model)
        before = {k: np.array(v) for k, v in ref.snapshot.items()}
        opt = optim.Adam(learning_rate=0.1)

        def loss_fn(m):
            out = sequence_logprobs(m, self.ids, self.cm, 1.0)
            return (out[0] * out[1]).sum()

        _, g = nn.value_and_grad(self.model, loss_fn)(self.model)
        opt.update(self.model, g)
        mx.eval(self.model.parameters())
        after = {k: np.array(v) for k, v in ref.snapshot.items()}
        moved = max(float(np.abs(np.array(v) - before[k]).max())
                    for k, v in dict(tree_flatten(self.model.trainable_parameters())).items())
        self.assertGreater(moved, 0, 'the optimizer step did nothing; test would be vacuous')
        for k in before:
            np.testing.assert_array_equal(before[k], after[k],
                                          err_msg=f'reference snapshot drifted for {k}')


class EntropyGradientTest(unittest.TestCase):
    """The bonus must actually raise entropy, not merely score it."""

    def test_a_step_on_the_entropy_term_increases_entropy(self):
        logits = mx.array(np.array([[[3.0, 0.5, 0.2, -1.0, 0.1]]], np.float32))
        mask = mx.ones((1, 1))

        def neg_entropy(x):
            return -(token_entropy(x) * mask).sum()

        before = float(token_entropy(logits)[0, 0].item())
        g = mx.grad(neg_entropy)(logits)
        stepped = logits - 0.05 * g          # descend the loss => raise entropy
        after = float(token_entropy(stepped)[0, 0].item())
        self.assertGreater(after, before,
                           f'entropy did not rise: {before:.5f} -> {after:.5f}')

    def test_only_masked_positions_contribute(self):
        ent = mx.array(np.array([[1.0, 2.0, 3.0]], np.float32))
        mask = mx.array(np.array([[1.0, 0.0, 1.0]], np.float32))
        self.assertEqual(float((ent * mask).sum().item()), 4.0)

    def test_entropy_gradient_is_zero_at_a_uniform_distribution(self):
        """Entropy is maximal there, so the bonus should stop pushing."""
        logits = mx.zeros((1, 1, 8))
        g = mx.grad(lambda x: -(token_entropy(x)).sum())(logits)
        self.assertLess(float(mx.abs(g).max().item()), 1e-6)


class LoggingIsInertTest(unittest.TestCase):
    """Measuring entropy at coef=0 must not perturb training in any way.

    The logging fix computes entropy even when it is not part of the objective,
    so arms with and without a bonus can be compared. If that measurement leaked
    into the graph it would silently change every control run.
    """

    def _grad(self, entropy_coef, supply_entropy):
        mx.random.seed(11)
        model = ToyModel()
        prompts, comps, adv = fixture(11)
        ids, cm = pad_batch(prompts, comps, pad_id=0)
        old, mask = sequence_logprobs_chunked(model, ids, cm, 2, 1.0)
        denom = float(mask.sum().item())

        def loss_fn(m):
            out = sequence_logprobs(m, ids, cm, 1.0, want_entropy=supply_entropy)
            ent = out[2] if supply_entropy else None
            return grpo_objective(out[0], old, old, adv, out[1], denom,
                                  0.2, 0.01, ent, entropy_coef)[0]

        _, g = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(g)
        return flat(g)

    def test_measuring_entropy_does_not_change_the_gradient(self):
        without = self._grad(entropy_coef=0.0, supply_entropy=False)
        measured = self._grad(entropy_coef=0.0, supply_entropy=True)
        for k in without:
            np.testing.assert_array_equal(
                without[k], measured[k],
                err_msg=f'entropy logging perturbed the gradient for {k}')

    def test_a_nonzero_coefficient_does_change_it(self):
        """Guard: otherwise the test above could pass by the term never applying."""
        base = self._grad(entropy_coef=0.0, supply_entropy=True)
        active = self._grad(entropy_coef=0.05, supply_entropy=True)
        self.assertTrue(any(not np.array_equal(base[k], active[k]) for k in base),
                        'entropy_coef had no effect; the inertness test is vacuous')

    def test_drawn_win_rate_counts_discarded_groups(self):
        from rl.trainer.grpo_trainer_mlx import _drawn_win_rate
        drawn = [[5.0, -1.0, -1.0, -1.0],   # kept (has spread)
                 [-1.0, -1.0, -1.0, -1.0]]  # discarded by dynamic sampling
        self.assertAlmostEqual(_drawn_win_rate(drawn, 5.0), 1 / 8)
        # the kept-only view would report 1/4 - twice as high
        self.assertAlmostEqual(_drawn_win_rate(drawn[:1], 5.0), 1 / 4)

    def test_drawn_win_rate_is_none_when_nothing_was_drawn(self):
        from rl.trainer.grpo_trainer_mlx import _drawn_win_rate
        self.assertIsNone(_drawn_win_rate([], 5.0))


class TrajectoryNormalisationTest(unittest.TestCase):
    """Per-trajectory normalisation must make length irrelevant.

    Token normalisation weights a trajectory by its length. In multi-turn that
    is systematically wrong: a trajectory stops on success, so failures run
    longer (measured 2.11x, 91.3% of token mass) and mean-centred advantages
    stop cancelling -- a standing downward push on the objective.
    """

    def _pg(self, lengths, advs, normalization):
        """Objective for trajectories of given lengths and advantages."""
        B, L = len(lengths), max(lengths)
        mask = np.zeros((B, L), np.float32)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1.0
        lp = np.full((B, L), -0.5, np.float32)
        denom = float(B) if normalization == 'trajectory' else float(mask.sum())
        loss, pg, _, _ = grpo_objective(
            mx.array(lp), mx.array(lp), mx.array(lp), mx.array(np.array(advs, np.float32)),
            mx.array(mask), denom, kl_coef=0.0, normalization=normalization)
        return float(loss.item())

    def test_token_normalisation_is_biased_by_length(self):
        """Equal and opposite advantages should cancel -- under token weighting
        they do not, when the negative one is longer."""
        biased = self._pg([2, 10], [+1.0, -1.0], 'token')
        self.assertGreater(abs(biased), 0.5, 'expected a large residual push')

    def test_trajectory_normalisation_cancels_regardless_of_length(self):
        for lengths in ([2, 10], [10, 2], [5, 5], [1, 50]):
            with self.subTest(lengths=lengths):
                self.assertAlmostEqual(self._pg(lengths, [+1.0, -1.0], 'trajectory'),
                                       0.0, places=5)

    def test_matches_token_mode_when_lengths_are_equal(self):
        """Single-turn completions are near-uniform, so the two should agree."""
        for advs in ([+1.0, -1.0], [+2.0, -0.5], [0.3, -0.3]):
            with self.subTest(advs=advs):
                self.assertAlmostEqual(self._pg([6, 6], advs, 'token'),
                                       self._pg([6, 6], advs, 'trajectory'), places=5)

    def test_empty_completion_row_does_not_produce_nan(self):
        v = self._pg([0, 5], [+1.0, -1.0], 'trajectory')
        self.assertFalse(np.isnan(v) or np.isinf(v))

    def test_microbatching_still_sums_to_the_whole_batch(self):
        """Gradient accumulation must survive the new normalisation."""
        mx.random.seed(5)
        model = ToyModel()
        prompts, comps, adv = fixture(5)
        ids, cm = pad_batch(prompts, comps, pad_id=0)
        old, _ = sequence_logprobs_chunked(model, ids, cm, 2, 1.0)
        denom = float(ids.shape[0])            # trajectories, not tokens

        def grads(microbatch):
            total = None
            for start in range(0, ids.shape[0], microbatch):
                sl = slice(start, min(start + microbatch, ids.shape[0]))

                def loss_fn(m):
                    out = sequence_logprobs(m, ids[sl], cm[sl], 1.0)
                    return grpo_objective(out[0], old[sl], old[sl], adv[sl], out[1],
                                          denom, normalization='trajectory')[0]

                _, g = nn.value_and_grad(model, loss_fn)(model)
                mx.eval(g)
                total = g if total is None else _add(total, g)
            return flat(total)

        whole, chunked = grads(len(comps)), grads(2)
        for k in whole:
            np.testing.assert_allclose(whole[k], chunked[k], rtol=2e-4, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
