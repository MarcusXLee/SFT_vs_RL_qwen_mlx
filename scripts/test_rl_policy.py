"""Verify the LoRA reference-policy trick against the real model.

The KL term in GRPO/PPO is measured against a frozen reference policy. Rather
than hold a second copy of 8B weights, every LoRA adapter's multiplicative
`scale` is zeroed, which should make the forward pass identical to the base
model. If that assumption is wrong the KL term is silently meaningless -- it
would compare the policy against itself (KL == 0, no regularisation) or against
something arbitrary.

Slow: loads the model once. Skipped unless the weights and an adapter exist.

    .venv-qwen/bin/python -m unittest discover -s scripts -p 'test_rl_policy*.py'
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]

MODEL = ROOT / 'models/Qwen3-VL-8B-Instruct'
ADAPTER = ROOT / 'train_ckpt/sft_gp_language'


@unittest.skipUnless(MODEL.is_dir() and (ADAPTER / 'adapter_config.json').exists(),
                     'needs the local model and a trained adapter')
class ReferencePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.trainer.utils import apply_lora_layers
        from rl.trainer.policy_mlx import lora_modules

        cls.mx = mx
        cls.model, processor = load(str(MODEL))
        cls.ids = mx.array([processor.tokenizer.encode(
            'Cards: 10, 5, 2, 4. Give a formula equal to 24.')])

        # Base logits BEFORE any adapter is attached.
        cls.base = cls._logits(cls.model, cls.ids)

        cls.model = apply_lora_layers(cls.model, str(ADAPTER))
        cls.model.eval()
        cls.adapters = lora_modules(cls.model)
        cls.tuned = cls._logits(cls.model, cls.ids)

    @staticmethod
    def _logits(model, ids):
        import mlx.core as mx
        out = model.language_model(ids).logits.astype(mx.float32)
        mx.eval(out)
        return out

    def test_adapters_were_actually_found(self):
        self.assertGreater(len(self.adapters), 0,
                           'lora_modules() found nothing; the KL term would be vacuous')
        self.assertEqual(len(self.adapters), 252)

    def test_the_adapter_changes_the_output(self):
        """If SFT changed nothing, the rest of this test would pass vacuously."""
        delta = float(self.mx.abs(self.tuned - self.base).max().item())
        self.assertGreater(delta, 1e-3, f'adapter had no effect (max delta {delta})')

    def test_zeroing_scales_restores_the_base_policy(self):
        from rl.trainer.policy_mlx import reference_policy
        with reference_policy(self.model):
            ref = self._logits(self.model, self.ids)
        delta = float(self.mx.abs(ref - self.base).max().item())
        self.assertLess(delta, 1e-4,
                        f'reference policy differs from the base model by {delta}')

    def test_scales_are_restored_afterwards(self):
        from rl.trainer.policy_mlx import reference_policy
        before = [m.scale for m in self.adapters]
        with reference_policy(self.model):
            pass
        self.assertEqual([m.scale for m in self.adapters], before)
        again = self._logits(self.model, self.ids)
        self.assertLess(float(self.mx.abs(again - self.tuned).max().item()), 1e-4,
                        'policy did not come back after the reference block')

    def test_scales_are_restored_even_on_exception(self):
        from rl.trainer.policy_mlx import reference_policy
        before = [m.scale for m in self.adapters]
        with self.assertRaises(RuntimeError):
            with reference_policy(self.model):
                raise RuntimeError('boom')
        self.assertEqual([m.scale for m in self.adapters], before)

    def test_frozen_reference_reproduces_the_policy_it_snapshotted(self):
        """The KL reference must be the SFT init, not the pre-SFT base.

        Zeroing adapter scales gives the base model, which is the wrong
        reference when RL starts from an SFT checkpoint: KL would pull back
        toward a model SFT deliberately moved away from, and starts large
        (0.67 nats/token was measured) instead of at zero.
        """
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten
        from rl.trainer.policy_mlx import FrozenReference

        reference = FrozenReference(self.model)

        # Perturb the adapters, as an optimizer step would.
        current = dict(tree_flatten(self.model.trainable_parameters()))
        bumped = {k: v + 0.05 for k, v in current.items()}
        self.model.update(tree_unflatten(list(bumped.items())))
        moved = self._logits(self.model, self.ids)
        self.assertGreater(float(self.mx.abs(moved - self.tuned).max().item()), 1e-4,
                           'perturbation had no effect; the test would be vacuous')

        # Inside the block we should see the snapshot, i.e. the SFT policy.
        with reference(self.model):
            ref = self._logits(self.model, self.ids)
        self.assertLess(float(self.mx.abs(ref - self.tuned).max().item()), 1e-4,
                        'frozen reference is not the snapshotted policy')

        # And the perturbed weights must come back afterwards.
        after = self._logits(self.model, self.ids)
        self.assertLess(float(self.mx.abs(after - moved).max().item()), 1e-4,
                        'policy weights were not restored')

        # Restore for any later test.
        self.model.update(tree_unflatten(list(current.items())))

    def test_frozen_reference_differs_from_zeroing_scales(self):
        """The two references are genuinely different objects here."""
        from rl.trainer.policy_mlx import FrozenReference, reference_policy
        reference = FrozenReference(self.model)
        with reference(self.model):
            frozen = self._logits(self.model, self.ids)
        with reference_policy(self.model):
            zeroed = self._logits(self.model, self.ids)
        self.assertGreater(float(self.mx.abs(frozen - zeroed).max().item()), 1e-3,
                           'SFT adapter is doing nothing, so the distinction is moot')

    def test_reference_policy_refuses_a_model_without_adapters(self):
        """Without adapters the KL term would be identically zero -- fail loudly."""
        from rl.trainer.policy_mlx import reference_policy
        import mlx.nn as nn
        with self.assertRaises(ValueError):
            with reference_policy(nn.Linear(4, 4)):
                pass


if __name__ == '__main__':
    unittest.main()
