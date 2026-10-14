"""Pin the batched-rollout token reconstruction against the real model.

The sequential path returns the exact sampled token ids; batch_generate returns
only text, so the batched path rebuilds ids with `_completion_ids`. If that
rebuild is wrong, GRPO scores a different sequence than the one that earned the
reward, and log-probs and importance ratios are silently corrupted.

Slow: loads the model once. Skipped unless the weights and an adapter exist.
A wider sweep (80 samples over three max_tokens settings) was run once and is
recorded in QWEN_EVAL.md; this keeps a fast version in the suite.

    .venv-qwen/bin/python -m unittest discover -s scripts -p 'test_rl_tokens*.py'
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
class TokenReconstructionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.trainer.utils import apply_lora_layers
        from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG

        cls.mx = mx
        cls.model, cls.processor = load(str(MODEL))
        cls.model = apply_lora_layers(cls.model, str(ADAPTER))
        cls.model.eval()
        question = Q_GeneralPoint_EQN_L_EG.format(
            cards=['K', '5', '2', '4'], target_number='24',
            face_card_msg="'J', 'Q', and 'K' count as '10'.")
        cls.prompt = apply_chat_template(
            cls.processor, cls.model.config,
            [{"role": "user", "content": [{"type": "text", "text": question}]}],
            num_images=0)

    def _probe(self, max_tokens):
        from rl.trainer.grpo_trainer_mlx import (
            GRPOConfig, GRPOTrainer, _declared_stop_ids)
        t = GRPOTrainer.__new__(GRPOTrainer)
        t.tokenizer = self.processor.tokenizer
        t.eos_id = t.tokenizer.eos_token_id
        t.stop_ids = _declared_stop_ids(self.model, t.eos_id)
        t.truncated = 0
        t.cfg = GRPOConfig(max_tokens=max_tokens)
        return t

    def _samples(self, max_tokens, n):
        from mlx_vlm import generate
        self.mx.random.seed(11)
        out = []
        for _ in range(n):
            r = generate(self.model, self.processor, self.prompt, image=None,
                         max_tokens=max_tokens, temperature=1.0, verbose=False)
            out.append((r.text, list(r.token_ids or []), getattr(r, 'finish_reason', None)))
        return out

    def test_normal_termination_reconstructs_exactly(self):
        probe = self._probe(128)
        for text, truth, reason in self._samples(128, 4):
            with self.subTest(reason=reason):
                self.assertEqual(reason, 'stop')
                self.assertEqual(probe._completion_ids(text), truth)

    def test_truncated_generation_reconstructs_exactly(self):
        """No stop token may be appended when generation was cut by the budget."""
        probe = self._probe(12)
        for text, truth, reason in self._samples(12, 3):
            with self.subTest(reason=reason):
                self.assertEqual(reason, 'length')
                self.assertEqual(probe._completion_ids(text), truth)
                self.assertNotIn(probe.eos_id, probe._completion_ids(text)[-1:])

    def test_normal_termination_uses_the_expected_stop_token(self):
        """The rebuild appends tokenizer.eos_token_id; confirm that is what ends a
        completion. This model also declares <|endoftext|> as a stop id, which
        was never observed terminating one."""
        probe = self._probe(128)
        for text, truth, _ in self._samples(128, 4):
            self.assertEqual(truth[-1], probe.eos_id)

    def test_declared_stop_ids_match_the_model_config(self):
        probe = self._probe(128)
        self.assertIn(probe.eos_id, probe.stop_ids)


if __name__ == '__main__':
    unittest.main()
