"""GP-VL GRPO pieces that can be checked without the model weights.

MODEL-FREE: loads only the processor (tokenizer + image preprocessor), never
the 8B weights, so it is safe to run while a GP-VL job holds the GPU
(memory: gpu-exclusive-during-vl-jobs). Everything that needs the model lives in
scripts/verify_vl_grpo.py.

  * the rollout's image preprocessing (batch_generate: process_image -> RGB,
    then prepare_inputs) gives the same ids and pixels as the evaluator's
    (generate(): the PNG path straight into prepare_inputs);
  * _templated_ids expands the image to one token per merged patch, and the
    multi-turn prefix invariant still holds with the image in the history;
  * the scorer refuses a row slice it did not prepare inputs for.
"""
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np  # noqa: E402

MODEL = ROOT / 'models/Qwen3-VL-8B-Instruct'


def load_processor_only(path=MODEL):
    """The processor mlx_vlm.load() builds, without loading any weights.

    Importing the model package registers mlx_vlm's own Qwen3-VL processor;
    without it AutoProcessor reaches for torchvision, absent here.
    """
    import mlx_vlm.models.qwen3_vl  # noqa: F401
    from mlx_vlm.utils import load_config, load_image_processor, load_processor
    config = load_config(path)
    processor = load_processor(path, True, eos_token_ids=config.get('eos_token_id'))
    image_processor = load_image_processor(path)
    if image_processor is not None:
        processor.image_processor = image_processor
    return processor, config


def _setup():
    from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
    from prompt_lib.prompts import Q_GeneralPoint_EQN_VL_EG
    from rl.trainer.grpo_vl_mlx import MultiTurnVLGRPOTrainer
    from rl.trainer.grpo_multiturn_mlx import MultiTurnConfig

    processor, config = load_processor_only()
    random.seed(5)
    env = GeneralPointEnv_oneline(language_only=False, resolution=1200, target_points=24,
                                  treat_face_cards_as_10=True, face_cards_color='black')
    obs, info = env.reset(seed=5)

    from types import SimpleNamespace
    model = SimpleNamespace(config=SimpleNamespace(**config))   # config only; no weights

    t = MultiTurnVLGRPOTrainer.__new__(MultiTurnVLGRPOTrainer)
    t.processor, t.tokenizer, t.model = processor, processor.tokenizer, model
    t.cfg = MultiTurnConfig(max_tokens=128)
    t.eos_id, t.truncated = processor.tokenizer.eos_token_id, 0
    t.image_token_id = config.get('image_token_id') or config.get('image_token_index')
    t.image_dir = Path(tempfile.mkdtemp())
    t._hand_seq, t._image_inputs, t._last_obs = 0, {}, obs
    question = Q_GeneralPoint_EQN_VL_EG.format(
        target_number='24', face_card_msg="'J', 'Q', and 'K' count as '10'.")
    return t, info, question, processor, config


class VLRolloutInputsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.t, cls.info, cls.question, cls.proc, cls.config = _setup()
        cls.first = cls.t._first_payload(cls.question)
        cls.ids, cls.text = cls.t._templated_ids(cls.first)

    def test_eval_and_rollout_preprocessing_agree(self):
        """generate() feeds the path; batch_generate feeds process_image(path)."""
        from mlx_vlm.utils import prepare_inputs, process_image, should_add_special_tokens
        path = self.t._current_image
        special = should_add_special_tokens(self.config['model_type'], self.proc)
        via_path = prepare_inputs(self.proc, images=[path], prompts=[self.text],
                                  image_token_index=self.t.image_token_id,
                                  add_special_tokens=special)
        pil = process_image(path, None, self.proc.image_processor)
        via_pil = prepare_inputs(self.proc, images=[pil], prompts=[self.text],
                                 image_token_index=self.t.image_token_id,
                                 add_special_tokens=special, pad_to_uniform_size=False)
        self.assertEqual(np.array(via_path['input_ids']).tolist(),
                         np.array(via_pil['input_ids']).tolist())
        self.assertTrue(np.array_equal(np.array(via_path['pixel_values']),
                                       np.array(via_pil['pixel_values'])),
                        'evaluation and rollout would see different pixels')
        self.assertEqual(np.array(via_path['input_ids']).reshape(-1).tolist(), self.ids,
                         '_templated_ids disagrees with the generation path')

    def test_image_expands_to_one_token_per_merged_patch(self):
        thw = np.array(self.t._image_inputs[self.t._current_image][1]).reshape(-1)
        merge = self.config['vision_config']['spatial_merge_size']
        expected = int(thw[0] * thw[1] * thw[2] // (merge * merge))
        self.assertEqual(self.ids.count(self.t.image_token_id), expected)
        self.assertGreater(expected, 1000)          # a real image, not one placeholder

    def test_prefix_invariant_with_image_in_history(self):
        """Each turn's re-templated ids extend the previous sequence exactly."""
        payload = [dict(m) for m in self.first]
        seq = list(self.ids)
        for turn, answer in enumerate([
                json.dumps({'cards': self.info['Plain Cards'], 'number': self.info['Numbers'],
                            'formula': '1+2=3'}),
                json.dumps({'cards': ['A', 'A', 'A', 'A'], 'number': [1, 1, 1, 1],
                            'formula': '(1+1)*1=2'})]):
            seq = seq + self.t._completion_ids(answer)
            payload = payload + [
                {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
                {'role': 'user', 'content': [{'type': 'text',
                                              'text': 'You failed this trial because x.'}]}]
            nxt, _ = self.t._templated_ids(payload)
            self.assertEqual(nxt[:len(seq)], seq, f'prefix broken at turn {turn + 2}')
            seq = nxt
        self.assertEqual(seq.count(self.t.image_token_id), self.ids.count(self.t.image_token_id),
                         'the image must appear once, in the first turn only')


class VLScorerSliceGuardTest(unittest.TestCase):

    def test_unprepared_slice_raises(self):
        import mlx.core as mx
        from rl.trainer.grpo_vl_mlx import MultiTurnVLGRPOTrainer
        from rl.trainer.grpo_multiturn_mlx import MultiTurnConfig
        t = MultiTurnVLGRPOTrainer.__new__(MultiTurnVLGRPOTrainer)
        t.cfg = MultiTurnConfig(microbatch=2)
        t._rows_images, t._image_inputs = [], {}
        for_slice = t._prepare_forward(mx.zeros((0, 5), dtype=mx.int32))
        with self.assertRaises(RuntimeError):
            for_slice(slice(0, 3))


if __name__ == '__main__':
    unittest.main()
