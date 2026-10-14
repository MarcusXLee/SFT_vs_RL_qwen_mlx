"""Regression tests for the Qwen evaluation path and the whitespace ablation.

Run:  .venv-qwen/bin/python -m unittest discover -s scripts -p 'test_qwen_*.py'

The point of these tests is comparison rigor. The whitespace flag must change
exactly one thing -- whether spaces inside a formula are fatal -- and must not
quietly rescue answers that are actually wrong.
"""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]

from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline  # noqa: E402
from utils_rl import REWARD_FN  # noqa: E402

CARDS = ['10', '5', '2', '4']
NUMS = [10, 5, 2, 4]
SOLUTION = '10+5*2+4'  # == 24


def fixed_env(strip):
    """An env pinned to one known hand, so only the formula under test varies."""
    env = GeneralPointEnv_oneline(language_only=True, resolution=64,
                                  strip_formula_whitespace=strip)
    env.reset()
    env.cards_num = list(NUMS)
    env.cards_without_suit = list(CARDS)
    env.solution = [SOLUTION]
    env.remaining_step = 5
    env.verify_info = None
    return env


def reward_for(formula, strip):
    env = fixed_env(strip)
    action = json.dumps({'cards': CARDS, 'number': NUMS, 'formula': f'{formula}=24'})
    _, reward, _, _, _ = env.step(action)
    return reward


class WhitespaceFlagTest(unittest.TestCase):
    def test_default_is_upstream_behaviour(self):
        """With the flag off, a space is still fatal -- the shipped semantics."""
        self.assertEqual(reward_for('10 + 5 * 2 + 4', strip=False),
                         REWARD_FN['ILLEGAL_FORMULA'])

    def test_unspaced_correct_answer_passes_either_way(self):
        """The flag must not alter the verdict on formulas that contain no spaces."""
        for strip in (False, True):
            with self.subTest(strip=strip):
                self.assertEqual(reward_for(SOLUTION, strip), REWARD_FN['CORRECT_SOLUTION'])

    def test_tolerant_accepts_spaced_correct_answer(self):
        self.assertEqual(reward_for('10 + 5 * 2 + 4', strip=True),
                         REWARD_FN['CORRECT_SOLUTION'])

    def test_tolerant_does_not_rescue_wrong_arithmetic(self):
        """(5-2)*(10-4) == 18. Removing spaces must not make it a success."""
        self.assertNotEqual(reward_for('(5 - 2) * (10 - 4)', strip=True),
                            REWARD_FN['CORRECT_SOLUTION'])

    def test_tolerant_does_not_merge_split_digits(self):
        """'1 0' is two numbers, not ten; joining them would invent a correct answer."""
        self.assertNotEqual(reward_for('1 0 + 5*2+4', strip=True),
                            REWARD_FN['CORRECT_SOLUTION'])

    def test_tolerant_does_not_invent_numbers_not_on_the_cards(self):
        self.assertNotEqual(reward_for('10 + 5 * 2 + 9', strip=True),
                            REWARD_FN['CORRECT_SOLUTION'])


class UpstreamEquivalenceTest(unittest.TestCase):
    """The strict arm must be behaviourally identical to the committed upstream env.

    Rather than trust inspection, this loads the env as it exists in git HEAD
    alongside the patched one and asserts they return the same (reward, message)
    for every formula. If someone widens the whitespace patch so it leaks into
    the default path, this fails.
    """

    @classmethod
    def setUpClass(cls):
        import shutil
        import subprocess
        import tempfile
        try:
            original = subprocess.run(
                ['git', 'show', 'HEAD:gym/gym_cards/envs/general_points_oneline.py'],
                capture_output=True, text=True, check=True, cwd=ROOT).stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise unittest.SkipTest(f'cannot read upstream env from git: {exc}')
        cls._tmp = Path(tempfile.mkdtemp())
        pkg = cls._tmp / 'gym_cards_upstream' / 'envs'
        pkg.mkdir(parents=True)
        (cls._tmp / 'gym_cards_upstream' / '__init__.py').write_text('')
        (pkg / '__init__.py').write_text('')
        (pkg / 'general_points_oneline.py').write_text(original)
        shutil.copytree(ROOT / 'gym/gym_cards/envs/img', pkg / 'img')
        sys.path.insert(0, str(cls._tmp))
        from gym_cards_upstream.envs.general_points_oneline import GeneralPointEnv_oneline
        cls.Upstream = GeneralPointEnv_oneline

    @classmethod
    def tearDownClass(cls):
        import shutil
        sys.path.remove(str(cls._tmp))
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _verdict(self, env_cls, formula, **kwargs):
        env = env_cls(language_only=True, resolution=64, **kwargs)
        env.reset()
        env.cards_num = list(NUMS)
        env.cards_without_suit = list(CARDS)
        env.solution = [SOLUTION]
        env.remaining_step = 5
        env.verify_info = None
        action = json.dumps({'cards': CARDS, 'number': NUMS, 'formula': formula})
        _, reward, _, _, info = env.step(action)
        return reward, info['Verify Info']

    def test_strict_arm_matches_upstream_exactly(self):
        formulas = ['10+5*2+4', '10 + 5 * 2 + 4', '(5-2)*(10-4)', '1 0 + 5*2+4',
                    '10+5*2+9', 'banana', '10/0+5*2+4', '(10+5)*2-4', '10+5*2+4=24']
        for formula in formulas:
            with self.subTest(formula=formula):
                self.assertEqual(
                    self._verdict(self.Upstream, formula),
                    self._verdict(GeneralPointEnv_oneline, formula,
                                  strip_formula_whitespace=False))


class ExactArithmeticTest(unittest.TestCase):
    """The verifier must agree with the solver that generated the answer.

    `eval` does float division, so eval("(2/5)*6*10") == 24.000000000000004.
    The env's solver uses Fraction, so it could emit gold solutions its own
    verifier rejected (55 of 16,983 across 300 hands before the fix).
    """

    def test_the_reported_case(self):
        from utils_rl import exact_eval
        self.assertNotEqual(eval("(2/5)*6*10"), 24)          # the bug
        self.assertEqual(exact_eval("(2/5)*6*10"), 24)       # the fix

    def test_division_solution_now_scores_correct(self):
        env = fixed_env(strip=False)
        env.cards_num = [2, 5, 6, 10]
        env.cards_without_suit = ['2', '5', '6', '10']
        env.solution = ['(2/5)*6*10']
        action = json.dumps({'cards': ['2', '5', '6', '10'], 'number': [2, 5, 6, 10],
                             'formula': '(2/5)*6*10=24'})
        _, reward, _, _, _ = env.step(action)
        self.assertEqual(reward, REWARD_FN['CORRECT_SOLUTION'])

    def test_exactness_does_not_accept_near_misses(self):
        """A formula that is merely close to 24 must still be wrong."""
        from utils_rl import exact_eval
        self.assertNotEqual(exact_eval("(2/5)*6*10-1"), 24)
        self.assertNotEqual(exact_eval("23"), 24)

    def test_division_by_zero_still_raises_for_the_caller_to_catch(self):
        from utils_rl import exact_eval
        with self.assertRaises(ZeroDivisionError):
            exact_eval("10/0+5*2+4")

    def test_integer_only_formulas_are_unaffected(self):
        from utils_rl import exact_eval
        for expr in ('10+5*2+4', '(5-2)*(10-4)', '4*3+2+10'):
            with self.subTest(expr=expr):
                self.assertEqual(exact_eval(expr), eval(expr))


class CardParsingTest(unittest.TestCase):
    """A well-formed JSON answer must not be scored as a card misread.

    re_match returns a real list when json.loads succeeds; robust_str_to_list
    used to call .replace on it, hit AttributeError, and return [] -- which in
    vision mode never equals the ground truth, so every step scored
    INCORRECT_VISION. It hid because the shipped prompt's example is itself
    invalid JSON, so models imitate it and take the regex fallback.
    """

    GT = ['K', '5', '2', '4']

    def test_list_input_from_valid_json(self):
        from utils_general import re_match, robust_str_to_list
        text = json.dumps({'cards': self.GT, 'number': [10, 5, 2, 4],
                           'formula': '10+5*2+4=24'})
        self.assertEqual(robust_str_to_list(re_match(text, 'cards')), self.GT)

    def test_numeric_entries_are_stringified(self):
        from utils_general import robust_str_to_list
        self.assertEqual(robust_str_to_list([10, 5, 2, 4]), ['10', '5', '2', '4'])

    def test_string_fallback_still_works(self):
        """The regex path (invalid JSON, as the shipped prompt encourages)."""
        from utils_general import re_match, robust_str_to_list
        text = '{\n "cards": [K, 5, 2, 4],\n "formula": "10+5*2+4=24",\n}'
        self.assertEqual(robust_str_to_list(re_match(text, 'cards')), self.GT)

    def test_wrong_length_is_still_rejected(self):
        from utils_general import robust_str_to_list
        self.assertEqual(robust_str_to_list(['K', '5', '2']), [])
        self.assertEqual(robust_str_to_list([]), [])

    def test_vision_mode_accepts_a_valid_json_solution_end_to_end(self):
        """The bug's actual consequence: reward -1.5 instead of +5."""
        env = GeneralPointEnv_oneline(language_only=False, resolution=64)
        env.reset()
        env.cards_num = [10, 5, 2, 4]
        env.cards_without_suit = list(self.GT)
        env.solution = ['10+5*2+4']
        env.remaining_step = 5
        env.verify_info = None
        action = json.dumps({'cards': self.GT, 'number': [10, 5, 2, 4],
                             'formula': '10+5*2+4=24'})
        _, reward, _, _, _ = env.step(action)
        self.assertEqual(reward, REWARD_FN['CORRECT_SOLUTION'])

    def test_vision_mode_still_catches_a_genuine_misread(self):
        env = GeneralPointEnv_oneline(language_only=False, resolution=64)
        env.reset()
        env.cards_num = [10, 5, 2, 4]
        env.cards_without_suit = list(self.GT)
        env.solution = ['10+5*2+4']
        env.remaining_step = 5
        env.verify_info = None
        action = json.dumps({'cards': ['Q', '5', '2', '4'], 'number': [10, 5, 2, 4],
                             'formula': '10+5*2+4=24'})
        _, reward, _, _, _ = env.step(action)
        self.assertEqual(reward, REWARD_FN['INCORRECT_VISION'])


class OutputPathTest(unittest.TestCase):
    """BaseEvaluator used str.replace('.', ...), which hits every dot.

    A dotted directory name (logs/run.sft/log.jsonl) was rewritten into the
    path, so results landed somewhere the caller never looked.
    """

    @staticmethod
    def _stamp(output_dir, daytime):
        import os
        stem, ext = os.path.splitext(output_dir)
        return f'{stem}_{daytime}{ext}'

    def test_dotted_directory_is_preserved(self):
        got = self._stamp('logs/ablation/run.sft/strict/log.jsonl', 'T')
        self.assertEqual(got, 'logs/ablation/run.sft/strict/log_T.jsonl')

    def test_matches_old_behaviour_for_the_shipped_configs(self):
        """Paths with a single dot must time-stamp exactly as before."""
        for path in ('logs/llama_gp_language/in-distribution.jsonl',
                     'logs/qwen_gp_vl/log.jsonl'):
            with self.subTest(path=path):
                self.assertEqual(self._stamp(path, 'T'), path.replace('.', '_T.'))

    def test_base_evaluator_uses_splitext(self):
        source = (ROOT / 'evaluation/evaluator/base_evaluator.py').read_text()
        self.assertIn('os.path.splitext(output_dir)', source)
        self.assertNotIn("output_dir.replace('.'", source)


class AdapterPathTest(unittest.TestCase):
    """Loading an SFT/RL checkpoint must fail loudly, never silently no-op."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__('shutil').rmtree, self.tmp, True)

    def _complete_adapter(self):
        (self.tmp / 'adapter_config.json').write_text('{"lora_parameters": {}}')
        weights = self.tmp / 'adapters.safetensors'
        weights.write_bytes(b'')
        return weights

    def test_empty_means_base_model(self):
        from evaluation.evaluator.qwen_evaluator import resolve_adapter_path
        for value in (None, ''):
            with self.subTest(value=value):
                self.assertIsNone(resolve_adapter_path(value))

    def test_directory_is_accepted(self):
        from evaluation.evaluator.qwen_evaluator import resolve_adapter_path
        self._complete_adapter()
        self.assertEqual(resolve_adapter_path(str(self.tmp)), str(self.tmp))

    def test_weights_file_resolves_to_its_directory(self):
        from evaluation.evaluator.qwen_evaluator import resolve_adapter_path
        weights = self._complete_adapter()
        self.assertEqual(resolve_adapter_path(str(weights)), str(self.tmp))

    def test_missing_path_raises(self):
        from evaluation.evaluator.qwen_evaluator import resolve_adapter_path
        with self.assertRaises(ValueError):
            resolve_adapter_path(str(self.tmp / 'nope'))

    def test_directory_without_config_raises(self):
        """A bare .safetensors with no config would otherwise load nothing."""
        from evaluation.evaluator.qwen_evaluator import resolve_adapter_path
        (self.tmp / 'adapters.safetensors').write_bytes(b'')
        with self.assertRaises(ValueError):
            resolve_adapter_path(str(self.tmp))


class EvaluatorWiringTest(unittest.TestCase):
    def test_qwen_evaluator_is_registered(self):
        from evaluation.evaluator import evaluator_init
        self.assertIn('QwenEvaluator', evaluator_init)

    def test_unknown_evaluator_names_the_alternatives(self):
        from evaluation.evaluator import evaluator_init
        with self.assertRaises(KeyError) as ctx:
            evaluator_init['NoSuchEvaluator']
        self.assertIn('QwenEvaluator', str(ctx.exception))

    def test_hf_generation_keys_are_translated_for_mlx(self):
        """The shipped configs say max_new_tokens; MLX wants max_tokens."""
        from evaluation.evaluator.qwen_evaluator import _GENERATION_KEY_ALIASES
        self.assertEqual(_GENERATION_KEY_ALIASES['max_new_tokens'], 'max_tokens')

    def test_shipped_qwen_configs_default_to_upstream_semantics(self):
        """Neither config may bake the tolerant flag in; the runner sets it."""
        import yaml
        for name in ('qwen_gp_language', 'qwen_gp_vl'):
            cfg = yaml.safe_load((ROOT / 'evaluation/configs' / f'{name}.yaml').read_text())
            with self.subTest(config=name):
                self.assertIs(cfg['env_config']['strip_formula_whitespace'], False)
                self.assertEqual(cfg['env_config']['verify_iter'], 5)
                self.assertTrue(cfg['prompt_config']['enable_verification'])


if __name__ == '__main__':
    unittest.main()
