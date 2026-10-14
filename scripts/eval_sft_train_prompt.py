"""Evaluate a language checkpoint with the prompt formatting its SFT data used.

The language SFT data stored the user prompt as a plain string. The evaluator
(and the GRPO rollout) send it as a content list, [{"type": "text", ...}], and
the chat template strips the text's leading and trailing newline on that path.
So SFT was trained on

    <|im_start|>user\\n\\n[Task Description] ... output the json.\\n<|im_end|>

and evaluated on

    <|im_start|>user\\n[Task Description] ... output the json.<|im_end|>

This runs the ordinary evaluation with ONE change: the opening user turn is a
plain string, exactly as in the SFT data. Verifier-feedback turns are unchanged;
the SFT data had none. Paired against the standard run on the same hands, the
difference measures how much the whitespace mismatch cost the SFT model.

    usage: same arguments as evaluation.launcher
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]

from evaluation.evaluator.qwen_evaluator import QwenEvaluator  # noqa: E402

_standard = QwenEvaluator.formulate_payload
_shown = []


def sft_data_format(self, question, obs=None):
    if obs is not None:                      # vision path: not what this measures
        return _standard(self, question, obs)
    self.payload = [{"role": "user", "content": question}]
    if not _shown:
        from mlx_vlm.prompt_utils import apply_chat_template
        text = apply_chat_template(self.processor, self.model_config, self.payload, num_images=0)
        print(f'[train-format prompt] head {text[:40]!r} ... tail {text[-50:]!r}', flush=True)
        _shown.append(True)


QwenEvaluator.formulate_payload = sft_data_format

if __name__ == '__main__':
    from evaluation.launcher import main
    from utils_general import load_config
    config, boxed = load_config()
    main(config, boxed)
