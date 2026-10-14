"""
Here we define the class for Qwen-VL evaluators backed by MLX (Apple silicon).

This mirrors LlamaEvaluator: it reuses BaseEvaluator's trajectory loop, so the
evaluation protocol (verify_iter retries, verifier feedback appended to the same
conversation, per-trajectory success) is identical to the published one. Only the
model backend differs -- MLX instead of transformers + deepspeed.

By Xinye, following the structure of llama_evaluator.py
"""

import os

import numpy as np
from PIL import Image

from evaluation.evaluator.base_evaluator import BaseEvaluator

# MLX generation keyword names differ from the HF ones used in the shipped configs.
_GENERATION_KEY_ALIASES = {"max_new_tokens": "max_tokens"}


def resolve_adapter_path(adapter_path):
    """Normalise a configured adapter location to the directory MLX expects.

    `mlx_vlm.trainer.utils.apply_lora_layers` wants the directory holding
    adapter_config.json + adapters.safetensors. Accept the weights file too,
    since that is what TrainingArgs.adapter_file names. Empty string / None
    means "no adapter", so the YAML key can stay present and unset.
    """
    if not adapter_path:
        return None
    if not os.path.exists(adapter_path):
        raise ValueError(f"adapter_path does not exist: {adapter_path}")
    directory = os.path.dirname(adapter_path) if os.path.isfile(adapter_path) else adapter_path
    config = os.path.join(directory, "adapter_config.json")
    if not os.path.exists(config):
        raise ValueError(
            f"no adapter_config.json in {directory}; "
            "point adapter_path at the directory produced by training")
    return directory


class QwenEvaluator(BaseEvaluator):
    def __init__(self, action_space, daytime, env_config, model, model_path, prompt_config,
                 generation_config, output_dir, seed=42, adapter_path=None, **kwargs):
        super(QwenEvaluator, self).__init__(action_space, daytime, env_config, prompt_config,
                                            generation_config, output_dir, seed, **kwargs)
        import mlx.core as mx
        from mlx_vlm import load

        if not mx.metal.is_available():
            raise RuntimeError("MLX needs a Metal device; run this from macOS on Apple silicon.")
        mx.random.seed(seed)
        self.mx = mx

        if not os.path.isdir(model_path):
            raise ValueError(f"model_path is not a local directory: {model_path}")
        self.model, self.processor = load(model_path)

        # An SFT/RL checkpoint is a LoRA adapter over the same base weights.
        # Empty string counts as "no adapter" so the YAML key can stay present.
        self.adapter_path = resolve_adapter_path(adapter_path)
        if self.adapter_path:
            from mlx_vlm.trainer.utils import apply_lora_layers
            self.model = apply_lora_layers(self.model, self.adapter_path)
            self.model.eval()
            print(f"[QwenEvaluator] loaded adapter: {self.adapter_path}")

        self.model_config = self.model.config

        # Translate HF-style generation keys so the shipped configs work unchanged.
        self.generation_config = {_GENERATION_KEY_ALIASES.get(k, k): v
                                  for k, v in dict(generation_config).items()}

        # Observations are written to disk once per trajectory; MLX loads images by path.
        self.image_dir = os.path.join(os.path.dirname(self.output_dir), "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.current_image_path = None

    def _save_observation(self, obs):
        """Persist the current card image and return its path (one per trajectory).

        Always overwrites. This is called once per trajectory, and the file name
        keys only on sample_num -- so skipping an existing file would silently
        feed the model a previous run's cards while scoring against this run's
        hand, whenever two runs share an output directory.
        """
        if isinstance(obs, np.ndarray):
            obs = Image.fromarray(obs)
        path = os.path.join(self.image_dir, f"sample_{self.sample_num:05d}.png")
        obs.save(path)
        return path

    def formulate_payload(self, question, obs=None):
        """Start a fresh conversation; the image marker rides on the first user turn."""
        self.payload = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        ]
        if obs is not None:
            self.payload[0]["content"].insert(0, {"type": "image"})

    def formulate_prompt(self, vision_res_dict, language_res_dict, prompt=None, obs=None, info=None):
        """Either open a new trajectory or append the verifier's feedback turn."""
        if 'gym_cards' in self.id:
            if info['Verify Info'] is not None:
                self.payload.append({"role": "user",
                                     "content": [{"type": "text",
                                                  "text": f"You failed this trial because {info['Verify Info']}"}]})
            else:
                question = prompt.format(**vision_res_dict, **language_res_dict, **self.oracle_arguments)
                self.formulate_payload(question, obs)
        elif 'gym_virl' in self.id:
            if info['Verify Info'] is None or 'Correct action' in info['Verify Info'] \
                    or 'step_limit_reached"' in info['Verify Info']:
                question = prompt.format(**vision_res_dict, **language_res_dict, **self.oracle_arguments)
                self.formulate_payload(question, obs)
            else:
                self.payload.append({"role": "user",
                                     "content": [{"type": "text",
                                                  "text": f"You failed this trial because {info['Verify Info']}"}]})

    def generate_one_response(self, vision_res_dict, language_res_dict, obs=None, info=None):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self.formulate_vision_arguments(vision_res_dict, info)
        if obs is not None:
            prompts, patterns = self.prompt_vision, self.pattern_vision
            if info['Verify Info'] is None:
                # New trajectory: the cards changed, so re-render the observation.
                self.current_image_path = self._save_observation(obs)
            images = [self.current_image_path]
        else:
            prompts, patterns = self.prompt_language, self.pattern_language
            images = None

        for prompt, pattern in zip(prompts, patterns):
            self.formulate_prompt(vision_res_dict, language_res_dict, prompt, obs, info)
            input_text = apply_chat_template(self.processor, self.model_config, self.payload,
                                             num_images=0 if images is None else len(images))
            result = generate(self.model, self.processor, input_text, image=images,
                              verbose=False, **self.generation_config)
            outputs = result.text
            language_res_dict[pattern] = outputs
            self.append_intermidiate_res(outputs)
        self.mx.clear_cache()

    def append_intermidiate_res(self, res):
        self.payload.append({"role": "assistant", "content": [{"type": "text", "text": res}]})

    def extract_final_action(self, language_res_dict, key_pattern=None):
        try:
            return language_res_dict[key_pattern]
        except KeyError:
            if 'gym_cards' in self.id:
                return language_res_dict['formula']
            return language_res_dict['action']
