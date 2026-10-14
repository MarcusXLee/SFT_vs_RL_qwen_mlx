"""Multi-turn GRPO for GP-VL: the model sees the rendered cards, not their names.

Everything that defines the RL algorithm is inherited unchanged from
MultiTurnGRPOTrainer / GRPOTrainer: rewards, advantages, the clipped objective
with k3 KL, entropy, trajectory normalisation, dynamic sampling, the
equal-length batching and token-count guard of _batch_generate. Upstream also
trains GP-VL with exactly the GP-L recipe (rl/configs/llama_gp_vl.yaml differs
from the language config only in the environment rendering and the prompt).

What differs, and why each piece is shaped as it is:

1. The conversation is the GP-VL evaluator's (QwenEvaluator): the image rides
   on the first user turn only, as a PNG path, and verifier feedback follows as
   text. Rendering is the GP-VL evaluation's: 1200 px, black face cards.

2. Token ids come from the processor, not tokenizer.encode. The processor
   expands the single <|image_pad|> into one token per merged vision patch
   (1444 for a 1200x1200 render). The scorer, the prefix invariant and the
   batch_generate token-count guard all need the expanded ids.

3. The scorer needs the image. The text path calls model.language_model on
   token ids, which would feed the image placeholders with no image at all.
   Here the image features, Qwen3-VL's deepstack features and the 3-D mRoPE
   positions are computed once per update, outside the gradient, and the
   language model is run on them for the old, reference and current passes.
   That is exact only because LoRA touches the language layers alone: the
   vision tower and the token embedding are frozen, so these inputs are the
   same for the policy, the reference and every optimisation step. __init__
   refuses to run if anything outside the language layers is trainable.

   It also keeps the vision tower out of the backward pass. With the vision
   tower inside it, VL SFT needed 97 GB at batch 1 (BUGS.md #29).

Status: verified on the GPU before any training (scripts/verify_vl_grpo.py,
logs/diagnostics/vl_grpo_verify_mb1.log; reproduced on a second run):
  * guard refuses the double-templated call, accepts message lists;
  * positions: 3,682 forwards over 73 batch_generate calls, 0 mismatches with
    the scorer's 3-D mRoPE positions;
  * fidelity: 5,468 sampled tokens, mean log-prob -0.134, none below -10 on
    any of 5 turns; 0/113 re-tokenisation mismatches; 0 truncated at 128;
  * precomputed-input logits bit-identical to the full forward (max |d| 0);
  * step-0: ref == old exactly; cur-vs-old noise as GP-L (k3 1.41e-4/token);
  * one update at microbatch 1: peak 67.6 GB of ~107.5. Use microbatch 1.
  * entropy: ratio 0.0115 at coef 0.01 (GP-L 0.0033). entropy_coef stays at
    GP-L's 0.002 -- see QWEN_EVAL.md "GP-VL GRPO" for why.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from rl.trainer.grpo_multiturn_mlx import MultiTurnGRPOTrainer
from rl.trainer.grpo_trainer_mlx import pad_sequences


class MultiTurnVLGRPOTrainer(MultiTurnGRPOTrainer):

    def __init__(self, *args, image_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_only_language_layers_train()
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        config = self.model.config
        self.image_token_id = (getattr(config, 'image_token_id', None)
                               or getattr(config, 'image_token_index', None))
        if self.image_token_id is None:
            raise ValueError('model config names no image token id')
        self._hand_seq = 0
        self._image_inputs = {}          # png path -> (pixel_values, image_grid_thw)
        self._rows_images = None

    # ------------------------------------------------------------ guards

    def _check_only_language_layers_train(self):
        """Precomputing image inputs is exact only if nothing upstream of the
        language layers can change during training."""
        from mlx.utils import tree_flatten
        names = [k for k, _ in tree_flatten(self.model.trainable_parameters())]
        outside = [k for k in names if not k.startswith('language_model.model.layers.')]
        if not names or outside:
            raise ValueError(
                'GP-VL GRPO precomputes the image features and token embeddings '
                'once per update, which is only exact when they cannot change. '
                f'Trainable parameters outside the language layers: {outside[:5]}')

    # ------------------------------------------------------------ rollout

    def _save_observation(self, obs):
        """The hand's render as a PNG, exactly as QwenEvaluator feeds it."""
        path = self.image_dir / f'hand_{self._hand_seq:06d}.png'
        self._hand_seq += 1
        (Image.fromarray(obs) if isinstance(obs, np.ndarray) else obs).save(path)
        return str(path)

    def _first_payload(self, question):
        obs = getattr(self, '_last_obs', None)
        if obs is None:
            raise RuntimeError('no rendered observation for this hand; the env must '
                               'be built with language_only=False')
        self._current_image = self._save_observation(obs)
        return [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": question}]}]

    def _call_images(self, n):
        return [self._current_image] * n

    def _templated_ids(self, payload, add_generation_prompt=True):
        """Processor-expanded ids for the conversation, image included.

        Also caches the image's pixel_values / grid, which the scorer reuses
        rather than re-processing the PNG.
        """
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs, process_image, should_add_special_tokens
        text = apply_chat_template(self.processor, self.model.config, payload,
                                   add_generation_prompt=add_generation_prompt,
                                   num_images=1)
        # Exactly the calls batch_generate makes (generate/ar.py: process_image
        # per image, then prepare_inputs in _generate_batch), so these ids are
        # the ones the rollout conditions on by construction.
        image = process_image(self._current_image, None,
                              getattr(self.processor, 'image_processor', None))
        inputs = prepare_inputs(
            self.processor, images=[image], prompts=[text],
            image_token_index=self.image_token_id,
            add_special_tokens=should_add_special_tokens(self.model.config.model_type,
                                                         self.processor),
            pad_to_uniform_size=False)
        ids = np.array(inputs['input_ids']).reshape(-1).tolist()
        if self._current_image not in self._image_inputs:
            self._image_inputs[self._current_image] = (
                inputs['pixel_values'], inputs['image_grid_thw'])
        return ids, text

    def collect(self):
        self._image_inputs = {}          # one update's hands at a time
        return super().collect()

    # ------------------------------------------------------------ update

    def _build_batch(self, sequences, masks, rows=None):
        ids, mask = pad_sequences(sequences, masks, self.pad_id)
        images = self.batch_images if rows is None else [self.batch_images[i] for i in rows]
        if len(images) != ids.shape[0]:
            raise RuntimeError(f'{ids.shape[0]} rows but {len(images)} images')
        self._rows_images = images
        return ids, mask

    def _prepare_forward(self, input_ids):
        """Image inputs per microbatch chunk, once per update, gradient-free.

        Chunks are exactly the row slices update() and sequence_logprobs_chunked
        use (both step by cfg.microbatch), and a request for any other slice
        raises rather than silently mis-aligning images and rows.
        """
        mb, n = self.cfg.microbatch, input_ids.shape[0]
        pre = {}
        for start in range(0, n, mb):
            stop = min(start + mb, n)
            paths = self._rows_images[start:stop]
            pv = mx.concatenate([self._image_inputs[p][0] for p in paths], axis=0)
            thw = mx.concatenate([self._image_inputs[p][1] for p in paths], axis=0)
            ids = input_ids[start:stop, :-1]
            f = self.model.get_input_embeddings(ids, pv, image_grid_thw=thw)
            deep = f.deepstack_visual_embeds
            if deep is not None:
                deep = [mx.stop_gradient(d) for d in deep]
            entry = (ids, mx.stop_gradient(f.inputs_embeds), f.position_ids,
                     f.visual_pos_masks, deep)
            mx.eval([x for x in entry if x is not None and not isinstance(x, list)]
                    + (deep or []))
            pre[(start, stop)] = entry
            mx.clear_cache()

        def for_slice(sl):
            key = (sl.start, sl.stop)
            if key not in pre:
                raise RuntimeError(f'logits requested for rows {key}, but image inputs '
                                   f'were prepared for {sorted(pre)}')
            ids, embeds, pos, vis, deep = pre[key]

            def logits_fn(model):
                return model.language_model(
                    ids, inputs_embeds=embeds, position_ids=pos,
                    visual_pos_masks=vis, deepstack_visual_embeds=deep).logits
            return logits_fn
        return for_slice
