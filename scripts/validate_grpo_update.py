"""Does one GRPO step move probabilities in the intended direction?

Every test so far checked the objective's arithmetic on synthetic arrays, or that
gradients flow at all. Neither would catch a sign error, a misaligned completion
mask, or wrong loss scaling on the real model -- faults whose only symptom is
"RL runs but does not learn".

This builds a FIXED group for one hand containing both correct and incorrect
completions, records each completion's log-probability, takes a single
optimisation step through the real `update()`, and checks that:

  * log p(correct)   goes UP
  * log p(incorrect) goes DOWN
  * the completion mask covers exactly the generated tokens and nothing else
  * advantages carry the sign the rewards imply
  * the loss denominator equals the batch's completion-token count

    .venv-qwen/bin/python scripts/validate_grpo_update.py [adapter_dir]
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer.utils import apply_lora_layers, find_all_linear_names, get_peft_model

from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG
from rl.trainer.grpo_trainer_mlx import (
    GRPOConfig, GRPOTrainer, group_advantages, pad_batch, sequence_logprobs)

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else 'train_ckpt/sft_gp_language'
CARDS, NUMS, GOLD = ['10', '5', '2', '4'], [10, 5, 2, 4], '10+5*2+4'
WRONG = ['10+5+2+4', '10*5-2-4', '10-5+2*4', '10/5+2+4']   # all valid JSON, all != 24


def answer(formula):
    return json.dumps({'cards': CARDS, 'number': NUMS, 'formula': f'{formula}=24'})


def main():
    model, processor = load(str(ROOT / 'models/Qwen3-VL-8B-Instruct'))
    if Path(ADAPTER, 'adapter_config.json').exists():
        model = apply_lora_layers(model, str(ADAPTER))
        cfg_lora = json.loads(Path(ADAPTER, 'adapter_config.json').read_text())
        if getattr(model.config, 'lora', None) is None:
            model.config.lora = cfg_lora
        print(f'adapter: {ADAPTER}')
    else:
        model = get_peft_model(model, find_all_linear_names(model.language_model),
                              rank=8, alpha=16.0, dropout=0.0, verbose=False)
        print('adapter: none (fresh LoRA)')
    tok = processor.tokenizer

    env = GeneralPointEnv_oneline(language_only=True, resolution=64,
                                  strip_formula_whitespace=True)
    env.reset()
    env.cards_num, env.cards_without_suit, env.solution = list(NUMS), list(CARDS), [GOLD]

    def reward_fn(text, info):
        env.remaining_step, env.verify_info = env.verify_iter, None
        return float(env.step(text)[1])

    question = Q_GeneralPoint_EQN_L_EG.format(
        cards=CARDS, target_number='24', face_card_msg="'J', 'Q', and 'K' count as '10'.")
    prompt = apply_chat_template(
        processor, model.config,
        [{"role": "user", "content": [{"type": "text", "text": question}]}], num_images=0)
    prompt_ids = tok.encode(prompt)

    texts = [answer(GOLD)] * 2 + [answer(w) for w in WRONG]          # 2 correct, 4 wrong
    completions = [tok.encode(t) + [tok.eos_token_id] for t in texts]
    rewards = np.array([[reward_fn(t, None) for t in texts]], np.float32)
    correct = rewards[0] == 5.0

    print(f'\ngroup of {len(texts)}: rewards {rewards[0].tolist()}')
    print(f'  correct completions: {int(correct.sum())}, incorrect: {int((~correct).sum())}')
    if correct.sum() == 0 or (~correct).sum() == 0:
        raise SystemExit('need both correct and incorrect completions; check GOLD/WRONG')

    # ---- structural checks before any step -----------------------------------
    input_ids, comp_mask = pad_batch([prompt_ids] * len(texts), completions, tok.pad_token_id or 0)
    mask_sums = np.array(comp_mask).sum(axis=1)
    expected = np.array([len(c) for c in completions], np.float32)
    print(f'\ncompletion mask covers exactly the generated tokens: '
          f'{np.allclose(mask_sums, expected)}  (mask {mask_sums.tolist()} vs len {expected.tolist()})')
    first_tok = [int(np.array(input_ids)[i, len(prompt_ids)]) for i in range(len(texts))]
    print(f'token at the prompt boundary == each completion\'s first token: '
          f'{first_tok == [c[0] for c in completions]}')

    adv = group_advantages(rewards)[0]
    print(f'advantages: {np.round(adv, 3).tolist()}')
    print(f'  correct rows positive: {bool((adv[correct] > 0).all())}, '
          f'incorrect rows negative: {bool((adv[~correct] < 0).all())}')

    def logp_per_row():
        lp, m = sequence_logprobs(model, input_ids, comp_mask, 1.0)
        mx.eval(lp)
        return np.array((lp * m).sum(axis=1))

    before = logp_per_row()

    # ---- one real update -----------------------------------------------------
    trainer = GRPOTrainer(
        model=model, processor=processor, env=env,
        prompt_template=Q_GeneralPoint_EQN_L_EG,
        oracle_arguments={'target_number': '24',
                          'face_card_msg': "'J', 'Q', and 'K' count as '10'."},
        reward_fn=reward_fn,
        config=GRPOConfig(group_size=len(texts), prompts_per_update=1, microbatch=3,
                          learning_rate=1e-5, kl_coef=0.01, temperature=1.0))
    metrics = trainer.update([prompt_ids] * len(texts), completions, rewards)
    after = logp_per_row()

    # The mask is shifted by one (comp_mask[:, 1:]) but the dropped column is a
    # prompt position, so it contributes no completion tokens: denom equals the
    # full completion-token count, and every generated token is scored.
    print(f'\nloss {metrics["loss"]:+.6f}   denom(completion tokens) '
          f'{metrics["completion_tokens"]:.0f}   expected {expected.sum():.0f}  '
          f'match: {abs(metrics["completion_tokens"] - expected.sum()) < 0.5}')
    print(f'\n{"row":>4} {"reward":>7} {"adv":>7} {"logp before":>12} {"logp after":>11} {"delta":>9}')
    for i in range(len(texts)):
        d = after[i] - before[i]
        print(f'{i:>4} {rewards[0][i]:>+7.1f} {adv[i]:>+7.2f} {before[i]:>12.3f} {after[i]:>11.3f} {d:>+9.4f}')

    d_correct = float((after - before)[correct].mean())
    d_wrong = float((after - before)[~correct].mean())
    print(f'\nmean delta log p  correct {d_correct:+.4f}   incorrect {d_wrong:+.4f}')
    ok = d_correct > 0 and d_wrong < 0
    print(f'\n{"PASS" if ok else "FAIL"}: one step '
          f'{"raised" if d_correct > 0 else "LOWERED"} the correct completions and '
          f'{"lowered" if d_wrong < 0 else "RAISED"} the incorrect ones')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
