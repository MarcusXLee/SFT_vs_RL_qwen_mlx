"""Launch single-turn GRPO on GeneralPoints with the local Qwen3-VL checkpoint.

Step 3 of the pipeline: the correctness smoke test for the RL loop before the
critic is added (see rl/trainer/grpo_trainer_mlx.py for why GRPO first).

    # from the SFT checkpoint (the paper's ordering: SFT init, then RL)
    .venv-qwen/bin/python scripts/train_grpo_qwen.py \
        --adapter train_ckpt/sft_gp_language --updates 50

    # from the base model, to test whether SFT init is needed at all here
    .venv-qwen/bin/python scripts/train_grpo_qwen.py --updates 50

What to watch: `reward` and `success` should climb, `kl/tok` should stay small,
and `groups` should not collapse to 0 (every group having identical reward means
no learning signal -- usually temperature too low, or the task too hard/easy).
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))

from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline  # noqa: E402
from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG, Q_GeneralPoint_EQN_VL_EG  # noqa: E402


def make_reward_fn(env):
    """Score a generated answer by asking the environment itself.

    An earlier version re-implemented the parse -> step_rewards path here. It
    drifted immediately: it never applied `strip_formula_whitespace`, so
    --strip-formula-whitespace was silently ignored during training while the
    evaluation honoured it, and the RL signal disagreed with the metric it was
    supposed to optimise.

    Calling env.step keeps exactly one implementation. The environment is still
    holding the hand the completions were sampled for, so only the per-attempt
    counters need resetting: each completion is scored as an independent first
    attempt (single-turn GRPO -- no verifier-feedback retries).
    """
    def reward_fn(text, info):
        env.remaining_step = env.verify_iter
        env.verify_info = None
        _, reward, _, _, _ = env.step(text)
        return float(reward)
    return reward_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, default=ROOT / 'models/Qwen3-VL-8B-Instruct')
    ap.add_argument('--adapter', type=Path, default=None,
                    help='SFT checkpoint to start from; omit to train from base')
    ap.add_argument('--out', type=Path, default=ROOT / 'train_ckpt/grpo_gp_language')
    ap.add_argument('--updates', type=int, default=50)
    ap.add_argument('--group-size', type=int, default=8)
    ap.add_argument('--prompts-per-update', type=int, default=4)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-6)
    ap.add_argument('--kl-coef', type=float, default=0.01)
    ap.add_argument('--clip-eps', type=float, default=0.2)
    ap.add_argument('--microbatch', type=int, default=4)
    ap.add_argument('--batch-rollout', action='store_true',
                    help='sample each group in one batched pass (much faster; '
                         'completions are re-tokenised, see _sample_group_batched)')
    ap.add_argument('--lora-rank', type=int, default=8)
    ap.add_argument('--lora-alpha', type=float, default=16.0)
    ap.add_argument('--target', type=int, default=24)
    ap.add_argument('--strip-formula-whitespace', action='store_true',
                    help='tolerant verifier (see QWEN_EVAL.md); default is the shipped one')
    ap.add_argument('--multi-turn', action='store_true',
                    help='roll out full verifier-feedback trajectories instead of one attempt')
    ap.add_argument('--reward-mode', choices=['final', 'sum'], default='final',
                    help="multi-turn trajectory reward: 'final' scores whether it ever "
                         "succeeded; 'sum' adds the per-attempt rewards")
    ap.add_argument('--vision', action='store_true',
                    help='GP-VL: the model sees the rendered cards (1200 px, black face '
                         'cards, as the GP-VL evaluation and upstream VL training). '
                         'Multi-turn only; see rl/trainer/grpo_vl_mlx.py')
    ap.add_argument('--max-turns', type=int, default=5,
                    help='attempts per trajectory; should match the env verify_iter')
    ap.add_argument('--entropy-coef', type=float, default=0.0,
                    help='bonus on policy entropy; 0.0 is an exact no-op')
    ap.add_argument('--normalize-advantage-std', action='store_true',
                    help='original GRPO advantage scaling (divide by group std); off by '
                         'default because it equalises real successes with varied failures')
    ap.add_argument('--dynamic-sampling', action='store_true',
                    help='discard groups with no reward spread and redraw (DAPO 3.2); '
                         'guarantees every update has signal, at the cost of extra rollouts')
    ap.add_argument('--dynamic-max-draws', type=int, default=0,
                    help='cap on hands drawn per update (0 -> prompts_per_update * 8)')
    ap.add_argument('--save-every', type=int, default=10,
                    help='checkpoint the adapter every N updates (0 disables)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-seed', type=int, default=42,
                    help='evaluation stream whose hands are withheld from training')
    ap.add_argument('--eval-traj', type=int, default=200,
                    help='how much of the eval stream to withhold (>= any eval num_traj)')
    ap.add_argument('--no-holdout', action='store_true',
                    help='train on all hands, including evaluation ones (not recommended: '
                         'the SFT data generator holds them out, so results stop being comparable)')
    args = ap.parse_args()

    if args.vision and not args.multi_turn:
        ap.error('--vision is implemented for multi-turn GRPO only (--multi-turn)')
    if args.temperature <= 0:
        ap.error('--temperature must be > 0, or every completion in a group is '
                 'identical and GRPO has no signal to learn from')

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.trainer.utils import (apply_lora_layers, find_all_linear_names,
                                       get_peft_model, print_trainable_parameters)
    from rl.trainer.grpo_trainer_mlx import GRPOConfig, GRPOTrainer
    from rl.trainer.grpo_multiturn_mlx import MultiTurnConfig, MultiTurnGRPOTrainer

    if not mx.metal.is_available():
        raise SystemExit('MLX needs a Metal device; run this from macOS on Apple silicon.')

    model, processor = load(str(args.model))
    if args.adapter:
        model = apply_lora_layers(model, str(args.adapter))
        # apply_lora_layers rebuilds the LoRA topology from the source config but
        # does not record it on the model, unlike get_peft_model. Without this the
        # trainer cannot write adapter_config.json for its own checkpoints, and
        # they would not be loadable (BUGS.md #14).
        source_config = Path(args.adapter) / 'adapter_config.json'
        if getattr(model.config, 'lora', None) is None and source_config.exists():
            model.config.lora = json.loads(source_config.read_text())
        print(f'resuming from adapter: {args.adapter}')
    else:
        model = get_peft_model(model, find_all_linear_names(model.language_model),
                               rank=args.lora_rank, alpha=args.lora_alpha,
                               dropout=0.0, verbose=False)
        print('training fresh LoRA adapters from the base model')
    print_trainable_parameters(model)

    # GP-VL renders exactly as evaluation/configs/qwen_gp_vl.yaml and upstream
    # rl/configs/llama_gp_vl.yaml: 1200 px, black face cards. The hand stream
    # does not depend on rendering or colour (checked 50/50 against the GP-VL
    # evaluation log), so the holdout below is the same set of hands.
    env_kwargs = (dict(language_only=False, resolution=1200, face_cards_color='black')
                  if args.vision else dict(language_only=True, resolution=64))
    env = GeneralPointEnv_oneline(
        target_points=args.target,
        treat_face_cards_as_10=True, verify_iter=args.max_turns,
        strip_formula_whitespace=args.strip_formula_whitespace, **env_kwargs)

    holdout = set()
    if not args.no_holdout:
        from scripts.make_sft_data import eval_hands
        holdout = eval_hands(args.eval_seed, args.eval_traj,
                             target_points=args.target, treat_face_cards_as_10=True)
        print(f'withholding {len(holdout)} evaluation hands from training')

    oracle = {'target_number': str(args.target),
              'face_card_msg': "'J', 'Q', and 'K' count as '10'."}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'run.json').write_text(json.dumps(vars(args), indent=2, default=str) + '\n')

    trainer_cls = MultiTurnGRPOTrainer if args.multi_turn else GRPOTrainer
    config_cls = MultiTurnConfig if args.multi_turn else GRPOConfig
    extra = dict(max_turns=args.max_turns, reward_mode=args.reward_mode) if args.multi_turn else {}
    trainer_extra = {}
    if args.vision:
        from rl.trainer.grpo_vl_mlx import MultiTurnVLGRPOTrainer
        trainer_cls = MultiTurnVLGRPOTrainer
        trainer_extra = dict(image_dir=args.out / 'rollout_images')
    print(f'rollout: {"multi-turn, reward=" + args.reward_mode if args.multi_turn else "single-turn"}'
          f'{", GP-VL (rendered cards)" if args.vision else ""}')

    trainer = trainer_cls(
        model=model, processor=processor, env=env,
        prompt_template=Q_GeneralPoint_EQN_VL_EG if args.vision else Q_GeneralPoint_EQN_L_EG,
        oracle_arguments=oracle, **trainer_extra,
        reward_fn=make_reward_fn(env),
        holdout=holdout,
        config=config_cls(
            group_size=args.group_size, prompts_per_update=args.prompts_per_update,
            temperature=args.temperature, max_tokens=args.max_tokens,
            clip_eps=args.clip_eps, kl_coef=args.kl_coef,
            learning_rate=args.lr, microbatch=args.microbatch,
            batch_rollout=args.batch_rollout, save_every=args.save_every,
            dynamic_sampling=args.dynamic_sampling,
            normalize_advantage_std=args.normalize_advantage_std,
            entropy_coef=args.entropy_coef, log_entropy=True,
            dynamic_max_draws=args.dynamic_max_draws,
            seed=args.seed, **extra),
        log_path=args.out / 'metrics.jsonl')

    trainer.train(args.updates)

    # Use the trainer's save_adapter so adapter_config.json is written too;
    # saving the weights alone produces a checkpoint apply_lora_layers cannot
    # load (BUGS.md #14, which this path was still reproducing).
    adapters = trainer.save_adapter(args.out / 'adapters.safetensors')
    print(f'\nadapter written to {adapters}')


if __name__ == '__main__':
    main()
