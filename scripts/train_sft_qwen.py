"""LoRA SFT for Qwen3-VL on GeneralPoints, using mlx_vlm's shipped SFT trainer.

Step 1 of the pipeline. This is the SFT arm of the paper's SFT-vs-RL comparison,
and it also teaches the space-free formula format that the shipped verifier
requires -- so a successful run should lift the *strict* evaluation off 0%.

Keep it short. The paper's Figure 19 point is that RL cannot rescue an
over-SFT'd checkpoint, so more iterations is not better.
"""
import argparse
import json
import os
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx_vlm import load  # noqa: E402
from mlx_vlm.trainer import VisionDataset, TrainingArgs, train  # noqa: E402
from mlx_vlm.trainer.utils import (  # noqa: E402
    find_all_linear_names, get_peft_model, print_trainable_parameters,
    not_supported_for_training,
)


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:            # make_sft_data stores image paths relative to the repo
        if 'images' in row:
            row['images'] = [str(ROOT / p) if not Path(p).is_absolute() else p
                             for p in row['images']]
    return rows


def check_lengths(dataset, max_seq_length, name):
    """Every example must fit, or the trainer silently drops it.

    mlx_vlm's iterate_batches skips (with only a log warning) any example whose
    image tokens would be cut by max_seq_length, since truncating them would
    break the feature/token alignment. A 1200 px card image is 1444 tokens and a
    GP-VL example ~1735, so the old 1024 default would have skipped every one.
    Text-only examples would instead be truncated, losing the answer.
    """
    lengths = [int(dataset[i]['input_ids'].size) for i in range(len(dataset))]
    if max(lengths) > max_seq_length:
        raise SystemExit(f'{name}: longest example is {max(lengths)} tokens but '
                         f'max_seq_length is {max_seq_length}; examples would be dropped '
                         f'or truncated. Raise --max-seq-length.')
    print(f'{name}: {len(lengths)} examples, {min(lengths)}..{max(lengths)} tokens '
          f'(all within max_seq_length {max_seq_length})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=ROOT / 'data/sft_gp_language')
    ap.add_argument('--model', type=Path, default=ROOT / 'models/Qwen3-VL-8B-Instruct')
    ap.add_argument('--out', type=Path, default=ROOT / 'train_ckpt/sft_gp_language')
    ap.add_argument('--iters', type=int, default=250)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--lora-rank', type=int, default=8)
    ap.add_argument('--lora-alpha', type=float, default=16.0)
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--max-seq-length', type=int, default=2048,
                    help='every example must fit (checked); a GP-VL example is ~1735 tokens')
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--steps-per-report', type=int, default=10)
    ap.add_argument('--steps-per-eval', type=int, default=50)
    ap.add_argument('--steps-per-save', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    # mlx_vlm's iterate_batches shuffles with np.random, which mx.random.seed
    # does not touch. Unseeded, two runs with the same --seed trained on
    # different example orders (BUGS.md #30). Seed all three.
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model, processor = load(str(args.model))
    if model.config.model_type in not_supported_for_training:
        raise SystemExit(f'{model.config.model_type} cannot be trained by mlx_vlm')
    config = model.config.__dict__

    # Adapt the language stack only; the vision tower stays frozen, for GP-VL
    # too. Qwen3-VL already reads the rendered cards perfectly (100% card
    # recognition in the GP-VL Init evaluation), so what SFT can change is the
    # reasoning and answer format -- and GP-L and GP-VL then differ only in input.
    modules = find_all_linear_names(model.language_model)
    model = get_peft_model(model, modules, rank=args.lora_rank, alpha=args.lora_alpha,
                           dropout=args.lora_dropout, verbose=False)
    print_trainable_parameters(model)

    make = lambda rows: VisionDataset(rows, config, processor, train_on_completions=True)
    train_ds = make(read_jsonl(args.data / 'train.jsonl'))
    val_ds = make(read_jsonl(args.data / 'valid.jsonl'))
    print(f'train {len(train_ds)} | valid {len(val_ds)}')
    check_lengths(train_ds, args.max_seq_length, 'train')
    check_lengths(val_ds, args.max_seq_length, 'valid')

    args.out.mkdir(parents=True, exist_ok=True)
    adapter = args.out / 'adapters.safetensors'
    (args.out / 'run.json').write_text(json.dumps({
        'model': str(args.model), 'data': str(args.data),
        'data_meta': json.loads((args.data / 'meta.json').read_text()),
        'iters': args.iters, 'batch_size': args.batch_size, 'lr': args.lr,
        'lora': {'rank': args.lora_rank, 'alpha': args.lora_alpha, 'dropout': args.lora_dropout},
        'grad_accum': args.grad_accum, 'grad_clip': args.grad_clip,
        'max_seq_length': args.max_seq_length, 'seed': args.seed,
    }, indent=2) + '\n')

    train(model=model,
          optimizer=optim.Adam(learning_rate=args.lr),
          train_dataset=train_ds,
          val_dataset=val_ds,
          args=TrainingArgs(
              batch_size=args.batch_size, iters=args.iters,
              steps_per_report=args.steps_per_report,
              steps_per_eval=args.steps_per_eval,
              steps_per_save=args.steps_per_save,
              val_batches=5, max_seq_length=args.max_seq_length,
              adapter_file=str(adapter), grad_checkpoint=True,
              learning_rate=args.lr, grad_clip=args.grad_clip,
              gradient_accumulation_steps=args.grad_accum, full_finetune=False),
          train_on_completions=True)
    print(f'\nadapter written to {adapter}')


if __name__ == '__main__':
    main()
