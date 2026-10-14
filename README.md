# SFT vs RL on Apple silicon: a Qwen3-VL + MLX reproduction

An independent reproduction of **[*SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training*](https://arxiv.org/abs/2501.17161)**
(Chu, Zhai, Yang, Tong, Xie, Schuurmans, Le, Levine and Ma, 2025) on the
**GeneralPoints** card-arithmetic task. The paper's Llama-3.2-Vision / CUDA
setup is replaced with **Qwen3-VL-8B-Instruct**, trained with LoRA in
**[MLX](https://github.com/ml-explore/mlx)** on a single 128 GB Mac.

> **Credit.** This repository is built on the authors' official code,
> [LeslieTrue/SFTvsRL](https://github.com/LeslieTrue/SFTvsRL): the GeneralPoints
> environment, verifier, prompts and evaluation loop are theirs. Everything that
> runs on MLX, the SFT/RL pipelines for Qwen3-VL, and the results below are this
> reproduction's. It is not affiliated with the authors. If you use the task or
> the idea, please cite their paper ([BibTeX below](#citation)).

## Result

On 200 held-out hands per setting, both modalities show the paper's pattern:
multi-turn RL beats SFT, and only RL improves on the untrained model under
distribution shift.

Success within 5 attempts with verifier feedback (%):

| | in-distribution | rule-OOD (J/Q/K = 11/12/13) | visual-OOD (red face cards) |
| --- | ---: | ---: | ---: |
| **GP-L**, Init / SFT / **RL** | 33.0 / 18.0 / **46.0** | 20.0 / 11.0 / **32.0** | — |
| **GP-VL**, Init / SFT / **RL** | 36.5 / 27.5 / **46.5** | 19.5 / 19.0 / **28.0** | 39.0 / 24.0 / **45.0** |

All comparisons are paired: the models saw the identical hands, and the tests
are exact McNemar.
- **RL beats SFT** in all five settings (p ≤ 0.018).
- **RL beats Init** in four of five (p ≤ 0.035). GP-VL visual-OOD is the
  exception: +6 points, p = 0.21.
- **SFT is below Init** in both GP-L settings and in GP-VL in-distribution and
  visual-OOD.
- **Where the gap comes from: revision.** SFT gains first-attempt accuracy but
  largely stops recovering after verifier feedback. RL restores that recovery.

**Where this differs from the paper.**
- **SFT here does not improve in-distribution performance,** whereas the
  paper's does. This is a property of our SFT stage (LoRA r = 8, 250 updates
  on 900 examples, on an already-capable instruct model), and is reported as
  such.
- **GP-VL RL solves fewer hands on the first attempt.** Against SFT this holds
  in every setting (p ≤ 0.002). Against Init it is significant on visual-OOD
  only (p = 0.016). All of GP-VL RL's gain comes from retries, whereas GP-L RL
  improves the first attempt too. A plausible cause, not verified: the `final`
  reward pays the same whether a hand is solved on attempt 1 or 5.
- **One seed per run.** These are single runs, not averages.

Full protocol, per-setting tables, recovery analysis and caveats:
**[QWEN_EVAL.md](QWEN_EVAL.md)**.

## What changed relative to the paper

| | paper | this reproduction |
| --- | --- | --- |
| model | Llama-3.2-11B-Vision-Instruct | Qwen3-VL-8B-Instruct (bf16) |
| framework / hardware | PyTorch + DeepSpeed, 8 × 80 GB GPUs | MLX, one Apple M4 Max, 128 GB |
| fine-tuning | full | LoRA, r = 8, language layers only |
| RL algorithm | PPO with a value head | multi-turn GRPO: groups of 8 full verifier-feedback trajectories, no critic |
| verifier | as shipped | whitespace-tolerant (as shipped, a correct `10 + 5*2 + 4 = 24` fails on spaces; see BUGS.md #1) |
| tasks | GeneralPoints, V-IRL | GeneralPoints only |

The RL recipe is the same for GP-L and GP-VL, as upstream's is:
- objective: clipped surrogate (0.2), k3 KL to the SFT model at 0.01,
  per-trajectory loss normalisation, Dr. GRPO advantages (no std scaling);
- reward and sampling: `final` success reward, dynamic sampling;
- optimisation: lr 1e-5, 60 updates of 4 hands × 8 trajectories.

## What was found along the way

Every defect is logged in **[BUGS.md](BUGS.md)** with how it was found and its
measured impact. The ones that matter most to anyone doing similar work:

- **mlx-vlm 0.7.2 `batch_generate` gives left-padded rows wrong RoPE positions**
  (#26). A padded row decodes at positions 17, −2, −1, 0 instead of 17, 18,
  19, 20.
  - Multi-turn RL batches conversations of different lengths, so the rollout
    sampled from a corrupted distribution, and every early multi-turn run hit a
    KL spike.
  - The fix batches only equal-length prompts and checks, on every call, that
    the sampler conditions on exactly the tokens the update scores.
  - Still present in mlx-vlm 0.7.8. Reported upstream with a tested one-line
    fix: [Blaizzy/mlx-vlm#2517](https://github.com/Blaizzy/mlx-vlm/issues/2517).
- **`batch_generate` applies the chat template itself,** so passing templated
  text templates it twice (#26). The measured impact was negligible.
- **The shipped verifier** rejects formulas with spaces (#1), rejects some of
  the solver's own solutions through float error (#2), and scores valid JSON as
  a card misread (#9).
- **SFT data** used a slightly different prompt than evaluation (#28).
  Re-measured: immaterial to the results.

## Setup

Apple silicon. GP-VL training needs a large GPU working set: VL SFT peaks at
about 97 GB. Python 3.11.

```bash
git clone https://github.com/MarcusXLee/SFT_vs_RL_qwen_mlx.git
cd SFT_vs_RL_qwen_mlx
python3.11 -m venv .venv-qwen
.venv-qwen/bin/pip install -r requirements-qwen.lock.txt     # mlx 0.32.2, mlx-vlm 0.7.2
.venv-qwen/bin/hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
```

The scripts put the repository and `gym/` on the path themselves, so nothing
else needs installing.

- **mlx-vlm version:** keep it at 0.7.2 unless you re-run
  `scripts/verify_rollout_fidelity.py` and `scripts/verify_vl_grpo.py`. The
  rollout guards were verified against this version's behaviour.
- **Original CUDA environment:** the upstream Llama/CUDA code is still here
  under its original paths. Its environment is `requirements.txt`, and
  LOCAL_SETUP.md records what it takes on a Mac.

## Reproduce

Each step writes into `data/`, `train_ckpt/` or `logs/`, all gitignored.

```bash
PY=.venv-qwen/bin/python

# 1. SFT data from the environment's solver; the 184 evaluation hands are held out
$PY scripts/make_sft_data.py -n 1000                                   # GP-L
$PY scripts/make_sft_data.py -n 1000 --vision --out data/sft_gp_vl     # GP-VL, rendered cards

# 2. SFT (LoRA r=8, 250 updates of 4 examples)
$PY scripts/train_sft_qwen.py                                          # GP-L
$PY scripts/train_sft_qwen.py --data data/sft_gp_vl --out train_ckpt/sft_gp_vl \
    --batch-size 1 --grad-accum 4 --iters 1000                         # GP-VL (memory)

# 3. multi-turn GRPO from the SFT checkpoint (GP-L ~9 h; GP-VL ~22 h)
COMMON="--multi-turn --reward-mode final --max-turns 5 --updates 60 --group-size 8 \
  --prompts-per-update 4 --temperature 1.0 --max-tokens 128 --lr 1e-5 --kl-coef 0.01 \
  --clip-eps 0.2 --entropy-coef 0.002 --dynamic-sampling --batch-rollout \
  --strip-formula-whitespace --save-every 10 --seed 0"
$PY scripts/train_grpo_qwen.py $COMMON --adapter train_ckpt/sft_gp_language \
    --out train_ckpt/mt_grpo_v5 --microbatch 4
$PY scripts/train_grpo_qwen.py $COMMON --vision --adapter train_ckpt/sft_gp_vl \
    --out train_ckpt/mt_grpo_vl_v1 --microbatch 1

# 4. evaluate (n=200; tolerant verifier, greedy, seed 42) and compare, paired
bash scripts/run_gp_l_indist_ood.sh                                    # GP-L: Init, SFT, RL
bash scripts/run_gp_vl_eval_suite.sh init= sft=train_ckpt/sft_gp_vl mt-vl=train_ckpt/mt_grpo_vl_v1
$PY scripts/compare_gp_runs.py Init=logs/ablation/qwen_gp_vl__init-n200/tolerant \
    SFT=logs/ablation/qwen_gp_vl__sft-n200/tolerant RL=logs/ablation/qwen_gp_vl__mt-vl-n200/tolerant
```

**Before training**, the acceptance checks confirm that the rollout samples
from the distribution the update scores:
- `scripts/verify_rollout_fidelity.py` (GP-L);
- `scripts/verify_vl_grpo.py` (GP-VL; needs the GPU to itself).

**Unit tests:**
`$PY -m unittest discover -s scripts -p 'test_rl_*.py'`. `test_rl_tokens` and
`test_rl_policy` load the model.

Two caveats on exact reproduction:
- The reported GP-L SFT checkpoint was trained on the earlier data format
  (BUGS.md #28).
- SFT batch order is now seeded, so a re-run trains on a different example
  order than the reported checkpoints (#30).

## Repository map

| path | what |
| --- | --- |
| `evaluation/evaluator/qwen_evaluator.py`, `evaluation/configs/qwen_*.yaml` | MLX evaluator on the paper's own evaluation loop |
| `rl/trainer/grpo_trainer_mlx.py` | GRPO core: objective, update, rollout guards |
| `rl/trainer/grpo_multiturn_mlx.py` | multi-turn trajectories with verifier feedback |
| `rl/trainer/grpo_vl_mlx.py` | GP-VL: rendered cards, image-aware scorer |
| `rl/trainer/policy_mlx.py`, `storage_mlx.py` | LoRA policy and KL reference; rollout storage (PPO port, unused by GRPO) |
| `scripts/` | data, training, evaluation, comparison, diagnostics, tests |
| `QWEN_EVAL.md` | protocol and all results |
| `BUGS.md` | every defect found, with evidence |
| `third_party/REFERENCES.md` | MLX RL implementations consulted (not dependencies) |

## Not done

- V-IRL
- PPO with a critic
- second seeds
- a reward variant that favours early success (to test the GP-VL first-attempt
  finding)

## Citation

Please cite the original paper:

```bibtex
@misc{chu2025sftmemorizesrlgeneralizes,
      title={SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training},
      author={Tianzhe Chu and Yuexiang Zhai and Jihan Yang and Shengbang Tong and Saining Xie and Dale Schuurmans and Quoc V. Le and Sergey Levine and Yi Ma},
      year={2025},
      eprint={2501.17161},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2501.17161},
}
```

## Acknowledgements

- **[LeslieTrue/SFTvsRL](https://github.com/LeslieTrue/SFTvsRL)**, the
  authors' official code. Environment, verifier, prompts and evaluation loop
  come from it, along with the original project page, checkpoints and data
  ([website](https://tianzhechu.com/SFTvsRL/),
  [models & data](https://huggingface.co/collections/tianzhechu/sftvsrl-models-and-data-6797ba6de522c7de7fcb80ba)).
- Credited by the original repository:
  [RL4VLM](https://github.com/RL4VLM/RL4VLM),
  [Llama3.2-Vision-Finetune](https://github.com/2U1/Llama3.2-Vision-Finetune),
  [V-IRL](https://virl-platform.github.io/).
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
  [MLX](https://github.com/ml-explore/mlx) and
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).
- [Cargo-Cult-Research/mlx-rl](https://github.com/Cargo-Cult-Research/mlx-rl)
  and [andrew-silva/mlx-rlhf](https://github.com/andrew-silva/mlx-rlhf), read as
  references (see `third_party/REFERENCES.md`).
