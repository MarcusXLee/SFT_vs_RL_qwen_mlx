# SFTvsRL local environment setup

Project: `/Users/xinyeli/Desktop/Paper/SFTvsRL`

The Mac-compatible development environment is installed. Core imports and a GeneralPoints game pass. **The entire repository is not fully runnable on this Mac:** eight optional backend modules and the existing LoRA merge script still fail import, and the original GPU training workflow has not been run.

## Use the environment

```bash
cd /Users/xinyeli/Desktop/Paper/SFTvsRL
source .venv/bin/activate
python scripts/check_environment.py
```

Select `/Users/xinyeli/Desktop/Paper/SFTvsRL/.venv/bin/python` as your editor's interpreter.

To reinstall the recorded dependencies:

```bash
cd /Users/xinyeli/Desktop/Paper/SFTvsRL
bash scripts/setup_macos.sh
```

The setup script can reuse the Python runtime under `.python/`; on another machine, install Python 3.11 first. It is specific to Apple silicon macOS. The verifier returns success for the core workflows and explicitly reports unavailable optional modules; success does not mean every module or runtime feature works.

## What was inspected and installed

- Read and statically scanned all 123 original Python files, including nested imports, alongside the repository's setup documentation, launch scripts, and configurations. This was an environment/dependency review, not a complete correctness review of every algorithm.
- Created an isolated Python 3.11.16 environment under `.venv/`, with a local Python runtime under `.python/`. System Python was not changed.
- Installed 176 locked distributions plus the editable `sftvsrl-gym` package and environment bootstrap tools.
- Core versions: PyTorch 2.5.1, torchvision 0.20.1, Transformers 4.45.2, PEFT 0.13.2, Accelerate 1.0.1, NumPy 1.26.4.
- Added missing common dependencies, including torchvision, ujson, sentencepiece, einops, timm, Gradio, OpenCLIP, DashScope, CLIP, and LightGlue. CLIP and LightGlue are pinned to the installed upstream Git commits.
- Added source paths inside this environment so the project and SFT modules resolve without a global PYTHONPATH change.

## Verification

- All 42 explicitly checked third-party import roots passed.
- 111 project modules imported successfully; nine failed as detailed below.
- All inspected Python source files passed syntax compilation under Python 3.11.
- The package consistency check reported `No broken requirements found`.
- GeneralPoints created a 256 × 256 RGB observation and accepted a valid solution, terminating with reward 5.
- RL launcher, evaluation launcher, SFT training module, gym_cards, and gym_virl imports passed.
- CUDA and MPS availability both returned false in this execution session; CPU operations were exercised.
- No checkpoints were downloaded, no training was started, and no remote evaluation requests were sent. V-IRL was import-tested but not exercised against its absent route/street-view dataset.

The EVA-CLIP server was deliberately not imported: importing it constructs a model and starts a public Gradio server. The setup module and checker itself were also excluded from application-module import testing.

## Repository changes

- `gym/setup.py`: replaced the invalid distribution name with `sftvsrl-gym`, discovered all subpackages, and included card images/fonts as package data.
- `sft/src/training/data.py`: moved the optional `decord` import into `encode_video`; text/image workflows now import without it, while video requests receive an explanatory error if it is absent.
- `.gitignore`: ignored `.venv/`, `.python/`, and `.cache/`.
- Added `requirements-macos.txt`, `requirements-macos.lock.txt`, `scripts/setup_macos.sh`, `scripts/check_environment.py`, and this setup guide as `LOCAL_SETUP.md`.
- Original `requirements.txt`, model settings, training logic, and launch configurations were retained.

## Outstanding import failures

| Module | Observed failure |
| --- | --- |
| `virl.perception.detector.glip` | NameError: name 'GLIPDemo' is not defined |
| `virl.perception.detector.grounding_dino` | ModuleNotFoundError: No module named 'groundingdino' |
| `virl.perception.mm_llm.blip2_local` | ModuleNotFoundError: No module named 'lavis' |
| `virl.perception.mm_llm.instructblip_local` | ModuleNotFoundError: No module named 'lavis' |
| `virl.perception.mm_llm.llava_client` | ModuleNotFoundError: No module named 'llava' |
| `virl.perception.mm_llm.minigemini` | ModuleNotFoundError: No module named 'mgm' |
| `virl.perception.recognizer.llava_client` | ModuleNotFoundError: No module named 'llava' |
| `virl.perception.recognizer.paddle_ocr` | ModuleNotFoundError: No module named 'PaddleOCR' |
| `merge_lora_weights` | ImportError: cannot import name 'modify_config_file' from 'utils' (/Users/xinyeli/Desktop/Paper/SFTvsRL/sft/src/utils.py) |

The LoRA merge issue is a missing repository function (`modify_config_file`), not an installable package. It was left explicit instead of inventing checkpoint-modification behavior.

## Other runtime dependencies not installed

Static scanning also identified deferred/optional imports. A successful module import does not guarantee those features can execute:

- `PaddleOCR`: `gym/virl/perception/recognizer/paddle_ocr.py:22`, `gym/virl/perception/recognizer/paddle_ocr.py:23`, `gym/virl/perception/recognizer/paddle_ocr.py:24`, `gym/virl/perception/recognizer/paddle_ocr.py:25`, `gym/virl/perception/recognizer/paddle_ocr.py:26`
- `bitsandbytes`: `sft/src/training/trainer.py:110`
- `decord`: `sft/src/training/data.py:18`
- `deepspeed`: `sft/src/training/train_utils.py:8`, `sft/src/training/train_utils.py:9`, `sft/src/training/trainer.py:28`, `sft/src/training/trainer.py:29`
- `diffusers`: `gym/virl/perception/mm_llm/minigemini.py:14`
- `eva_clip`: `gym/virl/perception/recognizer/eva_clip_client.py:11`, `gym/virl/perception/recognizer/eva_clip_server.py:13`
- `groundingdino`: `gym/virl/perception/detector/grounding_dino.py:6`, `gym/virl/perception/detector/grounding_dino.py:7`
- `lavis`: `gym/virl/perception/mm_llm/blip2_local.py:6`, `gym/virl/perception/mm_llm/instructblip_local.py:6`
- `llava`: `gym/virl/perception/mm_llm/llava_client.py:9`, `gym/virl/perception/mm_llm/llava_client.py:10`, `gym/virl/perception/mm_llm/llava_client.py:11`, `gym/virl/perception/mm_llm/llava_client.py:12`, `gym/virl/perception/mm_llm/llava_client.py:13`, `gym/virl/perception/recognizer/llava_client.py:12`, `gym/virl/perception/recognizer/llava_client.py:13`, `gym/virl/perception/recognizer/llava_client.py:14`, `gym/virl/perception/recognizer/llava_client.py:15`, `gym/virl/perception/recognizer/llava_client.py:16`
- `maskrcnn_benchmark`: `gym/virl/perception/detector/glip.py:22`, `gym/virl/perception/detector/glip.py:79`
- `mgm`: `gym/virl/perception/mm_llm/minigemini.py:26`, `gym/virl/perception/mm_llm/minigemini.py:27`, `gym/virl/perception/mm_llm/minigemini.py:28`, `gym/virl/perception/mm_llm/minigemini.py:29`, `gym/virl/perception/mm_llm/minigemini.py:30`
- `mplug_owl`: `gym/virl/perception/mm_llm/mplug_owl_client.py:19`, `gym/virl/perception/mm_llm/mplug_owl_client.py:20`
- `otter_ai`: `gym/virl/perception/mm_llm/otter.py:20`
- `paddleocr`: `gym/virl/perception/mm_llm/minigemini.py:19`

DeepSpeed 0.15.1 is excluded from the Mac requirements. The supplied RL and local model-evaluation code accesses the DeepSpeed plugin directly; RL checkpoint saving also invokes CUDA. Quantized training references bitsandbytes, and video datasets need decord. Optional V-IRL backends require their respective upstream research repositories and compatible model weights. These integrations were not claimed to be fully configured.

Reproducing the published training experiments requires the NVIDIA server setup described in the project README (eight GPUs with 80 GB memory each), the selected checkpoints, data paths, and any API credentials for API-based workflows. Import verification is not a substitute for a full experiment.

## Supporting files and references

- `environment-check.json`: complete module results, deferred imports, hardware checks, and game-test outcome.
- `package-imports.json`: explicit third-party import results.
- `source-audit.json`: source inventory, hashes, definitions, and import statements.
- [Official PyTorch version combinations](https://docs.pytorch.org/get-started/previous-versions/)
- [Official CLIP installation](https://github.com/openai/CLIP)
- [Official LightGlue installation](https://github.com/cvg/LightGlue)
- [Decord upstream](https://github.com/dmlc/decord)
