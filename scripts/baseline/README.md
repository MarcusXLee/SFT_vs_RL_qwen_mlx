> **Superseded — do not quote these numbers.**
>
> This harness is not from the paper's authors. It scores **one attempt per
> puzzle with no verifier feedback**, whereas the paper's protocol allows
> `verify_iter: 5` retries with the verifier's message fed back into the same
> conversation, and scores success **per trajectory**. Its scorer also
> normalises whitespace, which the real environment does not.
>
> On the same seed-42 puzzles the two protocols disagree sharply:
>
> | | this harness | paper's protocol (tolerant) | paper's protocol (strict) |
> | --- | ---: | ---: | ---: |
> | GP language | 2% | 35% | 0% |
> | GP vision | 8% | 30% | 0% |
>
> Use `scripts/run_qwen_gp_ablation.sh` instead; see `QWEN_EVAL.md`. What
> remains useful here is the fixed 50-puzzle dataset with rendered images under
> `output/qwen-generalpoints-seed42*/`, and the resume-on-restart behaviour that
> the paper's `BaseEvaluator` does not have.

# GeneralPoints baseline v2

Run from macOS Terminal (MLX cannot access Metal from this Codex execution session):

```bash
cd /Users/xinyeli/Desktop/Paper/SFTvsRL
bash scripts/run_generalpoints_baseline_v2.sh
```

This prepares/reuses `output/qwen-generalpoints-seed42-v2`, runs 100 new Qwen responses, and writes `summary.json`. Rerunning resumes saved responses. Do not start concurrent runs into the same output directory.

## Changes

- The scorer parses and validates arithmetic before removing whitespace. Spaces/tabs around operators no longer invalidate correct formulas; malformed split numbers such as `1 0` remain invalid.
- Corrected prompt copies use quoted card ranks and no trailing JSON commas. Output-field descriptions are outside the JSON example. Original prompt constants remain available for provenance.
- The revised dataset copies the exact original card lists, numbers, solutions, and images; it does not resample by seed. Only output-format instructions change.
- Prompt/scorer versions and source dataset hash are saved in the revised manifest. Summary files identify the prompt and scorer versions.
- Old summaries cannot be overwritten using the new scorer without an explicit alternate filename.

The original run is preserved. Its old responses have also been rescored into `output/qwen-generalpoints-seed42/summary.whitespace-v2.json` and `scored.whitespace-v2.json`: text-only 3/50 (6%), vision 8/50 (16%). **These are corrected scores for old responses, not results from the new prompts.**

The actual v2 launch attempted here was blocked before inference by `No Metal device available`; no new responses were produced. The user must run the Terminal command above to complete the GPU portion.

## Protocol held constant

50 paired puzzles, one attempt per mode, temperature 0, maximum 512 generated tokens, no verifier feedback, target 24, J/Q/K=10, 1200×1200 source images, same model and MLX environment. This remains a pilot; it does not establish that vision outperforms language. A larger held-out evaluation is a separate experiment.

## Checks

```bash
.venv/bin/python -m unittest discover -s scripts/baseline -p 'test_*.py'
.venv/bin/python scripts/baseline/data.py self-test
```

Regression tests cover whitespace, malformed/unsafe arithmetic, recognition requirements, protection of old results, and labeled partial summaries. Original file hashes and equality of all 50 puzzle/image pairs were verified; every revised prompt example parses as JSON.
