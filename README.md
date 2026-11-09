# MiniMax-H3 FL2VA on InfiniLM / InfiniCore / NineToothed

This repository contains the source changes and reproducible scripts for the MiniMax-H3 Base FL2VA competition path. Model weights and competition input data are intentionally not included; obtain them from the competition provider and pass their local paths to the runner.

## Reproduce from a clean checkout

> This project targets the competition's Hygon DCU environment. A normal CPU-only machine cannot build or run the formal inference path. The tested environment is gfx936, 8 cards, DTK 26.04, Python 3.11, and the vendor-provided PyTorch 2.9 runtime. gfx906/gfx938 source support is recorded, but this exact end-to-end test has only been completed on gfx936.

```bash
git clone https://github.com/<OWNER>/minimax-myxd.git
cd minimax-myxd

# Activate the competition-provided environment, then:
source scripts/setup_env.sh
scripts/preflight.py
python scripts/run_contract_tests.py
scripts/build_infinicore.sh
scripts/build_infinilm.sh

# Supply the official local model and public/sample first frame.
MODEL_PATH=/path/to/MiniMax-H3/diffusers \
FIRST_FRAME=/path/to/official-first-frame.jpg \
OUTPUT="$PWD/outputs/fl2va.mp4" \
scripts/run_fl2va.sh

# Structural/media check: expected 124 frames, 576x384, 24 FPS, with audio.
python scripts/verify_fl2va_output.py "$PWD/outputs/fl2va.mp4" --require-audio
```

The runner uses the required InfiniLM path (not `--legacy`), 8 cards, prompt `A dog falls asleep`, seed 42, 50 steps, and 124 frames. Do not lower the required work or use the included sample as a substitute for official evaluation inputs.

### What a passing test means

The commands above establish that the code builds, the official FL2VA entry point runs, and the output has the expected readable media structure. They do **not by themselves** establish competition correctness. A reviewer must also:

1. Visually inspect the generated video for first-frame conditioning, prompt adherence, and obvious artifacts.
2. Compare against the official corrected PyTorch reference and competition-provided reference using the supplied comparison program, for example:

   ```bash
   cd InfiniLM
   python examples/minimax_h3_compare.py \
     --model-path "$MODEL_PATH" --tasks fl2va \
     --reference-fl2va-video /path/to/reference.mp4 \
     --infinilm-fl2va-video "$OUTPUT" \
     --output-dir "$PWD/../outputs/compare"
   ```

3. Run all required competition cases and inspect their logs/results. A single successful smoke case is not a full preliminary pass.

The `--legacy` option is retained for reference/comparison use only; the formal runner deliberately does not use it. A chunked PyTorch SDPA helper is available for constrained comparison environments and is not part of formal timing.

## Current verified result

On 2026-11-09, the corrected native path completed gfx936 8-card inference with 50 configured steps, 124 decoded frames, 576x384, 24 FPS, and audio. Five sampled frames were visually inspected: the conditioned dog is recognizable and gradually lies down to sleep, without the earlier mosaic corruption. The sample video and raw log are in `validation/media/` and `validation/logs/`.

**Status limitation:** the corrected output has not yet been compared against a fresh corrected PyTorch reference and the official benchmark reference. Therefore this repository demonstrates successful inference and a visual smoke pass, not a claim that every preliminary quality criterion has passed.

## Repository map

- `InfiniLM/`, `InfiniCore/`: competition source trees with the adapter and target-platform changes.
- `patches/`: source patches against the upstream branch revisions recorded in `VERSION_MANIFEST.json`.
- `ninetoothed/`: pinned NineToothed integration and permitted release artifact/source used by the Hygon AOT build.
- `scripts/`: environment preflight, build, formal run, batch run, and media verification.
- `tests/`: submission contract and gfx936 profile checks.
- `validation/`: concise report, raw logs, and the single corrected sample result.

Weights, datasets, local credentials, caches, and compiled files are not tracked. See `LICENSE` files in the included upstream projects and `ninetoothed/vendor/LICENSE`; each upstream component retains its own license terms.
