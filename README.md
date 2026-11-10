# MiniMax-H3 FL2VA on InfiniLM / InfiniCore / NineToothed

This repository contains the source changes and reproducible scripts for the MiniMax-H3 Base FL2VA competition path. Model weights and competition input data are intentionally not included; obtain them from the competition provider and pass their local paths to the runner.

## Reproduce from a clean checkout

> This project targets the competition's Hygon DCU environment. A normal CPU-only machine cannot build or run the formal inference path. The tested environment is gfx936, 8 cards, DTK 26.04, Python 3.11, and the vendor-provided PyTorch 2.9 runtime. gfx906/gfx938 source support is recorded, but this exact end-to-end test has only been completed on gfx936.

```bash
git clone https://github.com/qazzc/minimax-myxd.git
cd minimax-myxd

# Activate the competition-provided Python 3.11 environment and configure `DTK_ROOT`,
# `ROCM_PATH`, or `HIP_PATH`, then:
source scripts/setup_env.sh
scripts/preflight.py
python scripts/run_contract_tests.py
scripts/build_infinicore.sh
scripts/build_infinilm.sh

# Supply the official local model and public/sample first frame.
export MODEL_PATH=../models/MiniMax-H3/diffusers
export DATA_ROOT=../data
export FIRST_FRAME="$DATA_ROOT/Golde33443.jpg"
export OUTPUT=outputs/fl2va.mp4
scripts/run_fl2va.sh

# Structural/media check: expected 124 frames, 576x384, 24 FPS, with audio.
python scripts/verify_fl2va_output.py outputs/fl2va.mp4 --require-audio
```

The runner uses the required InfiniLM path (not `--legacy`), 8 cards, prompt `A dog falls asleep`, seed 42, 50 steps, and 124 frames. Do not lower the required work or use the included sample as a substitute for official evaluation inputs.

### What a passing test means

The commands above establish that the code builds, the official FL2VA entry point runs, and the output has the expected readable media structure. They do **not by themselves** establish competition correctness. A reviewer must also:

1. Visually inspect the generated video for first-frame conditioning, prompt adherence, and obvious artifacts.
2. Compare against the official corrected PyTorch reference and competition-provided reference using the supplied comparison program, for example:

   ```bash
   python InfiniLM/examples/minimax_h3_compare.py \
     --model-path "$MODEL_PATH" --tasks fl2va \
     --reference-fl2va-video "$DATA_ROOT/reference.mp4" \
     --infinilm-fl2va-video "$OUTPUT" \
     --output-dir outputs/compare
   ```

3. Run all required competition cases and inspect their logs/results. A single successful smoke case is not a full preliminary pass.

The `--legacy` option is retained for reference/comparison use only; the formal runner deliberately does not use it. A chunked PyTorch SDPA helper is available for constrained comparison environments and is not part of formal timing. Runtime roots and external input paths are supplied through environment variables; the repository does not assume a particular home, workspace, Python, or data directory.

## Current verified result

On 2026-11-09, the candidate fixes completed three gfx936 8-card runs with 50 steps and 124 decoded frames. On 2026-11-10, an isolated clone rebuild of InfiniCore and InfiniLM passed, as did the official NineToothed SwiGLU test and media checks on all three candidate videos. A further full inference from the isolated rebuild remains pending because shared jobs occupied about 53 GiB per card. The correction history, measurements, and validation boundaries are recorded in [`validation/github-clone-2026-11-09.md`](validation/github-clone-2026-11-09.md) and [`validation/gfx936-2026-11-08.md`](validation/gfx936-2026-11-08.md). Run logs and generated videos are deliberately excluded from the repository.

**Status limitation:** the candidate's three runs were readable and scored 87.448%, 89.580%, and 89.746% SSIM against a fixed-seed PyTorch reference; repeat-to-repeat decoded RGB MAE was 3.57-4.95. The 2026-11-10 isolated build verifies compilation and the SwiGLU operator, not a fresh inference from that build. These results do not establish a competition pass: the official reference/threshold is not yet checked, and a fresh public-branch build/run remains required.

## Repository map

- `InfiniLM/`, `InfiniCore/`: competition source trees with the adapter and target-platform changes.
- `patches/`: source patches against the upstream branch revisions recorded in `VERSION_MANIFEST.json`.
- `ninetoothed/`: pinned NineToothed integration and permitted release artifact/source used by the Hygon AOT build.
- `scripts/`: environment preflight, build, formal run, batch run, and media verification.
- `tests/`: submission contract and gfx936 profile checks.
- `validation/`: concise verification reports and the public sample manifest.

Weights, datasets, local credentials, caches, and compiled files are not tracked. See `LICENSE` files in the included upstream projects and `ninetoothed/vendor/LICENSE`; each upstream component retains its own license terms.
