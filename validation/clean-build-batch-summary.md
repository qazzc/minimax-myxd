# Clean Build and Batch Summary

Date: 2026-11-08
Architecture: gfx936, 8 x Hygon BW1000
Submission source: clean extracted checkout
Install prefix: isolated gfx936 prefix

## Build and Operator Test

- InfiniCore build and install: exit 0. Main build: 513.525 s; Python binding build: 48.34 s.
- InfiniLM build and install: exit 0. Native build: 104.286 s.
- The generated AOT launcher translation units were compiled into the isolated build. Runtime map inspection showed `libinfiniop.so` loaded from the isolated install prefix.
- Official `InfiniCore/test/infiniop/swiglu.py --hygon`: exit 0, `Test passed!` for F16/BF16/F32, supported shapes/strides and in-place modes.

## Sequential FL2VA

The submission batch runner executed `validation/gfx936-two-case-smoke.tsv` sequentially. Both used 8 cards, 50 steps, seed 42/43 respectively, 384x576, and 124 requested frames.

| Case | Prompt | Exit | Output checks |
| --- | --- | ---: | --- |
| `public-sample-42` | `A dog falls asleep` | 0 | H.264/AAC, 576x384, 24 FPS, 124 decoded frames, non-black |
| `sequential-repeat-43` | `A dog slowly turns its head and falls asleep` | 0 | H.264/AAC, 576x384, 24 FPS, 124 decoded frames, non-black |

Official comparison script for the first case versus the PyTorch reference reported size PASS, length PASS, and Average SSIM `96.741%`.

## Evidence

Raw logs and generated videos are intentionally not versioned. The build, operator, batch, and comparison summaries above preserve the validation result without machine-specific file paths or large generated artifacts.
