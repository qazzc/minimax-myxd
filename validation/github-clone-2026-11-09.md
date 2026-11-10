# GitHub Clone Validation

Date: 2026-11-09
Architecture: gfx936, 8 Hygon DCUs
Public clone base: `d342f1bbb4ee8022c0ee5830aa085156ecf43710`
Remote worktree: isolated public-repository clone

## Scope and provenance

The test began from a clone of the public repository. Candidate fixes were then applied as uncommitted working-tree changes in that clone. Therefore these runs validate the candidate source changes, but **do not** prove that a clean checkout of the then-public `main` can reproduce them. A clean build and run from a public commit containing the fixes is still required.

The official weights and first-frame input were read from the provider files already present on the container. They are not included in this repository. The runner used the formal InfiniLM path, 8 cards, prompt `A dog falls asleep`, seed 42, 50 steps, 576x384, and 124 frames.

## Findings and candidate fixes

Initial cloned runs exposed black output and unstable image conditioning. The compatibility adapter called the VAE encode helper that sampled from the global RNG before the pipeline could pass its seeded generator. In addition, several PyTorch/InfiniCore and visual-feature handoffs used asynchronous copies without ensuring the producer buffers remained valid until consumers finished.

The candidate changes preserve the VAE Gaussian posterior until `sample(generator=...)`, synchronize PyTorch/native tensor handoffs, wait for visual-feature copies before their local source tensors are released, and keep the host position-ID source alive through the asynchronous host-to-device copy. The non-finite decoded-pixel check now fails early instead of writing a black/invalid video silently.

## Results

Three full candidate runs completed with the required 50-step configuration and `124/124` decoded-frame check. The official comparison utility reported matching 576x384 dimensions and 124 frames against the fixed-seed PyTorch reference.

| Run | Pipeline time | End-to-end time | SSIM vs. fixed-seed PyTorch |
| --- | ---: | ---: | ---: |
| visual-handoff-1 | 167.755 s | 264.081 s | 87.448% |
| visual-handoff-2 | 168.148 s | 248.653 s | 89.580% |
| visual-handoff-3 | 182.179 s | 280.377 s | 89.746% |

The image-conditioned native Qwen3-VL outputs were elementwise identical across three short repeated condition-encoder calls after the handoff fix. Across the three full videos, decoded RGB mean absolute error was 3.57-4.95, so full generation is close but not bitwise deterministic. Visual inspection found recognizable subject/prompt behavior without the earlier black or mosaic corruption.

Raw per-run logs were used during diagnosis but are not included in the submission branch. The measured values and validation limitations are retained above.

## Not yet passed

- The SSIM values above compare against a locally generated fixed-seed PyTorch reference, not the organizer's official reference output. No official acceptance threshold has been confirmed.
- These candidate fixes were uncommitted in the remote clone. Clean public clone -> build -> inference -> result check has not yet been completed with a commit that contains them.
- The first preflight was invoked with the container's default system Python rather than the competition virtual environment. Activating the competition Python 3.11 environment confirmed PyTorch 2.9, Triton 3.5.1 for DTK2604/Torch290, TorchVision 0.24.0, FlashAttention, and eight gfx936 cards. The isolated rebuild completed with that environment and its own install prefix; details follow below.
- This result covers gfx936 only. It does not establish NineToothed AOT or full inference support on gfx906 or gfx938.
- The three runs cover one public sample case only. Multi-case official evaluation and organizer-side quality review remain outstanding.

Accordingly, this is candidate-level recovery evidence, not a claim that the preliminary competition gate has passed.

## 2026-11-10 isolated rebuild follow-up

The initial system-Python preflight was not representative. A temporary Python 3.11 environment using the competition environment's preinstalled DTK2604 packages passed preflight: gfx936 on all eight devices, PyTorch 2.9.0, Triton `3.5.1+das.opt1.dtk2604.torch290`, TorchVision 0.24.0, and FlashAttention were confirmed.

The cloned candidate sources were rebuilt with separate paths:

- InfiniCore and InfiniLM used distinct Xmake build directories.
- Libraries were installed into a fresh architecture-specific prefix outside the default user install.
- NineToothed gfx936 SwiGLU AOT generation completed; full InfiniCore build and install exited 0.
- Official `InfiniCore/test/infiniop/swiglu.py --hygon` exited 0 and printed `Test passed!`.
- The corrected `scripts/build_infinilm.sh` configured the isolated build directory, built the existing `_infinilm` target, installed it, and exited 0. The built extension imported successfully; `ldd` resolved `libinfinicore_cpp_api.so`, `libinfiniop.so`, `libinfiniccl.so`, and `libinfinirt.so` from the isolated install prefix.
- The three existing candidate videos passed the repository media verifier: each decoded to `(124, 384, 576, 3)`, 24 FPS, with non-empty audio.

A new 50-step run was **not** started after the rebuild: concurrent external jobs were using about 52-53 GiB of each 64 GiB device and around 40% compute. The old candidate videos were generated before this isolated rebuild, so their media checks do not substitute for inference from the new prefix. Re-run the official 50-step entry from the isolated prefix once enough GPU memory is available.

Raw build and operator logs were used to verify these results but are not included in the branch. The isolated Xmake home must be configured to see the available package cache when building offline.
