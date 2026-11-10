#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_env="${SETUP_ENV:-$ROOT/scripts/setup_env.sh}"
if [[ -f "$source_env" ]]; then source "$source_env"; fi
export HYGON_ARCH="${HYGON_ARCH:-gfx936}"
export INFINI_ROOT="${INFINI_ROOT:-$HOME/.infini/$HYGON_ARCH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export FA_SO_PATH="${FA_SO_PATH:-$FLASH_ATTN_2_CUDA_SO}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model diffusers directory}"
FIRST_FRAME="${FIRST_FRAME:?Set FIRST_FRAME to the input image}"
OUTPUT="${OUTPUT:-outputs/fl2va.mp4}"
cd "$ROOT"
exec "$PYTHON_BIN" InfiniLM/examples/minimax_h3.py fl2va \
  --model-path="$MODEL_PATH" --text="A dog falls asleep" \
  --first-frame="$FIRST_FRAME" --gpus=0,1,2,3,4,5,6,7 \
  --num-frames=124 --steps=50 --seed=42 \
  --height=384 --width=576 --output="$OUTPUT"
