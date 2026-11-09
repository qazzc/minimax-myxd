#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/dtk/env.sh
source_env="${SETUP_ENV:-$ROOT/scripts/setup_env.sh}"
if [[ -f "$source_env" ]]; then source "$source_env"; fi
export HYGON_ARCH="${HYGON_ARCH:-$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx[0-9]+/{print $2; exit}' || true)}"
export HYGON_ARCH="${HYGON_ARCH:-gfx936}"
export INFINI_ROOT="${INFINI_ROOT:-$HOME/.infini/$HYGON_ARCH}"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"
export FA_SO_PATH="${FA_SO_PATH:-$(find "${VIRTUAL_ENV:-/workspace/envs/minimax-h3}"/lib/python*/site-packages -maxdepth 1 -name 'flash_attn_2_cuda*.so' -print -quit 2>/dev/null || true)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model diffusers directory}"
FIRST_FRAME="${FIRST_FRAME:?Set FIRST_FRAME to the input image}"
OUTPUT="${OUTPUT:-$ROOT/outputs/fl2va.mp4}"
cd "$ROOT/InfiniLM"
exec python examples/minimax_h3.py fl2va \
  --model-path="$MODEL_PATH" --text="A dog falls asleep" \
  --first-frame="$FIRST_FRAME" --gpus=0,1,2,3,4,5,6,7 \
  --num-frames=124 --steps=50 --seed=42 \
  --height=384 --width=576 --output="$OUTPUT"
