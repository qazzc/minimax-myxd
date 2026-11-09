#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HYGON_ARCH="${HYGON_ARCH:-$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx[0-9]+/{print $2; exit}' || true)}"
export HYGON_ARCH="${HYGON_ARCH:-gfx936}"
export INFINI_ROOT="${INFINI_ROOT:-$HOME/.infini/$HYGON_ARCH}"
export PATH="${HOME}/.local/bin:/opt/dtk/cuda/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"
export XMAKE_BUILD_JOBS="${XMAKE_BUILD_JOBS:-16}"
cd "$ROOT/InfiniLM"
xmake build -j "$XMAKE_BUILD_JOBS" infinicore_infer
xmake install infinicore_infer
xmake build -j "$XMAKE_BUILD_JOBS" _infinilm
xmake install _infinilm
python -m pip install -e . --no-deps --no-build-isolation
