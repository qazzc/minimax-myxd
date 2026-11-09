#!/usr/bin/env bash
set -euo pipefail
SUBMISSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HYGON_ARCH="${HYGON_ARCH:-$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx[0-9]+/{print $2; exit}' || true)}"
export HYGON_ARCH="${HYGON_ARCH:-gfx936}"
case "$HYGON_ARCH" in gfx906|gfx936|gfx938) ;; *) echo "Unsupported HYGON_ARCH=$HYGON_ARCH" >&2; return 2 2>/dev/null || exit 2;; esac
source /opt/dtk/env.sh
export XMAKE_ROOT=y
export PYTHONPATH="$SUBMISSION_ROOT/ninetoothed/vendor/src${PYTHONPATH:+:$PYTHONPATH}"
export INFINI_ROOT="${INFINI_ROOT:-$HOME/.infini/$HYGON_ARCH}"
export PATH="$HOME/.local/bin:/opt/dtk/cuda/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"
