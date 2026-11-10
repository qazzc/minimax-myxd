#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_env="${SETUP_ENV:-$ROOT/scripts/setup_env.sh}"
if [[ -f "$source_env" ]]; then source "$source_env"; fi

export XMAKE_ROOT=y
export HYGON_ARCH="${HYGON_ARCH:-gfx936}"
export INFINI_ROOT="${INFINI_ROOT:-$HOME/.infini/$HYGON_ARCH}"
export XMAKE_BUILD_JOBS="${XMAKE_BUILD_JOBS:-16}"
export NINETOOTHED_BUILD_JOBS="${NINETOOTHED_BUILD_JOBS:-2}"
export XMAKE_BUILDDIR="${XMAKE_BUILDDIR:-build-$HYGON_ARCH}"
XMAKE_FEATURE_ARGS=()

case "$HYGON_ARCH" in gfx906|gfx936|gfx938) ;; *) echo "Unsupported HYGON_ARCH=$HYGON_ARCH" >&2; exit 2;; esac
cd "$ROOT/InfiniCore"
xmake global --network=private || true
if [[ "$HYGON_ARCH" == gfx936 ]]; then
  export NINETOOTHED_BACKEND=triton
  export NINETOOTHED_PLATFORM=hygon-bw1000
  export NINETOOTHED_COMPUTE_ARCH=gfx936
  python "$ROOT/scripts/build_hygon_swiglu.py"
  XMAKE_FEATURE_ARGS+=(--hygon-ninetoothed-swiglu=y)
fi
xmake f -y -cv --hygon-dcu=y --hygon-arch="$HYGON_ARCH" --cuda="$DTK_CUDA_ROOT" --ccl=y --aten=y -m release --builddir="$XMAKE_BUILDDIR" "${XMAKE_FEATURE_ARGS[@]}"
xmake build -j "$XMAKE_BUILD_JOBS"
xmake install
xmake build -j "$XMAKE_BUILD_JOBS" _infinicore
xmake install _infinicore
python -m pip install -e . --no-deps --no-build-isolation
