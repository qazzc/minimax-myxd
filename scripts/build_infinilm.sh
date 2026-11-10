#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_env="${SETUP_ENV:-$ROOT/scripts/setup_env.sh}"
if [[ -f "$source_env" ]]; then source "$source_env"; fi
export HYGON_ARCH="${HYGON_ARCH:-gfx936}"
export INFINI_ROOT="${INFINI_ROOT:-$HOME/.infini/$HYGON_ARCH}"
export XMAKE_BUILD_JOBS="${XMAKE_BUILD_JOBS:-16}"
export XMAKE_BUILDDIR="${XMAKE_BUILDDIR:-$ROOT/InfiniLM/build-$HYGON_ARCH}"
cd "$ROOT/InfiniLM"
xmake f -y -cv -m release --builddir="$XMAKE_BUILDDIR"
xmake build -j "$XMAKE_BUILD_JOBS" _infinilm
xmake install _infinilm
python -m pip install -e . --no-deps --no-build-isolation
