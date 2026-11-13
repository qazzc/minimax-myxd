#!/usr/bin/env bash
set -euo pipefail
SUBMISSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HYGON_ARCH="${HYGON_ARCH:-$(rocminfo 2>&1 | awk '/Name:[[:space:]]+gfx[0-9]+/{print $2; exit}' || true)}"
export HYGON_ARCH="${HYGON_ARCH:-gfx936}"
case "$HYGON_ARCH" in
  gfx906|gfx936|gfx938) ;;
  *)
    echo "Unsupported HYGON_ARCH=$HYGON_ARCH" >&2
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 2; fi
    exit 2
    ;;
esac
export DTK_ROOT="${DTK_ROOT:-${ROCM_PATH:-${HIP_PATH:-}}}"
if [[ -n "$DTK_ROOT" && "${DTK_ROOT##*/}" == "hip" ]]; then DTK_ROOT="${DTK_ROOT%/hip}"; fi
if [[ -z "$DTK_ROOT" ]]; then
  echo "Set DTK_ROOT, ROCM_PATH, or HIP_PATH to the installed DTK root" >&2
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 2; fi
  exit 2
fi
if [[ -f "$DTK_ROOT/env.sh" ]]; then source "$DTK_ROOT/env.sh"; fi
if [[ -z "${DTK_CUDA_ROOT:-}" ]]; then
  for candidate in "$DTK_ROOT/cuda/cuda" "$DTK_ROOT/cuda/cuda-12" "$DTK_ROOT/cuda/cuda-11" "$DTK_ROOT/cuda"; do
    if [[ -d "$candidate" ]]; then DTK_CUDA_ROOT="$candidate"; break; fi
  done
fi
export DTK_CUDA_ROOT="${DTK_CUDA_ROOT:-$DTK_ROOT/cuda}"
HY_SMI_PATH="$(command -v hy-smi 2>&1 || true)"
if [[ -z "${HYHAL_ROOT:-}" && -n "$HY_SMI_PATH" ]]; then
  HYHAL_ROOT="$(cd "$(dirname "$HY_SMI_PATH")/.." && pwd)"
fi
export HYHAL_ROOT
export XMAKE_ROOT=y
export PYTHONPATH="$SUBMISSION_ROOT/InfiniLM/python:$SUBMISSION_ROOT/InfiniCore/python:$SUBMISSION_ROOT/ninetoothed/vendor/src${PYTHONPATH:+:$PYTHONPATH}"
export INFINI_ROOT="${INFINI_ROOT:-$HOME/.infini/$HYGON_ARCH}"
export PATH="$HOME/.local/bin:$DTK_CUDA_ROOT/bin:$PATH"
TORCH_LIB_DIR="${TORCH_LIB_DIR:-$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')}"
if [[ -z "${FLASH_ATTN_2_CUDA_SO:-}" ]]; then
  FLASH_ATTN_2_CUDA_SO="$(python -c 'import importlib.util; s=importlib.util.find_spec("flash_attn_2_cuda"); print(s.origin if s and s.origin else "")' || true)"
fi
export FLASH_ATTN_2_CUDA_SO
if [[ -n "$FLASH_ATTN_2_CUDA_SO" ]]; then
  export FA_SO_PATH="${FA_SO_PATH:-$FLASH_ATTN_2_CUDA_SO}"
fi
library_paths=("$INFINI_ROOT/lib" "$DTK_ROOT/lib" "$DTK_CUDA_ROOT/lib64" "$TORCH_LIB_DIR" "$DTK_ROOT/.hyhal/rocm_smi/lib" "$DTK_ROOT/.hyhal/rocm_smi")
if [[ -n "$HYHAL_ROOT" ]]; then library_paths+=("$HYHAL_ROOT/lib"); fi
export LD_LIBRARY_PATH="$(IFS=:; printf '%s' "${library_paths[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
