#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/dtk/env.sh
source_env="${SETUP_ENV:-$ROOT/scripts/setup_env.sh}"
if [[ -f "$source_env" ]]; then source "$source_env"; fi

export HYGON_ARCH="${HYGON_ARCH:-$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx[0-9]+/{print $2; exit}' || true)}"
export HYGON_ARCH="${HYGON_ARCH:-gfx936}"
export INFINI_ROOT="${INFINI_ROOT:-$HOME/.infini/$HYGON_ARCH}"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model diffusers directory}"
CASES_FILE="${CASES_FILE:?Set CASES_FILE to a tab-separated case manifest}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/batch-$(date +%Y%m%d-%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STEPS="${STEPS:-50}"
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-576}"
NUM_FRAMES="${NUM_FRAMES:-124}"

[[ -d "$MODEL_PATH" ]] || { echo "MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 2; }
[[ -f "$CASES_FILE" ]] || { echo "CASES_FILE does not exist: $CASES_FILE" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
SUMMARY="$OUTPUT_DIR/summary.tsv"
if [[ -e "$SUMMARY" ]]; then
  echo "Refusing to overwrite existing batch summary: $SUMMARY" >&2
  exit 2
fi
printf 'case_id\texit_code\toutput\tlog\n' > "$SUMMARY"

failed=0
while IFS=$'\t' read -r case_id prompt first_frame seed extra || [[ -n "${case_id:-}" ]]; do
  [[ -z "${case_id:-}" || "$case_id" == \#* ]] && continue
  if [[ -n "${extra:-}" || ! "$case_id" =~ ^[A-Za-z0-9._-]+$ || -z "${prompt:-}" || -z "${first_frame:-}" || ! "${seed:-}" =~ ^[0-9]+$ ]]; then
    echo "Invalid manifest row for case '$case_id'; expected case_id<TAB>prompt<TAB>first_frame<TAB>seed" >&2
    failed=1
    continue
  fi
  if [[ ! -f "$first_frame" ]]; then
    echo "Missing first frame for case '$case_id': $first_frame" >&2
    printf '%s\t%s\t%s\t%s\n' "$case_id" 2 "$OUTPUT_DIR/$case_id.mp4" "$OUTPUT_DIR/$case_id.log" >> "$SUMMARY"
    failed=1
    continue
  fi

  output="$OUTPUT_DIR/$case_id.mp4"
  log="$OUTPUT_DIR/$case_id.log"
  if [[ -e "$output" || -e "$log" ]]; then
    echo "Refusing to overwrite existing output/log for case '$case_id'" >&2
    printf '%s\t%s\t%s\t%s\n' "$case_id" 2 "$output" "$log" >> "$SUMMARY"
    failed=1
    continue
  fi

  printf '[batch] case=%s seed=%s steps=%s frames=%s prompt=%s\n' "$case_id" "$seed" "$STEPS" "$NUM_FRAMES" "$prompt"
  set +e
  "$PYTHON_BIN" "$ROOT/InfiniLM/examples/minimax_h3.py" fl2va \
    --model-path="$MODEL_PATH" --text="$prompt" \
    --first-frame="$first_frame" --gpus=0,1,2,3,4,5,6,7 \
    --num-frames="$NUM_FRAMES" --steps="$STEPS" --seed="$seed" \
    --height="$HEIGHT" --width="$WIDTH" --output="$output" \
    2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  printf '%s\t%s\t%s\t%s\n' "$case_id" "$status" "$output" "$log" >> "$SUMMARY"
  if [[ "$status" -ne 0 ]]; then failed=1; fi
done < "$CASES_FILE"

if [[ "$failed" -ne 0 ]]; then
  echo "[batch] one or more cases failed; see $SUMMARY" >&2
  exit 1
fi
echo "[batch] all cases passed; summary: $SUMMARY"
