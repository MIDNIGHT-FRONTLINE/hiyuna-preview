#!/bin/bash
# Run the ad-hoc neutral rating ladder across the given checkpoints, then build its page.
#
#   run_adhoc_rating.sh ep2 ep3 ep4 ep5
#
# Idempotent: the generator skips images that already exist, so re-running only fills gaps.
# 5 images per checkpoint at 1024, ~4 min each pass.
set -uo pipefail
cd "$(dirname "$0")"
source ../train/preview_config.sh

SET=$EVAL/rating_ladder_neutral.json
[ $# -gt 0 ] || { echo "usage: run_adhoc_rating.sh <ckpt-name>..." >&2; exit 2; }

for CK in "$@"; do
  DIT_CK=$OUT/$NAME-$CK.safetensors
  if [ ! -f "$DIT_CK" ]; then
    echo "[adhoc] skip $CK — no checkpoint at $DIT_CK" >> "$LOG"
    continue
  fi
  echo "[$(date '+%F %T')] [adhoc] neutral rating ladder on $CK" >> "$LOG"
  $PY "$EVAL/generate_preview_eval.py" --dit "$DIT_CK" --name "$CK" \
    --set usability --usability_json "$SET" >> "$LOG" 2>&1 \
    || echo "[adhoc] generation errors on $CK (non-fatal)" >> "$LOG"
done

$PY "$EVAL/build_adhoc_page.py" --set "$SET" --names "$@" >> "$LOG" 2>&1 \
  && echo "[$(date '+%F %T')] [adhoc] ADHOC_RATING_PAGE_READY: $EVAL/outputs/review_adhoc_rating.html" >> "$LOG" \
  || echo "[adhoc] page build FAILED" >> "$LOG"
