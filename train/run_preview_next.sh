#!/bin/bash
# Hiyuna preview FT orchestrator — 5 epochs on pristine Krea-2-Raw, with an eval + review
# page at every epoch boundary.
#
# Run: cd ~/Desktop/ft-preview/train && setsid nohup bash run_preview.sh > /dev/null 2>&1 < /dev/null &
# Log: ../logs/run_preview.log
#
# Structure: latent cache (sentinel-gated) -> per-epoch training segments (--resume chained,
# each supervised by watchdog.sh) -> after each segment, pin the terminal ckpt to
# k2-preview-ep<N>.safetensors and generate the usability set (real-usage prompts at
# 768/1024/1536, everything else 1024) + the style swap set + a review page.
#
# Segmenting rather than one long launch is what makes a page available at each boundary;
# a single GPU cannot sample and train at the same time. Completed epochs are detected by
# their pinned checkpoint, so re-running this script resumes wherever it left off.
#
# Resilience (added 2026-07-31 after an Xid 8 GPU lockup killed epoch 3 at 98%):
#   - in-epoch state every 500 steps, two retained -> a crash costs ~30 min, not ~7 h
#   - watchdog.sh restarts a segment on crash and on stall (heartbeat), resuming from the
#     newest state; watchdog.sh is deliberately run-agnostic and reused by the main FT.
set -uo pipefail
cd "$(dirname "$0")"
source ./preview_config.sh

export MUSUBI_TE_CACHE_OPTIONAL=1
mkdir -p "$OUT" "$LOGDIR" cache
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# ---- safety pin: refuse to start if the captioning orchestrator holds the GPU ----
if pgrep -f phase3_orchestrator > /dev/null; then
  log "ABORT: captioning orchestrator is running — stop it first"; exit 1
fi

log "=== preview run start: target $EPOCHS epochs, base=$(basename "$DIT") ==="
# sidecar version provenance (instructed): record it in the run log
$PY - <<'PYEOF' >> "$LOG" 2>&1
import sqlite3
db = sqlite3.connect("file:/home/haru/Desktop/ft-preview/caption/preview_materials.sqlite?mode=ro", uri=True)
meta = dict(db.execute("select key, value from meta"))
print(f"caption materials: sidecar_version={meta.get('sidecar_version')} "
      f"n_rows={meta.get('n_rows')} n_styled={meta.get('n_styled')} src={meta.get('source_pkg')}")
PYEOF

# ---- 0) latent cache (sentinel-gated: the keep set is frozen, so cache once) ----
SENTINEL=cache/.cache_complete
if [ -f "$SENTINEL" ]; then
  log "latent cache: sentinel present -> skipping scan"
else
  log "=== latent cache (9,439 images @1024^2 buckets) ==="
  $PY $DS/musubi-tuner/src/musubi_tuner/krea2_cache_latents.py \
    --dataset_config dataset.toml --vae "$VAE" --skip_existing >> "$LOG" 2>&1 \
    || { log "latent cache FAILED"; exit 1; }
  touch "$SENTINEL"
  log "latent cache complete (sentinel written)"
fi

# ---- 1..EPOCHS) per-epoch training segments + eval ----
for (( E=1; E<=EPOCHS; E++ )); do
  EP_CKPT=$OUT/$NAME-ep$E.safetensors
  if [ ! -f "$EP_CKPT" ]; then
    log "=== epoch $E segment (supervised; heartbeat ${HEARTBEAT_TIMEOUT}s, up to $MAX_ATTEMPTS attempts) ==="
    bash ./watchdog.sh --log "$LOG" --label "wd-ep$E" \
      --heartbeat-timeout "$HEARTBEAT_TIMEOUT" --max-attempts "$MAX_ATTEMPTS" \
      --retry-wait "$RETRY_WAIT" \
      -- bash ./train_segment.sh "$E" \
      || { log "epoch $E FAILED after all watchdog attempts — stopping"; exit 1; }

    # the segment's terminal save is the model-only DiT state_dict under the plain name;
    # pin it to this epoch before the next segment overwrites it.
    [ -f "$OUT/$NAME.safetensors" ] || { log "epoch $E: expected $OUT/$NAME.safetensors, missing"; exit 1; }
    mv "$OUT/$NAME.safetensors" "$EP_CKPT"
    # drop the in-epoch rolling artifacts: the terminal state supersedes them, ~24 GB each
    rm -rf "$OUT"/$NAME-step*-state "$OUT"/$NAME-step*.safetensors
    log "epoch $E ckpt: $EP_CKPT ($(du -h "$EP_CKPT" | cut -f1), model-only)"
  else
    log "epoch $E ckpt already present -> skipping training segment"
  fi

  # ---- eval: usability (sweep group at 3 tiers, rest at 1024) + style swap ----
  log "=== epoch $E eval ==="
  $PY "$EVAL/generate_preview_eval.py" --dit "$EP_CKPT" --name "ep$E" --set both >> "$LOG" 2>&1 \
    || log "epoch $E eval generation had errors (non-fatal, continuing)"
  NAMES=(); for (( K=1; K<=E; K++ )); do NAMES+=("ep$K"); done
  $PY "$EVAL/build_preview_review.py" --names "${NAMES[@]}" \
    --out "$EVAL/outputs/review_ep$E.html" >> "$LOG" 2>&1 \
    || log "epoch $E review page FAILED (non-fatal)"
  # how much the model still moved into this epoch (under-training vs converged)
  if [ "$E" -ge 2 ]; then
    $PY "$EVAL/convergence_delta.py" --names "${NAMES[@]}" \
      --json_out "$EVAL/outputs/convergence_ep$E.json" >> "$LOG" 2>&1 \
      || log "epoch $E convergence delta FAILED (non-fatal)"
  fi
  log "=== EPOCH_${E}_PAGE_READY: $EVAL/outputs/review_ep$E.html ==="

  # ad-hoc neutral rating ladder at the configured marks, retroactive over ep2..epE
  for MARK in $ADHOC_EPOCHS; do
    if [ "$E" -eq "$MARK" ]; then
      ADHOC=(); for (( K=2; K<=E; K++ )); do ADHOC+=("ep$K"); done
      log "=== adhoc neutral rating ladder at ep$E over ${ADHOC[*]} ==="
      bash "$EVAL/run_adhoc_rating.sh" "${ADHOC[@]}" || log "adhoc rating FAILED (non-fatal)"
    fi
  done
done

# ---- base reference last (GPU is free; does not delay any epoch page) ----
if [ ! -f "$EVAL/outputs/base_0step/style_swap/scene__notag.png" ]; then
  log "=== base_0step reference (pristine Raw, same frozen set/seeds) ==="
  $PY "$EVAL/generate_preview_eval.py" --dit "$DIT" --name base_0step --set both >> "$LOG" 2>&1 \
    || log "base_0step eval had errors (non-fatal)"
fi
ALL=(base_0step); for (( K=1; K<=EPOCHS; K++ )); do ALL+=("ep$K"); done
$PY "$EVAL/build_preview_review.py" --names "${ALL[@]}" \
  --out "$EVAL/outputs/review_all.html" >> "$LOG" 2>&1 || log "final page FAILED (non-fatal)"

log "=== PREVIEW_RUN_DONE ($EPOCHS epochs + base reference) ==="
