#!/bin/bash
# Hiyuna preview FT orchestrator — 3 epochs on pristine Krea-2-Raw, with an eval + review
# page at every epoch boundary.
#
# Run: cd ~/Desktop/ft-preview/train && setsid nohup bash run_preview.sh > /dev/null 2>&1 < /dev/null &
# Log: ../logs/run_preview.log
#
# Structure: latent cache (sentinel-gated) -> per-epoch training segments (--resume chained)
# -> after each segment, rename the segment's final ckpt to k2-preview-ep<N>.safetensors and
# generate the usability set (768/1024/1536) + style swap set (1024) + review page.
# Segmenting (rather than one 3-epoch launch) is what makes an epoch-1 page available while
# epochs 2-3 are still to come; eval cannot run concurrently with training on one GPU.
#
# Recipe = the validated pilot recipe, unchanged: full bf16 + Adafactor + fused backward +
# gradient checkpointing, LR 1e-5 constant (100-step warmup), krea2_shift timesteps, b4 @1024^2.
set -uo pipefail
cd "$(dirname "$0")"

DS=/home/haru/Desktop/diffsynth_research
PY=$DS/venv-musubi-gpu/bin/python
ACC=$DS/venv-musubi-gpu/bin/accelerate
PREVIEW=/home/haru/Desktop/ft-preview
EVAL=$PREVIEW/eval
LOGDIR=$PREVIEW/logs
OUT=$PREVIEW/train/ckpt
NAME=k2-preview
DIT=$DS/models_smoke/raw.safetensors                                    # pristine Krea-2-Raw
VAE=$DS/models_smoke/split_files/vae/qwen_image_vae.safetensors
TE=$DS/models_smoke/text_encoders/qwen3vl_4b_bf16.safetensors
MATERIALS=$PREVIEW/caption/preview_materials.sqlite
SYNTH_CFG=$PREVIEW/caption/caption_synth_preview_config.json
EPOCHS=3

export MUSUBI_TE_CACHE_OPTIONAL=1
mkdir -p "$OUT" "$LOGDIR" cache
LOG=$LOGDIR/run_preview.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# ---- safety pin: refuse to start if the captioning orchestrator holds the GPU ----
if pgrep -f phase3_orchestrator > /dev/null; then
  log "ABORT: captioning orchestrator is running — stop it first"; exit 1
fi

log "=== preview run start: $EPOCHS epochs, base=$(basename "$DIT") ==="
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

# The segment's terminal state always lands on this single fixed path
# (train_utils.LAST_STATE_NAME = "{output_name}-state"), overwritten by each segment —
# epoch-numbered states never appear here because the epoch-end save is gated on
# (epoch+1) < num_train_epochs, which is false for a segment's own last epoch.
STATE_DIR=$OUT/$NAME-state

# ---- 1..EPOCHS) per-epoch training segments + eval ----
for (( E=1; E<=EPOCHS; E++ )); do
  EP_CKPT=$OUT/$NAME-ep$E.safetensors
  if [ ! -f "$EP_CKPT" ]; then
    RESUME=()
    if [ "$E" -gt 1 ]; then
      [ -d "$STATE_DIR" ] || { log "epoch $E: no state at $STATE_DIR to resume from — abort"; exit 1; }
      RESUME=(--resume "$STATE_DIR")
      log "=== epoch $E segment (resume: $STATE_DIR) ==="
    else
      log "=== epoch $E segment (fresh from pristine Raw) ==="
    fi
    $ACC launch --num_cpu_threads_per_process 1 $DS/musubi-tuner/src/musubi_tuner/krea2_train.py \
      --dit "$DIT" \
      --vae "$VAE" \
      --dataset_config dataset.toml \
      --sdpa --mixed_precision bf16 --full_bf16 --gradient_checkpointing \
      --optimizer_type adafactor --learning_rate 1e-5 --fused_backward_pass \
      --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \
      --max_grad_norm 0 --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
      --timestep_sampling krea2_shift --weighting_scheme none \
      --max_data_loader_n_workers 4 --persistent_data_loader_workers \
      --te_online --text_encoder "$TE" \
      --caption_materials "$MATERIALS" \
      --caption_synth_config "$SYNTH_CFG" \
      --caption_synth_debug_log "$LOGDIR/synth_captions.tsv" \
      --caption_synth_seed 42 --seed 42 \
      --max_train_epochs "$E" --save_state \
      --mem_eff_save \
      --output_dir "$OUT" --output_name $NAME "${RESUME[@]}" >> "$LOG" 2>&1 \
      || { log "epoch $E training FAILED (see $LOG)"; exit 1; }
    # the segment's terminal save is the model-only DiT state_dict under the plain name;
    # pin it to this epoch before the next segment overwrites it.
    [ -f "$OUT/$NAME.safetensors" ] || { log "epoch $E: expected $OUT/$NAME.safetensors, missing"; exit 1; }
    mv "$OUT/$NAME.safetensors" "$EP_CKPT"
    log "epoch $E ckpt: $EP_CKPT ($(du -h "$EP_CKPT" | cut -f1), model-only)"
  else
    log "epoch $E ckpt already present -> skipping training segment"
  fi

  # ---- eval: usability (768/1024/1536) + style swap (1024) ----
  log "=== epoch $E eval: usability x3 tiers + style swap ==="
  $PY "$EVAL/generate_preview_eval.py" --dit "$EP_CKPT" --name "ep$E" --set both >> "$LOG" 2>&1 \
    || log "epoch $E eval generation had errors (non-fatal, continuing)"
  NAMES=(); for (( K=1; K<=E; K++ )); do NAMES+=("ep$K"); done
  $PY "$EVAL/build_preview_review.py" --names "${NAMES[@]}" \
    --out "$EVAL/outputs/review_ep$E.html" >> "$LOG" 2>&1 \
    || log "epoch $E review page FAILED (non-fatal)"
  log "=== EPOCH_${E}_PAGE_READY: $EVAL/outputs/review_ep$E.html ==="
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
