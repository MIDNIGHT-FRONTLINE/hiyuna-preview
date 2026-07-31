#!/bin/bash
# Shared configuration for the preview run — sourced by run_preview.sh and train_segment.sh
# so the recipe lives in exactly one place.

DS=/home/haru/Desktop/diffsynth_research
PY=$DS/venv-musubi-gpu/bin/python
ACC=$DS/venv-musubi-gpu/bin/accelerate
TRAINER=$DS/musubi-tuner/src/musubi_tuner/krea2_train.py

PREVIEW=/home/haru/Desktop/ft-preview
EVAL=$PREVIEW/eval
LOGDIR=$PREVIEW/logs
TRAINDIR=$PREVIEW/train
OUT=$TRAINDIR/ckpt
LOG=$LOGDIR/run_preview.log
NAME=k2-preview

DIT=$DS/models_smoke/raw.safetensors                                    # pristine Krea-2-Raw
VAE=$DS/models_smoke/split_files/vae/qwen_image_vae.safetensors
TE=$DS/models_smoke/text_encoders/qwen3vl_4b_bf16.safetensors
MATERIALS=$PREVIEW/caption/preview_materials.sqlite
SYNTH_CFG=$PREVIEW/caption/caption_synth_preview_config.json

EPOCHS=5                    # gate change 2026-07-31: run 3..5 unattended, page each boundary

# In-epoch checkpointing. save_last_n_steps is a STEP WINDOW, not a count: with a 500-step
# interval, a 500-step window retains exactly the two newest saves (see
# train_utils.get_remove_step_no). Caps a crash at ~30 min of lost compute instead of a
# whole epoch, at ~2 min of write per save.
SAVE_EVERY_N_STEPS=500
SAVE_WINDOW=500

# Watchdog tuning. The heartbeat must clear the longest legitimate silence — segment startup
# loads a 25.8 GB DiT plus an 8.3 GB text encoder, and each periodic save writes ~48 GB.
HEARTBEAT_TIMEOUT=900
MAX_ATTEMPTS=5
RETRY_WAIT=120

# Newest resume point: a segment's terminal state lands on "<name>-state", periodic in-epoch
# saves on "<name>-step<N>-state". Newest wins, so a mid-epoch crash resumes from the last
# periodic save rather than replaying the epoch.
latest_state() {
  local newest="" d
  for d in "$OUT/$NAME-state" "$OUT"/$NAME-step*-state; do
    [ -d "$d" ] || continue
    if [ -z "$newest" ] || [ "$d" -nt "$newest" ]; then newest="$d"; fi
  done
  printf '%s' "$newest"
}
