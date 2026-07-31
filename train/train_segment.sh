#!/bin/bash
# Run one preview training segment up to a given epoch. Invoked by watchdog.sh, which re-runs
# this verbatim on every retry — so the resume point is resolved HERE, at each start, and a
# restart automatically picks up the newest periodic state.
#
#   train_segment.sh <max_train_epochs>
#
# Recipe is the pilot-validated one, unchanged: full bf16 + Adafactor + fused backward pass +
# gradient checkpointing, LR 1e-5 constant after a 100-step warmup, batch 4 @1024^2.
set -uo pipefail
cd "$(dirname "$0")"
source ./preview_config.sh

E="${1:?usage: train_segment.sh <max_train_epochs>}"

RESUME=()
LATEST=$(latest_state)
if [ -n "$LATEST" ]; then
  RESUME=(--resume "$LATEST")
  echo "[$(date '+%F %T')] [segment] epoch target $E, resuming from $LATEST" >> "$LOG"
else
  echo "[$(date '+%F %T')] [segment] epoch target $E, fresh from $(basename "$DIT")" >> "$LOG"
fi

exec $ACC launch --num_cpu_threads_per_process 1 "$TRAINER" \
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
  --save_every_n_steps "$SAVE_EVERY_N_STEPS" \
  --save_last_n_steps "$SAVE_WINDOW" --save_last_n_steps_state "$SAVE_WINDOW" \
  --mem_eff_save \
  --output_dir "$OUT" --output_name "$NAME" "${RESUME[@]}" >> "$LOG" 2>&1
