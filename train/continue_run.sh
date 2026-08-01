#!/bin/bash
# Chain the epoch 6-10 extension onto the currently running 5-epoch orchestrator.
#
#   setsid nohup bash continue_run.sh <pid-of-running-run_preview.sh> > /dev/null 2>&1 < /dev/null &
#
# Why this exists: the running orchestrator captured EPOCHS=5 when it started, and bash reads a
# script lazily from a byte offset — editing run_preview.sh in place while it runs would corrupt
# whatever it has not read yet. So the extension is staged in run_preview_next.sh and swapped in
# here, once the old process is gone.
#
# Sequence: wait for the old orchestrator to exit -> run the ad-hoc neutral rating ladder over
# ep2..ep5 -> swap the updated orchestrator into place -> relaunch it. The relaunched run skips
# every epoch that already has a checkpoint and the base reference if it already exists, so it
# picks straight up at epoch 6.
set -uo pipefail
cd "$(dirname "$0")"
source ./preview_config.sh

OLD_PID="${1:?usage: continue_run.sh <pid of running run_preview.sh>}"
clog() { echo "[$(date '+%F %T')] [continue] $*" >> "$LOG"; }

clog "waiting for orchestrator pid $OLD_PID to finish its 5-epoch run"
while kill -0 "$OLD_PID" 2>/dev/null; do sleep 60; done
clog "orchestrator $OLD_PID exited"

# guard: only proceed if epoch 5 actually landed, otherwise the old run died early and a
# relaunch would silently restart training we did not intend to repeat unattended.
if [ ! -f "$OUT/$NAME-ep5.safetensors" ]; then
  clog "ABORT: $OUT/$NAME-ep5.safetensors missing — the 5-epoch run did not complete; not relaunching"
  exit 1
fi

clog "ad-hoc neutral rating ladder over ep2..ep5"
bash "$EVAL/run_adhoc_rating.sh" ep2 ep3 ep4 ep5 || clog "adhoc rating had errors (non-fatal)"

if [ -f run_preview_next.sh ]; then
  mv run_preview_next.sh run_preview.sh
  clog "swapped in the extended orchestrator (EPOCHS=$EPOCHS, adhoc at: $ADHOC_EPOCHS)"
fi

clog "relaunching orchestrator for epochs 6-$EPOCHS"
setsid nohup bash run_preview.sh > /dev/null 2>&1 < /dev/null &
clog "relaunched (pid $!)"
