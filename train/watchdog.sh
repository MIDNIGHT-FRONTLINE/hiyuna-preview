#!/bin/bash
# Generic training supervisor — restarts a training command on crash AND on stall.
#
# Written for the preview run after an Xid 8 / RC-watchdog GPU lockup killed a segment at
# 98% and left the card idle for hours because nothing restarted it. Kept free of any
# run-specific knowledge so the main fine-tune can reuse it as-is: it never builds the
# training command or the resume argument itself, it only supervises whatever it is given.
#
#   watchdog.sh --log FILE [options] -- <command...>
#
#   --log FILE                 log the supervised command appends to; its growth is the
#                              liveness signal (required)
#   --heartbeat-timeout SECS   restart if neither the log nor the step counter advances for
#                              this long (default 900 — must exceed the longest legitimate
#                              pause: model load and multi-GB checkpoint writes)
#   --max-attempts N           total attempts before giving up (default 5)
#   --retry-wait SECS          pause between attempts (default 120)
#   --poll SECS                liveness poll interval (default 60)
#   --label NAME               prefix for this watchdog's own log lines (default "watchdog")
#
# The command is expected to RESOLVE ITS OWN RESUME POINT at startup, because every restart
# re-runs it verbatim. Pair it with a launcher that picks the newest saved state.
#
# Exit: 0 if the command eventually succeeded, 1 if attempts were exhausted.
set -uo pipefail

LOG=""; HEARTBEAT=900; MAX_ATTEMPTS=5; RETRY_WAIT=120; POLL=60; LABEL="watchdog"
while [ $# -gt 0 ]; do
  case "$1" in
    --log) LOG="$2"; shift 2 ;;
    --heartbeat-timeout) HEARTBEAT="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --retry-wait) RETRY_WAIT="$2"; shift 2 ;;
    --poll) POLL="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "watchdog: unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$LOG" ] || { echo "watchdog: --log is required" >&2; exit 2; }
[ $# -gt 0 ] || { echo "watchdog: no command given" >&2; exit 2; }

wlog() { echo "[$(date '+%F %T')] [$LABEL] $*" >> "$LOG"; }

# Liveness fingerprint: log size plus the trainer's current step. Size alone already moves
# every step (progress bar), but the step counter distinguishes real training progress from
# a process that is merely spewing repeated errors.
progress_sig() {
  local sz step
  sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  step=$(tail -c 20000 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -oE "[0-9]+/[0-9]+ \[" | tail -1)
  printf '%s|%s' "$sz" "$step"
}

# Kill a stalled run leaves-first. Deliberately NOT a process-group kill: the supervised
# command shares this script's process group, so "kill -- -PGID" would take the watchdog
# down with it. Walking the tree reaches accelerate -> python -> dataloader workers, which
# is what actually holds the GPU.
kill_tree() {
  local pid=$1 sig=$2 child
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child" "$sig"; done
  kill "-$sig" "$pid" 2>/dev/null
}

for (( ATTEMPT=1; ATTEMPT<=MAX_ATTEMPTS; ATTEMPT++ )); do
  wlog "attempt $ATTEMPT/$MAX_ATTEMPTS: starting supervised command"
  "$@" &
  CHILD=$!

  LAST_SIG=$(progress_sig); LAST_MOVE=$(date +%s); STALLED=0
  while kill -0 "$CHILD" 2>/dev/null; do
    sleep "$POLL"
    SIG=$(progress_sig)
    NOW=$(date +%s)
    if [ "$SIG" != "$LAST_SIG" ]; then
      LAST_SIG="$SIG"; LAST_MOVE=$NOW
    elif [ $((NOW - LAST_MOVE)) -ge "$HEARTBEAT" ]; then
      wlog "STALL: no progress for $((NOW - LAST_MOVE))s (limit ${HEARTBEAT}s) — killing the process tree"
      kill_tree "$CHILD" TERM
      sleep 20
      kill_tree "$CHILD" KILL
      STALLED=1
      break
    fi
  done

  wait "$CHILD" 2>/dev/null; RC=$?
  [ "$STALLED" -eq 1 ] && RC=124  # conventional timeout code

  if [ "$RC" -eq 0 ]; then
    wlog "attempt $ATTEMPT succeeded"
    exit 0
  fi
  wlog "attempt $ATTEMPT failed (rc=$RC)$([ "$RC" -eq 124 ] && echo ' — stalled')"
  if [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; then
    wlog "waiting ${RETRY_WAIT}s, then restarting (the command re-resolves its own resume point)"
    sleep "$RETRY_WAIT"
  fi
done

wlog "GAVE UP after $MAX_ATTEMPTS attempts"
exit 1
