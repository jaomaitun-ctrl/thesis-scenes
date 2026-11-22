#!/usr/bin/env bash
# Run train_ppo.py until <name> reaches <total_steps>, restarting from the newest checkpoint
# after a crash (OOM kill, SOFA abort, ...).  Memory is normally handled inside train_ppo.py
# (env workers are respawned past --max-worker-mb / below --min-free-mb); this is the backstop.
#
#   rl/train_supervised.sh <name> <total_steps> [train_ppo args...]
#   e.g. rl/train_supervised.sh ppo_p44_server_v2_shape 200000 --envs 5 --w-shape 0.01
#
# Uses rl/run_server.sh when the server SOFA build exists, otherwise rl/run.sh (Mac).
# Log: rl/runs_<name>.log (appended).  Gives up after MAX_RESTARTS consecutive crashes.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=$1; TOTAL=$2; shift 2
MAX_RESTARTS=${MAX_RESTARTS:-5}
RUN=rl/run.sh; [ -x /srv/train/sofa/.pixi/envs/supported-plugins/bin/python ] && RUN=rl/run_server.sh
DIR=rl/runs/$NAME; LOG=rl/runs_${NAME}.log
fails=0
while true; do
  if [ -f "$DIR/final_model.zip" ]; then echo "=== $NAME finished" | tee -a "$LOG"; exit 0; fi
  CK=$(ls "$DIR"/ckpt_*_steps.zip 2>/dev/null | sort -V | tail -1)
  RES=(); [ -n "$CK" ] && RES=(--resume "$PWD/$CK")
  echo "=== $(date '+%F %T') start ${CK:+from $(basename "$CK")} (restart $fails)" | tee -a "$LOG"
  "$RUN" rl/train_ppo.py --name "$NAME" --total-steps "$TOTAL" ${RES[@]+"${RES[@]}"} "$@" 2>&1 \
    | grep --line-buffered -vE '^\[INFO\]' | tee -a "$LOG"
  [ -f "$DIR/final_model.zip" ] && continue
  fails=$((fails + 1))
  echo "=== $(date '+%F %T') crashed (${fails}/${MAX_RESTARTS})" | tee -a "$LOG"
  [ $fails -ge "$MAX_RESTARTS" ] && { echo "=== giving up" | tee -a "$LOG"; exit 1; }
  sleep 10
done
