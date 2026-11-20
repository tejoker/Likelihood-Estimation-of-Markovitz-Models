#!/usr/bin/env bash
# Control the running v2 experiment lane (experiments/run_v2.sh and everything it started).
#
#   bash experiments/v2ctl.sh status   list the lane's processes (STAT T = paused)
#   bash experiments/v2ctl.sh pause    freeze them in place (SIGSTOP): CPU and GPU are freed at once,
#                                      RAM and GPU memory stay allocated; resume continues exactly
#   bash experiments/v2ctl.sh resume   unfreeze (SIGCONT)
#   bash experiments/v2ctl.sh stop     end the lane (SIGTERM); finished blocks are checkpointed, only
#                                      the blocks in progress are lost. Restart later with
#                                      bash experiments/run_v2.sh >> results_v2/logs/v2.log 2>&1
#
# Run outside a sandbox that hides other processes.
set -u
descendants() { local c; for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done; }
# Roots: the lane script itself and the experiment processes, matched on their argv[0] (anchored),
# never on a shell whose command text merely mentions them.
roots() {
  pgrep -f '^bash experiments/run_v2(_rest|_gpu|_cpu|_precision|_s50|_handover)?\.sh' || true
  pgrep -f '^uv run (--[a-z]+ ?[^ ]* )*python experiments/v2\.py ' || true
  pgrep -f '^[^ ]*/python[0-9.]* experiments/v2\.py ' || true
}
lane_pids() {
  local p
  for p in $(roots); do echo "$p"; descendants "$p"; done | sort -un
}
P=$(lane_pids)
if [ -z "$P" ]; then echo "no v2 lane process found"; exit 0; fi
case "${1:-status}" in
  status) ps -o pid,stat,etime,rss,args -p "$(echo $P | tr ' ' ',')" | cut -c1-150 ;;
  pause)  kill -STOP $P && echo "paused $(echo $P | wc -w) processes at $(date +%T)" ;;
  resume) kill -CONT $P && echo "resumed $(echo $P | wc -w) processes at $(date +%T)" ;;
  stop)   kill -CONT $P 2>/dev/null; kill -TERM $(roots) 2>/dev/null; sleep 2; kill -TERM $P 2>/dev/null
          echo "stopped the lane at $(date +%T); finished blocks are kept. Restart by relaunching the lane script (e.g. experiments/run_v2_handover.sh, see docs/HANDOVER.md)" ;;
  *) echo "usage: $0 status|pause|resume|stop"; exit 2 ;;
esac
