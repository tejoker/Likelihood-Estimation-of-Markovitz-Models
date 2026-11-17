#!/usr/bin/env bash
# CPU lane of the remaining v2 steps (from 2026-11-17 11:30): E4 on the idle CPU cores while the GPU
# runs E7/E5 (run_v2_gpu.sh). CPU build of torch (root environment), SYNTHVAL_DEVICE=cpu; every
# other v2 setting as in experiments/v2.py. Launch detached:
#   setsid nohup bash experiments/run_v2_cpu.sh >> results_v2/logs/v2_cpu.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SYNTHVAL_DEVICE=cpu
RESERVE_MB=${RESERVE_MB:-2600}
WMAX=${WMAX:-10}
nw() {  # nw <MB per worker>: keep RESERVE_MB free, at most WMAX workers
  local avail n
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  n=$(( (avail - RESERVE_MB) / $1 ))
  [ "$n" -gt "$WMAX" ] && n=$WMAX
  [ "$n" -lt 1 ] && n=1
  echo "$n"
}
mkdir -p results_v2/logs
run() { echo "=== $(date '+%F %T') START [cpu] $*"; uv run --frozen python experiments/v2.py "$@"; echo "=== $(date '+%F %T') END (exit $?) [cpu] $*"; }
run e4 --part utility --workers $(nw 900)
run e4 --part tests,placebo --workers $(nw 900)
run e4 --part link
