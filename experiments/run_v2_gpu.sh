#!/usr/bin/env bash
# GPU lane of the remaining v2 steps (from 2026-11-17 11:30): E7, E5; E4 runs in the CPU lane
# (run_v2_cpu.sh). Plots once both lanes are done. Launch detached:
#   WMAX=5 setsid nohup bash experiments/run_v2_gpu.sh >> results_v2/logs/v2.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Worker counts are chosen before each part from MemAvailable so that at least RESERVE_MB stay
# free: about 1500 MB of unique memory per GPU worker (measured: RSS 2.1 GB, of which ~0.8 GB
# shared libraries), 2300 MB for kernel-heavy parts (E1, E5
# split), at most WMAX workers, at least 1.
RESERVE_MB=${RESERVE_MB:-2600}
WMAX=${WMAX:-4}
nw() {  # nw <MB per worker>
  local avail n
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  n=$(( (avail - RESERVE_MB) / $1 ))
  [ "$n" -gt "$WMAX" ] && n=$WMAX
  [ "$n" -lt 1 ] && n=1
  echo "$n"
}
mkdir -p results_v2/logs
run() { echo "=== $(date '+%F %T') START $*"; uv run --frozen --project envs/gpu python experiments/v2.py "$@"; echo "=== $(date '+%F %T') END (exit $?) $*"; }
# Power designs (E3 power/realvol, E5 split/gridpower/gridrealvol): critic banks (synthval.batched,
# SYNTHVAL_BATCHED=1), one process trains the 2 x n_seeds critics of a replication together; one
# worker already saturates the GPU (~0.8 tests/s, ~2.2 GB host RAM).
runb() { echo "=== $(date '+%F %T') START [batched] $*"; SYNTHVAL_BATCHED=1 uv run --frozen --project envs/gpu python experiments/v2.py "$@"; echo "=== $(date '+%F %T') END (exit $?) [batched] $*"; }
runb e7 --workers 1
runb e5 --part split --workers $(nw 2300)
runb e5 --part gridpower,gridrealvol --workers 1
echo "=== $(date '+%F %T') GPU lane done; waiting for the CPU lane (E4)"
while pgrep -f '^bash experiments/run_v2_cpu\.sh' > /dev/null; do sleep 60; done
for e in e1 e2 e2dep e3 e6 e7 e5 e4; do run $e --plots-only; done
