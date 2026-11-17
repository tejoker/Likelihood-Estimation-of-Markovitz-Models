#!/usr/bin/env bash
# Orchestrator from 2026-11-17 ~17:00: waits for the running 10-seed E7, then amendment A2 (50 seeds,
# results_v2_s50/), then E1 sim (fresh) and the E5 grid power designs (10 seeds, results_v2/), then tables.
# Launch detached:  WMAX=5 setsid nohup bash experiments/run_v2_s50.sh >> results_v2/logs/v2.log 2>&1 < /dev/null &
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
echo "=== $(date '+%F %T') waiting for the running 10-seed E7 to finish"
while pgrep -f '^[^ ]*/python[0-9.]* experiments/v2\.py e7' > /dev/null; do sleep 60; done
# --- amendment A2: 50 synthetic seeds, separate output tree
s50() { SYNTHVAL_N_SEEDS=50 SYNTHVAL_RESULTS_DIR=results_v2_s50 SYNTHVAL_TAB_DIR=results_v2_s50/tables SYNTHVAL_FIG_DIR=results_v2_s50/figures "$@"; }
s50 run e3 --part fit --workers $(nw 1500)
s50 runb e3 --part power,realvol --workers 1
s50 run e3 --part pooled --workers $(nw 1500)
s50 run e6 --part pooled --workers $(nw 1500)
SYNTHVAL_E7_GRID=s50 s50 runb e7 --workers 1
for e in e3 e6 e7; do SYNTHVAL_E7_GRID=s50 s50 run $e --plots-only; done
# --- remaining 10-seed steps
run e1 --part sim --workers $(nw 2300)
runb e5 --part gridpower,gridrealvol --workers 1
for e in e1 e2 e2dep e3 e6 e7 e5 e4; do run $e --plots-only; done
