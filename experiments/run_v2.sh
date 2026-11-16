#!/usr/bin/env bash
# v2 protocol runs (docs/PROTOCOL_v2.md) on the NVIDIA GPU, in the order fixed there.
# Host RAM is the limit (keep >= 2 GB free): a GPU worker needs about 1.3 GB for critic-only parts
# and about 2.1 GB when it also builds kernel matrices (E1, E5 split).
# Every part checkpoints; rerunning resumes. Log: results_v2/logs/v2.log
#     bash experiments/run_v2.sh            (outside a sandbox that hides /dev/nvidia*)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Worker counts are chosen before each part from MemAvailable so that at least RESERVE_MB stay
# free: about 1400 MB per worker for critic-only parts, 2200 MB for kernel-heavy parts (E1, E5
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
run e1 --part sim --workers $(nw 2200)
run e2 --part main --workers $(nw 1400)
run e2 --part e --workers $(nw 1400)
run e2 --part f,g --workers $(nw 1400)
run e2dep --workers $(nw 1400)
run e3 --part fit --workers $(nw 1400)
run e3 --part power,realvol --workers $(nw 1400)
run e3 --part pooled --workers $(nw 1400)
run e3 --part perticker,perticker_null --workers $(nw 1400)
run e3 --part witness,stylized,likelihood --workers $(nw 1400)
run e6 --part pooled,stylized --workers $(nw 1400)   # needs the e6 fit and E3 power/realvol/stylized
run e1 --part real,nonstat --workers $(nw 2200)
run e5 --part grid,gridsize,split,simpower --workers $(nw 2200)
run e5 --part gridpower,gridrealvol --workers $(nw 1400)
run e4 --part utility --workers $(nw 1400)
run e4 --part tests,placebo --workers $(nw 1400)
run e4 --part link
for e in e1 e2 e2dep e3 e6 e5 e4; do run $e --plots-only; done
