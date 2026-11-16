#!/usr/bin/env bash
# v2 protocol runs (docs/PROTOCOL_v2.md) on the NVIDIA GPU, in the order fixed there.
# Host RAM is the limit: each GPU worker needs about 1.4 GB (CUDA libraries), so W=5 on 15 GB.
# Every part checkpoints; rerunning resumes. Log: results_v2/logs/v2.log
#     bash experiments/run_v2.sh            (outside a sandbox that hides /dev/nvidia*)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${W:-5}
mkdir -p results_v2/logs
run() { echo "=== $(date '+%F %T') START $*"; uv run --frozen --project envs/gpu python experiments/v2.py "$@"; echo "=== $(date '+%F %T') END (exit $?) $*"; }
run e1 --part sim --workers $W
run e2 --part main --workers $W
run e2 --part e --workers $W
run e2 --part f,g --workers $W
run e2dep --workers $W
run e3 --part fit --workers $W
run e3 --part power,realvol --workers $W
run e3 --part pooled --workers $W
run e3 --part perticker,perticker_null --workers $W
run e3 --part witness,stylized,likelihood --workers $W
run e1 --part real,nonstat --workers $W
run e5 --part grid,gridsize,split,simpower --workers $W
run e5 --part gridpower,gridrealvol --workers $W
run e4 --part utility --workers $W
run e4 --part tests,placebo --workers $W
run e4 --part link
for e in e1 e2 e2dep e3 e5 e4; do run $e --plots-only; done
