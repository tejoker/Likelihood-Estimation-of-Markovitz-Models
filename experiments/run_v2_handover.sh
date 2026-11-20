#!/usr/bin/env bash
# Remaining v2 steps after the handover of 2026-11-20 (docs/HANDOVER.md): the E5 grid power design
# on the real volatility path (gridrealvol, 850 blocks, resumes from the checkpoints of the
# handover archive), then every v2 table and figure. Same commands and settings as the end of
# experiments/run_v2_s50.sh, which ran everything before it.
#
# Launch detached from the repository root (survives closing the terminal):
#   setsid nohup bash experiments/run_v2_handover.sh >> results_v2/logs/v2.log 2>&1 < /dev/null &
# Control: bash experiments/v2ctl.sh status|pause|resume|stop  (stop keeps finished blocks;
# relaunching the line above resumes from them).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results_v2/logs
run() { echo "=== $(date '+%F %T') START $*"; uv run --frozen --project envs/gpu python experiments/v2.py "$@"; echo "=== $(date '+%F %T') END (exit $?) $*"; }
# Critic banks (synthval.batched): one process trains the 2 x 10 critics of a replication together;
# one worker saturates the GPU (~2.3 GB host RAM, ~2.9 GB GPU memory on the original laptop).
runb() { echo "=== $(date '+%F %T') START [batched] $*"; SYNTHVAL_BATCHED=1 uv run --frozen --project envs/gpu python experiments/v2.py "$@"; echo "=== $(date '+%F %T') END (exit $?) [batched] $*"; }
runb e5 --part gridrealvol --workers 1
for e in e1 e2 e2dep e3 e6 e7 e5 e4; do run $e --plots-only; done
