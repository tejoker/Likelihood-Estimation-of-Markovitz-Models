#!/usr/bin/env bash
# Full paper runs, as actually executed (16 cores, ~6-7 GB RAM free for experiments).
# Three lanes run concurrently in separate shells:  bash experiments/run_full.sh lane{A,B,C}
# Each part checkpoints, so rerunning a lane resumes. Logs: results/logs/<lane>.log
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
run() { echo "=== $(date '+%F %T') START $*"; uv run python "$@"; echo "=== $(date '+%F %T') END (exit $?) $*"; }

laneA() {  # ground truth, then placebo  (E2 main is run first: --part main --workers 5)
  run experiments/e2_ground_truth.py --part main --workers 5
  run experiments/e2_ground_truth.py --part e --workers 5
  run experiments/e2_ground_truth.py --part f,g --workers 5
  run experiments/e1_placebo.py --part sim --workers 4
  run experiments/e1_placebo.py --part real,nonstat --workers 4
  run experiments/e1_placebo.py --plots-only
  run experiments/e2_ground_truth.py --plots-only
}
laneB() {  # S&P 500 univariate, then robustness
  run experiments/e3_sp500.py --part fit --workers 5
  run experiments/e3_sp500.py --part power,realvol --workers 5   # pre-specified power check, before real-data tests
  run experiments/e3_sp500.py --part pooled --workers 4
  run experiments/e3_sp500.py --part perticker,perticker_null --workers 5
  run experiments/e3_sp500.py --part witness,stylized,likelihood --workers 4
  run experiments/e3_sp500.py --plots-only
  run experiments/e5_robustness.py --part grid,split,simpower --workers 5
  run experiments/e5_robustness.py --plots-only
}
laneC() {  # Markowitz
  run experiments/e4_markowitz.py --part utility --workers 3
  run experiments/e4_markowitz.py --part tests --test-seeds 2 --workers 3
  run experiments/e4_markowitz.py --part placebo --placebo-reps 60 --workers 3
  run experiments/e4_markowitz.py --part link --test-seeds 2 --placebo-reps 60
  run experiments/e4_markowitz.py --plots-only --test-seeds 2 --placebo-reps 60
}
"$1"
