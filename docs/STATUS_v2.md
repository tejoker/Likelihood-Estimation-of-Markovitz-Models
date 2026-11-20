# v2 status: first results (snapshot of 2026-11-20, 07:15)

Interim report on the v2 program defined in [`PROTOCOL_v2.md`](PROTOCOL_v2.md). This is a
snapshot, not the paper: some runs are still going (listed at the end) and the numbers below
will move into the v2 paper once they finish. What remains to be done beyond computing is in
[`ROADMAP.md`](ROADMAP.md).

**v1 vs v2.** v1 is the paper tagged `paper-v1` (CPU, MLP critic, cut replication counts).
v2 changes three things, all fixed before any v2 run:
- the time-pooled CNN critic is the main test;
- the full replication counts from DESIGN section 4 are restored;
- a QuantGAN generator is added.

v2 runs on the GPU. Results are in `results_v2/` (10 synthetic seeds) and `results_v2_s50/`
(50 seeds, amendment A2).

## Headline

Question: can synthetic returns stand in for future real S&P 500 returns? Setup: 100 stocks,
TRAIN 2018-2022, TEST 2023-01-03 to 2025-06-16.

- **GBM is BAD.** It is rejected with p < 0.001 at 10 and at 50 seeds.
- **No other generator is rejected.** That covers GARCH-t, QuantGAN, Heston and Merton.
- **Certified tolerance.** With 50 seeds, the test catches a volatility error of +40% in
  99.5% of replications [97.2, 99.9]; with 10 seeds, in 92%. So these four generators are GOOD
  within a ±40% volatility tolerance. This is the directional reading, worked out after the
  results were seen and labelled post hoc (see below).
- **Pre-registered rule as fixed (amendment A1):** certifies only ±100%.
- **The pre-specified ±20% tolerance is not certified.** On realistic volatility paths (design
  D2), the rule has power 0.39-0.41, below the required 0.8. That is the INCONCLUSIVE of the
  pre-registered rule.
- **Ranking by distance from rejection:** GARCH-t and QuantGAN are furthest, Heston and Merton
  are borderline.

## 1. Verdicts on the S&P 500 (E3, E6)

The verdict comes from the anchored relative test: is the generator at least as close to the
TEST period as the real TRAIN past is? Its p-value is min(1, 2 × median p over seeds).

| Generator | p, 10 seeds | p, 50 seeds | Verdict at the pre-specified ±20% |
|---|---|---|---|
| GBM | < 0.001 | < 0.001 | **BAD** |
| Merton | 0.436 | 0.098 | not rejected (closest to BAD) |
| Heston | 0.560 | 0.122 | not rejected |
| QuantGAN | 0.265 | 0.766 | not rejected |
| GARCH-t | 0.819 | 0.823 | not rejected |
| Stationary bootstrap | n/a | n/a | relative test invalid: G copies windows the critic saw |

Files: `results_v2*/e3/e3_verdicts.csv`, `results_v2*/e6/e6_verdicts.csv`.

**Can the rule tell GOOD from INCONCLUSIVE?** It needs power ≥ 0.8 against a volatility error
of ×1.2:

| Design | 10 seeds | 50 seeds |
|---|---|---|
| D1: stationary GARCH-t, ×1.2 | power 1.00 | power 1.00 |
| D2: real TRAIN volatility path, ×1.2 | power 0.41 [0.34, 0.47] | power 0.39 [0.33, 0.46] |
| Boundary size (no error), D1 | 0.00 | 0.00 |
| Boundary size (no error), D2 | 0.02 | 0.00 |

D2 fails the power requirement, so no generator can be called GOOD at ±20%.

## 2. Certified tolerance (E7, amendment A1)

Instead of a single ±20% check, E7 finds the smallest volatility error that the data can rule
out with 80% power. Results at 10 seeds:
- 200 replications per cell;
- 400 replications on the decisive cells;
- 95% Clopper-Pearson intervals.

| Volatility error | D2 upward | D2 downward | D1 up / down |
|---|---|---|---|
| 10% | — | — | 0.965 / 0.985 |
| 20% | 0.405 | 0.00 | 1.00 / 1.00 |
| 30% | 0.415 | 0.00 | 1.00 / 1.00 |
| 35% | 0.61 | 0.02 | 1.00 / 1.00 |
| **40%** | **0.92** [0.89, 0.94] | 0.13 | 1.00 / 1.00 |
| 45% | 1.00 | 0.45 | 1.00 / 1.00 |
| 50% | 1.00 | 0.79 [0.75, 0.83] | 1.00 / 1.00 |
| 100% | 1.00 | 1.00 | 1.00 / 1.00 |

There are two readings, and both will be reported:
- **As pre-registered (A1):** power is required in every direction, which certifies ±100%.
- **Directional, post hoc:** in D2 the TEST period is about 20% calmer than TRAIN. A generator
  that is too calm is therefore closer to TEST, and it lies inside the anchored null. Requiring
  power against it was a mis-specification of A1. Dropping that requirement certifies ±40%.

**At 50 seeds** (amendment A2; 200 replications per cell; `results_v2_s50/e7tol/`):

| Volatility error | D2 upward | D1 up / down |
|---|---|---|
| 10% | — | 0.985 / 0.99 |
| 20% | 0.39 | 1.00 / 0.99 |
| 30% | 0.50 | 1.00 / 0.99 |
| 35% | 0.685 | 1.00 / 0.99 |
| **40%** | **0.995** [0.972, 0.999] | 1.00 / 0.99 |
| 45% and more | 1.00 | 1.00 / 0.99 |

- With 50 seeds, the certified tolerance is still ±40%, but the power at 40% rises from 0.92 to
  0.995. The 40% cut-off is sharp: power is 0.685 at 35%.
- Following A2, the 50-seed grid runs only the cells that matter for the directional reading.
  It has no D2 downward cells, so it gives no pre-registered A1 reading at 50 seeds.

## 3. Validity: does the test hold its size? (E1, E2, E2-dep)

**Real-data placebo.** Two halves of the same TEST period are compared at α = 0.05 (true
difference: none):

| Test and p-value | Rejection rate | Valid? |
|---|---|---|
| W-critic, HAC | 0.05 | yes |
| W-critic, fixed-b | 0.01 | yes |
| W-critic, permutation | 0.49 | no |
| MMD, wild bootstrap | 0.33 | no |
| Sig-MMD, wild bootstrap | 0.38 | no |
| KS | 0.46 | no |

- On real dependent data, only the time-block split with HAC inference controls size.
- The permutation and wild-bootstrap kernel tests are invalid there.
- Table: `results_v2/tables/e1_size_real.tex`.

**Real past vs real future, no generator** (`e1_nonstat.tex`): this compares real TRAIN with
real TEST.
- With HAC p-values, the critic tests do not separate the two periods: median p 0.34-0.41, and
  0-10% of repetitions reject.
- The permutation versions reject in 38-50% of repetitions.
- MMD and Sig-MMD with permutation, and KS, reject in 100%.

The anchored relative test is built on this reference comparison.

**Simulation with ground truth (E2, E2-dep):**
- Size is nominal on dependent families: 0.015-0.08 at α = 0.05 over 200 replications
  (`e2dep_size.tex`).
- Power is benchmarked against the exact likelihood-ratio oracle (`e2_power.tex`,
  `e2_llr.tex`).

**Simulated placebo** (E1, fresh run, 200 replications, α = 0.05; `e1_size_sim.tex`). The
panel has cross-sectionally dependent tickers:

| Test and p-value | Rejection rate |
|---|---|
| W-critic, HAC | 0.095 [0.062, 0.144] |
| W-critic, fixed-b | 0.045 |
| Anchored relative, HAC | 0.12 |
| Anchored relative, fixed-b | 0.025 |
| Permutation | 0.505 |
| Splitting by ticker | 0.93 |

- With independent synthetic tickers, the W-critic HAC test over-rejects (0.30). That is the
  known v1 problem, still open (roadmap B10).

## 4. Robustness (E5)

**Alternative train/test split** (`e5_split.tex`):
- Same verdicts: GBM BAD, the others INCONCLUSIVE.
- But the rule's boundary size in D2 rises to 0.28 [0.16, 0.42].
- This is the open problem: size control when the market regime shifts between TRAIN and TEST.
  See the roadmap.

**Calibration across critic settings** (`e5_calibration.tex`):
- The single-seed size of the absolute critic on GARCH-t placebo paths is 0.11-0.23 across 14
  settings (default 0.17).
- So the absolute test over-rejects on these paths. The decision criterion is the anchored
  relative test, whose rule size is in section 1.

**Pending:** the power-by-setting grid (`gridpower`, `gridrealvol`) is running.

## 5. Markowitz usefulness (E4)

**Distribution closeness does not predict portfolio usefulness.** Within a generator, no
test statistic has a significant correlation with the GMV variance ratio or the
certainty-equivalent losses:
- W-critic: Spearman ρ = -0.07 (p = 0.41);
- MMD²: ρ = -0.11 (p = 0.16).

The one quantity that does track GMV losses is the gap in average correlation between synthetic
and real data (`e4_link.tex`, `e4_link_pooled.tex`).

**Placebo size depends on the generator** (`e4_placebo.tex`):
- W-critic HAC holds size for i.i.d. multivariate generators: 0.03 (Gaussian) and 0.06
  (Student-t).
- It over-rejects for GARCH-type and bootstrap generators: 0.17-0.45.
- The over-rejection was already in v1 (0.13-0.27) and is larger with the pooled critic. Its
  cause is not established yet.

**Correlation regime shift:** the average pairwise correlation falls from 0.42 (TRAIN) to 0.27
(TEST), so a single-period Markowitz evaluation is fragile.

## 6. Neural generator: QuantGAN (E6)

- **Model:** pure-TCN generator after Wiese et al. (2020), with WGAN-GP and Lambert W
  preprocessing, trained on the pooled panel. Trained once on the GPU in 37 minutes; the
  specification was fixed in the protocol before its first evaluation.
- **Verdict:** not rejected (anchored p 0.265 at 10 seeds, 0.766 at 50).
- **Stylized facts** (`e6_stylized.tex`, `figures/e6_stylized.pdf`):
  - Over the TEST horizon, median per-ticker excess kurtosis is 3.8, against 7.1 in the real
    data (GARCH-t 3.9, Merton 5.2).
  - Over the full TRAIN length, it produces rare extreme outliers: pooled kurtosis 3207,
    against 12.9 in the real data.

## Still computing (no action needed)

| Run | Status |
|---|---|
| E7 at 50 seeds | done (2026-11-19, 23:00) |
| E1 simulation size, fresh run | done (2026-11-20, 00:15) |
| E5 power by setting, stationary design (`gridpower`, 690 blocks) | done (05:46) |
| E5 power by setting, real-volatility design (`gridrealvol`, 850 blocks) | 44 done; the rest is handed over to a second machine ([`HANDOVER.md`](HANDOVER.md)), ~27 h at the laptop's speed |
| All v2 tables and figures rebuilt | right after |

Control: `bash experiments/v2ctl.sh status|pause|resume|stop`.

## Where things are

| What | Path |
|---|---|
| v2 protocol and change log (amendments A1, A2) | [`docs/PROTOCOL_v2.md`](PROTOCOL_v2.md) |
| Results, 10 seeds | `results_v2/<experiment>/`, tables `results_v2/tables/`, figures `results_v2/figures/` |
| Results, 50 seeds (A2) | `results_v2_s50/` |
| Run log with environment and overrides | `results_v2*/v2_runs.jsonl` |
| v2 runner and lane scripts | `experiments/v2.py`, `experiments/run_v2*.sh` |
| GPU environment | `envs/gpu/` |
| v1 paper | `paper/main.pdf`, git tag `paper-v1` |

Checkpoints, logs and model-fit pickles are not committed. They are regenerated by the scripts.
