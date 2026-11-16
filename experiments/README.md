# Experiments

Five scripts, one per experiment of `docs/DESIGN.md` section 4. They share `common.py` (command
line, keyed seeds, checkpoints, Wilson intervals, timing) and the library module
`synthval.pipeline`, which holds the real-data pipeline of E1, E3, E4 and E5. Each script's
docstring is the detailed reference: designs, outputs and caveats.

## Conventions (all scripts)

```bash
uv run python experiments/<script>.py [--quick] [--part NAME ...] [--workers N] [--fresh] [--plots-only]
```

- `--part` can be repeated or comma-separated. By default all parts run in their canonical
  order.
- `--workers` defaults to `SYNTHVAL_WORKERS`, else 4.
- `--quick` runs a smoke test that writes only to `results/<exp>_quick/` (figures and tables
  included). It then times a few replications at the full settings and prints
  `ESTIMATED FULL-RUN WALL TIME ...` lines for 12 workers. Add `--no-probe` to skip that
  timing.
- **Full mode** writes:
  - `results/<exp>/*.csv`, `meta.json` and `meta_<part>.json`: configuration, git hash, code
    fingerprint, timing and package versions;
  - figures to `paper/figures/<exp>_*.pdf` and booktabs tables to `paper/tables/<exp>_*.tex`.
- **Seeds**: every replication's generator comes from a fixed root seed and the replication's
  key. Results therefore do not depend on the number of workers, the block size or which parts
  run. Checked during integration: E3 `pooled` at 3 and 2 workers, and E2 `main_a` at 2 and 3
  workers, gave identical CSVs (maximum numeric difference 0).
- **Checkpoints**: blocks of replications go to `results/<exp>/checkpoints/<part>/<hash>/`. An
  interrupted run resumes when you rerun the same command. The hash always covers the part's
  configuration. How much of the code it covers depends on the script:
  - E1 and E5 hash the synthval sources and the script (E5 also the E3 functions it calls and
    the fitted parameters);
  - E2 hashes the library files each part uses;
  - E3 hashes the fitted parameter table and warns when a model class changed after the fit;
  - E4 hashes only the configuration and the universes.

  When the code changes in a way the hash does not cover, run with `--fresh` to recompute.
- **Parallel processes**: when parts of one experiment run in separate processes, finish with
  `--plots-only`, so the combined tables see every part.
- **Threads**: BLAS/OpenMP threads are pinned to 1 and torch to 1 thread per worker. One worker
  uses one core.

Protocol (DESIGN 3.7), as checked in the quick outputs:

- E2 main and part e use independent windows with permutation p-values (the two-critic test
  uses the paired bootstrap).
- Every real-data or long-path test uses the dependent-window protocol (E1, E3, E4, E5 and E2
  part g):
  - learned tests: `split = time_block` with HAC p-values;
  - MMD and signature MMD: the wild bootstrap;
  - KS: `asymptotic_iid`, flagged invalid.
- E1 runs the pre-review ticker split (`absolute_ticker`, permutation) only as a comparison.

E1, E3 and E5 use the same pipeline settings, `synthval.pipeline` defaults (`test_kw = {}`,
`n_boot = 500`, `L = 20`, 100-ticker panel, `time_blocks = 10`). E5 checks its alignment with
E3 at start-up and, after both full runs, against E3's CSVs (`e5_e3_alignment.csv`). GARCH-t
fits use variance targeting with the 0.995 persistence cap in E3, E5 and E4 (CCC-GARCH-t
margins, factor GARCH-t), and in E1's factor panel.

## What each part does and the full-run command

### E1 `e1_placebo.py`: size of the real-data pipeline under H0

| part | what | full run |
|---|---|---|
| `sim` | primary placebo (DESIGN E1(a)): `FactorGARCHt` fitted on the real TRAIN panel. R_past* and R_new* are one contiguous stationary path. G* is an independent draw (`joint`) or has independent tickers with the same margins (`indep`). The exact E3 battery (6 absolute tests, ticker-split comparison, anchored and two-critic relative tests, R_new* vs R_past*), 200 replications per cell | `uv run python experiments/e1_placebo.py --part sim --workers 4` |
| `real` | ticker-half placebo on real data (E1(b)): 100 random splits within TEST and 100 within TRAIN; relative tests in the TRAIN split | `uv run python experiments/e1_placebo.py --part real --workers 4` |
| `nonstat` | real TEST vs real TRAIN absolute tests over 10 test seeds (E1(c)) | `uv run python experiments/e1_placebo.py --part nonstat --workers 4` |

Fixed-b HAC p-values are computed afterwards from stored fields (`--plots-only` recomputes
them).

### E2 `e2_ground_truth.py`: ground-truth size and power (independent windows)

| part | what | full run |
|---|---|---|
| `main` = `main_a..main_d` | families (a) vol level, (b) tails, (c) vol clustering, (d) jumps; n in {500, 2000}; 200 replications at H0 and 100 otherwise; critic, lrcritic, c2st, mmd, sigmmd, ks and the LR oracle (Monte Carlo nulls precomputed once per cell; family (c) uses a Monte Carlo GARCH density with K = 512) | `uv run python experiments/e2_ground_truth.py --part main --workers 6` |
| `e` | relative test at the boundary and in the interior: anchored vs two-critic, n in {500, 2000}, 200 replications; resample generator with a grouped split (size) and a naive split (leakage demonstration) | `uv run python experiments/e2_ground_truth.py --part e --workers 6` |
| `f` | learned vs exact log-likelihood ratio (GBM vs Merton, GARCH 0.98 vs 0.90); three LR-critic training budgets; 20 replications | `uv run python experiments/e2_ground_truth.py --part f --part g --workers 6` |
| `g` | windows cut from long GARCH paths: permutation vs time-block + HAC vs wild p-values, 200 replications | (with `f`, above) |

Run `main` in one process: families share the oracle null samples.

### E3 `e3_sp500.py`: S&P 500 univariate models

Run `fit` first. It caches the five fitted models per ticker, which every later part and E5
reuse. Run `power` and `realvol` (the pre-specified power check, DESIGN 1 item 3) **before**
looking at any real-data test result.

| part | what | full run |
|---|---|---|
| `fit` | GBM, Merton, GARCH-t (variance targeting), Heston (QE, SMM), stationary bootstrap per ticker on TRAIN; fit diagnostics, Heston moment match | `uv run python experiments/e3_sp500.py --part fit --workers 8` |
| `power` | design D1: power and boundary size of the verdict rule (twice-median anchored p over 10 seeds), per-ticker GARCH-t, vol x1.2, 200 replications | `uv run python experiments/e3_sp500.py --part power --workers 8` |
| `realvol` | design D2: the same rule on the real TRAIN volatility path (non-stationary boundary placebo), 200 replications | `uv run python experiments/e3_sp500.py --part realvol --workers 8` |
| `pooled` | per model x 10 synthetic seeds: 6 pooled absolute tests and the anchored and two-critic relative tests (memory: about 1 GB per worker, capped at 5) | `uv run python experiments/e3_sp500.py --part pooled --workers 4` |
| `perticker` | per ticker at L = 5: critic (HAC) and MMD (wild) on the headline synthetic path | `uv run python experiments/e3_sp500.py --part perticker,perticker_null --workers 8` |
| `perticker_null` | GARCH-t null pairs per ticker (1000 pairs), calibrating the per-ticker p-values before BH | (with `perticker`, above) |
| `witness` | refits the seed-0 pooled critic and stores its values on evaluation windows | `uv run python experiments/e3_sp500.py --part witness,stylized,likelihood --workers 5` |
| `stylized`, `likelihood` | stylized-fact table; predictive log-lik, Berkowitz, PIT-KS, Amisano-Giacomini (serial, seconds) | (with `witness`, above) |
| `agree`, `verdict` | derived from the CSVs (Kendall tau, Cohen kappa; GOOD/BAD/INCONCLUSIVE/INVALID); rerun by every invocation and by `--plots-only` | `uv run python experiments/e3_sp500.py --plots-only` |

### E4 `e4_markowitz.py`: Markowitz utility and the link with test statistics

| part | what | full run |
|---|---|---|
| `utility` | 30 universes x 30 assets, 5 MV generators (MVGaussian, MVStudentT, CCCGARCHt, MVBootstrap, FactorGARCHt), B = 20 synthetic TRAIN-length samples, sample / Ledoit-Wolf moments, 6 portfolios evaluated on real TEST against TRTR | `uv run python experiments/e4_markowitz.py --part utility --workers 8` |
| `tests` | multivariate windows (30, 5): W-critic (time blocks + HAC) and MMD (wild, fixed per-universe bandwidth), 10 synthetic seeds, plus the RealTRAIN reference row | `uv run python experiments/e4_markowitz.py --part tests --workers 8` |
| `placebo` | size of exactly the `tests` call under H0, 200 replications per generator; gives size-adjusted rejection shares | `uv run python experiments/e4_markowitz.py --part placebo --workers 8` |
| `link` | Spearman correlations between test quantities and utility losses (pooled, within-universe, within-generator, generator-level); serial, after the three parts above | `uv run python experiments/e4_markowitz.py --part link` |

### E5 `e5_robustness.py`: robustness of the E3 verdicts

Run E3 `fit` first so that E5 reads E3's pickles. Otherwise E5 refits identical models,
about 4 min.

| part | what | full run |
|---|---|---|
| `grid` | one-at-a-time grid (L in {5, 10, 60}, width x depth, steps, lr, GP weight, time blocks, train fraction, HAC lag rule; 14 configurations): pooled critic + anchored test, 10 seeds | `uv run python experiments/e5_robustness.py --part grid,gridsize,split,simpower --workers 7` |
| `gridsize` | absolute-test size per configuration (GARCH and factor designs, 200 replications each), McNemar vs default | (with `grid`) |
| `split` | alternative split TRAIN 2018-2021 / TEST 2022-2025: refits, full E3 battery (capped at 6 workers, 1 GB each), rule power D1/D2 (50 replications) | (with `grid`) |
| `simpower` | critic power on E2 alternatives: lr x steps, plus n_perm = 200 | (with `grid`) |
| `gridpower` | design D1 of the rule's power per configuration (50 replications; 200 for the default) | `uv run python experiments/e5_robustness.py --part gridpower --configs light --workers 7`, then the same with `--configs heavy` |
| `gridrealvol` | design D2 per configuration, same replication counts | `uv run python experiments/e5_robustness.py --part gridrealvol --workers 7` |

`--configs light` covers every configuration except `cap_large` and `steps3000`; `--configs
heavy` covers those two. `--power-reps` and `--power-reps-default` change the replication
counts; finished replications are reused.

### E2-dep `e2_dependence.py`: dependence-only alternatives (exact marginal)

Family `mix` (a fraction delta of the Q windows has L independent returns with the GARCH marginal)
and family `block` (Q windows are concatenations of independent GARCH blocks of length b in
{20, 10, 5, 2, 1}). The marginal law of every return is that of P, so KS has no power by
construction. All tests of E2 plus an approximate Monte Carlo LR oracle. Run as executed:
n = 2000, 100 replications under H0, 50 per alternative.
`uv run python experiments/e2_dependence.py --workers 4`

### E5 ablation `e5_architecture.py`: time-pooled CNN critic (post hoc)

The critic with `arch="pooled"` (`PooledConvCritic`) in place of the default MLP; everything else
unchanged and paired replication by replication with the experiment it is compared with.

| part | what | full run |
|---|---|---|
| `real` | E3 protocol via E5's grid/design machinery (results in `results/e5arch/`): absolute W-critic + anchored test at E3's 5 seeds; D1/D2 rule power and boundary size, 20 replications x 5 seeds | `uv run python experiments/e5_architecture.py --part real --workers 8` |
| `sim` | E2 families a-d at two alternative levels (+ H0 in a and c) and E2-dep mix/block at level 2, n as in E2 / E2-dep, 50 replications | `uv run python experiments/e5_architecture.py --part sim --workers 8` |

Tables: `e5_arch_sim.tex`, `e5_arch_real.tex` (+ notes). The pooled critic costs about ten times
the MLP per test (about 65 s against 6 s on one thread).

After all parts of an experiment: `uv run python experiments/<script>.py --plots-only`.

## Measured times and memory

All numbers come from the integration run of 2026-11-15, on the shared 16-core machine (load
average 11-13 from other jobs, including a concurrent full E2 `main` run at 5 workers). They are
therefore conservative.

- **Quick wall** is the measured `--quick` time.
- **Full estimate** is the script's own estimate from full-settings probe replications:
  measured seconds per replication x number of tasks, scheduled on 12 workers.
- **CPU h** is the summed task time, so wall time at w workers is about CPU h / w.

### Quick runs (`--quick`)

| script | workers | quick wall | notes |
|---|---|---|---|
| E2 | 2 (probes at 1) | about 6.5 min in three resumed runs | MemAvailable was about 1-2 GB then, so a memory watchdog stopped the 3-worker attempt; compute: oracle nulls 11 s, main 2.0 min, e 56 s, f 15 s, g 11 s; probes 2.4 min |
| E1 | 2 (probes at 1) | about 7.5 min in two runs | compute 1.6 min (sim 47 s, real 24 s, nonstat 13 s); full-settings probes 5.6 min serial (an earlier owner run at 3 workers: 104 s) |
| E3 | 3 | 3.4 min | fit 68 s, power 13 s, realvol 15 s, pooled 23 s, perticker 17 s, perticker_null 13 s, witness 10 s, probes |
| E4 | 3 | 3.1 min | utility 9 s, tests 48 s, placebo 42 s, probes |
| E5 | 3 | 4.5 min | grid 48 s, gridsize 15 s, split 33 s, simpower 11 s, gridpower 14 s, gridrealvol 14 s, probes 2.2 min |

Every quick run exited 0, wrote all its CSVs, tables and figures under `results/<exp>_quick/`,
and wrote nothing under `paper/`.

### Full runs: estimated wall time at 12 workers and memory

| part | full estimate at 12 workers | CPU h | memory per worker (peak RSS, full settings) |
|---|---|---|---|
| E1 sim | 30 min | 5.9 | 0.89 GB (+0.46 GB main process) |
| E1 real | 7 min | 1.2 | 0.88 GB |
| E1 nonstat | 37 s | 0.06 | 0.88 GB |
| E2 main (a-d, one process) | 62 min (+7.7 min oracle precompute) | 10.8 + 1.5 | about 0.6 GB at n = 2000 |
| E2 e | 64 min | 12.6 | about 0.6 GB |
| E2 f | 2 min | 0.26 | about 0.6 GB |
| E2 g | 3.5 min | 0.55 | about 0.6 GB |
| E3 fit | 3.9 min | 0.65 | 0.46 GB |
| E3 power | 71 min | 13.9 | 0.46 GB |
| E3 realvol | 41 min | 8.0 | 0.46 GB |
| E3 pooled | 5 min (12.6 min at the 4 workers that fit) | 0.79 | 0.91 GB (kernel matrices); capped at 5 workers |
| E3 perticker | 4.9 min | 0.78 | 0.46 GB |
| E3 perticker_null | 9.6 min | 1.56 | 0.46 GB |
| E3 witness, stylized, likelihood, agree, verdict | < 1 min | ~0 | 0.8 GB (witness) |
| E4 utility | 30 s | 0.05 | not probed separately |
| E4 tests | 102 min | 20.3 | about 0.35-0.45 GB (3 workers + main: 1.4 GB PSS in total) |
| E4 placebo | 48 min | 9.4 | same |
| E4 link | 48 s (serial) | 0.01 | main process |
| E5 grid | 22.5 min (+3.8 min fits if E3's are missing) | 3.7 | 0.46-0.6 GB |
| E5 gridsize | 86 min | 17.0 | 0.46-0.6 GB |
| E5 split | 41 min (+13 min fits and battery) | 5.3 + 0.8 | battery about 0.93 GB (capped at 6), power 0.6 GB |
| E5 simpower | 21 min | 4.0 | 0.46-0.6 GB |
| E5 gridpower | 4.35 h | 52.0 | 0.46-0.6 GB |
| E5 gridrealvol | 2.86 h | 34.2 | 0.46-0.6 GB |

Memory sources:

- E1, E3 and E5: the scripts' own peak-RSS probes. E1 resets the peak per task; E3 and E5
  report the process peak.
- E2: its docstring estimate.
- E4: total PSS of the process tree, sampled every 0.5 s during the quick run.

Total CPU time per experiment:

| experiment | CPU h |
|---|---|
| E1 | 7 |
| E2 | 26 |
| E3 | 26 |
| E4 | 30 |
| E5 | 118 |
| **total** | **about 205** |

At 12 workers this is about 17 h of wall time. Two parts exceed the 2.5 h per-part budget at
12 workers: E5 `gridpower` (4.35 h) and `gridrealvol` (2.86 h). Their replication counts are
the DESIGN minimum (50 per configuration) and E3's 200 for the default. The cheapest cut is to
run `--configs light` first and decide on `heavy` (`cap_large`, `steps3000`) separately.

## Schedule for 16 cores and about 5 GB of free RAM

On this machine RAM limits the run, not cores. Each worker holds a torch process (about
0.45 GB), and each script's main process adds about 0.45 GB. Keeping 15% headroom on 5 GB
(0.85 x 5 = 4.25 GB) allows:

- **about 8 workers** for critic-only parts (0.46-0.5 GB): E3 power, realvol, perticker,
  perticker_null, fit; E4.
- **6-7 workers** for E2 and E5 (0.6 GB).
- **4 workers** for the parts holding kernel matrices (0.9 GB): E1, E3 pooled, the E5 split
  battery.

So run **one experiment process at a time**; two processes at half the workers give the same
throughput.

Worker capping:

- E3 and E5 cap their workers from MemAvailable at the start of each part, so `--workers 12`
  is safe there.
- E1 only warns. E2 and E4 do not check, so pass `--workers` explicitly.

Before starting a job, check `grep MemAvailable /proc/meminfo`. If much more memory is free,
raise the worker counts towards 12; the times then approach the 12-worker column.

Recommended order (wall at the memory-feasible worker counts, about CPU h / workers):

| step | command(s) | workers | est. wall |
|---|---|---|---|
| 1 | E3 `fit` | 8 | 5 min |
| 2 | E3 `power`, then `realvol` (pre-specified power check, before any real-data result) | 8 | 1.7 h + 1.0 h |
| 3 | E3 `pooled` | 4 | 15 min |
| 4 | E3 `perticker,perticker_null`, then `witness,stylized,likelihood`, then `--plots-only` | 8 / 5 | 25 min |
| 5 | E2 `main` (already started by the lead on 2026-11-15 at 5 workers; it resumes from checkpoints) | 6 | about 2 h |
| 6 | E2 `e`, then `f,g`, then `--plots-only` | 6 | 2.1 h + 10 min |
| 7 | E1 `sim`, `real`, `nonstat`, then `--plots-only` | 4 | 1.8 h |
| 8 | E4 `utility`, `tests`, `placebo`, then `link` and `--plots-only` | 8 | 3.7 h |
| 9 | E5 `grid,gridsize,split,simpower` | 7 (split battery 4) | 4.5 h |
| 10 | E5 `gridpower --configs light`, then `--configs heavy` | 7 | 7.4 h in total |
| 11 | E5 `gridrealvol`, then `--plots-only` (E5 compares its default rows with E3's CSVs) | 7 | 4.9 h |

The total is about 30 h of wall time at about 5 GB free; E5 steps 9-11 take 17 h of it. With
12 workers' worth of memory (about 11 GB free) the same sequence takes about 17 h. Steps 1-4
must precede E5. E2, E1 and E4 are independent of E3 and of each other, so they can run in any
order or overnight.
