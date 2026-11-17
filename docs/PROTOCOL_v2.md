# Protocol v2 (fixed 2026-11-16, before any v2 run)

v1 is the protocol of `docs/DESIGN.md` as run on CPU on 2026-11-15 (change log entries of that
date), reported in the first version of the paper. v2 is the protocol of the improved version. It
is written down and committed before any v2 experiment is run; every deviation found later is
added to the change log at the end of this file with its date, never edited in place.

## What changes from v1

1. **Critic architecture.** Every learned test uses the time-pooled CNN
   (`synthval.twosample.PooledConvCritic`: two `Conv1d` blocks, 64 channels, kernel 5, mean over
   time, dense head of width 64), univariate windows `(L,)` and multivariate windows `(C, L)`
   alike: the W-critic, the LR-critic, C2ST, the anchored relative test and the two-critic test.
   All other settings are unchanged (`pipeline.CRITIC_KW`, DESIGN section 3.7). Set through
   `SYNTHVAL_ARCH_1D=pooled`, `SYNTHVAL_ARCH_2D=pooled`.
   *Reason:* the post-hoc ablation of v1 (`experiments/e5_architecture.py`, DESIGN change log
   2026-11-16) found much higher power at nominal size on every simulated design.
   *Known before v2:* that ablation already ran the pooled critic on the v1 S&P 500 TRAIN/TEST
   split (same five decisions as the MLP). The v2 real-data results on that split are therefore
   not blind; blind confirmation needs data not yet analysed (a further market or period,
   journal version).
2. **Device.** NVIDIA GPU (`SYNTHVAL_DEVICE=cuda`, `envs/gpu`, torch 2.15.0+cu132). Random draws
   (initial weights, mini-batches, interpolation weights, permutations, splits) stay on the CPU
   generators, so seeds and streams are those of v1; GPU arithmetic is not bit-identical to CPU
   arithmetic and the trained critics differ slightly. Exactness of the permutation p-values does
   not depend on the trained critic.
3. **Replications: the DESIGN section 4 values** (the compute cuts of 2026-11-15 are undone):
   E1 100 real-data splits; E2 part e at n in {500, 2000}; E2-dep n in {500, 2000} with 200 / 100
   replications; E3 10 synthetic seeds, 200 replications of each power design (D1, D2); E4 10
   test seeds and 200 placebo replications; E5 10 seeds, power 200 replications for the default
   and 50 for other configurations, split power 50, and the parts gridsize, gridpower and
   gridrealvol that v1 did not run.
4. **Outputs** go to `results_v2/` (tables and figures in `results_v2/tables`,
   `results_v2/figures`); v1 outputs in `results/` and `paper/tables` are never overwritten.

## What does not change

Data, panel, TRAIN/TEST split, window length L = 20 (L = 5 for per-ticker and E4 tests),
normalisation, fitted model specifications, the decision rule (BAD if the anchored
`min(1, 2 x median p)` over the synthetic seeds is at most 0.05; GOOD if not BAD and the rule
power is at least 0.8 under both D1 and D2; INCONCLUSIVE otherwise; INVALID for the bootstrap),
alpha = 0.05, all inference procedures (time-block split with 10 blocks, Driscoll-Kraay HAC with
the Andrews AR(1) lag, wild bootstrap for kernel tests), all seeds.

## Planned additions (each written here before it is run)

- Neural generators (QuantGAN-style TCN generator trained with WGAN-GP on pooled normalised TRAIN
  returns, and one further neural generator) evaluated with the E3 protocol as additional models.
  Specification to be added below before their first evaluation on TEST data.

### Neural generator 1: QuantGAN (fixed 2026-11-16, before its first evaluation on TEST data)

`synthval.models.neural.QuantGAN`, the "pure TCN" generator of Wiese, Knobloch, Korn and
Kretschmer (2020). Generator and discriminator are TCNs with skip connections: 7 temporal blocks
of two unpadded dilated causal convolutions with PReLUs, 80 channels, kernel size 1 then 2,
dilations 1, 1, 2, 4, 8, 16, 32, residual shortcuts, 1x1 output convolution; receptive field 127
days; 3-dimensional i.i.d. N(0, I) noise per day.

*Training data:* the pooled TRAIN log-returns of the 100 E3 panel tickers, each standardised by
its own TRAIN mean and std (ddof 1). One heavy-tail Lambert W x Gaussian transform is fitted by
maximum likelihood on the pooled standardised returns and inverted, and the result is
standardised again. The discriminator sees rolling 127-day windows (stride 1, within tickers,
sampled uniformly).

*Training:* WGAN-GP (gradient penalty 10, 5 critic steps per generator step, Adam lr 1e-4, betas
(0, 0.9), batch 64), 10,000 generator steps, fit seed `seed_int("fit", "QuantGAN")` of
`experiments/e6_neural.py`. The generator after the last step is used: no checkpoint selection,
no tuning on TEST data or on test results. TRAIN diagnostics are recorded only. If training yields
non-finite simulated returns, it is retrained once with the next fit seed, and this is reported.
The model is trained once; any other retraining after the first TEST evaluation is reported as a
deviation.

*Simulation:* a ticker's 615-day TEST path is the generator applied to 615 + 126 noise vectors
from that ticker's child generator in `pipeline.simulate_panel`, mapped back through the inverse
transforms and rescaled by the ticker's TRAIN std and mean. The path is stationary, with no
conditioning on the end of TRAIN.

*Evaluation:* `experiments/e6_neural.py`, E3's pooled protocol with E3's synthetic seeds and test
settings (absolute W-critic, anchored relative test, time blocks + HAC) and E3's decision rule
with E3's v2 power designs D1/D2.

*Deviations from the paper:*
- training pooled over the panel;
- WGAN-GP instead of the original GAN loss with the regulariser of Mescheder et al. (2018);
- no checkpoint selection;
- residual shortcuts;
- learning rate, batch size and step count as above (the paper does not report them).

*Role:* a test subject of the method, like GBM, not a contribution of the paper.

## Order of runs

E1 (sim) -> E2 (main, e, f, g) -> E2-dep -> E3 (all parts) -> E1 (real, nonstat) -> E5 (all
parts) -> E4 (all parts). Command for each:
`PYTHONPATH=src uv run --project envs/gpu python experiments/v2.py <exp> --part ... --workers N`.

## Change log

- 2026-11-16: protocol fixed.
- 2026-11-16: QuantGAN specification added (above) before any training or TEST evaluation;
  E6 (`experiments/e6_neural.py`) runs after E3's v2 power designs. Run-time settings:
  `run_v2.sh` uses 4 GPU workers (3 for the memory-heavy E1 and E5 split parts) to keep at
  least 2 GB of RAM free.
- 2026-11-16: `run_v2.sh` chooses the worker count of each part from available memory (keep
  2.6 GB free; 1400 MB per worker, 2200 MB for E1 and the E5 split). E1 `sim`: the 54 blocks
  computed between 14:34 and 16:14 were written under a library fingerprint that differs from
  the current one only by the additive QuantGAN code (new `models/neural.py`, `PANEL_MODELS`
  registry, a new branch of `pipeline.fit_panel` used only by panel models); none of it is
  called by E1, so these blocks were copied into the current checkpoint directory instead of
  being recomputed.
- 2026-11-16: computational change only (no change of method): the power designs (E3 `power`,
  `realvol`; E5 `split` power, `gridpower`, `gridrealvol`) train the 2 x n_seeds anchored
  critics of a replication as one bank of independent critics in one process
  (`synthval.batched`, `SYNTHVAL_BATCHED=1`, commit b24d9bc). Each critic keeps the initial
  weights and the mini-batch / interpolation draws of the sequential code (verified exactly);
  only floating-point rounding differs (grouped convolutions), which training amplifies as it
  does between thread counts or devices, so individual p-values are not reproducible across
  implementations but their distribution is (exactness check: uniform permutation p-values at
  the boundary). GPU benchmark (RTX 5050, E3 sizes, 1000 steps): 0.79 tests/s for one batched
  process (~2.2 GB host RAM) against 0.24 for one sequential process. E1 `sim` (finished
  17:08) is not rerun although E1's library fingerprint changed with the new package.
- 2026-11-17, amendment A1 (written after the v2 E3/E6 verdicts were known, before the
  experiment below ran): **certified tolerance**. The v2 verdicts are BAD for GBM and
  INCONCLUSIVE for Merton, GARCH-t, Heston and QuantGAN, because the rule's power against the
  pre-specified deviation (volatility x 1.2) is 1.00 under D1 but 0.41 under D2. To report what
  the data can resolve instead of a bare INCONCLUSIVE, E7 (`experiments/e7_tolerance.py`)
  measures the rule's power for other deviation sizes with E3's own replication functions,
  seeds and settings (only the volatility factor changes; alternative case only; 100
  replications per factor, paired with E3's): D2 at factors 1.3, 1.5, 2.0, 1/1.2, 1/1.3, 1/1.5,
  1/2.0 and D1 at 1/1.2, 1/1.3. Decision rule, fixed now: a relative volatility error t in
  {0.2, 0.3, 0.5, 1.0} is *resolved* if the rule's power is at least 0.8 under D1 and D2 for
  both 1 + t and 1/(1 + t) (D1 at upward factors >= 1.2 is E3's value at 1.2). A generator that
  the anchored rule does not reject is reported as GOOD at the smallest resolved t ("not
  detectably worse than the real past, within a volatility error of t"); BAD is unchanged.
  This uses no real-data quantity beyond the anchored rule already applied; the tolerance is a
  property of the data and the designs, the same for every non-rejected generator. The
  pre-specified verdicts (tolerance 0.2) are reported unchanged next to it. If more real data
  are obtained (Bloomberg export beyond 2025-06-17), the extended TEST period is analysed as a
  new, blind confirmation sample under the protocol fixed before its first use.
- 2026-11-17 11:41: computational change only. The GPU is saturated by the batched power
  designs (100% busy, ~49 W) while 90% of the CPU is idle, so E4 (all parts) runs on the CPU
  (`SYNTHVAL_DEVICE=cpu`, CPU build of torch 2.15.0, `experiments/run_v2_cpu.sh`) in parallel
  with E7/E5 on the GPU (`experiments/run_v2_gpu.sh`). Every other v2 setting is unchanged
  (pooled critic, seeds, replications); E4's results come from one device only. Both lanes
  run detached from the Claude Code session (`setsid nohup`).
- 2026-11-17 16:40, E7 corrections and precision extension.
  - **Bug fixed:** with one worker, E7's replications ran in the main process and left
    `pipeline.POWER_VOL_FACTOR` at the last cell's value, so the summary labelled E3's x1.2 rows
    with the wrong factor. Only the summary was affected; the raw results are unchanged.
  - **Missing cells:** amendment A1's rule needs D1 at the downward factors 1/1.5 and 1/2.0 for
    tolerances 0.5 and 1.0, which were not scheduled; they were added.
  - **A1 mis-specified for D2 (found, not hidden):** in D2 the new sample is the real TEST
    period, about 20% calmer than TRAIN. A generator made calmer (factor < 1) moves towards
    R_new, so it lies inside the anchored null, and the rule correctly never rejects it at
    factors 1/1.2 and 1/1.3 (0/100). A1 as written requires power there and therefore cannot
    resolve any tolerance below ~0.5. Both readings are reported: A1 as fixed, and a directional
    reading (D2 upward errors only, D1 both directions), labelled post hoc.
  - **Precision (compute is not a constraint):**
    - finer factors where the D2 power jumps (x1.35, x1.4, x1.45 and their inverses);
    - D1 at x1.1, 1/1.1, 1/1.35, 1/1.4, 1/1.45;
    - 400 replications on the cells that decide the tolerance, 100 elsewhere;
    - tolerance grid {0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0}.
  - **E1 `sim` recomputed** under the current code, in one run, instead of the blocks carried
    over across library fingerprints. The carried-over run stays for comparison.
  - **Order:** `experiments/run_v2_precision.sh` runs E7, then E1 sim, then the E5 grid power
    designs (resumed from checkpoints), then every table.
- 2026-11-17 ~17:00, **amendment A2 (50 synthetic seeds), fixed before any 50-seed run.**
  - **Why:** the verdict combines the anchored p-values of the synthetic seeds as
    min(1, 2 x median), and with 10 seeds it still depends on synthetic randomness. Merton's
    anchored statistic ranges from -0.34 to 0.83 across seeds; Merton was BAD with 5 seeds and
    INCONCLUSIVE with 10.
  - **Change:** the number of synthetic seeds of E3, E6 and E7 goes from 10 to 50
    (`SYNTHVAL_N_SEEDS=50`). Seeds 0-9 are those of the 10-seed run. The rule is re-estimated
    with the same 50 seeds:
    - E3 power designs D1/D2, 200 replications;
    - E7 tolerance cells D2 x1.3, 1.35, 1.4, 1.45, 1.5, 2.0 and D1 x1.1, 1/1.1, 200 replications
      each (`SYNTHVAL_E7_GRID=s50`).
  - **Tolerance:** the directional reading (D2 upward errors, D1 both directions). In D1 the
    power at larger deviations is bounded below by the nearest smaller measured deviation in
    the same direction.
  - **Unchanged:** everything else, including the decision rule and the models. E3's fitted
    models and the trained QuantGAN are copied, not refitted, so both seed counts use identical
    models (QuantGAN `fitted.pkl` sha256 81b57c23a661...).
  - **Outputs:** `results_v2_s50/`. The 10-seed results in `results_v2/` stay as they are, and
    both are reported.
  - **E5:** stays at 10 seeds (robustness grid).
  - **Order:** after the running 10-seed E7, the 50-seed E3 (fit from the copies, power
    designs, pooled), E6 and E7 run; then E1 sim and the E5 grid as planned.
