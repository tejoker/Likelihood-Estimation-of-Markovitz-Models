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

## Order of runs

E1 (sim) -> E2 (main, e, f, g) -> E2-dep -> E3 (all parts) -> E1 (real, nonstat) -> E5 (all
parts) -> E4 (all parts). Command for each:
`PYTHONPATH=src uv run --project envs/gpu python experiments/v2.py <exp> --part ... --workers N`.

## Change log

- 2026-11-16: protocol fixed.
