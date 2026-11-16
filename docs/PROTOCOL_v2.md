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
