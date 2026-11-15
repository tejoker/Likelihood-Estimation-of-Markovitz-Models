# synthval: is synthetic financial data good enough?

Code for the paper *Is Synthetic Financial Data Good Enough? A Wasserstein-Critic Two-Sample
Test, Likelihood Benchmarks for Markov Models, and Markowitz Portfolio Utility* (working title;
the repository keeps its historical name).

The paper turns the critic of a Wasserstein GAN, trained alone with no generator, into a formal
two-sample test of synthetic return data against real data:

- **Independent windows**: held-out permutation p-value, exact finite-sample level.
- **Dependent windows** (real data, long simulated paths): contiguous time-block train/eval
  split and a Driscoll-Kraay HAC p-value. MMD and signature MMD use the wild bootstrap here.
- **Anchored relative test** (the primary yes/no criterion): is the synthetic data at least as
  close to future real data as the real past is?
- **Decision rule**: BAD, GOOD or INCONCLUSIVE. GOOD also requires a pre-specified power check.

Size and power are validated against ground truth and against the Neyman-Pearson
likelihood-ratio oracle for Markov models where the likelihood is tractable (GBM, Merton
jump-diffusion, GARCH(1,1)-t). The test is compared with MMD, signature MMD, classifier
two-sample tests and KS. It is then applied to S&P 500 stocks (TRAIN 2018-2022, TEST
2023-01-03 to 2025-06-16) with:

- likelihood benchmarks (predictive log-likelihood, Berkowitz/PIT, Amisano-Giacomini);
- models whose likelihood is intractable (Heston, stationary bootstrap);
- Markowitz portfolios trained on synthetic data and tested on real data.

The binding specification is [`docs/DESIGN.md`](docs/DESIGN.md): interfaces, the inference
protocol, the experiments and a change log of every decision taken after review. The
statistical theory is in [`docs/theory.md`](docs/theory.md) and the literature map in
[`docs/literature.md`](docs/literature.md).

## Install

You need [uv](https://docs.astral.sh/uv/) (tested with uv 0.13.7) on Linux x86-64. uv installs
the pinned Python (>= 3.12) and every dependency from `uv.lock` into a project-local `.venv`.
Torch is the CPU build, taken from the PyTorch CPU index declared in `pyproject.toml`.

```bash
uv sync                  # runtime + dev dependencies (pytest, arch, scikit-learn)
```

Always run code with `uv run ...`; nothing is installed system-wide. The data file
`data/sp500_price_volume.csv` (Bloomberg export: PX_LAST and PX_VOLUME, 503 current S&P 500
constituents, 2018-01-02 to 2025-06-17) is in the repository. DESIGN section 2 gives the
cleaning rules: no back-filling, two anomalous tickers excluded, a 479-ticker complete universe,
and a seeded 100-ticker main panel.

## Tests

```bash
OMP_NUM_THREADS=1 uv run pytest -m "not slow"     # 298 tests, about 5 min on one core
uv run pytest -m slow                             # 19 slow statistical checks (size, recovery, real data)
uv run pytest tests/test_relative.py              # one module
```

The fast suite covers data cleaning, every model (fit, simulate, likelihood, cross-checks
against `arch`), every two-sample test (size and power at small n, HAC, wild bootstrap,
time-block split), the anchored and two-critic relative tests, the likelihood tools, portfolio
code and the shared real-data pipeline. Real-data tests skip when the CSV is missing.

## Experiments

Five scripts in `experiments/`, one per experiment of DESIGN section 4:

| script | question |
|---|---|
| `e1_placebo.py` | size of the real-data pipeline under H0 (factor-panel placebo, ticker-half placebo, R_past vs R_new) |
| `e2_ground_truth.py` | size and power against known laws and the likelihood-ratio oracle; relative-test boundary; learned vs true log-likelihood ratio; dependent windows |
| `e3_sp500.py` | S&P 500 univariate models: absolute and anchored relative tests, power check, verdicts, per-ticker tests, stylized facts, likelihood benchmarks |
| `e4_markowitz.py` | Markowitz utility of multivariate generators and its link with test statistics |
| `e5_robustness.py` | sensitivity of the E3 verdicts to tuning choices and to the TRAIN/TEST split |

```bash
uv run python experiments/e3_sp500.py --quick          # smoke test: results/e3_quick/ only
uv run python experiments/e3_sp500.py --part fit       # one part, full settings
uv run python experiments/e3_sp500.py --plots-only     # rebuild tables and figures from CSVs
```

Common options: `--quick`, `--part NAME` (repeatable), `--workers N` (default from the
`SYNTHVAL_WORKERS` environment variable, else 4), `--fresh` and `--plots-only`. Full runs write:

- results to `results/<exp>/*.csv` with `meta*.json`;
- figures to `paper/figures/` and LaTeX tables to `paper/tables/`;
- checkpoints to `results/<exp>/checkpoints/`, so an interrupted run resumes. Checkpoints are
  keyed by the configuration and, for most parts, the code version (see
  `experiments/README.md`).

`--quick` writes only to `results/<exp>_quick/` and prints the estimated full-run time.
[`experiments/README.md`](experiments/README.md) gives, for each part: what it does, the exact
full-run command, measured times, memory needs and a schedule for a 16-core machine with about
5 GB of free RAM.

## Layout

```
docs/          DESIGN.md (contract + change log), theory.md, literature.md
data/          sp500_price_volume.csv
src/synthval/  the library
  config.py      paths, TRAIN/TEST dates, panel seed, defaults
  data.py        loading, cleaning, universe, returns, windows, normalisation
  models/        univariate (GBM, Merton, GARCHt, Heston, StationaryBootstrap) and multivariate
                 (MVGaussian, MVStudentT, CCCGARCHt, MVStationaryBootstrap, FactorGARCHt) models;
                 get_model / get_mv_model registries
  twosample/     critic (W-critic), c2st (classifier and LR-critic), mmd, signature (sig-MMD),
                 ks, relative (anchored, two-critic, Bounliphone MMD), nets, base
  likelihood.py  predictive log-likelihood, PIT, Berkowitz, Vuong / Amisano-Giacomini, LR oracle
  pipeline.py    shared real-data pipeline of E1/E3/E4/E5: panel windows, fits, simulation,
                 dependent-window protocol, power check, decision rule
  portfolio.py   moment estimators (sample, Ledoit-Wolf), Markowitz portfolios, evaluation
  stylized.py    stylized-fact statistics
  plotting.py    one figure style for the paper
  utils.py       seeding, parallel_map, saving results with metadata, LaTeX tables
tests/         pytest suite (-m slow marks the long statistical checks)
experiments/   e1..e5 scripts, common.py (CLI, keyed seeds, checkpoints), README.md
results/       experiment outputs (<exp>/ full runs, <exp>_quick/ smoke tests)
paper/         LaTeX sources; figures/ and tables/ are generated by the experiments
legacy/        the original code of the project (see below)
```

## legacy/

`legacy/` holds the original code of this repository: a GAN and option-pricing scripts
(Black-Scholes, binomial CRR, Carr-Madan, finite differences, Heston Monte Carlo, Merton) and
their result CSVs. It is kept for reference only. Nothing in `src/`, `tests/` or `experiments/`
imports it, and its conventions are not used: for example, it mixes a daily sigma with
dt = 1/252, while synthval uses daily log-returns with dt = 1 everywhere.
