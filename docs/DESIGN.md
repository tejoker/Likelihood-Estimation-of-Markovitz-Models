# synthval: design contract

This is the contract every module and experiment builds against. If you need to deviate from an
interface here, keep the documented signature working and note the change at the bottom of this
file under "Change log", with a one-line reason.

## 1. The paper in one paragraph

**Working title:** *Is Synthetic Financial Data Good Enough? A Wasserstein-Critic Two-Sample Test,
Likelihood Benchmarks for Markov Models, and Markowitz Portfolio Utility.*

Synthetic return data is used for back-testing, data sharing and portfolio construction, but
"is this synthetic data good?" is usually answered with stylized facts, uncalibrated
discriminator scores, or formal tests whose size and power are unchecked. We turn the
discriminator of a Wasserstein GAN (the critic, trained alone, with no generator) into a formal
two-sample test: exact finite-sample level for independent windows (held-out permutation), and
Driscoll-Kraay HAC inference with a time-block train/eval split for serially and
cross-sectionally dependent real data. We add an *anchored relative* test that asks whether
synthetic data is at least as close to future real data as the real past is (markets are
non-stationary, so "identical to the future" is the wrong bar). We validate size and power against ground truth, and against the
Neyman-Pearson likelihood-ratio oracle where the likelihood is tractable (Markov models: GBM,
Merton jump-diffusion, GARCH(1,1)-t). We compare with MMD, signature MMD, classifier two-sample
tests (C2ST) and KS. On S&P 500 stocks (2018-2025) we compare critic verdicts with
likelihood-based evaluation (out-of-sample predictive log-likelihood, Berkowitz PIT test, Vuong
test), extend to models whose likelihood is intractable (Heston, stationary bootstrap), and test
whether the verdicts predict downstream usefulness for Markowitz mean-variance portfolios
(train on synthetic, test on real).

### Research questions
- **RQ1 (validity).** Does the critic test control its level under H0, in simulation and in
  real-data placebo comparisons?
- **RQ2 (power).** How close is its power to the likelihood-ratio oracle, and how does it compare
  with MMD, signature MMD, C2ST and KS, against the deviations that matter in finance (volatility
  level, fat tails, volatility clustering, jumps)?
- **RQ3 (relative goodness).** Is synthetic data at least as close to held-out real data as the
  real past is?
- **RQ4 (likelihood agreement).** Where likelihoods exist, do critic verdicts agree with
  likelihood-based rankings? What does the critic add where they do not exist?
- **RQ5 (utility).** Do verdicts predict Markowitz portfolio performance (TSTR)?

### Decision rule (the yes/no answer the paper delivers)
For a generator G calibrated on the real past R_past and evaluated against held-out real data R_new:
1. **Absolute test** H0: Law(G) = Law(R_new). Critic statistic; on real data time-block split +
   Driscoll-Kraay HAC p-value (permutation p-value only for independent windows).
2. **Anchored relative test** (primary; see §3.7 `method="anchored"`). One critic f is trained to
   separate R_new (positive class) from the balanced pool {G_tr, R_past_tr} (negative class).
   Statistic `Delta = mean f(R_past_ev) - mean f(G_ev)`; H1: Delta > 0 ("G is less like the
   future than the past is"). At the boundary Law(G) = Law(R_past), G_ev and R_past_ev are
   exchangeable given f, so the permutation p-value is exact for independent windows (HAC p-value
   on real data). The older two-critic version `D(G,R_new) - D(R_past,R_new)` is reported as a
   secondary statistic only: it is anti-conservative at that boundary (theory.md Prop 6.8).
3. **Power check (pre-specified, before any real-data run).** Simulate the full E3 relative-test
   pipeline at the real sample sizes and split, with R_past*, R_new* drawn from the per-ticker
   GARCHt fit (TRAIN, TEST lengths, independent of each other) and G* the same model with
   volatility x1.2. Power = rejection rate over >= 50 reps at alpha = 0.05.
4. **Verdict:** `BAD` if the anchored relative test rejects at alpha = 0.05; `GOOD` if it does not
   reject *and* the power check gives power >= 0.8; `INCONCLUSIVE` otherwise. The absolute test is
   reported alongside ("distinguishable from the future but no worse than history" is the typical
   real-world outcome and is a finding, not a failure).

## 2. Data

- File: `data/sp500_price_volume.csv`, Bloomberg export, `;`-separated. Line 1: `Date de début;2018-01-01`,
  line 2: `Date de fin;2025-06-17`, then a 2-level header (ticker, field) with fields `PX_LAST`
  and `PX_VOLUME`; first data row 2018-01-02. 503 tickers, current S&P 500 constituents
  (survivorship bias: must be stated as a limitation).
- Use `PX_LAST` only. **Never back-fill.** A ticker is in the *complete universe* if it has no
  missing `PX_LAST` from 2018-01-02 to 2025-06-17 (481 tickers; the last trading day in the file
  is 2025-06-16). Isolated internal gaps (<= 3 consecutive days) may be forward-filled; anything
  else excludes the ticker. Two data errors are excluded by a rule (`data.DATA_ANOMALIES`:
  max |log-return| > 1 or >= 20 zero-volume days): KDP (unadjusted special dividend, -1.72
  log-return on 2018-07-10) and AMCR (232 days of stale pre-listing quotes). Default universe:
  479 tickers. Report both in the paper's data section.
- Returns: daily log-returns `r_t = log P_t - log P_{t-1}`.
- Split (constants in `synthval.config`):
  - `TRAIN = ("2018-01-02", "2022-12-30")`: calibration ("real past").
  - `TEST  = ("2023-01-03", "2025-06-17")`: evaluation ("real new").
- Main panel: a seeded random subset of `N_PANEL = 100` tickers from the complete universe
  (`config.PANEL_SEED = 20251114`). The full universe is optional for robustness.
- Per-ticker scale normalisation for pooling: divide real *and* synthetic returns of ticker i by
  the standard deviation of ticker i's TRAIN returns (a fixed constant, same for both samples).
- Windows: non-overlapping by default (`make_windows(..., L, step=L)`), default `L = 20`.

## 3. Package layout (`src/synthval/`)

```
config.py        paths, dates, seeds, defaults (L, alpha, n_perm, workers)
data.py          loading, cleaning, returns, splits, windows, normalisation
stylized.py      stylized-fact statistics
models/
  __init__.py    MODEL_REGISTRY: dict[str, type] for univariate, MV_MODEL_REGISTRY for multivariate
  base.py        ReturnModel and MultiReturnModel base classes
  univariate.py  GBM, Merton, GARCHt, Heston, StationaryBootstrap
  multivariate.py MVGaussian, MVStudentT, CCCGARCHt, MVStationaryBootstrap
twosample/
  __init__.py    TEST_REGISTRY
  base.py        TestResult dataclass, permutation helpers
  nets.py        critic / classifier architectures
  critic.py      Wasserstein-critic test (the paper's method)
  c2st.py        classifier two-sample test
  mmd.py         Gaussian-kernel MMD test
  signature.py   truncated path signatures + signature-MMD test
  ks.py          KS test on pooled returns (baseline)
  relative.py    relative-similarity test
likelihood.py    predictive log-lik, PIT, Berkowitz test, Vuong test, LR oracle helpers
portfolio.py     moment estimators, Markowitz portfolios, out-of-sample evaluation
plotting.py      one shared matplotlib style for every paper figure
utils.py         seeding, parallel map, saving results with metadata
```

### 3.1 Conventions (all modules)
- Randomness: every stochastic function takes `rng: np.random.Generator` (keyword). No global
  `np.random.*` calls. Torch: seed with `torch.manual_seed(int(rng.integers(2**31 - 1)))` inside the
  function; call `torch.set_num_threads(1)` in worker processes (utils.parallel_map does this).
- Arrays: returns are `float64` numpy. Univariate series shape `(T,)`; paths `(n_paths, n_steps)`;
  multivariate `(T, N)`; window samples `(n, L)` univariate or `(n, N, L)` multivariate.
- Units: **daily** log-returns everywhere, real-world (P) measure. No annualisation inside models;
  `dt = 1` trading day. (The legacy code mixed daily sigma with dt = 1/252; never do that.)
- Type hints and numpy-style docstrings on public functions. No prints in library code (use
  `logging`).
- Dependencies available: numpy, pandas, scipy, statsmodels, matplotlib, torch (CPU), tabulate;
  dev: pytest, arch, scikit-learn (for cross-checks in tests only, not in library code, except
  where noted). Do **not** run `uv add`; if you need a package, say so in your report.
- Run everything with `uv run ...` (e.g. `uv run pytest tests/test_models_univariate.py`).

### 3.2 `config.py`
```python
ROOT: Path; DATA_CSV: Path; RESULTS_DIR: Path; FIG_DIR = ROOT/"paper/figures"; TAB_DIR = ROOT/"paper/tables"
TRAIN = ("2018-01-02", "2022-12-30"); TEST = ("2023-01-03", "2025-06-17")
N_PANEL = 100; PANEL_SEED = 20251114; WINDOW_L = 20; ALPHA = 0.05; N_PERM = 1000
def n_workers() -> int  # env SYNTHVAL_WORKERS, default 4
```

### 3.3 `data.py`
```python
def load_prices(path=config.DATA_CSV) -> pd.DataFrame            # Date index x ticker, PX_LAST, raw (NaN kept)
def complete_universe(prices, start, end, max_gap=3) -> list[str]
def clean_prices(prices, tickers, max_gap=3) -> pd.DataFrame       # ffill only internal gaps <= max_gap
def log_returns(prices) -> pd.DataFrame
def split(returns, period: tuple[str, str]) -> pd.DataFrame
def panel_tickers(n=config.N_PANEL, seed=config.PANEL_SEED) -> list[str]
def make_windows(x: np.ndarray, L: int, step: int | None = None) -> np.ndarray
    # x (T,) -> (n, L); x (T, N) -> (n, N, L); step defaults to L (non-overlapping)
def load_panel(tickers=None) -> tuple[pd.DataFrame, pd.DataFrame]   # (train_returns, test_returns)
```

### 3.4 `models/base.py`
```python
class ReturnModel:                      # univariate, daily log-returns
    name: str
    has_likelihood: bool                # True for GBM, Merton, GARCHt
    def fit(self, r: np.ndarray) -> "ReturnModel"
    def simulate(self, n_steps: int, n_paths: int, *, rng, burn_in: int = 250) -> np.ndarray  # (n_paths, n_steps) returns
    def loglik(self, r: np.ndarray) -> float              # total conditional log-lik; NotImplementedError if intractable
    def loglik_obs(self, r: np.ndarray) -> np.ndarray     # per-observation conditional log-densities (T,)
    def pit(self, r: np.ndarray) -> np.ndarray            # conditional CDF values u_t in (0,1); NotImplementedError if intractable
    def params(self) -> dict[str, float]
    def n_params(self) -> int
class MultiReturnModel:
    name: str
    def fit(self, R: np.ndarray) -> "MultiReturnModel"   # R (T, N)
    def simulate(self, n_steps: int, *, rng, burn_in: int = 250) -> np.ndarray  # (n_steps, N)
    def params(self) -> dict
```
For filtered models (GARCH) `loglik_obs`/`pit` on a *test* series must condition on the end
state of the training series: provide `loglik_obs(r, r_history=None)` / `pit(r, r_history=None)`
where `r_history` (the training returns) initialises the filter. `simulate(..., start_from_end=True)`
should start from the fitted end state when meaningful (GARCH, Heston), else from stationarity.

### 3.5 Univariate models (`models/univariate.py`)
| class | model (daily) | fit | likelihood |
|---|---|---|---|
| `GBM` | r_t iid N(mu, sigma^2) | closed-form MLE | exact |
| `Merton` | r_t = mu + sigma*e_t + sum_{k<=N_t} J_k, N_t~Poisson(lam), J~N(muJ, sJ^2) | MLE, truncated Poisson sum (n<=10 or until mass < 1e-12), multistart, bounds | exact (mixture) |
| `GARCHt` | r_t = mu + s_t z_t, s_t^2 = w + a eps_{t-1}^2 + b s_{t-1}^2, z ~ standardised Student-t(nu) | MLE (scipy, constraints a+b<1, nu>2.05), multistart; tests cross-check vs `arch`. Experiments use `variance_targeting=True` (w = (1-a-b) x sample variance; Engle-Mezrich 1996) with persistence cap 0.995, to avoid explosive simulated paths at near-IGARCH fits (RCL) | exact (filtered) |
| `Heston` | Daily step with Andersen (2008) QE scheme for v (no zero-variance sticking) and the matching log-return step: r_t = mu - (integrated var)/2 + ...; corr(z1,z2)=rho | simulated method of moments on: var, a robust tail moment (log-kurtosis or quantile ratio), ACF(r^2) lags 1,5,10,20, corr(r_t, r_{t+1}^2); common random numbers (QE driven by fixed normals/uniforms); Nelder-Mead with bounds | intractable |
| `StationaryBootstrap` | Politis-Romano (1994) on training returns, mean block length via Politis-White (2004) rule or default 20 | none | intractable |

### 3.6 Multivariate models (`models/multivariate.py`)
`MVGaussian` (iid, sample mean/cov), `MVStudentT` (iid, ECME/EM fit of df, location, scatter),
`CCCGARCHt` (GARCHt margins + constant correlation of standardised residuals, Gaussian copula on
innovations scaled to t margins, or multivariate-t innovations; document the choice),
`MVStationaryBootstrap` (same blocks for all assets, preserves cross-sectional dependence).

### 3.7 Two-sample tests (`twosample/`)
```python
@dataclass
class TestResult:
    name: str; statistic: float; pvalue: float; n_x: int; n_y: int
    extra: dict          # e.g. critic train curve, accuracy, bandwidth, eval-split sizes
class TwoSampleTest:     # every test implements this
    name: str
    def __call__(self, X, Y, *, rng, groups_x=None, groups_y=None, order_x=None, order_y=None) -> TestResult
```
- X = real sample, Y = synthetic sample, shapes `(n, L)` or `(n, N, L)` (or `(n, d)` features).
- `groups_x/groups_y` (optional int arrays): group labels (e.g. ticker ids). When given, the
  train/eval split for learned tests is done **by group** (no group in both halves).
- **CriticTest** (`critic.py`): split X and Y into train/eval halves (50/50 default). Train a
  critic f (WGAN-GP: maximise mean f(X_tr) - mean f(Y_tr) - lambda * GP, GP on interpolates
  between X_tr and Y_tr, lambda=10, Adam lr=1e-4, betas=(0.5, 0.9), batch 128, `n_steps` default
  1000 with optional early stopping on a held-out slice of the training half). Statistic
  `T = mean f(X_ev) - mean f(Y_ev)` (an estimate of a lower bound on W1 on the representation,
  given approx. 1-Lipschitz f). p-value: permutation of the pooled eval labels with f fixed
  (`n_perm` default 1000, one-sided, `(1 + #{T_b >= T}) / (1 + n_perm)`). Also return
  `extra["z"]` (asymptotic z from the difference of means) and `extra["W1_lower"] = T`.
  Architectures (`nets.py`): `MLPCritic(in_dim, hidden=128, depth=3)` and
  `ConvCritic(in_channels, L, hidden=64)` (1D CNN, LayerNorm/no BatchNorm, LeakyReLU); choose by
  input rank. Inputs are standardised with statistics of the *training half pooled* (same
  transform for both samples).
- **C2ST** (`c2st.py`): same split and architectures, BCE classifier; `statistic="accuracy"`
  (default; eval accuracy) or `statistic="logit"` (the **likelihood-ratio critic**: mean logit on
  real eval minus mean logit on synthetic eval; the optimal discriminator's logit is
  log p_real/p_synth (Goodfellow et al. 2014), so this is a learned log-likelihood-ratio
  statistic, cf. Cheng & Cloninger 2022). p-value by permutation of eval labels with the
  classifier fixed (or exact binomial for accuracy when n_perm=0). Expose
  `C2ST.fit_logit(X_tr, Y_tr) -> callable` so experiments can compare the learned logit with the
  true log-likelihood ratio in ground-truth studies. Lopez-Paz & Oquab (2017).
- **MMD** (`mmd.py`): unbiased MMD^2 with Gaussian kernel, bandwidth = median heuristic on the
  pooled sample (option: sum over 5 bandwidths x {0.25,0.5,1,2,4} median); permutation p-value
  (n_perm default 500). Subsample to at most `max_n=3000` per sample for memory. Gretton et al. (2012).
- **SigMMD** (`signature.py`): `signature(paths, depth)` for piecewise-linear paths via Chen's
  identity (vectorised numpy), input window of returns -> time-augmented path (t/L, cumulative
  return [, lead-lag optional]); depth 4 default; features standardised; MMD on signature
  features (Gaussian kernel on features, median heuristic) with permutation p-value.
  Chevyrev & Oberhauser (2022).
- **KS** (`ks.py`): two-sample KS on pooled flattened returns (baseline; note it ignores
  dependence and only sees the one-dimensional marginal).
- **RelativeTest** (`relative.py`): `relative_test(R_new, G, R_past, base: TwoSampleTest-like
  statistic, *, rng, n_boot=500)` returns `TestResult` with statistic
  `Delta = D(G, R_new) - D(R_past, R_new)`, one-sided p-value for H1: Delta > 0, and
  `extra["ci"]` (bootstrap 95% CI). For the critic version, train one critic per pair on the
  train halves and bootstrap the eval-set difference of means jointly (paired resampling of the
  shared R_new eval windows). For the MMD version follow Bounliphone et al. (2016).
- `TEST_REGISTRY = {"critic": CriticTest, "lrcritic": partial(C2ST, statistic="logit"), "c2st": C2ST, "mmd": MMDTest, "sigmmd": SigMMDTest, "ks": KSTest}`.
- Every learned test must also report `extra["z_hac"]`: a studentised statistic using a
  Driscoll-Kraay / Newey-West variance over eval windows ordered in time (Bartlett kernel,
  Andrews 1991 AR(1) plug-in bandwidth; justified by a CLT under mixing, Driscoll & Kraay 1998),
  plus its normal p-value `extra["p_hac"]`, when the caller passes `order_x`/`order_y`
  (time indices in window units or date ranks, not trading days); otherwise the i.i.d. z.

**Inference protocol (binding for experiments, after review).**
- *Independent windows* (E2: each window simulated from its own path with burn-in): permutation
  p-values, exact level. Default `inference="permutation"`.
- *Dependent windows* (E1, E3, E4, E5 on real data or on simulated long paths): learned tests use
  `time_blocks=10, inference="hac"` (contiguous time-block train/eval split from `order_*`, so
  train and eval cover disjoint dates; a ticker split shares dates and biases T, theory.md
  Prop 7.5); MMD / SigMMD use `inference="wild"` (Chwialkowski-Sejdinovic-Gretton 2014); KS is
  reported but marked invalid under dependence. Always pass `order_*` = date rank of each
  window's first day. Robust p-values are asymptotic and still somewhat oversized under strong
  dependence (pilot: 6-10% at nominal 5%), so E1 measures their real size with the exact E3
  pipeline and the paper reports it.
- Critic default `lr=1e-3` (changed from 1e-4 before any E2 run: under-training lost power).
  C2ST accuracy p-value uses randomized tie-breaking.

**Anchored relative test** (`relative_test(..., method="anchored")`, the primary relative test).
Inputs R_new, G, R_past (+ optional order/groups). Split G and R_past into train/eval (time
blocks when `order_*` given, else random halves). R_new is used only for training (all of it).
Train one learned critic (same architecture/optimiser as `CriticTest`, or C2ST logit) to separate
R_new (label 1) from the pool {G_tr, R_past_tr} with equal total weight on G_tr and R_past_tr.
Statistic `Delta = mean f(R_past_ev) - mean f(G_ev)`, one-sided H1: Delta > 0. p-value: permutation
of the G_ev / R_past_ev labels with f fixed (exact at Law(G) = Law(R_past) for independent
windows), and `extra["p_hac"]` (HAC/DK z, using `order_*`) for dependent windows; with
`inference="hac"` the reported p-value is `p_hac`. Also report `extra["ci"]` (normal CI with the
HAC se) and the two absolute statistics f-based means for diagnostics. `method="two_critic"`
keeps the existing implementation (secondary, documented as anti-conservative at the boundary).

### 3.8 `likelihood.py`
```python
def predictive_loglik(model, r_test, r_train) -> float            # mean per-obs out-of-sample log-lik
def berkowitz_test(u: np.ndarray) -> TestResult                    # PIT -> Phi^{-1}, LR test of N(0,1) iid vs AR(1) Gaussian (Berkowitz 2001)
def pit_ks_test(u) -> TestResult                                   # KS of PIT vs U(0,1) (Diebold, Gunther, Tay 1998)
def vuong_test(ll_a: np.ndarray, ll_b: np.ndarray, k_a=0, k_b=0) -> TestResult   # non-nested LR (Vuong 1989), HAC variance optional
def window_loglik(model, windows: np.ndarray) -> np.ndarray        # log-density of each window under a model (for LR oracle)
def lr_oracle_test(X, Y_model_true, model_null, model_alt, *, rng) -> TestResult
    # Neyman-Pearson oracle for ground-truth studies: statistic = sum log p_alt(x)/p_null(x) on X,
    # null distribution from samples simulated under the null model (Monte Carlo critical value)
```

### 3.9 `stylized.py`
`summary(r) -> dict`: mean, std, skew, excess kurtosis, ACF(r) lag 1..5, ACF(|r|) and ACF(r^2) at
lags 1, 5, 10, 20, 50, leverage corr(r_t, r_{t+k}^2) k=1..10 (summarised), Hill tail index (5% tails),
kurtosis of aggregated returns at horizons 1, 5, 20 (aggregational Gaussianity), max drawdown of
cumulative path. `acf(x, nlags)`. Cont (2001).

### 3.10 `portfolio.py`
```python
def sample_moments(R) -> (mu, Sigma); def ledoit_wolf(R) -> Sigma  (own implementation, LW 2004)
def gmv_weights(Sigma, long_only=False) -> w
def mean_variance_weights(mu, Sigma, gamma, long_only=True) -> w   # max w'mu - gamma/2 w'Sigma w, sum w = 1
def tangency_weights(mu, Sigma, long_only=True) -> w
def equal_weights(N) -> w
def evaluate(w, R_test) -> dict   # mean, vol, Sharpe (daily->annualised x sqrt(252) only here), max drawdown, CE utility
```
Long-only optimisation via `scipy.optimize.minimize(method="SLSQP")` with analytic gradients.

### 3.11 `utils.py` and `plotting.py`
- `seed_everything(seed)`, `spawn_rngs(seed, n) -> list[Generator]` (SeedSequence.spawn),
  `parallel_map(fn, items, n_workers=None)` (ProcessPoolExecutor, sets torch threads = 1,
  falls back to serial when n_workers == 1), `save_results(df_or_dict, name, exp, config: dict)`
  writes `results/<exp>/<name>.csv|json` and `results/<exp>/meta.json` (config, git hash, timing,
  package versions).
- `plotting.setup()` sets one style for all figures (serif font matching LaTeX, 9 pt, figure
  widths `ONE_COL = 3.4 in`, `TWO_COL = 7.0 in`, colour-blind-safe palette, one fixed colour per
  test and per model via `COLORS_TEST`, `COLORS_MODEL`). `savefig(fig, name)` writes PDF to
  `paper/figures/`.

## 4. Experiments (`experiments/`)
Each script: `uv run python experiments/eX_name.py [--quick]`. `--quick` finishes in < 5 min
(smoke test); default is the full paper configuration. Seeds fixed. Outputs: `results/eX/*.csv`,
`results/eX/meta.json`, figures `paper/figures/eX_*.pdf`, LaTeX tables `paper/tables/eX_*.tex`
(booktabs, no hand-edited numbers anywhere in the paper). Respect `SYNTHVAL_WORKERS`.

Compute: 16 cores, ~5 GB free RAM. Every script must set `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=
MKL_NUM_THREADS=1` and torch threads = 1 inside workers (`utils.parallel_map` does torch; set the
env vars at the top of the script before importing numpy), parallelise over replications with
`utils.parallel_map` and `SYNTHVAL_WORKERS` (default 4; full runs use 12), and checkpoint
per-replication results to `results/eX/` so an interrupted run resumes. Each script prints its
estimated full-run time after `--quick`.

- **E1 Placebo / size of the real-data pipeline.**
  (a) *Simulated panel placebo* (primary): a one-factor panel calibrated to the real panel
  (market factor m_t = equal-weight panel return, GARCHt; r_it = beta_i m_t + e_it with GARCHt
  idiosyncratic e_it, all fitted on TRAIN; common volatility regimes, which is what makes real
  windows dependent). "Real" and "synthetic" panels are independent draws from it (H0 true).
  Apply *exactly* the E3 pipeline (same sizes: 100 tickers x TEST length, L=20, vol
  normalisation, `time_blocks=10`, HAC / wild inference, anchored relative test with an R_past*
  of TRAIN length from the same model). 200 reps. Report rejection rates at alpha in
  {0.01, 0.05, 0.10} with Wilson 95% CIs for every test and p-value type (permutation, HAC/wild).
  (b) *Real-data placebo*: random halves of the panel tickers within TEST and within TRAIN
  (100 random splits each, same pipeline). Shared dates make this conservative; report it as
  secondary. (c) R_past vs R_new absolute tests (quantifies non-stationarity).
- **E2 Ground-truth size and power** (independent windows: each window its own path with
  burn-in 250, so permutation p-values are exact). "Real" from known P, "synthetic" from Q.
  Families (delta=0 is H0): (a) volatility level x(1+delta), delta in {0, .05, .1, .2, .3};
  (b) tails: Student-t nu=4 vs nu in {4, 6, 10, 30, inf} (variance matched); (c) volatility
  clustering: GARCHt persistence a+b in {0.98, 0.95, 0.90, 0.80, 0} vs P at 0.98, matched
  unconditional variance and nu; (d) jumps: Merton vs variance-matched GBM, mixing fraction
  delta in {0, .25, .5, .75, 1}. n in {500, 2000} windows of L=20; reps 200 at delta=0, 100
  otherwise. Tests: critic, lrcritic, c2st, mmd, sigmmd, ks, and the LR oracle (exact for
  (a),(b),(d); approximate for GARCH (c), say so). Outputs: size table with Wilson CIs, power
  curves (one panel per family, both n). Plus:
  (e) *relative-test boundary size*: P_past = P_new = G (and G = bootstrap-like resample of
  R_past*), anchored vs two-critic, 200 reps, n=500 and 2000; shows the anchored fix.
  (f) *learned vs true log-likelihood ratio*: for GBM-vs-Merton and GARCH(0.98)-vs-GARCH(0.90),
  scatter / Spearman correlation of the lrcritic logit with the exact log p/q on eval windows.
  (g) *dependent-window size*: windows cut from long GARCH paths, permutation vs HAC/time-block
  vs wild p-values, 200 reps, n=500 (documents why the real-data protocol differs).
- **E3 S&P 500 univariate.** Panel = `load_panel()` (100 tickers). Fit each univariate model per
  ticker on TRAIN (GBM, Merton, GARCHt with variance targeting, Heston QE, StationaryBootstrap).
  Simulate the TEST horizon with `start_from_end=True`, one path per ticker for the main test
  (balanced sizes; 3000 windows each side) and per-ticker vol normalisation with TRAIN std.
  Pooled absolute tests per model (critic, lrcritic, c2st, mmd-wild, sigmmd-wild, ks) with the
  dependent-window protocol; anchored relative test per model (R_past = real TRAIN windows,
  `order_*` = date ranks; synthetic order = the simulated date ranks); two-critic relative as
  secondary; repeat over 10 seeds of synthetic paths and report median p and rejection share.
  Power check per the decision rule (§1, item 3). Per-ticker tests at L=5 with BH (HAC). Stylized-facts table (real vs
  models, panel medians and IQR). Likelihood benchmark for GBM/Merton/GARCHt: predictive
  log-lik, Berkowitz and PIT-KS rejection rates, pairwise Amisano-Giacomini (HAC Vuong, k=0)
  tests. Agreement: Kendall tau between critic-statistic ranking and predictive-log-lik ranking
  of the 3 likelihood models, Cohen kappa of per-ticker verdicts (critic vs Berkowitz). Report
  fit diagnostics: Merton/GARCH at-bound tickers, Heston moment match (data vs simulated per
  moment, panel medians). Final verdict table (GOOD/BAD/INCONCLUSIVE).
- **E4 Markowitz utility.** 30 random universes x N=30 assets from the 479-name universe. For
  each MV generator: fit on TRAIN, simulate TRAIN-length synthetic samples (B=20 replicas),
  estimate moments (sample and Ledoit-Wolf), build GMV, long-only mean-variance (gamma in
  {1,5,20}) and 1/N; evaluate on real TEST; compare with the same portfolios estimated on real
  TRAIN. Utility loss = out-of-sample variance ratio and Sharpe/CE gap vs real-estimated
  portfolio. Report `portfolio.average_correlation` real vs each generator and CCC
  near-unit-root counts. Multivariate critic test (windows (N, L=5), dependent-window protocol)
  and MMD (wild) per universe and generator; Spearman correlation of distances / p-values with
  utility loss, universe-level bootstrap CI.
- **E5 Robustness.** Window length L in {5, 10, 20, 60}; critic capacity, n_steps, lr in
  {1e-4, 1e-3}; GP lambda in {1, 10}; time_blocks in {4, 10}; train fraction; n_perm; HAC lag
  rule (Andrews vs NW); alternative split (TRAIN 2018-2021 / TEST 2022-2025-06); sensitivity of
  the E3 verdicts (anchored relative test, critic, per model).

## 5. Paper (`paper/`)
`main.tex` (article class, 11pt, natbib author-year, booktabs, amsmath/amsthm, hyperref,
cleveref), `sections/{abstract,introduction,literature,methodology,theory,simulation,empirical,markowitz,robustness,conclusion,appendix}.tex`,
`references.bib` (only entries verified to exist), `figures/`, `tables/` (generated). English.
Compile with `latexmk -pdf main.tex` from `paper/`.

## Change log
- (agents append here)
- 2026-11-15 (lead, after Workflow-1 reviews):
  - Data: `DATA_ANOMALIES = ("AMCR","KDP")` excluded by rule; universe 479; last trading day 2025-06-16;
    `complete_universe`/`panel_tickers` take `exclude=` (default the anomalies).
  - Tests: critic `lr=1e-3`; HAC bandwidth Andrews AR(1) plug-in; C2ST accuracy p-value randomized;
    `time_blocks`/`inference="hac"` for learned tests and `inference="wild"` for MMD/SigMMD; the
    permutation p-value is exact only for i.i.d. windows (docstrings and paper must say so).
  - Real-data protocol: time-block split (a ticker split shares dates and biases T, theory Prop 7.5).
  - Relative test: new primary `method="anchored"` (exact at Law(G)=Law(R_past)); the two-critic
    version is anti-conservative at that boundary (theory Prop 6.8) and is secondary.
  - Decision rule: power check pre-specified (GARCHt panel, vol x1.2, power >= 0.8).
  - Models: Heston simulated with Andersen QE (Euler full truncation stuck at v=0 on 36-53% of days);
    GARCHt variance targeting + persistence cap 0.995 in experiments; MVStationaryBootstrap uses the
    |r| block rule (median over assets), like the univariate one; CCC correlations attenuated
    (documented limitation; E4 reports average correlations).
  - Likelihood: out-of-sample Vuong with k=0 and HAC = Amisano-Giacomini (2007); GARCH LR oracle is
    approximate (fixed initial variance), not a strict power bound.
  - z_hac citation: CLT under mixing + Driscoll-Kraay/Newey-West, not Chung & Romano (2013).
- 2026-11-15 (lead): C2ST/lrcritic early stopping keeps the first checkpoint (step 50) at n=500 because
  the MLP classifier overfits within ~16 epochs. Pilot (40 reps, alternatives outside the E2 grid:
  GARCH persistence 0.92, nu=8; configs lr 1e-4 / weight decay 1e-2 / no early stopping / min_steps=100)
  found no configuration with higher power at valid size, so the C2ST defaults are kept and the
  behaviour is reported (E2 records best_step) rather than tuned.
- 2026-11-15 (integration; deviations reported by the module and experiment owners, one line each):
  - Heston: vol-of-variance c bounded by sqrt(psi_c) = sqrt(1.5) ~ 1.2247 so QE stays in its quadratic branch (no zero-variance days; c > sqrt(psi_c) re-creates sticking at v = 0); `n_substeps` removed; simulated kurtosis stays below the data (sim/data median 0.37).
  - Heston SMM tail moment = quantile ratio (q99 - q01)/(q75 - q25) (`tail_quantile_ratio`, weight floored: sd capped at 10% of its value); log-kurtosis and MAD ratio dropped; moment list = var, tail_qr, ACF(r^2) lags 1/5/10/20, lev_1.
  - GARCHt: `persist_max` (default 0.995) is a constructor argument next to `variance_targeting=True`; `loglik_windows` integrates over K stationary initial-variance draws, and the E2 family (c) oracle uses that Monte Carlo density (K = 512), replacing the fixed-start approximation noted above.
  - `optimal_block_length` follows arch's Politis-White implementation exactly; GBM and the stationary bootstrap now report `fit_info_` (at_bound, fallback, block-length rule).
  - `relative_test` default `method="anchored"`; MMD/SigMMD bases ignore `method` (always Bounliphone); pool weights are applied by weighted mini-batch sampling, so the 1/2-1/2 G/R_past balance inside early-stopping train/val subsets is approximate; `relative_test_two_critic` alias.
  - New MV model `FactorGARCHt` (GARCHt factor = equal-weight panel return + GARCHt residuals, no likelihood) for E1(a) and as the fifth E4 generator; rolling log-vol correlation 0.32 vs 0.60 real (limitation); mean residual correlation is pinned near -1/(N-1), so rms and top-eigenvalue residual diagnostics are reported instead.
  - New module `synthval.pipeline` (not in section 3): shared real-data pipeline of E1/E3/E4/E5 (panel windows with TRAIN/TEST date ranks, fit_panel, simulate_panel, dependent/iid protocols, power_check, verdict, median_pvalue).
  - E1(a): R_past* and R_new* are one contiguous stationary factor path, G* an independent stationary draw (E3 starts G from the end of TRAIN); cells `joint` and `indep` (independent tickers, same margins).
  - E1: fixed-b HAC p-values (`hac_fixedb`, `hac_fixedb_eq`) computed post hoc as comparison rows (primary `hac` unchanged); the E1(b) table shows rates only (splits are dependent, no Wilson CIs).
  - E2(e): the resample generator runs with a source-grouped split (`resample`, the size row) and a random split (`resample_naive`, leakage demonstration, not a size); shifted cells (P_past = 0.95) kept beside the all-equal cell.
  - E2(f): LR-critic fitted with three training budgets (defaults with early stopping, 1000 steps, 3000 steps at lr 1e-4); calibration slope reported instead of R^2. E2: KS in family (c) flagged (oversized by within-window GARCH dependence).
  - E3: a single-seed anchored p-value is not calibrated on real data (non-stationary R_past); the verdict uses only p = min(1, 2 median) over 10 synthetic seeds, and power and boundary size are estimated for that rule.
  - E3 power check: 200 replications (DESIGN minimum 50), boundary G0 from stationarity, plus design D2 (`realvol`, real TRAIN volatility path); GOOD needs rule power >= 0.8 under both D1 and D2.
  - E3 per-ticker L = 5 tests: p-values calibrated against a pooled GARCH-t null (`perticker_null`, 10 pairs per ticker) before BH/BY/Storey.
  - E3/E5: relative tests are invalid for the stationary bootstrap (G copies R_past windows seen as pool negatives); its verdict is INVALID (E3) / n/a (E5).
  - E4: new `placebo` part (200 replications per generator) and size-adjusted rejection shares; FactorGARCHt as fifth generator; RealTRAIN reference = last 615 TRAIN days; MMD with a fixed per-universe bandwidth; 10 test seeds; link in four views (pooled, within-universe, within-generator, generator-level with n = 5).
  - E5 calls E3's power/realvol replication functions and verdict rule; new parts `gridsize` (absolute size per configuration, McNemar vs default), `gridpower`, `gridrealvol`; n_perm moved from the grid to `simpower`.
  - Compute: full-settings workers peak at 0.46-0.6 GB (critic-only) to 0.9 GB (kernel matrices: E1, E3 pooled, E5 split battery), so 12 workers do not fit in ~5 GB free RAM; E3/E5 cap workers from MemAvailable; E5 gridpower (4.35 h) and gridrealvol (2.86 h) exceed 2.5 h at 12 workers; schedule in experiments/README.md.
- 2026-11-15 (lead, compute budget; ~205 CPU h requested vs ~45 available today): E3 synthetic seeds
  10 -> 5 (verdict rule: min(1, 2 x median p) over 5 seeds), E3 power/realvol reps 200 -> 100;
  E5 runs grid, split (rule power 25 reps), simpower (50 reps); gridsize, gridpower, gridrealvol not run
  (E1 measures the size of the default real-data pipeline); E4 test seeds 2, placebo reps 60;
  E2 part e at n = 500 only (200 reps); E1 real-data placebo 50 splits. Commands: experiments/run_full.sh.
- 2026-11-15 (lead): new experiments/e2_dependence.py (E2 dependence-only families "mix" and "block",
  exact marginal, so KS has no power by construction) after preliminary E2 results showed pooled KS
  dominating every family that changes the marginal.
- 2026-11-16 (lead, post hoc): critic architecture ablation. A pilot on four E2 designs at n = 500 showed
  a time-pooled CNN critic (two Conv1d blocks, kernel 5, 64 channels, mean over time, dense head) far
  more powerful than the default MLP at the same size. New `arch="pooled"`
  (`synthval.twosample.nets.PooledConvCritic`; the default stays the MLP, so no earlier result changes)
  and experiments/e5_architecture.py: (sim) E2 families a-d at two alternative levels (+ H0 in a, c)
  and E2-dep mix/block at level 2, 50 replications (200 for the H0 cells, extended after 7/50
  rejections in family a; 0.06 at 200), paired with E2/E2-dep (same windows and test
  streams); (real) E3 protocol with the pooled critic via E5's grid and design machinery: absolute +
  anchored at E3's 5 seeds, D1/D2 rule power and boundary size with 20 replications x 5 seeds (paired
  with E3's first 20). Chosen after the MLP's real-data results were known: reported as a post-hoc
  ablation; the pre-specified verdicts remain those of the MLP critic.
