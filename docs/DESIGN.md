# synthval: design contract

This is the contract every module and experiment builds against. If you need to deviate from an
interface here, keep the documented signature working and note the change at the bottom of this
file under "Change log", with a one-line reason.

## 1. The paper in one paragraph

**Working title:** *Is Synthetic Financial Data Good Enough? A Wasserstein-Critic Two-Sample Test,
Likelihood Benchmarks for Markov Models, and Markowitz Portfolio Utility.*

Synthetic return data is used for back-testing, data sharing and portfolio construction, but
"is this synthetic data good?" is usually answered with eyeballed stylized facts or with tests
that have no power. We turn the discriminator of a Wasserstein GAN (the critic, trained alone,
with no generator) into a formal two-sample test with exact finite-sample level (held-out
permutation), and a *relative* test that asks whether synthetic data is at least as close to
future real data as the real past is (markets are non-stationary, so "identical to the future"
is the wrong bar). We validate size and power against ground truth, and against the
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
1. **Absolute test** H0: Law(G) = Law(R_new). Critic statistic, permutation p-value.
2. **Relative test** H0: d(G, R_new) <= d(R_past, R_new) vs H1: d(G, R_new) > d(R_past, R_new).
3. **Verdict:** `GOOD` if the relative test does not reject at level alpha *and* the power check
   (simulation at matched sample size) shows power >= 0.8 against a pre-specified deviation;
   `BAD` if the relative test rejects; `INCONCLUSIVE` otherwise. The absolute test is reported
   alongside ("distinguishable from the future but no worse than history" is the typical
   real-world outcome and is a finding, not a failure).

## 2. Data

- File: `data/sp500_price_volume.csv`, Bloomberg export, `;`-separated. Line 1: `Date de début;2018-01-01`,
  line 2: `Date de fin;2025-06-17`, then a 2-level header (ticker, field) with fields `PX_LAST`
  and `PX_VOLUME`; first data row 2018-01-02. 503 tickers, current S&P 500 constituents
  (survivorship bias: must be stated as a limitation).
- Use `PX_LAST` only. **Never back-fill.** A ticker is in the *complete universe* if it has no
  missing `PX_LAST` from 2018-01-02 to 2025-06-17 (about 481 tickers). Isolated internal gaps
  (<= 3 consecutive days) may be forward-filled; anything else excludes the ticker.
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
| `GARCHt` | r_t = mu + s_t z_t, s_t^2 = w + a eps_{t-1}^2 + b s_{t-1}^2, z ~ standardised Student-t(nu) | MLE (scipy, constraints a+b<1, nu>2.05), multistart; tests cross-check vs `arch` | exact (filtered) |
| `Heston` | Euler full-truncation, daily step: r_t = mu - v_t/2 + sqrt(v_t) z1, v_{t+1} = v_t + kappa(theta - v_t) + xi sqrt(v_t) z2, corr(z1,z2)=rho | simulated method of moments on: var, kurtosis, ACF(r^2) lags 1,5,10,20, corr(r_t, r_{t+1}^2); common random numbers; Nelder-Mead/L-BFGS-B with bounds | intractable |
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
  HAC/Newey-West variance over eval windows ordered in time (robustness to serial dependence
  between windows, cf. Chung & Romano 2013), plus its normal p-value `extra["p_hac"]`, when the
  caller passes `order_x`/`order_y` (time indices); otherwise the i.i.d. z.

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

- **E1 Placebo / size on real data.** Real-vs-real comparisons where H0 is plausible: random
  halves of the panel tickers in the same period (repeated with many random splits); and
  R_past vs R_new (quantifies non-stationarity, the reference for the relative test). Report
  empirical rejection rates per test at alpha in {0.01, 0.05, 0.10}.
- **E2 Ground-truth size and power.** "Real" sampled from known P, "synthetic" from Q.
  Families (each indexed by delta, delta=0 is H0): (a) volatility level x(1+delta);
  (b) tails: Student-t nu=4 vs nu in {4, 6, 10, 30, inf} (variance matched); (c) volatility
  clustering: GARCH persistence a+b in {0.98, 0.95, 0.90, 0.80, 0} at matched unconditional
  variance and tails; (d) jumps: Merton vs variance-matched GBM, mixing fraction delta.
  Sample sizes n in {500, 2000} windows of L=20; reps 200 at delta=0, 100 otherwise. Tests:
  critic, C2ST, MMD, SigMMD, KS, and the LR oracle where tractable. Output: size table, power
  curves (one panel per family).
- **E3 S&P 500 univariate.** Fit each univariate model per panel ticker on TRAIN; simulate the
  TEST horizon; pooled (vol-normalised, grouped by ticker) absolute and relative critic tests per
  model; per-ticker tests at L=5 with Benjamini-Hochberg; stylized-facts table (real vs models);
  likelihood benchmark for GBM/Merton/GARCHt: predictive log-lik, Berkowitz rejection rates,
  pairwise Vuong; agreement (Kendall tau of model rankings, Cohen kappa of per-ticker verdicts)
  between critic and likelihood-based evaluation. Final verdict table (GOOD/BAD/INCONCLUSIVE).
- **E4 Markowitz utility.** Random universes (e.g. 30 universes x N=30 assets from the complete
  universe). For each MV generator: fit on TRAIN, simulate TRAIN-length synthetic samples (B
  replicas), estimate moments (sample and Ledoit-Wolf), build GMV, long-only mean-variance and
  1/N; evaluate on real TEST; compare with the same portfolios estimated on real TRAIN.
  Utility loss = out-of-sample variance/Sharpe gap vs real-estimated portfolio. Also
  multivariate critic test (windows (N, L=5)) and MMD; correlate distances with utility loss
  (Spearman, with universe-level bootstrap CI).
- **E5 Robustness.** Window length L in {5, 10, 20, 60}; critic capacity and steps; GP lambda;
  train/eval split fraction; number of permutations; sensitivity of E3 verdicts.

## 5. Paper (`paper/`)
`main.tex` (article class, 11pt, natbib author-year, booktabs, amsmath/amsthm, hyperref,
cleveref), `sections/{abstract,introduction,literature,methodology,theory,simulation,empirical,markowitz,robustness,conclusion,appendix}.tex`,
`references.bib` (only entries verified to exist), `figures/`, `tables/` (generated). English.
Compile with `latexmk -pdf main.tex` from `paper/`.

## Change log
- (agents append here)
