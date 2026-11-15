"""E2: ground-truth size and power of the critic test and its competitors (DESIGN section 4, E2).

"Real" windows X come from a known law P and "synthetic" windows Y from a known law Q; delta = 0
is H0 (Q = P). Every window is simulated from its own path (GARCH: burn-in of 250 days from the
unconditional variance; i.i.d. models need none), so windows are i.i.d. and every permutation
p-value is exact (protocol "iid", DESIGN section 3.7).

Base processes (daily log-returns, mean 3e-4 per day, unconditional daily volatility 0.02, i.e.
about 32% a year, a typical single-stock level):

* i.i.d. standardised Student-t(nu) scaled to daily volatility 0.02 (``GARCHt`` with
  alpha = beta = 0), nu = 5 for family (a) and nu = 4 for family (b);
* GARCH(1,1)-t: alpha = 0.08, beta = 0.90 (persistence 0.98), nu = 5, omega set so that the
  unconditional variance is 0.02^2 (typical daily stock estimates: alpha 0.05-0.10,
  persistence 0.95-0.99, nu 4-8). Alternatives of lower persistence keep alpha = 0.08 (beta =
  persistence - 0.08), the same nu and the same unconditional variance; persistence 0 is the
  i.i.d. t(5) law;
* Merton jump-diffusion: lambda = 0.05 jumps per day (about 12.6 a year), jump sizes
  N(-0.01, 0.04^2), diffusion volatility set so that the total daily volatility is 0.02 (jumps
  carry 21% of the variance; excess kurtosis 2.7); drift set so that the mean is 3e-4;
* GBM: Gaussian i.i.d. returns with mean 3e-4 and volatility 0.02 (variance matched).

Families (grid index k = 0..4; k = 0 is H0):

(a) volatility level: P = t(5); Q = t(5) with volatility x (1 + delta), delta in
    {0, 0.05, 0.1, 0.2, 0.3};
(b) tails: P = t(4); Q = t(nu), nu in {4, 6, 10, 30, inf} (inf = Gaussian), variance matched;
(c) volatility clustering: P = GARCH-t persistence 0.98; Q = persistence in
    {0.98, 0.95, 0.90, 0.80, 0}, matched unconditional variance and nu;
(d) jumps: P = Merton; Q = window-level mixture (1 - delta) Merton + delta GBM (variance
    matched), delta in {0, 0.25, 0.5, 0.75, 1}: each Q window is a GBM window with probability
    delta, so the window density of Q is exactly (1 - delta) p_Merton + delta p_GBM.

n in {500, 2000} windows of L = 20 per sample; 200 replications at k = 0 and 100 otherwise.
The X sample of replication r is shared by all levels k of a (family, n) cell (common random
numbers; each level's rejection rate is still an unbiased estimate).

Tests: critic, lrcritic, c2st, mmd, sigmmd (permutation p-values, exact), ks (asymptotic; on
pooled returns it ignores the window structure and, for GARCH windows, within-window
dependence: returns of one window share the conditional variance, so the pooled returns are not
i.i.d. and KS is oversized in family (c); it is marked with a dagger in the tables and drawn
with hollow markers in the figure, and its power there is not size-adjusted), and the
Neyman-Pearson likelihood-ratio oracle (theory.md Prop. 5.5): the Monte
Carlo test of H0: Y ~ P against H1: Y ~ Q with statistic T = sum_i log q(y_i) / p(y_i). Its null
distribution does not depend on the replication, so it is computed once per (family, level, n)
cell from B = 2000 samples of n windows drawn under P (the same P samples serve every level and
both n, using the first n windows of each), and the oracle p-value is the Monte Carlo p-value
with randomised tie-breaking (exact; at k = 0 T = 0 identically and the p-value is uniform).
The oracle is exact for (a), (b), (d). For (c) the GARCH window density integrates over the law
of the conditional variance at the window start by Monte Carlo with K = 512 draws of that law
(the law after the same 250-day burn-in, ``GARCHt.stationary_variance_draws``), shared by the
observed and the null samples (so the Monte Carlo test stays exact and T converges to the exact
log-likelihood ratio as K grows); without that method the fixed-start approximation is used and
the oracle is labelled "approximate".

Sub-experiments (``--part``):

main_a .. main_d (alias ``main``)
    families (a)-(d) as above; outputs ``e2_power.pdf`` (Wilson 95% error bars),
    ``e2_size.tex``, ``e2_power.tex`` (rates at 5%), ``e2_power_ci.tex`` (rates with Wilson
    CIs) and the notes
    ``e2_size_note.tex`` / ``e2_power_note.tex`` (dagger marks, p-value types, oracle type).
    The C2ST / LR-critic rows record the early-stopping checkpoint (``best_step``) next to
    ``steps_run``.
e   relative test at the boundary and in the interior; anchored (permutation p-value) vs
    two-critic (paired bootstrap p-value) relative test with the W-critic; n in {500, 2000}
    windows per sample; 200 replications. Two scenarios:

    * ``equal`` (DESIGN): P_new = P_past = G = GARCH-t persistence 0.98 (boundary);
    * ``shift`` (the pilot of theory.md note 11.3): P_new = GARCH-t 0.98, P_past = 0.95, and
      G in {P_past (boundary: Law(G) = Law(R_past) != Law(R_new)), P_new (interior: better
      than the past), GARCH-t 0.90 and GBM (worse than the past), an i.i.d. resample of the
      R_past sample (DESIGN's "bootstrap-like resample of R_past*")}.

    The resample is run twice on the same G. ``resample``: every G window carries the index of
    its source R_past window as group label and R_past carries ``arange(n)`` in the same
    namespace (``groups_g`` / ``groups_past``), so copies stay on the same side of the
    train/eval split; the test is then conservative (G_ev are copies of R_past_ev windows).
    ``resample_naive``: random split, as a naive user would run it; G_ev then contains copies
    of windows the critic saw as pool negatives (train/eval leakage), exchangeability fails and
    the test rejects although G is "as good as history". This row documents the leakage (it is
    not a size); it is the situation of the stationary-bootstrap generator in E3.
    Outputs ``e2_relative_boundary.tex`` and ``e2_relative_boundary_note.tex``.
f   learned vs true log-likelihood ratio: GBM vs Merton (real P = Merton, synthetic Q = GBM)
    and GARCH 0.98 (P) vs 0.90 (Q); ``C2ST.fit_logit`` trained on n = 2000 windows per side,
    then Spearman / Pearson correlation and calibration slope (OLS of the logit on the exact
    value; 1 if calibrated) of the learned logit against the exact log p/q on 2000 fresh
    windows (1000 from each law; GARCH density with K = 2048 initial-variance draws), 20
    replications; also the W-critic's correlations with log p/q. The logit is fitted with
    three training budgets on the same data: the test defaults used in the main experiment
    (early stopping; the selected checkpoint ``best_step`` is recorded), 1000 steps without
    early stopping (the W-critic's budget) and 3000 steps at lr 1e-4 without early stopping,
    so that the result is not confounded with an early-stopping failure. Outputs
    ``e2_llr.tex``, ``e2_llr_note.tex`` and ``e2_llr_scatter.pdf`` (replication 0: the raw
    logit, which with balanced classes should equal log p/q, against the exact value, identity
    line; one row per training budget in ``F_SCATTER_VARIANTS``).
g   dependent windows: X and Y each cut into n = 500 consecutive windows from one long
    GARCH-t (0.98) path (independent paths, H0 true); random-split permutation p-values vs
    time-block (10 blocks) + HAC (learned tests; also the permutation p-value under the
    time-block split) vs wild bootstrap (MMD tests), for critic, lrcritic, mmd, sigmmd; 200
    replications; output ``e2_dependent_size.tex`` (MMD / Sig-MMD rows are labelled
    "permutation": they have no train/eval split).

Usage::

    uv run python experiments/e2_ground_truth.py --quick          # smoke test + timing estimate
    uv run python experiments/e2_ground_truth.py --part main --workers 12
    uv run python experiments/e2_ground_truth.py --part e --workers 12
    uv run python experiments/e2_ground_truth.py --part f --part g --workers 12
    uv run python experiments/e2_ground_truth.py --plots-only     # rebuild figures and tables

Memory: one main replication at n = 2000 peaks at about 0.6 GB per worker process, so 12
workers need about 7.3 GB. Run the main families in one process (``--part main``) and do not
start several 12-worker processes at once on a shared machine (split the workers instead).
When parts do run in parallel processes, each rebuilds only its own outputs at the end; run
``--plots-only`` once after all of them have finished so that the combined tables see every
family.

Checkpoints are keyed by the configuration *and* by a content hash of the library files each
part uses (``synthval/models`` and ``synthval/twosample``, ``synthval/config.py``), so a
resumed run never mixes library versions: after a library change the affected parts recompute.

Rejection rates are reported with Wilson 95% intervals; every result row records its p-value
type (``ptype``: permutation, hac, wild, timeblock_permutation, asymptotic, monte_carlo,
paired_bootstrap), and the tables state it.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import functools  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import common  # noqa: E402  (experiments/ is on sys.path when run as a script)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import synthval  # noqa: E402
from synthval import plotting, utils  # noqa: E402
from synthval.models.univariate import GARCHt, GBM, Merton  # noqa: E402
from synthval.twosample import (C2ST, CriticTest, KSTest, MMDTest, SigMMDTest,  # noqa: E402
                                relative_test)
from synthval.twosample.critic import perm_pvalues  # noqa: E402

logger = logging.getLogger("experiments.e2")

EXP = "e2"
ROOT_SEED = 20251115
PARTS = ("main_a", "main_b", "main_c", "main_d", "e", "f", "g")
ALIASES = {"main": ["main_a", "main_b", "main_c", "main_d"]}

# --------------------------------------------------------------------------------------------
# Base processes (daily units)
# --------------------------------------------------------------------------------------------
L = 20                # window length (days)
BURN = 250            # burn-in of every GARCH path (days)
MU = 3e-4             # mean daily log-return of every law
VOL = 0.02            # unconditional daily volatility of every base law
GARCH_ALPHA = 0.08
GARCH_NU = 5.0
MERTON_LAM, MERTON_MUJ, MERTON_SJ = 0.05, -0.01, 0.04
MERTON_SIGMA = math.sqrt(VOL ** 2 - MERTON_LAM * (MERTON_MUJ ** 2 + MERTON_SJ ** 2))
MERTON_MU = MU - MERTON_LAM * MERTON_MUJ
ALPHAS = (0.01, 0.05, 0.10)
TESTS = ("critic", "lrcritic", "c2st", "mmd", "sigmmd", "ks")
ALL_TESTS = TESTS + ("oracle",)
# KS on pooled returns assumes i.i.d. returns; within a GARCH window they share the conditional
# variance, so KS is oversized in these families (marked in tables and figures).
KS_INVALID_FAMILIES = ("c",)

# Library files whose content determines each kind of result (hashed into the checkpoint keys).
_LIB = Path(synthval.__file__).resolve().parent
_CODE_GROUPS = {"models": ("models/base.py", "models/univariate.py"),
                "tests": ("twosample/*.py", "config.py"),
                # part e uses only the W-critic and the relative test; part f the C2ST logit
                # and the W-critic (narrower sets avoid needless recomputation)
                "tests_e": ("twosample/base.py", "twosample/nets.py", "twosample/critic.py",
                            "twosample/relative.py", "config.py"),
                "tests_f": ("twosample/base.py", "twosample/nets.py", "twosample/critic.py",
                            "twosample/c2st.py", "config.py")}


@functools.lru_cache(maxsize=None)
def lib_fingerprint(group: str) -> str:
    """SHA-256 (16 hex digits) of the library files of ``group`` (paths and contents)."""
    files = sorted({f for pat in _CODE_GROUPS[group] for f in _LIB.glob(pat)})
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(_LIB).as_posix().encode() + b"\0"
                 + utils.file_sha256(f).encode() + b"\n")
    return h.hexdigest()[:16]


def code_keys(*groups: str) -> dict[str, str]:
    """``{"code_<group>": fingerprint}``: added to part configurations, so a checkpoint written
    with other library code is never reused (the affected part recomputes instead)."""
    return {f"code_{g}": lib_fingerprint(g) for g in groups}

# Full configuration (DESIGN section 4, E2) and the smoke-test configuration.
FULL = {
    "n_list": [500, 2000], "reps_h0": 200, "reps_h1": 100, "tests": {},
    "oracle_B": 2000, "oracle_K": 512, "oracle_chunk": 20,
    "e_n_list": [500], "e_reps": 200, "e_tests": {}, "e_n_boot": 500,
    "f_reps": 20, "f_n_train": 2000, "f_n_fresh": 2000, "f_tests": {}, "f_K": 2048,
    # LR-critic training budgets of part f (overrides of the C2ST defaults; "default" = the
    # configuration of the main experiment).
    "f_variants": {"default": {}, "fixed": {"early_stopping": False},
                   "long_lowlr": {"early_stopping": False, "n_steps": 3000, "lr": 1e-4}},
    "g_n": 500, "g_reps": 200, "g_tests": {}, "g_time_blocks": 10,
    "blocks": {"main": 10, "e": 2, "f": 1, "g": 5},
}
QUICK = {
    "n_list": [100, 200], "reps_h0": 3, "reps_h1": 3,
    "tests": {"n_steps": 200, "n_perm": 200, "mmd_n_perm": 200},
    "oracle_B": 200, "oracle_K": 64, "oracle_chunk": 50,
    "e_n_list": [100, 200], "e_reps": 2, "e_tests": {"n_steps": 200, "n_perm": 200},
    "e_n_boot": 200,
    "f_reps": 2, "f_n_train": 400, "f_n_fresh": 400,
    "f_tests": {"n_steps": 200, "n_perm": 200}, "f_K": 256,
    "f_variants": {"default": {}, "fixed": {"early_stopping": False},
                   "long_lowlr": {"early_stopping": False, "n_steps": 300, "lr": 1e-4}},
    "g_n": 100, "g_reps": 3, "g_tests": {"n_steps": 200, "n_perm": 200, "mmd_n_perm": 200},
    "g_time_blocks": 10,
    "blocks": {"main": 3, "e": 1, "f": 1, "g": 3},
}


def spec_t(nu: float, vol: float = VOL) -> dict:
    return {"kind": "t", "nu": float(nu), "vol": float(vol)}


def spec_gbm(vol: float = VOL) -> dict:
    return {"kind": "gbm", "vol": float(vol)}


def spec_garch(pers: float) -> dict:
    return {"kind": "garch", "pers": float(pers)}


def spec_merton() -> dict:
    return {"kind": "merton"}


def spec_mix(delta: float) -> dict:
    return {"kind": "mix", "delta": float(delta)}


def spec_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True)


@functools.lru_cache(maxsize=None)
def _model(key: str) -> Any:
    spec = json.loads(key)
    kind = spec["kind"]
    if kind == "t":
        return GARCHt(mu=MU, omega=spec["vol"] ** 2, alpha=0.0, beta=0.0, nu=spec["nu"])
    if kind == "gbm":
        return GBM(mu=MU, sigma=spec["vol"])
    if kind == "garch":
        p = spec["pers"]
        a = min(GARCH_ALPHA, p)
        return GARCHt(mu=MU, omega=(1.0 - p) * VOL ** 2, alpha=a, beta=p - a, nu=GARCH_NU)
    if kind == "merton":
        return Merton(mu=MERTON_MU, sigma=MERTON_SIGMA, lam=MERTON_LAM, muJ=MERTON_MUJ,
                      sJ=MERTON_SJ)
    raise ValueError(f"no single model for spec {spec}")


def model_of(spec: dict) -> Any:
    """The synthval model of a spec (cached per process)."""
    return _model(spec_key(spec))


def _persistent(spec: dict) -> bool:
    return spec["kind"] == "garch" and spec["pers"] > 0


def sim_windows(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` independent windows ``(n, L)``, each from its own path (GARCH: 250-day burn-in)."""
    if spec["kind"] == "mix":
        d = spec["delta"]
        is_g = rng.random(n) < d
        out = np.empty((n, L))
        ng = int(is_g.sum())
        if ng:
            out[is_g] = sim_windows(spec_gbm(), ng, rng)
        if ng < n:
            out[~is_g] = sim_windows(spec_merton(), n - ng, rng)
        return out
    m = model_of(spec)
    return m.simulate(L, n, rng=rng, burn_in=BURN if _persistent(spec) else 0)


def sim_long_path_windows(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` consecutive non-overlapping windows cut from one path of ``n * L`` days (part g)."""
    m = model_of(spec)
    return m.simulate(n * L, 1, rng=rng, burn_in=BURN).reshape(n, L)


HAS_MC_INIT = hasattr(GARCHt, "stationary_variance_draws")


@functools.lru_cache(maxsize=None)
def _sigma2_draws(key: str, K: int) -> np.ndarray | None:
    """Fixed draws of the GARCH conditional variance at a window's first day (same law as the
    simulation: 250 recursions from the unconditional variance); keyed, so identical in every
    process."""
    spec = json.loads(key)
    if not HAS_MC_INIT or K <= 0:
        return None
    rng = common.keyed_rng(ROOT_SEED, "sigma2_init", key, int(K))
    return model_of(spec).stationary_variance_draws(int(K), rng=rng, burn_in=BURN)


def oracle_type(fam: str, K: int) -> str:
    if fam != "c":
        return "exact"
    return f"mc_integrated_K{K}" if (HAS_MC_INIT and K > 0) else "approximate_fixed_start"


def logdens(spec: dict, W: np.ndarray, K: int, memo: dict | None = None) -> np.ndarray:
    """Log-density of each window under ``spec`` (exact; GARCH: Monte Carlo over K start variances).

    ``memo`` caches component densities of the same ``W`` (Merton / GBM inside mixtures).
    """
    memo = {} if memo is None else memo
    key = spec_key(spec)
    if key in memo:
        return memo[key]
    if spec["kind"] == "mix":
        d = spec["delta"]
        if d <= 0:
            out = logdens(spec_merton(), W, K, memo)
        elif d >= 1:
            out = logdens(spec_gbm(), W, K, memo)
        else:
            out = np.logaddexp(math.log1p(-d) + logdens(spec_merton(), W, K, memo),
                               math.log(d) + logdens(spec_gbm(), W, K, memo))
    else:
        m = model_of(spec)
        if _persistent(spec):
            s2 = _sigma2_draws(key, K)
            out = m.loglik_windows(W) if s2 is None else m.loglik_windows(W, sigma2_init=s2)
        else:
            out = m.loglik_windows(W)
    memo[key] = out
    return out


# --------------------------------------------------------------------------------------------
# Families of the main experiment
# --------------------------------------------------------------------------------------------
FAMILIES: dict[str, dict[str, Any]] = {
    "a": {"title": "(a) volatility level", "P": spec_t(5.0),
          "levels": [0.0, 0.05, 0.10, 0.20, 0.30],
          "Q": [spec_t(5.0, VOL * (1 + d)) for d in (0.0, 0.05, 0.10, 0.20, 0.30)],
          "ticks": ["0", "0.05", "0.1", "0.2", "0.3"],
          "xlabel": r"$\delta$ (volatility $\times(1+\delta)$)",
          "tex_ticks": ["0", "0.05", "0.1", "0.2", "0.3"], "param": r"$\delta$"},
    "b": {"title": "(b) tails", "P": spec_t(4.0),
          "levels": [4.0, 6.0, 10.0, 30.0, math.inf],
          "Q": [spec_t(4.0), spec_t(6.0), spec_t(10.0), spec_t(30.0), spec_gbm()],
          "ticks": ["4", "6", "10", "30", r"$\infty$"],
          "xlabel": r"$\nu$ of $Q$ ($P$: $\nu=4$)",
          "tex_ticks": ["4", "6", "10", "30", r"$\infty$"], "param": r"$\nu$"},
    "c": {"title": "(c) volatility clustering", "P": spec_garch(0.98),
          "levels": [0.98, 0.95, 0.90, 0.80, 0.0],
          "Q": [spec_garch(p) for p in (0.98, 0.95, 0.90, 0.80, 0.0)],
          "ticks": ["0.98", "0.95", "0.90", "0.80", "0"],
          "xlabel": r"persistence $\alpha+\beta$ of $Q$ ($P$: 0.98)",
          "tex_ticks": ["0.98", "0.95", "0.90", "0.80", "0"], "param": r"$\alpha+\beta$"},
    "d": {"title": "(d) jumps", "P": spec_merton(),
          "levels": [0.0, 0.25, 0.50, 0.75, 1.0],
          "Q": [spec_mix(d) for d in (0.0, 0.25, 0.50, 0.75, 1.0)],
          "ticks": ["0", "0.25", "0.5", "0.75", "1"],
          "xlabel": r"GBM mixing fraction $\delta$ ($P$: Merton)",
          "tex_ticks": ["0", "0.25", "0.5", "0.75", "1"], "param": r"$\delta$"},
}


def llr_windows(fam: str, k: int, W: np.ndarray, K: int,
                memo: dict | None = None) -> np.ndarray:
    """Per-window oracle log-likelihood ratio ``log q_k(w) - log p(w)`` (0 at k = 0)."""
    if k == 0:
        return np.zeros(W.shape[0])
    memo = {} if memo is None else memo
    F = FAMILIES[fam]
    return logdens(F["Q"][k], W, K, memo) - logdens(F["P"], W, K, memo)


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------
def make_tests(tc: dict, names: tuple[str, ...] = TESTS) -> dict[str, Any]:
    """Test instances: the TEST_REGISTRY defaults, with the overrides ``tc`` (quick mode)."""
    learned = {k: tc[k] for k in ("n_steps", "n_perm") if k in tc}
    mmd = {"n_perm": tc["mmd_n_perm"]} if "mmd_n_perm" in tc else {}
    factories = {
        "critic": lambda: CriticTest(**learned),
        "lrcritic": lambda: C2ST(statistic="logit", **learned),
        "c2st": lambda: C2ST(**learned),
        "mmd": lambda: MMDTest(**mmd),
        "sigmmd": lambda: SigMMDTest(**mmd),
        "ks": lambda: KSTest(),
    }
    return {n: factories[n]() for n in names}


def _attrs(t: Any) -> dict[str, Any]:
    """Scalar attributes of a test instance."""
    return {k: v for k, v in vars(t).items() if isinstance(v, (int, float, str, bool, type(None)))}


def test_settings(tc: dict) -> dict[str, dict]:
    """Scalar attributes of every configured test (recorded in the metadata)."""
    out = {}
    for name, t in make_tests(tc).items():
        attrs = _attrs(t)
        if hasattr(t, "mmd"):
            attrs.update({f"mmd.{k}": v for k, v in vars(t.mmd).items()
                          if isinstance(v, (int, float, str, bool, type(None)))})
        out[name] = attrs
    return out


def _ptype(test: Any) -> str:
    if isinstance(test, KSTest):
        return "asymptotic"
    inf = getattr(test, "inference", "permutation")
    if inf == "permutation" and getattr(test, "time_blocks", None):
        return "timeblock_permutation"
    return str(inf)


def _test_row(name: str, test: Any, res: Any, dt: float, err: str) -> dict[str, Any]:
    row: dict[str, Any] = {"test": name, "ptype": _ptype(test), "seconds": dt, "error": err}
    if res is None:
        row.update(pvalue=np.nan, statistic=np.nan)
        return row
    ex = res.extra
    row.update(pvalue=float(res.pvalue), statistic=float(res.statistic),
               steps_run=ex.get("steps_run", np.nan), z=ex.get("z", np.nan),
               p_hac=ex.get("p_hac", np.nan), p_perm=ex.get("p_perm", np.nan),
               **train_diagnostics(ex))
    return row


def train_diagnostics(info: dict[str, Any]) -> dict[str, Any]:
    """Early-stopping diagnostics of a learned test: the selected checkpoint ``best_step``
    (NaN without early stopping; ``steps_run`` alone hides which weights were kept) and the
    validation loss at the first and at the selected checkpoint."""
    vc = info.get("val_curve") or {}
    steps, vals = list(vc.get("step", [])), list(vc.get("bce", vc.get("objective", [])))
    best = info.get("best_step", np.nan)
    out = {"early_stopping": bool(vc), "best_step": best if best is not None else np.nan,
           "val_first": float(vals[0]) if vals else np.nan, "val_best": np.nan}
    if vals and best in steps:
        out["val_best"] = float(vals[steps.index(best)])
    return out


# --------------------------------------------------------------------------------------------
# Oracle null distributions
# --------------------------------------------------------------------------------------------
def _oracle_null_chunk(payload: dict, rng: np.random.Generator) -> np.ndarray:
    """T statistics of null samples ``b0..b1-1`` of a family: array ``(nb, 5, len(n_list))``.

    Null sample b is drawn with ``keyed_rng(ROOT_SEED, "oracle_null", fam, b)`` (independent
    of the chunking); T for size n sums the first n windows. With ``payload["path"]`` the chunk
    is also saved there (``.npy``, atomic) as a checkpoint.
    """
    del rng
    fam, b0, b1 = payload["fam"], payload["b0"], payload["b1"]
    path = payload.get("path")
    if path and Path(path).exists() and not payload.get("fresh", False):
        return np.load(path)
    n_list, K = payload["n_list"], payload["K"]
    n_max = max(n_list)
    P = FAMILIES[fam]["P"]
    W = np.concatenate([sim_windows(P, n_max, common.keyed_rng(ROOT_SEED, "oracle_null", fam, b))
                        for b in range(b0, b1)], axis=0)
    memo: dict = {}
    out = np.zeros((b1 - b0, 5, len(n_list)))
    for k in range(1, 5):
        llr = llr_windows(fam, k, W, K, memo).reshape(b1 - b0, n_max)
        cs = np.cumsum(llr, axis=1)
        for j, n in enumerate(n_list):
            out[:, k, j] = cs[:, n - 1]
    if path:
        tmp = f"{path}.{os.getpid()}.tmp.npy"
        np.save(tmp, out)
        os.replace(tmp, path)
    return out


def oracle_null_path(ctx: common.ExpContext, cfg: dict) -> Path:
    oc = {"B": cfg["oracle_B"], "K": cfg["oracle_K"], "n_list": cfg["n_list"],
          "families": {f: [FAMILIES[f]["P"], FAMILIES[f]["Q"]] for f in FAMILIES},
          "has_mc_init": HAS_MC_INIT, "burn": BURN, "L": L, "seed": ROOT_SEED,
          **code_keys("models")}
    ck = common.Checkpoint(ctx.ckpt_dir, "oracle", oc)
    return ck.dir


def compute_oracle_nulls(ctx: common.ExpContext, cfg: dict, fams: list[str]) -> dict[str, str]:
    """Compute (or load) the oracle null distribution of each family; returns CSV paths."""
    d = oracle_null_path(ctx, cfg)
    paths = {f: str(d / f"null_{f}.csv") for f in fams}
    todo = [f for f in fams if ctx.fresh or not Path(paths[f]).exists()]
    if not todo:
        return paths
    B, ch = cfg["oracle_B"], cfg["oracle_chunk"]
    items = [(f"{f}_{b0}", {"fam": f, "b0": b0, "b1": min(B, b0 + ch), "n_list": cfg["n_list"],
                            "K": cfg["oracle_K"], "fresh": ctx.fresh,
                            "path": str(d / f"chunk_{f}_{b0:05d}_{ch}.npy")})
             for f in todo for b0 in range(0, B, ch)]
    # GARCH chunks first (most expensive).
    items.sort(key=lambda it: it[1]["fam"] != "c")
    t0 = time.perf_counter()
    res = common.run_items(_oracle_null_chunk, items, part="oracle", root_seed=ROOT_SEED,
                           workers=ctx.workers)
    logger.info("oracle nulls for %s: %s", todo, common.fmt_seconds(time.perf_counter() - t0))
    for f in todo:
        arrs = [r for (key, p), r in zip(items, res) if p["fam"] == f]
        T = np.concatenate(arrs, axis=0)  # (B, 5, n)
        rows = [{"family": f, "k": k, "n": n, "b": b, "T": float(T[b, k, j])}
                for k in range(5) for j, n in enumerate(cfg["n_list"]) for b in range(T.shape[0])]
        pd.DataFrame(rows).to_csv(paths[f] + ".tmp", index=False)
        os.replace(paths[f] + ".tmp", paths[f])
    return paths


@functools.lru_cache(maxsize=None)
def _load_null(path: str) -> dict[tuple[int, int], np.ndarray]:
    df = pd.read_csv(path)
    return {(int(k), int(n)): g["T"].to_numpy() for (k, n), g in df.groupby(["k", "n"])}


# --------------------------------------------------------------------------------------------
# Part main: one replication
# --------------------------------------------------------------------------------------------
def _main_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    del rng  # streams are keyed explicitly below
    fam, n, k = cell["family"], int(cell["n"]), int(cell["k"])
    cfg = cell["_cfg"]
    F = FAMILIES[fam]
    X = sim_windows(F["P"], n, common.keyed_rng(ROOT_SEED, "main_X", fam, n, rep))
    Y = sim_windows(F["Q"][k], n, common.keyed_rng(ROOT_SEED, "main_Y", fam, n, k, rep))
    rows = []
    for name, test in make_tests(cfg["tests"]).items():
        trng = common.keyed_rng(ROOT_SEED, "main_test", fam, n, k, rep, name)
        res, dt, err = common.safe_test(test, X, Y, rng=trng)
        rows.append(_test_row(name, test, res, dt, err))
    # Neyman-Pearson oracle on the synthetic sample (H0: Y ~ P vs H1: Y ~ Q_k).
    t0 = time.perf_counter()
    T = float(llr_windows(fam, k, Y, cfg["oracle_K"]).sum())
    if cell.get("_null_path"):
        null = _load_null(cell["_null_path"])[(k, n)]
        p = perm_pvalues(T, null, rng=common.keyed_rng(ROOT_SEED, "oracle_u", fam, n, k, rep))
        pval, n_null = float(p["randomized"]), null.size
    else:  # timing probe: statistic only
        pval, n_null = np.nan, 0
    rows.append({"test": "oracle", "ptype": "monte_carlo", "seconds": time.perf_counter() - t0,
                 "error": "", "pvalue": pval, "statistic": T, "n_null": n_null,
                 "oracle_type": oracle_type(fam, cfg["oracle_K"])})
    return rows


def main_cells(cfg: dict, fams: list[str], null_paths: dict[str, str]) -> list[dict]:
    cells = []
    for fam in fams:
        F = FAMILIES[fam]
        for n in cfg["n_list"]:
            for k in range(5):
                cells.append({"cell": common.cell_key(fam=fam, n=n, k=k), "family": fam,
                              "n": n, "k": k, "level": F["levels"][k],
                              "level_label": F["ticks"][k],
                              "_cfg": _main_cfg(cfg), "_null_path": null_paths.get(fam, "")})
    return cells


def _main_cfg(cfg: dict) -> dict:
    return {"tests": cfg["tests"], "oracle_K": cfg["oracle_K"]}


def _main_config(cfg: dict) -> dict:
    """Everything (apart from cells and reps) that determines main-part results."""
    return {"seed": ROOT_SEED, "L": L, "burn": BURN, "families": {
        f: {"P": F["P"], "Q": F["Q"], "levels": F["levels"]} for f, F in FAMILIES.items()},
        "tests": test_settings(cfg["tests"]), "oracle_B": cfg["oracle_B"],
        "oracle_K": cfg["oracle_K"], "oracle_mc_init": HAS_MC_INIT,
        "base": {"mu": MU, "vol": VOL, "garch_alpha": GARCH_ALPHA, "garch_nu": GARCH_NU,
                 "merton": [MERTON_MU, MERTON_SIGMA, MERTON_LAM, MERTON_MUJ, MERTON_SJ]},
        **code_keys("models", "tests")}


def run_main(ctx: common.ExpContext, cfg: dict, fams: list[str]) -> dict[str, float]:
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    null_paths = compute_oracle_nulls(ctx, cfg, fams)
    timing["oracle_null_s"] = time.perf_counter() - t0
    cells = main_cells(cfg, fams, null_paths)
    t0 = time.perf_counter()
    df = common.run_cells(
        _main_rep, cells, part="main", ctx=ctx, root_seed=ROOT_SEED,
        n_reps=lambda c: cfg["reps_h0"] if c["k"] == 0 else cfg["reps_h1"],
        config=_main_config(cfg), block_size=cfg["blocks"]["main"],
        cost=lambda c: 1.0 + c["n"] / 2000.0)
    timing["main_s"] = time.perf_counter() - t0
    for fam in fams:
        sub = df[df["family"] == fam]
        ctx.save_csv(sub, f"e2_main_{fam}_raw", config={"part": f"main_{fam}",
                                                         **_main_config(cfg),
                                                         "n_list": cfg["n_list"],
                                                         "reps_h0": cfg["reps_h0"],
                                                         "reps_h1": cfg["reps_h1"]})
        ctx.write_part_meta(f"main_{fam}", {**_main_config(cfg), "n_list": cfg["n_list"],
                                            "reps_h0": cfg["reps_h0"],
                                            "reps_h1": cfg["reps_h1"]}, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Part e: relative test at the boundary and in the interior
# --------------------------------------------------------------------------------------------
E_P_NEW = spec_garch(0.98)
# Scenario -> law of R_past and the seed key of the R_past sample.
E_SCENARIOS: dict[str, dict[str, Any]] = {
    "equal": {"P_past": spec_garch(0.98), "seed": "e_past_equal",
              "tex": r"$P_{\rm new}=P_{\rm past}$ = GARCH 0.98"},
    "shift": {"P_past": spec_garch(0.95), "seed": "e_past",
              "tex": r"$P_{\rm new}$ = GARCH 0.98, $P_{\rm past}$ = GARCH 0.95"},
}
# (name, scenario, role, G): G is a spec, or "resample_grouped" / "resample_naive" (an i.i.d.
# resample of the R_past windows; both modes use the same resample, only the split differs).
E_GENS: list[tuple[str, str, str, dict | str]] = [
    ("equal", "equal", "boundary: G = P_past = P_new (GARCH 0.98)", spec_garch(0.98)),
    ("past", "shift", "boundary: G = P_past (GARCH 0.95) != P_new", spec_garch(0.95)),
    ("new", "shift", "interior: G = P_new (GARCH 0.98)", spec_garch(0.98)),
    ("garch90", "shift", "worse: GARCH 0.90", spec_garch(0.90)),
    ("gbm", "shift", "worse: GBM", spec_gbm()),
    ("resample", "shift", "resample of R_past, split by source window (conservative)",
     "resample_grouped"),
    ("resample_naive", "shift", "resample of R_past, random split (train/eval leakage)",
     "resample_naive"),
]
E_METHODS = ("anchored", "two_critic")
E_TEX = {"equal": r"boundary: $G\sim P_{\rm past}=P_{\rm new}$",
         "past": r"boundary: $G\sim P_{\rm past}$",
         "new": r"interior: $G\sim P_{\rm new}$",
         "garch90": "worse: GARCH 0.90", "gbm": "worse: GBM",
         "resample": r"resample of $R_{\rm past}$, split by source",
         "resample_naive": r"resample of $R_{\rm past}$, random split (leakage)"}
E_METHOD_TEX = {"anchored": "Anchored (permutation)", "two_critic": "Two-critic (paired bootstrap)"}


def _e_generator(gname: str, gen: dict | str, n: int, rep: int,
                 r_past: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Synthetic sample of a part-e generator and the group labels of its split."""
    if isinstance(gen, dict):
        return sim_windows(gen, n, common.keyed_rng(ROOT_SEED, "e_G", n, rep, gname)), {}
    src = common.keyed_rng(ROOT_SEED, "e_G", n, rep, "resample").integers(0, n, size=n)
    G = r_past[src]
    if gen == "resample_grouped":
        # Copies share their source window's label, so they stay on its side of the split.
        return G, {"groups_g": src, "groups_past": np.arange(n)}
    if gen == "resample_naive":
        return G, {}
    raise ValueError(f"unknown generator {gen!r}")


def _e_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    del rng
    n = int(cell["n"])
    cfg = cell["_cfg"]
    gens = [g for g in E_GENS if g[0] in cell.get("_gens", [g[0] for g in E_GENS])]
    r_new = sim_windows(E_P_NEW, n, common.keyed_rng(ROOT_SEED, "e_new", n, rep))
    r_pasts: dict[str, np.ndarray] = {}
    learned = {k: cfg["tests"][k] for k in ("n_steps", "n_perm") if k in cfg["tests"]}
    rows = []
    for gname, scen, role, gen in gens:
        if scen not in r_pasts:
            sc = E_SCENARIOS[scen]
            r_pasts[scen] = sim_windows(sc["P_past"], n,
                                        common.keyed_rng(ROOT_SEED, sc["seed"], n, rep))
        r_past = r_pasts[scen]
        G, groups = _e_generator(gname, gen, n, rep, r_past)
        for method in E_METHODS:
            trng = common.keyed_rng(ROOT_SEED, "e_test", n, rep, gname, method)
            res, dt, err = common.safe_test(relative_test, r_new, G, r_past,
                                            base=CriticTest(**learned), rng=trng,
                                            method=method, n_boot=cfg["n_boot"], **groups)
            row = {"G": gname, "scenario": scen, "G_role": role, "method": method,
                   "seconds": dt, "error": err}
            if res is None:
                row.update(pvalue=np.nan, statistic=np.nan,
                           ptype="permutation" if method == "anchored" else "paired_bootstrap")
            else:
                ex = res.extra
                ptype = ex.get("pvalue_type") or ex.get("method", "")
                row.update(pvalue=float(res.pvalue), statistic=float(res.statistic),
                           ptype=ptype, split=ex.get("split", ""), D_g=ex.get("D_g", np.nan),
                           D_past=ex.get("D_past", np.nan), z=ex.get("z", np.nan),
                           p_z=ex.get("p_z", np.nan), p_perm_stud=ex.get("p_perm_stud", np.nan),
                           p_boot=ex.get("p_boot", np.nan),
                           n_train_g=ex.get("n_train_g", np.nan),
                           n_eval_g=ex.get("n_eval_g", np.nan),
                           n_train_past=ex.get("n_train_past", np.nan),
                           n_eval_past=ex.get("n_eval_past", np.nan))
            rows.append(row)
    return rows


def _e_config(cfg: dict) -> dict:
    return {"seed": ROOT_SEED, "L": L, "burn": BURN, "P_new": E_P_NEW,
            "scenarios": {k: {"P_past": v["P_past"], "seed": v["seed"]}
                          for k, v in E_SCENARIOS.items()},
            "gens": [list(g) for g in E_GENS], "methods": list(E_METHODS),
            "base": test_settings(cfg["e_tests"])["critic"], "n_boot": cfg["e_n_boot"],
            **code_keys("models", "tests_e")}


def run_e(ctx: common.ExpContext, cfg: dict) -> dict[str, float]:
    cells = [{"cell": common.cell_key(n=n), "n": n,
              "_cfg": {"tests": cfg["e_tests"], "n_boot": cfg["e_n_boot"]}}
             for n in cfg["e_n_list"]]
    t0 = time.perf_counter()
    df = common.run_cells(_e_rep, cells, part="e", ctx=ctx, root_seed=ROOT_SEED,
                          n_reps=cfg["e_reps"], config=_e_config(cfg),
                          block_size=cfg["blocks"]["e"], cost=lambda c: 1.0 + c["n"] / 2000.0)
    timing = {"e_s": time.perf_counter() - t0}
    conf = {"part": "e", **_e_config(cfg), "n_list": cfg["e_n_list"], "reps": cfg["e_reps"]}
    ctx.save_csv(df, "e2_e_raw", config=conf)
    ctx.write_part_meta("e", conf, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Part f: learned vs exact log-likelihood ratio
# --------------------------------------------------------------------------------------------
F_COMPARISONS = {
    "gbm_merton": {"label": "GBM vs Merton", "P": spec_merton(), "Q": spec_gbm(),
                   "P_label": "Merton (real)", "Q_label": "GBM (synthetic)"},
    "garch98_garch90": {"label": "GARCH 0.98 vs 0.90", "P": spec_garch(0.98),
                        "Q": spec_garch(0.90), "P_label": "GARCH 0.98 (real)",
                        "Q_label": "GARCH 0.90 (synthetic)"},
}


F_SCATTER_VARIANTS = ("default", "fixed")  # LR-critic variants drawn in e2_llr_scatter.pdf


def f_logit_test(tc: dict, overrides: dict) -> C2ST:
    """LR-critic of part f: the test defaults, quick-mode overrides ``tc``, then ``overrides``."""
    kw = {k: tc[k] for k in ("n_steps", "n_perm") if k in tc}
    kw.update(overrides)
    return C2ST(statistic="logit", **kw)


def _tex_lr(lr: float) -> str:
    e = math.log10(lr)
    return f"$10^{{{int(round(e))}}}$" if abs(e - round(e)) < 1e-9 else f"{lr:g}"


def f_variant_label(t: C2ST, tex: bool = False) -> str:
    """Readable description of an LR-critic training budget."""
    lr = _tex_lr(t.lr) if tex else f"{t.lr:g}"
    if t.early_stopping:
        return f"early stopping (test default; max {t.n_steps} steps, lr {lr})"
    return f"{t.n_steps} steps, lr {lr}, no early stopping"


def _score_metrics(score: np.ndarray, exact: np.ndarray, cls: np.ndarray,
                   calibrated: bool) -> dict[str, float]:
    """Agreement of a learned score with the exact log p/q on the fresh windows.

    ``cal_slope`` / ``cal_intercept``: OLS of the score on the exact value (1 and 0 for a
    calibrated log-likelihood ratio; the exact value has no noise, so the slope is not attenuated
    towards 0 by a noisy score). ``rmse_raw`` and ``acc`` (the rule ``score > 0``) only for
    scores on the log-likelihood-ratio scale (LR-critic logits)."""
    def corr(fn: Any, x: np.ndarray, y: np.ndarray) -> float:
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            return float("nan")
        return float(fn(x, y).statistic)

    slope, icpt = np.polyfit(exact, score, 1)
    pe = corr(stats.pearsonr, score, exact)
    out = {"spearman": corr(stats.spearmanr, score, exact), "pearson": pe, "r2": pe ** 2,
           "cal_slope": float(slope), "cal_intercept": float(icpt),
           "spearman_P": corr(stats.spearmanr, score[cls == 1], exact[cls == 1]),
           "spearman_Q": corr(stats.spearmanr, score[cls == 0], exact[cls == 0]),
           "rmse_raw": np.nan, "acc": np.nan}
    if calibrated:
        out["rmse_raw"] = float(np.sqrt(np.mean((score - exact) ** 2)))
        out["acc"] = float(np.mean((score > 0) == (cls == 1)))
    return out


def _f_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    del rng
    comp = F_COMPARISONS[cell["comparison"]]
    cfg = cell["_cfg"]
    n_tr, n_fr, K = cfg["n_train"], cfg["n_fresh"], cfg["K"]
    key = (cell["comparison"], rep)
    X_tr = sim_windows(comp["P"], n_tr, common.keyed_rng(ROOT_SEED, "f_Xtr", *key))
    Y_tr = sim_windows(comp["Q"], n_tr, common.keyed_rng(ROOT_SEED, "f_Ytr", *key))
    h = n_fr // 2
    Z = np.concatenate([sim_windows(comp["P"], h, common.keyed_rng(ROOT_SEED, "f_ZP", *key)),
                        sim_windows(comp["Q"], n_fr - h,
                                    common.keyed_rng(ROOT_SEED, "f_ZQ", *key))])
    cls = np.r_[np.ones(h, dtype=int), np.zeros(n_fr - h, dtype=int)]
    memo: dict = {}
    exact = logdens(comp["P"], Z, K, memo) - logdens(comp["Q"], Z, K, memo)
    common_cols = {
        "sd_exact": float(exact.std()), "mean_exact_P": float(exact[cls == 1].mean()),
        "mean_exact_Q": float(exact[cls == 0].mean()),
        "acc_oracle": float(np.mean((exact > 0) == (cls == 1))),
        "exact_type": "exact" if not _persistent(comp["P"]) else
        (f"mc_integrated_K{K}" if HAS_MC_INIT else "approximate_fixed_start")}
    rows: list[dict[str, Any]] = []
    scores: dict[str, np.ndarray] = {}
    # LR-critic under each training budget, all on the same data; the fit seed is shared by
    # the variants (same initialisation and validation split), so only the budget differs.
    for v, ov in cfg["variants"].items():
        t = f_logit_test(cfg["tests"], ov)
        t0 = time.perf_counter()
        logit = t.fit_logit(X_tr, Y_tr, rng=common.keyed_rng(ROOT_SEED, "f_fit", *key))
        dt = time.perf_counter() - t0
        info = getattr(logit, "info", {})
        lhat = np.asarray(logit(Z), dtype=float)
        scores[f"logit_{v}"] = lhat
        rows.append({"score": "lrcritic", "variant": v, "settings": f_variant_label(t),
                     **_score_metrics(lhat, exact, cls, calibrated=True),
                     "steps_run": info.get("steps_run", np.nan), **train_diagnostics(info),
                     "fit_seconds": dt, **common_cols})
    t0 = time.perf_counter()
    critic, cinfo = CriticTest(**{k: cfg["tests"][k] for k in ("n_steps", "n_perm")
                                  if k in cfg["tests"]}).fit_critic(
        X_tr, Y_tr, rng=common.keyed_rng(ROOT_SEED, "f_critic", *key))
    dt = time.perf_counter() - t0
    fz = np.asarray(critic(Z), dtype=float)
    scores["critic"] = fz
    rows.append({"score": "critic", "variant": "default",
                 "settings": f"{cinfo.get('steps_run', '')} steps (test default)",
                 **_score_metrics(fz, exact, cls, calibrated=False),
                 "steps_run": cinfo.get("steps_run", np.nan), **train_diagnostics(cinfo),
                 "fit_seconds": dt, **common_cols})
    if rep == 0 and cell.get("_scatter_dir"):
        sc = pd.DataFrame({"comparison": cell["comparison"], "cls": cls, "exact": exact,
                           **scores})
        path = Path(cell["_scatter_dir"]) / f"scatter_{cell['comparison']}.csv"
        sc.to_csv(str(path) + f".{os.getpid()}.tmp", index=False)
        os.replace(str(path) + f".{os.getpid()}.tmp", path)
    return rows


def _f_config(cfg: dict) -> dict:
    return {"seed": ROOT_SEED, "L": L, "burn": BURN,
            "comparisons": {k: {"P": v["P"], "Q": v["Q"]} for k, v in F_COMPARISONS.items()},
            "n_train": cfg["f_n_train"], "n_fresh": cfg["f_n_fresh"], "K": cfg["f_K"],
            "lrcritic": {v: _attrs(f_logit_test(cfg["f_tests"], ov))
                         for v, ov in cfg["f_variants"].items()},
            "critic": test_settings(cfg["f_tests"])["critic"], "mc_init": HAS_MC_INIT,
            **code_keys("models", "tests_f")}


def run_f(ctx: common.ExpContext, cfg: dict) -> dict[str, float]:
    fc = _f_config(cfg)
    scatter_dir = common.Checkpoint(ctx.ckpt_dir, "f_scatter", fc).dir
    cells = [{"cell": common.cell_key(comparison=c), "comparison": c,
              "_cfg": {"tests": cfg["f_tests"], "n_train": cfg["f_n_train"],
                       "n_fresh": cfg["f_n_fresh"], "K": cfg["f_K"],
                       "variants": cfg["f_variants"]},
              "_scatter_dir": str(scatter_dir)} for c in F_COMPARISONS]
    if ctx.fresh:
        for c in F_COMPARISONS:
            (scatter_dir / f"scatter_{c}.csv").unlink(missing_ok=True)
    t0 = time.perf_counter()
    df = common.run_cells(_f_rep, cells, part="f", ctx=ctx, root_seed=ROOT_SEED,
                          n_reps=cfg["f_reps"], config=fc, block_size=cfg["blocks"]["f"])
    timing = {"f_s": time.perf_counter() - t0}
    conf = {"part": "f", **fc, "reps": cfg["f_reps"]}
    ctx.save_csv(df, "e2_f_raw", config=conf)
    scat = [pd.read_csv(scatter_dir / f"scatter_{c}.csv") for c in F_COMPARISONS
            if (scatter_dir / f"scatter_{c}.csv").exists()]
    if scat:
        ctx.save_csv(pd.concat(scat, ignore_index=True), "e2_llr_scatter", config=conf)
    ctx.write_part_meta("f", conf, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Part g: dependent windows (cut from long GARCH paths), H0 true
# --------------------------------------------------------------------------------------------
G_SPEC = spec_garch(0.98)
G_VARIANTS = ("critic", "lrcritic", "mmd", "sigmmd")


def _g_tests(tc: dict, time_blocks: int) -> list[tuple[str, Any]]:
    learned = {k: tc[k] for k in ("n_steps", "n_perm") if k in tc}
    mmd = {"n_perm": tc["mmd_n_perm"]} if "mmd_n_perm" in tc else {}
    return [
        ("critic", CriticTest(**learned)),
        ("critic", CriticTest(time_blocks=time_blocks, inference="hac", **learned)),
        ("lrcritic", C2ST(statistic="logit", **learned)),
        ("lrcritic", C2ST(statistic="logit", time_blocks=time_blocks, inference="hac",
                          **learned)),
        ("mmd", MMDTest(**mmd)),
        ("mmd", MMDTest(inference="wild", **mmd)),
        ("sigmmd", SigMMDTest(**mmd)),
        ("sigmmd", SigMMDTest(inference="wild", **mmd)),
    ]


def _g_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    del rng
    n = int(cell["n"])
    cfg = cell["_cfg"]
    X = sim_long_path_windows(G_SPEC, n, common.keyed_rng(ROOT_SEED, "g_X", n, rep))
    Y = sim_long_path_windows(G_SPEC, n, common.keyed_rng(ROOT_SEED, "g_Y", n, rep))
    order = np.arange(n)
    rows = []
    for name, test in _g_tests(cfg["tests"], cfg["time_blocks"]):
        pt = _ptype(test)
        trng = common.keyed_rng(ROOT_SEED, "g_test", n, rep, name, pt)
        res, dt, err = common.safe_test(test, X, Y, rng=trng, order_x=order, order_y=order)
        row = _test_row(name, test, res, dt, err)
        row["split"] = (res.extra.get("split", "") if res is not None else "")
        rows.append(row)
        if pt == "hac":  # same fit: the permutation p-value under the time-block split
            r2 = dict(row)
            r2.update(ptype="timeblock_permutation", seconds=0.0,
                      pvalue=row.get("p_perm", np.nan))
            rows.append(r2)
    return rows


def _g_config(cfg: dict) -> dict:
    return {"seed": ROOT_SEED, "L": L, "burn": BURN, "spec": G_SPEC,
            "time_blocks": cfg["g_time_blocks"],
            "tests": [(nm, _ptype(t), _attrs(t))
                      for nm, t in _g_tests(cfg["g_tests"], cfg["g_time_blocks"])],
            **code_keys("models", "tests")}


def run_g(ctx: common.ExpContext, cfg: dict) -> dict[str, float]:
    cells = [{"cell": common.cell_key(n=cfg["g_n"]), "n": cfg["g_n"],
              "_cfg": {"tests": cfg["g_tests"], "time_blocks": cfg["g_time_blocks"]}}]
    t0 = time.perf_counter()
    df = common.run_cells(_g_rep, cells, part="g", ctx=ctx, root_seed=ROOT_SEED,
                          n_reps=cfg["g_reps"], config=_g_config(cfg),
                          block_size=cfg["blocks"]["g"])
    timing = {"g_s": time.perf_counter() - t0}
    conf = {"part": "g", **_g_config(cfg), "n": cfg["g_n"], "reps": cfg["g_reps"]}
    ctx.save_csv(df, "e2_g_raw", config=conf)
    ctx.write_part_meta("g", conf, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Figures and tables (from the saved CSVs only)
# --------------------------------------------------------------------------------------------
TEST_ORDER = {t: i for i, t in enumerate(ALL_TESTS)}
LEARNED_TESTS = ("critic", "lrcritic", "c2st")
PTYPE_LABEL = {"permutation": "permutation (random split)",
               "timeblock_permutation": "time-block split, permutation",
               "hac": "time-block split, HAC", "wild": "wild bootstrap",
               "asymptotic": "asymptotic", "monte_carlo": "Monte Carlo",
               "paired_bootstrap": "paired bootstrap"}
PTYPE_SHORT = {"permutation": "perm.", "asymptotic": "asympt.", "monte_carlo": "MC",
               "hac": "HAC", "wild": "wild", "timeblock_permutation": "t.-b. perm.",
               "paired_bootstrap": "paired boot."}
ALPHA_COLS = {a: rf"$\alpha={a:g}$" for a in ALPHAS}
MARK_KS, MARK_ORACLE, MARK_NAN = r"$^{\dagger}$", r"$^{\ddagger}$", r"$^{\ast}$"


def _tex_alpha_label(a: float) -> str:
    return ALPHA_COLS[a]


def _ptype_label(test: str, ptype: str) -> str:
    """p-value label; "(random split)" only for learned tests (MMD has no train/eval split)."""
    if ptype == "permutation" and test not in LEARNED_TESTS:
        return "permutation"
    return PTYPE_LABEL.get(ptype, ptype)


def _oracle_is_exact(otype: Any) -> bool:
    return (not isinstance(otype, str)) or otype in ("", "exact")


def _marks(test: str, fam: str, otype: Any = "", n_nan: int = 0) -> str:
    """Footnote marks of a (test, family) cell: KS not valid, approximate / MC oracle, failures."""
    out = ""
    if test == "ks" and fam in KS_INVALID_FAMILIES:
        out += MARK_KS
    if test == "oracle" and not _oracle_is_exact(otype):
        out += MARK_ORACLE
    if n_nan > 0:
        out += MARK_NAN
    return out


def _warn_failures(tbl: pd.DataFrame, what: str, by: list[str]) -> int:
    """Log every cell with failed replications (NaN p-values); return their total number."""
    bad = tbl[tbl["n_nan"] > 0]
    for _, r in bad.iterrows():
        logger.warning("%s: %d of %d replications failed in %s (excluded from the rate)", what,
                       int(r["n_nan"]), int(r["n_nan"] + r["n_valid"]),
                       {k: r[k] for k in by})
    return int(bad["n_nan"].sum())


def save_note(ctx: common.ExpContext, name: str, sentences: list[str]) -> Path:
    """Write ``<tab_dir>/<name>.tex``: a generated table note (no hand-typed numbers)."""
    ctx.tab_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.tab_dir / f"{name}.tex"
    text = "% generated by experiments/e2_ground_truth.py\n" + "\n".join(sentences) + "\n"
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def _ks_size_sentence(tbl: pd.DataFrame) -> str:
    """Measured size of KS in the families where it is not valid (from the H0 cells)."""
    parts = []
    for fam in KS_INVALID_FAMILIES:
        s = tbl[(tbl["family"] == fam) & (tbl["k"] == 0) & (tbl["test"] == "ks")]
        for _, r in s.sort_values("n").iterrows():
            parts.append(f"$n={int(r['n'])}$: "
                         + common.fmt_rate_ci(r["rej_0.05"], r["lo_0.05"], r["hi_0.05"]))
    fams = ", ".join(f"({f})" for f in KS_INVALID_FAMILIES)
    msg = (f"{MARK_KS} KS on pooled returns assumes i.i.d. returns; within a GARCH window the "
           f"returns share the conditional variance, so KS is not a valid test in family {fams}")
    if parts:
        msg += " (rejection rate under $H_0$ at nominal 5\\%: " + "; ".join(parts) + ")"
    return msg + "; its power there is not size-adjusted."


def _oracle_sentence(df: pd.DataFrame) -> str:
    otypes = (df[df["test"] == "oracle"].groupby("family")["oracle_type"].first()
              if "oracle_type" in df else pd.Series(dtype=str))
    out = []
    for fam, ot in otypes.items():
        if _oracle_is_exact(ot):
            continue
        if ot.startswith("mc_integrated_K"):
            k = ot.removeprefix("mc_integrated_K")
            out.append(f"{MARK_ORACLE} Family ({fam}): the oracle's GARCH window density "
                       f"integrates over the conditional variance at the window start by Monte "
                       f"Carlo, $K={k}$ draws of its law after the 250-day burn-in "
                       f"(near-exact; the Monte Carlo test stays exact).")
        else:
            out.append(f"{MARK_ORACLE} Family ({fam}): approximate oracle (window density with "
                       f"a fixed start variance), not a strict power bound.")
    return " ".join(out)


def _main_ptype_sentence(df: pd.DataFrame) -> str:
    n_null = int(df.loc[df["test"] == "oracle", "n_null"].max()) if "n_null" in df else 0
    mc = f"Monte Carlo with {n_null} null samples" if n_null else "Monte Carlo"
    return ("$p$-values: permutation of the held-out evaluation labels with the network fixed "
            "(W-critic, LR-critic, C2ST; exact for independent windows), permutation (MMD, "
            f"Sig-MMD), asymptotic (KS), {mc} with randomised ties (LR oracle).")


def build_main_outputs(ctx: common.ExpContext, cfg: dict) -> None:
    raws = [ctx.load_csv(f"e2_main_{f}_raw", dtype={"level_label": str})
            for f in FAMILIES if ctx.has_csv(f"e2_main_{f}_raw")]
    if not raws:
        logger.info("no main results yet: skipping e2_power / e2_size")
        return
    df = pd.concat(raws, ignore_index=True)
    by = ["family", "n", "k", "level", "level_label", "test", "ptype"]
    tbl = common.rejection_table(df, by, alphas=ALPHAS)
    tbl["test_order"] = tbl["test"].map(TEST_ORDER)
    tbl = tbl.sort_values(["family", "n", "k", "test_order"]).drop(columns="test_order")
    tbl["oracle_type"] = ""
    if "oracle_type" in df:
        ot = df[df["test"] == "oracle"].groupby("family")["oracle_type"].first()
        tbl["oracle_type"] = np.where(tbl["test"] == "oracle", tbl["family"].map(ot), "")
    tbl["ks_invalid"] = (tbl["test"] == "ks") & tbl["family"].isin(KS_INVALID_FAMILIES)
    tbl = tbl.reset_index(drop=True)
    n_fail = _warn_failures(tbl, "main", ["family", "n", "k", "test"])
    ctx.save_csv(tbl, "e2_rejection")
    tbl["marks"] = [_marks(t, f, o, int(nn)) for t, f, o, nn in
                    zip(tbl["test"], tbl["family"], tbl["oracle_type"], tbl["n_nan"])]
    nan_note = ([f"{MARK_NAN} Some replications failed (NaN $p$-value) and are excluded; see "
                 "\\texttt{n\\_nan} in \\texttt{e2\\_rejection.csv}."] if n_fail else [])

    # Size table (k = 0, every test except the oracle, whose statistic is identically 0 there).
    size = tbl[(tbl["k"] == 0) & (tbl["test"] != "oracle")].copy()
    ctx.save_csv(size.drop(columns="marks").reset_index(drop=True), "e2_size")
    size["Family"] = size["family"].map(lambda f: FAMILIES[f]["title"])
    size["Test"] = [plotting.LABELS_TEST[t] + m for t, m in zip(size["test"], size["marks"])]
    size["$n$"] = size["n"]
    st = common.rate_ci_table(size, ["Family", "$n$", "Test"], ALPHAS, label=_tex_alpha_label)
    st.insert(0, "$N$", size["n_valid"].astype(int).to_numpy())
    st.insert(0, "$p$-value", [PTYPE_SHORT.get(p, p) for p in size["ptype"]])
    ctx.save_table(st, "e2_size", escape=False)
    save_note(ctx, "e2_size_note", [
        "Rejection rates under $H_0$ ($Q=P$) with Wilson 95\\% intervals; $N$ = valid "
        "replications.", _main_ptype_sentence(df), _ks_size_sentence(tbl)] + nan_note)

    # Power table: rejection rate at 5% (Wilson CIs in e2_power_ci.tex and e2_rejection.csv).
    pw = tbl.copy()
    pw["Family"] = pw["family"].map(lambda f: FAMILIES[f]["title"])
    pw["Level"] = [FAMILIES[f]["tex_ticks"][int(k)] for f, k in zip(pw["family"], pw["k"])]
    pw["$n$"] = pw["n"]
    pw["cell"] = [("--" if not np.isfinite(r) else f"{r:.2f}") + m
                  for r, m in zip(pw["rej_0.05"], pw["marks"])]
    wide = pw.pivot_table(index=["Family", "Level", "$n$", "k"], columns="test",
                          values="cell", aggfunc="first")
    wide = wide.reset_index().sort_values(["Family", "$n$", "k"])
    wide = wide.set_index(["Family", "$n$", "Level"]).drop(columns="k")
    wide = wide[[t for t in ALL_TESTS if t in wide.columns]].fillna("--")
    wide.columns = [plotting.LABELS_TEST[t] for t in wide.columns]
    ctx.save_table(wide, "e2_power", escape=False)
    h1 = tbl[tbl["k"] > 0]
    half = float(np.nanmax((h1["hi_0.05"] - h1["lo_0.05"]) / 2)) if len(h1) else float("nan")
    reps = tbl.groupby(tbl["k"] == 0)["n_valid"].max()
    save_note(ctx, "e2_power_note", [
        f"Rejection rates at $\\alpha=0.05$ over {int(reps.get(False, 0))} replications per "
        f"alternative ({int(reps.get(True, 0))} under $H_0$); Wilson 95\\% intervals in the "
        f"appendix table (largest half-width {half:.2f}).", _main_ptype_sentence(df),
        _ks_size_sentence(tbl), _oracle_sentence(df)] + nan_note)

    # Appendix table: every rate with its Wilson interval (columns: grid index k).
    ci = tbl.copy()
    ci["Family"] = [f"{FAMILIES[f]['title']} ({FAMILIES[f]['param']}: "
                    + ", ".join(FAMILIES[f]["tex_ticks"]) + ")" for f in ci["family"]]
    ci["Test"] = [plotting.LABELS_TEST[t] + m for t, m in zip(ci["test"], ci["marks"])]
    ci["$n$"] = ci["n"]
    ci["kcol"] = [f"$k={int(k)}$" + (" ($H_0$)" if k == 0 else "") for k in ci["k"]]
    ci["cell"] = [common.fmt_rate_ci(r, lo, hi, 2) for r, lo, hi in
                  zip(ci["rej_0.05"], ci["lo_0.05"], ci["hi_0.05"])]
    ci["test_order"] = ci["test"].map(TEST_ORDER)
    wci = ci.pivot_table(index=["family", "Family", "n", "$n$", "test_order", "Test"],
                         columns="kcol", values="cell", aggfunc="first")
    wci = wci.reset_index().sort_values(["family", "n", "test_order"])
    wci = wci.drop(columns=["family", "n", "test_order"]).set_index(["Family", "$n$", "Test"])
    wci = wci[sorted(wci.columns, key=lambda c: int(c.split("=")[1].split("$")[0]))].fillna("--")
    wci.columns.name = None
    ctx.save_table(wci, "e2_power_ci", escape=False)
    plot_power(ctx, tbl)


def plot_power(ctx: common.ExpContext, tbl: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plotting.setup()
    fig, axes = plt.subplots(2, 2, figsize=plotting.figsize("two", 0.78), sharey=True)
    n_vals = sorted(tbl["n"].unique())
    styles = {n: s for n, s in zip(n_vals, ["-", "--", ":", "-."])}
    # Each (test, n) series is shifted slightly along x so that the Wilson error bars of the
    # different series do not overlap; the oracle (the benchmark) stays on the grid.
    series = [(t, n) for t in ALL_TESTS if t != "oracle" for n in n_vals]
    width = 0.3
    dodge = {sn: (i - (len(series) - 1) / 2) * width / max(1, len(series) - 1)
             for i, sn in enumerate(series)}
    dodge.update({("oracle", n): 0.0 for n in n_vals})
    for ax, (fam, F) in zip(axes.ravel(), FAMILIES.items()):
        sub = tbl[tbl["family"] == fam]
        ax.axhline(0.05, color=plotting.OKABE_ITO["grey"], lw=0.6, ls=":")
        if sub.empty:
            ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center")
        for test in ALL_TESTS:
            invalid = test == "ks" and fam in KS_INVALID_FAMILIES
            col = plotting.COLORS_TEST[test]
            for n in n_vals:
                s = sub[(sub["test"] == test) & (sub["n"] == n)].sort_values("k")
                if s.empty:
                    continue
                is_or = test == "oracle"
                x = s["k"].to_numpy(dtype=float) + dodge[(test, n)]
                y = s["rej_0.05"].to_numpy(dtype=float)
                yerr = np.vstack([y - s["lo_0.05"].to_numpy(dtype=float),
                                  s["hi_0.05"].to_numpy(dtype=float) - y])
                a = 0.35 if invalid else 1.0
                # Wilson 95% interval of each rejection rate.
                ax.errorbar(x, y, yerr=np.clip(yerr, 0, None), fmt="none", ecolor=col,
                            elinewidth=0.6, capsize=0, alpha=0.5 * a, zorder=1)
                ax.plot(x, y, color=col, marker=plotting.MARKERS_TEST[test],
                        ls=styles[n], lw=1.4 if is_or else 1.0, ms=4.5 if is_or else 3.2,
                        alpha=a, zorder=3 if is_or else 2)
        notes = []
        if fam in KS_INVALID_FAMILIES and not sub.empty:
            notes.append("faded KS: not valid (oversized)")
        ot = sub.loc[sub["test"] == "oracle", "oracle_type"]
        if len(ot) and not _oracle_is_exact(ot.iloc[0]):
            o = ot.iloc[0]
            notes.append("oracle: MC density, $K=" + o.removeprefix("mc_integrated_K")
                         + "$" if o.startswith("mc_integrated_K") else "oracle: approximate")
        if notes:
            ax.set_title("\n".join(notes), loc="right", fontsize=6.5)
        ax.set_xticks(range(5), F["ticks"])
        ax.set_xlabel(F["xlabel"])
        ax.set_title(F["title"], loc="left")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.3, 4.3)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"rejection rate at $\alpha=0.05$")
    handles = [Line2D([], [], color=plotting.COLORS_TEST[t], marker=plotting.MARKERS_TEST[t],
                      ls="-", label=plotting.LABELS_TEST[t]) for t in ALL_TESTS]
    handles += [Line2D([], [], color="black", ls=styles[n], label=f"$n={n}$") for n in n_vals]
    handles += [Line2D([], [], color=plotting.OKABE_ITO["grey"], lw=0.6, marker="|", ms=6,
                       ls="", label="Wilson 95% CI")]
    ncol = math.ceil(len(handles) / 2)
    # Legends fill column by column: interleave so that each row reads left to right.
    rows = [handles[:ncol], handles[ncol:]]
    order = [h for j in range(ncol) for h in (rows[0][j:j + 1] + rows[1][j:j + 1])]
    fig.legend(handles=order, loc="outside lower center", ncol=ncol, frameon=False)
    ctx.savefig(fig, "e2_power")


def build_e_outputs(ctx: common.ExpContext, cfg: dict) -> None:
    if not ctx.has_csv("e2_e_raw"):
        return
    df = ctx.load_csv("e2_e_raw")
    if "scenario" not in df:  # raw file of an older script version: no scenario column
        logger.warning("e2_e_raw.csv has no scenario column (older version): rerun part e")
        return
    by = ["scenario", "G", "G_role", "method", "ptype", "n"]
    tbl = common.rejection_table(df, by, alphas=ALPHAS)
    order = {g: i for i, (g, *_rest) in enumerate(E_GENS)}
    mean_delta = df.groupby(["G", "method", "n"])["statistic"].mean().rename("mean_delta")
    tbl = tbl.merge(mean_delta.reset_index(), on=["G", "method", "n"], how="left")
    tbl["gord"] = tbl["G"].map(order)
    tbl = tbl.sort_values(["gord", "n", "method"]).drop(columns="gord").reset_index(drop=True)
    n_fail = _warn_failures(tbl, "part e", ["G", "method", "n"])
    ctx.save_csv(tbl, "e2_relative")
    t = tbl.copy()
    t["Scenario"] = t["scenario"].map(lambda s: E_SCENARIOS[s]["tex"])
    t["Generator $G$"] = t["G"].map(E_TEX)
    t["$n$"] = t["n"]
    for v, a in (("cell", 0.05), ("cell10", 0.10)):
        tag = f"{a:.2f}"
        t[v] = [common.fmt_rate_ci(r, lo, hi) + (MARK_NAN if nn else "") for r, lo, hi, nn in
                zip(t[f"rej_{tag}"], t[f"lo_{tag}"], t[f"hi_{tag}"], t["n_nan"])]
    t["gord"] = t["G"].map(order)
    wide = t.pivot_table(index=["gord", "Scenario", "Generator $G$", "$n$"], columns="method",
                         values=["cell", "cell10"], aggfunc="first")
    wide = wide.reset_index(level="gord", drop=True)
    cols = [(v, m, a) for m in E_METHODS for v, a in (("cell", 0.05), ("cell10", 0.10))
            if (v, m) in wide.columns]
    wide = wide[[(v, m) for v, m, _ in cols]]
    wide.columns = pd.MultiIndex.from_tuples([(E_METHOD_TEX[m], ALPHA_COLS[a])
                                              for _, m, a in cols])
    ctx.save_table(wide, "e2_relative_boundary", escape=False)
    nv = tbl["n_valid"]
    nvs = f"{int(nv.min())}" + (f"--{int(nv.max())}" if nv.max() > nv.min() else "")
    save_note(ctx, "e2_relative_boundary_note", [
        f"Rejection rates of the relative tests (W-critic base) with Wilson 95\\% intervals, "
        f"{nvs} valid replications per cell; independent windows. "
        "Anchored: permutation of the $G$/$R_{\\rm past}$ evaluation labels with the critic "
        "fixed (exact at $\\mathrm{Law}(G)=\\mathrm{Law}(R_{\\rm past})$). Two-critic: centred "
        "paired bootstrap (anti-conservative at that boundary).",
        "Resample rows: $G$ is an i.i.d. resample of the $R_{\\rm past}$ windows (the same $G$ "
        "in both rows). Split by source: every copy is on the train/eval side of its source "
        "window, so no evaluation window of $G$ was seen in training (the test is then "
        "conservative). Random split: evaluation copies of windows the critic saw as training "
        "negatives make $G$ look less like $R_{\\rm new}$ than $R_{\\rm past}$ does (train/eval "
        "leakage); this row is not a size."]
        + ([f"{MARK_NAN} Some replications failed and are excluded."] if n_fail else []))


def build_f_outputs(ctx: common.ExpContext, cfg: dict) -> None:
    if not ctx.has_csv("e2_f_raw"):
        return
    df = ctx.load_csv("e2_f_raw")
    if "score" not in df:  # raw file of an older script version (one row per replication)
        logger.warning("e2_f_raw.csv is from an older version of part f: rerun part f")
        return
    variants = list(cfg["f_variants"])
    rows = []
    for comp in F_COMPARISONS:
        g = df[df["comparison"] == comp]
        if g.empty:
            continue
        keys = [("lrcritic", v) for v in variants] + [("critic", "default")]
        for score, var in keys:
            s = g[(g["score"] == score) & (g["variant"] == var)]
            if s.empty:
                continue

            def med(c: str, full: bool = False, digits: int = 3, sub: pd.DataFrame = s) -> str:
                v = sub[c].to_numpy(dtype=float) if c in sub else np.array([np.nan])
                if not np.isfinite(v).any():
                    return "--"
                m = float(np.nanmedian(v))
                if not full:
                    return f"{m:.{digits}f}"
                return f"{m:.{digits}f} [{np.nanmin(v):.{digits}f}, {np.nanmax(v):.{digits}f}]"

            if score == "lrcritic":
                t = f_logit_test(cfg["f_tests"], cfg["f_variants"][var])
                label = "LR-critic: " + f_variant_label(t, tex=True)
                es = bool(s["early_stopping"].iloc[0]) if "early_stopping" in s else False
                steps = med("best_step", digits=0) if es else med("steps_run", digits=0)
                steps = steps + (" (selected)" if es and steps != "--" else "")
            else:
                label = f"W-critic: {med('steps_run', digits=0)} steps (test default)"
                steps = med("steps_run", digits=0)
            rows.append({"comparison": comp, "score": score, "variant": var,
                         "Comparison": F_COMPARISONS[comp]["label"], "Score": label,
                         "reps": len(s), "Spearman": med("spearman", full=True),
                         "Pearson": med("pearson"), "Cal. slope": med("cal_slope"),
                         "RMSE": med("rmse_raw"), "Accuracy": med("acc"), "Steps": steps,
                         "spearman_median": float(s["spearman"].median()),
                         "pearson_median": float(s["pearson"].median()),
                         "cal_slope_median": float(s["cal_slope"].median()),
                         "best_step_median": float(s["best_step"].median())
                         if "best_step" in s else np.nan,
                         "exact_type": s["exact_type"].iloc[0]})
        o = g[g["score"] == "lrcritic"].drop_duplicates("rep")
        rows.append({"comparison": comp, "score": "oracle", "variant": "",
                     "Comparison": F_COMPARISONS[comp]["label"],
                     "Score": r"exact $\log p/q$ (oracle)", "reps": len(o), "Spearman": "1",
                     "Pearson": "1", "Cal. slope": "1", "RMSE": "0",
                     "Accuracy": f"{float(o['acc_oracle'].median()):.3f}", "Steps": "--",
                     "exact_type": o["exact_type"].iloc[0] if len(o) else ""})
    summ = pd.DataFrame(rows)
    ctx.save_csv(summ, "e2_llr")
    tex = summ.set_index(["Comparison", "Score"])[["Spearman", "Pearson", "Cal. slope", "RMSE",
                                                   "Accuracy", "Steps"]]
    ctx.save_table(tex, "e2_llr", escape=False)
    reps = int(summ.loc[summ["score"] == "lrcritic", "reps"].max()) if len(summ) else 0
    save_note(ctx, "e2_llr_note", [
        f"Median over {reps} replications (Spearman: median [min, max]) of the agreement between "
        "a learned score and the exact log-likelihood ratio $\\log p(w)/q(w)$ on fresh windows "
        "(half from each law). Cal. slope: OLS slope of the score on the exact value (1 for a "
        "calibrated log-likelihood ratio). RMSE and Accuracy (rule score $>0$) only for the "
        "LR-critic logit, whose optimum with balanced classes is $\\log p/q$; the W-critic is "
        "identified only up to scale and shift. Steps: early-stopping checkpoint selected "
        "(median) or steps run. GARCH comparison: window density integrated over the start "
        "variance by Monte Carlo."])
    if ctx.has_csv("e2_llr_scatter"):
        plot_llr(ctx, ctx.load_csv("e2_llr_scatter"), df, cfg)


def plot_llr(ctx: common.ExpContext, sc: pd.DataFrame, raw: pd.DataFrame, cfg: dict) -> None:
    """Raw LR-critic logit against the exact log p/q (replication 0), identity line."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plotting.setup()
    comps = [c for c in F_COMPARISONS if c in set(sc["comparison"])]
    variants = [v for v in F_SCATTER_VARIANTS if f"logit_{v}" in sc]
    if not comps or not variants:
        return
    fig, axes = plt.subplots(len(variants), len(comps), squeeze=False,
                             figsize=plotting.figsize("two", 0.40 * len(variants)))
    col_p, col_q = plotting.OKABE_ITO["vermillion"], plotting.OKABE_ITO["blue"]
    for i, var in enumerate(variants):
        vlabel = f_variant_label(f_logit_test(cfg["f_tests"], cfg["f_variants"][var]))
        for ax, comp in zip(axes[i], comps):
            s = sc[sc["comparison"] == comp]
            meta = F_COMPARISONS[comp]
            y = s[f"logit_{var}"].to_numpy(dtype=float)
            x = s["exact"].to_numpy(dtype=float)
            for cls, col, lab in ((1, col_p, meta["P_label"]), (0, col_q, meta["Q_label"])):
                m = s["cls"].to_numpy() == cls
                ax.scatter(x[m], y[m], s=3, alpha=0.35, color=col, lw=0, label=lab,
                           rasterized=True)
            xlo, xhi = (float(q) for q in np.quantile(x, [0.002, 0.998]))
            ylo, yhi = (float(q) for q in np.quantile(np.r_[y, xlo, xhi], [0.002, 0.998]))
            ylo, yhi = min(ylo, xlo), max(yhi, xhi)
            ax.plot([xlo, xhi], [xlo, xhi], color="black", lw=0.9, ls="--", label="identity")
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            r0 = raw[(raw["comparison"] == comp) & (raw["rep"] == 0)
                     & (raw["score"] == "lrcritic") & (raw["variant"] == var)]
            if not r0.empty:
                ax.text(0.03, 0.97, f"Spearman {r0['spearman'].iloc[0]:.3f}\n"
                        f"calibration slope {r0['cal_slope'].iloc[0]:.2f}",
                        transform=ax.transAxes, va="top", ha="left")
            ax.set_title(f"{meta['label']}; LR-critic, {vlabel}", loc="left", fontsize=7.5)
            ax.set_xlabel(r"exact $\log p(w)/q(w)$")
            ax.set_ylabel(r"learned logit $\hat\ell(w)$")
            handles = [Line2D([], [], color=col_p, marker="o", ls="", ms=3,
                              label=meta["P_label"]),
                       Line2D([], [], color=col_q, marker="o", ls="", ms=3,
                              label=meta["Q_label"]),
                       Line2D([], [], color="black", ls="--", label="identity")]
            ax.legend(handles=handles, loc="lower right")
    ctx.savefig(fig, "e2_llr_scatter")


def build_g_outputs(ctx: common.ExpContext, cfg: dict) -> None:
    if not ctx.has_csv("e2_g_raw"):
        return
    df = ctx.load_csv("e2_g_raw")
    tbl = common.rejection_table(df, ["test", "ptype", "n"], alphas=ALPHAS)
    porder = {"permutation": 0, "timeblock_permutation": 1, "hac": 2, "wild": 3}
    tbl["o1"] = tbl["test"].map(TEST_ORDER)
    tbl["o2"] = tbl["ptype"].map(porder)
    tbl = tbl.sort_values(["o1", "o2"]).drop(columns=["o1", "o2"]).reset_index(drop=True)
    n_fail = _warn_failures(tbl, "part g", ["test", "ptype"])
    ctx.save_csv(tbl, "e2_dependent")
    t = tbl.copy()
    t["Test"] = t["test"].map(plotting.LABELS_TEST)
    t["$p$-value"] = [_ptype_label(te, p) + (MARK_NAN if nn else "")
                      for te, p, nn in zip(t["test"], t["ptype"], t["n_nan"])]
    out = common.rate_ci_table(t, ["Test", "$p$-value"], ALPHAS, label=_tex_alpha_label)
    out.insert(0, "$N$", t["n_valid"].astype(int).to_numpy())
    ctx.save_table(out, "e2_dependent_size", escape=False)
    if n_fail:
        save_note(ctx, "e2_dependent_size_note",
                  [f"{MARK_NAN} Some replications failed and are excluded."])


def build_outputs(ctx: common.ExpContext, cfg: dict, parts: list[str] | None = None) -> None:
    """Rebuild figures and tables from the saved CSVs: of ``parts`` (all if None)."""
    builders = [("main", build_main_outputs), ("e", build_e_outputs), ("f", build_f_outputs),
                ("g", build_g_outputs)]
    for key, fn in builders:
        if parts is not None and not any(p == key or p.startswith(f"{key}_") for p in parts):
            continue
        try:
            fn(ctx, cfg)
        except Exception:  # noqa: BLE001 - one broken output must not hide the others
            logger.exception("%s failed", fn.__name__)


# --------------------------------------------------------------------------------------------
# Timing probes (quick mode): a few replications at the FULL settings
# --------------------------------------------------------------------------------------------
def _probe_rep(cell: dict, rep: int, rng: np.random.Generator) -> dict[str, Any]:
    kind = cell["kind"]
    t0 = time.perf_counter()
    if kind == "main":
        _main_rep(cell, rep, rng)
    elif kind == "oracle":
        _oracle_null_chunk({"fam": cell["family"], "b0": 0, "b1": cell["chunk"],
                            "n_list": FULL["n_list"], "K": FULL["oracle_K"]}, rng)
    elif kind == "e":
        _e_rep(cell, rep, rng)
    elif kind == "f":
        _f_rep(cell, rep, rng)
    elif kind == "g":
        _g_rep(cell, rep, rng)
    return {"probe_seconds": time.perf_counter() - t0}


def run_probes(ctx: common.ExpContext, main_fams: list[str]) -> pd.DataFrame:
    """One replication of each heavy cell type at the full configuration (fresh, timing only)."""
    F = FULL
    cells: list[dict] = []
    for fam in main_fams:
        cells.append({"cell": f"oracle_{fam}", "kind": "oracle", "family": fam, "chunk": 4})
        for n in F["n_list"]:
            cells.append({"cell": f"main_{fam}_{n}", "kind": "main", "family": fam, "n": n,
                          "k": 4, "_cfg": _main_cfg(F), "_null_path": ""})
    if "e" in ctx.parts:
        for n in F["e_n_list"]:
            cells.append({"cell": f"e_{n}", "kind": "e", "n": n, "_gens": ["past"],
                          "_cfg": {"tests": F["e_tests"], "n_boot": F["e_n_boot"]}})
    if "f" in ctx.parts:
        cells.append({"cell": "f_garch", "kind": "f", "comparison": "garch98_garch90",
                      "_cfg": {"tests": F["f_tests"], "n_train": F["f_n_train"],
                               "n_fresh": F["f_n_fresh"], "K": F["f_K"],
                               "variants": F["f_variants"]},
                      "_scatter_dir": ""})
    if "g" in ctx.parts:
        cells.append({"cell": "g", "kind": "g", "n": F["g_n"],
                      "_cfg": {"tests": F["g_tests"], "time_blocks": F["g_time_blocks"]}})
    if not cells:
        return pd.DataFrame()
    cost = {"main": 3.0, "e": 4.0, "g": 3.0, "f": 2.0, "oracle": 1.0}
    return common.run_cells(_probe_rep, cells, part="probe", ctx=ctx, root_seed=ROOT_SEED,
                            n_reps=1, config={"probe": True, "full": FULL}, block_size=1,
                            cost=lambda c: cost[c["kind"]], fresh=True)


def report_estimates(probe: pd.DataFrame, main_fams: list[str], parts: list[str],
                     workers: int = 12) -> dict[str, float]:
    """Print measured per-replication times and estimated full-run wall times per part."""
    F = FULL
    est: dict[str, float] = {}
    if probe.empty:
        return est
    t = {r["cell"]: float(r["probe_seconds"]) for _, r in probe.iterrows()}
    all_main_tasks: list[float] = []
    all_oracle = 0.0
    for fam in main_fams:
        oracle_s = t[f"oracle_{fam}"] / 4 * F["oracle_B"]  # 4 null samples were timed
        tasks = []
        for n in F["n_list"]:
            s = t[f"main_{fam}_{n}"]
            common.print_rep_timing(f"main_{fam} n={n}", s, note="full settings, 6 tests + oracle")
            for k in range(5):
                reps = F["reps_h0"] if k == 0 else F["reps_h1"]
                tasks += common.block_tasks(s, reps, F["blocks"]["main"])
        print(f"TIMING oracle null {fam}: {oracle_s:.0f} s CPU for B={F['oracle_B']} samples "
              f"of {max(F['n_list'])} windows (K={F['oracle_K']})", flush=True)
        est[f"main_{fam}"] = common.print_wall_estimate(f"main_{fam}", tasks, workers=workers,
                                                        extra_s=oracle_s / workers)
        all_main_tasks += tasks
        all_oracle += oracle_s
    if len(main_fams) > 1:
        est["main"] = common.print_wall_estimate(f"main ({'+'.join(main_fams)}, one process)",
                                                 all_main_tasks, workers=workers,
                                                 extra_s=all_oracle / workers)
    if "e" in parts:
        tasks = []
        for n in F["e_n_list"]:
            s = t[f"e_{n}"] * len(E_GENS)  # probe ran one generator (both methods)
            common.print_rep_timing(f"e n={n}", s,
                                    note=f"{len(E_GENS)} generators x anchored + two-critic")
            tasks += common.block_tasks(s, F["e_reps"], F["blocks"]["e"])
        est["e"] = common.print_wall_estimate("e", tasks, workers=workers)
    if "f" in parts:
        s = t["f_garch"]
        common.print_rep_timing("f", s, note="GARCH comparison, fit_logit + critic + exact LLR")
        est["f"] = common.print_wall_estimate(
            "f", common.block_tasks(s, F["f_reps"] * len(F_COMPARISONS), 1), workers=workers)
    if "g" in parts:
        s = t["g"]
        common.print_rep_timing("g", s, note="8 test variants")
        est["g"] = common.print_wall_estimate(
            "g", common.block_tasks(s, F["g_reps"], F["blocks"]["g"]), workers=workers)
    return est


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------
def _add_args(parser: Any) -> None:
    parser.add_argument("--no-probe", action="store_true",
                        help="quick mode: skip the full-settings timing probes")


def main(argv: list[str] | None = None) -> None:
    ctx = common.parse_args(EXP, PARTS, description=__doc__, aliases=ALIASES,
                            add_arguments=_add_args, argv=argv)
    cfg = QUICK if ctx.quick else FULL
    if not HAS_MC_INIT:
        logger.warning("GARCHt.stationary_variance_draws not found: the GARCH oracle (family c) "
                       "uses the fixed-start approximation and is labelled 'approximate'")
    sw = common.Stopwatch()
    timing: dict[str, float] = {}
    ctx.write_meta({"quick": ctx.quick, "parts": ctx.parts, "plots_only": ctx.plots_only,
                    "cfg": cfg, "oracle_mc_init": HAS_MC_INIT})
    main_fams = [p.split("_")[1] for p in ctx.parts if p.startswith("main_")]
    if not ctx.plots_only:
        if main_fams:
            timing.update(run_main(ctx, cfg, main_fams))
        if "e" in ctx.parts:
            timing.update(run_e(ctx, cfg))
        if "f" in ctx.parts:
            timing.update(run_f(ctx, cfg))
        if "g" in ctx.parts:
            timing.update(run_g(ctx, cfg))
    sw.lap("compute")
    # Each process rebuilds only the outputs of the parts it ran (run --plots-only once at the
    # end when parts ran in parallel processes); --plots-only rebuilds everything.
    build_outputs(ctx, cfg, None if ctx.plots_only else ctx.parts)
    sw.lap("outputs")
    if ctx.quick and not ctx.plots_only and not ctx.args.no_probe:
        probe = run_probes(ctx, main_fams)
        sw.lap("probes")
        est = report_estimates(probe, main_fams, ctx.parts)
        timing.update({f"estimate_wall_s_{k}": v for k, v in est.items()})
    timing.update({f"wall_{k}_s": v for k, v in sw.laps.items()})
    ctx.write_meta(None, timing=timing)
    print(f"TIMING total wall (this run, {ctx.workers} workers): {common.fmt_seconds(sw.total())}",
          flush=True)


if __name__ == "__main__":
    main()
