"""E1: placebo size of the real-data pipeline (DESIGN section 4, E1; theory.md section 7.6).

Every test runs through :mod:`synthval.pipeline` with the E3 settings: 100-ticker panel, windows
of L = 20 days normalised by the real TRAIN standard deviation of each ticker, TRAIN / TEST date
ranks, ``protocol="dependent"`` (learned tests: 10 contiguous time blocks and the Driscoll-Kraay
HAC p-value; MMD / Sig-MMD: wild bootstrap; KS reported but invalid under dependence), anchored
relative test (primary) and two-critic relative test (secondary), both with the W-critic base.
The other p-values each call computes anyway are recorded too, so every p-value type gets its
own size estimate. Each test is called on its own (same random stream as one joint pipeline
call) inside ``common.safe_test``: a failing test gives a NaN row with its error message (counted
in ``n_nan``) instead of stopping the part.

Sub-experiments (``--part``)
----------------------------
sim (primary, DESIGN E1(a))
    One-factor panel ``FactorGARCHt`` (market factor = equal-weight panel return, GARCH(1,1)-t;
    r_it = a_i + b_i m_t + e_it with GARCH(1,1)-t residuals) fitted once on the real panel TRAIN.
    Each replication simulates one stationary path of TRAIN + TEST length (every recursion
    starts at its unconditional variance; 1000 burn-in days discarded) and cuts it into R_past*
    (the first TRAIN-length days) and R_new* (the rest), so R_past* and R_new* are one
    contiguous history like real TRAIN and TEST, and the last R_past* windows border the first
    R_new* dates (the adjacency the anchored test sees on real data). The synthetic TEST-length
    panel G* is an independent stationary draw, so both nulls hold exactly: Law(G*) =
    Law(R_new*) for the absolute tests and Law(G*) = Law(R_past*) (the boundary) for the
    relative tests. (E3 simulates G from the fitted end-of-TRAIN state, ``start_from_end=True``;
    here every panel starts from stationarity, so G* does not share R_new*'s starting state.)
    Windows exactly as in E3 (``RealPanel.synthetic_windows``: real TRAIN scale, TEST ranks for
    R_new* and G*, TRAIN ranks for R_past*; 3000 / 3000 / 6200 windows). Two cells, sharing
    R_past*, R_new* and the test seeds replication by replication (so compare them with the
    paired McNemar counts of ``e1_sim_paired.csv``, not with overlapping intervals):

    * ``joint``: G* is an independent draw of the whole factor panel (DESIGN E1(a) as written).
    * ``indep``: column i of G* is column i of its own independent panel draw, so each ticker
      has exactly the factor-model marginal law but tickers are independent, as for the
      univariate generators of E3 (the dependence asymmetry of theory.md section 7.6).

    Per replication: absolute tests R_new* vs G* (dependent protocol: HAC / wild, plus the
    time-block permutation and the plain permutation p-values); the learned tests once more under
    the pre-review ticker split with permutation and HAC p-values (``protocol="iid"`` with ticker
    groups; theory.md Prop. 7.5 predicts a bias); anchored and two-critic relative tests
    (R_new*, G*, R_past*); in the ``joint`` cell also R_new* vs R_past* absolute tests (the
    stationary reference for part nonstat); and the cross-sectional dependence of the three
    panels (mean pairwise return correlation, mean pairwise correlation of the rolling 20-day
    log-volatility). 200 replications per cell.
real (secondary, DESIGN E1(b))
    Random halves A / B of the panel tickers, 100 splits within TEST and 100 within TRAIN.
    Absolute tests (dependent protocol) of half A vs half B on the same dates. In the TRAIN
    split also the relative tests with R_new = all real TEST windows, G = TRAIN half B,
    R_past = TRAIN half A (a "generator" equal in law to the past: the boundary). Limits of this
    placebo (theory.md section 7.6): (i) the halves share dates; the HAC variance of the
    difference adds the two halves' variances and ignores their strong positive same-date
    covariance, so the HAC rows are near-vacuous by construction and cannot reveal oversize of
    the HAC p-value; the wild bootstrap uses the same multiplier for both halves on a date and
    does model that covariance, so the wild-bootstrap rows are the informative ones; (ii) a
    trained critic can learn the stock composition of each half (train and eval blocks hold the
    same tickers), which pushes towards rejection; (iii) all splits are random halves of one
    realised market path: rates are conditional on that path and the 100 splits are dependent,
    so binomial (Wilson) intervals do not apply. The LaTeX table shows rates only; the CSV keeps
    the Wilson columns, which assume independent splits.
nonstat (DESIGN E1(c))
    Absolute tests of R_new (real TEST, 3000 windows) vs R_past (real TRAIN, 6200 windows),
    dependent protocol, over 10 test seeds (critic initialisation, block assignment, kernel
    subsample): median statistic, median p, twice the median p (valid over seeds, theory.md
    Remark 8.5) and the share of seeds rejecting; next to the rejection rate of the same
    comparison in the stationary placebo (part sim, cell ``joint``, R_new* vs R_past*).

p-value types (``ptype``): ``hac`` (time-block split, Driscoll-Kraay z, normal reference: the
DESIGN protocol), ``hac_fixedb`` and ``hac_fixedb_eq`` (the same z against fixed-b references,
computed after the run from the stored z, eval-date counts and lags, see
:func:`fixedb_pvalue`), ``timeblock_permutation`` (same critic, permutation of the eval
labels), ``timeblock_perm_stud`` (anchored test, studentised permutation), ``paired_bootstrap``
(two-critic test), ``ticker_permutation`` and ``ticker_hac`` (ticker split), ``wild`` (wild
bootstrap), ``permutation`` (kernel tests), ``asymptotic`` (KS, i.i.d. asymptotics; invalid
under dependence).

Outputs: ``results/e1/e1_{sim,real,nonstat}_raw.csv`` (one row per test call, every p-value
column, ``error`` for failed calls, and the fixed-b inputs ``z_hac``, ``T_eval_x``,
``T_eval_y`` (eval dates per sample), ``H_x``, ``H_y`` (Bartlett lags), ``se_hac``; ``x`` is
the first sample of the statistic: X for absolute tests, R_past for the anchored test),
``e1_size_sim.csv``, ``e1_size_real.csv``, ``e1_nonstat.csv`` (rejection rates at alpha 0.01 /
0.05 / 0.10 with Wilson 95% intervals per test and ``ptype``), ``e1_sim_paired.csv`` (joint vs
indep, paired counts and exact McNemar p-values), ``e1_dependence.csv``, ``e1_factor_fit.csv``
(with the fourth-moment condition ``m4_cond = (a + b)^2 + (kappa - 1) a^2`` of every GARCH
component, kappa the kurtosis of the standardised t; finite fourth moment iff < 1); tables
``paper/tables/e1_size_sim.tex`` (main), ``e1_size_real.tex`` (rates only), ``e1_nonstat.tex``;
figure ``paper/figures/e1_pvalue_qq.pdf`` (QQ plot of the placebo p-values against the uniform,
log axes, p-values below 1e-3 drawn at 1e-3, grey band = pointwise 95% band of uniform order
statistics; solid ``joint``, dashed ``indep``; ``hac_fixedb_eq`` omitted for legibility). In
the tables, learned and relative tests use the time-block split unless "ticker split" is
stated, the protocol's p-value (HAC or wild bootstrap) is starred, and entries are "rate
[Wilson 95% interval]". The CSVs carry ``role`` (primary / secondary / comparison / invalid)
and ``ptype_label``.

Checkpoints are keyed by the part configuration *and* by the code: the SHA-256 of the synthval
sources (``utils.code_fingerprint``), of ``experiments/common.py`` and of this script, so a
resumed run never mixes blocks computed by different code versions. Quick mode injects two
failures (one test call in sim, one whole replication in real) to exercise the NaN-row path;
they show up as ``n_nan = 1`` in the quick tables.

Memory: a full-settings replication peaks at about 0.9 GB per worker in every part (two
6000 x 6000 kernel matrices plus the torch / data / model baseline; real TEST halves 0.55 GB),
plus about 0.45 GB for the main process. 12 workers need about 11 GB: with about 6 GB free run
each part with at most 5 workers. Quick mode prints the measured peak and the worker count that
fits the memory available at that moment; a full run logs a warning when ``--workers`` does not
fit.

Usage::

    uv run python experiments/e1_placebo.py --quick            # smoke test + timing estimate
    uv run python experiments/e1_placebo.py --part sim --workers 5
    uv run python experiments/e1_placebo.py --part real --workers 5
    uv run python experiments/e1_placebo.py --part nonstat --workers 5
    uv run python experiments/e1_placebo.py --plots-only       # rebuild figures and tables
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

from synthval import config, data, plotting, portfolio, utils  # noqa: E402
from synthval import pipeline as pl  # noqa: E402
from synthval.models import get_mv_model  # noqa: E402

logger = logging.getLogger("experiments.e1")

EXP = "e1"
ROOT_SEED = 20251121
PARTS = ("sim", "real", "nonstat")
ALPHAS = (0.01, 0.05, 0.10)
SIM_CELLS = ("joint", "indep")
SIM_BURN_IN = 1000   # stationary start of every placebo panel (0.995^1000 < 0.007)
PERIODS = ("test", "train")
LOGVOL_WINDOW = 20
GARCH_CAP = 0.995    # persistence cap of GARCHt with variance targeting (DESIGN section 3.5)
#: Draws of each fixed-b reference distribution (doubled by symmetry).
FIXEDB_DRAWS = 40_000
FIXEDB_SEED = 20251115
#: Peak resident memory per worker at the FULL settings (GB), measured by the quick-mode probes
#: on 2026-11-15 (two replications per cell, peak reset per task); only for the start-up warning.
MEM_PER_WORKER_GB = {"sim": 0.90, "real": 0.90, "nonstat": 0.90}

FULL: dict[str, Any] = {
    "n_tickers": config.N_PANEL, "sim_reps": 200, "real_splits": 50, "nonstat_seeds": 10,
    "test_kw": {}, "base_kw": {}, "n_boot": pl.N_BOOT, "ticker_split": True, "pastnew": True,
    "real_relative": True, "blocks": {"sim": 5, "real": 5, "nonstat": 1}, "inject": [],
}
_Q_TEST = {"n_steps": 200, "n_perm": 200}
QUICK: dict[str, Any] = {
    "n_tickers": 20, "sim_reps": 4, "real_splits": 3, "nonstat_seeds": 2,
    "test_kw": {"critic": _Q_TEST, "lrcritic": _Q_TEST, "c2st": _Q_TEST,
                "mmd": {"n_perm": 200}, "sigmmd": {"n_perm": 200}},
    "base_kw": _Q_TEST, "n_boot": 100, "ticker_split": True, "pastnew": True,
    "real_relative": True, "blocks": {"sim": 2, "real": 3, "nonstat": 1},
    # Exercise the failure path: one test call and one whole replication raise.
    "inject": [{"part": "sim", "cell": "synth=joint", "rep": 0, "test": "ks"},
               {"part": "real", "cell": "period=test", "rep": 2, "test": "*"}],
}
#: Timing probes (quick mode): replications per cell at the FULL settings.
PROBE_REPS = 2

# --------------------------------------------------------------------------------------------
# Labels, orders, styles
# --------------------------------------------------------------------------------------------
TEST_ORDER = ("critic", "lrcritic", "c2st", "mmd", "sigmmd", "ks", "relative_anchored",
              "relative_two_critic")
TEST_LABEL = {**plotting.LABELS_TEST, "relative_anchored": "Anchored relative",
              "relative_two_critic": "Two-critic relative"}
PTYPE_ORDER = ("hac", "hac_fixedb", "hac_fixedb_eq", "wild", "timeblock_permutation",
               "permutation", "timeblock_perm_stud", "paired_bootstrap", "ticker_hac",
               "ticker_permutation", "asymptotic")
PTYPE_LABEL = {
    "hac": "time blocks, HAC",
    "hac_fixedb": "time blocks, HAC, fixed-b reference (conservative)",
    "hac_fixedb_eq": "time blocks, HAC, fixed-b reference (two-sample, equal variance)",
    "wild": "wild bootstrap",
    "timeblock_permutation": "time blocks, permutation", "permutation": "permutation",
    "timeblock_perm_stud": "time blocks, stud. permutation",
    "paired_bootstrap": "time blocks, paired bootstrap",
    "ticker_hac": "ticker split, HAC", "ticker_permutation": "ticker split, permutation",
    "asymptotic": "asymptotic (i.i.d.; invalid under dependence)",
}
#: Short labels for the LaTeX tables: learned and relative tests use the time-block split
#: unless "ticker split" is stated; the protocol's (primary) p-value carries a star.
PTYPE_TEX = {
    "hac": "HAC", "hac_fixedb": "HAC, fixed-$b$ (cons.)",
    "hac_fixedb_eq": "HAC, fixed-$b$ (eq.\\ var.)", "wild": "wild bootstrap",
    "timeblock_permutation": "permutation", "permutation": "permutation",
    "timeblock_perm_stud": "stud. permutation", "paired_bootstrap": "paired bootstrap",
    "ticker_hac": "HAC, ticker split", "ticker_permutation": "permutation, ticker split",
    "asymptotic": "asymptotic i.i.d. (invalid)",
}
_O = plotting.OKABE_ITO
PTYPE_COLOR = {
    "hac": _O["vermillion"], "wild": _O["vermillion"], "hac_fixedb": _O["black"],
    "hac_fixedb_eq": _O["grey"], "timeblock_permutation": _O["orange"],
    "permutation": _O["orange"], "timeblock_perm_stud": _O["skyblue"],
    "paired_bootstrap": _O["purple"], "ticker_hac": _O["green"],
    "ticker_permutation": _O["blue"], "asymptotic": _O["black"],
}
#: p-value types left out of the QQ figure (still in the CSVs and tables).
QQ_SKIP = ("hac_fixedb_eq",)
#: Figure legend: one entry per colour (p-value types of the same colour never share a panel).
COLOR_LEGEND = (
    (("hac", "wild"), "protocol $p$-value: HAC (time blocks) / wild bootstrap"),
    (("hac_fixedb", "asymptotic"), "HAC, fixed-$b$ reference (conservative) / KS asymptotic"),
    (("timeblock_permutation", "permutation"), "permutation (time blocks for learned tests)"),
    (("timeblock_perm_stud",), "studentised permutation (time blocks)"),
    (("paired_bootstrap",), "paired bootstrap (time blocks)"),
    (("ticker_hac",), "HAC, ticker split"),
    (("ticker_permutation",), "permutation, ticker split"),
)
SIM_LABEL = {"joint": "factor panel (joint law)",
             "indep": "independent tickers (same margins)"}
SIM_TEX = {"joint": "joint", "indep": "indep."}
SIM_LS = {"joint": "-", "indep": "--"}
PERIOD_LABEL = {"test": "TEST, ticker halves", "train": "TRAIN, ticker halves"}
PERIOD_TEX = {"test": "TEST halves", "train": "TRAIN halves"}
ALPHA_COLS = {a: rf"$\alpha={a:g}$" for a in ALPHAS}
#: What the ``statistic`` column measures, per test (scales differ across tests).
STAT_UNITS = {"critic": "critic mean diff.", "lrcritic": "mean logit diff.",
              "c2st": "eval.\\ accuracy", "mmd": r"$\mathrm{MMD}^2_u$",
              "sigmmd": r"sig-$\mathrm{MMD}^2_u$", "ks": "KS $D$"}


def role(test: str, ptype: str) -> str:
    """Role of a (test, p-value type) pair in the paper's protocol."""
    if test == "ks":
        return "invalid"
    if ptype in ("hac", "wild"):
        return "secondary" if test == "relative_two_critic" else "primary"
    return "comparison"


def _tag(a: float) -> str:
    return f"{a:.2f}"


# --------------------------------------------------------------------------------------------
# Fingerprints (checkpoint keys) and dependence diagnostics
# --------------------------------------------------------------------------------------------
def _sha(obj: Any) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return [float(f"{v:.10g}") for v in o.ravel()]
        if isinstance(o, (np.floating, float)):
            return float(f"{float(o):.10g}")
        if isinstance(o, (np.integer,)):
            return int(o)
        return repr(o)
    return hashlib.sha256(json.dumps(obj, default=default, sort_keys=True).encode()).hexdigest()[:16]


def panel_fingerprint(panel: pl.RealPanel) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(panel.tickers).encode())
    for df in (panel.train, panel.test):
        h.update(np.ascontiguousarray(df.to_numpy(dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def model_fingerprint(model: Any) -> str:
    return _sha(model.params())


@functools.lru_cache(maxsize=1)
def code_key() -> dict[str, str]:
    """SHA-256 of the synthval sources, of ``common.py`` and of this script (checkpoint key)."""
    here = Path(__file__).resolve()
    return {"synthval": utils.code_fingerprint()["sha256"],
            "common.py": utils.file_sha256(here.parent / "common.py"),
            "e1_placebo.py": utils.file_sha256(here)}


def logvol_corr_mean(R: np.ndarray, w: int = LOGVOL_WINDOW) -> float:
    """Mean pairwise correlation of the rolling ``w``-day log-volatility (columns = tickers)."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape[0] < 2 * w or R.shape[1] < 2:
        return float("nan")
    sd = np.lib.stride_tricks.sliding_window_view(R, w, axis=0).std(axis=-1, ddof=1)
    if not np.all(sd > 0):
        return float("nan")
    C = np.corrcoef(np.log(sd), rowvar=False)
    N = C.shape[0]
    return float((C.sum() - np.trace(C)) / (N * (N - 1)))


def dependence_stats(**panels: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, R in panels.items():
        out[f"corr_{name}"] = float(portfolio.average_correlation(np.asarray(R)))
        out[f"logvol_{name}"] = logvol_corr_mean(R)
    return out


# --------------------------------------------------------------------------------------------
# Shared setup
# --------------------------------------------------------------------------------------------
def load_panel(cfg: dict) -> pl.RealPanel:
    n = cfg["n_tickers"]
    tickers = None if n == config.N_PANEL else data.panel_tickers(n)
    return pl.load_real_panel(tickers)


def fit_factor_model(panel: pl.RealPanel) -> Any:
    t0 = time.perf_counter()
    model = get_mv_model("FactorGARCHt").fit(panel.train.to_numpy(dtype=np.float64))
    logger.info("FactorGARCHt fitted on %d x %d TRAIN returns in %.1f s", panel.n_train,
                panel.n_tickers, time.perf_counter() - t0)
    return model


def m4_condition(alpha: Any, beta: Any, nu: Any) -> np.ndarray:
    """``(a + b)^2 + (kappa - 1) a^2`` of a GARCH(1,1) with standardised-t(nu) innovations.

    ``kappa = 3 (nu - 2) / (nu - 4)`` (infinite for nu <= 4) is the innovation kurtosis; the
    fourth moment of the GARCH process is finite iff the value is < 1 (He and Terasvirta
    1999). HAC consistency for witnesses that grow linearly in returns needs it (theory.md
    Assumption D3).
    """
    a, b, v = (np.asarray(x, dtype=np.float64) for x in (alpha, beta, nu))
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(v > 4.0, 3.0 * (v - 2.0) / (v - 4.0), np.inf)
    return (a + b) ** 2 + (kappa - 1.0) * a ** 2


def factor_fit_table(model: Any, panel: pl.RealPanel) -> pd.DataFrame:
    p = model.params()
    f = p["factor"]
    rows = [{"series": "factor", "a": np.nan, "loading": np.nan, **{k: float(f[k]) for k in f},
             "persistence": float(p["persistence"][0])}]
    for i, t in enumerate(panel.tickers):
        rows.append({"series": t, "a": float(p["a"][i]), "loading": float(p["loadings"][i]),
                     **{k: float(v[i]) for k, v in p["idio"].items()},
                     "persistence": float(p["persistence"][i + 1])})
    df = pd.DataFrame(rows)
    nu = df["nu"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["kurt_z"] = np.where(nu > 4.0, 3.0 * (nu - 2.0) / (nu - 4.0), np.inf)
    df["m4_cond"] = m4_condition(df["alpha"], df["beta"], df["nu"])
    df["finite_m4"] = df["m4_cond"] < 1.0
    df["at_cap"] = df["persistence"] >= GARCH_CAP - 1e-4
    info = {k: v for k, v in model.fit_info_.items()
            if isinstance(v, (int, float, np.integer, np.floating))}
    idio = df.iloc[1:]
    pers = idio["persistence"].to_numpy(float)
    summary = {
        "factor_alpha": float(df["alpha"].iloc[0]), "factor_beta": float(df["beta"].iloc[0]),
        "factor_nu": float(df["nu"].iloc[0]), "factor_kurt_z": float(df["kurt_z"].iloc[0]),
        "factor_m4_cond": float(df["m4_cond"].iloc[0]),
        "factor_finite_m4": bool(df["finite_m4"].iloc[0]),
        "n_idio_infinite_m4": int((~idio["finite_m4"]).sum()), "n_idio": int(len(idio)),
        "idio_persistence_min": float(pers.min()),
        "idio_persistence_median": float(np.median(pers)),
        "idio_persistence_max": float(pers.max()), "n_idio_at_cap": int(idio["at_cap"].sum()),
        "persistence_cap": GARCH_CAP,
    }
    df.attrs["fit_info"] = info
    df.attrs["summary"] = summary
    return df


def _settings(cfg: dict) -> dict[str, Any]:
    return {"test_kw": cfg["test_kw"], "base_kw": cfg["base_kw"], "n_boot": cfg["n_boot"],
            "ticker_split": cfg["ticker_split"], "pastnew": cfg["pastnew"],
            "real_relative": cfg["real_relative"], "burn_in": SIM_BURN_IN,
            "sim_layout": "R_past*+R_new* one contiguous path; G* independent stationary",
            "inject": list(cfg.get("inject", []))}


# --------------------------------------------------------------------------------------------
# Test calls: one test at a time, failures become NaN rows
# --------------------------------------------------------------------------------------------
_NAN = float("nan")


def _injected(s: dict, part: str, cell: str, rep: int) -> set[str]:
    """Tests (or ``"*"``: the whole replication) whose failure quick mode injects."""
    return {d["test"] for d in s.get("inject", ())
            if d["part"] == part and d["cell"] == cell and int(d["rep"]) == int(rep)}


def _raise_injected() -> None:
    raise RuntimeError("injected failure (quick mode: exercises the NaN-row path)")


def _per_date(order: np.ndarray | None) -> float:
    """Windows per date of a balanced sample (every date has the same count), else NaN."""
    if order is None:
        return _NAN
    _, c = np.unique(np.asarray(order), return_counts=True)
    return float(c[0]) if c.size and np.all(c == c[0]) else _NAN


def _nan_row(kind: str, test: str, err: str, secs: float = _NAN) -> dict[str, Any]:
    return {"kind": kind, "test": test, "statistic": _NAN, "pvalue": _NAN, "p_perm": _NAN,
            "p_hac": _NAN, "p_wild": _NAN, "z_hac": _NAN, "valid": test != "ks",
            "runtime_s": secs, "error": err}


def _abs_rows(kind: str, X: np.ndarray, Y: np.ndarray, *, order_x: np.ndarray,
              order_y: np.ndarray, rng: np.random.Generator,
              tests: tuple[str, ...] = pl.ABSOLUTE_TESTS, protocol: str = "dependent",
              test_kw: dict | None = None, groups_x: np.ndarray | None = None,
              groups_y: np.ndarray | None = None, fail: set[str] = frozenset()
              ) -> list[dict[str, Any]]:
    """``pl.absolute_tests`` one test at a time (same streams as one joint call).

    One integer is drawn from ``rng``; every test gets ``default_rng(that integer)``, from
    which ``absolute_tests`` draws its base seed, as a joint call with that generator would.
    Time-block rows also get the fixed-b inputs ``T_eval_x``, ``T_eval_y``, ``H_x``, ``H_y``
    and ``se_hac``.
    """
    seed = int(rng.integers(0, 2**63 - 1))
    mx, my = _per_date(order_x), _per_date(order_y)
    rows: list[dict[str, Any]] = []
    for name in tests:
        if name in fail:
            res, secs, err = common.safe_test(_raise_injected)
        else:
            res, secs, err = common.safe_test(
                pl.absolute_tests, X, Y, order_x=order_x, order_y=order_y, tests=(name,),
                protocol=protocol, rng=np.random.default_rng(seed), test_kw=test_kw,
                groups_x=groups_x, groups_y=groups_y, keep_extra=True)
        if res is None:
            rows.append(_nan_row(kind, name, err, secs))
            continue
        r = res[0]
        ex = r.pop("extra", None) or {}
        lags = ex.get("hac_lags")
        if r.get("split") == "time_block" and ex.get("hac") and lags is not None:
            r.update(H_x=int(lags[0]), H_y=int(lags[1]), T_eval_x=r["n_eval_x"] / mx,
                     T_eval_y=r["n_eval_y"] / my, se_hac=float(ex.get("se_hac", _NAN)))
        r["error"] = ""
        rows.append({"kind": kind, **r})
    return rows


def _rel_rows(R_new: np.ndarray, G: np.ndarray, R_past: np.ndarray, *,
              orders: tuple[np.ndarray, np.ndarray, np.ndarray], rng: np.random.Generator,
              base_kw: dict, n_boot: int, fail: set[str] = frozenset()
              ) -> list[dict[str, Any]]:
    """``pl.relative_tests`` one method at a time (same streams as one joint call).

    Anchored rows get the fixed-b inputs with ``x = R_past`` and ``y = G`` (``Delta = mean
    f(R_past_ev) - mean f(G_ev)``) and the split sizes (G blocks are shorter than R_past blocks
    and each G_tr window carries more training weight than an R_past_tr window).
    """
    seed = int(rng.integers(0, 2**63 - 1))
    m_g, m_p = _per_date(orders[1]), _per_date(orders[2])
    rows: list[dict[str, Any]] = []
    for method in pl.RELATIVE_METHODS:
        test = f"relative_{method}"
        if test in fail:
            res, secs, err = common.safe_test(_raise_injected)
        else:
            res, secs, err = common.safe_test(
                pl.relative_tests, R_new, G, R_past, orders=orders, base="critic",
                methods=(method,), protocol="dependent", rng=np.random.default_rng(seed),
                base_kw=base_kw, n_boot=n_boot, keep_extra=True)
        if res is None:
            rows.append(_nan_row("relative", test, err, secs))
            continue
        r = res[0]
        ex = r.pop("extra", None) or {}
        lags = ex.get("hac_lags")
        r["hac_lags"] = "" if lags is None else str(tuple(int(v) for v in lags))
        for k in ("n_train_g", "n_train_past", "n_eval_g", "n_eval_past", "n_eval_new"):
            r[k] = ex.get(k, _NAN)
        if method == "anchored" and ex.get("hac") and lags is not None:
            r.update(H_x=int(lags[0]), H_y=int(lags[1]), T_eval_x=ex["n_eval_past"] / m_p,
                     T_eval_y=ex["n_eval_g"] / m_g)
        r["error"] = ""
        rows.append({"kind": "relative", **r})
    return rows


# --------------------------------------------------------------------------------------------
# Part sim: simulated factor-panel placebo
# --------------------------------------------------------------------------------------------
def indep_panel(model: Any, n_steps: int, ss: np.random.SeedSequence, burn_in: int) -> np.ndarray:
    """Panel whose column i is column i of an independent draw of ``model`` (seed child i).

    Each ticker keeps exactly its marginal law under the factor model (as a process), while
    tickers are mutually independent, as in a panel of univariate generators.
    """
    N = int(model.n_assets_)
    out = np.empty((int(n_steps), N), dtype=np.float64)
    for i, child in enumerate(ss.spawn(N)):
        out[:, i] = pl.simulate_panel(model, n_steps, seed=child, start_from_end=False,
                                      burn_in=burn_in)[:, i]
    return out


def _sim_nan_rows(cell: dict, s: dict, err: str) -> list[dict[str, Any]]:
    rows = [_nan_row("absolute", t, err) for t in pl.ABSOLUTE_TESTS]
    if s["ticker_split"]:
        rows += [_nan_row("absolute_ticker", t, err) for t in pl.LEARNED_TESTS]
    rows += [_nan_row("relative", f"relative_{m}", err) for m in pl.RELATIVE_METHODS]
    if s["pastnew"] and cell["synth"] == "joint":
        rows += [_nan_row("pastnew", t, err) for t in pl.ABSOLUTE_TESTS]
    rows.append({"kind": "diag", "error": err})
    return rows


def _sim_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """One placebo replication (rows: absolute, absolute_ticker, relative, pastnew, diag)."""
    s, panel, model = cell["_settings"], cell["_panel"], cell["_model"]
    burn = s["burn_in"]
    fail = _injected(s, "sim", cell["cell"], rep)
    kseed = common.keyed_seed_sequence
    t0 = time.perf_counter()
    try:
        if "*" in fail:
            _raise_injected()
        # R_past* and R_new*: one contiguous stationary path (as real TRAIN -> TEST).
        P = pl.simulate_panel(model, panel.n_train + panel.n_test,
                              seed=kseed(ROOT_SEED, "sim", "R_past_new", rep),
                              start_from_end=False, burn_in=burn)
        Rp, Rn = P[:panel.n_train], P[panel.n_train:]
        if cell["synth"] == "joint":
            G = pl.simulate_panel(model, panel.n_test,
                                  seed=kseed(ROOT_SEED, "sim", "G_joint", rep),
                                  start_from_end=False, burn_in=burn)
        else:
            G = indep_panel(model, panel.n_test, kseed(ROOT_SEED, "sim", "G_indep", rep), burn)
        sim_s = time.perf_counter() - t0
        Wn, gn, on = panel.synthetic_windows(Rn, "test")
        Wg, gg, og = panel.synthetic_windows(G, "test")
        Wp, _, op = panel.synthetic_windows(Rp, "train")
        diag = dependence_stats(new=Rn, past=Rp, g=G)
    except Exception as exc:  # noqa: BLE001 - deterministic seeds: record, do not block the part
        err = f"{type(exc).__name__}: {exc}"
        logger.warning("sim %s rep %d failed before the tests: %s", cell["cell"], rep, err)
        return _sim_nan_rows(cell, s, err)
    trng = common.keyed_rng(ROOT_SEED, "sim", "tests", rep)  # shared by the two cells
    rows = _abs_rows("absolute", Wn, Wg, order_x=on, order_y=og, rng=trng,
                     test_kw=s["test_kw"], fail=fail)
    if s["ticker_split"]:
        rows += _abs_rows("absolute_ticker", Wn, Wg, order_x=on, order_y=og, rng=trng,
                          tests=pl.LEARNED_TESTS, protocol="iid", test_kw=s["test_kw"],
                          groups_x=gn, groups_y=gg)
    rows += _rel_rows(Wn, Wg, Wp, orders=(on, og, op), rng=trng, base_kw=s["base_kw"],
                      n_boot=s["n_boot"], fail=fail)
    if s["pastnew"] and cell["synth"] == "joint":
        rows += _abs_rows("pastnew", Wn, Wp, order_x=on, order_y=op, rng=trng,
                          test_kw=s["test_kw"])
    rows.append({"kind": "diag", "sim_seconds": sim_s, "error": "", **diag})
    return rows


def sim_cells(panel: pl.RealPanel, model: Any, cfg: dict) -> list[dict]:
    s = _settings(cfg)
    return [{"cell": f"synth={c}", "synth": c, "_settings": s, "_panel": panel, "_model": model}
            for c in SIM_CELLS]


def sim_config(panel: pl.RealPanel, model: Any, cfg: dict) -> dict[str, Any]:
    return {"part": "sim", "root_seed": ROOT_SEED, "pipeline": pl.pipeline_config(),
            "settings": _settings(cfg), "panel": panel_fingerprint(panel),
            "model": model_fingerprint(model), "n_tickers": panel.n_tickers,
            "n_train": panel.n_train, "n_test": panel.n_test, "L": panel.L, "code": code_key()}


def run_sim(ctx: common.ExpContext, cfg: dict, panel: pl.RealPanel, model: Any) -> dict:
    t0 = time.perf_counter()
    fit = factor_fit_table(model, panel)
    summ = fit.attrs["summary"]
    logger.info("factor GARCH: alpha %.3f beta %.3f nu %.2f, m4 condition %.3f (finite fourth "
                "moment iff < 1); idiosyncratic: %d of %d with infinite fourth moment, "
                "persistence min / median / max %.3f / %.3f / %.3f, %d at the %.3f cap",
                summ["factor_alpha"], summ["factor_beta"], summ["factor_nu"],
                summ["factor_m4_cond"], summ["n_idio_infinite_m4"], summ["n_idio"],
                summ["idio_persistence_min"], summ["idio_persistence_median"],
                summ["idio_persistence_max"], summ["n_idio_at_cap"], GARCH_CAP)
    ctx.save_csv(fit, "e1_factor_fit", config={"fit_info": fit.attrs["fit_info"],
                                               "summary": summ})
    conf = sim_config(panel, model, cfg)
    df = common.run_cells(_sim_rep, sim_cells(panel, model, cfg), part="sim", ctx=ctx,
                          root_seed=ROOT_SEED, n_reps=cfg["sim_reps"], config=conf,
                          block_size=cfg["blocks"]["sim"],
                          cost=lambda c: 1.4 if c["synth"] == "joint" else 1.0)
    secs = time.perf_counter() - t0
    _log_failures(df, "sim")
    ctx.save_csv(df, "e1_sim_raw", config=conf, timing={"sim_wall_s": secs})
    reps = df.groupby(["synth", "rep"])["rep_seconds"].first()
    for c in SIM_CELLS:
        if c in reps.index.get_level_values(0):
            common.print_rep_timing(f"sim {c} (this run's settings)", float(reps.loc[c].mean()))
    ctx.write_part_meta("sim", conf, {"wall_s": secs, "n_reps": cfg["sim_reps"]})
    return {"sim_wall_s": secs}


def _log_failures(df: pd.DataFrame, part: str) -> None:
    if "error" not in df:
        return
    err = df["error"].astype(object).fillna("").astype(str)
    bad = df[err != ""]
    if not bad.empty:
        logger.warning("[%s] %d failed test rows (NaN p-values, counted in n_nan): %s", part,
                       len(bad), "; ".join(sorted(set(bad["error"].astype(str)))[:5]))


# --------------------------------------------------------------------------------------------
# Part real: ticker-halves placebo on real data
# --------------------------------------------------------------------------------------------
def _halves(panel: pl.RealPanel, period: str, split: int) -> tuple[np.ndarray, np.ndarray]:
    perm = common.keyed_rng(ROOT_SEED, "real", period, "split", split).permutation(
        panel.n_tickers)
    h = panel.n_tickers // 2
    return np.sort(perm[:h]), np.sort(perm[h:])


def _real_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    s, panel, period = cell["_settings"], cell["_panel"], cell["period"]
    fail = _injected(s, "real", cell["cell"], rep)
    relative = period == "train" and s["real_relative"]
    A, B = _halves(panel, period, rep)
    try:
        if "*" in fail:
            _raise_injected()
        df, off = panel.returns(period), panel.offset(period)
        XA, _, oA = pl.panel_windows(df.iloc[:, A], panel.L, panel.step, scale=panel.scale,
                                     date_offset=off)
        XB, _, oB = pl.panel_windows(df.iloc[:, B], panel.L, panel.step, scale=panel.scale,
                                     date_offset=off)
        Xn, _, on = panel.windows("test") if relative else (None, None, None)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        logger.warning("real %s split %d failed before the tests: %s", period, rep, err)
        rows = [_nan_row("absolute", t, err) for t in pl.ABSOLUTE_TESTS]
        if relative:
            rows += [_nan_row("relative", f"relative_{m}", err) for m in pl.RELATIVE_METHODS]
    else:
        trng = common.keyed_rng(ROOT_SEED, "real", period, "tests", rep)
        rows = _abs_rows("absolute", XA, XB, order_x=oA, order_y=oB, rng=trng,
                         test_kw=s["test_kw"], fail=fail)
        if relative:
            rows += _rel_rows(Xn, XB, XA, orders=(on, oB, oA), rng=trng, base_kw=s["base_kw"],
                              n_boot=s["n_boot"], fail=fail)
    a0 = ",".join(str(int(i)) for i in A[:5])
    for r in rows:
        r["n_tickers_a"], r["n_tickers_b"], r["first_a"] = len(A), len(B), a0
    return rows


def real_cells(panel: pl.RealPanel, cfg: dict) -> list[dict]:
    s = _settings(cfg)
    return [{"cell": f"period={p}", "period": p, "_settings": s, "_panel": panel}
            for p in PERIODS]


def real_config(panel: pl.RealPanel, cfg: dict) -> dict[str, Any]:
    return {"part": "real", "root_seed": ROOT_SEED, "pipeline": pl.pipeline_config(),
            "settings": _settings(cfg), "panel": panel_fingerprint(panel),
            "n_tickers": panel.n_tickers, "L": panel.L, "code": code_key()}


def run_real(ctx: common.ExpContext, cfg: dict, panel: pl.RealPanel) -> dict:
    t0 = time.perf_counter()
    conf = real_config(panel, cfg)
    df = common.run_cells(_real_rep, real_cells(panel, cfg), part="real", ctx=ctx,
                          root_seed=ROOT_SEED, n_reps=cfg["real_splits"], config=conf,
                          block_size=cfg["blocks"]["real"],
                          cost=lambda c: 2.5 if c["period"] == "train" else 1.0)
    secs = time.perf_counter() - t0
    _log_failures(df, "real")
    ctx.save_csv(df, "e1_real_raw", config=conf, timing={"real_wall_s": secs})
    reps = df.groupby(["period", "rep"])["rep_seconds"].first()
    for p in PERIODS:
        if p in reps.index.get_level_values(0):
            common.print_rep_timing(f"real {p} (this run's settings)", float(reps.loc[p].mean()))
    ctx.write_part_meta("real", conf, {"wall_s": secs, "n_splits": cfg["real_splits"]})
    return {"real_wall_s": secs}


# --------------------------------------------------------------------------------------------
# Part nonstat: R_past vs R_new on the real panel
# --------------------------------------------------------------------------------------------
def _nonstat_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    s, panel = cell["_settings"], cell["_panel"]
    Xn, _, on = panel.windows("test")
    Xp, _, op = panel.windows("train")
    return _abs_rows("absolute", Xn, Xp, order_x=on, order_y=op, rng=rng,
                     test_kw=s["test_kw"], fail=_injected(s, "nonstat", cell["cell"], rep))


def nonstat_config(panel: pl.RealPanel, cfg: dict) -> dict[str, Any]:
    return {"part": "nonstat", "root_seed": ROOT_SEED, "pipeline": pl.pipeline_config(),
            "settings": _settings(cfg), "panel": panel_fingerprint(panel),
            "n_tickers": panel.n_tickers, "L": panel.L, "code": code_key()}


def run_nonstat(ctx: common.ExpContext, cfg: dict, panel: pl.RealPanel) -> dict:
    t0 = time.perf_counter()
    conf = nonstat_config(panel, cfg)
    cells = [{"cell": "real", "_settings": _settings(cfg), "_panel": panel}]
    df = common.run_cells(_nonstat_rep, cells, part="nonstat", ctx=ctx, root_seed=ROOT_SEED,
                          n_reps=cfg["nonstat_seeds"], config=conf,
                          block_size=cfg["blocks"]["nonstat"])
    secs = time.perf_counter() - t0
    _log_failures(df, "nonstat")
    ctx.save_csv(df, "e1_nonstat_raw", config=conf, timing={"nonstat_wall_s": secs})
    common.print_rep_timing("nonstat (this run's settings)",
                            float(df.groupby("rep")["rep_seconds"].first().mean()))
    ctx.write_part_meta("nonstat", conf, {"wall_s": secs, "n_seeds": cfg["nonstat_seeds"]})
    return {"nonstat_wall_s": secs}


# --------------------------------------------------------------------------------------------
# Fixed-b references for the HAC z (computed after the run from the stored inputs)
# --------------------------------------------------------------------------------------------
def _bartlett_var_of_mean(E: np.ndarray, H: int) -> np.ndarray:
    """Bartlett (Newey-West) variance of the mean of each demeaned row of ``E``, lag ``H``.

    Same weights ``1 - h / (H + 1)`` as ``critic.hac_var_mean`` (whose ``n / (n - 1)`` factor,
    with n the number of windows, is about 1 here and is left out).
    """
    T = E.shape[1]
    lrv = np.einsum("ij,ij->i", E, E)
    for h in range(1, min(int(H), T - 1) + 1):
        lrv += 2.0 * (1.0 - h / (H + 1.0)) * np.einsum("ij,ij->i", E[:, h:], E[:, :-h])
    return np.maximum(lrv, 0.0) / T**2


@functools.lru_cache(maxsize=64)
def fixedb_reference(T_x: int, H_x: int, T_y: int = 0, H_y: int = 0) -> np.ndarray:
    """Sorted draws of the fixed-b reference of a HAC t statistic (symmetrised).

    ``T_y = 0``: ``mean(x) / sqrt(V_x)`` for ``T_x`` i.i.d. N(0, 1) observations and the
    Bartlett variance with lag ``H_x``: the finite-T version of the Kiefer-Vogelsang (2005)
    fixed-b reference with ``b = (H_x + 1) / T_x``. Otherwise the two-sample statistic
    ``(mean(x) - mean(y)) / sqrt(V_x + V_y)`` with independent unit-variance samples (equal
    per-date variance).
    """
    rng = np.random.default_rng([FIXEDB_SEED, int(T_x), int(H_x), int(T_y), int(H_y)])
    X = rng.standard_normal((FIXEDB_DRAWS, int(T_x)))
    mx = X.mean(axis=1)
    v = _bartlett_var_of_mean(X - mx[:, None], H_x)
    num = mx
    if T_y:
        Y = rng.standard_normal((FIXEDB_DRAWS, int(T_y)))
        my = Y.mean(axis=1)
        v = v + _bartlett_var_of_mean(Y - my[:, None], H_y)
        num = mx - my
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num / np.sqrt(v)
    t = t[np.isfinite(t)]
    return np.sort(np.concatenate([t, -t]))


def _upper_tail(ref: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Monte Carlo upper-tail p-value ``(1 + #{ref >= z}) / (1 + len(ref))``; NaN for NaN z."""
    z = np.asarray(z, dtype=np.float64)
    p = (1.0 + ref.size - np.searchsorted(ref, z, side="left")) / (1.0 + ref.size)
    return np.where(np.isnan(z), np.nan, p)


def fixedb_pvalue(z: Any, T_x: int, H_x: int, T_y: int, H_y: int,
                  kind: str = "conservative") -> np.ndarray:
    """One-sided p-value of a two-sample HAC z against a fixed-b reference.

    ``kind="conservative"`` (ptype ``hac_fixedb``): single-series reference with
    ``T = min(T_x, T_y)`` eval dates and lag ``H = max(H_x, H_y)``, i.e. ``b = (H + 1) / T``
    (Kiefer and Vogelsang 2005; theory.md 7.4 and 11.5). The two-sample statistic's fixed-b
    limit is ``Z / sqrt(lam Q_bx + (1 - lam) Q_by)`` with ``lam`` the first sample's share of
    the variance; the single series with the larger ``b`` is its heaviest-tailed case, so this
    holds whatever ``lam`` is. ``kind="equal"`` (ptype ``hac_fixedb_eq``): the two-sample
    reference with each sample's own ``T`` and ``H`` and equal per-date variance, the fixed-b
    limit under an equal-law null with equal cross-sections (both E1 nulls), not in general.
    Both use a contiguous series, whereas the Driscoll-Kraay estimator runs on the time grid
    with the training blocks as zero-filled gaps (a further approximation). Fixed witnesses on
    the E1(a) design (400 replications, nominal 5%): absolute 0.033 (conservative) / 0.052
    (equal) / 0.095 (normal), anchored structure 0.037 / 0.068 / 0.104.
    """
    if kind == "conservative":
        ref = fixedb_reference(int(min(T_x, T_y)), int(max(H_x, H_y)))
    elif kind == "equal":
        ref = fixedb_reference(int(T_x), int(H_x), int(T_y), int(H_y))
    else:
        raise ValueError("kind must be 'conservative' or 'equal'")
    return _upper_tail(ref, z)


def add_fixedb(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``p_fixedb`` (conservative) and ``p_fixedb_eq`` (equal variance) where possible."""
    df = df.copy()
    df["p_fixedb"] = np.nan
    df["p_fixedb_eq"] = np.nan
    need = ["z_hac", "T_eval_x", "T_eval_y", "H_x", "H_y"]
    if not all(c in df.columns for c in need):
        return df
    v = df[need].apply(pd.to_numeric, errors="coerce")
    T = v[["T_eval_x", "T_eval_y"]].to_numpy(float)
    H = v[["H_x", "H_y"]].to_numpy(float)
    ok = (~np.isnan(v["z_hac"].to_numpy(float)) & np.all(np.isfinite(T), axis=1)
          & np.all(np.isfinite(H), axis=1))
    ok &= np.all(np.abs(T - np.round(T)) < 1e-6, axis=1) & np.all(np.round(T) >= 2, axis=1)
    if not ok.any():
        return df
    key = pd.DataFrame({"tx": np.round(T[ok, 0]).astype(int), "hx": H[ok, 0].astype(int),
                        "ty": np.round(T[ok, 1]).astype(int), "hy": H[ok, 1].astype(int)},
                       index=df.index[ok])
    z = v.loc[ok, "z_hac"]
    for (tx, hx, ty, hy), g in key.groupby(["tx", "hx", "ty", "hy"]):
        zz = z.loc[g.index].to_numpy(float)
        df.loc[g.index, "p_fixedb"] = fixedb_pvalue(zz, tx, hx, ty, hy, "conservative")
        df.loc[g.index, "p_fixedb_eq"] = fixedb_pvalue(zz, tx, hx, ty, hy, "equal")
    return df


# --------------------------------------------------------------------------------------------
# p-values in long format
# --------------------------------------------------------------------------------------------
_ID_COLS = ("synth", "period", "cell", "rep", "kind", "test", "statistic", "z_hac", "valid")


def melt_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (test call, p-value type) with columns ``ptype`` and ``p``."""
    df = add_fixedb(df[df["kind"] != "diag"])
    k, t = df["kind"], df["test"]
    learned, kernel = t.isin(pl.LEARNED_TESTS), t.isin(pl.KERNEL_TESTS)
    absd = k.isin(["absolute", "pastnew"])
    anch = (k == "relative") & (t == "relative_anchored")
    two = (k == "relative") & (t == "relative_two_critic")
    tick = k == "absolute_ticker"
    specs = [
        (absd & learned, "hac", "p_hac"),
        (absd & learned, "hac_fixedb", "p_fixedb"),
        (absd & learned, "hac_fixedb_eq", "p_fixedb_eq"),
        (absd & learned, "timeblock_permutation", "p_perm"),
        (absd & kernel, "wild", "p_wild"),
        (absd & kernel, "permutation", "p_perm"),
        (absd & ~learned & ~kernel, "asymptotic", "pvalue"),
        (tick, "ticker_hac", "p_hac"),
        (tick, "ticker_permutation", "p_perm"),
        (anch, "hac", "p_hac"),
        (anch, "hac_fixedb", "p_fixedb"),
        (anch, "hac_fixedb_eq", "p_fixedb_eq"),
        (anch, "timeblock_permutation", "p_perm"),
        (anch, "timeblock_perm_stud", "p_perm_stud"),
        (two, "hac", "p_hac"),
        (two, "paired_bootstrap", "p_boot"),
    ]
    ids = [c for c in _ID_COLS if c in df.columns]
    frames = []
    for mask, ptype, col in specs:
        if not mask.any() or col not in df.columns:
            continue
        sub = df.loc[mask, ids].copy()
        sub["ptype"] = ptype
        sub["p"] = pd.to_numeric(df.loc[mask, col], errors="coerce")
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ids)
    out["role"] = [role(a, b) for a, b in zip(out["test"], out["ptype"])]
    out["ptype_label"] = out["ptype"].map(PTYPE_LABEL)
    return out


_LONG_CACHE: dict[str, pd.DataFrame] = {}


def load_long(ctx: common.ExpContext, name: str) -> pd.DataFrame:
    """Long p-value table of ``results/<exp_id>/<name>.csv`` (cached within the process)."""
    key = str(ctx.results_dir / name)
    if key not in _LONG_CACHE:
        _LONG_CACHE[key] = melt_pvalues(ctx.load_csv(name))
    return _LONG_CACHE[key]


def _sort(tbl: pd.DataFrame, first: str | None = None, first_order: tuple = ()) -> pd.DataFrame:
    tbl = tbl.copy()
    keys = []
    if first is not None:
        tbl["_o0"] = tbl[first].map({v: i for i, v in enumerate(first_order)})
        keys.append("_o0")
    tbl["_o1"] = tbl["test"].map({v: i for i, v in enumerate(TEST_ORDER)})
    tbl["_o2"] = tbl["ptype"].map({v: i for i, v in enumerate(PTYPE_ORDER)})
    keys += ["_o1", "_o2"]
    return tbl.sort_values(keys).drop(columns=keys).reset_index(drop=True)


def _ptype_tex(test: str, ptype: str) -> str:
    """Short p-value label; the protocol's p-value of a valid test gets a star."""
    lab = PTYPE_TEX[ptype]
    return lab + r"$^{\star}$" if role(test, ptype) in ("primary", "secondary") else lab


def _tex_index(tbl: pd.DataFrame, first: str | None, first_label: dict | None,
               first_name: str) -> tuple[pd.DataFrame, list[str]]:
    t = tbl.copy()
    index = []
    if first is not None:
        t[first_name] = t[first].map(first_label)
        index.append(first_name)
    t["Test"] = t["test"].map(TEST_LABEL)
    t["$p$-value"] = [_ptype_tex(a, b) for a, b in zip(t["test"], t["ptype"])]
    return t, index + ["Test", "$p$-value"]


def _tex_rows(tbl: pd.DataFrame, first: str | None, first_label: dict | None,
              first_name: str) -> pd.DataFrame:
    """``rate [Wilson 95% CI]`` per alpha, indexed by (first, Test, p-value)."""
    t, index = _tex_index(tbl, first, first_label, first_name)
    return common.rate_ci_table(t, index, ALPHAS, label=lambda a: ALPHA_COLS[a])


def _tex_rates(tbl: pd.DataFrame, first: str | None, first_label: dict | None,
               first_name: str) -> pd.DataFrame:
    """Rejection rates only (no binomial interval), indexed by (first, Test, p-value)."""
    t, index = _tex_index(tbl, first, first_label, first_name)
    out = t[index].copy()
    for a in ALPHAS:
        out[ALPHA_COLS[a]] = t[f"rej_{_tag(a)}"].to_numpy(float)
    return out.set_index(index)


# --------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------
def build_sim_outputs(ctx: common.ExpContext, cfg: dict) -> None:
    if not ctx.has_csv("e1_sim_raw"):
        logger.info("no sim results: skipping e1_size_sim / e1_pvalue_qq")
        return
    raw = ctx.load_csv("e1_sim_raw")
    long = load_long(ctx, "e1_sim_raw")
    size = long[long["kind"].isin(["absolute", "absolute_ticker", "relative"])]
    tbl = common.rejection_table(size, ["synth", "test", "ptype", "ptype_label", "role"],
                                 pcol="p", alphas=ALPHAS)
    tbl = _sort(tbl, "synth", SIM_CELLS)
    ctx.save_csv(tbl, "e1_size_sim", config={
        "fixedb_draws": 2 * FIXEDB_DRAWS,
        "note": "hac_fixedb / hac_fixedb_eq are computed after the run from z_hac, T_eval_*, "
                "H_* (see fixedb_pvalue in experiments/e1_placebo.py)"})
    ctx.save_table(_tex_rows(tbl, "synth", SIM_TEX, "$G^*$"), "e1_size_sim", escape=False)
    paired = paired_table(size)
    if not paired.empty:
        ctx.save_csv(paired, "e1_sim_paired", config={
            "note": "joint and indep cells share R_past*, R_new* and the test seeds per rep: "
                    "paired counts and two-sided exact McNemar p-values on discordant pairs"})
    plot_qq(ctx, size)
    build_dependence(ctx, cfg, raw)


def paired_table(long: pd.DataFrame) -> pd.DataFrame:
    """Joint vs indep cell, paired by replication: discordant counts and exact McNemar p."""
    if "synth" not in long or not set(SIM_CELLS) <= set(long["synth"].dropna()):
        return pd.DataFrame()
    key = ["kind", "test", "ptype", "rep"]
    w = long.pivot_table(index=key, columns="synth", values="p", aggfunc="first")
    w = w.dropna(subset=list(SIM_CELLS)).reset_index()
    rows = []
    for (kind, test, ptype), g in w.groupby(["kind", "test", "ptype"]):
        row: dict[str, Any] = {"kind": kind, "test": test, "ptype": ptype,
                               "n_pairs": int(len(g))}
        for a in ALPHAS:
            rj = g["joint"].to_numpy(float) <= a
            ri = g["indep"].to_numpy(float) <= a
            b, c = int(np.sum(rj & ~ri)), int(np.sum(~rj & ri))
            row[f"rej_joint_{_tag(a)}"] = float(rj.mean())
            row[f"rej_indep_{_tag(a)}"] = float(ri.mean())
            row[f"joint_only_{_tag(a)}"] = b
            row[f"indep_only_{_tag(a)}"] = c
            row[f"mcnemar_p_{_tag(a)}"] = (float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
                                           if b + c else 1.0)
        rows.append(row)
    return _sort(pd.DataFrame(rows), "kind", ("absolute", "absolute_ticker", "relative"))


def build_dependence(ctx: common.ExpContext, cfg: dict, raw: pd.DataFrame) -> None:
    diag = raw[raw["kind"] == "diag"]
    rows = []
    try:
        panel = load_panel(cfg)
        for name, df in (("real TEST", panel.test), ("real TRAIN", panel.train)):
            R = df.to_numpy(dtype=np.float64)
            rows.append({"source": name, "n_reps": 1,
                         "corr_mean": float(portfolio.average_correlation(R)),
                         "corr_q25": np.nan, "corr_q75": np.nan,
                         "logvol_corr_mean": logvol_corr_mean(R),
                         "logvol_q25": np.nan, "logvol_q75": np.nan})
    except Exception:  # noqa: BLE001 - the data file may be missing in plots-only use
        logger.exception("could not load the real panel for the dependence summary")
    specs = [("placebo R_new* (TEST length)", "new", None),
             ("placebo R_past* (TRAIN length)", "past", None),
             ("placebo G*, joint", "g", "joint"), ("placebo G*, indep", "g", "indep")]
    for label, key, synth in specs:
        if f"corr_{key}" not in diag:
            continue
        d = diag if synth is None else diag[diag["synth"] == synth]
        if synth is None:
            d = d.drop_duplicates("rep")
        c, lv = d[f"corr_{key}"].to_numpy(float), d[f"logvol_{key}"].to_numpy(float)
        c, lv = c[np.isfinite(c)], lv[np.isfinite(lv)]
        if c.size == 0:
            continue
        rows.append({"source": label, "n_reps": int(c.size), "corr_mean": float(np.median(c)),
                     "corr_q25": float(np.quantile(c, 0.25)),
                     "corr_q75": float(np.quantile(c, 0.75)),
                     "logvol_corr_mean": float(np.median(lv)) if lv.size else np.nan,
                     "logvol_q25": float(np.quantile(lv, 0.25)) if lv.size else np.nan,
                     "logvol_q75": float(np.quantile(lv, 0.75)) if lv.size else np.nan})
    ctx.save_csv(pd.DataFrame(rows), "e1_dependence",
                 config={"note": "placebo rows: median and quartiles over replications of the "
                                 "mean pairwise return correlation and of the mean pairwise "
                                 "correlation of rolling 20-day log-volatility"})


def plot_qq(ctx: common.ExpContext, long: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plotting.setup()
    long = long[~long["ptype"].isin(QQ_SKIP)]
    tests = [t for t in TEST_ORDER if t in set(long["test"])]
    ncol = 4
    nrow = math.ceil(len(tests) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=plotting.figsize("two", 0.30 * nrow + 0.08),
                             sharex=True, sharey=True, squeeze=False)
    floor = 1e-3
    used: list[str] = []
    cells = [c for c in SIM_CELLS if c in set(long["synth"])]
    for ax, test in zip(axes.ravel(), tests):
        sub = long[long["test"] == test]
        n_max = 0
        for ptype in PTYPE_ORDER:
            for synth in cells:
                p = sub[(sub["ptype"] == ptype) & (sub["synth"] == synth)]["p"].to_numpy(float)
                p = np.sort(p[np.isfinite(p)])
                if p.size == 0:
                    continue
                n = p.size
                n_max = max(n_max, n)
                u = np.arange(1, n + 1) / (n + 1)
                ax.plot(u, np.clip(p, floor, 1.0), color=PTYPE_COLOR[ptype], ls=SIM_LS[synth],
                        lw=1.0 if synth == "joint" else 0.8, drawstyle="steps-post")
                if ptype not in used:
                    used.append(ptype)
        if n_max:
            i = np.arange(1, n_max + 1)
            u = i / (n_max + 1)
            lo = stats.beta.ppf(0.025, i, n_max + 1 - i)
            hi = stats.beta.ppf(0.975, i, n_max + 1 - i)
            ax.fill_between(u, np.clip(lo, floor, 1), np.clip(hi, floor, 1),
                            color=_O["grey"], alpha=0.25, lw=0, step="post")
        ax.plot([floor, 1], [floor, 1], color="black", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(floor, 1)
        ax.set_ylim(floor / 1.6, 1.05)  # lines clipped at the floor stay visible
        ax.set_title(TEST_LABEL.get(test, test), loc="left")
    for ax in axes.ravel()[len(tests):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("uniform quantile")
    for ax in axes[:, 0]:
        ax.set_ylabel("placebo $p$-value")
    handles = [Line2D([], [], color=PTYPE_COLOR[pts[0]], ls="-", label=lab)
               for pts, lab in COLOR_LEGEND if any(p in used for p in pts)]
    handles += [Line2D([], [], color="#555555", ls=SIM_LS[c], label=f"$G^*$: {SIM_LABEL[c]}")
                for c in cells]
    handles.append(Patch(color=_O["grey"], alpha=0.25, lw=0,
                         label="pointwise 95% band of uniform order statistics"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    ctx.savefig(fig, "e1_pvalue_qq")


def build_real_outputs(ctx: common.ExpContext) -> None:
    if not ctx.has_csv("e1_real_raw"):
        logger.info("no real results: skipping e1_size_real")
        return
    long = load_long(ctx, "e1_real_raw")
    tbl = common.rejection_table(long, ["period", "test", "ptype", "ptype_label", "role"],
                                 pcol="p", alphas=ALPHAS)
    tbl = _sort(tbl, "period", PERIODS)
    ctx.save_csv(tbl, "e1_size_real", config={
        "note": "rates are conditional on the one realised market path; the splits are random "
                "halves of the same panel and are dependent, so the Wilson columns (lo_*, "
                "hi_*), which assume independent splits, do not apply. HAC rows ignore the "
                "positive same-date covariance of the halves (near-vacuous by construction); "
                "the wild-bootstrap rows model it."})
    ctx.save_table(_tex_rates(tbl, "period", PERIOD_TEX, "Placebo"), "e1_size_real",
                   escape=False, digits=3)


def build_nonstat_outputs(ctx: common.ExpContext) -> None:
    if not ctx.has_csv("e1_nonstat_raw"):
        logger.info("no nonstat results: skipping e1_nonstat")
        return
    long = load_long(ctx, "e1_nonstat_raw")
    rows = []
    for (test, ptype), g in long.groupby(["test", "ptype"]):
        p = g["p"].to_numpy(float)
        p = p[np.isfinite(p)]
        rows.append({"test": test, "ptype": ptype, "role": role(test, ptype),
                     "n_seeds": int(p.size), "units": STAT_UNITS.get(test, ""),
                     "statistic_median": float(g["statistic"].median()),
                     "z_hac_median": (float(g["z_hac"].median())
                                      if ptype.startswith("hac") else np.nan),
                     "p_median": float(np.median(p)) if p.size else np.nan,
                     "p_2median": pl.median_pvalue(p) if p.size else np.nan,
                     "share_reject_0.05": float(np.mean(p <= 0.05)) if p.size else np.nan})
    summ = pd.DataFrame(rows)
    plac = None
    if ctx.has_csv("e1_sim_raw"):
        sl = load_long(ctx, "e1_sim_raw")
        sl = sl[sl["kind"] == "pastnew"]
        if not sl.empty:
            plac = common.rejection_table(sl, ["test", "ptype"], pcol="p", alphas=ALPHAS)
            plac = plac.rename(columns={c: f"placebo_{c}" for c in plac.columns
                                        if c not in ("test", "ptype")})
            summ = summ.merge(plac, on=["test", "ptype"], how="left")
    summ = _sort(summ)
    ctx.save_csv(summ, "e1_nonstat")
    t = summ.copy()
    t["Test"] = t["test"].map(TEST_LABEL)
    t["$p$-value"] = [_ptype_tex(a, b) for a, b in zip(t["test"], t["ptype"])]
    t = t.set_index(["Test", "$p$-value"])
    out = pd.DataFrame(index=t.index)
    out["Statistic"] = t["statistic_median"]
    out["Units"] = t["units"]
    out[r"$z_{\mathrm{HAC}}$"] = t["z_hac_median"]
    out["Median $p$"] = t["p_median"]
    out[r"$2\times$median $p$"] = t["p_2median"]
    out[r"Share $p\le 0.05$"] = t["share_reject_0.05"]
    if plac is not None and "placebo_rej_0.05" in t:
        out[r"Stationary placebo, $\alpha=0.05$"] = [
            common.fmt_rate_ci(r, lo, hi) for r, lo, hi in
            zip(t["placebo_rej_0.05"], t["placebo_lo_0.05"], t["placebo_hi_0.05"])]
    ctx.save_table(out, "e1_nonstat", escape=False, digits=3)


def build_outputs(ctx: common.ExpContext, cfg: dict) -> None:
    for fn, args in ((build_sim_outputs, (ctx, cfg)), (build_real_outputs, (ctx,)),
                     (build_nonstat_outputs, (ctx,))):
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 - one broken output must not hide the others
            logger.exception("%s failed", fn.__name__)


# --------------------------------------------------------------------------------------------
# Memory helpers (Linux /proc; NaN elsewhere)
# --------------------------------------------------------------------------------------------
def _proc_kb(path: str, field: str) -> float:
    try:
        with open(path, encoding="ascii") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return float(line.split()[1])
    except OSError:
        pass
    return _NAN


def mem_available_gb() -> float:
    return _proc_kb("/proc/meminfo", "MemAvailable") / 2**20


def rss_gb() -> float:
    return _proc_kb("/proc/self/status", "VmRSS") / 2**20


def _reset_peak_rss() -> bool:
    """Reset this process's peak RSS (``VmHWM``) so the next reading is per task."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def _peak_rss_gb() -> float:
    v = _proc_kb("/proc/self/status", "VmHWM")
    if np.isfinite(v):
        return v / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def check_memory(parts: list[str], workers: int) -> None:
    """Warn when ``workers`` full-settings workers may not fit in the available memory."""
    avail = mem_available_gb()
    if not np.isfinite(avail):
        return
    for part in parts:
        per = MEM_PER_WORKER_GB[part]
        need = workers * per + rss_gb()
        if need > 0.9 * avail:
            fit = max(1, int((0.85 * avail - rss_gb()) // per))
            logger.warning("[%s] %d workers x %.2f GB peak = %.1f GB, but only %.1f GB is "
                           "available: risk of swapping or OOM; use --workers %d or fewer",
                           part, workers, per, need, avail, fit)


# --------------------------------------------------------------------------------------------
# Timing probes (quick mode): PROBE_REPS replications of each cell type at the FULL settings
# --------------------------------------------------------------------------------------------
PROBE_CELLS = {"sim": ("sim_joint", "sim_indep"), "real": ("real_test", "real_train"),
               "nonstat": ("nonstat",)}


def _probe_rep(cell: dict, rep: int, rng: np.random.Generator) -> dict[str, Any]:
    reset = _reset_peak_rss()
    t0 = time.perf_counter()
    kind = cell["kind"]
    if kind == "sim":
        _sim_rep(cell, rep, rng)
    elif kind == "real":
        _real_rep(cell, rep, rng)
    else:
        _nonstat_rep(cell, rep, rng)
    return {"probe_seconds": time.perf_counter() - t0, "peak_rss_gb": _peak_rss_gb(),
            "rss_reset": reset}


def run_probes(ctx: common.ExpContext) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    panel = load_panel(FULL)
    model = fit_factor_model(panel) if "sim" in ctx.parts else None
    fit_s = time.perf_counter() - t0
    cells: list[dict] = []
    if "sim" in ctx.parts:
        cells += [{**c, "cell": f"sim_{c['synth']}", "kind": "sim"}
                  for c in sim_cells(panel, model, FULL)]
    if "real" in ctx.parts:
        cells += [{**c, "cell": f"real_{c['period']}", "kind": "real"}
                  for c in real_cells(panel, FULL)]
    if "nonstat" in ctx.parts:
        cells.append({"cell": "nonstat", "kind": "nonstat", "_settings": _settings(FULL),
                      "_panel": panel})
    if not cells:
        return pd.DataFrame(), fit_s
    cost = {"sim_joint": 5.0, "sim_indep": 4.0, "real_train": 3.0, "real_test": 1.0,
            "nonstat": 2.0}
    df = common.run_cells(_probe_rep, cells, part="probe", ctx=ctx, root_seed=ROOT_SEED,
                          n_reps=PROBE_REPS, config={"probe": True, "full": _settings(FULL)},
                          block_size=1, cost=lambda c: cost[c["cell"]], fresh=True)
    return df, fit_s


def report_estimates(probe: pd.DataFrame, fit_s: float, parts: list[str],
                     workers: int = 12) -> dict[str, float]:
    """Print per-replication timings, wall-time estimates (with a range) and memory needs."""
    est: dict[str, float] = {}
    if probe.empty:
        return est
    t = probe.groupby("cell")["probe_seconds"].agg(["mean", "min", "max", "count"])
    peak = probe.groupby("cell")["peak_rss_gb"].max()
    reset = bool(probe["rss_reset"].astype(bool).all())
    avail, main_gb = mem_available_gb(), rss_gb()
    F, bs = FULL, FULL["blocks"]
    n_reps = {"sim": F["sim_reps"], "real": F["real_splits"], "nonstat": F["nonstat_seeds"]}
    notes = {
        "sim_joint": "6 absolute + 3 ticker-split + 2 relative + R_new*/R_past* absolute",
        "sim_indep": "6 absolute + 3 ticker-split + 2 relative",
        "real_test": "6 absolute", "real_train": "6 absolute + 2 relative",
        "nonstat": "6 absolute, 3000 vs 6200",
    }
    all_tasks = {"mean": [], "min": [], "max": []}
    all_peak = 0.0
    for part in parts:
        cells = [c for c in PROBE_CELLS[part] if c in t.index]
        if not cells:
            continue
        tasks = {"mean": [], "min": [], "max": []}
        for c in cells:
            r = t.loc[c]
            common.print_rep_timing(
                c.replace("_", " "), float(r["mean"]),
                note=f"full settings: {notes[c]}; mean of {int(r['count'])} timed reps, "
                     f"range {r['min']:.1f}-{r['max']:.1f} s")
            for k in tasks:
                tasks[k] += common.block_tasks(float(r[k]), n_reps[part], bs[part])
        extra = fit_s if part == "sim" else 0.0
        est[part] = common.print_wall_estimate(part, tasks["mean"], workers=workers,
                                               extra_s=extra)
        lo = common.estimate_wall(tasks["min"], workers) + extra
        hi = common.estimate_wall(tasks["max"], workers) + extra
        print(f"ESTIMATED FULL-RUN WALL TIME RANGE {part} at {workers} workers: "
              f"{common.fmt_seconds(lo)} - {common.fmt_seconds(hi)} (fastest / slowest timed "
              "replication per cell)", flush=True)
        pk = float(max(peak.loc[c] for c in cells))
        all_peak = max(all_peak, pk)
        fit_w = (max(1, min(workers, int((0.85 * avail - main_gb) // pk)))
                 if np.isfinite(avail) else workers)
        print(f"ESTIMATED PEAK MEMORY {part} at {workers} workers: {workers * pk + main_gb:.1f} "
              f"GB ({pk:.2f} GB per worker{'' if reset else ' (upper bound: not reset)'} + "
              f"{main_gb:.2f} GB main process); MemAvailable now {avail:.1f} GB -> at most "
              f"{fit_w} workers", flush=True)
        if fit_w < workers:
            w_est = common.print_wall_estimate(f"{part} (memory-limited)", tasks["mean"],
                                               workers=fit_w, extra_s=extra)
            est[f"{part}_at_{fit_w}w"] = w_est
            hi_w = common.estimate_wall(tasks["max"], fit_w) + extra
            print(f"ESTIMATED FULL-RUN WALL TIME RANGE {part} at {fit_w} workers: "
                  f"{common.fmt_seconds(common.estimate_wall(tasks['min'], fit_w) + extra)} - "
                  f"{common.fmt_seconds(hi_w)}", flush=True)
        est[f"{part}_peak_gb_per_worker"] = pk
        for k in all_tasks:
            all_tasks[k] += tasks[k]
    if len(parts) > 1:
        est["all"] = common.print_wall_estimate(f"all parts ({'+'.join(parts)}, one process)",
                                                all_tasks["mean"], workers=workers,
                                                extra_s=fit_s)
        print(f"ESTIMATED FULL-RUN WALL TIME RANGE all parts at {workers} workers: "
              f"{common.fmt_seconds(common.estimate_wall(all_tasks['min'], workers) + fit_s)} - "
              f"{common.fmt_seconds(common.estimate_wall(all_tasks['max'], workers) + fit_s)}",
              flush=True)
    return est


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------
def _add_args(parser: Any) -> None:
    parser.add_argument("--no-probe", action="store_true",
                        help="quick mode: skip the full-settings timing probes")


def main(argv: list[str] | None = None) -> None:
    ctx = common.parse_args(EXP, PARTS, description=__doc__, add_arguments=_add_args,
                            argv=argv)
    cfg = QUICK if ctx.quick else FULL
    sw = common.Stopwatch()
    timing: dict[str, float] = {}
    ctx.write_meta({"quick": ctx.quick, "parts": ctx.parts, "plots_only": ctx.plots_only,
                    "cfg": cfg, "root_seed": ROOT_SEED, "sim_burn_in": SIM_BURN_IN,
                    "pipeline": pl.pipeline_config(), "code": code_key()})
    if not ctx.plots_only:
        if not ctx.quick:
            check_memory(ctx.parts, ctx.workers)
        panel = load_panel(cfg)
        logger.info("panel: %d tickers, %d TRAIN / %d TEST days", panel.n_tickers,
                    panel.n_train, panel.n_test)
        if "sim" in ctx.parts:
            model = fit_factor_model(panel)
            timing.update(run_sim(ctx, cfg, panel, model))
        if "real" in ctx.parts:
            timing.update(run_real(ctx, cfg, panel))
        if "nonstat" in ctx.parts:
            timing.update(run_nonstat(ctx, cfg, panel))
    sw.lap("compute")
    build_outputs(ctx, cfg)
    sw.lap("outputs")
    if ctx.quick and not ctx.plots_only and not ctx.args.no_probe:
        probe, fit_s = run_probes(ctx)
        sw.lap("probes")
        est = report_estimates(probe, fit_s, ctx.parts)
        timing.update({f"estimate_{k}": v for k, v in est.items()})
    timing.update({f"wall_{k}_s": v for k, v in sw.laps.items()})
    ctx.write_meta(None, timing=timing)
    print(f"TIMING total wall (this run, {ctx.workers} workers): "
          f"{common.fmt_seconds(sw.total())}", flush=True)


if __name__ == "__main__":
    main()
