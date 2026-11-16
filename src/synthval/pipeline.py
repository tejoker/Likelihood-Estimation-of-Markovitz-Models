"""Shared real-data pipeline for experiments E1, E3 and E5 (DESIGN sections 1, 2, 3.7 and 4).

One place for every step that the real-data experiments must do identically:

1. **Panel** (:func:`load_real_panel`): the default 100-ticker panel of :func:`synthval.data.
   load_panel`, its TRAIN ("real past") and TEST ("real new") daily log-returns and the per-ticker
   TRAIN standard deviation, the fixed scale that divides real *and* synthetic returns of a
   ticker before pooling (DESIGN section 2).
2. **Windows** (:func:`panel_windows`): non-overlapping windows of length ``L`` pooled over
   tickers, with the ticker index of each window (``groups``) and the **date rank of its first
   day** on the common trading calendar (``order``). TRAIN days have ranks ``0 .. T_train - 1``
   and TEST days ``T_train .. T_train + T_test - 1`` (``date_offset = T_train``), so TRAIN and
   TEST windows never share a rank and TEST follows TRAIN. Synthetic paths simulated over the
   TEST horizon get the TEST ranks (the simulated dates), as DESIGN section 4 (E3) prescribes.
   With ``step = L`` every per-period order is a multiple of ``L`` away from its period start, so
   :func:`synthval.twosample.time_grid` places it on a grid of one window per step (DESIGN
   section 3.7 asks for date ranks or window indices, never trading days with holes).
3. **Models** (:func:`fit_panel`, :func:`simulate_panel`): one univariate model per ticker fitted
   on TRAIN (parallel over tickers), and one synthetic path per ticker over a given horizon,
   deterministic given the seed (ticker ``i`` uses child ``i`` of ``SeedSequence(seed)``),
   starting from the fitted end state when the model has one (``start_from_end=True``: GARCHt,
   Heston). A fitted :class:`~synthval.models.MultiReturnModel` (e.g. the ``FactorGARCHt``
   panel of the E1 placebo) is accepted too.
4. **Tests** (:func:`absolute_tests`, :func:`relative_tests`) under one of two protocols:

   * ``protocol="dependent"`` (real data, long simulated paths; DESIGN section 3.7 "Inference
     protocol"): learned tests (critic, lrcritic, c2st) use a contiguous time-block train/eval
     split with ``time_blocks = 10`` built from ``order_*`` and report the Driscoll-Kraay /
     Newey-West HAC p-value (``inference="hac"``); MMD and signature MMD report the wild
     bootstrap p-value (``inference="wild"``); KS is reported but flagged ``valid = False``
     (it assumes i.i.d. returns). Relative tests use the same base settings and
     ``inference="hac"``.
   * ``protocol="iid"`` (independent windows): permutation p-values (random train/eval split;
     relative tests: anchored permutation, two-critic paired bootstrap).

   Both protocols also report the other kind of p-value where it is cheap (learned tests always
   carry ``p_perm`` and, with ``order_*``, ``p_hac``; kernel tests get both ``p_perm`` and
   ``p_wild`` when ``all_pvalues=True``), so E1 can tabulate the size of every p-value type.
5. **Decision rule** (DESIGN section 1): :func:`power_check` (pre-specified power of the
   anchored relative test, GARCHt reference panel, synthetic volatility x 1.2), and
   :func:`verdict` (``BAD`` / ``GOOD`` / ``INCONCLUSIVE``), with :func:`wilson_ci`,
   :func:`clopper_pearson_lower` and :func:`median_pvalue` (twice the median p-value over
   seeds, theory.md Remark 8.5).

:func:`evaluate_model` chains steps 3 and 4 for one fitted model and one seed (simulate the TEST
horizon, cut windows, run the absolute and relative tests) and is what E3 and E5 loop over.

Randomness: every stochastic function takes ``rng`` or ``seed``. Inside :func:`absolute_tests`
and :func:`relative_tests` each test gets its own generator derived from one draw of ``rng`` and
the test's name, so the result of a test does not depend on which other tests are run with it.

Defaults are module constants (``WINDOW_L``, ``ALPHA``, ``TIME_BLOCKS``, ``N_PERM``,
``N_PERM_KERNEL``, ``N_BOOT``, ``CRITIC_KW``, ``POWER_*``); experiment scripts should record
:func:`pipeline_config` in their ``meta.json``.
"""
from __future__ import annotations

import inspect
import logging
import math
import time
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from . import config, data
from .models import (MULTIVARIATE_MODELS, PANEL_MODELS, MultiReturnModel, ReturnModel,
                     get_model, get_panel_model)
from .twosample import TEST_REGISTRY, relative_test
from .utils import parallel_map, spawn_rngs

logger = logging.getLogger(__name__)

__all__ = [
    # constants
    "WINDOW_L", "ALPHA", "TIME_BLOCKS", "N_PERM", "N_PERM_KERNEL", "KERNEL_MAX_N", "N_BOOT",
    "CRITIC_KW", "DEFAULT_TEST_KW", "ABSOLUTE_TESTS", "LEARNED_TESTS", "KERNEL_TESTS",
    "RELATIVE_METHODS", "PROTOCOLS", "POWER_MODEL", "POWER_VOL_FACTOR", "POWER_N_REPS",
    "POWER_MIN", "BURN_IN",
    # panel and windows
    "RealPanel", "load_real_panel", "panel_windows",
    # models
    "fit_panel", "simulate_panel",
    # tests
    "protocol_kw", "make_test", "absolute_tests", "relative_tests", "evaluate_model",
    "relative_method_available",
    # decision rule
    "PowerCheckResult", "power_check", "verdict", "wilson_ci", "clopper_pearson_lower",
    "median_pvalue", "pipeline_config",
]

Protocol = Literal["dependent", "iid"]

# ----------------------------------------------------------------------------------------------
# Defaults (DESIGN sections 2, 3.7 and 1)
# ----------------------------------------------------------------------------------------------
#: Window length (non-overlapping windows, ``step = L``).
WINDOW_L: int = config.WINDOW_L
#: Level of every test and of the verdict.
ALPHA: float = config.ALPHA
#: Contiguous time blocks of the train/eval split of learned tests under the dependent protocol.
TIME_BLOCKS: int = 10
#: Evaluation-label permutations of the learned tests.
N_PERM: int = config.N_PERM
#: Permutations / wild-bootstrap replicates of MMD and signature MMD (DESIGN section 3.7: 500).
N_PERM_KERNEL: int = 500
#: Subsample size cap of the kernel tests (memory ``8 (2 max_n)^2`` bytes).
KERNEL_MAX_N: int = 3000
#: Paired-bootstrap replicates of the two-critic relative test.
N_BOOT: int = 500
#: Burn-in of simulations that start from stationarity.
BURN_IN: int = 250
#: Wasserstein critic settings (DESIGN section 3.7, with the post-review ``lr = 1e-3``).
CRITIC_KW: dict[str, Any] = {"n_steps": 1000, "batch_size": 128, "lr": 1e-3,
                             "betas": (0.5, 0.9), "gp_lambda": 10.0, "n_perm": N_PERM}
#: Constructor settings of each test of ``TEST_REGISTRY`` used by the pipeline (the protocol
#: settings of :func:`protocol_kw` and the caller's ``test_kw`` are applied on top).
DEFAULT_TEST_KW: dict[str, dict[str, Any]] = {
    "critic": CRITIC_KW,
    "lrcritic": {"n_perm": N_PERM},
    "c2st": {"n_perm": N_PERM},
    "mmd": {"n_perm": N_PERM_KERNEL, "max_n": KERNEL_MAX_N},
    "sigmmd": {"n_perm": N_PERM_KERNEL, "max_n": KERNEL_MAX_N},
    "ks": {},
}
#: Pooled absolute tests of E1 / E3.
ABSOLUTE_TESTS: tuple[str, ...] = ("critic", "lrcritic", "c2st", "mmd", "sigmmd", "ks")
LEARNED_TESTS: tuple[str, ...] = ("critic", "lrcritic", "c2st")
KERNEL_TESTS: tuple[str, ...] = ("mmd", "sigmmd")
#: Relative-test versions: anchored (primary) and two-critic (secondary).
RELATIVE_METHODS: tuple[str, ...] = ("anchored", "two_critic")
PROTOCOLS: tuple[str, ...] = ("dependent", "iid")
#: Power check (DESIGN section 1, item 3): reference model, deviation, replications, threshold.
POWER_MODEL: str = "GARCHt"
POWER_VOL_FACTOR: float = 1.2
POWER_N_REPS: int = 50
POWER_MIN: float = 0.8

_NAN = float("nan")


def pipeline_config() -> dict[str, Any]:
    """The pipeline defaults as a JSON-friendly dict (for ``utils.save_results`` metadata)."""
    return {
        "WINDOW_L": WINDOW_L, "ALPHA": ALPHA, "TIME_BLOCKS": TIME_BLOCKS, "N_PERM": N_PERM,
        "N_PERM_KERNEL": N_PERM_KERNEL, "KERNEL_MAX_N": KERNEL_MAX_N, "N_BOOT": N_BOOT,
        "BURN_IN": BURN_IN, "CRITIC_KW": {k: (list(v) if isinstance(v, tuple) else v)
                                          for k, v in CRITIC_KW.items()},
        "ABSOLUTE_TESTS": list(ABSOLUTE_TESTS), "RELATIVE_METHODS": list(RELATIVE_METHODS),
        "POWER_MODEL": POWER_MODEL, "POWER_VOL_FACTOR": POWER_VOL_FACTOR,
        "POWER_N_REPS": POWER_N_REPS, "POWER_MIN": POWER_MIN,
        "TRAIN": list(config.TRAIN), "TEST": list(config.TEST), "N_PANEL": config.N_PANEL,
        "PANEL_SEED": config.PANEL_SEED,
        # Runtime protocol (part of every checkpoint key that includes this dict): device and the
        # architectures chosen by arch="auto" (v1: cpu, mlp, conv; v2: cuda, pooled, pooled).
        **runtime_config(),
    }


def runtime_config() -> dict[str, Any]:
    """Device, ``arch="auto"`` choices and torch version of this process."""
    import torch

    from .twosample.nets import default_arch, get_device
    return {"DEVICE": str(get_device()), "ARCH_1D": default_arch(1), "ARCH_2D": default_arch(2),
            "TORCH": torch.__version__}


# ----------------------------------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------------------------------
def _scale_array(returns: pd.DataFrame | np.ndarray, scale: Any, n_cols: int) -> np.ndarray | None:
    """Per-column scales as a float array of length ``n_cols`` (``None``: no scaling)."""
    if scale is None:
        return None
    if isinstance(scale, pd.Series):
        if isinstance(returns, pd.DataFrame):
            missing = [c for c in returns.columns if c not in scale.index]
            if missing:
                raise KeyError(f"no scale for tickers {missing}")
            v = scale.loc[returns.columns].to_numpy(dtype=np.float64)
        else:
            v = scale.to_numpy(dtype=np.float64)
    else:
        v = np.asarray(scale, dtype=np.float64)
        if v.ndim == 0:
            v = np.full(n_cols, float(v))
    if v.shape != (n_cols,):
        raise ValueError(f"scale must have one value per column ({n_cols}), got shape {v.shape}")
    if not np.all(np.isfinite(v) & (v > 0)):
        raise ValueError("scales must be finite and positive")
    return v


def panel_windows(returns: pd.DataFrame | np.ndarray, L: int = WINDOW_L,
                  step: int | None = None, *, scale: pd.Series | np.ndarray | float | None,
                  date_offset: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised windows of a panel, pooled over tickers, with ticker and date-rank labels.

    Parameters
    ----------
    returns : DataFrame or ndarray, shape (T, N)
        Daily log-returns, one column per ticker (real or simulated).
    L : int, default ``WINDOW_L`` (20)
        Window length.
    step : int, optional
        Distance between window starts; default ``L`` (non-overlapping). An incomplete tail is
        dropped.
    scale : Series, ndarray (N,), float or None (keyword, required)
        Fixed per-ticker scale dividing the returns before windowing: the TRAIN standard
        deviation of the *real* returns of each ticker, for real and synthetic data alike
        (DESIGN section 2). A Series is aligned on the DataFrame columns; with an ndarray of
        returns it is used in its own order (columns must be in the same ticker order). Pass
        ``None`` explicitly to keep raw returns.
    date_offset : int, default 0
        Date rank of the first row of ``returns`` on the common trading calendar: ``0`` for
        TRAIN, ``len(train)`` for TEST and for synthetic paths over the TEST horizon.

    Returns
    -------
    W : ndarray, shape (N * n, L)
        Windows, ticker-major (all windows of column 0 in time order, then column 1, ...).
    groups : ndarray of int64, shape (N * n,)
        Column position of the ticker of each window.
    order : ndarray of int64, shape (N * n,)
        Date rank of each window's first day, ``date_offset + k * step`` for the ``k``-th window
        of a ticker; windows of different tickers on the same dates share the rank (the
        Driscoll-Kraay HAC and the wild bootstrap aggregate them).
    """
    a = np.asarray(returns, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 1:
        raise ValueError(f"returns must be 2-D (T, N) with N >= 1, got shape {a.shape}")
    v = _scale_array(returns, scale, a.shape[1])
    if v is not None:
        a = a / v
    step = L if step is None else int(step)
    if int(date_offset) != date_offset or date_offset < 0:
        raise ValueError("date_offset must be a non-negative integer")
    W, groups, pos = data.panel_windows(a, L, step)
    order = (int(date_offset) + pos * step).astype(np.int64)
    return W, groups.astype(np.int64), order


# ----------------------------------------------------------------------------------------------
# Real panel
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RealPanel:
    """TRAIN / TEST returns of a ticker panel and its fixed per-ticker scale.

    Attributes
    ----------
    train, test : DataFrame
        Date x ticker daily log-returns over ``config.TRAIN`` and ``config.TEST`` (same columns).
    scale : Series
        Per-ticker TRAIN standard deviation (``data.train_vol(train)``), the normalisation of real
        and synthetic windows.
    L : int
        Window length.
    step : int or None
        Window step (``None``: ``L``).
    """

    train: pd.DataFrame
    test: pd.DataFrame
    scale: pd.Series
    L: int = WINDOW_L
    step: int | None = None

    def __post_init__(self) -> None:
        if list(self.train.columns) != list(self.test.columns):
            raise ValueError("train and test must have the same columns in the same order")
        missing = [t for t in self.train.columns if t not in self.scale.index]
        if missing:
            raise KeyError(f"no scale for tickers {missing}")

    @property
    def tickers(self) -> list[str]:
        return list(self.train.columns)

    @property
    def n_tickers(self) -> int:
        return self.train.shape[1]

    @property
    def n_train(self) -> int:
        return self.train.shape[0]

    @property
    def n_test(self) -> int:
        return self.test.shape[0]

    @property
    def test_offset(self) -> int:
        """Date rank of the first TEST day on the common calendar (``= n_train``)."""
        return self.n_train

    def offset(self, period: Literal["train", "test"]) -> int:
        """Date rank of the first day of ``period``."""
        if period == "train":
            return 0
        if period == "test":
            return self.test_offset
        raise ValueError("period must be 'train' or 'test'")

    def returns(self, period: Literal["train", "test"]) -> pd.DataFrame:
        if period not in ("train", "test"):
            raise ValueError("period must be 'train' or 'test'")
        return self.train if period == "train" else self.test

    def scale_array(self) -> np.ndarray:
        """Scales in ticker (column) order."""
        return self.scale.loc[self.tickers].to_numpy(dtype=np.float64)

    def windows(self, period: Literal["train", "test"] = "test"
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Real windows ``(W, groups, order)`` of ``period`` (see :func:`panel_windows`)."""
        return panel_windows(self.returns(period), self.L, self.step, scale=self.scale,
                             date_offset=self.offset(period))

    def synthetic_windows(self, sim: np.ndarray | pd.DataFrame,
                          period: Literal["train", "test"] = "test"
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Windows of a simulated panel with the real scale and the date ranks of ``period``.

        ``sim`` has shape ``(n_steps, N)`` with columns in ticker order (as returned by
        :func:`simulate_panel` for models fitted on ``train``), or is a DataFrame with ticker
        columns. ``n_steps`` normally equals the length of ``period``, so synthetic windows have
        exactly the date ranks of the real windows of that period.
        """
        if isinstance(sim, pd.DataFrame):
            if set(sim.columns) != set(self.tickers):
                raise ValueError("sim columns must be the panel tickers")
            sim = sim.loc[:, self.tickers]
        a = np.asarray(sim, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != self.n_tickers:
            raise ValueError(f"sim must have shape (n_steps, {self.n_tickers}), got {a.shape}")
        if a.shape[0] != self.returns(period).shape[0]:
            logger.info("synthetic_windows: %d simulated days vs %d in %s", a.shape[0],
                        self.returns(period).shape[0], period)
        return panel_windows(a, self.L, self.step, scale=self.scale_array(),
                             date_offset=self.offset(period))


def load_real_panel(tickers: Sequence[str] | None = None, *, L: int = WINDOW_L,
                    step: int | None = None) -> RealPanel:
    """The default real panel (``data.load_panel``) with its per-ticker TRAIN standard deviation.

    Parameters
    ----------
    tickers : sequence of str, optional
        Default: ``data.panel_tickers()`` (``N_PANEL = 100`` tickers, ``PANEL_SEED``).
    L, step : int
        Window length and step used by :meth:`RealPanel.windows`.

    Returns
    -------
    RealPanel
        On the current file: 1258 TRAIN days (62 windows of 20 per ticker) and 615 TEST days
        (30 windows per ticker), so 6200 TRAIN and 3000 TEST windows for 100 tickers.
    """
    train, test = data.load_panel(tickers)
    return RealPanel(train=train, test=test, scale=data.train_vol(train), L=L, step=step)


# ----------------------------------------------------------------------------------------------
# Models: per-ticker fits and panel simulation
# ----------------------------------------------------------------------------------------------
def _fit_one(item: tuple[str, str, np.ndarray, dict[str, Any], Any]) -> tuple[str, Any, float]:
    """Worker: fit one univariate model on one ticker's returns."""
    model_name, ticker, r, model_kw, fit_seed = item
    model = get_model(model_name, **model_kw)
    t0 = time.perf_counter()
    if fit_seed is not None and "rng" in inspect.signature(model.fit).parameters:
        model.fit(r, rng=np.random.default_rng(fit_seed))
    else:
        model.fit(r)
    return ticker, model, time.perf_counter() - t0


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (bool, int, float, str, np.bool_, np.integer, np.floating))


def fit_panel(model_name: str, returns_train: pd.DataFrame, *, n_workers: int | None = None,
              fit_seed: int | None = None, **model_kw: Any
              ) -> tuple[dict[str, ReturnModel], pd.DataFrame]:
    """Fit one univariate model per ticker on TRAIN returns, in parallel over tickers.

    Parameters
    ----------
    model_name : str
        Univariate registry name (``models.UNIVARIATE_MODELS``: ``"GBM"``, ``"Merton"``,
        ``"GARCHt"``, ``"Heston"``, ``"Bootstrap"``). Multivariate models are fitted once on the
        whole panel (``models.get_mv_model(name).fit(train.to_numpy())``) and passed to
        :func:`simulate_panel` directly. Pooled panel models (``models.PANEL_MODELS``:
        ``"QuantGAN"``) are trained once on all tickers and returned as per-ticker views
        (:func:`_fit_pooled_panel`; ``n_workers`` is ignored).
    returns_train : DataFrame (T, N)
        TRAIN log-returns, one column per ticker (an ndarray gets tickers ``"0", "1", ...``).
    n_workers : int, optional
        Worker processes (``utils.parallel_map``; default ``config.n_workers()``).
    fit_seed : int, optional
        If given and the model's ``fit`` accepts ``rng`` (Heston), ticker ``i`` is fitted with
        child ``i`` of ``SeedSequence(fit_seed)``; by default each model uses its own
        deterministic default (Heston: ``default_rng(seed)``, common random numbers).
    **model_kw
        Constructor options (e.g. ``variance_targeting=True`` for GARCHt).

    Returns
    -------
    fitted : dict ticker -> fitted model
        In column order (the order :func:`simulate_panel` uses).
    table : DataFrame
        One row per ticker (index ``"ticker"``): the fitted parameters, ``at_bound`` (parameters
        at an optimiser bound, ``";"``-joined), ``n_at_bound``, every scalar entry of the
        model's ``fit_info_`` (e.g. ``success``, ``persistence``, ``uncond_ratio``, Heston's
        ``objective``, ``feller_ok``, ``frac_zero_var``; array entries such as Heston's
        ``m_data`` / ``m_sim`` stay on the model objects) and ``fit_seconds`` (wall time of
        ``fit`` in the worker). ``table.attrs`` holds ``model`` and ``model_kw``.
    """
    if model_name in MULTIVARIATE_MODELS:
        raise ValueError(f"{model_name} is multivariate: fit it once on the panel and pass the "
                         "fitted model to simulate_panel")
    if not isinstance(returns_train, pd.DataFrame):
        a = np.asarray(returns_train, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError("returns_train must be 2-D (T, N)")
        returns_train = pd.DataFrame(a, columns=[str(i) for i in range(a.shape[1])])
    tickers = list(returns_train.columns)
    if len(set(tickers)) != len(tickers):
        raise ValueError("duplicated ticker columns")
    if model_name in PANEL_MODELS:
        return _fit_pooled_panel(model_name, returns_train, fit_seed=fit_seed, **model_kw)
    seeds: list[Any] = ([None] * len(tickers) if fit_seed is None
                        else np.random.SeedSequence(int(fit_seed)).spawn(len(tickers)))
    items = [(model_name, t, returns_train[t].to_numpy(dtype=np.float64), dict(model_kw), s)
             for t, s in zip(tickers, seeds)]
    t0 = time.perf_counter()
    results = parallel_map(_fit_one, items, n_workers)
    logger.info("fit_panel(%s): %d tickers in %.1f s", model_name, len(tickers),
                time.perf_counter() - t0)
    fitted: dict[str, ReturnModel] = {}
    rows: dict[str, dict[str, Any]] = {}
    for ticker, model, secs in results:
        fitted[ticker] = model
        row: dict[str, Any] = {}
        try:
            row.update({k: float(v) for k, v in model.params().items()})
        except (RuntimeError, NotImplementedError):  # pragma: no cover - defensive
            pass
        info = dict(getattr(model, "fit_info_", {}) or {})
        at = [str(a) for a in info.pop("at_bound", [])]
        row["at_bound"] = ";".join(at)
        row["n_at_bound"] = len(at)
        info.pop("seconds", None)
        for k, v in info.items():
            if _is_scalar(v):
                row[k if k not in row else f"info_{k}"] = v.item() if hasattr(v, "item") else v
        row["fit_seconds"] = float(secs)
        rows[ticker] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "ticker"
    table.attrs["model"] = model_name
    table.attrs["model_kw"] = dict(model_kw)
    return fitted, table


def _fit_pooled_panel(model_name: str, returns_train: pd.DataFrame, *, fit_seed: int | None,
                      **model_kw: Any) -> tuple[dict[str, ReturnModel], pd.DataFrame]:
    """:func:`fit_panel` for pooled panel models (``models.PANEL_MODELS``, e.g. ``"QuantGAN"``).

    ONE model is trained (in this process, on ``nets.get_device()``) on all columns of
    ``returns_train``, each standardised by its own mean and std; ``fit_seed`` seeds
    ``default_rng(fit_seed)`` (default: the model's own ``seed``). Returns per-ticker views
    (``model.series_models(tickers)``, :class:`~synthval.models.ReturnModel` objects sharing the
    trained network), so :func:`simulate_panel` and :func:`evaluate_model` use them like
    per-ticker fits, and a table with one row per ticker: the ticker's ``mean`` / ``std``, the
    shared scalar ``fit_info_`` entries (same value in every row), ``at_bound``, ``n_at_bound``
    and ``fit_seconds`` = total fit time / number of tickers (so the column sums to the total).
    """
    model = get_panel_model(model_name, **model_kw)
    rng = None if fit_seed is None else np.random.default_rng(int(fit_seed))
    t0 = time.perf_counter()
    model.fit(returns_train.to_numpy(dtype=np.float64), rng=rng)
    secs = time.perf_counter() - t0
    tickers = list(returns_train.columns)
    fitted: dict[str, ReturnModel] = model.series_models(tickers)
    info = dict(getattr(model, "fit_info_", {}) or {})
    at = [str(a) for a in info.pop("at_bound", [])]
    info.pop("seconds", None)
    shared = {k: (v.item() if hasattr(v, "item") else v) for k, v in info.items()
              if _is_scalar(v)}
    rows: dict[str, dict[str, Any]] = {}
    for ticker, view in fitted.items():
        row: dict[str, Any] = {k: float(v) for k, v in view.params().items()}
        row["at_bound"] = ";".join(at)
        row["n_at_bound"] = len(at)
        for k, v in shared.items():
            row[k if k not in row else f"info_{k}"] = v
        row["fit_seconds"] = secs / len(tickers)
        rows[ticker] = row
    logger.info("fit_panel(%s, pooled): %d tickers in %.1f s", model_name, len(tickers), secs)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "ticker"
    table.attrs["model"] = model_name
    table.attrs["model_kw"] = dict(model_kw)
    table.attrs["pooled"] = True
    return fitted, table


def _simulate_kwargs(sim_fn: Any, start_from_end: bool, burn_in: int) -> dict[str, Any]:
    params = inspect.signature(sim_fn).parameters
    kw: dict[str, Any] = {"burn_in": burn_in} if "burn_in" in params else {}
    if "start_from_end" in params:
        kw["start_from_end"] = start_from_end
    elif start_from_end:
        logger.debug("%s.simulate has no start_from_end option", sim_fn)
    return kw


def simulate_panel(fitted: Mapping[str, ReturnModel] | MultiReturnModel, n_steps: int, *,
                   seed: int | np.random.SeedSequence | np.random.Generator,
                   start_from_end: bool = True, burn_in: int = BURN_IN) -> np.ndarray:
    """One synthetic path per ticker over ``n_steps`` days, deterministic given ``seed``.

    Parameters
    ----------
    fitted : mapping ticker -> fitted ReturnModel, or a fitted MultiReturnModel
        Per-ticker models (columns in mapping order, as returned by :func:`fit_panel`), or one
        multivariate panel model (e.g. ``FactorGARCHt``, whose ``simulate`` gives ``(n_steps,
        N)`` directly).
    n_steps : int
        Simulated days (``len(test)`` for the TEST horizon).
    seed : int, SeedSequence or Generator
        Ticker ``i`` uses child ``i`` of ``SeedSequence(seed)`` (``utils.spawn_rngs``), so the
        path of a ticker depends only on the seed and its position. A multivariate model uses
        ``default_rng(seed)``. Passing a Generator spawns from it (not reproducible across calls
        on the same generator).
    start_from_end : bool, default True
        Start from the fitted end-of-TRAIN state when the model has one (GARCHt: one-step
        variance forecast; Heston: EWMA variance); other models are stationary and ignore it.
    burn_in : int, default ``BURN_IN``
        Discarded days when starting from stationarity.

    Returns
    -------
    ndarray, shape (n_steps, N), float64
    """
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if isinstance(fitted, MultiReturnModel):
        rng = (seed if isinstance(seed, np.random.Generator)
               else np.random.default_rng(seed))
        kw = _simulate_kwargs(fitted.simulate, start_from_end, burn_in)
        out = np.asarray(fitted.simulate(n_steps, rng=rng, **kw), dtype=np.float64)
        if out.ndim != 2 or out.shape[0] != n_steps:
            raise ValueError(f"multivariate simulate returned shape {out.shape}")
        return np.ascontiguousarray(out)
    models = list(fitted.values())
    if not models:
        raise ValueError("no fitted models")
    rngs = spawn_rngs(seed, len(models))
    out = np.empty((n_steps, len(models)), dtype=np.float64)
    for j, (model, rng) in enumerate(zip(models, rngs)):
        kw = _simulate_kwargs(model.simulate, start_from_end, burn_in)
        out[:, j] = np.asarray(model.simulate(n_steps, 1, rng=rng, **kw))[0]
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("simulate_panel produced non-finite returns")
    return out


# ----------------------------------------------------------------------------------------------
# Test construction and per-test random streams
# ----------------------------------------------------------------------------------------------
def _check_protocol(protocol: str) -> None:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")


def protocol_kw(name: str, protocol: Protocol) -> dict[str, Any]:
    """Constructor settings that a protocol imposes on test ``name`` (DESIGN section 3.7).

    ``"dependent"``: learned tests ``time_blocks=TIME_BLOCKS, inference="hac"``; kernel tests
    ``inference="wild"``. ``"iid"``: learned tests ``time_blocks=None,
    inference="permutation"``; kernel tests ``inference="permutation"``. KS: nothing.
    """
    _check_protocol(protocol)
    if name in LEARNED_TESTS:
        if protocol == "dependent":
            return {"time_blocks": TIME_BLOCKS, "inference": "hac"}
        return {"time_blocks": None, "inference": "permutation"}
    if name in KERNEL_TESTS:
        return {"inference": "wild" if protocol == "dependent" else "permutation"}
    return {}


def make_test(name: str, protocol: Protocol = "dependent", **overrides: Any) -> Any:
    """Instantiate ``TEST_REGISTRY[name]`` with pipeline defaults, protocol settings, overrides.

    Precedence: ``DEFAULT_TEST_KW[name]`` < :func:`protocol_kw` < ``overrides`` (so E5 can vary
    e.g. ``time_blocks``, ``lr`` or ``hac_lags``).
    """
    if name not in TEST_REGISTRY:
        raise KeyError(f"unknown test {name!r}; options: {sorted(TEST_REGISTRY)}")
    kw = {**DEFAULT_TEST_KW.get(name, {}), **protocol_kw(name, protocol), **overrides}
    return TEST_REGISTRY[name](**kw)


def _draw_seed(rng: np.random.Generator) -> int:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return int(rng.integers(0, 2**63 - 1))


def _sub_rng(base_seed: int, key: str) -> np.random.Generator:
    """Generator determined by ``(base_seed, key)`` (independent of the other tests run)."""
    return np.random.default_rng(np.random.SeedSequence([base_seed, zlib.crc32(key.encode())]))


def _f(v: Any) -> float:
    """Float or NaN."""
    try:
        out = float(v)
    except (TypeError, ValueError):
        return _NAN
    return out


# ----------------------------------------------------------------------------------------------
# Absolute tests
# ----------------------------------------------------------------------------------------------
def absolute_tests(X: np.ndarray, Y: np.ndarray, *, order_x: np.ndarray | None = None,
                   order_y: np.ndarray | None = None, tests: Sequence[str] = ABSOLUTE_TESTS,
                   protocol: Protocol = "dependent", rng: np.random.Generator,
                   test_kw: Mapping[str, Mapping[str, Any]] | None = None,
                   groups_x: np.ndarray | None = None, groups_y: np.ndarray | None = None,
                   all_pvalues: bool = True, keep_extra: bool = False) -> list[dict[str, Any]]:
    """Pooled absolute two-sample tests ``H0: Law(X) = Law(Y)`` (X real, Y synthetic).

    Parameters
    ----------
    X, Y : arrays (n, L) (or (n, N, L))
        Real and synthetic windows (already normalised, see :func:`panel_windows`).
    order_x, order_y : int arrays, optional
        Date rank of each window's first day. Required by ``protocol="dependent"``.
    tests : sequence of str, default ``ABSOLUTE_TESTS``
        Names in ``TEST_REGISTRY``.
    protocol : {"dependent", "iid"}, default "dependent"
        See the module docstring and :func:`protocol_kw`.
    rng : numpy.random.Generator
        One integer is drawn from it; each test then uses its own generator derived from that
        integer and the test name.
    test_kw : mapping test name -> dict, optional
        Constructor overrides per test (applied after the protocol settings).
    groups_x, groups_y : int arrays, optional
        Ticker labels; used only by learned tests under ``protocol="iid"`` without time blocks
        (group train/eval split). Ignored under the dependent protocol (time blocks).
    all_pvalues : bool, default True
        For MMD / SigMMD, also compute the p-value of the other inference (permutation under the
        dependent protocol, wild bootstrap under the iid protocol when orders are given) by a
        second call with the same generator (same subsample and bandwidth); doubles their cost.
    keep_extra : bool, default False
        Add the test's ``extra`` dict to each row (training curves etc.; not CSV-friendly).

    Returns
    -------
    list of dict
        One row per test with keys ``test``, ``protocol``, ``statistic``, ``pvalue`` (the
        p-value the protocol uses), ``pvalue_type`` (``"hac"``, ``"wild"``, ``"permutation"``,
        ``"normal"`` or ``"asymptotic_iid"`` for KS), ``valid`` (False for KS under the dependent
        protocol, where it ignores serial dependence), ``p_perm``, ``p_hac`` (learned tests,
        with orders), ``p_wild`` (kernel tests), ``z``, ``z_hac`` (learned tests), ``n_x``,
        ``n_y``, ``n_eval_x``, ``n_eval_y``, ``split``, ``hac_lags``, ``wild_block``,
        ``runtime_s`` (seconds, including the second kernel call).
    """
    _check_protocol(protocol)
    if protocol == "dependent" and (order_x is None or order_y is None):
        raise ValueError("protocol='dependent' needs order_x and order_y (date ranks)")
    test_kw = {k: dict(v) for k, v in (test_kw or {}).items()}
    unknown = sorted(set(test_kw) - set(TEST_REGISTRY))
    if unknown:
        raise KeyError(f"test_kw for unknown tests {unknown}")
    for name in tests:
        if name not in TEST_REGISTRY:
            raise KeyError(f"unknown test {name!r}; options: {sorted(TEST_REGISTRY)}")
    base_seed = _draw_seed(rng)
    has_order = order_x is not None and order_y is not None
    rows: list[dict[str, Any]] = []
    for name in tests:
        kw = test_kw.get(name, {})
        t0 = time.perf_counter()
        test = make_test(name, protocol, **kw)
        call_kw: dict[str, Any] = {"order_x": order_x, "order_y": order_y}
        if (name in LEARNED_TESTS and protocol == "iid"
                and getattr(test, "time_blocks", None) is None):
            call_kw.update(groups_x=groups_x, groups_y=groups_y)
        res = test(X, Y, rng=_sub_rng(base_seed, name), **call_kw)
        ex = res.extra
        row: dict[str, Any] = {
            "test": name, "protocol": protocol, "statistic": float(res.statistic),
            "pvalue": float(res.pvalue), "pvalue_type": "", "valid": True,
            "p_perm": _NAN, "p_hac": _NAN, "p_wild": _NAN, "z": _NAN, "z_hac": _NAN,
            "n_x": int(res.n_x), "n_y": int(res.n_y), "n_eval_x": _NAN, "n_eval_y": _NAN,
            "split": "", "hac_lags": "", "wild_block": _NAN,
        }
        if name in LEARNED_TESTS:
            inference = getattr(test, "inference", "permutation")
            row["pvalue_type"] = ("hac" if inference == "hac"
                                  else "permutation" if test.n_perm > 0 else "normal")
            row.update(p_perm=_f(ex.get("p_perm")), z=_f(ex.get("z")),
                       n_eval_x=ex.get("n_eval_x", _NAN), n_eval_y=ex.get("n_eval_y", _NAN),
                       split=str(ex.get("split", "")))
            if ex.get("hac"):
                row.update(p_hac=_f(ex.get("p_hac")), z_hac=_f(ex.get("z_hac")),
                           hac_lags=str(ex.get("hac_lags")))
        elif name in KERNEL_TESTS:
            inference = getattr(test, "inference", "permutation")
            row["pvalue_type"] = inference
            other = None
            other_inf = "permutation" if inference == "wild" else "wild"
            if all_pvalues and (other_inf == "permutation" or has_order):
                other_test = make_test(name, protocol, **{**kw, "inference": other_inf})
                other = other_test(X, Y, rng=_sub_rng(base_seed, name), **call_kw)
            p_main, p_other = float(res.pvalue), (_NAN if other is None else float(other.pvalue))
            row["p_perm"] = p_main if inference == "permutation" else p_other
            row["p_wild"] = p_main if inference == "wild" else p_other
            wild_ex = ex if inference == "wild" else (other.extra if other is not None else {})
            row["wild_block"] = _f(wild_ex.get("wild_block"))
            row["n_eval_x"], row["n_eval_y"] = ex.get("n_used_x", _NAN), ex.get("n_used_y", _NAN)
        else:  # KS (and any other test without a learned witness or kernel)
            row["pvalue_type"] = "asymptotic_iid"
            row["valid"] = protocol == "iid"
        row["runtime_s"] = time.perf_counter() - t0
        if keep_extra:
            row["extra"] = ex
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------
# Relative tests
# ----------------------------------------------------------------------------------------------
def relative_method_available(method: str = "anchored") -> bool:
    """Whether the installed ``relative_test`` accepts ``method=<method>``."""
    params = inspect.signature(relative_test).parameters
    if "method" not in params:
        return method == "two_critic"
    return True


def relative_tests(R_new: np.ndarray, G: np.ndarray, R_past: np.ndarray, *,
                   orders: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]
                   = (None, None, None),
                   base: str | Any = "critic", methods: Sequence[str] = RELATIVE_METHODS,
                   protocol: Protocol = "dependent", rng: np.random.Generator,
                   base_kw: Mapping[str, Any] | None = None, n_boot: int = N_BOOT,
                   groups: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]
                   = (None, None, None),
                   keep_extra: bool = False) -> list[dict[str, Any]]:
    """Relative tests ``H1: G is further from R_new than R_past is`` (DESIGN sections 1, 3.7).

    Parameters
    ----------
    R_new, G, R_past : arrays (n, L)
        Real TEST windows, synthetic windows over the TEST horizon, real TRAIN windows.
    orders : (order_new, order_g, order_past)
        Date ranks of the windows' first days (required by the dependent protocol).
    base : str or configured test, default "critic"
        Base test name (built by :func:`make_test` with the protocol settings and ``base_kw``),
        or a configured instance (used as is). Kernel bases (``"mmd"``, ``"sigmmd"``) have no
        anchored version: they run once (Bounliphone et al. 2016) whatever ``methods`` says,
        with ``method`` set to ``extra["relative_method"]`` (or ``"mmd"``).
    methods : sequence of {"anchored", "two_critic"}, default ``RELATIVE_METHODS``
        ``"anchored"`` is the primary test of the verdict; ``"two_critic"`` is secondary
        (anti-conservative at ``Law(G) = Law(R_past)``, theory.md Prop. 6.8).
    protocol : {"dependent", "iid"}
        ``"dependent"``: base with ``time_blocks=TIME_BLOCKS`` and ``inference="hac"``, and the
        HAC p-value. ``"iid"``: anchored permutation / two-critic paired-bootstrap p-values.
    rng : numpy.random.Generator
        One integer is drawn; each method uses a generator derived from it and its name.
    base_kw : dict, optional
        Overrides of the base test's constructor settings (ignored for an instance).
    n_boot : int, default ``N_BOOT``
        Bootstrap replicates of the two-critic test.
    groups : (groups_new, groups_g, groups_past), optional
        Ticker labels; used only under ``protocol="iid"`` by a base without time blocks.
    keep_extra : bool, default False
        Add the result's ``extra`` dict to each row.

    Returns
    -------
    list of dict
        One row per method: ``test`` (``"relative_<method>"``), ``method``, ``base``,
        ``protocol``, ``statistic`` (``Delta``), ``pvalue``, ``pvalue_type``, ``valid``
        (``False`` for two-critic, whose size exceeds alpha at the boundary), ``p_perm``,
        ``p_perm_stud`` (anchored), ``p_hac``, ``p_boot`` (two-critic), ``z``, ``z_hac``,
        ``se_hac``, ``ci_lo``, ``ci_hi``, ``D_g``, ``D_past`` (diagnostic discrepancies;
        ``Delta = D_g - D_past``), ``n_new``, ``n_g``, ``n_past``, ``runtime_s``.

    Raises
    ------
    NotImplementedError
        If ``"anchored"`` is requested and the installed ``relative_test`` has no ``method``.
    """
    _check_protocol(protocol)
    order_new, order_g, order_past = orders
    if protocol == "dependent" and any(o is None for o in orders):
        raise ValueError("protocol='dependent' needs orders=(order_new, order_g, order_past)")
    for m in methods:
        if m not in RELATIVE_METHODS:
            raise ValueError(f"unknown relative method {m!r}; options: {RELATIVE_METHODS}")
    base_name = base if isinstance(base, str) else getattr(base, "name", type(base).__name__)
    kernel_base = base_name in KERNEL_TESTS
    run_methods = list(methods)[:1] if kernel_base else list(methods)
    has_method = "method" in inspect.signature(relative_test).parameters
    base_seed = _draw_seed(rng)
    inference = "hac" if protocol == "dependent" else "default"
    rows: list[dict[str, Any]] = []
    for method in run_methods:
        t0 = time.perf_counter()
        b = make_test(base, protocol, **dict(base_kw or {})) if isinstance(base, str) else base
        kw: dict[str, Any] = dict(n_boot=n_boot, order_new=order_new, order_g=order_g,
                                  order_past=order_past, inference=inference)
        if protocol == "iid" and getattr(b, "time_blocks", None) is None:
            kw.update(groups_new=groups[0], groups_g=groups[1], groups_past=groups[2])
        if has_method:
            kw["method"] = method
        elif method == "anchored" and not kernel_base:
            raise NotImplementedError("relative_test(method='anchored') is not available in "
                                      "the installed synthval.twosample.relative")
        res = relative_test(R_new, G, R_past, b, rng=_sub_rng(base_seed, f"relative:{method}"),
                            **kw)
        ex = res.extra
        label = (str(ex.get("relative_method", "mmd")) if kernel_base else method)
        if inference == "hac":
            ptype = "hac"
        elif kernel_base:
            ptype = "bootstrap" if ex.get("method") == "bootstrap" else "normal"
        elif method == "anchored":
            ptype = str(ex.get("pvalue_type", "permutation"))
        else:
            ptype = "bootstrap"
        ci = ex.get("ci", (_NAN, _NAN))
        row: dict[str, Any] = {
            "test": f"relative_{label}", "method": label, "base": base_name,
            "protocol": protocol, "statistic": float(res.statistic), "pvalue": float(res.pvalue),
            "pvalue_type": ptype, "valid": label != "two_critic",
            "p_perm": _f(ex.get("p_perm")), "p_perm_stud": _f(ex.get("p_perm_stud")),
            "p_hac": _f(ex.get("p_hac")), "p_boot": _f(ex.get("p_boot")),
            "z": _f(ex.get("z")), "z_hac": _f(ex.get("z_hac")), "se_hac": _f(ex.get("se_hac")),
            "ci_lo": _f(ci[0]), "ci_hi": _f(ci[1]),
            "D_g": _f(ex.get("D_g")), "D_past": _f(ex.get("D_past")),
            "n_new": int(np.shape(R_new)[0]), "n_g": int(np.shape(G)[0]),
            "n_past": int(np.shape(R_past)[0]), "runtime_s": time.perf_counter() - t0,
        }
        if keep_extra:
            row["extra"] = ex
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------
# One model, one seed: simulate, window, test (E3 / E5 inner loop)
# ----------------------------------------------------------------------------------------------
def evaluate_model(panel: RealPanel, fitted: Mapping[str, ReturnModel] | MultiReturnModel, *,
                   seed: int, tests: Sequence[str] = ABSOLUTE_TESTS,
                   relative_base: str | None = "critic",
                   relative_methods: Sequence[str] = RELATIVE_METHODS,
                   protocol: Protocol = "dependent",
                   test_kw: Mapping[str, Mapping[str, Any]] | None = None,
                   base_kw: Mapping[str, Any] | None = None, n_boot: int = N_BOOT,
                   start_from_end: bool = True, all_pvalues: bool = True,
                   keep_extra: bool = False) -> list[dict[str, Any]]:
    """Simulate the TEST horizon from ``fitted`` and run the absolute and relative tests.

    Synthetic returns (one path per ticker, ``simulate_panel(fitted, panel.n_test,
    seed=child 0 of SeedSequence(seed))``) are normalised with the real TRAIN std and get the
    TEST date ranks; ``R_new`` = real TEST windows, ``R_past`` = real TRAIN windows. The tests
    use ``default_rng(child 1)``. Each row of :func:`absolute_tests` / :func:`relative_tests`
    gets ``kind`` (``"absolute"`` / ``"relative"``) and ``seed``. ``relative_base=None`` skips
    the relative tests; ``tests=()`` skips the absolute ones.
    """
    ss_sim, ss_test = np.random.SeedSequence(int(seed)).spawn(2)
    sim = simulate_panel(fitted, panel.n_test, seed=ss_sim, start_from_end=start_from_end)
    Xn, _, on = panel.windows("test")
    Y, _, oy = panel.synthetic_windows(sim, "test")
    rng = np.random.default_rng(ss_test)
    rows: list[dict[str, Any]] = []
    if tests:
        for r in absolute_tests(Xn, Y, order_x=on, order_y=oy, tests=tests, protocol=protocol,
                                rng=rng, test_kw=test_kw, all_pvalues=all_pvalues,
                                keep_extra=keep_extra):
            rows.append({"kind": "absolute", "seed": int(seed), **r})
    if relative_base is not None and relative_methods:
        Xp, _, op = panel.windows("train")
        for r in relative_tests(Xn, Y, Xp, orders=(on, oy, op), base=relative_base,
                                methods=relative_methods, protocol=protocol, rng=rng,
                                base_kw=base_kw, n_boot=n_boot, keep_extra=keep_extra):
            rows.append({"kind": "relative", "seed": int(seed), **r})
    return rows


# ----------------------------------------------------------------------------------------------
# Binomial intervals and p-value aggregation
# ----------------------------------------------------------------------------------------------
def wilson_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson (1927) score interval for a binomial proportion ``k / n``."""
    if n <= 0:
        return _NAN, _NAN
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def clopper_pearson_lower(k: int, n: int, level: float = 0.95) -> float:
    """One-sided Clopper-Pearson (1934) lower confidence bound for ``k / n``."""
    if n <= 0:
        return _NAN
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    if k == 0:
        return 0.0
    return float(stats.beta.ppf(1.0 - level, k, n - k + 1))


def median_pvalue(pvalues: Sequence[float] | np.ndarray) -> float:
    """Valid combination of p-values over seeds: ``min(1, 2 * median)`` (theory.md Remark 8.5)."""
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return _NAN
    return float(min(1.0, 2.0 * np.median(p)))


def verdict(p_relative: float, power: float | None, alpha: float = ALPHA,
            power_min: float = POWER_MIN) -> str:
    """Decision rule of DESIGN section 1, item 4.

    ``"BAD"`` if the anchored relative test rejects (``p_relative <= alpha``); ``"GOOD"`` if it
    does not reject and ``power >= power_min``; ``"INCONCLUSIVE"`` otherwise (including a missing
    or NaN power). Pass ``PowerCheckResult.power`` for the DESIGN rule, or
    ``PowerCheckResult.cp_lower`` for the stricter Clopper-Pearson criterion of theory.md 11.7;
    pass :func:`median_pvalue` over seeds as ``p_relative`` when several seeds were run.
    """
    p = float(p_relative)
    if not np.isfinite(p):
        raise ValueError("p_relative must be finite")
    if p <= alpha:
        return "BAD"
    if power is not None and np.isfinite(power) and float(power) >= power_min:
        return "GOOD"
    return "INCONCLUSIVE"


# ----------------------------------------------------------------------------------------------
# Power check (DESIGN section 1, item 3)
# ----------------------------------------------------------------------------------------------
@dataclass
class PowerCheckResult:
    """Outcome of :func:`power_check`.

    Attributes
    ----------
    power : float
        Rejection rate ``n_reject / n_reps`` of the relative test at level ``alpha`` against the
        pre-specified deviation (synthetic volatility x ``vol_factor``).
    n_reject, n_reps : int
    ci : (float, float)
        Wilson 95% interval of ``power``.
    cp_lower : float
        One-sided 95% Clopper-Pearson lower bound of ``power``.
    alpha, vol_factor : float
    method, base, protocol : str
    reps : DataFrame
        One row per replication: ``rep``, ``pvalue``, ``statistic``, ``p_perm``, ``p_hac``,
        ``z_hac``, ``runtime_s`` (and ``boundary_*`` columns when the boundary size was run).
    boundary_size, boundary_ci, boundary_n_reject :
        With ``boundary=True``: rejection rate when ``G*`` is an independent draw from the
        reference law (``vol_factor = 1``), i.e. the size at ``Law(G) = Law(R_past)``
        (theory.md Definition 8.1), with its Wilson interval; NaN otherwise.
    seconds : float
        Wall time (fits excluded when ``fitted`` was given).
    settings : dict
        Configuration (sizes, model, seeds) for the experiment's metadata.
    """

    power: float
    n_reject: int
    n_reps: int
    ci: tuple[float, float]
    cp_lower: float
    alpha: float
    vol_factor: float
    method: str
    base: str
    protocol: str
    reps: pd.DataFrame
    boundary_size: float = _NAN
    boundary_ci: tuple[float, float] = (_NAN, _NAN)
    boundary_n_reject: int = 0
    seconds: float = _NAN
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def pvalues(self) -> np.ndarray:
        """Per-replication p-values of the relative test against the deviation."""
        return self.reps["pvalue"].to_numpy(dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        """Scalar summary (JSON-friendly)."""
        return {"power": self.power, "n_reject": self.n_reject, "n_reps": self.n_reps,
                "ci_lo": self.ci[0], "ci_hi": self.ci[1], "cp_lower": self.cp_lower,
                "alpha": self.alpha, "vol_factor": self.vol_factor, "method": self.method,
                "base": self.base, "protocol": self.protocol,
                "boundary_size": self.boundary_size, "boundary_ci_lo": self.boundary_ci[0],
                "boundary_ci_hi": self.boundary_ci[1],
                "boundary_n_reject": self.boundary_n_reject, "seconds": self.seconds}


def _power_rep(item: tuple[int, np.random.SeedSequence, dict[str, Any]]) -> dict[str, Any]:
    """Worker: one power-check replication."""
    rep, ss, cfg = item
    s_past, s_new, s_g, s_test, s_bd = ss.spawn(5)
    t0 = time.perf_counter()
    fitted, scale = cfg["fitted"], cfg["scale"]
    n_train, n_test, L, step = cfg["n_train"], cfg["n_test"], cfg["L"], cfg["step"]
    sfe, burn = cfg["start_from_end"], cfg["burn_in"]
    Rp = simulate_panel(fitted, n_train, seed=s_past, start_from_end=False, burn_in=burn)
    Rn = simulate_panel(fitted, n_test, seed=s_new, start_from_end=sfe, burn_in=burn)
    Gs = cfg["vol_factor"] * simulate_panel(fitted, n_test, seed=s_g, start_from_end=sfe,
                                            burn_in=burn)
    Wp, _, op = panel_windows(Rp, L, step, scale=scale, date_offset=0)
    Wn, _, on = panel_windows(Rn, L, step, scale=scale, date_offset=n_train)
    Wg, _, og = panel_windows(Gs, L, step, scale=scale, date_offset=n_train)
    rng = np.random.default_rng(s_test)
    common = dict(base=cfg["base"], methods=(cfg["method"],), protocol=cfg["protocol"],
                  base_kw=cfg["base_kw"], n_boot=cfg["n_boot"])
    r = relative_tests(Wn, Wg, Wp, orders=(on, og, op), rng=rng, **common)[0]
    out = {"rep": rep, "pvalue": r["pvalue"], "statistic": r["statistic"], "p_perm": r["p_perm"],
           "p_hac": r["p_hac"], "z_hac": r["z_hac"]}
    if cfg["boundary"]:
        G0 = simulate_panel(fitted, n_test, seed=s_bd, start_from_end=sfe, burn_in=burn)
        W0, _, o0 = panel_windows(G0, L, step, scale=scale, date_offset=n_train)
        r0 = relative_tests(Wn, W0, Wp, orders=(on, o0, op), rng=rng, **common)[0]
        out.update(boundary_pvalue=r0["pvalue"], boundary_statistic=r0["statistic"])
    out["runtime_s"] = time.perf_counter() - t0
    return out


def power_check(train_returns: pd.DataFrame, test_returns: pd.DataFrame, *,
                n_reps: int = POWER_N_REPS, seed: int, n_workers: int | None = None,
                vol_factor: float = POWER_VOL_FACTOR, base: str = "critic",
                protocol: Protocol = "dependent", method: str = "anchored",
                L: int = WINDOW_L, step: int | None = None, alpha: float = ALPHA,
                base_kw: Mapping[str, Any] | None = None,
                model_kw: Mapping[str, Any] | None = None,
                fitted: Mapping[str, ReturnModel] | None = None,
                scale: pd.Series | None = None, start_from_end: bool = True,
                boundary: bool = False, n_boot: int = N_BOOT,
                burn_in: int = BURN_IN) -> PowerCheckResult:
    """Pre-specified power of the relative test at the real sample sizes (DESIGN section 1.3).

    Reference design (theory.md Definition 8.1): per-ticker ``POWER_MODEL`` (GARCHt) fits on the
    real TRAIN returns. In each replication, independently: ``R_past*`` of TRAIN length (from
    stationarity, ``burn_in`` days discarded), ``R_new*`` of TEST length, and ``G*`` = another
    TEST-length path of the same model with returns multiplied by ``vol_factor`` (volatility
    x 1.2 by default). ``R_new*`` and ``G*`` start from the fitted end-of-TRAIN state when
    ``start_from_end`` (as the real TEST data and the E3 synthetic paths do), else from
    stationarity. All three are normalised with the real TRAIN std and windowed exactly as in
    E3 (TRAIN date ranks for ``R_past*``, TEST ranks for ``R_new*`` and ``G*``), and the
    relative test runs with the same protocol, base settings and split as E3.

    Parameters
    ----------
    train_returns, test_returns : DataFrame
        Real TRAIN and TEST returns of the panel (sizes and scales of the design).
    n_reps : int, default ``POWER_N_REPS`` (50; theory.md 11.7 recommends >= 200)
        Replications.
    seed : int
        Replication ``k`` uses child ``k`` of ``SeedSequence(seed)``, so results do not depend
        on ``n_workers``.
    n_workers : int, optional
        Processes for the fits and the replications (``utils.parallel_map``).
    vol_factor : float, default 1.2
        Deviation ``delta*`` of the generator. ``vol_factor=1`` estimates the boundary size.
    base, protocol, method, base_kw, n_boot :
        Relative test settings (as :func:`relative_tests`); ``method="anchored"`` is the verdict
        test.
    L, step : int
        Windows.
    alpha : float
        Level of the relative test.
    model_kw : dict, optional
        Options of the reference model (default: its constructor defaults, i.e. the E3 GARCHt).
    fitted : mapping ticker -> fitted model, optional
        Pre-fitted reference models (e.g. the E3 GARCHt fits) instead of fitting here.
    scale : Series, optional
        Per-ticker scale; default the real TRAIN std ``data.train_vol(train_returns)``.
    start_from_end : bool, default True
        Start ``R_new*`` and ``G*`` from the fitted end state (see above).
    boundary : bool, default False
        Also run, in each replication, the relative test with ``G*`` an unscaled independent
        draw (boundary size at ``Law(G) = Law(R_past)``); doubles the cost.
    burn_in : int
        Burn-in of stationary starts.

    Returns
    -------
    PowerCheckResult
    """
    if n_reps < 1:
        raise ValueError("n_reps must be >= 1")
    if vol_factor <= 0:
        raise ValueError("vol_factor must be positive")
    _check_protocol(protocol)
    t0 = time.perf_counter()
    tickers = list(train_returns.columns)
    if list(test_returns.columns) != tickers:
        raise ValueError("train_returns and test_returns must have the same columns")
    if fitted is None:
        fitted, _ = fit_panel(POWER_MODEL, train_returns, n_workers=n_workers,
                              **dict(model_kw or {}))
    else:
        missing = [t for t in tickers if t not in fitted]
        if missing:
            raise KeyError(f"no fitted model for tickers {missing}")
    fitted = {t: fitted[t] for t in tickers}
    sc = data.train_vol(train_returns) if scale is None else scale
    scale_arr = _scale_array(train_returns, sc, len(tickers))
    cfg = {"fitted": fitted, "scale": scale_arr, "n_train": train_returns.shape[0],
           "n_test": test_returns.shape[0], "L": L, "step": step, "vol_factor": float(vol_factor),
           "base": base, "base_kw": dict(base_kw or {}), "protocol": protocol, "method": method,
           "n_boot": n_boot, "boundary": bool(boundary), "start_from_end": bool(start_from_end),
           "burn_in": int(burn_in)}
    children = np.random.SeedSequence(int(seed)).spawn(int(n_reps))
    out = parallel_map(_power_rep, [(k, children[k], cfg) for k in range(n_reps)], n_workers)
    reps = pd.DataFrame(out).sort_values("rep").reset_index(drop=True)
    k = int((reps["pvalue"] <= alpha).sum())
    res = PowerCheckResult(
        power=k / n_reps, n_reject=k, n_reps=int(n_reps), ci=wilson_ci(k, n_reps),
        cp_lower=clopper_pearson_lower(k, n_reps), alpha=float(alpha),
        vol_factor=float(vol_factor), method=method,
        base=base if isinstance(base, str) else str(base), protocol=protocol, reps=reps,
        seconds=time.perf_counter() - t0,
        settings={"model": POWER_MODEL, "model_kw": dict(model_kw or {}), "seed": int(seed),
                  "n_tickers": len(tickers), "n_train": cfg["n_train"],
                  "n_test": cfg["n_test"], "L": L, "step": step, "base_kw": cfg["base_kw"],
                  "start_from_end": bool(start_from_end), "burn_in": int(burn_in)})
    if boundary:
        kb = int((reps["boundary_pvalue"] <= alpha).sum())
        res.boundary_n_reject = kb
        res.boundary_size = kb / n_reps
        res.boundary_ci = wilson_ci(kb, n_reps)
    return res
