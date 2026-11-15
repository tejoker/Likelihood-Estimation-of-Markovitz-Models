"""Stylized-fact statistics of daily log-returns (DESIGN section 3.9; Cont, 2001).

Every function takes a 1-D series of daily log-returns (finite ``float64``) and is
deterministic. Estimator conventions:

* ``std``: sample standard deviation (``ddof = 1``).
* ``skew`` / ``kurt``: bias-adjusted sample skewness G1 and *excess* kurtosis G2
  (``scipy.stats.skew/kurtosis`` with ``bias=False``), so a Gaussian sample has ``kurt ~ 0``.
* ACF: the standard biased estimator ``rho_k = sum_t (x_t - xbar)(x_{t+k} - xbar) /
  sum_t (x_t - xbar)^2`` (as in ``statsmodels.tsa.stattools.acf``).
* Leverage: Pearson correlation ``L(k) = corr(r_t, r_{t+k}^2)`` over the ``T - k`` available
  pairs, ``k = 1..10``; negative values mean that falls raise future volatility.
* Hill tail index on the ``5%`` largest ``|r|`` (uncentred; ``k = floor(0.05 T)`` order
  statistics above the threshold ``|r|_{(k+1)}``). ``hill_left`` / ``hill_right`` apply the same
  estimator to ``-r`` and ``r`` with the same ``k``.
* Aggregational Gaussianity: excess kurtosis of non-overlapping ``h``-day sums
  (``h = 1, 5, 20``; an incomplete tail block is dropped).
* Maximum drawdown of the cumulative log-price ``p_t = sum_{s<=t} r_s`` (with ``p_0 = 0``):
  ``max_t (max_{s<=t} p_s - p_t)``, in log units (``1 - exp(-mdd)`` is the loss in price).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "ACF_RET_LAGS",
    "ACF_VOL_LAGS",
    "LEVERAGE_LAGS",
    "AGG_HORIZONS",
    "HILL_FRACTION",
    "acf",
    "leverage",
    "hill",
    "aggregated_kurtosis",
    "max_drawdown",
    "summary",
    "panel_summary",
]

ACF_RET_LAGS: int = 5
ACF_VOL_LAGS: tuple[int, ...] = (1, 5, 10, 20, 50)
LEVERAGE_LAGS: int = 10
AGG_HORIZONS: tuple[int, ...] = (1, 5, 20)
HILL_FRACTION: float = 0.05


def _as_series(r: np.ndarray | pd.Series) -> np.ndarray:
    x = np.asarray(r, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or inf")
    return x


def acf(x: np.ndarray | pd.Series, nlags: int) -> np.ndarray:
    """Sample autocorrelation function at lags ``0..nlags``.

    Parameters
    ----------
    x : array_like, shape (T,)
        Finite series.
    nlags : int
        Largest lag.

    Returns
    -------
    ndarray, shape (nlags + 1,)
        ``rho_0 = 1, rho_1, ..., rho_nlags`` (biased estimator, FFT-based, identical to
        ``statsmodels.tsa.stattools.acf(x, nlags=nlags, fft=True)``). Lags ``>= T`` and every lag
        of a constant series are NaN.
    """
    if nlags < 0:
        raise ValueError("nlags must be >= 0")
    x = _as_series(x)
    T = x.size
    out = np.full(nlags + 1, np.nan)
    if T == 0:
        return out
    xc = x - x.mean()
    n_fft = 1 << int(np.ceil(np.log2(max(2 * T - 1, 1))))
    f = np.fft.rfft(xc, n_fft)
    acov = np.fft.irfft(f * np.conj(f), n_fft)[: min(nlags + 1, T)] / T
    if acov[0] <= 0:
        return out
    out[: acov.size] = acov / acov[0]
    out[0] = 1.0
    return out


def leverage(r: np.ndarray | pd.Series, max_lag: int = LEVERAGE_LAGS) -> np.ndarray:
    """Leverage correlations ``corr(r_t, r_{t+k}^2)`` for ``k = 1..max_lag``.

    Returns
    -------
    ndarray, shape (max_lag,)
        NaN where fewer than 3 pairs are available or a side is constant.
    """
    x = _as_series(r)
    out = np.full(max_lag, np.nan)
    for k in range(1, max_lag + 1):
        if x.size - k < 3:
            break
        a, b = x[:-k], x[k:] ** 2
        if a.std() > 0 and b.std() > 0:
            out[k - 1] = float(np.corrcoef(a, b)[0, 1])
    return out


def hill(x: np.ndarray | pd.Series, fraction: float = HILL_FRACTION) -> float:
    """Hill (1975) estimator of the tail index ``alpha`` of the right tail of ``x``.

    With ``x_(1) >= x_(2) >= ...`` the order statistics and ``k = floor(fraction * T)``,
    ``alpha_hat = 1 / ( mean_{i<=k} log x_(i) - log x_(k+1) )``. Pass ``|r|`` for the two-sided
    tail, ``-r`` for the left tail.

    Returns
    -------
    float
        Estimated tail index; NaN if ``k < 2`` or the threshold ``x_(k+1)`` is not positive.
    """
    v = _as_series(x)
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    k = int(np.floor(fraction * v.size))
    if k < 2:
        return float("nan")
    top = -np.partition(-v, k)[: k + 1]  # the k+1 largest values, unordered except index k
    top = np.sort(top)[::-1]
    u = top[k]
    if u <= 0:
        return float("nan")
    h = float(np.mean(np.log(top[:k])) - np.log(u))
    return 1.0 / h if h > 0 else float("nan")


def aggregated_kurtosis(r: np.ndarray | pd.Series,
                        horizons: tuple[int, ...] = AGG_HORIZONS) -> dict[int, float]:
    """Excess kurtosis of non-overlapping ``h``-day sums of returns, for each horizon ``h``.

    Returns
    -------
    dict
        ``{h: kurtosis}``; NaN when fewer than 4 complete blocks are available.
    """
    x = _as_series(r)
    out: dict[int, float] = {}
    for h in horizons:
        n = x.size // h
        if n < 4:
            out[h] = float("nan")
            continue
        agg = x[: n * h].reshape(n, h).sum(axis=1)
        out[h] = float(stats.kurtosis(agg, fisher=True, bias=False)) if agg.std() > 0 else float("nan")
    return out


def max_drawdown(r: np.ndarray | pd.Series) -> float:
    """Maximum drawdown of the cumulative log-price, in log units (non-negative)."""
    x = _as_series(r)
    p = np.concatenate(([0.0], np.cumsum(x)))
    return float(np.max(np.maximum.accumulate(p) - p))


def summary(r: np.ndarray | pd.Series) -> dict[str, float]:
    """Stylized facts of one series of daily log-returns.

    Parameters
    ----------
    r : array_like, shape (T,)
        Finite daily log-returns in time order.

    Returns
    -------
    dict
        Keys (all floats): ``n``, ``mean``, ``std``, ``skew``, ``kurt`` (excess);
        ``acf_r_1..acf_r_5``; ``acf_abs_k`` and ``acf_sq_k`` for ``k in (1, 5, 10, 20, 50)``;
        ``lev_1..lev_10`` and their average ``lev_mean``; ``hill``, ``hill_left``,
        ``hill_right``; ``kurt_agg_1``, ``kurt_agg_5``, ``kurt_agg_20``; ``max_drawdown``.
        Statistics that need more observations than available are NaN.
    """
    x = _as_series(r)
    T = x.size
    out: dict[str, float] = {"n": float(T)}
    out["mean"] = float(x.mean()) if T else float("nan")
    out["std"] = float(x.std(ddof=1)) if T > 1 else float("nan")
    nondeg = T > 3 and x.std() > 0
    out["skew"] = float(stats.skew(x, bias=False)) if nondeg else float("nan")
    out["kurt"] = float(stats.kurtosis(x, fisher=True, bias=False)) if nondeg else float("nan")

    rho = acf(x, ACF_RET_LAGS)
    for k in range(1, ACF_RET_LAGS + 1):
        out[f"acf_r_{k}"] = float(rho[k])
    m = max(ACF_VOL_LAGS)
    rho_abs, rho_sq = acf(np.abs(x), m), acf(x**2, m)
    for k in ACF_VOL_LAGS:
        out[f"acf_abs_{k}"] = float(rho_abs[k])
    for k in ACF_VOL_LAGS:
        out[f"acf_sq_{k}"] = float(rho_sq[k])

    lev = leverage(x, LEVERAGE_LAGS)
    for k in range(1, LEVERAGE_LAGS + 1):
        out[f"lev_{k}"] = float(lev[k - 1])
    out["lev_mean"] = float(np.mean(lev)) if np.all(np.isfinite(lev)) else float("nan")

    out["hill"] = hill(np.abs(x))
    out["hill_left"] = hill(-x)
    out["hill_right"] = hill(x)

    for h, v in aggregated_kurtosis(x, AGG_HORIZONS).items():
        out[f"kurt_agg_{h}"] = v
    out["max_drawdown"] = max_drawdown(x) if T else float("nan")
    return out


def _strip_edge_nan(s: pd.Series) -> np.ndarray:
    """Drop leading/trailing NaN (late listing); internal NaN would break the time structure."""
    v = s.to_numpy(dtype=np.float64)
    ok = np.flatnonzero(np.isfinite(v))
    if ok.size == 0:
        return v[:0]
    v = v[ok[0]: ok[-1] + 1]
    if not np.all(np.isfinite(v)):
        raise ValueError(f"column {s.name!r} has internal missing values")
    return v


def panel_summary(returns: pd.DataFrame | np.ndarray, *, per_ticker: bool = True) -> pd.DataFrame:
    """Per-ticker stylized facts with their cross-sectional median and inter-quartile range.

    Parameters
    ----------
    returns : DataFrame or ndarray, shape (T, N)
        Daily log-returns, one column per ticker (array columns are labelled ``0..N-1``).
        Leading/trailing NaN of a column are dropped; internal NaN raise ``ValueError``.
    per_ticker : bool
        If False, return only the four cross-sectional rows.

    Returns
    -------
    pd.DataFrame
        One column per statistic of :func:`summary`. Rows: one per ticker (if ``per_ticker``),
        followed by ``"median"``, ``"q25"``, ``"q75"`` and ``"iqr"`` (``q75 - q25``), computed
        across tickers ignoring NaN. ``df.attrs["tickers"]`` lists the ticker rows.
    """
    df = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(np.asarray(returns))
    if df.ndim != 2 or df.shape[1] == 0:
        raise ValueError("returns must be a non-empty (T, N) panel")
    per = pd.DataFrame.from_dict(
        {c: summary(_strip_edge_nan(df[c])) for c in df.columns}, orient="index")
    q25, q50, q75 = (per.quantile(q, numeric_only=True) for q in (0.25, 0.5, 0.75))
    agg = pd.DataFrame({"median": q50, "q25": q25, "q75": q75, "iqr": q75 - q25}).T
    out = pd.concat([per, agg]) if per_ticker else agg
    out.index.name = "ticker"
    out.attrs["tickers"] = list(per.index) if per_ticker else []
    return out
