"""Loading, cleaning, returns, sample splits, windows and normalisation (DESIGN section 2, 3.3).

The raw file is a Bloomberg export (``;``-separated, UTF-8, CRLF):

* line 1: ``Date de début;2018-01-01;;;...`` (requested start date),
* line 2: ``Date de fin;2025-06-17;;;...`` (requested end date),
* line 3: ``Date;MMM;MMM;AOS;AOS;...`` (ticker of each column),
* line 4: ``;PX_LAST;PX_VOLUME;PX_LAST;...`` (Bloomberg field of each column),
* lines 5+: ``YYYY-MM-DD;<value>;<value>;...``, an empty cell meaning "no price".

Only ``PX_LAST`` is used. Missing prices are **never back-filled**: a ticker with a late listing
keeps NaN before its first price and is excluded from the complete universe. Only internal gaps
of at most ``max_gap`` consecutive trading days are forward-filled (none occur in the current file).

Two tickers with a full price history are data errors and are excluded by default
(:data:`DATA_ANOMALIES`, found by the objective screen of :func:`quality_report`):

* ``KDP``: ``PX_LAST`` is not adjusted for the special dividend paid when the Keurig-Dr Pepper
  merger closed (123.66 on 2018-07-09, 22.19 on 2018-07-10 on a volume of 23.6M shares), which
  leaves one spurious log-return of -1.718 and triples the TRAIN standard deviation.
* ``AMCR``: before its NYSE listing on 2019-06-11 the series is a stale quote (361 days, only 71
  distinct prices, zero volume on 232 days); 298 of its 1258 TRAIN returns are exactly zero
  (268 of them before the listing), with runs of up to 18 days.

All returns are daily log-returns ``r_t = log P_t - log P_{t-1}`` (real-world measure, dt = 1
trading day).
"""
from __future__ import annotations

import csv
import functools
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_ANOMALIES",
    "ANOMALY_NOTES",
    "ANOMALY_MAX_ABS_RET",
    "ANOMALY_MIN_ZERO_VOLUME",
    "file_metadata",
    "load_prices",
    "load_volume",
    "quality_report",
    "complete_universe",
    "clean_prices",
    "log_returns",
    "split",
    "panel_tickers",
    "make_windows",
    "window_starts",
    "panel_windows",
    "load_panel",
    "train_vol",
    "normalise",
]

_FIELD = "PX_LAST"
_KNOWN_FIELDS = {"PX_LAST", "PX_VOLUME"}

#: Tickers of ``config.DATA_CSV`` with a complete price history that are data errors. They are
#: removed by default from :func:`complete_universe`, hence from :func:`panel_tickers`,
#: :func:`load_panel` and every universe drawn from them. Pass ``exclude=()`` to keep them (e.g.
#: for a sensitivity check). The list is exactly what the screen of :func:`quality_report` flags
#: (``anomaly`` column; a test checks the equality), not a hand-picked set.
DATA_ANOMALIES: tuple[str, ...] = ("AMCR", "KDP")

#: One-line reason per excluded ticker (for the data section of the paper).
ANOMALY_NOTES: dict[str, str] = {
    "AMCR": "stale quotes before the NYSE listing of 2019-06-11 (232 zero-volume days, "
            "298 zero TRAIN returns, longest zero run 18 days)",
    "KDP": "PX_LAST not adjusted for the special dividend of the Keurig-Dr Pepper merger "
           "(log-return -1.718 on 2018-07-10)",
}

#: Screen threshold 1: a daily ``|log-return|`` above 1 (price ratio beyond e ~ 2.72 in one
#: day) flags an unadjusted corporate action. The largest genuine moves in the file are about
#: 0.77 (oil producers on 2020-03-09, PCG on 2019-01-14, GL on 2024-04-11).
ANOMALY_MAX_ABS_RET: float = 1.0

#: Screen threshold 2: at least this many days with a price but zero volume flags stale quotes
#: (every ticker other than AMCR has at most one such day).
ANOMALY_MIN_ZERO_VOLUME: int = 20


# --------------------------------------------------------------------------------------------
# Raw file parsing
# --------------------------------------------------------------------------------------------
def _read_header_lines(path: Path, n: int = 4) -> list[list[str]]:
    """Return the first ``n`` lines of the file split on ``;`` (trailing CR stripped)."""
    rows: list[list[str]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        for _ in range(n):
            rows.append(next(reader))
    return rows


def file_metadata(path: str | Path = config.DATA_CSV) -> dict[str, str]:
    """Requested start/end dates written by Bloomberg in the two metadata lines.

    Parameters
    ----------
    path : str or Path
        Bloomberg CSV export.

    Returns
    -------
    dict
        ``{"start": "2018-01-01", "end": "2025-06-17"}``. These are the *requested* dates;
        the first and last trading days in the file may differ (2018-01-02 and 2025-06-16).
    """
    rows = _read_header_lines(Path(path), n=2)
    if not (rows[0][0].startswith("Date de d") and rows[1][0].startswith("Date de fin")):
        raise ValueError(f"{path}: unexpected metadata lines {rows[0][:2]!r}, {rows[1][:2]!r}")
    return {"start": rows[0][1].strip(), "end": rows[1][1].strip()}


@functools.lru_cache(maxsize=4)
def _parse_bloomberg(path: str, mtime_ns: int, size: int, field: str = _FIELD) -> pd.DataFrame:
    """Parse one field block of a Bloomberg export (cached; key includes mtime and size)."""
    del mtime_ns, size  # only part of the cache key, so an edited file is re-read
    p = Path(path)
    file_metadata(p)  # validates lines 1-2
    _, _, tick_row, field_row = _read_header_lines(p, n=4)
    if tick_row[0].strip() != "Date" or field_row[0].strip() != "":
        raise ValueError(f"{p}: lines 3-4 are not the (ticker, field) header")
    tick_row = [t.strip() for t in tick_row[1:]]
    field_row = [f.strip() for f in field_row[1:]]
    if len(tick_row) != len(field_row):
        raise ValueError(f"{p}: ticker and field header rows have different lengths")
    unknown = set(field_row) - _KNOWN_FIELDS
    if unknown:
        raise ValueError(f"{p}: unexpected Bloomberg fields {sorted(unknown)}")
    if field not in _KNOWN_FIELDS:
        raise ValueError(f"unknown field {field!r}")
    px_pos = [i for i, f in enumerate(field_row) if f == field]
    tickers = [tick_row[i] for i in px_pos]
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"{p}: duplicated {field} column for some ticker")

    # Column 0 is the date; the field of ticker j sits at raw column px_pos[j] + 1.
    raw = pd.read_csv(
        p,
        sep=";",
        skiprows=4,
        header=None,
        usecols=[0] + [i + 1 for i in px_pos],
        encoding="utf-8",
        dtype={0: str},
        na_values=[""],
        keep_default_na=False,
    )
    dates = pd.to_datetime(raw.iloc[:, 0].str.strip(), format="%Y-%m-%d")
    values = raw.iloc[:, 1:].to_numpy(dtype=np.float64)
    prices = pd.DataFrame(values, index=pd.DatetimeIndex(dates, name="Date"),
                          columns=pd.Index(tickers, name="ticker"))
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError(f"{p}: dates are not unique and increasing")
    if field == _FIELD and (prices <= 0).to_numpy().any():
        raise ValueError(f"{p}: non-positive {field} values")
    logger.debug("parsed %s: %d dates x %d tickers", p, *prices.shape)
    return prices


def load_prices(path: str | Path = config.DATA_CSV) -> pd.DataFrame:
    """Raw ``PX_LAST`` prices, Date index x ticker, NaN kept (no filling of any kind).

    The parsed file is cached in-process (``functools.lru_cache`` keyed by path, mtime and size),
    so repeated calls cost one DataFrame copy.

    Parameters
    ----------
    path : str or Path
        Bloomberg CSV export (default ``config.DATA_CSV``).

    Returns
    -------
    pd.DataFrame
        ``float64`` prices with a ``DatetimeIndex`` named ``"Date"`` and columns named
        ``"ticker"`` in file order. A fresh copy: mutating it does not touch the cache.
    """
    p = Path(path).resolve()
    st = os.stat(p)
    return _parse_bloomberg(str(p), st.st_mtime_ns, st.st_size, _FIELD).copy()


def load_volume(path: str | Path = config.DATA_CSV) -> pd.DataFrame:
    """Raw ``PX_VOLUME`` (shares traded), same layout as :func:`load_prices`.

    Used only for data-quality diagnostics (:func:`quality_report`), never for modelling.
    """
    p = Path(path).resolve()
    st = os.stat(p)
    return _parse_bloomberg(str(p), st.st_mtime_ns, st.st_size, "PX_VOLUME").copy()


def _longest_true_run(mask: np.ndarray) -> int:
    starts, stops = _nan_runs(mask)
    return int((stops - starts).max()) if starts.size else 0


def quality_report(path: str | Path = config.DATA_CSV, max_gap: int = 3) -> pd.DataFrame:
    """Per-ticker data-quality diagnostics and anomaly screen (for the data section of the paper).

    Parameters
    ----------
    path : str or Path
        Bloomberg CSV export.
    max_gap : int
        Gap rule of :func:`complete_universe`.

    Returns
    -------
    pd.DataFrame
        One row per ticker (index ``"ticker"``, file order) with columns

        * ``first_date``: first available price; ``n_missing``: missing ``PX_LAST`` values;
        * ``complete_history``: passes the availability rule of :func:`complete_universe`
          over ``config.FULL`` (no data-anomaly exclusion);
        * ``max_abs_ret``, ``date_max_abs_ret``: largest absolute daily log-return and its date;
        * ``n_zero_volume``: days with a price but zero volume (stale quotes);
        * ``n_zero_ret``, ``longest_zero_ret_run``: number and longest run of exactly zero
          returns;
        * ``anomaly``: the screen ``max_abs_ret > ANOMALY_MAX_ABS_RET`` or
          ``n_zero_volume >= ANOMALY_MIN_ZERO_VOLUME``. On the current file it flags exactly
          :data:`DATA_ANOMALIES` among complete-history tickers;
        * ``in_universe``: member of the default complete universe
          (``complete_history`` and not in :data:`DATA_ANOMALIES`).

    Notes
    -----
    Not flagged, but worth stating: ``SMCI`` has 117 exactly-zero TRAIN returns (9%, against a
    median of 4) because its split-adjusted prices (about 1.2 to 2.2 in TRAIN after the 10:1 split
    of 2024) are rounded to two decimals. It is a precision artefact rather than an error, and
    SMCI is not in the default panel.
    """
    prices = load_prices(path)
    volume = load_volume(path).reindex_like(prices)
    rets = log_returns(prices)
    complete = set(complete_universe(prices, *config.FULL, max_gap=max_gap, exclude=()))
    rows = {}
    for t in prices.columns:
        r = rets[t]
        has_px = prices[t].notna().to_numpy()
        absr = r.abs()
        zero = (r == 0).to_numpy()
        rows[t] = {
            "first_date": prices[t].first_valid_index(),
            "n_missing": int((~has_px).sum()),
            "complete_history": t in complete,
            "max_abs_ret": float(absr.max()),
            "date_max_abs_ret": absr.idxmax() if absr.notna().any() else pd.NaT,
            "n_zero_volume": int(((volume[t].to_numpy() == 0) & has_px).sum()),
            "n_zero_ret": int(zero.sum()),
            "longest_zero_ret_run": _longest_true_run(zero),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "ticker"
    out["anomaly"] = ((out["max_abs_ret"] > ANOMALY_MAX_ABS_RET)
                      | (out["n_zero_volume"] >= ANOMALY_MIN_ZERO_VOLUME))
    out["in_universe"] = out["complete_history"] & ~out.index.isin(DATA_ANOMALIES)
    return out


# --------------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------------
def _nan_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and stop (exclusive) indices of runs of True in a 1-D boolean array."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _admissible(col: np.ndarray, max_gap: int) -> bool:
    """True if ``col`` has a value at both ends and every NaN run is internal and <= max_gap."""
    if col.size == 0:
        return False
    starts, stops = _nan_runs(np.isnan(col))
    if starts.size == 0:
        return True
    if starts[0] == 0 or stops[-1] == col.size:  # leading or trailing gap
        return False
    return bool(np.all(stops - starts <= max_gap))


def complete_universe(prices: pd.DataFrame, start: str = config.FULL[0],
                      end: str = config.FULL[1], max_gap: int = 3, *,
                      exclude: Sequence[str] = DATA_ANOMALIES) -> list[str]:
    """Tickers with a usable ``PX_LAST`` history on ``[start, end]`` (inclusive).

    A ticker qualifies if it has a price on the first and on the last trading day of the period
    (so a late listing or an early delisting excludes it: nothing is back-filled or
    extrapolated), every internal run of missing prices has length ``<= max_gap``, and it is
    not in ``exclude`` (by default the data errors :data:`DATA_ANOMALIES`, so the default
    universe of the current file has 479 tickers: 481 with a complete history minus KDP and
    AMCR).

    Parameters
    ----------
    prices : pd.DataFrame
        Raw prices from :func:`load_prices`.
    start, end : str
        Inclusive period bounds (default: the whole file, ``config.FULL``).
    max_gap : int
        Longest admissible run of consecutive missing days (forward-filled by
        :func:`clean_prices`).
    exclude : sequence of str
        Tickers removed whatever their history (default :data:`DATA_ANOMALIES`; ``()`` applies
        the availability rule alone).

    Returns
    -------
    list of str
        Qualifying tickers, sorted alphabetically.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be >= 0")
    window = prices.loc[start:end]
    if window.empty:
        raise ValueError(f"no prices between {start} and {end}")
    drop = set(exclude)
    arr = window.to_numpy(dtype=np.float64)
    keep = [t for j, t in enumerate(window.columns)
            if t not in drop and _admissible(arr[:, j], max_gap)]
    return sorted(keep)


def clean_prices(prices: pd.DataFrame, tickers: Sequence[str], max_gap: int = 3) -> pd.DataFrame:
    """Select ``tickers`` and forward-fill only internal gaps of at most ``max_gap`` days.

    Leading NaN (before the first price), trailing NaN and internal gaps longer than ``max_gap``
    are left as NaN (never partially filled, never back-filled).

    Parameters
    ----------
    prices : pd.DataFrame
        Raw prices from :func:`load_prices`.
    tickers : sequence of str
        Columns to keep, in this order.
    max_gap : int
        Longest internal gap that is forward-filled.

    Returns
    -------
    pd.DataFrame
        Cleaned copy with columns ``tickers``.
    """
    tickers = list(tickers)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise KeyError(f"unknown tickers: {missing}")
    out = prices.loc[:, tickers].copy()
    arr = out.to_numpy(dtype=np.float64, copy=True)
    n_filled = 0
    for j in range(arr.shape[1]):
        col = arr[:, j]
        starts, stops = _nan_runs(np.isnan(col))
        for s, e in zip(starts, stops):
            if s > 0 and e < col.size and e - s <= max_gap:
                col[s:e] = col[s - 1]
                n_filled += e - s
    if n_filled:
        logger.info("clean_prices: forward-filled %d internal missing prices", n_filled)
    return pd.DataFrame(arr, index=out.index, columns=out.columns)


# --------------------------------------------------------------------------------------------
# Returns and splits
# --------------------------------------------------------------------------------------------
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log-returns ``log P_t - log P_{t-1}``; the first (undefined) row is dropped.

    NaN prices give NaN returns (no filling). The return dated ``t`` uses prices at ``t`` and the
    previous row, so the first TEST return (2023-01-03) uses the last TRAIN price; it is a TEST
    observation and carries no TEST information into TRAIN.
    """
    arr = prices.to_numpy(dtype=np.float64)
    if np.any(arr[~np.isnan(arr)] <= 0):
        raise ValueError("prices must be positive")
    lp = np.log(prices.astype(np.float64))
    return lp.diff().iloc[1:]


def split(returns: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    """Rows of ``returns`` dated within ``period = (start, end)``, both ends inclusive."""
    start, end = period
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(f"empty period {period}")
    return returns.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()


def panel_tickers(n: int | None = config.N_PANEL, seed: int = config.PANEL_SEED, *,
                  exclude: Sequence[str] = DATA_ANOMALIES) -> list[str]:
    """Seeded random subset of the complete universe (``config.FULL``, ``max_gap = 3``).

    The tickers with a complete history (availability rule only, alphabetically sorted; 481 in
    the current file) are shuffled once with ``default_rng(seed).permutation`` and the first
    ``n`` tickers not in ``exclude`` are kept. The draw is therefore deterministic and
    prefix-stable (the panel of size 50 is contained in the panel of size 100 for the same
    seed), and excluding a ticker replaces it by the next one in the permutation while the
    other selections stay unchanged. With the defaults, KDP and AMCR (in the draw without
    exclusion) are replaced by PCAR and TMUS.

    Parameters
    ----------
    n : int or None
        Panel size; ``None`` returns the whole universe minus ``exclude`` (with the default,
        equal to ``complete_universe(load_prices())``).
    seed : int
        Seed of the draw (default ``config.PANEL_SEED``).
    exclude : sequence of str
        Tickers never selected; default :data:`DATA_ANOMALIES`. ``exclude=()`` reproduces the
        draw that keeps the data errors (for sensitivity checks only).

    Returns
    -------
    list of str
        Selected tickers, sorted alphabetically.
    """
    drop = set(exclude)
    universe = complete_universe(load_prices(), *config.FULL, exclude=())
    eligible = [t for t in universe if t not in drop]
    if n is None:
        return eligible
    if not 1 <= n <= len(eligible):
        raise ValueError(f"n must be in [1, {len(eligible)}], got {n}")
    rng = np.random.default_rng(seed)
    order = [universe[i] for i in rng.permutation(len(universe))]
    return sorted([t for t in order if t not in drop][:n])


def load_panel(tickers: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TRAIN and TEST log-returns of a panel of tickers.

    Parameters
    ----------
    tickers : sequence of str, optional
        Default :func:`panel_tickers` (``N_PANEL`` tickers, ``PANEL_SEED``, without
        :data:`DATA_ANOMALIES`). Explicitly requesting an anomalous ticker logs a warning.

    Returns
    -------
    (train_returns, test_returns) : tuple of pd.DataFrame
        Date x ticker daily log-returns over ``config.TRAIN`` and ``config.TEST`` (inclusive),
        without any missing value.

    Raises
    ------
    ValueError
        If a requested ticker has missing returns in either period (e.g. a late listing);
        use :func:`complete_universe` to select admissible tickers.
    """
    if tickers is None:
        tickers = panel_tickers()
    else:
        tickers = list(tickers)
        flagged = sorted(set(tickers) & set(DATA_ANOMALIES))
        if flagged:
            logger.warning("load_panel: %s are known data errors (see data.ANOMALY_NOTES)",
                           flagged)
    rets = log_returns(clean_prices(load_prices(), tickers))
    train, test = split(rets, config.TRAIN), split(rets, config.TEST)
    bad = sorted(set(train.columns[train.isna().any()]) | set(test.columns[test.isna().any()]))
    if bad:
        raise ValueError(f"tickers with missing returns in TRAIN or TEST (not back-filled): {bad}")
    return train, test


# --------------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------------
def train_vol(train_returns: pd.DataFrame) -> pd.Series:
    """Per-ticker standard deviation (``ddof = 1``) of TRAIN daily log-returns.

    This is the fixed scale used to normalise real *and* synthetic returns of each ticker
    before pooling (DESIGN section 2).
    """
    vol = train_returns.std(axis=0, ddof=1)
    if vol.isna().any() or (vol <= 0).any():
        raise ValueError("train volatility must be finite and positive for every ticker")
    vol.name = "train_vol"
    return vol


def normalise(returns: pd.DataFrame | pd.Series | np.ndarray,
              vol: pd.Series | float | np.ndarray) -> pd.DataFrame | pd.Series | np.ndarray:
    """Divide returns by a fixed per-ticker scale (the TRAIN standard deviation).

    Parameters
    ----------
    returns : DataFrame, Series or ndarray
        * DataFrame (Date x ticker): ``vol`` must be a Series indexed by ticker that covers every
          column; each column is divided by its own ticker's scale.
        * Series or ndarray: divided by ``vol`` with numpy broadcasting (a scalar, or an array
          whose shape broadcasts, e.g. ``(N,)`` against ``(T, N)``).
    vol : Series, float or ndarray
        Strictly positive scales, typically :func:`train_vol` of the TRAIN returns.

    Returns
    -------
    Same type as ``returns``.
    """
    if isinstance(returns, pd.DataFrame):
        if not isinstance(vol, pd.Series):
            raise TypeError("normalising a DataFrame requires vol as a Series indexed by ticker")
        missing = [c for c in returns.columns if c not in vol.index]
        if missing:
            raise KeyError(f"no scale for tickers {missing}")
        v = vol.loc[returns.columns].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(v) & (v > 0)):
            raise ValueError("scales must be finite and positive")
        return returns.div(v, axis=1)
    v = np.asarray(vol, dtype=np.float64)
    if not np.all(np.isfinite(v) & (v > 0)):
        raise ValueError("scales must be finite and positive")
    if isinstance(returns, pd.Series):
        return returns / float(v) if v.ndim == 0 else returns / v
    return np.asarray(returns, dtype=np.float64) / v


# --------------------------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------------------------
def window_starts(T: int, L: int, step: int | None = None) -> np.ndarray:
    """Start indices ``0, step, 2 step, ...`` of the complete windows of length ``L`` in ``T``."""
    step = L if step is None else step
    if L < 1 or step < 1:
        raise ValueError("L and step must be >= 1")
    if T < L:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, T - L + 1, step, dtype=np.int64)


def make_windows(x: np.ndarray, L: int, step: int | None = None) -> np.ndarray:
    """Cut a series into windows of length ``L``, in time order.

    Parameters
    ----------
    x : ndarray, shape (T,) or (T, N)
        Univariate or multivariate series (DataFrames/Series are converted with ``np.asarray``).
    L : int
        Window length.
    step : int, optional
        Distance between window starts; default ``L`` (non-overlapping). An incomplete tail
        (fewer than ``L`` remaining observations) is dropped.

    Returns
    -------
    ndarray
        ``(n, L)`` for ``x`` of shape ``(T,)`` and ``(n, N, L)`` for ``(T, N)``, with
        ``n = (T - L) // step + 1`` (0 if ``T < L``). Window ``i`` is ``x[i*step : i*step + L]``
        (for 2-D input, transposed to asset x time). C-contiguous ``float64`` copy.

    Raises
    ------
    ValueError
        On non-finite input (windows must not silently contain missing values).
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim not in (1, 2):
        raise ValueError(f"x must have shape (T,) or (T, N), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("x contains NaN or inf")
    starts = window_starts(a.shape[0], L, step)
    if starts.size == 0:
        shape = (0, L) if a.ndim == 1 else (0, a.shape[1], L)
        return np.zeros(shape, dtype=np.float64)
    # sliding_window_view appends the window axis last: (T-L+1, L) or (T-L+1, N, L).
    view = np.lib.stride_tricks.sliding_window_view(a, L, axis=0)
    return np.ascontiguousarray(view[starts])


def panel_windows(returns: pd.DataFrame | np.ndarray, L: int,
                  step: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool univariate windows of every ticker of a panel, with group and time labels.

    Parameters
    ----------
    returns : DataFrame or ndarray, shape (T, N)
        Returns (typically already normalised), one column per ticker.
    L, step : int
        As in :func:`make_windows`.

    Returns
    -------
    X : ndarray, shape (N * n, L)
        Windows ordered ticker-major (all windows of column 0 in time order, then column 1, ...).
    groups : ndarray of int, shape (N * n,)
        Column position of the ticker of each window (for group-wise train/eval splits).
    order : ndarray of int, shape (N * n,)
        Window position in time (0, 1, ..., n-1), usable as ``order_x`` for HAC statistics.
    """
    a = np.asarray(returns, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("returns must be 2-D (T, N)")
    W = make_windows(a, L, step)  # (n, N, L)
    n, N, _ = W.shape
    X = np.ascontiguousarray(W.transpose(1, 0, 2).reshape(N * n, L))
    groups = np.repeat(np.arange(N, dtype=np.int64), n)
    order = np.tile(np.arange(n, dtype=np.int64), N)
    return X, groups, order
