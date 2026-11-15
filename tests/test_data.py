"""Tests for synthval.data, checked against the raw Bloomberg file."""
from __future__ import annotations

import csv
import math

import numpy as np
import pandas as pd
import pytest

from synthval import config, data

RAW = config.DATA_CSV
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="raw data file not available")

# Facts of the current file (data/sp500_price_volume.csv), established by direct inspection.
N_TICKERS = 503
N_PRICE_DAYS = 1874
N_COMPLETE = 481     # tickers with a complete PX_LAST history (availability rule only)
N_UNIVERSE = 479     # default complete universe: minus the data errors KDP and AMCR
N_TRAIN_DAYS = 1258  # returns dated 2018-01-03 .. 2022-12-30
N_TEST_DAYS = 615    # returns dated 2023-01-03 .. 2025-06-16
LATE_LISTINGS = {
    "DAY": "2018-04-26", "MRNA": "2018-12-07", "FOXA": "2019-03-12", "FOX": "2019-03-13",
    "DOW": "2019-03-20", "UBER": "2019-05-10", "CTVA": "2019-05-24", "CRWD": "2019-06-12",
    "CARR": "2020-03-19", "OTIS": "2020-03-19", "PLTR": "2020-09-30", "DASH": "2020-12-09",
    "ABNB": "2020-12-10", "EXE": "2021-02-10", "COIN": "2021-04-14", "CEG": "2022-01-19",
    "GEHC": "2022-12-15", "KVUE": "2023-05-04", "VLTO": "2023-10-04", "GEV": "2024-04-01",
    "SOLV": "2024-04-01", "SW": "2024-07-08",
}


def _raw_px_last(ticker: str, date: str) -> float | None:
    """PX_LAST of (ticker, date) read straight from the raw text lines (None if empty)."""
    with open(RAW, encoding="utf-8", newline="") as fh:
        lines = fh.read().split("\r\n")
    tick_row, field_row = lines[2].split(";"), lines[3].split(";")
    col = next(i for i, (t, f) in enumerate(zip(tick_row, field_row))
               if t == ticker and f == "PX_LAST")
    line = next(ln for ln in lines[4:] if ln.startswith(date + ";"))
    cell = line.split(";")[col]
    return float(cell) if cell else None


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return data.load_prices()


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------
def test_metadata_lines():
    assert data.file_metadata() == {"start": "2018-01-01", "end": "2025-06-17"}


def test_shape_index_columns(prices):
    assert prices.shape == (N_PRICE_DAYS, N_TICKERS)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index.name == "Date" and prices.columns.name == "ticker"
    assert prices.index[0] == pd.Timestamp("2018-01-02")
    assert prices.index[-1] == pd.Timestamp("2025-06-16")  # last trading day in the file
    assert prices.index.is_unique and prices.index.is_monotonic_increasing
    assert prices.columns.is_unique
    assert (prices.dtypes == np.float64).all()
    assert list(prices.columns[:4]) == ["MMM", "AOS", "ABT", "ABBV"]
    assert {"BRK-B", "BF-B", "GOOGL", "GOOG"} <= set(prices.columns)


@pytest.mark.parametrize("ticker,date,expected", [
    ("MMM", "2018-01-02", 197.02),
    ("AAPL", "2020-03-16", 60.55),   # split-adjusted (4:1 in Aug 2020)
    ("BRK-B", "2025-06-16", 490.23),
])
def test_spot_check_against_raw_line(prices, ticker, date, expected):
    raw = _raw_px_last(ticker, date)
    assert raw == expected
    assert prices.loc[date, ticker] == expected


def test_full_matrix_matches_independent_parse(prices):
    """Every PX_LAST cell equals an independent csv-module parse of the raw file."""
    with open(RAW, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    assert rows[0][0] == "Date de début" and rows[1][0] == "Date de fin"
    tick_row, field_row, body = rows[2], rows[3], rows[4:]
    assert all(len(r) == 1 + 2 * N_TICKERS for r in rows[2:])
    cols = [i for i, f in enumerate(field_row) if f == "PX_LAST"]
    assert [tick_row[i] for i in cols] == list(prices.columns)
    ref = np.array([[float(r[i]) if r[i] else np.nan for i in cols] for r in body])
    assert [r[0] for r in body] == [d.strftime("%Y-%m-%d") for d in prices.index]
    np.testing.assert_array_equal(prices.to_numpy(), ref)  # NaN positions equal too


def test_cache_returns_independent_copies(prices):
    p1 = data.load_prices()
    p1.iloc[0, 0] = -1.0
    p2 = data.load_prices()
    assert p2.iloc[0, 0] == 197.02
    assert data._parse_bloomberg.cache_info().hits >= 1


def test_volume_block():
    v = data.load_volume()
    assert v.shape == (N_PRICE_DAYS, N_TICKERS)
    assert v.loc["2018-01-02", "MMM"] == 3505476


# ---------------------------------------------------------------------------------------------
# No back-filling, complete universe, cleaning
# ---------------------------------------------------------------------------------------------
def test_late_listing_is_never_backfilled(prices):
    abnb = prices["ABNB"]
    assert _raw_px_last("ABNB", "2020-12-09") is None
    assert abnb.loc[:"2020-12-09"].isna().all()
    assert abnb.loc["2020-12-10"] == _raw_px_last("ABNB", "2020-12-10") == 144.71
    assert abnb.loc["2020-12-10":].notna().all()
    cleaned = data.clean_prices(prices, ["ABNB"])
    assert cleaned["ABNB"].loc[:"2020-12-09"].isna().all()
    r = data.log_returns(cleaned)
    assert r["ABNB"].loc[:"2020-12-10"].isna().all()  # first return needs two prices
    assert r["ABNB"].loc["2020-12-11":].notna().all()
    with pytest.raises(ValueError, match="ABNB"):
        data.load_panel(["ABNB"])


def test_first_price_dates_of_late_listings(prices):
    first = prices.apply(lambda s: s.first_valid_index())
    late = first[first > prices.index[0]]
    assert {t: d.strftime("%Y-%m-%d") for t, d in late.items()} == LATE_LISTINGS


def test_complete_universe_size(prices):
    u0 = data.complete_universe(prices, *config.FULL, exclude=())
    assert len(u0) == N_COMPLETE == N_TICKERS - len(LATE_LISTINGS)
    assert u0 == sorted(u0)
    assert not set(u0) & set(LATE_LISTINGS)
    assert prices[u0].notna().all().all()
    # The default universe drops the data errors, and only them.
    u = data.complete_universe(prices, *config.FULL)
    assert set(data.DATA_ANOMALIES) == {"KDP", "AMCR"}
    assert len(u) == N_UNIVERSE and set(u0) - set(u) == set(data.DATA_ANOMALIES)
    assert data.complete_universe(prices) == u and u == sorted(u)
    # Restricting the period to TEST admits the tickers listed before 2023.
    u_test = data.complete_universe(prices, *config.TEST)
    listed_before = {t for t, d in LATE_LISTINGS.items() if d <= "2023-01-03"}
    assert set(u_test) == set(u) | listed_before


def test_max_gap_rule_on_synthetic_prices():
    idx = pd.bdate_range("2020-01-01", periods=12, name="Date")
    nan = np.nan
    df = pd.DataFrame({
        "OK": np.arange(1.0, 13.0),
        "GAP3": [1, 2, nan, nan, nan, 6, 7, 8, 9, 10, 11, 12],
        "GAP4": [1, 2, nan, nan, nan, nan, 7, 8, 9, 10, 11, 12],
        "LEAD": [nan, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "TRAIL": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, nan],
        "TWO": [1, nan, 3, 4, nan, nan, 7, 8, 9, 10, 11, 12],
    }, index=idx, dtype=float)
    start, end = str(idx[0].date()), str(idx[-1].date())
    assert data.complete_universe(df, start, end, max_gap=3) == ["GAP3", "OK", "TWO"]
    assert data.complete_universe(df, start, end, max_gap=4) == ["GAP3", "GAP4", "OK", "TWO"]
    assert data.complete_universe(df, start, end, max_gap=0) == ["OK"]
    assert data.complete_universe(df, start, end, exclude=["GAP3"]) == ["OK", "TWO"]
    # A sub-period that starts after the leading gap admits LEAD.
    assert "LEAD" in data.complete_universe(df, str(idx[1].date()), end)

    c = data.clean_prices(df, list(df.columns), max_gap=3)
    np.testing.assert_array_equal(c["GAP3"].to_numpy()[2:5], [2.0, 2.0, 2.0])  # forward value
    assert c["GAP4"].iloc[2:6].isna().all()          # too long: left entirely as NaN
    assert np.isnan(c["LEAD"].iloc[0])               # never back-filled
    assert np.isnan(c["TRAIL"].iloc[-1])             # never extrapolated
    np.testing.assert_array_equal(c["TWO"].to_numpy()[[1, 4, 5]], [1.0, 4.0, 4.0])
    assert df["GAP3"].isna().sum() == 3              # input untouched


# ---------------------------------------------------------------------------------------------
# Returns and splits
# ---------------------------------------------------------------------------------------------
def test_log_returns_values(prices):
    r = data.log_returns(prices[["MMM", "AAPL"]])
    assert r.index[0] == pd.Timestamp("2018-01-03")
    assert len(r) == N_PRICE_DAYS - 1
    p0, p1 = _raw_px_last("AAPL", "2020-03-13"), _raw_px_last("AAPL", "2020-03-16")
    assert r.loc["2020-03-16", "AAPL"] == pytest.approx(math.log(p1) - math.log(p0), abs=1e-15)
    assert r.loc["2018-01-03", "MMM"] == 0.0  # 197.02 -> 197.02 in the raw file


def test_split_inclusive_and_partition(prices):
    u = data.complete_universe(prices)
    r = data.log_returns(data.clean_prices(prices, u))
    train, test = data.split(r, config.TRAIN), data.split(r, config.TEST)
    assert len(train) == N_TRAIN_DAYS and len(test) == N_TEST_DAYS
    assert train.index[0] == pd.Timestamp("2018-01-03")
    assert train.index[-1] == pd.Timestamp("2022-12-30")   # inclusive end
    assert test.index[0] == pd.Timestamp("2023-01-03")     # inclusive start
    assert test.index[-1] == pd.Timestamp("2025-06-16")
    assert len(train) + len(test) == len(r)
    assert train.index.intersection(test.index).empty
    assert not train.isna().any().any() and not test.isna().any().any()
    # the boundary-day return uses the last TRAIN price and is a TEST observation
    assert test.loc["2023-01-03", "MMM"] == pytest.approx(
        math.log(_raw_px_last("MMM", "2023-01-03")) - math.log(_raw_px_last("MMM", "2022-12-30")))


# ---------------------------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------------------------
def test_panel_tickers_deterministic_sorted(prices):
    a, b = data.panel_tickers(), data.panel_tickers()
    assert a == b
    assert len(a) == config.N_PANEL == len(set(a))
    assert a == sorted(a)
    universe = data.complete_universe(prices)
    assert set(a) <= set(universe)
    assert not set(a) & set(data.DATA_ANOMALIES)          # data errors excluded by default
    assert data.panel_tickers(seed=config.PANEL_SEED + 1) != a
    assert set(data.panel_tickers(50)) <= set(a)              # prefix-stable draw
    assert data.panel_tickers(None) == universe
    assert data.panel_tickers(None, exclude=()) == data.complete_universe(prices, exclude=())
    # Without the exclusion the same seed draws KDP and AMCR; excluding them swaps in the next
    # two names of the permutation (PCAR, TMUS) and leaves the other 98 selections unchanged.
    raw = data.panel_tickers(exclude=())
    assert {"KDP", "AMCR"} <= set(raw) and len(raw) == config.N_PANEL
    assert set(raw) - set(a) == {"KDP", "AMCR"} and set(a) - set(raw) == {"PCAR", "TMUS"}
    assert data.panel_tickers(exclude=["KDP", "AMCR"]) == a
    assert data.panel_tickers(exclude=[*data.DATA_ANOMALIES, "NOT_A_TICKER"]) == a
    with pytest.raises(ValueError):
        data.panel_tickers(N_UNIVERSE + 1)
    assert len(data.panel_tickers(N_COMPLETE, exclude=())) == N_COMPLETE


def test_load_panel(caplog):
    train, test = data.load_panel()
    assert list(train.columns) == data.panel_tickers() == list(test.columns)
    assert not set(train.columns) & set(data.DATA_ANOMALIES)
    assert train.shape == (N_TRAIN_DAYS, config.N_PANEL)
    assert test.shape == (N_TEST_DAYS, config.N_PANEL)
    assert np.isfinite(train.to_numpy()).all() and np.isfinite(test.to_numpy()).all()
    assert (train.dtypes == np.float64).all()
    with caplog.at_level("WARNING", logger="synthval.data"):
        kdp, _ = data.load_panel(["KDP", "MMM"])        # explicit request: loaded, with a warning
    assert "KDP" in caplog.text and list(kdp.columns) == ["KDP", "MMM"]


def test_train_vol_and_normalise():
    train, test = data.load_panel()
    vol = data.train_vol(train)
    assert list(vol.index) == list(train.columns)
    np.testing.assert_allclose(data.normalise(train, vol).std(ddof=1).to_numpy(), 1.0, rtol=1e-12)
    zt = data.normalise(test, vol)
    np.testing.assert_allclose(zt.to_numpy(), test.to_numpy() / vol.to_numpy(), rtol=0, atol=0)
    assert not np.allclose(zt.std(ddof=1).to_numpy(), 1.0)  # TEST scaled by TRAIN vol, not own
    # column order of the panel does not matter: alignment is by ticker
    rev = data.normalise(test[test.columns[::-1]], vol)
    pd.testing.assert_frame_equal(rev[zt.columns], zt)
    # arrays broadcast over the last axis; synthetic paths use a scalar
    np.testing.assert_allclose(data.normalise(test.to_numpy(), vol.to_numpy()), zt.to_numpy())
    paths = np.ones((3, 5))
    np.testing.assert_allclose(data.normalise(paths, 0.5), 2.0)
    with pytest.raises(KeyError):
        data.normalise(test, vol.iloc[1:])
    with pytest.raises(ValueError):
        data.normalise(paths, 0.0)


# ---------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------
def test_make_windows_univariate():
    x = np.arange(47, dtype=float)
    W = data.make_windows(x, 10)
    assert W.shape == (4, 10)                       # tail of 7 dropped
    np.testing.assert_array_equal(W[0], np.arange(10))
    np.testing.assert_array_equal(W[3], np.arange(30, 40))
    assert W.flags["C_CONTIGUOUS"] and W.dtype == np.float64
    W[0, 0] = 99.0
    assert x[0] == 0.0                              # a copy, not a view
    O = data.make_windows(x, 10, step=5)
    assert O.shape == ((47 - 10) // 5 + 1, 10)
    np.testing.assert_array_equal(O[:, 0], np.arange(0, 38, 5))
    assert data.make_windows(x, 47).shape == (1, 47)
    assert data.make_windows(x, 48).shape == (0, 48)
    np.testing.assert_array_equal(data.window_starts(47, 10), [0, 10, 20, 30])


def test_make_windows_multivariate():
    T, N, L = 43, 3, 5
    x = np.arange(T * N, dtype=float).reshape(T, N)
    W = data.make_windows(x, L)
    assert W.shape == (T // L, N, L)
    for i in range(W.shape[0]):
        for k in range(N):
            np.testing.assert_array_equal(W[i, k], x[i * L:(i + 1) * L, k])
    W2 = data.make_windows(pd.DataFrame(x), L, step=2)
    assert W2.shape == ((T - L) // 2 + 1, N, L)
    np.testing.assert_array_equal(W2[1, 2], x[2:7, 2])
    assert data.make_windows(x, T + 1).shape == (0, N, T + 1)


def test_make_windows_errors():
    with pytest.raises(ValueError):
        data.make_windows(np.array([0.0, np.nan, 1.0]), 2)
    with pytest.raises(ValueError):
        data.make_windows(np.zeros((2, 2, 2)), 1)
    with pytest.raises(ValueError):
        data.make_windows(np.zeros(5), 0)
    with pytest.raises(ValueError):
        data.make_windows(np.zeros(5), 2, step=0)


def test_panel_windows_on_real_panel():
    train, _ = data.load_panel()
    z = data.normalise(train, data.train_vol(train))
    L = config.WINDOW_L
    X, groups, order = data.panel_windows(z, L)
    n = N_TRAIN_DAYS // L
    assert X.shape == (config.N_PANEL * n, L)
    assert groups.shape == order.shape == (X.shape[0],)
    j, i = 7, 11
    row = j * n + i
    assert groups[row] == j and order[row] == i
    np.testing.assert_array_equal(X[row], z.iloc[i * L:(i + 1) * L, j].to_numpy())


def test_quality_report_flags_known_anomalies(prices):
    q = data.quality_report()
    assert len(q) == N_TICKERS
    assert int(q["complete_history"].sum()) == N_COMPLETE
    assert int(q["in_universe"].sum()) == N_UNIVERSE
    assert sorted(q.index[q["in_universe"]]) == data.complete_universe(prices)
    # The objective screen flags exactly the excluded tickers (nothing hand-picked).
    assert set(q.index[q["anomaly"]]) == set(data.DATA_ANOMALIES) == set(data.ANOMALY_NOTES)
    # KDP: special dividend of 2018-07-10 not adjusted in PX_LAST (log-return of -1.718).
    assert _raw_px_last("KDP", "2018-07-09") == 123.66 and _raw_px_last("KDP", "2018-07-10") == 22.19
    assert q.loc["KDP", "date_max_abs_ret"] == pd.Timestamp("2018-07-10")
    assert q.loc["KDP", "max_abs_ret"] == pytest.approx(math.log(123.66 / 22.19), abs=1e-12)
    others = q.loc[q["complete_history"]].drop(["KDP"])
    assert others["max_abs_ret"].max() < 0.8 < data.ANOMALY_MAX_ABS_RET   # genuine extremes
    # AMCR: stale zero-volume quotes before its NYSE listing (2019-06-11).
    assert q.loc["AMCR", "n_zero_volume"] == 232 and q.loc["AMCR", "longest_zero_ret_run"] == 18
    assert q["n_zero_volume"].drop("AMCR").max() <= 1
    assert q.loc[q["complete_history"], "longest_zero_ret_run"].drop("AMCR").max() <= 4
    v = data.load_volume()["AMCR"]
    assert (v.loc["2019-06-11":] > 0).all() and (v.loc[:"2019-06-10"] == 0).sum() == 232
    # The exclusion matters per ticker: KDP's TRAIN vol is ~3x its vol without the bad day.
    r = data.log_returns(prices[["KDP"]])["KDP"].loc[config.TRAIN[0]:config.TRAIN[1]]
    assert r.std() > 3 * r.drop(pd.Timestamp("2018-07-10")).std()
