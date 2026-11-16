"""Tables of the data appendix (Appendix D, label app:data) and of Section 6 (label sec:data).

Writes three bare tabulars (booktabs) to ``paper/tables``:

* ``manual_data_universe.tex``: construction of the ticker universe from the Bloomberg export
  (``synthval.data.quality_report``: tickers in the file, late listings, complete-history
  tickers, anomaly screen, default universe, main panel);
* ``manual_data_anomalies.tex``: the anomaly screen (``data.ANOMALY_MAX_ABS_RET``,
  ``data.ANOMALY_MIN_ZERO_VOLUME``) for the two excluded tickers (KDP, AMCR) and, for reference,
  the extreme values over the 479 tickers of the default universe;
* ``manual_data_panel.tex``: the 100 tickers of the main panel, read from
  ``results/e3/meta_fit.json`` (the tickers E3 actually fitted) and checked against
  ``data.panel_tickers()``.

The universe and anomaly tables are computed from the raw data file through ``synthval.data``
(no randomness); the panel table reads an experiment output.

Usage (from the repository root)::

    uv run python paper/scripts/data_appendix.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from synthval import config, data

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "paper" / "tables"
META_FIT = ROOT / "results" / "e3" / "meta_fit.json"


def num(v: float, d: int = 2) -> str:
    x = float(v)
    if not math.isfinite(x):
        return "--"
    s = f"{x:.{d}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return "$-$" + s[1:] if s.startswith("-") else s


def write(name: str, lines: list[str]) -> None:
    path = TAB / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


# ----------------------------------------------------------------------------------------------
def universe_tables() -> None:
    prices = data.load_prices()
    q = data.quality_report()
    n_file = int(q.shape[0])
    complete = q[q["complete_history"]]
    late = q[~q["complete_history"]]
    universe = q[q["in_universe"]]
    flagged = sorted(complete.index[complete["anomaly"]])
    if flagged != sorted(data.DATA_ANOMALIES):
        raise RuntimeError(f"screen flags {flagged}, DATA_ANOMALIES = {data.DATA_ANOMALIES}")
    if int(universe["n_missing"].sum()) != 0:
        raise RuntimeError("missing prices inside the default universe")
    first = pd.to_datetime(late["first_date"])
    panel = data.panel_tickers()
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Step & Tickers \\",
        r"\midrule",
        rf"Current S\&P~500 constituents in the Bloomberg export & {n_file} \\",
        rf"\quad no price on {prices.index[0].date()} (first price between "
        rf"{first.min().date()} and {first.max().date()}) & $-${late.shape[0]} \\",
        rf"Complete price history, {prices.index[0].date()} to {prices.index[-1].date()} "
        rf"& {complete.shape[0]} \\",
        rf"\quad flagged by the anomaly screen ({', '.join(sorted(flagged, reverse=True))}) "
        rf"& $-${len(flagged)} \\",
        rf"Default universe (E4 universes are drawn from it) & {universe.shape[0]} \\",
        rf"Main panel (seeded draw, seed {config.PANEL_SEED}; E1, E3, E5) & {len(panel)} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("manual_data_universe", lines)

    rets = data.log_returns(prices)
    train = data.split(rets, config.TRAIN)
    zero_train = (train == 0).sum()

    def zero_run(t: str) -> int:
        z = (train[t] == 0).to_numpy().astype(np.int8)
        d = np.diff(np.concatenate(([0], z, [0])))
        starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        return int((stops - starts).max()) if starts.size else 0

    rows = []
    for t in sorted(flagged, reverse=True):  # KDP, AMCR
        r = q.loc[t]
        rows.append(rf"{t} & {num(r['max_abs_ret'], 3)} ({pd.Timestamp(r['date_max_abs_ret']).date()})"
                    rf" & {int(r['n_zero_volume'])} & {int(zero_train[t])} & {zero_run(t)} \\")
    u = universe.index
    imax = universe["max_abs_ret"].idxmax()
    zt = zero_train[u]
    runs = pd.Series({t: zero_run(t) for t in u})
    rows.append(rf"Default universe (479), maximum & {num(universe.loc[imax, 'max_abs_ret'], 3)} "
                rf"({imax}, {pd.Timestamp(universe.loc[imax, 'date_max_abs_ret']).date()}) & "
                rf"{int(universe['n_zero_volume'].max())} & {int(zt.max())} ({zt.idxmax()}) & "
                rf"{int(runs.max())} \\")
    rows.append(rf"Default universe (479), median & {num(universe['max_abs_ret'].median(), 3)} & "
                rf"{num(universe['n_zero_volume'].median(), 0)} & {num(zt.median(), 0)} & "
                rf"{num(runs.median(), 0)} \\")
    if universe.shape[0] != 479:
        raise RuntimeError("the row labels assume a 479-ticker universe")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Ticker & Largest $\abs{r_{it}}$ (date) & Zero-volume days & Zero \TRAIN{} returns"
        r" & Longest zero run \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("manual_data_anomalies", lines)


# ----------------------------------------------------------------------------------------------
def panel_table(n_cols: int = 10) -> None:
    meta = json.loads(META_FIT.read_text(encoding="utf-8"))
    tickers = list(meta["config"]["tickers"])
    if sorted(tickers) != sorted(data.panel_tickers()):
        raise RuntimeError("E3 fitted tickers differ from data.panel_tickers()")
    tickers = sorted(tickers)
    n_rows = math.ceil(len(tickers) / n_cols)
    lines = [r"\begin{tabular}{" + "l" * n_cols + "}", r"\toprule"]
    for i in range(n_rows):
        cells = [tickers[j * n_rows + i] if j * n_rows + i < len(tickers) else ""
                 for j in range(n_cols)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("manual_data_panel", lines)


if __name__ == "__main__":
    universe_tables()
    panel_table()
