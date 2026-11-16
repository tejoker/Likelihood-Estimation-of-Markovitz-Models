"""Seed-to-seed reliability of the E4 test quantities (Section 8, label tab:manual-e4-reliability).

Reads ``results/e4/e4_tests.csv`` (written by ``experiments/e4_markowitz.py --part tests``: one
row per universe, generator, synthetic seed and test) and writes
``paper/tables/manual_e4_reliability.tex``.

For each of the four test quantities that enter the link of Section 8.4 (W-critic statistic
``T`` = column ``statistic`` of the critic rows; its studentised statistic ``z_HAC`` = ``zstat``
of the critic rows; ``MMD^2_u`` = ``statistic`` of the MMD rows; the wild-standardised MMD
statistic = ``zstat`` of the MMD rows) the table gives the Spearman correlation between the
values of the two synthetic seeds (seed 0 and seed 1):

* within each generator, across the 30 universes (the variation the within-generator view of
  the link uses), and their mean over the five generators (the link averages within-generator
  correlations the same way);
* pooled over the 150 (universe, generator) points;
* for the reference row (last 615 days of real TRAIN), whose two "seeds" use the same data and
  differ only in the randomness of the test (time-block split, critic initialisation and
  mini-batches for the critic; wild-bootstrap multipliers for MMD).

A low value means that the statistic of one seed hardly predicts that of the other, so that
its correlation with any other quantity across universes is attenuated by seed noise.

The header comment of the output also records two derived counts quoted in the text: the
number of training windows per sample of the critic (``n_x - n_eval_x`` of the critic rows) and
the number of trainable parameters of the E4 critic, the ``ConvCritic`` of
``src/synthval/twosample/nets.py`` for (N, L) = (30, 5) windows (two Conv1d blocks with 64
channels and kernel 5, each followed by LayerNorm([64, L]); dense head 64 * L -> 64 -> 1),
counted from that architecture without importing torch.

Usage (from the repository root)::

    uv run python paper/scripts/e4_reliability.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "e4" / "e4_tests.csv"
META = ROOT / "results" / "e4" / "meta_tests.json"
OUT = ROOT / "paper" / "tables" / "manual_e4_reliability.tex"

GENERATORS = ("MVGaussian", "MVStudentT", "CCCGARCHt", "MVBootstrap", "FactorGARCHt")
REF = "RealTRAIN"
LABELS = {
    "MVGaussian": "MV Gaussian", "MVStudentT": "MV Student-$t$", "CCCGARCHt": "CCC-GARCH-$t$",
    "MVBootstrap": "MV bootstrap", "FactorGARCHt": "Factor-GARCH-$t$",
}
QUANTITIES = (
    ("critic", "statistic", r"\WC{} $T$"),
    ("critic", "zstat", r"\WC{} $\zhac$"),
    ("mmd", "statistic", r"MMD$^2_u$"),
    ("mmd", "zstat", r"MMD, wild-std."),
)


def num(v: float) -> str:
    if not math.isfinite(v):
        return "--"
    return f"{v:.2f}".replace("-", "$-$")


def seed_pairs(d: pd.DataFrame, test: str, col: str) -> pd.DataFrame:
    """Wide table (generator, universe) x seed of one quantity."""
    w = d[d["test"] == test].pivot_table(index=["generator", "universe"], columns="seed",
                                         values=col)
    if list(w.columns) != [0, 1]:
        raise ValueError(f"expected seeds 0 and 1, got {list(w.columns)}")
    return w


def rho(w: pd.DataFrame) -> float:
    return float(stats.spearmanr(w[0], w[1]).statistic)


def conv_critic_params(c: int, L: int, hidden: int = 64, k: int = 5, n_conv: int = 2) -> int:
    """Trainable parameters of nets.ConvCritic (kernel min(k, L), LayerNorm([hidden, L]))."""
    k = max(1, min(k, L))
    n, ch = 0, c
    for _ in range(n_conv):
        n += ch * hidden * k + hidden          # Conv1d weight + bias
        n += 2 * hidden * L                    # LayerNorm([hidden, L]) weight + bias
        ch = hidden
    n += hidden * L * hidden + hidden          # Linear(hidden * L, hidden)
    n += hidden + 1                            # Linear(hidden, 1)
    return n


def main() -> None:
    d = pd.read_csv(SRC)
    crit = d[d["test"] == "critic"]
    n_train = (crit["n_x"] - crit["n_eval_x"]).agg(["min", "max"])
    cfg = json.loads(META.read_text(encoding="utf-8"))["config"]
    n_assets, L = int(cfg["n_assets"]), int(cfg["L"])
    if set(d["L"].unique()) != {L}:
        raise ValueError("window length in e4_tests.csv differs from meta_tests.json")
    params = conv_critic_params(n_assets, L)

    per_gen: dict[str, list[float]] = {g: [] for g in GENERATORS}
    mean_within, pooled, ref = [], [], []
    for test, col, _ in QUANTITIES:
        w = seed_pairs(d, test, col)
        vals = []
        for g in GENERATORS:
            r = rho(w.loc[g])
            per_gen[g].append(r)
            vals.append(r)
        mean_within.append(sum(vals) / len(vals))
        syn = w.drop(index=REF, level=0)
        if len(syn) != 150:
            raise ValueError(f"expected 150 synthetic points, got {len(syn)}")
        pooled.append(rho(syn))
        ref.append(rho(w.loc[REF]))

    lines = [
        "% Written by paper/scripts/e4_reliability.py from results/e4/e4_tests.csv.",
        f"% Critic training windows per sample (n_x - n_eval_x): {int(n_train['min'])} to "
        f"{int(n_train['max'])}; window dimension N x L = {n_assets} x {L} = {n_assets * L}; "
        f"trainable parameters of the E4 ConvCritic: {params}.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Seed-to-seed Spearman $\\rho$ & " + " & ".join(q[2] for q in QUANTITIES) + r" \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Within generator, across the 30 universes}} \\",
    ]
    for g in GENERATORS:
        lines.append(LABELS[g] + " & " + " & ".join(num(v) for v in per_gen[g]) + r" \\")
    lines.append(r"Mean over generators & " + " & ".join(num(v) for v in mean_within) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"Pooled, 150 points & " + " & ".join(num(v) for v in pooled) + r" \\")
    lines.append(r"Reference (same data in both seeds) & " + " & ".join(num(v) for v in ref)
                 + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print("\n".join(lines[:2]))


if __name__ == "__main__":
    main()
