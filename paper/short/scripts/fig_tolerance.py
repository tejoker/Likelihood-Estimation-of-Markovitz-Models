"""Figure 1 of the short paper: power of the verdict rule against a volatility error.

Reads the E7 power curves of the v2 runs and writes paper/short/figures/tolerance.pdf:
    results_v2/e7tol/e7_curve.csv      10 synthetic seeds (factor 1.2 and 1.0 are E3's runs)
    results_v2_s50/e7tol/e7_curve.csv  50 synthetic seeds (amendment A2)
Columns used: design, vol_factor, power, lo, hi (Wilson 95% interval).

From the repository root:
    uv run --project envs/gpu python paper/short/scripts/fig_tolerance.py [--png preview.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "short" / "figures" / "tolerance.pdf"

# Categorical slots 1-3 of the validated reference palette (light mode; all-pairs CVD and
# normal-vision checks pass). Two of them are below 3:1 contrast on white, so every series
# also carries a marker shape, a line style and a legend entry (grayscale print).
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, INK2, RULE = "#191919", "#666666", "#bdbdbd"


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d.sort_values(["design", "vol_factor"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=Path, default=None, help="also write a PNG preview here")
    args = ap.parse_args()

    c10 = load(ROOT / "results_v2" / "e7tol" / "e7_curve.csv")
    c50 = load(ROOT / "results_v2_s50" / "e7tol" / "e7_curve.csv")
    series = [
        # label, frame, design, colour, marker, line style
        ("D2, 10 seeds", c10, "D2", BLUE, "o", "-"),
        ("D2, 50 seeds", c50, "D2", ORANGE, "s", "--"),
        ("D1, 10 seeds", c10, "D1", AQUA, "^", ":"),
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
    })
    fig, ax = plt.subplots(figsize=(6.3, 2.3))

    ax.axhline(0.8, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(0.715, 0.815, "required power 0.8", color=INK2, fontsize=7.5, va="bottom")
    ax.axvline(1.0, color=RULE, lw=0.8, zorder=1)
    ax.text(0.70, 1.075, r"generator calmer than the past $\leftarrow$", color=INK2,
            fontsize=7.5, ha="center", va="bottom")
    ax.text(1.42, 1.075, r"$\rightarrow$ generator more volatile than the past", color=INK2,
            fontsize=7.5, ha="center", va="bottom")

    # Power curves on each side of the boundary; the boundary point itself (factor 1) is the
    # size of the rule, drawn as an unconnected marker so that no line suggests a power dip.
    for label, frame, design, colour, marker, ls in series:
        d = frame[frame["design"] == design]
        for side in (d[d["vol_factor"] < 1.0], d[d["vol_factor"] > 1.0]):
            if side.empty:
                continue
            x = side["vol_factor"].to_numpy()
            ax.fill_between(x, side["lo"], side["hi"], color=colour, alpha=0.12, lw=0, zorder=2)
            ax.plot(x, side["power"], color=colour, lw=1.6, ls=ls, marker=marker, ms=4.2,
                    mec="white", mew=0.6, zorder=3, solid_capstyle="round")
        b = d[d["vol_factor"] == 1.0]
        ax.plot(b["vol_factor"], b["power"], ls="none", marker=marker, ms=4.2, color=colour,
                mec="white", mew=0.6, zorder=4)
        ax.plot([], [], color=colour, lw=1.6, ls=ls, marker=marker, ms=4.2, mec="white",
                mew=0.6, label=label)

    # Selective value labels: the cell that decides the certified tolerance (factor 1.4).
    for frame, text_xy, seeds in ((c10, (1.60, 0.66), 10), (c50, (1.60, 0.88), 50)):
        row = frame[(frame["design"] == "D2") & (frame["vol_factor"].round(3) == 1.4)].iloc[0]
        ax.annotate(f"{row['power']:.3g} ({seeds} seeds)", (1.4, row["power"]), xytext=text_xy,
                    textcoords="data", fontsize=7.5, color=INK, ha="left", va="center",
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6,
                                "shrinkA": 1, "shrinkB": 3})

    ticks = [0.5, 1 / 1.5, 1 / 1.2, 1.0, 1.2, 1.4, 2.0]
    labels = ["1/2", "1/1.5", "1/1.2", "1", "1.2", "1.4", "2"]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(labels)
    ax.set_xlim(0.48, 2.08)
    ax.set_ylim(-0.03, 1.07)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("volatility factor of the generator relative to the boundary generator (log scale)")
    ax.set_ylabel("rule power")
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False,
              handlelength=2.6, columnspacing=2.2)

    fig.tight_layout(pad=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    if args.png is not None:
        fig.savefig(args.png, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
