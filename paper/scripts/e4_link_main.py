"""Condensed main-text version of the E4 link table (Section 8, label tab:e4-link).

Reads ``results/e4/e4_link.csv`` (written by ``experiments/e4_markowitz.py --part link``) and
writes ``paper/tables/manual_e4_link_main.tex``: Spearman correlations between five test
quantities and two utility losses in the four views of the link (pooled over the 150
(universe, generator) points, within universe, within generator, generator level), sample
moments. Panel A: replica-wise portfolios (the DESIGN measure) for the GMV variance ratio and the
mean-variance (gamma = 5) CE loss. Panel B: portfolios from the moments of the pooled replicas
(covariance misspecification only) for the GMV variance ratio. The full tables are the generated
``e4_link.tex`` and ``e4_link_pooled.tex`` (appendix). Numbers are rounded to two (correlations)
and three (p-values) decimals exactly as in the generated tables; p-values below 0.0005 are
printed as ``<0.001``.

Usage (from the repository root)::

    uv run python paper/scripts/e4_link_main.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "e4" / "e4_link.csv"
OUT = ROOT / "paper" / "tables" / "manual_e4_link_main.tex"

QUANTITIES = {
    "critic_stat": r"\WC{} $T$", "critic_z": r"\WC{} $\zhac$",
    "mmd_stat": r"MMD$^2_u$", "mmd_z": r"MMD, wild-std.", "corr_gap": r"$|\Delta$ avg.\ corr.$|$",
}
PANELS = (
    (r"Panel A: replica-wise portfolios", "replica",
     (("var_ratio", "gmv", r"GMV var.\ ratio"), ("ce_loss", "mv5", r"MV $\gamma=5$ CE loss"))),
    (r"Panel B: portfolios from the pooled replicas", "pooled",
     (("var_ratio", "gmv", r"GMV var.\ ratio"),)),
)


def num(v: float) -> str:
    if not math.isfinite(v):
        return "--"
    return f"{v:.2f}".replace("-", "$-$")


def pval(p: float) -> str:
    if not math.isfinite(p):
        return "--"
    s = f"{p:.3f}"
    return "$<$0.001" if s == "0.000" else s


def main() -> None:
    link = pd.read_csv(SRC)
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Utility loss & Test quantity & Pooled $\rho$ [95\% CI] & Within-univ.\ $\bar\rho$ ($p_u$)"
        r" & Within-gen.\ $\bar\rho$ ($p_g$) & Gen.-level $\rho$ ($p$) \\",
        r"\midrule",
    ]
    for pi, (title, agg, losses) in enumerate(PANELS):
        if pi:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{6}{l}{\emph{" + title + r"}} \\")
        for loss, port, llabel in losses:
            for qi, (q, qlabel) in enumerate(QUANTITIES.items()):
                d = link[(link["quantity"] == q) & (link["loss"] == loss)
                         & (link["est"] == "sample") & (link["port"] == port)
                         & (link["agg"] == agg)]
                if len(d) != 1:
                    raise ValueError(f"expected one row for {q}, {loss}, {port}, {agg}")
                r = d.iloc[0]
                cells = [
                    f"{num(r['rho_pooled'])} [{num(r['lo_pooled'])}, {num(r['hi_pooled'])}]",
                    f"{num(r['rho_within_u'])} ({pval(r['p_perm_within_u'])})",
                    f"{num(r['rho_within_g'])} ({pval(r['p_perm_within_g'])})",
                    f"{num(r['rho_gen_level'])} ({pval(r['p_exact_gen_level'])})",
                ]
                lines.append((llabel if qi == 0 else "") + " & " + qlabel + " & "
                             + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
