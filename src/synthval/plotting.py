"""One matplotlib style for every paper figure (DESIGN section 3.11).

Usage::

    from synthval import plotting
    plotting.setup()
    fig, ax = plt.subplots(figsize=plotting.figsize("one"))
    ax.plot(x, y, color=plotting.COLORS_TEST["critic"], label=plotting.LABELS_TEST["critic"])
    plotting.savefig(fig, "e2_power")          # -> paper/figures/e2_power.pdf

Colours come from the Okabe-Ito palette (Okabe & Ito, 2008), which stays distinguishable under
the common forms of colour-vision deficiency; markers give a second, colour-free cue. Text uses
matplotlib's own mathtext and bundled fonts, so no LaTeX installation is needed at run time.
Fonts are embedded as TrueType (``pdf.fonttype = 42``) and PDFs carry no creation date, so
re-running an experiment gives byte-stable figures for unchanged data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib as mpl
from matplotlib.figure import Figure

from . import config

logger = logging.getLogger(__name__)

__all__ = [
    "ONE_COL",
    "TWO_COL",
    "OKABE_ITO",
    "COLORS_TEST",
    "COLORS_MODEL",
    "MARKERS_TEST",
    "MARKERS_MODEL",
    "LABELS_TEST",
    "setup",
    "figsize",
    "savefig",
]

ONE_COL: float = 3.4   # inches, single-column figure width
TWO_COL: float = 7.0   # inches, full text width
GOLDEN: float = 0.618

#: Okabe-Ito colour-blind-safe palette (yellow omitted from defaults: too light on white).
OKABE_ITO: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}
_O = OKABE_ITO

#: One fixed colour per test; the paper's method (critic) gets the most salient colour and the
#: likelihood-ratio oracle is black (it is the benchmark, not a competitor).
COLORS_TEST: dict[str, str] = {
    "critic": _O["vermillion"],
    "lrcritic": _O["purple"],
    "c2st": _O["orange"],
    "mmd": _O["blue"],
    "sigmmd": _O["skyblue"],
    "ks": _O["green"],
    "oracle": _O["black"],
}

#: One fixed colour per model. A multivariate model shares the colour of its univariate
#: analogue (they never share a figure) and is told apart by its marker; real data is black.
COLORS_MODEL: dict[str, str] = {
    "GBM": _O["blue"],
    "Merton": _O["orange"],
    "GARCHt": _O["vermillion"],
    "Heston": _O["purple"],
    "Bootstrap": _O["green"],
    "MVGaussian": _O["blue"],
    "MVStudentT": _O["orange"],
    "CCCGARCHt": _O["vermillion"],
    "MVBootstrap": _O["green"],
    "Real": _O["black"],
}

MARKERS_TEST: dict[str, str] = {
    "critic": "o", "lrcritic": "D", "c2st": "s", "mmd": "^", "sigmmd": "v", "ks": "x",
    "oracle": "*",
}
MARKERS_MODEL: dict[str, str] = {
    "GBM": "o", "Merton": "s", "GARCHt": "^", "Heston": "D", "Bootstrap": "v",
    "MVGaussian": "P", "MVStudentT": "X", "CCCGARCHt": "<", "MVBootstrap": ">", "Real": "*",
}

#: Display names for legends and tables.
LABELS_TEST: dict[str, str] = {
    "critic": "W-critic", "lrcritic": "LR-critic", "c2st": "C2ST", "mmd": "MMD",
    "sigmmd": "Sig-MMD", "ks": "KS", "oracle": "LR oracle",
}

_FONTS: dict[str, tuple[list[str], str]] = {
    # name: (font.serif fallback list, mathtext.fontset); all fonts ship with matplotlib.
    "stix": (["STIXGeneral", "STIX Two Text", "DejaVu Serif"], "stix"),
    "dejavu": (["DejaVu Serif"], "dejavuserif"),
    "cm": (["cmr10", "DejaVu Serif"], "cm"),
}


def setup(font: Literal["stix", "dejavu", "cm"] = "stix", base_size: float = 9.0) -> None:
    """Apply the paper style to ``matplotlib.rcParams`` (idempotent).

    Parameters
    ----------
    font : {"stix", "dejavu", "cm"}
        Serif text font with a matching mathtext set: STIX (Times-like, default), DejaVu Serif,
        or Computer Modern (``cmr10``, closest to the default LaTeX font). All are bundled with
        matplotlib; ``text.usetex`` stays False.
    base_size : float
        Font size in points for text and labels (tick labels and legends are 1 pt smaller).
    """
    if font not in _FONTS:
        raise ValueError(f"font must be one of {sorted(_FONTS)}")
    serif, mathfont = _FONTS[font]
    cycle = [_O[c] for c in ("blue", "vermillion", "green", "orange", "purple", "skyblue",
                             "black")]
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": serif,
        "mathtext.fontset": mathfont,
        "axes.formatter.use_mathtext": True,  # cmr10 has no unicode minus; harmless otherwise
        "font.size": base_size,
        "axes.titlesize": base_size,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "legend.title_fontsize": base_size - 1,
        "figure.titlesize": base_size + 1,
        "figure.figsize": (ONE_COL, ONE_COL * GOLDEN),
        "figure.dpi": 150,
        "figure.constrained_layout.use": True,
        "axes.prop_cycle": mpl.cycler(color=cycle),
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figsize(width: Literal["one", "two"] | float = "one", aspect: float = GOLDEN,
            ) -> tuple[float, float]:
    """Figure size ``(w, w * aspect)`` in inches; ``width`` is ``"one"``, ``"two"`` or inches."""
    w = {"one": ONE_COL, "two": TWO_COL}.get(width, width) if isinstance(width, str) else width
    if isinstance(w, str):
        raise ValueError(f"width must be 'one', 'two' or a number, got {width!r}")
    return float(w), float(w) * aspect


def savefig(fig: Figure, name: str, *, fig_dir: Path | None = None,
            formats: tuple[str, ...] = ("pdf",), close: bool = False) -> Path:
    """Save ``fig`` as ``paper/figures/<name>.pdf`` (and optional extra formats).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        File stem, e.g. ``"e2_power"`` (a suffix such as ``.pdf`` is ignored).
    fig_dir : Path, optional
        Output directory (default ``config.FIG_DIR``); created if needed.
    formats : tuple of str
        File formats; the first one's path is returned.
    close : bool
        Close the figure after saving (frees memory in long experiment loops).

    Returns
    -------
    Path
        Path of the first written file.
    """
    out_dir = Path(fig_dir or config.FIG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem if Path(name).suffix.lower() in {".pdf", ".png", ".svg"} else name
    paths = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        metadata = {"CreationDate": None} if fmt == "pdf" else None
        fig.savefig(path, format=fmt, metadata=metadata)
        paths.append(path)
        logger.info("saved %s", path)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return paths[0]
