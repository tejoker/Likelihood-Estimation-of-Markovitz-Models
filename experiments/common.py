"""Shared infrastructure for the experiment scripts ``experiments/eX_*.py``.

Every experiment script follows the same conventions (DESIGN section 4):

* ``uv run python experiments/<script>.py [--quick] [--workers N] [--part NAME] [--plots-only]
  [--fresh]``.
* BLAS/OpenMP thread variables are set to 1 before numpy is imported (this module does it at
  import time too, so ``import common`` before ``import numpy`` is enough).
* Replications are run in worker processes with :func:`synthval.utils.parallel_map` (spawn
  start method: worker functions must live at module level and the script must guard its entry
  point with ``if __name__ == "__main__":``).
* Every replication's generator is derived from a fixed root seed and the replication's key
  ``(part, cell, rep)`` (:func:`keyed_rng`), so results do not depend on the number of workers,
  on the block size, or on which parts are run.
* Results are checkpointed per block of replications (:func:`run_cells`): an interrupted run
  resumes without recomputing finished blocks. The checkpoint directory is keyed by a hash of the
  part's configuration, so a changed configuration never reuses stale blocks (library code
  changes are *not* detected: pass ``--fresh`` to recompute).
* Full mode writes ``results/<exp>/*.csv`` and ``meta.json`` (:mod:`synthval.utils`), figures
  ``paper/figures/<exp>_*.pdf`` and tables ``paper/tables/<exp>_*.tex``. ``--quick`` writes
  everything (figures and tables included) under ``results/<exp>_quick/`` only.

Typical script skeleton::

    import os
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_k] = "1"
    import common                      # experiments/ is on sys.path when run as a script
    import numpy as np

    def one_rep(cell, rep, rng):       # module level: picklable for spawned workers
        ...
        return [{"test": "critic", "pvalue": p, "ptype": "permutation"}]

    def main():
        ctx = common.parse_args("e9", parts=("main",), description=__doc__)
        df = common.run_cells(one_rep, cells, part="main", ctx=ctx, root_seed=123,
                              n_reps=200, config={...})
        tab = common.rejection_table(df, ["cell", "test", "ptype"])
        ctx.save_csv(tab, "e9_size", config={...})

    if __name__ == "__main__":
        main()
"""
from __future__ import annotations

import os

#: Thread-count variables pinned to 1 (BLAS / OpenMP pools; one process per core instead).
THREAD_ENV_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def set_thread_env(n: int = 1) -> None:
    """Set the BLAS/OpenMP thread variables to ``n`` (call before numpy is imported)."""
    for k in THREAD_ENV_VARS:
        os.environ[k] = str(int(n))


set_thread_env(1)

import argparse  # noqa: E402  (after the thread variables, on purpose)
import dataclasses  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections.abc import Callable, Iterable, Mapping, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import synthval  # noqa: E402,F401  (sets MPLCONFIGDIR before matplotlib is imported)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from synthval import config, plotting, utils  # noqa: E402

logger = logging.getLogger("experiments")

__all__ = [
    "THREAD_ENV_VARS",
    "set_thread_env",
    "setup_logging",
    "ExpContext",
    "parse_args",
    "key_words",
    "keyed_seed_sequence",
    "keyed_rng",
    "config_hash",
    "cell_key",
    "Checkpoint",
    "run_cells",
    "run_items",
    "safe_test",
    "wilson_ci",
    "rejection_table",
    "fmt_rate_ci",
    "rate_ci_table",
    "Stopwatch",
    "estimate_wall",
    "print_rep_timing",
    "print_wall_estimate",
    "block_tasks",
    "fmt_seconds",
]


# --------------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    """Log to stderr with timestamps (idempotent). Library modules log under ``synthval.*``."""
    root = logging.getLogger()
    if not any(getattr(h, "_experiments", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        h._experiments = True  # type: ignore[attr-defined]
        root.addHandler(h)
    root.setLevel(level)
    for noisy in ("matplotlib", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path, _dt.datetime, _dt.date)):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return repr(obj)


def _dumps(obj: Any, **kw: Any) -> str:
    return json.dumps(obj, default=_json_default, **kw)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def config_hash(cfg: Mapping[str, Any], n: int = 12) -> str:
    """Short SHA-256 of the canonical JSON of ``cfg`` (sorted keys); identifies a configuration."""
    return hashlib.sha256(_dumps(cfg, sort_keys=True).encode()).hexdigest()[:n]


# --------------------------------------------------------------------------------------------
# Context: command line, output locations, saving
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class ExpContext:
    """Run context of an experiment script (built by :func:`parse_args`).

    Attributes
    ----------
    exp : str
        Experiment id, e.g. ``"e2"``.
    quick : bool
        Smoke-test mode: everything goes to ``results/<exp>_quick/``.
    workers : int
        Worker processes for :func:`run_cells`.
    parts : list of str
        Sub-experiments selected with ``--part`` (all by default).
    plots_only : bool
        Rebuild figures and tables from saved CSVs, compute nothing.
    fresh : bool
        Ignore (and overwrite) existing checkpoints.
    exp_id : str
        ``exp`` or ``exp + "_quick"``: the sub-directory of ``results/``.
    results_root, results_dir, fig_dir, tab_dir, ckpt_dir : Path
        ``results/``, ``results/<exp_id>/``, figure directory (``paper/figures`` or
        ``results/<exp>_quick/figures``), table directory, checkpoint directory
        (``results/<exp_id>/checkpoints``).
    args : argparse.Namespace
        All parsed arguments (including script-specific ones).
    """

    exp: str
    quick: bool
    workers: int
    parts: list[str]
    plots_only: bool
    fresh: bool
    exp_id: str
    results_root: Path
    results_dir: Path
    fig_dir: Path
    tab_dir: Path
    ckpt_dir: Path
    args: argparse.Namespace

    # -- saving ------------------------------------------------------------------------------
    def save_csv(self, df: pd.DataFrame, name: str, config: Mapping[str, Any] | None = None,
                 timing: Mapping[str, float] | None = None) -> Path:
        """Write ``results/<exp_id>/<name>.csv`` and update ``meta.json`` (utils.save_results)."""
        return utils.save_results(df, name, self.exp_id, config, results_dir=self.results_root,
                                  timing=timing)

    def save_json(self, obj: Mapping[str, Any], name: str,
                  config: Mapping[str, Any] | None = None) -> Path:
        """Write ``results/<exp_id>/<name>.json`` and update ``meta.json``."""
        return utils.save_results(dict(obj), name, self.exp_id, config,
                                  results_dir=self.results_root)

    def load_csv(self, name: str, **kw: Any) -> pd.DataFrame:
        """Read ``results/<exp_id>/<name>.csv`` (raises FileNotFoundError with a hint)."""
        path = self.results_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run the part that produces it first")
        return pd.read_csv(path, **kw)

    def has_csv(self, name: str) -> bool:
        return (self.results_dir / f"{name}.csv").exists()

    def write_meta(self, config: Mapping[str, Any] | None = None,
                   timing: Mapping[str, float] | None = None) -> Path:
        """Create or update ``results/<exp_id>/meta.json`` (utils.write_meta)."""
        return utils.write_meta(self.exp_id, config, results_dir=self.results_root,
                                timing=timing)

    def write_part_meta(self, part: str, config: Mapping[str, Any],
                        timing: Mapping[str, Any] | None = None) -> Path:
        """Write ``results/<exp_id>/meta_<part>.json`` (configuration, timing, provenance).

        ``meta.json`` is rewritten by every process, so when parts run in parallel processes the
        last one wins; this per-part file keeps each part's own record.
        """
        meta = {
            "experiment": self.exp, "part": part, "quick": self.quick,
            "written": _dt.datetime.now().isoformat(timespec="seconds"),
            "config": dict(config), "config_hash": config_hash(config),
            "timing": dict(timing or {}), "workers": self.workers, "argv": sys.argv,
            "git": utils.git_hash(), "code": utils.code_fingerprint(),
            "packages": utils.package_versions(),
            "platform": {"system": platform.system(), "release": platform.release(),
                         "machine": platform.machine(), "cpu_count": os.cpu_count()},
        }
        path = self.results_dir / f"meta_{part}.json"
        _atomic_write_text(path, _dumps(meta, indent=2) + "\n")
        return path

    def savefig(self, fig: Any, name: str) -> Path:
        """Save a figure as ``<fig_dir>/<name>.pdf`` with the paper style and close it."""
        return plotting.savefig(fig, name, fig_dir=self.fig_dir, close=True)

    def save_table(self, df: pd.DataFrame, name: str, **kwargs: Any) -> Path:
        """Write a booktabs table ``<tab_dir>/<name>.tex`` (utils.save_table_tex)."""
        return utils.save_table_tex(df, name, tab_dir=self.tab_dir, **kwargs)

    def checkpoint(self, part: str, cfg: Mapping[str, Any]) -> "Checkpoint":
        """Checkpoint store for ``part`` under the given configuration."""
        return Checkpoint(self.ckpt_dir, part, cfg, fresh=self.fresh)


def parse_args(exp: str, parts: Sequence[str], *, description: str | None = None,
               aliases: Mapping[str, Sequence[str]] | None = None,
               add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
               argv: Sequence[str] | None = None) -> ExpContext:
    """Parse the standard experiment command line and build the :class:`ExpContext`.

    Standard options: ``--quick``, ``--workers N`` (default env ``SYNTHVAL_WORKERS``, see
    :func:`synthval.config.n_workers`), ``--part NAME`` (repeatable or comma-separated; one of
    ``parts``, an alias, or ``all``; default all), ``--plots-only``, ``--fresh``, ``-v``.

    Parameters
    ----------
    exp : str
        Experiment id (``"e2"``).
    parts : sequence of str
        Names of the sub-experiments, in run order.
    description : str, optional
        Help text (e.g. the script's ``__doc__``).
    aliases : mapping, optional
        Extra names that expand to several parts, e.g. ``{"main": ["main_a", "main_b"]}``.
    add_arguments : callable, optional
        ``f(parser)`` adding script-specific options (available as ``ctx.args``).
    argv : sequence of str, optional
        Arguments (default ``sys.argv[1:]``).
    """
    aliases = {k: list(v) for k, v in (aliases or {}).items()}
    parser = argparse.ArgumentParser(
        description=(description or "").strip().split("\n\n")[0] or None,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="smoke test (< 5 min at 3 workers); writes only results/<exp>_quick/")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes (default: env SYNTHVAL_WORKERS, else 4)")
    names = list(parts) + sorted(aliases) + ["all"]
    parser.add_argument("--part", action="append", default=None, metavar="NAME",
                        help=f"sub-experiment(s) to run: {', '.join(names)} (default all; "
                             "repeat or comma-separate)")
    parser.add_argument("--plots-only", action="store_true",
                        help="rebuild figures and tables from the saved CSVs")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore existing checkpoints and recompute")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args(argv)

    selected: list[str] = []
    for item in args.part or ["all"]:
        for name in (s.strip() for s in item.split(",") if s.strip()):
            if name == "all":
                expand = list(parts)
            elif name in aliases:
                expand = aliases[name]
            elif name in parts:
                expand = [name]
            else:
                parser.error(f"unknown part {name!r}; choose from {', '.join(names)}")
            selected += [p for p in expand if p not in selected]
    selected = [p for p in parts if p in selected]  # canonical order

    workers = config.n_workers() if args.workers is None else int(args.workers)
    if workers < 1:
        parser.error("--workers must be >= 1")
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    exp_id = f"{exp}_quick" if args.quick else exp
    results_root = Path(config.RESULTS_DIR)
    results_dir = results_root / exp_id
    if args.quick:
        fig_dir, tab_dir = results_dir / "figures", results_dir / "tables"
    else:
        fig_dir, tab_dir = Path(config.FIG_DIR), Path(config.TAB_DIR)
    ctx = ExpContext(exp=exp, quick=bool(args.quick), workers=workers, parts=selected,
                     plots_only=bool(args.plots_only), fresh=bool(args.fresh), exp_id=exp_id,
                     results_root=results_root, results_dir=results_dir, fig_dir=fig_dir,
                     tab_dir=tab_dir, ckpt_dir=results_dir / "checkpoints", args=args)
    logger.info("%s: parts=%s quick=%s workers=%d plots_only=%s -> %s", exp, selected,
                ctx.quick, workers, ctx.plots_only, results_dir)
    return ctx


# --------------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------------
def key_words(*keys: Any) -> list[int]:
    """Map keys (str, int, float, tuples of those) to 32-bit words via SHA-256 of their repr.

    Stable across processes and Python versions (unlike ``hash``). ``1`` and ``1.0`` give
    different words (their reprs differ): use one type consistently for a given key.
    """
    words: list[int] = []
    for k in keys:
        d = hashlib.sha256(repr(k).encode()).digest()
        words += [int.from_bytes(d[0:4], "little"), int.from_bytes(d[4:8], "little")]
    return words


def keyed_seed_sequence(root_seed: int, *keys: Any) -> np.random.SeedSequence:
    """``SeedSequence`` determined by the root seed and the keys, e.g. ``(part, cell, rep)``."""
    return np.random.SeedSequence([int(root_seed), *key_words(*keys)])


def keyed_rng(root_seed: int, *keys: Any) -> np.random.Generator:
    """Generator determined by ``(root_seed, *keys)``; independent streams for distinct keys."""
    return np.random.default_rng(keyed_seed_sequence(root_seed, *keys))


def cell_key(**params: Any) -> str:
    """Readable, filesystem-safe key of a cell, e.g. ``cell_key(fam="a", n=500) -> "fam=a_n=500"``."""
    parts = []
    for k, v in params.items():
        s = f"{v:g}" if isinstance(v, float) else str(v)
        parts.append(f"{k}={s}")
    out = "_".join(parts)
    return "".join(c if (c.isalnum() or c in "=_-.+") else "-" for c in out)


# --------------------------------------------------------------------------------------------
# Checkpoints and the replication runner
# --------------------------------------------------------------------------------------------
class Checkpoint:
    """Block files of one part under ``<ckpt_dir>/<part>/<config hash>/``.

    A block is a CSV with the rows of a contiguous range of replications of one cell, written
    atomically by the worker that computed it. ``config.json`` in the same directory records the
    configuration whose hash names the directory.
    """

    def __init__(self, ckpt_dir: Path, part: str, cfg: Mapping[str, Any], *,
                 fresh: bool = False) -> None:
        self.part = part
        self.cfg = dict(cfg)
        self.hash = config_hash(self.cfg)
        self.dir = Path(ckpt_dir) / part / self.hash
        self.fresh = bool(fresh)
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.dir / "config.json"
        if not cfg_path.exists():
            _atomic_write_text(cfg_path, _dumps(self.cfg, indent=2, sort_keys=True) + "\n")

    def block_path(self, cell: str, start: int, stop: int) -> Path:
        return self.dir / f"{cell}__r{start:05d}-{stop:05d}.csv"

    def file(self, name: str) -> Path:
        """Path of an auxiliary checkpoint file (e.g. a Monte Carlo null distribution)."""
        return self.dir / name

    def done(self, path: Path) -> bool:
        return (not self.fresh) and path.exists()


def _blocks(n_reps: int, block_size: int) -> list[tuple[int, int]]:
    return [(s, min(n_reps, s + block_size)) for s in range(0, n_reps, block_size)]


def _run_block(item: tuple) -> dict[str, Any]:
    """Worker: run replications ``reps`` of one cell, write the block CSV, return timing."""
    fn, root_seed, part, cell, start, stop, path = item
    rows: list[dict[str, Any]] = []
    t_block = time.perf_counter()
    for rep in range(start, stop):
        rng = keyed_rng(root_seed, part, cell["cell"], rep)
        t0 = time.perf_counter()
        out = fn(cell, rep, rng)
        dt = time.perf_counter() - t0
        out_rows = [out] if isinstance(out, Mapping) else list(out)
        for r in out_rows:
            row = {k: v for k, v in cell.items()
                   if not k.startswith("_") and not isinstance(v, (list, dict, tuple))}
            row["rep"] = rep
            row["rep_seconds"] = dt
            row.update(r)
            rows.append(row)
    df = pd.DataFrame(rows)
    _atomic_write_text(Path(path), df.to_csv(index=False))
    return {"path": str(path), "cell": cell["cell"], "n_reps": stop - start,
            "seconds": time.perf_counter() - t_block}


def run_cells(fn: Callable[[dict, int, np.random.Generator], Any], cells: Sequence[dict], *,
              part: str, ctx: ExpContext, root_seed: int,
              n_reps: int | Callable[[dict], int], config: Mapping[str, Any],
              block_size: int = 10, cost: Callable[[dict], float] | None = None,
              workers: int | None = None, fresh: bool | None = None) -> pd.DataFrame:
    """Run ``fn(cell, rep, rng)`` for every cell and replication, with block checkpoints.

    Parameters
    ----------
    fn : callable
        Module-level function ``fn(cell, rep, rng) -> dict | list[dict]`` returning result
        rows for one replication. ``rng`` is ``keyed_rng(root_seed, part, cell["cell"], rep)``;
        derive further streams from it with ``rng.spawn`` or :func:`keyed_rng` with extra keys.
    cells : sequence of dict
        Cell parameters; each needs a unique string ``"cell"`` (see :func:`cell_key`). Scalar
        entries are copied into every result row, except keys starting with ``"_"`` (use those
        for configuration passed to ``fn``, e.g. ``"_cfg"``, or for paths).
    part : str
        Part name (checkpoint sub-directory and seed key).
    ctx : ExpContext
        Run context (checkpoint location, ``--fresh``, workers).
    root_seed : int
        Root seed of the experiment.
    n_reps : int or callable
        Replications per cell (or ``n_reps(cell)``).
    config : mapping
        Everything that determines the results apart from ``cells`` and ``n_reps`` (test
        settings, model parameters, ``block_size`` is added automatically); hashed to name the
        checkpoint directory.
    block_size : int
        Replications per block (unit of checkpointing and of parallel work).
    cost : callable, optional
        Relative cost of one replication of a cell; blocks are submitted most expensive first.
    workers : int, optional
        Default ``ctx.workers``.
    fresh : bool, optional
        Override ``ctx.fresh`` (``True``: recompute and overwrite every block, e.g. for timing
        probes).

    Returns
    -------
    DataFrame
        All rows of all requested cells and replications (from the block files), with the
        cell's scalar parameters, ``rep`` and ``rep_seconds`` (wall seconds of the whole
        replication in its worker).
    """
    cfg = dict(config)
    cfg["block_size"] = int(block_size)
    ck = Checkpoint(ctx.ckpt_dir, part, cfg, fresh=ctx.fresh if fresh is None else fresh)
    keys = [c["cell"] for c in cells]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate cell keys in part {part}")
    todo: list[tuple[float, tuple]] = []
    paths: list[Path] = []
    for cell in cells:
        n = int(n_reps(cell) if callable(n_reps) else n_reps)
        c = cost(cell) if cost is not None else 1.0
        for start, stop in _blocks(n, block_size):
            path = ck.block_path(cell["cell"], start, stop)
            paths.append(path)
            if not ck.done(path):
                todo.append((c * (stop - start),
                             (fn, int(root_seed), part, dict(cell), start, stop, str(path))))
    todo.sort(key=lambda t: -t[0])
    n_done = len(paths) - len(todo)
    logger.info("[%s] %d cells, %d blocks (%d from checkpoints, %d to run) in %s", part,
                len(cells), len(paths), n_done, len(todo), ck.dir)
    if todo:
        t0 = time.perf_counter()
        info = utils.parallel_map(_run_block, [t[1] for t in todo],
                                  n_workers=workers or ctx.workers)
        logger.info("[%s] ran %d blocks in %s (sum of block times %s)", part, len(info),
                    fmt_seconds(time.perf_counter() - t0),
                    fmt_seconds(sum(i["seconds"] for i in info)))
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_item(item: tuple) -> Any:
    fn, root_seed, part, key, payload = item
    return fn(payload, keyed_rng(root_seed, part, key))


def run_items(fn: Callable[[Any, np.random.Generator], Any], items: Sequence[tuple[str, Any]], *,
              part: str, root_seed: int, workers: int) -> list[Any]:
    """Parallel map of ``fn(payload, rng)`` over keyed items ``(key, payload)`` (no checkpoint).

    ``rng = keyed_rng(root_seed, part, key)``. Use for auxiliary computations whose results the
    caller checkpoints itself (e.g. Monte Carlo null distributions computed in chunks).
    """
    return utils.parallel_map(_run_item, [(fn, int(root_seed), part, k, p) for k, p in items],
                              n_workers=workers)


def safe_test(test: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float, str]:
    """Call a test, returning ``(result or None, seconds, error message or "")``.

    A rare numerical failure in one replication then costs one NaN row instead of the run.
    """
    t0 = time.perf_counter()
    try:
        res = test(*args, **kwargs)
        return res, time.perf_counter() - t0, ""
    except Exception as exc:  # noqa: BLE001 - recorded in the results, logged
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning("test %s failed: %s\n%s", getattr(test, "name", test), msg,
                       traceback.format_exc(limit=3))
        return None, time.perf_counter() - t0, msg


# --------------------------------------------------------------------------------------------
# Rejection rates and Wilson intervals
# --------------------------------------------------------------------------------------------
def wilson_ci(k: Any, n: Any, conf: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion ``k / n`` (vectorised).

    Wilson (1927); recommended over the Wald interval by Brown, Cai and DasGupta (2001). Returns
    ``(lo, hi)``; NaN where ``n == 0``.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = stats.norm.ppf(0.5 + conf / 2.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        den = 1.0 + z * z / n
        centre = (p + z * z / (2.0 * n)) / den
        half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
        lo = np.where(k <= 0, 0.0, np.clip(centre - half, 0.0, 1.0))  # exact ends (round-off)
        hi = np.where(k >= n, 1.0, np.clip(centre + half, 0.0, 1.0))
        lo = np.where(n > 0, lo, np.nan)
        hi = np.where(n > 0, hi, np.nan)
    return lo, hi


def rejection_table(df: pd.DataFrame, by: Sequence[str], *, pcol: str = "pvalue",
                    alphas: Sequence[float] = (0.01, 0.05, 0.10),
                    conf: float = 0.95) -> pd.DataFrame:
    """Rejection rates ``P(p <= alpha)`` with Wilson CIs, per group.

    Parameters
    ----------
    df : DataFrame
        One row per (replication, test, ...), with the p-value in ``pcol``.
    by : sequence of str
        Grouping columns (e.g. ``["family", "n", "test", "ptype"]``).
    pcol : str
        p-value column. NaN p-values (failed tests) are excluded and counted in ``n_nan``.
    alphas : sequence of float
        Nominal levels.
    conf : float
        Confidence level of the Wilson intervals.

    Returns
    -------
    DataFrame
        Columns ``by`` + ``n_valid`` (valid p-values), ``n_nan``, and per alpha ``rej_<a>``,
        ``lo_<a>``, ``hi_<a>`` (``<a>`` formatted with :func:`_alpha_tag`, e.g. ``0.05``).
    """
    rows = []
    for key, g in df.groupby(list(by), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        p = g[pcol].to_numpy(dtype=float)
        ok = np.isfinite(p)
        n = int(ok.sum())
        row = dict(zip(by, key))
        row.update(n_valid=n, n_nan=int((~ok).sum()))
        for a in alphas:
            k = int(np.sum(p[ok] <= a))
            lo, hi = wilson_ci(k, n, conf)
            tag = _alpha_tag(a)
            row[f"rej_{tag}"] = k / n if n else float("nan")
            row[f"lo_{tag}"] = float(lo)
            row[f"hi_{tag}"] = float(hi)
        rows.append(row)
    return pd.DataFrame(rows)


def _alpha_tag(a: float) -> str:
    return f"{a:.2f}" if round(a, 2) == a else f"{a:g}"


def fmt_rate_ci(rate: float, lo: float, hi: float, digits: int = 3) -> str:
    """``"0.050 [0.026, 0.094]"``; ``"--"`` if the rate is NaN."""
    if not np.isfinite(rate):
        return "--"
    return f"{rate:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def rate_ci_table(tbl: pd.DataFrame, index: Sequence[str],
                  alphas: Sequence[float] = (0.01, 0.05, 0.10), *, digits: int = 3,
                  label: Callable[[float], str] | None = None) -> pd.DataFrame:
    """String table ``rate [lo, hi]`` per alpha from :func:`rejection_table` output.

    ``label(alpha)`` names the columns (default ``"alpha=0.05"``); the result is indexed by
    ``index`` and ready for :meth:`ExpContext.save_table`.
    """
    label = label or (lambda a: f"alpha={_alpha_tag(a)}")
    out = tbl[list(index)].copy()
    for a in alphas:
        t = _alpha_tag(a)
        out[label(a)] = [fmt_rate_ci(r, lo, hi, digits)
                         for r, lo, hi in zip(tbl[f"rej_{t}"], tbl[f"lo_{t}"], tbl[f"hi_{t}"])]
    return out.set_index(list(index))


# --------------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------------
class Stopwatch:
    """Named wall-clock laps: ``sw = Stopwatch(); ...; sw.lap("main")``; ``sw.laps`` (seconds)."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self._last = self.t0
        self.laps: dict[str, float] = {}

    def lap(self, name: str) -> float:
        now = time.perf_counter()
        self.laps[name] = self.laps.get(name, 0.0) + (now - self._last)
        self._last = now
        return self.laps[name]

    def total(self) -> float:
        return time.perf_counter() - self.t0


def fmt_seconds(s: float) -> str:
    """``"42 s"``, ``"7.3 min"`` or ``"2.15 h"``."""
    if not np.isfinite(s):
        return "nan"
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def estimate_wall(task_seconds: Iterable[float], workers: int = 12,
                  overhead_s: float = 15.0) -> float:
    """Wall time of independent tasks on ``workers`` processes (LPT greedy schedule).

    Tasks are assigned longest first to the least-loaded worker; ``overhead_s`` adds pool
    start-up and data loading. Per-task times measured while other jobs share the machine are
    upper-biased, never lower.
    """
    loads = [0.0] * max(1, int(workers))
    for t in sorted((float(x) for x in task_seconds), reverse=True):
        i = int(np.argmin(loads))
        loads[i] += t
    return max(loads) + overhead_s


def print_rep_timing(label: str, per_rep_s: float, *, note: str = "") -> None:
    """Print ``TIMING <label>: <s> s per replication`` (one line, flushed)."""
    extra = f" ({note})" if note else ""
    print(f"TIMING {label}: {per_rep_s:.2f} s per replication{extra}", flush=True)


def print_wall_estimate(label: str, task_seconds: Iterable[float], *, workers: int = 12,
                        extra_s: float = 0.0, overhead_s: float = 15.0) -> float:
    """Print ``ESTIMATED FULL-RUN WALL TIME <label> at <workers> workers: ...`` and return seconds.

    ``task_seconds`` are the durations of the independent tasks (e.g. checkpoint blocks) of the
    full run; ``extra_s`` adds work done before them (e.g. a Monte Carlo null computed first).
    See :func:`estimate_wall`.
    """
    tasks = list(task_seconds)
    wall = estimate_wall(tasks, workers, overhead_s) + float(extra_s)
    cpu = sum(tasks)
    print(f"ESTIMATED FULL-RUN WALL TIME {label} at {workers} workers: {fmt_seconds(wall)} "
          f"({wall / 3600:.2f} h; {fmt_seconds(cpu)} CPU in {len(tasks)} tasks"
          + (f" + {fmt_seconds(extra_s)} precompute" if extra_s else "") + ")",
          flush=True)
    return wall


def block_tasks(per_rep_s: float, n_reps: int, block_size: int) -> list[float]:
    """Durations of the checkpoint blocks of one cell (``n_reps`` reps of ``per_rep_s`` s)."""
    return [per_rep_s * (stop - start) for start, stop in _blocks(int(n_reps), int(block_size))]
