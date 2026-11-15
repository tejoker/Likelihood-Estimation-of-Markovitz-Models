"""Seeding, process-parallel map, and saving results with metadata (DESIGN section 3.11).

Reproducibility contract
------------------------
Randomness flows only through ``np.random.Generator`` objects. To run ``n`` independent
replications, draw ``rngs = spawn_rngs(seed, n)`` once and give replication ``i`` the generator
``rngs[i]`` (e.g. ``parallel_map(fn, list(zip(params, rngs)))``). The result of replication ``i``
then depends only on ``(seed, i)``: it is identical serially and in parallel, for any number of
workers.

``parallel_map`` passes items **by value** in both modes (a pickled copy), so the caller's
generators are never advanced. Reusing the same ``rngs`` list in two calls therefore repeats the
same draws (common random numbers); for independent calls, spawn fresh generators, e.g.
``spawn_rngs(np.random.SeedSequence([seed, k]), n)`` for the ``k``-th call.
"""
from __future__ import annotations

import concurrent.futures as cf
from concurrent.futures.process import BrokenProcessPool
import datetime as _dt
import functools
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import platform
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import metadata as _metadata
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from . import config
from . import config as _cfg  # alias: save_results has a parameter named `config`

logger = logging.getLogger(__name__)

__all__ = [
    "seed_everything",
    "spawn_rngs",
    "parallel_map",
    "git_hash",
    "code_fingerprint",
    "file_sha256",
    "package_versions",
    "save_results",
    "write_meta",
    "to_latex_booktabs",
    "save_table_tex",
]

T = TypeVar("T")
R = TypeVar("R")

_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "torch", "arch",
             "scikit-learn")
_RUN_START = time.time()
_RUN_ID = f"{_dt.datetime.fromtimestamp(_RUN_START).strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"


# --------------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------------
def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python's ``random``, NumPy's legacy global state and torch; return a Generator.

    Library code must not rely on global state (every stochastic function takes ``rng``); this
    is a safety net for third-party code. ``PYTHONHASHSEED`` is set for child processes only.

    Parameters
    ----------
    seed : int
        Non-negative seed.

    Returns
    -------
    np.random.Generator
        ``np.random.default_rng(seed)``.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 - deliberate seeding of the legacy global state
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is a dependency
        pass
    return np.random.default_rng(seed)


def spawn_rngs(seed: int | np.random.SeedSequence | np.random.Generator,
               n: int) -> list[np.random.Generator]:
    """``n`` statistically independent generators from one seed (``SeedSequence.spawn``).

    The list is prefix-stable: ``spawn_rngs(s, 10)[:5]`` produces the same streams as
    ``spawn_rngs(s, 5)``.

    Parameters
    ----------
    seed : int, SeedSequence or Generator
        Root entropy. A Generator is spawned from (this advances its spawn counter).
    n : int
        Number of generators.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if isinstance(seed, np.random.Generator):
        return list(seed.spawn(n))
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(int(seed))
    return [np.random.default_rng(child) for child in ss.spawn(n)]


# --------------------------------------------------------------------------------------------
# Parallel map
# --------------------------------------------------------------------------------------------
_WORKER_THREAD_LIMITS: Any = None  # keeps the threadpoolctl limiter of a worker alive


def _worker_init(use_torch: bool) -> None:
    """Pool initializer: one BLAS/OpenMP/torch thread per worker process.

    The thread variables are already set in the environment the worker inherits, which is
    enough for a spawned worker (its BLAS starts after them). A forked worker inherits the
    parent's BLAS thread pools, already sized from the parent's environment (16 OpenBLAS
    threads on the development machine, which made Merton fits 10 times slower in forked
    workers): their size is set to 1 at run time with ``threadpoolctl`` when it is installed
    (a dependency of scikit-learn).
    """
    global _WORKER_THREAD_LIMITS
    for k in _THREAD_ENV:
        os.environ[k] = "1"
    try:
        from threadpoolctl import threadpool_limits

        _WORKER_THREAD_LIMITS = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - optional; spawned workers are covered by env vars
        pass
    except Exception as exc:  # pragma: no cover - never let thread limiting kill a worker
        logger.debug("threadpoolctl could not limit threads: %s", exc)
    if use_torch or "torch" in sys.modules:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:  # already set, or inter-op pool already started
            pass


def _call(fn: Callable[[T], R], item: T) -> R:
    """Worker-side trampoline: run ``fn(item)`` and make sure any exception reaches the parent.

    An exception that does not survive a pickle round trip (e.g. a custom exception whose
    ``__init__`` takes several arguments) would break the process pool and lose the message;
    it is replaced by a ``RuntimeError`` carrying its type, message and worker traceback.
    """
    try:
        return fn(item)
    except Exception as exc:
        try:
            pickle.loads(pickle.dumps(exc, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            raise RuntimeError(
                f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}\n"
                "(the original exception cannot be pickled back to the parent process)\n\n"
                f"Worker traceback:\n{traceback.format_exc()}") from None
        raise


def _by_value(item: T) -> T:
    """Pickled copy of ``item``: the serial path sees exactly what a worker process would."""
    return pickle.loads(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL))


def _annotate(exc: BaseException, index: int) -> None:
    exc.add_note(f"parallel_map: raised while processing item {index}")
    try:
        exc.item_index = index  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - exotic exception types
        pass


def parallel_map(fn: Callable[[T], R], items: Iterable[T], n_workers: int | None = None, *,
                 mp_context: str | None = None) -> list[R]:
    """Apply ``fn`` to every item, in worker processes, returning results in input order.

    Parameters
    ----------
    fn : callable
        Picklable (module-level) function of one argument. Scripts that call this must guard
        their entry point with ``if __name__ == "__main__":`` (workers re-import ``__main__``).
    items : iterable
        Arguments; materialised into a list. Pass ``(params, rng)`` tuples for stochastic work.
        Every item must be picklable: it is passed **by value** (a pickled copy) in both serial
        and parallel mode, so generators inside items are never advanced in the caller and the
        two modes give identical results even when an ``rngs`` list is reused. Reuse therefore
        repeats the same random streams (common random numbers); spawn new generators for
        independent calls.
    n_workers : int, optional
        Number of processes; default ``config.n_workers()`` (env ``SYNTHVAL_WORKERS``, 4).
        ``1`` (or a single item) runs serially in the calling process, with no pool.
    mp_context : str, optional
        Multiprocessing start method; default env ``SYNTHVAL_MP_START`` or ``"spawn"``.
        ``"spawn"`` gives fresh interpreters (about 2 s pool start-up when torch is imported),
        safe whatever torch/OpenMP state the parent holds; ``"fork"`` starts in ~0.1 s but can
        deadlock if the parent already used threaded libraries. (``"forkserver"`` needs Unix
        sockets, which the sandbox used for development blocks.)

    Returns
    -------
    list
        ``[fn(item) for item in items]``.

    Raises
    ------
    Exception
        The first exception raised by ``fn`` (original type, with the remote traceback chained
        as ``__cause__``), annotated with a note naming the item index and with attribute
        ``item_index``. Pending items are cancelled. In parallel mode an exception that cannot
        be pickled back to the parent arrives as a ``RuntimeError`` whose message holds the
        original type, message and worker traceback.
    RuntimeError
        If the pool breaks: a worker died (e.g. out of memory) or a result could not be
        unpickled in the parent.

    Notes
    -----
    Workers get ``OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_*_NUM_THREADS = 1`` in their environment from
    the start (so libraries imported while re-importing ``__main__`` already use one thread),
    and the initializer calls ``torch.set_num_threads(1)`` whenever torch is in use. The calling
    process' environment is restored on exit.
    """
    items = list(items)
    n_workers = config.n_workers() if n_workers is None else int(n_workers)
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if n_workers == 1 or len(items) <= 1:
        out: list[R] = []
        for i, item in enumerate(items):
            try:
                out.append(fn(_by_value(item)))
            except Exception as exc:
                _annotate(exc, i)
                raise
        return out

    n_proc = min(n_workers, len(items))
    ctx = mp.get_context(mp_context or os.environ.get("SYNTHVAL_MP_START", "spawn"))
    saved = {k: os.environ.get(k) for k in _THREAD_ENV}
    for k in _THREAD_ENV:
        os.environ[k] = "1"
    results: list[Any] = [None] * len(items)
    try:
        with cf.ProcessPoolExecutor(max_workers=n_proc, mp_context=ctx,
                                    initializer=_worker_init,
                                    initargs=("torch" in sys.modules,)) as ex:
            futures = {ex.submit(_call, fn, item): i for i, item in enumerate(items)}
            try:
                for fut in cf.as_completed(futures):
                    i = futures[fut]
                    try:
                        results[i] = fut.result()
                    except BrokenProcessPool as exc:
                        raise RuntimeError(
                            "parallel_map: the process pool broke (a worker died, e.g. out "
                            "of memory, or a result could not be unpickled in the parent); "
                            f"first affected item {i}") from exc
                    except Exception as exc:
                        _annotate(exc, i)
                        raise
            except BaseException:
                ex.shutdown(wait=True, cancel_futures=True)
                raise
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return results


# --------------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------------
_DIRTY_PATHS = ("src", "experiments")
_MAX_LISTED = 50


def git_hash(root: Path = config.ROOT, paths: Sequence[str] = _DIRTY_PATHS) -> dict[str, Any]:
    """Current commit and whether the code that produces results differs from it.

    Parameters
    ----------
    root : Path
        Repository root.
    paths : sequence of str
        Pathspecs whose state matters (default ``src`` and ``experiments``). A modified, staged,
        deleted **or untracked** file under them makes the tree dirty; files elsewhere (e.g.
        editor or shell files in the repository root) do not. Read-only: ``git status`` runs
        with ``--no-optional-locks``, so concurrent runs never take the index lock.

    Returns
    -------
    dict
        ``commit`` (40-hex SHA or ``None``), ``dirty`` (bool or ``None`` if git is unavailable),
        ``dirty_paths`` (the pathspecs), ``n_dirty_files`` and ``dirty_files`` (``git status
        --porcelain`` lines, at most 50).
    """
    info: dict[str, Any] = {"commit": None, "dirty": None, "dirty_paths": list(paths),
                            "n_dirty_files": None, "dirty_files": []}
    try:
        info["commit"] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, check=True).stdout.strip() or None
        status = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), "status", "--porcelain",
             "--untracked-files=all", "--", *paths],
            capture_output=True, text=True, timeout=10, check=True).stdout
        lines = [ln for ln in status.splitlines() if ln.strip()]
        info["dirty"] = bool(lines)
        info["n_dirty_files"] = len(lines)
        info["dirty_files"] = lines[:_MAX_LISTED]
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git metadata unavailable: %s", exc)
    return info


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes (read in 1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(functools.partial(fh.read, 1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=8)
def _file_sha256_cached(path: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size  # cache key only
    return file_sha256(path)


def code_fingerprint(package_dir: Path | None = None) -> dict[str, Any]:
    """Content hash of the library source, tying results to code even before a commit.

    The digest is the SHA-256 of the sequence ``(relative path, NUL, SHA-256 of contents, LF)``
    over every ``*.py`` file below ``package_dir`` in sorted path order, so it changes when any
    file is edited, added, removed or renamed, and does not depend on timestamps.

    Parameters
    ----------
    package_dir : Path, optional
        Default: the installed ``synthval`` package directory.

    Returns
    -------
    dict
        ``{"sha256": str, "n_files": int, "dir": str}``.
    """
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    h = hashlib.sha256()
    files = sorted(f for f in root.rglob("*.py") if "__pycache__" not in f.parts)
    for f in files:
        h.update(f.relative_to(root).as_posix().encode())
        h.update(b"\0" + bytes.fromhex(file_sha256(f)) + b"\n")
    return {"sha256": h.hexdigest(), "n_files": len(files), "dir": _rel(root)}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)


def _script_info() -> dict[str, str] | None:
    """Path and SHA-256 of the running script (``sys.argv[0]``), if it is a ``.py`` file."""
    if not sys.argv or not sys.argv[0]:
        return None
    p = Path(sys.argv[0])
    if p.suffix != ".py" or not p.is_file():
        return None
    return {"path": _rel(p), "sha256": file_sha256(p)}


def _data_info(path: Path = config.DATA_CSV) -> dict[str, Any] | None:
    """Path, size and SHA-256 of the raw data file (hash cached on path, mtime and size)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {"path": _rel(Path(path)), "bytes": st.st_size,
            "sha256": _file_sha256_cached(str(Path(path).resolve()), st.st_mtime_ns, st.st_size)}


def package_versions(packages: Sequence[str] = _PACKAGES) -> dict[str, str | None]:
    """Installed versions of the main dependencies (without importing them) and Python."""
    out: dict[str, str | None] = {"python": platform.python_version()}
    for p in packages:
        try:
            out[p] = _metadata.version(p)
        except _metadata.PackageNotFoundError:
            out[p] = None
    return out


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas/pathlib objects to JSON-serialisable Python objects."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="list"))
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def _current_meta(path: Path) -> dict[str, Any]:
    """``meta.json`` contents if written by the current run, else ``{}``."""
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return old if isinstance(old, dict) and old.get("run_id") == _RUN_ID else {}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_meta(exp: str, config_dict: Mapping[str, Any] | None = None, *,
               results_dir: Path | None = None, timing: Mapping[str, float] | None = None,
               files: Mapping[str, Mapping[str, Any]] | None = None) -> Path:
    """Create or update ``results/<exp>/meta.json`` for the current run.

    Entries written earlier by the same process (same run id) are merged; a ``meta.json`` left
    by a previous run is replaced. Recorded: experiment name, run id, start time, elapsed wall
    time since ``synthval.utils`` was imported, the run configuration, provenance (git commit
    and dirty flag over ``src``/``experiments`` including untracked files, a content hash of
    the ``synthval`` sources, path and hash of the running script, size and hash of the raw
    data file), package versions, platform, ``SYNTHVAL_WORKERS``, user timings and the files
    written so far (each with its own configuration, see :func:`save_results`).

    Parameters
    ----------
    exp : str
        Experiment id.
    config_dict : dict, optional
        Run-level configuration, stored as ``meta["config"]``. Replacing a different run-level
        configuration of the same run logs a warning.
    results_dir : Path, optional
        Root directory (default ``config.RESULTS_DIR``).
    timing : dict, optional
        Named durations in seconds, merged into ``meta["timing"]``.
    files : dict, optional
        ``{file name: info}`` merged into ``meta["files"]``.

    Returns
    -------
    Path
        Path of ``meta.json``.
    """
    out_dir = Path(results_dir or config.RESULTS_DIR) / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "meta.json"
    meta = _current_meta(path)
    meta.update({
        "experiment": exp,
        "run_id": _RUN_ID,
        "started": _dt.datetime.fromtimestamp(_RUN_START).isoformat(timespec="seconds"),
        "updated": _dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - _RUN_START, 3),
        "git": git_hash(),
        "code": code_fingerprint(),
        "script": _script_info(),
        "data": _data_info(),
        "packages": package_versions(),
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "cpu_count": os.cpu_count()},
        "workers": config.n_workers(),
        "argv": sys.argv,
    })
    if config_dict is not None:
        new_cfg = _jsonable(dict(config_dict))
        if "config" in meta and meta["config"] != new_cfg:
            logger.warning("write_meta(%s): replacing a different run-level config", exp)
        meta["config"] = new_cfg
    if timing:
        meta.setdefault("timing", {}).update(_jsonable(dict(timing)))
    if files:
        meta.setdefault("files", {}).update(_jsonable(dict(files)))
    _atomic_write(path, json.dumps(meta, indent=2, sort_keys=False) + "\n")
    return path


def save_results(obj: pd.DataFrame | Mapping[str, Any], name: str, exp: str,
                 config: Mapping[str, Any] | None = None, *, results_dir: Path | None = None,
                 timing: Mapping[str, float] | None = None) -> Path:
    """Write a result table or dict to ``results/<exp>/`` and update ``meta.json``.

    Parameters
    ----------
    obj : DataFrame or dict
        A DataFrame is written to ``<name>.csv`` (index included unless it is a default
        RangeIndex; floats in shortest round-trip form). A dict is written to ``<name>.json`` (numpy
        and pandas objects converted; NaN written as the ``NaN`` token, readable by Python).
    name : str
        File stem (a ``.csv``/``.json`` suffix is ignored).
    exp : str
        Experiment id, e.g. ``"e2"``.
    config : dict, optional
        Configuration that produced this file, stored as ``meta["files"][<file>]["config"]``.
        It also becomes the run-level ``meta["config"]`` if none is recorded yet; a different
        configuration in a later call never overwrites it (each file keeps its own).
    results_dir : Path, optional
        Root directory (default ``config.RESULTS_DIR``).
    timing : dict, optional
        Named durations in seconds, merged into ``meta.json["timing"]``.

    Returns
    -------
    Path
        Path of the written data file.
    """
    out_dir = Path(results_dir or _cfg.RESULTS_DIR) / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem if Path(name).suffix in (".csv", ".json") else name
    if isinstance(obj, pd.DataFrame):
        path = out_dir / f"{stem}.csv"
        keep_index = not (isinstance(obj.index, pd.RangeIndex) and obj.index.name is None)
        _atomic_write(path, obj.to_csv(index=keep_index))
        info = {"kind": "csv", "rows": int(obj.shape[0]), "cols": int(obj.shape[1])}
    elif isinstance(obj, Mapping):
        path = out_dir / f"{stem}.json"
        _atomic_write(path, json.dumps(_jsonable(obj), indent=2) + "\n")
        info = {"kind": "json", "keys": len(obj)}
    else:
        raise TypeError(f"save_results expects a DataFrame or a dict, got {type(obj).__name__}")
    info["written"] = _dt.datetime.now().isoformat(timespec="seconds")
    run_cfg = None
    if config is not None:
        info["config"] = _jsonable(dict(config))
        if "config" not in _current_meta(out_dir / "meta.json"):
            run_cfg = config
    write_meta(exp, run_cfg, results_dir=results_dir, timing=timing, files={path.name: info})
    logger.info("saved %s", path)
    return path


# --------------------------------------------------------------------------------------------
# LaTeX tables
# --------------------------------------------------------------------------------------------
_TEX_ESCAPE = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}"}


def _tex_escape(s: str) -> str:
    return "".join(_TEX_ESCAPE.get(c, c) for c in s)


def _fmt_number(v: Any, digits: int, na_rep: str, math_minus: bool) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)) or v is pd.NA:
        return na_rep
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        s = str(int(v))
    elif isinstance(v, (float, np.floating)):
        if math.isinf(v):
            return r"$\infty$" if v > 0 else r"$-\infty$"
        s = f"{float(v):.{digits}f}"
        if float(s) == 0.0:
            s = s.lstrip("-")  # no "-0.000"
    else:
        return str(v)
    if math_minus and s.startswith("-"):
        s = "$-$" + s[1:]
    return s


def to_latex_booktabs(df: pd.DataFrame, *, digits: int | Mapping[Any, int] = 3,
                      index: bool = True, escape: bool = True, na_rep: str = "--",
                      math_minus: bool = True, column_format: str | None = None,
                      caption: str | None = None, label: str | None = None) -> str:
    """Render a DataFrame as a booktabs ``tabular`` (optionally inside a ``table`` float).

    Parameters
    ----------
    df : DataFrame
        Single- or two-level columns; any number of index levels (repeated consecutive index
        labels are blanked for readability).
    digits : int or dict
        Decimals for float cells, globally or per column label (missing labels use 3).
        Integer cells are printed without decimals; NaN as ``na_rep``; ``-0.000`` as ``0.000``.
    index : bool
        Print the index as the leading column(s).
    escape : bool
        Escape LaTeX special characters in labels and string cells (set False to pass raw
        LaTeX such as ``$\\alpha$``).
    math_minus : bool
        Typeset negative numbers with a math minus (``$-$0.123``).
    column_format : str, optional
        Default ``l`` per index level and ``r`` per data column.
    caption, label : str, optional
        If either is given, wrap in ``\\begin{table}[t]\\centering ... \\end{table}``.

    Returns
    -------
    str
        LaTeX source (requires ``\\usepackage{booktabs}``).
    """
    esc = _tex_escape if escape else str

    def digits_for(col: Any) -> int:
        if isinstance(digits, Mapping):
            return int(digits.get(col, 3))
        return int(digits)

    n_idx = df.index.nlevels if index else 0
    two_level = isinstance(df.columns, pd.MultiIndex)
    if two_level and df.columns.nlevels != 2:
        raise NotImplementedError("at most two column levels are supported")
    colfmt = column_format or ("l" * n_idx + "r" * df.shape[1])

    lines = [rf"\begin{{tabular}}{{{colfmt}}}", r"\toprule"]
    idx_names = [esc("" if n is None else str(n)) for n in df.index.names] if index else []
    if two_level:
        top, cmid = [""] * n_idx, []
        lvl0 = list(df.columns.get_level_values(0))
        j = 0
        while j < len(lvl0):
            k = j
            while k + 1 < len(lvl0) and lvl0[k + 1] == lvl0[j]:
                k += 1
            span = k - j + 1
            top.append(rf"\multicolumn{{{span}}}{{c}}{{{esc(str(lvl0[j]))}}}")
            cmid.append(rf"\cmidrule(lr){{{n_idx + j + 1}-{n_idx + k + 1}}}")
            j = k + 1
        lines.append(" & ".join(top) + r" \\")
        lines.append(" ".join(cmid))
        sub = [esc(str(c)) for c in df.columns.get_level_values(1)]
        lines.append(" & ".join(idx_names + sub) + r" \\")
    else:
        lines.append(" & ".join(idx_names + [esc(str(c)) for c in df.columns]) + r" \\")
    lines.append(r"\midrule")

    prev: tuple = ()
    for key, row in zip(df.index, df.itertuples(index=False, name=None)):
        cells: list[str] = []
        if index:
            labels = key if isinstance(key, tuple) else (key,)
            shown = []
            for lvl, lab in enumerate(labels):
                same_prefix = len(prev) == len(labels) and prev[: lvl + 1] == labels[: lvl + 1]
                shown.append("" if same_prefix and lvl < len(labels) - 1 else esc(str(lab)))
            prev = labels
            cells.extend(shown)
        for col, v in zip(df.columns, row):
            if isinstance(v, str):
                cells.append(esc(v))
            else:
                cells.append(_fmt_number(v, digits_for(col), na_rep, math_minus))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    if caption is None and label is None:
        return body + "\n"
    head = [r"\begin{table}[t]", r"\centering"]
    if caption is not None:
        head.append(rf"\caption{{{caption}}}")
    if label is not None:
        head.append(rf"\label{{{label}}}")
    return "\n".join(head + [body, r"\end{table}"]) + "\n"


def save_table_tex(df: pd.DataFrame, name: str, *, tab_dir: Path | None = None,
                   **kwargs: Any) -> Path:
    """Write :func:`to_latex_booktabs` output to ``paper/tables/<name>.tex``.

    Parameters
    ----------
    df : DataFrame
        Table to render.
    name : str
        File stem, e.g. ``"e2_size"`` (a ``.tex`` suffix is ignored).
    tab_dir : Path, optional
        Output directory (default ``config.TAB_DIR``).
    **kwargs
        Passed to :func:`to_latex_booktabs` (``digits``, ``index``, ``escape``, ...).

    Returns
    -------
    Path
        Path of the ``.tex`` file.
    """
    out_dir = Path(tab_dir or config.TAB_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name[:-4] if name.endswith(".tex") else name
    path = out_dir / f"{stem}.tex"
    _atomic_write(path, to_latex_booktabs(df, **kwargs))
    logger.info("saved %s", path)
    return path
