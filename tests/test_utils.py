"""Tests for synthval.utils (seeding, parallel map, saving) and synthval.plotting."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthval import plotting, utils


# ---------------------------------------------------------------------------------------------
# Worker functions (module level so that spawned workers can unpickle them)
# ---------------------------------------------------------------------------------------------
def _draw(item: tuple[int, np.random.Generator]) -> tuple[int, float, float]:
    """A stochastic task: numpy draws plus torch draws seeded from the item's generator."""
    import torch

    i, rng = item
    a = float(rng.standard_normal(1000).sum())
    torch.manual_seed(int(rng.integers(2**31 - 1)))
    b = float(torch.randn(100, dtype=torch.float64).sum())
    return i, a, b


def _threads(_: int) -> tuple[int, str | None, int]:
    import torch

    return torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS"), os.getpid()


def _fail_on_three(i: int) -> int:
    if i == 3:
        raise ValueError("boom")
    return i * i


class _OddError(Exception):
    """An exception that pickles but cannot be unpickled (``__init__`` takes two arguments)."""

    def __init__(self, a: int, b: int) -> None:
        super().__init__(f"{a}-{b}")


def _raise_odd(i: int) -> int:
    if i == 2:
        raise _OddError(1, 2)
    return i


# ---------------------------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------------------------
def test_spawn_rngs_deterministic_and_prefix_stable():
    a = [g.random(5) for g in utils.spawn_rngs(123, 6)]
    b = [g.random(5) for g in utils.spawn_rngs(123, 6)]
    c = [g.random(5) for g in utils.spawn_rngs(123, 3)]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(a[:3], c):
        np.testing.assert_array_equal(x, y)
    assert len({tuple(x) for x in a}) == 6                 # distinct streams
    d = [g.random(5) for g in utils.spawn_rngs(124, 6)]
    assert not np.array_equal(a[0], d[0])
    assert utils.spawn_rngs(1, 0) == []


def test_spawn_rngs_from_generator_and_seedsequence():
    g1 = [g.random() for g in utils.spawn_rngs(np.random.default_rng(7), 3)]
    g2 = [g.random() for g in utils.spawn_rngs(np.random.default_rng(7), 3)]
    assert g1 == g2
    s1 = [g.random() for g in utils.spawn_rngs(np.random.SeedSequence(7), 3)]
    s2 = [g.random() for g in utils.spawn_rngs(7, 3)]
    assert s1 == s2


def test_seed_everything_reproducible():
    import random

    import torch

    out = []
    for _ in range(2):
        g = utils.seed_everything(2025)
        out.append((random.random(), np.random.random(), float(torch.rand(1)), g.random()))  # noqa: NPY002
    assert out[0] == out[1]


# ---------------------------------------------------------------------------------------------
# parallel_map
# ---------------------------------------------------------------------------------------------
def test_parallel_map_same_results_serial_and_parallel():
    items = list(enumerate(utils.spawn_rngs(42, 10)))
    serial = utils.parallel_map(_draw, items, n_workers=1)
    items = list(enumerate(utils.spawn_rngs(42, 10)))      # fresh generators
    parallel = utils.parallel_map(_draw, items, n_workers=4)
    assert [r[0] for r in parallel] == list(range(10))     # input order
    assert serial == parallel                              # bit-identical
    items3 = list(enumerate(utils.spawn_rngs(42, 10)))
    assert utils.parallel_map(_draw, items3, n_workers=3) == serial


def test_parallel_map_reused_generators_same_in_both_modes():
    """Items are passed by value: reusing an rngs list gives the same draws in both modes."""
    rngs = utils.spawn_rngs(0, 3)
    state0 = [g.bit_generator.state for g in rngs]
    items = list(enumerate(rngs))
    s1 = utils.parallel_map(_draw, items, n_workers=1)
    s2 = utils.parallel_map(_draw, items, n_workers=1)
    p1 = utils.parallel_map(_draw, items, n_workers=3)
    p2 = utils.parallel_map(_draw, items, n_workers=3)
    assert s1 == s2 == p1 == p2
    assert [g.bit_generator.state for g in rngs] == state0     # caller's generators untouched
    assert utils.parallel_map(_draw, items[:1], n_workers=4) == s1[:1]   # single-item path


def test_parallel_map_single_thread_workers_and_env_restored():
    before = os.environ.get("OMP_NUM_THREADS")
    res = utils.parallel_map(_threads, range(4), n_workers=2)
    assert all(n == 1 and omp == "1" for n, omp, _ in res)
    assert len({pid for *_, pid in res} - {os.getpid()}) >= 1   # ran in other processes
    assert os.environ.get("OMP_NUM_THREADS") == before


@pytest.mark.parametrize("n_workers", [1, 3])
def test_parallel_map_exception_names_item(n_workers):
    with pytest.raises(ValueError, match="boom") as info:
        utils.parallel_map(_fail_on_three, range(6), n_workers=n_workers)
    assert info.value.item_index == 3
    assert any("item 3" in note for note in info.value.__notes__)


def test_parallel_map_unpicklable_exception_keeps_message():
    with pytest.raises(_OddError, match="1-2"):
        utils.parallel_map(_raise_odd, range(4), n_workers=1)
    with pytest.raises(RuntimeError) as info:
        utils.parallel_map(_raise_odd, range(4), n_workers=2)
    msg = str(info.value)
    assert "_OddError: 1-2" in msg and "_raise_odd" in msg      # type, message, traceback
    assert "process pool broke" not in msg
    assert info.value.item_index == 2


def test_parallel_map_trivial_inputs():
    assert utils.parallel_map(_fail_on_three, [], n_workers=4) == []
    assert utils.parallel_map(_fail_on_three, [5], n_workers=4) == [25]
    with pytest.raises(ValueError):
        utils.parallel_map(_fail_on_three, [1, 2], n_workers=0)


# ---------------------------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------------------------
def test_save_results_and_meta(tmp_path):
    df = pd.DataFrame({"test": ["critic", "mmd"], "rej": [0.051, 1 / 3]})
    cfg = {"n": np.int64(500), "deltas": np.array([0.0, 0.1]), "seed": 1}
    p_csv = utils.save_results(df, "size", "e9", cfg, results_dir=tmp_path,
                               timing={"total_s": 1.5})
    assert p_csv == tmp_path / "e9" / "size.csv"
    back = pd.read_csv(p_csv)
    pd.testing.assert_frame_equal(back, df)                # full float precision, no index col
    p_json = utils.save_results({"stat": np.float64(0.25), "p": np.nan, "arr": np.arange(3)},
                                "summary.json", "e9", cfg, results_dir=tmp_path)
    assert p_json.name == "summary.json"
    d = json.loads(p_json.read_text())
    assert d["stat"] == 0.25 and d["arr"] == [0, 1, 2] and np.isnan(d["p"])
    meta = json.loads((tmp_path / "e9" / "meta.json").read_text())
    assert meta["experiment"] == "e9"
    assert meta["config"] == {"n": 500, "deltas": [0.0, 0.1], "seed": 1}
    assert set(meta["files"]) == {"size.csv", "summary.json"}
    assert meta["files"]["size.csv"]["config"] == meta["config"]
    assert meta["timing"] == {"total_s": 1.5}
    assert meta["packages"]["numpy"] == np.__version__
    assert meta["git"]["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", meta["git"]["commit"])
    assert meta["git"]["dirty_paths"] == ["src", "experiments"]
    assert re.fullmatch(r"[0-9a-f]{64}", meta["code"]["sha256"])
    assert meta["code"]["dir"] == "src/synthval" and meta["code"]["n_files"] >= 10
    if meta["data"] is not None:
        assert meta["data"]["path"] == "data/sp500_price_volume.csv"
        assert re.fullmatch(r"[0-9a-f]{64}", meta["data"]["sha256"])
    assert meta["elapsed_s"] >= 0


def test_meta_keeps_each_files_config(tmp_path):
    """A later call with another config neither overwrites the run config nor loses its own."""
    for i, cfg in enumerate([{"n": 1}, {"n": 2}, {"n": 2, "other": True}]):
        utils.save_results({"v": i}, f"r{i}", "e9", cfg, results_dir=tmp_path)
    meta = json.loads((tmp_path / "e9" / "meta.json").read_text())
    assert meta["config"] == {"n": 1}
    assert [meta["files"][f"r{i}.json"]["config"] for i in range(3)] == [
        {"n": 1}, {"n": 2}, {"n": 2, "other": True}]
    utils.write_meta("e9", {"run": "explicit"}, results_dir=tmp_path)   # explicit run config
    meta = json.loads((tmp_path / "e9" / "meta.json").read_text())
    assert meta["config"] == {"run": "explicit"} and len(meta["files"]) == 3


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
        capture_output=True, text=True, check=True).stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_hash_counts_untracked_code_only(tmp_path):
    """Untracked files under src/ or experiments/ make the tree dirty; root clutter does not."""
    repo = tmp_path / "repo"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "a.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "src/pkg/a.py")
    _git(repo, "commit", "-q", "-m", "init")
    head = _git(repo, "rev-parse", "HEAD").strip()
    info = utils.git_hash(repo)
    assert info["commit"] == head and info["dirty"] is False and info["n_dirty_files"] == 0
    (repo / ".bashrc").write_text("# clutter\n")                  # outside the code paths
    assert utils.git_hash(repo)["dirty"] is False
    (repo / "experiments").mkdir()
    (repo / "experiments" / "e1.py").write_text("print(1)\n")      # untracked experiment
    info = utils.git_hash(repo)
    assert info["dirty"] is True and info["dirty_files"] == ["?? experiments/e1.py"]
    (repo / "experiments" / "e1.py").unlink()
    (repo / "src" / "pkg" / "a.py").write_text("x = 2\n")          # modified tracked file
    assert utils.git_hash(repo)["dirty_files"] == [" M src/pkg/a.py"]
    assert utils.git_hash(tmp_path / "not_a_repo")["dirty"] is None


def test_code_fingerprint_tracks_content(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "sub" / "b.py").write_text("y = 1\n")
    f0 = utils.code_fingerprint(pkg)
    assert f0["n_files"] == 2 and utils.code_fingerprint(pkg) == f0
    (pkg / "sub" / "b.py").write_text("y = 2\n")
    f1 = utils.code_fingerprint(pkg)
    assert f1["sha256"] != f0["sha256"]
    (pkg / "sub" / "b.py").rename(pkg / "sub" / "c.py")               # rename changes it too
    assert utils.code_fingerprint(pkg)["sha256"] != f1["sha256"]
    (pkg / "notes.txt").write_text("ignored")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "z.py").write_text("ignored")
    assert utils.code_fingerprint(pkg)["n_files"] == 2


def test_meta_from_previous_run_is_replaced(tmp_path):
    (tmp_path / "e9").mkdir()
    (tmp_path / "e9" / "meta.json").write_text(json.dumps(
        {"run_id": "old", "files": {"stale.csv": {}}}))
    utils.save_results(pd.DataFrame({"a": [1]}), "x", "e9", {}, results_dir=tmp_path)
    meta = json.loads((tmp_path / "e9" / "meta.json").read_text())
    assert set(meta["files"]) == {"x.csv"}


def test_save_results_index_kept_when_meaningful(tmp_path):
    df = pd.DataFrame({"v": [1.0, 2.0]}, index=pd.Index(["GBM", "GARCHt"], name="model"))
    p = utils.save_results(df, "m", "e9", None, results_dir=tmp_path)
    pd.testing.assert_frame_equal(pd.read_csv(p, index_col="model"), df)
    with pytest.raises(TypeError):
        utils.save_results([1, 2], "bad", "e9", None, results_dir=tmp_path)


# ---------------------------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------------------------
def test_to_latex_booktabs_formatting():
    df = pd.DataFrame({"stat_1": [0.12345, -0.5, np.nan], "n": [500, 2000, 10],
                       "note": ["a_b", "50%", "x"], "p": [1e-4, -1e-5, 0.5]},
                      index=pd.Index(["GBM", "GARCH_t", "Real"], name="model"))
    tex = utils.to_latex_booktabs(df, digits={"stat_1": 2, "p": 3})
    lines = tex.strip().split("\n")
    assert lines[0] == r"\begin{tabular}{lrrrr}"
    assert lines[1] == r"\toprule"
    assert lines[2] == r"model & stat\_1 & n & note & p \\"
    assert lines[3] == r"\midrule"
    assert lines[4] == r"GBM & 0.12 & 500 & a\_b & 0.000 \\"
    assert lines[5] == r"GARCH\_t & $-$0.50 & 2000 & 50\% & 0.000 \\"   # no "-0.000"
    assert lines[6] == r"Real & -- & 10 & x & 0.500 \\"
    assert lines[-2:] == [r"\bottomrule", r"\end{tabular}"]


def test_to_latex_booktabs_multiindex_and_float_env():
    cols = pd.MultiIndex.from_product([["n=500", "n=2000"], ["0.05", "0.10"]])
    idx = pd.MultiIndex.from_tuples([("vol", "critic"), ("vol", "mmd"), ("tails", "critic")],
                                    names=["family", "test"])
    df = pd.DataFrame(np.arange(12).reshape(3, 4) / 10, index=idx, columns=cols)
    tex = utils.to_latex_booktabs(df, digits=2, caption="Size.", label="tab:size")
    assert tex.startswith("\\begin{table}[t]\n\\centering\n\\caption{Size.}\n\\label{tab:size}")
    assert r"\multicolumn{2}{c}{n=500} & \multicolumn{2}{c}{n=2000} \\" in tex
    assert r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}" in tex
    assert r"vol & critic & 0.00 & 0.10 & 0.20 & 0.30 \\" in tex
    assert r" & mmd & 0.40 & 0.50 & 0.60 & 0.70 \\" in tex       # repeated label blanked
    assert r"tails & critic & 0.80 & 0.90 & 1.00 & 1.10 \\" in tex
    assert tex.rstrip().endswith(r"\end{table}")


def test_save_table_tex(tmp_path):
    df = pd.DataFrame({"$\\alpha$": [0.05]})
    p = utils.save_table_tex(df, "e9_tab.tex", tab_dir=tmp_path, index=False, escape=False)
    assert p == tmp_path / "e9_tab.tex"
    assert p.read_text() == ("\\begin{tabular}{r}\n\\toprule\n$\\alpha$ \\\\\n\\midrule\n"
                             "0.050 \\\\\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------------
def test_plotting_palettes():
    assert set(plotting.COLORS_TEST) == {"critic", "lrcritic", "c2st", "mmd", "sigmmd", "ks",
                                         "oracle"}
    assert set(plotting.COLORS_MODEL) == {"GBM", "Merton", "GARCHt", "Heston", "Bootstrap",
                                          "MVGaussian", "MVStudentT", "CCCGARCHt",
                                          "MVBootstrap", "Real"}
    okabe = set(plotting.OKABE_ITO.values())
    for c in [*plotting.COLORS_TEST.values(), *plotting.COLORS_MODEL.values()]:
        assert c in okabe
    assert len(set(plotting.COLORS_TEST.values())) == len(plotting.COLORS_TEST)
    uni = ["GBM", "Merton", "GARCHt", "Heston", "Bootstrap", "Real"]
    assert len({plotting.COLORS_MODEL[m] for m in uni}) == len(uni)
    assert set(plotting.MARKERS_TEST) == set(plotting.COLORS_TEST)
    assert set(plotting.MARKERS_MODEL) == set(plotting.COLORS_MODEL)
    assert len(set(plotting.MARKERS_MODEL.values())) == len(plotting.MARKERS_MODEL)
    assert set(plotting.LABELS_TEST) == set(plotting.COLORS_TEST)
    assert plotting.figsize("two", 0.5) == (plotting.TWO_COL, plotting.TWO_COL * 0.5)
    assert (plotting.ONE_COL, plotting.TWO_COL) == (3.4, 7.0)


def test_plotting_setup_and_savefig(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotting.setup()
    assert matplotlib.rcParams["font.size"] == 9
    assert matplotlib.rcParams["text.usetex"] is False
    assert matplotlib.rcParams["font.family"] == ["serif"]
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    paths = []
    for _ in range(2):
        fig, ax = plt.subplots(figsize=plotting.figsize("one"))
        ax.plot([0, 1], [0, 1], color=plotting.COLORS_TEST["critic"], label=r"$\hat T$")
        ax.set_xlabel(r"$\delta$")
        ax.legend()
        paths.append(plotting.savefig(fig, "e9_demo.pdf", fig_dir=tmp_path, close=True))
    assert paths[0] == tmp_path / "e9_demo.pdf"
    raw = paths[0].read_bytes()
    assert raw.startswith(b"%PDF") and b"CreationDate" not in raw
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    p2 = plotting.savefig(fig, "e9_demo2", fig_dir=tmp_path, close=True)
    assert p2.read_bytes()[:4] == b"%PDF"
    with pytest.raises(ValueError):
        plotting.setup(font="comic")
