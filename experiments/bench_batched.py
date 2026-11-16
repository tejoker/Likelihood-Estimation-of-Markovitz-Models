"""Benchmark: K anchored relative tests, one by one vs as one critic bank (synthval.batched).

E3-like sizes by default (the power designs D1/D2): R_new = 100 tickers x 30 windows (3000),
G = 3000 windows (one independent draw per test), R_past = 100 x 62 windows (6200), L = 20; the
dependent protocol (10 time blocks, HAC); the v2 critic (time-pooled CNN), 1000 Adam steps,
batch 128, 1000 permutations. Everything runs in this one process on ``get_device()``
(``$SYNTHVAL_DEVICE``).

The sequential time per test is measured once on ``--seq-tests`` tests and scaled to K; the
batched time is measured for every K. For the tests run both ways, the largest differences of the
statistic and the p-value are printed (rounding of the batched training; the random streams are
identical). Peak host memory is ``VmHWM`` (reset before each measurement when the kernel allows
it); peak GPU memory is ``torch.cuda.max_memory_allocated``. Watch the GPU utilisation with
``nvidia-smi dmon -s u`` in another terminal.

    # GPU (outside a sandbox that hides /dev/nvidia*), from the repository root:
    export SYNTHVAL_DEVICE=cuda PYTHONPATH=src
    uv run --project envs/gpu python experiments/bench_batched.py
    # CPU smoke test with small settings:
    uv run python experiments/bench_batched.py --ks 1,4 --n-steps 20 --tickers 10 --seq-tests 2
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from synthval import pipeline as pl  # noqa: E402
from synthval.batched import AnchoredJob, anchored_relative_tests  # noqa: E402
from synthval.twosample.nets import get_device  # noqa: E402


def _reset_host_peak() -> bool:
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def _host_peak_mb() -> float:
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return float("nan")


def _gpu_reset() -> None:
    if torch.cuda.is_available() and get_device().type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _gpu_peak_mb() -> float:
    if torch.cuda.is_available() and get_device().type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**20
    return float("nan")


class Design:
    """E3-like windows: R_new and R_past fixed, one synthetic draw G per test."""

    def __init__(self, n_tickers: int, train_days: int, test_days: int, L: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.N, self.L, self.n_train, self.n_test = n_tickers, L, train_days, test_days
        self.vol = 0.01 * np.exp(0.3 * self.rng.standard_normal(n_tickers))
        self.Wp, _, self.op = self._windows(train_days, 0, 1.0, self.rng)
        self.Wn, _, self.on = self._windows(test_days, train_days, 1.0, self.rng)
        self._g: list[tuple[np.ndarray, np.ndarray]] = []

    def _windows(self, days: int, offset: int, factor: float, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = factor * self.vol * rng.standard_normal((days, self.N))
        return pl.panel_windows(r, self.L, scale=self.vol, date_offset=offset)

    def g(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Synthetic windows of test ``k`` (alternating vol x 1.2 and the boundary)."""
        while len(self._g) <= k:
            j = len(self._g)
            W, _, o = self._windows(self.n_test, self.n_train, 1.2 if j % 2 == 0 else 1.0,
                                    np.random.default_rng(10_000 + j))
            self._g.append((W, o))
        return self._g[k]

    def jobs(self, K: int) -> list[AnchoredJob]:
        out = []
        for k in range(K):
            G, og = self.g(k)
            out.append(AnchoredJob(self.Wn, G, self.Wp, rng=np.random.default_rng(20_000 + k),
                                   orders=(self.on, og, self.op)))
        return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", default="1,8,20,40,80", help="bank sizes K (comma-separated)")
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-perm", type=int, default=pl.N_PERM)
    ap.add_argument("--arch", default="pooled", choices=["pooled", "mlp"])
    ap.add_argument("--impl", default="grouped", choices=["grouped", "loop"])
    ap.add_argument("--max-bank", type=int, default=0, help="critics per bank (0: K)")
    ap.add_argument("--seq-tests", type=int, default=3,
                    help="tests timed sequentially (0: skip the sequential runs)")
    ap.add_argument("--tickers", type=int, default=100)
    ap.add_argument("--train-days", type=int, default=1258)
    ap.add_argument("--test-days", type=int, default=615)
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="torch.backends.cudnn.benchmark = True (autotuned convolution "
                         "algorithms; the experiments do not set it)")
    ap.add_argument("--out", default="", help="optional CSV path for the results")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    dev = get_device()
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    base_kw: dict[str, Any] = {"n_steps": args.n_steps, "batch_size": args.batch_size,
                               "n_perm": args.n_perm, "arch": args.arch}
    kw = dict(base="critic", protocol="dependent", base_kw=base_kw, n_boot=pl.N_BOOT)
    design = Design(args.tickers, args.train_days, args.test_days, args.L)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "-"
    print(f"device {dev} ({gpu}), torch {torch.__version__}, threads {args.threads}; "
          f"windows R_new {design.Wn.shape}, R_past {design.Wp.shape}, G {design.g(0)[0].shape}; "
          f"critic {args.arch} {args.n_steps} steps batch {args.batch_size}, impl {args.impl}",
          flush=True)

    # Warm-up (CUDA context, kernels) outside the measurements.
    anchored_relative_tests(design.jobs(2), base="critic", protocol="dependent",
                            base_kw={**base_kw, "n_steps": 3, "n_perm": 10}, impl=args.impl)

    seq_rows: list[dict[str, Any]] = []
    seq_s = seq_host = seq_gpu = float("nan")
    if args.seq_tests > 0:
        _reset_host_peak()
        _gpu_reset()
        t0 = time.perf_counter()
        for j in design.jobs(args.seq_tests):
            seq_rows.append(pl.relative_tests(j.R_new, j.G, j.R_past, orders=j.orders,
                                              methods=("anchored",), rng=copy.deepcopy(j.rng),
                                              **kw)[0])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        seq_s = (time.perf_counter() - t0) / args.seq_tests
        seq_host, seq_gpu = _host_peak_mb(), _gpu_peak_mb()
        print(f"sequential: {seq_s:.2f} s per test ({1.0 / seq_s:.3f} tests/s), peak host "
              f"{seq_host:.0f} MB, peak GPU {seq_gpu:.0f} MB", flush=True)

    out = []
    for K in ks:
        jobs = design.jobs(K)
        _reset_host_peak()
        _gpu_reset()
        t0 = time.perf_counter()
        rows = anchored_relative_tests(jobs, impl=args.impl, max_bank=args.max_bank or None,
                                       **kw)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        m = min(K, len(seq_rows))
        dstat = max((abs(a["statistic"] - b["statistic"]) for a, b in zip(seq_rows, rows[:m])),
                    default=float("nan"))
        dp = max((abs(a["pvalue"] - b["pvalue"]) for a, b in zip(seq_rows, rows[:m])),
                 default=float("nan"))
        rec = {"K": K, "device": str(dev), "arch": args.arch, "impl": args.impl,
               "n_steps": args.n_steps, "cudnn_benchmark": bool(args.cudnn_benchmark),
               "batched_s": dt, "batched_tests_per_s": K / dt,
               "seq_s_per_test": seq_s, "seq_tests_per_s": 1.0 / seq_s,
               "seq_s_extrapolated": K * seq_s, "speedup": K * seq_s / dt,
               "host_peak_mb": _host_peak_mb(), "gpu_peak_mb": _gpu_peak_mb(),
               "seq_host_peak_mb": seq_host, "seq_gpu_peak_mb": seq_gpu,
               "n_compared": m, "max_abs_dstat": dstat, "max_abs_dp": dp}
        out.append(rec)
        print(json.dumps(rec), flush=True)
    df = pd.DataFrame(out)
    cols = ["K", "batched_s", "batched_tests_per_s", "seq_tests_per_s", "speedup",
            "host_peak_mb", "gpu_peak_mb", "max_abs_dstat", "max_abs_dp"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
