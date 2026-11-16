"""Run an experiment under the v2 protocol (docs/PROTOCOL_v2.md): GPU, time-pooled critic, full
replication counts, separate output tree.

    PYTHONPATH=src uv run --project envs/gpu python experiments/v2.py <exp> [script arguments]

``<exp>`` is one of e1, e2, e2dep, e3, e4, e5, e5arch. The v1 scripts are not modified: this
runner sets the v2 environment *before* importing synthval, applies the v2 overrides of the
script's ``FULL`` configuration (the compute cuts of the v1 run are undone, DESIGN section 4
values), records the runtime environment and calls the script's ``main``.

Environment (defaults; an existing value is kept):
    SYNTHVAL_DEVICE=cuda          networks on the GPU (random draws stay on the CPU)
    SYNTHVAL_ARCH_1D=pooled       arch="auto" -> PooledConvCritic for (L,) windows
    SYNTHVAL_ARCH_2D=pooled       arch="auto" -> PooledConvCritic for (C, L) windows
    SYNTHVAL_RESULTS_DIR=results_v2, SYNTHVAL_TAB_DIR=results_v2/tables,
    SYNTHVAL_FIG_DIR=results_v2/figures
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_V2 = {
    "SYNTHVAL_DEVICE": "cuda",
    "SYNTHVAL_ARCH_1D": "pooled",
    "SYNTHVAL_ARCH_2D": "pooled",
    "SYNTHVAL_RESULTS_DIR": "results_v2",
    "SYNTHVAL_TAB_DIR": "results_v2/tables",
    "SYNTHVAL_FIG_DIR": "results_v2/figures",
}
for _k, _v in ENV_V2.items():
    os.environ.setdefault(_k, _v)
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

MODULES = {"e1": "e1_placebo", "e2": "e2_ground_truth", "e2dep": "e2_dependence",
           "e3": "e3_sp500", "e4": "e4_markowitz", "e5": "e5_robustness",
           "e5arch": "e5_architecture"}

#: v2 overrides of each script's FULL dict: the DESIGN section 4 values that the v1 run cut for
#: compute (DESIGN change log 2026-11-15), unless noted.
OVERRIDES: dict[str, dict] = {
    "e1": {"real_splits": 100},
    "e2": {"e_n_list": [500, 2000]},
    "e2dep": {"n_list": [500, 2000], "reps_h0": 200, "reps_h1": 100},
    "e3": {"n_seeds": 10, "power_reps": 200, "realvol_reps": 200},
    "e4": {},  # FULL already holds 10 test seeds and 200 placebo replications
    "e5": {"n_seeds": 10, "power_reps_default": 200, "power_reps": 50, "split_power_reps": 50},
    "e5arch": {},
}


def _gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                    "torch": torch.__version__}
        return {"gpu": None, "torch": torch.__version__}
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"gpu_error": repr(exc)}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        raise SystemExit(f"usage: experiments/v2.py <{'|'.join(MODULES)}> [script arguments]")
    exp, argv = sys.argv[1], sys.argv[2:]
    mod = importlib.import_module(MODULES[exp])
    over = OVERRIDES[exp]
    # Apply the overrides to the target and to every experiment module it imported (E5 reuses
    # E3's configuration, E5arch E2/E3/E5's), so that cross-module alignment checks hold.
    applied = {}
    for name, modname in MODULES.items():
        m = sys.modules.get(modname)
        if m is None or not hasattr(m, "FULL"):
            continue
        for k in OVERRIDES[name]:
            if k not in m.FULL:
                raise SystemExit(f"v2 override {k!r} is not a key of {modname}.FULL")
        m.FULL.update(OVERRIDES[name])
        applied[name] = OVERRIDES[name]
    from synthval import config
    from synthval.pipeline import runtime_config
    out = Path(config.RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"exp": exp, "module": MODULES[exp], "argv": argv, "overrides": applied,
           "env": {k: os.environ.get(k) for k in ENV_V2}, "runtime": runtime_config(),
           **_gpu_info(), "python": platform.python_version(),
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    log = out / "v2_runs.jsonl"
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"v2: {exp} -> {out} | {rec['runtime']} | {rec.get('gpu')}", flush=True)
    mod.main(argv)


if __name__ == "__main__":
    main()
