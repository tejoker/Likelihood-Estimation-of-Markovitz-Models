# Handover: finishing the v2 runs on another machine (2026-11-20)

**What is left:** one experiment step, then rebuilding the tables.
- The step is the E5 robustness grid on the real volatility path (`gridrealvol`): 850 blocks,
  of which 44 were done on the original laptop and are in the handover archive. That leaves
  806 blocks.
- On the laptop's GPU (RTX 5050 Laptop, ~50 W) a block takes about 2 minutes, so ~27 h. A
  desktop GPU should be several times faster.
- After it, the script rebuilds every v2 table and figure, which takes minutes.

**How resuming works:** the run is split into blocks, and each finished block is saved as a
checkpoint. The checkpoint folder is named by a hash of the code, the settings and the torch
version. If all three match what produced the archive, your run skips the 44 finished blocks
and continues. If anything differs, it silently starts from zero. Step 5 checks this.

## You need

- **Linux** with an **NVIDIA GPU**:
  - `nvidia-smi` must show `CUDA Version: 13.2` or higher;
  - at least 4 GB of GPU memory (the run uses about 3 GB);
  - at least 4 GB of free RAM (the run uses about 2.3 GB).
- About 7 GB of disk: the GPU Python environment alone is 5.6 GB.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), the Python package manager.
  It installs Python 3.12 and every package at the exact locked versions.
- Two files sent to you privately:
  - `synthval_v2_handover_2026-11-20.tar.gz` (36 MB);
  - its checksum file, `synthval_v2_handover_2026-11-20.tar.gz.sha256`.

## Steps

**1. Get the code.**
```bash
git clone --branch paper-v2 https://github.com/tejoker/Likelihood-Estimation-of-Markovitz-Models.git
cd Likelihood-Estimation-of-Markovitz-Models
```

**2. Install the GPU environment** (downloads about 3 GB; torch comes from download.pytorch.org):
```bash
uv sync --frozen --project envs/gpu
PYTHONPATH=src uv run --frozen --project envs/gpu python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected output: `2.15.0+cu132 True <your GPU name>`. Anything else: see Troubleshooting.

**3. Unpack the archive** at the repository root, after checking it:
```bash
(cd /path/to/folder/with/archive && sha256sum -c synthval_v2_handover_2026-11-20.tar.gz.sha256)
tar xzf /path/to/folder/with/archive/synthval_v2_handover_2026-11-20.tar.gz
ls results_v2/e5/checkpoints/gridrealvol/045e11f6c692/*.csv | wc -l     # must print 44
```

**4. Launch.** The run is detached, so it keeps going if you close the terminal:
```bash
setsid nohup bash experiments/run_v2_handover.sh >> results_v2/logs/v2.log 2>&1 < /dev/null &
```

**5. Check that it resumed** (within a minute of launching):
```bash
grep 'gridrealvol\] 14 cells' results_v2/logs/v2.log | tail -1
```
It must say `850 blocks (44 from checkpoints, 806 to run)`, with the folder ending in
`gridrealvol/045e11f6c692`.

If it says `0 from checkpoints` or names a different folder, something differs from the
original machine (code, torch version or settings). Stop the run with
`bash experiments/v2ctl.sh stop` and get in touch before going further.

**6. Monitor.**
```bash
ls results_v2/e5/checkpoints/gridrealvol/045e11f6c692/*.csv | wc -l   # progress, 850 = done
tail -f results_v2/logs/v2.log                                          # live log (Ctrl-C to quit)
bash experiments/v2ctl.sh status                                        # processes (STAT T = paused)
```

**7. Pause, stop and restart:**
- `bash experiments/v2ctl.sh pause` freezes the run and `resume` continues it. Nothing is lost.
- `bash experiments/v2ctl.sh stop` ends it. Finished blocks are kept; only the block in
  progress (about 2 minutes) is lost.
- After a stop or a reboot, run step 4 again; it resumes from the checkpoints.

**8. Check that it finished.** The last line of the log must be
`END (exit 0) e4 --plots-only`, and every step must have exit 0:
```bash
grep 'END (exit' results_v2/logs/v2.log | tail -9
```

**9. Send the results back.**
- **If you have push access** to the repository:
  ```bash
  git add results_v2
  git commit -m "v2: E5 gridrealvol and final tables (handover run on <your GPU>)"
  git push origin paper-v2
  ```
  `.gitignore` keeps checkpoints, logs and model-fit files out of git.
- **Otherwise**, send the results folder without checkpoints:
  ```bash
  tar czf results_v2_final.tar.gz --exclude='*/checkpoints' --exclude='*/fits/*.pkl' results_v2
  ```
- **In both cases**, also send `results_v2/logs/v2.log`. It is not in git, and it documents the
  run.

## Do not

- **Do not edit** anything in `src/synthval/`, `experiments/e3_sp500.py`,
  `experiments/e5_robustness.py` or `envs/gpu/`. Do not run `uv lock` or `uv add` there. The
  checkpoint keys hash the code and the torch version: any change silently restarts from zero,
  and the results would no longer match the pre-registered protocol (`docs/PROTOCOL_v2.md`).
- **Do not pass `--fresh`.** It discards the checkpoints.
- **Do not run two copies at once.** They would write to the same checkpoint folder.
- **Do not commit, upload or share the archive** or anything from `data/` that is not already
  in git. The price data are licensed, and the archive contains model fits that hold the real
  return series.

## Troubleshooting

- **`torch.cuda.is_available()` is False:** the driver is too old for CUDA 13.2, or no NVIDIA
  GPU is visible. `nvidia-smi` must work and show `CUDA Version` 13.2 or higher. Update the
  driver; do not change the torch version (see "Do not").
- **`uv sync` fails while downloading torch:** retry with a longer timeout:
  ```bash
  UV_HTTP_TIMEOUT=1800 UV_CONCURRENT_DOWNLOADS=1 uv sync --frozen --project envs/gpu
  ```
- **CUDA out of memory:** close other programs that use the GPU. The run needs about 3 GB.
- **The log shows a Python error:** stop the run and send `results_v2/logs/v2.log`.

Background: the protocol and its change log are in [`PROTOCOL_v2.md`](PROTOCOL_v2.md), and the
results so far in [`STATUS_v2.md`](STATUS_v2.md).
