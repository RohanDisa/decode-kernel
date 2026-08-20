# GPU runners

Local `python -m bench.bench` is unchanged. Modal is an alternate remote runner that writes the same JSON under `bench/results/`.

## Modal (scripted T4 / A10G / A100)

```bash
pip install modal
modal setup   # once per machine
modal run bench/modal_runner.py --gpu T4
modal run bench/modal_runner.py --gpu A10G --mode all
DK_MODAL_GPU=A100 modal run bench/modal_runner.py --mode all
python -m bench.plot
```

`--gpu` chooses the SKU and the `nvcc -arch` flag (`sm_75` / `sm_86` / `sm_80`). JSON `meta` includes `gpu_type`, `driver_version`, and full `nvidia-smi` output.

## Kaggle (recommended for interactive development)

1. New notebook, GPU T4 (or T4 x2 — use one).
2. Add this repo as a dataset, or clone it.
3. In the first cell:

```bash
%cd /kaggle/working
!pip install -e /kaggle/input/<this-repo> pytest matplotlib
# if cloned:
# !git clone <url> && pip install -e decode-kernel pytest matplotlib
```

4. Run `bash scripts/run_all.sh` from the repo root. Session timeouts: run ceilings + v0–v5 first, sweep second, ncu third.

5. Download `bench/results/*.json` and `bench/figures/*.png` before the session dies.

## Colab

Runtime → T4. Same commands. Colab Pro if you want an L4/A100 for the final pass.

## A100 rental (final numbers)

Two hours is enough. Lock clocks if `nvidia-smi -lgc` is permitted; if not, say so in `RESULTS.md`. Run `scripts/run_all.sh`, then `ncu` for v0 and v5 at 4096×4096 batch 1.

Report T4 and A100 side by side. The claim that should hold: **percent of measured HBM stays roughly constant, absolute latency drops with bandwidth.**
