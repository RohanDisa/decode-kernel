#!/usr/bin/env bash
# Full measurement pass for a Linux GPU box (Kaggle T4, Colab, A100 rental).
# Writes JSON under bench/results/ named with the GPU. Commit those files.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -e . pytest matplotlib -q
python -m pytest tests/test_quantize.py tests/test_correctness.py -q

python -m bench.ceilings
python -m bench.bench --shapes 4096x4096 --versions 0,1,2,3,4,5
python -m bench.bench --shapes 4096x11008,4096x14336 --versions 5 --iters 200
python -m bench.bench --sweep --shapes 4096x4096 --iters 200
python -m bench.ncu_collect || python -m bench.ncu_collect --print-only
python -m bench.plot

echo
echo "JSON and figures are under bench/results and bench/figures."
echo "Copy the medians into RESULTS.md. Do not type numbers from memory."
