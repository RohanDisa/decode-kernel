.PHONY: install test ceilings bench sweep plots ncu modal

install:
	pip install -e . pytest matplotlib

test:
	pytest tests/test_quantize.py -q
	pytest tests/test_correctness.py -q

ceilings:
	python -m bench.ceilings

bench:
	python -m bench.bench --shapes 4096x4096 --versions 0,1,2,3,4,5

sweep:
	python -m bench.bench --sweep --shapes 4096x4096 --iters 200

plots:
	python -m bench.plot

ncu:
	python -m bench.ncu_collect --print-only

# Alternate remote runner. GPU=T4|A10G|A100  MODE=bench|ceilings|sweep|all
GPU ?= T4
MODE ?= bench
modal:
	modal run bench/modal_runner.py --gpu $(GPU) --mode $(MODE)

install:
	pip install -e . pytest matplotlib

test:
	pytest tests/test_quantize.py -q
	pytest tests/test_correctness.py -q

ceilings:
	python -m bench.ceilings

bench:
	python -m bench.bench --shapes 4096x4096 --versions 0,1,2,3,4,5

sweep:
	python -m bench.bench --sweep --shapes 4096x4096 --iters 200

plots:
	python -m bench.plot

ncu:
	python -m bench.ncu_collect --print-only
