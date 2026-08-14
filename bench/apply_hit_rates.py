"""Recompute GEMV %HBM from existing latencies against a new ceiling JSON.

Does not rerun kernels. Writes hit-rate fields onto stages/sweep/diag JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bench.common import RESULTS_DIR, annotate_hit_rates, load_json, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    written = []
    for tag in ("T4", "A100", "A10G"):
        ce = args.results / f"ceilings_{tag}.json"
        if not ce.exists():
            continue
        ceilings = load_json(ce)
        if "hbm_read" not in ceilings:
            print(f"skip {ce.name}: no hbm_read")
            continue
        for pattern in (
            f"stages_{tag}.json",
            f"sweep_{tag}.json",
            f"diag_rows_{tag}.json",
            f"diag_wpr2_{tag}.json",
            f"diag_wpr4_{tag}.json",
        ):
            path = args.results / pattern
            if not path.exists():
                continue
            payload = load_json(path)
            annotate_hit_rates(payload, ceilings)
            save_json(path, payload)
            written.append(path)
            print(f"annotated {path.name}  read={ceilings['hbm_read']['gbps']:.1f} GB/s")
    if not written:
        raise SystemExit("no stage/sweep JSON found to annotate")


if __name__ == "__main__":
    main()
