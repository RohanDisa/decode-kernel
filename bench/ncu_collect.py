"""Collect Nsight Compute counters for each kernel stage.

ncu is the GPU equivalent of `perf`. These counters are the argument:

- dram__bytes_read/write          did we actually move 4x fewer bytes?
- dram__throughput % of peak      headline bandwidth utilization
- sm__throughput % of peak        should stay low at batch 1 (memory-bound)
- l1tex load sectors              coalescing quality
- sm__warps_active %              occupancy
- smsp__inst_executed             dequant instruction overhead

If `ncu` is not on PATH, this prints the command to run on the GPU machine.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from bench.common import RESULTS_DIR

METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
]


def ncu_cmd(out_csv: Path, version: int, m: int, k: int, batch: int) -> list[str]:
    metric_flag = ",".join(METRICS)
    return [
        "ncu",
        "--csv",
        "--metrics",
        metric_flag,
        "-o",
        str(out_csv.with_suffix("")),
        "--target-processes",
        "all",
        sys.executable,
        "-m",
        "bench.ncu_launch",
        "--version",
        str(version),
        "--m",
        str(m),
        "--k",
        str(k),
        "--batch",
        str(batch),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--versions", default="0,1,2,3,4,5")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--print-only", action="store_true")
    args = p.parse_args()

    versions = [int(v) for v in args.versions.split(",") if v]
    ncu = shutil.which("ncu")
    cmds = []
    for v in versions:
        out = RESULTS_DIR / f"ncu_v{v}_m{args.m}_k{args.k}_b{args.batch}"
        cmds.append(ncu_cmd(out, v, args.m, args.k, args.batch))

    if args.print_only or ncu is None:
        print("# ncu not found" if ncu is None else "# ncu commands")
        for c in cmds:
            print(" ".join(c))
        if ncu is None:
            raise SystemExit(0)
        if args.print_only:
            return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    for c in cmds:
        print("running:", " ".join(c))
        proc = subprocess.run(c, check=False, capture_output=True, text=True)
        log.append(
            {
                "cmd": c,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        )
    (RESULTS_DIR / "ncu_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"wrote {RESULTS_DIR / 'ncu_log.json'}")


if __name__ == "__main__":
    main()
