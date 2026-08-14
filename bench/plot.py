"""The three plots that carry the project.

1. Roofline, log-log, ridge marked, kernels far left of it.
2. Batch-size sweep: achieved bandwidth % and SM util % on the same axes.
3. Stage bars: DRAM throughput % against SM throughput %.

If measured JSON is missing, plot 1 is still drawn from the pre-registered
prediction so the argument is visible before a GPU run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bench.common import FIGURES_DIR, PREDICTION_PATH, RESULTS_DIR, hbm_roof, load_json

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _latest(results_dir: Path, pattern: str) -> Path | None:
    hits = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def plot_roofline(pred: dict, stages: dict | None, ceilings: dict | None, out: Path):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    if ceilings:
        bw_read, kind = hbm_roof(ceilings)
        bw = (bw_read or ceilings["hbm_copy"]["gbps"]) * 1e9
        peak = ceilings["cublas_gemm_fp16"]["tflops"] * 1e12
        gpu = ceilings["meta"]["gpu"]["name"]
        roof_label = f"roofline ({kind})"
        copy_gbps = ceilings.get("hbm_copy", {}).get("gbps")
    else:
        t4 = pred["ridge_point_estimate"]["T4"]
        bw = t4["achievable_hbm_gbs_estimate"] * 1e9
        peak = t4["peak_fp16_tflops"] * 1e12
        gpu = "T4 (estimated ceilings — replace after measurement)"
        roof_label = "roofline"
        copy_gbps = None

    ridge = peak / bw
    ai = np.logspace(-1, 3, 400)
    roof = np.minimum(bw * ai, peak)

    ax.loglog(ai, roof / 1e12, color="black", lw=2.0, label=roof_label)
    if copy_gbps:
        copy_roof = np.minimum(copy_gbps * 1e9 * ai, peak)
        ax.loglog(
            ai,
            copy_roof / 1e12,
            color="black",
            lw=1.0,
            ls=":",
            alpha=0.55,
            label="copy roof (read+write, reference)",
        )
    ax.axvline(ridge, color="black", ls="--", lw=1.0, alpha=0.7)
    ax.axvline(1.0, color="#4C78A8", ls=":", lw=1.4, label="fp16 GEMV  (1 FLOP/byte)")
    ax.axvline(4.0, color="#F58518", ls=":", lw=1.4, label="W4 GEMV  (4 FLOP/byte)")
    ax.annotate(
        f"ridge {ridge:.0f} FLOP/byte",
        xy=(ridge, peak / 1e12),
        xytext=(ridge * 1.15, peak / 1e12 * 0.35),
        fontsize=9,
    )

    if stages:
        for case in stages.get("cases", []):
            if case.get("batch") != 1:
                continue
            for k in case["kernels"]:
                if "tflops" not in k or k.get("skipped"):
                    continue
                ai = k.get("ai_flop_per_byte")
                if ai is None and k.get("precision") == "fp16":
                    ai = 1.0
                if ai is None:
                    continue
                ax.loglog(
                    ai,
                    k["tflops"],
                    marker="o",
                    markersize=8,
                    label=k["impl"],
                )

    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Achieved throughput (TFLOP/s)")
    ax.set_title(f"Roofline — decode GEMV sits left of the ridge\n{gpu}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0.3, 800)
    fig.savefig(out)
    plt.close(fig)


def plot_batch_sweep(sweep: dict, ncu: dict | None, ceilings: dict | None, out: Path):
    """HBM % vs batch. No SM series unless ncu is present — do not fake a crossover."""
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    hbm, kind = hbm_roof(ceilings)
    batches, reuse_pct, cublas_pct = [], [], []
    for case in sweep.get("cases", []):
        reuse = next((k for k in case["kernels"] if k.get("impl", "").startswith("v5")), None)
        cublas = next(
            (
                k
                for k in case["kernels"]
                if k.get("impl", "").startswith("cublas") and "gbps" in k
            ),
            None,
        )
        if reuse is None or "gbps" not in reuse or not hbm:
            continue
        batches.append(case["batch"])
        reuse_pct.append(100.0 * reuse["gbps"] / hbm)
        cublas_pct.append(100.0 * cublas["gbps"] / hbm if cublas else None)

    ax.plot(batches, reuse_pct, marker="o", color="#F58518", label="v5 weight-reuse")
    if any(v is not None for v in cublas_pct):
        ax.plot(
            batches,
            cublas_pct,
            marker="s",
            color="#4C78A8",
            label="cuBLAS fp16",
        )
    if ncu and "sm_pct_by_batch" in ncu:
        b2 = sorted(ncu["sm_pct_by_batch"])
        ax.plot(
            [int(x) for x in b2],
            [ncu["sm_pct_by_batch"][x] for x in b2],
            marker="^",
            color="#E45756",
            label="SM throughput % (ncu)",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Percent of measured read-only HBM")
    gpu = sweep.get("meta", {}).get("gpu", {}).get("name", "")
    extra = (
        f"% of {kind}"
        + (
            " — SM/DRAM crossover needs ncu (not available on Modal)"
            if not (ncu and "sm_pct_by_batch" in ncu)
            else " — Crossing of HBM % and SM % is the regime change"
        )
    )
    ax.set_title(f"Batch sweep — achieved HBM %\n{gpu}\n{extra}")
    ax.legend()
    ax.set_ylim(0, 110)
    fig.savefig(out)
    plt.close(fig)


def plot_batch_latency(sweep: dict, out: Path):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    batches, reuse_us, cublas_us = [], [], []
    for case in sweep.get("cases", []):
        reuse = next((k for k in case["kernels"] if k.get("impl", "").startswith("v5")), None)
        cublas = next(
            (
                k
                for k in case["kernels"]
                if k.get("impl", "").startswith("cublas") and "median_us" in k
            ),
            None,
        )
        if reuse is None or "median_us" not in reuse:
            continue
        batches.append(case["batch"])
        reuse_us.append(reuse["median_us"])
        cublas_us.append(cublas["median_us"] if cublas else None)
    ax.loglog(batches, reuse_us, marker="o", color="#F58518", label="v5 weight-reuse")
    if any(v is not None for v in cublas_us):
        ax.loglog(batches, cublas_us, marker="s", color="#4C78A8", label="cuBLAS fp16")
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Median latency (µs)")
    gpu = sweep.get("meta", {}).get("gpu", {}).get("name", "")
    ax.set_title(
        f"Batch sweep — latency\n{gpu}\n"
        "Reuse kernel tracks O(B). cuBLAS stays flat until the GEMM gets fat."
    )
    ax.legend()
    fig.savefig(out)
    plt.close(fig)


def plot_stage_bars(stages: dict, ncu: dict | None, ceilings: dict | None, out: Path):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    labels, dram, sm = [], [], []
    hbm, kind = hbm_roof(ceilings)
    case = None
    for c in stages.get("cases", []):
        if c.get("batch") == 1 and c.get("M") == 4096 and c.get("K") == 4096:
            case = c
            break
    if case is None:
        for c in stages.get("cases", []):
            if c.get("batch") == 1 and c.get("M") == 4096:
                case = c
                break
    if case is None and stages.get("cases"):
        case = stages["cases"][0]
    if case is None:
        return

    ncu_sm = (ncu or {}).get("sm_pct_by_impl", {})
    ncu_dram = (ncu or {}).get("dram_pct_by_impl", {})
    for k in case["kernels"]:
        impl = k.get("impl", "")
        if not impl.startswith("v"):
            continue
        labels.append(impl)
        if impl in ncu_dram:
            dram.append(ncu_dram[impl])
        elif hbm and "gbps" in k:
            dram.append(100.0 * k["gbps"] / hbm)
        else:
            dram.append(0.0)
        sm.append(ncu_sm.get(impl, 0.0))

    x = np.arange(len(labels))
    has_sm = any(v > 0 for v in sm)
    if has_sm:
        w = 0.36
        ax.bar(x - w / 2, dram, w, color="#4C78A8", label="DRAM throughput % of measured HBM")
        ax.bar(x + w / 2, sm, w, color="#E45756", label="SM throughput % (ncu)")
    else:
        ax.bar(x, dram, 0.6, color="#4C78A8", label="% of measured HBM (bytes/time)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percent of peak")
    ax.set_ylim(0, 100)
    gpu = stages.get("meta", {}).get("gpu", {}).get("name", "")
    title = (
        f"Stage bandwidth on {gpu}\n"
        f"% of measured {kind}, not the spec sheet and not D2D copy"
    )
    if not has_sm:
        title += "\nSM throughput omitted — ncu not collected yet"
    ax.set_title(title)
    ax.legend()
    fig.savefig(out)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = p.parse_args()
    results_dir = args.results
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pred = load_json(PREDICTION_PATH)
    ncu = None
    nc = _latest(results_dir, "ncu_*.json")
    if nc and nc.name != "ncu_log.json":
        try:
            ncu = load_json(nc)
        except Exception:
            ncu = None

    written = []
    for tag in ("T4", "A100"):
        st = results_dir / f"stages_{tag}.json"
        sw = results_dir / f"sweep_{tag}.json"
        ce = results_dir / f"ceilings_{tag}.json"
        stages = load_json(st) if st.exists() else None
        sweep = load_json(sw) if sw.exists() else None
        ceilings = load_json(ce) if ce.exists() else None
        if stages or ceilings:
            out = FIGURES_DIR / f"roofline_{tag}.png"
            plot_roofline(pred, stages, ceilings, out)
            written.append(out)
        if stages:
            out = FIGURES_DIR / f"stage_bars_{tag}.png"
            plot_stage_bars(stages, ncu, ceilings, out)
            written.append(out)
        if sweep:
            out = FIGURES_DIR / f"batch_sweep_{tag}.png"
            plot_batch_sweep(sweep, ncu, ceilings, out)
            written.append(out)
            out = FIGURES_DIR / f"batch_latency_{tag}.png"
            plot_batch_latency(sweep, out)
            written.append(out)

    # Keep untagged names pointing at T4 so older README links still resolve.
    t4_roof = FIGURES_DIR / "roofline_T4.png"
    if t4_roof.exists():
        (FIGURES_DIR / "roofline.png").write_bytes(t4_roof.read_bytes())
        written.append(FIGURES_DIR / "roofline.png")
    t4_bars = FIGURES_DIR / "stage_bars_T4.png"
    if t4_bars.exists():
        (FIGURES_DIR / "stage_bars.png").write_bytes(t4_bars.read_bytes())
        written.append(FIGURES_DIR / "stage_bars.png")

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
