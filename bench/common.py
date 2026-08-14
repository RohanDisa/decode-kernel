"""Shared benchmark helpers: CUDA events, L2 flush, GPU identity, JSON."""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import torch


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"name": "none", "capability": None}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": int(props.total_memory),
        "multi_processor_count": int(props.multi_processor_count),
        "l2_cache_bytes": int(getattr(props, "L2_cache_size", 0)),
        "clocks_locked": _clocks_locked(),
        "nvidia_smi": _smi_query(),
    }


def _smi_query() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,clocks.gr,clocks.mem,clocks_throttle_reasons.active",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).strip()
        return {"raw": out}
    except Exception as exc:
        return {"error": str(exc)}


def _clocks_locked() -> bool:
    # Application clocks are often forbidden on consumer / notebook SKUs.
    return False


def make_l2_flush(bytes_min: int = 64 * 1024 * 1024) -> torch.Tensor:
    """Buffer larger than T4 (4 MB) and A100 (40 MB) L2."""
    n = max(bytes_min, 64 * 1024 * 1024) // 4
    return torch.empty(n, device="cuda", dtype=torch.float32)


def flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


def time_cuda(
    fn: Callable[[], None],
    warmup: int = 50,
    iters: int = 1000,
    flush: Optional[torch.Tensor] = None,
    flush_every: bool = True,
) -> dict:
    """Time `fn` with CUDA events. Returns median/IQR latency in microseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_ms: list[float] = []

    for _ in range(warmup):
        if flush is not None and flush_every:
            flush_l2(flush)
        fn()
    torch.cuda.synchronize()

    for _ in range(iters):
        if flush is not None and flush_every:
            flush_l2(flush)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))

    us = sorted(ms * 1000.0 for ms in samples_ms)
    n = len(us)
    median = us[n // 2]
    q1 = us[n // 4]
    q3 = us[(3 * n) // 4]
    return {
        "median_us": median,
        "iqr_us": q3 - q1,
        "q1_us": q1,
        "q3_us": q3,
        "mean_us": statistics.fmean(us),
        "stdev_us": statistics.pstdev(us) if n > 1 else 0.0,
        "min_us": us[0],
        "max_us": us[-1],
        "warmup": warmup,
        "iters": iters,
        "l2_flushed": flush is not None,
    }


def bytes_moved_w4(m: int, k: int, batch: int) -> int:
    """Theoretical HBM traffic if W is read once: packed W + scales + X + Y."""
    packed = m * (k // 2)
    scales = m * (k // 128) * 2
    x = batch * k * 2
    y = batch * m * 2
    return packed + scales + x + y


def bytes_moved_fp16(m: int, k: int, batch: int) -> int:
    return m * k * 2 + batch * k * 2 + batch * m * 2


def flop_gemv(m: int, k: int, batch: int) -> int:
    return 2 * m * k * batch


def arithmetic_intensity_w4(m: int, k: int, batch: int) -> float:
    return flop_gemv(m, k, batch) / bytes_moved_w4(m, k, batch)


def hbm_roof(ceilings: dict | None) -> tuple[float | None, str]:
    """GEMV hit-rate denominator: read-only STREAM, else legacy copy."""
    if not ceilings:
        return None, "none"
    read = ceilings.get("hbm_read") or {}
    if "gbps" in read:
        return float(read["gbps"]), read.get("kind", "stream_read_sum")
    copy = ceilings.get("hbm_copy") or {}
    if "gbps" in copy:
        return float(copy["gbps"]), copy.get("kind", "d2d_copy")
    return None, "none"


def annotate_hit_rates(payload: dict, ceilings: dict, over_read_flag_pct: float = 90.0) -> dict:
    """Attach % of copy and % of read to each timed kernel. Does not retouch latency."""
    read_gbps, read_kind = hbm_roof(ceilings)
    copy = ceilings.get("hbm_copy") or {}
    copy_gbps = copy.get("gbps")
    payload["ceiling_denominator"] = "read" if ceilings.get("hbm_read") else "copy"
    payload["ceiling_kind"] = read_kind
    payload["hbm_read_gbps"] = read_gbps
    payload["hbm_copy_gbps"] = copy_gbps
    for case in payload.get("cases", []):
        for k in case.get("kernels", []):
            gbps = k.get("gbps")
            if gbps is None:
                continue
            if copy_gbps:
                k["hbm_pct_of_copy"] = 100.0 * gbps / copy_gbps
            if read_gbps:
                pct = 100.0 * gbps / read_gbps
                k["hbm_pct_of_read"] = pct
                k["hbm_pct"] = pct
                if k.get("precision") == "fp16" and pct > over_read_flag_pct:
                    k["flag_over_roof"] = True
                    k["flag_note"] = (
                        f"{pct:.1f}% of read-only STREAM exceeds {over_read_flag_pct:.0f}%; "
                        "do not report as a hit of the roof"
                    )
    return payload


@dataclass
class RunMeta:
    gpu: dict
    host: str
    platform: str
    torch: str
    cuda: Optional[str]
    timestamp: str
    git: Optional[str]


def collect_meta() -> RunMeta:
    git = None
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        pass
    return RunMeta(
        gpu=gpu_info(),
        host=platform.node(),
        platform=platform.platform(),
        torch=torch.__version__,
        cuda=getattr(torch.version, "cuda", None),
        timestamp=datetime.now(timezone.utc).isoformat(),
        git=git,
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PREDICTION_PATH = Path(__file__).resolve().parent / "prediction.json"
