"""Remote GPU runner via Modal. Local `python -m bench.bench` is unchanged.

The CUDA sources live in ``csrc/`` (this repo has no ``src/``). They are mounted
at ``/root/src`` inside the container, and ``bench/`` at ``/root/bench``, plus
the extra files ``setup.py`` needs so the PyTorch extension can link.

Usage (from the repo root, after `pip install modal && modal setup`):

    modal run bench/modal_runner.py --gpu T4
    modal run bench/modal_runner.py --gpu A10G --mode all
    DK_MODAL_GPU=A100 modal run bench/modal_runner.py

JSON lands in local ``bench/results/`` so ``python -m bench.plot`` works as-is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# This repo's CUDA lives in csrc/. Mounted remotely as /root/src as requested.
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = REPO_ROOT / "csrc"
LOCAL_BENCH = REPO_ROOT / "bench"
REMOTE_ROOT = "/root/decode-kernel"
REMOTE_SRC = "/root/src"
REMOTE_BENCH = "/root/bench"

GPU_TYPES = {
    "T4": {"modal_gpu": "T4", "arch": "sm_75"},
    "A10G": {"modal_gpu": "A10G", "arch": "sm_86"},
    "A100": {"modal_gpu": "A100", "arch": "sm_80"},
}

# index_url (not extra_index_url): PyPI's current torch is CUDA 13.0 and would
# win over the cu124 extra index, which is what broke the first Modal run.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("build-essential", "g++")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "PYTHONPATH": REMOTE_ROOT,
            "CC": "gcc",
            "CXX": "g++",
            "CUDAHOSTCXX": "g++",
            "DK_NVCC_CCBIN": "g++",
        }
    )
    .pip_install("numpy", "setuptools", "wheel", "ninja", "packaging")
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(str(LOCAL_SRC), remote_path=REMOTE_SRC)
    .add_local_dir(
        str(LOCAL_BENCH),
        remote_path=REMOTE_BENCH,
        ignore=["results", "figures", "__pycache__"],
    )
    # Repo layout so `pip install -e .` and `python -m bench.bench` resolve.
    .add_local_dir(str(LOCAL_SRC), remote_path=f"{REMOTE_ROOT}/csrc")
    .add_local_dir(
        str(LOCAL_BENCH),
        remote_path=f"{REMOTE_ROOT}/bench",
        ignore=["results", "figures", "__pycache__"],
    )
    .add_local_dir(str(REPO_ROOT / "include"), remote_path=f"{REMOTE_ROOT}/include")
    .add_local_dir(str(REPO_ROOT / "python"), remote_path=f"{REMOTE_ROOT}/python")
    .add_local_file(str(REPO_ROOT / "setup.py"), remote_path=f"{REMOTE_ROOT}/setup.py")
    .add_local_file(
        str(REPO_ROOT / "pyproject.toml"), remote_path=f"{REMOTE_ROOT}/pyproject.toml"
    )
    .add_local_file(str(REPO_ROOT / "README.md"), remote_path=f"{REMOTE_ROOT}/README.md")
    .add_local_file(str(REPO_ROOT / "LICENSE"), remote_path=f"{REMOTE_ROOT}/LICENSE")
)

app = modal.App("decode-kernel", image=image)


def resolve_gpu(name: str) -> dict:
    key = name.strip().upper().replace(" ", "")
    aliases = {"A100-40GB": "A100", "A10040GB": "A100", "A100-80GB": "A100"}
    key = aliases.get(key, key)
    if key not in GPU_TYPES:
        allowed = ", ".join(GPU_TYPES)
        raise SystemExit(f"unknown GPU {name!r}; choose one of: {allowed}")
    spec = dict(GPU_TYPES[key])
    spec["gpu_type"] = key
    return spec


def _run(
    cmd: list[str], log: list[str], cwd: str | None = None, env: dict | None = None
) -> None:
    log.append("$ " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        log.append(proc.stdout.rstrip())
    if proc.stderr:
        log.append(proc.stderr.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
        )


def _smi_identity() -> dict:
    driver = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    full = subprocess.check_output(["nvidia-smi"], text=True)
    return {"query": driver, "nvidia_smi": full}


def _stamp(payload: dict, gpu_type: str, arch: str, smi: dict, nvcc_log: str) -> dict:
    meta = payload.setdefault("meta", {})
    meta["gpu_type"] = gpu_type
    meta["nvcc_arch"] = arch
    meta["runner"] = "modal"
    meta["driver_version"] = smi["query"]
    meta["nvidia_smi"] = smi["nvidia_smi"]
    meta["nvcc_compile"] = nvcc_log
    return payload


def _torch_arch(sm: str) -> str:
    """sm_75 -> 7.5 for TORCH_CUDA_ARCH_LIST."""
    digits = sm.replace("sm_", "")
    return f"{digits[0]}.{digits[1:]}"


def _compiler_env(arch: str) -> dict[str, str]:
    """Match the CUDA 12.4 image: g++ host compiler, nvcc on PATH."""
    env = os.environ.copy()
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["CUDAHOSTCXX"] = "g++"
    env["DK_NVCC_CCBIN"] = "g++"
    env["CC"] = "gcc"
    env["CXX"] = "g++"
    env["DK_CUDA_ARCH"] = arch.replace("sm_", "")
    env["TORCH_CUDA_ARCH_LIST"] = _torch_arch(arch)
    env["PYTHONPATH"] = REMOTE_ROOT
    # Modal's standalone Python puts clang ahead of g++; nvcc must not pick it up.
    env["PATH"] = "/usr/bin:/usr/local/cuda/bin:" + env.get("PATH", "")
    return env


@app.function(gpu="T4", timeout=2 * 60 * 60)
def compile_and_bench(
    gpu_type: str,
    arch: str,
    mode: str,
    quick: bool,
    shapes: str,
) -> dict:
    """Compile with nvcc -O3 -arch=<sm_XX>, run the existing harness, return JSON."""
    log: list[str] = []
    env = _compiler_env(arch)

    smi = _smi_identity()
    log.append(f"gpu_type={gpu_type} nvcc_arch={arch}")
    log.append(smi["query"])
    log.append(smi["nvidia_smi"])

    cu = f"{REMOTE_SRC}/gemv_w4a16.cu"
    include = f"{REMOTE_ROOT}/include"
    nvcc_cmd = [
        "nvcc",
        "-O3",
        f"-arch={arch}",
        "-ccbin",
        "g++",
        "-std=c++17",
        "--use_fast_math",
        "-lineinfo",
        "-Xcompiler",
        "-fPIC",
        "-I",
        include,
        "-c",
        cu,
        "-o",
        "/tmp/gemv_w4a16.o",
    ]
    _run(nvcc_cmd, log, env=env)
    nvcc_log = " ".join(nvcc_cmd)

    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            REMOTE_ROOT,
            "--no-build-isolation",
            "--no-deps",
            "-q",
        ],
        log,
        env=env,
    )

    out_dir = Path("/tmp/dk-results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = gpu_type
    produced: list[Path] = []

    def py_mod(*args: str) -> None:
        _run([sys.executable, "-m", *args], log, cwd=REMOTE_ROOT, env=env)

    run_ceilings = mode in {"ceilings", "bench", "all", "diag"}
    run_stages = mode in {"bench", "all"}
    run_sweep = mode in {"sweep", "all"}
    run_stamp = mode in {"ceilings", "diag", "all"}
    run_diag = mode == "diag"

    if run_ceilings:
        dest = out_dir / f"ceilings_{tag}.json"
        py_mod("bench.ceilings", "--out", str(dest))
        produced.append(dest)

    if run_stamp:
        dest = out_dir / f"stamp_{tag}.json"
        py_mod("bench.stamp_check", "--out", str(dest))
        produced.append(dest)

    if run_stages:
        dest = out_dir / f"stages_{tag}.json"
        cmd = [
            "bench.bench",
            "--shapes",
            shapes,
            "--versions",
            "0,1,2,3,4,5",
            "--out",
            str(dest),
        ]
        if quick:
            cmd.append("--quick")
        py_mod(*cmd)
        produced.append(dest)

    if run_sweep:
        dest = out_dir / f"sweep_{tag}.json"
        cmd = ["bench.bench", "--sweep", "--shapes", shapes, "--out", str(dest)]
        if quick:
            cmd.append("--quick")
        else:
            cmd.extend(["--iters", "200"])
        py_mod(*cmd)
        produced.append(dest)

    if run_diag:
        # M=11008 rows, K=4096. Same bytes-ish as 4096x11008 but ~2.7x the warps.
        dest = out_dir / f"diag_rows_{tag}.json"
        cmd = [
            "bench.bench",
            "--shapes",
            "11008x4096",
            "--versions",
            "1",
            "--out",
            str(dest),
        ]
        if quick:
            cmd.append("--quick")
        py_mod(*cmd)
        produced.append(dest)

        dest = out_dir / f"diag_wpr2_{tag}.json"
        cmd = [
            "bench.bench",
            "--shapes",
            "4096x4096",
            "--versions",
            "1",
            "--warps-per-row",
            "2",
            "--out",
            str(dest),
        ]
        if quick:
            cmd.append("--quick")
        py_mod(*cmd)
        produced.append(dest)

        dest = out_dir / f"diag_wpr4_{tag}.json"
        cmd = [
            "bench.bench",
            "--shapes",
            "4096x4096",
            "--versions",
            "1",
            "--warps-per-row",
            "4",
            "--out",
            str(dest),
        ]
        if quick:
            cmd.append("--quick")
        py_mod(*cmd)
        produced.append(dest)

    files = {}
    for path in produced:
        payload = json.loads(path.read_text())
        _stamp(payload, gpu_type, arch, smi, nvcc_log)
        text = json.dumps(payload, indent=2) + "\n"
        path.write_text(text)
        files[path.name] = text

    return {
        "stdout": "\n".join(log) + "\n",
        "files": files,
        "gpu_type": gpu_type,
        "nvcc_arch": arch,
        "driver_version": smi["query"],
        "nvidia_smi": smi["nvidia_smi"],
    }


@app.local_entrypoint()
def main(
    gpu: str = "",
    mode: str = "bench",
    quick: bool = False,
    shapes: str = "4096x4096",
):
    """Run kernels on a remote GPU and write JSON into local bench/results/."""
    gpu_name = gpu or os.environ.get("DK_MODAL_GPU", "T4")
    spec = resolve_gpu(gpu_name)
    mode = mode.lower()
    if mode not in {"bench", "ceilings", "sweep", "all", "diag"}:
        raise SystemExit("mode must be bench, ceilings, sweep, all, or diag")

    print(
        f"Modal: gpu={spec['gpu_type']} arch={spec['arch']} "
        f"mode={mode} quick={quick}"
    )
    result = compile_and_bench.with_options(gpu=spec["modal_gpu"]).remote(
        gpu_type=spec["gpu_type"],
        arch=spec["arch"],
        mode=mode,
        quick=quick,
        shapes=shapes,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in result["files"].items():
        dest = RESULTS_DIR / name
        dest.write_text(text)
        print(f"wrote {dest}")

    sys.stdout.write(result["stdout"])
    print(
        f"SKU {result['gpu_type']} ({result['nvcc_arch']})  "
        f"driver {result['driver_version']}"
    )
