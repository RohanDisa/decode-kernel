from __future__ import annotations

import os
import shutil

from setuptools import find_packages, setup


def have_nvcc() -> bool:
    if os.environ.get("DK_SKIP_CUDA") == "1":
        return False
    if shutil.which("nvcc"):
        return True
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not home:
        return False
    exe = "nvcc.exe" if os.name == "nt" else "nvcc"
    return os.path.exists(os.path.join(home, "bin", exe))


def nvcc_arch_flags() -> list[str]:
    env = os.environ.get("DK_CUDA_ARCH")
    if env:
        return [f"-gencode=arch=compute_{a},code=sm_{a}" for a in env.split(",") if a]
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}{minor}"
            return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    except Exception:
        pass
    return [
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]


this_dir = os.path.dirname(os.path.abspath(__file__))
ext_modules = []
cmdclass = {}

if have_nvcc():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    cxx = ["/O2"] if os.name == "nt" else ["-O3"]
    nvcc = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-std=c++17",
        "--expt-relaxed-constexpr",
    ]
    ccbin = os.environ.get("DK_NVCC_CCBIN") or os.environ.get("CUDAHOSTCXX")
    if ccbin:
        nvcc = ["-ccbin", ccbin] + nvcc
    nvcc += nvcc_arch_flags()
    ext_modules = [
        CUDAExtension(
            name="decode_kernel._cuda",
            sources=[
                os.path.join("csrc", "bindings.cpp"),
                os.path.join("csrc", "gemv_w4a16.cu"),
            ],
            include_dirs=[os.path.join(this_dir, "include")],
            extra_compile_args={
                "cxx": cxx,
                "nvcc": nvcc,
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
else:
    print(
        "decode-kernel: nvcc not found; installing Python package only. "
        "Set CUDA_HOME and install the CUDA toolkit to build kernels."
    )

setup(
    name="decode-kernel",
    version="0.1.0",
    description="W4A16 fused-dequant GEMV kernels for LLM decode",
    packages=find_packages("python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    install_requires=["torch", "numpy", "matplotlib"],
    extras_require={"dev": ["pytest"], "modal": ["modal"]},
)
