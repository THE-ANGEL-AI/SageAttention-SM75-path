"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import subprocess
from packaging.version import parse, Version
from typing import List, Set
import warnings

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# Flags to track detected architectures
HAS_SM75 = False
HAS_SM80 = False
HAS_SM75 = False
HAS_SM86 = False
HAS_SM89 = False
HAS_SM90 = False
HAS_SM120 = False # Assuming 12.0 refers to Blackwell Gen 1 for setup purposes

# Supported NVIDIA GPU architectures.
# Removed specific version checks here, will do dynamically
# SUPPORTED_ARCHS = {"7.5", "8.0", "8.6", "8.9", "9.0", "12.0"} # Example, actual check below

# Common compiler flags.
COMMON_CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17"]
COMMON_NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    "--threads=8", # Adjust thread count if needed
    "-Xptxas=-v",
    "-diag-suppress=174", # suppress the specific warning
]

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_path = os.path.join(cuda_dir, "bin", "nvcc")
    if not os.path.exists(nvcc_path):
         nvcc_path = os.path.join(cuda_dir, "bin", "nvcc.exe") # Handle Windows path
    if not os.path.exists(nvcc_path):
        raise RuntimeError(f"nvcc not found at {nvcc_path}")

    try:
        nvcc_output = subprocess.check_output([nvcc_path, "-V"],
                                              universal_newlines=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run nvcc -V: {e.output}") from e
    
    output = nvcc_output.split()
    release_idx = -1
    for i, s in enumerate(output):
        if "release" in s:
            release_idx = i + 1
            break
    if release_idx == -1:
        raise RuntimeError("Could not parse nvcc -V output for CUDA version")
    
    version_str = output[release_idx].split(",")[0]
    nvcc_cuda_version = parse(version_str)
    return nvcc_cuda_version

# Get CUDA version
nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
print(f"Detected CUDA version: {nvcc_cuda_version}")

# Base CUDA requirement (adjust if necessary for SM75, e.g., 11.0)
MIN_CUDA_VERSION = Version("11.8") # Let's try 11.8 as a baseline supporting up to Ampere + some Ada/Hopper features
if nvcc_cuda_version < MIN_CUDA_VERSION:
    raise RuntimeError(f"CUDA {MIN_CUDA_VERSION} or higher is required to build the package (detected {nvcc_cuda_version}).")

# Iterate over all GPUs on the current machine.
# Iterate over all GPUs on the current machine. Also you can modify this part to specify the architecture if you want to build for specific GPU architectures.
compute_capabilities = set()
detected_cc = set() # Store detected CCs for later validation
device_count = torch.cuda.device_count()
if device_count > 0:
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        cc = f"{major}.{minor}"
        detected_cc.add(cc)
        if major < 7 or (major == 7 and minor < 5):
            warnings.warn(f"Skipping GPU {i} with compute capability {cc}. Requires SM75 (Turing) or higher.")
            continue
        # Allow building for detected capable GPUs
        compute_capabilities.add(cc)
else:
    warnings.warn("No CUDA GPUs were detected. Building kernels requires a CUDA-enabled device. "
                  "Attempting to build for common architectures [7.5, 8.0, 8.6, 8.9, 9.0]. "
                  "Specify TORCH_CUDA_ARCH_LIST environment variable to override.")
    # Fallback: Build for a common set if no GPUs detected
    # TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" # Example for env var
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0")
    compute_capabilities.update(arch.strip() for arch in arch_list_env.split(";") if arch.strip())


if not compute_capabilities:
    # This should only happen if detection failed AND fallback failed
     raise RuntimeError("No suitable compute capabilities found or specified. "
                        "Need SM75+ and CUDA environment.")

print(f"Building for compute capabilities: {compute_capabilities}")

# Architecture-specific flags and checks
arch_flags = []
enable_bf16 = True # Enable by default, disable specifically for SM75

for capability in sorted(list(compute_capabilities)):
    major, minor = map(int, capability.split('.'))
    arch_flag_base = f"{major}{minor}"

    # Architecture-specific CUDA version checks
# Architecture-specific flags and checks
arch_flags = []
enable_bf16 = True # Enable by default, disable specifically for SM75

for capability in sorted(list(compute_capabilities)):
    major, minor = map(int, capability.split('.'))
    arch_flag_base = f"{major}{minor}"

    # Architecture-specific CUDA version checks
    if major == 7 and minor == 5:
        HAS_SM75 = True
        # SM75 should work with CUDA 11.8+, no higher requirement needed from base
        enable_bf16 = False # Turing does not support BF16
        arch_flags.append("-gencode")
        arch_flags.append("arch=compute_75,code=sm_75")
    elif major == 8 and minor == 0:
        HAS_SM80 = True
        # SM80 requires CUDA 11.1+, covered by base 11.8
        arch_flags.append("-gencode",`n        f"arch=compute_80,code=sm_80")
    elif major == 8 and minor == 6:
        HAS_SM86 = True
        # SM86 requires CUDA 11.1+, covered by base 11.8
        arch_flags.append("-gencode",`n        f"arch=compute_86,code=sm_86")
    elif major == 8 and minor == 9:
        HAS_SM89 = True
        if nvcc_cuda_version < Version("12.0"): # FP8 MMA support added later, requires newer compiler
             # Relaxing slightly from 12.4, needs testing. Stricter check if issues arise.
             raise RuntimeError(f"Compute capability 8.9 requires CUDA 12.0 or higher (detected {nvcc_cuda_version}).")
        arch_flags.append("-gencode",`n        f"arch=compute_89,code=sm_89")
    elif major == 9 and minor == 0:
        HAS_SM90 = True
        if nvcc_cuda_version < Version("12.0"): # WGMMA requires CUDA 12.0+
            raise RuntimeError(f"Compute capability 9.0 requires CUDA 12.0 or higher (detected {nvcc_cuda_version}).")
        # Use 90a for WGMMA
        arch_flags.append("-gencode",`n        f"arch=compute_90,code=sm_90a")
    # Placeholder for future Blackwell check (SM120 / compute_120 ?)
    # elif major == 12 and minor == 0:
    #     HAS_SM120 = True
    #     if nvcc_cuda_version < Version("12.8"): # Example requirement
    #         raise RuntimeError(f"Compute capability 12.0 requires CUDA 12.8 or higher (detected {nvcc_cuda_version}).")
    #     arch_flags.append("-gencode",`n        f"arch=compute_120,code=sm_120a") # Hypothetical flag
    else:
        warnings.warn(f"Unhandled compute capability: {capability}. Skipping specific flags.")
        # Add generic PTX compilation for forward compatibility if desired
        # arch_flags.append(f"-gencode arch=compute_{arch_flag_base},code=compute_{arch_flag_base}")


# Common flags setup
CXX_FLAGS = COMMON_CXX_FLAGS[:]
NVCC_FLAGS = COMMON_NVCC_FLAGS[:]

# Add BF16 flag if needed by any selected architecture
if enable_bf16:
    CXX_FLAGS.append("-DENABLE_BF16")
    NVCC_FLAGS.append("-DENABLE_BF16")
else:
    print("BF16 support disabled (SM75 detected or only SM75 specified).")


# Add ABI flag
ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

# Append architecture-specific flags
NVCC_FLAGS.extend(arch_flags)

# Common flags setup
CXX_FLAGS = COMMON_CXX_FLAGS[:]
NVCC_FLAGS = COMMON_NVCC_FLAGS[:]

# Add BF16 flag if needed by any selected architecture
if enable_bf16:
    CXX_FLAGS.append("-DENABLE_BF16")
    NVCC_FLAGS.append("-DENABLE_BF16")
else:
    print("BF16 support disabled (SM75 detected or only SM75 specified).")


# Add ABI flag
ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

# Append architecture-specific flags
NVCC_FLAGS.extend(arch_flags)

# --- Define Extensions ---
ext_modules = []
# SM75 Kernel (INT8 QK, FP16 PV, FP32 Accum) - Only if SM75 is targeted
if HAS_SM75:
    qattn_sm75_extension = CUDAExtension(
        name="sageattention._qattn_sm75",
        sources=[
            "csrc/qattn/pybind_sm75.cpp",
            "csrc/qattn/qk_int_sv_f16_cuda_sm75.cu", # Needs to be created
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(qattn_sm75_extension)


ext_modules = []

# SM75 Kernel (INT8 QK, FP16 PV, FP32 Accum) - Only if SM75 is targeted
if HAS_SM75:
    qattn_sm75_extension = CUDAExtension(
        name="sageattention._qattn_sm75",
        sources=[
            "csrc/qattn/pybind_sm75.cpp",
            "csrc/qattn/qk_int_sv_f16_cuda_sm75.cu", # Needs to be created
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(qattn_sm75_extension)


# SM80+ Kernels (INT8 QK, FP16 PV) - Only if SM80+ is targeted
if HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM120: # Check all potentially capable archs
    qattn_sm80_extension = CUDAExtension(
        name="sageattention._qattn_sm80",
        sources=[
            "csrc/qattn/pybind_sm80.cpp",
            "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(qattn_sm80_extension)

# SM89+ Kernels (INT8 QK, FP8 PV) - Only if SM89+ is targeted
if HAS_SM89 or HAS_SM120: # Check all potentially capable archs
    qattn_sm89_extension = CUDAExtension(
        name="sageattention._qattn_sm89",
        sources=[
            "csrc/qattn/pybind_sm89.cpp",
            "csrc/qattn/qk_int_sv_f8_cuda_sm89.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(qattn_sm89_extension)

# SM90 Kernels (INT8 QK, FP8 PV, WGMMA) - Only if SM90 is targeted
if HAS_SM90:
    qattn_sm90_extension = CUDAExtension(
        name="sageattention._qattn_sm90",
        sources=[
            "csrc/qattn/pybind_sm90.cpp",
            "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
        # Link CUDA runtime library explicitly if needed for WGMMA/TMA features
        # Check if this is needed based on compiler errors
        extra_link_args=['-lcuda'], 
    )
    ext_modules.append(qattn_sm90_extension)

# Fused kernels (Quantization, etc.) - Always include if building any extension
if ext_modules: # Only build fused if we are building any qattn kernel
    fused_extension = CUDAExtension(
        name="sageattention._fused",
        sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(fused_extension)
else:
    print("No compatible GPU architecture detected or specified for SageAttention kernels. Skipping kernel compilation.")


# --- Setup Call ---
setup(
    name='sageattention',
    version='2.1.1', # Keep version or update as needed
    author='SageAttention team',
    license='Apache 2.0 License',
    description='Accurate and efficient plug-and-play low-bit attention.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thu-ml/SageAttention',
    packages=find_packages(),
    python_requires='>=3.9',
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[ # Add dependencies here if any are needed besides torch/triton
        'torch>=2.0.0', # Example: Specify torch version if needed
        # 'triton>=...' # Specify Triton version if needed by Python code
    ],


)
