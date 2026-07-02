from .core import sageattn, sageattn_varlen
from .core import sageattn_qk_int8_pv_fp16_triton
# SM75 CUDA kernel (INT8 QK + FP16 PV)
try:
    from .core import sageattn_qk_int8_pv_fp16_cuda_sm75
    SM75_CUDA_ENABLED = True
except ImportError:
    SM75_CUDA_ENABLED = False
# Other arch kernels — not built on SM75-only, but keep for compatibility
try:
    from .core import sageattn_qk_int8_pv_fp8_cuda
except ImportError:
    pass

__version__ = "2.1.1" # Or update if making a new release
