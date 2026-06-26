#!/bin/bash
# ============================================================
# Build SageAttention SM75 for Kaggle T4×2
# ============================================================
# Run from the SageAttention-SM75-path/ project root.
#
# Usage:
#   chmod +x scripts/build_sm75_kaggle.sh
#   bash scripts/build_sm75_kaggle.sh
# ============================================================
set -e

echo "========================================="
echo "SageAttention SM75 Build for Kaggle T4×2"
echo "========================================="

# 1. Verify CUDA environment
echo ""
echo "[1/4] Checking CUDA..."
nvcc --version || { echo "ERROR: nvcc not found. Is CUDA installed?"; exit 1; }
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "print(f'GPU count: {torch.cuda.device_count()}')"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} (sm{major}{minor})')
"

# 2. Verify Kaggle CUDA version monkey-patch is in place
echo ""
echo "[2/4] Verifying CUDA version monkey-patch..."
if grep -q "_check_cuda_version" setup.py 2>/dev/null; then
    echo "  ✓ Monkey-patch found in setup.py"
else
    echo "  ✗ WARNING: Monkey-patch not found in setup.py"
    echo "    Kaggle may fail to build due to CUDA version mismatch."
fi

# 3. Build extensions
echo ""
echo "[3/4] Building SageAttention SM75 extensions..."
cd "$(dirname "$0")/.."  # Go to project root

# Clean any stale build artifacts
rm -rf build/ dist/ *.egg-info 2>/dev/null || true

# Build in-place (extensions go to sageattention/ directory)
python setup.py build_ext --inplace 2>&1 | tee build_sm75.log

# 4. Verify build
echo ""
echo "[4/4] Verifying SM75 modules..."
python -c "
import torch
from sageattention import _qattn_sm75, _fused
print('  ✓ _qattn_sm75 loaded')
print('  ✓ _fused loaded')
print('  Available functions:')
for attr in dir(_qattn_sm75):
    if not attr.startswith('_'):
        print(f'    - {attr}')
" && echo "" && echo "=========================================" && echo "BUILD SUCCESSFUL" && echo "========================================="

echo ""
echo "Next steps:"
echo "  1. Verify correctness:  python test_sm75_kernel.py"
echo "  2. Run benchmark:       python bench/bench_sm75_output_store.py"
echo "  3. Integrate in ComfyUI: copy scripts/ltxv_sageattn_patch.py"
echo "     to ComfyUI/custom_nodes/ComfyUI-LTXVideo/"
