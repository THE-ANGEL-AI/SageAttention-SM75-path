"""
Micro-benchmark: smem_O staging vs direct output store for SM75 kernel.

Compares two output strategies in the SM75 SageAttention kernel:
  A) smem_O staging: RO_accum → smem_O → __syncthreads() → coalesced write to global
  B) Direct write: each thread writes RO_accum directly to global O (scattered)

Usage:
    python bench/bench_sm75_output_store.py

Requirements:
    - SM75 GPU (Turing: T4)
    - SageAttention built with SM75 support (both variants registered in pybind)
"""

import torch
import sys

torch.manual_seed(42)


def check_available():
    """Verify SM75 GPU + both kernel variants available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm{major}{minor}"
    print(f"GPU: {torch.cuda.get_device_name()} ({arch})")

    try:
        from sageattention import _qattn_sm75
        assert hasattr(_qattn_sm75, "qk_int8_sv_f16_accum_f32_attn_sm75")
        assert hasattr(_qattn_sm75, "qk_int8_sv_f16_accum_f32_attn_sm75_smem_o")
        print("✓ Both kernel variants found")
    except (ImportError, AssertionError):
        raise RuntimeError(
            "Both kernel variants not built. Rebuild with:\n"
            "  python setup.py build_ext --inplace"
        )


def quantize_per_warp(q, k):
    """Quantize Q/K via per_warp_int8_cuda."""
    from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
    q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(
        q.contiguous(), k.contiguous(),
        km=None, tensor_layout="HND",
        BLKQ=64, WARPQ=16, BLKK=64,
    )
    return q_int8, q_scale, k_int8, k_scale


def run_kernel(kernel_fn, q_int8, k_int8, v_fp16, o, q_scale, k_scale,
               sm_scale, is_causal=False, return_lse=False):
    """Call a kernel variant with pre-allocated output `o`."""
    _tensor_layout = 1  # HND
    _is_causal = 1 if is_causal else 0
    _qk_quant_gran = 2  # per_warp
    _return_lse = 1 if return_lse else 0
    return kernel_fn(
        q_int8, k_int8, v_fp16, o, q_scale, k_scale,
        _tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse
    )


def benchmark_variant(kernel_fn, q_int8, k_int8, v_fp16, o, q_scale, k_scale,
                      sm_scale, warmup=10, repeats=100):
    """Time a kernel variant using CUDA events."""
    # Warmup
    for _ in range(warmup):
        run_kernel(kernel_fn, q_int8, k_int8, v_fp16, o, q_scale, k_scale, sm_scale)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        run_kernel(kernel_fn, q_int8, k_int8, v_fp16, o, q_scale, k_scale, sm_scale)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeats  # ms per call


def run_benchmark():
    """Sweep seq lengths and head dims, compare both output strategies."""
    from sageattention import _qattn_sm75

    direct_fn = _qattn_sm75.qk_int8_sv_f16_accum_f32_attn_sm75
    smem_o_fn = _qattn_sm75.qk_int8_sv_f16_accum_f32_attn_sm75_smem_o

    device = torch.device("cuda")
    dtype = torch.float16

    configs = [
        # (batch, num_heads, seq_len, head_dim, label)
        (1, 4,  1024, 64,  "1×4×1K×64"),
        (1, 4,  1024, 128, "1×4×1K×128"),
        (1, 4,  2048, 64,  "1×4×2K×64"),
        (1, 4,  2048, 128, "1×4×2K×128"),
        (1, 4,  4096, 64,  "1×4×4K×64"),
        (1, 4,  4096, 128, "1×4×4K×128"),
        (1, 4,  8192, 64,  "1×4×8K×64"),
        (1, 4,  8192, 128, "1×4×8K×128"),
        (2, 8,  2048, 64,  "2×8×2K×64"),
        (2, 8,  4096, 64,  "2×8×4K×64"),
        (2, 8,  8192, 64,  "2×8×8K×64"),
        (4, 32, 1024, 128, "4×32×1K×128"),
        (4, 32, 2048, 128, "4×32×2K×128"),
    ]

    sm_scale_base = 1.0  # Will be set per config

    print("\n" + "=" * 85)
    print(f"{'Config':<18s} {'Direct (ms)':>12s} {'smem_O (ms)':>12s} {'Δ (ms)':>10s} {'Faster':>8s}")
    print("=" * 85)

    for batch, n_heads, seq_len, hd, label in configs:
        sm_scale = hd ** -0.5

        # Generate inputs
        q = torch.randn(batch, n_heads, seq_len, hd, dtype=dtype, device=device)
        k = torch.randn(batch, n_heads, seq_len, hd, dtype=dtype, device=device)
        v = torch.randn(batch, n_heads, seq_len, hd, dtype=dtype, device=device)

        # Quantize
        try:
            q_i8, qs, k_i8, ks = quantize_per_warp(q, k)
        except Exception as e:
            print(f"{label:<18s} {'SKIP (quant failed)':>40s}  {e}")
            continue

        # Pre-allocate output
        o = torch.empty(batch, n_heads, seq_len, hd, dtype=dtype, device=device)

        try:
            t_direct = benchmark_variant(
                direct_fn, q_i8, k_i8, v, o, qs, ks, sm_scale,
                warmup=5, repeats=50
            )
            t_smem_o = benchmark_variant(
                smem_o_fn, q_i8, k_i8, v, o, qs, ks, sm_scale,
                warmup=5, repeats=50
            )
        except Exception as e:
            print(f"{label:<18s} {'SKIP (kernel failed)':>40s}  {e}")
            continue

        delta = t_smem_o - t_direct
        ratio = t_smem_o / t_direct if t_direct > 0 else 0
        faster = "smem_O" if delta < 0 else "direct"

        print(f"{label:<18s} {t_direct:12.4f} {t_smem_o:12.4f} {delta:+10.4f} {faster:>8s}")

    print("=" * 85)
    print("Negative Δ = smem_O faster (better coalescing)")
    print("Positive Δ = direct faster (fewer syncs, less smem pressure)")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("SM75 Output Store Micro-Benchmark")
    print("smem_O staging vs direct global write")
    print("=" * 60)

    try:
        check_available()
    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)

    run_benchmark()
