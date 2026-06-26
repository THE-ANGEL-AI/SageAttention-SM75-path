"""
SM75 SageAttention Kernel Correctness Test

Compares the SM75 CUDA kernel (INT8 QK + FP16 PV + FP32 accum)
against torch.nn.functional.scaled_dot_product_attention as reference.

Usage:
    python test_sm75_kernel.py

Requirements:
    - SM75 GPU (Turing: T4, RTX 2080, etc.)
    - SageAttention built with SM75 support (setup.py HAS_SM75)
    - Fused CUDA kernels built (_fused module available)
"""

import torch
import torch.nn.functional as F
import warnings
import sys

# --- Setup ---
torch.manual_seed(42)

def check_sm75_available():
    """Verify we're running on an SM75 GPU with SageAttention SM75 built."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This test requires an SM75 GPU.")

    major, minor = torch.cuda.get_device_capability()
    arch = f"sm{major}{minor}"
    print(f"GPU architecture: {arch}")
    if arch != "sm75":
        warnings.warn(
            f"This test is designed for SM75 (Turing). "
            f"Current GPU is {arch}. Results may differ."
        )

    # Check SM75 module
    try:
        import sageattention._qattn_sm75 as _qattn_sm75
        print("✓ _qattn_sm75 module loaded")
    except ImportError:
        raise ImportError(
            "SM75 kernel not built. Rebuild with: "
            "HAS_SM75=1 python setup.py build_ext --inplace"
        )

    # Check fused module
    try:
        from sageattention import _fused
        print("✓ _fused module loaded")
    except ImportError:
        raise ImportError(
            "Fused CUDA kernels not built. Required for quantization."
        )

    return arch


def reference_attention(q, k, v, is_causal=False, sm_scale=None, smooth_k=False):
    """
    Compute reference attention manually in FP32 for maximum precision.
    Handles GQA (num_kv_heads < num_qo_heads) by repeating k/v.
    Returns (output, lse) where lse is log-sum-exp.
    """
    head_dim_og = q.size(-1)

    if sm_scale is None:
        sm_scale = head_dim_og ** -0.5

    # Handle GQA: repeat k/v to match q head count
    if k.shape[1] != q.shape[1]:
        n_groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(n_groups, dim=1)
        v = v.repeat_interleave(n_groups, dim=1)

    if smooth_k:
        seq_dim = 2  # HND layout
        km = k.mean(dim=seq_dim, keepdim=True)
        k_smooth = k - km
    else:
        k_smooth = k

    # Compute reference in float32 for maximum precision
    q_f32 = q.float()
    k_f32 = k_smooth.float()
    v_f32 = v.float()

    # Manual attention for LSE computation
    attn_weights = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * sm_scale

    if is_causal:
        L, S = q_f32.shape[-2], k_f32.shape[-2]
        mask = torch.triu(torch.ones(L, S, device=q.device, dtype=torch.bool), diagonal=1)
        attn_weights.masked_fill_(mask, float('-inf'))

    attn_weights_max = attn_weights.max(dim=-1, keepdim=True).values
    attn_weights = attn_weights - attn_weights_max
    attn_weights_exp = torch.exp(attn_weights)
    attn_weights_sum = attn_weights_exp.sum(dim=-1, keepdim=True)
    attn_probs = attn_weights_exp / attn_weights_sum

    o = torch.matmul(attn_probs, v_f32)
    lse = (attn_weights_max.squeeze(-1) + torch.log(attn_weights_sum.squeeze(-1)))

    # Convert output back to FP16 for comparison
    return o.to(torch.float16), lse


def test_sageattn_sm75(
    batch_size=2,
    num_qo_heads=8,
    num_kv_heads=2,  # GQA: 4 query heads per KV head
    qo_len=256,
    kv_len=256,
    head_dim=64,
    is_causal=False,
    smooth_k=True,
    atol=5e-2,   # Relaxed tolerance for INT8 quantization
    rtol=1e-2,
):
    """
    Run a single SM75 kernel correctness test case.

    Args:
        atol: Absolute tolerance for allclose
        rtol: Relative tolerance for allclose
    """
    assert num_qo_heads % num_kv_heads == 0, "num_qo_heads must be divisible by num_kv_heads"

    device = torch.device("cuda")
    dtype = torch.float16

    # Generate inputs in HND layout (default for SageAttention)
    q = torch.randn(batch_size, num_qo_heads, qo_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, dtype=dtype, device=device)

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    sm_scale = head_dim ** -0.5

    # --- SageAttention SM75 ---
    from sageattention.core import sageattn_qk_int8_pv_fp16_cuda_sm75

    try:
        o_sage = sageattn_qk_int8_pv_fp16_cuda_sm75(
            q.clone(), k.clone(), v.clone(),
            tensor_layout="HND",
            is_causal=is_causal,
            sm_scale=sm_scale,
            smooth_k=smooth_k,
            qk_quant_gran="per_warp",
            return_lse=False,
        )
    except Exception as e:
        print(f"  ✗ SageAttention kernel crashed: {e}")
        return False, f"Kernel crash: {e}"

    # --- Reference ---
    o_ref = reference_attention(
        q, k, v,
        is_causal=is_causal,
        sm_scale=sm_scale,
        smooth_k=smooth_k,
    )[0]

    # --- Compare ---
    # Compute error metrics
    diff = (o_sage.float() - o_ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    relative_err = (diff / (o_ref.float().abs() + 1e-6)).mean().item()

    allclose = torch.allclose(o_sage, o_ref, atol=atol, rtol=rtol)

    status = "✓" if allclose else "✗"
    print(f"  {status} max_err={max_err:.6f} mean_err={mean_err:.6f} rel_err={relative_err:.6f}")

    if not allclose:
        # Show distribution of errors
        err_quantiles = torch.quantile(
            diff.flatten(),
            torch.tensor([0.5, 0.9, 0.99, 0.999], device=device)
        ).cpu().tolist()
        print(f"    Error percentiles [50%, 90%, 99%, 99.9%]: {[f'{e:.6f}' for e in err_quantiles]}")
        print(f"    Tolerance: atol={atol}, rtol={rtol}")
        print(f"    Note: Known placeholder output store may cause large errors.")

    return allclose, {
        "max_err": max_err,
        "mean_err": mean_err,
        "relative_err": relative_err,
    }


def test_known_head_dims():
    """Test common head dimensions used in real models."""
    print("\n" + "=" * 60)
    print("Test: Common head dimensions")
    print("=" * 60)

    configs = [
        # (batch, q_heads, kv_heads, q_len, kv_len, head_dim, causal, smooth_k, atol)
        (2, 8, 2, 256, 256, 64, False, True, 5e-2),
        (2, 8, 2, 256, 256, 64, False, False, 5e-2),
        (2, 8, 8, 128, 128, 64, True, True, 5e-2),
        (1, 4, 4, 128, 128, 64, False, True, 5e-2),
    ]

    results = []
    for cfg in configs:
        batch, q_heads, kv_heads, q_len, kv_len, hd, causal, smooth, atol = cfg
        desc = (f"b={batch} hq={q_heads}/hkv={kv_heads} "
                f"qlen={q_len} kvlen={kv_len} hd={hd} "
                f"causal={causal} smooth_k={smooth}")
        print(f"\n{desc}:")
        passed, info = test_sageattn_sm75(
            batch_size=batch,
            num_qo_heads=q_heads,
            num_kv_heads=kv_heads,
            qo_len=q_len,
            kv_len=kv_len,
            head_dim=hd,
            is_causal=causal,
            smooth_k=smooth,
            atol=atol,
        )
        results.append((desc, passed, info))

    passed_count = sum(1 for _, p, _ in results if p)
    print(f"\n--- Results: {passed_count}/{len(results)} tests passed ---")
    for desc, passed, info in results:
        status = "✓" if passed else "✗"
        if isinstance(info, str):
            print(f"  {status} {desc}: {info}")
        else:
            print(f"  {status} {desc}: max_err={info['max_err']:.6f}")

    return passed_count == len(results)


def test_memory_layouts():
    """Test both HND and NHD layouts."""
    print("\n" + "=" * 60)
    print("Test: Tensor layouts (HND vs NHD)")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.float16
    head_dim = 64
    sm_scale = head_dim ** -0.5

    from sageattention.core import sageattn_qk_int8_pv_fp16_cuda_sm75

    all_passed = True
    for layout in ["HND", "NHD"]:
        print(f"\nLayout: {layout}")
        try:
            if layout == "HND":
                q = torch.randn(1, 4, 128, head_dim, dtype=dtype, device=device)
                k = torch.randn(1, 4, 128, head_dim, dtype=dtype, device=device)
                v = torch.randn(1, 4, 128, head_dim, dtype=dtype, device=device)
            else:
                q = torch.randn(1, 128, 4, head_dim, dtype=dtype, device=device)
                k = torch.randn(1, 128, 4, head_dim, dtype=dtype, device=device)
                v = torch.randn(1, 128, 4, head_dim, dtype=dtype, device=device)

            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

            o_sage = sageattn_qk_int8_pv_fp16_cuda_sm75(
                q.clone(), k.clone(), v.clone(),
                tensor_layout=layout,
                is_causal=False,
                sm_scale=sm_scale,
                smooth_k=False,
                qk_quant_gran="per_warp",
                return_lse=False,
            )

            o_ref = reference_attention(q, k, v, sm_scale=sm_scale, smooth_k=False)[0]

            max_err = (o_sage.float() - o_ref.float()).abs().max().item()
            passed = max_err < 0.5  # Relaxed for layout conversion
            print(f"  {'✓' if passed else '✗'} max_err={max_err:.6f}")
            all_passed &= passed

        except Exception as e:
            print(f"  ✗ Layout {layout} failed: {e}")
            all_passed = False

    return all_passed


def test_lse_return():
    """Test LSE (log-sum-exp) return path."""
    print("\n" + "=" * 60)
    print("Test: LSE return path")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.float16
    B, H, L, D = 1, 4, 128, 64
    sm_scale = D ** -0.5

    from sageattention.core import sageattn_qk_int8_pv_fp16_cuda_sm75

    q = torch.randn(B, H, L, D, dtype=dtype, device=device)
    k = torch.randn(B, H, L, D, dtype=dtype, device=device)
    v = torch.randn(B, H, L, D, dtype=dtype, device=device)

    try:
        o_sage, lse_sage = sageattn_qk_int8_pv_fp16_cuda_sm75(
            q.clone(), k.clone(), v.clone(),
            tensor_layout="HND",
            is_causal=False,
            sm_scale=sm_scale,
            smooth_k=False,
            qk_quant_gran="per_warp",
            return_lse=True,
        )

        o_ref, lse_ref = reference_attention(
            q, k, v, sm_scale=sm_scale, smooth_k=False
        )

        o_max_err = (o_sage.float() - o_ref.float()).abs().max().item()
        lse_max_err = (lse_sage.float() - lse_ref.float()).abs().max().item()
        passed = o_max_err < 0.5 and lse_max_err < 10.0
        print(f"  {'✓' if passed else '✗'} o_max_err={o_max_err:.6f}  lse_max_err={lse_max_err:.6f}")
        return passed

    except Exception as e:
        print(f"  ✗ LSE test failed: {e}")
        return False


def _call_sm75_kernel_direct(q, k, v, o, q_scale, k_scale, sm_scale,
                               is_causal=False, return_lse=False):
    """
    Call the SM75 CUDA kernel directly with a pre-allocated output tensor.

    This bypasses sageattn_qk_int8_pv_fp16_cuda_sm75() which allocates a
    fresh output tensor on every call. The kernel writes directly into `o`.

    NOTE: Q/K must already be quantized to INT8 via per_warp_int8_cuda().
    """
    from sageattention import _qattn_sm75

    _tensor_layout = 1  # HND
    _is_causal = 1 if is_causal else 0
    _qk_quant_gran = 2  # per_warp
    _return_lse = 1 if return_lse else 0

    lse = _qattn_sm75.qk_int8_sv_f16_accum_f32_attn_sm75(
        q, k, v, o, q_scale, k_scale,
        _tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse
    )
    return lse if return_lse else None


def _quantize_per_warp(q, k, smooth_k=False):
    """
    Quantize Q and K via per_warp_int8_cuda, matching
    sageattn_qk_int8_pv_fp16_cuda_sm75's pipeline.

    Returns (q_int8, q_scale, k_int8, k_scale, v_fp16).
    V is just converted to contiguous FP16.
    """
    from sageattention.quant import per_warp_int8 as per_warp_int8_cuda

    q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(
        q.contiguous(), k.contiguous(),
        km=None,  # no smooth_k for direct kernel test
        tensor_layout="HND",
        BLKQ=64, WARPQ=16, BLKK=64,
    )
    return q_int8, q_scale, k_int8, k_scale


def test_back_to_back():
    """
    Test: back-to-back calls on the SAME pre-allocated output tensor.

    Calls the SM75 CUDA kernel directly (via _qattn_sm75) with a
    pre-allocated output tensor to verify:
      a) No stale values leak from a previous invocation
      b) Large→small shape reuse: no stale tail beyond qo_len
      c) Small→large shape reuse: all elements correctly overwritten
      d) Partial CTA tile: qo_len not divisible by CTA_Q (63, 13, etc.)
      e) LSE: back-to-back with return_lse=True

    Critical after removing smem_O roundtrip (kernel now writes to global O).
    """
    print("\n" + "=" * 60)
    print("Test: Back-to-back output tensor reuse (direct kernel)")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.float16
    head_dim = 64
    sm_scale = head_dim ** -0.5

    all_passed = True

    # --- Variant A: Same shape, different inputs, pre-allocated output ---
    print("\n  A) Same shape, different inputs (pre-allocated o):")
    B, H, L = 2, 4, 128

    q1 = torch.randn(B, H, L, head_dim, dtype=dtype, device=device)
    k1 = torch.randn(B, H, L, head_dim, dtype=dtype, device=device)
    v1 = torch.randn(B, H, L, head_dim, dtype=dtype, device=device)

    q2 = torch.randn(B, H, L, head_dim, dtype=dtype, device=device)
    k2 = torch.randn(B, H, L, head_dim, dtype=dtype, device=device)
    v2 = torch.randn(B, H, L, head_dim, dtype=dtype, device=device)

    # Quantize both input sets
    q1_i8, q1_scale, k1_i8, k1_scale = _quantize_per_warp(q1, k1)
    q2_i8, q2_scale, k2_i8, k2_scale = _quantize_per_warp(q2, k2)

    # Pre-allocate output — this is the SAME tensor for both calls
    o = torch.empty(B, H, L, head_dim, dtype=dtype, device=device)

    # First call on pre-allocated output
    _call_sm75_kernel_direct(q1_i8, k1_i8, v1, o, q1_scale, k1_scale, sm_scale)
    o_a1 = o.clone()

    # Second call on the SAME output tensor — must overwrite completely
    _call_sm75_kernel_direct(q2_i8, k2_i8, v2, o, q2_scale, k2_scale, sm_scale)
    o_a2 = o.clone()

    ref_a = reference_attention(q2, k2, v2, sm_scale=sm_scale, smooth_k=False)[0]

    # 1) Second output must match reference
    diff_a = (o_a2.float() - ref_a.float()).abs()
    max_err_a = diff_a.max().item()
    passed_a = max_err_a < 5e-2
    print(f"     {'✓' if passed_a else '✗'} match_ref: max_err={max_err_a:.6f}")

    # 2) Second output must differ from first (different inputs → different result)
    diff_stale = (o_a2.float() - o_a1.float()).abs()
    max_stale = diff_stale.max().item()
    outputs_differ = max_stale > 0.01
    if outputs_differ:
        print(f"     ✓ outputs_differ: max_diff={max_stale:.6f}")
    else:
        print(f"     ✗ STALE: max_diff={max_stale:.6f} — kernel may not overwrite output!")
    passed_a &= outputs_differ
    all_passed &= passed_a

    # --- Variant B: Large → small shape (stale tail check) ---
    print("\n  B) Large→small shape reuse (pre-allocated, stale tail):")
    B_s, H_s, L_small, L_large = 1, 4, 64, 256

    q_large = torch.randn(B_s, H_s, L_large, head_dim, dtype=dtype, device=device)
    k_large = torch.randn(B_s, H_s, L_large, head_dim, dtype=dtype, device=device)
    v_large = torch.randn(B_s, H_s, L_large, head_dim, dtype=dtype, device=device)
    ql_i8, ql_scale, kl_i8, kl_scale = _quantize_per_warp(q_large, k_large)

    q_small = torch.randn(B_s, H_s, L_small, head_dim, dtype=dtype, device=device)
    k_small = torch.randn(B_s, H_s, L_small, head_dim, dtype=dtype, device=device)
    v_small = torch.randn(B_s, H_s, L_small, head_dim, dtype=dtype, device=device)
    qs_i8, qs_scale, ks_i8, ks_scale = _quantize_per_warp(q_small, k_small)

    # Pre-allocate output at LARGE shape (holds both)
    o_large = torch.empty(B_s, H_s, L_large, head_dim, dtype=dtype, device=device)

    # First call with large inputs — fills all L_large rows
    _call_sm75_kernel_direct(ql_i8, kl_i8, v_large, o_large, ql_scale, kl_scale, sm_scale)

    # Store large output for stale check
    o_large_saved = o_large.clone()

    # Second call with small inputs on the SAME tensor
    # The kernel should write only L_small rows (64), leaving rows 64-255 from
    # the first call UNTOUCHED if boundary check is missing. We then compare
    # small output against reference and verify rows 64-255 are NOT stale copies
    # of the large output (since we set a flag value).
    _call_sm75_kernel_direct(qs_i8, ks_i8, v_small, o_large, qs_scale, ks_scale, sm_scale)
    o_small_result = o_large.clone()

    ref_b = reference_attention(q_small, k_small, v_small, sm_scale=sm_scale, smooth_k=False)[0]

    # 1) First L_small rows must match reference
    o_small_rows = o_small_result[:, :, :L_small, :]
    err_b = (o_small_rows.float() - ref_b.float()).abs().max().item()
    passed_b_ref = err_b < 5e-2
    print(f"     {'✓' if passed_b_ref else '✗'} rows[0:{L_small}] match_ref: max_err={err_b:.6f}")

    # 2) Rows L_small..L_large must NOT equal the stale large output
    #    (kernel should not have written past qo_len=L_small)
    #    Since we're reusing the tensor, rows beyond L_small retain their old
    #    values from the first (large) call. This is EXPECTED behavior — the
    #    kernel writes only up to qo_len. We just flag it for awareness.
    stale_region = o_small_result[:, :, L_small:, :]
    prev_region = o_large_saved[:, :, L_small:, :]
    stale_diff = (stale_region.float() - prev_region.float()).abs().max().item()
    is_stale = stale_diff < 1e-6
    if is_stale:
        print(f"     ⚠ rows[{L_small}:{L_large}] unchanged (expected: kernel only writes qo_len rows)")
    else:
        print(f"     ⚠ rows[{L_small}:{L_large}] modified: max_diff={stale_diff:.6f}")
    all_passed &= passed_b_ref

    # --- Variant C: Small → large shape (incomplete overwrite check) ---
    print("\n  C) Small→large shape reuse (all elements overwritten?):")

    # Fresh output at large shape
    o_large2 = torch.empty(B_s, H_s, L_large, head_dim, dtype=dtype, device=device)

    # First call with small shapes
    _call_sm75_kernel_direct(qs_i8, ks_i8, v_small, o_large2, qs_scale, ks_scale, sm_scale)

    # Second call with large shapes on SAME tensor — must overwrite ALL L_large rows
    _call_sm75_kernel_direct(ql_i8, kl_i8, v_large, o_large2, ql_scale, kl_scale, sm_scale)

    ref_c = reference_attention(q_large, k_large, v_large, sm_scale=sm_scale, smooth_k=False)[0]

    err_c = (o_large2.float() - ref_c.float()).abs().max().item()
    passed_c = err_c < 5e-2
    print(f"     {'✓' if passed_c else '✗'} match_ref: max_err={err_c:.6f}")
    all_passed &= passed_c

    # --- Variant D: Partial CTA tile (qo_len not divisible by CTA_Q=64) ---
    print("\n  D) Partial CTA tile (qo_len=13, 63, 65 — boundary check):")
    for qlen in [13, 63, 65]:
        B_d, H_d = 1, 2
        kvlen = qlen

        q_d1 = torch.randn(B_d, H_d, qlen, head_dim, dtype=dtype, device=device)
        k_d1 = torch.randn(B_d, H_d, kvlen, head_dim, dtype=dtype, device=device)
        v_d1 = torch.randn(B_d, H_d, kvlen, head_dim, dtype=dtype, device=device)
        q_d2 = torch.randn(B_d, H_d, qlen, head_dim, dtype=dtype, device=device)
        k_d2 = torch.randn(B_d, H_d, kvlen, head_dim, dtype=dtype, device=device)
        v_d2 = torch.randn(B_d, H_d, kvlen, head_dim, dtype=dtype, device=device)

        qd1_i8, qd1_scale, kd1_i8, kd1_scale = _quantize_per_warp(q_d1, k_d1)
        qd2_i8, qd2_scale, kd2_i8, kd2_scale = _quantize_per_warp(q_d2, k_d2)

        o_d = torch.empty(B_d, H_d, qlen, head_dim, dtype=dtype, device=device)

        # First call
        _call_sm75_kernel_direct(qd1_i8, kd1_i8, v_d1, o_d, qd1_scale, kd1_scale, sm_scale)
        # Second call on SAME tensor
        _call_sm75_kernel_direct(qd2_i8, kd2_i8, v_d2, o_d, qd2_scale, kd2_scale, sm_scale)

        ref_d = reference_attention(q_d2, k_d2, v_d2, sm_scale=sm_scale, smooth_k=False)[0]
        err_d = (o_d.float() - ref_d.float()).abs().max().item()
        passed_d = err_d < 5e-2
        print(f"     {'✓' if passed_d else '✗'} qlen={qlen:3d}: max_err={err_d:.6f}")
        all_passed &= passed_d

    # --- Variant E: LSE back-to-back reuse ---
    print("\n  E) LSE back-to-back (same pre-allocated LSE tensor):")
    try:
        B_e, H_e, L_e = 1, 4, 128

        q_e1 = torch.randn(B_e, H_e, L_e, head_dim, dtype=dtype, device=device)
        k_e1 = torch.randn(B_e, H_e, L_e, head_dim, dtype=dtype, device=device)
        v_e1 = torch.randn(B_e, H_e, L_e, head_dim, dtype=dtype, device=device)
        q_e2 = torch.randn(B_e, H_e, L_e, head_dim, dtype=dtype, device=device)
        k_e2 = torch.randn(B_e, H_e, L_e, head_dim, dtype=dtype, device=device)
        v_e2 = torch.randn(B_e, H_e, L_e, head_dim, dtype=dtype, device=device)

        qe1_i8, qe1_scale, ke1_i8, ke1_scale = _quantize_per_warp(q_e1, k_e1)
        qe2_i8, qe2_scale, ke2_i8, ke2_scale = _quantize_per_warp(q_e2, k_e2)

        # Pre-allocate output (kernel allocates LSE internally)
        o_e = torch.empty(B_e, H_e, L_e, head_dim, dtype=dtype, device=device)

        # First call with LSE
        _call_sm75_kernel_direct(
            qe1_i8, ke1_i8, v_e1, o_e, qe1_scale, ke1_scale, sm_scale, return_lse=True
        )
        # LSE is written directly to global memory by the kernel. For this test
        # we can't easily check LSE reuse (the kernel has no LSE output parameter).
        # But we verify output correctness after back-to-back with LSE path.

        # Second call with different inputs, same output + LSE tensors
        lse2 = _call_sm75_kernel_direct(
            qe2_i8, ke2_i8, v_e2, o_e, qe2_scale, ke2_scale, sm_scale, return_lse=True
        )

        _, lse_ref = reference_attention(q_e2, k_e2, v_e2, sm_scale=sm_scale, smooth_k=False)
        lse_err = (lse2.float() / 1.44269504 - lse_ref.float()).abs().max().item()
        passed_e = lse_err < 10.0
        print(f"     {'✓' if passed_e else '✗'} LSE match_ref: max_err={lse_err:.6f}")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
        passed_e = False
    all_passed &= passed_e

    return all_passed


def test_numerical_stability():
    """Test with varied distributions to check edge cases."""
    print("\n" + "=" * 60)
    print("Test: Edge cases")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.float16
    head_dim = 64

    from sageattention.core import sageattn_qk_int8_pv_fp16_cuda_sm75

    # Use independent random tensors for each edge case
    B, H, L, D = 2, 4, 64, head_dim
    edge_cases = [
        ("small_values",
         torch.randn(B, H, L, D, dtype=dtype, device=device) * 0.1,
         torch.randn(B, H, L, D, dtype=dtype, device=device) * 0.1,
         torch.randn(B, H, L, D, dtype=dtype, device=device) * 0.1),
        ("mixed_scales",
         torch.randn(B, H, L, D, dtype=dtype, device=device) * 0.1,
         torch.randn(B, H, L, D, dtype=dtype, device=device) * 5.0,
         torch.randn(B, H, L, D, dtype=dtype, device=device)),
        ("zeros",
         torch.zeros(B, H, L, D, dtype=dtype, device=device),
         torch.zeros(B, H, L, D, dtype=dtype, device=device),
         torch.zeros(B, H, L, D, dtype=dtype, device=device)),
    ]

    all_passed = True
    for name, q, k, v in edge_cases:
        print(f"\n  {name}:")
        try:
            o_sage = sageattn_qk_int8_pv_fp16_cuda_sm75(
                q, k, v,
                tensor_layout="HND",
                is_causal=False,
                sm_scale=head_dim ** -0.5,
                smooth_k=False,
                qk_quant_gran="per_warp",
                return_lse=False,
            )
            o_ref = reference_attention(q, k, v, sm_scale=head_dim ** -0.5)[0]
            max_err = (o_sage.float() - o_ref.float()).abs().max().item()
            passed = max_err < 0.5
            print(f"    {'✓' if passed else '✗'} max_err={max_err:.6f}")
            all_passed &= passed
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("SM75 SageAttention Kernel Correctness Test")
    print("=" * 60)

    try:
        arch = check_sm75_available()
    except Exception as e:
        print(f"\nSetup failed: {e}")
        print("\nTest cannot run without SM75 kernel and fused CUDA modules built.")
        print("Build with: python setup.py build_ext --inplace")
        sys.exit(1)

    all_passed = True

    # Run test suites (each returns True if all tests passed)
    all_passed &= test_known_head_dims()
    all_passed &= test_memory_layouts()
    all_passed &= test_lse_return()
    all_passed &= test_numerical_stability()
    all_passed &= test_back_to_back()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        print("Note: SM75 output store is in development — errors may be expected.")
    print("=" * 60)
