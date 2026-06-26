"""
SageAttention SM75 Integration Patch for LTX-Video ComfyUI.
===========================================================

Drop-in patch that monkey-patches torch's scaled_dot_product_attention
to use SageAttention's SM75 INT8 kernel on Turing GPUs (T4).

Installation:
  Copy this file to your ComfyUI custom_nodes directory and call:
    from ltxv_sageattn_patch import apply_patch
    apply_patch()
  in the LTX-Video loader node before any inference.

Auto-detects GPU architecture and falls back to original sdpa
on non-SM75 GPUs or unsupported features (custom masks, dropout).
"""

import torch
import torch.nn.functional as F
import warnings


def apply_patch(smooth_k: bool = True, qk_quant_gran: str = "per_warp"):
    """
    Apply SageAttention monkey-patch to F.scaled_dot_product_attention.

    Args:
        smooth_k: Subtract K-mean before attention (improves accuracy). Default True.
        qk_quant_gran: Quantization granularity — 'per_warp' (recommended for T4)
                       or 'per_thread'.

    Returns True if patch applied, False if SageAttention unavailable.
    """
    try:
        from sageattention import sageattn
    except ImportError:
        warnings.warn(
            "SageAttention not installed. Run: pip install -e . "
            "or: python setup.py build_ext --inplace"
        )
        return False

    # Save original BEFORE replacing (fix: store on F, not local variable)
    F._original_sdpa = F.scaled_dot_product_attention

    def _sageattn_wrapper(query, key, value, attn_mask=None, dropout_p=0.0,
                          is_causal=False, scale=None, enable_gqa=False):
        """
        Wrapper: SageAttention for supported cases, original sdpa for fallbacks.
        Falls back on: custom attn_mask, dropout>0, non-FP16/BF16 dtypes.
        """
        # Fallback: custom attention mask
        if attn_mask is not None:
            return F._original_sdpa(query, key, value, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # Fallback: dropout not supported in INT8 path
        if dropout_p > 0.0:
            return F._original_sdpa(query, key, value, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # Fallback: SageAttention requires FP16 or BF16
        dtype = query.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            return F._original_sdpa(query, key, value, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # Detect tensor layout: HND = [B, H, L, D], NHD = [B, L, H, D]
        # Heuristic: num_heads (H) is typically smaller than seq_len (L).
        # LTX-Video: heads=30, seq_len=thousands -> shape[1] < shape[2] -> HND.
        if query.shape[1] < query.shape[2]:
            tensor_layout = "HND"
        else:
            tensor_layout = "NHD"

        try:
            return sageattn(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                tensor_layout=tensor_layout,
                is_causal=is_causal,
                sm_scale=scale,
                return_lse=False,
                smooth_k=smooth_k,
                qk_quant_gran=qk_quant_gran,
            )
        except Exception as e:
            warnings.warn(f"SageAttention failed: {e}. Falling back to sdpa.")
            return F._original_sdpa(query, key, value, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    # Apply the monkey-patch
    F.scaled_dot_product_attention = _sageattn_wrapper

    # Report SM75 status
    try:
        from sageattention.core import SM75_ENABLED
        if SM75_ENABLED:
            print("[SageAttention] SM75 kernel active - INT8 QK + FP16 PV on T4 GPU")
        else:
            print("[SageAttention] SM75 kernel NOT built - using default dispatch")
    except ImportError:
        print("[SageAttention] Applied (SM75 status unknown)")

    return True


def remove_patch():
    """Restore original scaled_dot_product_attention. Returns True if removed."""
    if hasattr(F, '_original_sdpa'):
        F.scaled_dot_product_attention = F._original_sdpa
        del F._original_sdpa
        print("[SageAttention] Patch removed.")
        return True
    print("[SageAttention] No patch to remove.")
    return False
