"""
ComfyUI Custom Node: SageAttention SM75 for NVIDIA T4 (Turing)

Uses ComfyUI's model patching system (add_object_patch) to safely redirect
scaled_dot_product_attention to SageAttention's INT8 kernel.

Unlike global monkey-patching, this ONLY affects the model that passes
through this node — other parts of the workflow are untouched.

Requires:
  - SageAttention-SM75 built and installed
  - NVIDIA T4 (SM75) or compatible GPU
"""

import torch
import torch.nn.functional as F
import warnings


# Cache: only import sageattn once
_sageattn = None
_sageattn_available = None


def _check_sageattn():
    """Lazy import sageattn. Returns (callable or None, available_bool)."""
    global _sageattn, _sageattn_available
    if _sageattn_available is not None:
        return _sageattn, _sageattn_available
    try:
        from sageattention import sageattn
        _sageattn = sageattn
        _sageattn_available = True
    except ImportError:
        _sageattn = None
        _sageattn_available = False
    return _sageattn, _sageattn_available


def _create_patched_sdpa(smooth_k=True, qk_quant_gran="per_warp"):
    """
    Creates a replacement for F.scaled_dot_product_attention that
    dispatches to SageAttention. Falls back to original on unsupported cases.
    """
    sageattn_fn, available = _check_sageattn()
    if not available:
        raise RuntimeError(
            "SageAttention not installed. Run: pip install -e /path/to/SageAttention-SM75-path"
        )

    _original_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False):
        # Fallback: custom attention mask
        if attn_mask is not None:
            return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                   dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # Fallback: dropout (not supported in INT8 path)
        if dropout_p > 0.0:
            return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                   dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # Fallback: non-FP16/BF16 tensor dtypes
        dtype = query.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                   dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # Detect layout: HND [B,H,L,D] vs NHD [B,L,H,D]
        # heads (shape[1]) is typically much smaller than seq_len (shape[2])
        if query.shape[1] < query.shape[2]:
            tensor_layout = "HND"
        else:
            tensor_layout = "NHD"

        try:
            return sageattn_fn(
                query.contiguous(), key.contiguous(), value.contiguous(),
                tensor_layout=tensor_layout,
                is_causal=is_causal,
                sm_scale=scale,
                return_lse=False,
                smooth_k=smooth_k,
                qk_quant_gran=qk_quant_gran,
            )
        except Exception as e:
            warnings.warn(f"SageAttention failed: {e}. Falling back to sdpa.")
            return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                   dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    return patched_sdpa


class SageAttentionNode:
    """
    Applies SageAttention INT8 attention to a MODEL.

    This node patches torch.nn.functional.scaled_dot_product_attention
    inside the model's execution context using add_object_patch.
    The patch only affects this specific model instance.

    Inputs:
      model          - The model to patch (MODEL)
      smooth_k       - Subtract K-mean before attention (default: True)
      enable         - Enable or disable the patch (default: True)

    Outputs:
      model          - Patched model (MODEL)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "smooth_k": ("BOOLEAN", {
                    "default": True,
                    "label_on": "smooth_k on (recommended)",
                    "label_off": "smooth_k off",
                }),
                "enable": ("BOOLEAN", {
                    "default": True,
                    "label_on": "SageAttention ON",
                    "label_off": "SageAttention OFF (passthrough)",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "SageAttention"
    DESCRIPTION = "Accelerates attention 2x on NVIDIA T4 using INT8 tensor cores. Automatically falls back for unsupported cases (masks, non-FP16)."

    def apply_patch(self, model, smooth_k=True, enable=True):
        if not enable:
            print("[SageAttention] Passthrough (enable=False)")
            return (model,)

        sageattn_fn, available = _check_sageattn()
        if not available:
            print("[SageAttention] ⚠ NOT installed — passthrough.")
            return (model,)

        # Clone model to avoid mutating shared references
        m = model.clone()

        # Create the patched sdpa function
        patched_fn = _create_patched_sdpa(
            smooth_k=smooth_k,
            qk_quant_gran="per_warp"
        )

        # Apply via ComfyUI's model patching system (safe: only this model)
        m.add_object_patch(
            "torch.nn.functional.scaled_dot_product_attention",
            patched_fn
        )

        print(f"[SageAttention] ✓ Applied (smooth_k={smooth_k})")
        return (m,)


class SageAttentionRemoveNode:
    """
    Removes SageAttention patch from a MODEL.
    Useful if you need to ensure a model uses original attention downstream.

    Inputs:
      model - The model to un-patch (MODEL)

    Outputs:
      model - Clean model with original attention (MODEL)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "remove_patch"
    CATEGORY = "SageAttention"
    DESCRIPTION = "Removes SageAttention patch — restores original scaled_dot_product_attention."

    def remove_patch(self, model):
        # Clone and remove the patch
        m = model.clone()
        m.add_object_patch(
            "torch.nn.functional.scaled_dot_product_attention",
            None  # None removes the patch
        )
        print("[SageAttention] ✗ Patch removed")
        return (m,)
