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
import logging

logger = logging.getLogger("SageAttention")

# Cache: only import sageattn once
_sageattn_fn = None
_sageattn_available = False
_sageattn_checked = False

# Keep a reference to the TRUE original sdpa before any patches
_TRUE_ORIGINAL_SDPA = torch.nn.functional.scaled_dot_product_attention


def _check_sageattn():
    """Lazy import sageattn. Returns (callable or None, available_bool)."""
    global _sageattn_fn, _sageattn_available, _sageattn_checked
    if _sageattn_checked:
        return _sageattn_fn, _sageattn_available
    _sageattn_checked = True
    try:
        from sageattention import sageattn
        _sageattn_fn = sageattn
        _sageattn_available = True
        logger.info("SageAttention imported successfully")
    except ImportError:
        _sageattn_fn = None
        _sageattn_available = False
        logger.warning("SageAttention not installed")
    return _sageattn_fn, _sageattn_available


def _create_patched_sdpa(smooth_k=True, qk_quant_gran="per_warp"):
    """
    Creates a replacement for F.scaled_dot_product_attention that
    dispatches to SageAttention. Falls back to TRUE original sdpa on unsupported cases.
    """
    sageattn_fn, available = _check_sageattn()
    if not available:
        raise RuntimeError("SageAttention not installed.")

    # Use the true original sdpa (captured at module load time), not whatever
    # F.scaled_dot_product_attention currently points to (may be another patch).
    _original = _TRUE_ORIGINAL_SDPA

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False):
        # Fallback: custom attention mask
        if attn_mask is not None:
            return _original(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # Fallback: dropout (not supported in INT8 path)
        if dropout_p > 0.0:
            return _original(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # Fallback: non-FP16/BF16 tensor dtypes
        dtype = query.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            return _original(query, key, value, attn_mask=attn_mask,
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
            logger.debug(f"SageAttention failed, falling back to sdpa: {e}")
            return _original(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    return patched_sdpa


class SageAttentionNode:
    """
    Applies SageAttention INT8 attention to a MODEL.

    Uses model.add_object_patch to redirect sdpa -> SageAttention.
    The patch only affects this model instance (not global).

    Inputs:
      model    - The model to patch (MODEL)
      smooth_k - Subtract K-mean before attention (default: True, recommended)
      enable   - Enable/disable patch (default: True)

    Outputs:
      model    - Patched model (MODEL)
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
    DESCRIPTION = "Accelerates attention 2x on T4 via INT8 tensor cores. Auto-fallback for masks/non-FP16."

    def apply_patch(self, model, smooth_k=True, enable=True):
        if not enable:
            logger.debug("Passthrough (enable=False)")
            return (model,)

        _, available = _check_sageattn()
        if not available:
            logger.warning("SageAttention NOT installed — passthrough")
            return (model,)

        # Clone model to avoid mutating shared references
        m = model.clone()

        # Create the patched sdpa function
        patched_fn = _create_patched_sdpa(
            smooth_k=smooth_k,
            qk_quant_gran="per_warp"
        )

        # Apply via ComfyUI's model patching system (safe: scoped to this model)
        m.add_object_patch(
            "torch.nn.functional.scaled_dot_product_attention",
            patched_fn
        )

        logger.info(f"SageAttention applied (smooth_k={smooth_k})")
        return (m,)


class SageAttentionRemoveNode:
    """
    Removes SageAttention patch from a MODEL, restoring original attention.

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
        m = model.clone()
        # Remove the patch by restoring the true original sdpa
        m.add_object_patch(
            "torch.nn.functional.scaled_dot_product_attention",
            _TRUE_ORIGINAL_SDPA
        )
        logger.info("SageAttention patch removed")
        return (m,)
