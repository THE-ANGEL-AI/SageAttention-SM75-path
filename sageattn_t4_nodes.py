"""
Bridge file for ComfyUI auto-discovery.

When this repo is cloned into ComfyUI/custom_nodes/SageAttention-T4/,
ComfyUI scans all .py files and imports them, checking for
NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.

This file re-exports the real node classes from comfyui_nodes/,
so the node appears automatically in ComfyUI's node menu under
category "SageAttention-T4" — no manual copying needed.
"""

from comfyui_nodes.sageattention_node import SageAttentionNode, SageAttentionRemoveNode

NODE_CLASS_MAPPINGS = {
    "SageAttentionT4_Apply": SageAttentionNode,
    "SageAttentionT4_Remove": SageAttentionRemoveNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SageAttentionT4_Apply": "🧠 SageAttention-T4 Apply (INT8 Turbo)",
    "SageAttentionT4_Remove": "🧠 SageAttention-T4 Remove",
}

# Also expose for programmatic use
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SageAttentionNode",
    "SageAttentionRemoveNode",
]
