"""
ComfyUI Custom Node: SageAttention SM75 (Turing T4 INT8 Attention)

Installing:
  Copy the comfyui_nodes/ folder into ComfyUI/custom_nodes/sageattention_sm75/

After restart, find nodes in menu: SageAttention

Provides:
  - SageAttention Apply (SM75)  — applies INT8 attention to MODEL
  - SageAttention Remove        — restores original attention
"""

from .sageattention_node import SageAttentionNode, SageAttentionRemoveNode

NODE_CLASS_MAPPINGS = {
    "SageAttentionApply": SageAttentionNode,
    "SageAttentionRemove": SageAttentionRemoveNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SageAttentionApply": "🧠 SageAttention Apply (SM75 T4 INT8)",
    "SageAttentionRemove": "🧠 SageAttention Remove",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
