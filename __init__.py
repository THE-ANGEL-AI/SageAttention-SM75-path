"""
SageAttention-T4 ComfyUI custom node package.

Автоматически обнаруживается ComfyUI при symlink в custom_nodes/.
Реэкспортирует ноды из sageattn_t4_nodes.py.
"""

from .sageattn_t4_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
