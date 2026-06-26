"""
ComfyUI Custom Node: SageAttention-T4 (Turing SM75 INT8 Attention)

Автор: THEANGELAI
Репозиторий: https://github.com/THE-ANGEL-AI/SageAttention-SM75-path

Авто-установка:
  Клонируйте репо в ComfyUI/custom_nodes/SageAttention-T4/
  → pip install -e . → перезапустите ComfyUI
  → нода появится автоматически (bridge: sageattn_t4_nodes.py)

Ноды:
  - SageAttention-T4 Apply (INT8 Turbo)  — ускоряет attention модели в ~2×
  - SageAttention-T4 Remove             — восстанавливает оригинальное внимание

NOTE: NODE_CLASS_MAPPINGS lives in root sageattn_t4_nodes.py for auto-discovery.
This __init__.py exports only the node classes for direct import.
"""

from .sageattention_node import SageAttentionNode, SageAttentionRemoveNode

__all__ = ["SageAttentionNode", "SageAttentionRemoveNode"]
