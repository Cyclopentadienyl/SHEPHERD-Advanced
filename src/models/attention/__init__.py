"""
# ==============================================================================
# Module: src/models/attention/__init__.py
# ==============================================================================
# Purpose: Attention backend selection
#
# Exports:
#   - AdaptiveAttentionBackend: Cross-platform attention backend selection
# ==============================================================================
"""
from src.models.attention.adaptive_backend import AdaptiveAttentionBackend

__all__ = [
    "AdaptiveAttentionBackend",
]
