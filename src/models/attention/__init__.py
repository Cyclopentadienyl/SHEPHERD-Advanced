"""
# ==============================================================================
# Module: src/models/attention/__init__.py
# ==============================================================================
# Purpose: Attention mechanisms for heterogeneous graphs
#
# Exports:
#   - AdaptiveAttentionBackend: Cross-platform attention backend selection
#   - FlexHeteroAttention: FlexAttention-based heterogeneous attention
#   - create_ontology_score_mod: Ontology-aware attention modifier
#   - create_species_score_mod: Species-aware attention modifier
# ==============================================================================
"""
from src.models.attention.adaptive_backend import AdaptiveAttentionBackend
from src.models.attention.flex_attention import (
    FlexHeteroAttention,
    create_ontology_score_mod,
    create_species_score_mod,
)

__all__ = [
    # Backends
    "AdaptiveAttentionBackend",
    # Heterogeneous attention
    "FlexHeteroAttention",
    # Score modifiers
    "create_ontology_score_mod",
    "create_species_score_mod",
]
