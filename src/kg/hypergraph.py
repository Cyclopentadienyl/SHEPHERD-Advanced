"""
Hypergraph Knowledge Graph Representation (PLANNED — NOT YET IMPLEMENTED)
========================================================================
Reserved for the hypergraph KG variant described in medical-kg-blueprint.md
L218-260 ("異質+超圖" / heterogeneous + hypergraph).

Difference from the heterogeneous graph in src/kg/graph.py:
  - Heterograph: edges are strictly pairwise
      (gene, encodes, protein)
      (drug, treats, disease)
  - Hypergraph: edges can connect three or more nodes simultaneously
      (gene, drug, disease) co-occurrence in a single clinical trial
      (phenotype_A, phenotype_B, phenotype_C) co-occurrence in one syndrome
      (variant, gene, disease, evidence_level) provenance hyperedge

Why a separate module rather than extending graph.py:
  Hyperedge data structures, message passing, and storage differ
  substantially from binary edges. Mixing them inside KnowledgeGraph
  would either bloat that class or force tagged-union dispatch in every
  edge accessor. A parallel class is cleaner.

Whether this will ship:
  Conditional. The hypergraph formulation is a research bet — it may
  outperform the heterograph baseline on multi-entity co-occurrence
  tasks (drug combinations, polygenic phenotypes), or it may not.
  Implementation gated on empirical evidence.

Status: planned, may or may not be implemented. Importing this module
yields an empty namespace; accessing any symbol raises NameError.
"""
