"""
Top-Level Preprocessing CLI (PLANNED — NOT YET IMPLEMENTED)
===========================================================
Reserved as a future command-line entry point that wraps
src/kg/preprocessing.py for batch / scriptable preprocessing of an
existing workspace (Laplacian PE, RWSE, degree features).

Current state of the world:
  - src/kg/preprocessing.py contains the actual preprocessing logic
    (compute_laplacian_pe, compute_rwse, compute_degree_features,
    preprocess_for_gnn). It is invoked automatically inside the GNN
    training path and exported via src/kg/__init__.py.
  - scripts/build_knowledge_graph.py already produces node_features.pt /
    edge_indices.pt as part of KG construction, which covers the
    common case.

Why this stub still exists:
  A standalone CLI would be useful when:
    - Users want to regenerate preprocessing artefacts without rebuilding
      the KG from scratch (e.g., after adjusting RWSE walk length)
    - PrimeKG-scale workspaces make on-the-fly preprocessing too slow
      during training startup
    - Caching preprocessed tensors as separate artefacts simplifies
      reproducibility / sharing

Status: planned. Implementation deferred until a concrete use case
demands it. This file is intentionally empty until then.
"""
