"""
What a built workspace holds, named once.
=========================================
`KnowledgeGraph.save_json` writes one file and `export_graph_data` writes three
more. Those four are one production event from one in-memory graph, and three
places need to agree on their names: the writer that produces them, the manifest
builder that binds their digests, and every consumer that verifies them.

The map lived in `src/evaluation/cohort.py`, which put a fact about
`src.kg`'s own output above `src.kg` in the layer order — so `sample_generator`
importing it broke the layered-architecture contract. It belongs here, at the
layer that writes the files, and the verifier above imports it.

Standard library only, so a consumer that needs the names does not pull in torch
to get them.

Module: src/kg/artifacts.py
"""
from __future__ import annotations

from typing import Dict

#: Manifest role → filename, for the artifacts a graph consumer reads.
#:
#: `kg.json` is the serialised graph; the other three are the PyG export a model
#: actually loads. Binding only the first left the tensors bound to nothing, and
#: `graph_fingerprint` does not close that — it is structural, so a same-shaped
#: `node_features.pt` from another workspace shares it.
GRAPH_ARTIFACTS: Dict[str, str] = {
    "kg": "kg.json",
    "node_features": "node_features.pt",
    "edge_indices": "edge_indices.pt",
    "num_nodes": "num_nodes.json",
}

__all__ = ["GRAPH_ARTIFACTS"]
