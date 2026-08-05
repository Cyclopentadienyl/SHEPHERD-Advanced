"""
Knowledge-graph storage backends — RESERVED package (no implementation yet).
============================================================================
Reserved home for pluggable KG persistence. Nothing imports this package.

The KG is currently persisted by ``src/kg/builder.py`` as files under
``data/workspaces/<kg>/`` (``kg.json``, ``node_features.pt``,
``edge_indices.pt``) and loaded by ``src/kg/data_loader.py``. That is the live
path and it works; this package exists for the case where the graph outgrows it
or has to live in a database.

Reserved module names (both empty):
  - ``file_storage.py`` — formalise the current on-disk layout behind an interface
  - ``graph_db.py``     — a graph-database backend

Status: this whole package belongs to a later build phase, which is why the
status is recorded once here rather than in two separate module docstrings.
No Protocol names these modules yet; they are kept as a documented reserved
home because the abstraction boundary is a deliberate part of the design.

Module: src/kg/storage/__init__.py
"""
