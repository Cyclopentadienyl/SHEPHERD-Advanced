"""
Knowledge-graph storage backends — the shared reader of a workspace's files.
============================================================================
`file_storage` is the one implementation (`read_graph_artifacts`,
`read_samples`); callers import that submodule directly, and this package's own
namespace exports nothing.

The KG is written by ``src/kg/builder.py`` as files under
``data/workspaces/<kg>/`` (``kg.json``, ``node_features.pt``,
``edge_indices.pt``, ``num_nodes.json``).

**Reading them is no longer duplicated.** ``src/kg/data_loader.py`` does *not*
read these files — it receives an already-loaded ``graph_data`` dictionary.
Measurement, serving and training read them through ``file_storage``, and
``scripts/setup_demo.py`` reads them through training; the last independent
copy, ``scripts/build_index.py``'s, went with its ``--checkpoint`` (K7).

Modules:
  - ``file_storage.py`` — the shared reader. **Started, not finished.** Its
    callers so far are `scripts/measure_scorer.py` (through
    `read_measurement_inputs`, contract M2.1, S7), serving,
    `src/inference/pipeline.py`, and training, `scripts/train_model.py` (S6 and
    S8).
  - ``graph_db.py``     — reserved, empty: a graph-database backend

**Scope guardrail for whoever implements ``file_storage.py`` (P1).** Its first
and only job is to become the single reader/writer of the current file layout,
replacing those six copies. It is **not** licence to build a ``Storage``
Protocol, a backend registry, a database adapter hierarchy or a migration
framework: this package's name says "backends" plural, and that plural is
aspirational. Those abstractions wait until a second real backend exists, at
which point the shape it needs will be known instead of guessed.

Status: this whole package belongs to a later build phase, which is why the
status is recorded once here rather than in two separate module docstrings.
No Protocol names these modules yet; they are kept as a documented reserved
home because the abstraction boundary is a deliberate part of the design.

Module: src/kg/storage/__init__.py
"""
