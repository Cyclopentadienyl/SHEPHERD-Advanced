"""
Knowledge-graph storage backends — RESERVED package (no implementation yet).
============================================================================
Reserved home for KG persistence. Nothing imports this package.

The KG is written by ``src/kg/builder.py`` as files under
``data/workspaces/<kg>/`` (``kg.json``, ``node_features.pt``,
``edge_indices.pt``, ``num_nodes.json``).

**Reading them is currently duplicated.** ``src/kg/data_loader.py`` does *not*
read these files — it receives an already-loaded ``graph_data`` dictionary. The
files are loaded independently by each consumer that needs them, and then handed
to the dataloader: ``src/inference/pipeline.py:579-606``,
``scripts/train_model.py``, ``scripts/evaluate_model.py``,
``scripts/build_index.py``, ``scripts/setup_demo.py``,
``scripts/spikes/validate_fast_subgraph.py`` and
``scripts/measure_scorer.py``. Every copy depends on the same filenames and the
same serialisation format, so a format change breaks all of them at once.

Reserved module names (both empty):
  - ``file_storage.py`` — one shared implementation of the layout above
  - ``graph_db.py``     — a graph-database backend

**Scope guardrail for whoever implements ``file_storage.py`` (P1).** Its first
and only job is to become the single reader/writer of the current file layout,
replacing those seven copies. It is **not** licence to build a ``Storage``
Protocol, a backend registry, a database adapter hierarchy or a migration
framework: this package's name says "backends" plural, and that plural is
aspirational. Those abstractions wait until a second real backend exists, at
which point the shape it needs will be known instead of guessed.

One consumer is exempt and stays duplicated on purpose:
``scripts/measure_scorer.py:load_legacy_mode_a_inputs`` must keep matching the
frozen evaluator until both are deleted together.

Status: this whole package belongs to a later build phase, which is why the
status is recorded once here rather than in two separate module docstrings.
No Protocol names these modules yet; they are kept as a documented reserved
home because the abstraction boundary is a deliberate part of the design.

Module: src/kg/storage/__init__.py
"""
