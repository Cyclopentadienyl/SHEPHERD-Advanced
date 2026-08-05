"""
Task heads — RESERVED package (no implementation yet).
======================================================
Reserved home for end-to-end task models that compose the encoder, GNN and
decoder into a single unit.

Named by ``DiagnosisModelProtocol`` (``src/core/protocols.py:632``), which
declares its implementation module as ``src/models/tasks/diagnosis.py`` — a
file that does not exist. Today that composition lives in
``src/models/gnn/shepherd_gnn.py``, which is the live diagnostic model.

The package is intentionally empty rather than absent: a Protocol names it, and
removing the package would leave that Protocol pointing into a directory that
is not there.

Module: src/models/tasks/__init__.py
"""
