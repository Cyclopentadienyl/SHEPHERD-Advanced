"""
Platform detection — RESERVED home (not yet implemented).
=========================================================
Reserved home for detecting the host platform (OS, CPU architecture, CUDA
availability and line) so callers branch on one answer instead of each
re-deriving it.

The concern is real and currently duplicated across 5 sites, each calling
``platform.system()`` / ``platform.machine()`` for its own purposes:

  - ``scripts/launch/shep_launch.py``
  - ``scripts/validate_installation.py``
  - ``scripts/debug_voyager_windows.py``
  - ``src/models/attention/adaptive_backend.py``
  - ``src/retrieval/vector_index.py``

This matters more than usual here: the project ships on Windows x86-64, Linux
x86-64 and Linux aarch64 (DGX Spark), and several backend choices depend on
which one it is.

Status: intentionally empty until the consolidation is done. Nothing imports
this module. It is kept as a documented reserved home rather than deleted
because the concern is real and currently homeless.

Module: src/utils/platform_detector.py
"""
