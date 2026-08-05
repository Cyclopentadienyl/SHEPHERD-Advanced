"""
Dependency version checking — RESERVED home (not yet implemented).
==================================================================
Reserved home for checking installed dependency versions against what the
project requires (torch / PyG and its native extensions, CUDA line, optional
retrieval backends), and for reporting a mismatch in one consistent way.

The concern is real and currently spread across 7 sites that each inspect
``__version__`` or parse a version string their own way, among them
``src/utils/pyg_native_check.py``, ``src/retrieval/backends/voyager_backend.py``,
``scripts/validate_installation.py``, ``scripts/validate_pyg_ext.py`` and
``scripts/build_pyg_arm.sh``.

Note the deliberate boundary with ``src/utils/pyg_native_check.py``, which is
implemented and stays: that module answers "do the compiled PyG extensions load
and produce correct results on this host", which is a functional check, not a
version comparison.

Status: intentionally empty until the consolidation is done. Nothing imports
this module. It is kept as a documented reserved home rather than deleted
because the concern is real and currently homeless.

Module: src/utils/version_checker.py
"""
