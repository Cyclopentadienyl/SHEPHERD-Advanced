"""
Shared logging configuration — RESERVED home (not yet implemented).
===================================================================
Reserved home for the project's *logging setup*: handler, format and level
policy applied once per process.

This is deliberately NOT about obtaining a logger. The live idiom for that is
stdlib ``logging.getLogger(__name__)``, used directly by 36 modules under
``src/``, and it is correct as it stands — this module would not replace it.

What is currently scattered is the setup call. ``logging.basicConfig(...)``
appears at 8 sites: ``src/api/main.py`` and seven entry-point scripts
(``build_index``, ``build_knowledge_graph``, ``compute_shortest_paths``,
``evaluate_model``, ``setup_demo``, ``test_gnn_inference``, ``train_model``),
each with its own format string and level. In an entry-point script that is a
defensible place for it; the point is that no module owns the policy, so the
eight differ.

Status: intentionally empty until that consolidation is done. Nothing imports
this module. It is kept as a documented reserved home rather than deleted
because the concern is real and currently homeless.

Module: src/utils/logger.py
"""
