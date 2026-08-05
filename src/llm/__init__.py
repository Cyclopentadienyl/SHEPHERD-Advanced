"""
LLM integration — RESERVED package (no implementation yet).
===========================================================
Reserved home for optional large-language-model support: natural-language
summarisation of a diagnosis, and free-text clinical note handling on the way
in. Nothing on any runtime path imports this package, and the diagnostic
pipeline does not depend on it — the ranking and its evidence chains come from
the GNN and the PathReasoner.

Named by two Protocols in ``src/core/protocols.py``:
  - ``LLMProtocol`` — the package as a whole
  - ``MedicalLLMProtocol`` — ``src/llm/medical_llm.py``, not yet created

Reserved module names (all empty):
  - ``interface.py``        — the boundary the rest of the codebase would call
  - ``model_loader.py``     — loading/quantising a local model
  - ``inference_engine.py`` — generation and batching
  - ``prompt_templates.py`` — prompt construction

Status: this whole package belongs to a later build phase, which is why the
status is recorded once here rather than in four separate module docstrings.
The files are kept, not deleted, because Protocols already define their
interfaces; deleting them would leave those Protocols pointing at nothing.

Module: src/llm/__init__.py
"""
