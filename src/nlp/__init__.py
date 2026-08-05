"""
Clinical NLP — RESERVED package (no implementation yet).
========================================================
Reserved home for turning free-text clinical descriptions into the HPO term
list the pipeline consumes. Nothing imports this package. The system's current
input contract is HPO terms supplied directly, so this is an input-convenience
layer, not a prerequisite.

Named by two Protocols in ``src/core/protocols.py``:
  - ``SymptomExtractorProtocol`` (:1007) — ``symptom_extractor.py``
  - ``HPOMatcherProtocol`` (:1037)       — ``hpo_matcher.py``

Reserved module names (all empty):
  - ``symptom_extractor.py`` — free text -> candidate symptom spans
  - ``hpo_matcher.py``       — spans -> HPO term IDs
  - ``entity_recognizer.py`` — clinical NER
  - ``clinical_bert.py``     — the encoder backing the above

Status: this whole package belongs to a later build phase, which is why the
status is recorded once here rather than in four separate module docstrings.
The files are kept, not deleted, because Protocols already define their
interfaces; deleting them would leave those Protocols pointing at nothing.

Module: src/nlp/__init__.py
"""
