"""
Medical interchange standards — RESERVED package (no implementation yet).
=========================================================================
Reserved home for exchanging data with hospital systems in standard formats:
FHIR bundles in and out, and mapping between MONDO and the coding systems a
hospital actually stores (ICD, SNOMED CT). Nothing imports this package.

Named by two Protocols in ``src/core/protocols.py``:
  - ``FHIRAdapterProtocol`` (:1260)      — ``fhir_adapter.py``
  - ``MedicalCodeMapperProtocol`` (:1282) — ``icd_mapper.py``

Reserved module names (all empty):
  - ``fhir_adapter.py``  — FHIR Bundle <-> internal representation
  - ``icd_mapper.py``    — MONDO <-> ICD
  - ``snomed_mapper.py`` — MONDO <-> SNOMED CT
  - ``hiss_adapter.py``  — hospital information system integration

Status: this whole package belongs to a later build phase, which is why the
status is recorded once here rather than in four separate module docstrings.
The files are kept, not deleted, because Protocols already define their
interfaces; deleting them would leave those Protocols pointing at nothing.

Module: src/medical_standards/__init__.py
"""
