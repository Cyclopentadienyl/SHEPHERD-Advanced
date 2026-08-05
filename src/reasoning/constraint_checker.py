"""
Diagnostic constraint checking — RESERVED home (not yet implemented).
=====================================================================
Reserved home for ``ConstraintCheckerProtocol`` (``src/core/protocols.py:783``):
validating a *prediction* against ontology constraints and adjusting candidate
scores accordingly —

  - ``check_prediction(patient_phenotypes, predicted_disease, kg, ontology)``
    -> ``(is_valid, violations, penalty_score)``
  - ``apply_constraints(candidates, patient_phenotypes, kg, ontology)``
    -> re-scored candidates

Not the same concern as ``src/ontology/constraints.py``, despite the similar
name. That module implements ``OntologyConstraintProtocol``
(``src/core/protocols.py:169``) and operates on a *phenotype set* against
ontology structure — ``validate_phenotype_set``, ``remove_redundant_ancestors``,
``get_implied_phenotypes``. It never sees a prediction. The names collide; the
concerns do not.

Status: intentionally empty until diagnostic constraint checking is built.
Nothing imports this module. It is kept as a documented reserved home rather
than deleted because a Protocol names it.

Module: src/reasoning/constraint_checker.py
"""
