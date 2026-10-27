"""
Task heads of a finetuned model bundle.
======================================
The patient encoder, the disease, gene and patients-like-me heads
(`heads.py`) and the one task-loss module (`losses.py`) of
`docs/working/scorer-retraining/README.md` §3.1. The bundle that composes them
with the graph side is `src.models.bundle`.

`DiagnosisModelProtocol` in `src.core.protocols` still names a planned
`src/models/tasks/diagnosis.py`; that Protocol is reviewed with the retirement
check (README §7, K7).

Module: src/models/tasks/__init__.py
"""
