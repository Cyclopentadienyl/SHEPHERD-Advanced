"""
Task heads of a finetuned model bundle.
======================================
The patient encoder, the disease, gene and patients-like-me heads
(`heads.py`) and the one task-loss module (`losses.py`) of
`docs/working/scorer-retraining/README.md` §3.1. The bundle that composes them
with the graph side is `src.models.bundle`.

In `src.core.protocols`, `DecoderProtocol` points at these heads and
`DiagnosisModelProtocol` at the bundle builder that composes them.

Module: src/models/tasks/__init__.py
"""
