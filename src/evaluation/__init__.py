"""
Offline measurement of the disease scorer.
==========================================
Work item B-0's harness. The offline evaluator it replaced
(`scripts/evaluate_model.py`, deleted; its code is at `7dab728`) did not measure
the deployed scorer: it scored a per-batch, 2-hop-expanded subgraph whose
candidate set was built from the answers. This package measures a task model
by the evaluator its protocol names (`protocols`, `task_measurement`). It
records each run on a manifest (`measurement`) and puts results in a ledger
beside the checkpoints (`sidecar`).

**This package measures. It does not correct.** Mode C, which measured a
checkpoint without a model spec by raw cosine against every disease, retired in
K8-6 (its command in K8-6a, its executor in K8-6b); a kept Mode C report is
still read as recorded. The served scorer is the task model's own.

Layering: `.import-linter.ini` places `src.evaluation` **above** `src.inference`,
so this package may import the served scoring primitives while production
inference can never import measurement code.

Module: src/evaluation/__init__.py
"""
