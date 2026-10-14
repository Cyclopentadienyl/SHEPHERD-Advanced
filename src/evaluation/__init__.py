"""
Offline measurement of the disease scorer.
==========================================
Work item B-0's harness. The offline evaluator it replaced
(`scripts/evaluate_model.py`, deleted; its code is at `7dab728`) did not measure
the deployed scorer: it scored a per-batch, 2-hop-expanded subgraph whose
candidate set was built from the answers. This package measures Mode C, the
formal Test's mode: every patient against every disease in the graph, with the
production model and raw cosine (`measurement.run_mode_c`). It records each run
on a manifest and puts results in a ledger beside the checkpoints
(`sidecar`).

**This package measures. It does not correct.** The served scorer still differs
from Mode C by design until B-1 (`docs/DISEASE_SCORER_POLICY.md`): candidates
discovered by path reachability, the eta mixture, and insertion-order ties.

Layering: `.import-linter.ini` places `src.evaluation` **above** `src.inference`,
so this package may import the served scoring primitives while production
inference can never import measurement code.

Module: src/evaluation/__init__.py
"""
