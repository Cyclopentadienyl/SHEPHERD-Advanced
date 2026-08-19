"""
Offline measurement of the disease scorer.
==========================================
Work item B-0's harness. It exists because the offline evaluator does not measure
the deployed scorer: it scores a per-batch, 2-hop-expanded subgraph whose
candidate set is built from the answers, with pure cosine and no shortest-path
term, truncated to twenty predictions. The deployed pipeline scores per-patient
*path-reachable* diseases with the eta mixture. Three separate differences, any
one of which makes the reported number describe a different system.

**This package measures. It does not correct.** Mode A deliberately preserves the
legacy candidate construction and sampling policy, because a control that has
been "improved" is no longer a control — the point is to tell a real effect from
a harness bug before anything is changed.

Layering: `.import-linter.ini` places `src.evaluation` **above** `src.inference`,
so this package may import the served scoring primitives while production
inference can never import measurement code.

Module: src/evaluation/__init__.py
"""
