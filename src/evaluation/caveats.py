"""
Standing caveats on measurement, in one place.
==============================================

De-duplicated from ``scripts/measure_scorer.py`` and
``scripts/calibrate_mode_a.py``, which carried the same 922 characters
byte-for-byte. The same class of duplication as ``DEPLOYMENT_RELATIONSHIPS``
before it moved to ``src/utils/provenance.py``: a claim spelled differently in
two places cannot be compared, and a claim that has to be corrected in two
places will eventually be corrected in one.

**Constants only. No imports beyond the standard library, no I/O.** These strings
are rendered into ``--help`` text, where ``argparse`` performs ``%`` substitution,
so every literal percent sign here is escaped as ``%%``.

Module: src/evaluation/caveats.py
"""
from __future__ import annotations

#: Why a metric measured on `val` is not an independent test result.
#:
#: **Permanent, and more important since the split became disease-disjoint, not
#: less.** Whatever `val` contains, the standard workflow selects the checkpoint
#: on it (``early_stopping_monitor=val_mrr``), so a number measured there is a
#: selection metric. That does not depend on how the split was cut.
SELECTION_CONTAMINATION = (
    "**Not an independent evaluation.** In the standard workflow `val` is the "
    "split early stopping and checkpoint selection run on "
    "(`early_stopping_monitor=val_mrr`), so a metric measured on it is a "
    "selection metric rather than an independent test estimate. This says "
    "nothing about how an arbitrary explicit checkpoint passed to this script "
    "was chosen — it may have been selected on some other workspace, or not "
    "selected at all."
)

#: Whether the split is disease-disjoint, which is now a property of the
#: workspace rather than of the code.
#:
#: **This one stopped being a global fact.** Before the allocation step,
#: `sample_generator` drew one pooled set and sliced it by index, so overlap was
#: permitted everywhere and total in the audited workspace — all 7,970 validation
#: diseases also in train (``docs/working/EVIDENCE_M4.json``). Generation now
#: consumes a disease allocation and the partitions are disjoint by construction,
#: but a workspace built before that change still has the old shape. A help
#: string cannot know which one it is being pointed at, so it must not assert
#: either.
WORKSPACE_DEPENDENT_OVERLAP = (
    "**Disease overlap is a property of the workspace, not of this script.** "
    "Workspaces generated from a disease allocation have disease-disjoint train "
    "and validation cohorts by construction; workspaces generated before that "
    "change were sliced at the sample level and may overlap completely — in the "
    "audited workspace the overlap was total, all 7,970 validation diseases also "
    "present in train (docs/working/EVIDENCE_M4.json). Establish "
    "which one you have with `scripts/audit_split_overlap.py`, and read "
    "`split_manifest.json` if the workspace has one. Overlap bounds what a `val` "
    "metric can be evidence of about unseen diseases; it does not by itself "
    "invalidate every sample-level claim."
)

#: The full `--split` help text. Assembled once so the two measurement entry
#: points cannot drift apart.
SPLIT_ARGUMENT_HELP = (
    "Which samples file to measure. **Required — there is no default.** "
    "Generated workspaces normally contain train and val only; a test split "
    "exists only where an evaluation protocol created one. Two distinct limits "
    "apply to `val`, and they are not the same limit. (1) "
    + SELECTION_CONTAMINATION
    + " (2) "
    + WORKSPACE_DEPENDENT_OVERLAP
)
