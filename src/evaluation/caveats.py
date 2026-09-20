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

#: What `val` is under the current pipeline, and what it still is not.
#:
#: **This used to be a workspace-dependent caveat and no longer is.** Before the
#: allocation step, `sample_generator` drew one pooled set and sliced it by index,
#: so a `val` cohort's disease overlap depended on which workspace you were
#: pointed at — all 7,970 validation diseases were also in train in the audited
#: one (``docs/working/EVIDENCE_M4.json``). Generation now consumes a disease
#: allocation, `src/evaluation/cohort.py` refuses a workspace without a manifest,
#: and a pre-allocation workspace therefore cannot reach a measurement at all. The
#: help no longer has to hedge about which regime it is describing.
GENERATED_VAL_IS_DISEASE_DISJOINT = (
    "**`val` is disease-disjoint from `train`, by construction.** Generation "
    "consumes a disease allocation, so no disease with labelled training examples "
    "appears in the validation cohort, and `split_manifest.json` records the cut. "
    "One channel a disease-level split does not close: two diseases with identical "
    "phenotype content can fall on opposite sides, which "
    "`scripts/audit_generator_fidelity.py` measures."
)

#: Why a supplied cohort's relationship to training is measured, not assumed.
SUPPLIED_COHORT_OVERLAP = (
    "**A supplied cohort's overlap with training is a measurement, not a "
    "contract.** An institutional or external patient set was not cut from this "
    "project's disease universe, so how much of it the model has labelled examples "
    "for is an open number — the upstream team reported 109 of 319 UDN diseases "
    "present in their simulated cohort. Establish it for yours with "
    "`scripts/audit_split_overlap.py`; it bounds what the cohort can be evidence "
    "of about unseen diseases."
)

#: Help for the cohort-kind argument, shared by every entry point that takes one.
COHORT_KIND_HELP = (
    "Whether this split was produced by this project's generator or supplied from "
    "outside. `generated` covers `train` and `val` and requires the workspace's "
    "`split_manifest.json`; `supplied` covers an institutional or external cohort, "
    "which carries no allocation and may not use a generated split's name. Stated "
    "rather than inferred from whether a manifest happens to be present, because "
    "that test cannot tell a supplied cohort from a workspace built before the "
    "allocation step."
)

#: The full `--split` help text. Assembled once so the two measurement entry
#: points cannot drift apart.
SPLIT_ARGUMENT_HELP = (
    "Which samples file to measure. **Required — there is no default.** "
    "Generated workspaces contain train and val only; a supplied cohort exists "
    "only where an evaluation protocol provided one, and must carry its own name. "
    "Two distinct limits apply, and they are not the same limit. (1) "
    + SELECTION_CONTAMINATION
    + " (2) "
    + GENERATED_VAL_IS_DISEASE_DISJOINT
    + " (3) "
    + SUPPLIED_COHORT_OVERLAP
)
