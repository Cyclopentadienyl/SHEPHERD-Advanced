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
#:
#: **Since M4 every measurement of a registered cohort version takes it** (R10,
#: contract §5.4): the report records the cohort's disease and phenotype-set
#: overlap with the training and validation inputs the checkpoint names, found
#: by digest, so the number travels with the result it qualifies.
SUPPLIED_COHORT_OVERLAP = (
    "**A supplied cohort's overlap with training is a measurement, not a "
    "contract.** An institutional or external patient set was not cut from this "
    "project's disease universe, so how much of it the model has labelled examples "
    "for is an open number — the upstream team reported 109 of 319 UDN diseases "
    "present in their simulated cohort. Each measurement of a registered cohort "
    "version records it as R10: disease and phenotype-set overlap with the training "
    "and validation inputs the checkpoint names, found by their digests in the "
    "workspace's own splits and then in each --training-evidence directory, and "
    "unverifiable, never zero, when those bytes are not found. "
    "`scripts/audit_split_overlap.py` audits a version against a workspace's "
    "generated splits. The overlap bounds what the cohort can be evidence of about "
    "unseen diseases."
)

#: The full `--split` help text: the generator's own splits, and nothing else.
#:
#: **A supplied cohort is no longer a split** (contract M4, decision 8). The loose
#: `<split>_samples.json` a workspace used to carry, and the `--cohort-kind` flag
#: that told the two apart, are gone: the kind of cohort is decided by which
#: argument names it, `--split` or `--cohort`, so it cannot be stated wrongly.
SPLIT_ARGUMENT_HELP = (
    "Which of the generator's splits to measure: train or val. Exactly one of "
    "--split and --cohort is required; there is no default. Needs --output. An "
    "institutional or external cohort is not a split of the workspace: it is "
    "imported as a registered cohort version (scripts/import_cohort.py) and "
    "measured with --cohort. A generated split's report is not registered in the "
    "ledger. Two distinct limits apply, and they are not the same limit. (1) "
    + SELECTION_CONTAMINATION
    + " (2) "
    + GENERATED_VAL_IS_DISEASE_DISJOINT
)

#: The full `--cohort` help text: a registered test cohort version (contract M4).
COHORT_ARGUMENT_HELP = (
    "The label of a registered test cohort version to measure "
    "(scripts/import_cohort.py), with --registry. Its samples are read once and "
    "must be the bytes its cohort.json binds, and the version must have been mapped "
    "against this workspace's graph and dataset record, or the run is refused "
    "before the checkpoint is read. The report is kept in the registry's report "
    "store, <registry>/versions/<label>/reports/<run id>/, so --output does not "
    "apply. "
    + SUPPLIED_COHORT_OVERLAP
)


# ---------------------------------------------------------------------------
# Recorded test results (M4): what the API, the UI and the CLI render beside a
# ledger record. Constants only; each surface renders them, none rewords them.
# ---------------------------------------------------------------------------

#: D3's label for a recorded result: what was ranked, by what, under which
#: policy. Shown with every recorded number so it cannot be quoted as another.
MODE_C_RESULT_LABEL = "Disease ranking by the model, approved scorer policy (Mode C)"

#: Why a recorded Mode C number is not the Diagnosis tab's performance yet.
#:
#: **Until B-1 the tab does not rank by Mode C.** The served scorer still mixes
#: the GNN score with the shortest-path term behind the path gate
#: (`docs/DISEASE_SCORER_POLICY.md`), so a Mode C number describes the model,
#: not what the tab shows a user. Removed when B-1 lands, not reworded.
PRE_B1_NOTE = (
    "This is the model's own Mode C disease ranking. Until B-1 ships, the Diagnosis "
    "tab ranks by the legacy GNN+SP mixture behind the path gate, so this number "
    "does not describe the tab's current ranking."
)

#: R10's four labels (contract §5.4), keyed as a record's `r10` block is: scope,
#: then label. Each has the name a surface shows and its one-line definition.
#:
#: **The definitions say what is counted and what is not.** Disease overlap counts
#: distinct diseases; phenotype-set overlap compares sets, so a case listing a term
#: twice and one listing it once are the same set but not the same scoring input.
R10_LABEL_TEXT = {
    "training": {
        "disease_overlap": {
            "name": "Training disease overlap",
            "definition": (
                "distinct test diseases that are also the disease of a case in the "
                "training inputs this checkpoint records, of all distinct test "
                "diseases, and the test cases whose disease is among them"
            ),
        },
        "phenotype_set_overlap": {
            "name": "Training phenotype-set overlap",
            "definition": (
                "test cases with the same disease and the same set of phenotypes as a "
                "case in the training inputs this checkpoint records, of all test "
                "cases; phenotype-set overlap ignores order and repetition, so the "
                "scoring inputs may differ"
            ),
        },
    },
    "validation": {
        "disease_overlap": {
            "name": "Validation disease overlap",
            "definition": (
                "distinct test diseases that are also the disease of a case in the "
                "validation inputs this checkpoint records, of all distinct test "
                "diseases, and the test cases whose disease is among them"
            ),
        },
        "phenotype_set_overlap": {
            "name": "Validation phenotype-set overlap",
            "definition": (
                "test cases with the same disease and the same set of phenotypes as a "
                "case in the validation inputs this checkpoint records, of all test "
                "cases; phenotype-set overlap ignores order and repetition, so the "
                "scoring inputs may differ"
            ),
        },
    },
}

#: What each R10 state says, and what it does not.
#:
#: **"none" is not "no leakage", and "unverifiable" is not a number.** Both are
#: statements about the inputs the checkpoint records and the evidence found for
#: them; a surface that shows a count for an unverifiable label, or reads "none"
#: as proof the model never saw these cases, says more than was measured.
R10_STATE_TEXT = {
    "overlap": "overlap",
    "none": (
        "none in the inputs this checkpoint records (not evidence about data "
        "outside them)"
    ),
    "unverifiable": (
        "unverifiable: the inputs this checkpoint records were not found or could "
        "not be read, so there is no count"
    ),
    "no_validation_inputs_recorded": "no validation inputs recorded",
}

#: What an empty answer from a ledger means (`src/evaluation/sidecar.py`).
RECORD_ABSENCE = (
    "This ledger holds no result for these weights; that does not mean they were "
    "never evaluated."
)

#: How each relation a report records (`RELATION_NAMES` in
#: `src/evaluation/measurement.py`) is named where a person reads it. Keyed by
#: those names; `tests/unit/test_registration.py` fails if the two differ.
RELATION_DISPLAY_NAMES = {
    "R1": "graph files match the workspace manifest",
    "R2": "generated split's samples match the workspace manifest",
    "R3": "model trained on this graph",
    "R5": "dataset record bound to this graph",
    "cohort_graph": "cohort version mapped against this graph",
}
