"""
Standing caveats on measurement, in one place.
==============================================

De-duplicated from ``scripts/measure_scorer.py`` and
``scripts/calibrate_mode_a.py``, which carried the same 922 characters
byte-for-byte. The same class of duplication as ``DEPLOYMENT_RELATIONSHIPS``
before it moved to ``src/utils/provenance.py``: a claim spelled differently in
two places cannot be compared, and a claim that has to be corrected in two
places will eventually be corrected in one.

**Constants, and pure lookups over them (`mode_label`, `served_scoring`, `mode_note`,
`metric_text`). No imports beyond the standard library, no I/O.** These strings
are rendered into ``--help`` text, where ``argparse`` performs ``%`` substitution,
so every literal percent sign here is escaped as ``%%``.

Module: src/evaluation/caveats.py
"""
from __future__ import annotations

from typing import Optional

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

#: Mode D's label for a recorded result (`docs/working/scorer-measurement/
#: PLAN_B05.md` §5): what the Diagnosis page serves today, which is not the
#: approved policy. `mode_label` adds the record's scoring mode and mixing
#: weight from its `score_semantics`.
_MODE_D_LABEL_HEAD = ("Served ranking today (Mode D): diseases reached by path discovery, "
                      "scored as the Diagnosis page scores them")
_MODE_D_LABEL_TAIL = "; not the approved scorer policy"
MODE_D_RESULT_LABEL = _MODE_D_LABEL_HEAD + _MODE_D_LABEL_TAIL

#: Shown beside a Mode D record, as `PRE_B1_NOTE` is beside a Mode C one.
MODE_D_NOTE = (
    "Mode D measures what the Diagnosis page serves today, over the cohort's import-eligible "
    "cases. Its report sets Mode C beside it on the same embeddings; a Mode C record shown "
    "here is a separate run. A Mode D number is not the model's approved ranking, and the "
    "cases excluded at import are counted beside it, not in it."
)

#: Each recorded mode's label, by the record's `scoring.mode`.
MODE_RESULT_LABELS = {"C": MODE_C_RESULT_LABEL, "D": MODE_D_RESULT_LABEL}

#: The note each mode's record is shown with.
MODE_RESULT_NOTES = {"C": PRE_B1_NOTE, "D": MODE_D_NOTE}


#: The label of a record whose mode cannot be read: never taken for Mode C.
MODE_NOT_RECORDED_LABEL = "Measurement mode not recorded"


def mode_label(mode: Optional[str], score_semantics: Optional[str] = None) -> str:
    """The label a record of `mode` is shown under. A Mode D record's served
    scoring mode and mixing weight are added from its `score_semantics`
    (``served:<scoring mode>:eta=<eta>:sp=<build id>``) when given. A mode this
    version does not know is named as such, and a mode that cannot be read
    (None) says so: neither is shown under another mode's label."""
    if not mode or not mode.strip():
        return MODE_NOT_RECORDED_LABEL
    label = MODE_RESULT_LABELS.get(mode)
    if label is None:
        return f"Mode {mode} (a measurement mode this version does not describe)"
    served = served_scoring(score_semantics) if mode == "D" else None
    return f"{_MODE_D_LABEL_HEAD} ({served}){_MODE_D_LABEL_TAIL}" if served else label


def served_scoring(score_semantics: Optional[str]) -> Optional[str]:
    """A Mode D record's served scoring mode and mixing weight, "hybrid, η = 0.5",
    from its `score_semantics` (``served:<scoring mode>:eta=<eta>:sp=<build id>``);
    None for any other value."""
    if not isinstance(score_semantics, str) or not score_semantics.startswith("served:"):
        return None
    fields = score_semantics.split(":")
    parts = dict(part.split("=", 1) for part in fields[2:] if "=" in part)
    return f"{fields[1]}, \u03b7 = {parts.get('eta', '?')}"


def mode_note(mode: Optional[str]) -> Optional[str]:
    """The note a record of `mode` is shown with, or None for a mode this
    version does not know or cannot read."""
    return MODE_RESULT_NOTES.get(mode) if mode and mode.strip() else None


#: Mode D's metric families by name prefix (PLAN_B05 §5), as a surface names
#: their case set: every import-eligible case (0 for one not reached), or the
#: reached cases only.
MODE_D_CASE_SETS = {"post_import_": "import-eligible", "reached_": "reached only"}


def _metric_words(name: str) -> str:
    if name == "mrr":
        return "MRR"
    if name == "mean_rank":
        return "mean rank"
    if name.startswith("hits@"):
        return f"Hits@{name[len('hits@'):]}"
    return name


def metric_text(mode: Optional[str], name: str) -> str:
    """A recorded metric as a surface names it, never without its mode: "Mode C
    MRR (untruncated)", "Mode C Hits@10", "Mode D MRR (import-eligible)", "Mode D
    mean rank (reached only)", "Mode D coverage"; "MRR (mode not recorded)" when
    the mode cannot be read."""
    if not mode or not mode.strip():
        return f"{_metric_words(name)} (mode not recorded)"
    if mode == "C" and name == "untruncated_mrr":
        return "Mode C MRR (untruncated)"
    if mode == "D":
        if name == "coverage":
            return "Mode D coverage"
        for prefix, case_set in MODE_D_CASE_SETS.items():
            if name.startswith(prefix):
                return f"Mode D {_metric_words(name[len(prefix):])} ({case_set})"
    return f"Mode {mode} {_metric_words(name)}"


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
    # The reason a surface gives only when it does not know the scope's evidence
    # state; otherwise it gives `R10_UNVERIFIABLE_REASON[evidence state]`.
    "unverifiable": (
        "unverifiable: the inputs this checkpoint records were not found or could "
        "not be read, or it names no such inputs, so there is no count"
    ),
    "no_validation_inputs_recorded": "no validation inputs recorded",
}

#: What an *unverifiable* label says, by its scope's evidence state (the
#: `EVIDENCE_STATES` of `src/evaluation/r10.py` other than "found"): contract
#: §5.4's three reasons. Each is the whole text for the label, in
#: `R10_STATE_TEXT`'s form, rendered in place of `R10_STATE_TEXT["unverifiable"]`.
#:
#: **One reason per state, because they call for different actions.** Bytes
#: that were not found may be found by measuring again with the directory that
#: holds them (`--training-evidence`); bytes found and not readable need the file
#: repaired. A checkpoint that names no such file at all is not a record the
#: pipeline wrote, and no directory can supply evidence for a digest that was
#: never recorded: telling that operator the inputs "were not found" sends them
#: looking for files that cannot help.
R10_UNVERIFIABLE_REASON = {
    "not_found": (
        "unverifiable: no file looked at holds the bytes this checkpoint records, "
        "so there is no count"
    ),
    "unreadable": (
        "unverifiable: the file holding the bytes this checkpoint records was found "
        "and could not be read, so there is no count"
    ),
    "not_recorded": (
        "unverifiable: this checkpoint names no such file, and its record is not one "
        "the pipeline wrote, so --training-evidence cannot help; there is no count"
    ),
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
