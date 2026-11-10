"""
Standing caveats on measurement, in one place.
==============================================

De-duplicated from ``scripts/measure_scorer.py`` and
``scripts/calibrate_mode_a.py``, which carried the same 922 characters
byte-for-byte. The same class of duplication as ``DEPLOYMENT_RELATIONSHIPS``
before it moved to ``src/utils/provenance.py``: a claim spelled differently in
two places cannot be compared, and a claim that has to be corrected in two
places will eventually be corrected in one.

**Constants, and pure lookups over them (`mode_label`, `mode_note`,
`metric_text`, `recorded_metric_lines`, `recorded_value_text`, `version_words`,
`exposure_text`), with the one name of each finetuned task (`TASK_NAMES`). No
imports beyond the standard library, no I/O.** These strings are rendered into
``--help`` text, where ``argparse`` performs ``%`` substitution, so every
literal percent sign here is escaped as ``%%``.

**One display for every recorded mode** (policy rev 15, R2; K8-6e.2). Words
exist only for the measurements this build describes (`MEASUREMENT_NAMES`). A
record of any other mode, a kept one included, is named by its own recorded
mode and shown as recorded: its metric names and values
(`recorded_metric_lines`) and its score semantics, never relabelled, hidden or
refused for its mode's name.

**Words by data kind** (research comparison §5 Stage 6). A record is read by
the kind of data it measured: a supplied cohort version's by its role and R10;
a KG edge set's and a generated split's by their own words (`version_words`,
never "cohort"), a KG edge set's run with its exposure (`EXPOSURE_READINGS`,
`exposure_text`), a generated split's with its generation identity
(`GENERATED_EXPOSURE_NOTE`). The KG edges' "held out" is scoped to the recorded
graph use and is never a patient cohort's holdout: no supplied cohort's or
generated split's words say it.

Module: src/evaluation/caveats.py
"""
from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping, Optional, Tuple

#: Why a metric measured on `val` is not an independent test result.
#:
#: **Permanent, and more important since the split became disease-disjoint, not
#: less.** Whatever `val` contains, the standard workflow selects the checkpoint
#: on it (by the finetuned task's selection metric, ``val_pooled_mrr``), so a
#: number measured there is a selection metric. That does not depend on how the
#: split was cut.
SELECTION_CONTAMINATION = (
    "**Not an independent evaluation.** In the standard workflow `val` is the "
    "split early stopping and checkpoint selection run on (by the finetuned "
    "task's selection metric, `val_pooled_mrr`), so a metric measured on it is "
    "a selection metric rather than an independent test estimate. This says "
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
#: **Since M4 every measurement of a supplied cohort version takes it** (R10,
#: contract §5.4): the report records the cohort's disease and phenotype-set
#: overlap with the training and validation inputs the checkpoint names, found
#: by digest, so the number travels with the result it qualifies.
SUPPLIED_COHORT_OVERLAP = (
    "**A supplied cohort's overlap with training is a measurement, not a "
    "contract.** An institutional or external patient set was not cut from this "
    "project's disease universe, so how much of it the model has labelled examples "
    "for is an open number — the upstream team reported 109 of 319 UDN diseases "
    "present in their simulated cohort. Each measurement of a supplied cohort "
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
    "measured with --cohort. A report from --split is not registered in the "
    "ledger: it has no registered version, so no data identity the ledger records "
    "results by (the same split kept with scripts/import_cohort.py --kind generated "
    "is measured with --cohort and registered); its report and ranks stay at "
    "--output. Two distinct limits apply, and they are "
    "not the same limit. (1) "
    + SELECTION_CONTAMINATION
    + " (2) "
    + GENERATED_VAL_IS_DISEASE_DISJOINT
)

#: The full `--cohort` help text: a registered data version of any kind (a
#: supplied cohort, contract M4; a KG edge set or a generated split, research
#: comparison Stages 4-5), and R10, which only a supplied cohort's measurement
#: records.
COHORT_ARGUMENT_HELP = (
    "The label of a registered data version to measure (scripts/import_cohort.py), "
    "with --registry: a supplied test cohort version, a KG edge set, or a generated "
    "split kept from a workspace, measured under its role as recorded. Its data are "
    "read once and must be the bytes its cohort.json binds, and the version must "
    "have been made against this workspace's graph and dataset record, or the run "
    "is refused before the checkpoint is read. The report is kept in the "
    "registry's report store, <registry>/versions/<label>/reports/<run id>/, so "
    "--output does not apply. "
    + SUPPLIED_COHORT_OVERLAP
)


# ---------------------------------------------------------------------------
# Recorded test results (M4): what the API, the UI and the CLI render beside a
# ledger record. Constants only; each surface renders them, none rewords them.
# ---------------------------------------------------------------------------

#: **Each finetuned task's name, the one every surface shows** (K6 decision 6):
#: the Test page's measurement (through `MEASUREMENT_NAMES`), the Models page's
#: filter and records, the Fine-tune page's options and the Diagnosis card's
#: name line read it, and `src.webui.components.renderers.record_kind_text`
#: builds on it. Keyed by the model spec's tasks (`src.models.bundle.spec.TASKS`,
#: which this standard-library-only module does not import;
#: `tests/unit/test_webui_renderers.py` pins the two equal).
TASK_NAMES = {"disease": "Disease", "gene": "Gene", "patients_like_me": "Patients-like-me"}

#: **Each measurement's name where a page offers or names one** (the Test
#: page's Measurement options and its job line), keyed by protocol name
#: (`src.evaluation.protocols.names`, pinned equal by
#: `tests/unit/test_kg_link_prediction.py`): a task's measurement by its task's
#: name (`TASK_NAMES`), and the common KG link-prediction measurement by its
#: own words (research comparison Stage 5). Never a task name: no task filter
#: or option reads it.
MEASUREMENT_NAMES = {**TASK_NAMES, "kg_link_prediction": "KG link prediction"}


def task_list_text(names: Iterable[str]) -> str:
    """Task names as one phrase, "Gene", "Gene and Patients-like-me",
    "Disease, Gene and Patients-like-me": the one way every page and the runner
    name a set of tasks (each name already a `TASK_NAMES` word)."""
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"

#: Each described measurement's label, by the record's `scoring.mode`.
#: A finetuned model bundle's disease task measurement
#: (`src.evaluation.task_measurement`): its own scoring semantics, never grouped
#: with a record of another mode.
DISEASE_TASK_RESULT_LABEL = ("Disease task (model bundle): every disease of the model's "
                             "graph view, average ranks")
DISEASE_TASK_NOTE = ("A score from the finetuned model's disease head, uncalibrated. Pooled "
                     "MRR is over every case's truth; coverage and the conservative reading "
                     "count the cases the model could not rank.")

GENE_TASK_RESULT_LABEL = ("Gene task (model bundle): each case's own candidate list, average "
                          "ranks")
GENE_TASK_NOTE = ("A score mixing the finetuned model's gene head with shortest-path "
                  "similarity, uncalibrated. Every causal gene is a positive; a rank depends on "
                  "the list's length, which the report gives.")

PATIENTS_LIKE_ME_RESULT_LABEL = ("Patients-like-me (model bundle): each patient against every "
                                 "other patient of the cohort, average ranks")
PATIENTS_LIKE_ME_NOTE = ("A similarity from the finetuned model's patient tower, uncalibrated. A "
                         "pair is relevant when the two patients share a causal gene; a patient "
                         "with no other relevant patient has no recall, not a recall of 0.")

#: The common KG link-prediction measurement (`task_measurement.run_kg_link_prediction`,
#: research comparison Stage 5): a base's and a finetune's LP head on one KG
#: edge set, its negatives rule-generated (never proven non-relations).
KG_LINK_PREDICTION_RESULT_LABEL = ("KG link prediction (model bundle): each positive edge of a "
                                   "KG edge set against its one rule-generated negative, over "
                                   "the version's message-passing graph")
KG_LINK_PREDICTION_NOTE = ("A raw score from the model's LP head (DistMult), uncalibrated; LP "
                           "accuracy decides at raw > 0. The negatives are rule-generated, never "
                           "proven non-relations. The report's exposure says how the measured "
                           "edges stand to the model's recorded graph use: a figure after "
                           "full-view exposure is never called held out.")

MODE_RESULT_LABELS = {"disease": DISEASE_TASK_RESULT_LABEL, "gene": GENE_TASK_RESULT_LABEL,
                      "patients_like_me": PATIENTS_LIKE_ME_RESULT_LABEL,
                      "kg_link_prediction": KG_LINK_PREDICTION_RESULT_LABEL}

#: The note each described measurement's record is shown with.
MODE_RESULT_NOTES = {"disease": DISEASE_TASK_NOTE, "gene": GENE_TASK_NOTE,
                     "patients_like_me": PATIENTS_LIKE_ME_NOTE,
                     "kg_link_prediction": KG_LINK_PREDICTION_NOTE}


#: The label of a record whose mode cannot be read: never taken for another mode.
MODE_NOT_RECORDED_LABEL = "Measurement mode not recorded"


def mode_label(mode: Optional[str]) -> str:
    """The label a record of `mode` is shown under: a described measurement's
    own label; any other mode named by its recorded value as one this version
    does not describe; a mode that cannot be read (None) says so. Neither of
    the last two is shown under another mode's label, and neither is refused."""
    if not mode or not mode.strip():
        return MODE_NOT_RECORDED_LABEL
    label = MODE_RESULT_LABELS.get(mode)
    if label is None:
        return f"Mode {mode} (a measurement mode this version does not describe)"
    return label


def mode_note(mode: Optional[str]) -> Optional[str]:
    """The note a record of `mode` is shown with, or None for a mode this
    version does not describe or cannot read."""
    return MODE_RESULT_NOTES.get(mode) if mode and mode.strip() else None


def recorded_value_text(value: Any) -> str:
    """A recorded value as `recorded_metric_lines` shows it: its JSON text,
    unrounded ("0.5", "3.0", a string quoted)."""
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=True)
    except (TypeError, ValueError):  # not JSON (never so from a parsed file)
        return repr(value)


def recorded_metric_lines(metrics: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """**The one rendering of "as recorded"** (R2): each metric of the mapping
    a page gives, in its order, a ``(name, value)`` pair: the name as written
    and the value's JSON text, unrounded (`recorded_value_text`). The text is
    the value as that mapping holds it: a report or usage entry keeps a whole
    number as written ("1"), while the API's record view holds every metric as
    a number with a fraction, so a page showing a record through it reads
    "1.0" for the same value. Every page wraps the pairs in its own markup and
    escapes them; nothing here names, rounds, orders or drops a metric by its
    name or its record's mode."""
    return [(str(name), recorded_value_text(value)) for name, value in metrics.items()]


#: A task measurement's metrics (`src.models.bundle.metrics.ranking_report`),
#: as a surface names them; a family that takes a cut-off k gets it appended.
_TASK_METRIC_WORDS = {"pooled_mrr": "pooled MRR", "macro_mrr": "macro MRR",
                      "best_positive_mrr": "best-positive MRR",
                      "conservative_pooled_mrr": "conservative pooled MRR",
                      "coverage": "coverage"}
_TASK_METRIC_FAMILIES = {"pooled_top": "pooled top-", "macro_recall@": "macro recall@",
                         "macro_precision@": "macro precision@"}

#: The KG link-prediction measurement's metrics (metric schema 3,
#: `src.evaluation.task_measurement.LP_METRICS`, pinned equal by
#: `tests/unit/test_kg_link_prediction.py`), as a surface names them after the
#: measurement's name: "KG link prediction ROC-AUC". Each metric's unit and
#: denominator are the report's (and a KG edge set's ledger record's
#: ``measured``), never restated here.
LP_METRIC_WORDS = {"lp_accuracy": "accuracy (raw > 0)", "lp_roc_auc": "ROC-AUC",
                   "lp_average_precision": "average precision",
                   "lp_loss": "loss (the LP head's hinge)"}


def _metric_words(name: str) -> str:
    if name in _TASK_METRIC_WORDS:
        return _TASK_METRIC_WORDS[name]
    for prefix, words in _TASK_METRIC_FAMILIES.items():
        if name.startswith(prefix):
            return f"{words}{name[len(prefix):]}"
    return name


def metric_text(mode: Optional[str], name: str) -> str:
    """A recorded metric as a surface names it, never without its mode: a
    described measurement's in words, by its measurement's name
    (`MEASUREMENT_NAMES`): "Gene pooled MRR", "Disease pooled top-10",
    "Patients-like-me macro recall@10", "KG link prediction ROC-AUC" (a name
    the measurement does not write keeps its recorded name after the
    measurement's); any other mode's by its recorded mode and the name as
    recorded ("Mode <mode> <name>"); "pooled MRR (mode not recorded)" when the
    mode cannot be read."""
    if not mode or not mode.strip():
        return f"{_metric_words(name)} (mode not recorded)"
    if mode in TASK_NAMES:
        return f"{TASK_NAMES[mode]} {_metric_words(name)}"
    if mode == "kg_link_prediction":
        return f"{MEASUREMENT_NAMES[mode]} {LP_METRIC_WORDS.get(name, name)}"
    return f"Mode {mode} {name}"


# ---------------------------------------------------------------------------
# Data kinds (research comparison §5 Stages 4-6): how a version of each kind,
# and a KG edge set's run's exposure, are said.
# ---------------------------------------------------------------------------
#: Each data kind as a sentence names it (`src.evaluation.cohort.COHORT_KINDS`'
#: words): a protocol's refusal (`src.evaluation.protocols`), a ledger
#: record's refusal (`src.evaluation.sidecar`) and `version_words`. Never
#: "cohort" for a KG edge set or a generated split.
DATA_KIND_WORDS = {"generated": "a generated split", "supplied": "a supplied cohort version",
                   "kg_edges": "a KG edge set"}


def version_words(kind: Any, role: Any) -> str:
    """A version of another kind than a supplied cohort as a sentence names
    it, by its kind and role: ``a KG edge set (test)``, ``a generated split
    (val)``. Never "cohort": a KG edge set holds edges, and a generated split
    is the workspace generator's, never a supplied cohort (research comparison
    §5 Stage 4). `src.evaluation.cohort_registry.version_words` is this
    function; every reader and refusal names a version by it."""
    if kind in ("kg_edges", "generated"):
        return f"{DATA_KIND_WORDS[kind]} ({role})"
    return f"a version of kind {kind!r}"


#: The version of the KG link-prediction exposure definition below: the
#: states, their readings and how each count is taken
#: (`src.evaluation.exposure`, which writes a run's exposure block by it). A
#: reader reads a block by the version it records.
EXPOSURE_DEFINITION_VERSION = "kg-lp-exposure/v1"

#: The states an exposure block holds, each read by `EXPOSURE_READINGS`.
HELD_OUT = "held_out"
IN_RECORDED_GRAPH_USE = "in_recorded_graph_use"
ANOTHER_SPLIT = "another_split"
AFTER_FULL_VIEW_EXPOSURE = "after_full_view_exposure"
NOT_RECORDED = "not_recorded"
EXPOSURE_STATES = (HELD_OUT, IN_RECORDED_GRAPH_USE, ANOTHER_SPLIT, AFTER_FULL_VIEW_EXPOSURE,
                   NOT_RECORDED)

#: Each state's reading; "{k}" and "{n}" are the block's counts. **A KG edge
#: set's "held out" is scoped to the recorded graph use** of the measured
#: model, never a patient cohort's holdout or an institutional acceptance's.
EXPOSURE_READINGS = {
    HELD_OUT: "held out relative to the recorded training and validation graph use",
    IN_RECORDED_GRAPH_USE: ("{k} of {n} measured edges were in the recorded training or "
                            "validation graph use"),
    ANOTHER_SPLIT: ("{k} of {n} measured edges were in this model's recorded training or "
                    "validation graph use, on another split of the same view"),
    AFTER_FULL_VIEW_EXPOSURE: "after full-view exposure",
    NOT_RECORDED: "exposure not recorded",
}

#: What every exposure block says of itself and of the negatives.
EXPOSURE_CAVEAT = (
    "Scoped to the recorded evidence: what the model's record and this KG edge set say, read by "
    "this build. It does not establish that no other exposure happened (another run, a "
    "person's choice, a file outside the record), and it is never a gate. The negatives are "
    "rule-generated (another tail of the same relation in the pair's batch, never the "
    "positive's own) and are never claimed as proven non-relations: a negative may be a true "
    "relation the graph does not hold.")

#: The unit of an exposure block's counts, said in the block.
MEASURED_EDGE_UNIT = ("positive edges scored: the (head, relation, tail) of each row of the "
                      "pairs file, a directed edge of the graph view")


def exposure_text(exposure: Any) -> str:
    """A KG edge set run's exposure block (a report's, its ledger record's,
    or the API's view of it, which keeps the block's shape) as a surface says
    it: for a block of this build's definition (`EXPOSURE_DEFINITION_VERSION`),
    its state's reading with its counts ("2 of 6 measured edges were in ...");
    for a block of another definition, its reading as recorded, naming that
    definition; "exposure not recorded" when there is no block or its state or
    counts do not read. Never "unseen", and never a gate."""
    if not isinstance(exposure, Mapping):
        return EXPOSURE_READINGS[NOT_RECORDED]
    version, state = exposure.get("definition_version"), exposure.get("state")
    if version == EXPOSURE_DEFINITION_VERSION and state in EXPOSURE_READINGS:
        edges = exposure.get("measured_edges")
        edges = edges if isinstance(edges, Mapping) else {}
        k, n = edges.get("in_recorded_graph_use"), edges.get("n")
        if "{k}" not in EXPOSURE_READINGS[state] or (type(k) is int and type(n) is int):
            return EXPOSURE_READINGS[state].format(k=k, n=n)
        return EXPOSURE_READINGS[NOT_RECORDED]
    reading = exposure.get("reading")
    if isinstance(version, str) and isinstance(reading, str) and reading.strip():
        return f"{reading} (as recorded under the exposure definition {version})"
    return EXPOSURE_READINGS[NOT_RECORDED]


#: How a generated split's run stands to a model's training, where a KG edge
#: set's run shows its exposure: by the split's generation identity and role,
#: which its record keeps (research comparison §5 Stage 6). Never a test set's
#: words, and never computed against the model's own record.
GENERATED_EXPOSURE_NOTE = (
    "A generated split's relation to a model's training is its allocation's: the generation "
    "and allocation below cut the workspace's patients into its train and val splits by "
    "disease. Whether the measured model was trained on this generation is not compared "
    "here; a val split is the one checkpoint selection reads.")


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
#: §5.4's three reasons and, from R10 definition version 2 (K8-6r.3), a search
#: left incomplete. Each is the whole text for the label, in `R10_STATE_TEXT`'s
#: form, rendered in place of `R10_STATE_TEXT["unverifiable"]`. A version-1
#: block's states read with these same words: "not_found" meant then what it
#: means now, every candidate read and compared.
#:
#: **One reason per state, because they call for different actions.** Bytes
#: that were not found may be found by measuring again with the directory that
#: holds them (`--training-evidence`); bytes found and not readable need the file
#: repaired; a place that could not be read (its path and error are in the
#: report) needs the account that measures to be able to read it. A checkpoint
#: that names no such file at all is not a record the pipeline wrote, and no
#: directory can supply evidence for a digest that was never recorded: telling
#: that operator the inputs "were not found" sends them looking for files that
#: cannot help.
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
    "search_incomplete": (
        "unverifiable: a place that may hold the bytes this checkpoint records could "
        "not be read, so there is no count. Make it readable to the account that "
        "measures, and measure again; the report names the place"
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
