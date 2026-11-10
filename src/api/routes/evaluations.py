"""
Recorded results for the loaded model: a read-only view of the ledger.
=======================================================================
`PLAN_TEST_RESULTS.md` D8 and §6 phase 1, step 4: the Diagnosis model status
shows what the ledger beside the loaded checkpoint holds for those weights,
each record with its run and how its condition group compares in the store.

Endpoint:
    GET /evaluations/loaded-model — the loaded model's recorded results

**Its own router, outside `pipeline.router`.** SPEC_4's C3 class covers the
reload, the UI configuration and checkpoint inspection; a deployment applying
B2 by unmounting or protecting that router as a whole keeps this view, which
reads the ledger, the registry's version manifests (each ``cohort.json`` once,
to find each shown record's version by the cohort digest its key carries) and
the report store of each shown record's version; no checkpoint, and no write
(D8).

**Found by the served read, filtered by its digest.** Application state is read
once, and both the ledger's directory and the digest the ledger is filtered by
come from that pipeline's own checkpoint read
(`DiagnosisPipeline.served_checkpoint_read`). A reload published meanwhile
therefore cannot pair one model's ledger with another model's digest, whichever
thread runs the handler: the route runs on the event loop, and the dashboard's
read-only view runs it on an event loop of its own in a worker thread
(`models_page._route_records`). It then ``await``s the file reads below, which
run in the threadpool (`run_in_threadpool`) so that the scans of the report
stores do not hold the event loop; they work from the identity already taken.
The digest used is echoed, and the panel shows the records only when it equals
the digest of the status it read. Because records are found by digest and not by
file name, a model replaced by other bytes never shows its predecessor's records,
and the same bytes moved together with their ledger still find theirs.

**The ledger beside the checkpoint**, ``Path(read.path).parent /
evaluations.json``, where the read's path is the checkpoint's resolved path as
the pipeline read it: the rule `register_report` writes by
(``Path(checkpoint).resolve().parent``), with the directory settled when the
checkpoint was read, not on each request. It is read on every
request through `read_ledger`, the one ledger reader, so a registration made
after a page loaded appears on the next load. Nothing comes from the client: no
query parameter, no path, no digest.

**Each record names its run** (policy rev 16, K8-6r.2; ``run_id``): the run
directory of the version's report store its report was registered from. A
record written before each run had its own record (every record of a schema-3
ledger, and those before a schema-4 ledger's run-identity boundary) has
``run_id`` null, shown as "not recorded": which run it came from was never
recorded, and nothing is invented for it. ``historical_overlap`` lists the
records it may overlap (`sidecar.possible_overlaps`): for a record naming its
run, the earlier merged records holding its report; for a merged record, the
named records whose report it holds. **The two kinds stay apart**: an overlap
is never confirmed, and a merged record and a named run are never counted as
two runs confirmed to be independent. A record whose run identity cannot be
read (at or after the boundary without a readable ``run_id``, or the boundary
itself unreadable) is not shown as a result; it is listed in
``unreadable_records`` with its reason, never shown as historical.

**Each record's group description** (``contest``; BACKLOG 36, the M5 review's
P2-6): how the reports its cohort version's store keeps under the record's
condition group compare, from the one implementation the dashboard shows too
(`src.api.services.record_states.contest_states`, computed for the records shown
only: the registry's version manifests read once each, then one scan of each
shown record's version store, no other store listed). **The state names stay
as machine values with a new meaning** (policy rev 16, Decision 3):
``contested`` says runs under these conditions differ, ``cannot_be_checked``
that the comparison is incomplete (no registry configured, a version the
registry does not list, a report of the group that cannot be compared), and
``uncontested`` neither. They describe and decide nothing: every record is
returned, none is hidden, ordered out or refused by them. A value that is not a
finite number is null, and a ``contested`` description whose details fail
validation is shown contested without them. Investigation notes are not shown
here, nor by the Models page; only the Test page lists them (BACKLOG 40).

**One reader, discriminated by data kind** (research comparison §5 Stage 6).
A record is read by its own ``cohort.kind`` into that kind's view, every kind
with the same common fields (its mode, metrics, relations, environment,
registration, run and digests): a supplied cohort version's as `RecordView`,
exactly as before (its cohort, counts and R10); a KG edge set's as
`KgEdgesRecordView` (its view, split, message-passing graph and pairs by
digest and rule, its counts with their units, no R10, the run's counts and
each metric's status, and its exposure, `ExposureView`); a generated split's
as `GeneratedRecordView` (its role and generation identity, its counts with
their units, no R10). A record whose kind this build does not read is listed
in ``unreadable_records``, naming the field and never its value, and hides no
other record. Each view keeps the record's own paths, so a refusal names the
key a hand edit of the ledger corrects.

**Aggregates only** (D7). A record is shown as its counts, metrics, relation
states, R10 states and counts, environment class, GPU and native build identity
(each component's version, native libraries' count and aggregate digest, and
source route; never a file name), dataset origin, registration, its run,
digests and group description (field names, run ids and
values; never a problem's words, which may name a server path: an unreadable
record's reason is built from field names and fixed words only). No path (a record's R10
evidence paths among them), no case id, no rank; a ledger problem names the
file, never its directory. A KG edge set's exposure is shown by its definition
version, state, reading, counts and sources' identities; its reason, which may
quote the model's record, and its caveat stay in the ledger (the surfaces say
the caveat in this build's words, `caveats.EXPOSURE_CAVEAT`).

**Expected failures are states; nothing else is swallowed.** A ledger that cannot
be read as a whole is ``ledger_not_read`` with its reason, and none of its
records are shown; a schema-4 ledger whose run-identity boundary cannot be read
is still read, its records naming a run shown and the others listed. A record
for these weights that does not have the shape registration writes, after a
hand edit (the sanctioned correction path), is not raised (a 500 here would
take the model status with it) and never shown as a result: **it is listed in
``unreadable_records``** (K8-6e.1; `read_records`, the one listing every reader
uses) by its index in the ledger, its registration sequence, report digest and
run where it still carries them readably, and the reason in one line of at most
300 characters that names the refused fields by their path in the ledger
record, never their values (`unreadable_reason`). ``records_not_shown`` keeps its meaning and equals the
list's length. The server log says the same line with the index. Any other
exception is a defect and reaches the client as one.

No write, no `find_checkpoint`, no `torch.load`.

Module: src/api/routes/evaluations.py
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from fastapi import APIRouter
from pydantic import Discriminator, Field, Tag, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from src.api.routes.diagnose import _SHA256, _Strict
from src.evaluation.sidecar import (
    LEDGER_FILENAME,
    REGISTERED_RELATIONS,
    RUN_ID_PATTERN,
    RecordIdentity,
    environment_class,
    indexed_records_for,
    possible_overlaps,
    read_ledger,
    record_identities,
)
# How long an unreadable record's reason may be: `problem_line`'s own bound,
# so the line it gives always fits the field.
from src.utils.checkpoint_paths import _PROBLEM_LIMIT as _REASON_LIMIT
from src.utils.run_environment import native_identity

if TYPE_CHECKING:
    from src.utils.fingerprint import ReadIdentity

logger = logging.getLogger(__name__)

router = APIRouter()

#: Where the ledger is, said without its path: this view names no path on the
#: server, and the checkpoint the status describes fixes the directory.
LEDGER_WHERE = "beside the loaded checkpoint"

#: States with no checkpoint digest, and so no ledger to look in.
_NOTHING_TO_LOOK_UP = ("no_model",)
#: R10 states that carry a count (spec §3.2); every other state carries none.
_COUNTED_R10_STATES = ("overlap", "none")

_Count = Annotated[int, Field(strict=True, ge=0)]
_Metric = Annotated[float, Field(strict=True, allow_inf_nan=False)]
_Digest = Annotated[str, Field(pattern=_SHA256)]
_Text = Annotated[str, Field(min_length=1)]
_RunId = Annotated[str, Field(pattern=f"^{RUN_ID_PATTERN.pattern}$")]
_Sequence = Annotated[int, Field(strict=True, ge=1)]
_R10State = Literal["overlap", "none", "unverifiable", "no_validation_inputs_recorded"]


# =============================================================================
# Response models
# =============================================================================
class LedgerView(_Strict):
    """The ledger an answer was read from, named without its path."""

    file: str = Field(..., description="The ledger's file name")
    where: str = Field(
        ..., description="Where it is: the directory of the checkpoint read the loaded "
        "model was built from")
    sha256: Optional[_Digest] = Field(
        ..., description="SHA-256 of the ledger bytes parsed; null when there is no "
        "file or it was not read")
    problem: Optional[str] = Field(
        ..., description="Why the ledger was not read, naming the file and not its "
        "directory; null otherwise")


class RecordCohort(_Strict):
    """The registered cohort version a result was measured on."""

    label: _Text
    role: _Text
    kind: Literal["supplied"]
    samples_sha256: _Digest
    source_sha256: _Digest
    version_manifest_sha256: _Digest = Field(
        ..., description="SHA-256 of the cohort version's manifest (`cohort.json`) the "
        "run read: the record's cohort digest, a key field. A label is unique only "
        "within one registry, so two imports of one label are two versions, and "
        "differ here")


class RecordCounts(_Strict):
    """The cohort's denominators (D6): every source case is ranked or excluded."""

    source_cases: _Count
    ranked_eligible: _Count
    excluded: Dict[str, _Count]
    n_excluded: _Count

    @model_validator(mode="after")
    def _the_counts_add_up(self) -> "RecordCounts":
        # The cohort version's own equalities (spec §2.3). Shown counts that do
        # not add up would give a coverage that no cohort had.
        if sum(self.excluded.values()) != self.n_excluded:
            raise ValueError("the exclusion counts do not add up to n_excluded")
        if self.ranked_eligible + self.n_excluded != self.source_cases:
            raise ValueError("ranked_eligible and n_excluded do not add up to source_cases")
        return self


def _check_r10_counts(state: str, k: Optional[int], n: int, *more: Optional[int]) -> None:
    """A count exactly where the state has one (spec §3.2): never a number for
    an unverifiable label, never a missing one for a counted label."""
    counts = (k, *more)
    if state not in _COUNTED_R10_STATES:
        if any(value is not None for value in counts):
            raise ValueError(f"an R10 label in state {state!r} carries a count")
        return
    if any(value is None for value in counts):
        raise ValueError(f"an R10 label in state {state!r} has no count")
    if k > n:
        raise ValueError("an R10 label counts more than its denominator")
    if state == "none" and any(counts):
        raise ValueError("an R10 label in state 'none' counts an overlap")
    if state == "overlap" and not all(counts):
        raise ValueError("an R10 label in state 'overlap' counts none")


class R10Evidence(_Strict):
    """Whether the inputs the checkpoint records were found; the file's path and
    any read error stay in the ledger. "search_incomplete" (R10 definition
    version 2): no file that was read holds them, and a place that may could
    not be read. A version-1 record's states mean what they meant then."""

    state: Literal["found", "not_found", "unreadable", "not_recorded", "search_incomplete"]


class R10DiseaseOverlap(_Strict):
    """``k`` of ``n`` distinct test diseases, and the test ``cases`` among them."""

    state: _R10State
    k: Optional[_Count]
    n: _Count
    cases: Optional[_Count]

    @model_validator(mode="after")
    def _counts_follow_the_state(self) -> "R10DiseaseOverlap":
        _check_r10_counts(self.state, self.k, self.n, self.cases)
        return self


class R10PhenotypeSetOverlap(_Strict):
    """``k`` of ``n`` test cases whose disease and phenotype set occur in scope."""

    state: _R10State
    k: Optional[_Count]
    n: _Count

    @model_validator(mode="after")
    def _counts_follow_the_state(self) -> "R10PhenotypeSetOverlap":
        _check_r10_counts(self.state, self.k, self.n)
        return self


class R10Scope(_Strict):
    evidence: R10Evidence
    disease_overlap: R10DiseaseOverlap
    phenotype_set_overlap: R10PhenotypeSetOverlap


class R10View(_Strict):
    """R10's four labels, by scope."""

    training: R10Scope
    validation: R10Scope

    @model_validator(mode="after")
    def _only_validation_can_be_unrecorded(self) -> "R10View":
        # A checkpoint with no training inputs recorded is not the pipeline's,
        # which makes its training labels unverifiable (spec §0.16), never this.
        for label in (self.training.disease_overlap, self.training.phenotype_set_overlap):
            if label.state == "no_validation_inputs_recorded":
                raise ValueError("a training label is 'no_validation_inputs_recorded'")
        return self


class RecordGpu(_Strict):
    """The GPU a CUDA run used, each fact null where it could not be read."""

    name: Optional[str]
    driver: Optional[str]


class RecordNativeLibraries(_Strict):
    """A component's native libraries, without their names: their count and
    the SHA-256 aggregate of their names and digests
    (`src.utils.run_environment.read_pyg_native` gives the method)."""

    state: _Text = Field(
        ..., description="recorded; not installed; cannot be read (the count is set where the "
        "file list was read); or not recorded. Shown as written")
    count: Optional[_Count]
    aggregate_sha256: Optional[_Digest]


class RecordComponentSource(_Strict):
    """Where a component came from: the deploy route of the source receipt,
    only where that receipt matched what was installed."""

    state: Literal["recorded", "not recorded"]
    route: Optional[_Text] = None

    @model_validator(mode="after")
    def _a_route_only_when_recorded(self) -> "RecordComponentSource":
        if (self.route is not None) != (self.state == "recorded"):
            raise ValueError("a route is given exactly when the source is recorded")
        return self


class RecordComponent(_Strict):
    """One named component of the measuring run's environment (E1), by its
    identity: what it is, its distribution version, its native libraries and
    its source's route. What was requested, installed, available, selected or
    in effect stays in the ledger record. The collector holds every component
    to this shape (`src.utils.run_environment.component_facts`), so a record
    it wrote is never refused here."""

    kind: _Text
    version: Optional[str]
    native: RecordNativeLibraries
    source: RecordComponentSource


class RecordRegistration(_Strict):
    sequence: Annotated[int, Field(strict=True, ge=1)]
    registered_at: _Text


class _RecordViewFields(_Strict):
    """The fields every kind's record view holds, in the order a supplied
    cohort's view always had them; each kind's view (`RecordView`,
    `KgEdgesRecordView`, `GeneratedRecordView`) states its own ``cohort``,
    ``counts`` and ``r10`` in their places and adds what only its kind
    holds. Never shown as itself."""

    cohort: Any
    counts: Any
    mode: _Text = Field(
        ..., description="the record's measurement mode (`scoring.mode`), as recorded: the "
        "protocol its report names. This build runs disease, gene and patients_like_me, each "
        "offered where a model's capabilities admit it, today a finetuned model that carries "
        "its task head (what the Test page runs), and kg_link_prediction, on a KG edge set, "
        "for every model with the LP head; a "
        "record of any other mode is read and returned the same way. A surface labels each "
        "record by its own mode")
    score_semantics: _Text = Field(
        ..., description="the record's `scoring.score_semantics`, as recorded: what its score "
        "is, in its measurement's own words (a task record: the task head's score). Returned "
        "whole, never split or interpreted")
    metrics: Dict[str, _Metric] = Field(
        ..., description="the record's metrics, as recorded: each name as its measurement "
        "wrote it, with its value. A task mode's: `pooled_mrr`, `pooled_top<k>`, "
        "`macro_recall@<k>`, `macro_precision@<k>`, `macro_mrr`, `best_positive_mrr` and "
        "`conservative_pooled_mrr` (each left out when it has no case to average), and "
        "`coverage`, the share of the cases measured that were ranked. KG link prediction's: "
        "`lp_accuracy`, `lp_roc_auc`, `lp_average_precision` and `lp_loss`, each that could "
        "be computed (a KG edge set's `measured.metrics` gives every metric's status, unit "
        "and denominator). None of them is the Trainer's validation figure")
    n_ranked: _Count = Field(
        ..., description="the record's `n_ranked`, as recorded: for a task mode, the cases "
        "ranked; for KG link prediction, the pairs scored")
    relations: Dict[str, Literal["verified", "not_applicable"]]
    r10: Any
    environment_class: str
    deterministic_algorithms: Optional[bool] = Field(
        ..., description="whether deterministic algorithms were on in the measuring run, as the "
        "record's regime says (a CPU run's environment class does not carry it); null when the "
        "record does not say")
    gpu: Optional[RecordGpu] = Field(..., description="null for a run off CUDA")
    native_identity: Optional[Dict[str, RecordComponent]] = Field(
        ..., description="The native build identity of each named component of the measuring "
        "run's environment (`environment.facts.components`, recorded since research-comparison "
        "Stage 2b; today PyG's native extensions), by name: its kind and distribution version, "
        "its native libraries' count and aggregate SHA-256 (or why not), and its source's "
        "deploy route where the source receipt matched what was installed. Two builds of one "
        "version differ in the digest. null for a record whose environment names no "
        "components (written before; shown as not recorded). Recorded, never compared or "
        "enforced (R11)")
    dataset_origin: Literal["files", "synthetic"]
    scoring_semantics_digest: _Digest
    measurement_semantics_digest: _Digest
    registration: RecordRegistration
    run_id: Optional[_RunId] = Field(
        ..., description="The run the record was registered from: its run directory in the "
        "version's report store (time, checkpoint and a random suffix; never a path). null for "
        "a record written before each run had its own record (shown as \"not recorded\"): "
        "which run it came from was never recorded")
    historical_overlap: List[_Sequence] = Field(
        ..., description="Registration sequences of the records this one may overlap, derived "
        "from the ledger: for a record naming its run, the records merged before each run had "
        "its own record that hold its report; for such a merged record, the records naming a "
        "run whose report it holds. Not confirmed: whether they are one execution was never "
        "recorded, and they are never counted as two independent runs. Empty when none")
    n_corroborating: _Count = Field(
        ..., description="Reports merged into this record before each run had its own record "
        "(K8-6r.2); 0 for every record that names its run. A merged report is not a confirmed "
        "independent run")
    source_artifact_digest: _Digest = Field(
        ..., description="SHA-256 of the report the record was built from")

    @field_validator("relations")
    @classmethod
    def _a_registered_results_states(cls, value: Dict[str, str]) -> Dict[str, str]:
        # Registration refuses any other states (`build_record`), so a record
        # carrying them was edited into a claim no measurement made.
        if value != REGISTERED_RELATIONS:
            raise ValueError("the relation states are not a registered result's")
        return value


class RecordView(_RecordViewFields):
    """One recorded result on a supplied cohort version, as a person reads it:
    aggregates and digests only."""

    cohort: RecordCohort
    counts: RecordCounts
    r10: R10View


def _units_follow_the_counts(counts: Dict[str, int], units: Dict[str, str]) -> None:
    """Every count with its unit, and a unit only for a count (the registry's
    rule for a version's counts, the evaluator's for a run's)."""
    if set(counts) != set(units):
        raise ValueError("the counts and their units do not name the same counts")


class KgEdgesGraphView(_Strict):
    """The graph view a KG edge set's split was cut from, by its digest."""

    digest: _Digest


class KgEdgesSplit(_Strict):
    """The recorded edge split a KG edge set is one part of: its rule, seed and
    digest, the view it was cut from, and each part's directed edges."""

    rule: _Text
    seed: Annotated[int, Field(strict=True)]
    digest: _Digest
    view_digest: _Digest
    edges: Dict[str, _Count]


class KgEdgesMessagePassing(_Strict):
    """The graph a KG edge set's run passes messages over: its split's part
    (train), its directed edges, and whether the measured edges are in it."""

    part: _Text
    edges: _Count
    includes_measured_edges: Annotated[bool, Field(strict=True)]


class KgEdgesPairs(_Strict):
    """The stored pairs a KG edge set's run scores, by the digest of the file's
    bytes and of the pairs, with their rule, seed and batch boundaries' size;
    the file's name and each batch's pairs stay in the ledger."""

    part: _Text
    rule: _Text
    seed: Annotated[int, Field(strict=True)]
    sha256: _Digest
    lp_pairs_digest: _Digest
    rows: _Count
    batch_size: _Count
    batches: _Count


class KgEdgesRecordCohort(_Strict):
    """The KG edge set a result was measured on: its label, its role (the
    split's part it is), its version's manifest digest, its view, split,
    message-passing graph and pairs. Never called a cohort where a person reads
    it ("a KG edge set (test)", `caveats.version_words`)."""

    label: _Text
    role: _Text
    kind: Literal["kg_edges"]
    version_manifest_sha256: _Digest = Field(
        ..., description="SHA-256 of the version's manifest (`cohort.json`) the run read: the "
        "record's cohort digest, a key field")
    graph_view: KgEdgesGraphView
    split: KgEdgesSplit
    message_passing: KgEdgesMessagePassing
    pairs: KgEdgesPairs


class MetricDenominator(_Strict):
    count: _Count
    unit: _Text


class LpMetricStatus(_Strict):
    """One metric of a KG link-prediction run: computed or not computable,
    with its unit and denominator. Why one could not be computed stays in the
    ledger; it is never a number here."""

    status: Literal["computed", "not_computable"]
    unit: _Text
    denominator: MetricDenominator


class KgEdgesMeasured(_Strict):
    """What a KG edge set's run measured beside its metrics: its counts, each
    with its unit, and each metric's status (the report's evidence, kept in the
    ledger record as ``measured``)."""

    counts: Dict[str, _Count]
    count_units: Dict[str, _Text]
    metrics: Dict[str, LpMetricStatus]

    @model_validator(mode="after")
    def _each_count_has_its_unit(self) -> "KgEdgesMeasured":
        _units_follow_the_counts(self.counts, self.count_units)
        return self


class ExposureEdges(_Strict):
    """``n`` measured edges, ``in_recorded_graph_use`` of them in the model's
    recorded training or validation graph use (null when that use could not
    be read)."""

    n: _Count
    in_recorded_graph_use: Optional[_Count]

    @model_validator(mode="after")
    def _k_of_n(self) -> "ExposureEdges":
        if self.in_recorded_graph_use is not None and self.in_recorded_graph_use > self.n:
            raise ValueError("more measured edges in the recorded graph use than measured")
        return self


class ExposureMessagePassing(_Strict):
    measured_edges_in_it: _Count
    includes_measured_edges: Annotated[bool, Field(strict=True)]


class ExposureModelSource(_Strict):
    """The model record an exposure was read from, by its checkpoint's digest,
    its learning stage and operation, and its recorded split's digest (null
    where the record names none)."""

    checkpoint_sha256: _Digest
    learning_stage: Optional[_Text]
    operation: Optional[_Text]
    split_digest: Optional[_Digest]


class ExposureVersionSource(_Strict):
    label: _Text
    kind: _Text
    role: _Text
    manifest_sha256: _Digest
    split_digest: _Digest


class ExposureSources(_Strict):
    model_record: ExposureModelSource
    version: ExposureVersionSource


class ExposureView(_Strict):
    """How a KG edge set's measured edges stand to the measured model's
    recorded graph use, as the run wrote it (research comparison F4): its
    definition version, state and reading, the counts, the split relation and
    the sources' identities. Scoped to the recorded evidence and never a gate;
    the run's reason and caveat stay in the ledger."""

    definition_version: _Text
    state: _Text = Field(
        ..., description="As recorded: under kg-lp-exposure/v1, held_out, "
        "in_recorded_graph_use, another_split, after_full_view_exposure or not_recorded")
    reading: _Text = Field(..., description="The run's reading of its state, as recorded")
    split: Optional[_Text] = Field(
        ..., description="Whether the model's recorded split is the version's or another split "
        "of the same view, as recorded; null when the model's record could not be read")
    measured_edges: ExposureEdges
    measurement_message_passing: ExposureMessagePassing
    sources: ExposureSources


class KgEdgesRecordView(_RecordViewFields):
    """One recorded result on a KG edge set, as a person reads it: its
    version's identity and counts with their units, the run's counts and each
    metric's status, and its exposure; no R10 (a KG edge set's relation to the
    model's training is its exposure)."""

    cohort: KgEdgesRecordCohort
    counts: Dict[str, _Count]
    r10: None
    count_units: Dict[str, _Text]
    measured: KgEdgesMeasured
    exposure: ExposureView

    @model_validator(mode="after")
    def _counts_units_and_sources_agree(self) -> "KgEdgesRecordView":
        _units_follow_the_counts(self.counts, self.count_units)
        if self.exposure.sources.version.manifest_sha256 != self.cohort.version_manifest_sha256:
            raise ValueError("its exposure names another version than its cohort digest")
        return self


class GeneratedSplitManifest(_Strict):
    """The workspace's split manifest a generated split was kept from, by
    digest (the version keeps it)."""

    sha256: _Digest


class GeneratedGeneration(_Strict):
    algorithm: _Text
    algorithm_version: Annotated[int, Field(strict=True)]


class GeneratedAllocation(_Strict):
    algorithm: _Text
    algorithm_version: Annotated[int, Field(strict=True)]
    seed: Annotated[int, Field(strict=True)]


class GeneratedFixtureFile(_Strict):
    sha256: _Digest


class GeneratedFixture(_Strict):
    """The functional gene fixture a generated split keeps, by its rule and
    its files' digests."""

    rule: _Text
    record: GeneratedFixtureFile
    samples: GeneratedFixtureFile


class GeneratedRecordCohort(_Strict):
    """The generated split a result was measured on: its label, its role as
    recorded (never shown as test unless it is), its version's manifest
    digest, and its generation identity (its samples, split manifest,
    generation definition and allocation, and the gene fixture it keeps, null
    when none). Never called a cohort where a person reads it ("a generated
    split (val)", `caveats.version_words`)."""

    label: _Text
    role: _Text
    kind: Literal["generated"]
    version_manifest_sha256: _Digest = Field(
        ..., description="SHA-256 of the version's manifest (`cohort.json`) the run read: the "
        "record's cohort digest, a key field")
    samples_sha256: _Digest
    split_manifest: GeneratedSplitManifest
    generation: GeneratedGeneration
    allocation: GeneratedAllocation
    task_fixture: Optional[GeneratedFixture]


class GeneratedRecordView(_RecordViewFields):
    """One recorded result on a generated split kept in the registry, as a
    person reads it: its role and generation identity and its counts with
    their units; no R10 (its relation to training is its allocation's)."""

    cohort: GeneratedRecordCohort
    counts: Dict[str, _Count]
    r10: None
    count_units: Dict[str, _Text]

    @model_validator(mode="after")
    def _each_count_has_its_unit(self) -> "GeneratedRecordView":
        _units_follow_the_counts(self.counts, self.count_units)
        return self


#: Each data kind's record view, by the record's ``cohort.kind``: the one
#: reader's discrimination (research comparison Stage 6).
RECORD_VIEWS: Dict[str, type] = {
    "supplied": RecordView,
    "kg_edges": KgEdgesRecordView,
    "generated": GeneratedRecordView,
}


def _view_kind(value: Any) -> Optional[str]:
    """A record view's data kind, from its ``cohort.kind``, as the API's
    union of views is told apart by."""
    cohort = value.get("cohort") if isinstance(value, dict) else getattr(value, "cohort", None)
    kind = cohort.get("kind") if isinstance(cohort, dict) else getattr(cohort, "kind", None)
    return kind if kind in RECORD_VIEWS else None


#: Any kind's record view, told apart by ``cohort.kind``.
AnyRecordView = Annotated[
    Union[Annotated[RecordView, Tag("supplied")],
          Annotated[KgEdgesRecordView, Tag("kg_edges")],
          Annotated[GeneratedRecordView, Tag("generated")]],
    Discriminator(_view_kind)]


class ContestRun(_Strict):
    """One run carrying a report that differs from another of its condition
    group: its run id (time, checkpoint and a random suffix; never a path) and
    its values of the fields that differ. Byte-identical runs are each named."""

    run: _Text
    values: Dict[str, Optional[_Metric]] = Field(
        ..., description="null where the run's report holds no finite number for the field "
        "(null, NaN, infinity, text, or an integer too large for a float, after a hand edit); "
        "the run's report still differs")


class RecordContest(_Strict):
    """How a record's condition group compares in its cohort version's report
    store, from the one implementation the dashboard shows too
    (`record_states.contest_states`). **A description that decides nothing**
    (policy rev 16, Decision 3): the state names stay as machine values, and
    their meaning changed at K8-6r.2."""

    state: Literal["uncontested", "contested", "cannot_be_checked"] = Field(
        ..., description="Machine values kept for compatibility; since K8-6r.2 they only "
        "describe and decide nothing (no record is hidden, ordered out or refused by them). "
        "uncontested: the reports the store keeps under the record's condition group agree; "
        "contested: runs under these conditions differ, every run kept as recorded; the value "
        "only describes the difference, and only when the group records deterministic "
        "algorithms on (the record's deterministic_algorithms is true) were its runs expected "
        "to agree, so the difference is flagged for investigation; cannot_be_checked: the "
        "comparison is incomplete (no cohort registry configured, a version the registry does "
        "not list, a report of the group that cannot be compared), never to be read as "
        "uncontested")
    differing: List[_Text] = Field(..., description="contested: the fields the reports "
                                   "differ on, when they can be shown; otherwise empty")
    runs: List[ContestRun] = Field(..., description="contested: every run carrying a "
                                   "differing report, byte-identical runs each named, when "
                                   "they can be shown; otherwise empty")

    @model_validator(mode="after")
    def _only_a_contested_key_names_runs(self) -> "RecordContest":
        if self.state != "contested" and (self.differing or self.runs):
            raise ValueError(f"state {self.state!r} with differing fields or runs")
        return self


class LoadedRecordView(RecordView):
    """A record on a supplied cohort version as this route shows it:
    `RecordView` and its group's description."""

    contest: RecordContest


class LoadedKgEdgesRecordView(KgEdgesRecordView):
    """A record on a KG edge set as this route shows it: `KgEdgesRecordView`
    and its group's description."""

    contest: RecordContest


class LoadedGeneratedRecordView(GeneratedRecordView):
    """A record on a generated split as this route shows it:
    `GeneratedRecordView` and its group's description."""

    contest: RecordContest


#: Each kind's view as this route shows it, by the view's class.
_LOADED_VIEWS: Dict[type, type] = {
    RecordView: LoadedRecordView,
    KgEdgesRecordView: LoadedKgEdgesRecordView,
    GeneratedRecordView: LoadedGeneratedRecordView,
}

#: Any kind's record as this route shows it, told apart by ``cohort.kind``.
AnyLoadedRecordView = Annotated[
    Union[Annotated[LoadedRecordView, Tag("supplied")],
          Annotated[LoadedKgEdgesRecordView, Tag("kg_edges")],
          Annotated[LoadedGeneratedRecordView, Tag("generated")]],
    Discriminator(_view_kind)]


class UnreadableRecord(_Strict):
    """A ledger record for these weights that the shared check (`_record_view`)
    cannot read as a result (K8-6e.1, policy rev 15 and rev 16): named by its
    place in the ledger and, where it still carries them readably, its
    registration sequence, report digest and run, with the reason in one
    bounded line that names the refused fields and never their values. It is
    never a result: it carries no metric, count or state. The ledger keeps it
    as written."""

    index: _Count = Field(
        ..., description="Its place in the ledger's records list, counted from 0: the number "
        "the writer's refusals name a record by")
    sequence: Optional[_Sequence] = Field(
        ..., description="Its registration sequence, or null when it carries none that reads "
        "(not an integer of at least 1)")
    source_artifact_digest: Optional[_Digest] = Field(
        ..., description="SHA-256 of the report it names, or null when it carries none that "
        "reads")
    run_id: Optional[_RunId] = Field(
        ..., description="The run it names, or null when it names none (a record written before "
        "each run had its own record) or its run cannot be read")
    reason: Annotated[str, Field(min_length=1, max_length=_REASON_LIMIT)] = Field(
        ..., description="Why it cannot be read as a result: the refused fields (at most three, "
        "then how many more), each by its path in the ledger record, and what is wrong with "
        "each, never their values (a key the record itself carries inside an object, such as a "
        "metric's name, is named as written); one line")


class LoadedModelEvaluations(_Strict):
    """What the ledger beside the loaded checkpoint holds for its weights, each
    record with its run and its group's description (`RecordContest`)."""

    state: Literal[
        "no_model", "no_ledger", "ledger_not_read", "no_records", "records",
    ] = Field(
        ..., description="no_model: no pipeline is served; "
        "no_ledger: no ledger beside the checkpoint; ledger_not_read: a ledger that "
        "cannot be read under this revision, so none of it is shown; no_records: the "
        "ledger holds no result for these weights; records: it holds at least one")
    checkpoint_sha256: Optional[_Digest] = Field(
        ..., description="The digest the ledger was filtered by, from the loaded "
        "model's checkpoint read; null when there is none")
    ledger: Optional[LedgerView]
    records: List[AnyLoadedRecordView] = Field(
        ..., description="Every readable record, sorted by registration sequence, each with "
        "its run (null: not recorded) and its group's description, read by its data kind "
        "(`cohort.kind`: supplied, kg_edges or generated)")
    records_not_shown: _Count = Field(
        ..., description="How many records for these weights cannot be read as a result "
        "(they do not have the shape registration writes, or their run identity cannot be "
        "read): always the length of unreadable_records, which lists each with its reason")
    unreadable_records: List[UnreadableRecord] = Field(
        ..., description="Each record for these weights that cannot be read as a result, in "
        "ledger order, with its index, what identifies it and the reason. Never a result; the "
        "ledger keeps it as written. A record that is not an object, or names no checkpoint "
        "digest, belongs to no model and is not listed here")

    @model_validator(mode="after")
    def _the_state_says_what_the_fields_hold(self) -> "LoadedModelEvaluations":
        looked_up = self.state not in _NOTHING_TO_LOOK_UP
        if looked_up != (self.checkpoint_sha256 is not None):
            raise ValueError(f"state {self.state!r} with checkpoint_sha256 {self.checkpoint_sha256!r}")
        if looked_up != (self.ledger is not None):
            raise ValueError(f"state {self.state!r} with ledger {self.ledger!r}")
        if self.ledger is not None:
            parsed = self.state in ("no_records", "records")
            if parsed != (self.ledger.sha256 is not None):
                raise ValueError(f"state {self.state!r} with ledger sha256 {self.ledger.sha256!r}")
            if (self.state == "ledger_not_read") != (self.ledger.problem is not None):
                raise ValueError(f"state {self.state!r} with ledger problem {self.ledger.problem!r}")
        if self.records_not_shown != len(self.unreadable_records):
            raise ValueError(f"records_not_shown {self.records_not_shown} with "
                             f"{len(self.unreadable_records)} unreadable record(s) listed")
        if (self.state == "records") != bool(self.records or self.unreadable_records):
            raise ValueError(f"state {self.state!r} with {len(self.records)} record(s) shown "
                             f"and {len(self.unreadable_records)} not readable")
        return self


# =============================================================================
# Reading
# =============================================================================
#: A step of a record that is not there: no field accepts it, so validation
#: refuses the record instead of a lookup raising `KeyError` or `TypeError`.
_NOT_THERE = object()


def _at(value: Any, *keys: str) -> Any:
    """``value[k1][k2]...``, or `_NOT_THERE` where a step is not an object or
    lacks its key."""
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return _NOT_THERE
        value = value[key]
    return value


def _pick(value: Any, fields: Iterable[str]) -> Any:
    """The named fields of an object, the rest left behind (an evidence path, the
    cohort's graph and mapping); a value that is not an object is passed on for
    validation to refuse."""
    if not isinstance(value, dict):
        return value
    return {name: value[name] for name in fields if name in value}


def _record_view(record: Dict[str, Any], identity: Optional[RecordIdentity] = None,
                 overlaps: Tuple[int, ...] = ()) -> Any:
    """A ledger record as its data kind's view (`RECORD_VIEWS`: `RecordView`,
    `KgEdgesRecordView` or `GeneratedRecordView`), or a ``ValueError`` saying
    why not. **The one reader by data kind** (research comparison Stage 6):
    the kind is the record's own ``cohort.kind``; a kind this build does not
    read is refused naming that field, never its value, and a record of a
    known kind is read by that kind's view alone.

    The view picks its fields out of the record by name, so what a record carries
    beyond them never reaches the response, and validates every one of them, so a
    record of the wrong shape is refused here and never shown half-read.

    ``identity`` is how its ledger's boundary reads it (`record_identities`):
    a record whose identity cannot be read is refused with that reason, a
    historical one has ``run_id`` null. ``overlaps`` are its possible overlaps
    (`possible_overlaps`). Every reader of a ledger passes both; without
    ``identity`` (a record read alone, outside its ledger) its own ``run_id``
    is shown as written.

    A field taken from another place in the record is listed in
    `_RECORD_PATHS`, so a refusal names the record's own path.
    """
    if identity is not None and identity.problem is not None:
        raise ValueError(identity.problem)
    run_id = (identity.run_id if identity is not None
              else record.get("run_id") if "run_id" in record else None)
    corroborating = record.get("corroborating_reports")
    if not isinstance(corroborating, list):
        raise ValueError("its corroborating_reports is not a list")
    if record.get("cohort_role") != _at(record, "cohort", "role"):
        raise ValueError("its cohort_role is not its cohort's role")
    kind = _at(record, "cohort", "kind")
    view = RECORD_VIEWS.get(kind) if isinstance(kind, str) else None
    if view is None:
        # Named by its field, never its value (K8-6e.1): a hand edit can put
        # a server path there.
        raise ValueError(
            "cohort.kind: missing" if kind is _NOT_THERE else
            f"cohort.kind: not a data kind this build reads ({', '.join(RECORD_VIEWS)})")
    # The version's manifest digest is the record's key field, not a field of its
    # cohort block; a record lacking it is refused like any other missing field.
    cohort = _pick(_at(record, "cohort"), view.model_fields["cohort"].annotation.model_fields)
    if isinstance(cohort, dict):
        cohort = {**cohort, "version_manifest_sha256": _at(record, "cohort_digest")}
    # The run environment's one projection; a record written before its
    # environment named components has none.
    components = _at(record, "environment", "facts", "components")
    common = {
        "mode": _at(record, "scoring", "mode"),
        "score_semantics": _at(record, "scoring", "score_semantics"),
        "metrics": _at(record, "metrics"),
        "n_ranked": _at(record, "n_ranked"),
        "relations": _at(record, "relations"),
        "environment_class": environment_class(record),
        "deterministic_algorithms": _recorded_determinism(record),
        "gpu": _pick(_at(record, "environment", "facts", "gpu"), RecordGpu.model_fields),
        "native_identity": None if components is _NOT_THERE else native_identity(components),
        "dataset_origin": _at(record, "dataset_record", "origin"),
        "scoring_semantics_digest": _at(record, "scoring_semantics_digest"),
        "measurement_semantics_digest": _at(record, "measurement_semantics_digest"),
        "registration": _pick(_at(record, "registration"), RecordRegistration.model_fields),
        "run_id": run_id,
        "historical_overlap": list(overlaps),
        "n_corroborating": len(corroborating),
        "source_artifact_digest": _at(record, "source_artifact_digest"),
    }
    return view(cohort=cohort, **_KIND_PARTS[kind](record, cohort), **common)


def _supplied_parts(record: Dict[str, Any], cohort: Any) -> Dict[str, Any]:
    """A supplied cohort version's own parts of its view: its counts and its
    R10 labels (each scope's evidence state, never its path)."""
    r10 = record.get("r10")
    return {
        "counts": _pick(_at(record, "cohort", "counts"), RecordCounts.model_fields),
        "r10": {
            scope: {
                "evidence": _pick(_at(r10, scope, "evidence"), R10Evidence.model_fields),
                "disease_overlap": _pick(_at(r10, scope, "disease_overlap"),
                                         R10DiseaseOverlap.model_fields),
                "phenotype_set_overlap": _pick(_at(r10, scope, "phenotype_set_overlap"),
                                               R10PhenotypeSetOverlap.model_fields),
            }
            for scope in R10View.model_fields
        },
    }


def _picked(value: Any, model: type, nested: Dict[str, type]) -> Any:
    """`value`'s fields of `model`, each field named in `nested` picked by its
    own model in turn: what a view shows of a block, the rest left in the
    ledger."""
    picked = _pick(value, model.model_fields)
    if isinstance(picked, dict):
        for name, inner in nested.items():
            if name in picked:
                picked[name] = _pick(picked[name], inner.model_fields)
    return picked


def _versioned_parts(record: Dict[str, Any], cohort: Any) -> Dict[str, Any]:
    """A KG edge set's or a generated split's parts every such view holds: its
    counts with their units, and no R10 (null)."""
    return {"counts": _at(record, "cohort", "counts"),
            "count_units": _at(record, "cohort", "count_units"),
            "r10": record.get("r10", _NOT_THERE)}


def _kg_edges_parts(record: Dict[str, Any], cohort: Any) -> Dict[str, Any]:
    """A KG edge set's own parts of its view: its view, split, message-passing
    graph and pairs (by digest and rule), its counts with their units, the
    run's counts and each metric's status, and its exposure, whose model
    record must be this record's checkpoint."""
    if isinstance(cohort, dict):
        for name, model in (("graph_view", KgEdgesGraphView), ("split", KgEdgesSplit),
                            ("message_passing", KgEdgesMessagePassing),
                            ("pairs", KgEdgesPairs)):
            if name in cohort:
                cohort[name] = _pick(cohort[name], model.model_fields)
    exposure = _picked(_at(record, "exposure"), ExposureView, {
        "measured_edges": ExposureEdges,
        "measurement_message_passing": ExposureMessagePassing,
        "sources": ExposureSources})
    sources = exposure.get("sources") if isinstance(exposure, dict) else None
    if isinstance(sources, dict):
        sources = {name: _pick(sources[name], model.model_fields)
                   for name, model in (("model_record", ExposureModelSource),
                                       ("version", ExposureVersionSource)) if name in sources}
        exposure["sources"] = sources
        if _at(sources, "model_record", "checkpoint_sha256") != record.get("checkpoint_digest"):
            raise ValueError("exposure.sources.model_record.checkpoint_sha256: not the record's "
                             "checkpoint_digest")
    measured = _picked(_at(record, "measured"), KgEdgesMeasured, {})
    if isinstance(measured, dict) and isinstance(measured.get("metrics"), dict):
        measured["metrics"] = {
            name: _picked(status, LpMetricStatus, {"denominator": MetricDenominator})
            for name, status in measured["metrics"].items()}
    return {**_versioned_parts(record, cohort), "measured": measured, "exposure": exposure}


def _generated_parts(record: Dict[str, Any], cohort: Any) -> Dict[str, Any]:
    """A generated split's own parts of its view: its generation identity (the
    split manifest by digest, the generation definition and allocation, the
    gene fixture it keeps) and its counts with their units."""
    if isinstance(cohort, dict):
        for name, model in (("split_manifest", GeneratedSplitManifest),
                            ("generation", GeneratedGeneration),
                            ("allocation", GeneratedAllocation)):
            if name in cohort:
                cohort[name] = _pick(cohort[name], model.model_fields)
        if isinstance(cohort.get("task_fixture"), dict):
            cohort["task_fixture"] = _picked(cohort["task_fixture"], GeneratedFixture, {
                "record": GeneratedFixtureFile, "samples": GeneratedFixtureFile})
    return _versioned_parts(record, cohort)


#: Each kind's own parts of its view, by the record's ``cohort.kind``.
_KIND_PARTS: Dict[str, Callable[[Dict[str, Any], Any], Dict[str, Any]]] = {
    "supplied": _supplied_parts,
    "kg_edges": _kg_edges_parts,
    "generated": _generated_parts,
}


def _recorded_determinism(record: Dict[str, Any]) -> Optional[bool]:
    """The record's own setting for deterministic algorithms (its regime), or
    None when it records none: read from the ledger, so it opens no checkpoint
    and a read-only request sees what an operator sees (M5-3)."""
    regime = record.get("regime")
    flag = regime.get("deterministic_algorithms") if isinstance(regime, dict) else None
    return flag if type(flag) is bool else None


def _ledger_view(sha256: Optional[str] = None, problem: Optional[str] = None) -> LedgerView:
    return LedgerView(file=LEDGER_FILENAME, where=LEDGER_WHERE, sha256=sha256, problem=problem)


def _ledger_problem(exc: Exception, ledger_path: Path) -> str:
    """Why the ledger was not read, with its directory left out: the response
    names no path on the server, and the file is the one beside the checkpoint.

    **An `OSError` is described by its type and its error text only, never
    `str(exc)`,** which quotes the filename with ``repr``: a directory name with
    a backslash, a newline or a byte that is not UTF-8 is escaped there, the
    quoted text no longer equals the path, and replacing the path would leave
    the server's directory in the answer. `read_ledger`'s own refusals name the
    path as written, so replacing it is enough for them.
    """
    if isinstance(exc, OSError):
        if exc.strerror:
            reason = f": {exc.strerror}"
        elif exc.errno is not None:
            reason = f": errno {exc.errno}"
        else:
            reason = ""
        return f"{LEDGER_FILENAME} could not be read ({type(exc).__name__}{reason})"
    return str(exc).replace(str(ledger_path), LEDGER_FILENAME)


def _recorded_results(read: "ReadIdentity") -> LoadedModelEvaluations:
    """What the ledger beside one checkpoint read holds for its digest.

    **The directory was fixed when the checkpoint was read:** the pipeline reads
    the checkpoint at its resolved path (`_load_model_from_checkpoint`), so
    ``read.path`` is the file whose bytes were loaded, and its parent is the
    directory they were in. It is not resolved again here: a link retargeted
    since, without a reload, would move the lookup away from the served bytes'
    ledger while the digest stayed theirs.
    """
    digest = read.sha256
    ledger_path = Path(read.path).parent / LEDGER_FILENAME
    try:
        ledger, ledger_sha = read_ledger(ledger_path)
    except (OSError, ValueError) as exc:
        return LoadedModelEvaluations(
            state="ledger_not_read", checkpoint_sha256=digest,
            ledger=_ledger_view(problem=_ledger_problem(exc, ledger_path)),
            records=[], records_not_shown=0, unreadable_records=[],
        )
    if ledger_sha is None:
        return LoadedModelEvaluations(
            state="no_ledger", checkpoint_sha256=digest, ledger=_ledger_view(),
            records=[], records_not_shown=0, unreadable_records=[],
        )

    viewed, unreadable = read_records(ledger, digest)
    for item in unreadable:
        logger.warning("A ledger record for checkpoint %s... is not shown: the ledger's record "
                       "%d: %s", digest[:12], item.index, item.reason)
    contest = _contest_states([record for record, _ in viewed])
    shown = [_LOADED_VIEWS[type(view)](**dict(view), contest=contest(record))
             for record, view in viewed]
    shown.sort(key=lambda view: view.registration.sequence)
    return LoadedModelEvaluations(
        state="records" if shown or unreadable else "no_records",
        checkpoint_sha256=digest, ledger=_ledger_view(sha256=ledger_sha),
        records=shown, records_not_shown=len(unreadable), unreadable_records=unreadable,
    )


def unreadable_reason(exc: ValueError) -> str:
    """Why the shared check refused a record, as one bounded line that names
    the refused fields and never their values (K8-6e.1).

    For pydantic's refusal, at most three of its errors, each "<field path>:
    missing" (a field the record lacks, `_NOT_THERE`) or "<field path>: <what
    is wrong>" (pydantic's words, its "Value error, " prefix left out), then
    how many more; the input it was given is never written (pydantic's own
    text carries it, with a documentation link, over several lines). **The
    field path is the ledger record's** (`_record_path`), not the API view's
    name for it: the key a hand edit of the ledger corrects. A key the record
    itself carries inside an object (a metric's name, an exclusion reason) is
    a step of that path and is named as written, as a readable record's
    metric names are shown. For any other `ValueError`, its own words:
    `_record_view`'s and the shared checks' refusals name a field, never its
    value. Either way through `problem_line`: one line, each character that
    is not printable escaped (a field name is a key a hand edit may have
    written), at most 300 characters."""
    from pydantic import ValidationError

    from src.utils.checkpoint_paths import problem_line

    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False)
        parts = []
        for error in errors[:3]:
            where = ".".join(_printable(step) for step in _record_path(error.get("loc") or ())
                             ) or "the record"
            if error.get("type") == "missing" or error.get("input") is _NOT_THERE:
                parts.append(f"{where}: missing")
                continue
            words = str(error.get("msg") or "is not valid")
            if words.startswith("Value error, "):
                words = words[len("Value error, "):]
            parts.append(f"{where}: {words}")
        if len(errors) > 3:
            parts.append(f"{len(errors) - 3} more")
        text = "; ".join(parts)
    else:
        text = str(exc)
    return problem_line(text) or "it does not have the shape registration writes"


#: Where each field `_record_view` takes from another place sits in the ledger
#: record, by the view's path: a reason names the record's own path, the key a
#: hand edit of the ledger corrects. Every other field has one path in both.
_RECORD_PATHS: Dict[Tuple[str, ...], Tuple[str, ...]] = {
    ("cohort", "version_manifest_sha256"): ("cohort_digest",),
    ("counts",): ("cohort", "counts"),
    ("count_units",): ("cohort", "count_units"),
    ("mode",): ("scoring", "mode"),
    ("score_semantics",): ("scoring", "score_semantics"),
    ("gpu",): ("environment", "facts", "gpu"),
    ("native_identity",): ("environment", "facts", "components"),
    ("dataset_origin",): ("dataset_record", "origin"),
}


def _record_path(loc: Sequence[Any]) -> Tuple[Any, ...]:
    """Pydantic's location in `RecordView` as the path in the ledger record
    (`_RECORD_PATHS`): the longest view path it starts with is replaced."""
    loc = tuple(loc)
    for size in (2, 1):
        moved = _RECORD_PATHS.get(loc[:size]) if len(loc) >= size else None
        if moved is not None:
            return moved + loc[size:]
    return loc


def _printable(step: Any) -> str:
    """One step of a field path as text, each character that is not
    printable escaped, so a newline in a hand-written key cannot end the line."""
    text = str(step)
    return text if text.isprintable() else "".join(
        c if c.isprintable() else repr(c)[1:-1] for c in text)


def read_records(ledger: Dict[str, Any], checkpoint_digest: str
                 ) -> Tuple[List[Tuple[Dict[str, Any], Any]], List[UnreadableRecord]]:
    """**The one listing of a ledger's records for one set of weights**
    (K8-6e.1), for every reader (this route, Models' operator view through
    the shared model list, `record_evaluation.py --show`): the records the
    shared check (`_record_view`) reads as results, each with its kind's view
    (`RECORD_VIEWS`), and every other record as an `UnreadableRecord` with its
    index and reason, in ledger order: a record of a kind this build does not
    read among them, record by record, never hiding another. Each record's run
    is read by the file's boundary, and its possible overlaps, from the whole
    ledger once (`record_reading`).

    Reading one record never waits on or hides another; nothing is repaired,
    edited or dropped (a record that is not an object, or names no checkpoint
    digest, belongs to no model: `indexed_records_for` does not give it)."""
    from src.evaluation.sidecar import _is_digest, _readable_sequence

    read_as = record_reading(ledger)
    viewed: List[Tuple[Dict[str, Any], RecordView]] = []
    unreadable: List[UnreadableRecord] = []
    for index, record in indexed_records_for(ledger, checkpoint_digest):
        identity, overlaps = read_as[id(record)]
        try:
            viewed.append((record, _record_view(record, identity, overlaps)))
        except ValueError as exc:
            source = record.get("source_artifact_digest")
            unreadable.append(UnreadableRecord(
                index=index, sequence=_readable_sequence(record),
                source_artifact_digest=source if _is_digest(source) else None,
                run_id=identity.run_id, reason=unreadable_reason(exc)))
    return viewed, unreadable


def record_reading(ledger: Dict[str, Any]) -> Dict[int, Tuple[RecordIdentity, Tuple[int, ...]]]:
    """Each record of `ledger`, by ``id()``, with how the file's boundary reads
    its run (`record_identities`) and its possible overlaps
    (`possible_overlaps`): the one reading the route and the Models page give
    `_record_view`. Reads nothing more than the ledger already parsed."""
    records = ledger.get("records") if isinstance(ledger.get("records"), list) else []
    return {id(record): (identity, overlaps)
            for record, identity, overlaps in zip(records, record_identities(ledger),
                                                  possible_overlaps(ledger))}


def _contest_states(records: List[Dict[str, Any]]) -> Callable[[Dict[str, Any]], RecordContest]:
    """Each shown record's group description, by the one implementation the
    dashboard uses (`record_states.contest_states`, with the versions the
    server's registry lists), without the problems' words, which may name a
    server path. It decides nothing.

    A record with no description read is ``cannot_be_checked``, never
    uncontested and never a 500 that would take the model status with it. **A
    contested description is never downgraded** by how its details can be
    shown: a run's value that is not a finite number is null
    (`_finite_or_null`), and a contested description whose fields or runs still
    cannot be carried is shown contested without them."""
    from src.api.services.record_states import CohortVersions, contest_states, key_text
    from src.evaluation.report_store import CONTEST_STATES
    from src.evaluation.sidecar import record_key

    found = contest_states(records, CohortVersions.listed(), operator=False) if records else {}

    def state(record: Dict[str, Any]) -> RecordContest:
        try:
            got = found.get(key_text(record_key(record)))
        except ValueError:  # a key this record does not have: nothing was read for it
            got = None
        if not isinstance(got, dict) or got.get("state") not in CONTEST_STATES:
            return RecordContest(state="cannot_be_checked", differing=[], runs=[])
        try:
            runs = [{"run": run.get("run"),
                     "values": {name: _finite_or_null(value)
                                for name, value in (run.get("values") or {}).items()}}
                    for run in got.get("runs") or [] if isinstance(run, dict)]
            return RecordContest(state=got["state"], differing=got.get("differing") or [],
                                 runs=runs)
        except Exception as exc:  # noqa: BLE001 -- pydantic's ValidationError included
            logger.warning("A record's group details are not shown: %s: %s",
                           type(exc).__name__, exc)
            if got["state"] == "contested":
                return RecordContest(state="contested", differing=[], runs=[])
            return RecordContest(state="cannot_be_checked", differing=[], runs=[])

    return state


def _finite_or_null(value: Any) -> Optional[float]:
    """A differing run's value as the view carries it: a finite number as a
    float; null for anything else a hand-edited report may hold there (null,
    NaN, infinity, text, an object, a boolean, or an integer too large for a
    float, which `math.isfinite` cannot take)."""
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _nothing_to_look_up(state: str) -> LoadedModelEvaluations:
    return LoadedModelEvaluations(
        state=state, checkpoint_sha256=None, ledger=None, records=[], records_not_shown=0,
        unreadable_records=[],
    )


@router.get("/evaluations/loaded-model", response_model=LoadedModelEvaluations)
async def get_loaded_model_evaluations() -> LoadedModelEvaluations:
    """What the ledger beside the loaded checkpoint holds for the weights the
    loaded model was built from, found by their digest. Read-only; takes no
    parameters."""
    from src.api.main import app_state

    # **Application state is read once, and everything below comes from that one
    # object**, so a reload published meanwhile cannot mix two models here,
    # whichever thread this runs on (see the module docstring).
    pipeline = app_state.pipeline
    if pipeline is None:
        return _nothing_to_look_up("no_model")
    read = pipeline.served_checkpoint_read()
    # The identity is taken; what follows reads files only (the ledger and the
    # report store of each shown record's version), so it runs off the loop.
    return await run_in_threadpool(_recorded_results, read)


__all__ = [
    "AnyLoadedRecordView",
    "AnyRecordView",
    "ContestRun",
    "ExposureView",
    "GeneratedRecordView",
    "KgEdgesRecordView",
    "LEDGER_WHERE",
    "LedgerView",
    "LoadedGeneratedRecordView",
    "LoadedKgEdgesRecordView",
    "LoadedModelEvaluations",
    "LoadedRecordView",
    "RECORD_VIEWS",
    "RecordContest",
    "RecordView",
    "UnreadableRecord",
    "get_loaded_model_evaluations",
    "read_records",
    "router",
    "unreadable_reason",
]
