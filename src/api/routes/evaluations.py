"""
Recorded results for the loaded model: a read-only view of the ledger.
=======================================================================
`PLAN_TEST_RESULTS.md` D8 and §6 phase 1, step 4: the Diagnosis model status
shows what the ledger beside the loaded checkpoint holds for those weights.

Endpoint:
    GET /evaluations/loaded-model — the loaded model's recorded results

**Its own router, outside `pipeline.router`.** SPEC_4's C3 class covers the
reload, the UI configuration and checkpoint inspection; a deployment applying
B2 by unmounting or protecting that router as a whole keeps this view, which
reads the ledger and nothing else (D8).

**Found by the served read, filtered by its digest.** Application state is read
once, and both the ledger's directory and the digest the ledger is filtered by
come from that pipeline's own checkpoint read
(`DiagnosisPipeline.served_checkpoint_read`). The handler is ``async def`` with
no ``await``, so it runs on the event-loop thread, where `publish_pipeline` runs
too: a reload cannot land between reading the pipeline and reading its identity.
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

**Aggregates only** (D7). A record is shown as its counts, metrics, relation
states, R10 states and counts, environment class and GPU, dataset origin,
registration and digests. No path (a record's R10 evidence paths among them),
no case id, no rank; a ledger problem names the file, never its directory.

**Expected failures are states; nothing else is swallowed.** A ledger that cannot
be read is ``ledger_not_read`` with its reason, and none of its records are
shown. A record for these weights that does not have the shape registration
writes, after a hand edit (the sanctioned correction path), is counted in
``records_not_shown`` rather than raised: a 500 here would take the model status
with it. Any other exception is a defect and reaches the client as one.

No write, no `find_checkpoint`, no `torch.load`.

Module: src/api/routes/evaluations.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Dict, Iterable, List, Literal, Optional

from fastapi import APIRouter
from pydantic import Field, field_validator, model_validator

from src.api.routes.diagnose import _SHA256, _Strict
from src.evaluation.sidecar import (
    LEDGER_FILENAME,
    REGISTERED_RELATIONS,
    environment_class,
    read_ledger,
    records_for,
)

if TYPE_CHECKING:
    from src.utils.fingerprint import ReadIdentity

logger = logging.getLogger(__name__)

router = APIRouter()

#: Where the ledger is, said without its path: this view names no path on the
#: server, and the checkpoint the status describes fixes the directory.
LEDGER_WHERE = "beside the loaded checkpoint"

#: States with no checkpoint digest, and so no ledger to look in.
_NOTHING_TO_LOOK_UP = ("no_model", "not_identified")
#: R10 states that carry a count (spec §3.2); every other state carries none.
_COUNTED_R10_STATES = ("overlap", "none")

_Count = Annotated[int, Field(strict=True, ge=0)]
_Metric = Annotated[float, Field(strict=True, allow_inf_nan=False)]
_Digest = Annotated[str, Field(pattern=_SHA256)]
_Text = Annotated[str, Field(min_length=1)]
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
    any read error stay in the ledger."""

    state: Literal["found", "not_found", "unreadable", "not_recorded"]


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


class RecordRegistration(_Strict):
    sequence: Annotated[int, Field(strict=True, ge=1)]
    registered_at: _Text


class RecordView(_Strict):
    """One recorded result, as a person reads it: aggregates and digests only."""

    cohort: RecordCohort
    counts: RecordCounts
    metrics: Dict[str, _Metric] = Field(
        ..., description="Mode C metrics over n_ranked cases; `untruncated_mrr` is not "
        "the trainer's top-20 `val_mrr`")
    n_ranked: _Count
    relations: Dict[str, Literal["verified", "not_applicable"]]
    r10: R10View
    environment_class: str
    deterministic_algorithms: Optional[bool] = Field(
        ..., description="whether deterministic algorithms were on in the measuring run, as the "
        "record's regime says (a CPU run's environment class does not carry it); null when the "
        "record does not say")
    gpu: Optional[RecordGpu] = Field(..., description="null for a run off CUDA")
    dataset_origin: Literal["files", "synthetic"]
    scoring_semantics_digest: _Digest
    measurement_semantics_digest: _Digest
    registration: RecordRegistration
    n_corroborating: _Count
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


class LoadedModelEvaluations(_Strict):
    """What the ledger beside the loaded checkpoint holds for its weights."""

    state: Literal[
        "no_model", "not_identified", "no_ledger", "ledger_not_read", "no_records",
        "records",
    ] = Field(
        ..., description="no_model: no pipeline, or one serving no model; "
        "not_identified: a model supplied in memory, no checkpoint read to look up; "
        "no_ledger: no ledger beside the checkpoint; ledger_not_read: a ledger that "
        "cannot be read under this revision, so none of it is shown; no_records: the "
        "ledger holds no result for these weights; records: it holds at least one")
    checkpoint_sha256: Optional[_Digest] = Field(
        ..., description="The digest the ledger was filtered by, from the loaded "
        "model's checkpoint read; null when there is none")
    ledger: Optional[LedgerView]
    records: List[RecordView] = Field(..., description="Sorted by registration sequence")
    records_not_shown: _Count = Field(
        ..., description="Records for these weights that do not have the shape "
        "registration writes")

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
        if (self.state == "records") != bool(self.records or self.records_not_shown):
            raise ValueError(f"state {self.state!r} with {len(self.records)} record(s) shown "
                             f"and {self.records_not_shown} not shown")
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


def _record_view(record: Dict[str, Any]) -> RecordView:
    """A ledger record as `RecordView`, or a ``ValueError`` saying why not.

    The view picks its fields out of the record by name, so what a record carries
    beyond them never reaches the response, and validates every one of them, so a
    record of the wrong shape is refused here and never shown half-read.
    """
    corroborating = record.get("corroborating_reports")
    if not isinstance(corroborating, list):
        raise ValueError("its corroborating_reports is not a list")
    if record.get("cohort_role") != _at(record, "cohort", "role"):
        raise ValueError("its cohort_role is not its cohort's role")
    r10 = record.get("r10")
    # The version's manifest digest is the record's key field, not a field of its
    # cohort block; a record lacking it is refused like any other missing field.
    cohort = _pick(_at(record, "cohort"), RecordCohort.model_fields)
    if isinstance(cohort, dict):
        cohort = {**cohort, "version_manifest_sha256": _at(record, "cohort_digest")}
    return RecordView(
        cohort=cohort,
        counts=_pick(_at(record, "cohort", "counts"), RecordCounts.model_fields),
        metrics=_at(record, "metrics"),
        n_ranked=_at(record, "n_ranked"),
        relations=_at(record, "relations"),
        r10={
            scope: {
                "evidence": _pick(_at(r10, scope, "evidence"), R10Evidence.model_fields),
                "disease_overlap": _pick(_at(r10, scope, "disease_overlap"),
                                         R10DiseaseOverlap.model_fields),
                "phenotype_set_overlap": _pick(_at(r10, scope, "phenotype_set_overlap"),
                                               R10PhenotypeSetOverlap.model_fields),
            }
            for scope in R10View.model_fields
        },
        environment_class=environment_class(record),
        deterministic_algorithms=_recorded_determinism(record),
        gpu=_pick(_at(record, "environment", "facts", "gpu"), RecordGpu.model_fields),
        dataset_origin=_at(record, "dataset_record", "origin"),
        scoring_semantics_digest=_at(record, "scoring_semantics_digest"),
        measurement_semantics_digest=_at(record, "measurement_semantics_digest"),
        registration=_pick(_at(record, "registration"), RecordRegistration.model_fields),
        n_corroborating=len(corroborating),
        source_artifact_digest=_at(record, "source_artifact_digest"),
    )


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
            records=[], records_not_shown=0,
        )
    if ledger_sha is None:
        return LoadedModelEvaluations(
            state="no_ledger", checkpoint_sha256=digest, ledger=_ledger_view(),
            records=[], records_not_shown=0,
        )

    shown: List[RecordView] = []
    not_shown = 0
    for record in records_for(ledger, digest):
        try:
            shown.append(_record_view(record))
        except ValueError as exc:
            not_shown += 1
            logger.warning(
                "A ledger record for checkpoint %s... is not shown: %s", digest[:12], exc
            )
    shown.sort(key=lambda view: view.registration.sequence)
    return LoadedModelEvaluations(
        state="records" if shown or not_shown else "no_records",
        checkpoint_sha256=digest, ledger=_ledger_view(sha256=ledger_sha),
        records=shown, records_not_shown=not_shown,
    )


def _nothing_to_look_up(state: str) -> LoadedModelEvaluations:
    return LoadedModelEvaluations(
        state=state, checkpoint_sha256=None, ledger=None, records=[], records_not_shown=0,
    )


@router.get("/evaluations/loaded-model", response_model=LoadedModelEvaluations)
async def get_loaded_model_evaluations() -> LoadedModelEvaluations:
    """What the ledger beside the loaded checkpoint holds for the weights the
    loaded model was built from, found by their digest. Read-only; takes no
    parameters."""
    from src.api.main import app_state

    # **No `await` in this handler, and none may be added before the identity
    # is taken.** The pipeline is read from application state once, on the
    # event-loop thread where a reload publishes, and everything below comes
    # from that one object.
    pipeline = app_state.pipeline
    if pipeline is None:
        return _nothing_to_look_up("no_model")
    read = pipeline.served_checkpoint_read()
    if read is None:
        absent = pipeline.served_identity().checkpoint.state == "absent"
        return _nothing_to_look_up("no_model" if absent else "not_identified")
    return _recorded_results(read)


__all__ = [
    "LEDGER_WHERE",
    "LedgerView",
    "LoadedModelEvaluations",
    "RecordView",
    "get_loaded_model_evaluations",
    "router",
]
