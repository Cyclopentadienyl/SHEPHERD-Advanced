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

**Each record is read by the one reader, by its data kind**
(`src.api.services.record_views`, moved there from this module in research
comparison Stage 8): a supplied cohort version's as `RecordView`, a KG edge
set's as `KgEdgesRecordView` (its exposure read apart from the result), a
generated split's as `GeneratedRecordView`, each with this route's group
description (`LoadedRecordView` and the rest, told apart by ``cohort.kind``).
A record whose kind this build does not read is listed in
``unreadable_records``, naming the field and never its value, and hides no
other record.

**Aggregates only** (D7). A record is shown as its view holds it (counts,
metrics, relation states, R10 states and counts, environment class, GPU and
native build identity, every recorded fact of each named component
(``components``, each text through the one D7 redaction, which replaces an
absolute path that starts its own word: `run_environment.recorded_components`;
a component's reader records no path, `ComponentReader`), dataset origin,
registration, its run and digests) and
its group description (field names, run ids and values; never a problem's
words, which may name a server path). No path (a record's R10 evidence paths
among them), no case id, no rank; a ledger problem names the file, never its
directory. A KG edge set's exposure is shown by its definition version, state,
reading, counts and sources' identities; its reason and its caveat stay in the
ledger (the surfaces say the caveat in this build's words,
`caveats.EXPOSURE_CAVEAT`).

**Expected failures are states; nothing else is swallowed.** A ledger that cannot
be read as a whole is ``ledger_not_read`` with its reason, and none of its
records are shown; a schema-4 ledger whose run-identity boundary cannot be read
is still read, its records naming a run shown and the others listed. A record
for these weights that does not have the shape registration writes, after a
hand edit (the sanctioned correction path), is not raised (a 500 here would
take the model status with it) and never shown as a result: **it is listed in
``unreadable_records``** (K8-6e.1; `record_views.read_records`, the one listing
every reader uses) with its index, what identifies it and the reason in one
bounded line naming fields, never values. ``records_not_shown`` keeps its
meaning and equals the list's length. The server log says the same line with
the index. Any other exception is a defect and reaches the client as one.

No write, no `find_checkpoint`, no `torch.load`.

Module: src/api/routes/evaluations.py
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, Dict, List, Literal, Optional, Union

from fastapi import APIRouter
from pydantic import Discriminator, Field, Tag, model_validator
from starlette.concurrency import run_in_threadpool

from src.api.services.record_views import (
    GeneratedRecordView,
    KgEdgesRecordView,
    RecordView,
    UnreadableRecord,
    _Count,
    _Digest,
    _Metric,
    _Text,
    _view_kind,
    ledger_problem,
    read_records,
)
from src.api.services.response_models import StrictModel as _Strict
from src.evaluation.sidecar import LEDGER_FILENAME, read_ledger

if TYPE_CHECKING:
    from src.utils.fingerprint import ReadIdentity

logger = logging.getLogger(__name__)

router = APIRouter()

#: Where the ledger is, said without its path: this view names no path on the
#: server, and the checkpoint the status describes fixes the directory.
LEDGER_WHERE = "beside the loaded checkpoint"

#: States with no checkpoint digest, and so no ledger to look in.
_NOTHING_TO_LOOK_UP = ("no_model",)


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
def _ledger_view(sha256: Optional[str] = None, problem: Optional[str] = None) -> LedgerView:
    return LedgerView(file=LEDGER_FILENAME, where=LEDGER_WHERE, sha256=sha256, problem=problem)


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
            ledger=_ledger_view(problem=ledger_problem(exc, ledger_path)),
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
    "ContestRun",
    "LEDGER_WHERE",
    "LedgerView",
    "LoadedGeneratedRecordView",
    "LoadedKgEdgesRecordView",
    "LoadedModelEvaluations",
    "LoadedRecordView",
    "RecordContest",
    "get_loaded_model_evaluations",
    "router",
]
