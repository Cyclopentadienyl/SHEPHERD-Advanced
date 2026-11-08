"""
The loaded model's recorded results, read-only (M4; PLAN_TEST_RESULTS D8).
===========================================================================
`GET /api/v1/evaluations/loaded-model` reads the ledger beside the checkpoint the
loaded model was built from and shows that ledger's records for the model's
digest. What is pinned here, on the real application and real served pipelines
(the session's trained disease child, copied into each test's directory), over
ledgers written by the real registration (`register_report`):

- every state: no pipeline, no ledger, a ledger that is not read (schema 2,
  damaged, unopenable, with no server directory in its problem whatever the
  directory's name), no records for these weights, records;
- a record that does not have the shape registration writes is listed in
  ``unreadable_records`` with its reason, never shown as a result, never a 500,
  and never hides a good one;
- records are found by digest: a model replaced by other bytes never shows its
  predecessor's records, and the same bytes moved with their ledger find theirs;
- the ledger is the one beside the served read, not beside a path in application
  state, its directory settled when the checkpoint was read (a link retargeted
  since moves nothing), and a reload landing mid-request cannot mix two models;
- two imports of one label are shown as two versions, by their manifest digests;
- aggregates only: no path, no case id, no rank in any answer, and no query
  parameter selects anything;
- the route opens the ledger once, each version manifest of the registry once
  (to find a record's version by its cohort digest; the version found is that
  read) and, for each record's D5 state, each report its version's store keeps
  once (BACKLOG 36); it opens no other file, lists no other version's store,
  writes nothing, and is mounted outside the C3 `pipeline` router; its one
  ``await`` comes after the identity is taken;
- `DiagnosisPipeline.served_checkpoint_read` returns the read the served
  identity names the checkpoint by; a served pipeline always has one (K7: no
  model is served without a checkpoint read), so the "not identified" state is
  gone.

On the dashboard, Research › Models' read-only view calls this route's handler
in process (`models_page._route_records`, pinned in `test_webui_models_page.py`);
Diagnosis shows the recorded tests as a state only (since M5-2).

Module: tests/unit/test_loaded_model_evaluations_api.py
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import shutil
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from src.evaluation.registration import MeasuredRun, register_report  # noqa: E402
from src.utils.fingerprint import ReadIdentity  # noqa: E402
from src.evaluation.sidecar import REGISTERED_RELATIONS  # noqa: E402
from tests.fixtures.opens import count_opens  # noqa: E402
from tests.fixtures.report_store import IMPORTED_AT, store_report  # noqa: E402
from tests.unit.test_evaluation_sidecar import (  # noqa: E402
    cohort_version,
    r10_block,
    registered_report,
)
from tests.unit.test_served_identity import (  # noqa: E402, F401 (fixtures found by name)
    _client,
    _full_pipeline,
    _with_val_pooled_mrr,
    app,
    served_child,
)

ROUTE = "/api/v1/evaluations/loaded-model"
WHEN = "2026-10-16T12:00:00Z"
OTHER_CHECKPOINT = "c" * 64
RESPONSE_KEYS = {"state", "checkpoint_sha256", "ledger", "records", "records_not_shown",
                 "unreadable_records"}
RECORD_VIEW_KEYS = {
    "cohort", "counts", "mode", "score_semantics", "metrics", "n_ranked", "relations", "r10",
    "environment_class",
    "deterministic_algorithms", "gpu", "dataset_origin", "scoring_semantics_digest", "measurement_semantics_digest",
    "registration", "run_id", "historical_overlap", "n_corroborating", "source_artifact_digest",
    "contest",
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report_for(checkpoint_sha, **manifest_overrides):
    """A registered cohort version's report (spec §4.1) of these weights, built
    by `MeasurementManifest` through the sidecar tests' helper."""
    return registered_report(
        artifact_digests={"checkpoint": checkpoint_sha, "samples": "5" * 64,
                          "node_features": "2" * 64, "edge_indices": "3" * 64,
                          "num_nodes": "4" * 64, "kg": "1" * 64},
        **manifest_overrides,
    )


def _r10_with_paths(tmp_path):
    """An R10 block whose evidence names files on this machine, as a real run's
    does: one found training file, and validation evidence that did not parse,
    with a problem naming its path. None of it may reach a response."""
    block = r10_block(training_state="overlap")
    block["training"]["evidence"]["path"] = str(tmp_path / "ws" / "train_samples.json")
    block["validation"] = {
        "recorded_sha256": "8" * 64,
        "evidence": {"state": "unreadable", "path": str(tmp_path / "ws" / "val_samples.json"),
                     "problem": f"{tmp_path / 'ws' / 'val_samples.json'} is not JSON"},
        "disease_overlap": {"state": "unverifiable", "k": None, "n": 50, "cases": None},
        "phenotype_set_overlap": {"state": "unverifiable", "k": None, "n": 80},
    }
    return block


def _register(report, checkpoint_dir, registry, *, beside=True, indent=2,
              imported_at=IMPORTED_AT):
    """Register `report` beside `checkpoint_dir` through the one registration
    function, from the registry's report store, the only place a report is
    registered from since M5-1: kept in a new run directory of its version under
    `registry` (`tests/fixtures/report_store.py`), its cohort block naming that
    version's digest, with `registry` as the deployment's (`environ`, BACKLOG
    37)."""
    path = store_report(registry, report, indent=indent, imported_at=imported_at)
    environ = {"SHEPHERD_REGISTRY": str(registry)}
    if beside:
        return register_report(path, checkpoint_dir=checkpoint_dir, registered_at=WHEN,
                               environ=environ)
    # As `--record` registers: the run's read of the checkpoint and the digest
    # of the report it published, so nothing is hashed against the directory.
    measured = MeasuredRun(
        ReadIdentity(checkpoint_dir / "best.pt",
                     report["manifest"]["artifact_digests"]["checkpoint"]),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return register_report(path, measured=measured, registered_at=WHEN, environ=environ)


def _version_digest(registry: Path, label: str = "mygene2-2026-10") -> str:
    """The manifest digest of a version the store fixture wrote."""
    from src.evaluation.cohort_registry import resolve_version

    return resolve_version(registry, label).identity.sha256


def _edit_ledger(path: Path, edit) -> None:
    """A hand edit, the sanctioned correction path: load, change, write back."""
    ledger = json.loads(path.read_text())
    edit(ledger)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True))


def _get(app):
    response = _client(app).get(ROUTE)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    return body


@pytest.fixture
def served(app, monkeypatch, served_child):
    """A real pipeline of the trained disease child, copied into this test's
    directory, built from its one checkpoint and published. Returns
    ``(pipeline, data_dir, checkpoint)``."""
    data_dir, checkpoint = served_child
    pipeline = _full_pipeline(data_dir, checkpoint)
    monkeypatch.setattr(app.app_state, "pipeline", pipeline)
    return pipeline, data_dir, checkpoint


# ---------------------------------------------------------------------------
# 1. The served read (spec §5.1)
# ---------------------------------------------------------------------------
def test_the_served_read_is_the_one_the_model_was_built_from(served):
    pipeline, _, checkpoint = served

    with count_opens() as opens:
        read = pipeline.served_checkpoint_read()

    assert read is not None
    assert read.sha256 == _sha(checkpoint)
    assert Path(read.path) == checkpoint.resolve()
    assert pipeline.served_identity().checkpoint.sha256 == read.sha256
    assert opens.opens(checkpoint) == 0, "no I/O: nothing is read or hashed again"


# ---------------------------------------------------------------------------
# 2. Nothing to look up
# ---------------------------------------------------------------------------
def test_no_pipeline_has_nothing_to_look_up(app):
    body = _get(app)

    assert body == {"state": "no_model", "checkpoint_sha256": None, "ledger": None,
                    "records": [], "records_not_shown": 0, "unreadable_records": []}


def test_the_not_identified_state_is_gone():
    """K7 (G3): no model is supplied in memory, so every served model has a
    checkpoint read to look up, and the response has no state for one that
    does not."""
    import typing

    from pydantic import ValidationError

    from src.api.routes.evaluations import LoadedModelEvaluations

    states = set(typing.get_args(LoadedModelEvaluations.model_fields["state"].annotation))
    assert states == {"no_model", "no_ledger", "ledger_not_read", "no_records", "records"}
    with pytest.raises(ValidationError):
        LoadedModelEvaluations(state="not_identified", checkpoint_sha256=None, ledger=None,
                               records=[], records_not_shown=0, unreadable_records=[])


# ---------------------------------------------------------------------------
# 3. The ledger beside the served checkpoint
# ---------------------------------------------------------------------------
def test_no_ledger_beside_the_checkpoint(app, served):
    _, _, checkpoint = served

    body = _get(app)

    assert body == {
        "state": "no_ledger", "checkpoint_sha256": _sha(checkpoint),
        "ledger": {"file": "evaluations.json", "where": "beside the loaded checkpoint",
                   "sha256": None, "problem": None},
        "records": [], "records_not_shown": 0, "unreadable_records": [],
    }


def test_the_loaded_checkpoints_records_are_shown(app, served, tmp_path, monkeypatch):
    pipeline, _, checkpoint = served
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(tmp_path / "reports"))
    digest = _sha(checkpoint)
    report = _report_for(digest, r10=_r10_with_paths(tmp_path))
    result = _register(report, checkpoint.parent, tmp_path / "reports")
    ledger = checkpoint.parent / "evaluations.json"
    record = result.record

    body = _get(app)

    assert body["state"] == "records"
    assert body["checkpoint_sha256"] == digest == pipeline.served_identity().checkpoint.sha256
    assert body["ledger"] == {"file": "evaluations.json", "where": "beside the loaded checkpoint",
                              "sha256": _sha(ledger), "problem": None}
    assert body["records_not_shown"] == 0
    (view,) = body["records"]
    assert set(view) == RECORD_VIEW_KEYS
    # The record names the run it was registered from (K8-6r.2), and overlaps
    # no record merged before each run had its own.
    (kept,) = (tmp_path / "reports").glob("versions/*/reports/*/report.json")
    assert (view["run_id"], view["historical_overlap"]) == (kept.parent.name, [])
    # Each record carries its own mode, so a surface labels it by that mode.
    assert (view["mode"], view["score_semantics"]) == (
        record["scoring"]["mode"], record["scoring"]["score_semantics"])
    assert view["mode"] == "disease"
    assert view["cohort"] == {
        "label": "mygene2-2026-10", "role": "mygene2", "kind": "supplied",
        "samples_sha256": "5" * 64, "source_sha256": "b" * 64,
        "version_manifest_sha256": record["cohort_digest"],
    }
    assert record["cohort_digest"] == _version_digest(tmp_path / "reports")
    assert view["counts"] == {
        "source_cases": 100, "ranked_eligible": 80,
        "excluded": {"ground_truth_absent": 12, "phenotype_unmappable": 8}, "n_excluded": 20,
    }
    assert view["metrics"] == report["authoritative_metrics"]
    assert view["n_ranked"] == 80
    assert view["relations"] == REGISTERED_RELATIONS
    assert view["r10"] == {
        "training": {
            "evidence": {"state": "found"},
            "disease_overlap": {"state": "overlap", "k": 3, "n": 50, "cases": 5},
            "phenotype_set_overlap": {"state": "none", "k": 0, "n": 80},
        },
        "validation": {
            "evidence": {"state": "unreadable"},
            "disease_overlap": {"state": "unverifiable", "k": None, "n": 50, "cases": None},
            "phenotype_set_overlap": {"state": "unverifiable", "k": None, "n": 80},
        },
    }
    assert view["environment_class"] == "CUDA, deterministic algorithms on"
    assert view["deterministic_algorithms"] is True     # the record's regime says so (M5-3)
    assert view["gpu"] == {"name": "NVIDIA GeForce RTX 4090", "driver": "560.35"}
    assert view["dataset_origin"] == "synthetic"
    assert view["scoring_semantics_digest"] == record["scoring_semantics_digest"]
    assert view["measurement_semantics_digest"] == record["measurement_semantics_digest"]
    assert view["registration"] == {"sequence": 1, "registered_at": WHEN}
    assert view["n_corroborating"] == 0
    (kept,) = (tmp_path / "reports").glob("versions/*/reports/*/report.json")
    assert view["source_artifact_digest"] == _sha(kept) == record["source_artifact_digest"]
    # D5's state, read from the version's report store (BACKLOG 36): one report.
    assert view["contest"] == {"state": "uncontested", "differing": [], "runs": []}


def test_kept_c_and_d_records_are_returned_as_recorded(tmp_path):
    """A guard (R2, K8-6e.2): the record view carries a kept Mode C and a kept
    Mode D record's mode, score semantics and metrics exactly as the ledger
    holds them, beside a task record: no name mapped, no value rounded, no
    score semantics split, and no record left out for its mode."""
    from src.api.routes.evaluations import _recorded_results
    from tests.unit.test_evaluation_sidecar import mode_c_report, mode_d_report

    weights = tmp_path / "runs" / "checkpoints" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"stand-in weights")
    digest = _sha(weights)
    registry = tmp_path / "registry"
    for report in (mode_c_report(), mode_d_report(), registered_report()):
        report["manifest"]["artifact_digests"]["checkpoint"] = digest
        assert _register(report, weights.parent, registry).outcome == "registered"
    ledger = json.loads((weights.parent / "evaluations.json").read_text())
    answer = _recorded_results(ReadIdentity(weights, digest))
    assert answer.state == "records" and answer.records_not_shown == 0
    assert [(v.mode, v.score_semantics, v.metrics) for v in answer.records] == [
        (r["scoring"]["mode"], r["scoring"]["score_semantics"], r["metrics"])
        for r in ledger["records"]]
    assert [v.mode for v in answer.records] == ["C", "D", "disease"]
    assert answer.records[1].score_semantics == "served:gnn_only:eta=1.0:sp=none"
    assert answer.records[0].metrics == {"untruncated_mrr": 0.5, "mean_rank": 3.0,
                                         "hits@1": 0.25, "hits@10": 0.75}


def test_no_path_case_id_or_rank_reaches_the_response(app, served, tmp_path):
    """D7, aggregates only. The record carries the R10 evidence paths and the
    cohort's graph digests; the view leaves both behind."""
    _, _, checkpoint = served
    _register(_report_for(_sha(checkpoint), r10=_r10_with_paths(tmp_path)),
              checkpoint.parent, tmp_path / "reports")
    stored = (checkpoint.parent / "evaluations.json").read_text()
    assert str(tmp_path) in stored, "control: the ledger itself holds the evidence paths"

    response = _client(app).get(ROUTE)
    text = response.text
    records = json.dumps(response.json()["records"])

    assert str(tmp_path) not in text
    for leaked in ("train_samples.json", "val_samples.json", "\"path\"", "case_id",
                   "patient_id", "sample_id", "\"ranks\"", "\"rank\"", "checkpoint_path",
                   "data_dir", "\"graph\"", "\"mapping\""):
        assert leaked not in text, leaked
    # The ledger block states its own problem (null here); a record carries none.
    assert "\"problem\"" not in records and "recorded_sha256" not in records


def _search_incomplete(tmp_path):
    """An R10 block as R10 definition version 2 writes it when the training
    bytes are held by no candidate that was read and one could not be read: its
    problem names that place on this machine."""
    block = r10_block()
    block["definition_version"] = 2
    place = tmp_path / "archive" / "run7_train_samples.json"
    block["training"]["evidence"] = {
        "state": "search_incomplete", "path": None,
        "problem": ("no candidate that was read holds the recorded bytes, and these places "
                    f"could not be looked in: {place} (PermissionError: Permission denied)"),
    }
    return block


def test_a_search_left_incomplete_is_shown_as_recorded_and_names_no_place(
    app, served, tmp_path
):
    """D10 (K8-6r.3). The API's `R10Evidence` takes "search_incomplete": the
    record is shown, its training labels *unverifiable* with no count, and the
    place that could not be read, which the ledger keeps, never reaches the
    response. HEAD: the record fails `R10Evidence` and is counted not shown."""
    _, _, checkpoint = served
    _register(_report_for(_sha(checkpoint), r10=_search_incomplete(tmp_path)),
              checkpoint.parent, tmp_path / "reports")

    response = _client(app).get(ROUTE)
    body = response.json()

    assert body["records_not_shown"] == 0
    (view,) = body["records"]
    assert view["r10"]["training"] == {
        "evidence": {"state": "search_incomplete"},
        "disease_overlap": {"state": "unverifiable", "k": None, "n": 50, "cases": None},
        "phenotype_set_overlap": {"state": "unverifiable", "k": None, "n": 80},
    }
    assert str(tmp_path) not in response.text and "PermissionError" not in response.text


def test_a_version_1_block_is_read_with_its_version_1_meaning(app, served, tmp_path):
    """D11 (guard; Decision 7). A record whose R10 block is definition version
    1, with a `not_found` scope, is read as written by the same reader: under
    version 1 "not_found" meant every candidate was read and compared, and it
    is shown as `not_found`, *unverifiable* with no count; nothing migrates it."""
    _, _, checkpoint = served
    block = r10_block()
    assert block["definition_version"] == 1
    assert block["training"]["evidence"]["state"] == "not_found"
    _register(_report_for(_sha(checkpoint), r10=block), checkpoint.parent,
              tmp_path / "reports")
    stored = (checkpoint.parent / "evaluations.json").read_bytes()

    (view,) = _get(app)["records"]

    assert view["r10"]["training"] == {
        "evidence": {"state": "not_found"},
        "disease_overlap": {"state": "unverifiable", "k": None, "n": 50, "cases": None},
        "phenotype_set_overlap": {"state": "unverifiable", "k": None, "n": 80},
    }
    assert (checkpoint.parent / "evaluations.json").read_bytes() == stored


def test_records_are_sorted_by_registration_sequence(app, served, tmp_path):
    _, _, checkpoint = served
    digest = _sha(checkpoint)
    _register(_report_for(digest), checkpoint.parent, tmp_path / "reports")
    _register(_report_for(digest, cohort_version=cohort_version(
        label="mygene2-2026-11", manifest_sha256="9" * 64)),
        checkpoint.parent, tmp_path / "reports")
    _edit_ledger(checkpoint.parent / "evaluations.json",
                 lambda ledger: ledger["records"].reverse())

    body = _get(app)

    assert [view["registration"]["sequence"] for view in body["records"]] == [1, 2]
    assert [view["cohort"]["label"] for view in body["records"]] == [
        "mygene2-2026-10", "mygene2-2026-11"]


def test_two_imports_of_one_label_are_shown_as_two_versions(app, served, tmp_path):
    """The same source imported under the same label into a recreated registry
    is another version: another manifest digest, so another record key. Catches
    a view that leaves the digest out, under which the two records read as one
    cohort measured twice (identical label, role, samples and source)."""
    _, _, checkpoint = served
    digest = _sha(checkpoint)
    _register(_report_for(digest), checkpoint.parent, tmp_path / "reports")
    # The recreated registry's import: the same label, imported at another time.
    _register(_report_for(digest), checkpoint.parent, tmp_path / "recreated",
              imported_at="2026-10-17T12:00:00Z")

    first, second = _get(app)["records"]

    assert ({**first["cohort"], "version_manifest_sha256": None}
            == {**second["cohort"], "version_manifest_sha256": None})
    assert first["cohort"]["version_manifest_sha256"] == _version_digest(tmp_path / "reports")
    assert second["cohort"]["version_manifest_sha256"] == _version_digest(
        tmp_path / "recreated")
    assert (first["cohort"]["version_manifest_sha256"]
            != second["cohort"]["version_manifest_sha256"])


def _old_merged_ledger(ledger_path: Path, source: Path, merged: Path) -> dict:
    """A schema 3 ledger as the writer before K8-6r.2 left it: one record
    built from `source`, naming no run, with `merged`'s digest merged into it
    as a corroborating report (what that writer did with equal values)."""
    from src.evaluation.sidecar import build_record, empty_ledger, write_ledger

    record = build_record(json.loads(source.read_bytes()), _sha(source))
    record["corroborating_reports"] = [_sha(merged)]
    record["registration"] = {"sequence": 1, "registered_at": "2026-10-15T12:00:00Z"}
    write_ledger(ledger_path, {**empty_ledger(), "schema_version": 3, "records": [record]}, None)
    return record


def test_an_old_merged_record_and_the_runs_whose_bytes_it_holds_are_kept_apart(
    app, served, tmp_path, monkeypatch
):
    """G1, A7 (d) (K8-6r.2): a record merged before each run had its own record
    is shown as written, its run null ("not recorded") and its merged report
    counted; each run that registers bytes it holds is a record of its own,
    naming its run. Both sides show the possible overlap (`historical_overlap`),
    never confirmed and never counted as two independent runs. The old record
    gains no run. HEAD: no `run_id` field, and both runs answered
    `already_registered` (one record)."""
    from tests.fixtures.report_store import store_report

    _, _, checkpoint = served
    registry = tmp_path / "reports"
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    report = _report_for(_sha(checkpoint))
    first = store_report(registry, report)                       # D1, run R1
    second = store_report(registry, report, indent=4)            # D2, run R2
    ledger_path = checkpoint.parent / "evaluations.json"
    old = _old_merged_ledger(ledger_path, first, second)
    environ = {"SHEPHERD_REGISTRY": str(registry)}
    for path in (second, first):
        result = register_report(path, checkpoint_dir=checkpoint.parent, registered_at=WHEN,
                                 environ=environ)
        assert result.outcome == "registered" and result.historical_overlap

    body = _get(app)

    assert body["records_not_shown"] == 0
    views = {view["registration"]["sequence"]: view for view in body["records"]}
    assert sorted(views) == [1, 2, 3]
    assert (views[1]["run_id"], views[1]["n_corroborating"], views[1]["historical_overlap"]) == (
        None, 1, [2, 3])
    assert (views[2]["run_id"], views[2]["n_corroborating"], views[2]["historical_overlap"]) == (
        second.parent.name, 0, [1])
    assert (views[3]["run_id"], views[3]["n_corroborating"], views[3]["historical_overlap"]) == (
        first.parent.name, 0, [1])
    ledger = json.loads(ledger_path.read_text())
    assert ledger["records"][0] == old and "run_id" not in ledger["records"][0]
    assert (ledger["schema_version"], ledger["run_identity_from_sequence"]) == (4, 2)


def test_every_run_of_a_group_is_its_own_record_and_the_description_names_each(
    app, served, tmp_path, monkeypatch
):
    """G1 (K8-6r.2): two byte-identical runs and a run whose value differs, all
    registered, are three records, each naming its run; the group's
    description names every run carrying a differing report, both
    byte-identical runs included. HEAD: the byte-identical run answered
    `already_registered` and the differing one was refused (one record)."""
    from tests.fixtures.report_store import store_bytes, version_of

    registry, kept, record = _registered_with_registry(served, tmp_path, monkeypatch)
    twin = store_bytes(registry, kept.read_bytes(), version=version_of(kept),
                       checkpoint_sha256=record["checkpoint_digest"])
    other = _disagreeing(kept, 0.4321)
    _, _, checkpoint = served
    environ = {"SHEPHERD_REGISTRY": str(registry)}
    for path in (twin, other):
        assert register_report(path, checkpoint_dir=checkpoint.parent, registered_at=WHEN,
                               environ=environ).outcome == "registered"

    views = _get(app)["records"]

    assert [view["run_id"] for view in views] == [kept.parent.name, twin.parent.name,
                                                  other.parent.name]
    for view in views:
        assert view["contest"]["state"] == "contested"
        assert sorted(run["run"] for run in view["contest"]["runs"]) == sorted(
            [kept.parent.name, twin.parent.name, other.parent.name])
        assert view["historical_overlap"] == []


@pytest.mark.parametrize("damage", ["no run_id", "boundary lost"])
def test_a_record_whose_run_cannot_be_read_is_not_shown_as_historical(
    app, served, tmp_path, monkeypatch, damage
):
    """A8, A12 (f) (K8-6r.2): a record at or after the run-identity boundary
    without its run, or every record that names no run once the boundary is
    lost, is counted as not shown (never shown with run "not recorded"); a
    record that still names its run is shown, and the ledger is read, never
    ``ledger_not_read``. HEAD: no schema 4, so the file was not read."""
    registry, kept, record = _registered_with_registry(served, tmp_path, monkeypatch)
    _, _, checkpoint = served
    twin = _disagreeing(kept, 0.4321)
    register_report(twin, checkpoint_dir=checkpoint.parent, registered_at=WHEN,
                    environ={"SHEPHERD_REGISTRY": str(registry)})
    ledger = checkpoint.parent / "evaluations.json"
    if damage == "no run_id":
        _edit_ledger(ledger, lambda data: data["records"][0].pop("run_id"))
    else:
        _edit_ledger(ledger, lambda data: (data.pop("run_identity_from_sequence"),
                                           data["records"][0].pop("run_id")))

    body = _get(app)

    assert body["state"] == "records" and body["records_not_shown"] == 1
    (view,) = body["records"]
    assert view["run_id"] == twin.parent.name
    # K8-6e.1 (rev6): listed with the reason its identity gives, its run null,
    # never as a historical record; with the boundary lost, the file is still
    # read (never ledger_not_read) and the reason says the boundary cannot be read.
    (item,) = body["unreadable_records"]
    assert (item["index"], item["sequence"], item["run_id"]) == (0, 1, None)
    assert item["source_artifact_digest"] == _sha(kept)
    assert item["reason"] == (
        "its run identity is missing" if damage == "no run_id" else
        "whether it names its run cannot be told: the file's run-identity boundary "
        "(run_identity_from_sequence) cannot be read (it is missing while the file holds 2 "
        "records)")


def test_records_for_other_weights_only_are_no_records(app, served, tmp_path):
    _, _, checkpoint = served
    _register(_report_for(OTHER_CHECKPOINT), checkpoint.parent, tmp_path / "reports",
              beside=False)
    ledger = checkpoint.parent / "evaluations.json"

    body = _get(app)

    assert body["state"] == "no_records"
    assert body["records"] == [] and body["records_not_shown"] == 0
    assert body["ledger"]["sha256"] == _sha(ledger)
    assert OTHER_CHECKPOINT not in json.dumps(body)


# ---------------------------------------------------------------------------
# 4. A ledger that is not read
# ---------------------------------------------------------------------------
def test_an_earlier_schema_ledger_is_not_read_and_shows_nothing(app, served):
    """§5: a schema 2 file is stated as not read, never shown as records, even
    when it holds a record that names these weights."""
    _, _, checkpoint = served
    ledger = checkpoint.parent / "evaluations.json"
    ledger.write_text(json.dumps({"schema_version": 2, "records": [
        {"checkpoint_digest": _sha(checkpoint), "metrics": {"untruncated_mrr": 0.9}},
    ]}))

    body = _get(app)

    assert body["state"] == "ledger_not_read"
    assert body["records"] == [] and body["records_not_shown"] == 0
    assert body["unreadable_records"] == []      # the whole file's state (K8-6e.1)
    problem = body["ledger"]["problem"]
    assert problem.startswith("evaluations.json is ledger schema 2")
    assert "evaluations.v2.json" in problem and "re-measure with this revision" in problem
    assert str(checkpoint.parent) not in json.dumps(body)
    assert body["ledger"]["sha256"] is None


@pytest.mark.parametrize("damage, says", [
    (b'{"schema_version": 3, "records": [', "does not parse as UTF-8 JSON"),
    (b"\xff\xfe not text", "does not parse as UTF-8 JSON"),
    (b"[]", "holds a JSON list"),
    (b'{"schema_version": 3}', "has no records list"),
    (b'{"schema_version": true, "records": []}', "ledger schema True"),
])
def test_a_damaged_ledger_is_not_read(app, served, damage, says):
    _, _, checkpoint = served
    (checkpoint.parent / "evaluations.json").write_bytes(damage)

    body = _get(app)

    assert body["state"] == "ledger_not_read"
    # A file unreadable as a whole is its own state (K8-6e.1): no record of it
    # is listed as unreadable, and nothing is recovered from it.
    assert body["records"] == [] and body["unreadable_records"] == []
    assert body["records_not_shown"] == 0
    assert body["ledger"]["problem"].startswith("evaluations.json")
    assert says in body["ledger"]["problem"]
    assert str(checkpoint.parent) not in json.dumps(body)


def test_a_ledger_that_cannot_be_opened_is_not_read(app, served):
    _, _, checkpoint = served
    (checkpoint.parent / "evaluations.json").mkdir()

    body = _get(app)

    assert body["state"] == "ledger_not_read"
    assert body["ledger"]["problem"].startswith(
        "evaluations.json could not be read (IsADirectoryError")
    assert str(checkpoint.parent) not in json.dumps(body)


@pytest.mark.parametrize("directory", ["back\\slash", "new\nline", "tab\tdir"],
                         ids=["backslash", "newline", "tab"])
def test_an_unopenable_ledger_names_no_directory_whatever_its_name(
    app, served_child, tmp_path, monkeypatch, directory
):
    """`str(OSError)` quotes the filename with ``repr``, which escapes these
    characters, so replacing the path in it left the server's directory in the
    answer. Catches a problem built from `str(exc)`."""
    data_dir, checkpoint = served_child
    base = tmp_path / directory
    base.mkdir()
    shutil.copy2(checkpoint, base / "best.pt")
    (base / "evaluations.json").mkdir()
    monkeypatch.setattr(app.app_state, "pipeline", _full_pipeline(data_dir, base / "best.pt"))

    response = _client(app).get(ROUTE)
    body = response.json()

    assert body["state"] == "ledger_not_read"
    assert body["ledger"]["problem"] == (
        "evaluations.json could not be read (IsADirectoryError: Is a directory)")
    for form in (str(tmp_path), repr(str(tmp_path))[1:-1], json.dumps(str(tmp_path))[1:-1]):
        assert form not in response.text
    assert directory not in json.dumps(body) and repr(directory)[1:-1] not in response.text


# ---------------------------------------------------------------------------
# 5. A record of the wrong shape is listed in unreadable_records with its reason,
#    never shown as a result, never a 500
# ---------------------------------------------------------------------------
def _set(path, value):
    def edit(record):
        target = record
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return edit


def _delete(*path):
    def edit(record):
        target = record
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return edit


MALFORMED = {
    "no cohort": _delete("cohort"),
    "no cohort digest": _delete("cohort_digest"),
    "cohort digest not hex": _set(("cohort_digest",), "a" * 63),
    "cohort a list": _set(("cohort",), []),
    "cohort kind generated": _set(("cohort", "kind"), "generated"),
    "samples digest not hex": _set(("cohort", "samples_sha256"), "not-a-digest"),
    "empty label": _set(("cohort", "label"), ""),
    "cohort_role is not the cohort's role": _set(("cohort_role",), "udn"),
    "exclusions do not add up to n_excluded": _set(
        ("cohort", "counts", "excluded", "ground_truth_absent"), 13),
    "ranked and excluded miss a case": _set(("cohort", "counts", "source_cases"), 101),
    "a negative count": _set(("cohort", "counts", "excluded", "ground_truth_absent"), -1),
    "a count as a bool": _set(("n_ranked",), True),
    "a metric as text": _set(("metrics", "pooled_mrr"), "0.5"),
    "a metric that is not finite": _set(("metrics", "pooled_top10"), float("nan")),
    "metrics a list": _set(("metrics",), [0.5]),
    "R2 verified": _set(("relations", "R2"), "verified"),
    "a relation missing": _delete("relations", "cohort_graph"),
    "unverifiable with a zero": _set(("r10", "training", "disease_overlap", "k"), 0),
    "overlap with no count": _set(("r10", "validation", "phenotype_set_overlap"),
                                  {"state": "overlap", "k": None, "n": 80}),
    "none that counts an overlap": _set(("r10", "training", "phenotype_set_overlap"),
                                        {"state": "none", "k": 4, "n": 80}),
    "overlap that counts none": _set(("r10", "training", "phenotype_set_overlap"),
                                     {"state": "overlap", "k": 0, "n": 80}),
    "more overlap than test cases": _set(("r10", "training", "phenotype_set_overlap"),
                                         {"state": "overlap", "k": 81, "n": 80}),
    "training with no validation inputs": _set(
        ("r10", "training", "disease_overlap"),
        {"state": "no_validation_inputs_recorded", "k": None, "n": 50, "cases": None}),
    "an unknown R10 state": _set(("r10", "training", "disease_overlap", "state"), "leaky"),
    "an unknown evidence state": _set(("r10", "training", "evidence", "state"), "lost"),
    "no validation scope": _delete("r10", "validation"),
    "r10 null": _set(("r10",), None),
    "a device neither CUDA nor CPU": _set(("regime", "device"), "tpu"),
    "determinism not a boolean": _set(("regime", "deterministic_algorithms"), "yes"),
    "no regime": _delete("regime"),
    "gpu a string": _set(("environment", "facts", "gpu"), "RTX 4090"),
    "no environment facts": _delete("environment", "facts"),
    "an unknown dataset origin": _set(("dataset_record", "origin"), "hospital"),
    "corroborating reports not a list": _set(("corroborating_reports",), 2),
    "registration with no sequence": _delete("registration", "sequence"),
    "sequence zero": _set(("registration", "sequence"), 0),
    "report digest not hex": _set(("source_artifact_digest",), "e" * 63),
    "no scoring digest": _delete("scoring_semantics_digest"),
    "scoring a list": _set(("scoring",), []),
    "environment a string": _set(("environment",), "x"),
    "dataset_record a number": _set(("dataset_record",), 3),
}


#: The reasons some edits above are listed with (K8-6e.1), pinned word for
#: word: the refused field's path in the ledger record (never the API view's
#: name for it) and what is wrong, never the value.
MALFORMED_REASONS = {
    "a metric as text": "metrics.pooled_mrr: Input should be a valid number",
    "no cohort digest": "cohort_digest: missing",
    "registration with no sequence": "registration.sequence: missing",
    "sequence zero": "registration.sequence: Input should be greater than or equal to 1",
    "report digest not hex": ("source_artifact_digest: String should match pattern "
                              "'^[0-9a-f]{64}$'"),
    "a device neither CUDA nor CPU": "the regime's device is neither CUDA nor CPU",
    "determinism not a boolean": "the regime's deterministic_algorithms is not a boolean",
    "exclusions do not add up to n_excluded": (
        "cohort.counts: the exclusion counts do not add up to n_excluded"),
    "scoring a list": "scoring.mode: missing; scoring.score_semantics: missing",
    "environment a string": "environment.facts.gpu: missing",
    "dataset_record a number": "dataset_record.origin: missing",
    "corroborating reports not a list": "its corroborating_reports is not a list",
    "r10 null": ("r10.training.evidence: missing; r10.training.disease_overlap: missing; "
                 "r10.training.phenotype_set_overlap: missing; 3 more"),
}
UNREADABLE_KEYS = {"index", "sequence", "source_artifact_digest", "run_id", "reason"}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_a_malformed_record_for_these_weights_is_listed_with_its_reason_not_shown(
    app, served, tmp_path, name, caplog
):
    """Each edit makes the record one registration never writes. Catches a view
    that shows it half-read, and one that raises: the route would 500 and the
    model status with it. K8-6e.1 (R3): it is listed in ``unreadable_records``
    at index 0 with a one-line reason of 1 to 300 characters naming no value,
    each refused field by its path in the ledger record (a reason that named
    the view's path, "gpu: missing" for a record whose ``environment`` is
    text, would send the operator to a key the file does not have); its sequence is null exactly where the edit took it, its report digest
    null exactly where the edit broke it, and its run is the one it names;
    ``records_not_shown`` counts it, and the server log says the same line.
    HEAD: no ``unreadable_records``, and the log carried pydantic's multi-line
    text with the input."""
    _, _, checkpoint = served
    _register(_report_for(_sha(checkpoint)), checkpoint.parent, tmp_path / "reports")
    ledger_path = checkpoint.parent / "evaluations.json"
    written = json.loads(ledger_path.read_text())["records"][0]
    _edit_ledger(ledger_path, lambda ledger: MALFORMED[name](ledger["records"][0]))

    body = _get(app)

    assert body["state"] == "records"
    assert body["records"] == []
    assert body["records_not_shown"] == 1
    assert "is not shown" in caplog.text
    (item,) = body["unreadable_records"]
    assert set(item) == UNREADABLE_KEYS
    assert item["index"] == 0
    assert item["sequence"] == (
        None if name in ("registration with no sequence", "sequence zero") else 1)
    assert item["source_artifact_digest"] == (
        None if name == "report digest not hex" else written["source_artifact_digest"])
    assert item["run_id"] == written["run_id"]
    reason = item["reason"]
    assert 1 <= len(reason) <= 300 and reason.isprintable()
    if name in MALFORMED_REASONS:
        assert reason == MALFORMED_REASONS[name]
    # Every field the reason names is a path the record registration wrote
    # has: the key a hand edit corrects, never the view's name for it.
    for part in reason.split("; "):
        if ": " in part and not part.startswith(("its ", "the ")):
            target = written
            for step in part.split(": ", 1)[0].split("."):
                assert isinstance(target, dict) and step in target, (part, step)
                target = target[step]
    assert "input_value" not in reason and "errors.pydantic.dev" not in reason
    assert f"the ledger's record 0: {reason}" in caplog.text
    assert "input_value" not in caplog.text
    assert str(tmp_path) not in json.dumps(body)


def test_the_unedited_record_is_shown_where_the_edits_are_not(app, served, tmp_path):
    """Control for the table above: the same registration, unedited, is shown."""
    _, _, checkpoint = served
    _register(_report_for(_sha(checkpoint)), checkpoint.parent, tmp_path / "reports")
    _edit_ledger(checkpoint.parent / "evaluations.json", lambda ledger: None)

    body = _get(app)

    assert len(body["records"]) == 1 and body["records_not_shown"] == 0


def test_a_malformed_record_does_not_hide_a_good_one(app, served, tmp_path):
    _, _, checkpoint = served
    digest = _sha(checkpoint)
    _register(_report_for(digest), checkpoint.parent, tmp_path / "reports")
    _register(_report_for(digest, cohort_version=cohort_version(
        label="mygene2-2026-11", manifest_sha256="9" * 64)),
        checkpoint.parent, tmp_path / "reports")
    _edit_ledger(checkpoint.parent / "evaluations.json",
                 lambda ledger: MALFORMED["no cohort"](ledger["records"][0]))

    body = _get(app)

    assert body["state"] == "records"
    assert [view["cohort"]["label"] for view in body["records"]] == ["mygene2-2026-11"]
    assert body["records_not_shown"] == 1
    assert [item["index"] for item in body["unreadable_records"]] == [0]


def _samples_file(path, rows):
    data = json.dumps([{"patient_id": p, "phenotype_ids": list(ph), "disease_id": d}
                       for p, ph, d in rows]).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _computed_r10(tmp_path, situation):
    """An R10 block from R10's own computation (`compute_r10`), in one of the
    situations the contract names."""
    from src.evaluation.r10 import compute_r10
    from src.kg.data_loader import DiagnosisSample

    test = [DiagnosisSample(patient_id="t0", phenotype_ids=[10, 11], disease_id=1),
            DiagnosisSample(patient_id="t1", phenotype_ids=[20], disease_id=2),
            DiagnosisSample(patient_id="t2", phenotype_ids=[30], disease_id=3)]
    train, val = tmp_path / "evidence" / "train_samples.json", tmp_path / "evidence" / "val_samples.json"
    if situation == "found, overlap and none":
        recorded = {"train_samples": _samples_file(train, [("p", [11, 10], 1)]),
                    "val_samples": _samples_file(val, [("q", [40], 9)])}
        return compute_r10(recorded, test, [train, val])
    if situation == "no validation inputs recorded":
        return compute_r10({"train_samples": _samples_file(train, [("p", [20], 2)])},
                           test, [train])
    if situation == "validation entry not a digest":
        recorded = {"train_samples": _samples_file(train, [("p", [20], 2)]),
                    "val_samples": {"sha256": "b" * 64}}
        return compute_r10(recorded, test, [train])
    if situation == "evidence not found":
        return compute_r10({"train_samples": "a" * 64, "val_samples": "b" * 64}, test, [])
    if situation == "search incomplete":
        train.mkdir(parents=True)  # a directory in a candidate's place: never read
        return compute_r10({"train_samples": "a" * 64}, test, [train])
    if situation == "evidence unreadable":
        train.parent.mkdir(parents=True, exist_ok=True)
        train.write_bytes(b"not json")
        digest = hashlib.sha256(b"not json").hexdigest()
        return compute_r10({"train_samples": digest}, test, [train])
    assert situation == "nothing recorded"
    return compute_r10(None, test, [])


@pytest.mark.parametrize("situation", [
    "found, overlap and none", "no validation inputs recorded",
    "validation entry not a digest", "evidence not found", "evidence unreadable",
    "search incomplete", "nothing recorded",
])
def test_every_block_r10_computes_is_shown_as_computed(tmp_path, situation):
    """The view's R10 rules are R10's own: a block `compute_r10` writes is never
    refused as malformed, and is shown as computed, less its paths."""
    from src.api.routes.evaluations import _record_view
    from src.evaluation.sidecar import append_record, build_record, empty_ledger

    block = _computed_r10(tmp_path, situation)
    record = build_record(_report_for(OTHER_CHECKPOINT, r10=block), "e" * 64)
    _, ledger = append_record(empty_ledger(), record,
                              run_id=f"20261016T120000Z-{OTHER_CHECKPOINT[:12]}-00000001",
                              registered_at=WHEN)

    view = _record_view(ledger["records"][0])

    assert view.r10.model_dump() == {
        scope: {"evidence": {"state": block[scope]["evidence"]["state"]},
                "disease_overlap": block[scope]["disease_overlap"],
                "phenotype_set_overlap": block[scope]["phenotype_set_overlap"]}
        for scope in ("training", "validation")
    }
    assert str(tmp_path) not in view.model_dump_json()


def test_a_record_that_is_not_an_object_or_names_no_digest_is_not_these_weights(
    app, served, tmp_path
):
    """`records_for` tolerates hand edits: neither is a record of these weights."""
    _, _, checkpoint = served
    _register(_report_for(_sha(checkpoint)), checkpoint.parent, tmp_path / "reports")
    _edit_ledger(checkpoint.parent / "evaluations.json",
                 lambda ledger: ledger["records"].extend(["a string", {"metrics": {}}]))

    body = _get(app)

    assert len(body["records"]) == 1 and body["records_not_shown"] == 0
    # They belong to no model (policy rev 15): no model's view lists them.
    assert body["unreadable_records"] == []


# ---------------------------------------------------------------------------
# 6. Found by digest: another model, moved bytes, a reload mid-request
# ---------------------------------------------------------------------------
def test_a_new_model_never_shows_the_previous_models_records(app, served, tmp_path,
                                                            monkeypatch):
    """Both checkpoints beside one ledger; only A's report is registered. After B
    is published, nothing of A's is shown."""
    pipeline_a, data_dir, checkpoint_a = served
    checkpoint_b = _with_val_pooled_mrr(checkpoint_a, checkpoint_a.parent / "other.pt", 0.25,
                                        shift=1.0)
    assert _sha(checkpoint_b) != _sha(checkpoint_a)
    result = _register(_report_for(_sha(checkpoint_a)), checkpoint_a.parent,
                       tmp_path / "reports")
    assert _get(app)["state"] == "records", "control: A's records are shown while A serves"

    monkeypatch.setattr(app.app_state, "pipeline", _full_pipeline(data_dir, checkpoint_b))
    body = _get(app)

    assert body["state"] == "no_records"
    assert body["checkpoint_sha256"] == _sha(checkpoint_b)
    text = json.dumps(body)
    assert _sha(checkpoint_a) not in text
    assert result.record["source_artifact_digest"] not in text


def test_the_same_bytes_moved_with_their_ledger_find_their_records(app, served, tmp_path,
                                                                   monkeypatch):
    _, data_dir, checkpoint = served
    _register(_report_for(_sha(checkpoint)), checkpoint.parent, tmp_path / "reports")
    moved = tmp_path / "elsewhere" / "checkpoints"
    moved.mkdir(parents=True)
    shutil.copy2(checkpoint, moved / "renamed.pt")
    shutil.copy2(checkpoint.parent / "evaluations.json", moved / "evaluations.json")
    (checkpoint.parent / "evaluations.json").unlink()

    monkeypatch.setattr(app.app_state, "pipeline", _full_pipeline(data_dir, moved / "renamed.pt"))
    body = _get(app)

    assert body["state"] == "records"
    assert body["checkpoint_sha256"] == _sha(checkpoint)
    assert body["ledger"]["sha256"] == _sha(moved / "evaluations.json")
    assert len(body["records"]) == 1


def test_the_ledger_is_the_one_beside_the_served_read_not_a_path_in_app_state(
    app, served, tmp_path, monkeypatch
):
    """Application state's checkpoint path is read separately from the pipeline
    and can belong to another load. Catches a route that looks there: the
    records sit beside that path, not beside the read."""
    _, _, checkpoint = served
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.copy2(checkpoint, elsewhere / "checkpoint.pt")
    _register(_report_for(_sha(checkpoint)), elsewhere, tmp_path / "reports")
    monkeypatch.setattr(app.app_state, "_current_checkpoint_path",
                        str(elsewhere / "checkpoint.pt"), raising=False)

    body = _get(app)

    assert body["state"] == "no_ledger"


def test_a_symlinked_checkpoint_reads_the_ledger_beside_its_target(app, served_child,
                                                                   tmp_path, monkeypatch):
    """`Path(checkpoint).resolve().parent` on both sides (spec §0.15):
    registration resolves the directory it writes into, and so does this."""
    data_dir, checkpoint = served_child
    _register(_report_for(_sha(checkpoint)), checkpoint.parent, tmp_path / "reports")
    links = tmp_path / "links"
    links.mkdir()
    (links / "served.pt").symlink_to(checkpoint)
    monkeypatch.setattr(app.app_state, "pipeline", _full_pipeline(data_dir, links / "served.pt"))

    body = _get(app)

    assert body["state"] == "records"


def test_a_link_retargeted_without_a_reload_still_finds_the_served_models_ledger(
    app, served_child, tmp_path, monkeypatch
):
    """The directory is settled when the checkpoint is read. The model is served
    through `current.pt -> run A`, then the link is pointed at run B to prepare
    the next reload, and nothing is reloaded: A is still served, so A's ledger
    is still the one read. Catches a lookup that resolves the link again on each
    request, which would answer `no_ledger` for A's intact records."""
    data_dir, checkpoint_a = served_child
    run_b = tmp_path / "run_b"
    run_b.mkdir()
    checkpoint_b = _with_val_pooled_mrr(checkpoint_a, run_b / "best.pt", 0.25, shift=1.0)
    assert _sha(checkpoint_b) != _sha(checkpoint_a)
    _register(_report_for(_sha(checkpoint_a)), checkpoint_a.parent, tmp_path / "reports")
    current = tmp_path / "models" / "current.pt"
    current.parent.mkdir()
    current.symlink_to(checkpoint_a)
    pipeline = _full_pipeline(data_dir, current)
    monkeypatch.setattr(app.app_state, "pipeline", pipeline)
    assert Path(pipeline.served_checkpoint_read().path) == checkpoint_a.resolve()
    assert _get(app)["state"] == "records", "control: A's records while the link names A"

    current.unlink()
    current.symlink_to(checkpoint_b)
    body = _get(app)

    assert body["checkpoint_sha256"] == _sha(checkpoint_a), "A is still the served model"
    assert body["state"] == "records"
    assert body["ledger"]["sha256"] == _sha(checkpoint_a.parent / "evaluations.json")
    assert len(body["records"]) == 1


def test_a_reload_landing_mid_request_cannot_mix_two_models(app, served, tmp_path,
                                                           monkeypatch):
    """The pipeline is read from application state once. Catches a route that
    reads it again after taking the identity: the answer would describe B's
    checkpoint with A's lookup, or the reverse."""
    pipeline_a, data_dir, checkpoint_a = served
    (tmp_path / "b").mkdir()
    checkpoint_b = _with_val_pooled_mrr(checkpoint_a, tmp_path / "b" / "b.pt", 0.25, shift=1.0)
    pipeline_b = _full_pipeline(data_dir, checkpoint_b)
    _register(_report_for(_sha(checkpoint_a)), checkpoint_a.parent, tmp_path / "reports")
    original = pipeline_a.served_checkpoint_read

    def read_then_reload():
        read = original()
        app.app_state.pipeline = pipeline_b
        return read

    monkeypatch.setattr(pipeline_a, "served_checkpoint_read", read_then_reload)

    body = _get(app)

    assert app.app_state.pipeline is pipeline_b, "the reload did land"
    assert body["checkpoint_sha256"] == _sha(checkpoint_a)
    assert body["state"] == "records"


def test_the_handler_awaits_nothing_before_the_identity_is_taken():
    """FastAPI runs a plain `def` route in a thread pool, where a reload's
    assignments could interleave with it; an `await` would yield to one. Neither
    may happen between reading the pipeline and taking its identity. Its one
    ``await`` comes after `served_checkpoint_read()` and hands the file reads
    (the ledger and the report stores, BACKLOG 36) to the thread pool with the
    read already taken."""
    from src.api.routes import evaluations

    handler = evaluations.get_loaded_model_evaluations
    assert inspect.iscoroutinefunction(handler)
    tree = ast.parse(textwrap.dedent(inspect.getsource(handler)))
    awaits = [node for node in ast.walk(tree)
              if isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith))]
    (identity,) = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and getattr(node.func, "attr", None) == "served_checkpoint_read"]
    (only,) = awaits
    assert only.lineno > identity.lineno
    assert ast.unparse(only) == "await run_in_threadpool(_recorded_results, read)"


# ---------------------------------------------------------------------------
# 7. Read-only, nothing from the client, its own router
# ---------------------------------------------------------------------------
def test_no_query_parameter_selects_anything(app, served, tmp_path):
    """The route declares no parameter; one sent anyway changes nothing. Catches
    a parameter that would let a client point the reader at a file or digest."""
    _, _, checkpoint = served
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _register(_report_for(OTHER_CHECKPOINT), elsewhere, tmp_path / "reports", beside=False)
    client = _client(app)

    plain = client.get(ROUTE).json()
    probed = client.get(ROUTE, params={
        "path": str(elsewhere / "evaluations.json"), "ledger": str(elsewhere),
        "checkpoint": str(elsewhere / "x.pt"), "checkpoint_sha256": OTHER_CHECKPOINT,
        "digest": OTHER_CHECKPOINT,
    }).json()

    assert probed == plain
    assert plain["state"] == "no_ledger"
    operation = app.app.openapi()["paths"][ROUTE]
    assert set(operation) == {"get"}
    assert not operation["get"].get("parameters")
    assert "requestBody" not in operation["get"]


def test_the_route_opens_the_ledger_the_manifests_and_the_shown_versions_reports_only(
        app, served, tmp_path, monkeypatch):
    """D8 and EB-4: everything one request opens, under the test's directories.
    The ledger, once; each version manifest of the registry,
    ``versions/*/cohort.json``, once (to find the record's version by the cohort
    digest its key carries; the version found is that read, never read again);
    and, for D5's state, each report the record's version store keeps, once
    (BACKLOG 36). Nothing else: no checkpoint, and no other version's store, not
    even listed, although an unrelated version keeps a report beside it. No
    `torch.load`, no directory hash (`find_checkpoint`), no write, the report
    store's usage history included."""
    import copy

    import src.evaluation.sidecar as sidecar
    import src.utils.fingerprint as fingerprint
    from tests.fixtures.report_store import store_bytes, store_version

    _, _, checkpoint = served
    registry = tmp_path / "reports"
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    _register(_report_for(_sha(checkpoint)), checkpoint.parent, registry)
    ledger = checkpoint.parent / "evaluations.json"
    (kept,) = registry.glob("versions/*/reports/*/report.json")
    unrelated = copy.deepcopy(cohort_version())
    unrelated["label"] = "unrelated-2026-11"
    other = store_version(registry, unrelated)
    store_bytes(registry, b"{}", version=other)
    manifests = sorted(registry.glob("versions/*/cohort.json"))
    assert len(manifests) == 2
    before = {path.name: path.stat().st_mtime_ns for path in checkpoint.parent.iterdir()}
    store_before = {path: path.stat().st_mtime_ns
                    for path in (tmp_path / "reports").rglob("*") if path.is_file()}
    ledger_bytes = ledger.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("the read-only view must not load or hash a checkpoint")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(sidecar, "find_checkpoint", forbidden)
    monkeypatch.setattr(fingerprint, "file_sha256", forbidden)
    with count_opens() as opens:
        body = _get(app)

    assert body["state"] == "records"
    assert body["records"][0]["contest"]["state"] == "uncontested"
    here = str(tmp_path.resolve())
    opened = {path: n for path, n in opens.counts.items() if n and path.startswith(here)}
    assert opened == {str(ledger.resolve()): 1, str(kept.resolve()): 1,
                      **{str(manifest.resolve()): 1 for manifest in manifests}}
    assert opens.opens(checkpoint) == 0
    assert opens.listed(other.directory / "reports") == 0
    assert opens.listed(kept.parents[1]) == 1
    assert {path.name: path.stat().st_mtime_ns for path in checkpoint.parent.iterdir()} == before
    assert {path: path.stat().st_mtime_ns for path in (tmp_path / "reports").rglob("*")
            if path.is_file()} == store_before
    assert ledger.read_bytes() == ledger_bytes


def test_the_route_module_names_no_checkpoint_reader_or_writer():
    """The static half of the test above: a call reached only on a path that
    test does not take is still found."""
    from src.api.routes import evaluations

    tree = ast.parse(Path(evaluations.__file__).read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {alias.name.split(".")[0] for node in ast.walk(tree)
              if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    for name in ("find_checkpoint", "file_sha256", "read_once", "write_ledger", "torch",
                 "append_record", "register_report", "open", "write_text", "write_bytes",
                 "mkdir", "unlink"):
        assert name not in names, name


def test_the_route_is_mounted_outside_the_pipeline_router(app):
    """A deployment that unmounts or protects `pipeline.router` (SPEC_4 B2, the
    C3 class) keeps this view."""
    from src.api.routes import evaluations, pipeline

    assert evaluations.router is not pipeline.router
    assert all("evaluations" not in route.path for route in pipeline.router.routes)
    mounted = [route for route in app.app.routes if getattr(route, "path", None) == ROUTE]
    assert len(mounted) == 1 and mounted[0].methods == {"GET"}


def test_an_answer_whose_fields_contradict_its_state_is_refused():
    """The response model holds the route to its states, so a defect in the
    route is a 500, never a misleading answer."""
    from pydantic import ValidationError

    from src.api.routes.evaluations import LoadedModelEvaluations, _ledger_view

    good = dict(state="no_ledger", checkpoint_sha256="a" * 64, ledger=_ledger_view(),
                records=[], records_not_shown=0, unreadable_records=[])
    LoadedModelEvaluations(**good)
    item = {"index": 0, "sequence": None, "source_artifact_digest": None, "run_id": None,
            "reason": "n_ranked: missing"}
    for wrong in (
        {"state": "records"},
        {"state": "no_records"},
        {"state": "ledger_not_read"},
        {"state": "no_model"},
        {"checkpoint_sha256": None},
        {"ledger": None},
        {"records_not_shown": 1},
        {"ledger": _ledger_view(problem="evaluations.json is not a ledger")},
    ):
        with pytest.raises(ValidationError):
            LoadedModelEvaluations(**{**good, **wrong})
    # K8-6e.1: the count and the list never disagree, and a listed item makes
    # the state "records" (never "no_records"). HEAD: no field.
    listed = dict(good, state="records", ledger=_ledger_view(sha256="b" * 64))
    LoadedModelEvaluations(**{**listed, "records_not_shown": 1, "unreadable_records": [item]})
    for wrong in (
        {"records_not_shown": 1, "unreadable_records": []},
        {"records_not_shown": 0, "unreadable_records": [item]},
        {"records_not_shown": 1, "unreadable_records": [item], "state": "no_records"},
        {"records_not_shown": 1, "unreadable_records": [{**item, "reason": ""}]},
        {"records_not_shown": 1, "unreadable_records": [{**item, "reason": "x" * 301}]},
        {"records_not_shown": 1, "unreadable_records": [{**item, "index": -1}]},
        {"records_not_shown": 1, "unreadable_records": [{**item, "sequence": 0}]},
        {"records_not_shown": 1, "unreadable_records": [{**item, "run_id": "/srv/x"}]},
    ):
        with pytest.raises(ValidationError):
            LoadedModelEvaluations(**{**listed, **wrong})


# ---------------------------------------------------------------------------
# 8. Each record's D5 state (BACKLOG 36; the M5 review's P2-6)
# ---------------------------------------------------------------------------
def _registered_with_registry(served, tmp_path, monkeypatch):
    """One report of the served weights registered, its registry the
    deployment's (`SHEPHERD_REGISTRY`), as the service is configured."""
    _, _, checkpoint = served
    registry = tmp_path / "reports"
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    result = _register(_report_for(_sha(checkpoint)), checkpoint.parent, registry)
    (kept,) = registry.glob("versions/*/reports/*/report.json")
    return registry, kept, result.record


def _disagreeing(kept: Path, value) -> Path:
    """Another report under the same key, kept in the same store, whose MRR is
    `value`: a hand edit D5 admits can happen to plain files (NaN, null, text
    and an object included)."""
    from tests.fixtures.report_store import store_bytes, version_of

    report = json.loads(kept.read_bytes())
    report["authoritative_metrics"]["pooled_mrr"] = value
    data = json.dumps(report, indent=2, sort_keys=True, allow_nan=True).encode()
    return store_bytes(kept.parents[4], data, version=version_of(kept),
                       checkpoint_sha256=report["manifest"]["artifact_digests"]["checkpoint"])


def _operator_state(record):
    """The dashboard's state for the same record: the one implementation."""
    from src.api.services.record_states import CohortVersions, contest_states, key_text
    from src.evaluation.sidecar import record_key

    return contest_states([record], CohortVersions.listed())[key_text(record_key(record))]


def test_a_registry_whose_versions_cannot_be_listed_is_the_records_named_cause(
    app, served, tmp_path, monkeypatch
):
    """The verification of EB-4 (R-5): when the registry's ``versions/``
    cannot be listed at all, every record is "cannot be checked" and the
    operator is told that cause, from the listing's own refusal -- never that
    the record's version "is not one the server's registry lists", which names
    the wrong one. The route still shows the state without the words. The same
    registry listable is the control."""
    registry, kept, record = _registered_with_registry(served, tmp_path, monkeypatch)
    assert _operator_state(record)["state"] == "uncontested"
    original = Path.iterdir

    def iterdir(self):
        if self.name == "versions":
            raise PermissionError(13, "Permission denied", str(self))
        return original(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    state = _operator_state(record)

    assert state["state"] == "cannot_be_checked"
    (problem,) = state["problems"]
    assert problem.startswith("the registry's cohort versions cannot be listed: ")
    assert "cannot be listed (PermissionError" in problem
    assert "not one the server's registry lists" not in problem
    (view,) = _get(app)["records"]
    assert view["contest"]["state"] == "cannot_be_checked"
    assert str(tmp_path) not in json.dumps(view)


def test_a_contested_record_is_marked_with_what_differs_and_names_no_path(
    app, served, tmp_path, monkeypatch
):
    """The reviewer's counter-example: a record registered before another run
    under its key disagreed stays, marked contested, with the fields and the
    runs (ids, never a path), as the dashboard shows it."""
    registry, kept, record = _registered_with_registry(served, tmp_path, monkeypatch)
    other = _disagreeing(kept, 0.4321)

    body = _get(app)

    (view,) = body["records"]
    assert view["contest"]["state"] == "contested"
    assert view["contest"]["differing"] == ["pooled_mrr"]
    runs = {run["run"]: run["values"] for run in view["contest"]["runs"]}
    assert runs == {kept.parent.name: {"pooled_mrr": record["metrics"]["pooled_mrr"]},
                    other.parent.name: {"pooled_mrr": 0.4321}}
    assert str(tmp_path) not in json.dumps(body)
    operator = _operator_state(record)
    assert {k: operator[k] for k in ("state", "differing", "runs")} == view["contest"]


@pytest.mark.parametrize("value", [float("nan"), None, "0.5", {"k": 1}, 10**400, -10**400],
                         ids=["nan", "null", "text", "object", "huge_int", "huge_negative_int"])
def test_a_contested_state_is_never_downgraded_by_a_value_it_cannot_show(
    app, served, tmp_path, monkeypatch, value
):
    """The critic's break: a disagreeing report holding NaN, null, text or an
    object is still contested for the dashboard, so the route says contested
    too -- the value shown as null -- never "cannot be checked". The
    verification's break: an integer too large for a float (``10**400``), which
    `math.isfinite` cannot take, is null too, never a 500."""
    registry, kept, record = _registered_with_registry(served, tmp_path, monkeypatch)
    other = _disagreeing(kept, value)

    (view,) = _get(app)["records"]

    assert _operator_state(record)["state"] == "contested"
    assert view["contest"]["state"] == "contested"
    runs = {run["run"]: run["values"] for run in view["contest"]["runs"]}
    assert runs[other.parent.name] == {"pooled_mrr": None}


@pytest.mark.parametrize("break_run", [
    lambda run: run.__setitem__("run", None),
    lambda run: run.__setitem__("values", [0.4321]),
], ids=["run_without_id", "values_not_a_mapping"])
def test_a_contested_state_whose_details_cannot_be_shown_stays_contested(
    app, served, tmp_path, monkeypatch, break_run
):
    """Whatever fails in carrying a contested state's details -- the view's
    validation (a run with no id) or building them at all (values that are not
    a mapping) -- the state is shown contested without them, never a 500."""
    import src.api.services.record_states as record_states

    registry, kept, record = _registered_with_registry(served, tmp_path, monkeypatch)
    _disagreeing(kept, 0.4321)
    real = record_states.contest_states

    def broken_runs(*args, **kwargs):
        found = real(*args, **kwargs)
        for state in found.values():
            for run in state["runs"]:
                break_run(run)
        return found

    monkeypatch.setattr(record_states, "contest_states", broken_runs)
    (view,) = _get(app)["records"]
    assert view["contest"] == {"state": "contested", "differing": [], "runs": []}


@pytest.mark.parametrize("case", ["no_registry", "unreadable_report", "unlisted_version"])
def test_a_state_that_cannot_be_read_is_cannot_be_checked_never_uncontested(
    app, served, tmp_path, monkeypatch, case
):
    """No registry configured, or a version the server's registry does not
    list: "cannot_be_checked", with no server path in the answer, and never a
    500. Another run's report in the store that cannot be read is that run's
    own problem (K8-6r.2, C1): this record's group reads uncontested."""
    from tests.fixtures.report_store import store_bytes, version_of

    registry, kept, record = _registered_with_registry(served, tmp_path, monkeypatch)
    if case == "no_registry":
        monkeypatch.delenv("SHEPHERD_REGISTRY")
    elif case == "unreadable_report":
        store_bytes(registry, b"{damaged", version=version_of(kept),
                    checkpoint_sha256=record["checkpoint_digest"])
    else:
        other = tmp_path / "other-registry"
        other.mkdir()
        monkeypatch.setenv("SHEPHERD_REGISTRY", str(other))

    body = _get(app)

    (view,) = body["records"]
    expected = "uncontested" if case == "unreadable_report" else "cannot_be_checked"
    assert view["contest"] == {"state": expected, "differing": [], "runs": []}
    assert str(tmp_path) not in json.dumps(body)


def test_the_state_descriptions_say_their_meaning_changed_and_decide_nothing():
    """G9 (K8-6r.2, Decision 3): the group description's machine values stay
    (`uncontested`, `contested`, `cannot_be_checked`), and the API's own
    descriptions say they changed meaning and decide nothing; `run_id` and
    `historical_overlap` are described. HEAD: "the record stays, marked", no
    run field."""
    from src.api.routes.evaluations import RecordContest, RecordView

    state = RecordContest.model_fields["state"]
    assert set(state.annotation.__args__) == {"uncontested", "contested", "cannot_be_checked"}
    assert "since K8-6r.2 they only describe and decide nothing" in state.description
    assert "runs under these conditions differ" in state.description
    assert "stays, marked" not in state.description
    assert "not recorded" in RecordView.model_fields["run_id"].description
    assert "Not confirmed" in RecordView.model_fields["historical_overlap"].description


def test_reading_a_schema_3_ledger_shows_its_records_as_merged_and_leaves_its_bytes(
    app, served, tmp_path
):
    """A12 (a), G1 (K8-6r.2): the route reads a schema 3 ledger as written, each
    record shown with `run_id` null ("not recorded"), and never rewrites the
    file. HEAD: no `run_id` field (a guard on the bytes)."""
    from src.evaluation.sidecar import append_record, build_record, empty_ledger, write_ledger

    _, _, checkpoint = served
    record = build_record(_report_for(_sha(checkpoint)), "e" * 64)
    _, ledger = append_record(empty_ledger(), record,
                              run_id=f"20261016T120000Z-{_sha(checkpoint)[:12]}-00000001",
                              registered_at=WHEN)
    old = {**ledger, "schema_version": 3, "records": [
        {k: v for k, v in ledger["records"][0].items() if k != "run_id"}]}
    old.pop("run_identity_from_sequence")
    path = checkpoint.parent / "evaluations.json"
    write_ledger(path, old, None)
    before = path.read_bytes()

    body = _get(app)

    (view,) = body["records"]
    assert (view["run_id"], view["historical_overlap"], view["n_corroborating"]) == (None, [], 0)
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# 9. Records the shared check cannot read: listed by every reader, each with
#    its reason (K8-6e.1, R3; policy rev 15 and rev 16)
# ---------------------------------------------------------------------------
def _operator_entry(tmp_path, checkpoint, digest):
    """What Models' operator view and the Diagnosis card read for the served
    checkpoint, through the shared model list (`model_list.read_recorded`): the
    checkpoint placed in a workspace's run layout by a link, which the listing
    resolves to the directory of the ledger beside it."""
    from src.webui.components import model_list as ml

    link = tmp_path / "runs" / "gatv2" / "r1" / "checkpoints" / "best.pt"
    link.parent.mkdir(parents=True)
    link.symlink_to(checkpoint)
    item = {"architecture": "gatv2", "run": "r1", "filename": "best.pt", "sha256": digest,
            "problem": None}
    ledgers, contest = ml.read_recorded(tmp_path, [item])
    key = ml.item_key(item)
    return ledgers[key], contest, ml.recorded_tests(ledgers, contest)[key]


def _text(markup):
    import html
    import re

    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", markup)).split())


def test_a_mixed_ledger_lists_its_unreadable_record_the_same_way_in_every_reader(
    app, served, tmp_path, monkeypatch
):
    """The mixed ledger of B rev 6 §3.5 (iv) (R3): a Mode C, a Mode D and a
    disease record (sequences 1-3), then a disease record that lost
    ``n_ranked`` (sequence 4, its own report digest). The API shows records
    1, 2 and 3 and lists the fourth at index 3 with "#4", its digest, its run
    and "n_ranked: missing"; the request leaves the ledger's bytes as they
    were. Models' operator details (through the shared listing) and its
    read-only view (through this route) list the same item and never point to
    the server's log; the row and the Diagnosis card read "Recorded tests: 3 ·
    1 not readable". Registering a new disease, C and D report appends records
    5, 6 and 7, and the unreadable record stays as written at index 3. HEAD: no
    ``unreadable_records``, "the server's log says why", and the row counted
    4 recorded tests."""
    from src.webui.components import diagnosis_card
    from src.webui.components import models_page as mp
    from src.webui.components import renderers as r
    from tests.unit.test_evaluation_sidecar import mode_c_report, mode_d_report

    _, _, checkpoint = served
    digest = _sha(checkpoint)
    registry = tmp_path / "reports"
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    digests = _report_for(digest)["manifest"]["artifact_digests"]
    reports = [mode_c_report(artifact_digests=digests), mode_d_report(artifact_digests=digests),
               _report_for(digest), _report_for(digest, batch_size=8)]
    results = [_register(report, checkpoint.parent, registry) for report in reports]
    assert [result.outcome for result in results] == ["registered"] * 4
    ledger_path = checkpoint.parent / "evaluations.json"
    _edit_ledger(ledger_path, lambda ledger: ledger["records"][3].pop("n_ranked"))
    bad = json.loads(ledger_path.read_text())["records"][3]
    before = ledger_path.read_bytes()

    body = _get(app)

    assert ledger_path.read_bytes() == before, "reading never edits the ledger"
    assert body["state"] == "records"
    assert [v["registration"]["sequence"] for v in body["records"]] == [1, 2, 3]
    assert [v["mode"] for v in body["records"]] == ["C", "D", "disease"]
    assert body["records_not_shown"] == 1
    assert body["unreadable_records"] == [{
        "index": 3, "sequence": 4, "source_artifact_digest": bad["source_artifact_digest"],
        "run_id": bad["run_id"], "reason": "n_ranked: missing"}]

    # Models, operator: the shared listing classified the records once.
    entry, contest, recorded = _operator_entry(tmp_path, checkpoint, digest)
    assert entry["state"] == "records" and len(entry["records"]) == 3
    assert entry["unreadable"] == body["unreadable_records"]
    operator, count, _contested, unreadable = mp._records_for_item(entry, {"contest": contest})
    assert (count, unreadable) == (3, 1)
    # Models, read-only: the route's answer.
    readonly = mp.readonly_parts({"readonly": True, "answer": body, "served_sha": digest,
                                  "status_at": "now", "contest": {}})["records"]
    row = f"3 #4 {bad['source_artifact_digest'][:12]} n_ranked: missing"
    for shown in (operator, readonly):
        said = _text(shown)
        assert ("▲ 1 record for these weights cannot be read as a result. The ledger keeps it "
                "as written.") in said
        assert "Index Registration Report Reason " + row in said
        assert "server's log" not in said and "not shown:" not in said
        assert 'aria-label="Records that cannot be read"' in shown
        assert str(tmp_path) not in shown
    # The row and the Diagnosis card count results only, and name the rest.
    assert (recorded.count, recorded.unreadable) == (3, 1)
    assert r.recorded_tests_text(recorded).startswith("Recorded tests: 3 · 1 not readable")
    card = _text(diagnosis_card._recorded_html({"recorded": recorded._asdict()}, grey=False))
    assert card.startswith("Recorded tests: 3 · 1 not readable")

    # Appending goes on past it; the unreadable record stays as written.
    more = [_report_for(digest, batch_size=16), mode_c_report(artifact_digests=digests,
                                                              batch_size=16),
            mode_d_report(artifact_digests=digests, batch_size=2)]
    appended = [_register(report, checkpoint.parent, registry) for report in more]
    assert [result.record["registration"]["sequence"] for result in appended] == [5, 6, 7]
    assert json.loads(ledger_path.read_text())["records"][3] == bad
    after = _get(app)
    assert [v["registration"]["sequence"] for v in after["records"]] == [1, 2, 3, 5, 6, 7]
    assert after["unreadable_records"] == body["unreadable_records"]


def test_a_reason_names_the_field_never_its_value(app, served, tmp_path):
    """Reason safety (K8-6e.1): a device set to a server path of over 500
    characters with a lone surrogate reads "the regime's device is neither CUDA
    nor CPU", the path nowhere in the answer; a metric whose key holds a lone
    surrogate and runs long gives a reason of at most 300 characters that
    encodes as UTF-8; a run id set to a path is named as a field, not echoed.
    HEAD: the values were echoed into the server log and no item was listed."""
    _, _, checkpoint = served
    digest = _sha(checkpoint)
    for _ in range(3):
        _register(_report_for(digest), checkpoint.parent, tmp_path / "reports")
    path = str(tmp_path / ("secret-dir-" + "d" * 500) / "case-0042") + "\udc80"
    key = "pooled\udc80" + "k" * 400

    def edit(ledger):
        ledger["records"][0]["regime"]["device"] = path
        ledger["records"][1]["metrics"][key] = "0.5"
        ledger["records"][2]["run_id"] = path

    _edit_ledger(checkpoint.parent / "evaluations.json", edit)

    response = _client(app).get(ROUTE)

    assert response.status_code == 200, response.text
    body = response.json()
    first, second, third = body["unreadable_records"]
    assert first["reason"] == "the regime's device is neither CUDA nor CPU"
    # pydantic carries the key it refuses without the lone surrogate (as
    # replacement characters); either way the line is bounded and encodes.
    assert len(second["reason"]) == 300 and second["reason"].startswith("metrics.pooled")
    assert second["reason"].endswith("...") and "\udc80" not in second["reason"]
    second["reason"].encode("utf-8")
    assert third["reason"] == (
        "its run identity cannot be read: its run_id is not a run directory's name")
    assert third["run_id"] is None
    for form in ("secret-dir", "case-0042", str(tmp_path)):
        assert form not in response.text
    # The pages show the same items: Models' operator details (the shared
    # listing) and its read-only view (this answer) name no path either.
    from src.webui.components import models_page as mp

    entry, contest, _recorded = _operator_entry(tmp_path, checkpoint, digest)
    operator = mp._records_for_item(entry, {"contest": contest})[0]
    readonly = mp.readonly_parts({"readonly": True, "answer": body, "served_sha": digest,
                                  "status_at": "now", "contest": {}})["records"]
    for shown in (operator, readonly):
        assert "the regime&#x27;s device is neither CUDA nor CPU" in shown
        for form in ("secret-dir", "case-0042", str(tmp_path)):
            assert form not in shown


def test_an_unreadable_historical_record_is_listed_with_no_run(app, served, tmp_path):
    """(viii) (K8-6e.1 on K8-6r.2's record model): an unreadable record of a
    schema 3 file, written before each run had its own record, is listed with
    ``run_id`` null; it is never shown as a result. HEAD: no item."""
    from src.evaluation.sidecar import build_record, empty_ledger, write_ledger

    _, _, checkpoint = served
    record = build_record(_report_for(_sha(checkpoint)), "e" * 64)
    record.update(registration={"sequence": 1, "registered_at": WHEN},
                  corroborating_reports=[], metrics=[0.5])
    path = checkpoint.parent / "evaluations.json"
    write_ledger(path, {**empty_ledger(), "schema_version": 3, "records": [record]}, None)

    body = _get(app)

    assert body["records"] == []
    assert body["unreadable_records"] == [{
        "index": 0, "sequence": 1, "source_artifact_digest": "e" * 64, "run_id": None,
        "reason": "metrics: Input should be a valid dictionary"}]


def test_the_record_path_helpers_annotations_resolve():
    """P3 of the K8-6 completion review (2026-11-08). `_record_path` annotates
    `Sequence`, which the module did not import: annotations are deferred, so
    serving was unaffected, but resolving its type hints raised `NameError`.
    HEAD: `NameError: name 'Sequence' is not defined`."""
    import typing
    from collections.abc import Sequence

    from src.api.routes import evaluations

    hints = typing.get_type_hints(evaluations._record_path)
    assert typing.get_origin(hints["loc"]) is Sequence
