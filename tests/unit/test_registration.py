"""Registration: one implementation for `--record` and `record_evaluation.py`.

`src/evaluation/registration.register_report` turns a kept measurement report
into a record in the ledger beside its checkpoint (`PLAN_TEST_RESULTS.md` D5, §6
phase 1 step 2). What is pinned here:

  - **the acceptance cases at this level**: not registering writes nothing; the
    same run's same report again is a no-op that leaves the ledger's bytes as
    they were; a damaged report, a generated split's report, a schema 2 ledger
    and a ledger that is a JSON list are refused with one message that says
    nothing was written;
  - **since K8-6r.2 (policy rev 16), registration by the verified run**: every
    run is its own record, whatever its bytes or values (rows A1-A3, A6, A7,
    A9, A11, B2); the same run's same report is ``already_registered`` (A4); the
    same run's other bytes are refused, since a run's report is sealed (A5,
    A10); what keeps the write from completing safely refuses it (A8); the
    ledger is schema 4 from its first append (A12); another report of the store
    never blocks, and the selected report's own problems still refuse (C1-C5);
    `report_state` agrees with the writer;
  - **BACKLOG 18**: the report and the ledger are each parsed and identified from
    one read, so a report replaced while it is registered is recorded as the
    bytes read, never as one file's numbers under the other's digest; and the
    interleaving that erased a record (writer A reads, writer B appends, A
    writes) is refused by the stale-write guard when B takes no lock. The
    guard is a path hash at write time, not compare-and-swap: the second line;
  - **matrix F (K8-6r.1)**: one registration at a time per ledger, across
    threads and processes (`tests/fixtures/concurrent_writers.py`): two or
    four at once all persist with their own sequences, the same run resent at
    once is one record, a busy ledger is refused saying nothing was written
    and registers once free, a registration inside another's section is
    excluded, and a written ledger whose usage history is busy or fails says
    so and stands; the ledger's lock is taken before the usage history's;
  - **where the ledger is**: beside the checkpoint file's resolved path,
    whether the caller names the file (`--checkpoint`, any name), a directory
    searched by digest (`--checkpoint-dir`) or the measurement's own read; a
    linked checkpoint files beside the file the link names;
  - **the CLI**: the checkpoint must be beside the ledger, every refusal exits
    non-zero with one message and no traceback, and `--show` reads the ledger
    once;
  - **the standing text** a surface renders beside a record (`caveats.py`);
  - **since M5-1 (D5, 2026-10-19)**: only a report the registry's report store
    holds is registered, found through the registry's structure (a copy, a path
    off that structure, a link out of the store refused; a registry moved as a
    whole once `SHEPHERD_REGISTRY` names its new place, a link to a stored
    report accepted), naming its version exactly and
    kept in a run directory named for its checkpoint; a registration is in the
    version's usage history; `report_state` reads kept and registered from
    disk. (Rev 15's D5 block -- runs that disagree under one key, or a store
    that cannot be checked, refusing every registration under it -- is
    withdrawn from K8-6r.2.)

The reports are built by `MeasurementManifest` itself
(`tests/unit/test_evaluation_sidecar.registered_report`) and kept in a report
store by `tests/fixtures/report_store.py`.

Module: tests/unit/test_registration.py
"""
from __future__ import annotations

import hashlib
import json
import re
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation.registration import (
    MeasuredRun,
    RegistrationRefused,
    RegistrationResult,
    ReportNotAsPublished,
    group_note,
    historical_note,
    register_report,
)
from src.utils.fingerprint import ReadIdentity
from src.evaluation.sidecar import build_record, empty_ledger, read_ledger, write_ledger
from tests.fixtures.report_store import store_bytes, store_report, version_of
from tests.unit.test_evaluation_sidecar import (
    DISEASE_METRICS,
    generated_report,
    mode_d_report,
    r10_block,
    registered_manifest,
    registered_report,
)

REPO = Path(__file__).resolve().parents[2]
WHEN = "2026-10-16T12:00:00Z"
WEIGHTS = b"the weights"
DIGEST = hashlib.sha256(WEIGHTS).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _deployment_registry(tmp_path, monkeypatch):
    """``tmp_path/registry``, where `_report` keeps its reports, is the
    deployment's registry (`SHEPHERD_REGISTRY`): a report is registered only
    from it (BACKLOG 37)."""
    (tmp_path / "registry").mkdir(exist_ok=True)
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(tmp_path / "registry"))


@pytest.fixture
def checkpoints(tmp_path):
    """A run's checkpoint directory holding the weights the reports measured."""
    directory = tmp_path / "runs" / "shepherd" / "r1" / "checkpoints"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(WEIGHTS)
    return directory


def _report(tmp_path, **overrides) -> Path:
    """A report of a registered cohort version, measuring `WEIGHTS`, kept in a
    new run directory of its version's report store under ``tmp_path/registry``
    (`tests/fixtures/report_store.py`): since M5-1 only a report the store holds
    is registered. A disease task measurement's by default (mode ``disease``,
    metric schema 2: `registered_report`)."""
    report = registered_report(**overrides)
    report["manifest"]["artifact_digests"]["checkpoint"] = DIGEST
    return store_report(tmp_path / "registry", report)


def _measured(report_path, ledger_dir, *, checkpoint_sha256=DIGEST) -> MeasuredRun:
    """What `--record` passes: the run's read of `ledger_dir/best.pt` and the
    digest of the report it published (here, the report file as it is; a path
    that is not a file has no bytes, and the registration refuses it on reading
    before any digest is compared)."""
    published = _sha(report_path) if Path(report_path).is_file() else "0" * 64
    return MeasuredRun(ReadIdentity(Path(ledger_dir) / "best.pt", checkpoint_sha256),
                       published)


def _register(report_path, ledger_dir, *, beside=False, **kwargs) -> RegistrationResult:
    """Through the one registration function: as `--record` registers -- the
    run's own checkpoint read and published report, `ledger_dir/best.pt` not
    hashed again -- or, with `beside`, as `record_evaluation --checkpoint-dir`
    does, searching the directory by the report's digest."""
    kwargs.setdefault("registered_at", WHEN)
    if beside:
        return register_report(report_path, checkpoint_dir=ledger_dir, **kwargs)
    return register_report(report_path, measured=_measured(report_path, ledger_dir), **kwargs)


def _listing(directory: Path):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def _again_in_another_run(report: Path) -> Path:
    """The same report bytes kept in a second run directory of its version: a
    second run whose report is byte-identical (two CPU runs of one test)."""
    return store_bytes(Path(report).parents[4], Path(report).read_bytes(),
                       version=version_of(report), checkpoint_sha256=DIGEST)


def _historical(ledger_path: Path, report: Path, corroborating=(), *, sequence=1,
                registered_at=WHEN) -> dict:
    """A schema 3 ledger as the writer before K8-6r.2 left it, written at
    `ledger_path`: one record built from `report`, naming no run, with
    `corroborating` digests merged into it. Returns that record."""
    record = build_record(json.loads(Path(report).read_text()), _sha(report))
    record["corroborating_reports"] = list(corroborating)
    record["registration"] = {"sequence": sequence, "registered_at": registered_at}
    ledger = {**empty_ledger(), "schema_version": 3, "records": [record]}
    write_ledger(ledger_path, ledger, None)
    return record


# ---------------------------------------------------------------------------
# The acceptance cases
# ---------------------------------------------------------------------------
def test_a_report_is_registered_and_the_record_returned(tmp_path, checkpoints):
    report = _report(tmp_path)

    result = _register(report, checkpoints)

    assert result.outcome == "registered"
    assert result.ledger_path == checkpoints.resolve() / "evaluations.json"
    assert result.record["source_artifact_digest"] == _sha(report)
    assert result.record["registration"] == {"sequence": 1, "registered_at": WHEN}
    ledger, _ = read_ledger(result.ledger_path)
    assert ledger["records"] == [result.record]


def test_the_time_of_registration_is_now_unless_given(tmp_path, checkpoints):
    from datetime import datetime, timezone

    report = _report(tmp_path)
    result = register_report(report, measured=_measured(report, checkpoints))

    when = datetime.strptime(result.record["registration"]["registered_at"],
                             "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert abs((datetime.now(timezone.utc) - when).total_seconds()) < 60


def test_not_registering_writes_nothing(tmp_path, checkpoints, capsys):
    """A kept report that nobody registers leaves the checkpoint directory as it
    was, and reading what the ledger holds writes nothing either."""
    import scripts.record_evaluation as record_evaluation

    _report(tmp_path)
    before = _listing(checkpoints)

    record_evaluation.main(["--checkpoint-dir", str(checkpoints), "--show", DIGEST])

    assert json.loads(capsys.readouterr().out) == []
    assert _listing(checkpoints) == before


def test_the_same_report_again_is_a_no_op_and_the_ledger_bytes_stay(tmp_path, checkpoints):
    report = _report(tmp_path)
    first = _register(report, checkpoints)
    before = first.ledger_path.read_bytes()

    again = _register(report, checkpoints, registered_at="2027-01-01T00:00:00Z")

    assert again.outcome == "already_registered"
    assert again.record == first.record
    assert first.ledger_path.read_bytes() == before


def test_another_run_with_the_same_result_is_its_own_record_with_its_environment(
    tmp_path, checkpoints
):
    """A2 (K8-6r.2): byte-different (measured again after the checkpoint moved,
    under another driver, finding other R10 evidence), every compared field
    equal: a record of its own, with its own run, environment and R10; the
    first record keeps its own. Rev 15's merge into the first record
    (the old merge into the first record) is withdrawn. HEAD: merged, one record."""
    first = _register(_report(tmp_path), checkpoints)
    moved = registered_manifest()["environment"]
    moved["facts"]["gpu"]["driver"] = "999.0"
    other = _report(tmp_path, checkpoint_path="/moved/best.pt", data_dir="/moved",
                    environment=moved, r10=r10_block(training_state="overlap"))

    result = _register(other, checkpoints)

    assert result.outcome == "registered"
    assert result.record["run_id"] == other.parent.name != first.record["run_id"]
    assert result.record["source_artifact_digest"] == _sha(other)
    assert result.record["environment"] == moved
    assert result.record["r10"] == r10_block(training_state="overlap")
    assert result.record["corroborating_reports"] == []
    assert result.group_differences == () and group_note(result) is None
    ledger, _ = read_ledger(result.ledger_path)
    assert [r["registration"]["sequence"] for r in ledger["records"]] == [1, 2]
    assert ledger["records"][0] == first.record, "the first record keeps its own"


def test_two_runs_whose_values_differ_are_both_registered_and_described(
    tmp_path, checkpoints
):
    """A1 (K8-6r.2): two runs of one group, `pooled_mrr` 0.5 and 0.5001,
    deterministic algorithms off: two records (sequences 1 and 2), each with
    its own run, metrics, environment and R10; the second's group note names
    `pooled_mrr` and record 1, and asks no investigation. The first, sent
    again, answers ``already_registered`` and its note names record 2. HEAD:
    both refused as runs that disagree."""
    environment = registered_manifest()["environment"]
    environment["facts"]["gpu"]["driver"] = "999.0"
    one = _report(tmp_path, metrics={**DISEASE_METRICS, "pooled_mrr": 0.5},
                  deterministic_algorithms=False)
    two = _report(tmp_path, metrics={**DISEASE_METRICS, "pooled_mrr": 0.5001},
                  deterministic_algorithms=False, environment=environment,
                  r10=r10_block(training_state="overlap"))

    first = _register(one, checkpoints)
    second = _register(two, checkpoints)

    assert (first.outcome, second.outcome) == ("registered", "registered")
    ledger, _ = read_ledger(second.ledger_path)
    assert [(r["registration"]["sequence"], r["run_id"]) for r in ledger["records"]] == [
        (1, one.parent.name), (2, two.parent.name)]
    assert [r["metrics"]["pooled_mrr"] for r in ledger["records"]] == [0.5, 0.5001]
    assert ledger["records"][1]["environment"] == environment != ledger["records"][0][
        "environment"]
    assert ledger["records"][1]["r10"] == r10_block(training_state="overlap")
    (other,) = second.group_differences
    assert (other.sequence, other.run_id, other.historical, other.differing,
            other.deterministic) == (1, one.parent.name, False, ("pooled_mrr",), False)
    assert group_note(second) == (
        f"Runs under these conditions differ: record 2 (this run) differs from record 1 (run "
        f"{one.parent.name}) on pooled_mrr. Each run is kept as recorded.")
    again = _register(one, checkpoints)
    assert again.outcome == "already_registered" and again.record == first.record
    assert [d.sequence for d in again.group_differences] == [2]


# ---------------------------------------------------------------------------
# D5: runs that disagree under one key block every registration under it
# ---------------------------------------------------------------------------
_OTHER_METRICS = {**DISEASE_METRICS, "pooled_mrr": 0.9, "pooled_top1": 0.5}


def test_runs_that_differ_never_block_a_registration_through_any_entry_point(
    tmp_path, checkpoints
):
    """Policy rev 16 (G9 at this level): a registered run, then runs of its
    group with other results, through the run's own registration, a directory
    search and a named file: each registers as its own run; the first, sent
    again, is ``already_registered`` and its record stays as it was. HEAD:
    every one refused "runs disagree -- investigate"."""
    first_report = _report(tmp_path)
    first = _register(first_report, checkpoints)
    before = first.record

    results = [
        _register(_report(tmp_path, metrics=_OTHER_METRICS), checkpoints),
        _register(_report(tmp_path, metrics=_OTHER_METRICS), checkpoints, beside=True),
        register_report(_report(tmp_path, metrics=_OTHER_METRICS),
                        checkpoint=checkpoints / "best.pt", registered_at=WHEN),
    ]

    assert [(r.outcome, r.record["registration"]["sequence"]) for r in results] == [
        ("registered", 2), ("registered", 3), ("registered", 4)]
    assert all("pooled_mrr" in r.group_differences[0].differing for r in results)
    again = _register(first_report, checkpoints, beside=True)
    assert again.outcome == "already_registered" and again.record == before


def test_a_group_whose_runs_differ_never_affects_another_group(tmp_path, checkpoints):
    """B3 (guard): two runs of one group differ in the store, neither
    registered; a run of another group (another batch size is another
    measurement) registers as before."""
    _report(tmp_path), _report(tmp_path, metrics=_OTHER_METRICS)
    other_key = _report(tmp_path, batch_size=8)

    result = _register(other_key, checkpoints)

    assert result.outcome == "registered"
    assert len(read_ledger(result.ledger_path)[0]["records"]) == 1


def test_two_differing_runs_nothing_was_registered_under_both_register(tmp_path, checkpoints):
    """Policy rev 16: within one group, no run is chosen and none is refused:
    two runs that differ, neither registered, both register, each its own
    record. HEAD: both refused as runs that disagree, and nothing written."""
    one, two = _report(tmp_path), _report(tmp_path, metrics=_OTHER_METRICS)

    outcomes = [_register(report, checkpoints).outcome for report in (one, two)]

    assert outcomes == ["registered", "registered"]
    ledger, _ = read_ledger(checkpoints / "evaluations.json")
    assert [r["run_id"] for r in ledger["records"]] == [one.parent.name, two.parent.name]


def test_an_investigation_note_decides_nothing(tmp_path, checkpoints):
    """The note is kept in the store with the group's reports, never in the
    ledger; the group's differing runs register with it there, and it stays.
    HEAD: the second run refused as runs that disagree."""
    from src.evaluation.report_store import add_investigation_note, read_investigation_notes
    from src.evaluation.sidecar import record_key

    one, two = _report(tmp_path), _report(tmp_path, metrics=_OTHER_METRICS)
    key = record_key(build_record(json.loads(one.read_text()), _sha(one)))
    add_investigation_note(version_of(one), key, who="technician", what="driver differs")

    assert [_register(r, checkpoints).outcome for r in (one, two)] == ["registered"] * 2
    notes, problems = read_investigation_notes(version_of(one), key)
    assert [note["what"] for note in notes] == ["driver differs"] and problems == []
    assert "driver differs" not in (checkpoints / "evaluations.json").read_text()


def test_another_runs_broken_report_never_blocks_a_complete_run(tmp_path, checkpoints):
    """C1 (K8-6r.2, F2): another model's run directory holds broken JSON. The
    complete run registers, reading only its own report; the broken bytes are
    unchanged; the store lists the broken run with its own reason only, and the
    complete run's group reads uncontested. HEAD: refused, its disagreement not checked,
    naming the broken file."""
    from src.evaluation.report_store import scan_report_store
    from tests.fixtures.opens import count_opens

    report = _report(tmp_path)
    broken = store_bytes(tmp_path / "registry", b"{broken", version=version_of(report),
                         checkpoint_sha256="0" * 64)
    before = broken.read_bytes()

    with count_opens() as opens:
        result = _register(report, checkpoints)

    assert result.outcome == "registered"
    assert opens.opens(broken) == 0, "another report is never read to register this one"
    assert broken.read_bytes() == before
    scan = scan_report_store(version_of(report))
    assert scan.run_status(report.parent.name, _sha(report)).state == "uncontested"
    status = scan.run_status(broken.parent.name, None)
    assert status.state == "cannot_be_checked" and status.run_ids == (broken.parent.name,)
    (problem,) = status.problems
    assert str(broken) in problem and "does not parse" in problem


# ---------------------------------------------------------------------------
# Only a report the registry's report store holds (D5, 2026-10-19)
# ---------------------------------------------------------------------------
def test_a_copy_outside_the_report_store_is_refused_saying_how_to_register(
    tmp_path, checkpoints, capsys
):
    """A byte-identical copy of a stored report, anywhere else, is refused
    before it is read, saying to register the store's original; the original
    registers through the CLI as it is printed."""
    import scripts.record_evaluation as record_evaluation
    from tests.fixtures.opens import count_opens

    original = _report(tmp_path)
    copy = tmp_path / "shared" / "report.json"
    copy.parent.mkdir()
    copy.write_bytes(original.read_bytes())

    for named in ({"checkpoint_dir": checkpoints}, {"checkpoint": checkpoints / "best.pt"}):
        with count_opens() as opens, pytest.raises(RegistrationRefused) as refused:
            register_report(copy, registered_at=WHEN, **named)
        message = str(refused.value)
        assert f"{copy} is not a report the registry's report store holds" in message
        assert "Register the store's original instead" in message
        assert "Nothing was written" in message
        assert opens.opens(copy) == 0, "the copy was read"
    with pytest.raises(SystemExit, match="is not a report the registry's report store holds"):
        record_evaluation.main(["--checkpoint-dir", str(checkpoints), "--report", str(copy)])
    assert not (checkpoints / "evaluations.json").exists()

    assert record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                                   "--report", str(original)]) == 0


@pytest.mark.parametrize("where", ["not_report_json", "not_a_run_dir", "no_reports_dir",
                                   "no_versions_dir", "link_out_of_the_store"])
def test_a_path_that_is_not_the_stores_structure_is_refused(tmp_path, checkpoints, where):
    """Each step of `<registry>/versions/<label>/reports/<run id>/report.json`
    is the store's own structure, and a path missing any is refused, never
    guessed at. A `report.json` in the store that is a link to a file elsewhere
    is that file, which is not in the store."""
    original = _report(tmp_path)
    data = original.read_bytes()
    run_dir = original.parent
    path = {
        "not_report_json": run_dir / "report-copy.json",
        "not_a_run_dir": run_dir.parent / "backup" / "report.json",
        "no_reports_dir": run_dir.parents[1] / "kept" / run_dir.name / "report.json",
        "no_versions_dir": (tmp_path / "other-root" / "elsewhere" / run_dir.parents[1].name
                            / "reports" / run_dir.name / "report.json"),
        "link_out_of_the_store": None,
    }[where]
    if path is None:
        outside = tmp_path / "outside.json"
        outside.write_bytes(data)
        original.unlink()
        original.symlink_to(outside)
        path = original
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    with pytest.raises(RegistrationRefused, match="is not a report the registry's report "
                                                  "store holds"):
        _register(path, checkpoints, beside=True)
    assert not (checkpoints / "evaluations.json").exists()


def test_a_registry_moved_as_a_whole_registers_once_the_setting_names_its_new_place(
    tmp_path, checkpoints, monkeypatch
):
    """The registry is found from the report's path where it now is, and its
    version is the same version by digest; **it is the deployment's once
    `SHEPHERD_REGISTRY` names its new place** (BACKLOG 37). Before that it is
    refused, saying the setting resolves elsewhere, and nothing is written."""
    import shutil

    from src.evaluation.registration import NotFromTheDeploymentRegistry

    report = _report(tmp_path)
    moved = tmp_path / "elsewhere" / "registry"
    moved.parent.mkdir()
    shutil.move(str(tmp_path / "registry"), str(moved))
    (tmp_path / "registry").mkdir()               # the setting's old place, now empty
    relative = report.relative_to(tmp_path / "registry")

    with pytest.raises(NotFromTheDeploymentRegistry, match="is not the deployment's"):
        _register(moved / relative, checkpoints, beside=True)
    assert not (checkpoints / "evaluations.json").exists()

    monkeypatch.setenv("SHEPHERD_REGISTRY", str(moved))
    result = _register(moved / relative, checkpoints, beside=True)

    assert result.outcome == "registered"
    assert result.record["cohort_digest"] == version_of(moved / relative).identity.sha256


def test_a_link_to_a_stored_report_registers_its_original(tmp_path, checkpoints):
    """A link elsewhere that names a report in the store is that report: its
    resolved path is the store's original."""
    report = _report(tmp_path)
    link = tmp_path / "pointer.json"
    link.symlink_to(report)

    result = _register(link, checkpoints, beside=True)

    assert result.record["source_artifact_digest"] == _sha(report)


@pytest.mark.parametrize("edit", ["another_label_store", "edited_counts"])
def test_a_report_must_name_the_version_it_is_kept_under(tmp_path, checkpoints, edit):
    """The report's cohort block must be the block of the version holding it
    (`cohort_version_block`): a report moved into another version's store, or
    whose counts were edited (its manifest digest left as it was), is refused
    naming what differs."""
    from tests.unit.test_evaluation_sidecar import cohort_version

    report = _report(tmp_path)
    data = json.loads(report.read_text())
    if edit == "another_label_store":
        other = store_report(tmp_path / "registry", registered_report(
            cohort_version=cohort_version(label="mygene2-2026-11")))
        target = other.parent.parent / report.parent.name / "report.json"
        target.parent.mkdir()
        target.write_text(json.dumps(data))
        fragment = "differing in: label, manifest_sha256"
    else:
        data["manifest"]["cohort_version"]["counts"]["n_excluded"] += 1
        target = report
        target.write_text(json.dumps(data))
        fragment = "differing in: counts"

    with pytest.raises(RegistrationRefused, match=fragment) as refused:
        _register(target, checkpoints, beside=True)

    assert "registered only under the version it measured" in str(refused.value)
    assert not (checkpoints / "evaluations.json").exists()


def test_a_report_in_a_run_directory_named_for_other_weights_is_refused(
    tmp_path, checkpoints
):
    """A run directory is named for the checkpoint it measured
    (`create_report_dir`); a report moved into another checkpoint's run
    directory is not where the measurement kept it."""
    from tests.fixtures.report_store import new_run_id

    report = _report(tmp_path)
    moved = report.parents[1] / new_run_id("d" * 64) / "report.json"
    moved.parent.mkdir()
    report.rename(moved)

    with pytest.raises(RegistrationRefused, match="is named for dddddddddddd"):
        _register(moved, checkpoints, beside=True)
    assert not (checkpoints / "evaluations.json").exists()


def test_a_registration_is_appended_to_the_versions_usage_history(tmp_path, checkpoints):
    """D4 and G6 (K8-6r.2): each registration that wrote a ledger is in the
    version's usage history with its own run and sequence, whichever entry
    point made it; the same run's report again writes nothing there either. An
    entry written before K8-6r.2 with the retired outcome ``corroborated`` is
    read as written. HEAD: the second run was merged into sequence 1."""
    from src.evaluation.report_store import USAGE_FILE, _append_entry, read_usage_history

    report = _report(tmp_path)
    old = {"kind": "registered", "at": "2026-10-20T00:00:00Z", "run_id": "kept-as-written",
           "report_sha256": "b" * 64, "checkpoint_sha256": DIGEST, "outcome": "corroborated",
           "sequence": 1}
    _append_entry(version_of(report), USAGE_FILE, old)
    first = _register(report, checkpoints, beside=True)
    again = _register(report, checkpoints, beside=True)
    other_report = _report(tmp_path, data_dir="/elsewhere")
    other = _register(other_report, checkpoints, beside=True)

    entries, problems = read_usage_history(version_of(report))
    assert problems == []
    assert entries[0] == old, "an old entry is read as written"
    assert [(e["kind"], e["outcome"], e["sequence"], e["run_id"]) for e in entries[1:]] == [
        ("registered", "registered", 1, report.parent.name),
        ("registered", "registered", 2, other_report.parent.name)]
    assert entries[1]["report_sha256"] == _sha(report)
    assert entries[1]["checkpoint_sha256"] == DIGEST
    assert again.outcome == "already_registered"
    assert (first.usage_history_problem, again.usage_history_problem,
            other.usage_history_problem) == (None, None, None)


def test_a_failed_usage_history_append_is_reported_with_the_registration(
    tmp_path, checkpoints, monkeypatch, caplog
):
    """F-i (a), K8-6r.1: the ledger and the usage history are two files. The
    ledger written and its usage-history entry not: the outcome stays
    ``registered``, the record stands (never rolled back), and the result and
    the CLI say "the usage history was not written" with the reason."""
    import logging

    import scripts.record_evaluation as record_evaluation
    import src.evaluation.registration as registration

    def fail(*args, **kwargs):
        raise PermissionError(13, "Permission denied", "usage.jsonl")

    monkeypatch.setattr(registration, "record_registration", fail)
    report = _report(tmp_path)

    result = _register(report, checkpoints, beside=True)

    assert result.outcome == "registered"
    assert result.usage_history_problem.startswith("the usage history was not written: ")
    assert "Permission denied" in result.usage_history_problem
    assert f"The record in {result.ledger_path} stands" in result.usage_history_problem
    assert read_ledger(result.ledger_path)[0]["records"], "the registration stands"
    with caplog.at_level(logging.INFO, logger=record_evaluation.logger.name):
        assert record_evaluation.main(["--checkpoint-dir", str(checkpoints), "--report",
                                       str(_report(tmp_path, batch_size=8))]) == 0
    said = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Registered record 2 ") for m in said)
    assert any(m.startswith("NOTE: the usage history was not written: ") for m in said)
    assert len(read_ledger(result.ledger_path)[0]["records"]) == 2, "never rolled back"


@pytest.mark.parametrize(
    "data,fragment",
    [
        (b'{"manifest": {"mode": "C", "spl', "does not parse as UTF-8 JSON"),
        (b"[1, 2, 3]", "JSON list, not a measurement report"),
        (b'{"manifest": "\xff\xfe"}', "does not parse as UTF-8 JSON"),
        (b'{"hello": "world"}', "no manifest object"),
    ],
    ids=["truncated", "a-list", "not-utf8", "not-a-report"],
)
def test_a_damaged_report_is_refused_and_nothing_is_written(
    tmp_path, checkpoints, data, fragment
):
    report = store_bytes(tmp_path / "registry", data, checkpoint_sha256=DIGEST)
    before = _listing(checkpoints)

    with pytest.raises(RegistrationRefused, match=fragment) as refused:
        _register(report, checkpoints)

    assert str(report) in str(refused.value)
    assert "Nothing was written" in str(refused.value)
    assert _listing(checkpoints) == before


def test_a_missing_report_or_a_directory_is_refused(tmp_path, checkpoints):
    with pytest.raises(RegistrationRefused, match="does not exist.*Nothing was written"):
        _register(tmp_path / "absent.json", checkpoints)
    a_directory = _report(tmp_path).parents[1] / "20261016T120000Z-aaaaaaaaaaaa-00000000"
    (a_directory / "report.json").mkdir(parents=True)
    with pytest.raises(RegistrationRefused, match="could not be read.*Nothing was written"):
        _register(a_directory / "report.json", checkpoints)
    assert not (checkpoints / "evaluations.json").exists()


def test_a_generated_splits_report_is_refused(tmp_path, checkpoints):
    report = store_report(tmp_path / "registry", generated_report())

    with pytest.raises(RegistrationRefused, match="generated split's report is not registered"):
        _register(report, checkpoints)
    assert not (checkpoints / "evaluations.json").exists()


@pytest.mark.parametrize(
    "ledger,fragment",
    [
        ({"schema_version": 2, "records": []}, "schema 2.*evaluations.v2.json"),
        ([], "JSON list, not an object"),
    ],
    ids=["schema-2", "a-list"],
)
def test_a_ledger_this_revision_does_not_read_is_refused_naming_it(
    tmp_path, checkpoints, ledger, fragment
):
    path = checkpoints / "evaluations.json"
    path.write_text(json.dumps(ledger))
    before = path.read_bytes()

    with pytest.raises(RegistrationRefused, match=fragment) as refused:
        _register(_report(tmp_path), checkpoints)

    assert str(path) in str(refused.value)
    assert path.read_bytes() == before


def test_a_ledger_directory_that_does_not_exist_is_not_created(tmp_path):
    absent = tmp_path / "no-such-run" / "checkpoints"

    with pytest.raises(RegistrationRefused, match="is not a directory"):
        _register(_report(tmp_path), absent)
    assert not absent.exists() and not absent.parent.exists()


def test_a_report_of_other_weights_is_refused_when_they_must_be_beside(tmp_path, checkpoints):
    (checkpoints / "best.pt").write_bytes(b"other weights")

    with pytest.raises(RegistrationRefused, match="holds no checkpoint whose bytes"):
        _register(_report(tmp_path), checkpoints, beside=True)
    assert not (checkpoints / "evaluations.json").exists()


def test_the_measurements_own_registration_does_not_rehash_the_directory(
    tmp_path, checkpoints, monkeypatch
):
    """`--record` trusts the checkpoint read the run just made (spec §0.15)."""
    import src.evaluation.registration as registration

    def forbidden(*args, **kwargs):
        raise AssertionError("the checkpoints were hashed again")

    monkeypatch.setattr(registration, "find_checkpoint", forbidden)
    monkeypatch.setattr(registration, "checkpoint_matches", forbidden)

    assert _register(_report(tmp_path), checkpoints).outcome == "registered"


def test_a_ledger_that_cannot_be_written_is_a_refusal(tmp_path, checkpoints, monkeypatch):
    """A read-only checkpoint directory fails the registration, not with a
    traceback; the report stays where it was."""
    import src.evaluation.sidecar as sidecar

    def readonly(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(checkpoints))

    monkeypatch.setattr(sidecar, "create_staging_file", readonly)
    report = _report(tmp_path)

    with pytest.raises(RegistrationRefused, match="Permission denied.*Nothing was written"):
        _register(report, checkpoints)
    assert report.exists()
    assert not (checkpoints / "evaluations.json").exists()


def _no_create_in(directory: Path, monkeypatch) -> None:
    """A directory this account cannot write, as the kernel refuses it: every
    open that would create a file there fails with EACCES. (Tests may run as
    root, which a directory's mode does not stop.)"""
    real_open = os.open

    def guarded(path, flags, *args, **kwargs):
        if flags & os.O_CREAT and Path(os.fsdecode(path)).parent == directory:
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded)


def test_a_checkpoint_directory_this_account_cannot_write_says_permission_denied(
    tmp_path, checkpoints, monkeypatch
):
    """K8-6r.1 review: the ledger's lock file cannot be created there. The
    refusal names that cause (``Permission denied``), never a missing lock
    file; nothing is written and the report stays."""
    report = _report(tmp_path)
    _no_create_in(checkpoints, monkeypatch)

    with pytest.raises(RegistrationRefused) as refused:
        _register(report, checkpoints)
    said = str(refused.value)
    assert (f"the ledger's lock {checkpoints / 'evaluations.json.lock'} could not be opened or "
            "taken (PermissionError: [Errno 13] Permission denied") in said
    assert "No such file" not in said and said.endswith("Nothing was written to "
                                                         f"{checkpoints / 'evaluations.json'}.")
    assert report.exists()
    assert sorted(p.name for p in checkpoints.iterdir()) == ["best.pt"]


def test_a_held_report_from_a_directory_this_account_cannot_write_is_already_registered(
    tmp_path, checkpoints, monkeypatch
):
    """K8-6r.1 review: a ledger written before the lock existed (no lock file
    beside it), in a directory this account cannot write. Whether it holds the
    report is a read, which takes no lock, so the answer stays
    ``already_registered`` with nothing written. Another report, which would
    need a write, is refused naming the cause."""
    report = _report(tmp_path)
    assert _register(report, checkpoints).outcome == "registered"
    (checkpoints / "evaluations.json.lock").unlink()
    before = _listing(checkpoints)
    _no_create_in(checkpoints, monkeypatch)

    result = _register(report, checkpoints)

    assert (result.outcome, result.record["registration"]["sequence"]) == ("already_registered", 1)
    with pytest.raises(RegistrationRefused, match="Permission denied.*Nothing was written"):
        _register(_report(tmp_path, batch_size=8), checkpoints)
    assert _listing(checkpoints) == before


# ---------------------------------------------------------------------------
# BACKLOG 18: one read each, and the interleaving that erased a record
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["rename", "rewrite"])
def test_a_report_replaced_after_its_read_is_recorded_as_the_bytes_read(
    tmp_path, checkpoints, monkeypatch, how
):
    """Report A is read; B replaces it at once. The record carries A's numbers
    under A's digest -- never B's digest beside A's data."""
    from tests.fixtures.replacement import replace_after_read

    report = _report(tmp_path)
    bytes_a = report.read_bytes()
    b = registered_report(metrics={**DISEASE_METRICS, "pooled_mrr": 0.1, "pooled_top1": 0.0,
                                   "pooled_top10": 0.1})
    b["manifest"]["artifact_digests"]["checkpoint"] = DIGEST
    # B names the version it is kept under, as a measurement's report would.
    b["manifest"]["cohort_version"]["manifest_sha256"] = version_of(report).identity.sha256
    bytes_b = json.dumps(b).encode("utf-8")
    replaced = replace_after_read(monkeypatch, after=report, target=report, data=bytes_b,
                                  how=how)

    result = _register(report, checkpoints)

    assert replaced.fired == 1
    assert report.read_bytes() == bytes_b, "the replacement happened"
    assert result.record["source_artifact_digest"] == hashlib.sha256(bytes_a).hexdigest()
    assert result.record["metrics"] == json.loads(bytes_a)["authoritative_metrics"]
    # B sent now as this run is other bytes than its record holds: refused,
    # since a run's report is sealed when it is published (A5).
    with pytest.raises(RegistrationRefused, match="a run's report is sealed when it is published"):
        _register(report, checkpoints)


def test_the_report_and_the_ledger_are_each_read_once(tmp_path, checkpoints):
    """The ledger is opened once to be parsed and identified; a write adds the
    stale-write guard's hash by path, which compares and identifies nothing
    parsed."""
    from tests.fixtures.opens import count_opens

    report = _report(tmp_path)
    ledger = checkpoints / "evaluations.json"
    write_ledger(ledger, empty_ledger(), None)

    measured = _measured(report, checkpoints)  # the run's own digest, taken before
    with count_opens() as opens:
        assert register_report(report, measured=measured,
                               registered_at=WHEN).outcome == "registered"
    assert opens.opens(report) == 1
    assert opens.opens(ledger) == 2, "the read, and the write guard"

    with count_opens() as opens:
        assert register_report(report, measured=measured,
                               registered_at=WHEN).outcome == "already_registered"
    assert opens.opens(report) == 1
    assert opens.opens(ledger) == 1, "nothing written, so no guard"


def test_neither_raw_buffer_outlives_its_parse(tmp_path, checkpoints, monkeypatch):
    """The readers' release check (contract M2.1, S2): no read's bytes outlive
    its parse. Every read goes through `read_once`, in this order: the version's
    `cohort.json` (the report store is located through it, since M5-1), the
    report, and the ledger; the store holds no other report to compare."""
    from tests.fixtures.release import check_release

    report = _report(tmp_path)
    write_ledger(checkpoints / "evaluations.json", empty_ledger(), None)

    result, reads = check_release(monkeypatch, lambda: _register(report, checkpoints))

    assert result.outcome == "registered"
    assert [p.name for p in reads] == ["cohort.json", "report.json", "evaluations.json"]


def test_no_path_hash_reaches_the_record(tmp_path, checkpoints, monkeypatch):
    """Under poisoned path hashes the record still names the report by the
    digest of the bytes parsed. Only the write guard hashes a path, so only the
    ledger keeps its true digest."""
    from tests.fixtures.path_forms import POISON, poison_path_forms

    report = _report(tmp_path)
    poison_path_forms(monkeypatch, true_digest_for=[checkpoints / "evaluations.json"])

    first = _register(report, checkpoints)
    again = _register(report, checkpoints)

    assert first.record["source_artifact_digest"] == _sha(report) != POISON
    assert again.outcome == "already_registered"


def test_another_writers_append_between_this_read_and_this_write_is_refused(
    tmp_path, checkpoints, monkeypatch
):
    """F-f (K8-6r.1): the interleaving that erased a record (map, BACKLOG 18),
    writer A reads the ledger, writer B appends, A writes, in three parts.

    **The control.** With A's digest taken by path after B's write -- the old
    order, the ledger parsed and then hashed again -- A's write passes the
    guard and B's record is gone. **B through `register_report`** inside A's
    section is now excluded by the ledger's lock: B waits out its (patched,
    short) grace and is refused as busy, nothing written; A registers; B's
    report is kept and registers afterwards with the next sequence. **B taking
    no lock** (a raw `write_ledger` between A's read and A's write, as a hand
    edit or an older build would) is caught by the stale-write guard, the
    second line: A is refused "changed since it was read" and B's write stays.
    """
    import src.evaluation.sidecar as sidecar
    from src.evaluation.registration import LedgerBusy
    from src.evaluation.sidecar import append_record, ledger_digest
    from tests.fixtures.replacement import hook_read_once

    ledger_path = checkpoints / "evaluations.json"
    seed = _report(tmp_path)
    report_a = _report(tmp_path, batch_size=8)
    report_b = _report(tmp_path, batch_size=16)
    _register(seed, checkpoints)
    seeded = ledger_path.read_bytes()

    def batches():
        return [r["regime"]["batch_size"] for r in read_ledger(ledger_path)[0]["records"]]

    # The control: the old order erases B's append.
    parsed, _ = read_ledger(ledger_path)
    _register(report_b, checkpoints)
    record_a = build_record(json.loads(report_a.read_text()), _sha(report_a))
    _, with_a = append_record(parsed, record_a, run_id=report_a.parent.name, registered_at=WHEN)
    write_ledger(ledger_path, with_a, expected_digest=ledger_digest(ledger_path))
    assert batches() == [32, 8], "the old order erased B's record"

    # B through register_report, inside A's section: excluded, then registered.
    ledger_path.write_bytes(seeded)
    b_bytes = report_b.read_bytes()
    told = []

    def writer_b(path):
        if told or os.path.realpath(path) != os.path.realpath(ledger_path):
            return
        try:
            told.append(_register(report_b, checkpoints))
        except LedgerBusy as exc:
            told.append(exc)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sidecar, "LEDGER_LOCK_GRACE_SECONDS", 0.2)
        hook_read_once(patch, on_return=writer_b)
        first = _register(report_a, checkpoints)

    (busy,) = told
    assert isinstance(busy, LedgerBusy), "B ran inside A's section and was excluded"
    assert "the ledger is busy" in str(busy) and "register it again" in str(busy)
    assert f"Nothing was written to {ledger_path} this time." in str(busy)
    assert (first.outcome, first.record["registration"]["sequence"]) == ("registered", 2)
    assert batches() == [32, 8]
    assert report_b.read_bytes() == b_bytes, "B's report is kept"
    later = _register(report_b, checkpoints)
    assert (later.outcome, later.record["registration"]["sequence"]) == ("registered", 3)
    assert batches() == [32, 8, 16]

    # B taking no lock: the stale-write guard refuses A, and B's write stays.
    ledger_path.write_bytes(seeded)
    fired = []

    def raw_writer_b(path):
        if fired or os.path.realpath(path) != os.path.realpath(ledger_path):
            return
        fired.append(path)
        current, digest = read_ledger(ledger_path)
        record_b = build_record(json.loads(report_b.read_text()), _sha(report_b))
        _, with_b = append_record(current, record_b, run_id=report_b.parent.name,
                                  registered_at=WHEN)
        write_ledger(ledger_path, with_b, expected_digest=digest)

    hook_read_once(monkeypatch, on_return=raw_writer_b)
    with pytest.raises(RegistrationRefused, match="changed since it was read") as refused:
        _register(report_a, checkpoints)

    assert fired, "writer B ran between A's read and A's write"
    assert "Nothing was written" in str(refused.value)
    assert batches() == [32, 16], "B's record stays and A's is not written"


# ---------------------------------------------------------------------------
# Matrix F (K8-6r.1): registrations at once, across threads and processes
# ---------------------------------------------------------------------------
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="flock and spawn are tested on POSIX")


def _hold_after_the_guard(monkeypatch, parties: int):
    """Hold each registration right after `write_ledger`'s guard reads the
    ledger's digest -- between the check and the replace -- until ``parties``
    are there, or 1 s passes (then it goes on alone)."""
    import threading

    import src.evaluation.sidecar as sidecar

    barrier = threading.Barrier(parties)
    real = sidecar.ledger_digest

    def ledger_digest(path):
        digest = real(path)
        try:
            barrier.wait(1.0)
        except threading.BrokenBarrierError:
            pass
        return digest

    monkeypatch.setattr(sidecar, "ledger_digest", ledger_digest)


def _held(ledger_path: Path) -> dict:
    """What the ledger holds: each record's sequence by its report's digest."""
    return {r["source_artifact_digest"]: r["registration"]["sequence"]
            for r in read_ledger(ledger_path)[0]["records"]}


def test_two_threads_registering_two_runs_at_once_both_persist(
    tmp_path, checkpoints, monkeypatch
):
    """F-a: two threads, two different runs, held together right after the
    guard's real read. Both persist with sequences 1 and 2, each returned
    sequence is the one the ledger holds for that report, and both reports are
    unchanged. (Before K8-6r.1 both answered ``registered`` as record 1 and the
    ledger kept one record.)"""
    import threading

    reports = [_report(tmp_path, batch_size=8), _report(tmp_path, batch_size=16)]
    before = [r.read_bytes() for r in reports]
    _hold_after_the_guard(monkeypatch, 2)
    results: dict = {}

    def register(report):
        try:
            results[report] = _register(report, checkpoints)
        except Exception as exc:  # noqa: BLE001 - reported by the assertion below
            results[report] = exc

    threads = [threading.Thread(target=register, args=(r,)) for r in reports]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "a registration hung"

    held = _held(checkpoints / "evaluations.json")
    answered = {_sha(r): (results[r].outcome, results[r].record["registration"]["sequence"])
                for r in reports}
    assert sorted(answered.values()) == [("registered", 1), ("registered", 2)]
    assert held == {digest: sequence for digest, (_o, sequence) in answered.items()}
    assert [r.read_bytes() for r in reports] == before


def _register_at_once(tmp_path, checkpoints, reports, *, held_after_guard=True):
    """Each report registered by a process of its own into the ledger beside
    ``checkpoints``, all starting together; with ``held_after_guard``, each is
    held right after the guard's read until the others arrive or 1 s passes."""
    from tests.fixtures.concurrent_writers import CONTEXT, answers_from, processes, register

    n = len(reports)
    start = CONTEXT.Barrier(n)
    barrier = CONTEXT.Barrier(n) if held_after_guard else None
    queue = CONTEXT.Queue()
    with processes(*[(register, (f"P{i}", str(tmp_path / "registry"), str(report),
                                 str(checkpoints), start, barrier, queue))
                     for i, report in enumerate(reports, start=1)]):
        answers = answers_from(queue, n)
    return sorted(answers, key=lambda answer: answer["name"])


def _every_success_persists(answers, ledger_path):
    """Every ``registered`` answer is in the ledger with its own sequence and
    report; a refusal is only a busy ledger that says nothing was written."""
    held = _held(ledger_path)
    registered = [a for a in answers if a["outcome"] == "registered"]
    for answer in registered:
        assert held.get(answer["source"]) == answer["sequence"], answer
    assert len(held) == len(registered) == len({a["sequence"] for a in registered})
    for answer in answers:
        if answer["outcome"] == "refused":
            assert answer["kind"] == "LedgerBusy" and "Nothing was written" in answer["message"]
    return sorted(a["sequence"] for a in registered)


@POSIX
def test_two_processes_registering_two_runs_at_once_both_persist(tmp_path, checkpoints):
    """F-b: two separate processes (one ``SHEPHERD_REGISTRY``), two different
    runs, held together right after the guard's read: every success persists
    with its own sequence and report. (Before K8-6r.1 both answered
    ``registered`` as record 1 and the ledger kept one record.)"""
    reports = [_report(tmp_path, batch_size=8), _report(tmp_path, batch_size=16)]

    answers = _register_at_once(tmp_path, checkpoints, reports)

    assert _every_success_persists(answers, checkpoints / "evaluations.json") == [1, 2]
    assert {a["source"] for a in answers} == {_sha(r) for r in reports}


@POSIX
def test_four_processes_registering_at_once_get_sequences_one_to_four(tmp_path, checkpoints):
    """F-b's four-process variant: sequences 1 to 4, each in the ledger with
    its own report."""
    reports = [_report(tmp_path, batch_size=size) for size in (8, 16, 24, 40)]

    answers = _register_at_once(tmp_path, checkpoints, reports)

    assert _every_success_persists(answers, checkpoints / "evaluations.json") == [1, 2, 3, 4]


@POSIX
def test_the_same_run_resent_at_once_by_two_processes_is_one_record(tmp_path, checkpoints):
    """F-c: one report registered by two processes at once: one answers
    ``registered``, the other ``already_registered``, and the ledger holds one
    record. (Before K8-6r.1 both answered ``registered``.)"""
    report = _report(tmp_path)

    answers = _register_at_once(tmp_path, checkpoints, [report, report])

    assert sorted(a["outcome"] for a in answers) == ["already_registered", "registered"]
    assert {(a["source"], a["sequence"]) for a in answers} == {(_sha(report), 1)}
    assert _held(checkpoints / "evaluations.json") == {_sha(report): 1}


@POSIX
def test_a_busy_ledger_is_refused_explicitly_and_registers_once_free(
    tmp_path, checkpoints, monkeypatch
):
    """F-d: another process holds ``evaluations.json.lock`` past the (patched)
    grace. The registration is refused as `LedgerBusy`, a `RegistrationRefused`
    saying the ledger is busy, nothing was written this time, the report is kept
    and it may be registered again; no ledger appears and the report's bytes are
    unchanged. Once the holder lets go, the same call registers."""
    import src.evaluation.sidecar as sidecar
    from src.evaluation.registration import LedgerBusy
    from src.evaluation.sidecar import ledger_lock_path
    from tests.fixtures.concurrent_writers import CONTEXT, WAIT, hold_lock, processes

    monkeypatch.setattr(sidecar, "LEDGER_LOCK_GRACE_SECONDS", 0.3)
    ledger_path = checkpoints / "evaluations.json"
    report = _report(tmp_path)
    before = report.read_bytes()
    held, release = CONTEXT.Event(), CONTEXT.Event()
    with processes((hold_lock, (str(ledger_lock_path(ledger_path)), held, release))) as (holder,):
        assert held.wait(WAIT), "the holder did not take the ledger's lock"
        with pytest.raises(LedgerBusy) as busy:
            _register(report, checkpoints, beside=True)
        release.set()
        holder.join(WAIT)

    said = str(busy.value)
    assert isinstance(busy.value, RegistrationRefused)
    assert f"{report} is not registered: the ledger is busy" in said
    assert "for the 0.3 s this one waited" in said
    assert "The report stays where it is kept; register it again." in said
    assert said.endswith(f"Nothing was written to {ledger_path} this time.")
    assert not ledger_path.exists() and report.read_bytes() == before
    assert sorted(p.name for p in checkpoints.iterdir()) == [
        "best.pt", "evaluations.json.lock"]

    again = _register(report, checkpoints, beside=True)
    assert (again.outcome, again.record["registration"]["sequence"]) == ("registered", 1)
    assert ledger_path.with_name("evaluations.json.lock").exists(), "the lock file is kept"


@POSIX
def test_a_written_ledger_with_a_busy_usage_history_says_so_and_stands(
    tmp_path, checkpoints, monkeypatch, caplog
):
    """F-i (b): the version's usage-history lock is held by another process
    past its (patched) grace. The ledger is written and the outcome stays
    ``registered``; `usage_history_problem` says the usage history was not
    written, naming the busy lock; the CLI prints it and exits 0; the ledger is
    never rolled back and the usage history holds no entry of these."""
    import logging

    import scripts.record_evaluation as record_evaluation
    import src.evaluation.report_store as report_store
    from tests.fixtures.concurrent_writers import CONTEXT, WAIT, hold_lock, processes

    monkeypatch.setattr(report_store, "APPEND_LOCK_GRACE_SECONDS", 0.3)
    report = _report(tmp_path)
    other = _report(tmp_path, batch_size=8)
    usage_lock = report.parents[1] / "usage.jsonl.lock"
    held, release = CONTEXT.Event(), CONTEXT.Event()
    with processes((hold_lock, (str(usage_lock), held, release))) as (holder,):
        assert held.wait(WAIT), "the holder did not take the usage history's lock"
        result = _register(report, checkpoints, beside=True)
        with caplog.at_level(logging.INFO, logger=record_evaluation.logger.name):
            code = record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                                           "--report", str(other)])
        release.set()
        holder.join(WAIT)

    assert (result.outcome, result.record["registration"]["sequence"]) == ("registered", 1)
    problem = result.usage_history_problem
    assert problem.startswith(f"the usage history was not written: it was busy (another writer "
                              f"held {usage_lock} for the 0.3 s this one waited). ")
    assert f"The record in {result.ledger_path} stands" in problem
    assert code == 0
    said = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("NOTE: the usage history was not written: it was busy (")
               for m in said)
    assert len(read_ledger(result.ledger_path)[0]["records"]) == 2, "never rolled back"
    assert not (report.parents[1] / "usage.jsonl").exists()


@POSIX
def test_the_ledger_lock_is_taken_before_the_usage_lock_never_after(
    tmp_path, checkpoints, monkeypatch
):
    """F-j: lock order GPU, ledger, usage history, never the other way, and a
    registration on its own takes no GPU lock. With the version's usage lock
    held by another process, a registration takes the ledger's lock, writes the
    ledger, then waits for the usage lock inside it and reports F-i (b); it
    never deadlocks, and the order of acquisition is recorded through the
    shared primitive as each caller holds it."""
    from contextlib import contextmanager

    import src.evaluation.report_store as report_store
    import src.evaluation.sidecar as sidecar
    from src.utils import gpu_lock
    from tests.fixtures.concurrent_writers import CONTEXT, WAIT, hold_lock, processes

    monkeypatch.setattr(report_store, "APPEND_LOCK_GRACE_SECONDS", 0.3)
    ledger_path = checkpoints / "evaluations.json"
    events = []

    def recording(module, name):
        real = module.exclusive_lock

        @contextmanager
        def lock(path, **kwargs):
            events.append(("try", name, ledger_path.exists()))
            with real(path, **kwargs):
                events.append(("held", name, ledger_path.exists()))
                try:
                    yield path
                finally:
                    events.append(("left", name, ledger_path.exists()))

        monkeypatch.setattr(module, "exclusive_lock", lock)

    recording(sidecar, "ledger")
    recording(report_store, "usage")
    monkeypatch.setattr(gpu_lock.GpuLock, "acquire",
                        lambda *a, **k: pytest.fail("a registration took the GPU lock"))
    report = _report(tmp_path)
    usage_lock = report.parents[1] / "usage.jsonl.lock"
    held, release = CONTEXT.Event(), CONTEXT.Event()
    with processes((hold_lock, (str(usage_lock), held, release))) as (holder,):
        assert held.wait(WAIT)
        result = _register(report, checkpoints, beside=True)
        release.set()
        holder.join(WAIT)

    assert events == [("try", "ledger", False), ("held", "ledger", False),
                      ("try", "usage", True), ("left", "ledger", True)]
    assert result.outcome == "registered" and "it was busy" in result.usage_history_problem
    assert _held(ledger_path) == {_sha(report): 1}


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------
def test_the_cli_requires_the_weights_beside_the_ledger(tmp_path):
    """A record filed in a directory that does not hold those weights is filed
    under the wrong address, and the reader's reason for looking there is gone."""
    import scripts.record_evaluation as record_evaluation

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(SystemExit, match="holds no checkpoint whose bytes"):
        record_evaluation.main(["--checkpoint-dir", str(elsewhere),
                                "--report", str(_report(tmp_path))])
    assert not (elsewhere / "evaluations.json").exists()


def test_duplicate_checkpoint_bytes_are_acceptable(tmp_path, checkpoints):
    """The digest is the identity, not the name."""
    import scripts.record_evaluation as record_evaluation

    (checkpoints / "epoch-3.pt").write_bytes(WEIGHTS)

    assert record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                                   "--report", str(_report(tmp_path))]) == 0
    assert (checkpoints / "evaluations.json").exists()


# ---------------------------------------------------------------------------
# Where the ledger is: beside the checkpoint file's resolved path (decision 15)
# ---------------------------------------------------------------------------
@pytest.fixture
def linked(tmp_path, checkpoints):
    """`served/best.pt`, a link to the run's checkpoint, in a directory that
    holds no real weights: a deployment's view of them."""
    served = tmp_path / "served"
    served.mkdir()
    (served / "best.pt").symlink_to(checkpoints / "best.pt")
    return served


@pytest.mark.parametrize("named", ["checkpoint_dir", "checkpoint", "measured"])
def test_a_linked_checkpoint_files_its_ledger_beside_the_file_the_link_names(
    tmp_path, checkpoints, linked, named
):
    """Findings 0/2/13/24/25: whether the link's directory, the link, or a
    measurement's read of it is named, the ledger is the one the read-only view
    reads for a model served through the link -- the directory of its resolved
    path -- never a second ledger beside the link."""
    report = _report(tmp_path)
    where = {"checkpoint_dir": {"checkpoint_dir": linked},
             "checkpoint": {"checkpoint": linked / "best.pt"},
             # A run that read the link resolves it when it reads (M4 decision 15).
             "measured": {"measured": _measured(report, (linked / "best.pt").resolve().parent)},
             }[named]

    result = register_report(report, registered_at=WHEN, **where)

    assert result.outcome == "registered"
    assert result.ledger_path == checkpoints.resolve() / "evaluations.json"
    assert result.ledger_path.is_file()
    assert not (linked / "evaluations.json").exists()


@pytest.mark.parametrize("flag", ["--checkpoint-dir", "--checkpoint"])
def test_the_cli_files_a_linked_checkpoints_ledger_beside_its_target(
    tmp_path, checkpoints, linked, capsys, flag
):
    """The same through the CLI, and `--show --checkpoint <link>` reads that
    ledger, the one the service reads for the link."""
    import scripts.record_evaluation as record_evaluation

    named = linked if flag == "--checkpoint-dir" else linked / "best.pt"
    assert record_evaluation.main([flag, str(named), "--report", str(_report(tmp_path))]) == 0

    assert (checkpoints / "evaluations.json").is_file()
    assert not (linked / "evaluations.json").exists()
    capsys.readouterr()
    assert record_evaluation.main(["--checkpoint", str(linked / "best.pt"),
                                   "--show", DIGEST]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1


def test_the_cli_registers_against_a_named_file_whatever_its_name(tmp_path, checkpoints):
    """Finding 14: `--checkpoint` hashes that one file against the report, so a
    checkpoint not named `*.pt` -- which a directory search does not consider --
    is registered, beside itself."""
    import scripts.record_evaluation as record_evaluation

    renamed = checkpoints / "best.ckpt"
    (checkpoints / "best.pt").rename(renamed)
    report = _report(tmp_path)
    with pytest.raises(SystemExit, match="holds no checkpoint whose bytes"):
        record_evaluation.main(["--checkpoint-dir", str(checkpoints), "--report", str(report)])

    assert record_evaluation.main(["--checkpoint", str(renamed), "--report", str(report)]) == 0

    ledger, _ = read_ledger(checkpoints / "evaluations.json")
    assert [r["source_artifact_digest"] for r in ledger["records"]] == [_sha(report)]


@pytest.mark.parametrize("case", ["other-weights", "missing", "a-directory"])
def test_the_cli_refuses_a_named_file_that_is_not_the_reports_checkpoint(
    tmp_path, checkpoints, case
):
    """A ledger beside other weights is a record filed under the wrong address:
    the named file must hold the report's checkpoint bytes, and nothing is
    written otherwise."""
    import scripts.record_evaluation as record_evaluation

    named = checkpoints / {"other-weights": "other.pt", "missing": "absent.pt",
                           "a-directory": "sub"}[case]
    if case == "other-weights":
        named.write_bytes(b"other weights")
    elif case == "a-directory":
        named.mkdir()
    fragment = {"other-weights": "does not hold the weights", "missing": "does not exist",
                "a-directory": "is not a file"}[case]

    with pytest.raises(SystemExit, match=fragment) as refused:
        record_evaluation.main(["--checkpoint", str(named), "--report", str(_report(tmp_path))])

    assert str(refused.value.code).endswith(
        f"Nothing was written to {checkpoints.resolve() / 'evaluations.json'}.")
    assert not (checkpoints / "evaluations.json").exists()


def test_a_call_naming_neither_or_both_places_is_a_callers_error(tmp_path, checkpoints):
    """Exactly one of the checkpoint file and the directory names where the
    ledger goes, and a directory is always searched by hashing."""
    report = _report(tmp_path)
    with pytest.raises(TypeError, match="exactly one"):
        register_report(report)
    with pytest.raises(TypeError, match="exactly one"):
        register_report(report, checkpoint=checkpoints / "best.pt", checkpoint_dir=checkpoints)
    with pytest.raises(TypeError, match="exactly one"):
        register_report(report, checkpoint_dir=checkpoints,
                        measured=_measured(report, checkpoints))
    with pytest.raises(TypeError, match="exactly one"):
        register_report(report, checkpoint=checkpoints / "best.pt",
                        measured=_measured(report, checkpoints))
    with pytest.raises(TypeError, match="MeasuredRun"):
        register_report(report, measured=(checkpoints / "best.pt", _sha(report)))
    assert not (checkpoints / "evaluations.json").exists()


@pytest.mark.parametrize("where", [[], ["--checkpoint", "x.pt", "--checkpoint-dir", "."]],
                         ids=["neither", "both"])
def test_the_cli_needs_exactly_one_of_checkpoint_and_checkpoint_dir(tmp_path, where):
    import scripts.record_evaluation as record_evaluation

    with pytest.raises(SystemExit) as refused:
        record_evaluation.main([*where, "--show", DIGEST])
    assert refused.value.code == 2


def test_the_cli_records_and_then_reads_back(tmp_path, checkpoints, capsys):
    import scripts.record_evaluation as record_evaluation

    record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                            "--report", str(_report(tmp_path))])
    capsys.readouterr()
    record_evaluation.main(["--checkpoint-dir", str(checkpoints), "--show", DIGEST])

    shown = json.loads(capsys.readouterr().out)
    assert len(shown) == 1
    assert shown[0]["cohort"]["label"] == "mygene2-2026-10"
    assert shown[0]["scoring"]["mode"] == "disease"


def test_the_cli_show_reads_the_ledger_once(tmp_path, checkpoints, capsys):
    import scripts.record_evaluation as record_evaluation
    from tests.fixtures.opens import count_opens

    record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                            "--report", str(_report(tmp_path))])
    with count_opens() as opens:
        record_evaluation.main(["--checkpoint-dir", str(checkpoints), "--show", DIGEST])

    assert opens.opens(checkpoints / "evaluations.json") == 1


def test_the_cli_shows_an_empty_result_for_unknown_weights(tmp_path, capsys):
    import scripts.record_evaluation as record_evaluation

    record_evaluation.main(["--checkpoint-dir", str(tmp_path), "--show", "z" * 64])

    assert json.loads(capsys.readouterr().out) == []


def test_the_cli_refuses_to_show_a_ledger_it_does_not_read(tmp_path):
    import scripts.record_evaluation as record_evaluation

    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps({"schema_version": 2, "records": []}))

    with pytest.raises(SystemExit, match=f"{path} is ledger schema 2"):
        record_evaluation.main(["--checkpoint-dir", str(tmp_path), "--show", DIGEST])


def test_the_cli_refuses_a_report_that_is_not_a_measurement_artifact(tmp_path, checkpoints):
    import scripts.record_evaluation as record_evaluation

    report = store_bytes(tmp_path / "registry", json.dumps({"hello": "world"}).encode())

    with pytest.raises(SystemExit, match="not a measurement report"):
        record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                                "--report", str(report)])


def test_the_cli_refuses_a_report_from_an_earlier_revision(tmp_path, checkpoints):
    import scripts.record_evaluation as record_evaluation

    report = registered_report()
    report["manifest"]["python_seed"] = 0
    report["manifest"]["artifact_digests"]["checkpoint"] = DIGEST
    path = store_report(tmp_path / "registry", report)

    with pytest.raises(SystemExit, match="python_seed"):
        record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                                "--report", str(path)])
    assert not (checkpoints / "evaluations.json").exists()


def test_the_cli_needs_exactly_one_of_report_and_show(tmp_path):
    import scripts.record_evaluation as record_evaluation

    with pytest.raises(SystemExit, match="exactly one"):
        record_evaluation.main(["--checkpoint-dir", str(tmp_path)])


@pytest.mark.parametrize("data", [b'{"manifest": {', b"[]"], ids=["truncated", "a-list"])
def test_a_refusal_exits_non_zero_with_one_message_and_no_traceback(
    tmp_path, checkpoints, data
):
    """Run as an operator runs it: a damaged report is a message, not a crash."""
    report = store_bytes(tmp_path / "registry", data, checkpoint_sha256=DIGEST)

    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "record_evaluation.py"),
         "--checkpoint-dir", str(checkpoints), "--report", str(report)],
        capture_output=True, text=True, timeout=120, cwd=tmp_path,
    )

    assert done.returncode != 0
    assert "Traceback" not in done.stderr
    lines = [line for line in done.stderr.splitlines() if line.strip()]
    assert len(lines) == 1 and "Nothing was written" in lines[0], done.stderr
    assert not (checkpoints / "evaluations.json").exists()


def test_the_script_reads_and_hashes_nothing_itself():
    """Everything it records comes from `register_report` and `read_ledger`, so
    the static pin's scope can take it with an empty site list."""
    from tests.unit.test_provenance_pins import _sites_of

    path = REPO / "scripts" / "record_evaluation.py"
    sites = _sites_of(path.read_text(encoding="utf-8"), str(path)).sites

    assert not sites, dict(sites)


# ---------------------------------------------------------------------------
# The standing text rendered beside a record
# ---------------------------------------------------------------------------
def test_every_relation_has_a_display_name_and_no_other_name_has_one():
    from src.evaluation.caveats import RELATION_DISPLAY_NAMES
    from src.evaluation.measurement import RELATION_NAMES

    assert tuple(RELATION_DISPLAY_NAMES) == RELATION_NAMES
    assert all(RELATION_DISPLAY_NAMES.values())


def test_the_r10_labels_are_the_four_of_the_contract():
    """Two scopes, two labels each, keyed as a record's `r10` block is; the
    phenotype-set definitions say the scoring inputs may differ."""
    from src.evaluation.caveats import R10_LABEL_TEXT

    assert set(R10_LABEL_TEXT) == {"training", "validation"}
    for scope, labels in R10_LABEL_TEXT.items():
        assert set(labels) == {"disease_overlap", "phenotype_set_overlap"}
        for label in labels.values():
            assert set(label) == {"name", "definition"}
            assert label["name"].lower().startswith(scope)
        assert ("phenotype-set overlap ignores order and repetition, so the scoring "
                "inputs may differ") in labels["phenotype_set_overlap"]["definition"]


def test_the_r10_states_say_neither_zero_nor_no_leakage():
    """`unverifiable` is never a number, and `none` is not proof the model never
    saw these cases (brief §3)."""
    from src.evaluation.caveats import R10_STATE_TEXT

    assert set(R10_STATE_TEXT) == {"overlap", "none", "unverifiable",
                                   "no_validation_inputs_recorded"}
    assert not any(c.isdigit() for c in R10_STATE_TEXT["unverifiable"])
    assert "no count" in R10_STATE_TEXT["unverifiable"]
    assert "leak" not in R10_STATE_TEXT["none"]
    assert "not evidence about data outside them" in R10_STATE_TEXT["none"]


def test_the_result_label_the_mode_c_note_and_the_absence_sentence():
    """The Mode C note is dated history (owner decision 4, K7): its figures'
    denominator, and that the number does not describe the current diagnosis
    model; no claim of what the page serves today."""
    from src.evaluation.caveats import MODE_C_RESULT_LABEL, PRE_B1_NOTE, RECORD_ABSENCE

    assert MODE_C_RESULT_LABEL == (
        "Disease ranking by the model, approved scorer policy (Mode C)")
    assert PRE_B1_NOTE == (
        "Mode C (October 2026, before the task models) ranks every disease by the model's raw "
        "cosine; its figures are over the cases ranked. The Diagnosis page never ranked by "
        "Mode C, so this number does not describe the current diagnosis model.")
    assert RECORD_ABSENCE == (
        "This ledger holds no result for these weights; that does not mean they were "
        "never evaluated.")


# ---------------------------------------------------------------------------
# An automatic registration is bound to its own run (reviewer's P2, 2026-10-17)
# ---------------------------------------------------------------------------
def test_a_run_registers_only_the_report_bytes_it_published(tmp_path, checkpoints):
    """`--record` passes the digest `publish_bytes` returned. A file at the
    report's path holding any other bytes -- here a valid report of the same
    checkpoint, written over it after publication -- is refused as not the
    run's report, before it is interpreted, and nothing is written."""
    published = _report(tmp_path)
    measured = _measured(published, checkpoints)
    replacement = json.loads(published.read_text())
    replacement["authoritative_metrics"]["pooled_top1"] = (
        replacement["authoritative_metrics"]["pooled_top1"] / 2 + 0.25)
    published.write_text(json.dumps(replacement, indent=1))
    assert _sha(published) != measured.report_sha256

    with pytest.raises(ReportNotAsPublished) as refused:
        register_report(published, measured=measured, registered_at=WHEN)

    message = str(refused.value)
    assert "is not the report this run published" in message
    assert measured.report_sha256[:12] in message and _sha(published)[:12] in message
    assert "re-measure" in message
    assert message.endswith(
        f"Nothing was written to {checkpoints.resolve() / 'evaluations.json'}.")
    assert not (checkpoints / "evaluations.json").exists()


def test_a_run_registers_only_a_report_of_the_checkpoint_it_read(tmp_path, checkpoints):
    """The report's bytes are the published ones, but it names other weights
    than the run's checkpoint read: refused as filed under the wrong address,
    not as a replaced file, and nothing is written."""
    report = _report(tmp_path)
    other = hashlib.sha256(b"other weights").hexdigest()

    with pytest.raises(RegistrationRefused) as refused:
        register_report(report, measured=_measured(report, checkpoints,
                                                   checkpoint_sha256=other),
                        registered_at=WHEN)

    assert not isinstance(refused.value, ReportNotAsPublished)
    message = str(refused.value)
    assert f"names checkpoint {DIGEST[:12]}" in message
    assert f"this run read {other[:12]}" in message
    assert not (checkpoints / "evaluations.json").exists()


def test_a_run_whose_report_and_checkpoint_are_its_own_registers_without_hashing(
    tmp_path, checkpoints, monkeypatch
):
    """The control: the published bytes and the run's read agree, the record is
    filed beside the read, and the checkpoint is not hashed again."""
    import src.evaluation.registration as registration

    def no_hashing(*args, **kwargs):
        raise AssertionError("a run's own checkpoint read is not hashed again")

    monkeypatch.setattr(registration, "checkpoint_matches", no_hashing)
    monkeypatch.setattr(registration, "find_checkpoint", no_hashing)
    report = _report(tmp_path)

    result = register_report(report, measured=_measured(report, checkpoints),
                             registered_at=WHEN)

    assert result.outcome == "registered"
    assert result.record["source_artifact_digest"] == _sha(report)
    assert result.ledger_path == checkpoints.resolve() / "evaluations.json"


@pytest.mark.parametrize("lost", ["not_utf8", "damaged_json", "missing", "a_directory"])
def test_a_run_never_gets_back_a_lost_report_as_kept(tmp_path, checkpoints, lost):
    """The reviewer's P3 (2026-10-17): whatever stands at the report's path
    instead of the published bytes -- bytes that are not even UTF-8, damaged
    JSON, nothing, or a directory -- an automatic registration refuses it as not
    the run's report (`ReportNotAsPublished`), compared before anything is
    decoded, so no entry point calls it kept. Nothing is written."""
    published = _report(tmp_path)
    measured = _measured(published, checkpoints)
    published.unlink()
    if lost == "not_utf8":
        published.write_bytes(b"\xff\xfe\x00 not a report")
    elif lost == "damaged_json":
        published.write_bytes(b"{broken")
    elif lost == "a_directory":
        published.mkdir()

    with pytest.raises(ReportNotAsPublished) as refused:
        register_report(published, measured=measured, registered_at=WHEN)

    message = str(refused.value)
    assert "parse" not in message, "the bytes were compared, not decoded"
    assert "re-measure" in message
    assert not (checkpoints / "evaluations.json").exists()


def test_a_runs_own_report_replaced_by_a_link_out_of_the_store_is_lost(tmp_path, checkpoints):
    """The reviewer's P3 (2026-10-20): the run published its report into the
    store at this path, so a path that now resolves elsewhere -- here a link to
    a byte-identical copy outside any store -- was replaced: refused as not the
    run's report (`ReportNotAsPublished`), never as an outside copy whose
    store original could still be registered. Nothing is written."""
    published = _report(tmp_path)
    measured = _measured(published, checkpoints)
    outside = tmp_path / "outside.json"
    outside.write_bytes(published.read_bytes())
    published.unlink()
    published.symlink_to(outside)

    with pytest.raises(ReportNotAsPublished) as refused:
        register_report(published, measured=measured, registered_at=WHEN)

    message = str(refused.value)
    assert f"{published} is no longer a report the store holds" in message
    assert "replaced or moved after it was written" in message and "re-measure" in message
    assert "Register the store's original" not in message
    assert not (checkpoints / "evaluations.json").exists()
    # By hand, the same path is an outside copy, refused as such.
    with pytest.raises(RegistrationRefused) as by_hand:
        register_report(published, checkpoint=checkpoints / "best.pt", registered_at=WHEN)
    assert not isinstance(by_hand.value, ReportNotAsPublished)
    assert "Register the store's original instead" in str(by_hand.value)


def test_a_report_nested_too_deeply_to_build_is_refused_not_raised(tmp_path, checkpoints):
    """A hand-edited report that parses but nests deeper than `build_record`
    can copy is one refusal, not a `RecursionError` traceback."""
    report = _report(tmp_path)
    deep = json.loads(report.read_text())
    node = deep["manifest"]["r10"]
    for _ in range(600):
        node["x"] = {}
        node = node["x"]
    report.write_text(json.dumps(deep))

    with pytest.raises(RegistrationRefused, match="nests too deeply to be read as a report"):
        register_report(report, checkpoint=checkpoints / "best.pt", registered_at=WHEN)
    assert not (checkpoints / "evaluations.json").exists()


def test_a_report_nested_too_deeply_is_refused_whatever_the_recursion_limit(tmp_path, checkpoints):
    """The depth is counted, not recursed: a raised recursion limit (a library
    may raise it) or a deep caller stack never changes the answer."""
    import sys

    from src.evaluation.sidecar import MAX_REPORT_DEPTH

    report = _report(tmp_path)
    deep = json.loads(report.read_text())
    node = deep["manifest"]["r10"]
    for _ in range(MAX_REPORT_DEPTH):
        node["x"] = {}
        node = node["x"]
    report.write_text(json.dumps(deep))
    before = sys.getrecursionlimit()
    sys.setrecursionlimit(100_000)
    try:
        with pytest.raises(RegistrationRefused, match="nests too deeply to be read as a report"):
            register_report(report, checkpoint=checkpoints / "best.pt", registered_at=WHEN)
    finally:
        sys.setrecursionlimit(before)
    assert not (checkpoints / "evaluations.json").exists()


def test_the_nesting_count_stops_exactly_at_its_limit():
    from src.evaluation.sidecar import _nests_deeper_than

    def nested(levels):
        value = {}
        for _ in range(levels - 1):
            value = {"x": value}
        return value

    assert not _nests_deeper_than(nested(64), 64)
    assert _nests_deeper_than(nested(65), 64)
    assert _nests_deeper_than([[[]]], 2) and not _nests_deeper_than([[[]]], 3)


def test_a_later_registration_still_names_a_missing_report_plainly(tmp_path, checkpoints):
    """A registration by hand has no run behind it: a missing report is the
    ordinary refusal, not a run's lost report."""
    with pytest.raises(RegistrationRefused) as refused:
        register_report(tmp_path / "absent.json", checkpoint=checkpoints / "best.pt",
                        registered_at=WHEN)
    assert not isinstance(refused.value, ReportNotAsPublished)
    assert "re-measure" not in str(refused.value)


# ---------------------------------------------------------------------------
# Kept? Registered? Read from disk (`report_state`), never from a job's exit
# ---------------------------------------------------------------------------
def test_report_state_reads_kept_and_registered_from_the_store_and_the_ledger(
    tmp_path, checkpoints
):
    from src.evaluation.registration import report_state

    report = _report(tmp_path)
    published = _sha(report)

    state = report_state(report, checkpoint=checkpoints / "best.pt", expected_sha256=published)
    assert (state.kept, state.registered) == ("kept", "not_registered")
    assert state.report_sha256 == published and state.sequence is None
    assert state.ledger_path == checkpoints.resolve() / "evaluations.json"

    _register(report, checkpoints, beside=True)
    state = report_state(report, checkpoint=checkpoints / "best.pt")
    assert (state.kept, state.registered, state.sequence) == ("kept", "registered", 1)

    # Another run with equal values is not this run (K8-6r.2): not registered
    # until it is, then its own record.
    another = _report(tmp_path, data_dir="/elsewhere")
    state = report_state(another, checkpoint=checkpoints / "best.pt")
    assert (state.registered, state.sequence) == ("not_registered", None)
    _register(another, checkpoints, beside=True)
    state = report_state(another, checkpoint=checkpoints / "best.pt")
    assert (state.registered, state.sequence) == ("registered", 2)


def test_report_state_never_calls_a_replaced_or_missing_report_kept(tmp_path, checkpoints):
    """The run's published digest decides: other bytes at the path are not its
    report, and a report removed after its registration is still registered
    (by the digest the run published), but not kept."""
    from src.evaluation.registration import report_state

    report = _report(tmp_path)
    published = _sha(report)
    _register(report, checkpoints, beside=True)
    data = json.loads(report.read_text())
    report.write_text(json.dumps(data, indent=3))

    replaced = report_state(report, checkpoint=checkpoints / "best.pt",
                            expected_sha256=published)
    assert replaced.kept == "not_kept" and "replaced or changed" in replaced.why_kept
    assert replaced.registered == "registered", "looked up by the digest the run published"

    report.unlink()
    gone = report_state(report, checkpoint=checkpoints / "best.pt", expected_sha256=published)
    assert (gone.kept, gone.registered) == ("not_kept", "registered")
    unknown = report_state(report, checkpoint=checkpoints / "best.pt")
    assert (unknown.kept, unknown.registered) == ("not_kept", "cannot_be_checked")


def test_report_state_says_cannot_be_checked_rather_than_no(tmp_path, checkpoints):
    """A ledger that cannot be read is not "not registered"; a path outside a
    store is not a kept report."""
    from src.evaluation.registration import report_state

    report = _report(tmp_path)
    (checkpoints / "evaluations.json").write_text(
        json.dumps({"schema_version": 2, "records": []}))
    state = report_state(report, checkpoint=checkpoints / "best.pt")
    assert (state.kept, state.registered) == ("kept", "cannot_be_checked")
    assert "schema 2" in state.why_registered

    copy = tmp_path / "copy.json"
    copy.write_bytes(report.read_bytes())
    outside = report_state(copy, checkpoint=checkpoints / "best.pt")
    assert outside.kept == "not_kept" and "not in a registry's report store" in outside.why_kept


# ---------------------------------------------------------------------------
# A kept Mode D report: registered by the general rules under its own mode
# (K8-6c, policy rev 15)
# ---------------------------------------------------------------------------
def _mode_d(tmp_path) -> Path:
    """A kept Mode D report measuring `WEIGHTS`, in the version's report store:
    Mode D no longer runs (K7), so it is built (`mode_d_report`)."""
    report = mode_d_report()
    report["manifest"]["artifact_digests"]["checkpoint"] = DIGEST
    return store_report(tmp_path / "registry", report)


@pytest.mark.parametrize("entry", ["measured", "checkpoint_dir", "checkpoint", "place"])
def test_a_kept_mode_d_report_registers_by_the_general_rules(tmp_path, checkpoints, entry):
    """Policy rev 15 (K8-6c): no rule refuses a report because its mode is D.
    `register_report`, the one writer behind `--record`, `record_evaluation`
    and the Test page's Register, registers a kept Mode D report by the rules
    every report meets, under its own mode: the ledger beside the checkpoint
    holds its record, mode D; the version's usage history has its
    `registered` line; the store reads the report as before, under its own
    key, uncontested; and the report is kept and registered."""
    from src.evaluation.registration import report_state
    from src.evaluation.report_store import read_usage_history, scan_report_store

    report = _mode_d(tmp_path)
    before_scan = scan_report_store(version_of(report))
    (key,) = before_scan.reports
    how = {"measured": {"measured": _measured(report, checkpoints)},
           "checkpoint_dir": {"checkpoint_dir": checkpoints},
           "checkpoint": {"checkpoint": checkpoints / "best.pt"},
           "place": {"place": lambda manifest: checkpoints / "best.pt",
                     "expected_report_sha256": _sha(report)}}[entry]

    result = register_report(report, registered_at=WHEN, **how)

    assert result.outcome == "registered"
    assert result.ledger_path == checkpoints.resolve() / "evaluations.json"
    assert result.record["scoring"]["mode"] == "D"
    assert result.record["source_artifact_digest"] == _sha(report)
    assert result.record["registration"] == {"sequence": 1, "registered_at": WHEN}
    ledger, _digest = read_ledger(result.ledger_path)
    assert ledger["records"] == [result.record]
    entries, problems = read_usage_history(version_of(report))
    assert problems == []
    assert [(e["kind"], e["outcome"], e["sequence"], e["report_sha256"]) for e in entries] == [
        ("registered", "registered", 1, _sha(report))]
    after_scan = scan_report_store(version_of(report))
    assert after_scan.reports == before_scan.reports
    ((_run_id, record),) = after_scan.reports[key].values()
    assert record["scoring"]["mode"] == "D" and after_scan.state(key) == "uncontested"
    state = report_state(report, checkpoint=checkpoints / "best.pt")
    assert (state.kept, state.registered) == ("kept", "registered")


def test_a_mode_d_record_already_in_a_ledger_stays_and_a_task_report_still_registers(
        tmp_path, checkpoints):
    """A Mode D record a ledger already holds (Batch 3's, written before each
    run had its own record: schema 3, no run) stays as written and is read with
    it. Its report registered again from its run directory is its own record
    now (A11's trade-off, K8-6r.2), noted as a possible overlap with record 1;
    sent once more it answers ``already_registered`` and writes nothing, neither
    the ledger nor the usage history; and none of it stops the next
    registration."""
    from src.evaluation.sidecar import LEDGER_FILENAME

    kept = _mode_d(tmp_path)
    ledger_path = checkpoints / LEDGER_FILENAME
    old = _historical(ledger_path, kept)

    named = _register(kept, checkpoints, beside=True, registered_at="2027-01-01T00:00:00Z")
    assert (named.outcome, named.record["registration"]["sequence"]) == ("registered", 2)
    assert named.record["scoring"]["mode"] == "D" and named.record["run_id"] == kept.parent.name
    assert named.historical_overlap == (
        "possible overlap with record 1, merged before each run had its own record: not "
        "recorded whether this run is one of them",)
    before, usage = ledger_path.read_bytes(), _usage(tmp_path)
    again = _register(kept, checkpoints, beside=True)
    assert again.outcome == "already_registered" and again.record == named.record
    assert ledger_path.read_bytes() == before and _usage(tmp_path) == usage
    result = _register(_report(tmp_path), checkpoints, beside=True)

    assert result.outcome == "registered"
    ledger, _digest = read_ledger(ledger_path)
    assert ledger["records"][0] == old
    assert [r["scoring"]["mode"] for r in ledger["records"]] == ["D", "D", "disease"]


def test_a_kept_mode_d_report_whose_version_block_was_edited_is_refused_like_any_report(
        tmp_path, checkpoints):
    """The general checks hold for a Mode D report as for a task report
    (`test_a_report_must_name_the_version_it_is_kept_under`): one whose
    cohort-version block was edited (its counts, its manifest digest left as
    it was) is refused with the version block's reason, naming what differs,
    and nothing is written."""
    report = _mode_d(tmp_path)
    data = json.loads(report.read_text())
    data["manifest"]["cohort_version"]["counts"]["n_excluded"] += 1
    report.write_text(json.dumps(data))
    usage = _usage(tmp_path)

    with pytest.raises(RegistrationRefused, match="differing in: counts") as refused:
        _register(report, checkpoints, beside=True)

    assert "registered only under the version it measured" in str(refused.value)
    assert "Mode D" not in str(refused.value)
    assert not (checkpoints / "evaluations.json").exists()
    assert _usage(tmp_path) == usage


def test_a_kept_mode_d_reports_record_keeps_its_own_mode_and_semantics(tmp_path, checkpoints):
    """No conversion (policy rev 15): the record registered from a kept Mode D
    report is the one `build_record` makes of it, as for any report: its mode
    "D", which is no protocol of this build, its scoring as recorded, and the
    `measurement_semantics_digest` of its own manifest."""
    from src.evaluation import protocols
    from src.evaluation.sidecar import SCORING_SEMANTICS_FIELDS, measurement_semantics_digest

    report = _mode_d(tmp_path)
    data = json.loads(report.read_text())
    manifest = data["manifest"]

    record = _register(report, checkpoints, beside=True).record

    assert record["scoring"]["mode"] == manifest["mode"] == "D"
    assert "D" not in protocols.names()
    assert record["scoring"] == {field: manifest[field] for field in SCORING_SEMANTICS_FIELDS}
    assert record["measurement_semantics_digest"] == measurement_semantics_digest(manifest)
    appended = ("registration", "corroborating_reports", "run_id")  # `append_record`'s
    built = build_record(data, _sha(report))
    assert {k: v for k, v in record.items() if k not in appended} == built
    assert (record["registration"]["sequence"], record["corroborating_reports"],
            record["run_id"]) == (1, [], report.parent.name)


# ---------------------------------------------------------------------------
# Without a checkpoint: the server's list of kept reports (M5-1)
# ---------------------------------------------------------------------------
def test_report_state_without_a_checkpoint_reads_the_ledger_the_report_names(
    tmp_path, checkpoints
):
    """A list of a version's kept reports knows no checkpoint file: the
    ledger read is the one beside the checkpoint path the report records, from
    the same read. Found there, the report is registered; not found, that is
    all that can be said: never "not registered"."""
    from src.evaluation.registration import report_state

    report = _report(tmp_path, checkpoint_path=str(checkpoints / "best.pt"))
    ledger = checkpoints.resolve() / "evaluations.json"
    state = report_state(report)
    assert (state.kept, state.registered) == ("kept", "cannot_be_checked")
    assert state.why_registered.startswith(
        f"not registered beside the checkpoint this run read (there is no {ledger});")
    assert "another copy of the same weights" in state.why_registered
    assert state.ledger_path == ledger

    _register(report, checkpoints, beside=True)
    state = report_state(report)
    assert (state.kept, state.registered, state.sequence) == ("kept", "registered", 1)


def test_a_report_registered_beside_a_copy_is_never_listed_as_not_registered(
    tmp_path, checkpoints
):
    """`record_evaluation --checkpoint` files the record beside whichever file
    with the measured bytes it is given, a served copy say. Given that copy,
    the report reads as registered; the list, which knows only the checkpoint
    the run read, cannot rule that out and says so (review of M5-1 WP2)."""
    from src.evaluation.registration import report_state

    report = _report(tmp_path, checkpoint_path=str(checkpoints / "best.pt"))
    release = tmp_path / "release"
    release.mkdir()
    (release / "best.pt").write_bytes(WEIGHTS)
    _register(report, release, beside=True)

    served = report_state(report, checkpoint=release / "best.pt")
    assert (served.registered, served.sequence) == ("registered", 1)
    measured = report_state(report, checkpoint=checkpoints / "best.pt")
    assert measured.registered == "not_registered", "scoped by the checkpoint given"

    listed = report_state(report)
    assert listed.registered == "cannot_be_checked"
    assert str(checkpoints.resolve() / "evaluations.json") in listed.why_registered
    assert "cannot be checked here" in listed.why_registered


def test_report_state_without_a_checkpoint_never_guesses_a_ledger(tmp_path, checkpoints):
    """A relative recorded path meant whatever that run's working directory
    made of it; a report that cannot be read names nothing. Either way the
    registration cannot be checked, and says why; kept is still answered."""
    from src.evaluation.registration import report_state

    relative = _report(tmp_path, checkpoint_path="runs/shepherd/r1/checkpoints/best.pt")
    state = report_state(relative)
    assert (state.kept, state.registered) == ("kept", "cannot_be_checked")
    assert "relative path" in state.why_registered and state.ledger_path is None

    gone = _report(tmp_path, checkpoint_path=str(checkpoints / "best.pt"))
    gone.unlink()
    state = report_state(gone)
    assert (state.kept, state.registered) == ("not_kept", "cannot_be_checked")


# ---------------------------------------------------------------------------
# The report that was shown (the M5 review's P2-1)
# ---------------------------------------------------------------------------
def _usage(tmp_path) -> bytes:
    """The version's usage history as bytes ("" when there is none)."""
    files = list((tmp_path / "registry").glob("versions/*/reports/usage.jsonl"))
    return files[0].read_bytes() if files else b""


def test_the_digest_shown_is_compared_on_the_read_registered_before_it_is_parsed(
    tmp_path, checkpoints
):
    """`expected_report_sha256`: other bytes than the ones shown are refused
    from the digest of the one read, before they are decoded -- a plain
    refusal, not `ReportNotAsPublished`, which belongs to a run's own report
    -- and nothing is written; the digest shown registers."""
    report = _report(tmp_path)
    usage_before = _usage(tmp_path)

    with pytest.raises(RegistrationRefused) as refused:
        register_report(report, checkpoint_dir=checkpoints, expected_report_sha256="0" * 64)
    assert not isinstance(refused.value, ReportNotAsPublished)
    said = str(refused.value)
    assert f"{report} is not the report that was shown: it now holds bytes {_sha(report)[:12]}" in said
    assert "0" * 12 + "... was shown" in said and "select it again" in said
    assert said.endswith("Nothing was written.")      # the directory is not searched yet
    assert not (checkpoints / "evaluations.json").exists()
    assert _usage(tmp_path) == usage_before

    damaged = report.read_bytes()[:20]
    shown = _sha(report)
    report.write_bytes(damaged)
    with pytest.raises(RegistrationRefused, match="is not the report that was shown") as refused:
        register_report(report, checkpoint_dir=checkpoints, expected_report_sha256=shown)
    assert "not a measurement report" not in str(refused.value), "it was parsed"


def test_the_digest_shown_registers_the_report_it_names(tmp_path, checkpoints):
    report = _report(tmp_path)

    result = register_report(report, checkpoint_dir=checkpoints,
                             expected_report_sha256=_sha(report), registered_at=WHEN)

    assert result.outcome == "registered"
    assert result.record["source_artifact_digest"] == _sha(report)


@pytest.mark.parametrize("digest", ["ABC", "A" * 64, "a" * 63, 123, b"0" * 64])
def test_a_digest_that_is_not_a_sha256_is_a_callers_error(tmp_path, checkpoints, digest):
    report = _report(tmp_path)
    with pytest.raises(TypeError, match="64 lowercase hex"):
        register_report(report, checkpoint_dir=checkpoints, expected_report_sha256=digest)
    assert not (checkpoints / "evaluations.json").exists()


def test_a_run_and_a_shown_digest_together_are_a_callers_error(tmp_path, checkpoints):
    """`measured` carries the digest its run published; a second one would be
    two answers to one question."""
    report = _report(tmp_path)
    with pytest.raises(TypeError, match="measured carries the digest"):
        register_report(report, measured=_measured(report, checkpoints),
                        expected_report_sha256=_sha(report))


# ---------------------------------------------------------------------------
# Only from the deployment's registry (BACKLOG 37; the M5 review's P2-7)
# ---------------------------------------------------------------------------
def _whole_copy(tmp_path) -> Path:
    import shutil

    copy = tmp_path / "registry-copy"
    shutil.copytree(tmp_path / "registry", copy, symlinks=True)
    return copy


def test_a_copy_of_the_whole_registry_is_refused_before_anything_in_it_is_read(
    tmp_path, checkpoints, monkeypatch
):
    """The reviewer's counter-example: a copy holds the same bytes in the same
    structure, so only where it is tells it apart. Refused from the path's shape
    (its `cohort.json` never resolved), nothing written beside the checkpoint,
    nothing appended to the copy's usage history; the original registers."""
    from src.evaluation import cohort_registry
    from src.evaluation.registration import NotFromTheDeploymentRegistry

    report = _report(tmp_path)
    copy = _whole_copy(tmp_path)
    in_copy = copy / report.relative_to(tmp_path / "registry")
    copy_before = {p: p.read_bytes() for p in copy.rglob("*") if p.is_file()}
    resolved = []
    real = cohort_registry.resolve_version
    monkeypatch.setattr(cohort_registry, "resolve_version",
                        lambda *a, **k: resolved.append(a) or real(*a, **k))

    with pytest.raises(NotFromTheDeploymentRegistry) as refused:
        _register(in_copy, checkpoints, beside=True)

    said = str(refused.value)
    assert f"{in_copy} is not registered: the registry {copy} is not the deployment's" in said
    assert (f"SHEPHERD_REGISTRY ({tmp_path / 'registry'}) resolves to "
            f"{(tmp_path / 'registry').resolve()}") in said
    assert "Nothing was written" in said and "is a copy" not in said
    assert resolved == [], "something in the copy was read"
    assert not (checkpoints / "evaluations.json").exists()
    assert {p: p.read_bytes() for p in copy.rglob("*") if p.is_file()} == copy_before
    assert _register(report, checkpoints, beside=True).outcome == "registered"


def test_a_runs_own_report_in_another_registry_is_not_called_lost(tmp_path, checkpoints,
                                                                 monkeypatch):
    """For `--record`, the run's report is where it published it: the refusal is
    of the registry, not `ReportNotAsPublished`."""
    from src.evaluation.registration import NotFromTheDeploymentRegistry

    report = _report(tmp_path)
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(_whole_copy(tmp_path)))
    with pytest.raises(NotFromTheDeploymentRegistry) as refused:
        _register(report, checkpoints)
    assert not isinstance(refused.value, ReportNotAsPublished)


@pytest.mark.parametrize("setting, said", [
    (None, "SHEPHERD_REGISTRY is not set here, so which registry is the deployment's cannot "
           "be told"),
    ("relative/registry", "SHEPHERD_REGISTRY is refused: the registry 'relative/registry' is "
                          "not an absolute path"),
], ids=["unset", "refused"])
def test_with_no_usable_setting_every_registration_is_refused_saying_so(
    tmp_path, checkpoints, monkeypatch, setting, said
):
    """A fresh shell without the service's settings: refused, never "any
    registry", with words true on the command line (nothing about features
    that are off) and nothing written."""
    report = _report(tmp_path)
    if setting is None:
        monkeypatch.delenv("SHEPHERD_REGISTRY")
    else:
        monkeypatch.setenv("SHEPHERD_REGISTRY", setting)

    with pytest.raises(RegistrationRefused) as refused:
        _register(report, checkpoints, beside=True)

    assert said in str(refused.value)
    assert "features" not in str(refused.value)
    assert not (checkpoints / "evaluations.json").exists()


def test_a_link_to_the_deployments_registry_is_that_registry(tmp_path, checkpoints,
                                                             monkeypatch):
    """Compared by resolved path: the setting through a link, or the report
    reached through one, names the same registry."""
    report = _report(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "registry", target_is_directory=True)

    monkeypatch.setenv("SHEPHERD_REGISTRY", str(alias))
    assert _register(report, checkpoints, beside=True).outcome == "registered"
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(tmp_path / "registry"))
    through = alias / report.relative_to(tmp_path / "registry")
    assert _register(through, checkpoints, beside=True).outcome == "already_registered"


def test_the_settings_passed_say_which_registry_is_the_deployments(tmp_path, checkpoints,
                                                                   monkeypatch):
    """`environ` (the page's settings) is read, not the process's."""
    report = _report(tmp_path)
    monkeypatch.delenv("SHEPHERD_REGISTRY")

    result = _register(report, checkpoints, beside=True,
                       environ={"SHEPHERD_REGISTRY": str(tmp_path / "registry")})

    assert result.outcome == "registered"


def test_record_evaluation_refuses_a_report_in_a_copy_of_the_registry(tmp_path, checkpoints):
    import scripts.record_evaluation as record_evaluation

    report = _report(tmp_path)
    in_copy = _whole_copy(tmp_path) / report.relative_to(tmp_path / "registry")

    with pytest.raises(SystemExit) as refused:
        record_evaluation.main(["--report", str(in_copy), "--checkpoint-dir", str(checkpoints)])

    assert "is not the deployment's" in str(refused.value.code)
    assert not (checkpoints / "evaluations.json").exists()


# ---------------------------------------------------------------------------
# Registration by the verified run (K8-6r.2, policy rev 16): rows A, B and C
# ---------------------------------------------------------------------------
def test_byte_identical_reports_of_two_runs_are_two_records(tmp_path, checkpoints):
    """A3: the same bytes in two run directories are two runs. Run 2 reads
    ``not_registered`` before its registration and ``registered`` after;
    record 2 has the same source digest and its own run. HEAD: run 2 read
    ``registered`` and answered ``already_registered``."""
    from src.evaluation.registration import report_state

    one = _report(tmp_path)
    two = _again_in_another_run(one)
    first = _register(one, checkpoints)

    before = report_state(two, checkpoint=checkpoints / "best.pt")
    second = _register(two, checkpoints)
    after = report_state(two, checkpoint=checkpoints / "best.pt")

    assert (before.registered, after.registered, after.sequence) == (
        "not_registered", "registered", 2)
    assert second.outcome == "registered"
    assert second.record["source_artifact_digest"] == first.record["source_artifact_digest"]
    assert (first.record["run_id"], second.record["run_id"]) == (one.parent.name,
                                                                 two.parent.name)
    assert second.group_differences == () and second.historical_overlap == ()


@pytest.mark.parametrize("entry", ["measured", "checkpoint", "checkpoint_dir", "place"])
def test_the_same_runs_report_again_writes_nothing_through_every_entry_point(
    tmp_path, checkpoints, entry
):
    """A4 (guard): the same run's report again, through `measured`,
    `checkpoint`, `checkpoint_dir` and `place`: ``already_registered``; the
    ledger's and the usage history's bytes unchanged."""
    report = _report(tmp_path)
    first = _register(report, checkpoints)
    ledger, usage = first.ledger_path.read_bytes(), _usage(tmp_path)
    how = {"measured": {"measured": _measured(report, checkpoints)},
           "checkpoint": {"checkpoint": checkpoints / "best.pt"},
           "checkpoint_dir": {"checkpoint_dir": checkpoints},
           "place": {"place": lambda manifest: checkpoints / "best.pt",
                     "expected_report_sha256": _sha(report)}}[entry]

    again = register_report(report, registered_at="2027-01-01T00:00:00Z", **how)

    assert again.outcome == "already_registered"
    assert again.record["registration"] == {"sequence": 1, "registered_at": WHEN}
    assert first.ledger_path.read_bytes() == ledger and _usage(tmp_path) == usage


@pytest.mark.parametrize("overrides", [{"data_dir": "/elsewhere"}, {"batch_size": 16}],
                         ids=["equal-values-other-data-dir", "another-group"])
def test_a_runs_report_replaced_after_its_registration_is_refused(
    tmp_path, checkpoints, overrides
):
    """A5: the same run's report.json replaced after registration, (a) with
    equal values and another `data_dir`, (b) with another group (batch size
    16): refused, naming record 1 and both digests, since a run's report is
    sealed when it is published; the ledger and the usage history unchanged.
    HEAD: (a) merged into record 1, (b) a new record."""
    from src.evaluation.registration import report_state

    report = _report(tmp_path)
    first = _register(report, checkpoints, beside=True)
    sealed = _sha(report)
    replacement = registered_report(**overrides)
    replacement["manifest"]["artifact_digests"]["checkpoint"] = DIGEST
    replacement["manifest"]["cohort_version"]["manifest_sha256"] = (
        version_of(report).identity.sha256)
    report.write_text(json.dumps(replacement))
    ledger, usage = first.ledger_path.read_bytes(), _usage(tmp_path)

    with pytest.raises(RegistrationRefused) as refused:
        _register(report, checkpoints, beside=True)

    said = str(refused.value)
    assert (f"run {report.parent.name} is already registered as record 1 from report "
            f"{sealed[:12]}...; this report ({_sha(report)[:12]}...) is other bytes: a run's "
            "report is sealed when it is published, so this file was replaced after it was "
            "registered") in said
    assert said.endswith(f"Nothing was written to {first.ledger_path}.")
    assert first.ledger_path.read_bytes() == ledger and _usage(tmp_path) == usage
    state = report_state(report, checkpoint=checkpoints / "best.pt")
    assert state.registered == "not_registered" and "sealed" in state.why_registered


def test_deterministic_runs_whose_values_differ_are_kept_and_flagged_for_investigation(
    tmp_path, checkpoints
):
    """A6: deterministic algorithms on in both runs, values differ: both kept;
    the group note says so and asks for an investigation. HEAD: refused."""
    one = _report(tmp_path)
    two = _report(tmp_path, metrics={**DISEASE_METRICS, "pooled_mrr": 0.4501})

    first, second = _register(one, checkpoints), _register(two, checkpoints)

    assert (first.outcome, second.outcome) == ("registered", "registered")
    (other,) = second.group_differences
    assert other.deterministic is True
    assert group_note(second).endswith(
        "This run and record 1 both record deterministic algorithms on, so they were "
        "expected to agree: investigate.")


@pytest.mark.parametrize("first_registered_at", [
    "2026-10-15T08:00:00Z", WHEN, "2099-01-01T00:00:00Z", "not a time",
], ids=["earlier", "same-second", "future", "unreadable"])
def test_runs_whose_bytes_an_old_merged_record_holds_each_get_their_own_record(
    tmp_path, checkpoints, first_registered_at
):
    """A7 (Decision 1): a schema 3 ledger whose record 1 names no run and holds
    D1 with corroborating [D2]. (a) D1 and (b) D2, each from its own run
    directory, become records of their own with their runs, whatever record
    1's time; each result's historical note names record 1 as not confirmed;
    `report_state` reads ``not_registered`` before (with the overlap note) and
    ``registered`` after; each resent answers ``already_registered``. (c) D3 of
    the same group is its own record with no note. (d) Record 1 stays equal as
    a JSON value, with no run, environment or R10 added. HEAD: (a) and (b)
    ``already_registered``; D3 merged into record 1."""
    from src.evaluation.registration import report_state

    d1 = _report(tmp_path)
    d2 = _report(tmp_path, data_dir="/elsewhere")
    d3 = _report(tmp_path, data_dir="/third")
    ledger_path = checkpoints / "evaluations.json"
    old = _historical(ledger_path, d1, [_sha(d2)], registered_at=first_registered_at)
    overlap = ("possible overlap with record 1, merged before each run had its own record: "
               "not recorded whether this run is one of them")

    for sequence, report in ((2, d1), (3, d2)):
        before = report_state(report, checkpoint=checkpoints / "best.pt")
        assert before.registered == "not_registered"
        assert ("its bytes equal a report merged into record 1 before each run had its own "
                "record; registering it adds its own record") in before.why_registered
        result = _register(report, checkpoints)
        assert (result.outcome, result.record["registration"]["sequence"],
                result.record["run_id"]) == ("registered", sequence, report.parent.name)
        assert result.historical_overlap == (overlap,)
        assert historical_note(result) == overlap[:1].upper() + overlap[1:] + "."
        after = report_state(report, checkpoint=checkpoints / "best.pt")
        assert (after.registered, after.sequence) == ("registered", sequence)
        assert _register(report, checkpoints).outcome == "already_registered"
    third = _register(d3, checkpoints)
    assert (third.outcome, third.historical_overlap) == ("registered", ())

    ledger, _ = read_ledger(ledger_path)
    assert ledger["records"][0] == old
    assert ledger["schema_version"] == 4 and ledger["run_identity_from_sequence"] == 2


@pytest.mark.parametrize("boundary", ["absent", "x", 0, 99], ids=[
    "missing", "not-an-integer", "below-1", "above-next"])
def test_a_ledger_whose_boundary_is_lost_still_reads_and_refuses_only_the_unsure_write(
    tmp_path, checkpoints, boundary
):
    """A8 (fact-check of revision 6) through the registration: a schema 4
    ledger whose run-identity boundary is lost while it holds records. A run no
    record names is refused naming the field, the ledger unchanged, and
    `report_state` reads ``cannot_be_checked`` with the same reason; the run a
    record names with its report answers ``already_registered`` and reads
    ``registered``; the file still reads as a ledger."""
    from src.evaluation.registration import report_state

    named = _report(tmp_path)
    other = _report(tmp_path, batch_size=8)
    first = _register(named, checkpoints)
    data = json.loads(first.ledger_path.read_text())
    if boundary == "absent":
        del data["run_identity_from_sequence"]
    else:
        data["run_identity_from_sequence"] = boundary
    first.ledger_path.write_text(json.dumps(data))
    before = first.ledger_path.read_bytes()

    with pytest.raises(RegistrationRefused) as refused:
        _register(other, checkpoints)
    assert ("the file's run-identity boundary (run_identity_from_sequence) cannot be read"
            in str(refused.value))
    assert first.ledger_path.read_bytes() == before
    state = report_state(other, checkpoint=checkpoints / "best.pt")
    assert state.registered == "cannot_be_checked"
    assert "restore the boundary from a trusted backup" in state.why_registered
    assert _register(named, checkpoints).outcome == "already_registered"
    assert report_state(named, checkpoint=checkpoints / "best.pt").registered == "registered"
    assert read_ledger(first.ledger_path)[0]["records"][0]["run_id"] == named.parent.name


@pytest.mark.parametrize("lost", [("registration", "run_id"), ("registration",)],
                         ids=["sequence-and-run", "sequence-only"])
def test_a_record_without_its_sequence_reads_as_the_writer_answers(tmp_path, checkpoints,
                                                                     lost):
    """A8 (review of K8-6r.2): run one's record lost its registration
    sequence (and, in the first case, its run). A run no intact record names
    is refused with that record named, and `report_state` reads
    ``cannot_be_checked`` with the same reason (before the fix: ``not_registered``,
    "holds no record naming this run", while its registration was refused). A
    run whose intact record is found reads ``registered`` and answers
    ``already_registered``, nothing written (before the fix: refused). Run one
    itself, when its record still names it, is this run's damaged identity
    record (clarification 1): refused and ``cannot_be_checked``, never
    ``already_registered`` as a record with no sequence."""
    from src.evaluation.registration import report_state

    one = _report(tmp_path)
    two = _report(tmp_path, batch_size=8)
    three = _report(tmp_path, batch_size=16)
    first = _register(one, checkpoints)
    _register(two, checkpoints)
    data = json.loads(first.ledger_path.read_text())
    for field in lost:
        del data["records"][0][field]
    first.ledger_path.write_text(json.dumps(data))
    before = first.ledger_path.read_bytes()
    best = checkpoints / "best.pt"
    why = "the ledger's record 0 has no registration sequence"

    for report in ([one, three] if "run_id" in lost else [three]):
        state = report_state(report, checkpoint=best)
        assert state.registered == "cannot_be_checked" and why in state.why_registered
        with pytest.raises(RegistrationRefused, match=why):
            _register(report, checkpoints)
    if "run_id" not in lost:
        damaged = "names run .* but its registration sequence cannot be read"
        state = report_state(one, checkpoint=best)
        assert state.registered == "cannot_be_checked"
        assert re.search(damaged, state.why_registered)
        with pytest.raises(RegistrationRefused, match=damaged):
            _register(one, checkpoints)
    for report in [two]:
        assert report_state(report, checkpoint=best).registered == "registered"
        assert _register(report, checkpoints).outcome == "already_registered"
    assert first.ledger_path.read_bytes() == before


@POSIX
def test_two_byte_identical_runs_registered_at_once_by_two_processes_are_two_records(
    tmp_path, checkpoints
):
    """A9 (with K8-6r.1's harness): two processes register two runs whose
    reports are byte-identical, at once: two records, each with its run.
    HEAD: one ``registered``, the other ``already_registered``."""
    one = _report(tmp_path)
    two = _again_in_another_run(one)

    answers = _register_at_once(tmp_path, checkpoints, [one, two])

    assert sorted((a["outcome"], a["sequence"]) for a in answers) == [
        ("registered", 1), ("registered", 2)]
    assert {a["run_id"] for a in answers} == {one.parent.name, two.parent.name}
    ledger, _ = read_ledger(checkpoints / "evaluations.json")
    assert sorted(r["run_id"] for r in ledger["records"]) == sorted(
        [one.parent.name, two.parent.name])


@pytest.mark.parametrize("swap", [False, True], ids=["historical-first", "named-first"])
def test_this_runs_record_is_found_before_any_old_merged_record_holding_its_bytes(
    tmp_path, checkpoints, swap
):
    """A10 (REVIEW-P5 §3.1): record 1 is historical, holding D1 with
    corroborating [D2]; record 2 names run R1 with D3. R1's report offered as
    D2, then as D1: refused as a replaced report naming record 2, the ledger
    unchanged, and `report_state` reads ``not_registered`` with that reason;
    R1 resent as D3 answers ``already_registered`` with record 2 and reads
    ``registered``. (e) The same with record 2 placed first. HEAD: no run
    identity; all three ``already_registered`` by digest."""
    from src.evaluation.registration import report_state

    d1 = _report(tmp_path)
    d2 = _report(tmp_path, data_dir="/elsewhere")
    r1 = _report(tmp_path, data_dir="/third")
    d3 = r1.read_bytes()
    ledger_path = checkpoints / "evaluations.json"
    _historical(ledger_path, d1, [_sha(d2)])
    named = _register(r1, checkpoints)
    assert named.record["registration"]["sequence"] == 2
    if swap:
        data = json.loads(ledger_path.read_text())
        data["records"].reverse()
        ledger_path.write_text(json.dumps(data))
    before = ledger_path.read_bytes()

    for other in (d2, d1):
        r1.write_bytes(other.read_bytes())
        with pytest.raises(RegistrationRefused) as refused:
            _register(r1, checkpoints)
        assert (f"run {r1.parent.name} is already registered as record 2 from report "
                f"{named.record['source_artifact_digest'][:12]}...") in str(refused.value)
        state = report_state(r1, checkpoint=checkpoints / "best.pt")
        assert state.registered == "not_registered" and "sealed" in state.why_registered
        assert ledger_path.read_bytes() == before
    r1.write_bytes(d3)
    again = _register(r1, checkpoints)
    assert again.outcome == "already_registered" and again.record == named.record
    state = report_state(r1, checkpoint=checkpoints / "best.pt")
    assert (state.registered, state.sequence) == ("registered", 2)
    assert ledger_path.read_bytes() == before


@pytest.mark.parametrize("usage", ["entry", "absent", "unreadable", "other-entry"])
def test_a_run_registered_before_run_identity_and_sent_again_gets_a_named_record(
    tmp_path, checkpoints, usage
):
    """A11 (the trade-off, pinned): a run registered under schema 3 (record 1
    holds its D1, no run) sent again from its own run directory is registered
    as record 2 naming the run, its note naming record 1 as a possible
    overlap; sent again it answers ``already_registered`` from record 2;
    record 1 is unchanged. With the version's usage history holding a
    ``registered`` entry for that run, record 1's sequence and D1, the note
    cites it; with the usage file absent, unreadable or without the entry, the
    outcome and the record are the same. HEAD: ``already_registered``."""
    from src.evaluation.report_store import USAGE_FILE, record_registration

    report = _report(tmp_path)
    old = _historical(checkpoints / "evaluations.json", report)
    usage_path = version_of(report).directory / "reports" / USAGE_FILE
    if usage == "entry":
        record_registration(version_of(report), run_id=report.parent.name,
                            report_sha256=_sha(report), checkpoint_sha256=DIGEST,
                            outcome="registered", sequence=1)
    elif usage == "unreadable":
        usage_path.write_bytes(b"\xff\xfe not utf-8\n")
    elif usage == "other-entry":
        record_registration(version_of(report), run_id=report.parent.name,
                            report_sha256=_sha(report), checkpoint_sha256=DIGEST,
                            outcome="registered", sequence=7)

    result = _register(report, checkpoints)

    assert (result.outcome, result.record["registration"]["sequence"],
            result.record["run_id"]) == ("registered", 2, report.parent.name)
    (note,) = result.historical_overlap
    assert note.startswith("possible overlap with record 1, merged before each run had its "
                           "own record: not recorded whether this run is one of them")
    cited = (f" (the usage history names run {report.parent.name}'s registration into "
             "record 1)")
    assert note.endswith(cited) == (usage == "entry")
    again = _register(report, checkpoints)
    assert again.outcome == "already_registered" and again.record == result.record
    assert read_ledger(checkpoints / "evaluations.json")[0]["records"][0] == old


def test_a_ledger_a_registration_creates_starts_at_schema_4_with_boundary_1(
    tmp_path, checkpoints
):
    """A12 (c) and (g): no file, or the empty file `write_ledger(path,
    empty_ledger(), None)` writes (no boundary on disk, not damaged): the first
    registration answers ``registered`` with sequence 1 and writes schema 4
    with `run_identity_from_sequence` 1."""
    other_dir = tmp_path / "other" / "checkpoints"
    other_dir.mkdir(parents=True)
    (other_dir / "best.pt").write_bytes(WEIGHTS)
    write_ledger(other_dir / "evaluations.json", empty_ledger(), None)
    assert "run_identity_from_sequence" not in json.loads(
        (other_dir / "evaluations.json").read_text())

    for directory in (checkpoints, other_dir):
        result = _register(_report(tmp_path), directory)
        on_disk = json.loads(result.ledger_path.read_text())
        assert (result.outcome, result.record["registration"]["sequence"]) == ("registered", 1)
        assert (on_disk["schema_version"], on_disk["run_identity_from_sequence"]) == (4, 1)


def test_reading_a_schema_3_ledger_through_report_state_leaves_its_bytes(tmp_path, checkpoints):
    """A12 (a) (guard): reading never rewrites a file."""
    from src.evaluation.registration import report_state

    report = _report(tmp_path)
    _historical(checkpoints / "evaluations.json", report)
    before = (checkpoints / "evaluations.json").read_bytes()

    report_state(report, checkpoint=checkpoints / "best.pt")
    read_ledger(checkpoints / "evaluations.json")

    assert (checkpoints / "evaluations.json").read_bytes() == before


@pytest.mark.parametrize("overrides", [{"batch_size": 8}, {"deterministic_algorithms": False}],
                         ids=["batch-size", "deterministic-setting"])
def test_changed_conditions_register_as_they_are(tmp_path, checkpoints, overrides):
    """B1 (guard): another batch size or deterministic setting is its own
    record, with its values as reported. (Another cohort version or checkpoint:
    `test_evaluation_sidecar`'s separate-measurement tests.)"""
    first = _register(_report(tmp_path), checkpoints)
    other = _register(_report(tmp_path, **overrides), checkpoints)

    assert (first.outcome, other.outcome) == ("registered", "registered")
    assert other.record["regime"][next(iter(overrides))] == next(iter(overrides.values()))
    assert len(read_ledger(other.ledger_path)[0]["records"]) == 2


def test_a_seed_change_the_manifest_does_not_record_is_one_group_two_records(
    tmp_path, checkpoints
):
    """B2: two runs whose manifests differ in nothing recorded (a seed the
    manifest does not record) and whose metrics differ: the same group, each
    its own record. HEAD: the second refused, as in A1."""
    from src.evaluation.sidecar import record_key

    one = _register(_report(tmp_path), checkpoints)
    two = _register(_report(tmp_path, metrics={**DISEASE_METRICS, "pooled_top1": 0.3}),
                    checkpoints)

    assert record_key(one.record) == record_key(two.record)
    assert (one.outcome, two.outcome) == ("registered", "registered")
    assert two.group_differences[0].differing == ("pooled_top1",)


def test_a_same_group_report_this_revision_cannot_compare_is_a_note_never_a_block(
    tmp_path, checkpoints
):
    """C2: a report of the same group whose manifest has a field the current
    format lacks. The complete run registers; the group's description names
    that file as ``cannot_be_checked``. HEAD: refused, its disagreement not checked."""
    from src.evaluation.report_store import scan_report_store

    report = _report(tmp_path)
    odd = json.loads(report.read_text())
    odd["manifest"]["retired_field"] = 1
    other = store_bytes(tmp_path / "registry", json.dumps(odd).encode(),
                        version=version_of(report), checkpoint_sha256=DIGEST)

    result = _register(report, checkpoints)

    assert result.outcome == "registered"
    scan = scan_report_store(version_of(report))
    status = scan.run_status(report.parent.name, _sha(report))
    assert status.state == "cannot_be_checked"
    (problem,) = status.problems
    assert str(other) in problem and "retired_field" in problem


def test_a_link_a_run_directory_out_of_the_store_or_a_deep_report_never_blocks(
    tmp_path, checkpoints
):
    """C3: a report.json that is a link, a run directory a link takes out of
    the store, and another run's report too deep to build: the complete run
    registers, and each of the three is listed by its own run with its own
    reason. HEAD: refused, its disagreement not checked."""
    from src.evaluation.report_store import scan_report_store
    from tests.fixtures.report_store import new_run_id

    report = _report(tmp_path)
    reports = report.parents[1]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "report.json").write_bytes(report.read_bytes())
    linked_file = reports / new_run_id("1" * 64)
    linked_file.mkdir()
    (linked_file / "report.json").symlink_to(elsewhere / "report.json")
    linked_dir = reports / new_run_id("2" * 64)
    linked_dir.symlink_to(elsewhere, target_is_directory=True)
    deep = store_bytes(tmp_path / "registry", ("[" * 80 + "]" * 80).encode(),
                       version=version_of(report), checkpoint_sha256="3" * 64)

    assert _register(report, checkpoints).outcome == "registered"

    scan = scan_report_store(version_of(report))
    assert scan.run_status(report.parent.name, _sha(report)).state == "uncontested"
    for run in (linked_file.name, linked_dir.name, deep.parent.name):
        status = scan.run_status(run, scan.digests.get(run))
        assert status.state == "cannot_be_checked" and status.run_ids == (run,), run
        assert len(status.problems) == 1, status.problems


def test_a_store_that_cannot_be_listed_never_stops_a_named_verified_report(
    tmp_path, checkpoints, monkeypatch
):
    """C5 (Decision 10): the store's listing fails. The command line's
    registration of an explicitly named report it verifies reads only that
    report and registers; the store's description of every group, and of each
    run, says the store could not be listed. HEAD: the registration's scan
    made it a refusal, its disagreement not checked."""
    from pathlib import Path as _Path

    from src.evaluation.report_store import scan_report_store

    report = _report(tmp_path)
    real = _Path.iterdir

    def unlistable(self):
        if self.name == "reports":
            raise PermissionError(13, "Permission denied", str(self))
        return real(self)

    monkeypatch.setattr(_Path, "iterdir", unlistable)

    result = register_report(report, checkpoint=checkpoints / "best.pt", registered_at=WHEN)

    assert result.outcome == "registered"
    scan = scan_report_store(version_of(report))
    from src.evaluation.sidecar import record_key

    status = scan.status(record_key(result.record))
    assert status.state == "cannot_be_checked"
    assert "could not be listed" in status.problems[0]
    run = scan.run_status(report.parent.name, _sha(report))
    assert run.state == "cannot_be_checked" and "could not be listed" in run.problems[0]


# ---------------------------------------------------------------------------
# K8-6r.2's surfaces: the CLIs' notes, `--show`, the job summary
# ---------------------------------------------------------------------------
def test_record_evaluation_prints_the_group_note_and_the_historical_note(
    tmp_path, checkpoints, caplog
):
    """G4, A7 (K8-6r.2) through record_evaluation: a run whose values differ
    from a registered run of its group prints "Registered record 2" and the
    group note, which names record 1, its run and what differs; a run whose
    bytes a record merged before each run had its own record holds prints the
    historical note, never "already holds". Both are information: each run is
    registered. HEAD: the differing run was refused ("runs disagree") and the
    merged bytes answered "already holds this report"."""
    import logging

    import scripts.record_evaluation as record_evaluation

    first = _report(tmp_path)
    assert record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                                   "--report", str(first)]) == 0
    differing = _report(tmp_path, metrics={**DISEASE_METRICS, "pooled_mrr": 0.5001})
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=record_evaluation.logger.name):
        assert record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                                       "--report", str(differing)]) == 0
    said = caplog.text
    assert "Registered record 2" in said
    assert (f"Runs under these conditions differ: record 2 (this run) differs from record 1 "
            f"(run {first.parent.name}) on pooled_mrr. Each run is kept as recorded.") in said

    merged = _report(tmp_path, data_dir="/elsewhere")
    old_checkpoints = tmp_path / "old-ledger"
    _historical(old_checkpoints / "evaluations.json", merged)
    (old_checkpoints / "best.pt").write_bytes(WEIGHTS)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=record_evaluation.logger.name):
        assert record_evaluation.main(["--checkpoint-dir", str(old_checkpoints),
                                       "--report", str(merged)]) == 0
    said = caplog.text
    assert "Registered record 2" in said and "already holds" not in said
    assert ("Possible overlap with record 1, merged before each run had its own record: not "
            "recorded whether this run is one of them") in said


def test_show_prints_each_records_run_and_the_possible_overlaps(tmp_path, checkpoints, caplog):
    """G8, A12 (a), A8 (K8-6r.2): `record_evaluation --show` prints every record
    as written (JSON) and, for each, its run: "not recorded" for a record
    merged before each run had its own record, with the possible overlap on
    both sides; reading a schema 3 file leaves its bytes; a record whose run
    identity cannot be read is said with its reason, never as historical.
    HEAD: no run or overlap was said."""
    import logging

    import scripts.record_evaluation as record_evaluation

    d1 = _report(tmp_path)
    d2 = _again_in_another_run(d1)
    ledger_path = checkpoints / "evaluations.json"
    _historical(ledger_path, d1, [_sha(d2)])
    before = ledger_path.read_bytes()
    with caplog.at_level(logging.INFO, logger=record_evaluation.logger.name):
        assert record_evaluation.main(["--checkpoint-dir", str(checkpoints), "--show",
                                       DIGEST]) == 0
    assert ledger_path.read_bytes() == before, "reading a schema 3 file never rewrites it"
    assert "record 1: run not recorded (registered before each run had its own record)" in (
        caplog.text)

    _register(d2, checkpoints)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=record_evaluation.logger.name):
        assert record_evaluation.main(["--checkpoint-dir", str(checkpoints), "--show",
                                       DIGEST]) == 0
    said = caplog.text
    assert ("record 1: run not recorded (registered before each run had its own record); "
            "possible overlap with record 2 (not recorded)") in said
    assert (f"record 2: run {d2.parent.name}; possible overlap with record 1 "
            "(not recorded)") in said

    data = json.loads(ledger_path.read_text())
    data["records"][1].pop("run_id")
    data["records"][0]["metrics"] = "/srv/x/case-0042"
    ledger_path.write_text(json.dumps(data))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=record_evaluation.logger.name):
        assert record_evaluation.main(["--checkpoint-dir", str(checkpoints), "--show",
                                       DIGEST]) == 0
    # K8-6e.1: each record the readers' one listing cannot read is said by its
    # index with that listing's reason, never as a result and never with the
    # value; the JSON still prints both as written. HEAD: the identity line
    # only, and the record with malformed metrics read like a result.
    assert ("record 2 (the ledger's record 1) cannot be read as a result: its run identity "
            "is missing") in caplog.text
    assert ("record 1 (the ledger's record 0) cannot be read as a result: metrics: Input "
            "should be a valid dictionary") in caplog.text
    assert "case-0042" not in caplog.text


@pytest.mark.parametrize("case", ["another run's byte-identical report", "an old merged record"])
def test_the_job_summary_reads_a_run_registered_only_by_its_own_record(
    tmp_path, checkpoints, case
):
    """G5 (K8-6r.2), with the real `report_state`: a run whose byte-identical
    report another run registered reads not registered until it is registered
    itself; a run whose bytes only a record merged before each run had its own
    record holds reads not registered, the summary carrying the possible
    overlap, never "it was registered". HEAD: both read "it was registered"."""
    from types import SimpleNamespace

    from src.api.services.training_manager import TrainingManager

    first = _report(tmp_path)
    run = _again_in_another_run(first)
    if case == "another run's byte-identical report":
        _register(first, checkpoints)
    else:
        _historical(checkpoints / "evaluations.json", first)
    manager = TrainingManager()
    job = SimpleNamespace(inputs={"checkpoint": str(checkpoints / "best.pt"), "record": False},
                          stop_signal=None)

    said = manager._report_outcome(job, run, _sha(run), {"complete": True}, {})

    assert said["kept"] is True and said["registered"] is False
    assert "it was not registered" in said["summary"]
    assert "it was registered" not in said["summary"]
    if case == "an old merged record":
        assert ("its bytes equal a report merged into record 1 before each run had its own "
                "record; registering it adds its own record") in said["summary"]
        from src.evaluation.registration import report_state

        state = report_state(run, checkpoint=checkpoints / "best.pt")
        assert (state.registered, state.historical_overlap) == ("not_registered", (1,))
    _register(run, checkpoints)
    again = manager._report_outcome(job, run, _sha(run), {"complete": True}, {})
    assert again["registered"] is True and "it was registered" in again["summary"]
