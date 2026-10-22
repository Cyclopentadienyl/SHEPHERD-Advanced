"""The report store (M5-1): disagreement by key, notes and the usage history.

`src/evaluation/report_store.py` (`PLAN_TEST_RESULTS.md` D4, D5 as amended
2026-10-19). What is pinned here, on reports built by `MeasurementManifest` and
kept in a store by `tests/fixtures/report_store.py`:

- **a key's state** comes from the ledger's own comparison (`differing_fields`):
  agreeing reports (and byte-identical copies) are uncontested, any compared
  field that differs is contested and named, R10 and the environment are not
  compared; a key no report is under is uncontested;
- **"cannot be checked", never uncontested**: a report that cannot be read, a
  report whose key cannot be told, a run directory or ``report.json`` that is a
  link out of the store leave every key unable to be checked; a report that
  parses but cannot be compared leaves only its own key so; an interrupted run
  directory (no ``report.json``) and entries that are not run directories are
  passed over;
- a report the caller already read is not read again (`known`);
- **investigation notes** are appended with their key, only for a key the store
  holds, and change no state;
- **the usage history** is append-only, one JSON line per entry, aggregates
  only, refuses a link in place of the file, and names a damaged line; an
  entry after an interrupted append starts its own line, and a failed write is
  taken back (or says it could not be);
- a report too deeply nested to build leaves its key "cannot be checked";
- the registry's helpers it rests on: run ids, the store directory's checks.

The registration's use of all this is in `tests/unit/test_registration.py`.

Module: tests/unit/test_report_store.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.evaluation.report_store import (
    CONTEST_STATES,
    USAGE_KINDS,
    add_investigation_note,
    read_investigation_notes,
    read_usage_history,
    record_model_load,
    record_registration,
    record_test_run,
    scan_report_store,
)
from src.evaluation.sidecar import build_record, record_key
from tests.fixtures.report_store import new_run_id, store_bytes, store_report, version_of
from tests.unit.test_evaluation_sidecar import environment_block, r10_block, registered_report

WHEN = datetime(2026, 10, 20, 9, 0, 0, tzinfo=timezone.utc)
OTHER_METRICS = {"untruncated_mrr": 0.9, "mean_rank": 1.5, "hits@1": 0.25, "hits@10": 0.75}


def _stored(tmp_path, **overrides) -> Path:
    return store_report(tmp_path / "registry", registered_report(**overrides))


def _key(path: Path):
    data = path.read_bytes()
    return record_key(build_record(json.loads(data), hashlib.sha256(data).hexdigest()))


# ---------------------------------------------------------------------------
# A key's state
# ---------------------------------------------------------------------------
def test_the_states_are_the_three_d5_names():
    assert CONTEST_STATES == ("uncontested", "contested", "cannot_be_checked")


def test_agreeing_reports_are_uncontested_whatever_their_r10_and_environment(tmp_path):
    """Corroboration in the ledger is agreement in the store: R10 and the
    environment are not compared; paths differ; a byte-identical copy is one
    report."""
    first = _stored(tmp_path)
    second = _stored(tmp_path, data_dir="/elsewhere", r10=r10_block(training_state="overlap"),
                     environment=environment_block(driver="999.0"))
    copy = first.parents[1] / new_run_id("c" * 64) / "report.json"
    copy.parent.mkdir()
    copy.write_bytes(first.read_bytes())

    status = scan_report_store(version_of(first)).status(_key(first))

    assert status.state == "uncontested"
    assert status.run_ids == tuple(sorted([first.parent.name, second.parent.name,
                                           copy.parent.name])), "every run, copies included"
    assert status.differing == () and status.problems == ()


def test_reports_that_differ_on_a_compared_field_are_contested_and_named(tmp_path):
    first = _stored(tmp_path)
    second = _stored(tmp_path, metrics=OTHER_METRICS)
    other_key = _stored(tmp_path, batch_size=8, metrics=OTHER_METRICS)

    scan = scan_report_store(version_of(first))
    status = scan.status(_key(first))

    assert status.state == scan.state(_key(first)) == "contested"
    assert status.differing == ("mean_rank", "untruncated_mrr")
    assert status.disagreeing_runs == tuple(sorted([first.parent.name, second.parent.name]))
    assert scan.state(_key(other_key)) == "uncontested", "another batch size is another key"


def test_a_difference_outside_the_metrics_is_named_by_its_field(tmp_path):
    first = _stored(tmp_path)
    _stored(tmp_path, n_ranked=79)

    status = scan_report_store(version_of(first)).status(_key(first))

    assert status.state == "contested" and status.differing == ("n_ranked",)


def test_a_key_no_report_is_under_is_uncontested(tmp_path):
    first = _stored(tmp_path)

    status = scan_report_store(version_of(first)).status(("a" * 64, "r", "b" * 64, "c" * 64))

    assert (status.state, status.run_ids) == ("uncontested", ())


@pytest.mark.parametrize("damage", ["not_json", "not_a_report", "unreadable_run_dir_link"])
def test_a_report_whose_key_cannot_be_told_leaves_every_key_unable_to_be_checked(
    tmp_path, damage
):
    first = _stored(tmp_path)
    version = version_of(first)
    if damage == "unreadable_run_dir_link":
        outside = tmp_path / "outside-run"
        outside.mkdir()
        (outside / "report.json").write_bytes(first.read_bytes())
        link = first.parents[1] / new_run_id("c" * 64)
        link.symlink_to(outside, target_is_directory=True)
        named = str(link)
    else:
        named = str(store_bytes(tmp_path / "registry",
                                b"{broken" if damage == "not_json" else b'{"hello": 1}',
                                version=version))

    status = scan_report_store(version).status(_key(first))

    assert status.state == "cannot_be_checked"
    assert any(named in problem for problem in status.problems), status.problems


def test_a_report_json_that_is_a_link_cannot_be_checked(tmp_path):
    first = _stored(tmp_path)
    outside = tmp_path / "elsewhere.json"
    outside.write_bytes(_stored(tmp_path, metrics=OTHER_METRICS).read_bytes())
    linked = first.parents[1] / new_run_id("c" * 64) / "report.json"
    linked.parent.mkdir()
    linked.symlink_to(outside)

    scan = scan_report_store(version_of(first))

    assert scan.state(_key(first)) == "contested", "the second stored report still disagrees"
    assert any(str(linked) in p and "is a link" in p for _run, p in scan.unplaced)
    other_key = scan.status(("a" * 64, "r", "b" * 64, "c" * 64))
    assert other_key.state == "cannot_be_checked", "the link could be under any key"


def test_a_report_that_parses_but_cannot_be_compared_affects_only_its_own_key(tmp_path):
    """A report from an earlier revision (a field this one does not record)
    names its key; only that key cannot be checked."""
    first = _stored(tmp_path)
    old = registered_report()
    old["manifest"]["python_seed"] = 0
    stale = store_report(tmp_path / "registry", old)
    other_key = _stored(tmp_path, batch_size=8)

    scan = scan_report_store(version_of(first))

    assert scan.state(_key(first)) == "cannot_be_checked"
    assert any(str(stale) in p and "python_seed" in p
               for p in scan.status(_key(first)).problems)
    assert scan.state(_key(other_key)) == "uncontested"


def test_every_report_the_scan_read_is_named_by_its_run_id(tmp_path):
    """EB-3: a report that cannot be compared and one whose key cannot be told
    are named by their run ids, so a run's state is found by its id
    (`run_key`, `run_status`), never read out of a problem's words: the one
    built is under its key; the one that cannot be compared under its key with
    its own problem; the one whose key cannot be told has no key and cannot be
    checked; a run directory with no report has no state."""
    first = _stored(tmp_path)
    version = version_of(first)
    old = registered_report()
    old["manifest"]["python_seed"] = 0
    stale = store_report(tmp_path / "registry", old)
    broken = store_bytes(tmp_path / "registry", b"{broken", version=version)
    interrupted = first.parents[1] / new_run_id("c" * 64)
    interrupted.mkdir()

    scan = scan_report_store(version)

    key = _key(first)
    (unbuilt,) = scan.unbuilt[key]
    assert unbuilt[0] == stale.parent.name and "python_seed" in unbuilt[1]
    (unplaced,) = scan.unplaced
    assert unplaced[0] == broken.parent.name and str(broken) in unplaced[1]
    assert scan.run_key(first.parent.name) == (key, None)
    assert scan.run_key(stale.parent.name) == (key, unbuilt[1])
    assert scan.run_key(broken.parent.name) == (None, unplaced[1])
    assert scan.run_key(interrupted.name) is None and scan.run_status(interrupted.name) is None
    assert scan.run_status(first.parent.name) == scan.status(key)
    assert scan.run_status(stale.parent.name).state == "cannot_be_checked"
    lost = scan.run_status(broken.parent.name)
    assert (lost.state, lost.run_ids, lost.problems) == (
        "cannot_be_checked", (broken.parent.name,), (unplaced[1],))


def test_a_report_nested_too_deeply_to_copy_cannot_be_checked_and_does_not_crash(tmp_path):
    """The reviewer's P3 (2026-10-20): a hand-edited report that parses but
    nests deeper than `build_record` can copy is a report that cannot be
    compared -- its key cannot be checked -- not a crash of every scan."""
    first = _stored(tmp_path)
    stored = _stored(tmp_path)
    deep = json.loads(stored.read_text())
    node = deep["manifest"]["r10"]
    for _ in range(600):
        node["x"] = {}
        node = node["x"]
    stored.write_text(json.dumps(deep))

    status = scan_report_store(version_of(first)).status(_key(first))

    assert status.state == "cannot_be_checked"
    assert any(str(stored) in p and "nests too deeply" in p for p in status.problems)


def test_a_manifest_too_deep_to_read_a_key_from_gives_no_key(monkeypatch):
    import src.evaluation.sidecar as sidecar

    def too_deep(manifest):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(sidecar, "measurement_semantics_digest", too_deep)
    assert sidecar.report_key(registered_report()) is None


def test_an_interrupted_run_and_entries_that_are_not_runs_are_passed_over(tmp_path):
    first = _stored(tmp_path)
    reports = first.parents[1]
    (reports / new_run_id("c" * 64)).mkdir()                    # interrupted writes
    interrupted = reports / new_run_id("c" * 64)
    interrupted.mkdir()
    (interrupted / "ranks.json").write_text("[]")
    (reports / "backup").mkdir()
    (reports / "backup" / "report.json").write_text("{broken")   # not a run directory
    (reports / "usage.jsonl").write_text("")

    assert scan_report_store(version_of(first)).state(_key(first)) == "uncontested"


def test_a_version_with_no_report_store_has_no_reports(tmp_path):
    from tests.fixtures.report_store import store_version
    from tests.unit.test_evaluation_sidecar import cohort_version

    version = store_version(tmp_path / "registry", cohort_version())

    scan = scan_report_store(version)

    assert scan.reports == {} and scan.unplaced == ()


def test_a_report_the_caller_read_is_not_read_again(tmp_path):
    from tests.fixtures.opens import count_opens

    first = _stored(tmp_path)
    second = _stored(tmp_path, metrics=OTHER_METRICS)
    data = first.read_bytes()
    record = build_record(json.loads(data), hashlib.sha256(data).hexdigest())

    with count_opens() as opens:
        status = scan_report_store(version_of(first),
                                   known=(first.parent.name, record)).status(record_key(record))

    assert opens.opens(first) == 0 and opens.opens(second) == 1
    assert status.state == "contested"


# ---------------------------------------------------------------------------
# Investigation notes
# ---------------------------------------------------------------------------
def test_a_note_is_kept_with_its_key_and_changes_no_state(tmp_path):
    first = _stored(tmp_path)
    _stored(tmp_path, metrics=OTHER_METRICS)
    version, key = version_of(first), _key(first)

    entry = add_investigation_note(version, key, who=" technician ", what=" driver 560 vs 565 ",
                                   at=WHEN)

    assert entry == {"kind": "investigation_note",
                     "key": {"checkpoint_digest": key[0], "cohort_role": key[1],
                             "cohort_digest": key[2], "measurement_semantics_digest": key[3]},
                     "who": "technician", "at": "2026-10-20T09:00:00Z",
                     "what": "driver 560 vs 565"}
    notes, problems = read_investigation_notes(version, key)
    assert notes == [entry] and problems == []
    assert read_investigation_notes(version, ("a" * 64, "r", "b" * 64, "c" * 64))[0] == []
    assert scan_report_store(version).state(key) == "contested"
    assert (first.parents[1] / "notes.jsonl").is_file()


@pytest.mark.parametrize("bad", ["no_report_under_key", "empty_who", "empty_what", "bad_key"])
def test_a_note_is_refused_without_its_parts_or_a_report_under_its_key(tmp_path, bad):
    first = _stored(tmp_path)
    version, key = version_of(first), _key(first)
    kwargs = {"who": "t", "what": "w"}
    if bad == "no_report_under_key":
        key = ("a" * 64, "r", "b" * 64, "c" * 64)
    elif bad == "empty_who":
        kwargs["who"] = "  "
    elif bad == "empty_what":
        kwargs["what"] = ""
    else:
        key = ("a" * 64,)

    with pytest.raises(ValueError):
        add_investigation_note(version, key, **kwargs)
    assert not (first.parents[1] / "notes.jsonl").exists()


# ---------------------------------------------------------------------------
# The usage history
# ---------------------------------------------------------------------------
def test_the_usage_history_appends_each_kind_in_order(tmp_path):
    first = _stored(tmp_path)
    version = version_of(first)

    run = record_test_run(version, run_id=first.parent.name, report_sha256="e" * 64,
                          checkpoint_sha256="c" * 64, metrics={"untruncated_mrr": 0.5},
                          n_ranked=80, deterministic=True, device="cuda",
                          record_requested=False, at=WHEN)
    registered = record_registration(version, run_id=first.parent.name,
                                     report_sha256="e" * 64, checkpoint_sha256="c" * 64,
                                     outcome="registered", sequence=1, at=WHEN)
    load = record_model_load(version, checkpoint_sha256="c" * 64, metric="untruncated_mrr",
                             scoring_semantics_digest="s" * 64, loaded=False,
                             why="the reload was refused", at=WHEN)

    entries, problems = read_usage_history(version)
    assert entries == [run, registered, load] and problems == []
    assert [e["kind"] for e in entries] == list(USAGE_KINDS)
    lines = (first.parents[1] / "usage.jsonl").read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(line) for line in lines)


def test_the_usage_history_holds_aggregates_only(tmp_path):
    first = _stored(tmp_path)
    record_test_run(version_of(first), run_id=first.parent.name, report_sha256="e" * 64,
                    checkpoint_sha256="c" * 64, metrics={"hits@1": 0.25}, n_ranked=80,
                    deterministic=False, device="cpu", record_requested=True, at=WHEN)

    text = (first.parents[1] / "usage.jsonl").read_text()

    for forbidden in ("case-", "patient_id", "phenotype", str(tmp_path)):
        assert forbidden not in text, forbidden


def test_the_usage_history_is_never_truncated_and_names_a_damaged_line(tmp_path):
    first = _stored(tmp_path)
    version = version_of(first)
    path = first.parents[1] / "usage.jsonl"
    path.write_text('{"kind": "test_run"}\nnot json\n[1]\n{"kind": "model_lo')

    record_registration(version, run_id=first.parent.name, report_sha256="e" * 64,
                        checkpoint_sha256="c" * 64, outcome="registered", sequence=1, at=WHEN)
    entries, problems = read_usage_history(version)

    assert path.read_text().startswith(
        '{"kind": "test_run"}\nnot json\n[1]\n{"kind": "model_lo\n{')
    # The unfinished line an interrupted append left is closed, not continued:
    # the entry appended after it is on a line of its own and is read.
    assert [e["kind"] for e in entries] == ["test_run", "registered"]
    assert [p.split(" line ")[1] for p in problems] == [
        "2 is not a JSON object", "3 is not a JSON object", "4 is not a JSON object"]


def test_an_entry_after_an_interrupted_append_starts_its_own_line(tmp_path):
    """The reviewer's P3 (2026-10-20): a torn append (a crash, a full disk)
    leaves the file without its last newline; the next entry is not lost to it,
    and the torn line is named."""
    first = _stored(tmp_path)
    version = version_of(first)
    record_test_run(version, run_id=first.parent.name, report_sha256="e" * 64,
                    checkpoint_sha256="c" * 64, metrics={"hits@1": 0.25}, n_ranked=80,
                    deterministic=False, device="cpu", record_requested=False, at=WHEN)
    path = first.parents[1] / "usage.jsonl"
    with open(path, "ab") as handle:
        handle.write(b'{"kind":"test_run","at":"2026')

    load = record_model_load(version, checkpoint_sha256="a" * 64, metric="hits@1",
                             scoring_semantics_digest="b" * 64, loaded=True, at=WHEN)
    entries, problems = read_usage_history(version)

    assert [e["kind"] for e in entries] == ["test_run", "model_load"]
    assert entries[1] == load
    assert len(problems) == 1 and "line 2 is not a JSON object" in problems[0]
    assert path.read_bytes().endswith(b'"at":"2026\n' + json.dumps(
        load, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def _failing_write_to(path, monkeypatch, *, after: int):
    """Make `os.write` to `path` write `after` bytes and then fail, as a full
    disk does; every other descriptor is written as usual."""
    import os

    real_write = os.write
    inode = path.stat().st_ino

    def write(fd, data):
        if os.fstat(fd).st_ino != inode:
            return real_write(fd, data)
        if after:
            real_write(fd, bytes(data[:after]))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "write", write)


def test_a_failed_append_is_taken_back_leaving_no_part_of_its_entry(tmp_path, monkeypatch):
    first = _stored(tmp_path)
    version = version_of(first)
    record_registration(version, run_id=first.parent.name, report_sha256="e" * 64,
                        checkpoint_sha256="c" * 64, outcome="registered", sequence=1, at=WHEN)
    path = first.parents[1] / "usage.jsonl"
    before = path.read_bytes()
    _failing_write_to(path, monkeypatch, after=10)

    with pytest.raises(OSError, match="No space left on device") as failed:
        record_model_load(version, checkpoint_sha256="a" * 64, metric="hits@1",
                          scoring_semantics_digest="b" * 64, loaded=True, at=WHEN)
    monkeypatch.undo()

    assert path.read_bytes() == before
    assert "cut back" not in str(failed.value)
    assert read_usage_history(version) == ([json.loads(before)], [])


def test_a_failed_append_that_cannot_be_taken_back_says_so(tmp_path, monkeypatch):
    import os

    first = _stored(tmp_path)
    version = version_of(first)
    record_registration(version, run_id=first.parent.name, report_sha256="e" * 64,
                        checkpoint_sha256="c" * 64, outcome="registered", sequence=1, at=WHEN)
    path = first.parents[1] / "usage.jsonl"
    size = path.stat().st_size
    _failing_write_to(path, monkeypatch, after=10)
    monkeypatch.setattr(os, "ftruncate",
                        lambda fd, length: (_ for _ in ()).throw(OSError(5, "I/O error")))

    with pytest.raises(OSError) as failed:
        record_model_load(version, checkpoint_sha256="a" * 64, metric="hits@1",
                          scoring_semantics_digest="b" * 64, loaded=True, at=WHEN)

    message = str(failed.value)
    assert "No space left on device" in message
    assert f"could not be cut back to the {size} bytes it had" in message
    assert "may hold part of this entry" in message


def test_an_appended_file_that_is_a_link_is_refused(tmp_path):
    from src.evaluation.cohort_registry import RegistryRefused

    first = _stored(tmp_path)
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("")
    (first.parents[1] / "usage.jsonl").symlink_to(target)

    with pytest.raises(RegistryRefused, match="not a plain file"):
        record_registration(version_of(first), run_id=first.parent.name,
                            report_sha256="e" * 64, checkpoint_sha256="c" * 64,
                            outcome="registered", sequence=1, at=WHEN)
    assert target.read_text() == ""


def test_a_time_without_a_zone_is_refused(tmp_path):
    first = _stored(tmp_path)
    with pytest.raises(ValueError, match="no time zone"):
        record_registration(version_of(first), run_id="r", report_sha256="e" * 64,
                            checkpoint_sha256="c" * 64, outcome="registered", sequence=1,
                            at=datetime(2026, 10, 20))


# ---------------------------------------------------------------------------
# The registry's helpers the store rests on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("20261016T120000Z-abcdef012345-0a1b2c3d", True),
    ("20261016T120000Z-ABCDEF012345-0a1b2c3d", False),
    ("20261016T120000Z-abcdef01234-0a1b2c3d", False),
    ("usage.jsonl", False), ("notes.jsonl", False), ("backup", False), (None, False),
])
def test_a_run_id_is_the_name_create_report_dir_gives(name, expected):
    from src.evaluation.cohort_registry import is_run_id

    assert is_run_id(name) is expected


def test_create_report_dir_names_a_run_by_the_pattern(tmp_path):
    from src.evaluation.cohort_registry import RUN_ID_PATTERN, create_report_dir

    first = _stored(tmp_path)
    run_id, run_dir = create_report_dir(version_of(first), "c" * 64, WHEN)

    assert RUN_ID_PATTERN.fullmatch(run_id).group(1) == "c" * 12
    assert run_dir.parent == first.parents[1].resolve()


def test_a_report_store_that_is_a_link_out_of_its_version_is_refused(tmp_path):
    from src.evaluation.cohort_registry import RegistryRefused, report_store_directory
    from tests.fixtures.report_store import store_version
    from tests.unit.test_evaluation_sidecar import cohort_version

    version = store_version(tmp_path / "registry", cohort_version())
    outside = tmp_path / "outside-reports"
    outside.mkdir()
    (version.directory / "reports").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RegistryRefused, match="outside"):
        report_store_directory(version)
    scan = scan_report_store(version)
    assert scan.state(("a" * 64, "r", "b" * 64, "c" * 64)) == "cannot_be_checked"
