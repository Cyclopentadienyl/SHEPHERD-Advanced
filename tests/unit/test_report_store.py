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
- **one append at a time per store file** (K8-6r.1): one writer's failed append
  never cuts another's line, across threads and processes and across two
  ledgers sharing one version's usage history; the lock file is kept, a busy
  lock refuses after the grace, notes and usage history lock apart, and a lock
  file that is a link is refused (by name, and not followed when swapped in);
- a report too deeply nested to build leaves its key "cannot be checked";
- the registry's helpers it rests on: run ids, the store directory's checks.

The registration's use of all this is in `tests/unit/test_registration.py`.

Module: tests/unit/test_report_store.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.evaluation.report_store import (
    CONTEST_STATES,
    USAGE_KINDS,
    add_investigation_note,
    entry_mode,
    read_investigation_notes,
    read_usage_history,
    record_registration,
    record_test_run,
    scan_report_store,
)
from src.evaluation.sidecar import build_record, record_key
from tests.fixtures.report_store import (
    kept_model_load,
    new_run_id,
    store_bytes,
    store_report,
    version_of,
)
from tests.unit.test_evaluation_sidecar import (
    DISEASE_METRICS,
    environment_block,
    r10_block,
    registered_report,
)

WHEN = datetime(2026, 10, 20, 9, 0, 0, tzinfo=timezone.utc)
OTHER_METRICS = {**DISEASE_METRICS, "pooled_mrr": 0.9, "pooled_top1": 0.5}


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
    assert status.differing == ("pooled_mrr", "pooled_top1")
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
    first_sha, stale_sha = _sha(first), _sha(stale)
    assert scan.run_key(first.parent.name, first_sha) == (key, None)
    assert scan.run_key(stale.parent.name, stale_sha) == (key, unbuilt[1])
    assert scan.run_key(broken.parent.name, None) == (None, unplaced[1])
    assert scan.run_key(interrupted.name, None) is None
    assert scan.run_status(interrupted.name, None) is None
    assert scan.run_status(first.parent.name, first_sha) == scan.status(key)
    assert scan.run_status(stale.parent.name, stale_sha).state == "cannot_be_checked"
    lost = scan.run_status(broken.parent.name, None)
    assert (lost.state, lost.run_ids, lost.problems) == (
        "cannot_be_checked", (broken.parent.name,), (unplaced[1],))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("now", ["other bytes", "no report"])
def test_a_lookup_by_run_id_is_bound_to_the_bytes_the_caller_read(tmp_path, now):
    """The verification of EB-3 (R-1): a run's state and its key's state come
    from two reads of its ``report.json`` (the registration's helper's, the
    scan's). A report replaced between them -- by other bytes, or by none the
    caller could read -- is given no key and is ``cannot_be_checked``, naming
    both reads: never the scan's key or state for bytes the caller did not
    read. The bytes the scan read are its own control."""
    first = _stored(tmp_path)
    scan = scan_report_store(version_of(first))
    read = None if now == "no report" else hashlib.sha256(b"other bytes").hexdigest()

    assert scan.digests == {first.parent.name: _sha(first)}
    assert scan.run_key(first.parent.name, _sha(first)) == (_key(first), None)
    assert scan.run_status(first.parent.name, _sha(first)).state == "uncontested"
    assert scan.run_key(first.parent.name, read) is None
    replaced = scan.run_status(first.parent.name, read)
    assert (replaced.state, replaced.run_ids) == ("cannot_be_checked", (first.parent.name,))
    (problem,) = replaced.problems
    assert _sha(first)[:12] in problem and "replaced between the two reads" in problem
    assert (read[:12] if read else "no report") in problem


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
    """A test run and a registration, then a ``model_load`` line kept from
    before By test score retired (K8-6d; nothing writes one now): all three
    kinds are read back in order."""
    first = _stored(tmp_path)
    version = version_of(first)

    run = record_test_run(version, run_id=first.parent.name, report_sha256="e" * 64,
                          checkpoint_sha256="c" * 64, metrics={"untruncated_mrr": 0.5},
                          n_ranked=80, deterministic=True, device="cuda",
                          record_requested=False, mode="C", at=WHEN)
    registered = record_registration(version, run_id=first.parent.name,
                                     report_sha256="e" * 64, checkpoint_sha256="c" * 64,
                                     outcome="registered", sequence=1, at=WHEN)
    load = kept_model_load(version, checkpoint_sha256="c" * 64, metric="untruncated_mrr",
                           scoring_semantics_digest="s" * 64, loaded=False,
                           why="the reload was refused", at=run["at"])

    entries, problems = read_usage_history(version)
    assert entries == [run, registered, load] and problems == []
    assert [e["kind"] for e in entries] == list(USAGE_KINDS)
    lines = (first.parents[1] / "usage.jsonl").read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(line) for line in lines)


def test_a_test_run_names_its_mode_and_an_entry_from_before_the_key_reads_as_mode_c(tmp_path):
    """`record_test_run` requires the run's mode (K7: no writer default records
    "C" for a mode it was not told; a call without one is a TypeError and
    writes nothing), and `entry_mode` reads it back. An entry written before
    the key existed (Batch 3's M4/M5 registries, every one Mode C) reads as Mode
    C; a key that is present but empty or not a string is no mode, never C."""
    first = _stored(tmp_path)
    version = version_of(first)
    run = dict(run_id=first.parent.name, report_sha256="e" * 64, checkpoint_sha256="c" * 64,
               metrics={"pooled_mrr": 0.25}, n_ranked=8, deterministic=True, device="cpu",
               record_requested=False, at=WHEN)

    with pytest.raises(TypeError, match="mode"):
        record_test_run(version, **run)
    assert read_usage_history(version) == ([], [])
    for mode in ("disease", "C", "D"):
        record_test_run(version, mode=mode, **run)

    entries, problems = read_usage_history(version)
    assert [entry_mode(entry) for entry in entries] == ["disease", "C", "D"] and not problems
    before_the_key = {name: value for name, value in entries[0].items() if name != "mode"}
    assert entry_mode(before_the_key) == "C"
    for damaged in (None, "", " ", 7):
        assert entry_mode({**entries[2], "mode": damaged}) is None


def test_the_usage_history_holds_aggregates_only(tmp_path):
    first = _stored(tmp_path)
    record_test_run(version_of(first), run_id=first.parent.name, report_sha256="e" * 64,
                    checkpoint_sha256="c" * 64, metrics={"hits@1": 0.25}, n_ranked=80,
                    deterministic=False, device="cpu", record_requested=True, mode="C",
                    at=WHEN)

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
                    deterministic=False, device="cpu", record_requested=False, mode="C",
                    at=WHEN)
    path = first.parents[1] / "usage.jsonl"
    with open(path, "ab") as handle:
        handle.write(b'{"kind":"test_run","at":"2026')

    later = record_test_run(version, run_id=first.parent.name, report_sha256="a" * 64,
                            checkpoint_sha256="a" * 64, metrics={"hits@1": 0.5}, n_ranked=80,
                            deterministic=True, device="cpu", record_requested=False, mode="C",
                            at=WHEN)
    entries, problems = read_usage_history(version)

    assert [e["kind"] for e in entries] == ["test_run", "test_run"]
    assert entries[1] == later
    assert len(problems) == 1 and "line 2 is not a JSON object" in problems[0]
    assert path.read_bytes().endswith(b'"at":"2026\n' + json.dumps(
        later, sort_keys=True, separators=(",", ":")).encode() + b"\n")


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
        record_test_run(version, run_id=first.parent.name, report_sha256="a" * 64,
                        checkpoint_sha256="a" * 64, metrics={"hits@1": 0.5}, n_ranked=80,
                        deterministic=True, device="cpu", record_requested=False, mode="C",
                        at=WHEN)
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
        record_test_run(version, run_id=first.parent.name, report_sha256="a" * 64,
                        checkpoint_sha256="a" * 64, metrics={"hits@1": 0.5}, n_ranked=80,
                        deterministic=True, device="cpu", record_requested=False, mode="C",
                        at=WHEN)

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


# ---------------------------------------------------------------------------
# One append at a time per store file (K8-6r.1, Decision 4)
# ---------------------------------------------------------------------------
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="flock and spawn are tested on POSIX")


def _seeded_usage(tmp_path):
    """A version whose usage history holds one entry (run "seed")."""
    first = _stored(tmp_path)
    version = version_of(first)
    record_registration(version, run_id="seed", report_sha256="0" * 64,
                        checkpoint_sha256="0" * 64, outcome="registered", sequence=1, at=WHEN)
    return version, first.parents[1] / "usage.jsonl"


def test_one_writers_failed_append_never_cuts_another_writers_line_threads(
    tmp_path, monkeypatch
):
    """F-g, two threads: writer A has read the file's size when writer B
    appends to the same ``usage.jsonl``; A's write then fails (ENOSPC, after
    part of its entry) and A undoes it. With the lock B waits until A has cut
    back its own part and let go, then appends: the file ends with B's line
    whole and nothing of A's, and A's error names the failed write. (Before
    K8-6r.1, B returned success and A's undo cut B's line.)"""
    import os
    import threading

    version, usage = _seeded_usage(tmp_path)
    seeded = usage.read_bytes()
    inode = usage.stat().st_ino
    sized, b_done = threading.Event(), threading.Event()
    real_write, real_truncate = os.write, os.ftruncate
    a_thread: list = []

    def ours(fd):
        return threading.current_thread() in a_thread and os.fstat(fd).st_ino == inode

    def write(fd, data):
        if not ours(fd):
            return real_write(fd, data)
        sized.set()
        b_done.wait(1.0)
        real_write(fd, bytes(data[:10]))
        raise OSError(28, "No space left on device")

    def ftruncate(fd, length):
        if ours(fd):
            b_done.wait(1.0)
        return real_truncate(fd, length)

    failures: dict = {}

    def writer_a():
        try:
            record_test_run(version, run_id="A", report_sha256="a" * 64,
                            checkpoint_sha256="a" * 64, metrics={"hits@1": 0.5}, n_ranked=8,
                            deterministic=True, device="cpu", record_requested=False,
                            mode="disease", at=WHEN)
        except OSError as exc:
            failures["A"] = exc

    def writer_b():
        try:
            assert sized.wait(30)
            record_registration(version, run_id="B", report_sha256="b" * 64,
                                checkpoint_sha256="b" * 64, outcome="registered",
                                sequence=2, at=WHEN)
        except Exception as exc:  # noqa: BLE001 - reported by the assertion below
            failures["B"] = exc
        finally:
            b_done.set()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "write", write)
        patch.setattr(os, "ftruncate", ftruncate)
        a, b = threading.Thread(target=writer_a), threading.Thread(target=writer_b)
        a_thread.append(a)
        a.start()
        b.start()
        a.join(30)
        b.join(30)
    assert not a.is_alive() and not b.is_alive(), "a writer hung"

    assert set(failures) == {"A"} and "No space left on device" in str(failures["A"])
    assert usage.read_bytes().startswith(seeded)
    entries, problems = read_usage_history(version)
    assert [e["run_id"] for e in entries] == ["seed", "B"] and problems == []


@POSIX
def test_one_writers_failed_append_never_cuts_another_writers_line_processes(tmp_path):
    """F-g, two separate processes (`tests/fixtures/concurrent_writers.py`):
    the same interleaving, A's write and undo each waiting for B. B's line
    stays whole, nothing of A's is left, and A says its write failed."""
    from tests.fixtures.concurrent_writers import (
        CONTEXT,
        answers_from,
        append_registration,
        failing_append,
        processes,
    )

    version, usage = _seeded_usage(tmp_path)
    seeded = usage.read_bytes()
    registry, label = str(tmp_path / "registry"), version.manifest["label"]
    sized, b_done, queue = CONTEXT.Event(), CONTEXT.Event(), CONTEXT.Queue()
    with processes((failing_append, ("A", registry, label, sized, [b_done], queue)),
                   (append_registration, ("B", registry, label, "B", sized, b_done, queue))):
        answers = {a["name"]: a for a in answers_from(queue, 2)}

    assert answers["A"]["outcome"] == "failed"
    assert "No space left on device" in answers["A"]["message"]
    assert answers["B"]["outcome"] == "appended"
    assert usage.read_bytes().startswith(seeded)
    entries, problems = read_usage_history(version)
    assert [e["run_id"] for e in entries] == ["seed", "B"] and problems == []


def _checkpoint_dir(tmp_path, name: str, weights: bytes):
    directory = tmp_path / "runs" / "shepherd" / name / "checkpoints"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(weights)
    return directory


def _report_of(tmp_path, weights: bytes, **overrides) -> Path:
    report = registered_report(**overrides)
    report["manifest"]["artifact_digests"]["checkpoint"] = hashlib.sha256(weights).hexdigest()
    return store_report(tmp_path / "registry", report)


@POSIX
def test_two_ledgers_sharing_one_versions_usage_history(tmp_path):
    """F-h: two checkpoints in two directories, each with its own ledger,
    register reports of one cohort version at once (two processes) while a
    third process appends a ``test_run`` entry: each ledger holds its own
    record with its own sequence, and ``usage.jsonl`` holds both
    ``registered`` entries and the ``test_run`` entry, each a whole line. Then
    the same with the third writer's append failing after the registrations'
    entries were written (F-g's mechanism across ledgers): their entries stay.
    (Before K8-6r.1 that failure cut both registrations' entries.)"""
    from src.evaluation.sidecar import read_ledger
    from tests.fixtures.concurrent_writers import (
        CONTEXT,
        answers_from,
        append_test_run,
        failing_append,
        processes,
        register,
    )

    one, two = b"weights one", b"weights two"
    first_dir, second_dir = _checkpoint_dir(tmp_path, "r1", one), _checkpoint_dir(tmp_path, "r2", two)
    reports = [_report_of(tmp_path, one), _report_of(tmp_path, two),
               _report_of(tmp_path, one, batch_size=8), _report_of(tmp_path, two, batch_size=8)]
    version = version_of(reports[0])
    registry, label = str(tmp_path / "registry"), version.manifest["label"]
    usage = reports[0].parents[1] / "usage.jsonl"

    # At once: two registrations into two ledgers and a test run.
    start, queue = CONTEXT.Barrier(3), CONTEXT.Queue()
    with processes((register, ("R1", registry, str(reports[0]), str(first_dir), start, None,
                               queue)),
                   (register, ("R2", registry, str(reports[1]), str(second_dir), start, None,
                               queue)),
                   (append_test_run, ("T", registry, label, "T-run", start, queue))):
        answers = {a["name"]: a for a in answers_from(queue, 3)}
    assert [(answers[n]["outcome"], answers[n]["sequence"], answers[n]["problem"])
            for n in ("R1", "R2")] == [("registered", 1, None)] * 2
    assert answers["T"]["outcome"] == "appended"

    # The test-run writer's append fails after both registrations wrote theirs.
    sized, done_1, done_2 = CONTEXT.Event(), CONTEXT.Event(), CONTEXT.Event()
    queue = CONTEXT.Queue()
    with processes((register, ("R3", registry, str(reports[2]), str(first_dir), None, None,
                               queue, sized, done_1)),
                   (register, ("R4", registry, str(reports[3]), str(second_dir), None, None,
                               queue, sized, done_2)),
                   (failing_append, ("A", registry, label, sized, [done_1, done_2], queue))):
        answers = {a["name"]: a for a in answers_from(queue, 3)}
    assert [(answers[n]["outcome"], answers[n]["sequence"], answers[n]["problem"])
            for n in ("R3", "R4")] == [("registered", 2, None)] * 2
    assert answers["A"]["outcome"] == "failed"

    for directory, mine in ((first_dir, reports[0::2]), (second_dir, reports[1::2])):
        records = read_ledger(directory / "evaluations.json")[0]["records"]
        assert [(r["source_artifact_digest"], r["registration"]["sequence"]) for r in records] \
            == [(hashlib.sha256(r.read_bytes()).hexdigest(), n) for n, r in enumerate(mine, 1)]
    entries, problems = read_usage_history(version)
    assert problems == [] and usage.read_bytes().endswith(b"\n")
    assert sorted((e["kind"], e["run_id"]) for e in entries) == sorted(
        [("test_run", "T-run")] + [("registered", r.parent.name) for r in reports])


def _usage_lock(path: Path) -> Path:
    return path.parents[1] / "usage.jsonl.lock"


def _test_run(version, run_id="r"):
    return record_test_run(version, run_id=run_id, report_sha256="a" * 64,
                           checkpoint_sha256="a" * 64, metrics={"hits@1": 0.5}, n_ranked=8,
                           deterministic=True, device="cpu", record_requested=False,
                           mode="disease", at=WHEN)


def test_an_append_holds_its_files_lock_which_is_kept(tmp_path, monkeypatch):
    """`_append_entry`'s lock: two opens of one lock file exclude each other
    (another holder here is a second open in this process), so an append past
    the (patched) grace raises `FileLockBusy` naming the lock file, nothing
    appended; the lock file is kept after use and after a failed append; and a
    notes append and a usage append of one version do not exclude each other
    (two files, two locks)."""
    import src.evaluation.report_store as report_store
    from src.evaluation.report_store import STORE_LOCK_FILES
    from src.utils.file_lock import FileLockBusy, exclusive_lock

    assert STORE_LOCK_FILES == ("usage.jsonl.lock", "notes.jsonl.lock")
    monkeypatch.setattr(report_store, "APPEND_LOCK_GRACE_SECONDS", 0.2)
    first = _stored(tmp_path)
    version = version_of(first)
    lock = _usage_lock(first)
    _test_run(version, "kept-after-use")
    assert lock.is_file(), "the lock file is kept after use"
    usage = first.parents[1] / "usage.jsonl"
    before = usage.read_bytes()

    with exclusive_lock(lock, grace=0):
        with pytest.raises(FileLockBusy) as busy:
            _test_run(version, "while-held")
        assert str(busy.value) == (f"{lock} is busy: another writer held this lock for the "
                                   "0.2 s this one waited")
        assert usage.read_bytes() == before
        add_investigation_note(version, _key(first), who="ops", what="a note", at=WHEN)

    _failing_write_to(usage, monkeypatch, after=10)
    with pytest.raises(OSError, match="No space left on device"):
        _test_run(version, "failed")
    monkeypatch.undo()
    assert lock.is_file() and usage.read_bytes() == before, "kept after an exception"
    _test_run(version, "after")
    assert [e["run_id"] for e in read_usage_history(version)[0]] == ["kept-after-use", "after"]
    assert len(read_investigation_notes(version)[0]) == 1


@pytest.mark.parametrize("target", ["outside", "dangling"])
def test_a_store_lock_file_that_is_a_link_is_refused(tmp_path, target):
    """A ``usage.jsonl.lock`` that is a link, to a file outside the registry or
    to nothing, is refused by name, naming it: nothing is appended, and the
    link and its target are as they were (no file created through it)."""
    import os

    from src.evaluation.cohort_registry import RegistryRefused

    first = _stored(tmp_path)
    outside = tmp_path / "outside.lock"
    if target == "outside":
        outside.write_text("kept")
    lock = _usage_lock(first)
    lock.symlink_to(outside)

    with pytest.raises(RegistryRefused, match="not a plain file") as refused:
        _test_run(version_of(first))

    assert str(lock) in str(refused.value) and "Nothing was written." in str(refused.value)
    assert not (first.parents[1] / "usage.jsonl").exists()
    assert lock.is_symlink() and os.readlink(lock) == str(outside)
    assert (outside.read_text() == "kept") if target == "outside" else not outside.exists()


@POSIX
def test_a_store_lock_link_swapped_in_after_the_check_is_not_followed(tmp_path, monkeypatch):
    """The by-name check and the open are two steps: a link swapped in between
    fails to open (``O_NOFOLLOW``), an `OSError` naming the lock file, rather
    than creating or locking a file outside the registry. Nothing is
    appended."""
    import src.evaluation.report_store as report_store

    first = _stored(tmp_path)
    outside = tmp_path / "outside" / "usage.jsonl.lock"
    outside.parent.mkdir()
    lock = _usage_lock(first)
    real = report_store.exclusive_lock

    def swapped(path, **kwargs):
        lock.symlink_to(outside)
        return real(path, **kwargs)

    monkeypatch.setattr(report_store, "exclusive_lock", swapped)
    with pytest.raises(OSError) as failed:
        _test_run(version_of(first))

    assert str(lock) in str(failed.value)
    assert not outside.exists(), "a file was created through the link"
    assert not (first.parents[1] / "usage.jsonl").exists()


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
