"""Registration: one implementation for `--record` and `record_evaluation.py`.

`src/evaluation/registration.register_report` turns a kept measurement report
into a record in the ledger beside its checkpoint (`PLAN_TEST_RESULTS.md` D5, §6
phase 1 step 2). What is pinned here:

  - **the acceptance cases at this level**: not registering writes nothing; the
    same report again is a no-op that leaves the ledger's bytes as they were;
    another report with the same result corroborates; a contradiction is refused
    naming what differs, ledger unchanged; a damaged report, a generated split's
    report, a schema 2 ledger and a ledger that is a JSON list are refused with
    one message that says nothing was written;
  - **BACKLOG 18**: the report and the ledger are each parsed and identified from
    one read, so a report replaced while it is registered is recorded as the
    bytes read, never as one file's numbers under the other's digest; and the
    interleaving that erased a record (writer A reads, writer B appends, A
    writes) is refused by the stale-write guard. The guard is a path hash at
    write time and its window stays open: this is not compare-and-swap;
  - **where the ledger is**: beside the checkpoint file's resolved path,
    whether the caller names the file (`--checkpoint`, any name), a directory
    searched by digest (`--checkpoint-dir`) or the measurement's own read; a
    linked checkpoint files beside the file the link names;
  - **the CLI**: the checkpoint must be beside the ledger, every refusal exits
    non-zero with one message and no traceback, and `--show` reads the ledger
    once;
  - **the standing text** a surface renders beside a record (`caveats.py`).

The reports are built by `MeasurementManifest` itself
(`tests/unit/test_evaluation_sidecar.registered_report`).

Module: tests/unit/test_registration.py
"""
from __future__ import annotations

import hashlib
import json
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
    corroboration_note,
    register_report,
)
from src.utils.fingerprint import ReadIdentity
from src.evaluation.sidecar import empty_ledger, read_ledger, write_ledger
from tests.unit.test_evaluation_sidecar import (
    generated_report,
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


@pytest.fixture
def checkpoints(tmp_path):
    """A run's checkpoint directory holding the weights the reports measured."""
    directory = tmp_path / "runs" / "shepherd" / "r1" / "checkpoints"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(WEIGHTS)
    return directory


def _report(tmp_path, name="report.json", **overrides) -> Path:
    """A report file of a registered cohort version, measuring `WEIGHTS`."""
    report = registered_report(**overrides)
    report["manifest"]["artifact_digests"]["checkpoint"] = DIGEST
    path = tmp_path / "reports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    return path


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


def test_another_report_with_the_same_result_corroborates(tmp_path, checkpoints):
    """Byte-different (measured again after the checkpoint moved, under another
    driver) with every compared field equal: its digest joins the first record."""
    first = _register(_report(tmp_path), checkpoints)
    moved = registered_manifest()["environment"]
    moved["facts"]["gpu"]["driver"] = "999.0"
    other = _report(tmp_path, "moved.json", checkpoint_path="/moved/best.pt",
                    data_dir="/moved", environment=moved)

    result = _register(other, checkpoints)

    assert result.outcome == "corroborated"
    assert result.record["source_artifact_digest"] == first.record["source_artifact_digest"]
    assert result.record["corroborating_reports"] == [_sha(other)]
    assert result.excluded_differences == ("environment",)
    assert result.record["environment"] == first.record["environment"], "the record keeps its own"
    ledger, _ = read_ledger(result.ledger_path)
    assert len(ledger["records"]) == 1


def test_a_corroboration_says_which_uncompared_fields_this_report_measured_differently(
    tmp_path, checkpoints
):
    """Finding 3: a re-run that found the training evidence the first did not
    (and ran under another driver) corroborates, and the record keeps its own
    R10 and environment -- so the result names both fields, and the one note
    both entry points print says the record kept its own values and names the
    report holding this run's. A re-run that agrees on them too has nothing to
    say."""
    report = _report(tmp_path)
    first = _register(report, checkpoints)
    same = _register(_report(tmp_path, "same.json", data_dir="/elsewhere"), checkpoints)
    environment = registered_manifest()["environment"]
    environment["facts"]["gpu"]["driver"] = "999.0"
    other = _report(tmp_path, "found.json", data_dir="/again",
                    r10=r10_block(training_state="overlap"), environment=environment)

    result = _register(other, checkpoints)

    assert same.outcome == result.outcome == "corroborated"
    assert same.excluded_differences == ()
    assert result.excluded_differences == ("r10", "environment")
    assert result.record["r10"] == first.record["r10"], "the record keeps its own labels"
    assert corroboration_note(result, other) == (
        "This report's r10 and environment differ from record 1's; they are not "
        f"compared, so the record keeps its own and this report's are in {other}.")
    assert corroboration_note(same, tmp_path / "same.json") is None
    assert corroboration_note(first, report) is None


def test_the_cli_prints_the_note_when_a_corroborating_report_found_other_r10(
    tmp_path, checkpoints, caplog
):
    import logging

    import scripts.record_evaluation as record_evaluation

    _register(_report(tmp_path), checkpoints)
    other = _report(tmp_path, "found.json", data_dir="/again",
                    r10=r10_block(training_state="overlap"))

    with caplog.at_level(logging.INFO, logger=record_evaluation.logger.name):
        assert record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                                       "--report", str(other)]) == 0

    said = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("This report agrees with record 1") for m in said)
    assert (f"This report's r10 differ from record 1's; they are not compared, so the "
            f"record keeps its own and this report's are in {other}.") in said


def test_a_contradiction_is_refused_naming_what_differs_and_the_ledger_stays(
    tmp_path, checkpoints
):
    first = _register(_report(tmp_path), checkpoints)
    before = first.ledger_path.read_bytes()
    edited = _report(tmp_path, "edited.json", metrics={
        "untruncated_mrr": 0.9, "mean_rank": 1.5, "hits@1": 0.25, "hits@10": 0.75})

    with pytest.raises(RegistrationRefused) as refused:
        _register(edited, checkpoints)

    message = str(refused.value)
    assert "disagrees on: mean_rank, untruncated_mrr" in message
    assert message.endswith(f"Nothing was written to {first.ledger_path}.")
    assert first.ledger_path.read_bytes() == before


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
    report = tmp_path / "report.json"
    report.write_bytes(data)
    before = _listing(checkpoints)

    with pytest.raises(RegistrationRefused, match=fragment) as refused:
        _register(report, checkpoints)

    assert str(report) in str(refused.value)
    assert "Nothing was written" in str(refused.value)
    assert _listing(checkpoints) == before


def test_a_missing_report_or_a_directory_is_refused(tmp_path, checkpoints):
    with pytest.raises(RegistrationRefused, match="does not exist.*Nothing was written"):
        _register(tmp_path / "absent.json", checkpoints)
    with pytest.raises(RegistrationRefused, match="could not be read.*Nothing was written"):
        _register(tmp_path, checkpoints)
    assert not (checkpoints / "evaluations.json").exists()


def test_a_generated_splits_report_is_refused(tmp_path, checkpoints):
    report = tmp_path / "val.json"
    report.write_text(json.dumps(generated_report()))

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
    b = registered_report(metrics={"untruncated_mrr": 0.1, "mean_rank": 9.0,
                                   "hits@1": 0.0, "hits@10": 0.1})
    b["manifest"]["artifact_digests"]["checkpoint"] = DIGEST
    bytes_b = json.dumps(b).encode("utf-8")
    replaced = replace_after_read(monkeypatch, after=report, target=report, data=bytes_b,
                                  how=how)

    result = _register(report, checkpoints)

    assert replaced.fired == 1
    assert report.read_bytes() == bytes_b, "the replacement happened"
    assert result.record["source_artifact_digest"] == hashlib.sha256(bytes_a).hexdigest()
    assert result.record["metrics"] == json.loads(bytes_a)["authoritative_metrics"]
    # B registered now is B: a different result under the same key, refused.
    with pytest.raises(RegistrationRefused, match="disagrees on"):
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
    """The readers' release check (contract M2.1, S2): the report's bytes are
    gone before the ledger is read, and the ledger's before the function
    returns. Both reads go through `read_once`, in that order."""
    from tests.fixtures.release import check_release

    report = _report(tmp_path)
    write_ledger(checkpoints / "evaluations.json", empty_ledger(), None)

    result, reads = check_release(monkeypatch, lambda: _register(report, checkpoints))

    assert result.outcome == "registered"
    assert [p.name for p in reads] == ["report.json", "evaluations.json"]


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
    """The interleaving that erased a record (map, BACKLOG 18): writer A reads
    the ledger, writer B appends, A writes.

    **The control first.** With A's digest taken by path after B's write -- the
    old order, the ledger parsed and then hashed again -- A's write passes the
    guard and B's record is gone. **Then the fix:** A's digest is the one its
    read returned, so the guard refuses A and B's record stays. Still
    single-writer: the guard's own window, between its hash and the rename, is
    not closed.
    """
    from src.evaluation.sidecar import append_record, build_record, ledger_digest
    from tests.fixtures.replacement import hook_read_once

    ledger_path = checkpoints / "evaluations.json"
    seed = _report(tmp_path, "seed.json")
    report_a = _report(tmp_path, "a.json", batch_size=8)
    report_b = _report(tmp_path, "b.json", batch_size=16)
    _register(seed, checkpoints)
    seeded = ledger_path.read_bytes()

    # The control: the old order erases B's append.
    parsed, _ = read_ledger(ledger_path)
    _register(report_b, checkpoints)
    record_a = build_record(json.loads(report_a.read_text()), _sha(report_a))
    _, with_a = append_record(parsed, record_a, registered_at=WHEN)
    write_ledger(ledger_path, with_a, expected_digest=ledger_digest(ledger_path))
    batches = [r["regime"]["batch_size"] for r in read_ledger(ledger_path)[0]["records"]]
    assert batches == [32, 8], "the old order erased B's record"

    # The fix: the same interleaving through register_report is refused.
    ledger_path.write_bytes(seeded)
    fired = []

    def writer_b(path):
        if fired or os.path.realpath(path) != os.path.realpath(ledger_path):
            return
        fired.append(path)
        _register(report_b, checkpoints)

    hook_read_once(monkeypatch, on_return=writer_b)
    with pytest.raises(RegistrationRefused, match="changed since it was read") as refused:
        _register(report_a, checkpoints)

    assert fired, "writer B ran between A's read and A's write"
    assert "Nothing was written" in str(refused.value)
    batches = [r["regime"]["batch_size"] for r in read_ledger(ledger_path)[0]["records"]]
    assert batches == [32, 16], "B's record stays and A's is not written"


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
    assert shown[0]["scoring"]["mode"] == "C"


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

    report = tmp_path / "not_a_report.json"
    report.write_text(json.dumps({"hello": "world"}))

    with pytest.raises(SystemExit, match="not a measurement report"):
        record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                                "--report", str(report)])


def test_the_cli_refuses_a_report_from_an_earlier_revision(tmp_path, checkpoints):
    import scripts.record_evaluation as record_evaluation

    report = registered_report()
    report["manifest"]["python_seed"] = 0
    report["manifest"]["artifact_digests"]["checkpoint"] = DIGEST
    path = tmp_path / "old.json"
    path.write_text(json.dumps(report))

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
    report = tmp_path / "report.json"
    report.write_bytes(data)

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


def test_the_result_label_the_pre_b1_note_and_the_absence_sentence():
    from src.evaluation.caveats import MODE_C_RESULT_LABEL, PRE_B1_NOTE, RECORD_ABSENCE

    assert MODE_C_RESULT_LABEL == (
        "Disease ranking by the model, approved scorer policy (Mode C)")
    assert PRE_B1_NOTE == (
        "This is the model's own Mode C disease ranking. Until B-1 ships, the Diagnosis "
        "tab ranks by the legacy GNN+SP mixture behind the path gate, so this number "
        "does not describe the tab's current ranking.")
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
    replacement["authoritative_metrics"]["hits@1"] = (
        replacement["authoritative_metrics"]["hits@1"] / 2 + 0.25)
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


def test_a_later_registration_still_names_a_missing_report_plainly(tmp_path, checkpoints):
    """A registration by hand has no run behind it: a missing report is the
    ordinary refusal, not a run's lost report."""
    with pytest.raises(RegistrationRefused) as refused:
        register_report(tmp_path / "absent.json", checkpoint=checkpoints / "best.pt",
                        registered_at=WHEN)
    assert not isinstance(refused.value, ReportNotAsPublished)
    assert "re-measure" not in str(refused.value)
