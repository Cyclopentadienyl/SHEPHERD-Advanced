"""
Registering a measurement report in the ledger beside its checkpoint.
=====================================================================
`PLAN_TEST_RESULTS.md` D5 and §6 phase 1, step 2: a report is registered
automatically right after its run (`scripts/measure_scorer.py --record`) or later
by hand (`scripts/record_evaluation.py --report`), and **both go through
`register_report`**. Two implementations of "turn a report into a ledger record"
could disagree about what a report says; one cannot.

**Each file is parsed and identified from one read** (BACKLOG item 18). The report
is read once, and the record is built from those bytes and names them by their
digest; the ledger is read once, and the write's stale-write guard compares the
file with the digest of what was parsed. A report replaced while it was being
registered therefore cannot be recorded as one file's numbers under the other's
digest, and another writer's append between this read and this write is refused
rather than erased.

**Still single-writer.** The guard compares the file by path just before the
replacement, and the window between that comparison and the replacement stays
open (`write_ledger`). This is not compare-and-swap and makes no claim of safety
for concurrent writers.

**One place for the ledger, whichever way a report arrives:** beside the
checkpoint file's resolved path, the rule the read-only view reads by. The
caller names that file, or a directory searched for it by the report's digest;
a checkpoint reached through a link files its ledger beside the file the link
names, so a record registered by hand is one the service shows, and the same
weights never get a second ledger beside the link.

**An automatic registration is bound to the run that asks for it.** `--record`
passes what that run itself holds (`MeasuredRun`): the identity of its checkpoint
read and the digest of the report bytes it published. The report is read again
here, as for any registration, and refused unless those are its bytes and it
measured that checkpoint, so a file replaced between the publication and this
read is never registered in the run's name (the reviewer's P2 of 2026-10-17).

**Every expected failure is one refusal** (`RegistrationRefused`) whose message
names the file and says nothing was written: a missing or damaged report, a file
that is not a report, a generated split's report, a ledger this revision does not
read, a contradiction, a stale write, a checkpoint file or directory that does
not hold the weights the report measured, and, for an automatic registration, a
report file that is no longer the one the run published (`ReportNotAsPublished`).
The entry points print it and exit; neither shows a traceback for these.

Module: src/evaluation/registration.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

from src.evaluation.sidecar import (
    CORROBORATION_KEEPS_FIRST,
    LEDGER_FILENAME,
    append_record,
    build_record,
    checkpoint_matches,
    find_checkpoint,
    read_ledger,
    record_key,
    write_ledger,
)
from src.utils.fingerprint import ReadIdentity


class RegistrationRefused(ValueError):
    """A report was not registered, and nothing was written."""


class ReportNotAsPublished(RegistrationRefused):
    """An automatic registration could not read back the bytes the run
    published: the report file holds other bytes (replaced or changed after
    its publication, damaged included), is gone, or cannot be read. Nothing was
    written, and the report the run published may no longer exist, so it is not
    offered for a later registration."""


class _NotTheExpectedBytes(Exception):
    """`_read_report` read bytes whose digest is not the one expected; raised
    before they are decoded."""

    def __init__(self, sha256: str) -> None:
        super().__init__(sha256)
        self.sha256 = sha256


class MeasuredRun(NamedTuple):
    """What a measurement holds about itself when it registers its own report
    (`measure_scorer.py --record`): the identity of its checkpoint read (the path
    it read, resolved when it was read, and the digest of those bytes) and the
    digest `publish_bytes` returned for the report bytes it wrote. Taken from
    the run's own reads and writes; nothing is hashed again to make it."""

    checkpoint: ReadIdentity
    report_sha256: str


class RegistrationResult(NamedTuple):
    """What a registration did: one of ``REGISTRATION_OUTCOMES``, the record as the
    ledger now holds it, the ledger's path, and -- for ``corroborated`` only --
    which of ``CORROBORATION_KEEPS_FIRST`` this report measured differently.

    ``excluded_differences`` is empty unless this report corroborated a record
    and its own R10 or environment differ from the record's. The record keeps
    its own values (each record's labels are true of its own report), so a
    surface that reported plain agreement would hide that this report found,
    for instance, training evidence the first run did not (`corroboration_note`).
    """

    outcome: str
    record: Dict[str, Any]
    ledger_path: Path
    excluded_differences: Tuple[str, ...] = ()


def corroboration_note(result: RegistrationResult, report_path: Path) -> Optional[str]:
    """The one sentence both entry points print when a corroborating report's
    own values differ from the record's in a field that is not compared; `None`
    when there is nothing to say.

    Names the fields, the record and the report, never the values: the report
    holds them, and the record keeps its own.
    """
    if result.outcome != "corroborated" or not result.excluded_differences:
        return None
    fields = " and ".join(result.excluded_differences)
    sequence = result.record["registration"]["sequence"]
    return (
        f"This report's {fields} differ from record {sequence}'s; they are not "
        f"compared, so the record keeps its own and this report's are in {report_path}."
    )


def _read_report(path: Path, expected_sha256: Optional[str] = None) -> Tuple[Any, str]:
    """A report parsed from one read, and the digest of exactly those bytes.

    With ``expected_sha256``, the digest of that read is compared **before the
    bytes are decoded or parsed**, and other bytes raise `_NotTheExpectedBytes`:
    a file that is not the expected report is never interpreted, whether it is
    another valid report or damaged JSON.

    Decoded as UTF-8, as the measurement writes it; the bytes are released before
    the text is parsed. Not UTF-8 or not JSON is a ``ValueError``; whether what
    parsed is a report is `build_record`'s to decide.
    """
    from src.utils.fingerprint import read_once

    read = read_once(path)
    digest = read.sha256
    if expected_sha256 is not None and digest != expected_sha256:
        del read
        raise _NotTheExpectedBytes(digest)
    try:
        text = read.data.decode("utf-8")
        del read
        parsed = json.loads(text)
    except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError are both ValueErrors
        raise ValueError(
            f"it does not parse as UTF-8 JSON ({type(exc).__name__}: {exc})"
        ) from exc
    return parsed, digest


def _ledger_beside(checkpoint: Path) -> Path:
    """The ledger of the checkpoint at this path: ``evaluations.json`` in the
    directory of its resolved path (M4 decision 15), the rule the read-only view
    reads by. A link files its ledger beside the file it names."""
    return Path(checkpoint).resolve().parent / LEDGER_FILENAME


def register_report(
    report_path: Path,
    *,
    checkpoint: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    measured: Optional[MeasuredRun] = None,
    registered_at: Optional[str] = None,
) -> RegistrationResult:
    """Register one measurement report in the ledger beside the checkpoint it
    measured.

    **The ledger is always beside the checkpoint file's resolved path**
    (``Path(file).resolve().parent / "evaluations.json"``, M4 decision 15): the
    rule the read-only view reads by, for the checkpoint it serves. A checkpoint
    reached through a link files its ledger beside the file the link names,
    where the view looks; a ledger beside the link would be one the service
    never reads, and a second ledger for the same weights. Nothing creates a
    directory for a ledger.

    Exactly one of ``checkpoint``, ``checkpoint_dir`` and ``measured`` says which
    file that is:

    - ``checkpoint_dir`` (``record_evaluation.py --checkpoint-dir``): a file in
      that directory whose bytes are the report's checkpoint must exist
      (`find_checkpoint`, ``*.pt``); the ledger goes beside the match's resolved
      path.
    - ``checkpoint`` (``record_evaluation.py --checkpoint``): that one file,
      whatever its name, must hold the report's checkpoint bytes
      (`checkpoint_matches`).
    - ``measured`` (``measure_scorer.py --record``): the run's own checkpoint read
      and the digest of the report it published (`MeasuredRun`). The checkpoint
      is not hashed again, since the run read and measured those bytes; instead
      the report read here must be the bytes the run published
      (`ReportNotAsPublished` otherwise) and must name the checkpoint that run
      read. **Both are compared, not assumed**: a report replaced after its
      publication by another valid report -- another checkpoint's, or this
      checkpoint's on another cohort version -- is refused, never filed in the
      run's name. The ledger goes beside the read's path, which the run resolved
      when it read it.

    ``registered_at`` is the time the registration is made, now in UTC unless a
    caller passes one.

    Returns the outcome — ``registered``, ``already_registered`` (nothing written)
    or ``corroborated`` — the record as the ledger now holds it, and, for a
    corroboration, the fields not compared on which this report differs. Every
    expected failure raises ``RegistrationRefused``; a call naming none, or more
    than one, of ``checkpoint``, ``checkpoint_dir`` and ``measured`` is a
    caller's error (``TypeError``).
    """
    named = [arg for arg in (checkpoint, checkpoint_dir, measured) if arg is not None]
    if len(named) != 1:
        raise TypeError(
            "register_report takes exactly one of checkpoint, checkpoint_dir and measured"
        )
    if measured is not None and not isinstance(measured, MeasuredRun):
        raise TypeError("measured is the run's MeasuredRun")
    report_path = Path(report_path)
    # Known up front for a named file or a run's read; for a directory, once the
    # file in it is.
    ledger_path: Optional[Path] = None
    if checkpoint is not None:
        ledger_path = _ledger_beside(checkpoint)
    elif measured is not None:
        ledger_path = _ledger_beside(measured.checkpoint.path)

    def refused(problem: str, kind: type = RegistrationRefused) -> RegistrationRefused:
        problem = problem.rstrip()
        if not problem.endswith("."):
            problem += "."
        where = f" to {ledger_path}" if ledger_path is not None else ""
        return kind(f"{problem} Nothing was written{where}.")

    def not_a_directory(directory: Path) -> RegistrationRefused:
        return refused(
            f"{directory} is not a directory. The ledger lives beside the checkpoint "
            "it describes, and no directory is created for it."
        )

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir).resolve()
        if not checkpoint_dir.is_dir():
            raise not_a_directory(checkpoint_dir)
    elif not ledger_path.parent.is_dir():
        raise not_a_directory(ledger_path.parent)

    # **For a run's own report, every way of not getting back the published
    # bytes is one refusal** (`ReportNotAsPublished`): the run cannot vouch that
    # its report still exists, so it is never described as kept.
    lost = ReportNotAsPublished if measured is not None else RegistrationRefused
    lost_tail = (" The run's own report may no longer exist; re-measure to register "
                 "this measurement." if measured is not None else "")
    try:
        report, source_digest = _read_report(
            report_path, measured.report_sha256 if measured is not None else None
        )
    except _NotTheExpectedBytes as exc:
        # Compared on the bytes just read, before they are decoded: whatever the
        # file now holds -- another report or damaged JSON -- it is not the
        # report this run wrote, so nothing it says is registered in its name.
        raise refused(
            f"{report_path} is not the report this run published: it holds bytes "
            f"{exc.sha256[:12]}..., and the run wrote {measured.report_sha256[:12]}... "
            "The file was replaced or changed after it was written." + lost_tail,
            lost,
        ) from None
    except FileNotFoundError as exc:
        gone = (f"{report_path} does not exist: the run's own report is gone; re-measure to "
                "register this measurement." if measured is not None
                else f"{report_path} does not exist.")
        raise refused(gone, lost) from exc
    except OSError as exc:
        raise refused(
            f"{report_path} could not be read ({type(exc).__name__}: {exc})." + lost_tail,
            lost,
        ) from exc
    except ValueError as exc:
        raise refused(f"{report_path} is not a measurement report: {exc}.") from exc
    try:
        record = build_record(report, source_digest)
    except ValueError as exc:
        raise refused(f"{report_path}: {exc}") from exc
    # Released before the ledger is read: the record holds what is kept of it.
    del report
    digest = record["checkpoint_digest"]

    # **The ledger has to be beside the weights it describes.** Without this it
    # can be written into any directory, including one holding no checkpoint at
    # all, and a reader looking up a digest beside the weights they were handed
    # would find a record filed under the wrong address.
    if checkpoint_dir is not None:
        try:
            beside = find_checkpoint(checkpoint_dir, digest)
        except OSError as exc:
            raise refused(
                f"the checkpoints in {checkpoint_dir} could not be read "
                f"({type(exc).__name__}: {exc})."
            ) from exc
        if beside is None:
            raise refused(
                f"{checkpoint_dir} holds no checkpoint whose bytes are "
                f"{digest[:12]}..., which is what {report_path} measured (a directory "
                "is searched for *.pt files). Point at the directory holding those "
                "weights, or name the file itself; a ledger beside a different "
                "checkpoint is a record filed under the wrong address."
            )
        ledger_path = _ledger_beside(beside)
    elif measured is not None:
        if digest != measured.checkpoint.sha256:
            raise refused(
                f"{report_path} names checkpoint {digest[:12]}..., but this run read "
                f"{measured.checkpoint.sha256[:12]}... ({measured.checkpoint.path}); a "
                "ledger beside a different checkpoint is a record filed under the "
                "wrong address."
            )
    else:
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise refused(f"{checkpoint} does not exist.")
        if not checkpoint.is_file():
            raise refused(f"{checkpoint} is not a file, so it is not a checkpoint.")
        try:
            matches = checkpoint_matches(checkpoint, digest)
        except OSError as exc:
            raise refused(
                f"{checkpoint} could not be read ({type(exc).__name__}: {exc})."
            ) from exc
        if not matches:
            raise refused(
                f"{checkpoint} does not hold the weights {report_path} measured "
                f"({digest[:12]}...); a ledger beside a different checkpoint is a "
                "record filed under the wrong address."
            )

    try:
        ledger, digest_at_read = read_ledger(ledger_path)
    except (OSError, ValueError) as exc:
        raise refused(str(exc)) from exc

    when = registered_at if registered_at is not None else (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    try:
        outcome, updated = append_record(ledger, record, registered_at=when)
    except ValueError as exc:
        raise refused(f"{report_path} is not registered: {exc}") from exc

    key = record_key(record)
    held = next(r for r in updated["records"] if record_key(r) == key)
    differences: Tuple[str, ...] = ()
    if outcome == "corroborated":
        differences = tuple(field for field in CORROBORATION_KEEPS_FIRST
                            if held.get(field) != record[field])
    if outcome == "already_registered":
        return RegistrationResult(outcome, held, ledger_path)

    try:
        write_ledger(ledger_path, updated, expected_digest=digest_at_read)
    except (OSError, ValueError) as exc:
        raise refused(f"{report_path} is not registered: {exc}") from exc
    return RegistrationResult(outcome, held, ledger_path, differences)


__all__ = ["MeasuredRun", "RegistrationRefused", "RegistrationResult",
           "ReportNotAsPublished", "corroboration_note", "register_report"]
