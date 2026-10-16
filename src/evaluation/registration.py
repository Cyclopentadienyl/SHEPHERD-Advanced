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

**Every expected failure is one refusal** (`RegistrationRefused`) whose message
names the file and says nothing was written: a missing or damaged report, a file
that is not a report, a generated split's report, a ledger this revision does not
read, a contradiction, a stale write, a checkpoint that is not beside the ledger.
The entry points print it and exit; neither shows a traceback for these.

Module: src/evaluation/registration.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

from src.evaluation.sidecar import (
    LEDGER_FILENAME,
    append_record,
    build_record,
    find_checkpoint,
    read_ledger,
    record_key,
    write_ledger,
)


class RegistrationRefused(ValueError):
    """A report was not registered, and nothing was written."""


class RegistrationResult(NamedTuple):
    """What a registration did: one of ``REGISTRATION_OUTCOMES``, the record as the
    ledger now holds it, and the ledger's path."""

    outcome: str
    record: Dict[str, Any]
    ledger_path: Path


def _read_report(path: Path) -> Tuple[Any, str]:
    """A report parsed from one read, and the digest of exactly those bytes.

    Decoded as UTF-8, as the measurement writes it; the bytes are released before
    the text is parsed. Not UTF-8 or not JSON is a ``ValueError``; whether what
    parsed is a report is `build_record`'s to decide.
    """
    from src.utils.fingerprint import read_once

    read = read_once(path)
    digest = read.sha256
    try:
        text = read.data.decode("utf-8")
        del read
        parsed = json.loads(text)
    except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError are both ValueErrors
        raise ValueError(
            f"it does not parse as UTF-8 JSON ({type(exc).__name__}: {exc})"
        ) from exc
    return parsed, digest


def register_report(
    report_path: Path,
    *,
    ledger_dir: Path,
    verify_checkpoint_beside: bool,
    registered_at: Optional[str] = None,
) -> RegistrationResult:
    """Register one measurement report in ``<ledger_dir>/evaluations.json``.

    ``ledger_dir`` is the directory of the checkpoint the report measured,
    resolved here, so the ledger is found the same way on the write side and the
    read side (the API reads the ledger beside the checkpoint it serves). It must
    exist: nothing creates a directory for a ledger.

    ``verify_checkpoint_beside`` asks for a file in ``ledger_dir`` whose bytes are
    the checkpoint the report measured (`find_checkpoint`), for a registration
    made later from a report alone. ``measure_scorer --record`` passes ``False``:
    it read that checkpoint moments earlier and takes its directory from that
    read, so hashing the directory's checkpoints again would only cost time.

    ``registered_at`` is the time the registration is made, now in UTC unless a
    caller passes one.

    Returns the outcome — ``registered``, ``already_registered`` (nothing written)
    or ``corroborated`` — and the record as the ledger now holds it. Every
    expected failure raises ``RegistrationRefused``.
    """
    report_path = Path(report_path)
    ledger_dir = Path(ledger_dir).resolve()
    ledger_path = ledger_dir / LEDGER_FILENAME

    def refused(problem: str) -> RegistrationRefused:
        problem = problem.rstrip()
        if not problem.endswith("."):
            problem += "."
        return RegistrationRefused(f"{problem} Nothing was written to {ledger_path}.")

    if not ledger_dir.is_dir():
        raise refused(
            f"{ledger_dir} is not a directory. The ledger lives beside the checkpoint "
            "it describes, and no directory is created for it."
        )

    try:
        report, source_digest = _read_report(report_path)
    except FileNotFoundError as exc:
        raise refused(f"{report_path} does not exist.") from exc
    except OSError as exc:
        raise refused(
            f"{report_path} could not be read ({type(exc).__name__}: {exc})."
        ) from exc
    except ValueError as exc:
        raise refused(f"{report_path} is not a measurement report: {exc}.") from exc
    try:
        record = build_record(report, source_digest)
    except ValueError as exc:
        raise refused(f"{report_path}: {exc}") from exc
    # Released before the ledger is read: the record holds what is kept of it.
    del report

    if verify_checkpoint_beside:
        # **The ledger has to be beside the weights it describes.** Without this it
        # can be written into any directory, including one holding no checkpoint
        # at all, and a reader looking up a digest in the directory they were
        # handed would find a record filed under the wrong address.
        try:
            beside = find_checkpoint(ledger_dir, record["checkpoint_digest"])
        except OSError as exc:
            raise refused(
                f"the checkpoints in {ledger_dir} could not be read "
                f"({type(exc).__name__}: {exc})."
            ) from exc
        if beside is None:
            raise refused(
                f"{ledger_dir} holds no checkpoint whose bytes are "
                f"{record['checkpoint_digest'][:12]}..., which is what {report_path} "
                "measured. Point at the directory holding those weights; a ledger "
                "beside a different checkpoint is a record filed under the wrong "
                "address."
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
    if outcome == "already_registered":
        return RegistrationResult(outcome, held, ledger_path)

    try:
        write_ledger(ledger_path, updated, expected_digest=digest_at_read)
    except (OSError, ValueError) as exc:
        raise refused(f"{report_path} is not registered: {exc}") from exc
    return RegistrationResult(outcome, held, ledger_path)


__all__ = ["RegistrationRefused", "RegistrationResult", "register_report"]
