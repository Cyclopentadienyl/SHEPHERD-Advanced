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

**Every expected failure is one refusal** (`RegistrationRefused`) whose message
names the file and says nothing was written: a missing or damaged report, a file
that is not a report, a generated split's report, a ledger this revision does not
read, a contradiction, a stale write, a checkpoint file or directory that does
not hold the weights the report measured.
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


class RegistrationRefused(ValueError):
    """A report was not registered, and nothing was written."""


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
    measured_read: bool = False,
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

    Exactly one of ``checkpoint`` and ``checkpoint_dir`` says which file that is:

    - ``checkpoint_dir`` (``record_evaluation.py --checkpoint-dir``): a file in
      that directory whose bytes are the report's checkpoint must exist
      (`find_checkpoint`, ``*.pt``); the ledger goes beside the match's resolved
      path.
    - ``checkpoint`` with ``measured_read=False`` (``record_evaluation.py
      --checkpoint``): that one file, whatever its name, must hold the report's
      checkpoint bytes (`checkpoint_matches`).
    - ``checkpoint`` with ``measured_read=True`` (``measure_scorer.py
      --record``): the caller read and measured these bytes in this run, so they
      are not hashed again; pass the path of that read.

    ``registered_at`` is the time the registration is made, now in UTC unless a
    caller passes one.

    Returns the outcome — ``registered``, ``already_registered`` (nothing written)
    or ``corroborated`` — the record as the ledger now holds it, and, for a
    corroboration, the fields not compared on which this report differs. Every
    expected failure raises ``RegistrationRefused``; a call naming neither or
    both of ``checkpoint`` and ``checkpoint_dir``, or ``measured_read`` without
    ``checkpoint``, is a caller's error (``TypeError``).
    """
    if (checkpoint is None) == (checkpoint_dir is None):
        raise TypeError("register_report takes exactly one of checkpoint and checkpoint_dir")
    if measured_read and checkpoint is None:
        raise TypeError(
            "measured_read applies to checkpoint= only: a directory is searched by the "
            "report's digest, so its checkpoints are always hashed"
        )
    report_path = Path(report_path)
    # Known up front for a named file; for a directory, once the file in it is.
    ledger_path: Optional[Path] = (
        _ledger_beside(checkpoint) if checkpoint is not None else None
    )

    def refused(problem: str) -> RegistrationRefused:
        problem = problem.rstrip()
        if not problem.endswith("."):
            problem += "."
        where = f" to {ledger_path}" if ledger_path is not None else ""
        return RegistrationRefused(f"{problem} Nothing was written{where}.")

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
    elif not measured_read:
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


__all__ = ["RegistrationRefused", "RegistrationResult", "corroboration_note",
           "register_report"]
