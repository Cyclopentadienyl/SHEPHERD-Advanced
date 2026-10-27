"""
The report store: where a cohort version's reports are kept, found and compared.
================================================================================
`docs/working/PLAN_TEST_RESULTS.md` D4 and D5 (as amended 2026-10-19), M5's
boundaries. A registered cohort version's report store is its ``reports/``
directory (`src.evaluation.cohort_registry.report_store_directory`):

    <registry>/versions/<label>/reports/
        <run id>/ranks.json, <run id>/report.json   one measurement each
        usage.jsonl                                 the version's usage history
        notes.jsonl                                 investigation notes

**A report is registered only from here** (D5, 2026-10-19). Whether another run
under a report's key disagrees with it can be checked only against the store
that keeps the key's reports, which a copy elsewhere cannot give.
`locate_store_report` therefore finds a report's version through the
registry's own structure -- ``<registry>/versions/<label>/reports/<run id>/
report.json``, read from the report's resolved path -- and resolves that
version (`resolve_version`), refusing whatever does not fit rather than
guessing or falling back to a copy.

**And only from the deployment's registry** (BACKLOG 37): the registry that
structure names must be the one ``SHEPHERD_REGISTRY`` names, compared by
resolved path (`server_settings.deployment_registry_refusal`), checked before
anything in it is read. A copy of the whole registry holds the same bytes and
would pass every check above while keeping its own disagreement block and usage
history, which the service never reads. A registry moved as a whole still works
once ``SHEPHERD_REGISTRY`` names its new place; its version is the same version
by the digest of its ``cohort.json``.

**Disagreement is found with the ledger's own comparison** (`scan_report_store`).
Every report the store keeps is built into a record by `build_record`, grouped
by `record_key` and compared with `differing_fields`, the comparison
`append_record` makes: runs under one key "disagree" in the store exactly when
the second would contradict the first in a ledger. A key's state is one of
`CONTEST_STATES`; a report that cannot be read or built makes its key (or,
when even its key cannot be told, every key of the version) **cannot be
checked, never uncontested**. Registration refuses both a contested key and one
that cannot be checked (`src/evaluation/registration.py`).

**Investigation notes** (who, when, what) are kept here with the key's reports,
never in the ledger, and nothing consults them to lift the block: a note
records what was found; it decides nothing. "Who" is what the operator types;
under deployment mode M1 nothing verifies it.

**The usage history** is one append-only file per version, written by the
shared code (the measurement, the registration function, the load from an
ordered list), so a run started from the command line counts as one started
from the Test tab. It makes the use of a cohort for selection visible; it does
not prevent it (D4).

**What it rests on: plain files and one writer at a time** (D2, D7). Deleting a
disagreeing report lifts the block, and the files can be edited; the tools
refuse, they do not make the refusal tamper-proof. Each append writes one line
at the end of a file opened for appending, after closing a line an interrupted
append left open, and a write that fails is taken back (`_append_entry`);
nothing locks the store: a scan and a registration that follows it are not
atomic with a run publishing a new report into the same version at that moment.

Module: src/evaluation/report_store.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

if TYPE_CHECKING:
    from src.evaluation.cohort_registry import CohortVersionRead

#: The two append-only files a report store keeps beside its run directories.
USAGE_FILE = "usage.jsonl"
NOTES_FILE = "notes.jsonl"
#: Both: the store's own files, which are not runs
#: (`cohort_registry.list_report_runs` leaves them out).
STORE_FILES = (USAGE_FILE, NOTES_FILE)

#: What the usage history records (D4): every measurement of the version, every
#: registration of one of its reports, and every model loaded from a list
#: ordered by its score.
USAGE_KINDS: Tuple[str, ...] = ("test_run", "registered", "model_load")

#: A key's state in the store (D5): every report under it agrees (or there is at
#: most one); two of them disagree; or whether they do cannot be checked.
CONTEST_STATES: Tuple[str, ...] = ("uncontested", "contested", "cannot_be_checked")


class NotInReportStore(ValueError):
    """The path is not a report the registry's report store holds, so it is not
    registered (D5): a copy, wherever it is, cannot show whether another run
    under its key disagrees. ``why`` is the reason alone, without the advice to
    register the store's original: for a run's own report, which it published
    into the store, that advice would not be true."""

    def __init__(self, message: str, why: str = "") -> None:
        super().__init__(message)
        self.why = why


class NotTheDeploymentRegistry(ValueError):
    """The registry a report is kept in, or a ``--cohort`` run would measure in,
    is not the deployment's (`server_settings.deployment_registry_refusal`:
    compared by resolved path with ``SHEPHERD_REGISTRY``, which may also be
    unset or refused). Nothing in that registry was read."""


class _NotTheExpectedBytes(Exception):
    """`read_report` read bytes whose digest is not the one expected; raised
    before they are decoded."""

    def __init__(self, sha256: str) -> None:
        super().__init__(sha256)
        self.sha256 = sha256


class StoreReport(NamedTuple):
    """A report the store holds: the registry it was found in (resolved), the
    version's label and resolution, the run id, and the report's resolved path."""

    registry: Path
    label: str
    run_id: str
    path: Path
    version: "CohortVersionRead"


def _utc(at: Optional[datetime]) -> str:
    when = at if at is not None else datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError(f"{when!r} has no time zone; a recorded time must say which it is in")
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Reading one report
# ---------------------------------------------------------------------------
def read_report(path: Path, expected_sha256: Optional[str] = None) -> Tuple[Any, str]:
    """A report parsed from one read, and the digest of exactly those bytes.

    With ``expected_sha256``, the digest of that read is compared **before the
    bytes are decoded or parsed**, and other bytes raise `_NotTheExpectedBytes`:
    a file that is not the expected report is never interpreted, whether it is
    another valid report or damaged JSON.

    Decoded as UTF-8, as the measurement writes it; the bytes are released before
    the text is parsed. Not UTF-8 or not JSON is a ``ValueError``; whether what
    parsed is a report is `build_record`'s to decide. The one reader of a kept
    report: the registration, the store's scan and `report_state` all read
    through it.
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
    except RecursionError as exc:
        raise ValueError(f"it nests too deeply to parse ({exc})") from None
    return parsed, digest


# ---------------------------------------------------------------------------
# Where a report is: the registry's own structure
# ---------------------------------------------------------------------------
def _not_in_store(given: Path, why: str) -> str:
    return (
        f"{given} is not a report the registry's report store holds: {why}. A report is "
        "registered only from the store that keeps it, "
        "<registry>/versions/<label>/reports/<run id>/report.json, because only the "
        "store can show whether another run under its key disagrees (D5); a copy, which "
        "can still be read and shared, cannot. Register the store's original instead: the "
        "path the measurement printed as 'Report:'."
    )


def store_report_parts(report_path: Any) -> Tuple[Path, str, str, Path]:
    """The registry root, version label, run id and resolved path of a report in a
    report store, read from the path's shape alone, or `NotInReportStore`.

    **The resolved path is what is read,** so a link to a store report is its
    original in the store, and a ``report.json`` in the store that is a link to a
    file elsewhere is that file, which is not in a store. Nothing is opened; the
    caller resolves the version (`locate_store_report`).
    """
    from src.evaluation.cohort_registry import REPORT_FILE, REPORTS_DIR, VERSIONS_DIR, is_run_id

    given = Path(report_path)
    try:
        resolved = given.resolve()
    except (OSError, RuntimeError) as exc:  # a link loop: RuntimeError before Python 3.13
        why = f"it cannot be resolved ({type(exc).__name__}: {exc})"
        raise NotInReportStore(_not_in_store(given, why), why) from None
    run_dir = resolved.parent
    reports = run_dir.parent
    version_dir = reports.parent
    versions = version_dir.parent
    if resolved.name != REPORT_FILE:
        why = f"its file name is {resolved.name!r}, not {REPORT_FILE}"
    elif not is_run_id(run_dir.name):
        why = (f"the directory holding it, {run_dir.name!r}, is not a run directory the "
               "measurement makes (<UTC time>-<12 hex of the checkpoint digest>-<8 hex>)")
    elif reports.name != REPORTS_DIR:
        why = f"its run directory is not in a version's {REPORTS_DIR}/ directory"
    elif versions.name != VERSIONS_DIR:
        why = f"its version directory is not in a registry's {VERSIONS_DIR}/ directory"
    else:
        return versions.parent, version_dir.name, run_dir.name, resolved
    if str(resolved) != os.path.abspath(given):
        why += f" (it resolves to {resolved})"
    raise NotInReportStore(_not_in_store(given, why), why)


def locate_store_report(report_path: Any, *,
                        environ: Optional[Mapping[str, str]] = None) -> StoreReport:
    """The report store holding this report: its registry and version, found
    through the registry's structure and resolved, never guessed.

    The path's shape names the registry, the label and the run
    (`store_report_parts`). **That registry must be the deployment's**
    (`deployment_registry_refusal`, with ``SHEPHERD_REGISTRY`` read from
    `environ`, `os.environ` when `None`), checked from the path's shape before
    anything in it is read. The version is then resolved there
    (`resolve_version`: the registry absolute, outside the repository and
    existing; the version's ``cohort.json`` read once and checked; its directory
    checked against the registry). That the file sits in the store is not proof
    of what it is: the caller still reads it and compares it with the version
    and its checkpoint.

    Raises:
        NotInReportStore: the path is not shaped as a store report.
        NotTheDeploymentRegistry: the registry that shape names is not the
            deployment's, or which one is cannot be told.
        RegistryRefused, CohortVersionRefused: the registry or version that
            shape names cannot be read (`resolve_version`).
    """
    from src.config.server_settings import deployment_registry_refusal
    from src.evaluation.cohort_registry import resolve_version

    root, label, run_id, resolved = store_report_parts(report_path)
    why = deployment_registry_refusal(root, environ)
    if why is not None:
        raise NotTheDeploymentRegistry(why)
    version = resolve_version(root, label)
    if resolved.parent.parent.parent != version.directory:
        why = (f"the version {label!r} resolves to {version.directory}, not the directory it "
               "sits in")
        raise NotInReportStore(_not_in_store(Path(report_path), why), why)
    return StoreReport(version.directory.parent.parent, label, run_id, resolved, version)


# ---------------------------------------------------------------------------
# Disagreement under one key (D5)
# ---------------------------------------------------------------------------
class KeyStatus(NamedTuple):
    """One key's state in a version's report store (`CONTEST_STATES`).

    ``run_ids`` are the readable reports under the key; for ``contested``,
    ``differing`` names what they disagree on (the metric names when only the
    metrics do) and ``disagreeing_runs`` which runs; for ``cannot_be_checked``,
    ``problems`` says why, naming each file."""

    state: str
    run_ids: Tuple[str, ...]
    differing: Tuple[str, ...] = ()
    disagreeing_runs: Tuple[str, ...] = ()
    problems: Tuple[str, ...] = ()


class StoreScan(NamedTuple):
    """Every report a version's store keeps, built into records and grouped by
    key (`scan_report_store`).

    ``reports`` maps a key to its distinct reports, by the report's digest, as
    ``(run id, record)``: byte-identical copies are one report, compared once.
    ``runs`` maps a key to every run directory holding a readable report under
    it, copies included. ``unbuilt`` maps a key to the reports under it that
    cannot be compared, as ``(run id, problem)``; ``unplaced`` holds the reports
    whose key cannot be told, as ``(run id, problem)`` (the run id None when the
    store itself could not be listed), which leave every key of the version
    unable to be checked. **Every report the scan read is named by its run id**,
    so a run's state is found by its id (`run_status`), never by reading ids out
    of a problem's words. ``digests`` maps each run id whose ``report.json``
    the scan read to the digest of those bytes, so a lookup by run id is bound
    to the bytes the caller read there (`run_key`, `run_status`).
    """

    version: "CohortVersionRead"
    reports: Dict[Tuple[Any, ...], Dict[str, Tuple[str, Dict[str, Any]]]]
    unbuilt: Dict[Tuple[Any, ...], Tuple[Tuple[str, str], ...]]
    unplaced: Tuple[Tuple[Optional[str], str], ...]
    runs: Dict[Tuple[Any, ...], Tuple[str, ...]]
    digests: Dict[str, str]

    def status(self, key: Tuple[Any, ...]) -> KeyStatus:
        """The key's state: ``contested`` when any two of its reports differ on
        a compared field; otherwise ``cannot_be_checked`` when a report that
        may be under it could not be compared; otherwise ``uncontested``, also
        for a key no report in the store is under."""
        from src.evaluation.sidecar import difference_names, differing_fields

        entries = list(self.reports.get(tuple(key), {}).values())
        run_ids = self.runs.get(tuple(key), ())
        differing: List[str] = []
        disagreeing: set = set()
        for index, (run_a, first) in enumerate(entries):
            for run_b, second in entries[index + 1:]:
                fields = differing_fields(first, second)
                if fields:
                    for name in difference_names(first, second, fields):
                        if name not in differing:
                            differing.append(name)
                    disagreeing |= {run_a, run_b}
        if differing:
            return KeyStatus("contested", run_ids, tuple(differing),
                             tuple(sorted(disagreeing)))
        problems = (tuple(problem for _run, problem in self.unbuilt.get(tuple(key), ()))
                    + self._unplaced_problems())
        if problems:
            return KeyStatus("cannot_be_checked", run_ids, problems=problems)
        return KeyStatus("uncontested", run_ids)

    def state(self, key: Tuple[Any, ...]) -> str:
        """The key's state alone, one of `CONTEST_STATES` (for D4's ordering)."""
        return self.status(key).state

    def _unplaced_problems(self) -> Tuple[str, ...]:
        return tuple(problem for _run, problem in self.unplaced)

    def _placed(self, run_id: str) -> Optional[Tuple[Optional[Tuple[Any, ...]], Optional[str]]]:
        for key, run_ids in self.runs.items():
            if run_id in run_ids:
                return tuple(key), None
        for key, entries in self.unbuilt.items():
            for run, problem in entries:
                if run == run_id:
                    return tuple(key), problem
        for run, problem in self.unplaced:
            if run == run_id:
                return None, problem
        return None

    def _other_bytes(self, run_id: str, sha256: Optional[str]) -> bool:
        """Whether the scan read other bytes in run directory `run_id` than the
        caller's read found there (`sha256`, None when it read none)."""
        read = self.digests.get(run_id)
        return read is not None and read != sha256

    def run_key(self, run_id: str,
                sha256: Optional[str]) -> Optional[Tuple[Optional[Tuple[Any, ...]], Optional[str]]]:
        """Where the scan placed the report in run directory `run_id`:
        ``(key, None)`` for a report it built, ``(key, problem)`` for one under
        a key it could not compare, ``(None, problem)`` for one whose key cannot
        be told; None when it read no report there (no ``report.json``: an
        interrupted or refused run), **or other bytes than the caller's read**:
        `sha256` is the digest of the ``report.json`` the caller read in that
        directory (None when it read none), so no key is given for bytes the
        caller did not read (a report replaced between the two reads)."""
        if self._other_bytes(run_id, sha256):
            return None
        return self._placed(run_id)

    def run_status(self, run_id: str, sha256: Optional[str]) -> Optional[KeyStatus]:
        """D5's state of the key the report in run directory `run_id` is under
        (`status`), for the bytes the caller read there (`sha256`, as
        `run_key`); ``cannot_be_checked`` for a report whose key cannot be told,
        with the store's unplaced problems, and for one the scan read as other
        bytes, saying so; None when the scan read no report there."""
        placed = self._placed(run_id)
        if placed is None:
            return None
        if self._other_bytes(run_id, sha256):
            return KeyStatus("cannot_be_checked", (run_id,), problems=(
                f"the report in {run_id} is not the one the store's comparison read (it read "
                f"{self.digests[run_id][:12]}..., this read "
                f"{sha256[:12] + '...' if sha256 else 'no report'}): it was replaced between "
                "the two reads",))
        key, _problem = placed
        if key is None:
            return KeyStatus("cannot_be_checked", (run_id,), problems=self._unplaced_problems())
        return self.status(key)


def scan_report_store(
    version: "CohortVersionRead",
    *,
    known: Optional[Tuple[str, Dict[str, Any]]] = None,
) -> StoreScan:
    """Read every report a version's store keeps, once each, and group the
    records `build_record` makes of them by key.

    Run directories are taken by name (`RUN_ID_PATTERN`), each resolved and
    checked (`report_run_directory`); one without ``report.json`` is an
    interrupted write, which nothing registers, and is passed over. A
    ``report.json`` that is a link, a report that cannot be read or parsed,
    and a run directory a link takes out of the store are problems whose key
    cannot be told; a report that parses but that `build_record` refuses is a
    problem of the key its manifest names (`report_key`), when it names one.

    ``known`` is a report already read by the caller, as ``(run id, record)``:
    its run directory is not read again, so a registration reads the report it
    registers once (BACKLOG item 18).
    """
    from src.evaluation.cohort_registry import (
        REPORT_FILE,
        RegistryRefused,
        is_run_id,
        report_run_directory,
        report_store_directory,
    )
    from src.evaluation.sidecar import build_record, record_key, report_key

    grouped: Dict[Tuple[Any, ...], Dict[str, Tuple[str, Dict[str, Any]]]] = {}
    unbuilt: Dict[Tuple[Any, ...], List[Tuple[str, str]]] = {}
    unplaced: List[Tuple[Optional[str], str]] = []

    runs: Dict[Tuple[Any, ...], List[str]] = {}
    digests: Dict[str, str] = {}

    def add(run_id: str, record: Dict[str, Any]) -> None:
        key = record_key(record)
        grouped.setdefault(key, {}).setdefault(record["source_artifact_digest"],
                                               (run_id, record))
        runs.setdefault(key, []).append(run_id)
        digests[run_id] = record["source_artifact_digest"]

    if known is not None:
        add(*known)
    names: List[str] = []
    reports: Optional[Path] = None
    try:
        reports = report_store_directory(version, done="Nothing is read from there.")
        if reports is not None:
            names = sorted(entry.name for entry in reports.iterdir())
    except (RegistryRefused, OSError) as exc:
        unplaced.append((None, f"the report store of {version.manifest.get('label')!r} could "
                               f"not be listed ({type(exc).__name__}: {exc})"))
    for name in names:
        if not is_run_id(name) or (known is not None and name == known[0]):
            continue
        try:
            run_dir = report_run_directory(reports, name)
        except RegistryRefused as exc:
            unplaced.append((name, str(exc)))
            continue
        if not run_dir.is_dir():
            continue
        path = run_dir / REPORT_FILE
        if not os.path.lexists(path):
            continue
        if path.is_symlink():
            unplaced.append((name, f"{path} is a link, so the report it reaches is not one this "
                                   "store keeps"))
            continue
        try:
            report, digest = read_report(path)
        except (OSError, ValueError) as exc:
            unplaced.append((name, f"{path} could not be read: {type(exc).__name__}: {exc}"))
            continue
        digests[name] = digest
        try:
            record = build_record(report, digest)
        except (ValueError, RecursionError) as exc:
            # A report nested deeper than `build_record` can copy (a hand edit)
            # is one that cannot be compared, like any other it refuses.
            key = report_key(report)
            problem = (f"{path} cannot be compared: "
                       + ("it nests too deeply to be read" if isinstance(exc, RecursionError)
                          else str(exc)))
            if key is None:
                unplaced.append((name, problem))
            else:
                unbuilt.setdefault(key, []).append((name, problem))
            continue
        del report
        add(name, record)
    return StoreScan(version, grouped, {key: tuple(p) for key, p in unbuilt.items()},
                     tuple(unplaced), {key: tuple(sorted(r)) for key, r in runs.items()},
                     digests)


# ---------------------------------------------------------------------------
# The append-only files: usage history and investigation notes
# ---------------------------------------------------------------------------
def _append_entry(version: "CohortVersionRead", filename: str, entry: Dict[str, Any]) -> None:
    """One JSON object as one line at the end of a store file, written and
    flushed to disk.

    The store directory is resolved and checked first (`report_store_directory`,
    created when absent), and the file must be a plain file there: a link is
    refused (``O_NOFOLLOW`` where the platform has it, and a check by name
    everywhere), so an entry never lands outside the registry. Opened for
    appending; nothing locks it, so it is one writer at a time, as the ledger is.

    **An entry always starts a line of its own.** A file whose last byte is not
    a newline ends in an append that was interrupted (a crash, a power loss), and
    the entry is written after a newline that closes it, so the reader names that
    line and still reads this one. **A write that fails is taken back**: the file
    is cut back to the size it had before, so a failed append leaves no part of
    its entry; if even that fails, the error says the end of the file may hold
    part of it. The file is never cut below the size it had when opened.
    """
    from src.evaluation.cohort_registry import RegistryRefused, report_store_directory

    line = (json.dumps(entry, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("ascii")
    reports = report_store_directory(version, create=True)
    path = reports / filename
    if path.is_symlink() or (os.path.lexists(path) and not path.is_file()):
        raise RegistryRefused(
            f"{path} is not a plain file, so nothing is appended to it: an entry written "
            "through a link could land outside the registry. Replace it with the file it "
            "names. Nothing was written."
        )
    # Read and write: the last byte is read to see whether the file ends a line.
    # With O_APPEND every write still goes to the end, wherever the offset is.
    flags = (os.O_RDWR | os.O_APPEND | os.O_CREAT
             | getattr(os, "O_BINARY", 0)      # Windows: no newline translation
             | getattr(os, "O_NOINHERIT", 0)   # Windows: not inherited by children
             | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NOFOLLOW", 0))   # POSIX: refuse a link swapped in since
    fd = os.open(path, flags, 0o666)
    try:
        size = os.fstat(fd).st_size
        if size:
            os.lseek(fd, size - 1, os.SEEK_SET)
            if os.read(fd, 1) != b"\n":
                line = b"\n" + line
        try:
            view = memoryview(line)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        except OSError as exc:
            try:
                os.ftruncate(fd, size)  # Windows too (`_chsize_s`)
            except OSError as undo:
                raise OSError(
                    exc.errno,
                    f"{exc.strerror or exc}; and {path} could not be cut back to the "
                    f"{size} bytes it had ({type(undo).__name__}: {undo}), so its end may "
                    "hold part of this entry, which a reader names as a line that is not "
                    "an entry",
                ) from exc
            raise
    finally:
        os.close(fd)


def _read_entries(
    version: "CohortVersionRead", filename: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Every entry of a store file, in the order written, and the problems
    found: a line that is not a JSON object is named by its number, never
    skipped silently and never guessed at. No file is no entries."""
    from src.evaluation.cohort_registry import report_store_directory
    from src.utils.fingerprint import read_once

    reports = report_store_directory(version, done="Nothing is read from there.")
    if reports is None:
        return [], []
    path = reports / filename
    try:
        read = read_once(path)
    except FileNotFoundError:
        return [], []
    try:
        text = read.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [], [f"{path} is not UTF-8 ({exc}); none of its entries is read"]
    del read
    lines = text.split("\n")
    finished = lines[-1] == ""
    if finished:
        lines.pop()
    entries: List[Dict[str, Any]] = []
    problems: List[str] = []
    for number, line in enumerate(lines, start=1):
        try:
            entry = json.loads(line)
        except (ValueError, RecursionError):
            entry = None
        if isinstance(entry, dict):
            entries.append(entry)
            continue
        tail = (" (its last line is unfinished: an interrupted append)"
                if number == len(lines) and not finished else "")
        problems.append(f"{path} line {number} is not a JSON object{tail}")
    return entries, problems


def _usage(kind: str, at: Optional[datetime], **fields: Any) -> Dict[str, Any]:
    return {"kind": kind, "at": _utc(at), **fields}


def record_test_run(
    version: "CohortVersionRead",
    *,
    run_id: str,
    report_sha256: str,
    checkpoint_sha256: str,
    metrics: Mapping[str, Any],
    n_ranked: int,
    deterministic: bool,
    device: str,
    record_requested: bool,
    at: Optional[datetime] = None,
    mode: str = "C",
) -> Dict[str, Any]:
    """Append one measurement of the version to its usage history: when, which
    weights, which report, its measurement mode and its result (D7: "when it
    ran, on which checkpoint, and its result"), aggregates only. Written by the
    measurement right after its report is published, so a run from the command
    line counts as one from the Test tab. Returns the entry.

    **`mode` is an added key** (Batch 3, Mode D): an entry written before it has
    none and reads as Mode C, the only mode that wrote any (`entry_mode`)."""
    entry = _usage("test_run", at, run_id=run_id, report_sha256=report_sha256,
                   checkpoint_sha256=checkpoint_sha256, mode=str(mode), metrics=dict(metrics),
                   n_ranked=n_ranked, deterministic=bool(deterministic), device=str(device),
                   record_requested=bool(record_requested))
    _append_entry(version, USAGE_FILE, entry)
    return entry


def entry_mode(entry: Mapping[str, Any]) -> Optional[str]:
    """A `test_run` entry's measurement mode: its `mode`; "C" for an entry
    written before the key existed (the key absent), when Mode C was the only
    mode that wrote; None when the key is there and is not a non-empty string (a
    hand edit), which is never taken for Mode C."""
    if "mode" not in entry:
        return "C"
    mode = entry.get("mode")
    return mode if isinstance(mode, str) and mode.strip() else None


def record_registration(
    version: "CohortVersionRead",
    *,
    run_id: str,
    report_sha256: str,
    checkpoint_sha256: str,
    outcome: str,
    sequence: int,
    at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Append one registration of a report of the version (``registered`` or
    ``corroborated``: a ledger was written) to its usage history. Written by the
    registration function, whichever entry point called it."""
    entry = _usage("registered", at, run_id=run_id, report_sha256=report_sha256,
                   checkpoint_sha256=checkpoint_sha256, outcome=outcome, sequence=sequence)
    _append_entry(version, USAGE_FILE, entry)
    return entry


def record_model_load(
    version: "CohortVersionRead",
    *,
    checkpoint_sha256: str,
    metric: str,
    scoring_semantics_digest: str,
    loaded: bool,
    why: Optional[str] = None,
    at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Append one manual Load of a model chosen from a list ordered by this
    version's score (D4; §4.9 question 1) to its usage history: which weights,
    which metric and scoring group ordered the list, and whether the load took
    effect (with why, when it did not). Recording it does not undo that the
    version took part in choosing a model. For the load handler (M5-2)."""
    entry = _usage("model_load", at, checkpoint_sha256=checkpoint_sha256, metric=metric,
                   scoring_semantics_digest=scoring_semantics_digest, loaded=bool(loaded),
                   why=why)
    _append_entry(version, USAGE_FILE, entry)
    return entry


def read_usage_history(
    version: "CohortVersionRead",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The version's usage history, oldest first, and any problem found reading
    it (a line that is not an entry is named, not skipped)."""
    return _read_entries(version, USAGE_FILE)


def _key_tuple(key: Any) -> Tuple[Any, ...]:
    from src.evaluation.sidecar import KEY_FIELDS

    if isinstance(key, Mapping):
        key = tuple(key.get(field) for field in KEY_FIELDS)
    key = tuple(key) if isinstance(key, (tuple, list)) else ()
    if len(key) != len(KEY_FIELDS) or not all(isinstance(part, str) and part for part in key):
        raise ValueError(f"a key is the four fields {', '.join(KEY_FIELDS)}, as record_key "
                         "gives them. Nothing was written.")
    return key


def add_investigation_note(
    version: "CohortVersionRead",
    key: Any,
    *,
    who: str,
    what: str,
    at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Keep an investigation note with a key's reports (D5): who, when and what
    was found. Never in the ledger, and nothing reads it to decide anything: a
    note does not lift the block, delete a report or change a record.

    ``key`` is `record_key`'s tuple (or a mapping of `KEY_FIELDS`), and a report
    in this version's store must be under it, so a note is kept with the reports
    it concerns. ``who`` is what the operator gives; nothing verifies it under
    deployment mode M1. Returns the entry written.

    Raises:
        ValueError: the key, ``who`` or ``what`` is not given, or no report in
            the store is under the key. Nothing was written.
    """
    from src.evaluation.sidecar import KEY_FIELDS

    key = _key_tuple(key)
    for name, value in (("who", who), ("what", what)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"an investigation note needs {name!r}, a non-empty text. "
                             "Nothing was written.")
    scan = scan_report_store(version)
    if key not in scan.reports and key not in scan.unbuilt:
        raise ValueError(
            f"no report in the store of cohort version {version.manifest.get('label')!r} is "
            "under this key, and a note is kept with the reports it concerns. Nothing was "
            "written."
        )
    entry = {"kind": "investigation_note", "key": dict(zip(KEY_FIELDS, key, strict=True)),
             "who": who.strip(), "at": _utc(at), "what": what.strip()}
    _append_entry(version, NOTES_FILE, entry)
    return entry


def read_investigation_notes(
    version: "CohortVersionRead", key: Any = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The investigation notes kept in the version's store, oldest first, those
    of one key when it is given, and any problem found reading them."""
    from src.evaluation.sidecar import KEY_FIELDS

    entries, problems = _read_entries(version, NOTES_FILE)
    if key is None:
        return entries, problems
    wanted = dict(zip(KEY_FIELDS, _key_tuple(key), strict=True))
    return [entry for entry in entries if entry.get("key") == wanted], problems


__all__ = [
    "CONTEST_STATES",
    "KeyStatus",
    "NOTES_FILE",
    "STORE_FILES",
    "NotInReportStore",
    "NotTheDeploymentRegistry",
    "StoreReport",
    "StoreScan",
    "USAGE_FILE",
    "USAGE_KINDS",
    "add_investigation_note",
    "locate_store_report",
    "read_investigation_notes",
    "read_report",
    "read_usage_history",
    "record_model_load",
    "record_registration",
    "record_test_run",
    "scan_report_store",
    "store_report_parts",
]
