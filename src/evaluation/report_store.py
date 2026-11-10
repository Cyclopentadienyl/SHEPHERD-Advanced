"""
The report store: where a cohort version's reports are kept, found and compared.
================================================================================
`docs/working/PLAN_TEST_RESULTS.md` D4 and D5 (as amended 2026-10-19), M5's
boundaries, policy rev 16 (K8-6r.2). A registered cohort version's report store
is its ``reports/`` directory
(`src.evaluation.cohort_registry.report_store_directory`):

    <registry>/versions/<label>/reports/
        <run id>/ranks.json, <run id>/report.json   one measurement each (a KG
                                                    edge set's run: scores.bin,
                                                    then report.json)
        usage.jsonl                                 the version's usage history
        notes.jsonl                                 investigation notes
        usage.jsonl.lock, notes.jsonl.lock          their locks (K8-6r.1)

**A report is registered only from here** (D5, 2026-10-19). A run is its run
directory in the version's store: the ledger's record names it, and the same
run is recognised by it. A copy elsewhere has no run, so it is not registered.
`locate_store_report` therefore finds a report's version through the
registry's own structure -- ``<registry>/versions/<label>/reports/<run id>/
report.json``, read from the report's resolved path -- and resolves that
version (`resolve_version`), refusing whatever does not fit rather than
guessing or falling back to a copy.

**And only from the deployment's registry** (BACKLOG 37): the registry that
structure names must be the one ``SHEPHERD_REGISTRY`` names, compared by
resolved path (`server_settings.deployment_registry_refusal`), checked before
anything in it is read. A copy of the whole registry holds the same bytes and
would pass every check above while keeping its own store and usage history,
which the service never reads. A registry moved as a whole still works
once ``SHEPHERD_REGISTRY`` names its new place; its version is the same version
by the digest of its ``cohort.json``.

**The store scan lists; it never gates** (`scan_report_store`; policy rev 16,
from K8-6r.2). Every report the store keeps is built into a record by
`build_record`, grouped by its condition group (`record_key`) and listed by its
run, byte-identical runs included. A group's description (`StoreScan.status`,
one of `CONTEST_STATES`) compares its distinct reports with `differing_fields`:
``contested`` means runs under these conditions differ, ``cannot_be_checked``
that the comparison is incomplete (a report of the group that cannot be built,
or a store that cannot be listed), ``uncontested`` neither. The names stay as
machine values; they **describe and decide nothing**: no registration reads
the scan, and no surface disables, hides or refuses a run by them. A report
that cannot be read, a link, or a run directory a link takes out of the store
is listed by its run with its own reason only; it never marks another run or
another group.

**Investigation notes** (who, when, what) are kept here with the group's
reports, never in the ledger: a note records what was found; it decides
nothing. "Who" is what the operator types; under deployment mode M1 nothing
verifies it.

**The usage history** is one append-only file per version, written by the
shared code (the measurement, the registration function), so a run started from
the command line counts as one started from the Test tab. It makes the use of a
cohort visible; it does not prevent it (D4). A ``model_load`` entry, written by
a Load from a list ordered by test score until that ordering retired (K8-6d), is
kept and read as written; nothing writes one now.

**What it rests on: plain files, and one append at a time per file** (D2, D7;
K8-6r.1). The files can be edited; the tools refuse, they do not make a refusal
tamper-proof. Each append holds that file's lock (``usage.jsonl.lock``, ``notes.jsonl.lock``,
beside it, never removed) across processes while it writes one line at the end
of the file, after closing a line an interrupted append left open, and while it
takes back a write that fails (`_append_entry`), so one writer's failed append
never removes another writer's line. The lock is cooperative, and per file: a
scan is not atomic with a run publishing a new report into the same version at
that moment, and lists what it read.

Module: src/evaluation/report_store.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from src.utils.file_lock import exclusive_lock

if TYPE_CHECKING:
    from src.evaluation.cohort_registry import CohortVersionRead

#: The two append-only files a report store keeps beside its run directories.
USAGE_FILE = "usage.jsonl"
NOTES_FILE = "notes.jsonl"
#: Both: the store's own files, which are not runs
#: (`cohort_registry.list_report_runs` leaves them out).
STORE_FILES = (USAGE_FILE, NOTES_FILE)
#: Each store file's lock, beside it (`_append_entry`): the file's name with
#: this added. Never removed, never read, and not runs either
#: (`list_report_runs` leaves them out with `STORE_FILES`).
STORE_LOCK_SUFFIX = ".lock"
STORE_LOCK_FILES = tuple(name + STORE_LOCK_SUFFIX for name in STORE_FILES)

#: How long an append waits for another writer of the same store file before
#: it raises `FileLockBusy`, nothing written; read at each call.
APPEND_LOCK_GRACE_SECONDS = 10.0

#: What the usage history records (D4): every measurement of the version and every
#: registration of one of its reports; ``model_load``, a model loaded from a list
#: once ordered by its score, stays a kind its kept entries are read as (no longer
#: written, K8-6d).
USAGE_KINDS: Tuple[str, ...] = ("test_run", "registered", "model_load")

#: A condition group's description in the store (`StoreScan.status`), kept as
#: machine values with a new meaning since K8-6r.2 (policy rev 16): its runs
#: agree (or there is at most one); runs under these conditions differ; or the
#: comparison is incomplete. **None decides anything**: no registration, display,
#: query or export is refused, disabled or hidden by it.
CONTEST_STATES: Tuple[str, ...] = ("uncontested", "contested", "cannot_be_checked")


class NotInReportStore(ValueError):
    """The path is not a report the registry's report store holds, so it is not
    registered (D5): a copy, wherever it is, is no run of the store, and a
    record names the run it came from. ``why`` is the reason alone, without the advice to
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
        "<registry>/versions/<label>/reports/<run id>/report.json, because a record names "
        "the run directory its report was kept in (D5); a copy, which can still be read and "
        "shared, is no run. Register the store's original instead: the path the "
        "measurement printed as 'Report:'."
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
# The store scan: every run listed, each group described (policy rev 16)
# ---------------------------------------------------------------------------
class KeyStatus(NamedTuple):
    """One condition group's description in a version's report store
    (`CONTEST_STATES`, machine values that decide nothing).

    ``run_ids`` are the runs whose readable reports are under the group,
    byte-identical ones included; for ``contested`` (runs under these
    conditions differ), ``differing`` names what differs (the metric names when
    only the metrics do) and ``disagreeing_runs`` every run carrying a differing
    report; for ``cannot_be_checked`` (the comparison is incomplete),
    ``problems`` says why, naming each file."""

    state: str
    run_ids: Tuple[str, ...]
    differing: Tuple[str, ...] = ()
    disagreeing_runs: Tuple[str, ...] = ()
    problems: Tuple[str, ...] = ()


class StoreScan(NamedTuple):
    """Every report a version's store keeps, built into records, grouped by
    condition group and listed by run (`scan_report_store`).

    ``reports`` maps a group to its runs, ``{run id: (run id, record)}``: every
    run is named, byte-identical ones included. ``runs`` maps a group to the
    same run ids, sorted. ``unbuilt`` maps a group to the reports under it that
    cannot be compared, as ``(run id, problem)``: a note on that group's
    description. ``unplaced`` holds the reports whose group cannot be told, as
    ``(run id, problem)``, each listed by its run with its own reason only, and
    the store itself when it could not be listed (run id None), which every
    group's description then names. **Every report the scan read is named by
    its run id**, so a run's state is found by its id (`run_status`), never by
    reading ids out of a problem's words. ``digests`` maps each run id whose
    ``report.json`` the scan read to the digest of those bytes, so a lookup by
    run id is bound to the bytes the caller read there (`run_key`,
    `run_status`).
    """

    version: "CohortVersionRead"
    reports: Dict[Tuple[Any, ...], Dict[str, Tuple[str, Dict[str, Any]]]]
    unbuilt: Dict[Tuple[Any, ...], Tuple[Tuple[str, str], ...]]
    unplaced: Tuple[Tuple[Optional[str], str], ...]
    runs: Dict[Tuple[Any, ...], Tuple[str, ...]]
    digests: Dict[str, str]

    def status(self, key: Tuple[Any, ...]) -> KeyStatus:
        """The group's description: ``contested`` when any two of its distinct
        reports differ on a compared field (each distinct digest compared once;
        every run carrying a differing one named); otherwise
        ``cannot_be_checked`` when a report of the group could not be compared
        or the store could not be listed; otherwise ``uncontested``, also for a
        group no report in the store is under. Decides nothing."""
        from src.evaluation.sidecar import difference_names, differing_fields

        runs_of: Dict[str, List[str]] = {}
        first: Dict[str, Dict[str, Any]] = {}
        for run_id, record in self.reports.get(tuple(key), {}).values():
            digest = record.get("source_artifact_digest")
            runs_of.setdefault(digest, []).append(run_id)
            first.setdefault(digest, record)
        run_ids = self.runs.get(tuple(key), ())
        distinct = list(first.items())
        differing: List[str] = []
        disagreeing: set = set()
        for index, (digest_a, record_a) in enumerate(distinct):
            for digest_b, record_b in distinct[index + 1:]:
                fields = differing_fields(record_a, record_b)
                if fields:
                    for name in difference_names(record_a, record_b, fields):
                        if name not in differing:
                            differing.append(name)
                    disagreeing |= set(runs_of[digest_a]) | set(runs_of[digest_b])
        if differing:
            return KeyStatus("contested", run_ids, tuple(differing),
                             tuple(sorted(disagreeing)))
        problems = (tuple(problem for _run, problem in self.unbuilt.get(tuple(key), ()))
                    + self._unplaced_problems())
        if problems:
            return KeyStatus("cannot_be_checked", run_ids, problems=problems)
        return KeyStatus("uncontested", run_ids)

    def state(self, key: Tuple[Any, ...]) -> str:
        """The group's description alone, one of `CONTEST_STATES`."""
        return self.status(key).state

    def _unplaced_problems(self) -> Tuple[str, ...]:
        """The store-level problem only (run id None: the store could not be
        listed). A run's own unreadable report is never another group's."""
        return tuple(problem for run, problem in self.unplaced if run is None)

    def _placed(self, run_id: str) -> Optional[Tuple[Optional[Tuple[Any, ...]], Optional[str]]]:
        for key, run_ids in self.runs.items():
            if run_id in run_ids:
                return tuple(key), None
        for key, entries in self.unbuilt.items():
            for run, problem in entries:
                if run == run_id:
                    return tuple(key), problem
        for run, problem in self.unplaced:
            if run is not None and run == run_id:
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
        a group it could not compare, ``(None, problem)`` for one whose group
        cannot be told; None when it read no report there (no ``report.json``:
        an interrupted or refused run), **or other bytes than the caller's
        read**: `sha256` is the digest of the ``report.json`` the caller read in
        that directory (None when it read none), so no group is given for bytes
        the caller did not read (a report replaced between the two reads). The
        run's record is then ``reports[key][run_id]``."""
        if self._other_bytes(run_id, sha256):
            return None
        return self._placed(run_id)

    def run_status(self, run_id: str, sha256: Optional[str]) -> Optional[KeyStatus]:
        """The description of the group the report in run directory `run_id`
        is under (`status`), for the bytes the caller read there (`sha256`, as
        `run_key`); for a report whose group cannot be told, ``cannot_be_checked``
        with **that run's own reason only**, and for one the scan read as other
        bytes, saying so; for any run of a store that could not be listed,
        ``cannot_be_checked`` saying so; otherwise None when the scan read no
        report there."""
        placed = self._placed(run_id)
        if placed is None:
            listing = self._unplaced_problems()
            return (KeyStatus("cannot_be_checked", (run_id,), problems=listing)
                    if listing else None)
        if self._other_bytes(run_id, sha256):
            return KeyStatus("cannot_be_checked", (run_id,), problems=(
                f"the report in {run_id} is not the one the store's comparison read (it read "
                f"{self.digests[run_id][:12]}..., this read "
                f"{sha256[:12] + '...' if sha256 else 'no report'}): it was replaced between "
                "the two reads",))
        key, problem = placed
        if key is None:
            return KeyStatus("cannot_be_checked", (run_id,), problems=(problem,))
        return self.status(key)


def scan_report_store(version: "CohortVersionRead") -> StoreScan:
    """Read every report a version's store keeps, once each, build the record
    `build_record` makes of each, and list them by condition group and run.

    **It lists; it never gates** (policy rev 16): no registration reads it, and
    a report it cannot read is listed by its run with its own reason, kept as
    written, never repaired and never assigned to a model by guessing.

    Run directories are taken by name (`RUN_ID_PATTERN`), each resolved and
    checked (`report_run_directory`); one without ``report.json`` is an
    interrupted write, which nothing registers, and is passed over. A
    ``report.json`` that is a link, a report that cannot be read or parsed,
    and a run directory a link takes out of the store are listed by their run
    (``unplaced``); a report that parses but that `build_record` refuses is a
    note on the group its manifest names (`report_key`), when it names one, and
    otherwise listed by its run likewise.
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
    digests: Dict[str, str] = {}

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
        if not is_run_id(name):
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
        grouped.setdefault(record_key(record), {})[name] = (name, record)
    return StoreScan(version, grouped, {key: tuple(p) for key, p in unbuilt.items()},
                     tuple(unplaced), {key: tuple(sorted(runs)) for key, runs in grouped.items()},
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
    appending.

    **One append at a time per file, across processes** (K8-6r.1): the file's
    lock (``<file>.lock`` beside it, `STORE_LOCK_FILES`, never removed) is held
    from the size read to the failure undo, so no other writer appends between
    the size recorded and a cut back to it. A lock not obtained within
    `APPEND_LOCK_GRACE_SECONDS` raises `FileLockBusy` (an `OSError`) naming it,
    nothing written, and each caller reports it as it reports a failed write.
    The lock file is refused like the data file when it is a link: by name
    first, then opened with ``O_NOFOLLOW`` where the platform has it, so a link
    swapped in after the check fails to open rather than being followed. This
    calls nothing that takes a ledger or GPU lock.

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
    lock = reports / (filename + STORE_LOCK_SUFFIX)
    if lock.is_symlink() or (os.path.lexists(lock) and not lock.is_file()):
        raise RegistryRefused(
            f"{lock} is not a plain file, so {path} is not locked through it: a lock taken "
            "through a link could create or lock a file outside the registry. Replace it "
            "with a plain file or remove it. Nothing was written."
        )
    # Read and write: the last byte is read to see whether the file ends a line.
    # With O_APPEND every write still goes to the end, wherever the offset is.
    flags = (os.O_RDWR | os.O_APPEND | os.O_CREAT
             | getattr(os, "O_BINARY", 0)      # Windows: no newline translation
             | getattr(os, "O_NOINHERIT", 0)   # Windows: not inherited by children
             | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NOFOLLOW", 0))   # POSIX: refuse a link swapped in since
    with exclusive_lock(lock, grace=APPEND_LOCK_GRACE_SECONDS, nofollow=True):
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
                        f"{size} bytes it had ({type(undo).__name__}: {undo}), so its end "
                        "may hold part of this entry, which a reader names as a line that "
                        "is not an entry",
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
    mode: str,
    at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Append one measurement of the version to its usage history: when, which
    weights, which report, its measurement mode and its result (D7: "when it
    ran, on which checkpoint, and its result"), aggregates only. Written by the
    measurement right after its report is published, so a run from the command
    line counts as one from the Test tab. Returns the entry.

    **`mode` is required**, the run's own manifest mode: nothing is recorded
    under a mode the caller did not name. An entry written before the key
    existed (Batch 3) has none, and reads as a mode not recorded (`entry_mode`):
    no mode is inferred for it (K8-6e.2)."""
    entry = _usage("test_run", at, run_id=run_id, report_sha256=report_sha256,
                   checkpoint_sha256=checkpoint_sha256, mode=str(mode), metrics=dict(metrics),
                   n_ranked=n_ranked, deterministic=bool(deterministic), device=str(device),
                   record_requested=bool(record_requested))
    _append_entry(version, USAGE_FILE, entry)
    return entry


def entry_mode(entry: Mapping[str, Any]) -> Optional[str]:
    """A `test_run` entry's measurement mode: its `mode` when that is a
    non-empty string; None otherwise, absent (an entry written before the key
    existed) or damaged (a hand edit) alike. No mode is ever inferred."""
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
    """Append one registration of a report of the version (``registered``: a
    ledger was written, naming the run) to its usage history. Written by the
    registration function, whichever entry point called it. An entry written
    before K8-6r.2 with the retired outcome ``corroborated`` is read as written.
    The usage history decides nothing: registration may cite it in a note."""
    entry = _usage("registered", at, run_id=run_id, report_sha256=report_sha256,
                   checkpoint_sha256=checkpoint_sha256, outcome=outcome, sequence=sequence)
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
    """Keep an investigation note with a group's reports (D5): who, when and
    what was found. Never in the ledger, and nothing reads it to decide
    anything: a note does not delete a report or change a record.

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
    "APPEND_LOCK_GRACE_SECONDS",
    "STORE_FILES",
    "STORE_LOCK_FILES",
    "STORE_LOCK_SUFFIX",
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
    "record_registration",
    "record_test_run",
    "scan_report_store",
    "store_report_parts",
]
