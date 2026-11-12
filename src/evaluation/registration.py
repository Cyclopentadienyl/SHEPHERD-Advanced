"""
Registering a measurement report in the ledger beside its checkpoint.
=====================================================================
`PLAN_TEST_RESULTS.md` D5 and §6 phase 1, step 2: a report is registered
automatically right after its run (`scripts/measure_scorer.py --record`) or later
by hand (`scripts/record_evaluation.py --report`), and **both go through
`register_report`**. Two implementations of "turn a report into a ledger record"
could disagree about what a report says; one cannot.

**Only a report the registry's report store holds** (D5, amended 2026-10-19).
The report's version is found through the registry's own structure,
``<registry>/versions/<label>/reports/<run id>/report.json``, and resolved
(`src.evaluation.report_store.locate_store_report`); a report anywhere else is
refused, saying to register the store's original, and nothing guesses where a
copy came from. Its place is not taken as proof: the report must name that
version exactly (`cohort_version_block`, the block the measurement writes from
the same version), its run directory must be named for the checkpoint it
measured, and the checkpoint checks below still hold.

**And only from the deployment's registry** (BACKLOG 37): the registry the
report's path names must be the one ``SHEPHERD_REGISTRY`` names, compared by
resolved path (`server_settings.deployment_registry_refusal`), so a copy of the
whole registry -- the same bytes, its own report store and usage history
-- is refused before anything in it is read (`NotFromTheDeploymentRegistry`).
Unset, every registration is refused saying to set it. A registry moved as a
whole still works once ``SHEPHERD_REGISTRY`` names its new place.

**Registered by the verified run** (policy rev 16, from K8-6r.2). A run is its
run directory in the version's report store (`StoreReport.run_id`), verified
with its cohort version, its report's digest and its checkpoint. The ledger
looks for this run first (`sidecar.find_run`): the same run with the same report
bytes is ``already_registered`` and nothing is written; the same run with other
bytes is refused, since a run's report is sealed when it is published. Any
other verified run gets a record of its own, whatever its bytes, its values or
the times involved. **Only the selected run is read**: no other report in the
store is opened, and none can block or mark this registration. How this run
relates to its condition group (`RegistrationResult.group_differences`) and to
records written before each run had its own record
(`RegistrationResult.historical_overlap`) is described after the decision and
decides nothing; the usage history may be cited in that description as a clue,
never as the authority.

**Each file is parsed and identified from one read** (BACKLOG item 18). The report
is read once, and the record is built from those bytes and names them by their
digest; the ledger is read once, and the write's stale-write guard
compares the file with the digest of what was parsed. A report replaced while it
was being registered therefore cannot be recorded as one file's numbers under
the other's digest, and another writer's append between this read and this
write is refused rather than erased.

**One registration at a time per ledger, across processes** (K8-6r.1). The
section from reading the ledger to writing the new file and its usage-history
entry -- the read, whether the report is already held, the sequence, the write
and the entry -- runs while the ledger's lock is held (`sidecar.ledger_lock`,
``evaluations.json.lock`` beside the ledger). It is taken after every check
that reads the report or hashes the checkpoint. A registration that cannot take
it within the grace is refused as busy (`LedgerBusy`): nothing was written this
time, the report stays kept, and it may be registered again. The stale-write
guard stays as the second line, for a writer that takes no lock. The lock is
cooperative.

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

**A registration from the Test page is bound to the report it showed**
(``expected_report_sha256``, the M5 review's P2-1): the digest shown is
compared on the one read registered here, before its bytes are decoded, so a
file replaced after it was shown and before that read -- by another valid
report or damaged bytes -- is refused and never parsed; a file replaced after
that read is registered as the bytes the read holds, the ones shown, and the
replacement is never recorded. **That read is the page's only one** (round 3
of the review fixes, F2): where the ledger goes is found from it too, by the
page's own placement passed in (``place``). The page's answer for whether a
report is registered in the ledger its Register would write comes from one
read in the same way (`report_state`'s ``place``), so the page neither reads
nor judges a report itself, to register it or to say whether it is
registered.

**A registration is in the version's usage history** (D4): a ledger written is
appended there by this function, inside the ledger's lock, so a registration
from the command line counts as one from the Test tab. The ledger and the usage
history are two files: if that append fails or finds the usage history busy
after the ledger was written, the record stands and the result says "the usage
history was not written" with the reason (`usage_history_problem`); the ledger
is never rolled back.

**Every expected failure is one refusal** (`RegistrationRefused`) whose message
names the file and says nothing was written: a missing or damaged report, a file
that is not a report or not in a report store, a report in a registry other
than the deployment's (`NotFromTheDeploymentRegistry`), a report that is no
longer the one shown, a ledger another registration is writing (`LedgerBusy`),
a generated split's report, a
report that does not name the version it is kept under, a ledger this revision
does not read, a ledger the write cannot complete safely in (a record without a
readable registration sequence, an unreadable run-identity boundary, this run's
own identity record damaged), this run already registered from other bytes (a
replaced report), a stale write, a checkpoint file or directory that does not
hold the weights the report measured, and, for an automatic registration, a
report file that is no longer the one the run published
(`ReportNotAsPublished`). No refusal comes from another run's report or from
how this run's results compare with another's. The entry points print it and
exit; neither shows a traceback for these.

**Kept? Registered?** `report_state` answers both for one report from disk, the
way the publication contract defines them, for the job runner and the Test tab
(M5's acceptance: never inferred from a job's exit alone).

Module: src/evaluation/registration.py
"""
from __future__ import annotations

import os
import re
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, NamedTuple, Optional, Tuple

from src.evaluation.report_store import (
    NotInReportStore,
    NotTheDeploymentRegistry,
    _NotTheExpectedBytes,
    locate_store_report,
    read_report,
    read_usage_history,
    record_registration,
    store_report_parts,
)
from src.evaluation.sidecar import (
    LEDGER_FILENAME,
    GroupDifference,
    HistoricalNote,
    append_record,
    build_record,
    checkpoint_matches,
    find_checkpoint,
    find_run,
    group_differences,
    held_record,
    historical_overlap,
    ledger_lock,
    ledger_lock_path,
    read_ledger,
    sealed_run_refusal,
    sequence_problem,
    write_ledger,
)
from src.utils.file_lock import FileLockBusy
from src.utils.fingerprint import ReadIdentity
from src.utils.interrupts import interrupt_deferred

#: Whether a report was kept, as the publication contract defines it: its
#: ``report.json`` is in the report store, holding the bytes the run published
#: when that digest is known (`report_state`).
KEPT_STATES: Tuple[str, ...] = ("kept", "not_kept", "cannot_be_checked")

#: Whether a report is registered: a record in the ledger beside its checkpoint
#: names this run with these report bytes (`report_state`, through
#: `sidecar.find_run`). A record written before each run had its own record that
#: holds the same bytes is a possible overlap, never "registered".
REGISTERED_STATES: Tuple[str, ...] = ("registered", "not_registered", "cannot_be_checked")

_SAYS_NOTHING_WRITTEN = re.compile(r"\s*Nothing was written\.?\s*$")

#: How ``usage_history_problem`` starts: the ledger was written and the usage
#: history was not (policy rev 16). The job runner's summary repeats a
#: registered run's own words when they carry it.
USAGE_HISTORY_NOT_WRITTEN = "the usage history was not written"

#: A report digest as the lists show it: 64 lowercase hexadecimal digits.
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RegistrationRefused(ValueError):
    """A report was not registered, and nothing was written."""


class LedgerBusy(RegistrationRefused):
    """Another registration held the ledger's lock for the whole grace
    (`sidecar.ledger_lock`), so this one read nothing from the ledger and wrote
    nothing to it this time. The report stays where it is kept and is not
    marked invalid: registering it again is expected to work once the other
    registration has finished."""


class NotFromTheDeploymentRegistry(RegistrationRefused):
    """The report is kept in a registry other than the deployment's -- the one
    ``SHEPHERD_REGISTRY`` names, compared by resolved path -- or the variable is
    unset or refused, so which registry is the deployment's cannot be told
    (`server_settings.deployment_registry_refusal`). Nothing in that registry
    was read. A run's own report stays where it was published, but no command
    to register it from there is offered: it would be refused the same way."""


class ReportNotAsPublished(RegistrationRefused):
    """An automatic registration could not read back the bytes the run
    published: the report file holds other bytes (replaced or changed after
    its publication, damaged included), is gone, cannot be read, or no longer
    resolves into the report store it was published into (replaced by a link
    to a file elsewhere, say). Nothing was written, and the report the run
    published may no longer exist, so it is not offered for a later
    registration."""


class MeasuredRun(NamedTuple):
    """What a measurement holds about itself when it registers its own report
    (`measure_scorer.py --record`): the identity of its checkpoint read (the path
    it read, resolved when it was read, and the digest of those bytes) and the
    digest `publish_bytes` returned for the report bytes it wrote. Taken from
    the run's own reads and writes; nothing is hashed again to make it."""

    checkpoint: ReadIdentity
    report_sha256: str


class RegistrationResult(NamedTuple):
    """What a registration did: one of ``REGISTRATION_OUTCOMES`` (``registered``
    or ``already_registered``), the record the ledger holds for this run (the
    one appended or the one found, `sidecar.held_record`), the ledger's path,
    and two descriptions that decide nothing (Decision 11):

    ``group_differences``: each other record of this run's condition group
    whose compared fields differ, with its sequence, its run (or none recorded)
    and the difference names (`sidecar.group_differences`; `group_note` words
    it). Both CLIs print it.

    ``usage_history_problem`` is `None` unless the ledger was written and the
    entry in the version's usage history could not be appended (a failed write,
    or that file's lock busy); it then says "the usage history was not written"
    with the reason, and the registration stands: the ledger is never rolled
    back.

    ``historical_overlap``: one sentence per record written before each run had
    its own record that may hold this report or whose relation cannot be
    checked (`sidecar.historical_overlap`), citing the version's usage history
    where it names this run's registration into that record; `historical_note`
    words them. Both CLIs print it.
    """

    outcome: str
    record: Dict[str, Any]
    ledger_path: Path
    group_differences: Tuple[GroupDifference, ...] = ()
    usage_history_problem: Optional[str] = None
    historical_overlap: Tuple[str, ...] = ()


class ReportState(NamedTuple):
    """Whether one report was kept and whether it is registered, each one of
    `KEPT_STATES` and `REGISTERED_STATES` with the reason in words.

    ``report_sha256`` is the digest of the bytes found (or the one expected,
    when the report could not be read); ``sequence`` the ledger record's when
    registered; ``ledger_path`` the ledger that was (or would be) read, `None`
    when no checkpoint was given and the report names none to find it by.
    ``historical_overlap``: when no record names this run, the sequences of
    the records merged before each run had its own record that hold these
    bytes (a possible overlap, never "registered"; the reason says it too), so
    a page can show it without reading words. ``replaced_after_registration``:
    a record names this run with other bytes (``not_registered``; the reason
    names the record and both digests, `sealed_run_refusal`), likewise.
    """

    kept: str
    why_kept: str
    registered: str
    why_registered: str
    report_sha256: Optional[str]
    sequence: Optional[int]
    ledger_path: Optional[Path]
    historical_overlap: Tuple[int, ...] = ()
    replaced_after_registration: bool = False


def group_note(result: RegistrationResult) -> Optional[str]:
    """The group note both entry points print: how this run's record differs
    from the other records of its condition group, or `None` when none differs.

    Names the records, their runs (a record written before each run had its own
    record: "not recorded") and the differing names, never the values: each
    record keeps its own. When both runs record deterministic algorithms on,
    says so and asks for an investigation. Informational: every run is kept.
    """
    if not result.group_differences:
        return None
    sequence = result.record.get("registration", {}).get("sequence")
    parts = []
    for other in result.group_differences:
        run = other.run_id or ("not recorded" if other.historical else "cannot be read")
        shown = (f"record {other.sequence}" if other.sequence is not None
                 else f"the record at index {other.index}")
        parts.append(f"{shown} (run {run}) on {', '.join(other.differing)}")
    note = (f"Runs under these conditions differ: record {sequence} (this run) differs from "
            + "; ".join(parts) + ". Each run is kept as recorded.")
    deterministic = [f"record {other.sequence}" if other.sequence is not None
                     else f"the record at index {other.index}"
                     for other in result.group_differences if other.deterministic]
    if deterministic:
        note += (f" This run and {', '.join(deterministic)} both record deterministic "
                 "algorithms on, so they were expected to agree: investigate.")
    return note


def historical_note(result: RegistrationResult) -> Optional[str]:
    """The historical note both entry points print: this run's possible overlap
    with records written before each run had its own record, or `None`.
    Informational: such a record never decided this outcome."""
    if not result.historical_overlap:
        return None
    return "; ".join(text[:1].upper() + text[1:] for text in result.historical_overlap) + "."


def _with_usage_clues(version: Any, notes: Tuple[HistoricalNote, ...], run_id: str,
                      checkpoint_sha256: str, report_sha256: str) -> Tuple[str, ...]:
    """Each note's sentence, a possible overlap citing the version's usage
    history when it holds a ``registered`` entry naming this run, its checkpoint,
    its report and that historical record's sequence. **A clue, never the
    authority**: read only when there is a possible overlap to note, and an
    absent, unreadable or silent usage history adds nothing and changes
    nothing."""
    entries: list = []
    if any(note.possible_overlap and note.sequence is not None for note in notes):
        try:
            entries, _problems = read_usage_history(version)
        except Exception:  # noqa: BLE001 - a clue that cannot be read is no clue
            entries = []
    texts = []
    for note in notes:
        text = note.text
        if note.possible_overlap and note.sequence is not None and any(
                isinstance(entry, dict) and entry.get("kind") == "registered"
                and entry.get("run_id") == run_id
                and entry.get("checkpoint_sha256") == checkpoint_sha256
                and entry.get("report_sha256") == report_sha256
                and type(entry.get("sequence")) is int and entry["sequence"] == note.sequence
                for entry in entries):
            text += (f" (the usage history names run {run_id}'s registration into record "
                     f"{note.sequence})")
        texts.append(text)
    return tuple(texts)


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
    place: Optional[Callable[[Mapping[str, Any]], Path]] = None,
    expected_report_sha256: Optional[str] = None,
    registered_at: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> RegistrationResult:
    """Register one measurement report in the ledger beside the checkpoint it
    measured.

    **The report must be one the registry's report store holds**
    (``<registry>/versions/<label>/reports/<run id>/report.json``,
    `locate_store_report`), naming that version exactly and measured on the
    checkpoint its run directory is named for. Its run is that run directory's
    id: the ledger's record names it (``run_id``), and the same run is
    recognised by it (`sidecar.find_run`). **No other report of the store is
    read**, so none blocks or marks this registration. **The registry must be the
    deployment's**: the one ``SHEPHERD_REGISTRY`` names, read from ``environ``
    (`os.environ` when `None`), compared by resolved path before anything in the
    registry is read (`NotFromTheDeploymentRegistry`, also when it is unset).

    **The ledger is always beside the checkpoint file's resolved path**
    (``Path(file).resolve().parent / "evaluations.json"``, M4 decision 15): the
    rule the read-only view reads by, for the checkpoint it serves. A checkpoint
    reached through a link files its ledger beside the file the link names,
    where the view looks; a ledger beside the link would be one the service
    never reads, and a second ledger for the same weights. Nothing creates a
    directory for a ledger.

    Exactly one of ``checkpoint``, ``checkpoint_dir``, ``measured`` and ``place``
    says which file that is:

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
    - ``place`` (the Test page's Register): a callable given the report's
      manifest as parsed here, from the one read compared with
      ``expected_report_sha256``, which returns the checkpoint file the ledger
      goes beside, found the caller's way (the page: in the server's lists),
      or raises ``ValueError`` saying why it names none, which is refused.
      That file must then hold the report's checkpoint bytes
      (`checkpoint_matches`), as with ``checkpoint``. Called after the report,
      its version and its run directory have been checked, so a refused report
      lists nothing. The caller never reads the report itself.

    ``expected_report_sha256``: the digest of the report as it was shown when it
    was chosen (the Test page's Register), with ``checkpoint``,
    ``checkpoint_dir`` or ``place``. The one read below is compared with it
    **before its bytes are decoded**, so a file replaced or changed after it
    was shown -- by another valid report or damaged bytes -- is refused and
    never parsed, the check ``measured`` makes with the digest its run
    published. Not with ``measured``, which carries its own; that, or a value
    that is not 64 lowercase hex digits, is a caller's error (``TypeError``).

    ``registered_at`` is the time the registration is made, now in UTC unless a
    caller passes one.

    **The ledger is held from its read to the usage-history entry**
    (`sidecar.ledger_lock`, K8-6r.1), after every check of the report and the
    checkpoint: another registration of the same ledger, from any process,
    waits for it. One that waits past the grace is refused as `LedgerBusy`,
    nothing written; the usage history's own lock is taken inside it, never
    the other way, and no GPU lock is taken here. Where this account may not
    create or open the lock file (a checkpoint directory it cannot write),
    nothing can be written: a report the ledger already holds still answers
    ``already_registered``, read without the lock as every reader reads, and
    anything else is refused naming that cause.

    Returns the outcome — ``registered`` (this run's own record) or
    ``already_registered`` (this run with these bytes is held; nothing written)
    — the record the ledger holds for this run, the group's differences and the
    historical note (informational), and, when the ledger was written and the
    version's usage history could not be appended, why
    (`RegistrationResult.usage_history_problem`). Every expected failure raises
    ``RegistrationRefused``; a call naming none, or more than one, of
    ``checkpoint``, ``checkpoint_dir``, ``measured`` and ``place`` is a caller's
    error (``TypeError``).
    """
    named = [arg for arg in (checkpoint, checkpoint_dir, measured, place) if arg is not None]
    if len(named) != 1:
        raise TypeError(
            "register_report takes exactly one of checkpoint, checkpoint_dir, measured and place"
        )
    if measured is not None and not isinstance(measured, MeasuredRun):
        raise TypeError("measured is the run's MeasuredRun")
    if expected_report_sha256 is not None:
        if measured is not None:
            raise TypeError("measured carries the digest of the report its run published; "
                            "expected_report_sha256 is for a report chosen by the digest shown")
        if (not isinstance(expected_report_sha256, str)
                or _SHA256.fullmatch(expected_report_sha256) is None):
            raise TypeError("expected_report_sha256 is a SHA-256 in 64 lowercase hex digits")
    expected = measured.report_sha256 if measured is not None else expected_report_sha256
    report_path = Path(report_path)
    # Known up front for a named file or a run's read; for a directory or a
    # placement, once the file is found.
    ledger_path: Optional[Path] = None
    if checkpoint is not None:
        ledger_path = _ledger_beside(checkpoint)
    elif measured is not None:
        ledger_path = _ledger_beside(measured.checkpoint.path)

    def refused(problem: str, kind: type = RegistrationRefused) -> RegistrationRefused:
        # A refusal from a reader that already says nothing was written (the
        # registry's own) says it once, naming the ledger.
        problem = _SAYS_NOTHING_WRITTEN.sub("", problem.rstrip())
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
    elif ledger_path is not None and not ledger_path.parent.is_dir():
        raise not_a_directory(ledger_path.parent)

    # **For a run's own report, every way of not getting back the published
    # bytes is one refusal** (`ReportNotAsPublished`): the run cannot vouch that
    # its report still exists, so it is never described as kept.
    lost = ReportNotAsPublished if measured is not None else RegistrationRefused
    lost_tail = (" The run's own report may no longer exist; re-measure to register "
                 "this measurement." if measured is not None else "")
    gone = (f"{report_path} does not exist: the run's own report is gone; re-measure to "
            "register this measurement." if measured is not None
            else f"{report_path} does not exist.")
    if not os.path.lexists(report_path):
        raise refused(gone, lost)

    # **Only from the report store** (D5, 2026-10-19): the version is found
    # through the registry's structure and resolved, or the report is refused.
    try:
        stored = locate_store_report(report_path, environ=environ)
    except NotTheDeploymentRegistry as exc:
        # A refusal of where the report is kept, not of the report: a run's own
        # report stays as it was published (not `ReportNotAsPublished`).
        raise refused(
            f"{report_path} is not registered: {exc}. A report is registered only from the "
            "deployment's registry (D5): another registry keeps its own report store "
            "and usage history, which the service never reads.",
            NotFromTheDeploymentRegistry,
        ) from None
    except NotInReportStore as exc:
        if measured is not None:
            # The run published its report into the store at this very path, so
            # a path that no longer resolves there (now a link to a file
            # elsewhere, say) was replaced after the publication: not the run's
            # report, and not one to offer for a later registration either.
            raise refused(
                f"{report_path} is no longer a report the store holds: {exc.why}. This run "
                "published its report there, so the file was replaced or moved after it was "
                "written." + lost_tail,
                lost,
            ) from None
        raise refused(str(exc)) from None
    except ValueError as exc:  # RegistryRefused, CohortVersionRefused
        raise refused(
            f"{report_path} is in a report store whose registry or version cannot be read: "
            f"{exc}"
        ) from exc

    try:
        report, source_digest = read_report(stored.path, expected)
    except _NotTheExpectedBytes as exc:
        if measured is None:
            # Compared on the bytes just read, before they are decoded: the file
            # no longer holds the report that was shown and chosen.
            raise refused(
                f"{report_path} is not the report that was shown: it now holds bytes "
                f"{exc.sha256[:12]}..., and {expected[:12]}... was shown. It was replaced or "
                "changed after it was shown; select it again."
            ) from None
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
    except RecursionError:  # nested deeper than a report is, from a hand edit
        raise refused(f"{report_path} nests too deeply to be read as a report.") from None

    # **The report names the version it is kept under, exactly**: the block the
    # measurement writes from a version (`cohort_version_block`), compared with
    # that block of the version the store holds it under. A report moved into
    # another version's store, or whose block was edited, is refused here.
    from src.evaluation.cohort_registry import (
        RUN_ID_PATTERN,
        SUPPLIED,
        cohort_version_block,
        version_words,
    )

    try:
        expected_block = cohort_version_block(stored.version)
    except ValueError as exc:
        raise refused(f"{report_path} cannot be checked against its version: {exc}") from exc
    block = report["manifest"]["cohort_version"]
    if block != expected_block:
        fields = sorted(f for f in set(block) | set(expected_block)
                        if block.get(f) != expected_block.get(f))
        # Named by its kind: a KG edge set or a generated split is never a
        # "cohort version" where a person reads it.
        kept_under = ("cohort version" if stored.version.kind == SUPPLIED else
                      version_words(stored.version.kind, stored.version.manifest.get("role")))
        raise refused(
            f"{report_path} is kept under {kept_under} {stored.label!r} "
            f"({expected_block['manifest_sha256'][:12]}...), but it records another version, "
            f"differing in: {', '.join(fields)}. A report is registered only under the "
            "version it measured."
        )
    # Released before anything else is read: the record holds what is kept of
    # it, and a placement the manifest it is given.
    placed_from = report["manifest"] if place is not None else None
    del report
    digest = record["checkpoint_digest"]
    # **Its run directory is named for the checkpoint it measured**
    # (`create_report_dir`): a report moved into another run's directory is not
    # where the measurement kept it.
    named_for = RUN_ID_PATTERN.fullmatch(stored.run_id).group(1)
    if named_for != digest[:12]:
        raise refused(
            f"{report_path} measured checkpoint {digest[:12]}..., but its run directory "
            f"{stored.run_id} is named for {named_for}...: it is not where the "
            "measurement kept that run's report."
        )

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
        if place is not None:
            # **Where the ledger goes, found from this read** (the Test page's
            # Register): the caller is given the manifest parsed from the bytes
            # just compared with the digest shown, never a read of its own, and
            # the file it names is checked below as a named file is.
            try:
                checkpoint = Path(place(placed_from))
            except ValueError as exc:
                raise refused(str(exc)) from exc
            del placed_from
            ledger_path = _ledger_beside(checkpoint)
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

    def lock_refused(exc: OSError) -> RegistrationRefused:
        return refused(
            f"{report_path} is not registered: the ledger's lock "
            f"{ledger_lock_path(ledger_path)} could not be opened or taken "
            f"({type(exc).__name__}: {exc})."
        )

    # **One registration at a time per ledger** (K8-6r.1): from the read to the
    # usage-history entry, under the ledger's lock. Only the lock's own failures
    # are mapped here; whatever the section raises keeps its own refusal.
    with ExitStack() as held_ledger:
        # Set when this account may not create or open the lock file (a
        # checkpoint directory it cannot write): then nothing can be written
        # here, but whether the ledger already holds this report is a read, and
        # readers take no lock (the replace is atomic). So that answer stays as
        # it was, and anything that would write is refused with this cause.
        cannot_lock: Optional[PermissionError] = None
        try:
            held_ledger.enter_context(ledger_lock(ledger_path))
        except FileLockBusy as exc:
            raise LedgerBusy(
                f"{report_path} is not registered: the ledger is busy (another "
                f"registration was writing {ledger_path} for the {exc.grace:g} s this one "
                "waited). The report stays where it is kept; register it again. Nothing "
                f"was written to {ledger_path} this time."
            ) from None
        except PermissionError as exc:
            cannot_lock = exc
        except OSError as exc:
            raise lock_refused(exc) from exc

        try:
            ledger, digest_at_read = read_ledger(ledger_path)
        except (OSError, ValueError) as exc:
            raise refused(str(exc)) from exc

        when = registered_at if registered_at is not None else (
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        # **By the verified run** (K8-6r.2): this run first, among the records
        # that name a run; then its own record. Nothing here reads another
        # report, and no description below decides anything.
        try:
            outcome, updated = append_record(ledger, record, run_id=stored.run_id,
                                             registered_at=when)
        except ValueError as exc:
            raise refused(f"{report_path} is not registered: {exc}") from exc

        held = held_record(updated, record, stored.run_id)
        differences = group_differences(updated, held)
        overlap = historical_overlap(updated, held)
        problem: Optional[str] = None
        if outcome == "already_registered":
            notes = _with_usage_clues(stored.version, overlap, stored.run_id, digest,
                                      source_digest)
            return RegistrationResult(outcome, held, ledger_path, differences, None, notes)
        if cannot_lock is not None:
            raise lock_refused(cannot_lock) from cannot_lock

        # **The ledger record and its history entry are one step against a
        # cancel** (M5-3 review EVAL-1, `interrupt_deferred`): a SIGINT arriving
        # between them is held until both are written, so a registration never
        # lacks its "registered" entry because of a cancel.
        with interrupt_deferred():
            try:
                write_ledger(ledger_path, updated, expected_digest=digest_at_read)
            except (OSError, ValueError) as exc:
                raise refused(f"{report_path} is not registered: {exc}") from exc

            # **In the version's usage history** (D4), after the ledger was
            # written and inside its lock (the order is ledger, then usage
            # history, never the other way): a failure here, the usage
            # history's lock busy included, does not undo the registration, so
            # it is reported with it.
            try:
                record_registration(
                    stored.version, run_id=stored.run_id, report_sha256=source_digest,
                    checkpoint_sha256=digest, outcome=outcome,
                    sequence=held["registration"]["sequence"],
                )
            except FileLockBusy as exc:
                problem = (f"{USAGE_HISTORY_NOT_WRITTEN}: it was busy (another "
                           f"writer held {exc.path} for the {exc.grace:g} s this one "
                           f"waited). The record in {ledger_path} stands")
            except (OSError, ValueError) as exc:
                problem = (f"{USAGE_HISTORY_NOT_WRITTEN}: {type(exc).__name__}: "
                           f"{exc}. The record in {ledger_path} stands")
    # A written registration's clue is read after the ledger's lock is
    # released; an already registered run's, above, under it (reading the usage
    # history takes no lock, so the order ledger, then usage history, holds
    # either way). It decides nothing.
    notes = _with_usage_clues(stored.version, overlap, stored.run_id, digest, source_digest)
    return RegistrationResult(outcome, held, ledger_path, differences, problem, notes)


def _recorded_checkpoint(report: Any) -> Tuple[Optional[Path], Optional[str]]:
    """The checkpoint a report's manifest says the run read
    (``manifest.checkpoint_path``), for `report_state` called without one, or
    `None` and why there is no ledger to read. Only an absolute path is
    taken: a relative one meant whatever the run's working directory made of
    it, which nothing here knows."""
    if report is None:
        return None, ("no checkpoint to find its ledger by: the report could not be read, "
                      "and no checkpoint was given")
    manifest = report.get("manifest") if isinstance(report, dict) else None
    recorded = manifest.get("checkpoint_path") if isinstance(manifest, dict) else None
    if not isinstance(recorded, str) or not recorded:
        return None, ("the report records no checkpoint path, and no checkpoint was given, so "
                      "which ledger to read is not known")
    if not Path(recorded).is_absolute():
        return None, (f"the report records its checkpoint as the relative path {recorded!r}, "
                      "which depended on where that run was started, and no checkpoint was "
                      "given, so which ledger to read is not known")
    return Path(recorded), None


def report_state(
    report_path: Path,
    *,
    checkpoint: Optional[Path] = None,
    place: Optional[Callable[[Mapping[str, Any]], Optional[Path]]] = None,
    expected_sha256: Optional[str] = None,
) -> ReportState:
    """Whether one report was kept, and whether it is registered, read from
    disk as the publication contract defines them -- never from a job's exit.

    - **Kept** (`KEPT_STATES`): the path is a report in a report store
      (`store_report_parts`, its shape; the version is not resolved, since a
      kept report stays kept when its version cannot be read), and its
      ``report.json``, the run's commit point, is there: holding the bytes
      ``expected_sha256`` names when that is given (the digest the run
      published, from its outcome file), or parsing as JSON otherwise. No file
      is ``not_kept``; other bytes than the run published are ``not_kept`` (the
      run's report was replaced); a file that cannot be read is
      ``cannot_be_checked``.
    - **Registered** (`REGISTERED_STATES`): the ledger beside ``checkpoint``'s
      resolved path -- the file the run was given, where `register_report`
      files it -- holds a record that names **this run** (the report's run
      directory) with **these bytes**, found by the writer's own lookup
      (`sidecar.find_run`), so the state agrees with what a registration would
      answer. The digest is the one read, or ``expected_sha256`` when the
      report could not be read, so a registered report that was later removed
      still reads as registered. A record naming this run with other bytes is
      ``not_registered``, saying the report was replaced after it was
      registered. A record written before each run had its own record that
      holds these bytes is never "registered": the state is as without it, and
      the reason adds that registering adds this run's own record. No ledger,
      or no such record, is ``not_registered``; a ledger that cannot be read,
      a lookup that cannot tell (this run's identity record damaged, the
      run-identity boundary unreadable), a record whose registration sequence
      cannot be read while none names this run (`sequence_problem`: its
      registration would be refused), a report outside a report store (it
      names no run), or no digest to look for, is ``cannot_be_checked``.
    - **Without** ``checkpoint`` (the server's list of a version's kept
      reports, which knows no checkpoint file; M5-1), the ledger is the one
      beside the checkpoint the report's manifest records, taken from the same
      read when that path is absolute (`_recorded_checkpoint`). A record there
      naming the report is ``registered``. **None there is**
      ``cannot_be_checked``, **never** ``not_registered``: `register_report`
      files a record in the ledger beside whichever copy of the weights it is
      given (`record_evaluation --checkpoint` takes any file with the same
      bytes), and only the copy the run read is known here, so a registration
      beside another copy cannot be ruled out. The reason names the ledger
      looked in and says so. A report that cannot be read, or records no
      absolute path, has no ledger to read: ``cannot_be_checked``, saying why.
      A checkpoint moved since the run is looked for where the run read it,
      and the reason names that ledger.
    - **With** ``place`` (the Test page's answer for the ledger its Register
      would write; round 3 of the review fixes, its verification's A-3): a
      callable given the report's manifest as parsed here, from the one read
      (compared with ``expected_sha256`` before it was decoded), which returns
      the checkpoint file whose ledger is read, as ``checkpoint`` would name
      it, found the caller's way, or None when it names none. It is called only
      for a kept report, so bytes other than the ones expected are never
      decoded to place a ledger; when it is not called, or names none, the
      ledger is looked for as without ``checkpoint``. As `register_report`'s
      ``place``: the caller never reads the report itself. Not with
      ``checkpoint`` (a caller's error, ``TypeError``).

    Reads the report once (`read_report`) and the ledger once (`read_ledger`);
    writes nothing. A link retargeted between the run and this call moves the
    ledger looked in, as it would move a later registration.
    """
    if checkpoint is not None and place is not None:
        raise TypeError("report_state takes at most one of checkpoint and place")
    ledger_path = _ledger_beside(checkpoint) if checkpoint is not None else None
    report_path = Path(report_path)
    digest: Optional[str] = None
    report: Any = None
    run_id: Optional[str] = None
    try:
        _root, _label, run_id, _resolved = store_report_parts(report_path)
    except NotInReportStore:
        kept, why_kept = "not_kept", (f"{report_path} is not in a registry's report store, so it "
                                      "is not a kept report")
        digest = expected_sha256
    else:
        try:
            report, digest = read_report(report_path, expected_sha256)
        except _NotTheExpectedBytes as exc:
            kept, why_kept = "not_kept", (
                f"{report_path} holds bytes {exc.sha256[:12]}..., not the "
                f"{expected_sha256[:12]}... the run published: the run's report was replaced "
                "or changed")
            digest = expected_sha256
        except FileNotFoundError:
            kept, why_kept = "not_kept", f"there is no {report_path}"
            digest = expected_sha256
        except OSError as exc:
            kept, why_kept = "cannot_be_checked", (
                f"{report_path} could not be read ({type(exc).__name__}: {exc})")
            digest = expected_sha256
        except ValueError as exc:
            kept, why_kept = "not_kept", f"{report_path} is not a report: {exc}"
            digest = expected_sha256
        else:
            kept, why_kept = "kept", f"{report_path} is in the report store"
            if place is not None:
                # **The ledger placed from this read**: the caller is given the
                # manifest parsed from the bytes just compared, never a read of
                # its own.
                manifest = report.get("manifest") if isinstance(report, dict) else None
                checkpoint = place(manifest if isinstance(manifest, dict) else {})
                if checkpoint is not None:
                    ledger_path = _ledger_beside(checkpoint)

    if digest is None:
        return ReportState(kept, why_kept, "cannot_be_checked",
                           "no report digest to look for: the report could not be read and "
                           "no published digest was given", None, None, ledger_path)
    if checkpoint is None:
        recorded, no_ledger = _recorded_checkpoint(report)
        if recorded is not None:
            try:
                ledger_path = _ledger_beside(recorded)
            except (OSError, RuntimeError) as exc:  # a link loop: RuntimeError before 3.13
                no_ledger = (f"the checkpoint the report records, {recorded}, cannot be "
                             f"resolved ({type(exc).__name__}), so its ledger cannot be found")
        if ledger_path is None:
            return ReportState(kept, why_kept, "cannot_be_checked", no_ledger, digest, None,
                               None)
    if run_id is None:
        return ReportState(kept, why_kept, "cannot_be_checked", (
            f"{report_path} is not in a report store, so it names no run, and a record is "
            "found by the run it names"), digest, None, ledger_path)
    try:
        ledger, _ = read_ledger(ledger_path)
    except (OSError, ValueError) as exc:
        return ReportState(kept, why_kept, "cannot_be_checked", str(exc), digest, None,
                           ledger_path)
    # **The writer's own lookup, first** (`find_run`): this run, among the
    # records that name a run; never a record found by its bytes alone.
    cohort, measured = _report_identity(report)
    found = find_run(ledger, run_id, cohort, measured)
    if found.problem is not None:
        return ReportState(kept, why_kept, "cannot_be_checked",
                           f"{ledger_path}: {found.problem}", digest, None, ledger_path)
    if found.record is not None:
        registration = found.record.get("registration")
        sequence = registration.get("sequence") if isinstance(registration, dict) else None
        if found.record.get("source_artifact_digest") == digest:
            return ReportState(kept, why_kept, "registered",
                               f"record {sequence} in {ledger_path} names this run and report",
                               digest, sequence if type(sequence) is int else None,
                               ledger_path)
        return ReportState(kept, why_kept, "not_registered",
                           f"{ledger_path}: {sealed_run_refusal(found, run_id, digest)}",
                           digest, None, ledger_path, replaced_after_registration=True)
    # **What would refuse the write** (`append_record`'s step 0) leaves the
    # state unknown too, so the state and the writer agree.
    uncertain = sequence_problem(ledger)
    if uncertain is not None:
        return ReportState(kept, why_kept, "cannot_be_checked", f"{ledger_path}: {uncertain}",
                           digest, None, ledger_path)
    looked = (f"{ledger_path} holds no record naming this run" if ledger_path.exists()
              else f"there is no {ledger_path}")
    # A note, not a state: a record merged before each run had its own record
    # may hold these bytes, and registering this run adds its own record.
    held = [note for note in historical_overlap(ledger, {"source_artifact_digest": digest})
            if note.possible_overlap]
    if held:
        shown = ", ".join(f"record {note.sequence}" if note.sequence is not None
                          else f"the record at index {note.index}" for note in held)
        looked += (f"; its bytes equal a report merged into {shown} before each run had its "
                   "own record; registering it adds its own record")
    overlap = tuple(note.sequence for note in held if note.sequence is not None)
    if checkpoint is None:
        return ReportState(kept, why_kept, "cannot_be_checked", (
            f"not registered beside the checkpoint this run read ({looked}); a ledger beside "
            "another copy of the same weights, where a registration naming that copy is filed, "
            "is not looked for without the checkpoint, so whether this report is registered "
            "cannot be checked here"), digest, None, ledger_path, overlap)
    return ReportState(kept, why_kept, "not_registered", looked, digest, None, ledger_path,
                       overlap)


def _report_identity(report: Any) -> Tuple[Optional[str], Optional[str]]:
    """The cohort version and checkpoint digests a report names, each None when
    it cannot be read (the report itself unread): `find_run` then does not
    filter by it, and the run id alone decides."""
    manifest = report.get("manifest") if isinstance(report, dict) else None
    if not isinstance(manifest, dict):
        return None, None
    block = manifest.get("cohort_version")
    digests = manifest.get("artifact_digests")
    cohort = block.get("manifest_sha256") if isinstance(block, dict) else None
    measured = digests.get("checkpoint") if isinstance(digests, dict) else None
    return ((cohort if isinstance(cohort, str) and _SHA256.fullmatch(cohort) else None),
            (measured if isinstance(measured, str) and _SHA256.fullmatch(measured) else None))


__all__ = ["KEPT_STATES", "LedgerBusy", "MeasuredRun",
           "NotFromTheDeploymentRegistry", "REGISTERED_STATES", "RegistrationRefused",
           "RegistrationResult", "ReportNotAsPublished", "ReportState",
           "USAGE_HISTORY_NOT_WRITTEN",
           "group_note", "historical_note", "register_report", "report_state"]
