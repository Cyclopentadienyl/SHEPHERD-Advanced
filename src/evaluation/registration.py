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
whole registry -- the same bytes, its own disagreement block and usage history
-- is refused before anything in it is read (`NotFromTheDeploymentRegistry`).
Unset, every registration is refused saying to set it. A registry moved as a
whole still works once ``SHEPHERD_REGISTRY`` names its new place.

**While runs under the report's key disagree, nothing under the key is
registered** (D5), checked or unchecked, the same report again included: the
store's reports are compared with the ledger's own comparison
(`StoreScan.status`), and a key whose state cannot be checked is refused too,
never read as uncontested. The reports stay in the store for the investigation;
a note kept there records what was found and does not lift the block.

**Each file is parsed and identified from one read** (BACKLOG item 18). The report
is read once, and the record is built from those bytes and names them by their
digest; the store's other reports are read once each for the comparison, never
this one again; the ledger is read once, and the write's stale-write guard
compares the file with the digest of what was parsed. A report replaced while it
was being registered therefore cannot be recorded as one file's numbers under
the other's digest, and another writer's append between this read and this
write is refused rather than erased.

**Still single-writer.** The guard compares the file by path just before the
replacement, and the window between that comparison and the replacement stays
open (`write_ledger`). This is not compare-and-swap and makes no claim of safety
for concurrent writers; nor is the store's comparison atomic with a run
publishing a new report into the version at that moment.

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
appended there by this function, so a registration from the command line counts
as one from the Test tab. If that append fails after the ledger was written, the
result says so (`usage_history_problem`) rather than claiming either.

**Every expected failure is one refusal** (`RegistrationRefused`) whose message
names the file and says nothing was written: a missing or damaged report, a file
that is not a report or not in a report store, a report in a registry other
than the deployment's (`NotFromTheDeploymentRegistry`), a report that is no
longer the one shown, a generated split's report, a
report that does not name the version it is kept under, runs that disagree
(`RunsDisagree`) or a store that cannot be checked
(`DisagreementNotChecked`), a ledger this revision does not read, a
contradiction, a stale write, a checkpoint file or directory that does not hold
the weights the report measured, and, for an automatic registration, a report
file that is no longer the one the run published (`ReportNotAsPublished`). The
entry points print it and exit; neither shows a traceback for these.

**Kept? Registered?** `report_state` answers both for one report from disk, the
way the publication contract defines them, for the job runner and the Test tab
(M5's acceptance: never inferred from a job's exit alone).

Module: src/evaluation/registration.py
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, NamedTuple, Optional, Tuple

from src.evaluation.report_store import (
    NotInReportStore,
    NotTheDeploymentRegistry,
    _NotTheExpectedBytes,
    locate_store_report,
    read_report,
    record_registration,
    scan_report_store,
    store_report_parts,
)
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
from src.utils.interrupts import interrupt_deferred

#: Whether a report was kept, as the publication contract defines it: its
#: ``report.json`` is in the report store, holding the bytes the run published
#: when that digest is known (`report_state`).
KEPT_STATES: Tuple[str, ...] = ("kept", "not_kept", "cannot_be_checked")

#: Whether a report is registered: a record in the ledger beside its checkpoint
#: names it, as its source or a corroborating report (`report_state`).
REGISTERED_STATES: Tuple[str, ...] = ("registered", "not_registered", "cannot_be_checked")

_SAYS_NOTHING_WRITTEN = re.compile(r"\s*Nothing was written\.?\s*$")

#: A report digest as the lists show it: 64 lowercase hexadecimal digits.
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RegistrationRefused(ValueError):
    """A report was not registered, and nothing was written."""


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


class RunsDisagree(RegistrationRefused):
    """Reports in the store under this report's key disagree (D5), so nothing
    under the key is registered while they do; `status` is the key's state.
    Registering the report later is refused the same way until that changes."""

    def __init__(self, message: str, status: Any) -> None:
        super().__init__(message)
        self.status = status


class DisagreementNotChecked(RegistrationRefused):
    """Whether reports in the store under this report's key disagree cannot be
    checked (D5: never read as uncontested); `status` says why."""

    def __init__(self, message: str, status: Any) -> None:
        super().__init__(message)
        self.status = status


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

    ``usage_history_problem`` is `None` unless the ledger was written and the
    entry in the version's usage history could not be appended; it then says
    why, and the registration stands.
    """

    outcome: str
    record: Dict[str, Any]
    ledger_path: Path
    excluded_differences: Tuple[str, ...] = ()
    usage_history_problem: Optional[str] = None


class ReportState(NamedTuple):
    """Whether one report was kept and whether it is registered, each one of
    `KEPT_STATES` and `REGISTERED_STATES` with the reason in words.

    ``report_sha256`` is the digest of the bytes found (or the one expected,
    when the report could not be read); ``sequence`` the ledger record's when
    registered; ``ledger_path`` the ledger that was (or would be) read, `None`
    when no checkpoint was given and the report names none to find it by.
    """

    kept: str
    why_kept: str
    registered: str
    why_registered: str
    report_sha256: Optional[str]
    sequence: Optional[int]
    ledger_path: Optional[Path]


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
    checkpoint its run directory is named for; and **no report in the store
    under its key may disagree with another** (D5), nor may that be uncheckable
    (`RunsDisagree`, `DisagreementNotChecked`). **The registry must be the
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
      its version, its run directory and D5's block have been checked, so a
      refused report lists nothing. The caller never reads the report itself.

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

    Returns the outcome — ``registered``, ``already_registered`` (nothing written)
    or ``corroborated`` — the record as the ledger now holds it, for a
    corroboration the fields not compared on which this report differs, and,
    when the ledger was written and the version's usage history could not be
    appended, why (`RegistrationResult.usage_history_problem`). Every
    expected failure raises ``RegistrationRefused``; a call naming none, or more
    than one, of ``checkpoint``, ``checkpoint_dir``, ``measured`` and ``place``
    is a caller's error (``TypeError``).
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

    def refused(problem: str, kind: type = RegistrationRefused,
                **fields: Any) -> RegistrationRefused:
        # A refusal from a reader that already says nothing was written (the
        # registry's own) says it once, naming the ledger.
        problem = _SAYS_NOTHING_WRITTEN.sub("", problem.rstrip())
        if not problem.endswith("."):
            problem += "."
        where = f" to {ledger_path}" if ledger_path is not None else ""
        return kind(f"{problem} Nothing was written{where}.", **fields)

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
            "deployment's registry (D5): another registry keeps its own report store, "
            "disagreement block and usage history, which the service never reads.",
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
    from src.evaluation.cohort_registry import RUN_ID_PATTERN, cohort_version_block

    try:
        expected_block = cohort_version_block(stored.version)
    except ValueError as exc:
        raise refused(f"{report_path} cannot be checked against its version: {exc}") from exc
    block = report["manifest"]["cohort_version"]
    if block != expected_block:
        fields = sorted(f for f in set(block) | set(expected_block)
                        if block.get(f) != expected_block.get(f))
        raise refused(
            f"{report_path} is kept under cohort version {stored.label!r} "
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

    # **D5: no registration under a key whose runs disagree**, checked or
    # unchecked, and none where that cannot be checked. This report stands for
    # its own run directory; every other report the store keeps is read once.
    key = record_key(record)
    block = key_refusal(stored.version, key, known=(stored.run_id, record))
    if block is not None:
        raise refused(str(block), type(block), status=block.status)

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

    held = next(r for r in updated["records"] if record_key(r) == key)
    differences: Tuple[str, ...] = ()
    if outcome == "corroborated":
        differences = tuple(field for field in CORROBORATION_KEEPS_FIRST
                            if held.get(field) != record[field])
    if outcome == "already_registered":
        return RegistrationResult(outcome, held, ledger_path)

    # **The ledger record and its history entry are one step against a cancel**
    # (M5-3 review EVAL-1, `interrupt_deferred`): a SIGINT arriving between them
    # is held until both are written, so a registration never lacks its
    # "registered" entry because of a cancel.
    with interrupt_deferred():
        try:
            write_ledger(ledger_path, updated, expected_digest=digest_at_read)
        except (OSError, ValueError) as exc:
            raise refused(f"{report_path} is not registered: {exc}") from exc

        # **In the version's usage history** (D4), after the ledger was written:
        # a failure here does not undo the registration, so it is reported with
        # it.
        problem: Optional[str] = None
        try:
            record_registration(
                stored.version, run_id=stored.run_id, report_sha256=source_digest,
                checkpoint_sha256=digest, outcome=outcome,
                sequence=held["registration"]["sequence"],
            )
        except (OSError, ValueError) as exc:
            problem = (f"the registration was written to {ledger_path}, but the cohort "
                       f"version's usage history could not be appended "
                       f"({type(exc).__name__}: {exc})")
    return RegistrationResult(outcome, held, ledger_path, differences, problem)


def key_refusal(
    version: Any,
    key: Tuple[Any, ...],
    *,
    known: Optional[Tuple[str, Dict[str, Any]]] = None,
) -> Optional[RegistrationRefused]:
    """D5's block on a key of a version's report store, as the refusal a
    registration under it gets -- `RunsDisagree` while runs under it disagree,
    `DisagreementNotChecked` while that cannot be checked -- or `None` when
    nothing blocks it.

    **Returned, not raised**, so the caller says it in its own place:
    `register_report` raises it (adding that nothing was written), and a
    measurement without ``--record`` says why it offers no register-later
    command rather than printing one that would be refused. ``known`` is a
    report the caller already read (`scan_report_store`). ``key`` is
    `record_key`'s tuple.
    """
    status = scan_report_store(version, known=known).status(key)
    if status.state == "uncontested":
        return None
    checkpoint, _role, cohort, semantics = key
    if status.state == "contested":
        return RunsDisagree(
            f"runs disagree — investigate. Under this report's key (checkpoint "
            f"{checkpoint[:12]}... on cohort version {version.manifest.get('label')} "
            f"({cohort[:12]}...), measurement semantics {semantics[:12]}...) the report "
            f"store holds reports that disagree on: {', '.join(status.differing)} (runs "
            f"{', '.join(status.disagreeing_runs)}). While they do, no report under this key "
            "is registered, checked or unchecked: registering one would be choosing between "
            "results. Every report stays in the store; compare their environment facts "
            "first. A deterministic re-run is another key.",
            status,
        )
    problems = "; ".join(status.problems).rstrip()
    return DisagreementNotChecked(
        f"whether runs under this report's key disagree cannot be checked, so no report "
        f"under the key is registered: {problems}{'' if problems.endswith('.') else '.'}",
        status,
    )


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
      files it -- holds a record naming this report's digest, as its source or
      among its corroborating reports. The digest is the one read, or
      ``expected_sha256`` when the report could not be read, so a registered
      report that was later removed still reads as registered. No ledger, or no
      such record, is ``not_registered``; a ledger that cannot be read, or no
      digest to look for, is ``cannot_be_checked``.
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
    try:
        store_report_parts(report_path)
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
    try:
        ledger, _ = read_ledger(ledger_path)
    except (OSError, ValueError) as exc:
        return ReportState(kept, why_kept, "cannot_be_checked", str(exc), digest, None,
                           ledger_path)
    for record in ledger["records"]:
        if not isinstance(record, dict):
            continue
        corroborating = record.get("corroborating_reports")
        if (record.get("source_artifact_digest") == digest
                or (isinstance(corroborating, list) and digest in corroborating)):
            registration = record.get("registration")
            sequence = registration.get("sequence") if isinstance(registration, dict) else None
            return ReportState(kept, why_kept, "registered",
                               f"record {sequence} in {ledger_path} names this report",
                               digest, sequence if type(sequence) is int else None,
                               ledger_path)
    looked = (f"{ledger_path} holds no record naming this report" if ledger_path.exists()
              else f"there is no {ledger_path}")
    if checkpoint is None:
        return ReportState(kept, why_kept, "cannot_be_checked", (
            f"not registered beside the checkpoint this run read ({looked}); a ledger beside "
            "another copy of the same weights, where a registration naming that copy is filed, "
            "is not looked for without the checkpoint, so whether this report is registered "
            "cannot be checked here"), digest, None, ledger_path)
    return ReportState(kept, why_kept, "not_registered", looked, digest, None, ledger_path)


__all__ = ["DisagreementNotChecked", "KEPT_STATES", "MeasuredRun",
           "NotFromTheDeploymentRegistry", "REGISTERED_STATES", "RegistrationRefused",
           "RegistrationResult", "ReportNotAsPublished", "ReportState", "RunsDisagree",
           "corroboration_note", "key_refusal", "register_report", "report_state"]
