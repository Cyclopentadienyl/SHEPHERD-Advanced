"""
Ordering checkpoints by a test score: D4's rules, as one pure function.
=======================================================================
`docs/working/PLAN_TEST_RESULTS.md` D4 and §4.9 question 1 (answered
2026-10-19). A list ordered by test score invites choosing by it, and a cohort
used to choose stops being an independent test (EC §4.1 condition 6). The
owner asked for ordering as an advanced option, so the aim is to make that use
comparable and visible, not to forbid it:

- **Only for a cohort version whose declared purpose allows it** -- its role,
  listed by the deployment in ``SHEPHERD_ORDERING_ROLES`` and passed in here
  (`ordering_offered`); ``institutional_acceptance`` never, even when listed,
  until the institution's formal evaluation rules exist. The role is the
  version's, never inferred from the source's name, and a record on the version
  naming another role stops the ordering rather than trusting the caller's.
- **Only comparable numbers together:** one metric over the records of one
  cohort version (its manifest digest) and one scoring semantics (D4's seven
  fields, `scoring_semantics_digest`). A checkpoint with no record on the
  version is "not tested on this cohort version"; one whose records there are
  all under other scoring semantics is "tested under different scoring
  semantics". Neither is interleaved by a number.
- **Mode C records only:** the metrics a list is ordered by are Mode C's; a
  scoring group of another mode (Mode D, the served ranking) is not ordered,
  its records shown and never ranked against each other (PLAN_B05 §6).
- **One number per checkpoint, by a fixed rule** (`chosen_record`): its first
  registered record from a deterministic regime, else its first registered
  record (lowest registration sequence in its ledger). A deterministic record
  takes precedence by a protocol choice declared in advance, not as a claim
  that it is more accurate. The checkpoint's other records in the group are
  returned beside it.
- **Contested means not ordered** (D5): when the record the rule picks is
  contested in the report store, or its state cannot be checked, the checkpoint
  is listed as not ordered, and the next record is never promoted -- contesting
  a score can take a checkpoint out of the order but cannot put a better number
  in.

**Pure.** It reads no file and no setting: the caller passes the records (each
checkpoint's from its one ledger), the roles the deployment allows and each
key's state from the report store (`StoreScan.state`). **Automatic selection
never comes here** (`select_auto_run_checkpoint` stays on validation): an
import-linter contract keeps the reload route and the selection module from
importing this module or the ledger's, and a test pins that a ledger does not
change what automatic selection chooses.

Module: src/evaluation/ordering.py
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from src.config.server_settings import NEVER_ORDERED_ROLES

#: `NEVER_ORDERED_ROLES` (imported above): roles for which ordering by test
#: score is never offered, whatever the deployment lists (§4.9 question 1:
#: until the institution's formal evaluation rules exist, ordering by the
#: acceptance cohort's score *is* selecting by it). The deployment settings'
#: one constant, never a copy (M5-3 review ARCH-2).

#: The metrics a Mode C record carries under metric schema 1
#: (`src/evaluation/measurement.py`, `_authoritative`), the ones a list can be
#: ordered by. Mean rank is better when lower; the rest when higher.
ORDERABLE_METRICS: Tuple[str, ...] = (
    "untruncated_mrr", "hits@1", "hits@5", "hits@10", "hits@20", "hits@50", "hits@100",
    "mean_rank",
)
LOWER_IS_BETTER: Tuple[str, ...] = ("mean_rank",)

#: How each row of an ordering is placed, in the order the rows are listed.
ROW_STATES: Tuple[str, ...] = (
    "ordered", "contested", "contested_unchecked", "unreadable", "two_ledgers",
    "different_scoring", "not_tested",
)

#: What a row that is not ordered says, by state.
ROW_LABELS: Dict[str, str] = {
    "contested": "contested — not ordered",
    "contested_unchecked": "contested status cannot be checked — not ordered",
    "unreadable": "its records cannot be read as registration writes them — not ordered",
    "two_ledgers": "records in two ledgers — not ordered",
    "different_scoring": "tested under different scoring semantics",
    "not_tested": "not tested on this cohort version",
}


class Candidate(NamedTuple):
    """One checkpoint of the list: its digest, every record its ledger holds for
    it (`records_for`), and which ledger that is (an opaque name, for example
    its resolved path; required, so two ledgers are never one by default). Two
    candidates with one digest and two ledgers are never merged: each ledger's
    sequence orders its own records only. One digest and one ledger named twice
    is one checkpoint listed twice, and must carry the same records."""

    checkpoint_sha256: str
    records: Tuple[Mapping[str, Any], ...]
    ledger: str


class Row(NamedTuple):
    """One checkpoint as the ordering places it.

    ``state`` is one of `ROW_STATES`; ``label`` is `ROW_LABELS`' text when it is
    not ordered. For an ordered row, ``position`` (ties share one) and ``value``
    are set. ``chosen`` is the record D4's rule picked, when there is one, and
    ``others`` the checkpoint's other records in the group, by sequence, shown
    beside it. ``regime_differs`` marks an ordered row whose numerical regime is
    not the one most ordered rows share (every ordered row when no single regime
    is the most common)."""

    checkpoint_sha256: str
    state: str
    label: Optional[str]
    position: Optional[int]
    value: Optional[float]
    chosen: Optional[Mapping[str, Any]]
    others: Tuple[Mapping[str, Any], ...]
    regime_differs: bool


class Ordering(NamedTuple):
    """The result: whether ordering is offered for the version's role and, when
    it is not, why; the group ordered (cohort digest, scoring digest, metric);
    and every candidate as a `Row`, ordered rows first."""

    offered: bool
    why: str
    cohort_digest: str
    scoring_semantics_digest: str
    metric: str
    rows: Tuple[Row, ...]


def ordering_offered(role: Any, ordering_roles: Collection[str]) -> Tuple[bool, str]:
    """Whether ordering by test score is offered for a cohort version of this
    role (`server_settings.offers_ordering`, the one rule), and why not when it
    is not."""
    from src.config import server_settings

    if server_settings.offers_ordering(role, ordering_roles):
        return True, ""
    if role in server_settings.NEVER_ORDERED_ROLES:
        return False, (f"ordering by test score is never offered for role {role!r}: until the "
                       "institution's formal evaluation rules exist, ordering by its score is "
                       "selecting by it")
    return False, (f"ordering by test score is not offered for role {role!r}: the "
                   "deployment does not list it in SHEPHERD_ORDERING_ROLES")


def _sequence(record: Mapping[str, Any]) -> Optional[int]:
    registration = record.get("registration")
    sequence = registration.get("sequence") if isinstance(registration, Mapping) else None
    return sequence if type(sequence) is int and sequence >= 1 else None


def _deterministic(record: Mapping[str, Any]) -> Optional[bool]:
    regime = record.get("regime")
    flag = regime.get("deterministic_algorithms") if isinstance(regime, Mapping) else None
    return flag if type(flag) is bool else None


def chosen_record(records: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """D4's fixed rule over one checkpoint's records in one group: the first
    registered record from a deterministic regime if there is one, otherwise the
    first registered record. `None` when there is no record, or when any of them
    lacks a readable sequence or determinism flag: the rule is not applied
    around a record it cannot read."""
    if not records:
        return None
    if any(_sequence(r) is None or _deterministic(r) is None for r in records):
        return None
    deterministic = [r for r in records if _deterministic(r)]
    return min(deterministic or records, key=_sequence)


def _mode(record: Mapping[str, Any]) -> Optional[str]:
    scoring = record.get("scoring")
    return scoring.get("mode") if isinstance(scoring, Mapping) else None


def _readable_mode(mode: Any) -> bool:
    return isinstance(mode, str) and bool(mode.strip())


def _value(record: Mapping[str, Any], metric: str) -> Optional[float]:
    metrics = record.get("metrics")
    value = metrics.get(metric) if isinstance(metrics, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _regime_text(record: Mapping[str, Any]) -> str:
    return json.dumps(record.get("regime"), sort_keys=True, default=str)


def order_by_test_score(
    candidates: Sequence[Candidate],
    *,
    cohort_digest: str,
    scoring_semantics_digest: str,
    metric: str,
    role: str,
    ordering_roles: Collection[str],
    contested_state: Callable[[Tuple[Any, ...]], str],
) -> Ordering:
    """Order checkpoints by one metric of their records on one cohort version
    under one scoring semantics, by D4's rules (see the module docstring).

    ``role`` is the version's role and ``ordering_roles`` the roles the
    deployment lists (``SHEPHERD_ORDERING_ROLES``, read by the caller). **The
    records' own role decides too:** when any record on ``cohort_digest`` names
    another role than ``role`` (a caller's stale role, say), nothing is ordered,
    so ``institutional_acceptance`` is never ordered on a caller's word;
    ``contested_state`` gives a record key's state in the version's report store
    (`StoreScan.state`: ``uncontested``, ``contested`` or
    ``cannot_be_checked``; anything else is taken as cannot be checked).

    Ordered rows come first, best first (higher is better except for
    `LOWER_IS_BETTER`), tied values sharing a position and listed by checkpoint
    digest; then the rows that are not ordered, by `ROW_STATES` and digest.
    When ordering is not offered for the role, no row is returned.

    Raises:
        ValueError: ``metric`` is not one of `ORDERABLE_METRICS`, or two
            candidates name one digest and one ledger with different records
            (a caller's error, not a property of the records).
    """
    from src.evaluation.sidecar import record_key

    if metric not in ORDERABLE_METRICS:
        raise ValueError(f"{metric!r} is not a metric a list is ordered by "
                         f"({', '.join(ORDERABLE_METRICS)})")
    offered, why = ordering_offered(role, ordering_roles)
    if not offered:
        return Ordering(False, why, cohort_digest, scoring_semantics_digest, metric, ())

    # One entry per checkpoint digest, keeping each distinct ledger apart; one
    # ledger named twice for a digest is the same file listed twice, never a
    # second set of records written over the first.
    by_digest: Dict[str, Dict[Any, Tuple[Mapping[str, Any], ...]]] = {}
    for candidate in candidates:
        records = tuple(r for r in candidate.records if isinstance(r, Mapping)
                        and r.get("checkpoint_digest") == candidate.checkpoint_sha256)
        ledgers = by_digest.setdefault(candidate.checkpoint_sha256, {})
        if candidate.ledger in ledgers and ledgers[candidate.ledger] != records:
            raise ValueError(
                f"two candidates name checkpoint {candidate.checkpoint_sha256[:12]}... and "
                f"ledger {candidate.ledger!r} with different records; name each ledger by "
                "its own path, so records of two ledgers are never taken for one's"
            )
        ledgers[candidate.ledger] = records

    # **The records' role, not only the caller's**: a version has one role, and
    # a record on it naming another means the role passed is not the version's.
    roles = {r.get("cohort_role") for ledgers in by_digest.values()
             for records in ledgers.values() for r in records
             if r.get("cohort_digest") == cohort_digest}
    if roles - {role}:
        named = ", ".join(sorted(repr(r) for r in roles - {role}))
        return Ordering(False, (
            f"ordering by test score is not offered: records on cohort version "
            f"{cohort_digest[:12]}... name role {named}, not the role {role!r} it was asked "
            "for; a version has one role, and ordering follows the records' own"),
            cohort_digest, scoring_semantics_digest, metric, ())

    # **Mode C records only** (PLAN_B05 §6): the metrics a list is ordered by
    # are Mode C's, and a record of another mode (Mode D, the served ranking) is
    # shown, never ordered. One scoring group is one mode (`mode` is among its
    # fields), so a group whose records name another mode is not ordered at all.
    # A record whose mode cannot be read never stops the whole group: it makes
    # only its own checkpoint unreadable, below.
    modes = {_mode(r) for ledgers in by_digest.values() for records in ledgers.values()
             for r in records if r.get("cohort_digest") == cohort_digest
             and r.get("scoring_semantics_digest") == scoring_semantics_digest}
    modes = {m for m in modes if _readable_mode(m)}
    if modes - {"C"}:
        named = ", ".join(sorted(modes - {"C"}))
        return Ordering(False, (
            f"ordering by test score is not offered for this scoring group: its records are "
            f"mode {named}, not Mode C; only Mode C records are ordered, and the others are "
            "shown, never ordered"),
            cohort_digest, scoring_semantics_digest, metric, ())

    unordered: List[Row] = []
    ordered: List[Tuple[float, str, Mapping[str, Any], Tuple[Mapping[str, Any], ...]]] = []

    def row(digest: str, state: str, chosen=None, others=()) -> Row:
        return Row(digest, state, ROW_LABELS[state], None, None, chosen, tuple(others), False)

    for digest, ledgers in by_digest.items():
        on_version = {ledger: tuple(r for r in records if r.get("cohort_digest") == cohort_digest)
                      for ledger, records in ledgers.items()}
        in_group = {ledger: tuple(r for r in records
                                  if r.get("scoring_semantics_digest") == scoring_semantics_digest)
                    for ledger, records in on_version.items()}
        holding = [ledger for ledger, records in in_group.items() if records]
        if len(holding) > 1:
            unordered.append(row(digest, "two_ledgers"))
            continue
        if not holding:
            state = ("different_scoring" if any(on_version.values()) else "not_tested")
            unordered.append(row(digest, state))
            continue
        group = in_group[holding[0]]
        # A record whose mode cannot be read is not taken for Mode C's: the rule
        # is not applied around it, as for an unreadable sequence (`chosen_record`).
        if not all(_readable_mode(_mode(r)) for r in group):
            unordered.append(row(digest, "unreadable"))
            continue
        chosen = chosen_record(group)
        if chosen is None:
            unordered.append(row(digest, "unreadable"))
            continue
        others = tuple(sorted((r for r in group if r is not chosen), key=_sequence))
        try:
            state = contested_state(record_key(chosen))
        except ValueError:
            unordered.append(row(digest, "unreadable", chosen, others))
            continue
        if state == "contested":
            unordered.append(row(digest, "contested", chosen, others))
            continue
        if state != "uncontested":
            unordered.append(row(digest, "contested_unchecked", chosen, others))
            continue
        value = _value(chosen, metric)
        if value is None:
            unordered.append(row(digest, "unreadable", chosen, others))
            continue
        ordered.append((value, digest, chosen, others))

    lower = metric in LOWER_IS_BETTER
    ordered.sort(key=lambda item: (item[0] if lower else -item[0], item[1]))
    regimes = Counter(_regime_text(chosen) for _, _, chosen, _ in ordered)
    most = max(regimes.values(), default=0)
    modal = [text for text, count in regimes.items() if count == most]
    rows: List[Row] = []
    position = 0
    previous: Optional[float] = None
    for index, (value, digest, chosen, others) in enumerate(ordered, start=1):
        if value != previous:
            position, previous = index, value
        differs = len(modal) > 1 or _regime_text(chosen) != modal[0]
        rows.append(Row(digest, "ordered", None, position, value, chosen, others,
                        differs if len(ordered) > 1 else False))
    unordered.sort(key=lambda r: (ROW_STATES.index(r.state), r.checkpoint_sha256))
    return Ordering(True, "", cohort_digest, scoring_semantics_digest, metric,
                    tuple(rows + unordered))


__all__ = [
    "Candidate",
    "LOWER_IS_BETTER",
    "NEVER_ORDERED_ROLES",
    "ORDERABLE_METRICS",
    "Ordering",
    "ROW_LABELS",
    "ROW_STATES",
    "Row",
    "chosen_record",
    "order_by_test_score",
    "ordering_offered",
]
