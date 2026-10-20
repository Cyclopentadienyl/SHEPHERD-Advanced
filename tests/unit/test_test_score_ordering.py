"""D4: ordering checkpoints by a test score, and automatic selection kept apart.

`src/evaluation/ordering.py` (`PLAN_TEST_RESULTS.md` D4; §4.9 question 1,
answered 2026-10-19). What is pinned here, on records `build_record` makes from
reports `MeasurementManifest` builds:

- **offered by role only**: for a role the deployment lists, never for
  ``institutional_acceptance`` even when listed, and with no rows otherwise;
  records on the version naming another role than the caller's stop it;
- **comparable numbers only**: one cohort version, one scoring semantics; the
  others are labelled "not tested on this cohort version" or "tested under
  different scoring semantics", never interleaved;
- **D4's fixed rule**: the first registered deterministic record, else the
  first registered; the rest returned beside it; a record the rule cannot read
  is not picked around;
- **contested means not ordered**, the next record never promoted, and a state
  that cannot be checked is not ordered either;
- direction (mean rank lower is better), ties sharing a position, two ledgers
  for one digest (and one ledger named twice with other records refused), the
  regime mark;
- **automatic selection never reads test results**: a ledger beside the
  checkpoints that favours another file does not change what
  `select_auto_run_checkpoint` or `select_run_checkpoint` choose (the
  import-linter contract `selection-ignores-test-results` keeps the modules
  apart).

Module: tests/unit/test_test_score_ordering.py
"""
from __future__ import annotations

import hashlib
import json

import pytest

from src.evaluation.ordering import (
    NEVER_ORDERED_ROLES,
    ROW_LABELS,
    Candidate,
    chosen_record,
    order_by_test_score,
    ordering_offered,
)
from src.evaluation.sidecar import build_record, record_key, scoring_semantics_digest
from tests.unit.test_evaluation_sidecar import (
    cohort_version,
    registered_manifest,
    registered_report,
)

ROLE = "mygene2"
COHORT = "a" * 64
SCORING = scoring_semantics_digest(registered_manifest())


def _record(checkpoint: str, *, sequence: int = 1, mrr: float = 0.5, mean_rank: float = 3.0,
            deterministic: bool = True, cohort: str = COHORT, **overrides):
    """A ledger record of `checkpoint`, as `append_record` leaves it."""
    report = registered_report(
        metrics={"untruncated_mrr": mrr, "mean_rank": mean_rank, "hits@1": 0.25,
                 "hits@10": 0.75},
        cohort_version=cohort_version(manifest_sha256=cohort),
        deterministic_algorithms=deterministic,
        **overrides,
    )
    report["manifest"]["artifact_digests"]["checkpoint"] = checkpoint
    digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    return {**build_record(report, digest), "corroborating_reports": [],
            "registration": {"sequence": sequence, "registered_at": "2026-10-20T09:00:00Z"}}


def _order(candidates, *, metric="untruncated_mrr", role=ROLE, roles=(ROLE,), states=None):
    states = states or {}
    return order_by_test_score(
        candidates, cohort_digest=COHORT, scoring_semantics_digest=SCORING, metric=metric,
        role=role, ordering_roles=roles,
        contested_state=lambda key: states.get(key, "uncontested"),
    )


def _c(digest, *records, ledger="ledger-1"):
    return Candidate(digest, tuple(records), ledger)


A, B, C, D = ("a1" * 32, "b2" * 32, "c3" * 32, "d4" * 32)


# ---------------------------------------------------------------------------
# Offered by role
# ---------------------------------------------------------------------------
def test_ordering_is_offered_only_for_a_listed_role():
    assert ordering_offered("mygene2", ("mygene2",)) == (True, "")
    offered, why = ordering_offered("mygene2", ())
    assert not offered and "SHEPHERD_ORDERING_ROLES" in why
    ordering = _order([_c(A, _record(A))], roles=())
    assert (ordering.offered, ordering.rows) == (False, ())


def test_institutional_acceptance_is_never_ordered_even_when_listed():
    assert NEVER_ORDERED_ROLES == ("institutional_acceptance",)
    ordering = _order([_c(A, _record(A))], role="institutional_acceptance",
                      roles=("institutional_acceptance", ROLE))
    assert not ordering.offered and ordering.rows == ()
    assert "never offered" in ordering.why


@pytest.mark.parametrize("records_role", ["institutional_acceptance", "another_listed_role"])
def test_the_records_own_role_decides_not_only_the_callers(records_role):
    """The reviewer's P3 (2026-10-20): a caller passing a listed role (stale UI
    state, say) for a version whose records carry another role orders nothing,
    so ``institutional_acceptance`` is never ordered on a caller's word."""
    acceptance = {**_record(A), "cohort_role": records_role}

    ordering = _order([_c(A, acceptance), _c(B, _record(B))],
                      roles=(ROLE, "another_listed_role"))

    assert not ordering.offered and ordering.rows == ()
    assert repr(records_role) in ordering.why and "records' own" in ordering.why
    # Records of another version say nothing about this one's role.
    other_version = {**_record(C, cohort="9" * 64), "cohort_role": "institutional_acceptance"}
    assert _order([_c(B, _record(B)), _c(C, other_version)]).offered


def test_an_unknown_metric_is_a_callers_error():
    with pytest.raises(ValueError, match="not a metric"):
        _order([], metric="val_mrr")


# ---------------------------------------------------------------------------
# Comparable numbers only, and the order
# ---------------------------------------------------------------------------
def test_checkpoints_are_ordered_best_first_with_the_rest_labelled_after():
    ordering = _order([
        _c(A, _record(A, mrr=0.4)),
        _c(B, _record(B, mrr=0.7)),
        _c(C, _record(C, cohort="9" * 64)),                       # another version only
        _c(D, _record(D, score_semantics="another scorer")),      # other scoring semantics
        _c("e5" * 32),                                            # no record at all
    ])

    assert ordering.offered
    assert [(r.checkpoint_sha256, r.state, r.position, r.value) for r in ordering.rows] == [
        (B, "ordered", 1, 0.7), (A, "ordered", 2, 0.4),
        (D, "different_scoring", None, None),
        (C, "not_tested", None, None), ("e5" * 32, "not_tested", None, None),
    ]
    assert ordering.rows[2].label == "tested under different scoring semantics"
    assert ordering.rows[3].label == "not tested on this cohort version"


def test_mean_rank_is_ordered_lowest_first_and_ties_share_a_position():
    ordering = _order([_c(A, _record(A, mean_rank=4.0)), _c(B, _record(B, mean_rank=2.0)),
                       _c(C, _record(C, mean_rank=2.0))], metric="mean_rank")

    assert [(r.checkpoint_sha256, r.position) for r in ordering.rows] == [
        (B, 1), (C, 1), (A, 3)]


# ---------------------------------------------------------------------------
# D4's fixed rule
# ---------------------------------------------------------------------------
def test_the_first_registered_deterministic_record_is_chosen_over_an_earlier_one():
    early = _record(A, sequence=1, mrr=0.9, deterministic=False)
    det_late = _record(A, sequence=3, mrr=0.5, batch_size=16)
    det_first = _record(A, sequence=2, mrr=0.4, batch_size=8)

    assert chosen_record([early, det_late, det_first]) is det_first
    (row,) = _order([_c(A, early, det_late, det_first)]).rows
    assert row.chosen is det_first and row.value == 0.4
    assert [r["registration"]["sequence"] for r in row.others] == [1, 3]


def test_without_a_deterministic_record_the_first_registered_is_chosen():
    first = _record(A, sequence=1, mrr=0.3, deterministic=False)
    later = _record(A, sequence=2, mrr=0.8, deterministic=False, batch_size=8)

    assert chosen_record([later, first]) is first


def test_a_record_the_rule_cannot_read_is_not_picked_around():
    good = _record(A, sequence=2)
    broken = {**_record(A, sequence=1, batch_size=8), "registration": {}}

    assert chosen_record([good, broken]) is None
    (row,) = _order([_c(A, good, broken)]).rows
    assert row.state == "unreadable" and row.label == ROW_LABELS["unreadable"]


def test_a_chosen_record_without_the_metric_is_not_ordered():
    record = _record(A)
    record["metrics"] = {"hits@1": 0.25}
    (row,) = _order([_c(A, record)]).rows
    assert row.state == "unreadable"


# ---------------------------------------------------------------------------
# Contested means not ordered (D5)
# ---------------------------------------------------------------------------
def test_a_contested_choice_is_not_ordered_and_the_next_record_is_not_promoted():
    chosen = _record(A, sequence=1, mrr=0.2)
    better = _record(A, sequence=2, mrr=0.99, batch_size=8)
    states = {record_key(chosen): "contested"}

    ordering = _order([_c(A, chosen, better), _c(B, _record(B, mrr=0.5))], states=states)

    assert [(r.checkpoint_sha256, r.state) for r in ordering.rows] == [
        (B, "ordered"), (A, "contested")]
    assert ordering.rows[1].label == "contested — not ordered"
    assert ordering.rows[1].chosen is chosen and ordering.rows[1].value is None


@pytest.mark.parametrize("state", ["cannot_be_checked", "something else"])
def test_a_choice_whose_state_cannot_be_checked_is_not_ordered(state):
    record = _record(A)
    (row,) = _order([_c(A, record)], states={record_key(record): state}).rows
    assert row.state == "contested_unchecked"
    assert row.label == "contested status cannot be checked — not ordered"


# ---------------------------------------------------------------------------
# Ledgers and regimes
# ---------------------------------------------------------------------------
def test_one_digest_with_records_in_two_ledgers_is_not_ordered():
    ordering = _order([_c(A, _record(A), ledger="run-1"), _c(A, _record(A, sequence=4),
                                                             ledger="run-2")])
    (row,) = ordering.rows
    assert row.state == "two_ledgers"


def test_one_file_listed_twice_from_one_ledger_is_one_checkpoint():
    record = _record(A)
    (row,) = _order([_c(A, record), _c(A, record)]).rows
    assert row.state == "ordered"


def test_one_ledger_named_twice_with_other_records_is_a_callers_error_not_a_merge():
    """The reviewer's P3 (2026-10-20): two ledgers for one digest under one
    name -- the same weights in two run directories, ledgers not named -- are
    refused, never one written over the other; and a candidate names its
    ledger, so none are one ledger by default."""
    with pytest.raises(ValueError, match="different records"):
        _order([_c(A, _record(A, mrr=0.9)), _c(A, _record(A, mrr=0.1))])
    with pytest.raises(TypeError):
        Candidate(A, (_record(A),))


def test_a_row_whose_regime_differs_from_the_rest_is_marked():
    ordering = _order([_c(A, _record(A, mrr=0.6)), _c(B, _record(B, mrr=0.5)),
                       _c(C, _record(C, mrr=0.4, device="cpu", cuda_executed=False))])

    assert [(r.checkpoint_sha256, r.regime_differs) for r in ordering.rows] == [
        (A, False), (B, False), (C, True)]
    (alone,) = _order([_c(A, _record(A))]).rows
    assert alone.regime_differs is False


# ---------------------------------------------------------------------------
# Automatic selection never reads test results
# ---------------------------------------------------------------------------
def test_a_ledger_does_not_change_what_automatic_selection_chooses(tmp_path):
    """Two checkpoints of one run, the validation metric favouring `best.pt`; a
    ledger beside them whose test result favours `other.pt`. Both selection
    functions choose `best.pt`, with or without the ledger."""
    import os

    from src.evaluation.sidecar import LEDGER_FILENAME, empty_ledger, write_ledger
    from src.utils.checkpoint_io import read_checkpoint, write_checkpoint
    from src.utils.checkpoint_paths import (
        ranking_score_detail,
        select_auto_run_checkpoint,
        select_run_checkpoint,
    )

    checkpoints = tmp_path / "runs" / "hgt" / "20261020-090000" / "checkpoints"
    best = checkpoints / "best.pt"
    other = checkpoints / "other.pt"
    best_digest = write_checkpoint({"state_dict": {}, "logs": {"val_mrr": 0.6}}, best)
    other_digest = write_checkpoint({"state_dict": {}, "logs": {"val_mrr": 0.2}}, other)
    os.utime(best, (1, 1))  # older, so recency would not favour it

    def score(path):
        logs = read_checkpoint(path, map_location="cpu", weights_only=True).checkpoint["logs"]
        return ranking_score_detail(logs)[1]

    def chosen():
        return (select_auto_run_checkpoint(tmp_path, score_fn=score)[0],
                select_run_checkpoint(tmp_path, "hgt", score_fn=score)[0])

    without = chosen()
    ledger = empty_ledger()
    ledger["records"] = [
        {**_record(other_digest, mrr=0.99), "checkpoint_digest": other_digest},
        {**_record(best_digest, sequence=2, mrr=0.01), "checkpoint_digest": best_digest},
    ]
    write_ledger(checkpoints / LEDGER_FILENAME, ledger, None)

    assert without == chosen() == (best, best)
