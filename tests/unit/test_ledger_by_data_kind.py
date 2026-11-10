"""
Registration, the ledger and the one reader by data kind.
=========================================================
Research comparison §5 Stage 6 (owner's O8; the plan review's F2, F4 and its
§5 items 3-4 and 10). A ledger record is built per data kind through one
function (`sidecar.build_record`, `_kind_parts`), in the same schema 4 file,
under the same writer's rules, and read back by one reader discriminated by
the record's own ``cohort.kind`` (`evaluations.read_records`). What is pinned
here, on reports built in memory (each kind's real runs are measured and
registered by the command line in `test_kg_link_prediction.py`,
`test_generated_measurement.py` and the integration tests):

- **each kind's record by its own rules**: a supplied cohort's exactly as
  before (the strict golden P0-1 pins its bytes); a KG edge set's cohort
  block (view, split, message-passing graph, pairs, counts with their units),
  no R10, the run's counts and metric statuses (``measured``) and its exposure
  block as written (F4); a generated split's role and generation identity, its
  counts with their units, no R10. The key fields keep their names and
  meanings for every kind;
- **refusals only where a record cannot be valid**: a block naming a field
  its kind does not carry, a kind its block contradicts, R10 on another kind,
  a KG edge set's run without its exposure or its metric statuses;
- **the writer's rules for every kind**: the same run's same bytes are
  ``already_registered``, its other bytes refused, and other runs, with equal
  bytes or different numbers, recorded each; the exposure never tells two
  runs of one group apart;
- **the one reader**: each kind read as its own view, a record of a kind
  this build does not read listed unreadable record by record (naming the
  field, never its value) and never hiding another; no case id, server path or
  R10 evidence path in any view; the loaded-model route answers every kind;
  `record_evaluation --show`, Models and the Test page say each kind in its
  own words, R10 only for a supplied cohort and the exposure only for a KG
  edge set;
- **automatic selection never reads test results**, whatever their kind
  (`selection-ignores-test-results`; `test_task_first_selection.py`).

Module: tests/unit/test_ledger_by_data_kind.py
"""
from __future__ import annotations

import copy
import json
import re
from types import SimpleNamespace

import pytest

from tests.unit.test_evaluation_sidecar import (
    CHECKPOINT,
    DISEASE_METRICS,
    REPORT_A,
    REPORT_B,
    RUN_1,
    RUN_2,
    RUN_3,
    WHEN,
    _path_like,
    _strings,
    r10_block,
    registered_manifest,
    registered_report,
)

GRAPH = {"kg": "1" * 64, "node_features": "2" * 64, "edge_indices": "3" * 64,
         "num_nodes": "4" * 64}
KG_METRICS = {"lp_accuracy": 0.5, "lp_roc_auc": 0.625, "lp_average_precision": 0.75,
              "lp_loss": 0.875}
KINDS = ("supplied", "kg_edges", "generated")


def kg_edges_block(**overrides):
    """A KG edge set's report block (`cohort_registry.cohort_version_block`):
    its view, split, message-passing graph, pairs and counts with their units,
    aggregates and digests only."""
    from src.evaluation.cohort_registry import KG_EDGES_COUNT_UNITS

    block = {
        "label": "kg-test", "role": "test", "kind": "kg_edges", "manifest_sha256": "8" * 64,
        "graph": dict(GRAPH), "dataset_record_sha256": "d" * 64,
        "graph_view": {"digest": "a1" * 32,
                       "rule": {"rule": "reference/v1", "deduplicate": True,
                                "drop_native_self_loops": True, "exclude_relations": [],
                                "largest_component": True}},
        "split": {"rule": "facts-80/10/10/v1", "seed": 0, "digest": "b2" * 32,
                  "view_digest": "a1" * 32, "edges": {"train": 50, "val": 6, "test": 8}},
        "message_passing": {"part": "train", "edges": 50, "includes_measured_edges": False},
        "pairs": {"part": "test", "rule": "fixed-lp-pairs/v1", "seed": 0, "batch_size": 512,
                  "batches": 1, "batch_pairs": [6], "rows": 6, "file": "pairs.bin",
                  "layout": {"columns": ["head", "relation", "tail", "negative_tail"],
                             "dtype": "<i8", "order": "rows"},
                  "sha256": "c3" * 32, "lp_pairs_digest": "e4" * 32},
        "counts": {"positive_edges": 8, "pairs": 6, "edges_without_negative": 2,
                   "message_passing_edges": 50},
        "count_units": dict(KG_EDGES_COUNT_UNITS),
    }
    block.update(overrides)
    return block


def exposure_block(state="held_out", k=0, n=6, reason=None):
    """A KG edge set's run's exposure block (`exposure.kg_edges_exposure`)."""
    from src.evaluation.caveats import (
        EXPOSURE_CAVEAT,
        EXPOSURE_DEFINITION_VERSION,
        EXPOSURE_READINGS,
        MEASURED_EDGE_UNIT,
    )

    return {
        "definition_version": EXPOSURE_DEFINITION_VERSION, "state": state,
        "reading": EXPOSURE_READINGS[state].format(k=k, n=n), "reason": reason,
        "sources": {
            "model_record": {"checkpoint_sha256": CHECKPOINT, "learning_stage": "base",
                             "operation": "new",
                             "fields_read": ["graph_exposure", "data_roles.split"],
                             "graph_exposure": {"base": {"training": {
                                 "supervision": "train edges"}}},
                             "split_digest": "b2" * 32},
            "version": {"label": "kg-test", "kind": "kg_edges", "role": "test",
                        "manifest_sha256": "8" * 64, "split_digest": "b2" * 32}},
        "split": "the version's split",
        "measured_edges": {"n": n, "unit": MEASURED_EDGE_UNIT,
                           "in_model_parts": {"train": k, "val": 0, "test": n - k},
                           "in_recorded_graph_use": k,
                           "recorded_graph_use": {"base.training.supervision": ["train"]}},
        "measurement_message_passing": {
            "graph": "the KG edge set's message-passing graph: its split's train part",
            "measured_edges_in_it": 0, "includes_measured_edges": False},
        "caveat": EXPOSURE_CAVEAT,
    }


def kg_edges_report(metrics=None, not_finite=0, exposure=None, **manifest_overrides):
    """A KG edge set's link-prediction report, as `task_measurement` writes
    it: the manifest of a kg_edges version (no R10), metric schema 3 with each
    metric's status, the run's counts with their units, its exposure block and
    its scores file's entry."""
    from src.evaluation.task_measurement import LP_COUNT_UNITS, lp_metric_statuses

    values, statuses = lp_metric_statuses(dict(metrics or KG_METRICS), pairs=6,
                                          not_finite=not_finite)
    manifest = registered_manifest(**{
        "mode": "kg_link_prediction", "metric_schema_version": 3,
        "candidate_construction": "each positive edge of the KG edge set against its one "
                                  "negative tail, as its pairs file stores them",
        "score_semantics": "LP head: DistMult raw score sum(h * w_r * t); decision raw > 0",
        "canonical_tie_policy_version": "average-rank/v1", "batch_size": 7, "split": "test",
        "cohort_kind": "kg_edges", "n_samples": 8, "cohort_version": kg_edges_block(),
        "r10": None,
        "artifact_digests": {"checkpoint": CHECKPOINT, "pairs": "c3" * 32, **GRAPH},
        **manifest_overrides})
    return {
        "manifest": manifest, "authoritative_metrics": values, "n_ranked": 6,
        "n_ground_truth_absent": 0,
        "sampler_evidence": {
            "graph_view": "a1" * 32,
            "message_passing": {"part": "train", "split_digest": "b2" * 32, "edges": 50,
                                "measured_edges_in_it": 0, "includes_measured_edges": False},
            "pairs": {"rule": "fixed-lp-pairs/v1", "rows": 6, "lp_pairs_digest": "e4" * 32,
                      "pairs_scored_at_once": 7},
            "counts": {"positive_edges": 8, "pairs": 6, "edges_without_negative": 2,
                       "pairs_scored": 6 - not_finite, "pairs_not_finite": not_finite,
                       "message_passing_edges": 50},
            "count_units": dict(LP_COUNT_UNITS),
            "metrics": statuses},
        "exposure": exposure or exposure_block(),
        "scores": {"file": "scores.bin", "sha256": "f5" * 32, "rows": 6, "bytes": 102},
    }


def generated_block(**overrides):
    """A registered generated split's report block: its role and generation
    identity, its counts with their units, the gene fixture it keeps (none
    here); never a sample."""
    from src.evaluation.cohort_registry import GENERATED_COUNT_UNITS

    block = {
        "label": "gen-val", "role": "val", "kind": "generated", "manifest_sha256": "9" * 64,
        "graph": dict(GRAPH), "dataset_record_sha256": "d" * 64, "samples_sha256": "5" * 64,
        "split_manifest": {"file": "split_manifest.json", "schema_version": 3,
                           "sha256": "6" * 64},
        "split_artifacts": {"train_samples": "7" * 64, "val_samples": "5" * 64},
        "generation": {"algorithm": "coverage-first-then-replacement", "algorithm_version": 1,
                       "num_train": 20, "num_val": 5, "min_phenotypes": 2,
                       "max_phenotypes": 15, "phenotype_drop_rate": 0.3},
        "allocation": {"algorithm": "uniform-without-replacement", "algorithm_version": 1,
                       "seed": 42, "eligible_diseases": 4, "val_fraction_requested": 0.2},
        "counts": {"samples": 5, "diseases": 1},
        "count_units": dict(GENERATED_COUNT_UNITS),
        "task_fixture": None,
    }
    block.update(overrides)
    return block


def generated_report(metrics=None, **manifest_overrides):
    """A disease task report of a registered generated split (val): its block,
    no R10."""
    return registered_report(metrics, 5, **{
        "split": "val", "cohort_kind": "generated", "n_samples": 5,
        "cohort_version": generated_block(), "r10": None, **manifest_overrides})


REPORTS = {"supplied": registered_report, "kg_edges": kg_edges_report,
           "generated": generated_report}


def _record(kind, digest=REPORT_A, **report):
    from src.evaluation.sidecar import build_record

    return build_record(REPORTS[kind](**report), digest)


def _ledger(*records):
    """A ledger holding `records` (each ``(record, run_id)``), appended in order."""
    from src.evaluation.sidecar import append_record, empty_ledger

    ledger = empty_ledger()
    for record, run_id in records:
        outcome, ledger = append_record(ledger, record, run_id=run_id, registered_at=WHEN)
        assert outcome == "registered"
    return ledger


# ---------------------------------------------------------------------------
# Each kind's record, by its own rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_each_kinds_record_is_built_by_its_own_rules(kind):
    """The cohort block copies the kind's fields (`RECORD_COHORT_FIELDS`)
    beside the kind; the key fields keep their meanings (the version's role,
    its manifest's digest); R10 is a supplied cohort's mapping and null for
    every other kind; a KG edge set's run keeps its counts and metric statuses
    and its exposure block as written; every field of a record is a key,
    compared or excluded field, and none escapes the comparison by oversight."""
    from src.evaluation.sidecar import (
        KEY_FIELDS,
        KIND_RECORD_FIELDS,
        RECORD_COHORT_FIELDS,
        RECORD_COMPARED_FIELDS,
        RECORD_EXCLUDED_FIELDS,
        REGISTERED_RELATIONS,
        build_record,
    )

    report = REPORTS[kind]()
    block = report["manifest"]["cohort_version"]
    record = build_record(report, REPORT_A)

    assert record["cohort"] == {"kind": kind, **{field: block[field]
                                                 for field in RECORD_COHORT_FIELDS[kind]}}
    assert (record["cohort_role"], record["cohort_digest"]) == (block["role"],
                                                                block["manifest_sha256"])
    assert record["relations"] == REGISTERED_RELATIONS
    assert (record["r10"] is None) == (kind != "supplied")
    others = {f for k, fields in KIND_RECORD_FIELDS.items() if k != kind for f in fields}
    every = set(KEY_FIELDS) | set(RECORD_COMPARED_FIELDS) | set(RECORD_EXCLUDED_FIELDS)
    assert set(record) | {"run_id", "corroborating_reports", "registration"} == (
        every - (others - set(KIND_RECORD_FIELDS[kind])))
    if kind == "kg_edges":
        assert record["exposure"] == report["exposure"]
        assert record["exposure"] is not report["exposure"]
        assert record["measured"] == {key: report["sampler_evidence"][key]
                                      for key in ("counts", "count_units", "metrics")}
        assert {name: s["status"] for name, s in record["measured"]["metrics"].items()} == (
            dict.fromkeys(KG_METRICS, "computed"))
    else:
        assert "exposure" not in record and "measured" not in record
    if kind == "generated":
        for field in ("split_manifest", "split_artifacts", "generation", "allocation"):
            assert record["cohort"][field] == block[field]


def test_a_metric_that_could_not_be_computed_is_a_status_in_the_record():
    """A KG edge set's run with a raw score that is not finite computes no
    whole-set metric: the record holds no number for any, and its
    ``measured`` says each is not computable, so a reader of the ledger alone
    never reads a missing metric as one it can guess."""
    record = _record("kg_edges", not_finite=1)

    assert record["metrics"] == {}
    assert {s["status"] for s in record["measured"]["metrics"].values()} == {"not_computable"}
    assert record["measured"]["counts"]["pairs_not_finite"] == 1


@pytest.mark.parametrize("kind, edit, says", [
    ("kg_edges", lambda r: r["manifest"]["cohort_version"].__setitem__("edges", [[0, 1, 2]]),
     "names edges, which a KG edge set's report block does not carry, so it is not a KG edge "
     "set this revision records"),
    ("generated", lambda r: r["manifest"]["cohort_version"].__setitem__("samples", [{"id": 1}]),
     "names samples, which a generated split's report block does not carry, so it is not a "
     "generated split this revision records"),
    ("kg_edges", lambda r: r["manifest"]["cohort_version"].pop("count_units"),
     "lacks count_units"),
    ("generated", lambda r: r["manifest"]["cohort_version"].__setitem__("generation", None),
     "cohort_version.generation is not an object"),
    ("kg_edges", lambda r: r["manifest"].__setitem__("cohort_kind", "generated"),
     "records cohort_kind 'generated' beside a cohort version of kind 'kg_edges'"),
    ("supplied", lambda r: r["manifest"].__setitem__("cohort_kind", "kg_edges"),
     "records cohort_kind 'kg_edges' beside a cohort version of kind 'supplied'"),
    ("kg_edges", lambda r: r["manifest"].__setitem__("r10", r10_block()),
     "R10 is a supplied cohort version's only"),
    ("generated", lambda r: r["manifest"].__setitem__("r10", r10_block()),
     "R10 is a supplied cohort version's only"),
    ("kg_edges", lambda r: r.pop("exposure"), "its exposure block is not an object"),
    ("kg_edges", lambda r: r["exposure"].pop("definition_version"),
     "its exposure.definition_version is not a non-empty string"),
    ("kg_edges", lambda r: r["sampler_evidence"].pop("metrics"),
     "its sampler_evidence.metrics is not an object"),
    ("generated", lambda r: r["manifest"].__setitem__("split", "test"),
     "names the cohort role 'test' and its cohort version the role 'val'"),
    ("kg_edges", lambda r: r["exposure"]["sources"]["model_record"].__setitem__(
        "checkpoint_sha256", "e" * 64),
     "its exposure.sources.model_record.checkpoint_sha256 is not the checkpoint its "
     "artifact_digests name, so the report contradicts itself"),
    ("kg_edges", lambda r: r["exposure"]["sources"]["version"].__setitem__(
        "manifest_sha256", "e" * 64),
     "its exposure.sources.version.manifest_sha256 is not its cohort_version's "
     "manifest_sha256, so the report contradicts itself"),
], ids=["kg-edges-listed", "generated-samples-listed", "kg-no-units", "generated-no-generation",
        "kind-and-block-disagree", "supplied-beside-kg-block", "kg-with-r10",
        "generated-with-r10", "kg-no-exposure", "kg-exposure-no-definition",
        "kg-no-metric-statuses", "generated-role-said-test", "kg-exposure-another-checkpoint",
        "kg-exposure-another-version"])
def test_a_record_that_cannot_be_valid_is_refused_by_name(kind, edit, says):
    """Aggregates only, and each kind's block exactly: an edge or a sample
    cannot ride into a record on a field nobody classified; a report whose
    kind, role or R10 contradicts its block, or a KG edge set's run without its
    exposure or metric statuses, is refused naming what is wrong, and so is a
    KG edge set's exposure naming another checkpoint or version than the
    report measured. A KG edge set or a generated split is never called a
    cohort version in its refusal. Never a refusal for the role itself: a
    generated split's role is recorded as it is."""
    from src.evaluation.sidecar import build_record

    report = REPORTS[kind]()
    edit(report)
    with pytest.raises(ValueError, match=re.escape(says)) as refused:
        build_record(report, REPORT_A)
    if kind != "supplied":
        assert "cohort version this revision records" not in str(refused.value)


# ---------------------------------------------------------------------------
# The writer's rules, the same for every kind
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ("kg_edges", "generated"))
def test_the_writers_rules_hold_for_every_kind(kind):
    """The same run's same bytes are ``already_registered`` and nothing is
    written; the same run's other bytes are refused (a run's report is sealed
    when it is published); another run with equal bytes, and another with
    different numbers, are recorded each, in one condition group, the
    difference named and never refused."""
    from src.evaluation.sidecar import (
        append_record,
        group_differences,
        held_record,
        record_key,
    )

    first = _record(kind)
    ledger = _ledger((first, RUN_1))
    assert append_record(ledger, first, run_id=RUN_1, registered_at=WHEN) == (
        "already_registered", ledger)
    with pytest.raises(ValueError, match=f"run {RUN_1} is already registered as record 1"):
        append_record(ledger, _record(kind, REPORT_B), run_id=RUN_1, registered_at=WHEN)

    outcome, ledger = append_record(ledger, copy.deepcopy(first), run_id=RUN_2,
                                    registered_at=WHEN)
    assert outcome == "registered"
    metrics = (dict(KG_METRICS, lp_roc_auc=0.5) if kind == "kg_edges"
               else dict(DISEASE_METRICS, pooled_mrr=0.25))
    other = _record(kind, REPORT_B, metrics=metrics)
    outcome, ledger = append_record(ledger, other, run_id=RUN_3, registered_at=WHEN)
    assert outcome == "registered"
    assert [r["run_id"] for r in ledger["records"]] == [RUN_1, RUN_2, RUN_3]
    assert len({record_key(r) for r in ledger["records"]}) == 1
    held = held_record(ledger, other, RUN_3)
    (first_differs, second_differs) = group_differences(ledger, held)
    expected = ["lp_roc_auc"] if kind == "kg_edges" else ["pooled_mrr"]
    assert list(first_differs.differing) == list(second_differs.differing) == expected


def test_the_exposure_never_tells_two_runs_of_one_group_apart():
    """The exposure block is a function of the keyed checkpoint and version
    under its definition: outside the comparison, so two runs whose blocks
    were worded differently (another build's reason) are one group and agree;
    a run's own counts (``measured``) are compared, so a run with a score that
    is not finite is described as differing."""
    from src.evaluation.sidecar import differing_fields, record_key

    one = _record("kg_edges")
    worded = _record("kg_edges", REPORT_B,
                     exposure=exposure_block(reason="another build's words"))
    assert record_key(one) == record_key(worded)
    assert differing_fields(one, worded) == []

    not_finite = _record("kg_edges", REPORT_B, not_finite=1)
    assert record_key(one) == record_key(not_finite)
    assert differing_fields(one, not_finite) == ["metrics", "measured"]


def test_measured_is_compared_by_its_facts_never_its_words():
    """``measured`` keeps the run's units and reasons as its report wrote
    them, for a reader holding only the ledger, but the comparison reads its
    facts only (its counts and each metric's status and denominator count):
    two runs whose results agree are never described as differing because a
    later build reworded a unit or a reason (seen in review: a reworded pair
    unit gave ``differing: ['measured']``), while two runs whose numbers of
    pairs without a finite score differ, and whose metrics are both absent,
    are still described as differing."""
    from src.evaluation.sidecar import differing_fields, record_key

    one = _record("kg_edges")
    reworded = _record("kg_edges", REPORT_B)
    reworded["measured"]["count_units"]["pairs"] = "another build's words for the pairs"
    for status in reworded["measured"]["metrics"].values():
        status["unit"] = "another build's unit"
        status["denominator"]["unit"] = "another build's denominator"
    assert record_key(one) == record_key(reworded)
    assert reworded["measured"] != one["measured"]
    assert differing_fields(one, reworded) == []

    once = _record("kg_edges", REPORT_A, not_finite=1)
    reasoned = copy.deepcopy(once)
    for status in reasoned["measured"]["metrics"].values():
        status["reason"] = "another build's reason"
    assert differing_fields(once, reasoned) == []
    twice = _record("kg_edges", REPORT_B, not_finite=2)
    assert once["metrics"] == twice["metrics"] == {}
    assert differing_fields(once, twice) == ["measured"]


# ---------------------------------------------------------------------------
# The one reader, by data kind
# ---------------------------------------------------------------------------
def _mixed_ledger():
    """Records of every kind for one checkpoint, and three more a hand edit
    of the file then left in shapes no build reads (the sanctioned correction
    path edits the file): a kind no build reads (its value a server path), a
    record with no kind, a KG edge set's record carrying R10."""
    kinds = (*KINDS, "supplied", "kg_edges", "kg_edges")
    ledger = _ledger(*((_record(kind, f"{n:x}" * 64), f"20261110T00000{n}Z-cccccccccccc-0000000{n}")
                       for n, kind in enumerate(kinds, start=1)))
    edited = ledger["records"]
    edited[3]["cohort"]["kind"] = "/srv/private/cases"
    del edited[4]["cohort"]["kind"]
    edited[5]["r10"] = r10_block()
    return ledger


def test_one_reader_reads_each_kind_and_lists_what_it_cannot_read_record_by_record():
    """Each record is read by its own kind into that kind's view, in ledger
    order; a record of a kind this build does not read, or of no kind, or
    whose fields are not its kind's, is listed unreadable with its index,
    sequence and run, the reason naming the field and never its value, and
    never hides another record (mixed-ledger acceptance (a))."""
    from src.api.routes.evaluations import (
        GeneratedRecordView,
        KgEdgesRecordView,
        RecordView,
        read_records,
    )

    ledger = _mixed_ledger()
    viewed, unreadable = read_records(ledger, CHECKPOINT)

    assert [type(view) for _record, view in viewed] == [
        RecordView, KgEdgesRecordView, GeneratedRecordView]
    assert [view.cohort.kind for _record, view in viewed] == list(KINDS)
    assert [(item.index, item.sequence) for item in unreadable] == [(3, 4), (4, 5), (5, 6)]
    assert [item.run_id for item in unreadable] == [
        r["run_id"] for r in ledger["records"][3:]]
    assert unreadable[0].reason == (
        "cohort.kind: not a data kind this build reads (supplied, kg_edges, generated)")
    assert unreadable[1].reason == "cohort.kind: missing"
    assert unreadable[2].reason == "r10: Input should be None"
    assert "/srv/private" not in json.dumps([item.model_dump() for item in unreadable])


def _another_definition(block):
    """An exposure block of another definition (a later build's), in its own
    shape: none of this definition's counts or split relation."""
    return {"definition_version": "kg-lp-exposure/v2", "state": "held_out",
            "reading": "held out relative to a later build's evidence",
            "sources": {"model_record": {"checkpoint_sha256": CHECKPOINT}}}


def _unhex_split_digest(block):
    """A not_recorded block whose model record's split digest, as that record
    wrote it, is not a digest (the measurement copies it as written)."""
    block = exposure_block(state="not_recorded", reason="the record's split does not rebuild")
    block["sources"]["model_record"]["split_digest"] = "not-a-digest"
    block["split"] = None
    return block


@pytest.mark.parametrize("change, problem", [
    (_another_definition, "exposure.split: missing; exposure.measured_edges: missing; "
                          "exposure.measurement_message_passing: missing; 4 more"),
    (_unhex_split_digest, "exposure.sources.model_record.split_digest: String should match "
                          "pattern '^[0-9a-f]{64}$'"),
    (lambda block: None, "exposure: missing"),
], ids=["another-definition", "split-digest-not-a-digest", "no-exposure-in-the-ledger"])
def test_an_exposure_this_build_cannot_read_never_hides_its_result(change, problem):
    """F4 and §4.2: the exposure is never a gate. A KG edge set's run whose
    exposure block this build cannot read as its view (another definition's
    shape, a model record's split digest copied as written, a hand edit that
    removed it) is registered (the writer checks only that its sources, where
    named, are this run's) and read: its metrics and counts are shown, its
    exposure reads "exposure not recorded", and the problem names fields,
    never values. Seen in review: such a record was registered and then
    listed unreadable by every reader."""
    from scripts.record_evaluation import _run_lines
    from src.api.routes.evaluations import KgEdgesRecordView, read_records
    from src.webui.components import models_page as mp

    block = change(exposure_block())
    ledger = _ledger((_record("kg_edges", exposure=block), RUN_1))
    if block is None:  # a hand edit of the ledger removed it
        del ledger["records"][0]["exposure"]
    else:
        assert ledger["records"][0]["exposure"] == block
    ((_held, view),), unreadable = read_records(ledger, CHECKPOINT)

    assert unreadable == [] and isinstance(view, KgEdgesRecordView)
    assert view.metrics == KG_METRICS and view.measured.counts["pairs_scored"] == 6
    assert view.exposure is None and view.exposure_problem == problem
    assert "not-a-digest" not in view.model_dump_json()
    (line,) = _run_lines(ledger, CHECKPOINT)
    assert line.endswith(f"; a KG edge set (test); exposure: exposure not recorded (its block "
                         f"does not read: {problem})")
    reading = __import__("src.api.routes.evaluations",
                         fromlist=["record_reading"]).record_reading(ledger)
    identity, overlaps = reading[id(ledger["records"][0])]
    shown = _text(mp.records_table_html([mp.record_view(
        ledger["records"][0], [identity.run_id, identity.historical, identity.problem,
                               list(overlaps)])], {}))
    assert "Exposure: exposure not recorded" in shown
    assert f"Its block does not read: {problem}" in shown
    assert "lp_roc_auc 0.625" in shown


@pytest.mark.parametrize("path, own", [
    (("sources", "model_record", "checkpoint_sha256"), "checkpoint_digest"),
    (("sources", "version", "manifest_sha256"), "cohort_digest"),
])
def test_an_exposure_naming_another_checkpoint_or_version_makes_the_record_unreadable(path,
                                                                                    own):
    """A record whose exposure names another checkpoint or version than its
    own contradicts itself: the one reader lists it unreadable, naming the
    field and never its value, and the writer refuses such a report
    (`test_a_record_that_cannot_be_valid_is_refused_by_name`), so the writer
    never appends a record its reader refuses for it. Here a hand edit of the
    ledger made it."""
    from src.api.routes.evaluations import read_records

    ledger = _ledger((_record("kg_edges"), RUN_1))
    held = ledger["records"][0]["exposure"]
    for key in path[:-1]:
        held = held[key]
    held[path[-1]] = "e" * 64
    viewed, (item,) = read_records(ledger, CHECKPOINT)

    assert viewed == []
    assert item.reason == f"exposure.{'.'.join(path)}: not the record's {own}"
    assert "e" * 64 not in item.reason


def test_no_view_carries_a_case_id_a_server_path_or_an_r10_evidence_path():
    """D7: a supplied record's R10 evidence path, a report's checkpoint path,
    and a KG edge set's exposure reason quoting a path (a hand-edited model
    record) stay in the ledger; no view of any kind carries a path-like string
    or a case id, and a KG edge set's exposure view carries no reason or
    caveat text."""
    from src.api.routes.evaluations import read_records
    from src.evaluation.sidecar import append_record

    supplied = _record("supplied", r10=r10_block(training_state="overlap"))
    assert supplied["r10"]["training"]["evidence"]["path"] == "/abs/train_samples.json"
    kg = _record("kg_edges", REPORT_B, exposure=exposure_block(
        state="not_recorded", reason="the record's graph_exposure.base says '/srv/ws/x'"))
    ledger = _ledger((supplied, RUN_1), (kg, RUN_2))
    ledger = append_record(ledger, _record("generated", "a" * 64), run_id=RUN_3,
                           registered_at=WHEN)[1]
    viewed, unreadable = read_records(ledger, CHECKPOINT)
    assert unreadable == [] and len(viewed) == 3

    for view in (view for _record_, view in viewed):
        dumped = view.model_dump(mode="json")
        paths = [(at, text) for at, text in _strings(dumped) if _path_like(text)]
        assert paths == [], paths
        assert "case_id" not in json.dumps(dumped)
    exposure = viewed[1][1].model_dump(mode="json")["exposure"]
    assert "reason" not in exposure and "caveat" not in exposure
    assert exposure["state"] == "not_recorded" and exposure["measured_edges"]["n"] == 6


def test_the_loaded_model_route_answers_every_kind_through_its_union(tmp_path, monkeypatch):
    """The route's answer holds each kind's view with its group's description,
    told apart by ``cohort.kind``; the answer validates and serialises, and
    the API's schema names each kind's view."""
    from src.api.routes import evaluations
    from src.evaluation.sidecar import LEDGER_FILENAME, write_ledger

    monkeypatch.delenv("SHEPHERD_REGISTRY", raising=False)
    write_ledger(tmp_path / LEDGER_FILENAME, _mixed_ledger(), None)
    read = SimpleNamespace(path=str(tmp_path / "last.pt"), sha256=CHECKPOINT)
    answer = evaluations._recorded_results(read)

    assert answer.state == "records" and answer.records_not_shown == 3
    assert [type(view) for view in answer.records] == [
        evaluations.LoadedRecordView, evaluations.LoadedKgEdgesRecordView,
        evaluations.LoadedGeneratedRecordView]
    body = json.loads(answer.model_dump_json())
    assert [view["cohort"]["kind"] for view in body["records"]] == list(KINDS)
    assert all(view["contest"]["state"] == "cannot_be_checked" for view in body["records"])
    assert evaluations.LoadedModelEvaluations.model_validate(body) == answer

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(evaluations.router)
    schemas = app.openapi()["components"]["schemas"]
    for name in ("LoadedRecordView", "LoadedKgEdgesRecordView", "LoadedGeneratedRecordView",
                 "ExposureView"):
        assert name in schemas, name


def test_show_says_each_kind_in_its_own_words():
    """`record_evaluation --show`'s lines come from the one reader: a KG edge
    set's record says its version and its exposure, a generated split's its
    version, a supplied cohort's its run only, as before; a record the reader
    cannot read says why, never its value."""
    from scripts.record_evaluation import _run_lines

    lines = _run_lines(_mixed_ledger(), CHECKPOINT)

    assert lines[0] == "record 1: run 20261110T000001Z-cccccccccccc-00000001"
    assert lines[1] == ("record 2: run 20261110T000002Z-cccccccccccc-00000002; a KG edge set "
                        "(test); exposure: held out relative to the recorded training and "
                        "validation graph use")
    assert lines[2] == ("record 3: run 20261110T000003Z-cccccccccccc-00000003; a generated "
                        "split (val)")
    assert lines[3] == ("record 4 (the ledger's record 3) cannot be read as a result: "
                        "cohort.kind: not a data kind this build reads (supplied, kg_edges, "
                        "generated)")
    assert not any("/srv/private" in line for line in lines)


def _text(markup: str) -> str:
    import html

    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", markup))).strip()


def test_models_shows_each_kind_by_the_one_display_with_its_exposure():
    """Models (O8 ii): a KG edge set's record is shown by the one generic
    display (every metric as recorded in one column; no figure chosen for it),
    named "A KG edge set (test)" under "Data version", with its exposure in
    this build's words, its message passing and pairs, and its counts with
    their units; a generated split's with its generation identity and how it
    stands to training. Neither shows R10 or a cohort's words; a table of
    supplied cohorts' records reads as it always did. Every value is
    escaped."""
    from src.api.routes.evaluations import record_reading
    from src.webui.components import models_page as mp

    ledger = _ledger((_record("supplied"), RUN_1), (_record("kg_edges", REPORT_B), RUN_2),
                     (_record("generated", "a" * 64), RUN_3))
    reading = record_reading(ledger)
    views = []
    for record in ledger["records"]:
        identity, overlaps = reading[id(record)]
        views.append(mp.record_view(record, [identity.run_id, identity.historical,
                                             identity.problem, list(overlaps)]))
    supplied, kg, generated = (_text(mp.records_table_html([v], {})) for v in views)

    assert "Cohort version" in supplied and "R10, training inputs" in supplied
    assert "Exposure" not in supplied and "Data version" not in supplied
    assert "Data version" in kg and "A KG edge set (test)" in kg
    assert "Recorded metrics" in kg and "lp_roc_auc 0.625" in kg
    assert ("Exposure: held out relative to the recorded training and validation graph use"
            in kg)
    assert "Definition kg-lp-exposure/v1; the version's split" in kg
    assert "6 of 6 stored pairs scored" in kg and "8 positive edges" in kg
    assert "positive_edges 8: directed graph-view edges of the role's part" in kg
    assert "R10" not in kg and "cohort version" not in kg.lower().replace("data version", "")
    assert "A generated split (val)" in generated and "Generation:" in generated
    assert "allocation uniform-without-replacement v1 with seed 42" in generated
    assert "5 generated patients of 1 disease" in generated
    assert "R10" not in generated and "Exposure:" not in generated
    assert "held out" not in generated and "held out" not in supplied

    # A train or val result is never named a test result (F2): the table, the
    # segment and the read-only title say "Recorded results" for every kind.
    # Seen in review: a generated val record under "Recorded test results".
    for view in views:
        markup = mp.records_table_html([view], {})
        assert 'aria-label="Recorded results, ' in markup
        assert "test result" not in markup.lower()
    assert mp.READONLY_TITLE == "Recorded results of the model in use"

    block = _text(mp.records_block_html(views[1:2], {}))
    assert "R10 labels" not in block
    assert mp.PAGE_HELP["models-exposure"][1].endswith(
        __import__("src.evaluation.caveats", fromlist=["EXPOSURE_CAVEAT"]).EXPOSURE_CAVEAT)

    hostile = copy.deepcopy(views[1])
    hostile["exposure"]["definition_version"] = "<b>kg-lp-exposure/v9</b>"
    hostile["exposure"]["reading"] = "<img src=x onerror=alert(1)>"
    markup = mp.records_table_html([hostile], {})
    assert "<img" not in markup and "&lt;img src=x onerror=alert(1)&gt;" in markup


def test_the_test_page_says_a_kg_edge_set_and_a_generated_split_in_their_words():
    """The Test page's result shows the record the ledger would hold: a KG
    edge set's with its pairs scored and its exposure, a generated split's
    with its generation identity, named by kind and role; neither with R10 or
    a supplied cohort's counts."""
    from src.webui.components import test_page as tp

    kg = _text(tp._record_html(_record("kg_edges"), {}, {}))
    generated = _text(tp._record_html(_record("generated"), {}, {}))
    supplied = _text(tp._record_html(_record("supplied"), {}, {}))

    assert "a KG edge set (test)" in kg and "6 of 6 stored pairs scored" in kg
    assert ("Exposure held out relative to the recorded training and validation graph use "
            "(definition kg-lp-exposure/v1)") in kg
    assert "a generated split (val)" in generated and "Generation" in generated
    assert "5 generated patients of 1 disease" in generated
    assert "share of the cases measured that were ranked" in generated
    for text in (kg, generated):
        assert "R10" not in text and "source cases" not in text
        assert "import-eligible" not in text
    assert "R10, training" in supplied and "import-eligible cases" in supplied
