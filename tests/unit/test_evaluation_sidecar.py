"""EVALUATION_COHORTS §6.5 — evaluation records beside the checkpoint, schemas 3 and 4.

A result written into a `.pt` changes its SHA-256, and the M1-M5 chain cites
checkpoints by digest, so the record has to live outside the weights it is about.

**The load-bearing property is what absence means.** A ledger holding no record
for a checkpoint says only that *this ledger* holds none. These tests pin that
wording, because the tempting shortcut -- treating a missing record as "never
evaluated" -- is a claim the system cannot support and would be believed.

**The reports here are built by `MeasurementManifest` itself** (`registered_report`
below, through `dataclasses.asdict`), so the record's classification is checked
against the real dataclass: a field added to the manifest reaches these reports
and is refused by `build_record` until it is classified.

**Since K8-6r.2 (policy rev 16) a record names its run**, and the writer looks
for this run first: the same run's same bytes are ``already_registered``, the
same run's other bytes are refused (a sealed run), and any other run gets its
own record, whatever its bytes or values. Records written before each run had
its own record are read as written and decide nothing; the ledger is schema 4
from its first append. The acceptance rows A1-A12 are pinned below at the
writer's level ("Run identity") and through `register_report` in
`tests/unit/test_registration.py`.

The registration entry points, the BACKLOG 18 reads and the CLI are in
`tests/unit/test_registration.py`.

Module: tests/unit/test_evaluation_sidecar.py
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict

import pytest

from src.evaluation.sidecar import (
    ENVIRONMENT_CLASSES,
    KEY_FIELDS,
    LEDGER_SCHEMA_VERSION,
    NON_SEMANTIC_FIELDS,
    RECORD_COMPARED_FIELDS,
    RECORD_EXCLUDED_FIELDS,
    REGIME_FIELDS,
    REGISTERED_RELATIONS,
    SCORING_SEMANTICS_FIELDS,
    SEMANTIC_MANIFEST_FIELDS,
    append_record,
    build_record,
    checkpoint_matches,
    empty_ledger,
    environment_class,
    find_checkpoint,
    ledger_digest,
    measurement_semantics_digest,
    read_ledger,
    record_key,
    records_for,
    scoring_semantics_digest,
    write_ledger,
)

CHECKPOINT = "c" * 64
REPORT_A = "e" * 64
REPORT_B = "f" * 64
WHEN = "2026-10-16T12:00:00Z"
#: Run directories of `CHECKPOINT`, as `create_report_dir` names them.
RUN_1 = "20261016T120000Z-cccccccccccc-00000001"
RUN_2 = "20261016T120001Z-cccccccccccc-00000002"
RUN_3 = "20261016T120002Z-cccccccccccc-00000003"


def cohort_version(**overrides):
    """A registered cohort version's report block (spec §4.1): aggregates only."""
    block = {
        "label": "mygene2-2026-10",
        "role": "mygene2",
        "manifest_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "source_format": "shepherd-cases/1",
        "samples_sha256": "5" * 64,
        "graph": {"kg": "1" * 64, "node_features": "2" * 64,
                  "edge_indices": "3" * 64, "num_nodes": "4" * 64},
        "dataset_record_sha256": "d" * 64,
        "mapping": {"phenotype_normalisation_version": 1, "phenotype_namespace": "HP",
                    "disease_namespace": "MONDO", "partial_phenotypes": "keep"},
        "counts": {"source_cases": 100, "ranked_eligible": 80,
                   "excluded": {"ground_truth_absent": 12, "phenotype_unmappable": 8},
                   "n_excluded": 20, "cases_with_lost_terms": 9, "terms_unmappable": 14,
                   "repeats_removed": 3, "cases_with_repeats": 2},
    }
    block.update(overrides)
    return block


def r10_block(training_state="unverifiable"):
    """An R10 block in the shape of spec §3.2, whose labels' states follow their
    counts: ``training_state`` is the disease overlap's state; under ``overlap`` the
    phenotype sets are disjoint (``none``, the contract's own example), and under
    ``none`` both labels count 0."""
    unverifiable = training_state == "unverifiable"
    if unverifiable:
        disease = {"state": "unverifiable", "k": None, "n": 50, "cases": None}
        phenotype = {"state": "unverifiable", "k": None, "n": 80}
    elif training_state == "overlap":
        disease = {"state": "overlap", "k": 3, "n": 50, "cases": 5}
        phenotype = {"state": "none", "k": 0, "n": 80}
    else:
        disease = {"state": training_state, "k": 0, "n": 50, "cases": 0}
        phenotype = {"state": training_state, "k": 0, "n": 80}
    return {
        "definition_version": 1,
        "candidates_examined": 2,
        "training": {
            "recorded_sha256": "7" * 64,
            "evidence": {"state": "not_found" if unverifiable else "found",
                         "path": None if unverifiable else "/abs/train_samples.json",
                         "problem": None},
            "disease_overlap": disease,
            "phenotype_set_overlap": phenotype,
        },
        "validation": {
            "recorded_sha256": None,
            "evidence": {"state": "not_recorded", "path": None, "problem": None},
            "disease_overlap": {"state": "no_validation_inputs_recorded", "k": None,
                                "n": 50, "cases": None},
            "phenotype_set_overlap": {"state": "no_validation_inputs_recorded", "k": None,
                                      "n": 80},
        },
    }


def environment_block(driver="560.35"):
    return {
        "facts": {"python": "3.12.3", "torch": "2.10.0", "cudnn": 91501,
                  "gpu": {"name": "NVIDIA GeForce RTX 4090", "capability": [8, 9],
                          "driver": driver}},
        "code": {"state": "recorded", "revision": "9" * 40, "dirty": False,
                 "modified_tracked": 0, "untracked": 0},
    }


def registered_manifest(**overrides):
    """A registered cohort version's manifest, as `MeasurementManifest` builds it
    and `ModeResult.to_dict` writes it (`dataclasses.asdict`).

    **By default a disease task measurement's** (`--mode disease`): its `mode`
    is the protocol asked, set here as `scripts/measure_scorer.build_manifest`
    sets it from `--mode`; the evaluator's values are
    `task_measurement.disease_semantics` at the default `--batch-size` (32), as
    `build_manifest(..., semantics=)` takes them; every other field is built
    there the same way for every protocol. A Mode C manifest is
    `mode_c_report`'s, a Mode D one `mode_d_report`'s."""
    from src.evaluation.measurement import MeasurementManifest, expected_relations
    from src.evaluation.task_measurement import disease_semantics

    fields = dict(
        mode="disease",
        **disease_semantics(32),
        split="mygene2",
        cohort_kind="supplied",
        n_samples=80,
        checkpoint_path="/srv/ws/runs/shepherd/r1/checkpoints/best.pt",
        data_dir="/srv/ws",
        graph_fingerprint={"num_nodes": {"disease": 4}},
        artifact_digests={"checkpoint": CHECKPOINT, "samples": "5" * 64,
                          "node_features": "2" * 64, "edge_indices": "3" * 64,
                          "num_nodes": "4" * 64, "kg": "1" * 64},
        dataset_record={"binding": "verified", "origin": "synthetic", "sha256": "d" * 64,
                        "sources": [], "incomplete_by_design": ["no ontology files"]},
        cuda_executed=True,
        software_revision="9" * 40,
        torch_version="2.10.0",
        cuda_version="13.0",
        device="cuda",
        dtype="torch.float32",
        amp_enabled=False,
        amp_dtype=None,
        torch_compile_wrapped=False,
        deterministic_algorithms=True,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        cohort_version=cohort_version(),
        relations=expected_relations(registered=True),
        r10=r10_block(),
        environment=environment_block(),
    )
    fields.update(overrides)
    return asdict(MeasurementManifest(**fields))


#: Pooled top-k of the default disease report over `task_measurement.TASK_KS`.
_DISEASE_TOP_K = {1: 0.25, 3: 0.5, 5: 0.75, 10: 0.75, 20: 1.0, 50: 1.0, 100: 1.0}

#: A disease task report's metrics: the names `ranking_report` writes over
#: `TASK_KS`, in its order, as `task_measurement._mode_result` keeps them (the
#: numbers only). Its 80 cases are all ranked, each with one truth (20 each at
#: ranks 1, 2, 4 and 20), so macro recall is pooled top-k and every MRR is the
#: same (0.45).
DISEASE_METRICS = {
    "coverage": 1.0,
    "pooled_mrr": 0.45,
    **{f"pooled_top{k}": v for k, v in _DISEASE_TOP_K.items()},
    **{name: value for k, v in _DISEASE_TOP_K.items()
       for name, value in ((f"macro_recall@{k}", v), (f"macro_precision@{k}", v / k))},
    "macro_mrr": 0.45,
    "best_positive_mrr": 0.45,
    "conservative_pooled_mrr": 0.45,
}


def registered_report(metrics=None, n_ranked=80, **manifest_overrides):
    """A registered cohort version's measurement report, as the measurement
    writes it: the manifest and the result fields of `ModeResult.to_dict`.
    By default a disease task measurement's (`registered_manifest`,
    `DISEASE_METRICS`)."""
    return {
        "manifest": registered_manifest(**manifest_overrides),
        "authoritative_metrics": dict(metrics or DISEASE_METRICS),
        "n_ranked": n_ranked,
        "n_ground_truth_absent": 0,
        "sampler_evidence": {"candidate_columns": {"min": 4, "max": 4}},
    }


def generated_report():
    """A generated split's report: a valid manifest, and not a registrable one."""
    from src.evaluation.measurement import expected_relations

    report = registered_report()
    report["manifest"] = registered_manifest(
        split="val", cohort_kind="generated", cohort_version=None,
        relations=expected_relations(registered=False), r10=None,
        artifact_digests={"checkpoint": CHECKPOINT, "samples": "5" * 64,
                          "split_manifest": "6" * 64},
    )
    return report


#: A kept Mode C report's semantic values, as `scripts/measure_scorer.py --mode C`
#: wrote them (metric schema 1, `build_shepherd_model`, raw cosine).
MODE_C_SEMANTICS = {
    "mode": "C",
    "metric_schema_version": 1,
    "candidate_construction": "every disease in the knowledge graph",
    "batch_size": 32,
    "score_semantics": "raw cosine, no eta mixture and no shortest-path term",
    "model_construction": "production (build_shepherd_model)",
    "canonical_tie_policy_version": "score-desc-then-global-id-asc/v1",
}

#: A Mode C report's metrics (metric schema 1).
MODE_C_METRICS = {"untruncated_mrr": 0.5, "mean_rank": 3.0, "hits@1": 0.25, "hits@10": 0.75}


def mode_c_report(metrics=None, n_ranked=80, **manifest_overrides):
    """A Mode C report of a registered cohort version: Mode C's semantic values
    on the default manifest's other fields, and its metric schema 1 names.

    **The explicit builder for a test whose subject is a Mode C report or
    record** (its caveat label, its words, By test score, which orders Mode C
    groups only): every other test reads the disease default of
    `registered_report`, so retiring Mode C does not reach it by accident. K8-6e
    makes this report the subject of the generic refusal of a report or record
    this build does not read."""
    return registered_report(metrics or MODE_C_METRICS, n_ranked,
                             **{**MODE_C_SEMANTICS, **manifest_overrides})


#: A kept Mode D report's semantic values, as Batch 3's measurement wrote them
#: (its GB10 run of 2026-10-27, `docs/working/batch3-acceptance-2026-10-27/`).
#: Mode D no longer runs (K7), so the readers of its kept reports are tested on
#: reports built here.
MODE_D_SEMANTICS = {
    "mode": "D",
    "metric_schema_version": 2,
    "batch_size": 1,
    "candidate_construction": "served-path-discovery/v1:settings=5451e2f8486b",
    "score_semantics": "served:gnn_only:eta=1.0:sp=none",
    "canonical_tie_policy_version": "served-stable-discovery-order/v1",
}


def mode_d_report(n_ranked=56, n_eligible=80, **manifest_overrides):
    """A kept Mode D report of a registered cohort version, in the shape Batch
    3 kept: Mode D's semantic values on Mode C's manifest fields, its metric
    families over `n_eligible` import-eligible cases (`post_import_*`,
    `coverage`) and over the `n_ranked` reached ones (`reached_*`), and the
    blocks `build_record` ignores."""
    metrics = {"post_import_mrr": 0.35, "post_import_hits@1": 0.2,
               "post_import_hits@10": n_ranked / n_eligible * 0.9,
               "coverage": n_ranked / n_eligible, "reached_mrr": 0.5,
               "reached_mean_rank": 2.43, "reached_hits@1": 0.29, "reached_hits@10": 0.9}
    report = mode_c_report(metrics, n_ranked,
                           **{"n_samples": n_eligible, **MODE_D_SEMANTICS, **manifest_overrides})
    del report["n_ground_truth_absent"], report["sampler_evidence"]
    report.update(
        outcomes={"ranked": n_ranked, "not_reached": n_eligible - n_ranked, "refused": 0,
                  "no_candidates": 0, "n_import_eligible": n_eligible},
        import_counts={"ranked_eligible": n_eligible},
        served={"scoring_mode": "gnn_only"},
        comparison={"statement": "Mode C and Mode D are different ranking policies."},
    )
    return report


def _registered(report=None, digest=REPORT_A, ledger=None, run_id=RUN_1):
    """A ledger holding the record of `report`, registered once as run `run_id`."""
    record = build_record(report or registered_report(), digest)
    return append_record(ledger or empty_ledger(), record, run_id=run_id,
                         registered_at=WHEN)[1]


# ---------------------------------------------------------------------------
# What absence means, and the schema
# ---------------------------------------------------------------------------
def test_a_directory_with_no_ledger_reads_as_an_empty_one_with_no_digest(tmp_path):
    """The first record has to be appendable to a directory that has none, and
    `None` is the claim "I read no file" the write guard checks."""
    ledger, digest = read_ledger(tmp_path / "evaluations.json")

    assert ledger["records"] == []
    assert ledger["schema_version"] == LEDGER_SCHEMA_VERSION == 4
    assert "run_identity_from_sequence" not in ledger, "set by the first append"
    assert digest is None


def test_the_ledger_states_what_an_absent_record_does_and_does_not_mean():
    """This provenance system is not closed; an evaluation can happen outside it."""
    text = empty_ledger()["what_this_is"]

    assert "holds no result" in text
    assert "not that they were never evaluated" in text


def test_the_ledger_is_parsed_and_identified_from_one_read(tmp_path):
    """BACKLOG 18: the digest returned is of the bytes parsed, from one open."""
    from tests.fixtures.opens import count_opens

    path = tmp_path / "evaluations.json"
    write_ledger(path, _registered(), None)

    with count_opens() as opens:
        ledger, digest = read_ledger(path)

    assert opens.opens(path) == 1
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(ledger["records"]) == 1


def test_an_unknown_schema_version_is_refused_rather_than_reinterpreted(tmp_path):
    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps({"schema_version": 99, "records": []}))

    with pytest.raises(ValueError, match=f"{path}.*Read it with the revision that wrote it"):
        read_ledger(path)


@pytest.mark.parametrize("version", [1, 2])
def test_an_earlier_schema_is_refused_naming_the_file_and_what_to_do(tmp_path, version):
    """A schema 2 record is keyed by a samples digest and lacks the relation
    states, counts and registration; it is archived, not read beside schema 3.

    **What to do is what can be done** (finding 20): no report written before
    schema 3 can be registered under it, and this text reaches the model status
    and `--show`, where no current report is at hand -- so it says the results
    are not carried over and to re-measure, never "register again"."""
    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps({"schema_version": version, "records": []}))

    with pytest.raises(ValueError) as refused:
        read_ledger(path)

    message = str(refused.value)
    assert message == (
        f"{path} is ledger schema {version}; this code reads schemas 3 and 4. Archive it, "
        f"for example as evaluations.v{version}.json; it is read with the revision that "
        "wrote it. Its results are not carried over: re-measure with this revision and "
        "register the new reports.")
    assert "register again" not in message


def test_a_boolean_schema_version_is_not_read_as_one(tmp_path):
    """`True == 1` in Python; a ledger claiming `true` is not schema 1 or 3."""
    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps({"schema_version": True, "records": []}))

    with pytest.raises(ValueError, match="schema True"):
        read_ledger(path)


@pytest.mark.parametrize("version", [3, 4])
def test_a_ledger_without_a_records_list_is_refused(tmp_path, version):
    """Schema 3 is still read as a ledger, and judged as one (K8-6r.2)."""
    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps({"schema_version": version}))

    with pytest.raises(ValueError, match=f"{path} has no records list"):
        read_ledger(path)


@pytest.mark.parametrize(
    "data,fragment",
    [
        (b'{"schema_version": 3, "rec', "UTF-8 JSON"),
        (b"[]", "JSON list, not an object"),
        (b'{"schema_version": 3, "records": [], "x": "\xff"}', "UTF-8 JSON"),
    ],
    ids=["truncated", "a-list", "not-utf8"],
)
def test_a_damaged_ledger_is_refused_naming_the_file(tmp_path, data, fragment):
    """A ledger that does not parse, or parses into something else, is a
    `ValueError` naming the file: never an `AttributeError` or a traceback."""
    path = tmp_path / "evaluations.json"
    path.write_bytes(data)

    with pytest.raises(ValueError, match=fragment) as refused:
        read_ledger(path)
    assert str(path) in str(refused.value)


# ---------------------------------------------------------------------------
# The manifest's four M4 fields
# ---------------------------------------------------------------------------
def test_every_manifest_field_is_decided_one_way_or_the_other():
    """The claim "this list holds everything that can move a metric" is only worth
    making if it is checkable.

    Score-affecting fields were missing from the first version while the
    docstring claimed completeness. A field added to or removed from
    `MeasurementManifest` now fails here until someone decides
    which side it falls on, so omission has to be a decision rather than an
    oversight.
    """
    from dataclasses import fields

    from src.evaluation.measurement import MeasurementManifest

    declared = {f.name for f in fields(MeasurementManifest)}
    semantic, excluded = set(SEMANTIC_MANIFEST_FIELDS), set(NON_SEMANTIC_FIELDS)

    assert not declared - semantic - excluded, "undecided manifest field(s)"
    assert not semantic & excluded, "a field cannot be both"
    assert not (semantic | excluded) - declared, "names no manifest field has"
    assert all(NON_SEMANTIC_FIELDS.values()), "every exclusion states its reason"
    assert {"cohort_version", "relations", "r10", "environment"} <= excluded


def test_the_registered_relation_states_are_the_manifests_own():
    """The sidecar takes them from the manifest's module: R2 does not apply to a
    registered cohort, the cohort's graph binding does not apply to a generated
    split, and every other relation is verified."""
    from src.evaluation.measurement import (
        RELATION_NAMES,
        RELATION_STATES,
        expected_relations,
    )

    assert REGISTERED_RELATIONS == expected_relations(registered=True)
    assert tuple(REGISTERED_RELATIONS) == RELATION_NAMES
    assert set(REGISTERED_RELATIONS.values()) <= set(RELATION_STATES)
    assert expected_relations(registered=False) == {
        "R1": "verified", "R2": "verified", "R3": "verified", "R5": "verified",
        "cohort_graph": "not_applicable",
    }


#: Every state a registered run's relation does not earn, by relation.
_UNEARNED = [
    (name, state)
    for name, earned in REGISTERED_RELATIONS.items()
    for state in ("failed", "not_checked", "not_applicable", "verified")
    if state != earned
]


@pytest.mark.parametrize("name,state", _UNEARNED, ids=lambda v: str(v))
def test_a_manifest_carries_no_relation_state_its_run_did_not_earn(name, state):
    """Generated: R2 verified, cohort_graph not applicable. Registered: the
    reverse. R1, R3, R5 verified. Any other state is refused when the manifest
    is built, so no report can carry it."""
    from src.evaluation.measurement import expected_relations

    relations = expected_relations(registered=True)
    relations[name] = state

    with pytest.raises(ValueError, match="relation states"):
        registered_manifest(relations=relations)


def test_a_manifest_with_a_relation_missing_or_added_is_refused():
    from src.evaluation.measurement import expected_relations

    missing = expected_relations(registered=True)
    del missing["R5"]
    added = {**expected_relations(registered=True), "R10": "verified"}

    for relations in (missing, added):
        with pytest.raises(ValueError, match="relation states"):
            registered_manifest(relations=relations)


def test_a_supplied_cohort_without_a_registered_version_has_no_manifest():
    """Nothing binds its samples to the graph scored, so its report could not
    state that relation (M4 §0.8, §0.9)."""
    from src.evaluation.measurement import expected_relations

    with pytest.raises(ValueError, match="no registered cohort version"):
        registered_manifest(cohort_version=None, r10=None,
                            relations=expected_relations(registered=False))


def test_a_cohort_version_is_a_supplied_cohort():
    with pytest.raises(ValueError, match="is a supplied cohort"):
        registered_manifest(cohort_kind="generated")


@pytest.mark.parametrize("registered", [True, False])
def test_r10_is_recorded_for_a_registered_version_and_only_for_one(registered):
    from src.evaluation.measurement import expected_relations

    if registered:
        overrides = {"r10": None}
    else:
        overrides = {"split": "val", "cohort_kind": "generated", "cohort_version": None,
                     "relations": expected_relations(registered=False),
                     "r10": r10_block()}
    with pytest.raises(ValueError, match="R10 is recorded for a registered"):
        registered_manifest(**overrides)


# ---------------------------------------------------------------------------
# The key and the two semantics digests
# ---------------------------------------------------------------------------
def test_the_semantics_digest_is_unchanged_in_content():
    """M4 classifies its four fields as non-semantic and moves `software_revision`
    to `code_revision`, which names the same commit: the digest is the schema 2
    formula over the same 19 fields and 4 artifact roles."""
    manifest = registered_manifest()
    nineteen = (
        "mode", "cohort_kind", "candidate_construction", "batch_size", "score_semantics",
        "model_construction", "canonical_tie_policy_version", "metric_schema_version",
        "software_revision", "torch_version", "cuda_version", "device", "dtype",
        "amp_enabled", "amp_dtype", "torch_compile_wrapped", "deterministic_algorithms",
        "cudnn_deterministic", "cudnn_benchmark",
    )
    payload = {field: manifest.get(field) for field in nineteen}
    payload["artifacts"] = {
        role: manifest["artifact_digests"].get(role)
        for role in ("node_features", "edge_indices", "num_nodes", "split_manifest")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    assert SEMANTIC_MANIFEST_FIELDS == nineteen
    assert measurement_semantics_digest(manifest) == hashlib.sha256(
        encoded.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "field,value",
    [
        ("cohort_version", cohort_version(label="another-label")),
        ("r10", r10_block("overlap")),
        ("environment", environment_block(driver="999.0")),
    ],
)
def test_the_m4_fields_do_not_enter_the_semantics_digest(field, value):
    """Recorded, not keyed (spec §0.12): a re-run that found other R10 evidence,
    or ran under another driver, is the same measurement."""
    base = registered_manifest()

    assert measurement_semantics_digest(base) == measurement_semantics_digest(
        {**base, field: value})


def test_the_scoring_and_regime_fields_partition_the_semantic_list():
    """D4's seven scoring fields, and the twelve of the regime, cover the semantic
    list exactly once."""
    scoring, regime = set(SCORING_SEMANTICS_FIELDS), set(REGIME_FIELDS)

    assert scoring | regime == set(SEMANTIC_MANIFEST_FIELDS)
    assert not scoring & regime
    assert len(SCORING_SEMANTICS_FIELDS) == 7 and len(REGIME_FIELDS) == 12
    assert scoring == {"mode", "cohort_kind", "candidate_construction", "score_semantics",
                       "model_construction", "canonical_tie_policy_version",
                       "metric_schema_version"}


@pytest.mark.parametrize("field", SCORING_SEMANTICS_FIELDS)
def test_each_scoring_field_changes_the_scoring_digest(field):
    base = registered_manifest()

    assert scoring_semantics_digest(base) != scoring_semantics_digest(
        {**base, field: "something else"})


@pytest.mark.parametrize("field", REGIME_FIELDS)
def test_a_regime_field_changes_the_measurement_but_not_the_scoring(field):
    """Two runs at another batch size are two measurements of the same scoring:
    D4 groups them and shows the difference."""
    base = registered_manifest()
    other = {**base, field: "something else"}

    assert scoring_semantics_digest(base) == scoring_semantics_digest(other)
    assert measurement_semantics_digest(base) != measurement_semantics_digest(other)


@pytest.mark.parametrize(
    "differing",
    [
        {"mode": "D"},
        {"batch_size": 8},
        {"amp_enabled": True, "amp_dtype": "torch.bfloat16"},
        {"software_revision": "8" * 40},
        {"device": "cpu", "cuda_executed": False},
        {"metric_schema_version": 3},
        {"canonical_tie_policy_version": "v2"},
        {"candidate_construction": "path-reachable diseases"},
    ],
    ids=lambda d: "-".join(sorted(d)),
)
def test_runs_differing_in_anything_score_affecting_are_separate_records(differing):
    """Not one measurement with two answers -- two measurements.

    The earlier key was (checkpoint, cohort, mode, tie policy), and every case
    here except the first collided under it and was refused as a contradiction.
    `batch_size` is the sharpest: it is the measurement's chunk size (the cases
    ranked per chunk), which can move scores at floating-point level, so two
    honest runs at different sizes are two measurements.
    """
    second = build_record(registered_report(**differing), REPORT_B)

    outcome, ledger = append_record(_registered(), second, run_id=RUN_2, registered_at=WHEN)
    assert outcome == "registered"
    assert len(ledger["records"]) == 2


def test_a_changed_graph_artifact_is_a_separate_measurement():
    """The graph tensors are as much a part of a measurement as the weights."""
    digests = {**registered_manifest()["artifact_digests"], "node_features": "0" * 64}
    other = build_record(registered_report(artifact_digests=digests), REPORT_B)

    assert append_record(_registered(), other, run_id=RUN_2,
                         registered_at=WHEN)[0] == "registered"


def test_two_cohort_versions_with_identical_samples_are_two_cohorts():
    """The cohort digest is the version manifest's, which binds the source too
    (spec §0.11): the same samples from another source are another cohort."""
    other = build_record(
        registered_report(cohort_version=cohort_version(manifest_sha256="0" * 64,
                                                        source_sha256="1" * 64)),
        REPORT_B)

    assert append_record(_registered(), other, run_id=RUN_2,
                         registered_at=WHEN)[0] == "registered"


def test_a_record_missing_a_key_field_is_refused():
    with pytest.raises(ValueError, match="missing key field"):
        record_key({"checkpoint_digest": CHECKPOINT})


# ---------------------------------------------------------------------------
# What a record carries
# ---------------------------------------------------------------------------
def test_the_record_is_derived_from_the_manifest_not_asserted():
    """A run's own manifest says which checkpoint, cohort version and role it was."""
    report = registered_report()
    record = build_record(report, REPORT_A)
    manifest = report["manifest"]

    assert record["checkpoint_digest"] == CHECKPOINT
    assert record["cohort_role"] == "mygene2" == manifest["cohort_version"]["role"]
    assert record["cohort_digest"] == "a" * 64, "the version manifest's digest"
    assert record["measurement_semantics_digest"] == measurement_semantics_digest(manifest)
    assert record["scoring_semantics_digest"] == scoring_semantics_digest(manifest)
    assert record["source_artifact_digest"] == REPORT_A
    assert record["metrics"] == report["authoritative_metrics"]
    assert record["n_ranked"] == 80


def test_the_record_carries_what_the_view_and_d4_need():
    """Data source, cohort version, scoring semantics, regime, relation states,
    counts with their denominators, R10 and the environment (brief §3)."""
    record = build_record(registered_report(), REPORT_A)
    cohort = cohort_version()

    assert record["scoring"] == {f: registered_manifest()[f] for f in SCORING_SEMANTICS_FIELDS}
    assert set(record["regime"]) == set(REGIME_FIELDS)
    assert record["cohort"] == {
        "kind": "supplied", "label": cohort["label"], "role": cohort["role"],
        "samples_sha256": cohort["samples_sha256"],
        "source_sha256": cohort["source_sha256"], "source_format": "shepherd-cases/1",
        "graph": cohort["graph"], "mapping": cohort["mapping"], "counts": cohort["counts"],
    }
    assert record["cohort"]["counts"]["source_cases"] == 100
    assert record["cohort"]["counts"]["excluded"] == {
        "ground_truth_absent": 12, "phenotype_unmappable": 8}
    assert record["relations"] == REGISTERED_RELATIONS
    assert record["dataset_record"] == {"origin": "synthetic", "sha256": "d" * 64}
    assert record["r10"] == r10_block()
    assert record["environment"] == environment_block()


def test_a_completed_record_is_key_compared_and_excluded_fields_exactly():
    """The split is checkable: a field outside all three fails here, so none can
    escape the comparison by oversight. The run the ledger assigns (schema 4,
    K8-6r.2) is an excluded field: it tells runs apart and is never compared."""
    record = _registered()["records"][0]
    key, compared, excluded = set(KEY_FIELDS), set(RECORD_COMPARED_FIELDS), set(
        RECORD_EXCLUDED_FIELDS)

    assert set(record) == key | compared | excluded
    assert record["run_id"] == RUN_1 and "run_id" in excluded
    assert not key & compared and not key & excluded and not compared & excluded
    assert all(len(reason) > 20 for reason in RECORD_EXCLUDED_FIELDS.values())


def test_build_record_leaves_the_registration_to_the_ledger():
    """The run, like the registration, is the ledger's to assign: no manifest
    field is added, so no kept report becomes unreadable (K8-6r.2)."""
    record = build_record(registered_report(), REPORT_A)

    assert "corroborating_reports" not in record and "registration" not in record
    assert "run_id" not in record


def _strings(value, at=()):
    """Every string in a JSON value, with the keys and indices that lead to it."""
    if isinstance(value, str):
        yield at, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, (*at, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, (*at, index))


def _path_like(text: str) -> bool:
    """An absolute path on either platform, a home- or dot-relative path, or a
    Windows separator: what a run's location looks like in a string."""
    import re

    return (text.startswith(("/", "~", "./", "../")) or "\\" in text
            or re.match(r"^[A-Za-z]:[\\/]", text) is not None)


def test_the_record_keeps_no_run_paths_no_prose_and_no_case_ids():
    """Finding 28: what the sidecar's docstring says a record keeps, exactly.

    - **No run paths:** the manifest's `checkpoint_path` and `data_dir` stay in
      the report.
    - **R10's evidence paths are kept** (the read-only view strips them): every
      string in the record is walked, and a path-like one is allowed only at
      `r10.<scope>.evidence.path`, so a path copied anywhere else fails here.
    - **No prose:** neither the manifest's explanation nor any standing text.
    - **No case ids.**
    """
    from src.evaluation import caveats

    report = registered_report(r10=r10_block(training_state="overlap"))
    record = build_record(report, REPORT_A)
    text = json.dumps(record)

    paths = {at: value for at, value in _strings(record) if _path_like(value)}
    assert paths == {("r10", "training", "evidence", "path"): "/abs/train_samples.json"}
    assert all(len(at) == 4 and at[0] == "r10" and at[2:] == ("evidence", "path")
               for at in paths)
    assert report["manifest"]["checkpoint_path"] not in text
    assert report["manifest"]["data_dir"] not in text
    assert "manifest" not in record
    assert "why_this_is_here" not in text
    standing = [caveats.DISEASE_TASK_RESULT_LABEL, caveats.DISEASE_TASK_NOTE,
                caveats.MODE_C_RESULT_LABEL, caveats.PRE_B1_NOTE, caveats.RECORD_ABSENCE,
                *caveats.R10_STATE_TEXT.values(), *caveats.R10_UNVERIFIABLE_REASON.values(),
                *(label["definition"] for labels in caveats.R10_LABEL_TEXT.values()
                  for label in labels.values())]
    assert not [words for words in standing if len(words) > 20 and words in text]
    assert "case_id" not in text and "exclusions" not in text


def test_a_cohort_version_naming_a_field_nobody_classified_is_refused():
    """Aggregates only: a case list cannot ride into a record on an unclassified
    field."""
    block = cohort_version(exclusions=[{"case_id": "c17"}])

    with pytest.raises(ValueError, match="names exclusions"):
        build_record(registered_report(cohort_version=block), REPORT_A)


def test_build_record_neither_changes_nor_aliases_the_report():
    report = registered_report()
    before = copy.deepcopy(report)
    record = build_record(report, REPORT_A)
    record["cohort"]["counts"]["source_cases"] = -1
    record["r10"]["training"]["evidence"]["state"] = "edited"
    record["metrics"]["pooled_top1"] = -1.0

    assert report == before


# ---------------------------------------------------------------------------
# What build_record refuses
# ---------------------------------------------------------------------------
def test_a_generated_splits_report_is_not_registered():
    """The ledger records results on registered test cohort versions (D3), and
    does not yet support a generated split's data identity: the refusal says
    so, and no longer calls the result selection data, a lesser kind of result
    (K8-6r.4, policy rev 16). The refusal itself is unchanged."""
    with pytest.raises(ValueError, match="generated split's report is not registered") as refused:
        build_record(generated_report(), REPORT_A)
    message = str(refused.value)
    assert ("the ledger does not yet support a generated split's data identity (it has no "
            "registered cohort version), not because such a result counts for less") in message
    assert "its report and ranks stay at --output" in message
    assert "selection data" not in message


@pytest.mark.parametrize("field", ["python_seed", "legacy_truncation_k", "num_workers"])
def test_a_manifest_carrying_a_field_this_revision_does_not_record_is_refused(field):
    """A report written before the Mode A/B retirement's R2, or by a retired mode,
    carries fields the current manifest does not. Hashed under the current list
    it would lose what told its runs apart, so it is refused rather than read."""
    report = registered_report()
    report["manifest"][field] = 0

    with pytest.raises(ValueError, match=f"records {field}.*not read"):
        build_record(report, REPORT_A)


@pytest.mark.parametrize("field", ["cohort_version", "relations", "r10", "environment"])
def test_a_report_from_before_m4_is_refused_naming_what_it_lacks(field):
    """A schema 2 era report lacks the M4 fields; it is re-measured, not read."""
    report = registered_report()
    del report["manifest"][field]

    with pytest.raises(ValueError, match=f"lacks {field}.*Re-measure"):
        build_record(report, REPORT_A)


@pytest.mark.parametrize("name", ["R1", "R2", "R3", "R5", "cohort_graph"])
def test_a_report_whose_relation_states_were_edited_is_refused(name):
    """The measurement cannot write one, so such a file was edited or is not a
    report."""
    report = registered_report()
    relations = report["manifest"]["relations"]
    relations[name] = "verified" if relations[name] == "not_applicable" else "failed"

    with pytest.raises(ValueError, match=f"relation states.*{name}"):
        build_record(report, REPORT_A)


@pytest.mark.parametrize(
    "edit,fragment",
    [
        (lambda r: r.__setitem__("manifest", []), "no manifest object"),
        (lambda r: r.pop("manifest"), "no manifest object"),
        (lambda r: r.pop("authoritative_metrics"), "authoritative_metrics is not an object"),
        (lambda r: r.__setitem__("n_ranked", "80"), "n_ranked"),
        (lambda r: r.__setitem__("n_ranked", True), "n_ranked"),
        (lambda r: r["manifest"].__setitem__("cohort_kind", "generated"),
         "cohort_kind 'generated' beside a cohort version"),
        (lambda r: r["manifest"].__setitem__("split", "other-role"), "contradicts itself"),
        (lambda r: r["manifest"]["cohort_version"].pop("counts"), "lacks counts"),
        (lambda r: r["manifest"]["cohort_version"].__setitem__("graph", None),
         "cohort_version.graph is not an object"),
        (lambda r: r["manifest"]["cohort_version"].__setitem__("label", ""),
         "cohort_version.label"),
        (lambda r: r["manifest"]["artifact_digests"].pop("checkpoint"), "no checkpoint"),
        (lambda r: r["manifest"]["dataset_record"].pop("origin"), "dataset_record.origin"),
        (lambda r: r["manifest"].__setitem__("r10", None), "r10 block is not an object"),
        (lambda r: r["manifest"].__setitem__("environment", []),
         "environment is not an object"),
    ],
    ids=["manifest-a-list", "no-manifest", "no-metrics", "n-ranked-a-string",
         "n-ranked-a-bool", "generated-kind", "role-mismatch", "cohort-lacks-counts",
         "graph-not-object", "empty-label", "no-checkpoint", "no-origin", "no-r10",
         "environment-a-list"],
)
def test_a_report_of_the_wrong_shape_is_a_value_error_never_a_key_error(edit, fragment):
    report = registered_report()
    edit(report)

    with pytest.raises(ValueError, match=fragment):
        build_record(report, REPORT_A)


@pytest.mark.parametrize("report", [[], "text", 3, None], ids=repr)
def test_a_json_value_that_is_not_an_object_is_not_a_report(report):
    with pytest.raises(ValueError, match="not a measurement report"):
        build_record(report, REPORT_A)


@pytest.mark.parametrize("digest", [None, "", "abc", "E" * 64])
def test_the_report_digest_is_required(digest):
    """The record names the report by the digest of the bytes it was parsed from;
    there is no record without one."""
    with pytest.raises(ValueError, match="not a SHA-256"):
        build_record(registered_report(), digest)


# ---------------------------------------------------------------------------
# Registration by run (K8-6r.2)
# ---------------------------------------------------------------------------
def test_a_first_record_is_registered_with_sequence_one():
    """Its run named, and the new file at schema 4 with its run-identity
    boundary at that first record (A12 (c), at the writer's level)."""
    outcome, ledger = append_record(
        empty_ledger(), build_record(registered_report(), REPORT_A), run_id=RUN_1,
        registered_at=WHEN)
    record = ledger["records"][0]

    assert outcome == "registered"
    assert record["registration"] == {"sequence": 1, "registered_at": WHEN}
    assert record["corroborating_reports"] == []
    assert record["run_id"] == RUN_1
    assert ledger["schema_version"] == 4 and ledger["run_identity_from_sequence"] == 1


def test_the_sequence_continues_from_the_largest_in_the_file():
    """A gap left by a hand edit is not refilled: a sequence is never repeated."""
    ledger = _registered()
    ledger["records"][0]["registration"]["sequence"] = 5
    ledger["run_identity_from_sequence"] = 5
    second = build_record(registered_report(batch_size=8), REPORT_B)

    _, updated = append_record(ledger, second, run_id=RUN_2, registered_at=WHEN)
    assert updated["records"][1]["registration"]["sequence"] == 6


def test_the_same_report_again_is_a_no_op():
    """The same run's same report: a ledger rebuilt from the same reports twice
    is the same ledger."""
    ledger = _registered()
    again = build_record(registered_report(), REPORT_A)

    outcome, same = append_record(ledger, again, run_id=RUN_1,
                                  registered_at="2027-01-01T00:00:00Z")
    assert outcome == "already_registered"
    assert same is ledger


def test_differing_fields_is_the_compared_list_in_order():
    """The one comparison, shared by the report store's description of a group
    and the registration's group note (K8-6r.2). Only compared fields count; a
    field one side lacks differs; R10, the environment and the run never do."""
    from src.evaluation.sidecar import difference_names, differing_fields

    first = build_record(registered_report(), REPORT_A)
    other = build_record(registered_report(
        n_ranked=79, metrics={**DISEASE_METRICS, "pooled_mrr": 0.9},
        r10=r10_block(training_state="overlap"), environment=environment_block("999.0")),
        REPORT_B)

    assert differing_fields(first, first) == []
    assert differing_fields(first, other) == ["metrics", "n_ranked"]
    assert difference_names(first, other, ["metrics"]) == ["pooled_mrr"]
    assert difference_names(first, other, ["metrics", "n_ranked"]) == ["metrics", "n_ranked"]
    lacking = {k: v for k, v in first.items() if k != "dataset_record"}
    assert differing_fields(first, lacking) == ["dataset_record"]


def test_the_store_description_and_the_group_note_share_one_comparison(monkeypatch, tmp_path):
    """Replacing `differing_fields` changes what the store calls a group whose
    runs differ and what the registration's group note names alike: neither
    keeps a comparison of its own, and neither refuses (K8-6r.2)."""
    import src.evaluation.sidecar as sidecar
    from src.evaluation.report_store import scan_report_store
    from src.evaluation.sidecar import group_differences, held_record
    from tests.fixtures.report_store import store_report, version_of

    path = store_report(tmp_path / "registry", registered_report())
    # The same result again, from other report bytes (another workspace path).
    store_report(tmp_path / "registry", registered_report(data_dir="/elsewhere"))
    monkeypatch.setattr(sidecar, "differing_fields", lambda a, b: ["n_ranked"])

    second = build_record(registered_report(), REPORT_B)
    outcome, ledger = append_record(_registered(), second, run_id=RUN_2, registered_at=WHEN)
    assert outcome == "registered"
    (other,) = group_differences(ledger, held_record(ledger, second, RUN_2))
    assert (other.sequence, other.run_id, other.differing) == (1, RUN_1, ("n_ranked",))
    scan = scan_report_store(version_of(path))
    (key,) = scan.reports
    assert scan.state(key) == "contested"


def test_report_key_is_the_records_key_and_none_without_its_parts():
    """For a report `build_record` accepts, the lenient key is its record's key;
    for one it refuses, the key is still read where the manifest names it, and
    `None` where it does not."""
    from src.evaluation.sidecar import report_key

    report = registered_report()
    assert report_key(report) == record_key(build_record(report, REPORT_A))
    old = registered_report()
    old["manifest"]["python_seed"] = 0
    assert report_key(old) is not None
    for broken in ({}, [], {"manifest": []},
                   {"manifest": {**registered_manifest(), "cohort_version": None}},
                   {"manifest": {**registered_manifest(), "artifact_digests": {}}}):
        assert report_key(broken) is None, broken


def test_one_group_with_two_results_keeps_both_and_names_the_metrics():
    """Identical semantics, different numbers, two runs: both kept, each its
    own record, the difference described (A1 at the writer's level). Rev 15's
    contradiction refusal is withdrawn (policy rev 16)."""
    from src.evaluation.sidecar import group_differences, held_record

    ledger = _registered()
    before = copy.deepcopy(ledger)
    other = build_record(
        registered_report(metrics={**DISEASE_METRICS, "pooled_mrr": 0.9}), REPORT_B)

    outcome, updated = append_record(ledger, other, run_id=RUN_2, registered_at=WHEN)

    assert outcome == "registered" and ledger == before
    assert [r["run_id"] for r in updated["records"]] == [RUN_1, RUN_2]
    (first,) = group_differences(updated, held_record(updated, other, RUN_2))
    assert (first.sequence, first.run_id, first.differing) == (1, RUN_1, ("pooled_mrr",))


@pytest.mark.parametrize(
    "report,named",
    [
        (lambda: registered_report(n_ranked=79), "n_ranked"),
        (lambda: registered_report(
            cohort_version=cohort_version(counts={**cohort_version()["counts"],
                                                  "source_cases": 101})), "cohort"),
        (lambda: registered_report(dataset_record={
            **registered_manifest()["dataset_record"], "origin": "files"}),
         "dataset_record"),
    ],
    ids=["n-ranked", "cohort-counts", "dataset-origin"],
)
def test_a_difference_outside_the_metrics_is_registered_and_named(report, named):
    from src.evaluation.sidecar import group_differences, held_record

    record = build_record(report(), REPORT_B)
    outcome, updated = append_record(_registered(), record, run_id=RUN_2, registered_at=WHEN)

    assert outcome == "registered"
    (first,) = group_differences(updated, held_record(updated, record, RUN_2))
    assert first.differing == (named,)


def test_append_never_changes_what_it_was_given():
    ledger, record = _registered(), build_record(registered_report(batch_size=8), REPORT_B)
    ledger_before, record_before = copy.deepcopy(ledger), copy.deepcopy(record)

    append_record(ledger, record, run_id=RUN_2, registered_at=WHEN)
    assert ledger == ledger_before and record == record_before


@pytest.mark.parametrize("field,value", [
    ("registration", {"sequence": 1, "registered_at": WHEN}),
    ("corroborating_reports", []),
    ("run_id", RUN_1),
])
def test_a_record_carrying_what_the_ledger_assigns_is_refused(field, value):
    record = {**build_record(registered_report(), REPORT_A), field: value}

    with pytest.raises(ValueError, match=f"carries {field}, which only the ledger assigns"):
        append_record(empty_ledger(), record, run_id=RUN_1, registered_at=WHEN)


@pytest.mark.parametrize("run_id", ["", "run-1", RUN_1 + "x", None, 7])
def test_a_registration_needs_a_run_directorys_name(run_id):
    with pytest.raises(ValueError, match="is not a run directory's name"):
        append_record(empty_ledger(), build_record(registered_report(), REPORT_A),
                      run_id=run_id, registered_at=WHEN)


@pytest.mark.parametrize(
    "damage,fragment",
    [
        (lambda r: r.pop("registration"), "record 0 has no registration sequence"),
        (lambda r: r["registration"].__setitem__("sequence", "1"),
         "no registration sequence"),
    ],
    ids=["no-registration", "sequence-a-string"],
)
def test_a_hand_edit_that_broke_a_sequence_refuses_the_append(damage, fragment):
    """Whatever run the new record is: the next sequence cannot be allocated
    safely past a broken one (A8, kept: what makes the write itself
    uncertain)."""
    ledger = _registered()
    damage(ledger["records"][0])
    before = copy.deepcopy(ledger)

    with pytest.raises(ValueError, match=fragment):
        append_record(ledger, build_record(registered_report(batch_size=8), REPORT_B),
                      run_id=RUN_2, registered_at=WHEN)
    assert ledger == before


@pytest.mark.parametrize("damage", ["not-an-object", "no-sequence"])
def test_a_damaged_record_elsewhere_refuses_a_write_never_a_found_run(damage):
    """Step 0 guards a write (review of K8-6r.2): with another record damaged,
    the same run's same report, found intact, answers ``already_registered``
    with nothing written, and its other bytes are refused as a replaced report,
    as `find_run` (so `report_state`) reads them; a run with no record is still
    refused, since its sequence cannot be allocated safely. Before the fix the
    repeat was refused with the damaged record's reason."""
    ledger = _registered(registered_report(batch_size=8), REPORT_B,
                         _registered(), RUN_2)
    if damage == "not-an-object":
        ledger["records"].insert(0, "junk")
    else:
        ledger["records"][1].pop("registration")  # run 2's record
    before = copy.deepcopy(ledger)

    outcome, kept = append_record(ledger, build_record(registered_report(), REPORT_A),
                                  run_id=RUN_1, registered_at=WHEN)
    assert (outcome, kept) == ("already_registered", before)
    with pytest.raises(ValueError, match="is already registered as record 1"):
        append_record(ledger, build_record(registered_report(), REPORT_C), run_id=RUN_1,
                      registered_at=WHEN)
    with pytest.raises(ValueError, match="cannot be allocated safely"):
        append_record(ledger, build_record(registered_report(), REPORT_C), run_id=RUN_3,
                      registered_at=WHEN)
    assert ledger == before


def test_this_runs_record_without_its_sequence_is_damaged_identity_never_registered():
    """The lookup runs first (review of K8-6r.2), so this run's own record with
    its registration lost by a hand edit must not answer ``already_registered``
    (it would name no record) nor let a new record in: it is this run's damaged
    identity record (clarification 1), refused with its index and the remedy,
    for the same bytes and for other bytes alike."""
    ledger = _registered()
    ledger["records"][0].pop("registration")
    before = copy.deepcopy(ledger)
    for report in (REPORT_A, REPORT_C):
        with pytest.raises(ValueError, match=r"record 0 names run .* registration sequence "
                                             r"cannot be read.*trusted backup"):
            append_record(ledger, build_record(registered_report(), report), run_id=RUN_1,
                          registered_at=WHEN)
    assert ledger == before


def test_another_runs_record_without_a_group_field_never_refuses_the_append():
    """K8-6r.2: a lost group field on another run's record no longer refuses
    every append (HEAD refused "record 0: record is missing key field"). The
    record is kept as written; the readers list it with its reason."""
    ledger = _registered()
    ledger["records"][0].pop("cohort_digest")
    kept = copy.deepcopy(ledger["records"][0])

    outcome, updated = append_record(
        ledger, build_record(registered_report(batch_size=8), REPORT_B), run_id=RUN_2,
        registered_at=WHEN)

    assert outcome == "registered"
    assert updated["records"][0] == kept and updated["records"][1]["run_id"] == RUN_2


def test_a_registration_needs_its_time():
    with pytest.raises(ValueError, match="time it was made"):
        append_record(empty_ledger(), build_record(registered_report(), REPORT_A),
                      run_id=RUN_1, registered_at="")


# ---------------------------------------------------------------------------
# Lookup and the environment class
# ---------------------------------------------------------------------------
def test_records_for_returns_only_that_checkpoints_results():
    ledger = _registered()
    digests = {**registered_manifest()["artifact_digests"], "checkpoint": "0" * 64}
    ledger = append_record(ledger, build_record(
        registered_report(artifact_digests=digests), REPORT_B),
        run_id="20261016T120001Z-000000000000-00000002", registered_at=WHEN)[1]

    assert len(records_for(ledger, CHECKPOINT)) == 1
    assert records_for(ledger, "f" * 64) == []


def test_records_for_tolerates_hand_edited_records():
    """Correction is by editing the file; a record that lost its digest is not
    this checkpoint's, and the lookup does not raise."""
    ledger = _registered()
    ledger["records"] += [{"metrics": {}}, "not a record", None]

    assert len(records_for(ledger, CHECKPOINT)) == 1
    assert records_for({"records": "not a list"}, CHECKPOINT) == []


@pytest.mark.parametrize(
    "device,deterministic,expected",
    [
        ("cuda", True, ENVIRONMENT_CLASSES[0]),
        ("cuda:0", False, ENVIRONMENT_CLASSES[1]),
        ("cpu", False, ENVIRONMENT_CLASSES[2]),
    ],
)
def test_the_environment_class_of_a_record_and_of_a_regime(device, deterministic, expected):
    record = build_record(
        registered_report(device=device, deterministic_algorithms=deterministic), REPORT_A)

    assert environment_class(record) == expected
    assert environment_class(record["regime"]) == expected


def test_the_environment_classes_are_the_specified_phrases():
    assert ENVIRONMENT_CLASSES == (
        "CUDA, deterministic algorithms on",
        "CUDA, deterministic algorithms off",
        "CPU (development; a different measurement from a CUDA run)",
    )


@pytest.mark.parametrize(
    "regime", [{"device": "mps"}, {}, {"device": "cuda"},
               {"device": "cuda", "deterministic_algorithms": "yes"}, []],
    ids=["mps", "no-device", "no-flag", "flag-a-string", "a-list"],
)
def test_an_environment_with_no_class_is_refused_not_guessed(regime):
    with pytest.raises(ValueError):
        environment_class(regime)


# ---------------------------------------------------------------------------
# Writing, and the stale-write guard
# ---------------------------------------------------------------------------
def test_a_written_ledger_reads_back_identically(tmp_path):
    path = tmp_path / "checkpoints" / "evaluations.json"
    ledger = _registered()
    write_ledger(path, ledger, None)

    assert read_ledger(path) == (ledger, hashlib.sha256(path.read_bytes()).hexdigest())


def test_a_failed_write_leaves_the_previous_ledger_intact(tmp_path, monkeypatch):
    """The records an interrupted append would destroy are not recoverable from
    the checkpoints."""
    import src.evaluation.sidecar as sidecar

    path = tmp_path / "evaluations.json"
    first = _registered()
    write_ledger(path, first, None)
    before = path.read_bytes()

    def _explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sidecar.json, "dump", _explode)
    with pytest.raises(OSError):
        write_ledger(path, _registered(registered_report(batch_size=8), REPORT_B, first, RUN_2),
                     read_ledger(path)[1])

    assert path.read_bytes() == before
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")], (
        "the temporary file must not be left behind"
    )


def test_a_writer_holding_a_stale_read_is_refused(tmp_path):
    """Stale-write detection, the second line behind the ledger's lock.

    **What this proves:** a writer that read the ledger, had another writer
    replace it, and then writes, is refused instead of erasing that append.
    That is a writer that takes no lock (a hand edit, an older build).

    **What it does not prove:** safety under overlapping writers on its own.
    Two writers whose check-then-replace windows interleave both pass the
    comparison. Registrations exclude each other through `ledger_lock` (K8-6r.1,
    pinned below and in tests/unit/test_registration.py); this check is what
    stays for a writer that takes no lock.
    """
    path = tmp_path / "evaluations.json"
    write_ledger(path, _registered(), None)
    first, stale = read_ledger(path)

    other = _registered(registered_report(batch_size=8), REPORT_B, first, RUN_2)
    write_ledger(path, other, stale)

    with pytest.raises(ValueError, match="changed since it was read"):
        write_ledger(path, first, stale)
    assert len(read_ledger(path)[0]["records"]) == 2


def test_the_ledger_lock_excludes_a_second_holder_and_is_kept(tmp_path, monkeypatch):
    """`ledger_lock` (K8-6r.1): ``evaluations.json.lock`` beside the ledger.
    Two opens of it exclude each other (in one process too), a second taker
    is refused after the grace (`FileLockBusy`, naming the lock file), the lock
    is released when the block ends normally or by an exception, and the lock
    file is kept."""
    import src.evaluation.sidecar as sidecar
    from src.evaluation.sidecar import ledger_lock, ledger_lock_path
    from src.utils.file_lock import FileLockBusy

    monkeypatch.setattr(sidecar, "LEDGER_LOCK_GRACE_SECONDS", 0.1)
    ledger = tmp_path / "evaluations.json"
    lock = ledger_lock_path(ledger)
    assert lock == tmp_path / "evaluations.json.lock"

    with ledger_lock(ledger):
        with pytest.raises(FileLockBusy) as busy:
            with ledger_lock(ledger):
                pytest.fail("a second holder took the ledger's lock")
        assert str(busy.value) == (f"{lock} is busy: another writer held this lock for the "
                                   "0.1 s this one waited")
    with ledger_lock(ledger):
        pass
    with pytest.raises(RuntimeError, match="inside"):
        with ledger_lock(ledger):
            raise RuntimeError("inside")
    with ledger_lock(ledger):
        assert lock.is_file()
    assert lock.is_file() and not ledger.exists(), "the lock file is kept; no ledger made"


def test_a_writer_expecting_no_file_is_refused_when_one_appeared(tmp_path):
    """`None` is a claim -- *I read no file* -- not a request to skip the check.
    A ledger created by another writer in the meantime is that writer's append."""
    path = tmp_path / "evaluations.json"
    write_ledger(path, _registered(), None)

    with pytest.raises(ValueError, match="expected no file to exist"):
        write_ledger(path, empty_ledger(), None)


def test_the_expected_digest_cannot_be_skipped_by_omission():
    """Required rather than defaulted: a caller that forgets it would otherwise
    silently lose the detection."""
    import inspect

    parameter = inspect.signature(write_ledger).parameters["expected_digest"]
    assert parameter.default is inspect.Parameter.empty


def test_the_write_guard_is_a_path_hash_of_the_file_now(tmp_path):
    """Stated, not hidden: the guard compares the file by path at write time
    with the digest of what was read. It is not compare-and-swap."""
    path = tmp_path / "evaluations.json"
    write_ledger(path, _registered(), None)

    assert ledger_digest(path) == read_ledger(path)[1]
    assert ledger_digest(tmp_path / "absent.json") is None


def test_checkpoint_matches_compares_one_file_by_its_bytes_whatever_its_name(tmp_path):
    """The one comparison `find_checkpoint` and `record_evaluation --checkpoint`
    share: any file name, a link by its target's bytes, nothing that is not a
    regular file."""
    weights = tmp_path / "best.ckpt"
    weights.write_bytes(b"the weights")
    link = tmp_path / "served.pt"
    link.symlink_to(weights)
    digest = hashlib.sha256(b"the weights").hexdigest()

    assert checkpoint_matches(weights, digest)
    assert checkpoint_matches(link, digest)
    assert not checkpoint_matches(weights, "0" * 64)
    assert not checkpoint_matches(tmp_path / "absent.pt", digest)
    assert not checkpoint_matches(tmp_path, digest)


def test_find_checkpoint_compares_through_checkpoint_matches(tmp_path, monkeypatch):
    import src.evaluation.sidecar as sidecar

    (tmp_path / "a.pt").write_bytes(b"x")
    (tmp_path / "b.pt").write_bytes(b"y")
    seen = []
    monkeypatch.setattr(sidecar, "checkpoint_matches",
                        lambda path, digest: seen.append(path.name) or path.name == "b.pt")

    assert find_checkpoint(tmp_path, "0" * 64) == tmp_path / "b.pt"
    assert seen == ["a.pt", "b.pt"]


def test_find_checkpoint_identifies_the_weights_by_their_bytes(tmp_path):
    """The digest is the identity, not the name; nothing else matches."""
    (tmp_path / "a.pt").write_bytes(b"other weights")
    (tmp_path / "best.pt").write_bytes(b"the weights")
    digest = hashlib.sha256(b"the weights").hexdigest()

    assert find_checkpoint(tmp_path, digest) == tmp_path / "best.pt"
    assert find_checkpoint(tmp_path, "0" * 64) is None
    assert find_checkpoint(tmp_path / "missing", digest) is None


def test_the_ledger_writes_ascii_so_its_bytes_parse_as_utf8(tmp_path):
    """The writer keeps `ensure_ascii`; a label outside ASCII round-trips through
    the one read, which decodes UTF-8."""
    path = tmp_path / "evaluations.json"
    record = build_record(
        registered_report(cohort_version=cohort_version(label="測試-2026")), REPORT_A)
    write_ledger(path, append_record(empty_ledger(), record, run_id=RUN_1,
                                     registered_at=WHEN)[1], None)

    assert path.read_bytes().isascii()
    assert read_ledger(path)[0]["records"][0]["cohort"]["label"] == "測試-2026"


# ---------------------------------------------------------------------------
# Run identity (K8-6r.2, policy rev 16): the writer's acceptance rows
# ---------------------------------------------------------------------------
REPORT_C = "9" * 64


def _historical_ledger(registered_at=WHEN, corroborating=(REPORT_B,)):
    """A schema 3 ledger as the writer before K8-6r.2 left it: record 1 names no
    run, holds source `REPORT_A` and corroborating `REPORT_B` (D1 and D2)."""
    old = build_record(registered_report(), REPORT_A)
    old["corroborating_reports"] = list(corroborating)
    old["registration"] = {"sequence": 1, "registered_at": registered_at}
    return {"schema_version": 3, "what_this_is": empty_ledger()["what_this_is"],
            "records": [old]}


def test_the_run_id_pattern_is_the_registrys():
    """Restated in the ledger module (standard library only), pinned equal."""
    from src.evaluation.cohort_registry import RUN_ID_PATTERN as REGISTRYS
    from src.evaluation.sidecar import RUN_ID_PATTERN

    assert RUN_ID_PATTERN.pattern == REGISTRYS.pattern


def test_byte_identical_reports_of_two_runs_are_two_records():
    """A3 at the writer's level: equal bytes do not make two runs one. HEAD
    answered ``already_registered`` for the second run by its digest."""
    ledger = _registered()
    again = build_record(registered_report(), REPORT_A)

    outcome, updated = append_record(ledger, again, run_id=RUN_2, registered_at=WHEN)

    assert outcome == "registered"
    first, second = updated["records"]
    assert first["source_artifact_digest"] == second["source_artifact_digest"] == REPORT_A
    assert (first["run_id"], second["run_id"]) == (RUN_1, RUN_2)
    assert second["registration"]["sequence"] == 2


def test_equal_values_under_another_driver_are_two_records_each_with_its_environment():
    """A2 at the writer's level: HEAD merged the second report into the first
    record (merged as a corroborating report), which kept the first run's environment."""
    environment = environment_block(driver="999.0")
    other = build_record(registered_report(environment=environment, r10=r10_block("overlap")),
                         REPORT_B)

    outcome, updated = append_record(_registered(), other, run_id=RUN_2, registered_at=WHEN)

    assert outcome == "registered"
    first, second = updated["records"]
    assert first["environment"] == environment_block() and second["environment"] == environment
    assert first["r10"] == r10_block() and second["r10"] == r10_block("overlap")
    assert first["corroborating_reports"] == second["corroborating_reports"] == []


@pytest.mark.parametrize("overrides", [
    {"data_dir": "/elsewhere"},
    {"batch_size": 16},
], ids=["equal-values", "another-group"])
def test_a_sealed_runs_report_replaced_by_other_bytes_is_refused(overrides):
    """A5 at the writer's level: run 1's report replaced after its registration,
    with equal values or another group, is refused naming record 1 and both
    digests; the ledger is unchanged. HEAD: merged into record 1, or a new record."""
    ledger = _registered()
    before = copy.deepcopy(ledger)
    replaced = build_record(registered_report(**overrides), REPORT_B)

    with pytest.raises(ValueError) as refused:
        append_record(ledger, replaced, run_id=RUN_1, registered_at=WHEN)

    assert str(refused.value) == (
        f"run {RUN_1} is already registered as record 1 from report {REPORT_A[:12]}...; this "
        f"report ({REPORT_B[:12]}...) is other bytes: a run's report is sealed when it is "
        "published, so this file was replaced after it was registered")
    assert ledger == before


def test_deterministic_runs_whose_values_differ_are_both_kept_and_flagged():
    """A6 at the writer's level: deterministic algorithms on in both runs, the
    values differ: both kept, the difference flagged for an investigation."""
    from src.evaluation.sidecar import group_differences, held_record

    other = build_record(registered_report(
        metrics={**DISEASE_METRICS, "pooled_mrr": 0.4501}), REPORT_B)
    assert other["regime"]["deterministic_algorithms"] is True

    outcome, updated = append_record(_registered(), other, run_id=RUN_2, registered_at=WHEN)

    assert outcome == "registered" and len(updated["records"]) == 2
    (first,) = group_differences(updated, held_record(updated, other, RUN_2))
    assert first.deterministic is True and first.differing == ("pooled_mrr",)
    off = build_record(registered_report(deterministic_algorithms=False,
                                         metrics={**DISEASE_METRICS, "pooled_mrr": 0.5}),
                       REPORT_C)
    _, two = append_record(_registered(registered_report(deterministic_algorithms=False)), off,
                           run_id=RUN_2, registered_at=WHEN)
    (first,) = group_differences(two, held_record(two, off, RUN_2))
    assert first.deterministic is False


@pytest.mark.parametrize("first_registered_at", [
    "2026-10-15T08:00:00Z", WHEN, "2026-10-17T00:00:00Z", "yesterday", "2099-01-01T00:00:00Z",
    None,
], ids=["earlier", "same-second", "later", "unreadable", "future", "absent"])
@pytest.mark.parametrize("digest", [REPORT_A, REPORT_B], ids=["D1-source", "D2-corroborating"])
def test_a_run_whose_bytes_an_old_merged_record_holds_gets_its_own_record(
    first_registered_at, digest
):
    """A7 (a) and (b) at the writer's level (Decision 1): record 1 names no run
    and holds D1 with corroborating [D2]; a run sending D1, or D2, becomes a
    record of its own, whatever the times; its historical note names record 1
    as not confirmed; the run resent answers ``already_registered``; record 1 is
    unchanged. No time is compared. HEAD: ``already_registered`` by digest."""
    from src.evaluation.sidecar import held_record, historical_overlap

    ledger = _historical_ledger()
    if first_registered_at is None:
        del ledger["records"][0]["registration"]["registered_at"]
    else:
        ledger["records"][0]["registration"]["registered_at"] = first_registered_at
    old = copy.deepcopy(ledger["records"][0])
    record = build_record(registered_report(), digest)

    outcome, updated = append_record(ledger, record, run_id=RUN_2, registered_at=WHEN)

    assert outcome == "registered"
    assert updated["records"][0] == old, "record 1 stays as written, with no run"
    assert "run_id" not in updated["records"][0]
    held = held_record(updated, record, RUN_2)
    assert held is updated["records"][1] and held["source_artifact_digest"] == digest
    (note,) = historical_overlap(updated, held)
    assert (note.sequence, note.possible_overlap) == (1, True)
    assert note.text == ("possible overlap with record 1, merged before each run had its own "
                         "record: not recorded whether this run is one of them")
    assert append_record(updated, record, run_id=RUN_2, registered_at=WHEN) == (
        "already_registered", updated)


@pytest.mark.parametrize("metrics", [None, {**DISEASE_METRICS, "pooled_mrr": 0.9}],
                         ids=["equal-values", "other-values"])
def test_another_report_of_an_old_records_group_is_its_own_record_with_no_note(metrics):
    """A7 (c): D3 of record 1's group, with equal or other values, is its own
    record; it holds no digest record 1 holds, so there is no historical note.
    HEAD merged the equal one into record 1 and refused the other."""
    from src.evaluation.sidecar import held_record, historical_overlap

    ledger = _historical_ledger()
    record = build_record(registered_report(metrics=metrics), REPORT_C)

    outcome, updated = append_record(ledger, record, run_id=RUN_2, registered_at=WHEN)

    assert outcome == "registered" and len(updated["records"]) == 2
    assert historical_overlap(updated, held_record(updated, record, RUN_2)) == ()
    assert updated["records"][0] == ledger["records"][0]


def test_the_first_append_to_a_schema_3_file_writes_schema_4_and_its_boundary():
    """A12 (b): the earlier records stay equal as JSON values, with no run; the
    boundary is the appended record's sequence. Reading never rewrote it."""
    ledger = _historical_ledger()
    ledger["records"][0]["registration"]["sequence"] = 4
    before = copy.deepcopy(ledger)

    _, updated = append_record(ledger, build_record(registered_report(), REPORT_C),
                               run_id=RUN_1, registered_at=WHEN)

    assert updated["schema_version"] == 4
    assert updated["run_identity_from_sequence"] == 5 == updated["records"][1][
        "registration"]["sequence"]
    assert updated["records"][0] == before["records"][0] and ledger == before


def _read_ledger_701d743(path):
    """`read_ledger` as it was at `701d743` (src/evaluation/sidecar.py:421-480),
    copied: the read a writer that knows only schema 3 makes before it merges by
    group (REVIEW-P5:57-67). Only its version check is reached here."""
    from src.utils.fingerprint import read_once

    read = read_once(path)
    digest = read.sha256
    ledger = json.loads(read.data.decode("utf-8"))
    if not isinstance(ledger, dict):
        raise ValueError(f"{path} is not a ledger")
    version = ledger.get("schema_version")
    if type(version) is int and version in (1, 2):
        raise ValueError(f"{path} is ledger schema {version}; this code reads schema 3.")
    if type(version) is not int or version != 3:
        raise ValueError(
            f"{path} is ledger schema {version!r}; this code reads 3. Read it with the "
            "revision that wrote it rather than reinterpreting its fields under new rules.")
    if not isinstance(ledger.get("records"), list):
        raise ValueError(f"{path} has no records list")
    return ledger, digest


def test_a_writer_that_knows_only_schema_3_refuses_a_schema_4_file_at_read(tmp_path):
    """A12 (d): the bump protects the changed write contract. The `701d743`
    writer, given a file holding run-identity records, would merge another run's
    bytes into them; it refuses the schema 4 file at read and writes nothing."""
    path = tmp_path / "evaluations.json"
    write_ledger(path, _registered(), None)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="is ledger schema 4; this code reads 3"):
        _read_ledger_701d743(path)
    assert path.read_bytes() == before
    assert read_ledger(path)[0]["records"][0]["run_id"] == RUN_1


def test_schema_5_is_refused_and_schemas_3_and_4_read_as_written(tmp_path):
    """A12 (a) and (e) at the reader: a schema 3 file reads with its bytes
    unchanged; a schema 5 file is refused as an unknown version."""
    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps(_historical_ledger()))
    before = path.read_bytes()

    assert read_ledger(path)[0]["schema_version"] == 3
    assert path.read_bytes() == before
    path.write_text(json.dumps({"schema_version": 5, "records": []}))
    with pytest.raises(ValueError, match="schema 5; this code reads schemas 3 and 4"):
        read_ledger(path)


def test_a_record_reads_by_the_boundary_never_by_its_run_id_key_alone():
    """A12 (f): below the boundary a record is historical ("not recorded"),
    whatever it carries; at or after it, a record without a readable run is an
    unreadable item, never historical."""
    from src.evaluation.sidecar import RecordIdentity, record_identities

    ledger = _registered(ledger=_historical_ledger(), run_id=RUN_2, digest=REPORT_C)
    ledger = _registered(registered_report(batch_size=8), REPORT_B, ledger, RUN_3)
    ledger["records"][0]["run_id"] = RUN_1  # a hand edit below the boundary
    ledger["records"][2].pop("run_id")
    assert ledger["run_identity_from_sequence"] == 2

    assert record_identities(ledger) == [
        RecordIdentity(None, True, None),
        RecordIdentity(RUN_2, False, None),
        RecordIdentity(None, False, "its run identity is missing"),
    ]
    assert [i.historical for i in record_identities(_historical_ledger())] == [True]


def test_an_empty_schema_4_file_reads_as_a_ledger_and_takes_its_first_record(tmp_path):
    """A12 (g): the file `write_ledger(path, empty_ledger(), None)` writes has no
    boundary on disk and is not damaged: it reads with no records and nothing
    unreadable, its boundary is 1, and the first append writes boundary 1."""
    from src.evaluation.sidecar import record_identities, run_identity_boundary

    path = tmp_path / "evaluations.json"
    write_ledger(path, empty_ledger(), None)
    ledger, _ = read_ledger(path)

    assert "run_identity_from_sequence" not in json.loads(path.read_text())
    assert ledger["records"] == [] and record_identities(ledger) == []
    assert run_identity_boundary(ledger) == (1, None)
    outcome, updated = append_record(ledger, build_record(registered_report(), REPORT_A),
                                     run_id=RUN_1, registered_at=WHEN)
    assert (outcome, updated["records"][0]["registration"]["sequence"],
            updated["run_identity_from_sequence"]) == ("registered", 1, 1)


# A8: the append's boundaries ------------------------------------------------
def _named_ledger():
    """Schema 4: historical record 1 (D1, corroborating [D2]); record 2 names
    run 1 with D3 (`REPORT_C`); the boundary is 2."""
    return _registered(digest=REPORT_C, ledger=_historical_ledger(), run_id=RUN_1)


@pytest.mark.parametrize("damage,fragment", [
    (lambda rs: rs[1].pop("run_id"),
     "the ledger's record 1 should name its run and does not, so whether it is this run "
     "cannot be told"),
    (lambda rs: rs[1].__setitem__("run_id", "not-a-run"),
     "the ledger's record 1 should name its run and does not"),
    (lambda rs: rs[1].pop("cohort_digest"),
     f"the ledger's record 1 names run {RUN_1} but its cohort digest cannot be read"),
    (lambda rs: rs[1].__setitem__("checkpoint_digest", 7),
     f"the ledger's record 1 names run {RUN_1} but its checkpoint digest cannot be read"),
    (lambda rs: rs[1].pop("source_artifact_digest"),
     f"record 2 (the ledger's record 1) names run {RUN_1} but its report digest cannot be "
     "read, so whether this report is the one registered cannot be told"),
    (lambda rs: rs.append({**copy.deepcopy(rs[1]),
                           "registration": {"sequence": 3, "registered_at": WHEN}}),
     f"the ledger's records 1, 2 all name run {RUN_1}"),
], ids=["no-run-id", "unreadable-run-id", "this-run-no-cohort", "this-run-bad-checkpoint",
        "this-run-no-report-digest", "two-records-name-this-run"])
def test_this_runs_damaged_identity_refuses_the_append_naming_the_record(damage, fragment):
    """A8 (refused): a record at or after the boundary that may be this run, or
    a record naming this run without a readable cohort, checkpoint or report
    digest (its own reason, never the replacement message), or two records
    naming it. Nothing guessed; the ledger unchanged. Run 1 offers D3 (its own
    bytes) so a found record would otherwise answer ``already_registered``; a
    new run (run 2) is refused as well where the record may be it."""
    ledger = _named_ledger()
    damage(ledger["records"])
    before = copy.deepcopy(ledger)

    with pytest.raises(ValueError) as refused:
        append_record(ledger, build_record(registered_report(), REPORT_C), run_id=RUN_1,
                      registered_at=WHEN)

    assert fragment in str(refused.value)
    assert "sealed" not in str(refused.value)
    assert ledger == before


@pytest.mark.parametrize("boundary", [
    "absent", "two", 0, 9,
], ids=["missing", "not-an-integer", "below-1", "above-next"])
def test_a_lost_boundary_refuses_a_run_no_record_names_and_still_answers_a_named_one(
    boundary
):
    """A8 (fact-check of revision 6): with the boundary lost while the file holds
    records, a run no readable run id names is refused naming the field; the
    run a record names with this report answers ``already_registered``; the
    records still read: named ones as results, the others as unreadable items
    with the boundary's reason, none historical."""
    from src.evaluation.sidecar import record_identities

    ledger = _named_ledger()
    if boundary == "absent":
        del ledger["run_identity_from_sequence"]
    else:
        ledger["run_identity_from_sequence"] = boundary
    before = copy.deepcopy(ledger)

    with pytest.raises(ValueError) as refused:
        append_record(ledger, build_record(registered_report(batch_size=8), REPORT_B),
                      run_id=RUN_2, registered_at=WHEN)
    assert ("the file's run-identity boundary (run_identity_from_sequence) cannot be read"
            in str(refused.value))
    assert "restore the boundary from a trusted backup" in str(refused.value)
    assert ledger == before
    assert append_record(ledger, build_record(registered_report(), REPORT_C), run_id=RUN_1,
                         registered_at=WHEN) == ("already_registered", ledger)
    old, named = record_identities(ledger)
    assert named.run_id == RUN_1 and not named.historical
    assert not old.historical and old.problem.startswith(
        "whether it names its run cannot be told: the file's run-identity boundary")


def test_what_another_record_lacks_never_refuses_a_verified_run():
    """A8 (proceeds): a record at or after the boundary without a readable run
    in another checkpoint; a historical record without its corroborating list
    or its source digest, whatever its group, and one without a group field.
    HEAD refused the lost group field everywhere and the lost list in the key.
    The historical note follows (ii)(b)3: this report's group or an unreadable
    group whose digests cannot be read is noted, naming the field; another
    readable group gets no note; an unreadable group whose digests read only
    if it holds this report's digest."""
    from src.evaluation.sidecar import held_record, historical_overlap

    other_weights = {**registered_manifest()["artifact_digests"], "checkpoint": "0" * 64}
    ledger = _historical_ledger()
    lost_list = build_record(registered_report(), "1" * 64)
    lost_list["registration"] = {"sequence": 2, "registered_at": WHEN}       # no list
    lost_source = build_record(registered_report(batch_size=8), "2" * 64)  # another group
    lost_source.pop("source_artifact_digest")
    lost_source.update(corroborating_reports=[], registration={"sequence": 3,
                                                               "registered_at": WHEN})
    no_group = build_record(registered_report(), "3" * 64)
    no_group.pop("cohort_digest")
    no_group.update(corroborating_reports="lost", registration={"sequence": 4,
                                                                "registered_at": WHEN})
    no_group_holding = build_record(registered_report(), "4" * 64)
    no_group_holding.pop("cohort_role")
    no_group_holding.update(corroborating_reports=[REPORT_C],
                            registration={"sequence": 5, "registered_at": WHEN})
    ledger["records"] += [lost_list, lost_source, no_group, no_group_holding]
    ledger = _registered(registered_report(artifact_digests=other_weights), "5" * 64,
                         ledger, "20261016T120003Z-000000000000-00000004")
    ledger["records"][5].pop("run_id")  # at the boundary, another checkpoint
    before = copy.deepcopy(ledger)
    record = build_record(registered_report(), REPORT_C)

    outcome, updated = append_record(ledger, record, run_id=RUN_1, registered_at=WHEN)

    assert outcome == "registered"
    assert updated["records"][:6] == before["records"]
    notes = historical_overlap(updated, held_record(updated, record, RUN_1))
    assert [(n.sequence, n.text) for n in notes] == [
        (2, "its relation to record 2 cannot be checked: corroborating_reports cannot be read"),
        (4, "its relation to record 4 cannot be checked: corroborating_reports cannot be read"),
        (5, "possible overlap with record 5, merged before each run had its own record: not "
            "recorded whether this run is one of them"),
    ]


@pytest.mark.parametrize("field,value", [
    ("cohort_digest", None), ("checkpoint_digest", "not a digest"), ("cohort_role", ""),
    ("measurement_semantics_digest", 7),
], ids=["cohort-null", "checkpoint-not-a-digest", "role-empty", "semantics-a-number"])
def test_a_group_field_that_cannot_be_read_is_a_group_that_cannot_be_read(field, value):
    """A8, final pass (review of K8-6r.2): a historical record whose group
    field is present but cannot be read is of a group that cannot be read, as
    one whose field is missing: whose corroborating list is lost, its relation
    is noted as unknown. Before the fix it read as another readable group and
    got no note. The run is registered either way."""
    from src.evaluation.sidecar import held_record, historical_overlap

    ledger = _historical_ledger()
    ledger["records"][0][field] = value
    del ledger["records"][0]["corroborating_reports"]
    record = build_record(registered_report(), REPORT_B)

    outcome, updated = append_record(ledger, record, run_id=RUN_1, registered_at=WHEN)

    assert outcome == "registered"
    notes = historical_overlap(updated, held_record(updated, record, RUN_1))
    assert [(n.sequence, n.text) for n in notes] == [
        (1, "its relation to record 1 cannot be checked: corroborating_reports cannot be read")]


@pytest.mark.parametrize("swap", [False, True], ids=["historical-first", "named-first"])
def test_this_runs_exact_lookup_comes_before_any_old_merged_record(swap):
    """A10 (rev6): record 1 is historical (D1, corroborating [D2]); record 2
    names run 1 with D3. Run 1 offered with D2 or D1 is refused as a replaced
    report naming record 2 -- record 1's digests never shadow it -- and run 1
    resent with D3 answers ``already_registered``, nothing written; `held_record`
    and `find_run` agree. (e) The same with record 2 placed first in the file.
    HEAD: no run identity, all three ``already_registered`` by digest."""
    from src.evaluation.sidecar import find_run, held_record

    ledger = _named_ledger()
    if swap:
        ledger["records"].reverse()
    before = copy.deepcopy(ledger)
    named_index = 0 if swap else 1

    for digest in (REPORT_B, REPORT_A):
        with pytest.raises(ValueError) as refused:
            append_record(ledger, build_record(registered_report(), digest), run_id=RUN_1,
                          registered_at=WHEN)
        assert str(refused.value).startswith(
            f"run {RUN_1} is already registered as record 2 from report {REPORT_C[:12]}...; "
            f"this report ({digest[:12]}...) is other bytes")
    resent = build_record(registered_report(), REPORT_C)
    assert append_record(ledger, resent, run_id=RUN_1, registered_at=WHEN) == (
        "already_registered", ledger)
    assert held_record(ledger, resent, RUN_1) is ledger["records"][named_index]
    found = find_run(ledger, RUN_1, resent["cohort_digest"], resent["checkpoint_digest"])
    assert (found.index, found.problem) == (named_index, None)
    assert ledger == before
