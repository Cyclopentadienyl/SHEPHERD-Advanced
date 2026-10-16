"""EVALUATION_COHORTS §6.5 — evaluation records beside the checkpoint, schema 3.

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
    and `ModeResult.to_dict` writes it (`dataclasses.asdict`)."""
    from src.evaluation.measurement import MeasurementManifest, expected_relations

    fields = dict(
        mode="C",
        split="mygene2",
        metric_schema_version=1,
        cohort_kind="supplied",
        n_samples=80,
        candidate_construction="every disease in the knowledge graph",
        batch_size=32,
        score_semantics="raw cosine, no eta mixture and no shortest-path term",
        model_construction="production (build_shepherd_model)",
        canonical_tie_policy_version="score-desc-then-global-id-asc/v1",
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


def registered_report(metrics=None, n_ranked=80, **manifest_overrides):
    """A registered cohort version's measurement report, as the measurement
    writes it: the manifest and the result fields of `ModeResult.to_dict`."""
    return {
        "manifest": registered_manifest(**manifest_overrides),
        "authoritative_metrics": dict(metrics or {
            "untruncated_mrr": 0.5, "mean_rank": 3.0, "hits@1": 0.25, "hits@10": 0.75,
        }),
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


def _registered(report=None, digest=REPORT_A, ledger=None):
    """A ledger holding the record of `report`, registered once."""
    record = build_record(report or registered_report(), digest)
    return append_record(ledger or empty_ledger(), record, registered_at=WHEN)[1]


# ---------------------------------------------------------------------------
# What absence means, and the schema
# ---------------------------------------------------------------------------
def test_a_directory_with_no_ledger_reads_as_an_empty_one_with_no_digest(tmp_path):
    """The first record has to be appendable to a directory that has none, and
    `None` is the claim "I read no file" the write guard checks."""
    ledger, digest = read_ledger(tmp_path / "evaluations.json")

    assert ledger["records"] == []
    assert ledger["schema_version"] == LEDGER_SCHEMA_VERSION == 3
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
    states, counts and registration; it is archived, not read beside schema 3."""
    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps({"schema_version": version, "records": []}))

    with pytest.raises(ValueError) as refused:
        read_ledger(path)

    message = str(refused.value)
    assert str(path) in message
    assert f"schema {version}" in message
    assert f"evaluations.v{version}.json" in message and "register again" in message


def test_a_boolean_schema_version_is_not_read_as_one(tmp_path):
    """`True == 1` in Python; a ledger claiming `true` is not schema 1 or 3."""
    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps({"schema_version": True, "records": []}))

    with pytest.raises(ValueError, match="schema True"):
        read_ledger(path)


def test_a_ledger_without_a_records_list_is_refused(tmp_path):
    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps({"schema_version": LEDGER_SCHEMA_VERSION}))

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
    """Restated in the sidecar because it imports no torch; the two must agree."""
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
        {"metric_schema_version": 2},
        {"canonical_tie_policy_version": "v2"},
        {"candidate_construction": "path-reachable diseases"},
    ],
    ids=lambda d: "-".join(sorted(d)),
)
def test_runs_differing_in_anything_score_affecting_are_separate_records(differing):
    """Not one measurement with two answers -- two measurements.

    The earlier key was (checkpoint, cohort, mode, tie policy), and every case
    here except the first collided under it and was refused as a contradiction.
    `batch_size` is the sharpest: it is Mode C's chunk size, which can move scores
    at floating-point level, so two honest runs at different sizes are two
    measurements.
    """
    second = build_record(registered_report(**differing), REPORT_B)

    outcome, ledger = append_record(_registered(), second, registered_at=WHEN)
    assert outcome == "registered"
    assert len(ledger["records"]) == 2


def test_a_changed_graph_artifact_is_a_separate_measurement():
    """The graph tensors are as much a part of a measurement as the weights."""
    digests = {**registered_manifest()["artifact_digests"], "node_features": "0" * 64}
    other = build_record(registered_report(artifact_digests=digests), REPORT_B)

    assert append_record(_registered(), other, registered_at=WHEN)[0] == "registered"


def test_two_cohort_versions_with_identical_samples_are_two_cohorts():
    """The cohort digest is the version manifest's, which binds the source too
    (spec §0.11): the same samples from another source are another cohort."""
    other = build_record(
        registered_report(cohort_version=cohort_version(manifest_sha256="0" * 64,
                                                        source_sha256="1" * 64)),
        REPORT_B)

    assert append_record(_registered(), other, registered_at=WHEN)[0] == "registered"


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
    """D5's split is checkable: a field outside all three fails here, so none can
    escape the duplicate rule by oversight."""
    record = _registered()["records"][0]
    key, compared, excluded = set(KEY_FIELDS), set(RECORD_COMPARED_FIELDS), set(
        RECORD_EXCLUDED_FIELDS)

    assert set(record) == key | compared | excluded
    assert not key & compared and not key & excluded and not compared & excluded
    assert all(len(reason) > 20 for reason in RECORD_EXCLUDED_FIELDS.values())


def test_build_record_leaves_the_registration_to_the_ledger():
    record = build_record(registered_report(), REPORT_A)

    assert "corroborating_reports" not in record and "registration" not in record


def test_the_record_keeps_no_prose_no_paths_and_no_case_ids():
    """The report stays the source: paths and case-level data stay there, and
    standing text is rendered from `caveats.py`, never stored."""
    report = registered_report()
    text = json.dumps(build_record(report, REPORT_A))

    assert "why_this_is_here" not in text
    assert report["manifest"]["checkpoint_path"] not in text
    assert report["manifest"]["data_dir"] not in text
    assert "manifest" not in build_record(report, REPORT_A)


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
    record["metrics"]["hits@1"] = -1.0

    assert report == before


# ---------------------------------------------------------------------------
# What build_record refuses
# ---------------------------------------------------------------------------
def test_a_generated_splits_report_is_not_registered():
    """The ledger records results on registered test cohort versions (D3); a
    generated `val` is selection data (spec §0.9)."""
    with pytest.raises(ValueError, match="generated split's report is not registered"):
        build_record(generated_report(), REPORT_A)


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
# Registration, duplicates and contradiction (D5)
# ---------------------------------------------------------------------------
def test_a_first_record_is_registered_with_sequence_one():
    outcome, ledger = append_record(
        empty_ledger(), build_record(registered_report(), REPORT_A), registered_at=WHEN)
    record = ledger["records"][0]

    assert outcome == "registered"
    assert record["registration"] == {"sequence": 1, "registered_at": WHEN}
    assert record["corroborating_reports"] == []


def test_the_sequence_continues_from_the_largest_in_the_file():
    """A gap left by a hand edit is not refilled: a sequence is never repeated."""
    ledger = _registered()
    ledger["records"][0]["registration"]["sequence"] = 5
    second = build_record(registered_report(batch_size=8), REPORT_B)

    _, updated = append_record(ledger, second, registered_at=WHEN)
    assert updated["records"][1]["registration"]["sequence"] == 6


def test_the_same_report_again_is_a_no_op():
    """A ledger rebuilt from the same reports twice is the same ledger."""
    ledger = _registered()
    again = build_record(registered_report(), REPORT_A)

    outcome, same = append_record(ledger, again, registered_at="2027-01-01T00:00:00Z")
    assert outcome == "already_registered"
    assert same is ledger


def test_a_report_already_corroborating_is_a_no_op():
    ledger = append_record(
        _registered(), build_record(registered_report(), REPORT_B), registered_at=WHEN)[1]

    outcome, same = append_record(
        ledger, build_record(registered_report(), REPORT_B), registered_at=WHEN)
    assert outcome == "already_registered" and same is ledger


@pytest.mark.parametrize(
    "overrides",
    [
        {"checkpoint_path": "/moved/best.pt", "data_dir": "/moved"},
        {"r10": r10_block("overlap")},
        {"environment": environment_block(driver="999.0")},
    ],
    ids=["moved-files", "other-r10-evidence", "other-driver"],
)
def test_another_report_agreeing_on_every_compared_field_corroborates(overrides):
    """A re-run, or a checkpoint measured again after a move with unchanged bytes:
    other report bytes, the same result. Its digest joins the first record's
    corroborating reports; the first record keeps its own R10 and environment."""
    ledger = _registered()
    before = copy.deepcopy(ledger)
    second = build_record(registered_report(**overrides), REPORT_B)

    outcome, updated = append_record(ledger, second, registered_at=WHEN)
    assert outcome == "corroborated"
    assert len(updated["records"]) == 1
    first = updated["records"][0]
    assert first["corroborating_reports"] == [REPORT_B]
    assert first["source_artifact_digest"] == REPORT_A
    assert first["r10"] == before["records"][0]["r10"]
    assert first["environment"] == before["records"][0]["environment"]
    assert ledger == before, "the input ledger is not changed"


def test_one_key_with_two_results_is_refused_naming_the_metrics():
    """Identical semantics, different numbers: one of the two runs is wrong about
    what it measured. That is a finding, not a merge conflict."""
    ledger = _registered()
    before = copy.deepcopy(ledger)
    contradicting = build_record(
        registered_report(metrics={"untruncated_mrr": 0.9, "mean_rank": 3.0,
                                   "hits@1": 0.25, "hits@10": 0.75}),
        REPORT_B)

    with pytest.raises(ValueError, match=r"disagrees on: untruncated_mrr\. "):
        append_record(ledger, contradicting, registered_at=WHEN)
    assert ledger == before


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
def test_a_difference_outside_the_metrics_is_refused_naming_the_field(report, named):
    with pytest.raises(ValueError, match=f"disagrees on: {named}"):
        append_record(_registered(), build_record(report(), REPORT_B), registered_at=WHEN)


def test_the_refusal_says_the_ledger_cannot_choose():
    contradicting = build_record(registered_report(n_ranked=79), REPORT_B)

    with pytest.raises(ValueError, match="cannot choose between them"):
        append_record(_registered(), contradicting, registered_at=WHEN)


def test_append_never_changes_what_it_was_given():
    ledger, record = _registered(), build_record(registered_report(batch_size=8), REPORT_B)
    ledger_before, record_before = copy.deepcopy(ledger), copy.deepcopy(record)

    append_record(ledger, record, registered_at=WHEN)
    assert ledger == ledger_before and record == record_before


def test_a_record_carrying_its_own_registration_is_refused():
    record = {**build_record(registered_report(), REPORT_A),
              "registration": {"sequence": 1, "registered_at": WHEN}}

    with pytest.raises(ValueError, match="only the ledger assigns"):
        append_record(empty_ledger(), record, registered_at=WHEN)


@pytest.mark.parametrize(
    "damage,fragment",
    [
        (lambda r: r.pop("cohort_digest"), "record 0: record is missing key field"),
        (lambda r: r.pop("registration"), "record 0 has no registration sequence"),
        (lambda r: r["registration"].__setitem__("sequence", "1"),
         "no registration sequence"),
    ],
    ids=["no-key", "no-registration", "sequence-a-string"],
)
def test_a_hand_edit_that_broke_a_record_refuses_the_append(damage, fragment):
    """Whatever key the new record has: the next sequence cannot be computed
    without guessing past a broken record."""
    ledger = _registered()
    damage(ledger["records"][0])

    with pytest.raises(ValueError, match=fragment):
        append_record(ledger, build_record(registered_report(batch_size=8), REPORT_B),
                      registered_at=WHEN)


def test_a_registration_needs_its_time():
    with pytest.raises(ValueError, match="time it was made"):
        append_record(empty_ledger(), build_record(registered_report(), REPORT_A),
                      registered_at="")


# ---------------------------------------------------------------------------
# Lookup and the environment class
# ---------------------------------------------------------------------------
def test_records_for_returns_only_that_checkpoints_results():
    ledger = _registered()
    digests = {**registered_manifest()["artifact_digests"], "checkpoint": "0" * 64}
    ledger = append_record(ledger, build_record(
        registered_report(artifact_digests=digests), REPORT_B), registered_at=WHEN)[1]

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
        write_ledger(path, _registered(registered_report(batch_size=8), REPORT_B, first),
                     read_ledger(path)[1])

    assert path.read_bytes() == before
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")], (
        "the temporary file must not be left behind"
    )


def test_a_writer_holding_a_stale_read_is_refused(tmp_path):
    """Best-effort stale-write detection, and only that.

    **What this proves:** a writer that read the ledger, had another writer
    replace it, and then writes, is refused instead of erasing that append.
    That is the ordinary accident -- two terminals, one after the other.

    **What it does not prove, and what the implementation does not provide:**
    safety under overlapping writers. Two processes whose check-then-replace
    windows interleave both pass the comparison and the second replacement still
    erases the first. Closing that needs real mutual exclusion, which is not
    built; the ledger is single-writer by operational rule.
    """
    path = tmp_path / "evaluations.json"
    write_ledger(path, _registered(), None)
    first, stale = read_ledger(path)

    other = _registered(registered_report(batch_size=8), REPORT_B, first)
    write_ledger(path, other, stale)

    with pytest.raises(ValueError, match="changed since it was read"):
        write_ledger(path, first, stale)
    assert len(read_ledger(path)[0]["records"]) == 2


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
    write_ledger(path, append_record(empty_ledger(), record, registered_at=WHEN)[1], None)

    assert path.read_bytes().isascii()
    assert read_ledger(path)[0]["records"][0]["cohort"]["label"] == "測試-2026"
