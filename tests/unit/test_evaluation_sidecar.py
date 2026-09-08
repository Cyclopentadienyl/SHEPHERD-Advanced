"""EVALUATION_COHORTS §6.5 — evaluation records beside the checkpoint.

A result written into a `.pt` changes its SHA-256, and the M1-M5 chain cites
checkpoints by digest, so the record has to live outside the weights it is about.

**The load-bearing property is what absence means.** A ledger holding no record
for a checkpoint says only that *this ledger* holds none. These tests pin that
wording, because the tempting shortcut -- treating a missing record as "never
evaluated" -- is a claim the system cannot support and would be believed.

Module: tests/unit/test_evaluation_sidecar.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.sidecar import (
    LEDGER_SCHEMA_VERSION,
    find_checkpoint,
    ledger_digest,
    measurement_semantics_digest,
    append_record,
    build_record,
    empty_ledger,
    read_ledger,
    record_key,
    records_for,
    write_ledger,
)


def _report(**overrides):
    """A measurement artifact in the shape `MeasurementResult.to_dict` writes."""
    manifest = {
        "mode": "A",
        "split": "val",
        "cohort_kind": "generated",
        "metric_schema_version": 1,
        "batch_size": 3,
        "candidate_construction": "2-hop subgraph",
        "device": "cuda",
        "dtype": "float32",
        "deterministic_algorithms": True,
        "canonical_tie_policy_version": "v1",
        "artifact_digests": {
            "checkpoint": "c" * 64,
            "samples": "s" * 64,
            "split_manifest": "m" * 64,
        },
        "software_revision": "abc1234",
        "cuda_executed": True,
        "torch_version": "2.5.0",
        "amp_enabled": False,
    }
    manifest.update(overrides.pop("manifest", {}))
    if overrides.pop("supplied", False):
        manifest["artifact_digests"].pop("split_manifest")
        manifest["cohort_kind"] = "supplied"
        manifest["split"] = "test"
    report = {
        "manifest": manifest,
        "authoritative_metrics": {"mrr": 0.5, "hits_at_1": 0.25},
        "n_ranked": 100,
        "n_ground_truth_absent": 0,
    }
    report.update(overrides)
    return report


# ---------------------------------------------------------------------------
# What absence means
# ---------------------------------------------------------------------------
def test_a_directory_with_no_ledger_reads_as_an_empty_one(tmp_path):
    """The first record has to be appendable to a directory that has none."""
    ledger = read_ledger(tmp_path / "evaluations.json")

    assert ledger["records"] == []
    assert ledger["schema_version"] == LEDGER_SCHEMA_VERSION


def test_the_ledger_states_what_an_absent_record_does_and_does_not_mean(tmp_path):
    """This provenance system is not closed; an evaluation can happen outside it."""
    text = empty_ledger()["what_this_is"]

    assert "holds no result" in text
    assert "not that they were never evaluated" in text


def test_an_unknown_schema_version_is_refused_rather_than_reinterpreted(tmp_path):
    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps({"schema_version": 99, "records": []}))

    with pytest.raises(ValueError, match="Read it with the revision that wrote it"):
        read_ledger(path)


def test_a_ledger_without_a_records_list_is_refused(tmp_path):
    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps({"schema_version": LEDGER_SCHEMA_VERSION}))

    with pytest.raises(ValueError, match="no records list"):
        read_ledger(path)


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "differing",
    [
        {"mode": "C"},
        {"batch_size": 8},
        {"amp_enabled": True, "amp_dtype": "bfloat16"},
        {"software_revision": "def5678"},
        {"device": "cpu"},
        {"metric_schema_version": 2},
        {"canonical_tie_policy_version": "v2"},
        {"candidate_construction": "full disease set"},
    ],
    ids=lambda d: "-".join(sorted(d)),
)
def test_runs_differing_in_anything_score_affecting_are_separate_records(differing):
    """Not one measurement with two answers -- two measurements.

    The earlier key was (checkpoint, cohort, mode, tie policy), and every case
    here except the first collided under it and was refused as a contradiction.
    `batch_size` is the sharpest: the manifest documents it as semantics rather
    than performance, because Mode A's candidate universe is the batch's subgraph.
    """
    ledger = append_record(empty_ledger(), build_record(_report(), None))
    second = build_record(_report(manifest=differing), None)

    assert len(append_record(ledger, second)["records"]) == 2


def test_a_changed_graph_artifact_is_a_separate_measurement(tmp_path):
    """The graph tensors are as much a part of a measurement as the weights."""
    other = _report()
    other["manifest"]["artifact_digests"]["node_features"] = "n" * 64
    ledger = append_record(empty_ledger(), build_record(_report(), None))

    assert len(append_record(ledger, build_record(other, None))["records"]) == 2


def test_a_descriptive_field_does_not_change_the_semantics_digest():
    """The digest is over a named list, not the whole manifest, so a field that
    cannot move a number must not silence the contradiction check."""
    base = _report()["manifest"]
    embellished = {**base, "n_samples": 999, "data_dir": "/somewhere/else"}

    assert measurement_semantics_digest(base) == measurement_semantics_digest(embellished)


def test_a_semantic_field_a_manifest_lacks_does_not_collide_with_one_carrying_it():
    """Hashed as null rather than skipped: a manifest that dropped `batch_size`
    must not produce the digest of one that recorded a value for it."""
    base = _report()["manifest"]
    without = {k: v for k, v in base.items() if k != "batch_size"}

    assert measurement_semantics_digest(base) != measurement_semantics_digest(without)


def test_the_same_measurement_recorded_twice_does_not_duplicate(tmp_path):
    """A ledger rebuilt from the same artifacts twice is the same ledger."""
    record = build_record(_report(), "d" * 64)
    ledger = append_record(empty_ledger(), record)
    again = append_record(ledger, record)

    assert again is ledger
    assert len(again["records"]) == 1


def test_one_key_with_two_answers_is_refused_and_names_the_difference(tmp_path):
    """Identical semantics, different numbers: one of the two runs is wrong about
    what it measured. That is a finding, not a merge conflict."""
    ledger = append_record(empty_ledger(), build_record(_report(), None))
    contradicting = build_record(
        _report(authoritative_metrics={"mrr": 0.9, "hits_at_1": 0.25}), None
    )

    with pytest.raises(ValueError, match="disagree on: mrr"):
        append_record(ledger, contradicting)


def test_the_refusal_states_that_nothing_was_written(tmp_path):
    ledger = append_record(empty_ledger(), build_record(_report(), None))
    with pytest.raises(ValueError, match="Nothing was written"):
        append_record(
            ledger, build_record(_report(authoritative_metrics={"mrr": 0.9}), None)
        )


def test_a_record_missing_a_key_field_is_refused(tmp_path):
    with pytest.raises(ValueError, match="missing key field"):
        record_key({"checkpoint_digest": "c" * 64})


# ---------------------------------------------------------------------------
# What a record carries
# ---------------------------------------------------------------------------
def test_the_record_is_derived_from_the_manifest_not_asserted(tmp_path):
    """A run's own manifest says which checkpoint, cohort and mode it was."""
    record = build_record(_report(), "d" * 64)

    assert record["checkpoint_digest"] == "c" * 64
    assert record["cohort_digest"] == "s" * 64
    assert record["cohort_role"] == "val"
    assert record["mode"] == "A"
    assert record["metrics"] == {"mrr": 0.5, "hits_at_1": 0.25}
    assert record["source_artifact_digest"] == "d" * 64


def test_the_cohort_section_separates_the_two_kinds(tmp_path):
    """A generated cohort is disease-disjoint by construction and names the cut it
    came from; a supplied cohort carries no allocation because nobody cut it, and
    its overlap with training is an open measurement."""
    generated = build_record(_report(), None)
    supplied = build_record(_report(supplied=True), None)

    assert generated["cohort"]["kind"] == "generated"
    assert generated["cohort"]["split_manifest_digest"] == "m" * 64
    assert supplied["cohort"]["kind"] == "supplied"
    assert supplied["cohort"]["split_manifest_digest"] is None


def test_the_kind_is_read_from_the_manifest_not_inferred_from_the_roles(tmp_path):
    """Absence of a digest could mean several things later; the field means one.

    The report here is deliberately inconsistent — it claims `supplied` while
    carrying a `split_manifest` digest — which `resolve_cohort` makes unreachable
    in practice. It is what distinguishes reading the field from inferring the
    kind, and the two implementations agree on every consistent input.
    """
    inconsistent = _report(manifest={"cohort_kind": "supplied"})
    record = build_record(inconsistent, None)

    assert record["cohort"]["kind"] == "supplied"
    assert record["cohort"]["split_manifest_digest"] == "m" * 64


def test_a_supplied_cohort_states_that_it_has_no_allocation(tmp_path):
    record = build_record(_report(supplied=True), None)

    assert "split_manifest_digest" in record["cohort"]
    assert record["cohort"]["split_manifest_digest"] is None


def test_the_record_does_not_copy_the_manifest(tmp_path):
    """It points back at the artifact by digest and stops. A second copy of a
    manifest is a second thing to keep in step with the first."""
    record = build_record(_report(), "d" * 64)

    assert "manifest" not in record
    assert "artifact_digests" not in record
    assert set(record["runtime"]) == {
        "software_revision", "cuda_executed", "torch_version", "amp_enabled"
    }


# ---------------------------------------------------------------------------
# Lookup and writing
# ---------------------------------------------------------------------------
def test_records_for_returns_only_that_checkpoints_results(tmp_path):
    ledger = append_record(empty_ledger(), build_record(_report(), None))
    other = _report()
    other["manifest"]["artifact_digests"]["checkpoint"] = "e" * 64
    ledger = append_record(ledger, build_record(other, None))

    assert len(records_for(ledger, "c" * 64)) == 1
    assert records_for(ledger, "f" * 64) == []


def test_a_written_ledger_reads_back_identically(tmp_path):
    path = tmp_path / "checkpoints" / "evaluations.json"
    ledger = append_record(empty_ledger(), build_record(_report(), "d" * 64))
    write_ledger(path, ledger)

    assert read_ledger(path) == ledger


def test_a_failed_write_leaves_the_previous_ledger_intact(tmp_path, monkeypatch):
    """The records an interrupted append would destroy are not recoverable from
    the checkpoints."""
    import src.evaluation.sidecar as sidecar

    path = tmp_path / "evaluations.json"
    first = append_record(empty_ledger(), build_record(_report(), None))
    write_ledger(path, first)
    before = path.read_bytes()

    def _explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sidecar.json, "dump", _explode)
    with pytest.raises(OSError):
        write_ledger(path, append_record(first, build_record(
            _report(manifest={"mode": "C"}), None)))

    assert path.read_bytes() == before
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")], (
        "the temporary file must not be left behind"
    )


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------
def _checkpoint_dir(tmp_path, *, digest_of=b"the weights"):
    """A directory holding a checkpoint whose bytes the report claims to measure."""
    import hashlib

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    (checkpoints / "model-1.pt").write_bytes(digest_of)
    return checkpoints, hashlib.sha256(digest_of).hexdigest()


def test_the_ledger_must_be_beside_the_weights_it_describes(tmp_path):
    """A record filed in a directory that does not hold those weights is filed
    under the wrong address, and the reader's reason for looking there is gone."""
    import scripts.record_evaluation as record_evaluation

    empty = tmp_path / "elsewhere"
    empty.mkdir()
    report_path = tmp_path / "mode_a.json"
    report_path.write_text(json.dumps(_report()))

    with pytest.raises(SystemExit, match="holds no checkpoint whose bytes"):
        record_evaluation.main(["--checkpoint-dir", str(empty),
                                "--report", str(report_path)])
    assert not (empty / "evaluations.json").exists()


def test_duplicate_checkpoint_bytes_are_acceptable(tmp_path):
    """The digest is the identity, not the name."""
    import scripts.record_evaluation as record_evaluation

    checkpoints, digest = _checkpoint_dir(tmp_path)
    (checkpoints / "best.pt").write_bytes(b"the weights")
    report = _report()
    report["manifest"]["artifact_digests"]["checkpoint"] = digest
    report_path = tmp_path / "mode_a.json"
    report_path.write_text(json.dumps(report))

    record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                            "--report", str(report_path)])
    assert (checkpoints / "evaluations.json").exists()


def test_a_concurrent_writer_is_refused_rather_than_silently_erased(tmp_path):
    """Two processes read one ledger, both append, and the second replace erases
    the first with no trace. Optimistic concurrency turns that into a refusal."""
    from src.evaluation.sidecar import write_ledger as _write

    path = tmp_path / "evaluations.json"
    first = append_record(empty_ledger(), build_record(_report(), None))
    _write(path, first)
    stale = ledger_digest(path)

    other = append_record(first, build_record(_report(manifest={"mode": "C"}), None))
    _write(path, other, expected_digest=stale)

    with pytest.raises(ValueError, match="changed since it was read"):
        _write(path, first, expected_digest=stale)
    assert len(read_ledger(path)["records"]) == 2


def test_the_cli_records_and_then_reads_back(tmp_path, capsys):
    import scripts.record_evaluation as record_evaluation

    checkpoints, digest = _checkpoint_dir(tmp_path)
    report = _report()
    report["manifest"]["artifact_digests"]["checkpoint"] = digest
    report_path = tmp_path / "mode_a.json"
    report_path.write_text(json.dumps(report))

    record_evaluation.main(["--checkpoint-dir", str(checkpoints),
                            "--report", str(report_path)])
    capsys.readouterr()
    record_evaluation.main(["--checkpoint-dir", str(checkpoints), "--show", digest])

    shown = json.loads(capsys.readouterr().out)
    assert len(shown) == 1 and shown[0]["mode"] == "A"


def test_the_cli_shows_an_empty_result_for_unknown_weights(tmp_path, capsys):
    import scripts.record_evaluation as record_evaluation

    record_evaluation.main(["--checkpoint-dir", str(tmp_path), "--show", "z" * 64])

    assert json.loads(capsys.readouterr().out) == []


def test_the_cli_refuses_a_report_that_is_not_a_measurement_artifact(tmp_path):
    import scripts.record_evaluation as record_evaluation

    report_path = tmp_path / "not_a_report.json"
    report_path.write_text(json.dumps({"hello": "world"}))

    with pytest.raises(SystemExit, match="not a measurement artifact"):
        record_evaluation.main(["--checkpoint-dir", str(tmp_path),
                                "--report", str(report_path)])


def test_the_cli_needs_exactly_one_of_report_and_show(tmp_path):
    import scripts.record_evaluation as record_evaluation

    with pytest.raises(SystemExit, match="exactly one"):
        record_evaluation.main(["--checkpoint-dir", str(tmp_path)])
