"""
A sample id that is not a JSON integer is refused, not converted.
=================================================================
Retirement assessment §8 item 8, decided by the owner on 2026-10-12. Every caller
turns sample ids into an index tensor, which truncates `3.7` to `3` and reads
`true` as `1`, so a converted id names a node the file did not. `read_samples`,
the reader every caller shares, refuses such an id before any caller converts
it, naming the file, the patient and the field.

Shown at the Mode C entry point (`main`): the run is refused before a model is
built or anything is written. The training fields go through the same reader
and are shown on it directly.

Module: tests/unit/test_measurement_integer_ids.py
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402

BAD = [
    ("disease_id", 1.0),
    ("disease_id", 2.7),
    ("disease_id", True),
    ("disease_id", "2"),
    ("phenotype_ids", [0, 1.5]),
    ("phenotype_ids", [False, 1]),
    ("phenotype_ids", ["1"]),
    ("phenotype_ids", 1),
]


def _rewrite_first_sample(path, field, value):
    rows = json.loads(path.read_text())
    rows[0][field] = value
    path.write_text(json.dumps(rows))
    return rows[0]["patient_id"]


@pytest.mark.parametrize("field, value", BAD, ids=[f"{f}={v!r}" for f, v in BAD])
def test_mode_c_refuses_a_sample_id_that_is_not_an_integer(tmp_path, monkeypatch, field, value):
    import scripts.measure_scorer as cli
    import src.models.gnn.shepherd_gnn as shepherd_gnn

    data_dir, checkpoint = build_workspace(tmp_path / "ws")
    samples_path = data_dir / "test_samples.json"
    patient = _rewrite_first_sample(samples_path, field, value)

    def built(*args, **kwargs):
        raise AssertionError("a model was built from samples that should have been refused")

    monkeypatch.setattr(shepherd_gnn, "build_shepherd_model", built)
    output = tmp_path / "out" / "measurement.json"

    with pytest.raises(ValueError) as refused:
        cli.main([
            "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
            "--split", "test", "--cohort-kind", "supplied", "--output", str(output),
            "--device", "cpu",
        ])

    message = str(refused.value)
    assert str(samples_path) in message
    assert repr(patient) in message
    assert field in message
    assert "refused rather than converted" in message
    assert not output.parent.exists()


def test_integer_ids_still_measure(tmp_path):
    """The refusal leaves an ordinary cohort alone: the fixture's integer ids run."""
    import scripts.measure_scorer as cli

    data_dir, checkpoint = build_workspace(tmp_path / "ws")
    output = tmp_path / "out" / "measurement.json"

    assert cli.main([
        "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "test", "--cohort-kind", "supplied", "--output", str(output),
        "--device", "cpu",
    ]) == 0
    assert json.loads(output.read_text())["n_ranked"] > 0


@pytest.mark.parametrize("field, value", [
    ("candidate_disease_ids", [0, 2.0]),
    ("gene_ids", [True]),
    ("gene_ids", "0"),
])
def test_the_training_fields_are_refused_by_the_same_reader(tmp_path, field, value):
    from src.kg.storage.file_storage import read_samples

    data_dir, _ = build_workspace(tmp_path / "ws")
    path = data_dir / "test_samples.json"
    _rewrite_first_sample(path, field, value)

    with pytest.raises(ValueError, match=f"{field}.*refused rather than converted"):
        read_samples(data_dir, "test", training_fields=True)
    # Without the training fields they are not read, so not refused.
    read_samples(data_dir, "test")
