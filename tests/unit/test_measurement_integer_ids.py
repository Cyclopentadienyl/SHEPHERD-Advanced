"""
A sample id that is not a JSON integer is refused, not converted.
=================================================================
Retirement assessment §8 item 8, decided by the owner on 2026-10-12. Every caller
turns sample ids into an index tensor, which truncates `3.7` to `3` and reads
`true` as `1`, so a converted id names a node the file did not. `read_samples`,
the reader every caller shares, refuses such an id before any caller converts
it, naming the file, the patient and the field.

Shown at the measurement entry point (`main --mode disease`, measuring the
session's trained disease child, `trained_disease_child` in tests/conftest.py)
on the generated `val` split of a copy of its workspace: the run is refused
before a model is built or anything is written. A registered cohort version's
samples cannot carry such an id: the importer writes integers, and a
`samples.json` edited afterwards is refused by the digest its `cohort.json`
binds before it is parsed, so the id is never converted there either. The
training fields go through the same reader and are shown on it directly.

The same reader refuses a case that lists a phenotype twice (N2), checked after
the ids are known to be integers; one test shows it at this entry point.

Module: tests/unit/test_measurement_integer_ids.py
"""
from __future__ import annotations

import json
import shutil

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


@pytest.fixture
def child_copy(trained_disease_child, tmp_path):
    """The session's trained disease child, ``(data_dir, checkpoint)``, its
    workspace copied into this test's directory to be edited; the checkpoint is
    only read."""
    data_dir = shutil.copytree(trained_disease_child["data_dir"], tmp_path / "ws")
    return data_dir, trained_disease_child["disease"]


def _measure(cli, data_dir, checkpoint, output):
    return cli.main([
        "--mode", "disease", "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--split", "val", "--output", str(output), "--device", "cpu",
    ])


@pytest.mark.parametrize("field, value", BAD, ids=[f"{f}={v!r}" for f, v in BAD])
def test_measurement_refuses_a_sample_id_that_is_not_an_integer(
    child_copy, tmp_path, monkeypatch, field, value
):
    import scripts.measure_scorer as cli
    import src.models.bundle.record as bundle_record

    data_dir, checkpoint = child_copy
    samples_path = data_dir / "val_samples.json"
    patient = _rewrite_first_sample(samples_path, field, value)

    def built(*args, **kwargs):
        raise AssertionError("a model was built from samples that should have been refused")

    monkeypatch.setattr(bundle_record, "load_bundle", built)
    output = tmp_path / "out" / "measurement.json"

    with pytest.raises(ValueError) as refused:
        _measure(cli, data_dir, checkpoint, output)

    message = str(refused.value)
    assert str(samples_path) in message
    assert repr(patient) in message
    assert field in message
    assert "refused rather than converted" in message
    assert not output.parent.exists()


def test_measurement_refuses_a_case_not_in_phenotype_normal_form(
    child_copy, tmp_path, monkeypatch
):
    """N2, through the same reader: a case that lists a phenotype twice is refused,
    not normalised on the way, before a model is built or anything is written,
    naming the file, the patient and the producer that rebuilds it."""
    import scripts.measure_scorer as cli
    import src.models.bundle.record as bundle_record

    data_dir, checkpoint = child_copy
    samples_path = data_dir / "val_samples.json"
    phenotypes = json.loads(samples_path.read_text())[0]["phenotype_ids"]
    patient = _rewrite_first_sample(samples_path, "phenotype_ids", phenotypes * 2)
    monkeypatch.setattr(bundle_record, "load_bundle",
                        lambda *a, **k: pytest.fail("a model was built"))
    output = tmp_path / "out" / "measurement.json"

    with pytest.raises(ValueError, match="not in phenotype normal form") as refused:
        _measure(cli, data_dir, checkpoint, output)

    message = str(refused.value)
    assert str(samples_path) in message
    assert repr(patient) in message
    assert "scripts/build_knowledge_graph.py --generate-samples" in message
    assert not output.parent.exists()


def test_integer_ids_still_measure(trained_disease_child, tmp_path):
    """The refusal leaves an ordinary cohort alone: the workspace's integer ids run."""
    import scripts.measure_scorer as cli

    output = tmp_path / "out" / "measurement.json"

    assert _measure(cli, trained_disease_child["data_dir"], trained_disease_child["disease"],
                    output) == 0
    assert json.loads(output.read_text())["n_ranked"] > 0


@pytest.mark.parametrize("field, value", [("disease_id", 1.0), ("phenotype_ids", [True])])
def test_a_versions_samples_edited_to_a_non_integer_id_are_refused_by_digest(
    trained_disease_child, tmp_path, monkeypatch, field, value
):
    """The registered path's form of the rule: the edit changes the bytes, so the
    version's `samples.json` is refused against the digest its `cohort.json`
    binds, before it is parsed or any id converted, and nothing is written. The
    version is the graph source imported over the session child's workspace
    (read, not changed) into a registry of this test's own."""
    import scripts.measure_scorer as cli
    import src.models.bundle.record as bundle_record
    from tests.fixtures.registered_cohort import graph_source_cases, import_version, write_source

    data_dir = trained_disease_child["data_dir"]
    registry = tmp_path / "registry"
    registry.mkdir()
    version = import_version(registry, data_dir,
                             write_source(tmp_path / "cases.json", graph_source_cases(data_dir)))
    # The deployment's registry: a --cohort run measures only there (BACKLOG 37).
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    _rewrite_first_sample(version.samples_path, field, value)
    monkeypatch.setattr(bundle_record, "load_bundle",
                        lambda *a, **k: pytest.fail("a model was built"))

    with pytest.raises(ValueError, match="is not the samples") as refused:
        cli.main([
            "--mode", "disease", "--checkpoint", str(trained_disease_child["disease"]),
            "--data-dir", str(data_dir), "--cohort", version.manifest["label"],
            "--registry", str(registry), "--device", "cpu",
        ])

    assert "refused rather than converted" not in str(refused.value), "it was parsed"
    assert not (version.directory / "reports").exists()


@pytest.mark.parametrize("field, value", [
    ("gene_ids", [True]),
    ("gene_ids", "0"),
])
def test_the_gene_fields_are_refused_by_the_same_reader(tmp_path, field, value):
    from src.kg.storage.file_storage import read_samples

    data_dir, _ = build_workspace(tmp_path / "ws")
    path = data_dir / "test_samples.json"
    _rewrite_first_sample(path, field, value)

    with pytest.raises(ValueError, match=f"{field}.*refused rather than converted"):
        read_samples(data_dir, "test", gene_fields=True)
    # Without the gene fields they are not read, so not refused.
    read_samples(data_dir, "test")
