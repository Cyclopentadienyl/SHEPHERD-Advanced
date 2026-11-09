"""The gene task's functional fixture (README §8; provenance contract §9.3):
the declared rule, the digest binding to its workspace build, and the samples
reader's gene fields.

Module: tests/unit/test_task_fixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from src.kg.storage.file_storage import UnparseableFile, read_samples_file  # noqa: E402
from src.kg.task_fixture import (  # noqa: E402
    FIXTURE_LABEL,
    TaskFixtureError,
    fixture_dir,
    read_gene_fixture,
)
from tests.fixtures.registered_cohort import write_demo_workspace  # noqa: E402


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return write_demo_workspace(tmp_path_factory.mktemp("fixture") / "ws")


def _manifest_sha(data_dir):
    from src.kg.artifacts import read_split_manifest

    return read_split_manifest(data_dir).identity.sha256


def _genes_of(data_dir):
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.kg.task_fixture import disease_genes

    graph = read_graph_artifacts(data_dir, map_location="cpu").graph_data
    return disease_genes(graph["edge_index_dict"]), int(graph["num_nodes_dict"]["gene"])


def test_the_rule_one_causal_gene_of_the_disease_plus_distractors(workspace, tmp_path):
    import shutil

    from derive_task_fixture import derive

    data_dir = tmp_path / "ws"
    shutil.copytree(workspace, data_dir)
    record = derive(data_dir, seed=3, distractors=4)
    assert record["label"] == FIXTURE_LABEL and record["seed"] == 3
    genes_of, num_genes = _genes_of(data_dir)
    read = read_gene_fixture(data_dir, manifest_sha256=_manifest_sha(data_dir))
    rows = read.samples["train"].samples + read.samples["val"].samples
    assert rows
    for row in rows:
        own = set(genes_of[row.disease_id])
        assert len(row.causal_gene_ids) == 1 and row.causal_gene_ids[0] in own
        distractors = set(row.gene_ids) - set(row.causal_gene_ids)
        assert row.causal_gene_ids[0] in row.gene_ids and not distractors & own
        assert len(distractors) == min(4, num_genes - len(own))
        assert row.gene_ids == sorted(row.gene_ids)
    # Same workspace and seed: the same bytes.
    first = {s: e["sha256"] for s, e in record["files"].items()}
    again = derive(data_dir, seed=3, distractors=4, replace=True)
    assert {s: e["sha256"] for s, e in again["files"].items()} == first


def test_a_fixture_is_refused_for_another_build_or_changed_files(workspace, tmp_path):
    import shutil

    from derive_task_fixture import derive

    data_dir = tmp_path / "ws"
    shutil.copytree(workspace, data_dir)
    derive(data_dir)
    with pytest.raises(TaskFixtureError, match="already|exists"):
        derive(data_dir)
    with pytest.raises(TaskFixtureError, match="another build"):
        read_gene_fixture(data_dir, manifest_sha256="0" * 64)
    path = fixture_dir(data_dir) / "train_samples.json"
    rows = json.loads(path.read_text())
    rows[0]["causal_gene_ids"] = rows[0]["gene_ids"][-1:]
    path.write_text(json.dumps(rows))
    with pytest.raises(TaskFixtureError, match="not the one its record names"):
        read_gene_fixture(data_dir, manifest_sha256=_manifest_sha(data_dir))


def test_an_absent_fixture_names_the_command(tmp_path):
    with pytest.raises(TaskFixtureError, match="derive_task_fixture"):
        read_gene_fixture(tmp_path, manifest_sha256="0" * 64)


def test_the_reader_keeps_every_causal_gene_and_reads_old_rows(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps([
        {"patient_id": "two", "phenotype_ids": [1], "disease_id": 0, "causal_gene_ids": [2, 5],
         "gene_ids": [2, 3, 5]},
        {"patient_id": "old", "phenotype_ids": [1], "disease_id": 0},
    ]))
    two, old = read_samples_file(path, gene_fields=True).samples
    assert two.causal_gene_ids == [2, 5] and two.gene_ids == [2, 3, 5]
    assert old.causal_gene_ids is None and old.gene_ids is None
    plain = read_samples_file(path).samples[0]
    assert plain.causal_gene_ids is None


def test_a_non_integer_causal_gene_is_refused(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{"patient_id": "p", "phenotype_ids": [1], "disease_id": 0,
                                 "causal_gene_ids": [2.0]}]))
    with pytest.raises(UnparseableFile, match="causal_gene_ids"):
        read_samples_file(path, gene_fields=True)


def test_the_record_read_alone_says_absent_another_build_or_bound(workspace, tmp_path):
    """`read_fixture_record` is the record half of `read_gene_fixture` (the
    Fine-tune start's and the readiness list's check): each refusal carries its
    state, and a record of this build is returned with the identity of its
    bytes, no sample file read."""
    import shutil

    from derive_task_fixture import derive

    from src.kg.task_fixture import FIXTURE_RECORD, FIXTURE_TASKS, read_fixture_record
    from src.utils.fingerprint import file_sha256

    assert FIXTURE_TASKS == ("gene", "patients_like_me")
    data_dir = tmp_path / "ws"
    shutil.copytree(workspace, data_dir)
    with pytest.raises(TaskFixtureError, match="derive_task_fixture") as absent:
        read_fixture_record(data_dir, manifest_sha256=_manifest_sha(data_dir))
    assert absent.value.state == "absent"

    derive(data_dir)
    record, identity = read_fixture_record(data_dir, manifest_sha256=_manifest_sha(data_dir))
    assert record["workspace"]["split_manifest"] == _manifest_sha(data_dir)
    assert identity.sha256 == file_sha256(fixture_dir(data_dir) / FIXTURE_RECORD)
    for split in ("train", "val"):  # the record half reads no sample file
        (fixture_dir(data_dir) / f"{split}_samples.json").unlink()
    assert read_fixture_record(data_dir, manifest_sha256=_manifest_sha(data_dir))[0] == record

    with pytest.raises(TaskFixtureError, match="another build") as other:
        read_fixture_record(data_dir, manifest_sha256="0" * 64)
    assert other.value.state == "another_build"
    # A record whose parts have another shape is refused as unreadable, never
    # an AttributeError that the readiness list and the Fine-tune start miss.
    for part, value in (("workspace", "x"), ("workspace", [1]), ("files", 5),
                        ("files", {"train": "x", "val": {}})):
        (fixture_dir(data_dir) / FIXTURE_RECORD).write_text(json.dumps({**record, part: value}))
        with pytest.raises(TaskFixtureError, match="not a readable record") as shaped:
            read_fixture_record(data_dir, manifest_sha256=_manifest_sha(data_dir))
        assert shaped.value.state == "unreadable"
    (fixture_dir(data_dir) / FIXTURE_RECORD).write_text("{not json")
    with pytest.raises(TaskFixtureError, match="not a readable record") as torn:
        read_fixture_record(data_dir, manifest_sha256=_manifest_sha(data_dir))
    assert torn.value.state == "unreadable"
