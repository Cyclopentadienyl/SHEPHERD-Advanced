"""Serving a gene or patients-like-me finetune (README §8; PLAN_TEST_RESULTS
§8.3, §8.4; Spec 1's result union with `gene_task/v1` and
`patients_like_me/v1`).

The pipeline built as the service builds it serves a gene child: the request's
candidate genes ranked, unmapped ones counted; and a patients-like-me child:
a reference cohort version's patients ranked, the query's own case id left
out, the version named on the result. A patients-like-me model without a
reference, or a reference given to another model, is refused.

Module: tests/unit/test_task_serving.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from src.core.types import (  # noqa: E402
    GENE_TASK,
    PATIENTS_LIKE_ME,
    GeneTaskCandidate,
    NodeID,
    PatientCandidate,
    PatientPhenotypes,
)
from src.models.bundle import reference_spec  # noqa: E402
from src.models.bundle.spec import InputSpec  # noqa: E402
from tests.fixtures.registered_cohort import write_demo_workspace  # noqa: E402


def _base_spec():
    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=8),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=4, output=4, heads=2)
    ).validate()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    import train_model
    from derive_task_fixture import derive

    root = tmp_path_factory.mktemp("serve-tasks")
    data_dir = write_demo_workspace(root / "workspace")
    derive(data_dir, distractors=4)
    train_model.train(train_model.TrainConfig(
        data_dir=str(data_dir), run_dir=str(root / "base"), learning_stage="base",
        model_spec=_base_spec().to_dict(), num_epochs=1, batch_size=32, num_neighbors=[4, 4],
        device="cpu", save_top_k=1),
        {"learning_stage", "model_spec", "num_epochs", "batch_size", "num_neighbors", "device",
         "save_top_k"})
    base = root / "base" / "checkpoints" / "last.pt"
    for task in ("gene", "patients_like_me"):
        train_model.train(train_model.TrainConfig(
            data_dir=str(data_dir), run_dir=str(root / task), learning_stage="finetune",
            task=task, finetune_from=str(base), num_epochs=1, batch_size=4,
            num_neighbors=[3, 3], sparse_sample_size=4, device="cpu", save_top_k=1),
            {"learning_stage", "task", "finetune_from", "num_epochs", "batch_size",
             "num_neighbors", "sparse_sample_size", "device", "save_top_k"})
    return {"data_dir": data_dir, "gene": root / "gene" / "checkpoints" / "last.pt",
            "patients_like_me": root / "patients_like_me" / "checkpoints" / "last.pt"}


def _pipeline(trained, task):
    from src.inference.pipeline import build_served_pipeline

    return build_served_pipeline(trained["data_dir"] / "kg.json",
                                 checkpoint_path=str(trained[task]),
                                 data_dir=str(trained["data_dir"]), device="cpu")


def _reference(trained):
    from src.inference.pipeline import ReferenceCohort
    from src.kg.artifacts import read_split_manifest
    from src.kg.task_fixture import read_gene_fixture

    fixture = read_gene_fixture(trained["data_dir"], manifest_sha256=read_split_manifest(
        trained["data_dir"]).identity.sha256)
    samples = fixture.samples["train"]
    return ReferenceCohort({"label": "fixture-train", "samples_sha256": samples.identity.sha256},
                           samples.samples)


def test_a_gene_child_ranks_the_submitted_candidates(trained):
    pipeline = _pipeline(trained, "gene")
    identity = pipeline.served_identity()
    assert identity.scoring_mode == "gene_task" and identity.settings.eta is None
    patient = PatientPhenotypes(patient_id="g1", phenotypes=["HP:0001250", "HP:0001263"],
                                candidate_genes=["SCN1A", "disgenet:PTEN", "MECP2", "NOT_A_GENE"])
    result = pipeline.run(patient, top_k=10)
    assert result.result_kind == GENE_TASK
    assert len(result.candidates) == 3
    assert all(isinstance(c, GeneTaskCandidate) for c in result.candidates)
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert {c.gene_name for c in result.candidates} == {"SCN1A", "PTEN", "MECP2"}
    assert any("3 of 4 candidate genes" in w for w in result.warnings)
    first = result.candidates[0]
    assert first.score == pytest.approx(0.5 * first.embsim + 0.5 * first.sp_similarity, rel=1e-5)


def test_a_gene_child_without_candidates_ranks_nothing(trained):
    result = _pipeline(trained, "gene").run(
        PatientPhenotypes(patient_id="g2", phenotypes=["HP:0001250"]), top_k=5)
    assert result.candidates == [] and any("candidate_genes" in w for w in result.warnings)


def test_a_patients_like_me_child_ranks_the_reference_cohort(trained):
    pipeline = _pipeline(trained, "patients_like_me")
    reference = _reference(trained)
    own = reference.samples[0]
    patient = PatientPhenotypes(patient_id=own.patient_id, phenotypes=["HP:0001250"])
    result = pipeline.run(patient, top_k=50, reference_cohort=reference)
    assert result.result_kind == PATIENTS_LIKE_ME
    assert result.reference_cohort["label"] == "fixture-train"
    assert all(isinstance(c, PatientCandidate) for c in result.candidates)
    assert own.patient_id not in {c.case_id for c in result.candidates}
    assert len(result.candidates) == len(reference.samples) - 1
    sims = [c.similarity for c in result.candidates]
    assert sims == sorted(sims, reverse=True) and all(s <= 0 for s in sims)
    assert all(c.causal_genes and c.phenotypes for c in result.candidates)


def test_a_reference_is_required_by_patients_like_me_and_refused_elsewhere(trained):
    with pytest.raises(ValueError, match="reference cohort"):
        _pipeline(trained, "patients_like_me").run(
            PatientPhenotypes(patient_id="q", phenotypes=["HP:0001250"]))
    with pytest.raises(ValueError, match="patients-like-me model only"):
        _pipeline(trained, "gene").run(
            PatientPhenotypes(patient_id="q", phenotypes=["HP:0001250"], candidate_genes=["SCN1A"]),
            reference_cohort=_reference(trained))


def _registry_with_genes(trained, tmp_path):
    """A reference cohort version imported with causal genes: the fixture's
    train rows, written as a shepherd-cases/1 source in the graph's ids."""
    import json

    from src.kg.graph import KnowledgeGraph
    from tests.fixtures.registered_cohort import import_version

    data_dir = trained["data_dir"]
    kg = KnowledgeGraph.read_json(data_dir / "kg.json").kg
    reverse = {t: {i: k for k, i in m.items()} for t, m in kg.get_node_id_mapping().items()}
    cases = []
    for row in _reference(trained).samples:
        cases.append({
            "case_id": row.patient_id,
            "phenotypes": [reverse["phenotype"][i].split(":", 1)[1] for i in row.phenotype_ids],
            "disease": reverse["disease"][row.disease_id].split(":", 1)[1],
            "causal_genes": [kg.get_node(NodeID.from_string(reverse["gene"][g])).name
                             for g in row.causal_gene_ids]})
    source = tmp_path / "reference.json"
    source.write_text(json.dumps({"format": "shepherd-cases/1", "cases": cases}))
    registry = tmp_path / "registry"
    registry.mkdir()
    import_version(registry, data_dir, source, label="reference-1")
    return registry


def test_the_api_resolves_a_reference_cohort_and_validates_the_response(trained, tmp_path,
                                                                        monkeypatch):
    from dataclasses import asdict

    from fastapi import HTTPException

    from src.api.routes.diagnose import DiagnoseResponse, _candidate_body, _reference_cohort

    pipeline = _pipeline(trained, "patients_like_me")
    with pytest.raises(HTTPException, match="SHEPHERD_REGISTRY"):
        monkeypatch.delenv("SHEPHERD_REGISTRY", raising=False)
        _reference_cohort("reference-1", "patients_like_me", pipeline)
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(_registry_with_genes(trained, tmp_path)))
    with pytest.raises(HTTPException, match="name one"):
        _reference_cohort(None, "patients_like_me", pipeline)
    with pytest.raises(HTTPException, match="not one"):
        _reference_cohort("reference-1", "disease_task", pipeline)
    reference = _reference_cohort("reference-1", "patients_like_me", pipeline)
    assert reference.identity["label"] == "reference-1"
    assert all(s.causal_gene_ids for s in reference.samples)
    result = pipeline.run(PatientPhenotypes(patient_id="q1", phenotypes=["HP:0001250"]),
                          top_k=5, reference_cohort=reference)
    body = {
        "result_kind": result.result_kind, "served_identity": result.served_identity.to_dict(),
        "phenotype_input": asdict(result.phenotype_input),
        "candidates": [_candidate_body(c, False, False) for c in result.candidates],
        "tie_policy": result.tie_policy, "reference_cohort": result.reference_cohort,
        "result_schema_version": 2, "session_id": "s", "patient_id": "q1",
        "timestamp": "2026-10-27T00:00:00", "summary": None, "inference_time_ms": 1.0,
        "warnings": [],
    }
    response = DiagnoseResponse.model_validate(body)
    assert response.root.result_kind == "patients_like_me/v1"
    assert response.root.reference_cohort.label == "reference-1"
    assert response.root.candidates[0].causal_genes


def test_a_gene_response_validates(trained):
    from dataclasses import asdict

    from src.api.routes.diagnose import DiagnoseResponse, _candidate_body

    result = _pipeline(trained, "gene").run(PatientPhenotypes(
        patient_id="g3", phenotypes=["HP:0001250"], candidate_genes=["SCN1A", "PTEN"]))
    body = {
        "result_kind": result.result_kind, "served_identity": result.served_identity.to_dict(),
        "phenotype_input": asdict(result.phenotype_input),
        "candidates": [_candidate_body(c, False, False) for c in result.candidates],
        "tie_policy": result.tie_policy, "result_schema_version": 2, "session_id": "s",
        "patient_id": "g3", "timestamp": "2026-10-27T00:00:00", "summary": None,
        "inference_time_ms": 1.0, "warnings": [],
    }
    assert DiagnoseResponse.model_validate(body).root.result_kind == "gene_task/v1"
