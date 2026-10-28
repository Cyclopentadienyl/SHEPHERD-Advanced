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
from tests.fixtures.registered_cohort import write_demo_workspace  # noqa: E402
from tests.fixtures.trained_children import train_children  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("serve-tasks")
    return train_children(write_demo_workspace(root / "workspace"),
                          ("gene", "patients_like_me"), runs=root)


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
    # The gene model's own η and SP bounds are named; its distances are
    # computed from its view, no table (README §6; PLAN_TEST_RESULTS §8.3).
    assert identity.scoring_mode == "gene_task" and identity.settings.eta == 0.5
    assert identity.shortest_paths.state == "computed"
    patient = PatientPhenotypes(patient_id="g1", phenotypes=["HP:0001250", "HP:0001263"],
                                candidate_genes=["SCN1A", "disgenet:PTEN", "MECP2", "NOT_A_GENE"])
    result = pipeline.run(patient, top_k=10)
    assert result.result_kind == GENE_TASK and result.sp_bounds == (1.0, 6.84)
    assert len(result.candidates) == 3
    assert all(isinstance(c, GeneTaskCandidate) for c in result.candidates)
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert {c.gene_name for c in result.candidates} == {"SCN1A", "PTEN", "MECP2"}
    assert any("3 of 4 candidate genes" in w and "1 map to no gene" in w
               for w in result.warnings)
    # Two identifiers of one gene are one gene, said as a repeat, not unmapped.
    repeated = pipeline.run(PatientPhenotypes(
        patient_id="g1", phenotypes=["HP:0001250"], candidate_genes=["SCN1A", "scn1a"]))
    assert len(repeated.candidates) == 1
    assert any("1 repeat an earlier entry" in w and "no gene" not in w
               for w in repeated.warnings)
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
    assert any("self-exclusion" in w for w in result.warnings)
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
    with pytest.raises(HTTPException, match="SHEPHERD_REGISTRY") as caught:
        monkeypatch.delenv("SHEPHERD_REGISTRY", raising=False)
        _reference_cohort("reference-1", "patients_like_me", pipeline)
    assert caught.value.status_code == 503  # the server's configuration
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
        "tie_policy": result.tie_policy, "sp_bounds": list(result.sp_bounds),
        "result_schema_version": 2, "session_id": "s",
        "patient_id": "g3", "timestamp": "2026-10-27T00:00:00", "summary": None,
        "inference_time_ms": 1.0, "warnings": [],
    }
    response = DiagnoseResponse.model_validate(body).root
    assert response.result_kind == "gene_task/v1" and response.sp_bounds == (1.0, 6.84)
    assert response.served_identity.settings.eta == 0.5


def test_a_wrong_reference_cohort_is_refused(trained, tmp_path, monkeypatch):
    """README §8 regression 7: a version mapped against another graph, and one
    whose samples changed after import, are refused before anything is ranked."""
    from fastapi import HTTPException

    from src.api.routes.diagnose import _reference_cohort
    from src.evaluation.cohort_registry import resolve_version
    from tests.fixtures.registered_cohort import import_version

    pipeline = _pipeline(trained, "patients_like_me")
    registry = _registry_with_genes(trained, tmp_path)
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    other = write_demo_workspace(tmp_path / "other", feature_seed=7)
    import_version(registry, other, tmp_path / "reference.json", label="other-graph")
    with pytest.raises(HTTPException, match="node_features") as caught:
        _reference_cohort("other-graph", "patients_like_me", pipeline)
    assert caught.value.status_code == 422
    samples = resolve_version(registry, "reference-1").samples_path
    samples.write_text(samples.read_text().replace('"', '"', 1) + " ")
    with pytest.raises(HTTPException, match="(?i)sha-?256|digest"):
        _reference_cohort("reference-1", "patients_like_me", pipeline)


def test_the_summary_names_each_kinds_rows():
    from src.api.routes.diagnose import _summary

    assert _summary(3, 2, None, "gene_task/v1").startswith("Found 3 candidate genes")
    assert _summary(3, 2, None, "patients_like_me/v1").startswith("Found 3 similar patients")
    assert _summary(3, 2, None, "disease_task/v1").startswith("Found 3 candidate diagnoses")


def test_a_reference_patients_context_names_its_causal_genes_outside_the_graph(trained):
    """The serving consumer of the declared truth (review of 52cb900, P2-1): a
    reference patient's recorded causal genes include those that name no
    single gene of the graph, as the cohort recorded them."""
    from types import SimpleNamespace

    pipeline = _pipeline(trained, "patients_like_me")
    row = SimpleNamespace(phenotype_ids=[0], disease_id=0, causal_gene_ids=[0],
                          causal_genes_unmapped=["GENE_X"])
    candidate = pipeline._patient_candidate(1, "r1", -1.0, row)
    assert len(candidate.causal_genes) == 2 and candidate.causal_genes[-1] == "GENE_X"


def test_a_reload_by_task_publishes_a_model_without_eta(trained, monkeypatch):
    """A patients-like-me model scores by its own head alone (eta None, as a
    disease model does): the reload's status reports that, and the reload
    publishes the model, built as the service builds it."""
    import src.api.main as api
    from src.api.routes.pipeline import PipelineReloadRequest, run_reload

    published = []
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    monkeypatch.setattr(api, "publish_pipeline", published.append)
    monkeypatch.setattr(api.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(api.app_state, "real_pipeline_requested", False, raising=False)
    result = run_reload(PipelineReloadRequest(
        data_dir=str(trained["data_dir"]), task="patients_like_me",
        checkpoint_path=str(trained["patients_like_me"])))
    assert result.success, result.message
    assert result.status.eta_effective is None and len(published) == 1
    assert published[0].config["served_identity"]["scoring_mode"] == "patients_like_me"
