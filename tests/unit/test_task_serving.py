"""Serving a gene or patients-like-me finetune (README §8; PLAN_TEST_RESULTS
§8.3, §8.4; Spec 1's result union with `gene_task/v1` and
`patients_like_me/v1`).

The pipeline built as the service builds it serves a gene child: the request's
candidate genes ranked, unmapped ones counted; and a patients-like-me child:
a reference cohort version's patients ranked, the query's own case id left
out, the version named on the result. A patients-like-me model without a
reference, or a reference given to another model, is refused. Its pipeline's
bound inputs, which the reference cohort's binding reads, are the reads its
build took (`test_bound_inputs_are_the_reads_the_build_took`, served on this
child since K8-2a). Each task's served identity and status validate against the
API's schemas, the gene model's η its own and the others' null (K8-4, API1,
API2).

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
    TASK_SCORING_MODES,
    GeneTaskCandidate,
    NodeID,
    PatientCandidate,
    PatientPhenotypes,
)
from tests.fixtures.registered_cohort import write_demo_workspace  # noqa: E402
from tests.fixtures.trained_children import (  # noqa: E402
    RULES_1_RETIRED_SETTINGS,
    train_children,
)


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


def test_a_rules_1_child_written_before_k8_7_still_serves(trained, tmp_path):
    """TM2 (Q19 (a)): retiring the ShepherdGNN settings is no rules change. A
    rules-1 child written before K8-7 records them among its requested
    settings; serving reads its model record, not those, so it serves as a
    child of this build does, with the same ranking."""
    from tests.fixtures.trained_children import as_rules_1

    # The child as the rules-1 build wrote it (research comparison Stage 9a
    # raised the rules to 2; `as_rules_1` is that build's record of the run).
    payload = as_rules_1(torch.load(trained["gene"], map_location="cpu", weights_only=True))
    assert payload["training_rules"] == 1
    requested = {**payload["config"]["requested"], **RULES_1_RETIRED_SETTINGS}
    older = tmp_path / "written-before-k8-7.pt"
    torch.save({**payload, "config": {**payload["config"], "requested": requested}}, older)
    patient = PatientPhenotypes(patient_id="g1", phenotypes=["HP:0001250", "HP:0001263"],
                                candidate_genes=["SCN1A", "disgenet:PTEN", "MECP2"])

    served = [_pipeline({**trained, "gene": path}, "gene").run(patient, top_k=10)
              for path in (trained["gene"], older)]

    assert [(c.gene_name, c.score) for c in served[1].candidates] == [
        (c.gene_name, c.score) for c in served[0].candidates]
    assert len(served[1].candidates) == 3


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


def test_bound_inputs_are_the_reads_the_build_took(trained):
    """Against an independent account of the files: the four graph digests and
    the dataset record, which the reference cohort's binding reads (K7: the
    fields only Mode D read are gone), of the patients-like-me child, the model
    that binding serves (`_reference_cohort`)."""
    import hashlib
    from dataclasses import fields

    from src.inference.pipeline import BoundInputs
    from src.kg.artifacts import GRAPH_TENSOR_ROLES
    from src.kg.provenance import PROVENANCE_FILENAME

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    data_dir = trained["data_dir"]
    bound = _pipeline(trained, "patients_like_me").bound_inputs()
    assert [field.name for field in fields(BoundInputs)] == ["graph", "dataset_record"]
    assert bound.graph["kg"] == sha(data_dir / "kg.json")
    for role in GRAPH_TENSOR_ROLES:
        files = [p for p in data_dir.iterdir() if p.stem == role]
        assert len(files) == 1 and bound.graph[role] == sha(files[0]), role
    assert bound.dataset_record.identity.sha256 == sha(data_dir / PROVENANCE_FILENAME)


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


def _response(result, patient_id, **kind_fields):
    """The API's response to `result` (`DiagnoseResponse`), its kind's own
    result-level fields given by the caller, as the route assembles it."""
    from dataclasses import asdict

    from src.api.routes.diagnose import DiagnoseResponse, _candidate_body

    return DiagnoseResponse.model_validate({
        "result_kind": result.result_kind, "served_identity": result.served_identity.to_dict(),
        "phenotype_input": asdict(result.phenotype_input),
        "candidates": [_candidate_body(c, False, False) for c in result.candidates],
        "tie_policy": result.tie_policy,
        "result_schema_version": result.result_schema_version, "session_id": "s",
        "patient_id": patient_id, "timestamp": "2026-10-27T00:00:00", "summary": None,
        "inference_time_ms": 1.0, "warnings": list(result.warnings), **kind_fields})


def _patients_response(trained, tmp_path, monkeypatch):
    """A patients-like-me child's response for a registered reference cohort
    version imported with causal genes."""
    from src.api.routes.diagnose import _reference_cohort

    pipeline = _pipeline(trained, "patients_like_me")
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(_registry_with_genes(trained, tmp_path)))
    reference = _reference_cohort("reference-1", "patients_like_me", pipeline)
    result = pipeline.run(PatientPhenotypes(patient_id="q1", phenotypes=["HP:0001250"]),
                          top_k=5, reference_cohort=reference)
    return _response(result, "q1", reference_cohort=result.reference_cohort)


def _gene_response(trained):
    result = _pipeline(trained, "gene").run(PatientPhenotypes(
        patient_id="g3", phenotypes=["HP:0001250"], candidate_genes=["SCN1A", "PTEN"]))
    return _response(result, "g3", sp_bounds=list(result.sp_bounds))


#: The status fields that left with the served mixture and the shortest-path
#: table (K8-4, API2): absent from the status and the pipeline's configuration.
RETIRED_STATUS_FIELDS = {"sp_ready", "eta_configured", "sp_max_hops", "sp_hop_bound_source",
                         "sp_kg_binding", "fingerprint_warnings"}


@pytest.mark.parametrize("task", ["disease", "gene", "patients_like_me"])
def test_each_tasks_identity_and_status_validate_with_its_own_eta(
        trained, trained_disease_child, tmp_path, monkeypatch, task):
    """API1, API2 (K8-4): each task's child, built as the service builds it.
    Its served identity validates against the API's mirror and inside the
    response to one of its results; its status validates as
    `PipelineStatusResponse` and carries none of the retired fields. The
    identity's `settings.eta` is the gene spec's own η for a gene model (0.5 in
    the demo) and null for a disease or patients-like-me model, which mix
    nothing; the status's `scoring_mode` and `eta_effective` are the
    identity's."""
    from src.api.routes.diagnose import ServedIdentity
    from src.api.routes.pipeline import PipelineStatusResponse, _status_of

    children = trained_disease_child if task == "disease" else trained
    pipeline = _pipeline(children, task)
    identity = pipeline.served_identity().to_dict()
    config = pipeline.get_pipeline_config()
    status = _status_of(config, str(children["data_dir"]), str(children[task]))

    eta = pipeline.model.task_head.spec.gene.eta if task == "gene" else None
    assert (task != "gene") or eta == 0.5
    assert identity["settings"]["eta"] == eta
    assert identity["shortest_paths"] == {
        "state": "computed" if task == "gene" else "absent"}
    assert ServedIdentity.model_validate(identity).model_dump() == identity
    if task == "gene":
        response = _gene_response(trained)
    elif task == "patients_like_me":
        response = _patients_response(trained, tmp_path, monkeypatch)
    else:
        result = pipeline.run(PatientPhenotypes(patient_id="d1",
                                                phenotypes=["HP:0001250", "HP:0001263"]),
                              top_k=3, include_explanations=False)
        response = _response(result, "d1")
    assert response.root.served_identity.model_dump() == identity
    as_json = status.model_dump(mode="json")
    assert PipelineStatusResponse.model_validate(as_json).model_dump(mode="json") == as_json
    assert status.scoring_mode == identity["scoring_mode"] == TASK_SCORING_MODES[task]
    assert status.eta_effective == identity["settings"]["eta"]
    assert not RETIRED_STATUS_FIELDS & set(as_json)
    assert not RETIRED_STATUS_FIELDS & set(config)


def test_the_api_resolves_a_reference_cohort_and_validates_the_response(trained, tmp_path,
                                                                        monkeypatch):
    from fastapi import HTTPException

    from src.api.routes.diagnose import _reference_cohort

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
    response = _response(result, "q1", reference_cohort=result.reference_cohort)
    assert response.root.result_kind == "patients_like_me/v1"
    assert response.root.reference_cohort.label == "reference-1"
    assert response.root.candidates[0].causal_genes


def test_a_gene_response_validates(trained):
    response = _gene_response(trained).root
    assert response.result_kind == "gene_task/v1" and response.sp_bounds == (1.0, 6.84)
    assert response.served_identity.settings.eta == 0.5


def test_the_diagnosis_page_and_exports_read_both_kinds(trained, tmp_path, monkeypatch):
    """The page's contract with the real bodies (as `test_disease_serving`'s):
    a gene child's and a patients-like-me child's responses, through
    `model_dump`, draw every row in the table and the details, and their CSV
    and report carry the kind's values and the reference cohort version."""
    import csv
    import io

    from src.webui.components import diagnosis_panel as dp

    gene = _gene_response(trained).model_dump(mode="json")
    table = dp.results_body_html(gene, 1)
    assert table.count("<tr data-key=") == len(gene["candidates"]) == 2
    assert "Gene task model" in table and ">SP similarity</th>" in table
    for candidate in gene["candidates"]:
        assert f'title="{candidate["gene_id"]}">{candidate["gene_name"]}<' in table
        assert dp._num(candidate["score"]) in table
    assert "1.000 to 6.840" in dp.details_outputs(gene, 1)["details_html"]["value"]
    rows = list(csv.DictReader(io.StringIO(dp._build_results_csv(gene))))
    assert [r["gene_name"] for r in rows] == [c["gene_name"] for c in gene["candidates"]]
    assert {(r["sp_bounds_lower"], r["sp_bounds_upper"]) for r in rows} == {("1.0000", "6.8400")}
    assert rows[0]["served_scoring_mode"] == "gene_task" and rows[0]["served_eta"] == "0.5"
    report = dp._build_results_report_md(gene)
    assert dp.GENE_SCORE_NOTE in report and "a gene task model discovers no paths" in report

    patients = _patients_response(trained, tmp_path, monkeypatch).model_dump(mode="json")
    reference = patients["reference_cohort"]
    table = dp.results_body_html(patients, 1)
    assert table.count("<tr data-key=") == len(patients["candidates"]) > 0
    assert f"version <code>{reference['version_sha256'][:12]}</code>" in table
    first = patients["candidates"][0]
    details = dp.details_outputs(patients, first["rank"])["details_html"]["value"]
    assert all(f"<code>{term}</code>" in details for term in first["phenotypes"])
    rows = list(csv.DictReader(io.StringIO(dp._build_results_csv(patients))))
    assert [r["case_id"] for r in rows] == [c["case_id"] for c in patients["candidates"]]
    assert rows[0]["reference_cohort_version_sha256"] == f"sha256:{reference['version_sha256']}"
    assert rows[0]["reference_cohort_patients"] == str(reference["patients"])
    assert rows[0]["causal_genes"] == "; ".join(first["causal_genes"])
    report = dp._build_results_report_md(patients)
    assert f"`sha256:{reference['samples_sha256']}`" in report
    assert dp.SIMILARITY_NOTE in report and "**Recorded HPO terms**" in report


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
