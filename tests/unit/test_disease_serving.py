"""Serving a finetuned disease bundle (README §1, §3, K3; provenance §9.3
hazard 3; Spec 1's result union with `disease_task/v1`).

The pipeline built as the service builds it serves a disease child: every
disease of the model's graph view ranked by the disease head, ties by view id,
paths only explaining the rows shown; the served identity says
`disease_task`, no mixing weight and no shortest-path table; the API's response
validates. A base is refused: it serves no diagnosis. `run` is `rank()` cut at
`top_k` (`test_run_is_rank_cut_at_top_k`), its outcomes are `rank()`'s
(`test_the_three_outcomes_are_the_three_returns_of_run`), and the service
builds through `build_served_pipeline`
(`test_the_service_and_build_served_pipeline_build_the_same_pipeline`); these
three are served on a disease child since K8-2a, and were the mixture's before.

Module: tests/unit/test_disease_serving.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from src.core.types import DISEASE_TASK, DiseaseTaskCandidate, PatientPhenotypes  # noqa: E402
from tests.fixtures.registered_cohort import write_demo_workspace  # noqa: E402
from tests.fixtures.trained_children import train_children  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("serve")
    trained = train_children(write_demo_workspace(root / "workspace"), ("disease",), runs=root)
    return {"data_dir": trained["data_dir"], "base": trained["base"],
            "child": trained["disease"]}


def _pipeline(trained, checkpoint):
    from src.inference.pipeline import build_served_pipeline

    return build_served_pipeline(trained["data_dir"] / "kg.json", checkpoint_path=str(checkpoint),
                                 data_dir=str(trained["data_dir"]), device="cpu")


def test_a_disease_child_ranks_every_disease(trained):
    pipeline = _pipeline(trained, trained["child"])
    identity = pipeline.served_identity()
    assert identity.scoring_mode == "disease_task"
    assert identity.settings.eta is None and identity.shortest_paths.state == "absent"
    assert identity.checkpoint.state == "verified"

    patient = PatientPhenotypes(patient_id="p1", phenotypes=["HP:0001250", "HP:0001263"])
    ranking = pipeline.rank(patient)
    n_diseases = int(pipeline._bundle_view.type_nodes("disease").numel())
    assert ranking.outcome == "ranked" and len(ranking.candidates) == n_diseases
    scores = [c.score for c in ranking.candidates]
    assert scores == sorted(scores, reverse=True) and all(s <= 0 for s in scores)

    result = pipeline.run(patient, top_k=3, include_explanations=True)
    assert result.result_kind == DISEASE_TASK and len(result.candidates) == 3
    assert all(isinstance(c, DiseaseTaskCandidate) for c in result.candidates)
    assert [c.rank for c in result.candidates] == [1, 2, 3]
    assert result.tie_policy == "score-desc-then-view-id-asc/v1"
    assert "uncalibrated" in result.label
    assert all(c.explanation for c in result.candidates)
    # The order is the model's: the same as rank()'s first three.
    assert [c.disease_id for c in result.candidates] == [
        c.disease_id for c in ranking.candidates[:3]]


def _response(trained):
    """The API's response to a diagnosis served by the disease child."""
    from dataclasses import asdict

    from src.api.routes.diagnose import DiagnoseResponse, _candidate_body

    pipeline = _pipeline(trained, trained["child"])
    result = pipeline.run(PatientPhenotypes(patient_id="p2", phenotypes=["HP:0001250"]),
                          top_k=2, include_explanations=False)
    body = {
        "result_kind": result.result_kind, "served_identity": result.served_identity.to_dict(),
        "phenotype_input": asdict(result.phenotype_input),
        "candidates": [_candidate_body(c, False, True) for c in result.candidates],
        "tie_policy": result.tie_policy,
        "result_schema_version": result.result_schema_version, "session_id": "s",
        "patient_id": "p2", "timestamp": "2026-10-27T00:00:00", "summary": None,
        "inference_time_ms": 1.0, "warnings": [],
    }
    return DiagnoseResponse.model_validate(body)


def test_the_api_response_validates(trained):
    response = _response(trained)
    assert response.root.result_kind == "disease_task/v1"
    assert response.root.candidates[0].score <= 0


def test_the_diagnosis_page_and_exports_read_the_kind(trained):
    """The page's table, details, CSV and report show the head's score as the
    Score, keep the evidence columns, and say the identity in the kind's words:
    no mixing weight and no shortest-path table to miss."""
    import csv
    import io

    from src.webui.components import diagnosis_panel as dp

    result = _response(trained).model_dump(mode="json")
    first = result["candidates"][0]
    table = dp.candidates_table_html(result["candidates"], 1, DISEASE_TASK)
    assert "Score" in table and "Model term" not in table and "Evidence" in table
    assert dp._num(first["score"]) in table
    assert f"Score <b>{dp._num(first['score'])}</b>" in dp.details_header_html(
        first, DISEASE_TASK)

    rows = list(csv.DictReader(io.StringIO(dp._build_results_csv(result))))
    assert {"score", "evidence_label"} <= set(rows[0]) and "served_score" not in rows[0]
    assert rows[0]["result_kind"] == DISEASE_TASK and float(rows[0]["score"]) <= 0
    assert rows[0]["served_scoring_mode"] == "disease_task"

    report = dp._build_results_report_md(result)
    assert "uncalibrated" in report and "NO SHORTEST-PATH TABLE" not in report
    assert "a disease task model has no terms to mix" in report
    assert "a disease task model scores without one" in report


def test_a_base_is_refused(trained):
    from src.inference.pipeline import PipelineBuildError

    with pytest.raises(PipelineBuildError, match="not a finetune"):
        _pipeline(trained, trained["base"])


def test_a_phenotype_outside_the_view_is_reported_unknown_not_used(trained):
    """The mapping step admits only what the served model can score: a graph
    phenotype outside the model's graph view is listed as unknown, warned, and
    never reported as used."""
    pipeline = _pipeline(trained, trained["child"])
    view = pipeline._bundle_view
    index = pipeline._node_id_to_idx["phenotype"]
    outside = [key for key, row in index.items()
               if int(view.view_ids("phenotype", torch.tensor([row]))[0]) < 0]
    assert outside, "the demo view excludes phenotypes outside its largest component"
    term = outside[0].split(":", 1)[1]
    result = pipeline.run(PatientPhenotypes(patient_id="p3", phenotypes=["HP:0001250", term]),
                          top_k=3)
    assert term in result.phenotype_input.unknown
    assert term not in result.phenotype_input.used
    assert any("outside the model's graph view" in w for w in result.warnings)


@pytest.fixture(scope="module")
def served(trained):
    """The disease child served once, for the tests that only read it, and the
    patients they rank: the graph's own cases (`graph_source_cases`), an id
    the graph lacks alone, one with a case and a repeat, and 150 ids."""
    from tests.fixtures.registered_cohort import graph_source_cases

    cases = [case["phenotypes"] for case in graph_source_cases(trained["data_dir"])]
    patients = cases + [["HP:9999999"], ["HP:9999999"] + cases[0] + cases[0][:1],
                        [f"HP:{i:07d}" for i in range(150)]]
    return {"pipeline": _pipeline(trained, trained["child"]), "patients": patients}


def _patient(phenotypes, patient_id="p"):
    return PatientPhenotypes(patient_id=patient_id, phenotypes=list(phenotypes))


def _row(candidate):
    """A candidate as `rank()` makes it: [disease, rank, name, the head's
    score]. The evidence fields are `run`'s, built on the rows it keeps after
    the cut (paths explain, they never rank)."""
    assert isinstance(candidate, DiseaseTaskCandidate), type(candidate).__name__
    return [str(candidate.disease_id), candidate.rank, candidate.disease_name, candidate.score]


@pytest.mark.parametrize("top_k", [1, 2, 5, 1000])
def test_run_is_rank_cut_at_top_k(served, top_k):
    """Field for field, the candidates `run` returns are `rank()`'s first
    `top_k`; the warnings and the phenotype account are the same."""
    pipeline = served["pipeline"]
    ranked = 0
    for phenotypes in served["patients"]:
        ranking = pipeline.rank(_patient(phenotypes))
        result = pipeline.run(_patient(phenotypes), top_k=top_k, include_explanations=False)
        assert [_row(c) for c in result.candidates] == \
            [_row(c) for c in ranking.candidates[:top_k]]
        assert result.warnings == ranking.warnings
        assert result.phenotype_input == ranking.phenotype_input
        ranked += ranking.outcome == "ranked"
    assert ranked, "no patient was ranked: the cut was never taken"


def test_the_three_outcomes_are_the_three_returns_of_run(served):
    """`refused` is a refused validation (no candidates, its errors as the
    warnings); `ranked` holds every candidate, ranked 1..N. A disease child
    has no `no_candidates`: once anything maps, its head ranks every disease
    of the view (`test_a_disease_child_ranks_every_disease`)."""
    pipeline = served["pipeline"]
    refused = pipeline.rank(_patient(["HP:9999999"]))
    assert refused.outcome == "refused" and refused.candidates == []
    assert refused.warnings == ["No valid phenotypes after validation"]

    ranked = pipeline.rank(_patient(served["patients"][0]))
    assert ranked.outcome == "ranked" and ranked.candidates
    assert [c.rank for c in ranked.candidates] == list(range(1, len(ranked.candidates) + 1))
    scores = [c.score for c in ranked.candidates]
    assert scores == sorted(scores, reverse=True)


def test_the_service_and_build_served_pipeline_build_the_same_pipeline(trained):
    """The API's `build_pipeline` builds through `build_served_pipeline`, so both
    serve the same identity from the same files: the disease child's, scored by
    its task."""
    from src.api import main as api
    from src.inference.pipeline import build_served_pipeline

    workspace = trained["data_dir"]
    checkpoint = str(trained["child"])
    direct = build_served_pipeline(workspace / "kg.json", checkpoint_path=checkpoint,
                                   data_dir=str(workspace), device="cpu")
    bundle = api.build_pipeline(kg_path=str(workspace / "kg.json"), checkpoint_path=checkpoint,
                                data_dir=str(workspace), device="cpu")
    assert direct.served_identity() == bundle.pipeline.served_identity()
    assert direct.served_identity().scoring_mode == "disease_task"
