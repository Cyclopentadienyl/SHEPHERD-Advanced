"""Serving a finetuned disease bundle (README §1, §3, K3; provenance §9.3
hazard 3; Spec 1's result union with `disease_task/v1`).

The pipeline built as the service builds it serves a disease child: every
disease of the model's graph view ranked by the disease head, ties by view id,
paths only explaining the rows shown; the served identity says
`disease_task`, no mixing weight and no shortest-path table; the API's response
validates. A base is refused: it serves no diagnosis.

Module: tests/unit/test_disease_serving.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from src.core.types import DISEASE_TASK, DiseaseTaskCandidate, PatientPhenotypes  # noqa: E402
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

    root = tmp_path_factory.mktemp("serve")
    data_dir = write_demo_workspace(root / "workspace")
    train_model.train(train_model.TrainConfig(
        data_dir=str(data_dir), run_dir=str(root / "base"), learning_stage="base",
        model_spec=_base_spec().to_dict(), num_epochs=1, batch_size=32, num_neighbors=[4, 4],
        device="cpu", save_top_k=1),
        {"learning_stage", "model_spec", "num_epochs", "batch_size", "num_neighbors", "device",
         "save_top_k"})
    train_model.train(train_model.TrainConfig(
        data_dir=str(data_dir), run_dir=str(root / "child"), learning_stage="finetune",
        task="disease", finetune_from=str(root / "base" / "checkpoints" / "last.pt"),
        num_epochs=1, batch_size=8, num_neighbors=[3, 3], candidate_sample_size=8,
        sparse_sample_size=8, device="cpu", save_top_k=1),
        {"learning_stage", "task", "finetune_from", "num_epochs", "batch_size",
         "num_neighbors", "candidate_sample_size", "sparse_sample_size", "device",
         "save_top_k"})
    return {"data_dir": data_dir, "base": root / "base" / "checkpoints" / "last.pt",
            "child": root / "child" / "checkpoints" / "last.pt"}


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
        "tie_policy": result.tie_policy, "result_schema_version": 2, "session_id": "s",
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
