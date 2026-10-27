"""Task measurement of a finetuned disease bundle through the measurement entry
(`scripts/measure_scorer.py --mode disease`; PLAN_TEST_RESULTS §8, README
E1–E6, provenance §9.3 hazards 2 and 3).

A base and a disease child are trained on the demo workspace; the child is
measured on a generated split and on a registered cohort version, the latter
registered in the ledger beside the checkpoint under the task's own scoring
semantics. A base, and a checkpoint without a model spec, are refused before
anything is written.

Module: tests/unit/test_disease_measurement.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from src.models.bundle import reference_spec  # noqa: E402
from src.models.bundle.spec import InputSpec  # noqa: E402
from tests.fixtures.registered_cohort import (  # noqa: E402
    LABEL,
    graph_source_cases,
    import_version,
    write_demo_workspace,
    write_fixture_checkpoint,
    write_source,
)


def _base_spec():
    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=8),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=4, output=4, heads=2)
    ).validate()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    import train_model

    root = tmp_path_factory.mktemp("measure")
    data_dir = write_demo_workspace(root / "workspace")
    registry = root / "registry"
    registry.mkdir()
    import_version(registry, data_dir, write_source(root / "cases.json",
                                                     graph_source_cases(data_dir)))
    base_run = root / "base"
    train_model.train(train_model.TrainConfig(
        data_dir=str(data_dir), run_dir=str(base_run), learning_stage="base",
        model_spec=_base_spec().to_dict(), num_epochs=1, batch_size=32, num_neighbors=[4, 4],
        device="cpu", save_top_k=1),
        {"learning_stage", "model_spec", "num_epochs", "batch_size", "num_neighbors", "device",
         "save_top_k"})
    child_run = root / "child"
    train_model.train(train_model.TrainConfig(
        data_dir=str(data_dir), run_dir=str(child_run), learning_stage="finetune",
        task="disease", finetune_from=str(base_run / "checkpoints" / "last.pt"),
        num_epochs=1, batch_size=8, num_neighbors=[3, 3], candidate_sample_size=8,
        sparse_sample_size=8, device="cpu", save_top_k=1),
        {"learning_stage", "task", "finetune_from", "num_epochs", "batch_size",
         "num_neighbors", "candidate_sample_size", "sparse_sample_size", "device",
         "save_top_k"})
    return {"root": root, "data_dir": data_dir, "registry": registry,
            "base": base_run / "checkpoints" / "last.pt",
            "child": child_run / "checkpoints" / "last.pt"}


def test_generated_split_report(trained, tmp_path):
    import measure_scorer

    out = tmp_path / "report.json"
    code = measure_scorer.main(["--mode", "disease", "--checkpoint", str(trained["child"]),
                                "--data-dir", str(trained["data_dir"]), "--split", "val",
                                "--output", str(out), "--device", "cpu"])
    assert code == 0
    report = json.loads(out.read_text())
    manifest = report["manifest"]
    assert manifest["mode"] == "disease" and manifest["metric_schema_version"] == 2
    assert manifest["canonical_tie_policy_version"] == "average-rank/v1"
    assert manifest["model_construction"].startswith("model bundle")
    metrics = report["authoritative_metrics"]
    assert 0 < metrics["pooled_mrr"] <= 1 and "macro_recall@10" in metrics
    assert metrics["coverage"] <= 1
    ranks = json.loads((tmp_path / "report_ranks.json").read_text())
    assert len(ranks) == manifest["n_samples"]


def test_registered_cohort_is_recorded_under_task_semantics(trained, monkeypatch):
    import measure_scorer
    from src.evaluation.sidecar import read_ledger

    monkeypatch.setenv("SHEPHERD_REGISTRY", str(trained["registry"]))
    code = measure_scorer.main(["--mode", "disease", "--checkpoint", str(trained["child"]),
                                "--data-dir", str(trained["data_dir"]), "--cohort", LABEL,
                                "--registry", str(trained["registry"]), "--device", "cpu",
                                "--record"])
    assert code == 0
    ledger, _ = read_ledger(trained["child"].parent / "evaluations.json")
    (record,) = ledger["records"]
    assert record["scoring"]["mode"] == "disease"
    assert record["scoring"]["candidate_construction"] == "every disease of the model's graph view"
    assert "pooled_mrr" in record["metrics"]


def test_a_base_is_refused_before_anything_is_written(trained, tmp_path):
    import measure_scorer

    out = tmp_path / "base.json"
    with pytest.raises(ValueError, match="not a finetune"):
        measure_scorer.main(["--mode", "disease", "--checkpoint", str(trained["base"]),
                             "--data-dir", str(trained["data_dir"]), "--split", "val",
                             "--output", str(out), "--device", "cpu"])
    assert not out.exists()


def test_a_checkpoint_without_a_spec_is_history(trained, tmp_path):
    import measure_scorer

    old = tmp_path / "old.pt"
    write_fixture_checkpoint(trained["data_dir"], old)
    out = tmp_path / "old.json"
    with pytest.raises(ValueError, match="written before the model spec"):
        measure_scorer.main(["--mode", "disease", "--checkpoint", str(old),
                             "--data-dir", str(trained["data_dir"]), "--split", "val",
                             "--output", str(out), "--device", "cpu"])
    assert not out.exists()
