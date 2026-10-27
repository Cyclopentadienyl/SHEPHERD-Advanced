"""The disease finetune: binding, batches, the joint step and validation, and
`scripts/train_model.py --stage finetune` from a base (README A6, A7, A10,
C3, F7, U6, U9, G4; contract §9.2; regressions 1, 2, 7, 8).

Module: tests/unit/test_disease_finetune.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from src.kg.graph_view import build_graph_view, split_facts  # noqa: E402
from src.models.bundle import build_bundle, finetune_spec_from_parent, reference_spec  # noqa: E402
from src.models.bundle.record import load_bundle  # noqa: E402
from src.models.bundle.spec import InputSpec, PatientEncoderSpec  # noqa: E402
from src.training.finetune_task import (  # noqa: E402
    DiseaseTask,
    FinetuneSampler,
    bind_disease_cases,
)
from src.training.lp_task import BaseLPSampler  # noqa: E402
from tests.fixtures.base_workspace import write_base_workspace  # noqa: E402


def _base_spec():
    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=8),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=4, output=4, heads=2)
    ).validate()


def _finetune_spec():
    spec = finetune_spec_from_parent(_base_spec(), "disease")
    return dataclasses.replace(spec, patient_encoder=PatientEncoderSpec(
        transformer_layers=1, transformer_heads=2, feedforward=16)).validate()


def _view(workspace):
    from src.kg.storage.file_storage import read_graph_artifacts

    graph = read_graph_artifacts(workspace, map_location="cpu").graph_data
    view = build_graph_view(graph)
    return view, split_facts(view, 0)


def _samples(*rows):
    return [SimpleNamespace(patient_id=p, phenotype_ids=ph, disease_id=d) for p, ph, d in rows]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return write_base_workspace(tmp_path_factory.mktemp("ft"))


def test_binding_drops_and_counts(workspace):
    view, _ = _view(workspace)
    binding = bind_disease_cases(view, _samples(("a", [0, 1], 0), ("b", [], 1), ("c", [2], 3)))
    assert [c.case_id for c in binding.cases] == ["a", "c"]
    assert binding.excluded == [("b", "no_phenotype_in_view")]
    assert binding.counts["excluded_no_phenotype_in_view"] == 1
    assert int(binding.cases[0].truths[0]) == int(view.view_ids("disease", torch.tensor([0])))


def _task_setup(workspace):
    view, split = _view(workspace)
    train = split.mask("train")
    cases = bind_disease_cases(view, _samples(*[(f"p{i}", [i, i + 1], i % 8) for i in range(10)]))
    lp = BaseLPSampler(view.num_nodes, view.edge_index(train), view.rel[train],
                       view.edge_index(train), view.rel[train], sizes=[3, 3], batch_size=8, seed=0)
    sampler = FinetuneSampler(view, cases.cases, lp, batch_size=4, candidate_sample_size=3,
                              sparse_sample_size=5, seed=0)
    task = DiseaseTask(view=view, val_cases=cases.cases[:4], val_excluded=1,
                       full_edges=view.edge_index(), full_relations=view.rel,
                       task_loss_weight=0.1, lp_loss_weight=0.9)
    return view, sampler, task


def test_batch_carries_truths_among_candidates(workspace):
    view, sampler, _ = _task_setup(workspace)
    batch = next(iter(sampler))
    assert bool(batch.positive.any(dim=1).all())
    seeds = batch.lp.node_ids[:batch.lp.num_seeds]
    cands = seeds[batch.candidate_index]
    assert bool((view.node_type[cands] == view.node_types.index("disease")).all())
    assert batch.phenotype_mask.sum() == sum(int(m.sum()) for m in batch.phenotype_mask)


def test_joint_step_weights_task_and_lp_and_reaches_every_part(workspace):
    view, sampler, task = _task_setup(workspace)
    torch.manual_seed(0)
    bundle = build_bundle(_finetune_spec(), view).train()
    batch = next(iter(sampler))
    step = task.training_step(bundle, batch, torch.device("cpu"))
    assert "lp" in step.terms
    assert step.loss.item() == pytest.approx(0.1 * step.terms["task"] + 0.9 * step.terms["lp"],
                                             rel=1e-5)
    step.loss.backward()
    for part in (bundle.input, bundle.encoder, bundle.patient_encoder, bundle.task_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    assert bundle.lp_head.relation_weights.grad.abs().sum() > 0


def test_validation_ranks_every_disease_with_average_ranks(workspace):
    view, _, task = _task_setup(workspace)
    bundle = build_bundle(_finetune_spec(), view)
    report = task.validate(bundle, torch.device("cpu"))
    assert report["ranked_units"] == 4 and report["units"] == 5
    assert report["coverage"] == pytest.approx(4 / 5)
    assert 0 < report["pooled_mrr"] <= 1
    assert report["conservative_pooled_mrr"] < report["pooled_mrr"]
    outcomes = task.rank_cases(bundle, task.val_cases, torch.device("cpu"))
    assert all(o.candidates == int(view.type_nodes("disease").numel()) for o in outcomes)


# --------------------------------------------------------------------------- the CLI


def _base_run(workspace, run_dir):
    import train_model

    config = train_model.TrainConfig(
        data_dir=str(workspace), run_dir=str(run_dir), learning_stage="base",
        model_spec=_base_spec().to_dict(), num_epochs=1, batch_size=16, num_neighbors=[4, 4],
        device="cpu", save_top_k=1)
    train_model.train(config, {"learning_stage", "model_spec", "num_epochs", "batch_size",
                               "num_neighbors", "device", "save_top_k"})
    return run_dir / "checkpoints" / "last.pt"


def _finetune_config(workspace, run_dir, parent, **kw):
    import train_model

    fields = dict(data_dir=str(workspace), run_dir=str(run_dir), learning_stage="finetune",
                  task="disease", finetune_from=str(parent), num_epochs=2, batch_size=4,
                  num_neighbors=[3, 3], candidate_sample_size=4, sparse_sample_size=5,
                  device="cpu", save_top_k=1)
    fields.update(kw)
    explicit = set(fields) - {"data_dir", "run_dir"}
    return train_model.TrainConfig(**fields), explicit


def test_finetune_from_a_base(workspace, tmp_path):
    import train_model

    parent = _base_run(workspace, tmp_path / "base")
    config, explicit = _finetune_config(workspace, tmp_path / "child", parent)
    metrics = train_model.train(config, explicit)
    assert "val_pooled_mrr" in metrics and "val_macro_recall@10" in metrics
    last = torch.load(tmp_path / "child" / "checkpoints" / "last.pt", map_location="cpu",
                      weights_only=True)
    bundle, record = load_bundle(last, task="disease")
    assert record.operation == "finetune" and record.family_id == "base"
    assert record.parent["run"] == "base"
    assert record.weight_transfer["initialised_new"] == ["patient_encoder", "task_head"]
    assert record.loss_identity["task_loss_weight"] == 0.1
    assert record.scoring_identity["candidates"] == "every disease of the graph view"
    base = torch.load(parent, map_location="cpu", weights_only=True)
    assert record.data_roles["split"]["digest"] == base["data_roles"]["split"]["digest"]
    assert record.model_spec.encoder == load_bundle(base)[1].model_spec.encoder
    assert last["config"]["requested"]["learning_rate"] == 1e-5
    assert record.parent["sha256"] == __import__("hashlib").sha256(parent.read_bytes()).hexdigest()
    assert "train_samples" in last["training_input_digests"]
    with pytest.raises(Exception, match="not a base"):
        load_bundle(last, learning_stage="base")


def test_finetune_refusals(workspace, tmp_path):
    import train_model
    from src.utils.run_plan import RunPlanRefused

    parent = _base_run(workspace, tmp_path / "base")
    child_cfg, explicit = _finetune_config(workspace, tmp_path / "child", parent, num_epochs=1)
    train_model.train(child_cfg, explicit)
    child = tmp_path / "child" / "checkpoints" / "last.pt"
    # A finetune parent of a finetune: children never chain.
    config, explicit = _finetune_config(workspace, tmp_path / "x", child)
    with pytest.raises(RunPlanRefused, match="never chain"):
        train_model.train(config, explicit)
    # A parent without a model spec.
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    payload.pop("model_spec")
    old = tmp_path / "old.pt"
    torch.save(payload, old)
    config, explicit = _finetune_config(workspace, tmp_path / "y", old)
    with pytest.raises(RunPlanRefused, match="written before the model spec"):
        train_model.train(config, explicit)
    # No task, or both a base and a resume parent.
    config, explicit = _finetune_config(workspace, tmp_path / "z", parent, task=None)
    with pytest.raises(ValueError, match="names its task"):
        train_model.train(config, explicit)
    config, explicit = _finetune_config(workspace, tmp_path / "w", parent, resume_from=str(child))
    with pytest.raises(ValueError, match="exactly one"):
        train_model.train(config, explicit)
    for name in ("x", "y", "z", "w"):
        assert not (tmp_path / name).exists()


def test_a_paused_finetune_resumes_with_its_lineage(workspace, tmp_path):
    import train_model

    parent = _base_run(workspace, tmp_path / "base")
    config, explicit = _finetune_config(workspace, tmp_path / "a", parent, num_epochs=2,
                                        pause_after_epoch=1)
    train_model.train(config, explicit)
    paused = tmp_path / "a" / "checkpoints" / "last.pt"
    resumed = train_model.TrainConfig(data_dir=str(workspace), run_dir=str(tmp_path / "b"),
                                      learning_stage="finetune", task="disease",
                                      resume_from=str(paused), device="cpu")
    train_model.train(resumed, {"learning_stage", "task", "resume_from", "device"})
    last = torch.load(tmp_path / "b" / "checkpoints" / "last.pt", map_location="cpu",
                      weights_only=True)
    _, record = load_bundle(last, task="disease")
    assert record.operation == "resume" and record.family_id == "base"
    assert record.parent["run"] == "a"
    assert last["progress"]["completed_epochs"] == 2
    view = json.loads((tmp_path / "b" / "graph_view.json").read_text())
    assert view["digest"] == record.graph_view.digest
