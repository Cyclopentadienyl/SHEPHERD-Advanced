"""The gene finetune: binding, upsampling, the joint step, validation and
`scripts/train_model.py --stage finetune --task gene` (README A8, U9, U14, §6,
F9, F10, C8; regressions 2, 8, 10).

Module: tests/unit/test_gene_finetune.py
"""
from __future__ import annotations

import dataclasses
import shutil
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
from src.models.tasks.gene import (  # noqa: E402
    bind_gene_cases,
    distance_table,
    neighbour_table,
)
from src.training.finetune_task import GeneSampler, GeneTask  # noqa: E402
from src.training.lp_task import BaseLPSampler  # noqa: E402
from tests.fixtures.registered_cohort import write_demo_workspace  # noqa: E402


def _base_spec():
    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=8),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=4, output=4, heads=2)
    ).validate()


def _gene_spec():
    spec = finetune_spec_from_parent(_base_spec(), "gene")
    return dataclasses.replace(spec, patient_encoder=PatientEncoderSpec(
        transformer_layers=1, transformer_heads=2, feedforward=16)).validate()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    import train_model
    from derive_task_fixture import derive

    root = tmp_path_factory.mktemp("gene")
    data_dir = write_demo_workspace(root / "ws")
    derive(data_dir, distractors=4)
    train_model.train(train_model.TrainConfig(
        data_dir=str(data_dir), run_dir=str(root / "base"), learning_stage="base",
        model_spec=_base_spec().to_dict(), num_epochs=1, batch_size=32, num_neighbors=[4, 4],
        device="cpu", save_top_k=1),
        {"learning_stage", "model_spec", "num_epochs", "batch_size", "num_neighbors", "device",
         "save_top_k"})
    return {"data_dir": data_dir, "base": root / "base" / "checkpoints" / "last.pt"}


def _setup(trained, *, upsample=2):
    from src.kg.artifacts import read_split_manifest
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.kg.task_fixture import read_gene_fixture

    data_dir = trained["data_dir"]
    view = build_graph_view(read_graph_artifacts(data_dir, map_location="cpu").graph_data)
    train = split_facts(view, 0).mask("train")
    fixture = read_gene_fixture(data_dir,
                                manifest_sha256=read_split_manifest(data_dir).identity.sha256)
    cases = bind_gene_cases(view, fixture.samples["train"].samples).cases
    val = bind_gene_cases(view, fixture.samples["val"].samples).cases
    phenotypes = torch.cat([c.phenotypes for c in cases + val])
    distances = distance_table(view, phenotypes, view.type_nodes("gene"))
    neighbours = neighbour_table(view, 3)
    lp = BaseLPSampler(view.num_nodes, view.edge_index(train), view.rel[train],
                       view.edge_index(train), view.rel[train], sizes=[3, 3], batch_size=8, seed=0)
    sampler = GeneSampler(view, cases, lp, neighbours, batch_size=4, upsample_size=upsample,
                          sparse_sample_size=4, seed=0, augmentation_k=3)
    task = GeneTask(view=view, val_cases=val, val_excluded=0, full_edges=view.edge_index(),
                    full_relations=view.rel, distances=distances, neighbours=neighbours,
                    task_loss_weight=0.6, lp_loss_weight=0.4)
    return view, sampler, task, cases


def test_binding_keeps_every_causal_gene_and_names_exclusions(trained):
    view, _, _, _ = _setup(trained)
    rows = [SimpleNamespace(patient_id="two", phenotype_ids=[0, 1], disease_id=0,
                            causal_gene_ids=[0, 1], gene_ids=[0, 1, 2]),
            SimpleNamespace(patient_id="off", phenotype_ids=[0], disease_id=0,
                            causal_gene_ids=[3], gene_ids=[0, 1]),
            SimpleNamespace(patient_id="none", phenotype_ids=[0], disease_id=0,
                            causal_gene_ids=None, gene_ids=[0])]
    binding = bind_gene_cases(view, rows)
    assert [c.case_id for c in binding.cases] == ["two"]
    assert binding.cases[0].truths.numel() == 2
    assert dict(binding.excluded) == {"off": "truth_not_in_candidates",
                                      "none": "no_causal_gene_or_candidate_list"}


def test_upsampling_adds_rare_genes_and_an_epoch_does_not_depend_on_the_last(trained):
    view, sampler, _, cases = _setup(trained)
    batch = next(iter(sampler))
    assert batch.counts["upsampled"] > 0
    by_id = {c.case_id: c for c in cases}
    order = torch.randperm(len(cases), generator=torch.Generator().manual_seed(0)).tolist()
    for row, i in enumerate(order[:4]):
        case = by_id[cases[i].case_id]
        listed = batch.gene_view[row][batch.gene_mask[row]]
        extra = listed[~torch.isin(listed, case.candidates)]
        assert torch.equal(listed[:case.candidates.numel()], case.candidates)
        assert extra.numel() <= 2 and not bool(torch.isin(extra, case.truths).any())
    # Regression 8: epoch 1 drawn after epoch 0 equals epoch 1 drawn fresh, as
    # after a resume (the counter is reset at each epoch start).
    sampler.set_epoch(0)
    list(iter(sampler))
    sampler.set_epoch(1)
    after = [b.gene_view for b in sampler]
    _, fresh, _, _ = _setup(trained)
    fresh.set_epoch(1)
    assert all(torch.equal(a, b.gene_view) for a, b in zip(after, fresh, strict=True))


def test_two_causal_genes_are_positives_and_the_joint_step_reaches_every_part(trained):
    view, sampler, task, cases = _setup(trained, upsample=0)
    two = dataclasses.replace(cases[0], truths=cases[0].candidates[:2])
    batch = sampler.batch([two, cases[1]], torch.Generator().manual_seed(0))
    assert int(batch.positive[0].sum()) == 2
    torch.manual_seed(0)
    bundle = build_bundle(_gene_spec(), view).train()
    step = task.training_step(bundle, batch, torch.device("cpu"))
    assert step.counts["task_anchors"] == 2
    expected = 0.6 * step.terms["task"] + 0.4 * step.terms.get("lp", 0.0)
    assert step.loss.item() == pytest.approx(expected, rel=1e-5)
    step.loss.backward()
    for part in (bundle.input, bundle.encoder, bundle.patient_encoder, bundle.task_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    assert bundle.task_head.decoder_weight.grad.abs().sum() > 0


def test_validation_ranks_each_case_own_list(trained):
    view, _, task, _ = _setup(trained)
    bundle = build_bundle(_gene_spec(), view)
    report = task.validate(bundle, torch.device("cpu"))
    assert report["ranked_units"] == len(task.val_cases) and 0 < report["pooled_mrr"] <= 1
    outcomes = task.rank_cases(bundle, task.val_cases, torch.device("cpu"))
    assert [o.candidates for o in outcomes] == [int(c.candidates.numel()) for c in task.val_cases]


def _gene_config(trained, run_dir, **kw):
    import train_model

    fields = dict(data_dir=str(trained["data_dir"]), run_dir=str(run_dir),
                  learning_stage="finetune", task="gene", finetune_from=str(trained["base"]),
                  num_epochs=1, batch_size=4, num_neighbors=[3, 3], sparse_sample_size=4,
                  device="cpu", save_top_k=1)
    fields.update(kw)
    explicit = set(fields) - {"data_dir", "run_dir"}
    return train_model.TrainConfig(**fields), explicit


def test_gene_finetune_from_a_base(trained, tmp_path):
    import train_model

    config, explicit = _gene_config(trained, tmp_path / "child")
    metrics = train_model.train(config, explicit)
    assert "val_pooled_mrr" in metrics
    last = torch.load(tmp_path / "child" / "checkpoints" / "last.pt", map_location="cpu",
                      weights_only=True)
    bundle, record = load_bundle(last, task="gene")
    assert record.operation == "finetune"
    assert record.loss_identity["task_loss"] == "multi_similarity"
    assert record.loss_identity["task_loss_weight"] == 0.6
    assert (record.scoring_identity["eta"], record.scoring_identity["sp_lower"],
            record.scoring_identity["sp_upper"]) == (0.5, 1.0, 6.84)
    assert record.data_roles["task_fixture"]["label"] == "functional fixture"
    tables = record.data_roles["derived_tables"]
    assert len(tables["sp_distances"]["digest"]) == 64
    assert tables["augmentation_neighbours"]["k"] == 3
    assert {"task_fixture", "task_train_samples", "task_val_samples"} <= set(
        last["training_input_digests"])
    assert last["config"]["requested"]["candidate_upsample_size"] == 2
    assert last["config"]["requested"]["learning_rate"] == 1e-5


def test_a_gene_finetune_without_its_fixture_is_refused_before_output(trained, tmp_path):
    import train_model
    from src.kg.task_fixture import TaskFixtureError

    bare = tmp_path / "ws"
    shutil.copytree(trained["data_dir"], bare, ignore=shutil.ignore_patterns("derived"))
    config, explicit = _gene_config(dict(trained, data_dir=bare), tmp_path / "child")
    with pytest.raises(TaskFixtureError, match="derive_task_fixture"):
        train_model.train(config, explicit)
    assert not (tmp_path / "child").exists()


def test_gene_measurement_over_the_fixture_split(trained, tmp_path):
    import json

    import measure_scorer
    import train_model

    config, explicit = _gene_config(trained, tmp_path / "child")
    train_model.train(config, explicit)
    out = tmp_path / "report.json"
    code = measure_scorer.main(["--mode", "gene", "--checkpoint",
                                str(tmp_path / "child" / "checkpoints" / "last.pt"),
                                "--data-dir", str(trained["data_dir"]), "--split", "val",
                                "--output", str(out), "--device", "cpu"])
    assert code == 0
    report = json.loads(out.read_text())
    manifest = report["manifest"]
    assert manifest["mode"] == "gene"
    assert manifest["candidate_construction"] == "each case's own candidate list, no upsampling"
    assert "eta=0.5" in manifest["score_semantics"] and "[1.0, 6.84]" in manifest["score_semantics"]
    assert "task_fixture" in manifest["artifact_digests"]
    metrics = report["authoritative_metrics"]
    assert 0 < metrics["pooled_mrr"] <= 1 and metrics["coverage"] == 1.0
