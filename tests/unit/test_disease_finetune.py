"""The disease finetune: binding, batches, the joint step and validation, and
`scripts/train_model.py --stage finetune` from a base (README A6, A7, A10,
C3, F7, U6, U9, G4; contract §9.2; regressions 1, 2, 7, 8). A parent whose
record names the graph two ways is refused before a run is written (K8-4
review P2-2).

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
from src.models.bundle.metrics import INPUT_EXCLUDED, QueryOutcome  # noqa: E402
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
    task = DiseaseTask(view=view, val_cases=cases.cases[:4], val_excluded=[QueryOutcome("x", INPUT_EXCLUDED, n_positives=1)],
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


def test_the_patient_batches_follow_the_run_seed(workspace):
    """The same seed draws the same patient batches, another seed others: each
    batch's seeds (its patients' nodes, candidates and sparse sample) are the
    sampler's own draws. The neighbours PyG's sampler adds follow the process's
    generator, which training seeds (test_training_seed.py), so they are left out."""
    view, split = _view(workspace)
    train = split.mask("train")
    cases = bind_disease_cases(view, _samples(*[(f"p{i}", [i, i + 1], i % 8)
                                                for i in range(10)])).cases

    def batches(seed):
        lp = BaseLPSampler(view.num_nodes, view.edge_index(train), view.rel[train],
                           view.edge_index(train), view.rel[train], sizes=[3, 3],
                           batch_size=8, seed=seed)
        sampler = FinetuneSampler(view, cases, lp, batch_size=4, candidate_sample_size=3,
                                  sparse_sample_size=5, seed=seed)
        return [b.lp.node_ids[:b.lp.num_seeds].tolist() for b in sampler]

    assert batches(0) == batches(0)
    assert batches(0) != batches(1)


def test_a_node_that_is_not_a_seed_of_the_batch_is_refused_never_indexed(workspace):
    """A truth or candidate the batch did not seed has no row of its own; asked
    for, it is refused, never given another seed's index (or one past the last)."""
    _, sampler, _ = _task_setup(workspace)
    lp = next(iter(sampler)).lp
    seeds = lp.node_ids[:lp.num_seeds]
    assert lp.node_ids.numel() > lp.num_seeds, "every node is a seed, so this proves nothing"
    for not_a_seed in (lp.node_ids[lp.num_seeds:lp.num_seeds + 1], seeds[-1:] + 1):
        assert not bool(torch.isin(not_a_seed, seeds).any())
        with pytest.raises(ValueError, match="not a seed of this batch"):
            lp.seed_index(not_a_seed)


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


def test_validation_ranks_every_disease_with_average_ranks(workspace, monkeypatch):
    view, _, task = _task_setup(workspace)
    bundle = build_bundle(_finetune_spec(), view)
    report = task.validate(bundle, torch.device("cpu"))
    assert report["ranked_units"] == 4 and report["units"] == 5
    assert report["coverage"] == pytest.approx(4 / 5)
    assert 0 < report["pooled_mrr"] <= 1
    assert report["conservative_pooled_mrr"] < report["pooled_mrr"]
    outcomes = task.rank_cases(bundle, task.val_cases, torch.device("cpu"))
    assert all(o.candidates == int(view.type_nodes("disease").numel()) for o in outcomes)
    # A program-correctness check of chunking and padding, not a requirement that
    # runs or batch sizes give equal results: cases of one, two and three
    # phenotypes, padded together in one chunk, score as each does alone within
    # float rounding (the tolerance below; why `batch_size` is recorded).
    from src.models.tasks.scoring import full_view_vectors, score_cases

    mixed = bind_disease_cases(view, _samples(("m1", [3], 0), ("m2", [4, 5], 1),
                                              ("m3", [6, 7, 8], 2))).cases
    z = full_view_vectors(bundle, view.edge_index(), view.rel, torch.device("cpu"))
    diseases = view.type_nodes("disease")
    torch.testing.assert_close(score_cases(bundle, z, mixed, diseases),
                               score_cases(bundle, z, mixed, diseases, case_chunk=1),
                               rtol=0, atol=1e-5)
    # Exact ties share their average rank in the ranker validation and measurement
    # share (`rank_cases`, E3): the `average-rank/v1` every task manifest records.
    # With every disease scored alike, each truth ranks (n + 1) / 2 wherever it
    # stands in the view, never by its place in the candidate order.
    import src.models.tasks.scoring as scoring

    n = int(view.type_nodes("disease").numel())
    monkeypatch.setattr(scoring, "score_cases",
                        lambda bundle, z, cases, candidates, **_: torch.zeros(len(cases), n))
    tied = task.rank_cases(bundle, task.val_cases, torch.device("cpu"))
    assert [o.ranks for o in tied] == [[(n + 1) / 2]] * len(task.val_cases)
    # A higher score ranks first (E3's direction): a truth scored above every
    # other disease ranks 1 in the same ranker.
    monkeypatch.setattr(scoring, "score_cases", lambda bundle, z, cases, candidates, **_:
                        torch.stack([torch.isin(candidates, c.truths).float() for c in cases]))
    top = task.rank_cases(bundle, task.val_cases, torch.device("cpu"))
    assert [o.ranks for o in top] == [[1.0]] * len(task.val_cases)


def test_validation_reports_the_pooled_mrr_an_independent_computation_gives(workspace):
    """The value the Trainer logs and selects by (`val_pooled_mrr`) against a
    computation written here, not taken from the task scorer: node vectors by
    the bundle's own full-view pass over every edge of the view (G4), each
    case's patient vector by its patient encoder and patient tower, every
    disease through the disease tower, the score the negative squared distance
    of the L2-normalised pair, and each truth's rank counted (blueprint §3.2's
    oracle rule; the role `test_trainer_validation_values.py` held for the
    ShepherdGNN pass until K8-7)."""
    view, _, task = _task_setup(workspace)
    torch.manual_seed(0)
    bundle = build_bundle(_finetune_spec(), view)
    report = task.validate(bundle, torch.device("cpu"))

    bundle.eval()
    with torch.no_grad():
        z = bundle.node_vectors_full(view.edge_index(), edge_type=view.rel)
        diseases = view.type_nodes("disease")
        candidates = bundle.task_head.disease_tower(z[diseases]).double()
        candidates = candidates / candidates.norm(dim=1, keepdim=True)
        reciprocal = []
        for case in task.val_cases:
            mask = torch.ones(1, case.phenotypes.numel(), dtype=torch.bool)
            pooled, _ = bundle.patient_encoder(z[case.phenotypes].unsqueeze(0), mask)
            patient = bundle.task_head.patient_tower(pooled)[0].double()
            patient = patient / patient.norm()
            scores = [-float(((patient - c) ** 2).sum()) for c in candidates]
            at = int((diseases == case.truths[0]).nonzero())
            others = scores[:at] + scores[at + 1:]
            assert all(abs(s - scores[at]) > 1e-4 for s in others), "no near tie at the truth"
            reciprocal.append(1.0 / (1 + sum(s > scores[at] for s in others)))
    assert report["pooled_mrr"] == pytest.approx(sum(reciprocal) / len(reciprocal))
    assert report["conservative_pooled_mrr"] == pytest.approx(
        sum(reciprocal) / (len(reciprocal) + 1))


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


def test_a_setting_stated_at_its_default_is_recorded_as_the_operator_s(workspace, tmp_path):
    """A finetune that states `use_amp=True`, TrainConfig's own default and not
    the disease recipe's value (False), records it as the operator set it:
    requested True, in the recipe record's `operator_set`, not filled from the
    recipe. What the operator stated is the set the caller passes, never
    inferred from which values differ from a default."""
    import train_model

    parent = _base_run(workspace, tmp_path / "base")
    config, explicit = _finetune_config(workspace, tmp_path / "child", parent, num_epochs=1,
                                        use_amp=True)
    assert config.use_amp == train_model.TrainConfig().use_amp, "stated at its default"
    train_model.train(config, explicit)

    last = torch.load(tmp_path / "child" / "checkpoints" / "last.pt", map_location="cpu",
                      weights_only=True)
    recipe = last["config"]["effective"]["loaders"]["train"]["recipe"]
    assert last["config"]["requested"]["use_amp"] is True
    assert "use_amp" in recipe["operator_set"]
    assert "use_amp" not in recipe["reference_filled"]


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
    with pytest.raises(ValueError, match="resume_from and finetune_from name two parents"):
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


def _contradicting(source, target, part):
    """`source` written to `target` with only `part`'s ``kg`` claim changed,
    and the claim it no longer matches."""
    payload = torch.load(source, map_location="cpu", weights_only=True)
    other = "training_input_digests" if part == "data_roles" else "data_roles"
    kept = payload[other]["kg"]
    assert payload[part]["kg"] == kept
    payload[part]["kg"] = "0" * 64
    torch.save(payload, target)
    return kept


def test_a_base_whose_training_record_kg_contradicts_its_data_roles_is_refused_before_the_child_run_is_written(
        workspace, tmp_path):
    """The reviewer's K8-4 P2-2, training counterexample (probe.py,
    ``finetune_conflicting_kg``): a real base whose ``data_roles.kg`` is right
    and whose ``training_input_digests.kg`` is another digest. The finetune
    compares only ``data_roles.kg`` with the workspace, so before the fix a
    child trained and recorded the contradicting parent. Now the shared record
    reader refuses the parent before the child's run directory is written."""
    import train_model
    from src.utils.run_plan import RunPlanRefused

    base = _base_run(workspace, tmp_path / "base")
    conflicting = tmp_path / "conflicting.pt"
    kept = _contradicting(base, conflicting, "training_input_digests")
    config, explicit = _finetune_config(workspace, tmp_path / "child", conflicting, num_epochs=1)
    with pytest.raises(RunPlanRefused, match="the record contradicts itself") as refused:
        train_model.train(config, explicit)
    assert (f"data_roles.kg {kept[:12]}... differs from training_input_digests.kg "
            f"{'0' * 12}...") in str(refused.value)
    assert not (tmp_path / "child").exists()


def test_a_resume_parent_whose_data_roles_kg_contradicts_its_training_record_is_refused_before_the_run_is_written(
        workspace, tmp_path):
    """The other direction, on the resume entry: a paused child whose
    ``data_roles.kg`` is another digest. Resume compares the training record
    with the workspace and never read ``data_roles.kg``, so before the fix the
    run went on from it. Now it is refused before the new run is written."""
    import train_model
    from src.utils.run_plan import RunPlanRefused

    parent = _base_run(workspace, tmp_path / "base")
    config, explicit = _finetune_config(workspace, tmp_path / "a", parent, num_epochs=2,
                                        pause_after_epoch=1)
    train_model.train(config, explicit)
    paused = tmp_path / "a" / "checkpoints" / "last.pt"
    kept = _contradicting(paused, paused, "data_roles")
    resumed = train_model.TrainConfig(data_dir=str(workspace), run_dir=str(tmp_path / "b"),
                                      learning_stage="finetune", task="disease",
                                      resume_from=str(paused), device="cpu")
    with pytest.raises(RunPlanRefused, match="the record contradicts itself") as refused:
        train_model.train(resumed, {"learning_stage", "task", "resume_from", "device"})
    assert (f"data_roles.kg {'0' * 12}... differs from training_input_digests.kg "
            f"{kept[:12]}...") in str(refused.value)
    assert not (tmp_path / "b").exists()
