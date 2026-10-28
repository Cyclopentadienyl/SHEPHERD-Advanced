"""Patients-like-me: binding, F11 batches, self-exclusion by case identity, the
joint step, validation states and `scripts/train_model.py --task
patients_like_me` with its measurement (README A9, F9, F11, E1, E4, E5;
regression 10).

Module: tests/unit/test_patients_like_me.py
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
from src.models.bundle.metrics import NO_OTHER_POSITIVE, NO_RELIABLE_TRUTH, RANKED  # noqa: E402
from src.models.bundle.record import load_bundle  # noqa: E402
from src.models.bundle.spec import InputSpec, PatientEncoderSpec  # noqa: E402
from src.models.tasks.patients_like_me import (  # noqa: E402
    BoundPatient,
    anchor_groups,
    bind_patients,
    positive_matrix,
    rank_patients,
)
from src.models.tasks.scoring import full_view_vectors  # noqa: E402
from src.training.finetune_task import PatientsSampler, PatientsTask  # noqa: E402
from src.training.lp_task import BaseLPSampler  # noqa: E402
from tests.fixtures.registered_cohort import write_demo_workspace  # noqa: E402


def _base_spec():
    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=8),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=4, output=4, heads=2)
    ).validate()


def _plm_spec():
    spec = finetune_spec_from_parent(_base_spec(), "patients_like_me")
    return dataclasses.replace(spec, patient_encoder=PatientEncoderSpec(
        transformer_layers=1, transformer_heads=2, feedforward=16)).validate()


def _p(case_id, genes, phenotypes=(0,)):
    return BoundPatient(case_id, torch.tensor(list(phenotypes)), frozenset(genes))


def test_f11_anchors_need_two_others_sharing_a_gene():
    """Three patients share gene 7; a singleton (gene 9) and a pair (gene 8)
    do not anchor, and are counted as non-anchoring."""
    cases = [_p("a", {7}), _p("b", {7}), _p("c", {7, 8}), _p("d", {8}), _p("e", {9})]
    anchors, partners = anchor_groups(cases)
    assert anchors == [0, 1, 2]
    assert partners[2] == [0, 1, 3] and partners[3] == [2] and partners[4] == []


def test_positives_intersect_causal_genes_and_never_pair_a_case_with_itself():
    queries = [_p("a", {7}), _p("b", {7, 8})]
    reference = [_p("b", {7, 8}), _p("c", {8}), _p("a", {7})]
    assert positive_matrix(queries, reference).tolist() == [[True, False, False],
                                                            [False, True, True]]


def test_binding_keeps_genes_outside_the_view_and_refuses_repeated_ids():
    view = SimpleNamespace(view_ids=lambda t, local: torch.where(local < 3, local, -1),
                           node_types=["phenotype", "gene"])
    rows = [SimpleNamespace(patient_id="a", phenotype_ids=[0, 5], causal_gene_ids=[999]),
            SimpleNamespace(patient_id="b", phenotype_ids=[5], causal_gene_ids=[1]),
            SimpleNamespace(patient_id="c", phenotype_ids=[1], causal_gene_ids=None)]
    binding = bind_patients(view, rows)
    assert [c.case_id for c in binding.cases] == ["a", "c"]
    assert binding.cases[0].genes == frozenset({999}) and not binding.cases[1].genes
    assert binding.excluded == [("b", "no_phenotype_in_view")]
    assert binding.counts["no_causal_gene"] == 1
    with pytest.raises(ValueError, match="appears twice"):
        bind_patients(view, rows[:1] * 2)


def test_excluded_patients_keep_their_e4_state_and_their_partners_stay_relevant():
    """E4/E5: a patient left out of the reference keeps the state its truth
    gives it (no reliable truth, no other positive, or input excluded with
    every same-gene patient as a positive), and a bound patient's excluded
    partners are relevant items it could not rank."""
    view = SimpleNamespace(view_ids=lambda t, local: torch.where(local < 3, local, -1),
                           node_types=["phenotype", "gene"])
    row = lambda pid, phen, genes: SimpleNamespace(  # noqa: E731
        patient_id=pid, phenotype_ids=phen, causal_gene_ids=genes)
    rows = [row("a", [0], [7]), row("b", [1], [7]), row("z", [5], [7]),
            row("x", [5], None), row("y", [5], [99])]
    binding = bind_patients(view, rows)
    states = {o.query_id: (o.status, o.n_positives) for o in binding.excluded_outcomes()}
    assert states == {"z": ("input_excluded", 2), "x": (NO_RELIABLE_TRUTH, 0),
                      "y": (NO_OTHER_POSITIVE, 0)}
    assert binding.unranked_positives == {"a": 1, "b": 1}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    import train_model
    from derive_task_fixture import derive

    root = tmp_path_factory.mktemp("plm")
    data_dir = write_demo_workspace(root / "ws")
    derive(data_dir, distractors=4)
    train_model.train(train_model.TrainConfig(
        data_dir=str(data_dir), run_dir=str(root / "base"), learning_stage="base",
        model_spec=_base_spec().to_dict(), num_epochs=1, batch_size=32, num_neighbors=[4, 4],
        device="cpu", save_top_k=1),
        {"learning_stage", "model_spec", "num_epochs", "batch_size", "num_neighbors", "device",
         "save_top_k"})
    return {"data_dir": data_dir, "base": root / "base" / "checkpoints" / "last.pt"}


def _setup(trained):
    from src.kg.artifacts import read_split_manifest
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.kg.task_fixture import read_gene_fixture

    data_dir = trained["data_dir"]
    view = build_graph_view(read_graph_artifacts(data_dir, map_location="cpu").graph_data)
    train = split_facts(view, 0).mask("train")
    fixture = read_gene_fixture(data_dir,
                                manifest_sha256=read_split_manifest(data_dir).identity.sha256)
    cases = bind_patients(view, fixture.samples["train"].samples).cases
    val = bind_patients(view, fixture.samples["val"].samples).cases
    lp = BaseLPSampler(view.num_nodes, view.edge_index(train), view.rel[train],
                       view.edge_index(train), view.rel[train], sizes=[3, 3], batch_size=8, seed=0)
    sampler = PatientsSampler(view, cases, lp, batch_size=4, upsample_size=1,
                              sparse_sample_size=4, seed=0)
    task = PatientsTask(view=view, val_cases=val, val_excluded=[], full_edges=view.edge_index(),
                        full_relations=view.rel, task_loss_weight=0.3, lp_loss_weight=0.7)
    return view, sampler, task


def test_every_anchor_has_two_positives_in_its_batch_and_the_step_reaches_every_part(trained):
    view, sampler, task = _setup(trained)
    batch = next(iter(sampler))
    # F11: the loss's queries are exactly the batch's anchors, each with at
    # least two positives in the batch; the step averages over those alone.
    query = batch.query
    assert int(query.sum()) == batch.counts["anchors"] == batch.counts["queries"]
    assert bool((batch.positive[query].sum(dim=1) >= 2).all())
    assert not bool(batch.positive.diagonal().any())
    torch.manual_seed(0)
    bundle = build_bundle(_plm_spec(), view).train()
    step = task.training_step(bundle, batch, torch.device("cpu"))
    assert step.counts["task_queries"] == int(query.sum())
    expected = 0.3 * step.terms["task"] + 0.7 * step.terms.get("lp", 0.0)
    assert step.loss.item() == pytest.approx(expected, rel=1e-5)
    step.loss.backward()
    for part in (bundle.input, bundle.encoder, bundle.patient_encoder, bundle.task_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())


def test_a_partner_with_one_positive_is_a_reference_patient_not_a_query():
    """F11, overlapping genes: A={7,8} anchors with B={7} and C={8}; B and C
    each have one positive in the batch, so A is the only NCA query (review
    of 52cb900, P2-2)."""
    from src.models.tasks.losses import nca_loss

    cases = [_p("A", {7, 8}), _p("B", {7}), _p("C", {8})]
    view = SimpleNamespace(num_nodes=10, node_types=["gene", "phenotype", "disease"],
                           type_nodes=lambda t: {"gene": torch.tensor([7, 8]),
                                                 "phenotype": torch.tensor([0]),
                                                 "disease": torch.tensor([9])}[t])
    sampler = PatientsSampler(view, cases, None, batch_size=1, upsample_size=0,
                              sparse_sample_size=0, seed=0)
    # The LP draw is not under test; the sampler's own batch logic is.
    sampler._lp = lambda seeds, rng: SimpleNamespace(counts={}, seed_index=lambda ids: ids)
    assert [c.case_id for c in sampler.cases] == ["A"]
    batch = sampler.batch(sampler.cases, torch.Generator().manual_seed(0))
    rows = batch.positive.sum(dim=1).tolist()
    assert sorted(rows) == [1, 1, 2]
    assert batch.query.tolist() == [r == 2 for r in rows] and int(batch.query.sum()) == 1
    assert (batch.counts["queries"], batch.counts["non_anchoring_patients"]) == (1, 2)
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    loss = nca_loss(vectors[batch.query], vectors, batch.positive[batch.query], scale=1.0,
                    eps=1e-12, exclude=torch.eye(3, dtype=torch.bool)[batch.query])
    assert loss.units == 1


def test_declared_genes_outside_the_graph_pair_by_identifier_through_import_and_measurement(
        trained, tmp_path):
    """Review of 52cb900, P2-1: a declared causal gene the graph lacks is a
    truth, not "no causal gene"; two patients share it when they declared the
    same identifier (the importer's normalised form). Import, reader, binding
    and measurement, with the same identifier, different ones, a partial
    mapping and a patient that declares none."""
    from src.evaluation.cohort_registry import map_cases, parse_source
    from src.evaluation.task_measurement import run_patients_like_me_measurement
    from src.kg.storage.file_storage import read_samples_file

    view, _, _ = _setup(trained)
    case = lambda cid, genes: {"case_id": cid, "phenotypes": ["HP:0001250"],  # noqa: E731
                               "disease": "MONDO:0011073", **({"causal_genes": genes}
                                                              if genes else {})}
    source = {"format": "shepherd-cases/1", "cases": [
        case("a", ["GENE_X"]), case("b", [" gene_x"]), case("c", ["GENE_Y"]),
        case("d", ["SCN1A", "GENE_X"]), case("e", ["SCN1A"]), case("f", None)]}
    mapped = map_cases(parse_source(json.dumps(source).encode()),
                       {"phenotype": {"hpo:HP:0001250": 0},
                        "disease": {"mondo:MONDO:0011073": 0}}, "keep", {"SCN1A": 0})
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(mapped.samples))
    rows = read_samples_file(path, gene_fields=True).samples
    assert [r.causal_genes_unmapped for r in rows] == [["GENE_X"], ["GENE_X"], ["GENE_Y"],
                                                       ["GENE_X"], None, None]
    binding = bind_patients(view, rows)
    assert binding.counts["no_causal_gene"] == 1
    assert binding.counts["with_causal_genes_outside_graph"] == 4
    ids = [c.case_id for c in binding.cases]
    pairs = {(ids[i], ids[j]) for i, j in positive_matrix(binding.cases, binding.cases)
             .nonzero().tolist()}
    assert pairs == {("a", "b"), ("b", "a"), ("a", "d"), ("d", "a"), ("b", "d"), ("d", "b"),
                     ("d", "e"), ("e", "d")}
    result = run_patients_like_me_measurement(
        bundle=build_bundle(_plm_spec(), view), view=view, samples=rows,
        manifest=SimpleNamespace(amp_enabled=False, amp_dtype=None),
        device=torch.device("cpu"), batch_size=8)
    states = result.sampler_evidence["units_by_status"]
    assert (states["ranked"], states[NO_OTHER_POSITIVE], states[NO_RELIABLE_TRUTH]) == (4, 1, 1)
    assert result.sampler_evidence["conservative_denominator_pairs"] == 8


def test_every_member_node_is_a_seed_and_the_anchors_lists_are_upsampled(trained):
    """U9, U14: the batch's patients seed their phenotypes, causal genes,
    candidate genes and diseases, and each anchor gets its rare genes."""
    view, sampler, _ = _setup(trained)
    sampler.sparse_sample_size = 0
    sampler.start_epoch()
    anchors = sampler.cases[:4]
    batch = sampler.batch(anchors, torch.Generator().manual_seed(0))
    seeds = set(batch.lp.node_ids[:batch.lp.num_seeds].tolist())
    members = [p for p in sampler.patients
               if any(p.case_id == a.case_id for a in anchors)]
    for p in members:
        assert set(p.truths.tolist()) <= seeds and set(p.candidates.tolist()) <= seeds
        assert p.disease < 0 or p.disease in seeds
    assert batch.counts["upsampled"] > 0


def test_a_cohort_without_causal_genes_is_refused_before_anything_is_written(trained):
    from src.evaluation.task_measurement import run_patients_like_me_measurement

    view, _, _ = _setup(trained)
    rows = [SimpleNamespace(patient_id=f"p{i}", phenotype_ids=[0], causal_gene_ids=None)
            for i in range(3)]
    with pytest.raises(ValueError, match="no patient of the cohort declares a causal gene"):
        run_patients_like_me_measurement(bundle=build_bundle(_plm_spec(), view), view=view,
                                         samples=rows, manifest=None,
                                         device=torch.device("cpu"), batch_size=8)


def test_ranking_excludes_the_self_by_case_id_whatever_the_order(trained):
    view, _, task = _setup(trained)
    bundle = build_bundle(_plm_spec(), view)
    z = full_view_vectors(bundle, view.edge_index(), view.rel, torch.device("cpu"))
    val = task.val_cases
    forward = {o.query_id: o for o in rank_patients(bundle, z, val, val)}
    backward = {o.query_id: o for o in rank_patients(bundle, z, val, list(reversed(val)))}
    assert forward.keys() == backward.keys()
    for case_id, outcome in forward.items():
        assert outcome.status == backward[case_id].status
        assert sorted(outcome.ranks) == sorted(backward[case_id].ranks)
        if outcome.status == RANKED:
            assert outcome.candidates == len(val) - 1


def test_states_are_kept_apart(trained):
    view, _, _ = _setup(trained)
    lonely, no_truth = _p("x", {1}), _p("y", set())
    bundle = build_bundle(_plm_spec(), view)
    z = full_view_vectors(bundle, view.edge_index(), view.rel, torch.device("cpu"))
    outcomes = rank_patients(bundle, z, [lonely, no_truth], [lonely, no_truth])
    assert [o.status for o in outcomes] == [NO_OTHER_POSITIVE, NO_RELIABLE_TRUTH]


def _plm_config(trained, run_dir, **kw):
    import train_model

    fields = dict(data_dir=str(trained["data_dir"]), run_dir=str(run_dir),
                  learning_stage="finetune", task="patients_like_me",
                  finetune_from=str(trained["base"]), num_epochs=1, batch_size=4,
                  num_neighbors=[3, 3], sparse_sample_size=4, device="cpu", save_top_k=1)
    fields.update(kw)
    return train_model.TrainConfig(**fields), set(fields) - {"data_dir", "run_dir"}


def test_patients_like_me_finetune_and_measurement(trained, tmp_path):
    import measure_scorer
    import train_model

    config, explicit = _plm_config(trained, tmp_path / "child")
    metrics = train_model.train(config, explicit)
    assert "val_pooled_mrr" in metrics
    last_path = tmp_path / "child" / "checkpoints" / "last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    _, record = load_bundle(last, task="patients_like_me")
    assert record.loss_identity["task_loss"] == "nca"
    assert record.loss_identity["task_loss_weight"] == 0.3
    assert record.scoring_identity["task"] == "patients_like_me"
    assert record.data_roles["task_fixture"]["label"] == "functional fixture"
    assert last["config"]["requested"]["learning_rate"] == 5e-5

    out = tmp_path / "report.json"
    code = measure_scorer.main(["--mode", "patients_like_me", "--checkpoint", str(last_path),
                                "--data-dir", str(trained["data_dir"]), "--split", "val",
                                "--output", str(out), "--device", "cpu"])
    assert code == 0
    report = json.loads(out.read_text())
    assert report["manifest"]["mode"] == "patients_like_me"
    assert "self excluded" in report["manifest"]["candidate_construction"]
    assert "task_fixture" in report["manifest"]["artifact_digests"]
