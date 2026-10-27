"""The base stage's sampler, step and validation, and the Trainer delegating to
them (README C4, F1, F3, F5, F6, G4; §8 regressions 2 and 3).

Module: tests/unit/test_base_lp_task.py
"""
from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from src.kg.graph_view import build_graph_view, split_facts  # noqa: E402
from src.models.bundle import build_bundle, reference_spec  # noqa: E402
from src.models.bundle.record import ModelRecord, load_bundle  # noqa: E402
from src.models.bundle.spec import InputSpec  # noqa: E402
from src.training.lp_task import BaseLPSampler, BaseLPTask  # noqa: E402


def _ei(pairs):
    return torch.tensor(pairs, dtype=torch.long).T.reshape(2, -1)


def _sampler(n, edges, rels, sizes=(-1, -1), batch_size=8, seed=0, mp=None):
    mp_edges, mp_rels = mp if mp is not None else (edges, rels)
    return BaseLPSampler(n, mp_edges, mp_rels, edges, rels, sizes=list(sizes),
                         batch_size=batch_size, seed=seed)


# --------------------------------------------------------------------------- C4 / F6


def test_one_neighbour_source_enumerates_both_draws():
    # Node 0 has one supervision edge 0→1 (and its twin 1→0).
    edges = _ei([(0, 1), (1, 0)])
    rels = torch.tensor([0, 1])
    sampler = _sampler(2, edges, rels)
    seen = {"real": 0, "self": 0}
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        b = sampler.batch(torch.tensor([0]), g)
        if b.counts["self_draws"]:
            assert b.counts["lp_positives"] == 0
            seen["self"] += 1
        else:
            assert b.counts["lp_positives"] == 1
            assert int(b.node_ids[b.head[0]]) == 0 and int(b.node_ids[b.tail[0]]) == 1
            assert int(b.relation[0]) == 0
            seen["real"] += 1
    # Probability 1/2 each: both happen, neither dominates.
    assert 60 < seen["real"] < 140 and 60 < seen["self"] < 140


def test_isolated_source_draws_itself_and_is_counted():
    edges = _ei([(0, 1), (1, 0)])
    sampler = _sampler(3, edges, torch.tensor([0, 1]))
    b = sampler.batch(torch.tensor([2]), torch.Generator().manual_seed(0))
    assert b.counts == {"source_visits": 1, "self_draws": 1, "isolated_sources": 1,
                        "drawn_pairs": 0, "lp_positives": 0, "lp_without_negative": 0,
                        "edges_filtered": 0}


def test_every_relation_between_the_drawn_pair_is_a_positive():
    # 0→1 by two relations; one draw makes two positives.
    edges = _ei([(0, 1), (0, 1), (1, 0), (1, 0)])
    rels = torch.tensor([0, 2, 1, 3])
    sampler = _sampler(2, edges, rels)
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        b = sampler.batch(torch.tensor([0]), g)
        if not b.counts["self_draws"]:
            assert sorted(b.relation.tolist()) == [0, 2]
            return
    pytest.fail("no real draw in 50 tries")


def test_an_epoch_draws_the_same_stream_after_a_restart():
    edges = _ei([(i, (i + 1) % 10) for i in range(10)] + [((i + 1) % 10, i) for i in range(10)])
    rels = torch.tensor([0] * 10 + [1] * 10)
    a = _sampler(10, edges, rels, batch_size=4, seed=5)
    b = _sampler(10, edges, rels, batch_size=4, seed=5)
    a.set_epoch(3)
    b.set_epoch(3)
    for x, y in zip(a, b):
        assert torch.equal(x.node_ids, y.node_ids) and torch.equal(x.negative, y.negative)
        assert torch.equal(x.edge_index, y.edge_index)


def test_each_epoch_visits_every_node_once():
    edges = _ei([(i, (i + 1) % 10) for i in range(10)] + [((i + 1) % 10, i) for i in range(10)])
    sampler = _sampler(10, edges, torch.zeros(20, dtype=torch.long), batch_size=3)
    visits = sum(b.counts["source_visits"] for b in sampler)
    assert visits == 10 and len(sampler) == 4


# --------------------------------------------------------------------------- F5


def test_drawn_pair_edges_leave_message_passing_both_directions():
    # Triangle 0-1-2 with twins; the message-passing graph is the same edges.
    pairs = [(0, 1), (1, 2), (2, 0)]
    edges = _ei(pairs + [(b, a) for a, b in pairs])
    rels = torch.tensor([0, 0, 0, 1, 1, 1])
    sampler = _sampler(3, edges, rels)
    g = torch.Generator().manual_seed(3)
    for _ in range(50):
        b = sampler.batch(torch.tensor([0]), g)
        if b.counts["self_draws"]:
            continue
        t = int(b.node_ids[b.tail[0]])
        gsrc = b.node_ids[b.edge_index[0]].tolist()
        gdst = b.node_ids[b.edge_index[1]].tolist()
        for s, d in zip(gsrc, gdst):
            assert {s, d} != {0, t}
        assert b.counts["edges_filtered"] >= 2
        assert sum(b.edges_per_hop) == b.edge_index.size(1)
        return
    pytest.fail("no real draw in 50 tries")


# --------------------------------------------------------------------------- task on a toy view


def _toy_view():
    n = 40
    pairs = [(i, (i + 1) % n) for i in range(n)] + [(i, (i + 7) % n) for i in range(0, n, 2)]
    data = {"num_nodes_dict": {"gene": n},
            "edge_index_dict": {("gene", "link", "gene"): _ei(pairs),
                                ("gene", "rev_link", "gene"): _ei([(b, a) for a, b in pairs])}}
    view = build_graph_view(data)
    return view, split_facts(view, seed=1)


def _spec():
    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=8),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=4, output=4, heads=2)
    ).validate()


def _task_and_sampler(view, split, seed=0):
    train, val = split.mask("train"), split.mask("val")
    task = BaseLPTask(num_nodes=view.num_nodes, train_edges=view.edge_index(train),
                      train_relations=view.rel[train], val_edges=view.edge_index(val),
                      val_relations=view.rel[val], seed=seed, val_batch_size=4)
    sampler = _sampler(view.num_nodes, view.edge_index(train), view.rel[train],
                       sizes=(5, 5), batch_size=6, seed=seed)
    return task, sampler


def test_validation_is_fixed_and_passes_messages_over_training_edges():
    view, split = _toy_view()
    task, _ = _task_and_sampler(view, split)
    torch.manual_seed(0)
    bundle = build_bundle(_spec(), view)
    a = task.validate(bundle, torch.device("cpu"))
    b = task.validate(bundle, torch.device("cpu"))
    assert a == b
    assert a["val_edges"] == int(split.mask("val").sum())
    # The vectors validation uses are the training-edge full view: recomputed
    # from them here, the counts agree.
    bundle.eval()
    z = bundle.node_vectors_full(view.edge_index(split.mask("train")))
    pos = bundle.lp_head.raw_score(z[task.val_head], task.val_rel, z[task.val_tail])
    neg = bundle.lp_head.raw_score(z[task.val_head], task.val_rel, z[task.val_neg_tail])
    correct = int((pos > 0).sum()) + int((neg <= 0).sum())
    assert a["lp_accuracy"] == pytest.approx(correct / (pos.numel() + neg.numel()))
    assert bool((task.val_neg_tail != task.val_tail).all())


def test_a_split_whose_validation_has_no_negative_is_refused():
    edges = _ei([(0, 1), (1, 0)])
    with pytest.raises(ValueError, match="cannot be validated"):
        BaseLPTask(num_nodes=2, train_edges=edges, train_relations=torch.tensor([0, 1]),
                   val_edges=edges, val_relations=torch.tensor([0, 1]), seed=0)


def test_one_step_reaches_table_layers_and_drawn_relations():
    view, split = _toy_view()
    task, sampler = _task_and_sampler(view, split)
    torch.manual_seed(0)
    bundle = build_bundle(_spec(), view).train()
    batch = next(iter(sampler))
    step = task.training_step(bundle, batch, torch.device("cpu"))
    assert step.loss is not None
    step.loss.backward()
    assert bundle.input.table.weight.grad.abs().sum() > 0
    for conv in bundle.encoder.convs:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in conv.parameters())
    # A relation is in the loss when it has a positive with a negative: two
    # distinct tails of it in the batch (F1). Only those vectors move.
    in_loss = {r for r in set(batch.relation.tolist())
               if batch.tail[batch.relation == r].unique().numel() > 1}
    assert in_loss == set(batch.relation[batch.negative >= 0].tolist())
    ok = batch.negative >= 0
    assert bool((batch.tail[batch.negative[ok]] != batch.tail[ok]).all())
    assert in_loss
    for r in range(len(view.relations)):
        touched = bool(bundle.lp_head.relation_weights.grad[r].abs().sum() > 0)
        assert touched == (r in in_loss)


def test_step_without_a_pair_makes_no_loss():
    view, split = _toy_view()
    task, sampler = _task_and_sampler(view, split)
    bundle = build_bundle(_spec(), view)
    # An isolated-like batch: a source whose only draw is itself.
    g = torch.Generator().manual_seed(0)
    batch = sampler.batch(torch.tensor([0]), g)
    batch.negative = torch.full_like(batch.negative, -1)
    step = task.training_step(bundle, batch, torch.device("cpu"))
    assert step.loss is None and step.counts["lp_pairs"] == 0


def _trainer(tmp_path, bundle, task, sampler, epochs=2):
    from src.training.trainer import Trainer, TrainerConfig

    config = TrainerConfig(device="cpu", num_epochs=epochs, learning_rate=1e-3, weight_decay=0.0,
                           scheduler_type="none", use_amp=False, early_stopping=False,
                           early_stopping_monitor=task.selection_metric,
                           checkpoint_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"),
                           save_top_k=1, seed=0)
    return Trainer(model=bundle, train_dataloader=sampler, config=config, task=task)


def test_trainer_runs_the_base_stage_and_writes_bundle_checkpoints(tmp_path):
    view, split = _toy_view()
    task, sampler = _task_and_sampler(view, split)
    torch.manual_seed(0)
    spec = _spec()
    bundle = build_bundle(spec, view)
    trainer = _trainer(tmp_path, bundle, task, sampler)
    trainer.model_record = ModelRecord(
        model_spec=spec, graph_view=bundle.view, operation="new", family_id="run-x",
        data_roles={"graph_view": view.digest, "split": split.digest},
        graph_exposure={"base": {"message_passing": "train", "supervision": "train",
                                 "validation_message_passing": "train"}},
        loss_identity={"terms": ["lp"], "lp_loss_weight": 1.0})
    before = bundle.input.table.weight.detach().clone()
    final = trainer.train()
    assert "val_lp_accuracy" in final and "val_lp_roc_auc" in final
    assert not torch.equal(before, bundle.input.table.weight)
    names = [type(c).__name__ for c in trainer.callbacks.callbacks]
    assert "EarlyStopping" not in names
    assert trainer.state.completed_epochs == 2
    last = torch.load(tmp_path / "ckpt" / "last.pt", map_location="cpu", weights_only=True)
    again, record = load_bundle(last, learning_stage="base", view_digest=view.digest)
    assert record.data_roles["split"] == split.digest
    best = [p for p in (tmp_path / "ckpt").glob("model-*.pt")]
    assert len(best) == 1 and "val_lp_accuracy" not in best[0].name


def test_trainer_skips_steps_with_nothing_to_learn(tmp_path):
    view, split = _toy_view()
    task, _ = _task_and_sampler(view, split)
    bundle = build_bundle(_spec(), view)

    class Empty:
        def __iter__(self):
            g = torch.Generator().manual_seed(0)
            sampler = _sampler(view.num_nodes, view.edge_index(split.mask("train")),
                               view.rel[split.mask("train")], sizes=(5, 5))
            b = sampler.batch(torch.tensor([0]), g)
            b.negative = torch.full_like(b.negative, -1)
            yield b

        def __len__(self):
            return 1

    trainer = _trainer(tmp_path, bundle, task, Empty(), epochs=1)
    before = {k: v.clone() for k, v in bundle.state_dict().items()}
    metrics = trainer._train_epoch(0)
    assert metrics["train_steps_skipped"] == 1
    assert trainer.state.optimizer_updates == 0
    for k, v in bundle.state_dict().items():
        assert torch.equal(before[k], v)


def test_an_undefined_validation_metric_is_logged_as_none(tmp_path):
    view, split = _toy_view()
    task, sampler = _task_and_sampler(view, split)
    bundle = build_bundle(_spec(), view)
    trainer = _trainer(tmp_path, bundle, task, sampler, epochs=1)
    task.validate = lambda model, device: {"lp_accuracy": 0.5, "lp_roc_auc": None, "note": "x"}
    logs = trainer._validate(0)
    assert logs == {"val_lp_accuracy": 0.5, "val_lp_roc_auc": None}
