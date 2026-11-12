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


def test_another_seed_draws_another_stream():
    """The half the restart test cannot show: a constant stream reproduces too.
    The run's seed is what the sampler's own generator starts from; no worker
    process draws anything."""
    edges = _ei([(i, (i + 1) % 10) for i in range(10)] + [((i + 1) % 10, i) for i in range(10)])
    rels = torch.tensor([0] * 10 + [1] * 10)
    a = _sampler(10, edges, rels, batch_size=4, seed=5)
    b = _sampler(10, edges, rels, batch_size=4, seed=6)
    assert [x.node_ids.tolist() for x in a] != [y.node_ids.tolist() for y in b]


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


def test_validation_takes_its_pairs_and_scores_from_the_one_lp_core():
    """Base validation is the LP core's (README §3.2): its pairs are
    `fixed_lp_pairs` of the val edges under the split's seed in its batches,
    and its dict is `score_lp_pairs`' metrics over the training edges, then
    its counts, which name the edges dropped for having no negative."""
    from src.models.bundle.lp_head import fixed_lp_pairs, score_lp_pairs

    view, split = _toy_view()
    task, _ = _task_and_sampler(view, split, seed=2)
    train, val = split.mask("train"), split.mask("val")
    pairs = fixed_lp_pairs(view.edge_index(val), view.rel[val], seed=2, batch_size=4)
    assert task.val_pairs.digest == pairs.digest
    assert task.val_counts == {"val_edges": pairs.counts["edges"],
                               "val_without_negative": pairs.counts["without_negative"]}
    torch.manual_seed(0)
    bundle = build_bundle(_spec(), view)
    scored = score_lp_pairs(bundle, pairs, view.edge_index(train), view.rel[train],
                            torch.device("cpu"))
    assert task.validate(bundle, torch.device("cpu")) == {**scored.metrics, **task.val_counts}


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


def test_validation_calls_each_hook_once_and_tracks_the_monitored_metric():
    """The Trainer's validation lifecycle around a task's report, which stays when
    the ShepherdGNN pass retires (moved in K8-7 from
    test_trainer_validation_characterization.py §3-4): one begin and one end
    hook per call, the end hook handed exactly the metrics returned; the
    monitored metric's best value and the epoch it came from; an equal value is
    not an improvement in mode "max"; every call is kept in the history the
    checkpoint's `state` records; a monitored name the task does not report
    leaves the best unset and is still kept in the history."""
    from src.training.callbacks import Callback
    from src.training.trainer import Trainer, TrainerConfig

    class Recorder(Callback):
        def __init__(self):
            self.events = []

        def on_validation_begin(self, trainer, **kwargs):
            self.events.append("begin")

        def on_validation_end(self, trainer, logs, **kwargs):
            self.events.append(("end", dict(logs)))

    view, split = _toy_view()
    task, sampler = _task_and_sampler(view, split)
    reports = iter([0.5, 0.5, 0.75, 0.25])
    task.validate = lambda model, device: {"lp_accuracy": next(reports)}
    recorder = Recorder()
    config = TrainerConfig(device="cpu", num_epochs=1, scheduler_type="none", use_amp=False,
                           early_stopping=False, early_stopping_monitor=task.selection_metric,
                           early_stopping_mode="max", seed=0)
    trainer = Trainer(model=build_bundle(_spec(), view), train_dataloader=sampler,
                      config=config, callbacks=[recorder], task=task)

    first = trainer._validate(3)
    assert recorder.events == ["begin", ("end", first)]
    assert (trainer.state.best_metric, trainer.state.best_epoch) == (0.5, 3)
    trainer._validate(4)
    assert trainer.state.best_epoch == 3, "an equal value is not an improvement"
    trainer._validate(5)
    trainer._validate(6)
    assert (trainer.state.best_metric, trainer.state.best_epoch) == (0.75, 5)
    assert [h["val_lp_accuracy"] for h in trainer.state.val_metric_history] == [
        0.5, 0.5, 0.75, 0.25]
    assert recorder.events[::2] == ["begin"] * 4
    assert [logs for _, logs in recorder.events[1::2]] == trainer.state.val_metric_history

    trainer.config.early_stopping_monitor = "val_not_reported"
    trainer.state.best_metric = None
    task.validate = lambda model, device: {"lp_accuracy": 0.9}
    trainer._validate(7)
    assert trainer.state.best_metric is None
    assert len(trainer.state.val_metric_history) == 5


def test_a_trainer_given_no_monitor_selects_by_its_task_s_selection_metric(tmp_path):
    """C1c (K8-7 map default 1): `TrainerConfig.early_stopping_monitor` is None
    unless stated, and the Trainer then selects by its task's own selection
    metric (`selection_metric`; the base stage's is `val_lp_accuracy`): early
    stopping, the best-k checkpoints and their file names, and the best value
    validation tracks all follow it. The config handed in is left as given."""
    from src.training.trainer import Trainer, TrainerConfig

    view, split = _toy_view()
    task, sampler = _task_and_sampler(view, split)
    config = TrainerConfig(device="cpu", num_epochs=1, scheduler_type="none", use_amp=False,
                           checkpoint_dir=str(tmp_path / "ckpt"),
                           log_dir=str(tmp_path / "logs"), seed=0)
    assert config.early_stopping_monitor is None, "no metric is named by default"

    trainer = Trainer(model=build_bundle(_spec(), view), train_dataloader=sampler,
                      config=config, task=task)

    assert task.selection_metric == "val_lp_accuracy"
    assert trainer.config.early_stopping_monitor == "val_lp_accuracy"
    assert config.early_stopping_monitor is None, "the caller's config is unchanged"
    selecting = {type(c).__name__: c.config for c in trainer.callbacks.callbacks
                 if type(c).__name__ in ("EarlyStopping", "ModelCheckpoint")}
    assert {name: c.monitor for name, c in selecting.items()} == {
        "EarlyStopping": "val_lp_accuracy", "ModelCheckpoint": "val_lp_accuracy"}
    assert "{val_lp_accuracy:.4f}" in selecting["ModelCheckpoint"].filename

    task.validate = lambda model, device: {"lp_accuracy": 0.75, "lp_roc_auc": 0.5}
    trainer._validate(0)
    assert (trainer.state.best_metric, trainer.state.best_epoch) == (0.75, 0)
