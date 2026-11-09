"""
P0-5: a base learns and passes messages on the train part, and validates on the val part.
========================================================================================
`docs/working/research-comparison/README.md` §5 Stage 1 (behavioural pin
P0-5), before the shared LP core and the shared graph input (F6) are wired:
a tiny CPU base run through `scripts/train_model.py`'s own `train` (the one
trained-child helper, `tests/fixtures/trained_children.train_children`, at
the demo's sizes on the demo workspace), with the bundle's two message-passing
entries, the base step and the base task's construction observed (each
wrapped, then called as it was):

- every edge given to message passing in training
  (`ModelBundle.node_vectors_sampled`) is an edge of the split's train part;
- every supervised pair (every positive of every step,
  `BaseLPTask.training_step`) is an edge of the train part;
- validation's message passing (`ModelBundle.node_vectors_full`, every call)
  uses exactly the train part's edges, and its fixed pairs
  (`val_head`, `val_rel`, `val_tail`) are edges of the val part.

The split is the run's own record (``graph_split.pt``), checked to be the
split `split_facts` makes of the view the run recorded. An edge is named by
(source, target, relation) in view ids; the view deduplicates, so each names
one edge, and a part's edges are a set.

Module: tests/unit/test_pin_p0_5_base_split_use.py
"""
from __future__ import annotations

import dataclasses
import inspect
from collections import Counter

import pytest

torch = pytest.importorskip("torch")


def _triples(heads, tails, relations):
    return list(zip(heads.tolist(), tails.tolist(), relations.tolist(), strict=True))


def test_p0_5_a_base_run_learns_on_train_edges_and_validates_on_val_edges(
        tmp_path, monkeypatch):
    """P0-5: in a real base run, training's message passing and supervision
    use train-part edges only, validation's message passing uses exactly the
    train part, and validation's pairs are val-part edges."""
    from src.kg.graph_view import build_graph_view, split_facts
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.models.bundle.builder import ModelBundle
    from src.models.bundle.record import read_model_record
    from src.training.lp_task import BaseLPTask
    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.trained_children import BASE_RUN, train_children

    sampled, full, supervised, tasks = [], [], [], []

    def observe(owner, name, record):
        original = getattr(owner, name)
        signature = inspect.signature(original)

        def observed(*args, **kwargs):
            record(signature.bind(*args, **kwargs).arguments)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, observed)

    observe(ModelBundle, "node_vectors_sampled", lambda a: sampled.append(_triples(
        a["node_ids"][a["edge_index"][0]].cpu(), a["node_ids"][a["edge_index"][1]].cpu(),
        a["edge_type"].cpu())))
    observe(ModelBundle, "node_vectors_full", lambda a: full.append(_triples(
        a["edge_index"][0].cpu(), a["edge_index"][1].cpu(), a["edge_type"].cpu())))
    observe(BaseLPTask, "training_step", lambda a: supervised.append(_triples(
        a["batch"].node_ids[a["batch"].head].cpu(), a["batch"].node_ids[a["batch"].tail].cpu(),
        a["batch"].relation.cpu())))
    init = BaseLPTask.__init__

    def constructed(self, *args, **kwargs):
        init(self, *args, **kwargs)
        tasks.append(self)

    monkeypatch.setattr(BaseLPTask, "__init__", constructed)

    data_dir = write_demo_workspace(tmp_path / "data")
    trained = train_children(data_dir, tasks=())

    # The run's split, as it recorded it, over the view it recorded.
    run_dir = data_dir / "runs" / "gatv2" / BASE_RUN
    kept = torch.load(run_dir / "graph_split.pt", map_location="cpu", weights_only=True)
    record = read_model_record(torch.load(trained["base"], map_location="cpu",
                                          weights_only=True))
    rule = dataclasses.asdict(record.model_spec.graph_view)
    rule["exclude_relations"] = list(rule["exclude_relations"])
    view = build_graph_view(read_graph_artifacts(data_dir, map_location="cpu").graph_data,
                            rule=rule)
    assert view.digest == kept["record"]["view_digest"] == record.data_roles["graph_view"][
        "digest"]
    split = split_facts(view, kept["record"]["seed"])
    assert torch.equal(split.part, kept["part"])
    assert kept["record"]["digest"] == split.digest == record.data_roles["split"]["digest"]
    edges = _triples(view.edge_index()[0], view.edge_index()[1], view.rel)
    assert len(set(edges)) == len(edges), "an edge triple names more than one edge"
    part = {name: {e for e, keep in zip(edges, split.mask(name).tolist(), strict=True)
                   if keep}
            for name in ("train", "val", "test")}
    assert all(part.values()), {name: len(p) for name, p in part.items()}

    # Training: message passing and supervision on train edges only.
    assert sampled and all(sampled), "no sampled message passing was observed"
    for call in sampled:
        assert set(call) <= part["train"], set(call) - part["train"]
    assert supervised and any(supervised), "no supervised pair was observed"
    for step in supervised:
        assert set(step) <= part["train"], set(step) - part["train"]

    # Validation: message passing over exactly the train part, pairs on val edges.
    (task,) = tasks
    assert full, "no validation message passing was observed"
    for call in full:
        assert Counter(call) == Counter(part["train"])
    pairs = _triples(task.val_head, task.val_tail, task.val_rel)
    assert pairs and set(pairs) <= part["val"], set(pairs) - part["val"]
    assert task.val_counts["val_edges"] == len(part["val"])
