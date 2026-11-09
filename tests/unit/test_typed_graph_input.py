"""
Node types through the shared graph input (F6), and a small cross-species graph (E4).
=====================================================================================
`docs/working/research-comparison/README.md` §3 (F6), §5 Stage 1, §6 (E2, E4).
An input module with a type embedding (`input.type_embedding`) needs each
node's view type. The formal callers passed only relation ids, so a typed
model failed at its first training step, in base validation and in the shared
full-view entry. Both shared graph inputs now carry the view's own type table
(`GraphView.node_type`). A step's subgraph (`LPBatch`) carries its nodes'
types in the order of its node ids. The full-view entry (`full_view_vectors`)
gets every view node's type in the view's order. Each edge's relation comes
in edge order beside them. All of it goes through the one sampled and the one
full forward.

Through the formal paths, on CPU, with `encoder.relation_edge_dim` off and on:

- a typed base and its disease, gene and patients-like-me children are
  trained by `scripts/train_model.py`'s `train`
  (`tests/fixtures/trained_children.train_children`, the demo workspace, which
  holds a mouse gene and its ortholog relations). Every sampled and full-view
  call of the bundle gets the view's types for exactly its nodes, in their
  order, and each message-passing edge's own view relation; the types agree
  with the relations' endpoint types;
- each checkpoint reloads (`load_bundle`): its full-view vectors are those its
  last validation computed, the base reproduces the validation metrics it
  logged, and the types are read (other types give other vectors; none is
  refused);
- each child measures through `scripts/measure_scorer.py` (`--split val`)
  and the disease child serves (`build_served_pipeline`), both on the
  vectors validation computed;
- the reference input (no type embedding) reads no type: its vectors are the
  same, bit for bit, with the view's types, other types or none;
- types or relation ids that are not one per node or edge are refused.

**E4, a small synthetic cross-species graph** (ten human genes, one mouse
gene, ten phenotypes, ortholog and association relations and their reverse
twins). The graph view keeps every node and relation, its types and both
mappings; reverse twins share one fact and one split part; the LP pairs keep
relation identity; a typed base step and validation run through the formal
pieces (`BaseLPSampler`, `BaseLPTask`, `fixed_lp_pairs`,
`full_view_vectors`). The bundle has no ortholog gate and runs one message
pass per vector computation. Real HP-MP mappings and any cross-species effect
are not tested here.

Module: tests/unit/test_typed_graph_input.py
"""
from __future__ import annotations

import dataclasses
import inspect
import json
from collections import Counter
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

TASKS = ("disease", "gene", "patients_like_me")
CPU = torch.device("cpu")


def _observe(patch: Any, owner: Any, name: str, calls: List[Dict[str, Any]]) -> None:
    """Wrap `owner.name` (a `ModelBundle` vector entry): each call is recorded
    with its run (the bundle's task, or "base"), its arguments and its output,
    detached CPU copies, then answered as before."""
    original = getattr(owner, name)
    signature = inspect.signature(original)

    def copy(value: Any) -> Any:
        return value.detach().cpu().clone() if torch.is_tensor(value) else value

    def observed(*args: Any, **kwargs: Any) -> Any:
        arguments = signature.bind(*args, **kwargs).arguments
        out = original(*args, **kwargs)
        bundle = arguments.pop("self")
        calls.append({"entry": name, "run": bundle.spec.task or "base",
                      "args": {k: copy(v) for k, v in arguments.items()}, "out": copy(out)})
        return out

    patch.setattr(owner, name, observed)


def _triples(heads: Any, tails: Any, relations: Any) -> List[tuple]:
    return list(zip(heads.tolist(), tails.tolist(), relations.tolist(), strict=True))


def _view_edges(view: Any, mask: Any = None) -> Counter:
    ei = view.edge_index(mask)
    return Counter(_triples(ei[0], ei[1], view.rel if mask is None else view.rel[mask]))


def _types_agree_with_relations(view: Any, triples: List[tuple], node_type: Any,
                                position: Dict[int, int]) -> bool:
    """Each edge's relation names the view types of its two ends, read from
    `node_type` at the ends' positions."""
    for source, target, relation in triples:
        src, _, dst = view.relations[relation].split("|")
        if (view.node_types[int(node_type[position[source]])],
                view.node_types[int(node_type[position[target]])]) != (src, dst):
            return False
    return True


def _typed_spec(relation_edge_dim: Any) -> Any:
    from tests.fixtures.trained_children import small_base_spec

    spec = small_base_spec()
    return dataclasses.replace(
        spec, input=dataclasses.replace(spec.input, type_embedding=True),
        encoder=dataclasses.replace(spec.encoder, relation_edge_dim=relation_edge_dim)).validate()


@pytest.fixture(scope="module", params=[None, 3], ids=["relations_off", "relations_on"])
def typed(request: Any, tmp_path_factory: Any) -> Dict[str, Any]:
    """A typed base (`input.type_embedding` on, `encoder.relation_edge_dim`
    the param) and one typed child per task, trained by `train`, every call of
    the bundle's two vector entries observed; the view and split the base
    recorded."""
    from src.kg.graph_view import build_graph_view
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.models.bundle.builder import ModelBundle
    from src.models.bundle.record import read_model_record
    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.trained_children import BASE_RUN, train_children

    spec = _typed_spec(request.param)
    calls: List[Dict[str, Any]] = []
    data_dir = write_demo_workspace(tmp_path_factory.mktemp("typed") / "data")
    with pytest.MonkeyPatch.context() as patch:
        for name in ("node_vectors_sampled", "node_vectors_full"):
            _observe(patch, ModelBundle, name, calls)
        trained = train_children(data_dir, tasks=TASKS, base_model_spec=spec.to_dict())
    record = read_model_record(torch.load(trained["base"], map_location="cpu",
                                          weights_only=True))
    rule = dataclasses.asdict(record.model_spec.graph_view)
    rule["exclude_relations"] = list(rule["exclude_relations"])
    view = build_graph_view(read_graph_artifacts(data_dir, map_location="cpu").graph_data,
                            rule=rule)
    split = torch.load(data_dir / "runs" / "gatv2" / BASE_RUN / "graph_split.pt",
                       map_location="cpu", weights_only=True)
    return {"spec": spec, "trained": trained, "calls": calls, "view": view, "split": split,
            "relation_edge_dim": request.param}


def _calls(typed: Dict[str, Any], run: str, entry: str) -> List[Dict[str, Any]]:
    return [c for c in typed["calls"] if c["run"] == run and c["entry"] == entry]


def _message_passing(typed: Dict[str, Any], run: str) -> Any:
    """The full-view edges of `run`'s validation: the train part for the
    base, the whole view for a finetune (G4)."""
    return typed["split"]["part"] == 0 if run == "base" else None


# --------------------------------------------------------------------------- formal paths


def test_the_demo_view_is_a_cross_species_graph(typed):
    """The typed runs below are cross-species runs: the demo view keeps its
    mouse gene and the ortholog relation with its reverse twin."""
    view = typed["view"]
    assert "mouse_gene" in view.node_types and view.type_nodes("mouse_gene").numel() == 1
    forward = view.relations.index("gene|human_mouse_ortholog|mouse_gene")
    assert int(view.reverse_relation[forward]) == view.relations.index(
        "mouse_gene|rev_human_mouse_ortholog|gene")


def test_every_formal_call_gets_the_views_types_and_relations_in_order(typed):
    """Base step and validation, and each finetune's step and validation:
    every sampled call's types are the view's at its node ids, in their order,
    and its edges are train-part view edges with their own relations; every
    full-view call's types are the view's whole table, in its node order, over
    exactly its validation graph's edges."""
    view = typed["view"]
    train = _view_edges(view, typed["split"]["part"] == 0)
    for run in ("base", *TASKS):
        sampled = _calls(typed, run, "node_vectors_sampled")
        full = _calls(typed, run, "node_vectors_full")
        assert sampled and full, run
        for call in sampled:
            args = call["args"]
            ids, ei = args["node_ids"], args["edge_index"]
            assert args["node_type"] is not None, run
            assert torch.equal(args["node_type"], view.node_type[ids]), run
            triples = _triples(ids[ei[0]], ids[ei[1]], args["edge_type"])
            assert set(triples) <= set(train), run
            position = {int(v): i for i, v in enumerate(ids.tolist())}
            assert _types_agree_with_relations(view, triples, args["node_type"], position), run
        expected = _view_edges(view, _message_passing(typed, run))
        for call in full:
            args = call["args"]
            assert torch.equal(args["node_type"], view.node_type), run
            triples = _triples(args["edge_index"][0], args["edge_index"][1], args["edge_type"])
            assert Counter(triples) == expected, run
            identity = {i: i for i in range(view.num_nodes)}
            assert _types_agree_with_relations(view, triples, args["node_type"], identity), run


def test_each_typed_checkpoint_reloads_and_reproduces_its_validation(typed):
    """Save and reload: each checkpoint holds the type table's weights (and
    the relation embedding's exactly when the spec has one) and loads through
    `load_bundle`; with the view's types its full-view vectors are those its
    last validation computed. The reloaded base reproduces the validation
    metrics it logged. Other types give other vectors, and none is refused."""
    from src.config.recipe import BASE_FIXED
    from src.models.bundle.record import load_bundle
    from src.models.tasks.scoring import full_view_vectors
    from src.training.lp_task import BaseLPTask

    view, spec, split = typed["view"], typed["spec"], typed["split"]
    for run in ("base", *TASKS):
        payload = torch.load(typed["trained"][run], map_location="cpu", weights_only=True)
        state = payload["state_dict"]
        assert state["input.type_embedding.weight"].shape == (len(view.node_types),
                                                              spec.input.dim)
        assert ("encoder.relation_embedding.weight" in state) == (
            typed["relation_edge_dim"] is not None)
        stage = "base" if run == "base" else "finetune"
        bundle, record = load_bundle(payload, learning_stage=stage,
                                     task=None if run == "base" else run,
                                     view_digest=view.digest)
        assert record.model_spec.input == spec.input
        assert record.model_spec.encoder == spec.encoder
        mask = _message_passing(typed, run)
        edges, relations = view.edge_index(mask), view.rel if mask is None else view.rel[mask]
        z = full_view_vectors(bundle, edges, relations, CPU, node_type=view.node_type)
        assert torch.equal(z, _calls(typed, run, "node_vectors_full")[-1]["out"]), run
        other = (view.node_type + 1) % len(view.node_types)
        assert not torch.equal(full_view_vectors(bundle, edges, relations, CPU,
                                                 node_type=other), z), run
        with pytest.raises(ValueError, match="needs node types"):
            full_view_vectors(bundle, edges, relations, CPU)
        if run == "base":
            val = split["part"] == 1
            task = BaseLPTask(
                num_nodes=view.num_nodes, train_edges=edges, train_relations=relations,
                val_edges=view.edge_index(val), val_relations=view.rel[val],
                seed=split["record"]["seed"],
                val_batch_size=BASE_FIXED["validation_batch_size"], node_type=view.node_type)
            logged = {k: v for k, v in payload["logs"].items() if k.startswith("val_")}
            assert {f"val_{k}": v for k, v in task.validate(bundle, CPU).items()} == logged


@pytest.mark.parametrize("task", TASKS)
def test_each_typed_child_measures_through_the_measurement_entry(typed, task, tmp_path,
                                                                 monkeypatch):
    """`scripts/measure_scorer.py --split val` measures each typed child: its
    one full-view call gets the view's types over the whole view, and its
    vectors are those the child's validation computed."""
    import scripts.measure_scorer as cli
    from src.models.bundle.builder import ModelBundle

    calls: List[Dict[str, Any]] = []
    _observe(monkeypatch, ModelBundle, "node_vectors_full", calls)
    output = tmp_path / "run" / "measurement.json"
    assert cli.main([
        "--mode", task, "--checkpoint", str(typed["trained"][task]),
        "--data-dir", str(typed["trained"]["data_dir"]), "--split", "val",
        "--output", str(output), "--batch-size", "3", "--device", "cpu",
    ]) == 0
    report = json.loads(output.read_text())
    assert report["manifest"]["mode"] == task and report["n_ranked"] > 0
    (call,) = calls
    assert torch.equal(call["args"]["node_type"], typed["view"].node_type)
    assert torch.equal(call["out"], _calls(typed, task, "node_vectors_full")[-1]["out"])


def test_a_typed_child_serves(typed):
    """The served pipeline of the typed disease child computes its vectors
    with the view's types: those the child's validation computed."""
    from tests.fixtures.trained_children import serve

    pipeline = serve(typed["trained"], "disease")
    assert torch.equal(pipeline._bundle_z.cpu(),
                       _calls(typed, "disease", "node_vectors_full")[-1]["out"])


# --------------------------------------------------------------------------- the reference and refusals

VIEW = {"digest": "d" * 64, "num_nodes": 5, "node_types": ["disease", "gene", "phenotype"],
        "relations": ["disease|assoc|phenotype", "phenotype|rev_assoc|disease"]}
#: VIEW's nodes' types, and a star (seed 0, hop 1 the four leaves) of its edges.
NODE_TYPE = torch.tensor([0, 1, 1, 2, 2])
STAR = torch.tensor([(1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (0, 2), (0, 3), (0, 4)]).T
STAR_RELATIONS = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1])


def _bundle(type_embedding: bool, relation_edge_dim: Any = None) -> Any:
    from src.models.bundle import build_bundle, reference_spec
    from src.models.bundle.spec import InputSpec

    spec = reference_spec("base")
    spec = dataclasses.replace(
        spec, input=InputSpec(dim=6, type_embedding=type_embedding),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=3, output=2, heads=2,
                                    relation_edge_dim=relation_edge_dim)).validate()
    torch.manual_seed(0)
    return build_bundle(spec, VIEW).eval()


@pytest.mark.parametrize("relation_edge_dim", [None, 3])
def test_the_reference_input_reads_no_type(relation_edge_dim):
    """With no type embedding (the reference recipe), the types change
    nothing: the sampled and full vectors are the same, bit for bit, with the
    view's types, other types or none, and the state holds no type weights."""
    bundle = _bundle(False, relation_edge_dim)
    assert not any("type_embedding" in key for key in bundle.state_dict())
    other = (NODE_TYPE + 1) % 3
    with torch.no_grad():
        full = [bundle.node_vectors_full(STAR, edge_type=STAR_RELATIONS, node_type=t)
                for t in (None, NODE_TYPE, other)]
        sampled = [bundle.node_vectors_sampled(torch.arange(5), STAR, [1, 4, 0], [4, 4],
                                               edge_type=STAR_RELATIONS, node_type=t)
                   for t in (None, NODE_TYPE, other)]
    assert all(torch.equal(full[0], z) for z in full[1:])
    assert all(torch.equal(sampled[0], z) for z in sampled[1:])


def test_types_and_relations_not_one_per_node_or_edge_are_refused():
    """A type list that is not one per node, or relation ids that are not one
    per message-passing edge, are refused by the one input module and the
    encoder, sampled and full; a type table that is not one int64 id per view
    node is refused by the sampler and the base task when they are built."""
    from src.training.lp_task import BaseLPSampler, BaseLPTask

    bundle = _bundle(True, relation_edge_dim=3)
    hops = ([1, 4, 0], [4, 4])
    with torch.no_grad():
        with pytest.raises(ValueError, match="node types are given for"):
            bundle.node_vectors_full(STAR, edge_type=STAR_RELATIONS, node_type=NODE_TYPE[:4])
        with pytest.raises(ValueError, match="node types are given for"):
            bundle.node_vectors_sampled(torch.arange(5), STAR, *hops, edge_type=STAR_RELATIONS,
                                        node_type=NODE_TYPE[:2])
        with pytest.raises(ValueError, match="relation ids are given for"):
            bundle.node_vectors_full(STAR, edge_type=STAR_RELATIONS[:7], node_type=NODE_TYPE)
        with pytest.raises(ValueError, match="relation ids are given for"):
            bundle.node_vectors_sampled(torch.arange(5), STAR, *hops,
                                        edge_type=torch.cat([STAR_RELATIONS, STAR_RELATIONS]),
                                        node_type=NODE_TYPE)
    edges = torch.tensor([[0, 1, 0, 2], [3, 4, 4, 3]])
    relations = torch.zeros(4, dtype=torch.long)
    for table in (NODE_TYPE[:4], NODE_TYPE.float()):
        with pytest.raises(ValueError, match="node type table"):
            BaseLPSampler(5, edges, relations, edges, relations, sizes=[2, 2], batch_size=4,
                          seed=0, node_type=table)
        with pytest.raises(ValueError, match="node type table"):
            BaseLPTask(num_nodes=5, train_edges=edges, train_relations=relations,
                       val_edges=edges, val_relations=relations, seed=0, node_type=table)


# --------------------------------------------------------------------------- E4: a cross-species graph

ORTHOLOG = "gene|human_mouse_ortholog|mouse_gene"
REV_ORTHOLOG = "mouse_gene|rev_human_mouse_ortholog|gene"


def _cross_species_graph() -> Dict[str, Any]:
    """Ten human genes, one mouse gene, ten phenotypes, as a workspace exports
    them (`KnowledgeGraph.export_to_model`): each gene has four phenotypes,
    genes 0 and 5 are orthologs of the mouse gene, and every relation has its
    `rev_` twin."""
    phenotypes = [(g, (g + k) % 10) for g in range(10) for k in (0, 1, 3, 7)]
    graph: Dict[str, Any] = {"num_nodes_dict": {"gene": 10, "mouse_gene": 1, "phenotype": 10},
                             "edge_index_dict": {}}
    for (src, rel, dst), pairs in ((("gene", "gene_has_phenotype", "phenotype"), phenotypes),
                                   (("gene", "human_mouse_ortholog", "mouse_gene"),
                                    [(0, 0), (5, 0)])):
        tensor = torch.tensor(pairs, dtype=torch.long).T
        graph["edge_index_dict"][(src, rel, dst)] = tensor
        graph["edge_index_dict"][(dst, "rev_" + rel, src)] = tensor.flip(0)
    return graph


def test_a_cross_species_view_keeps_species_relations_mappings_and_twins():
    """The view keeps every node of every species and every relation, with
    each edge's relation naming its ends' types; the raw-to-view and
    view-to-raw mappings and each type's local ids round trip; each reverse
    twin is in its forward edge's fact and split part."""
    from src.kg.graph_view import build_graph_view, split_facts

    view = build_graph_view(_cross_species_graph())
    assert view.node_types == ["gene", "mouse_gene", "phenotype"]
    assert view.num_nodes == 21 and set(view.counts["nodes_excluded_by_type"].values()) == {0}
    assert {ORTHOLOG, REV_ORTHOLOG} <= set(view.relations) and len(view.relations) == 4
    for i, name in enumerate(view.relations):
        twin = int(view.reverse_relation[i])
        assert twin >= 0 and int(view.reverse_relation[twin]) == i, name
    ortholog = view.relations.index(ORTHOLOG)
    assert int((view.rel == ortholog).sum()) == 2
    assert int((view.rel == view.relations.index(REV_ORTHOLOG)).sum()) == 2
    triples = _triples(view.src, view.dst, view.rel)
    identity = {i: i for i in range(view.num_nodes)}
    assert _types_agree_with_relations(view, triples, view.node_type, identity)

    assert torch.equal(view.raw_to_view[view.view_to_raw], torch.arange(view.num_nodes))
    for t, count in _cross_species_graph()["num_nodes_dict"].items():
        ids = view.view_ids(t, torch.arange(count))
        assert bool((ids >= 0).all()), t
        assert torch.equal(view.local_index(ids), torch.arange(count)), t
        assert bool((view.node_type[ids] == view.node_types.index(t)).all()), t
        assert torch.equal(view.view_to_raw[ids], torch.arange(count) + view.type_offsets[t]), t
    (mouse,) = view.view_ids("mouse_gene", torch.tensor([0])).tolist()
    assert view.node_types[int(view.node_type[mouse])] == "mouse_gene"

    split = split_facts(view, seed=0)
    part = {edge: int(p) for edge, p in zip(triples, split.part.tolist(), strict=True)}
    facts = Counter(view.fact.tolist())
    assert set(facts.values()) == {2}
    for (source, target, relation), p in part.items():
        twin = (target, source, int(view.reverse_relation[relation]))
        assert part[twin] == p
        assert int(view.fact[triples.index(twin)]) == int(view.fact[triples.index(
            (source, target, relation))])


@pytest.mark.parametrize("relation_edge_dim", [None, 3])
def test_a_typed_base_runs_on_the_cross_species_graph_through_the_formal_pieces(
        relation_edge_dim, monkeypatch):
    """The base's sampler, step and validation (`BaseLPSampler`,
    `BaseLPTask`) and the LP core's pairs on the cross-species view, with a
    typed bundle: each batch carries its nodes' view types, the step learns
    the type table, validation scores the val part's pairs over the train
    part with the view's types, and the mouse gene's vector reads its own
    type. The LP pairs of the test part keep relation identity, each negative
    of its relation's target type. The bundle has its three graph-side parts
    and no gate, and each vector computation is one encoder pass."""
    from src.kg.graph_view import build_graph_view, split_facts
    from src.models.bundle import build_bundle
    from src.models.bundle.encoders import GATv2Encoder
    from src.models.bundle.lp_head import fixed_lp_pairs
    from src.models.tasks.scoring import full_view_vectors
    from src.training.lp_task import BaseLPSampler, BaseLPTask

    passes: Counter = Counter()
    for name in ("forward_sampled", "forward_full"):
        original = getattr(GATv2Encoder, name)

        def counted(self, *args, _original=original, _name=name, **kwargs):
            passes[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(GATv2Encoder, name, counted)

    view = build_graph_view(_cross_species_graph())
    split = split_facts(view, seed=0)
    train, val, test = (split.mask(n) for n in ("train", "val", "test"))
    torch.manual_seed(0)
    bundle = build_bundle(_typed_spec(relation_edge_dim), view)
    assert {name for name, _ in bundle.named_children()} == {"input", "encoder", "lp_head"}
    assert not any("gate" in key or "ortholog" in key for key in bundle.state_dict())

    sampler = BaseLPSampler(view.num_nodes, view.edge_index(train), view.rel[train],
                            view.edge_index(train), view.rel[train], sizes=[4, 4],
                            batch_size=32, seed=0, node_type=view.node_type)
    task = BaseLPTask(num_nodes=view.num_nodes, train_edges=view.edge_index(train),
                      train_relations=view.rel[train], val_edges=view.edge_index(val),
                      val_relations=view.rel[val], seed=0, node_type=view.node_type)
    batch = next(iter(sampler))
    assert torch.equal(batch.node_type, view.node_type[batch.node_ids])
    bundle.train()
    step = task.training_step(bundle, batch, CPU)
    assert step.loss is not None and passes == {"forward_sampled": 1}
    step.loss.backward()
    assert bundle.input.type_embedding.weight.grad.abs().sum() > 0

    metrics = task.validate(bundle, CPU)
    assert passes == {"forward_sampled": 1, "forward_full": 1}
    assert metrics["lp_positives"] > 0

    (mouse,) = view.view_ids("mouse_gene", torch.tensor([0])).tolist()
    as_gene = view.node_type.clone()
    as_gene[mouse] = view.node_types.index("gene")
    z = full_view_vectors(bundle, view.edge_index(train), view.rel[train], CPU,
                          node_type=view.node_type)
    z_gene = full_view_vectors(bundle, view.edge_index(train), view.rel[train], CPU,
                               node_type=as_gene)
    assert not torch.equal(z[mouse], z_gene[mouse])

    pairs = fixed_lp_pairs(view.edge_index(test), view.rel[test], seed=0, batch_size=512)
    test_edges = set(_triples(view.src[test], view.dst[test], view.rel[test]))
    assert pairs.counts["pairs"] > 0
    assert set(_triples(pairs.head, pairs.tail, pairs.relation)) <= test_edges
    for relation, negative in zip(pairs.relation.tolist(), pairs.negative_tail.tolist(),
                                  strict=True):
        target_type = view.relations[relation].split("|")[2]
        assert view.node_types[int(view.node_type[negative])] == target_type
