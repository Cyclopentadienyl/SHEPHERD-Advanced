"""
HGT's conv sees its input in its own node-type order.
====================================================
BACKLOG item 20 (a); docs/working/PLAN_HGT_LINEAR_PATHS.md §1. PyG's grouped
branch of `HeteroDictLinear` (`pyg_lib.ops.grouped_matmul`) builds its inputs in
its layers' order and stores the outputs under the input dict's keys, so when the
two orders differ a node type receives another type's projection — silently more
often than not. It runs when `torch_geometric.backend.use_segment_matmul` is
`True`, or by PyG's default from ten node types up. `HeteroGNNLayer` passes the
conv its input in the conv's own order, so the grouped branch and the per-type
loop compute the same thing. The edges likewise go in the conv's edge-type order
(plan §1.7): its relation linears cut their input by edge-type index in arrival
order, so another order is wrong on both kernel paths.

Every test spies on the grouped kernel. The cases meant to run the grouped
branch require the spy to fire, so none of them passes without the branch
running; the others require it not to fire.

Module: tests/unit/test_hgt_node_type_order.py
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyg_typing = pytest.importorskip("torch_geometric.typing")

pytestmark = pytest.mark.skipif(
    not pyg_typing.WITH_GMM, reason="pyg_lib's grouped matmul is not available"
)

#: The production node types in the order the shipped build adds them
#: (`scripts/build_knowledge_graph.py`), with reverse relations so that every type
#: is a destination, as in the production graph.
METADATA = (
    ["disease", "phenotype", "gene"],
    [
        ("gene", "associated_with", "disease"),
        ("disease", "rev_associated_with", "gene"),
        ("phenotype", "associated_with", "gene"),
        ("gene", "rev_associated_with", "phenotype"),
        ("disease", "has_phenotype", "phenotype"),
        ("phenotype", "rev_has_phenotype", "disease"),
    ],
)
#: The order the training loader hands the types over: its seed types first.
LOADER_ORDER = ["phenotype", "disease", "gene"]
HIDDEN = 32


@pytest.fixture
def grouped(monkeypatch):
    """Counts calls into the grouped kernel, and sets the backend flag."""
    import pyg_lib
    import torch_geometric.backend as backend

    calls = []
    real = pyg_lib.ops.grouped_matmul

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pyg_lib.ops, "grouped_matmul", spy)

    def set_flag(value):
        monkeypatch.setattr(backend, "use_segment_matmul", value)

    return calls, set_flag


def _layer(metadata=METADATA):
    from src.models.gnn.layers import HeteroGNNLayer

    torch.manual_seed(0)
    layer = HeteroGNNLayer(HIDDEN, metadata, conv_type="hgt", num_heads=4)
    layer.eval()
    return layer


def _inputs(rows, edge_types, order, seed=1):
    """Features in `order`, and random edges between the types that have rows."""
    generator = torch.Generator().manual_seed(seed)
    x_dict = {
        node_type: torch.randn(rows[node_type], HIDDEN, generator=generator)
        for node_type in order
    }
    edges = {}
    for src, rel, dst in edge_types:
        if rows[src] and rows[dst]:
            edges[(src, rel, dst)] = torch.stack([
                torch.randint(rows[src], (12,), generator=generator),
                torch.randint(rows[dst], (12,), generator=generator),
            ])
        else:
            edges[(src, rel, dst)] = torch.empty(2, 0, dtype=torch.long)
    return x_dict, edges


def _forward(layer, x_dict, edges):
    """Outputs and the gradients of the inputs and every parameter."""
    layer.zero_grad()
    leaves = {k: v.clone().requires_grad_(True) for k, v in x_dict.items()}
    out = layer(leaves, edges)
    sum(o.sum() for o in out.values() if o.numel()).backward()
    grads = {f"x.{k}": v.grad for k, v in leaves.items() if v.grad is not None}
    grads.update({n: p.grad for n, p in layer.named_parameters() if p.grad is not None})
    return {k: v.detach() for k, v in out.items()}, grads


def test_both_linears_are_built_in_the_conv_node_order():
    layer = _layer()

    assert layer._hgt_node_order == tuple(METADATA[0])
    assert tuple(layer.conv.kqv_lin.lins) == layer._hgt_node_order
    assert tuple(layer.conv.out_lin.lins) == layer._hgt_node_order


@pytest.mark.parametrize("flag", [None, False])
def test_the_output_does_not_depend_on_the_input_order(grouped, flag):
    """Below ten node types and with the flag off, the per-type loop runs; the
    layer's output is the same whichever order the caller uses."""
    calls, set_flag = grouped
    set_flag(flag)
    layer = _layer()
    rows = {"disease": 7, "phenotype": 5, "gene": 3}
    in_loader_order, edges = _inputs(rows, METADATA[1], LOADER_ORDER)
    in_metadata_order = {k: in_loader_order[k] for k in METADATA[0]}

    first, _ = _forward(layer, in_loader_order, edges)
    second, _ = _forward(layer, in_metadata_order, edges)

    assert calls == [], "the grouped branch ran where it should not"
    for node_type in METADATA[0]:
        assert torch.equal(first[node_type], second[node_type]), node_type


@pytest.mark.parametrize("rows", [
    {"disease": 7, "phenotype": 5, "gene": 3},   # silently wrong before the fix
    {"disease": 5, "phenotype": 7, "gene": 3},   # an IndexError before the fix
    {"disease": 5, "phenotype": 5, "gene": 3},   # silently wrong before the fix
    {"disease": 0, "phenotype": 0, "gene": 3},
], ids=["7-5-3", "5-7-3", "5-5-3", "0-0-3"])
def test_the_grouped_branch_agrees_with_the_loop(grouped, rows):
    """Input in the loader's order, as training hands it over."""
    calls, set_flag = grouped
    layer = _layer()
    x_dict, edges = _inputs(rows, METADATA[1], LOADER_ORDER)

    set_flag(False)
    loop_out, loop_grads = _forward(layer, x_dict, edges)
    assert calls == []
    set_flag(True)
    grouped_out, grouped_grads = _forward(layer, x_dict, edges)

    assert calls, "the grouped branch did not run"
    assert grouped_out.keys() == loop_out.keys()
    for node_type, value in loop_out.items():
        torch.testing.assert_close(grouped_out[node_type], value)
    # A type with no rows leaves its linear out of one branch's graph (no
    # gradient) and in the other's (a zero gradient); both mean zero.
    for name in loop_grads.keys() | grouped_grads.keys():
        reference = loop_grads.get(name, grouped_grads.get(name))
        torch.testing.assert_close(
            grouped_grads.get(name, torch.zeros_like(reference)),
            loop_grads.get(name, torch.zeros_like(reference)),
        )


@pytest.mark.parametrize("flag", [False, True])
def test_the_output_does_not_depend_on_the_edge_order(grouped, flag):
    """Edges in the reverse of the conv's order give what the conv's order gives,
    on either kernel path of the relation linears."""
    calls, set_flag = grouped
    set_flag(flag)
    layer = _layer()
    rows = {"disease": 7, "phenotype": 5, "gene": 3}
    x_dict, in_conv_order = _inputs(rows, METADATA[1], METADATA[0])
    reversed_order = dict(reversed(list(in_conv_order.items())))
    assert list(reversed_order) != list(in_conv_order)

    first, first_grads = _forward(layer, x_dict, in_conv_order)
    second, second_grads = _forward(layer, x_dict, reversed_order)

    assert bool(calls) == flag
    for node_type in METADATA[0]:
        assert torch.equal(first[node_type], second[node_type]), node_type
    assert first_grads.keys() == second_grads.keys()
    for name, value in first_grads.items():
        assert torch.equal(second_grads[name], value), name


def test_an_edge_type_the_conv_does_not_know_is_still_refused():
    layer = _layer()
    rows = {"disease": 7, "phenotype": 5, "gene": 3}
    x_dict, edges = _inputs(rows, METADATA[1], METADATA[0])
    edges[("gene", "unknown", "disease")] = edges[("gene", "associated_with", "disease")]

    with pytest.raises(KeyError):
        layer(x_dict, edges)


def test_ten_node_types_take_the_grouped_branch_by_default_and_agree_with_the_loop(grouped):
    """PyG's default takes the grouped branch from ten node types up; the input
    order here is the reverse of the conv's."""
    calls, set_flag = grouped
    types = [f"t{i}" for i in range(10)]
    edge_types = []
    for a, b in zip(types, types[1:], strict=False):
        edge_types += [(a, "to", b), (b, "rev_to", a)]
    layer = _layer((types, edge_types))
    rows = {t: 3 + i for i, t in enumerate(types)}
    x_dict, edges = _inputs(rows, edge_types, list(reversed(types)))

    set_flag(False)
    loop_out, _ = _forward(layer, x_dict, edges)
    assert calls == []
    set_flag(None)
    default_out, _ = _forward(layer, x_dict, edges)

    assert calls, "PyG's default did not take the grouped branch at ten types"
    for node_type, value in loop_out.items():
        torch.testing.assert_close(default_out[node_type], value)


def test_the_formal_build_paths_agree_across_branches(grouped, tmp_path):
    """The formal paths: the model `train_model` builds over a workspace, on a
    batch from the loader training uses, whose type order differs from the
    model's; its checkpoint, as the Trainer writes it, rebuilt by serving's
    `build_shepherd_model` over the full graph; and the same checkpoint through
    `build_index`."""
    import scripts.train_model as train_model
    from scripts.build_index import extract_embeddings_from_checkpoint
    from src.kg.data_loader import DataLoaderConfig, create_diagnosis_dataloader
    from src.kg.storage.file_storage import read_graph_artifacts, read_samples
    from src.models.gnn.shepherd_gnn import build_shepherd_model
    from src.training.trainer import Trainer, TrainerConfig
    from tests.fixtures.synthetic_workspace import build_workspace

    calls, set_flag = grouped
    data_dir, _ = build_workspace(tmp_path)
    graph_data = read_graph_artifacts(data_dir).graph_data
    torch.manual_seed(0)
    model = train_model.create_model_from_config(
        train_model.TrainConfig(conv_type="hgt", hidden_dim=HIDDEN, num_layers=2,
                                num_heads=4, dropout=0.0),
        graph_data,
    )
    model.eval()
    batch = next(iter(create_diagnosis_dataloader(
        samples=read_samples(data_dir, "test").samples, graph_data=graph_data,
        config=DataLoaderConfig(batch_size=3, num_workers=0, shuffle=False),
    )))
    assert list(batch["subgraph_x_dict"]) != list(graph_data["num_nodes_dict"]), (
        "the loader's order matches the model's, so this proves nothing"
    )

    checkpoint_path = tmp_path / "hgt.pt"
    Trainer(model, train_dataloader=[], config=TrainerConfig(device="cpu")).save_checkpoint(
        checkpoint_path
    )
    served = build_shepherd_model(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True), graph_data
    )
    assert served.config.conv_type == "hgt"

    paths = {
        "training batch": lambda: model(
            batch["subgraph_x_dict"], batch["subgraph_edge_index_dict"]
        ),
        "serving, full graph": lambda: served(
            graph_data["x_dict"], graph_data["edge_index_dict"]
        ),
        "build_index": lambda: {
            node_type: torch.from_numpy(value)
            for node_type, value in extract_embeddings_from_checkpoint(
                str(checkpoint_path), str(data_dir), device="cpu"
            )[0].items()
        },
    }
    for path, embed in paths.items():
        set_flag(False)
        with torch.no_grad():
            loop_out = embed()
        assert calls == [], path
        set_flag(True)
        with torch.no_grad():
            grouped_out = embed()
        assert calls, f"{path}: the grouped branch did not run"
        calls.clear()

        assert loop_out, path
        for node_type, value in loop_out.items():
            torch.testing.assert_close(grouped_out[node_type], value, msg=path)
