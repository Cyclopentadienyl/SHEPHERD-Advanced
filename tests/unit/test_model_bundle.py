"""The model spec, the one builder and the reference encoder (README §2.8, §3, §4).

- the spec is plain data, read strictly: another version, an unknown key or a
  finetune whose parent is not a base is refused (regression 7);
- the builder reads the spec and the view's shape only; built widths follow
  the spec (C1) and every parameter is one module's (§3.1);
- the graph-mode norm equals its formula and differs from a per-node norm;
  chunked full-view inference equals the one-shot forward for two chunk sizes,
  and the sampled rule equals the full rule when the sample is the whole graph
  (regression 4);
- save and reload through the builder: same names, same outputs, strict load
  (regression 6).

Module: tests/unit/test_model_bundle.py
"""
from __future__ import annotations

import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")

from src.models.bundle import (  # noqa: E402
    ModelSpec,
    SpecError,
    build_bundle,
    finetune_spec_from_parent,
    reference_spec,
)
from src.models.bundle.spec import EncoderSpec, GeneScoringSpec, InputSpec  # noqa: E402

VIEW = {"digest": "d" * 64, "num_nodes": 5, "node_types": ["disease", "gene", "phenotype"],
        "relations": ["disease|assoc|phenotype", "phenotype|rev_assoc|disease"]}


def _small(stage="base", task=None, **encoder):
    spec = reference_spec(stage, task)
    enc = dict(layers=2, hidden=3, output=2, heads=2)
    enc.update(encoder)
    return dataclasses.replace(spec, input=InputSpec(dim=6),
                               encoder=dataclasses.replace(spec.encoder, **enc)).validate()


# --------------------------------------------------------------------------- spec


def test_spec_round_trips_as_plain_json():
    for spec in (reference_spec("base"), reference_spec("finetune", "gene")):
        data = json.loads(json.dumps(spec.to_dict()))
        assert ModelSpec.from_dict(data) == spec


def test_spec_refuses_other_version_unknown_and_missing_keys():
    data = reference_spec("base").to_dict()
    with pytest.raises(SpecError, match="version"):
        ModelSpec.from_dict({**data, "spec_version": 0})
    with pytest.raises(SpecError, match="unknown key"):
        ModelSpec.from_dict({**data, "encoder": {**data["encoder"], "gate": True}})
    trimmed = dict(data)
    del trimmed["lp_head"]
    with pytest.raises(SpecError, match="lacks"):
        ModelSpec.from_dict(trimmed)


def test_finetune_never_chains_and_inherits_the_graph_side():
    base = _small()
    child = finetune_spec_from_parent(base, "disease")
    assert child.encoder == base.encoder and child.input == base.input
    with pytest.raises(SpecError, match="never chain"):
        finetune_spec_from_parent(child, "gene")


def test_gene_sp_bounds_refused_unless_upper_above_lower():
    with pytest.raises(SpecError, match="upper > lower"):
        GeneScoringSpec(sp_lower=2.0, sp_upper=2.0).validate()


def test_differences_from_reference_name_each_field():
    spec = _small()
    diffs = spec.differences_from_reference()
    assert diffs["input.dim"] == (2048, 6)
    assert diffs["encoder.layers"] == (3, 2)
    assert reference_spec("base").differences_from_reference() == {}


# --------------------------------------------------------------------------- builder


def test_reference_widths_follow_the_source():
    bundle = build_bundle(reference_spec("finetune", "disease"), VIEW)
    widths = [conv.out_channels * conv.heads for conv in bundle.encoder.convs]
    assert widths == [2048, 1024, 512]
    assert bundle.encoder.convs[0].in_channels == 2048
    assert bundle.lp_head.relation_weights.shape == (2, 512)
    assert bundle.task_head.patient_tower.first.in_features == 512
    # C2: a norm after every layer but the last.
    assert len(bundle.encoder.layer_norms) == len(bundle.encoder.batch_norms) == 2
    # U4: the attention settings are explicit.
    conv = bundle.encoder.convs[0]
    assert conv.add_self_loops and not conv.share_weights and conv.dropout == 0.0
    assert conv.res is None


def test_base_has_no_task_parts_and_unbuilt_kinds_refuse():
    bundle = build_bundle(_small(), VIEW)
    assert bundle.patient_encoder is None and bundle.task_head is None
    with pytest.raises(SpecError, match="not yet behind"):
        build_bundle(_small(kind="hgt"), VIEW)
    with pytest.raises(SpecError, match="lacks"):
        build_bundle(_small(), {"num_nodes": 5})


def test_relation_aware_gatv2_takes_edge_types():
    bundle = build_bundle(_small(relation_edge_dim=3), VIEW)
    assert bundle.encoder.uses_relations
    assert bundle.encoder.relation_embedding.weight.shape == (2, 3)


# --------------------------------------------------------------------------- norm and inference


def test_graph_mode_norm_is_its_formula_not_per_node():
    from torch_geometric.nn import LayerNorm

    x = torch.tensor([[1.0, 2.0], [3.0, 10.0], [-4.0, 0.0]])
    norm = LayerNorm(2, mode="graph").eval()
    flat = x.flatten()
    mean = flat.mean()
    std = ((flat - mean) ** 2).mean().sqrt()
    expected = (x - mean) / (std + 1e-5)
    assert torch.allclose(norm(x), expected, atol=1e-6)
    per_node = LayerNorm(2, mode="node").eval()(x)
    assert not torch.allclose(norm(x), per_node, atol=1e-3)


def _graph(n=10):
    # Two blocks with different scales, edges across them, a shared neighbour
    # (node 0 feeds both blocks).
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
             (0, 5), (0, 7), (2, 9), (6, 1), (8, 3)]
    both = edges + [(b, a) for a, b in edges]
    return torch.tensor(both, dtype=torch.long).T


def _one_shot(encoder, x, edge_index):
    # The whole graph in one forward of each layer, written independently of
    # the encoder's routing.
    for i, conv in enumerate(encoder.convs):
        x = conv(x, edge_index)
        if i < len(encoder.convs) - 1:
            x = encoder.layer_norms[i](x)
            x = torch.nn.functional.leaky_relu(x, 0.01)
            x = encoder.batch_norms[i](x)
    return x


@pytest.mark.parametrize("chunk_size", [3, 7])
def test_chunked_full_view_equals_one_shot(chunk_size):
    torch.manual_seed(0)
    view = {**VIEW, "num_nodes": 10}
    bundle = build_bundle(_small(layers=3), view).eval()
    with torch.no_grad():
        bundle.input.table.weight[5:] *= 4.0
        bundle.input.table.weight[5:] += 3.0
        for bn in bundle.encoder.batch_norms:
            bn.running_mean.uniform_(-0.5, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
    ei = _graph()
    expected = _one_shot(bundle.encoder, bundle.input.table.weight, ei)
    got = bundle.node_vectors_full(ei, chunk_size=chunk_size)
    assert torch.allclose(got, expected, atol=1e-5)


def test_full_view_inference_refuses_training_mode():
    bundle = build_bundle(_small(), VIEW)
    with pytest.raises(RuntimeError, match="evaluation mode"):
        bundle.node_vectors_full(_graph(5)[:, :4])


def test_sampled_rule_equals_full_rule_when_the_sample_is_the_graph():
    # A star: seed 0, hop 1 = the four leaves, hop 2 adds no node. Layer 1's
    # destinations are then every node, so its norm sees the full node set.
    torch.manual_seed(1)
    bundle = build_bundle(_small(layers=2), VIEW).eval()
    hop1 = [(1, 0), (2, 0), (3, 0), (4, 0)]
    hop2 = [(0, 1), (0, 2), (0, 3), (0, 4)]
    sampled_ei = torch.tensor(hop1 + hop2, dtype=torch.long).T
    with torch.no_grad():
        sampled = bundle.node_vectors_sampled(torch.arange(5), sampled_ei, [1, 4, 0], [4, 4])
        full = bundle.node_vectors_full(sampled_ei)
    assert sampled.shape == (1, bundle.width)
    assert torch.allclose(sampled[0], full[0], atol=1e-5)


def test_sampled_rule_needs_one_hop_per_layer():
    bundle = build_bundle(_small(layers=2), VIEW)
    with pytest.raises(ValueError, match="2 hops"):
        bundle.node_vectors_sampled(torch.arange(5), torch.zeros(2, 0, dtype=torch.long),
                                    [1, 4], [4])


def test_gradient_reaches_the_table_every_layer_and_drawn_relations():
    torch.manual_seed(2)
    bundle = build_bundle(_small(layers=2), VIEW).train()
    # Seeds 0 and 4, hop 1 nodes 1 2 3 (local order 0 4 1 2 3). Hop-1 edges
    # 1→0, 2→0, 3→4; hop-2 edges 0→1, 0→2, 2→3; local ids below.
    z = bundle.node_vectors_sampled(torch.tensor([0, 4, 1, 2, 3]),
                                    torch.tensor([[2, 3, 4, 0, 0, 3], [0, 0, 1, 2, 3, 4]]),
                                    [2, 3, 0], [3, 3])
    rel = torch.tensor([0])
    pos = bundle.lp_head.score(z[0:1], rel, z[1:2])
    neg = bundle.lp_head.score(z[1:2], rel, z[0:1]) * 0.5
    bundle.lp_head.hinge(pos, neg).backward()
    assert bundle.input.table.weight.grad is not None
    assert bundle.input.table.weight.grad.abs().sum() > 0
    for conv in bundle.encoder.convs:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in conv.parameters())
    assert bundle.lp_head.relation_weights.grad[0].abs().sum() > 0
    assert bundle.lp_head.relation_weights.grad[1].abs().sum() == 0


# --------------------------------------------------------------------------- save and reload


def test_save_and_reload_through_the_builder(tmp_path):
    torch.manual_seed(3)
    # Width 4 x 2 = 8 takes the patient Transformer's 8 heads.
    spec = finetune_spec_from_parent(_small(output=4), "patients_like_me")
    bundle = build_bundle(spec, VIEW).eval()
    path = tmp_path / "bundle.pt"
    torch.save({"model_spec": spec.to_dict(), "view": bundle.view.record(),
                "model_state_dict": bundle.state_dict()}, path)
    loaded = torch.load(path, weights_only=True)
    again = build_bundle(ModelSpec.from_dict(loaded["model_spec"]), loaded["view"]).eval()
    again.load_state_dict(loaded["model_state_dict"], strict=True)
    assert list(again.state_dict()) == list(bundle.state_dict())
    ei = _graph(5)[:, :8].clamp(max=4)
    ei = ei[:, ei[0] != ei[1]]
    with torch.no_grad():
        assert torch.equal(again.node_vectors_full(ei), bundle.node_vectors_full(ei))


def test_reload_refuses_weights_that_do_not_match_the_spec():
    spec = _small()
    bundle = build_bundle(spec, VIEW)
    other = build_bundle(_small(hidden=5), VIEW)
    with pytest.raises(RuntimeError):
        other.load_state_dict(bundle.state_dict(), strict=True)


def test_spec_encoder_defaults_are_the_reference():
    enc = EncoderSpec()
    assert (enc.layers, enc.hidden, enc.output, enc.heads, enc.dropout) == (3, 256, 128, 4, 0.3)
    assert enc.width == 512
