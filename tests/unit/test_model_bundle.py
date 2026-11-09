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
  (regression 6);
- every field holds the type the spec declares for it (one rule, pinned to
  every declared field): a `bool` true or false, never "false", "true", 1 or 0
  (the reviewer's K8-1 recheck: "false" built as true); an `int` an integer,
  never a boolean, a float or a string; a `float` a finite number, an integer
  included, never a boolean, a string, NaN or infinity; a `str` a string; a
  tuple a list of its element type, of its fixed length; a mapping of strings;
  `Optional` also null (the recheck's follow-up: `transformer_layers: true`
  built one layer and `attention_slope: "0.2"` failed in the first step);
- weights the spec shares (`encoder.share_weights`) are one set: a state whose
  shared keys agree loads, one holding two different values is refused by
  every load (serving's and measurement's `load_bundle`, a finetune's parent
  transfer, a resume's restore).

Module: tests/unit/test_model_bundle.py
"""
from __future__ import annotations

import dataclasses
import json
import re

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


# --------------------------------------------------------------------------- declared types
#: Every field the spec declares, by section ("" for the top level), with its
#: declared type as written; sections themselves are walked, not listed.
DECLARED_FIELDS = {
    ("", "learning_stage"): "str",
    ("", "sources"): "Dict[str, str]",
    ("graph_view", "rule"): "str",
    ("graph_view", "drop_native_self_loops"): "bool",
    ("graph_view", "deduplicate"): "bool",
    ("graph_view", "largest_component"): "bool",
    ("graph_view", "exclude_relations"): "Tuple[str, ...]",
    ("input", "kind"): "str",
    ("input", "dim"): "int",
    ("input", "init"): "str",
    ("input", "type_embedding"): "bool",
    ("encoder", "kind"): "str",
    ("encoder", "layers"): "int",
    ("encoder", "hidden"): "int",
    ("encoder", "output"): "int",
    ("encoder", "heads"): "int",
    ("encoder", "first_hidden_multiplier"): "int",
    ("encoder", "dropout"): "float",
    ("encoder", "attention_dropout"): "float",
    ("encoder", "attention_slope"): "float",
    ("encoder", "add_self_loops"): "bool",
    ("encoder", "share_weights"): "bool",
    ("encoder", "concat"): "bool",
    ("encoder", "residual"): "bool",
    ("encoder", "norm"): "str",
    ("encoder", "activation_slope"): "float",
    ("encoder", "relation_edge_dim"): "Optional[int]",
    ("lp_head", "kind"): "str",
    ("lp_head", "margin"): "float",
    ("lp_head", "relation_init"): "str",
    ("patient_encoder", "transformer_layers"): "int",
    ("patient_encoder", "transformer_heads"): "int",
    ("patient_encoder", "feedforward"): "int",
    ("patient_encoder", "dropout"): "float",
    ("task_head", "task"): "str",
    ("task_head", "projection_slope"): "float",
    ("task_head", "distance_normalisation_eps"): "float",
    ("task_head", "softmax_scale"): "float",
    ("task_head", "multi_similarity"): "Tuple[float, float, float, float]",
    ("task_head.gene", "eta"): "float",
    ("task_head.gene", "sp_lower"): "float",
    ("task_head.gene", "sp_upper"): "float",
    ("task_head.gene", "sp_aggregation"): "str",
    ("task_head.gene", "augmentation"): "bool",
    ("task_head.gene", "augmentation_k"): "int",
    ("task_head.gene", "augmentation_omega"): "float",
}

#: Values each declared type refuses (booleans: `test_a_boolean_setting_refuses_anything_else`).
WRONG_VALUES = {
    "int": [True, 3.0, "3", None],
    "Optional[int]": [True, 3.0, "3"],
    "float": [True, "0.1", None, float("nan"), float("inf"), 10**400],
    "str": [1, True, None, ["x"]],
    "Tuple[str, ...]": ["disease|assoc|phenotype", {"disease|assoc|phenotype": 1}, [1], [None]],
    "Tuple[float, float, float, float]": [
        "abcd", ["a", "b", "c", "d"], [1.0, 20.0, True, 0.4], [1.0, 20.0, 1.0, "0.4"],
        [1.0, 20.0, 1.0], [1.0, 20.0, 1.0, 0.4, 1.0]],
    "Dict[str, str]": [[["encoder.layers", "x"]], {"encoder.layers": 1}, "x"],
}

#: An integer each `float` field takes within its range (an integer is a number).
INTEGER_FOR_FLOAT = {("encoder", "dropout"): 0, ("encoder", "attention_dropout"): 0,
                     ("patient_encoder", "dropout"): 0, ("task_head.gene", "sp_upper"): 7}


def _declared_fields():
    """(section, field, declared type) of every field the spec's dataclasses
    declare, walking each section field (`Optional[...]` of one too)."""
    from dataclasses import fields

    from src.models.bundle import spec as spec_module

    def walk(cls, section):
        for f in fields(cls):
            inner = f.type[len("Optional["):-1] if f.type.startswith("Optional[") else f.type
            kind = getattr(spec_module, inner, None)
            if isinstance(kind, type) and dataclasses.is_dataclass(kind):
                yield from walk(kind, f"{section}.{f.name}".lstrip("."))
            else:
                yield section, f.name, f.type

    return {(section, name): declared for section, name, declared in walk(ModelSpec, "")}


def _with(section, name, value):
    """A gene child's spec as plain data with `section`.`name` set to `value`."""
    data, holder = _spec_data(section)
    holder[name] = value
    return data


def _path(section, name):
    return f"{section}.{name}" if section else name


def test_every_declared_field_is_listed():
    """The table above is every field of every section with its declared type,
    so the checks below cover each one; a field added or retyped fails here
    until it is listed."""
    assert _declared_fields() == DECLARED_FIELDS
    assert {key for key, kind in DECLARED_FIELDS.items() if kind == "bool"} == {
        (section, name) for section, name, _ in BOOLEAN_FIELDS}


@pytest.mark.parametrize("section, name", [key for key, kind in DECLARED_FIELDS.items()
                                           if kind != "bool"],
                         ids=[_path(*key) for key, kind in DECLARED_FIELDS.items()
                              if kind != "bool"])
def test_a_field_takes_its_declared_type(section, name):
    """Each declared type's own values pass and read back as written: an int
    field an integer, a float field an integer or a float, a string field its
    string, a tuple a list of its elements, an optional one null too, the
    sources a mapping of strings."""
    kind = DECLARED_FIELDS[(section, name)]
    current = _spec_data(section)[1][name]
    accepted = {
        "int": lambda: [current, current + 1],
        "Optional[int]": lambda: [None, 3],
        "float": lambda: [float(current), INTEGER_FOR_FLOAT.get((section, name), 1)],
        "str": lambda: [current],
        "Tuple[str, ...]": lambda: [[], ["disease|assoc|phenotype"]],
        "Tuple[float, float, float, float]": lambda: [[1, 20, 1, 0], [1.0, 20.0, 1.0, 0.4]],
        "Dict[str, str]": lambda: [{}, {"encoder.layers": "operator-set"}],
    }[kind]()
    for value in accepted:
        data = _with(section, name, value)
        assert ModelSpec.from_dict(data).to_dict() == data, (kind, value)


@pytest.mark.parametrize("section, name, value", [
    (*key, value) for key, kind in DECLARED_FIELDS.items() if kind != "bool"
    for value in WRONG_VALUES[kind]],
    ids=[f"{_path(*key)}-{index}" for key, kind in DECLARED_FIELDS.items() if kind != "bool"
         for index, _ in enumerate(WRONG_VALUES[kind])])
def test_a_field_refuses_another_type(section, name, value):
    """The regression: before the rule only some fields were type-checked, so
    patient_encoder.transformer_layers true built one layer, dropout "0.1",
    encoder.attention_slope "0.2", lp_head.margin true, task_head.softmax_scale
    "1.0" and multi_similarity ["a", "b", "c", "d"] were accepted (the slope
    failing in the run's first step). Now each is refused, naming the field."""
    with pytest.raises(SpecError, match=rf"^{re.escape(_path(section, name))}(\[[^\]]*\])? must "):
        ModelSpec.from_dict(_with(section, name, value))


# --------------------------------------------------------------------------- booleans
#: Every field the spec declares `bool`, by section, with the value each must
#: hold here (`encoder.concat` is refused false by its own rule, README C1).
BOOLEAN_FIELDS = [
    ("graph_view", "drop_native_self_loops", (True, False)),
    ("graph_view", "deduplicate", (True, False)),
    ("graph_view", "largest_component", (True, False)),
    ("input", "type_embedding", (True, False)),
    ("encoder", "add_self_loops", (True, False)),
    ("encoder", "share_weights", (True, False)),
    ("encoder", "concat", (True,)),
    ("encoder", "residual", (True, False)),
    ("task_head.gene", "augmentation", (True, False)),
]


def _spec_data(section):
    """A gene child's spec as plain data (it has every section), and the
    mapping that holds `section`'s fields (the spec itself for "")."""
    data = finetune_spec_from_parent(_small(), "gene").to_dict()
    holder = data
    for part in filter(None, section.split(".")):
        holder = holder[part]
    return data, holder


@pytest.mark.parametrize("section, name, values", BOOLEAN_FIELDS,
                         ids=[f"{s}.{n}" for s, n, _ in BOOLEAN_FIELDS])
def test_a_boolean_setting_takes_true_and_false(section, name, values):
    for value in values:
        data, holder = _spec_data(section)
        holder[name] = value
        spec = ModelSpec.from_dict(data)
        assert spec.to_dict() == data


@pytest.mark.parametrize("value", ["false", "true", 1, 0, None],
                         ids=["str-false", "str-true", "one", "zero", "null"])
@pytest.mark.parametrize("section, name, values", BOOLEAN_FIELDS,
                         ids=[f"{s}.{n}" for s, n, _ in BOOLEAN_FIELDS])
def test_a_boolean_setting_refuses_anything_else(section, name, values, value):
    """The regression: before the fix only graph_view's flags and
    input.type_embedding were checked, so encoder.add_self_loops "false"
    built a layer with self-loops, encoder.residual "false" one with a
    residual, and task_head.gene.augmentation "false" augmented."""
    data, holder = _spec_data(section)
    holder[name] = value
    with pytest.raises(SpecError, match=rf"{section}\.{name} must be a boolean"):
        ModelSpec.from_dict(data)


def test_a_spec_built_in_code_is_held_to_the_same_rule():
    """A spec built with `dataclasses.replace`, not read from a dict, passes
    the same gate (`ModelSpec.validate`), which the builder calls: a value of
    another type, a list where a tuple is declared and a mapping in place of a
    section are refused."""
    spec = _small()
    child = finetune_spec_from_parent(spec, "disease")
    for built, message in [
        (dataclasses.replace(spec, encoder=dataclasses.replace(spec.encoder, residual="false")),
         "encoder.residual must be a boolean"),
        (dataclasses.replace(spec, encoder=dataclasses.replace(spec.encoder,
                                                               attention_slope="0.2")),
         "encoder.attention_slope must be a finite number"),
        (dataclasses.replace(child, patient_encoder=dataclasses.replace(
            child.patient_encoder, transformer_layers=True)),
         "patient_encoder.transformer_layers must be an integer"),
        (dataclasses.replace(child, task_head=dataclasses.replace(
            child.task_head, multi_similarity=[1.0, 20.0, 1.0, 0.4])),
         r"task_head.multi_similarity must be a list \(a tuple in a spec built in code\)"),
        (dataclasses.replace(spec, lp_head=spec.lp_head.__dict__),
         "lp_head must be a LPHeadSpec section"),
    ]:
        with pytest.raises(SpecError, match=message):
            build_bundle(built, VIEW)


# --------------------------------------------------------------------------- shared weights
def _shared_child():
    """A disease child whose GATv2 layers share their source and target
    transforms, and its model record as a checkpoint carries it."""
    from src.models.bundle.record import MODEL_RECORD_VERSION, ModelRecord

    spec = finetune_spec_from_parent(_small(output=4, share_weights=True), "disease")
    bundle = build_bundle(spec, VIEW)
    record = ModelRecord(model_spec=spec, graph_view=bundle.view,
                         data_roles={"kg": "k" * 64}, graph_exposure={}, loss_identity={},
                         operation="finetune", family_id="r",
                         parent={"sha256": "a" * 64, "run": "r"})
    return bundle, {"record_version": MODEL_RECORD_VERSION, **record.to_payload()}


def _contradicted(state):
    """`state` with the first layer's two shared transforms set apart."""
    state = {key: value.clone() for key, value in state.items()}
    state["encoder.convs.0.lin_l.weight"].fill_(1.0)
    state["encoder.convs.0.lin_r.weight"].fill_(2.0)
    return state


def test_consistent_shared_weights_load(tmp_path):
    """A state as training writes it (both keys name the one tensor, saved and
    read back) loads, through every load."""
    from src.models.bundle.record import load_bundle

    bundle, record = _shared_child()
    conv = bundle.encoder.convs[0]
    assert conv.lin_l is conv.lin_r
    path = tmp_path / "child.pt"
    torch.save({**record, "state_dict": bundle.state_dict()}, path)
    payload = torch.load(path, weights_only=True)
    loaded, _ = load_bundle(payload, learning_stage="finetune", task="disease")
    assert torch.equal(loaded.encoder.convs[0].lin_l.weight, conv.lin_l.weight)
    # Shared values that agree, NaN included, are one set.
    state = {key: value.clone() for key, value in payload["state_dict"].items()}
    for key in ("encoder.convs.0.lin_l.weight", "encoder.convs.0.lin_r.weight"):
        state[key][0, 0] = float("nan")
    build_bundle(bundle.spec, VIEW).load_state_dict(state, strict=True)


def test_contradictory_shared_weights_are_refused_before_serving_or_finetune():
    """The regression: before the fix every load took whichever of the two
    values came last. Now serving and measurement (`load_bundle`), a
    finetune's or a branch's parent transfer and a resume's restore refuse,
    before anything is copied."""
    import sys
    from pathlib import Path

    from src.models.bundle.record import ModelRecordError, load_bundle

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from train_model import _transfer_parent_weights

    bundle, record = _shared_child()
    state = _contradicted(bundle.state_dict())
    message = ("two different values for one shared parameter "
               r"\(encoder.convs.0.lin_l.weight and encoder.convs.0.lin_r.weight\)")

    with pytest.raises(ModelRecordError, match=message):
        load_bundle({**record, "state_dict": state}, learning_stage="finetune", task="disease")
    # A finetune of a base whose shared weights disagree: the child's graph
    # side is loaded from the parent.
    base_state = {k: v for k, v in state.items()
                  if not k.startswith(("patient_encoder.", "task_head."))}
    child = build_bundle(bundle.spec, VIEW)
    before = {k: v.clone() for k, v in child.state_dict().items()}
    with pytest.raises(RuntimeError, match=message):
        _transfer_parent_weights(child, base_state, new_parts=("patient_encoder", "task_head"))
    assert all(torch.equal(before[k], v) for k, v in child.state_dict().items())
    # A resume's restore loads the parent's state into the built bundle.
    with pytest.raises(RuntimeError, match=message):
        build_bundle(bundle.spec, VIEW).load_state_dict(state)


def test_a_union_of_two_types_is_not_read_as_one():
    """The type rule reads one declared type, optionally None. A field
    declared as a union of two types gets a SpecError naming it, never a bare
    unpacking error."""
    from typing import Optional, Union

    from src.models.bundle.spec import SpecError, _require_type

    _require_type("x", None, Optional[int])
    _require_type("x", 3, Optional[int])
    with pytest.raises(SpecError, match="declared as"):
        _require_type("x", 3, Union[int, float])
