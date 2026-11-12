"""
A model's capability description, and protocols judged by it (Stage 3, E2, E5).
===============================================================================
`docs/working/research-comparison/README.md` §5 Stage 3, §6 E2 and E5;
`src/models/bundle/capabilities.py`, `src/models/bundle/spec.py`
(`EncoderDeclaration`), `src/evaluation/protocols.py`.

- **One description per model, from its validated spec and its built
  components**: for every record the spec can write, the representations, the
  heads with their named scoring semantics, the input needs and the
  computations its encoder declares; plain data, the same on every read; the
  built bundle agrees with it (width, relation and type reading).
- **GATv2's rules stay in GATv2's declaration**: its width is output x heads,
  its sampler takes one hop per layer and its concat rule refuses, in the same
  sentences as before.
- **A test-only encoder double** (declared here, never shipped) with another
  output width and another sampling need goes through the spec's checks, the
  hop check, the capability description, the one listing reader, the shared
  protocol judgment, the Test page's options and the runner's judgment
  without any gatv2 name or layers == hops rule firing; and its build is still
  refused, having no implementation.
- **An encoder kind this build does not implement** is read, offers no
  computation, is refused by a protocol in words that say so, is refused by
  the hop check (so by the base start, whatever its hop list) in the builder's
  own sentence, and is refused at build; a declared kind that offers no
  sampled computation is refused by the hop check too; a kind outside the list
  is refused at read.
- **E5**: a model offering two task heads is taken by both task protocols; an
  input no evaluator supplies refuses; the data kinds are judged, a generated
  split's counts are not.

Module: tests/unit/test_model_capabilities.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from src.evaluation import protocols
from src.models.bundle import capabilities as cap
from src.models.bundle import spec as spec_module
from src.models.bundle.capabilities import model_capabilities
from src.models.bundle.spec import (
    FULL_VIEW,
    SAMPLED,
    TASKS,
    ModelSpec,
    SpecError,
    ViewShape,
    encoder_not_built,
    hops_problem,
    reference_spec,
)
from tests.fixtures.capabilities import DoubleDeclaration, declare_encoder

#: A view shape as a checkpoint's record names it.
VIEW = ViewShape("d" * 64, 6, ("disease", "gene", "phenotype"), ("a|r|b", "b|r_rev|a"))
#: A supplied cohort version's counts that list causal and candidate genes.
GENES = {"source_cases": 3, "genes": {"cases_with_causal_genes": 2,
                                      "cases_with_candidate_genes": 2}}


def _writable() -> list:
    """Every (learning stage, task) the spec can write, with its reference spec."""
    return [("base", None, reference_spec("base"))] + [
        ("finetune", task, reference_spec("finetune", task)) for task in TASKS]


def _plain(value: Any) -> bool:
    """Strings, integers, None and tuples of them, nothing else."""
    if value is None or isinstance(value, str):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    return isinstance(value, tuple) and all(_plain(v) for v in value)


# --------------------------------------------------------------------------- the description
def test_every_writable_model_is_described_from_its_spec():
    for stage, task, spec in _writable():
        described = model_capabilities(spec)
        assert tuple(described) == cap.CAPABILITY_FIELDS
        assert all(_plain(v) for v in described.values()), described
        assert described["learning_stage"] == stage and described["encoder"] == "gatv2"
        heads = described["heads"]
        # The LP head's semantics named, never read off the attribute name.
        assert heads[0] == (cap.LINK_PREDICTION, "distmult_tanh",
                            "distmult_raw_score/tanh_hinge/v1")
        if task is None:
            assert described["representations"] == (cap.NODE_VECTORS,) and len(heads) == 1
        else:
            assert described["representations"] == (cap.NODE_VECTORS,
                                                     cap.PATIENT_REPRESENTATION)
            assert heads[1:] == ((cap.TASK, task, cap.TASK_HEAD_SEMANTICS[task]),)
        assert described["node_inputs"] == (cap.NODE_IDS,)
        assert described["relation_inputs"] == (cap.RELATION_IDS,)
        assert described["computation"] == (FULL_VIEW, SAMPLED)
        assert described["sampled_hops"] == spec.encoder.layers == 3
        assert described["node_vector_width"] == spec.encoder.output * spec.encoder.heads == 512
        # The same on every read, JSON included (a list's answer).
        assert model_capabilities(spec) == described
        assert json.dumps(described) == json.dumps(model_capabilities(spec))


def test_every_task_and_lp_head_kind_states_its_scoring_semantics():
    assert set(cap.TASK_HEAD_SEMANTICS) == set(TASKS)
    assert set(cap.LP_HEAD_SEMANTICS) == {reference_spec("base").lp_head.kind}
    assert len(set(cap.TASK_HEAD_SEMANTICS.values())) == len(TASKS)


def test_the_input_needs_follow_the_spec():
    """Node types where the input adds a type embedding; each edge's relation
    where the encoder reads it (GATv2 with `relation_edge_dim`)."""
    spec = reference_spec("finetune", "disease")
    typed = replace(spec, input=replace(spec.input, type_embedding=True)).validate()
    assert model_capabilities(typed)["node_inputs"] == (cap.NODE_IDS, cap.NODE_TYPES)
    aware = replace(spec, encoder=replace(spec.encoder, relation_edge_dim=8)).validate()
    assert model_capabilities(aware)["relation_inputs"] == (cap.EDGE_RELATION_IDS,
                                                            cap.RELATION_IDS)
    # The evaluators supply both, so neither changes what is offered.
    for described in (model_capabilities(typed), model_capabilities(aware)):
        assert protocols.available(described) == (("disease", "kg_link_prediction"), None)


def test_the_built_bundle_agrees_with_its_description():
    """The one builder's components say what the description says: the node
    vectors' width, whether the input reads types and whether the encoder
    reads each edge's relation."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.models.bundle.builder import build_bundle

    small = replace(reference_spec("base"), input=replace(reference_spec("base").input, dim=8))
    for spec in (small,
                 replace(small, input=replace(small.input, type_embedding=True)),
                 replace(small, encoder=replace(small.encoder, relation_edge_dim=4,
                                                hidden=4, output=3, heads=2))):
        spec = spec.validate()
        described = model_capabilities(spec)
        bundle = build_bundle(spec, VIEW)
        assert bundle.width == described["node_vector_width"]
        assert (bundle.input.type_embedding is not None) == (
            cap.NODE_TYPES in described["node_inputs"])
        assert bundle.encoder.uses_relations == (
            cap.EDGE_RELATION_IDS in described["relation_inputs"])
        assert bundle.encoder.uses_node_types == (cap.NODE_TYPES in described["node_inputs"]
                                                  and not spec.input.type_embedding)


def test_a_head_whose_semantics_this_build_does_not_state_never_fails_a_listing(monkeypatch):
    """Reading never depends on the capability judgment: a head whose scoring
    semantics this build does not state is described with None, so the one
    listing reader still lists the model; no protocol scores with that head,
    and a protocol that does not read it still takes the model."""
    spec = reference_spec("finetune", "disease")
    monkeypatch.setattr(cap, "LP_HEAD_SEMANTICS", {})
    item = _listed(spec)
    assert item["record"] == "model" and item["record_problem"] is None
    assert item["capabilities"]["heads"][0] == (cap.LINK_PREDICTION, "distmult_tanh", None)
    assert all(_plain(v) for v in item["capabilities"].values())
    assert protocols.available(item["capabilities"]) == (("disease",), None)
    monkeypatch.setattr(cap, "TASK_HEAD_SEMANTICS", {})
    item = _listed(spec)
    assert item["record"] == "model"
    assert item["capabilities"]["heads"][1] == (cap.TASK, "disease", None)
    assert protocols.available(item["capabilities"])[0] == ()
    assert protocols.model_refusal("disease", item["capabilities"]) is not None


def test_the_built_encoder_reads_what_its_declaration_says(monkeypatch):
    """A built encoder's width and the inputs it reads are its declaration's
    answers, not a second statement of them: a declaration that says another
    thing is followed."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.models.bundle.encoders import GATv2Encoder

    class _ReadsTypes(spec_module.GATv2Declaration):
        def reads_node_types(self, spec: Any) -> bool:
            return True

    encoder_spec = replace(reference_spec("base").encoder, hidden=4, output=3, heads=2)
    built = GATv2Encoder(encoder_spec, 8, 2)
    assert (built.width, built.uses_node_types, built.uses_relations) == (6, False, False)
    monkeypatch.setattr(spec_module, "ENCODER_DECLARATIONS", {"gatv2": _ReadsTypes()})
    assert GATv2Encoder(encoder_spec, 8, 2).uses_node_types is True


# --------------------------------------------------------------------------- GATv2's own rules
def test_gatv2_keeps_its_rules_and_its_sentences():
    """The width, the one-hop-per-layer rule and the concat rule are GATv2's
    declaration's, said as before."""
    encoder = reference_spec("base").encoder
    assert encoder.width == encoder.output * encoder.heads
    assert hops_problem([15, 10, 5], encoder) is None
    assert hops_problem([15, 10], encoder) == (
        "num_neighbors needs 3 numbers, one hop per encoder layer, not 2.")
    with pytest.raises(SpecError, match=r"^encoder\.concat must be true: widths are output x "
                                        r"heads \(README C1\)$"):
        replace(encoder, concat=False).validate()
    for kind in ("hgt", "sage", "hetero_gat"):
        with pytest.raises(SpecError, match=r"^encoder\.relation_edge_dim applies to gatv2 only"):
            replace(encoder, kind=kind, relation_edge_dim=4).validate()


# --------------------------------------------------------------------------- a test-only double
@pytest.fixture
def double(monkeypatch):
    """The test-only double (`tests.fixtures.capabilities.DoubleDeclaration`)
    declared for this test only, and a maker of its specs."""
    declare_encoder(monkeypatch, DoubleDeclaration())

    def make(stage: str = "finetune", task: str | None = "disease") -> ModelSpec:
        spec = reference_spec(stage, task)
        return replace(spec, encoder=replace(spec.encoder, kind="double", layers=3, hidden=96,
                                             output=32, heads=2, concat=False)).validate()

    return make


def _listed(spec: ModelSpec) -> dict:
    """The one listing reader's summary of a checkpoint whose record names `spec`."""
    from src.models.bundle.record import (
        MODEL_RECORD_VERSION,
        ModelRecord,
        listed_record,
    )

    lineage = ({"operation": "new", "family_id": "root"} if spec.learning_stage == "base" else
               {"operation": "finetune", "family_id": "root",
                "parent": {"sha256": "a" * 64, "run": "root"}})
    record = ModelRecord(model_spec=spec, graph_view=VIEW, data_roles={}, graph_exposure={},
                         loss_identity={}, **lineage)
    return listed_record({"record_version": MODEL_RECORD_VERSION, **record.to_payload()})


def test_a_test_only_double_passes_the_shared_checks_without_a_gatv2_rule(double, tmp_path):
    """Another width (hidden, not output x heads) and another sampling need
    (two hops for three layers): the spec reads (no concat rule), the hop
    check asks the double's declaration, the description carries both, the
    listing carries the description, and the protocol judgment, the Test
    page's options and Start, and the runner's judgment take it as they take
    a GATv2 disease child. No sentence names gatv2 or one hop per layer."""
    pytest.importorskip("gradio")
    pytest.importorskip("fastapi")
    from src.api.services.training_manager import TestRunInputs, _new_bundle_run
    from src.webui.components import test_page as tp

    spec = double()
    assert spec.encoder.width == 96 != spec.encoder.output * spec.encoder.heads
    said = []
    assert hops_problem([15, 10], spec.encoder) is None
    said.append(hops_problem([15, 10, 5], spec.encoder))
    assert said[-1] == "num_neighbors needs 2 numbers, one hop per sampled ring the double " \
                       "declares, not 3."
    described = model_capabilities(spec)
    assert (described["encoder"], described["node_vector_width"], described["sampled_hops"],
            described["computation"]) == ("double", 96, 2, (FULL_VIEW, SAMPLED))

    item = _listed(spec)
    assert item["record"] == "model" and item["capabilities"] == described
    assert protocols.available(item["capabilities"]) == (("disease", "kg_link_prediction"), None)
    assert tp.measurement_choices(item) == [("Disease", "disease"),
                                            ("KG link prediction", "kg_link_prediction")]
    version = {"kind": protocols.SUPPLIED, "counts": GENES}
    assert tp.measure_refusal(item, version, "disease") is None
    for name in ("gene", "patients_like_me"):
        said.append(tp.measure_refusal(item, version, name))
        assert said[-1] == protocols.model_refusal(name, described) == (
            f"the {name} measurement takes a finetune model with the {name} head; this "
            "checkpoint is a finetune model with the disease head. It can be measured by "
            "disease, kg_link_prediction.")
    registry = tmp_path / "registry"
    registry.mkdir()
    inputs = TestRunInputs(
        registry=registry, version_label="v1", version_sha256="e" * 64,
        workspace=tmp_path / "ws", checkpoint=tmp_path / "ws" / "c.pt",
        checkpoint_sha256="c" * 64, protocol="disease", learning_stage="finetune",
        task="disease", capabilities=item["capabilities"], version_kind=protocols.SUPPLIED,
        version_counts=GENES)
    assert inputs.reports_dir() == registry.resolve() / "versions" / "v1" / "reports"

    # The base start's check asks the double's sampling need, not its layers.
    base = double("base", None)
    start = {"learning_stage": "base", "data_dir": str(tmp_path), "model_spec": base.to_dict()}
    assert _new_bundle_run({**start, "num_neighbors": [15, 10]}) == ("double", None)
    kind, why = _new_bundle_run({**start, "num_neighbors": [15, 10, 5]})
    said.append(why)
    assert kind is None and why == said[0]
    # The double's base is taken by the KG link-prediction protocol as a
    # GATv2 base is: its LP head's semantics and its full-view computation.
    assert protocols.available(model_capabilities(base)) == (("kg_link_prediction",), None)
    assert not any("gatv2" in s or "encoder layer" in s for s in said), said


def test_the_double_has_no_implementation_so_its_build_is_refused(double):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.models.bundle.builder import build_bundle

    with pytest.raises(SpecError) as refused:
        build_bundle(double(), VIEW)
    assert str(refused.value) == encoder_not_built("double")


# --------------------------------------------------------------------------- kinds not built
def test_a_kind_this_build_does_not_implement_offers_no_computation():
    """`hgt` is a kind the spec names and this build implements no encoder
    of: its record is read and listed, its model offers no computation, so a
    protocol refuses it in words that say so (never as a missing head), no hop
    list fits it (the builder's sentence), it has no width, and its build is
    refused."""
    spec = reference_spec("finetune", "disease")
    hgt = replace(spec, encoder=replace(spec.encoder, kind="hgt")).validate()
    described = model_capabilities(hgt)
    assert (described["computation"], described["sampled_hops"],
            described["node_vector_width"]) == ((), None, None)
    said = ("the disease measurement needs full-view computation, which this checkpoint's "
            "model (hgt encoder) does not offer in this build.")
    assert protocols.model_refusal("disease", described) == said
    assert protocols.available(described) == ((), said)
    assert _listed(hgt)["capabilities"] == described
    for hops in ([1], [15, 10, 5]):
        assert hops_problem(hops, hgt.encoder) == f"{encoder_not_built('hgt')}."
    with pytest.raises(SpecError, match="not yet behind the GraphEncoder contract"):
        hgt.encoder.width  # noqa: B018
    pytest.importorskip("torch_geometric")
    from src.models.bundle.builder import build_bundle

    with pytest.raises(SpecError) as refused:
        build_bundle(hgt, VIEW)
    assert str(refused.value) == encoder_not_built("hgt")


def test_a_declared_encoder_kind_is_exactly_a_built_one():
    """Decision S5: the architectures offered as trainable are the implemented
    declarations, `ENCODER_DECLARATIONS`, pinned equal to the builder's
    implementations (`builder._ENCODERS`), so a list read from the
    declarations never offers a kind the builder refuses, and a built kind is
    never missing from it. `ENCODER_KINDS` names more (kinds a spec may name
    and a run refuses), never the options."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.config.model_types import ENCODER_KINDS
    from src.models.bundle import builder

    assert set(spec_module.ENCODER_DECLARATIONS) == set(builder._ENCODERS) == {"gatv2"}
    assert set(spec_module.ENCODER_DECLARATIONS) < set(ENCODER_KINDS)


@pytest.mark.parametrize("kind", ["hgt", "sage", "hetero_gat"])
def test_the_base_start_refuses_a_kind_this_build_does_not_implement(kind, tmp_path):
    """A run that cannot be built is refused at the start, in the builder's
    own sentence, whatever its hop list (a list of the reference GATv2's three
    hops included) and whatever its concat setting (GATv2's rule is not
    another kind's): research comparison §4 invariant 2."""
    pytest.importorskip("fastapi")
    from src.api.services.training_manager import _new_bundle_run

    base = reference_spec("base")
    start = {"learning_stage": "base", "data_dir": str(tmp_path)}
    for concat in (True, False):
        spec = replace(base, encoder=replace(base.encoder, kind=kind, concat=concat)).validate()
        for hops in ([15, 10, 5], [15, 10], [5]):
            assert _new_bundle_run({**start, "model_spec": spec.to_dict(),
                                    "num_neighbors": hops}) == (
                None, f"{encoder_not_built(kind)}."), (concat, hops)


def test_a_declared_kind_without_sampled_computation_fits_no_hop_list(monkeypatch, tmp_path):
    """A declaration that offers full-view computation only (a test-only
    double) has no sampled hops in its description, and the hop check, so the
    base start, refuses any hop list in words that say so, naming no gatv2
    rule."""
    pytest.importorskip("fastapi")
    from src.api.services.training_manager import _new_bundle_run

    class _FullViewOnly(DoubleDeclaration):
        kind = "full_only"
        computation = (FULL_VIEW,)

        def sampled_hops(self, spec: Any) -> int | None:
            return None

    declare_encoder(monkeypatch, _FullViewOnly())
    base = reference_spec("base")
    spec = replace(base, encoder=replace(base.encoder, kind="full_only", concat=False)).validate()
    described = model_capabilities(spec)
    assert (described["computation"], described["sampled_hops"]) == ((FULL_VIEW,), None)
    said = ("encoder kind 'full_only' offers no sampled computation in this build, which "
            "training's sampler needs.")
    for hops in ([15, 10, 5], [15, 10]):
        assert hops_problem(hops, spec.encoder) == said
        assert _new_bundle_run({"learning_stage": "base", "data_dir": str(tmp_path),
                                "model_spec": spec.to_dict(), "num_neighbors": hops}) == (
            None, said)
    assert "gatv2" not in said and "encoder layer" not in said


def test_a_kind_outside_the_list_is_refused_when_read():
    with pytest.raises(SpecError, match="unknown encoder kind 'double'"):
        replace(reference_spec("base").encoder, kind="double").validate()


# --------------------------------------------------------------------------- E5 and data kinds
def test_a_model_offering_two_task_heads_is_taken_by_both_task_protocols():
    """No layer assumes one task per model: a description carrying the
    disease and patients-like-me heads (a double) is taken by both, and a
    third names both heads."""
    described = model_capabilities(reference_spec("finetune", "disease"))
    two = {**described, "heads": (*described["heads"], protocols.task_head("patients_like_me"))}
    assert protocols.available(two) == (("disease", "patients_like_me", "kg_link_prediction"),
                                        None)
    assert protocols.model_refusal("gene", two) == (
        "the gene measurement takes a finetune model with the gene head; this checkpoint is a "
        "finetune model with the disease and patients_like_me heads. It can be measured by "
        "disease, patients_like_me, kg_link_prediction.")


def test_an_input_no_evaluator_supplies_is_refused():
    described = model_capabilities(reference_spec("finetune", "disease"))
    needs = {**described, "node_inputs": (*described["node_inputs"], "node_features")}
    said = ("this checkpoint's model needs node_features, which the disease measurement does "
            "not supply.")
    assert protocols.model_refusal("disease", needs) == said
    assert protocols.available(needs) == ((), said)


def test_the_data_kind_is_judged_and_a_generated_split_has_no_counts_to_judge():
    tasks = [name for name, row in protocols.PROTOCOLS.items() if row.head[0] == protocols.TASK]
    assert tasks == ["disease", "gene", "patients_like_me"]
    for name in tasks:
        assert protocols.inputs_refusal(name, protocols.GENERATED, None) is None
        assert protocols.inputs_refusal(name, protocols.SUPPLIED, GENES) is None
        # A kind no protocol names is said in neutral words, never as a cohort.
        assert protocols.inputs_refusal(name, "unknown_kind", GENES) == (
            f"the {name} measurement does not measure data of kind 'unknown_kind'; it measures "
            "a generated split or a supplied cohort version.")
        assert protocols.inputs_refusal(name, None, GENES) == (
            f"the {name} measurement does not measure data of no kind this build reads; it "
            "measures a generated split or a supplied cohort version.")
    # The counts are a registered version's: a gene measurement of one listing none is refused.
    assert protocols.inputs_refusal("gene", protocols.SUPPLIED, None).startswith(
        "this cohort version lists no causal genes")
    # The KG link-prediction protocol measures a KG edge set only, and says so.
    for kind, words in (("unknown_kind", "data of kind 'unknown_kind'"),
                        (None, "data of no kind this build reads")):
        assert protocols.inputs_refusal("kg_link_prediction", kind, {"pairs": 3}) == (
            f"the kg_link_prediction measurement does not measure {words}; it measures a KG "
            "edge set.")


def test_a_protocol_takes_exactly_the_writable_models_its_words_name():
    """The words a refusal uses for what a task protocol takes are the words it
    uses for the writable models it does take, and for no other; the KG
    link-prediction protocol's name the LP head's scoring semantics, which
    every writable model carries, and it takes every one."""
    for _stage, _task, spec in _writable():
        described = model_capabilities(spec)
        for protocol in protocols.PROTOCOLS.values():
            accepted = protocols.model_refusal(protocol.name, described) is None
            if protocol.head[0] == protocols.TASK:
                assert accepted == (protocols._takes(protocol) == protocols._described(described))
            else:
                assert accepted and protocol.head[2] in protocols._takes(protocol)
                assert protocol.head[2] in {head[2] for head in described["heads"]}
