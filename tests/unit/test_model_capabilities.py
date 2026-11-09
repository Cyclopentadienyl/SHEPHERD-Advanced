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
  computation, is refused by a protocol in words that say so, has no hops to
  check and is refused at build; a kind outside the list is refused at read.
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
    EncoderDeclaration,
    ModelSpec,
    SpecError,
    ViewShape,
    encoder_not_built,
    hops_problem,
    reference_spec,
)

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
    # The task evaluators supply both, so neither changes what is offered.
    for described in (model_capabilities(typed), model_capabilities(aware)):
        assert protocols.available(described) == (("disease",), None)


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
class _DoubleDeclaration(EncoderDeclaration):
    """A test-only encoder declaration, never shipped: its node vectors are
    `hidden` wide, its sampler takes two hops whatever its layers, it reads
    no relation feature and has no concat rule. No implementation exists."""

    kind = "double"
    computation = (FULL_VIEW, SAMPLED)
    hops_words = "one hop per sampled ring the double declares"

    def width(self, spec: Any) -> int:
        return spec.hidden

    def sampled_hops(self, spec: Any) -> int | None:
        return 2


@pytest.fixture
def double(monkeypatch):
    """The double declared for this test only: a kind of the spec's list and a
    declaration of this build's, both read when a judgment is made."""
    monkeypatch.setattr(spec_module, "ENCODER_KINDS", (*spec_module.ENCODER_KINDS, "double"))
    monkeypatch.setattr(spec_module, "ENCODER_DECLARATIONS",
                        {**spec_module.ENCODER_DECLARATIONS, "double": _DoubleDeclaration()})

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
    assert protocols.available(item["capabilities"]) == (("disease",), None)
    assert tp.measurement_choices(item) == [("Disease", "disease")]
    version = {"kind": protocols.SUPPLIED, "counts": GENES}
    assert tp.measure_refusal(item, version, "disease") is None
    for name in ("gene", "patients_like_me"):
        said.append(tp.measure_refusal(item, version, name))
        assert said[-1] == protocols.model_refusal(name, described) == (
            f"the {name} measurement takes a finetune model with the {name} head; this "
            "checkpoint is a finetune model with the disease head. It can be measured by disease.")
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
    said.append(protocols.available(model_capabilities(base))[1])
    assert said[-1] == protocols.BASE_MESSAGE
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
    protocol refuses it in words that say so (never as a missing head), it has
    no hops for the start to check and no width, and its build is refused."""
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
    assert hops_problem([1], hgt.encoder) is None
    with pytest.raises(SpecError, match="not yet behind the GraphEncoder contract"):
        hgt.encoder.width  # noqa: B018
    pytest.importorskip("torch_geometric")
    from src.models.bundle.builder import build_bundle

    with pytest.raises(SpecError) as refused:
        build_bundle(hgt, VIEW)
    assert str(refused.value) == encoder_not_built("hgt")


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
    assert protocols.available(two) == (("disease", "patients_like_me"), None)
    assert protocols.model_refusal("gene", two) == (
        "the gene measurement takes a finetune model with the gene head; this checkpoint is a "
        "finetune model with the disease and patients_like_me heads. It can be measured by "
        "disease, patients_like_me.")


def test_an_input_no_evaluator_supplies_is_refused():
    described = model_capabilities(reference_spec("finetune", "disease"))
    needs = {**described, "node_inputs": (*described["node_inputs"], "node_features")}
    said = ("this checkpoint's model needs node_features, which the disease measurement does "
            "not supply.")
    assert protocols.model_refusal("disease", needs) == said
    assert protocols.available(needs) == ((), said)


def test_the_data_kind_is_judged_and_a_generated_split_has_no_counts_to_judge():
    for name in protocols.names():
        assert protocols.inputs_refusal(name, protocols.GENERATED, None) is None
        assert protocols.inputs_refusal(name, protocols.SUPPLIED, GENES) is None
        assert protocols.inputs_refusal(name, "kg_edges", GENES) == (
            f"the {name} measurement does not measure a cohort of kind 'kg_edges'; it measures "
            "a generated split or a supplied cohort version.")
        assert protocols.inputs_refusal(name, None, GENES) == (
            f"the {name} measurement does not measure a cohort of no kind this build reads; it "
            "measures a generated split or a supplied cohort version.")
    # The counts are a registered version's: a gene measurement of one listing none is refused.
    assert protocols.inputs_refusal("gene", protocols.SUPPLIED, None).startswith(
        "this cohort version lists no causal genes")


def test_a_protocol_takes_exactly_the_writable_models_its_words_name():
    """The words a refusal uses for what a protocol takes are the words it
    uses for the writable models it does take, and for no other."""
    for _stage, _task, spec in _writable():
        described = model_capabilities(spec)
        for protocol in protocols.PROTOCOLS.values():
            accepted = protocols.model_refusal(protocol.name, described) is None
            assert accepted == (protocols._takes(protocol) == protocols._described(described))
