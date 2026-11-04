"""The model record in a bundle checkpoint (contract §9.1–§9.3; regressions 6, 7).

The training payload writes the record beside M3a's and marks it with record
version 2; the one reader refuses a checkpoint without a model spec (history,
never run), another version, a record that disagrees with its spec, the wrong
stage or task (checked first), another graph view, weights that do not
match the spec, and a record that names one input two ways (K8-4 review P2-2).
Both writers carry the record, weights-only files included, and everything
loads with `weights_only=True`. The reader is total over what the safe loader
reads (K8-4r2 recheck, P2): a value it cannot convert or walk (an infinite
count, a list holding itself, one nested without end or shared past counting)
is refused naming its field, never raised as another error.

Module: tests/unit/test_model_record.py
"""
from __future__ import annotations

import copy
import dataclasses
import re

import pytest

torch = pytest.importorskip("torch")

from src.models.bundle import build_bundle, finetune_spec_from_parent, reference_spec  # noqa: E402
from src.models.bundle.record import (  # noqa: E402
    MODEL_RECORD_VERSION,
    PLAIN_DEPTH,
    PLAIN_VALUES,
    ModelRecord,
    ModelRecordError,
    load_bundle,
    read_model_record,
)
from src.models.bundle.spec import InputSpec, SpecError, ViewShape  # noqa: E402
from src.training.callbacks import ModelCheckpoint, ModelCheckpointConfig  # noqa: E402
from src.utils.run_environment import assert_primitive  # noqa: E402

VIEW = {"digest": "e" * 64, "num_nodes": 4, "node_types": ["gene", "phenotype"],
        "relations": ["gene|has|phenotype", "phenotype|rev_has|gene"]}


def _base_spec():
    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=4),
        encoder=dataclasses.replace(spec.encoder, layers=1, hidden=2, output=4, heads=2)
    ).validate()


def _record(spec=None, **kw):
    spec = spec or _base_spec()
    bundle = build_bundle(spec, VIEW)
    fields = dict(model_spec=spec, graph_view=bundle.view, operation="new", family_id="run-a",
                  data_roles={"kg": "k" * 64, "split": "s" * 64},
                  graph_exposure={"base": {"message_passing": "train_edges"}},
                  loss_identity={"terms": ["lp"], "lp_loss_weight": 1.0})
    fields.update(kw)
    return bundle, ModelRecord(**fields)


def _payload(bundle, record):
    return {"record_version": MODEL_RECORD_VERSION, "state_dict": bundle.state_dict(),
            **record.to_payload()}


def test_round_trip_and_strict_build():
    bundle, record = _record()
    again, read = load_bundle(_payload(bundle, record), learning_stage="base")
    assert read.family_id == "run-a" and read.task is None
    for (n1, t1), (n2, t2) in zip(again.state_dict().items(), bundle.state_dict().items()):
        assert n1 == n2 and torch.equal(t1, t2)


def test_checkpoint_without_spec_is_history():
    with pytest.raises(ModelRecordError, match="written before the model spec"):
        read_model_record({"record_version": 1, "state_dict": {}})


def test_other_version_and_inconsistent_record_refused():
    bundle, record = _record()
    payload = _payload(bundle, record)
    with pytest.raises(ModelRecordError, match="record version"):
        read_model_record({**payload, "record_version": 1})
    with pytest.raises(ModelRecordError, match="differs from its model spec"):
        read_model_record({**payload, "task": "disease"})
    with pytest.raises(ModelRecordError, match="lacks"):
        read_model_record({k: v for k, v in payload.items() if k != "graph_exposure"})


def test_stage_and_task_checked_first_then_view():
    base = _base_spec()
    spec = finetune_spec_from_parent(base, "disease")
    bundle, record = _record(spec, operation="finetune",
                             parent={"sha256": "p" * 64, "run": "run-a"})
    payload = _payload(bundle, record)
    with pytest.raises(ModelRecordError, match="not a base one"):
        load_bundle(payload, learning_stage="base")
    with pytest.raises(ModelRecordError, match="not 'gene'"):
        load_bundle(payload, task="gene")
    with pytest.raises(ModelRecordError, match="graph view"):
        load_bundle(payload, task="disease", view_digest="f" * 64)
    loaded, read = load_bundle(payload, task="disease", view_digest="e" * 64)
    assert read.parent["run"] == "run-a" and loaded.task_head is not None


def test_weights_not_matching_the_spec_refused():
    bundle, record = _record()
    payload = _payload(bundle, record)
    state = dict(payload["state_dict"])
    state.pop("lp_head.relation_weights")
    with pytest.raises(ModelRecordError, match="do not match"):
        load_bundle({**payload, "state_dict": state})


def test_lineage_rules():
    _, resumed = _record(operation="resume")
    with pytest.raises(ModelRecordError, match="parent"):
        resumed.validate()
    _, task_root = _record(finetune_spec_from_parent(_base_spec(), "gene"))
    with pytest.raises(ModelRecordError, match="new run is a base"):
        task_root.validate()


def test_a_record_part_of_the_wrong_kind_is_refused_whole():
    """Each part is plain data of its kind (a parent names its file's SHA-256
    and its run), or the record is refused with a reason, never half read or
    raised as another error out of a list."""
    spec = finetune_spec_from_parent(_base_spec(), "disease")
    bundle, record = _record(spec, operation="finetune",
                             parent={"sha256": "p" * 64, "run": "run-a"})
    payload = _payload(bundle, record)
    with pytest.raises(ModelRecordError, match="the record's parent is neither a mapping"):
        read_model_record({**payload, "parent": "p" * 64})
    for parent in ({"sha256": "p" * 64}, {"sha256": "p" * 64, "run": 3},
                   {"sha256": None, "run": "run-a"}):
        with pytest.raises(ModelRecordError, match="SHA-256 of its checkpoint file and its run"):
            read_model_record({**payload, "parent": parent})
    for key, value in (("data_roles", None), ("graph_exposure", ["base"]),
                       ("weight_transfer", "all"), ("selection", 3)):
        with pytest.raises(ModelRecordError, match=f"the record's {key} is"):
            read_model_record({**payload, key: value})
    for spec_form in (["not", "a", "mapping"], {"learning_stage": "finetune"}):
        with pytest.raises(ModelRecordError, match="the model spec cannot be read"):
            read_model_record({**payload, "model_spec": spec_form})


def test_a_branch_records_its_changes_and_its_weights():
    """BACKLOG 23: a branch names each changed setting, from the parent's value
    to the new one, beside the exact parent, and how its weights were loaded;
    no other operation records changes."""
    parent = {"sha256": "p" * 64, "run": "run-a"}
    changes = {"learning_rate": {"parent": 1e-4, "new": 1e-3}}
    transfer = {"loaded_from_parent": ["encoder"], "initialised_new": [], "trainable": "all",
                "optimizer": "new"}
    _, branch = _record(operation="branch", parent={**parent, "changes": changes},
                        weight_transfer=transfer)
    branch.validate()
    for broken in ({**parent}, {**parent, "changes": {}},
                   {**parent, "changes": {"learning_rate": 1e-3}}):
        _, record = _record(operation="branch", parent=broken, weight_transfer=transfer)
        with pytest.raises(ModelRecordError, match="parent.changes"):
            record.validate()
    _, untransferred = _record(operation="branch", parent={**parent, "changes": changes})
    with pytest.raises(ModelRecordError, match="how its parent's weights were loaded"):
        untransferred.validate()
    _, resumed = _record(operation="resume", parent={**parent, "changes": changes})
    with pytest.raises(ModelRecordError, match="only a branch records changed settings"):
        resumed.validate()


# --------------------------------------------------------------------------- one input, one claim


def _claims_payload(*, fixture=True, cases=("t" * 64, "v" * 64)):
    """A resumed child's record as `train_model` writes it, naming every input
    it names twice the same way: a gene child trained on a task fixture
    (`fixture`), whose task cases are the fixture's, or a disease child, whose
    task cases are the workspace's samples. The view's rule is stored as the
    view keeps it (`dict(view.rule)`: exclusions sorted), the spec's as the
    operator wrote it."""
    spec = finetune_spec_from_parent(_base_spec(), "gene" if fixture else "disease")
    train, val = cases
    data_roles = {
        "kg": "k" * 64,
        "graph_view": {"digest": VIEW["digest"], "rule": {
            **spec.to_dict()["graph_view"],
            "exclude_relations": sorted(spec.graph_view.exclude_relations)}},
        "split": {"rule": "facts/80-10-10", "seed": 0, "view_digest": VIEW["digest"],
                  "digest": "s" * 64, "edges": {"train": 8, "val": 1, "test": 1}},
        "task_data": {"train": {"samples": train, "binding": {}},
                      "val": {"samples": val, "binding": {}}},
    }
    digests = {"kg": "k" * 64, "node_features": "n" * 64, "edge_indices": "i" * 64,
               "num_nodes": "u" * 64, "split_manifest": "m" * 64, "train_samples": train,
               "val_samples": val, "resume_checkpoint": "p" * 64}
    if fixture:
        data_roles["task_fixture"] = {"record": "f" * 64, "label": "fixture", "rule": "r",
                                      "seed": 0}
        digests.update(train_samples="x" * 64, val_samples="y" * 64, task_fixture="f" * 64,
                       task_train_samples=train, task_val_samples=val)
    _, record = _record(spec, operation="resume", parent={"sha256": "p" * 64, "run": "run-b"},
                        data_roles=data_roles)
    return {"record_version": MODEL_RECORD_VERSION, **record.to_payload(),
            "training_input_digests": digests}


def _at(payload, name):
    value = payload
    for key in name.split("."):
        value = value[key]
    return value


def _with_changed(payload, name):
    """`payload` with only the claim at `name` changed: a digest to another
    digest, a view rule to one that keeps every component."""
    changed = copy.deepcopy(payload)
    *parents, leaf = name.split(".")
    part = _at(changed, ".".join(parents))
    value = part[leaf]
    part[leaf] = ({**value, "largest_component": not value["largest_component"]}
                  if isinstance(value, dict) else "0" * 64)
    return changed


def _named(payload, name):
    value = _at(payload, name)
    return f"{name} {value[:12]}..." if isinstance(value, str) else name


@pytest.mark.parametrize("first, second, fixture", [
    ("data_roles.kg", "training_input_digests.kg", True),
    ("graph_view.digest", "data_roles.graph_view.digest", True),
    ("graph_view.digest", "data_roles.split.view_digest", True),
    ("data_roles.graph_view.rule", "model_spec.graph_view", True),
    ("data_roles.task_data.train.samples", "training_input_digests.task_train_samples", True),
    ("data_roles.task_data.val.samples", "training_input_digests.task_val_samples", True),
    ("data_roles.task_data.train.samples", "training_input_digests.train_samples", False),
    ("data_roles.task_data.val.samples", "training_input_digests.val_samples", False),
    ("data_roles.task_fixture.record", "training_input_digests.task_fixture", True),
    ("parent.sha256", "training_input_digests.resume_checkpoint", True),
])
def test_a_record_naming_one_input_two_ways_is_refused_in_either_direction(
        first, second, fixture):
    """K8-4 review P2-2: serving trusts the training record and the top-level
    view, training trusts `data_roles`, so a record whose two claims about one
    input disagree passed whichever check its entry made. The one reader now
    refuses it, whichever of the two claims was changed, naming both."""
    payload = _claims_payload(fixture=fixture)
    read_model_record(payload)
    for name in (first, second):
        changed = _with_changed(payload, name)
        with pytest.raises(ModelRecordError, match="the record contradicts itself") as refused:
            read_model_record(changed)
        phrase = f"{_named(changed, first)} differs from {_named(changed, second)}"
        assert re.search(r"(?:: |; )" + re.escape(phrase), str(refused.value)), str(refused.value)


def test_a_record_as_the_trainer_writes_it_reads_with_its_exclusions_in_two_orders():
    """The view keeps its exclusions sorted (`build_graph_view`), the spec keeps
    the operator's order, and `train_model` records each from its own source:
    the two copies of one rule are compared in the view's form, so a valid
    model whose exclusions were not written sorted is not refused."""
    payload = _claims_payload()
    payload["model_spec"]["graph_view"]["exclude_relations"] = [
        "gene|causes|disease", "disease|onset|phenotype"]
    payload["data_roles"]["graph_view"]["rule"]["exclude_relations"] = [
        "disease|onset|phenotype", "gene|causes|disease"]
    record = read_model_record(payload)
    assert record.model_spec.graph_view.exclude_relations == (
        "gene|causes|disease", "disease|onset|phenotype")


def test_two_records_on_one_graph_with_different_training_cases_both_read():
    """The rule concerns one file's own claims: two children of one graph
    trained on different cases each name their inputs one way, so both read
    (and keep training, measuring and comparing)."""
    for fixture in (True, False):
        one = read_model_record(_claims_payload(fixture=fixture))
        other = read_model_record(_claims_payload(fixture=fixture, cases=("a" * 64, "b" * 64)))
        assert one.data_roles["kg"] == other.data_roles["kg"]
        assert one.data_roles["task_data"] != other.data_roles["task_data"]


# --------------------------------------------------------------------------- totality


def _holding_itself(kind):
    if kind == "list":
        value = []
        value.append(value)
    else:
        value = {}
        value["self"] = value
    return value


def _nested(depth):
    value = []
    for _ in range(depth):
        value = [value]
    return value


def _shared(width, depth):
    """One list held `width` times at each of `depth` levels: a few objects,
    `width ** depth` values once walked out."""
    value = ["x"]
    for _ in range(depth):
        value = [value] * width
    return value


@pytest.mark.parametrize("field", ["data_roles.graph_view.rule.largest_component",
                                   "data_roles.kg"])
@pytest.mark.parametrize("make, refused", [
    (lambda: _holding_itself("list"), "{field}[0] is a list that contains itself"),
    (lambda: _holding_itself("mapping"), "{field}.self is a dict that contains itself"),
    (lambda: _nested(10_000), "{field}{levels} nests deeper than " + f"{PLAIN_DEPTH} levels"),
    (lambda: _shared(4, 8), "{whole} holds more than " + f"{PLAIN_VALUES} values"),
], ids=["list-holding-itself", "mapping-holding-itself", "nested-10000-deep",
        "one-list-shared-4^8-times"])
def test_a_compared_value_the_walk_cannot_finish_is_refused_naming_where_it_stopped(
        field, make, refused):
    """The reviewer's K8-4r2 recheck, P2 (``cyclic_rule``): the walk that reads a
    compared value as plain data recursed into a list holding itself until
    `RecursionError`, which no entry expects, and the whole model list answered
    HTTP 500; a value nested ten thousand deep did the same, and one list held
    many times over at each level would be walked out for ever. Now each is
    refused as the record's error, naming the field and where the walk stopped:
    on its own path again, `PLAIN_DEPTH` levels down, or past `PLAIN_VALUES`
    values in all (the compared claim named)."""
    payload = _claims_payload()
    *parents, leaf = field.split(".")
    _at(payload, ".".join(parents))[leaf] = make()
    whole = "data_roles.graph_view.rule" if "rule" in field else field
    levels = "[0]" * (PLAIN_DEPTH - field.count(".") + whole.count("."))
    with pytest.raises(ModelRecordError) as caught:
        read_model_record(payload)
    expected = refused.format(field=field, whole=whole, levels=levels)
    assert f"the record's {expected}" in str(caught.value), str(caught.value)


@pytest.mark.parametrize("field, make, refused", [
    ("family_id", lambda old: "\udc80run", "family_id is not text: character 0"),
    ("parent.run", lambda old: "run-\ud800", "parent.run is not text: character 4"),
    ("parent.sha256", lambda old: "\udc80" + old[1:], "parent.sha256 is not text: character 0"),
    ("data_roles.kg", lambda old: "\udc80" + old[1:],
     "data_roles.kg is not text: character 0 is a lone surrogate, so it is not compared"),
    ("training_input_digests.kg", lambda old: old[:-1] + "\ud800",
     "training_input_digests.kg is not text: character 63 is a lone surrogate, so it is not "
     "compared"),
    ("data_roles.graph_view.rule.exclude_relations", lambda old: ["\udc80"],
     "data_roles.graph_view.rule.exclude_relations[0] is not text: character 0"),
    ("data_roles.graph_view.rule", lambda old: {**old, "\udc80": True},
     "data_roles.graph_view.rule has a key that is not text: character 0"),
], ids=["family-id", "parent-run", "parent-sha256", "compared-claim", "other-claim",
        "compared-list", "compared-key"])
def test_a_string_the_record_shows_or_compares_that_is_not_text_is_refused_naming_it(
        field, make, refused):
    """The K8-4r3 verifier, left open 1: the safe loader reads any string a
    pickle holds, a lone surrogate among them, which no answer can encode. The
    reader accepted one in the names a list shows (the family, the parent's
    run and file) and copied a compared claim holding one into its reason, so
    one such file made every model list answer HTTP 500. Now the reader
    refuses each, as it does a value that is not plain data, naming the field
    and never echoing the value, and its reason is text."""
    payload = _claims_payload()
    *parents, leaf = field.split(".")
    part = _at(payload, ".".join(parents)) if parents else payload
    part[leaf] = make(part[leaf])
    with pytest.raises(ModelRecordError) as caught:
        read_model_record(payload)
    assert f"the record's {refused}" in str(caught.value), repr(str(caught.value))
    str(caught.value).encode("utf-8")


def test_a_contradiction_shows_a_claim_that_is_not_printable_escaped():
    """The reader's reason named a contradicting claim by its first twelve
    characters as written, so a claim of control characters (text, but not
    printable) went into every list's answer raw. Now such a claim is shown
    escaped and bounded (`_shown`), and a digest as before."""
    payload = _claims_payload(fixture=False)
    payload["data_roles"]["kg"] = "\x00\n" + "0" * 62
    with pytest.raises(ModelRecordError) as caught:
        read_model_record(payload)
    reason = str(caught.value)
    assert ("data_roles.kg '\\x00\\n0000000000'... differs from training_input_digests.kg "
            f"{'k' * 12}...") in reason, repr(reason)
    assert "\x00" not in reason and "\n" not in reason


@pytest.mark.parametrize("field, refused", [
    ("record_version", "checkpoint record version"),
    ("learning_stage", "the record's learning stage or task differs from its model spec"),
    ("task", "the record's learning stage or task differs from its model spec"),
    ("operation", "unknown operation"),
])
@pytest.mark.parametrize("make", [lambda: _holding_itself("list"), lambda: _nested(10_000)],
                         ids=["list-holding-itself", "nested-10000-deep"])
def test_a_value_judged_against_the_readers_own_is_judged_by_its_kind_without_a_walk(
        field, refused, make):
    """A field the reader holds its own value for (the record version, the
    stage and task its spec names) or a closed list of (the operation) is
    judged directly by that value's kind: a list differs from a number or a
    name without being walked, and the reason shows it however deep it nests.
    Before the fix each raised `RecursionError` from the walk or from the
    reason's own `repr`."""
    payload = _claims_payload()
    payload[field] = make()
    with pytest.raises(ModelRecordError, match=re.escape(refused)):
        read_model_record(payload)


@pytest.mark.parametrize("field, value", [
    ("num_nodes", float("inf")), ("num_nodes", float("-inf")), ("num_nodes", float("nan")),
    ("num_nodes", True), ("num_nodes", 4.0), ("num_nodes", "4"), ("num_nodes", b"4"),
    ("num_nodes", torch.tensor([4])), ("num_nodes", torch.tensor([4, 4])), ("num_nodes", 0),
    ("digest", None), ("digest", 7), ("digest", b"e" * 64), ("digest", ""),
    ("node_types", "gene"), ("node_types", [1]), ("node_types", torch.tensor([1, 2])),
    ("relations", []), ("relations", ["a|b|c", None]), ("relations", _holding_itself("list")),
])
def test_a_graph_view_record_field_is_judged_before_anything_is_made_of_it(field, value):
    """The reviewer's K8-4r2 recheck, P2 (``infinite_node_count``): the view's
    read converted before it checked, so `int(inf)` raised `OverflowError` out
    of the reader, a tensor of two values raised from PyTorch, and a count
    given as text or a boolean, a digest that is not text, or names given as a
    string were cast into a view no writer made. Each field is now judged as
    the kind the view writes, nothing cast, and refused naming it."""
    record = {**VIEW, field: value}
    with pytest.raises(SpecError, match=re.escape(f"graph_view.{field}")):
        ViewShape.from_record(record)
    bundle, model = _record()
    with pytest.raises(ModelRecordError, match=re.escape(f"graph_view.{field}")):
        read_model_record({**_payload(bundle, model), "graph_view": record})


@pytest.mark.parametrize("section, value, refused", [
    ("graph_view", torch.tensor([1, 2]), "model spec section 'graph_view' is Tensor"),
    ("task_head", 1.5, "model spec section 'task_head' is float"),
    ("encoder", {torch.tensor([1, 2]): 1, torch.tensor([3, 4]): 2},
     "model spec section 'encoder' has a key that is not a name (Tensor)"),
    ("encoder.dropout", _nested(10_000),
     "encoder.dropout must be a finite number, not [[[[...]]]]"),
    ("spec_version", _nested(10_000), "model spec version [[[[...]]]] is not 1"),
    ("graph_view.exclude_relations", [_shared(16, 12)],
     "graph_view.exclude_relations[0] must be a string, not [[[[...], [...],"),
])
def test_a_spec_value_the_reader_cannot_ask_or_show_is_refused_naming_its_field(
        section, value, refused):
    """A spec section is read as a mapping of names before anything is asked of
    it: a tensor or a number given for one made `dict()` raise an error naming
    no field, and keys that are tensors made sorting the unknown keys raise
    from PyTorch. A container in a reason is shown a few levels and values
    deep: one nested past the interpreter's depth made `repr` raise
    `RecursionError`, and one list held 16 times at each of 12 levels would be
    written out value by value (16^12, in C, never returning; not run at
    14fe7a8 for that reason)."""
    spec = finetune_spec_from_parent(_base_spec(), "disease")
    bundle, model = _record(spec, operation="finetune", parent={"sha256": "p" * 64, "run": "a"})
    payload = _payload(bundle, model)
    *parents, leaf = section.split(".")
    target = _at(payload["model_spec"], ".".join(parents)) if parents else payload["model_spec"]
    target[leaf] = value
    with pytest.raises(ModelRecordError, match=re.escape(refused)):
        read_model_record(payload)


def _trainer(tmp_path, model):
    from src.training.trainer import Trainer, TrainerConfig

    config = TrainerConfig(device="cpu", scheduler_type="cosine", warmup_steps=0, use_amp=False,
                           num_epochs=2, checkpoint_dir=str(tmp_path / "ckpt"),
                           log_dir=str(tmp_path / "logs"))
    return Trainer(model=model, train_dataloader=[{}] * 2, config=config)


def test_both_writers_carry_the_record(tmp_path):
    bundle, record = _record()
    trainer = _trainer(tmp_path, bundle)
    trainer.model_record = record
    by_trainer, by_callback = tmp_path / "t.pt", tmp_path / "c.pt"
    trainer.save_checkpoint(by_trainer)
    ModelCheckpoint(ModelCheckpointConfig(dirpath=str(tmp_path)))._save_checkpoint(
        trainer, by_callback, epoch=0, logs={"val_pooled_mrr": 0.1})
    for path in (by_trainer, by_callback):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["record_version"] == MODEL_RECORD_VERSION
        assert payload["operation"] == "new" and payload["learning_stage"] == "base"
        load_bundle(payload, learning_stage="base")
        for key, value in payload.items():
            if not key.endswith("state_dict"):
                assert_primitive(value, key)


def test_trainer_without_a_record_keeps_the_m3a_version(tmp_path):
    from src.utils.checkpoint_io import CHECKPOINT_RECORD_VERSION

    trainer = _trainer(tmp_path, torch.nn.Linear(2, 1))
    path = tmp_path / "old.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["record_version"] == CHECKPOINT_RECORD_VERSION
    assert "model_spec" not in payload


def test_record_reads_without_torch():
    import subprocess
    import sys

    code = ("import sys; from src.models.bundle.record import read_model_record; "
            "from src.models.bundle import ModelSpec, ViewShape; "
            "assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
