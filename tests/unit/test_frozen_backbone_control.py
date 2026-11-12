"""The frozen-backbone disease control (research comparison Stage 9b; owner's O2
and O4a; the reviewer's method note, example B).

`frozen_parts` "graph_side" keeps a finetune's input module, encoder and LP
head as its base left them: `ModelBundle.freeze` turns their parameters'
gradients off before the Trainer builds its optimizer (so no group holds
them), and keeps them in eval for the whole run, whoever calls `train()`, so
BatchNorm uses and never updates its running statistics and dropout is off.
Only the patient encoder and the task head train. Both arms keep the recipe's
loss coefficients; the control's LP term is recorded as having no trainable
parameter, never dropped.

**On CPU, after real training steps** (the demo workspace, a base and its
children trained by `scripts/train_model.py`'s own `train`):

- every frozen parameter and buffer (BatchNorm's running mean, variance and
  batch count included) is bit-identical to the base's, for a disease, gene
  and patients-like-me child; the downstream parts learned; in the full arm
  the graph side changed;
- a whole-model `train()` never switches a frozen part back;
- the record (`weight_transfer`) says each part trainable or frozen, the
  frozen parts' forward mode and the SHA-256 of the downstream weights as
  training started: equal for two arms of one base and seed, different when
  the downstream initialisation differs, and the digest of what the build
  starts from;
- a resume and an extension keep the freeze with the same optimizer groups and
  record their stage's weight transfer, their parent's unchanged; a branch
  keeps the freeze too, and another freeze stated on a continuation is
  refused, by the job runner's preview and the process in one sentence;
- a loss that reaches no trainable parameter is skipped only in a model with a
  frozen part; with nothing frozen it fails, as before;
- the job runner's Fine-tune start and the CLI take the setting, and the run
  the runner starts trains as previewed;
- both arms are ordinary disease children measured by the one disease
  evaluator, registered, and listed and exported by the results service with
  their freeze among the recorded learning settings, and their weight transfer
  and loss identity as recorded (the downstream start's digest equal for the
  two arms, in Compare and in the export); a child continued from a rules-1
  parent shows the decoded freeze and its source there too.

Not claimed: that the control's numbers are better or worse (no accuracy
condition), or identical weights across an interruption (never promised).

Module: tests/unit/test_frozen_backbone_control.py
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import io
import json
import logging
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")  # the results service's views are the API's models

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from tests.fixtures.registered_cohort import (  # noqa: E402
    LABEL,
    graph_source_cases,
    import_version,
    write_demo_workspace,
    write_source,
)
from tests.fixtures.trained_children import (  # noqa: E402
    BASE_RUN,
    as_rules_1,
    train_children,
)

GRAPH_SIDE = ("input", "encoder", "lp_head")
DOWNSTREAM = ("patient_encoder", "task_head")
#: The gene and patients-like-me children's sizes (`trained_children`'s).
SMALL = {"batch_size": 4, "num_neighbors": [3, 3], "sparse_sample_size": 4}
#: The runs the fixture trains, by name: the two arms of one base and seed,
#: the control with another seed, the control paused after one of two epochs,
#: and a frozen gene and patients-like-me child.
RUNS = {"control": "20261010-000000_control_aaaaaa", "full": "20261011-000000_full_bbbbbb",
        "control_seed34": "20261012-000000_control34_cccccc",
        "paused": "20261013-000000_controlp_dddddd",
        "gene": "20261014-000000_gene_eeeeee", "patients_like_me": "20261015-000000_plm_ffffff"}


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _train(**fields):
    import train_model

    return train_model.train(train_model.TrainConfig(**fields), set(fields))


def _child(workspace, run, task="disease", **settings):
    from scripts.setup_demo import demo_task_training

    sizes = demo_task_training()["disease"] if task == "disease" else SMALL
    base = workspace / "runs" / "gatv2" / BASE_RUN / "checkpoints" / "last.pt"
    fields = {"data_dir": str(workspace), "run_dir": str(workspace / "runs" / "gatv2" / run),
              "learning_stage": "finetune", "task": task, "finetune_from": str(base),
              "device": "cpu", "save_top_k": 1, "num_epochs": 2, **sizes, **settings}
    _train(**fields)
    return workspace / "runs" / "gatv2" / run / "checkpoints" / "last.pt"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A workspaces root holding one demo workspace (``lab``), a registry with
    a supplied cohort version of it, a base, and the runs `RUNS` names."""
    root = tmp_path_factory.mktemp("frozen")
    workspace = write_demo_workspace(root / "roots" / "lab")
    registry = root / "registry"
    registry.mkdir()
    import_version(registry, workspace, write_source(root / "cases.json",
                                                      graph_source_cases(workspace)))
    trained = train_children(workspace, tasks=())
    from derive_task_fixture import derive

    derive(workspace, distractors=4)
    files = {
        "base": trained["base"],
        "control": _child(workspace, RUNS["control"], frozen_parts="graph_side"),
        "full": _child(workspace, RUNS["full"], frozen_parts="none"),
        "control_seed34": _child(workspace, RUNS["control_seed34"], frozen_parts="graph_side",
                                 seed=34, num_epochs=1),
        "paused": _child(workspace, RUNS["paused"], frozen_parts="graph_side",
                         pause_after_epoch=1),
        "gene": _child(workspace, RUNS["gene"], task="gene", frozen_parts="graph_side",
                       num_epochs=1),
        "patients_like_me": _child(workspace, RUNS["patients_like_me"],
                                   task="patients_like_me", frozen_parts="graph_side",
                                   num_epochs=1),
    }
    return {"root": root, "roots": root / "roots", "workspace": workspace,
            "registry": registry, "files": files}


def _state(payload, parts):
    return {k: v for k, v in payload["state_dict"].items() if k.split(".", 1)[0] in parts}


def _initial_bundle(payload):
    """The bundle a finetune starts from, as `train` builds it: the run's seed
    applied, the spec built, its base's graph side loaded."""
    from src.models.bundle.builder import build_bundle
    from src.models.bundle.record import read_model_record
    from src.training.trainer import apply_seed

    record = read_model_record(payload)
    apply_seed(int(payload["config"]["requested"]["seed"]))
    return build_bundle(record.model_spec, record.graph_view)


# ---------------------------------------------------------------------------
# The freeze, after real training steps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("run", ["control", "gene", "patients_like_me"])
def test_every_frozen_parameter_and_buffer_is_the_bases_and_the_downstream_parts_learn(
        world, run):
    """Bit-identical, BatchNorm's running statistics and batch count
    included: in eval for the whole run, they never moved."""
    base = _load(world["files"]["base"])
    child = _load(world["files"][run])
    frozen = _state(child, GRAPH_SIDE)
    assert frozen.keys() == _state(base, GRAPH_SIDE).keys()
    buffers = [k for k in frozen if k.endswith(("running_mean", "running_var",
                                                "num_batches_tracked"))]
    assert len(buffers) == 3, "the demo encoder's one BatchNorm"
    for name, value in frozen.items():
        assert torch.equal(value, base["state_dict"][name]), name
    # The downstream parts learned: their weights are not those training
    # started from, which the record's digest names.
    from src.models.bundle.builder import build_bundle
    from src.models.bundle.record import read_model_record

    record = read_model_record(child)
    trained = build_bundle(record.model_spec, record.graph_view)
    trained.load_state_dict(child["state_dict"])
    assert trained.parts_sha256(DOWNSTREAM) != record.weight_transfer[
        "downstream_initial_sha256"]
    assert child["progress"]["completed_epochs"] >= 1


def test_the_full_arm_trains_its_graph_side_and_the_control_its_downstream_parts_only(world):
    """The full finetune's graph side moved (its weights and its BatchNorm
    statistics); in the control only downstream parameters differ from where
    training started."""
    base = _load(world["files"]["base"])
    full, control = _load(world["files"]["full"]), _load(world["files"]["control"])
    moved = [k for k, v in _state(full, GRAPH_SIDE).items()
             if not torch.equal(v, base["state_dict"][k])]
    assert any(k.endswith("running_mean") for k in moved)
    assert any(k.startswith("encoder.convs") for k in moved)
    assert any(k.startswith("input.") for k in moved) and any(
        k.startswith("lp_head.") for k in moved)
    start = _initial_bundle(control)
    start.load_state_dict({**start.state_dict(), **_state(base, GRAPH_SIDE)})
    initial = start.state_dict()
    changed = {k.split(".", 1)[0] for k, v in control["state_dict"].items()
               if not torch.equal(v, initial[k])}
    assert changed == set(DOWNSTREAM)


def test_the_frozen_parts_are_in_no_optimizer_group(world):
    """The optimizer was built after the freeze: its two groups hold exactly
    the downstream parameters, the full arm's every parameter."""
    from src.models.bundle.record import read_model_record

    control, full = _load(world["files"]["control"]), _load(world["files"]["full"])
    record = read_model_record(control)
    counts = {"all": 0, "downstream": 0}
    from src.models.bundle.builder import build_bundle

    bundle = build_bundle(record.model_spec, record.graph_view)
    for name, _parameter in bundle.named_parameters():
        counts["all"] += 1
        counts["downstream"] += name.split(".", 1)[0] in DOWNSTREAM
    in_groups = {name: sum(len(g["params"]) for g in payload["optimizer_state_dict"][
        "param_groups"]) for name, payload in (("control", control), ("full", full))}
    assert in_groups == {"control": counts["downstream"], "full": counts["all"]}
    assert len(control["optimizer_state_dict"]["param_groups"]) == 2


# ---------------------------------------------------------------------------
# The bundle's freeze, at its root
# ---------------------------------------------------------------------------
#: A view's shape for a bundle built in a test (`build_bundle` takes a record).
VIEW = {"digest": "d" * 64, "num_nodes": 6, "node_types": ["disease", "gene", "phenotype"],
        "relations": ["disease|assoc|phenotype", "phenotype|rev_assoc|disease"]}
#: A two-hop sampled subgraph of `VIEW`'s six nodes, in the sampler's hop
#: order: seeds 0 and 1, their sources 2 and 3, theirs 4 and 5.
SAMPLED = (torch.arange(6), torch.tensor([[2, 3, 4, 5, 4, 5], [0, 1, 2, 3, 3, 2]]),
           [2, 2, 2], [2, 4])


def _finetune_bundle(dropout=0.3):
    """The demo's disease bundle (its base spec's sizes) over `VIEW`, its
    encoder's dropout `dropout`."""
    from src.models.bundle.builder import build_bundle
    from src.models.bundle.spec import finetune_spec_from_parent
    from tests.fixtures.trained_children import small_base_spec

    base = small_base_spec()
    base = dataclasses.replace(base, encoder=dataclasses.replace(base.encoder,
                                                                 dropout=dropout)).validate()
    return build_bundle(finetune_spec_from_parent(base, "disease"), VIEW)


def test_a_whole_model_train_never_switches_a_frozen_part_back():
    """`train()`, `eval()` then `train()` again, as the Trainer calls them
    each epoch: every module of a frozen part stays in eval, every other
    follows the call; a frozen part's parameters have no gradient, and its
    forward in a training step is the evaluation forward (dropout off,
    BatchNorm's running statistics used and not updated)."""
    bundle = _finetune_bundle()
    bundle.freeze(GRAPH_SIDE)
    for _ in range(2):
        bundle.train()
        assert bundle.training is True
        for part in bundle.parts:
            modes = {module.training for module in getattr(bundle, part).modules()}
            assert modes == ({False} if part in GRAPH_SIDE else {True}), part
        bundle.eval()
        assert not any(module.training for module in bundle.modules())
    bundle.train()
    assert all(not p.requires_grad for part in GRAPH_SIDE
               for p in getattr(bundle, part).parameters())
    assert all(p.requires_grad for part in DOWNSTREAM for p in getattr(bundle, part).parameters())
    assert bundle.frozen == GRAPH_SIDE
    # A sampled forward of the frozen graph side, in train(): deterministic
    # (dropout 0.3 is off), its statistics untouched, no autograd graph.
    batch_norm = bundle.encoder.batch_norms[0]
    before = {k: v.clone() for k, v in batch_norm.state_dict().items()}
    first = bundle.node_vectors_sampled(*SAMPLED)
    second = bundle.node_vectors_sampled(*SAMPLED)
    assert torch.equal(first, second) and not first.requires_grad
    assert all(torch.equal(v, before[k]) for k, v in batch_norm.state_dict().items())
    # The same bundle unfrozen: dropout on, statistics updated.
    free = _finetune_bundle()
    free.train()
    stats = free.encoder.batch_norms[0].num_batches_tracked.clone()
    torch.manual_seed(0)
    first = free.node_vectors_sampled(*SAMPLED)
    second = free.node_vectors_sampled(*SAMPLED)
    assert not torch.equal(first, second) and first.requires_grad
    assert free.encoder.batch_norms[0].num_batches_tracked == stats + 2


def test_a_freeze_is_refused_when_it_names_no_part_leaves_nothing_or_comes_twice():
    from src.config.recipe import FROZEN_PARTS
    from src.models.bundle.builder import GRAPH_SIDE_PARTS, PARTS

    assert FROZEN_PARTS["graph_side"] == GRAPH_SIDE_PARTS == GRAPH_SIDE
    assert all(set(parts) <= set(PARTS) for parts in FROZEN_PARTS.values())
    from src.config.training_fields import by_name

    assert set(by_name("frozen_parts").choices) == set(FROZEN_PARTS)
    bundle = _finetune_bundle()
    with pytest.raises(ValueError, match="no part trunk to freeze"):
        bundle.freeze(["trunk"])
    with pytest.raises(ValueError, match="leaves no part of this bundle to train"):
        bundle.freeze(PARTS)
    assert bundle.frozen == () and all(p.requires_grad for p in bundle.parameters())
    bundle.freeze(GRAPH_SIDE)
    with pytest.raises(ValueError, match="frozen already"):
        bundle.freeze(DOWNSTREAM)


def test_the_trainer_builds_its_groups_without_the_frozen_parameters():
    from src.training.callbacks import Callback
    from src.training.trainer import Trainer, TrainerConfig

    class _Quiet(Callback):
        pass

    class _Task:
        selection_metric = "val_pooled_mrr"

    bundle = _finetune_bundle()
    bundle.freeze(GRAPH_SIDE)
    trainer = Trainer(model=bundle, train_dataloader=[], config=TrainerConfig(device="cpu"),
                      task=_Task(), callbacks=[_Quiet()])
    grouped = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert grouped == {id(p) for part in DOWNSTREAM for p in getattr(bundle, part).parameters()}


def test_a_step_whose_loss_reaches_no_trainable_parameter_is_skipped_and_counted():
    """A gene step without a task term, its LP term over a frozen graph side
    alone, has nothing to learn from: no backward (which would fail), no
    update, counted, its terms logged; a step with a trainable term trains."""
    from src.training.callbacks import Callback
    from src.training.lp_task import StepResult
    from src.training.trainer import Trainer, TrainerConfig

    logged = []

    class _Logs(Callback):
        def on_batch_end(self, trainer, batch_idx, logs=None, **kwargs):
            logged.append(dict(logs or {}))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Linear(2, 1)
            self.free = torch.nn.Linear(2, 1)
            self.frozen.requires_grad_(False)

    class _Task:
        selection_metric = "val_pooled_mrr"

        def training_step(self, model, batch, device):
            x = torch.ones(1, 2)
            if batch == "frozen only":
                return StepResult(model.frozen(x).sum(), {"lp": 1.5}, {"task_absent": 1})
            return StepResult(model.free(x).sum() + model.frozen(x).sum(), {"task": 2.0}, {})

    model = _Model()
    before = model.free.weight.detach().clone()
    trainer = Trainer(model=model, train_dataloader=["frozen only", "both"],
                      config=TrainerConfig(device="cpu", max_grad_norm=0.0), task=_Task(),
                      callbacks=[_Logs()])
    metrics = trainer._train_epoch(0)
    assert metrics["train_steps_skipped"] == 1 and trainer.state.optimizer_updates == 1
    assert logged[0] == {"batch_skipped": 1, "loss_lp": 1.5, "count_task_absent": 1}
    assert not torch.equal(model.free.weight, before)


def test_a_loss_without_a_gradient_still_fails_in_a_model_with_nothing_frozen():
    """The skip is the freeze's alone: in a model every parameter of which
    trains (the full arm, any run without a freeze), a loss that reaches no
    trainable parameter is a fault (a forward run without gradients, a term
    detached by mistake), and it fails at its backward as before Stage 9b,
    never finishing a run with no update."""
    from src.training.callbacks import Callback
    from src.training.lp_task import StepResult
    from src.training.trainer import Trainer, TrainerConfig

    class _Quiet(Callback):
        pass

    class _Task:
        selection_metric = "val_pooled_mrr"

        def training_step(self, model, batch, device):
            with torch.no_grad():
                return StepResult(model(torch.ones(1, 2)).sum(), {"task": 2.0}, {})

    trainer = Trainer(model=torch.nn.Linear(2, 1), train_dataloader=["detached"],
                      config=TrainerConfig(device="cpu", max_grad_norm=0.0), task=_Task(),
                      callbacks=[_Quiet()])
    with pytest.raises(RuntimeError, match="does not require grad"):
        trainer._train_epoch(0)
    assert trainer.state.optimizer_updates == 0


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def test_the_weight_transfer_records_each_part_its_forward_and_the_downstream_start(world):
    from src.models.bundle.builder import PARTS_DIGEST_RULE
    from src.models.bundle.record import read_model_record

    control = read_model_record(_load(world["files"]["control"]))
    full = read_model_record(_load(world["files"]["full"]))
    assert control.weight_transfer == {
        "loaded_from_parent": sorted(GRAPH_SIDE), "initialised_new": list(DOWNSTREAM),
        "trainable": list(DOWNSTREAM), "optimizer": "new",
        "parts": {"input": "frozen", "encoder": "frozen", "lp_head": "frozen",
                  "patient_encoder": "trainable", "task_head": "trainable"},
        "frozen_forward": "eval",
        "downstream_initial_sha256": control.weight_transfer["downstream_initial_sha256"],
        "downstream_digest_rule": PARTS_DIGEST_RULE}
    assert full.weight_transfer["trainable"] == "all"
    assert set(full.weight_transfer["parts"].values()) == {"trainable"}
    assert full.weight_transfer["frozen_forward"] is None
    # Both arms keep the recipe's coefficients; the control's LP term trains
    # nothing, said so, not dropped.
    for record in (control, full):
        assert record.loss_identity["terms"] == ["task", "lp"]
    assert ({k: v for k, v in control.loss_identity.items()
             if k != "terms_without_trainable_parameters"}
            == {k: v for k, v in full.loss_identity.items()
                if k != "terms_without_trainable_parameters"})
    assert control.loss_identity["terms_without_trainable_parameters"] == ["lp"]
    assert full.loss_identity["terms_without_trainable_parameters"] == []
    requested = _load(world["files"]["control"])["config"]["requested"]
    assert (requested["lp_loss_weight"], requested["task_loss_weight"]) == (0.9, 0.1)


def test_the_downstream_start_is_checked_by_its_digest_not_inferred_from_a_seed(world):
    """Two arms of one base and seed start from the same downstream weights,
    and their records say so; another seed starts elsewhere, and its record
    says that. The digest is that of what the build starts from."""
    from src.models.bundle.record import read_model_record

    digests = {name: read_model_record(_load(world["files"][name])).weight_transfer[
        "downstream_initial_sha256"] for name in ("control", "full", "control_seed34")}
    assert digests["control"] == digests["full"]
    assert digests["control_seed34"] != digests["control"]
    start = _initial_bundle(_load(world["files"]["control"]))
    assert start.parts_sha256(DOWNSTREAM) == digests["control"]
    # Any changed value, a buffer included, changes a digest.
    bundle = _finetune_bundle()
    digest = bundle.parts_sha256(GRAPH_SIDE)
    bundle.encoder.batch_norms[0].num_batches_tracked += 1
    assert bundle.parts_sha256(GRAPH_SIDE) != digest
    with pytest.raises(ValueError, match="no part trunk"):
        bundle.parts_sha256(["trunk"])


# ---------------------------------------------------------------------------
# Continuing a frozen run, preview and process alike
# ---------------------------------------------------------------------------
def _groups(checkpoint):
    return [{**{k: v for k, v in group.items() if k != "params"}, "params": list(group["params"])}
            for group in checkpoint["optimizer_state_dict"]["param_groups"]]


def _preview(fields):
    from src.api.services.training_manager import TrainingManager

    return TrainingManager().preview_continuation(
        fields["resume_from"], device=fields["device"], extend_epochs=fields.get("extend_epochs"),
        requested={k: v for k, v in fields.items() if k != "resume_from"})


@pytest.mark.parametrize("operation", ["resume", "extend", "branch"])
def test_a_continued_or_branched_frozen_run_keeps_its_freeze(world, tmp_path, operation, caplog):
    """A resume and an extension inherit the freeze, rebuild the same
    optimizer groups and restore the parent's optimizer into them, record
    their stage's weight transfer (their parent's, unchanged), and the
    preview says what the process does; a branch keeps the freeze with a new
    optimizer, its record naming the downstream weights it starts from (its
    parent's). The graph side stays the base's throughout."""
    from src.models.bundle.builder import build_bundle
    from src.models.bundle.record import read_model_record

    parent_path = world["files"]["paused" if operation == "resume" else "control"]
    parent = _load(parent_path)
    fields = {"data_dir": str(world["workspace"]), "run_dir": str(tmp_path / "run"),
              "learning_stage": "finetune", "task": "disease", "device": "cpu"}
    if operation == "branch":
        fields.update(branch_from=str(parent_path),
                      learning_rate=parent["config"]["requested"]["learning_rate"] * 2)
    else:
        fields["resume_from"] = str(parent_path)
    if operation == "extend":
        fields["extend_epochs"] = 1
    preview = _preview(fields) if operation != "branch" else None
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _train(**fields)
    said = [record.getMessage() for record in caplog.records]
    child = _load(tmp_path / "run" / "checkpoints" / "last.pt")
    record = read_model_record(child)

    assert child["config"]["requested"]["frozen_parts"] == "graph_side"
    assert record.loss_identity["terms_without_trainable_parameters"] == ["lp"]
    base = _load(world["files"]["base"])
    for name, value in _state(child, GRAPH_SIDE).items():
        assert torch.equal(value, base["state_dict"][name]), name
    if operation == "branch":
        transfer = record.weight_transfer
        assert transfer["parts"] == read_model_record(parent).weight_transfer["parts"]
        assert transfer["trainable"] == list(DOWNSTREAM) and transfer["initialised_new"] == []
        own = build_bundle(record.model_spec, record.graph_view)
        own.load_state_dict(parent["state_dict"])
        assert transfer["downstream_initial_sha256"] == own.parts_sha256(DOWNSTREAM)
        assert [len(g) for g in _groups(child)] == [len(g) for g in _groups(parent)]
        assert [len(g["params"]) for g in _groups(child)] == [
            len(g["params"]) for g in _groups(parent)]
        return
    # A continuation records its stage's transfer, its parent's unchanged, so
    # the checkpoint a paused arm ends with names the downstream weights its
    # stage started from, as the arm trained without a pause does.
    assert record.weight_transfer == read_model_record(parent).weight_transfer
    assert record.weight_transfer["downstream_initial_sha256"] == read_model_record(
        _load(world["files"]["control"])).weight_transfer["downstream_initial_sha256"]
    groups, parents = _groups(child), _groups(parent)
    assert [g["params"] for g in groups] == [g["params"] for g in parents]
    assert [g["weight_decay"] for g in groups] == [g["weight_decay"] for g in parents]
    assert preview["ok"], preview.get("reason")
    assert preview["inherited"]["frozen_parts"] == "graph_side" and preview["decoded"] == {}
    assert f"{preview['action']}." in said
    assert ("Inherited from the parent: " + ", ".join(
        f"{name}={value!r}" for name, value in sorted(preview["inherited"].items()))) in said


def test_another_freeze_stated_on_a_frozen_runs_continuation_is_refused_alike(world, tmp_path):
    from src.utils.run_plan import RunPlanRefused

    fields = {"data_dir": str(world["workspace"]), "run_dir": str(tmp_path / "run"),
              "learning_stage": "finetune", "task": "disease", "device": "cpu",
              "resume_from": str(world["files"]["paused"]), "frozen_parts": "none"}
    preview = _preview(fields)
    assert preview["ok"] is False
    assert "frozen_parts: the parent's 'graph_side', requested 'none'" in preview["reason"]
    with pytest.raises(RunPlanRefused) as refused:
        _train(**fields)
    assert str(refused.value) == preview["reason"]
    assert not (tmp_path / "run").exists()


def test_a_frozen_record_that_contradicts_its_parts_is_refused(world, tmp_path):
    """The one transfer check, on the parts a frozen record names: a record
    whose parts say another freeze than its setting contradicts itself."""
    payload = copy.deepcopy(_load(world["files"]["control"]))
    payload["weight_transfer"]["parts"]["lp_head"] = "trainable"
    edited = world["files"]["control"].with_name(f"edited-{tmp_path.name}.pt")
    torch.save(payload, edited)
    fields = {"data_dir": str(world["workspace"]), "run_dir": str(tmp_path / "run"),
              "learning_stage": "finetune", "task": "disease", "device": "cpu",
              "resume_from": str(edited), "extend_epochs": 1}
    preview = _preview(fields)
    assert preview["ok"] is False and preview["reason"].endswith(
        "records a weight transfer with frozen parts ['encoder', 'input'] beside frozen_parts "
        "'graph_side', which freezes ['encoder', 'input', 'lp_head'], so its record "
        "contradicts itself."), preview["reason"]


# ---------------------------------------------------------------------------
# A new frozen run: the CLI, the job runner's Fine-tune start, the process
# ---------------------------------------------------------------------------
def test_the_cli_takes_the_freeze_as_the_operators_setting(monkeypatch):
    import train_model

    monkeypatch.setattr(sys, "argv", ["train_model.py", "--stage", "finetune", "--task",
                                      "disease", "--finetune-from", "base.pt",
                                      "--frozen-parts", "graph_side"])
    config, explicit = train_model.load_config_with_explicit(train_model.parse_args())
    assert config.frozen_parts == "graph_side" and "frozen_parts" in explicit
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--frozen-parts", "partial"])
    with pytest.raises(SystemExit):
        train_model.parse_args()


def test_the_job_runners_fine_tune_start_takes_the_freeze_and_its_run_trains_as_started(
        world, monkeypatch, tmp_path):
    """The Fine-tune start checks the freeze by the process's own check and
    passes it to the child as asked; the child's process, given that
    configuration, trains the control."""
    import shutil

    import train_model
    import yaml

    from src.api.services.training_manager import FinetuneInputs, TrainingManager
    from tests.fixtures.job_processes import FakeProcess
    from tests.unit.test_job_runner import Popens

    monkeypatch.delenv("SHEPHERD_DEVICE", raising=False)
    workspace = Path(shutil.copytree(world["workspace"], tmp_path / "ws"))
    base = workspace / "runs" / "gatv2" / BASE_RUN / "checkpoints" / "last.pt"
    import subprocess

    real_popen = subprocess.Popen
    process = FakeProcess(pid=1)
    calls = Popens(process)
    monkeypatch.setattr("subprocess.Popen", calls)
    from scripts.setup_demo import demo_task_training

    manager = TrainingManager()
    # The demo's disease sizes, its two hops among them: the start checks
    # each child's hop list against its base's encoder before any child
    # starts, as the child's process does (decision S4).
    started = manager.start_finetune(FinetuneInputs(
        data_dir=str(workspace), finetune_from=str(base), tasks=("disease",), device="cpu",
        run_name="", settings={"frozen_parts": "graph_side", "num_epochs": 1,
                               **demo_task_training()["disease"]}))
    assert started["success"], started
    (argv,) = calls.calls
    config_path = Path(argv[argv.index("--config") + 1])
    config = yaml.safe_load(config_path.read_text())
    assert config["frozen_parts"] == "graph_side"
    kept = tmp_path / "config.yaml"
    kept.write_text(yaml.safe_dump(config))
    process.finish(0)
    monkeypatch.setattr("subprocess.Popen", real_popen)
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--config", str(kept),
                                      *argv[argv.index("--expect-parent-sha256"):]])
    assert train_model._train_main(train_model.parse_args()) == 0
    child = _load(Path(config["run_dir"]) / "checkpoints" / "last.pt")
    assert child["weight_transfer"]["frozen_forward"] == "eval"
    for name, value in _state(child, GRAPH_SIDE).items():
        assert torch.equal(value, _load(base)["state_dict"][name]), name


# ---------------------------------------------------------------------------
# Both arms: one disease evaluator, registered, listed and exported
# ---------------------------------------------------------------------------
def test_both_arms_are_measured_registered_listed_and_exported_with_their_freeze(
        world, tmp_path, monkeypatch):
    """Each arm is an ordinary disease child: measured by `measure_scorer
    --mode disease` on the registered cohort and recorded, then listed by the
    results service, its freeze among the recorded learning settings, in the
    export and in Compare's rows. A child extended from the full arm's rules-1
    form records the decoded freeze, and its source shows in its effective
    settings (the recipe record), read through the same listing."""
    import measure_scorer

    from src.api.services import results
    from src.evaluation.sidecar import read_ledger
    from src.webui.components.results_segment import compare_fields

    # The deployment's settings, as a measuring shell has them: its
    # workspaces root (the GPU lock's) and its registry.
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(world["roots"]))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(world["registry"]))
    full = world["files"]["full"]
    rules_1 = full.with_name("rules1.pt")
    torch.save(as_rules_1(_load(full)), rules_1)
    extended_run = world["workspace"] / "runs" / "gatv2" / "20261016-000000_fullx_aaaaab"
    _train(data_dir=str(world["workspace"]), run_dir=str(extended_run),
           learning_stage="finetune", task="disease", device="cpu", resume_from=str(rules_1),
           extend_epochs=1)
    checkpoints = {"control": world["files"]["control"], "full": full,
                   "decoded": extended_run / "checkpoints" / "last.pt"}
    for name, path in checkpoints.items():
        assert measure_scorer.main([
            "--mode", "disease", "--checkpoint", str(path), "--data-dir", str(world["workspace"]),
            "--cohort", LABEL, "--registry", str(world["registry"]), "--device", "cpu",
            "--record"]) == 0, name
        ledger, _ = read_ledger(path.parent / "evaluations.json")
        assert [r["scoring"]["mode"] for r in ledger["records"]] == ["disease"]

    found = results.query(results.explicit_sources(
        workspaces_roots=[("a", world["roots"])], registries=[("main", world["registry"])]))
    rows = {}
    for row in found:
        if row["result"] == "unreadable" or row["protocol"] != "disease":
            continue
        for name, path in checkpoints.items():
            if row["model"]["checkpoint_sha256"] == _sha(path):
                rows[name] = row
    assert set(rows) == set(checkpoints)
    assert {name: row["registered"] for name, row in rows.items()} == dict.fromkeys(
        checkpoints, "registered")
    settings = {name: row["model"]["learning_settings"] for name, row in rows.items()}
    assert {name: s["frozen_parts"] for name, s in settings.items()} == {
        "control": "graph_side", "full": "none", "decoded": "none"}
    assert {k: v for k, v in settings["control"].items() if k != "frozen_parts"} == {
        k: v for k, v in settings["full"].items() if k != "frozen_parts"}
    recipe = rows["decoded"]["model"]["effective_settings"]["loaders"]["train"]["recipe"]
    assert recipe["decoded_from_parent"] == {"frozen_parts": (
        "none, decoded from rules 1: rules 1 had no freeze and trained every part (its weight "
        "transfer records trainable 'all')")}
    # Compare shows the freeze as a learning setting of its own row, and the
    # decode's source in the effective settings it was recorded with, as the
    # row of that one entry (a nested setting is one row per entry).
    shown = {name: {(group, label): value for group, label, value, _why in compare_fields(row)}
             for name, row in rows.items()}
    assert shown["control"][("Training settings", "Learning setting · Frozen parts")] == (
        "graph_side")
    assert shown["full"][("Training settings", "Learning setting · Frozen parts")] == "none"
    assert shown["decoded"][(
        "Training settings", "Effective setting · Loaders › Train › Recipe › Decoded from parent "
        "› Frozen parts")] == recipe["decoded_from_parent"]["frozen_parts"]
    # The weight transfer and the loss identity as each record holds them,
    # through the same listing: the two arms start from one downstream digest,
    # checked where they are compared; the control's parts frozen and its LP
    # term without a trainable parameter; the child extended from the rules-1
    # form records its stage's transfer as that form holds it (no parts), and
    # its own loss identity, written by this build.
    transfers = {name: row["model"]["weight_transfer"] for name, row in rows.items()}
    for name, path in checkpoints.items():
        assert transfers[name] == _load(path)["weight_transfer"], name
        assert rows[name]["model"]["loss_identity"] == _load(path)["loss_identity"], name
    digest = ("Training settings", "Weight transfer · Downstream initial SHA-256")
    assert shown["control"][digest] == shown["full"][digest] == (
        transfers["control"]["downstream_initial_sha256"])
    from src.webui.components.results_segment import field_words

    assert {part: shown["control"][("Training settings",
                                    f"Weight transfer · Parts › {field_words(part)}")]
            for part in GRAPH_SIDE + DOWNSTREAM} == {
        **dict.fromkeys(GRAPH_SIDE, "frozen"), **dict.fromkeys(DOWNSTREAM, "trainable")}
    assert shown["control"][("Training settings",
                             "Weight transfer · Frozen parts' forward mode")] == "eval"
    untrained = ("Training settings", "Loss · Terms without a trainable parameter")
    assert (shown["control"][untrained], shown["full"][untrained]) == (["lp"], [])
    assert transfers["decoded"] == {"loaded_from_parent": sorted(GRAPH_SIDE),
                                    "initialised_new": list(DOWNSTREAM), "trainable": "all",
                                    "optimizer": "new"}
    assert shown["decoded"][untrained] == []
    assert not any(label.startswith("Weight transfer · Parts")
                   for _group, label in shown["decoded"])
    # The export: the same rows, the freeze in each one's learning settings.
    table = list(csv.DictReader(io.StringIO(results.to_csv(list(rows.values())))))
    by_model = {}
    for line in table:
        by_model.setdefault(line["checkpoint_sha256"], json.loads(line["learning_settings"]))
    assert {name: by_model[f"sha256:{_sha(path)}"]["frozen_parts"]
            for name, path in checkpoints.items()} == {
        "control": "graph_side", "full": "none", "decoded": "none"}
    assert "decoded from rules 1" in next(
        line["effective_settings"] for line in table
        if line["checkpoint_sha256"] == f"sha256:{_sha(checkpoints['decoded'])}")
    exported = {line["checkpoint_sha256"]: (json.loads(line["weight_transfer"]),
                                            json.loads(line["loss_identity"]))
                for line in table}
    assert {name: exported[f"sha256:{_sha(path)}"] for name, path in checkpoints.items()} == {
        name: (transfers[name], rows[name]["model"]["loss_identity"]) for name in checkpoints}
