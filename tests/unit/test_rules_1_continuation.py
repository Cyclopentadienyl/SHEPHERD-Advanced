"""Rules-1 runs continued by their released semantics (research comparison
Stage 9a; owner's O4 and O4a; review F1).

Training rules 2 add one learning setting, `frozen_parts`. A parent is judged
by what this build's one executor can faithfully run, through **one settings
decode** (`src.utils.run_plan.decode_learning_settings`) that the job runner's
preview (`TrainingManager.preview_continuation`) and the training process's
resume, extend and branch all reach. A rules-1 record reads as this build's
settings with `frozen_parts` "none", from what the record itself shows (rules
1 had no freeze; a finetune's weight transfer records "all").

**Case by case, on CPU** (the demo workspace and `trained_children`): a base
and a disease, gene and patients-like-me finetune, each trained by this build,
paused and finished, each beside its rules-1 form (`as_rules_1`, the record
the rules-1 build wrote of the same run, checked field by field against
checkpoints the rules-1 code itself wrote when this stage was built). Each is
resumed, extended and branched from both forms, through `train_model.train`,
and the two children compared: rules 2 and `frozen_parts` "none" recorded on
both, the decode's source on the rules-1 child only (there `frozen_parts` has
one source, the decode, not also "inherited"), and the same optimizer groups,
scheduler state, mixed precision and scaler state, early stopping, validation
(a base's on its split's pairs), progress, settings and recipe record. The
preview says what the process logs, word for word: the same plan sentence for
both forms, and the same decoded setting with its source, which the process
logs in a sentence of its own. Every case is compatible: no rules-1
continuation retires. The parents' files are never rewritten.

**Refused, with the reason, by the preview and the process in the same
sentence:** a rules-1 record that names a freeze, a record of either rules
whose weight transfer contradicts its freeze (one check: "all" exactly when
nothing is frozen), a rules-1 record missing another current setting (nothing
is filled for it), a rules-2 record with an unknown freeze, a base recorded
with a freeze, a frozen run whose transfer records every part trainable, a
resume that states another freeze (a plain continuation never swaps it; a
branch keeps it too), rules
this build does not know, and K4's records (no rules). Mixed precision on,
which the CPU cannot run, is shown at the plan: decided for a rules-1 parent
as for its rules-2 form.

Not claimed: identical weights across an interruption (never promised), or
between the two children (not compared).

Module: tests/unit/test_rules_1_continuation.py
"""
from __future__ import annotations

import copy
import hashlib
import logging
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from src.training import lp_task  # noqa: E402
from tests.fixtures.trained_children import as_rules_1, train_children  # noqa: E402

#: Base validation's task, as built, which `_recording` wraps once per use.
_BASE_LP_TASK = lp_task.BaseLPTask

TASKS = ("disease", "gene", "patients_like_me")
MODELS = ("base",) + TASKS
OPERATIONS = ("resume", "extend", "branch")
#: The decode's source of a rules-1 parent's freeze, as the plan carries it
#: (`decoded`), the process logs it and a continued run records it
#: (`decoded_from_parent`): a base records no weight transfer, a finetune its
#: own.
DECODED = "none, decoded from rules 1: rules 1 had no freeze and trained every part"
DECODED_FINETUNE = DECODED + " (its weight transfer records trainable 'all')"
#: The paused base's own settings beside the demo's: a cosine schedule with
#: warmup and an early-stopping patience, so a resume rebuilds a schedule and
#: restores early stopping's state.
PAUSED_BASE = {"scheduler_type": "cosine", "warmup_steps": 2, "early_stopping_patience": 3}


def _recording(pairs):
    """Base validation's task, recording the digest of each instance's fixed
    validation pairs (`LPPairs.digest`) into `pairs`."""

    class Recording(_BASE_LP_TASK):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            pairs.append(self.val_pairs.digest)

    return Recording


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _train(**fields):
    import train_model

    return train_model.train(train_model.TrainConfig(**fields), set(fields))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Per model, a finished run (one epoch) and a paused one (one of two
    epochs; the base's of three), trained by this build, each with its
    rules-1 form saved beside it in its own checkpoint directory (so the files
    its record names resolve alike), and the base validation pairs' digest of
    every base the fixture trained."""
    from scripts.setup_demo import demo_task_training
    from tests.fixtures.registered_cohort import write_demo_workspace

    pairs = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lp_task, "BaseLPTask", _recording(pairs))
        workspace = write_demo_workspace(tmp_path_factory.mktemp("rules1") / "data")
        trained = train_children(workspace, TASKS)
        demo = demo_task_training()
        sizes = {"gene": {"batch_size": 4, "num_neighbors": [3, 3], "sparse_sample_size": 4},
                 "patients_like_me": {"batch_size": 4, "num_neighbors": [3, 3],
                                      "sparse_sample_size": 4},
                 "disease": demo["disease"]}
        runs_dir = workspace / "runs" / "gatv2"
        paused = {}
        run = runs_dir / "20261005-000000_basep_eeeeee"
        _train(data_dir=str(workspace), run_dir=str(run), learning_stage="base", num_epochs=3,
               pause_after_epoch=1, device="cpu", save_top_k=1, **demo["base"], **PAUSED_BASE)
        paused["base"] = run / "checkpoints" / "last.pt"
        for n, task in enumerate(TASKS):
            run = runs_dir / f"2026100{6 + n}-000000_{task}p_ffffff"
            _train(data_dir=str(workspace), run_dir=str(run), learning_stage="finetune",
                   task=task, finetune_from=str(trained["base"]), num_epochs=2,
                   pause_after_epoch=1, device="cpu", save_top_k=1, **sizes[task])
            paused[task] = run / "checkpoints" / "last.pt"
    files = {}
    for model in MODELS:
        for state, path in (("finished", trained[model]), ("paused", paused[model])):
            rules_1 = path.with_name("rules1.pt")
            torch.save(as_rules_1(_load(path)), rules_1)
            files[(model, state)] = {2: path, 1: rules_1}
    assert len(set(pairs)) == 1 and len(pairs) == 2, "two bases over one split, one pair set"
    return {"workspace": workspace, "files": files, "base_pairs": pairs[0]}


def _fields(runs, model, operation, parent, run_dir):
    """What the CLI states for the case: the stage and task, the parent and,
    for an extension one epoch, for a branch a learning rate twice the
    parent's (on every stage's list)."""
    fields = {"data_dir": str(runs["workspace"]), "run_dir": str(run_dir),
              "learning_stage": "base" if model == "base" else "finetune", "device": "cpu"}
    if model != "base":
        fields["task"] = model
    if operation == "branch":
        rate = _load(parent)["config"]["requested"]["learning_rate"]
        fields.update(branch_from=str(parent), learning_rate=rate * 2)
    else:
        fields["resume_from"] = str(parent)
    if operation == "extend":
        fields["extend_epochs"] = 1
    return fields


def _preview(fields):
    """The job runner's preview of a continuation, as `start_training` asks it
    (`requested`: the request's fields but its parent)."""
    from src.api.services.training_manager import TrainingManager

    return TrainingManager().preview_continuation(
        fields["resume_from"], device=fields["device"], extend_epochs=fields.get("extend_epochs"),
        requested={k: v for k, v in fields.items() if k != "resume_from"})


def _groups(checkpoint):
    return [{**{k: v for k, v in group.items() if k != "params"},
             "params": len(group["params"])}
            for group in checkpoint["optimizer_state_dict"]["param_groups"]]


def _trainer_config(checkpoint):
    """The Trainer's configuration as recorded, without its locations."""
    return {k: v for k, v in checkpoint["config"].items()
            if k not in ("requested", "effective", "checkpoint_dir", "log_dir")}


def _continue(runs, model, operation, rules, tmp_path, monkeypatch, caplog):
    """Continue the case's parent of `rules` through the training process;
    return the child's last checkpoint, the plan sentences the process
    logged, the base validation pairs' digests it drew and the preview."""
    parent = runs["files"][(model, "paused" if operation == "resume" else "finished")][rules]
    fields = _fields(runs, model, operation, parent, tmp_path / f"rules{rules}")
    preview = _preview(fields) if operation != "branch" else None
    pairs = []
    monkeypatch.setattr(lp_task, "BaseLPTask", _recording(pairs))
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _train(**fields)
    said = [record.getMessage() for record in caplog.records]
    return _load(tmp_path / f"rules{rules}" / "checkpoints" / "last.pt"), said, pairs, preview


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("model", MODELS)
def test_a_rules_1_parent_continues_as_its_rules_2_form_does(runs, model, operation, tmp_path,
                                                             monkeypatch, caplog):
    """The acceptance case: the rules-1 parent continues (no retirement) and
    its child equals the rules-2 form's child in everything the one executor
    decides, but the decode's recorded source."""
    from src.utils.checkpoint_io import TRAINING_RULES_VERSION

    state = "paused" if operation == "resume" else "finished"
    files = runs["files"][(model, state)]
    before = {rules: _sha(path) for rules, path in files.items()}
    parent = _load(files[2])
    old_parent = _load(files[1])
    assert old_parent["training_rules"] == 1 and "frozen_parts" not in (
        old_parent["config"]["requested"])

    child_1, said_1, pairs_1, preview_1 = _continue(runs, model, operation, 1, tmp_path,
                                                    monkeypatch, caplog)
    child_2, said_2, pairs_2, preview_2 = _continue(runs, model, operation, 2, tmp_path,
                                                    monkeypatch, caplog)
    decoded = DECODED if model == "base" else DECODED_FINETUNE

    # Continued, under the current rules, with the decoded setting and its
    # source recorded; the parents' files unchanged.
    assert child_1["training_rules"] == child_2["training_rules"] == TRAINING_RULES_VERSION == 2
    assert child_1["operation"] == child_2["operation"] == operation
    requested_1, requested_2 = child_1["config"]["requested"], child_2["config"]["requested"]
    assert requested_1["frozen_parts"] == requested_2["frozen_parts"] == "none"
    assert requested_1 == requested_2
    recipe_1 = dict(child_1["config"]["effective"]["loaders"]["train"]["recipe"])
    recipe_2 = dict(child_2["config"]["effective"]["loaders"]["train"]["recipe"])
    assert recipe_1.pop("decoded_from_parent") == {"frozen_parts": decoded}
    assert recipe_2.pop("decoded_from_parent") == {}
    # One source per setting: the decoded freeze is not also inherited (a
    # base's recipe has no freeze to list; a finetune's lists it as inherited
    # from a rules-2 parent, which records it).
    inherited_2 = recipe_2.pop("inherited_from_parent")
    assert recipe_1.pop("inherited_from_parent") == [n for n in inherited_2
                                                     if n != "frozen_parts"]
    assert ("frozen_parts" in inherited_2) is (model != "base")
    assert recipe_1 == recipe_2
    assert {rules: _sha(path) for rules, path in files.items()} == before

    # The one executor decided alike: optimizer groups, schedule, mixed
    # precision and its scaler, early stopping, validation, progress.
    assert _groups(child_1) == _groups(child_2)
    assert [g["params"] for g in _groups(child_1)] == [g["params"] for g in _groups(parent)]
    assert ("scheduler_state_dict" in child_1) == ("scheduler_state_dict" in child_2)
    assert child_1.get("scheduler_state_dict") == child_2.get("scheduler_state_dict")
    assert "scaler_state_dict" not in child_1 and "scaler_state_dict" not in child_2
    assert "scaler_state_dict" not in parent, "mixed precision is off on the CPU"
    effective_1 = {k: v for k, v in child_1["config"]["effective"].items() if k != "loaders"}
    effective_2 = {k: v for k, v in child_2["config"]["effective"].items() if k != "loaders"}
    assert effective_1 == effective_2
    assert effective_1["amp"] == parent["config"]["effective"]["amp"]
    assert _trainer_config(child_1) == _trainer_config(child_2)
    assert child_1["config"]["early_stopping"] is parent["config"]["early_stopping"] is (
        parent["config"]["requested"]["early_stopping_patience"] is not None)
    assert child_1["callbacks"].get("early_stopping") == child_2["callbacks"].get(
        "early_stopping")
    loaders_1 = child_1["config"]["effective"]["loaders"]
    assert loaders_1["val"] == child_2["config"]["effective"]["loaders"]["val"] == (
        parent["config"]["effective"]["loaders"]["val"])
    assert child_1["progress"] == child_2["progress"]
    if model == "base":
        assert pairs_1 == pairs_2 == [runs["base_pairs"]], "the split's validation pairs"
    if operation == "resume" and model == "base":
        assert child_1["config"]["early_stopping"] is True
        assert "scheduler_state_dict" in child_1, "the cosine schedule rebuilt and restored"

    # The preview says what the process does, in its words: the plan's
    # sentence alike for both forms, the decoded setting in its own.
    from src.utils.run_plan import decoded_sentence

    said_decoded = decoded_sentence({"frozen_parts": decoded})
    assert said_decoded == f"Not in the parent's record: frozen_parts {decoded}."
    assert said_decoded in said_1
    assert not any(s.startswith("Not in the parent's record") for s in said_2)
    if operation == "branch":
        (action,) = [s for s in said_1 if s.startswith("Branch rules1.pt:")]
        (control,) = [s for s in said_2 if s.startswith("Branch last.pt:")]
        assert action == control.replace("last.pt", "rules1.pt")
        return
    assert preview_1["ok"] and preview_2["ok"], (preview_1.get("reason"), preview_2.get("reason"))
    assert f"{preview_1['action']}." in said_1 and f"{preview_2['action']}." in said_2
    assert preview_1["action"] == preview_2["action"]
    assert preview_1["decoded"] == {"frozen_parts": decoded} and preview_2["decoded"] == {}
    assert "frozen_parts" not in preview_1["inherited"]
    assert preview_2["inherited"]["frozen_parts"] == "none"
    assert preview_1["inherited"] == {name: value for name, value in preview_2["inherited"].items()
                                      if name != "frozen_parts"}
    assert ("Inherited from the parent: " + ", ".join(
        f"{name}={value!r}" for name, value in sorted(preview_1["inherited"].items()))) in said_1
    assert (preview_1["mode"], preview_1["start_epoch"], preview_1["end_epoch"]) == (
        preview_2["mode"], preview_2["start_epoch"], preview_2["end_epoch"])


# ---------------------------------------------------------------------------
# Refused with the reason: the preview and the process in one sentence
# ---------------------------------------------------------------------------
def _edited(runs, tmp_path, model, rules, edit, state="paused"):
    """A copy of the case's parent of `rules`, edited, beside its run's files."""
    source = runs["files"][(model, state)][rules]
    payload = copy.deepcopy(_load(source))
    edit(payload)
    target = source.with_name(f"edited-{tmp_path.name}.pt")
    torch.save(payload, target)
    return target


def _set_requested(name, value):
    return lambda payload: payload["config"]["requested"].__setitem__(name, value)


@pytest.mark.parametrize("model, rules, edit, said", [
    ("disease", 1, _set_requested("frozen_parts", "none"),
     "records frozen_parts 'none' under training rules 1, which had no freeze setting, so its "
     "record contradicts its rules."),
    ("gene", 1, lambda p: p["weight_transfer"].__setitem__("trainable", "task_head"),
     "records a weight transfer with trainable 'task_head' under training rules 1, which "
     "trained every part ('all'), so its record contradicts its rules."),
    ("gene", 2, lambda p: p["weight_transfer"].__setitem__("trainable", "task_head"),
     "records a weight transfer with trainable 'task_head' beside frozen_parts 'none', which "
     "trains every part ('all'), so its record contradicts itself."),
    ("patients_like_me", 1, lambda p: p["config"]["requested"].pop("max_grad_norm"),
     "does not record max_grad_norm, so the run cannot inherit them."),
    ("base", 1, lambda p: p["config"]["requested"].pop("warmup_steps"),
     "does not record warmup_steps, so the run cannot inherit them."),
    ("disease", 2, _set_requested("frozen_parts", "partial"),
     "cannot be continued with its recorded freeze: frozen_parts must be one of ['none', "
     "'graph_side'], not 'partial'."),
    ("disease", 2, lambda p: p["config"]["requested"].pop("frozen_parts"),
     "does not record frozen_parts, so the run cannot inherit them."),
    ("base", 2, _set_requested("frozen_parts", "graph_side"),
     "cannot be continued with its recorded freeze: frozen_parts 'graph_side' freezes the input "
     "module, the encoder and the LP head, every part a base run has, so nothing would train."),
    ("gene", 2, _set_requested("frozen_parts", "graph_side"),
     "records a weight transfer with trainable 'all' beside frozen_parts 'graph_side', which "
     "keeps some parts from training, so its record contradicts itself."),
    ("disease", 1, lambda p: p.__setitem__("training_rules", 3),
     "was trained under other training rules (rules 3), which this build does not know (it "
     "continues rules 1, 2), so what its settings mean is not guessed."),
    ("base", 1, lambda p: p.pop("training_rules"),
     "was trained under other training rules (none recorded, K4's): this build turns early "
     "stopping on exactly when a patience is in effect"),
], ids=["r1-names-a-freeze", "r1-transfer-not-all", "r2-none-transfer-not-all",
        "r1-missing-setting", "r1-base-missing",
        "r2-unknown-freeze", "r2-missing-freeze", "r2-base-frozen", "r2-frozen-transfer-all",
        "unknown-rules", "k4-no-rules"])
def test_what_the_decode_cannot_run_is_refused_alike(runs, tmp_path, model, rules, edit, said):
    """Each refusal comes from the one decode, before anything is written, in
    the same sentence from the preview and from the process; the file refused
    is not changed."""
    from src.utils.run_plan import RunPlanRefused

    path = _edited(runs, tmp_path, model, rules, edit)
    digest = _sha(path)
    fields = _fields(runs, model, "resume", path, tmp_path / "run")
    preview = _preview(fields)
    assert preview["ok"] is False and said in preview["reason"], preview["reason"]
    with pytest.raises(RunPlanRefused) as refused:
        _train(**fields)
    assert str(refused.value) == preview["reason"]
    assert str(refused.value).startswith(f"The parent {path.name}")
    assert not (tmp_path / "run").exists() and _sha(path) == digest


def test_a_plain_continuation_never_swaps_the_freeze(runs, tmp_path):
    """A resume and an extension inherit the parent's freeze and refuse
    another stated one, preview and process alike; a branch keeps it too
    (`frozen_parts` is not on a branch's list): another freeze is a new
    finetune from the base."""
    from src.config.recipe import branch_settings
    from src.utils.run_plan import RunPlanRefused

    assert "frozen_parts" not in branch_settings("finetune", "disease")
    hint = ("The freeze (frozen_parts) is kept by every continuation and branch: to train with "
            "another, finetune the base again (finetune_from).")
    for state, operation in (("paused", "resume"), ("finished", "extend")):
        parent = runs["files"][("disease", state)][1]
        fields = {**_fields(runs, "disease", operation, parent, tmp_path / operation),
                  "frozen_parts": "graph_side"}
        preview = _preview(fields)
        assert preview["ok"] is False and preview["reason"].endswith(hint), preview["reason"]
        assert "frozen_parts: the parent's 'none', requested 'graph_side'" in preview["reason"]
        with pytest.raises(RunPlanRefused) as refused:
            _train(**fields)
        assert str(refused.value) == preview["reason"]
    parent = runs["files"][("disease", "finished")][1]
    fields = {**_fields(runs, "disease", "branch", parent, tmp_path / "branch"),
              "frozen_parts": "graph_side"}
    with pytest.raises(RunPlanRefused, match="this request changes others: frozen_parts: the "
                                             r"parent's 'none', requested 'graph_side'\.") as why:
        _train(**fields)
    assert str(why.value).endswith(hint)
    assert not any((tmp_path / name).exists() for name in ("resume", "extend", "branch"))


@pytest.mark.parametrize("model", ["base", "disease"])
def test_mixed_precision_is_decided_for_a_rules_1_parent_as_for_its_rules_2_form(runs, model):
    """AMP takes effect on CUDA only, so the CPU cannot run a parent that
    trained with it: the plan decides for the rules-1 form exactly as for the
    rules-2 form (resumed on CUDA, refused on the CPU with the same reason),
    the decode beside it."""
    from src.models.bundle.record import MODEL_RECORD_VERSION
    from src.utils.run_plan import RunPlanRefused, plan_continuation

    def on_cuda(payload):
        payload = copy.deepcopy(payload)
        payload["config"]["requested"]["use_amp"] = True
        payload["config"]["effective"]["amp"] = {"enabled": True, "dtype": "float16"}
        return payload

    plans, refusals = {}, {}
    for rules, path in runs["files"][(model, "paused")].items():
        record = on_cuda(_load(path))
        request = {**record["config"]["requested"], "frozen_parts": "none", "run_dir": "",
                   "resume_from": "x", "extend_epochs": None, "extend_lr": None,
                   "pause_after_epoch": None}
        plans[rules] = plan_continuation(record, Path("last.pt"), request, set(), device="cuda",
                                         record_version=MODEL_RECORD_VERSION)
        with pytest.raises(RunPlanRefused) as refused:
            plan_continuation(record, Path("last.pt"), request, set(), device="cpu",
                              record_version=MODEL_RECORD_VERSION)
        refusals[rules] = str(refused.value)
    assert refusals[1] == refusals[2] and "mixed precision on" in refusals[1]
    assert plans[1].settings == plans[2].settings and plans[1].settings["use_amp"] is True
    assert plans[1].inherited == {name: value for name, value in plans[2].inherited.items()
                                  if name != "frozen_parts"}
    assert plans[1].decoded == {"frozen_parts": DECODED if model == "base" else DECODED_FINETUNE}
    assert plans[2].decoded == {}


def test_a_new_run_is_checked_for_its_freeze_by_the_process_and_the_job_runner(
        runs, tmp_path, monkeypatch):
    """A new run's freeze, by the one check (`frozen_parts_problem`), first
    in both: a base with a freeze is refused (nothing would train), by the
    training process and the job runner's base start (`start_training`) in
    the same words; nothing is written or started. A base given the freeze's
    "none" is refused as a setting a base does not read, by the process in
    words true of it. (A finetune's freeze trains: Stage 9b,
    tests/unit/test_frozen_backbone_control.py.)"""
    import subprocess

    from src.api.services.training_manager import TrainingManager
    from src.config.recipe import frozen_parts_problem

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("a run was started"))
    with pytest.raises(ValueError) as base_refused:
        _train(data_dir=str(runs["workspace"]), run_dir=str(tmp_path / "base"),
               learning_stage="base", device="cpu", frozen_parts="graph_side",
               candidate_sample_size=3)
    assert str(base_refused.value) == frozen_parts_problem("base", "graph_side")
    assert "every part a base run has, so nothing would train" in str(base_refused.value)
    started = TrainingManager().start_training({
        "data_dir": str(runs["workspace"]), "learning_stage": "base", "device": "cpu",
        "frozen_parts": "graph_side", "candidate_sample_size": 3})
    assert started["success"] is False and started["invalid_settings"] is True
    assert started["error"] == f"{base_refused.value} Nothing was started."
    with pytest.raises(ValueError, match=r"^frozen_parts configure a finetune, which a base run "
                                         r"does not read\. Remove them\.$"):
        _train(data_dir=str(runs["workspace"]), run_dir=str(tmp_path / "base"),
               learning_stage="base", device="cpu", frozen_parts="none")
    assert frozen_parts_problem("finetune", "graph_side") is None
    assert not (tmp_path / "base").exists()


def test_the_continue_drawer_says_the_plan_and_lists_the_decoded_freeze_once(runs, tmp_path):
    """The Continue drawer, from the job runner's real preview of a rules-1
    finetune: its plan line is the same short sentence as for the rules-2
    form (no decode clause, no internal names: WebUI skill §5), and the freeze
    is listed once, among the settings taken from the checkpoint, with the
    decode's own words as its source, never also as an inherited value."""
    import html
    import re

    from src.webui.components import training_console as tc

    previews = {}
    for rules in (1, 2):
        parent = runs["files"][("disease", "finished")][rules]
        previews[rules] = _preview(_fields(runs, "disease", "extend", parent, tmp_path / "run"))
        assert previews[rules]["ok"], previews[rules].get("reason")
    shown = {rules: tc._check_html(preview, tc.CONTINUE_FROM_PATH)
             for rules, preview in previews.items()}
    for rules, text in shown.items():
        (plan,) = re.findall(r"<p>Plan read from the checkpoint: (.*?)\.</p>", text)
        plan = html.unescape(re.sub(r"<[^>]+>", "", plan))
        assert plan == previews[rules]["action"] == previews[2]["action"]
        assert "frozen_parts" not in plan and "rules" not in plan and len(plan.split()) <= 20
        assert text.count("frozen_parts") == 1
        count = len(previews[rules]["inherited"]) + len(previews[rules]["decoded"])
        assert f"({count}; fixed when continuing)" in text
    assert (f"<tr><td>frozen_parts</td><td>{html.escape(DECODED_FINETUNE, quote=True)}</td></tr>"
            in shown[1])
    assert "<tr><td>frozen_parts</td><td><code>&#x27;none&#x27;</code></td></tr>" in shown[2]
