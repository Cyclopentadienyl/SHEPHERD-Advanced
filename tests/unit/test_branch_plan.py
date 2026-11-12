"""How a branch is planned from its parent's record (`run_plan.plan_branch`;
contract §9.2, BACKLOG 23), without torch.

Any end state may be branched; each explicit setting that differs from the
parent's and is on the stage's closed list is a change, recorded from the
parent's value to the new one, and one set to the parent's own value is not.
A request that changes none is a same-settings branch (ops-wiring round,
decision B1): every setting the parent's, its changes `{}`. Refused: a
setting off the list (the refusal names the list), a parent without recorded
settings or with weights only, another record version, an extension, and
mixed precision that would take effect otherwise on this device, whether or
not anything changed. A resume that changes a setting points to the branch.

Module: tests/unit/test_branch_plan.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config.model_types import WRITTEN_BEFORE_SPEC
from src.config.recipe import branch_settings
from src.utils.checkpoint_io import TRAINING_RULES_VERSION
from src.utils.run_plan import RunPlanRefused, plan_branch, plan_continuation

RECORDED = {"learning_rate": 1e-4, "seed": 1, "split_seed": 0, "num_epochs": 2,
            "eval_every_n_epochs": 1, "early_stopping_patience": None, "use_amp": False,
            "amp_dtype": "float16", "model_spec": {"kind": "spec"}}


def _parent(end="finished", **changes):
    payload = {"record_version": 2, "training_rules": TRAINING_RULES_VERSION,
               "checkpoint_kind": "training_state", "learning_stage": "base", "task": None,
               "config": {"requested": dict(RECORDED),
                          "effective": {"amp": {"enabled": False, "dtype": "float16"}}},
               "progress": {"end": end, "completed_epochs": 1,
                            "stage": {"kind": "original", "start_epoch": 0, "target_epochs": 2,
                                      "schedule": {"type": "constant"}}}}
    payload.update(changes)
    return payload


def _plan(parent, explicit=(), *, device="cpu", **request):
    full = {**RECORDED, "branch_from": "parent.pt", "device": device, "extend_epochs": None,
            "extend_lr": None, **request}
    return plan_branch(parent, Path("runs/gatv2/base/checkpoints/last.pt"), full,
                       set(explicit) | set(request), device=device, record_version=2)


@pytest.mark.parametrize("end", [None, "paused", "finished", "stopped_early"])
def test_any_end_state_is_branched_and_each_change_recorded(end):
    plan = _plan(_parent(end), ("seed",), learning_rate=1e-3, num_epochs=4, seed=1)
    assert plan.changes == {"learning_rate": {"parent": 1e-4, "new": 1e-3},
                            "num_epochs": {"parent": 2, "new": 4}}
    assert plan.settings["learning_rate"] == 1e-3 and plan.settings["split_seed"] == 0
    assert "seed" not in plan.changes and "seed" not in plan.inherited, "set to its own value"
    assert plan.inherited["model_spec"] == {"kind": "spec"}
    assert plan.action == ("Branch last.pt: learning_rate 0.0001 -> 0.001; num_epochs 2 -> 4")


@pytest.mark.parametrize("patience, parent", [(3, None), (None, 5), (7, 5)])
def test_early_stopping_is_changed_by_a_branch(patience, parent):
    """Decision B3: early stopping's patience is on a branch's list, on or
    off or another patience, recorded as a change; a branch builds a fresh
    Trainer, so nothing of the parent's early stopping carries over."""
    recorded = {**RECORDED, "early_stopping_patience": parent}
    plan = _plan(_parent(config={"requested": recorded,
                                 "effective": {"amp": {"enabled": False, "dtype": "float16"}}}),
                 early_stopping_patience=patience)
    assert plan.changes == {"early_stopping_patience": {"parent": parent, "new": patience}}
    assert plan.settings["early_stopping_patience"] == patience


@pytest.mark.parametrize("request_, fragment", [
    (dict(eval_every_n_epochs=2, learning_rate=1e-3), "but those on its list"),
    (dict(split_seed=1), "but those on its list"),
    (dict(model_spec={"kind": "other"}), "but those on its list"),
    (dict(extend_epochs=2, learning_rate=1e-3), "a branch starts a new run"),
    (dict(extend_epochs=2), "a branch starts a new run"),
    (dict(extend_lr=1e-5), "a branch starts a new run"),
])
def test_a_request_off_the_list_or_extending_is_refused(request_, fragment):
    with pytest.raises(RunPlanRefused, match=fragment) as refused:
        _plan(_parent(), **request_)
    if "list" in fragment:
        assert ", ".join(sorted(branch_settings("base"))) in str(refused.value)


@pytest.mark.parametrize("end", [None, "paused", "finished", "stopped_early"])
@pytest.mark.parametrize("explicit", [(), ("seed", "learning_rate")])
def test_a_branch_that_changes_nothing_is_a_same_settings_branch(end, explicit):
    """Decision B1: nothing stated, or settings stated at the parent's own
    values, is a new run with every setting of the parent's, its changes
    `{}`; never a resume, whatever the parent's end."""
    plan = _plan(_parent(end), explicit)
    assert plan.changes == {}
    assert {name: plan.settings[name] for name in RECORDED} == RECORDED
    assert set(plan.inherited) == set(RECORDED) - set(explicit)
    assert plan.action == ("Branch last.pt with every setting of its parent (no setting "
                           "changed): a new run from its weights, with a new optimizer and "
                           "schedule")


@pytest.mark.parametrize("parent, fragment", [
    (_parent(checkpoint_kind="weights"), "holds weights only"),
    (_parent(record_version=1), "written before the model spec"),
    (_parent(config={}), "does not record the settings"),
])
def test_a_parent_without_its_settings_is_refused(parent, fragment):
    with pytest.raises(RunPlanRefused, match=fragment):
        _plan(parent, learning_rate=1e-3)


@pytest.mark.parametrize("parent", [[], None, "x", 3])
def test_a_read_that_is_not_a_record_is_refused_not_raised_on(parent):
    """A direct caller may hand the planner a read that is no checkpoint record
    at all: it is refused with the reason, for a resume, an extension and a
    branch alike, never an AttributeError."""
    for extend in (None, 1):
        with pytest.raises(RunPlanRefused, match="holds no checkpoint record"):
            plan_continuation(parent, Path("/runs/p/checkpoints/last.pt"),
                              {**RECORDED, "extend_epochs": extend}, set(), device="cpu",
                              record_version=2)
    with pytest.raises(RunPlanRefused, match="holds no checkpoint record"):
        _plan(parent, learning_rate=1e-3)


@pytest.mark.parametrize("rules", [None, TRAINING_RULES_VERSION + 1])
def test_a_parent_trained_under_other_rules_is_neither_continued_nor_branched(rules):
    """The reviewer's K5 review, P2-1: a resume, an extension and a branch keep
    their parent's training rules, so a bundle parent of other rules (none
    recorded: written before K5) is refused by the one check all three use."""
    parent = _parent(end="paused")
    if rules is None:
        parent.pop("training_rules")
    else:
        parent["training_rules"] = rules
    with pytest.raises(RunPlanRefused, match="other training rules"):
        _plan(parent, learning_rate=1e-3)
    for extend in (None, 1):
        with pytest.raises(RunPlanRefused, match="other training rules"):
            plan_continuation(parent, Path("/runs/p/checkpoints/last.pt"),
                              {**RECORDED, "extend_epochs": extend}, set(), device="cpu",
                              record_version=2)
    # Record version 1 (contract M3a's ShepherdGNN checkpoint) retired with its
    # writer (K8-7): the version is checked first, and every caller states the
    # version it continues.
    shepherd = _parent(end="paused", record_version=1)
    shepherd.pop("training_rules")
    with pytest.raises(RunPlanRefused, match=re.escape(
            f"The parent last.pt: {WRITTEN_BEFORE_SPEC}")):
        plan_continuation(shepherd, Path("/runs/p/checkpoints/last.pt"), dict(RECORDED),
                          set(), device="cpu", record_version=2)
    with pytest.raises(TypeError, match="record_version"):
        plan_continuation(shepherd, Path("/runs/p/checkpoints/last.pt"), dict(RECORDED),
                          set(), device="cpu")


def test_mixed_precision_must_take_effect_alike_unless_it_is_changed():
    trained_on_cuda = _parent(config={"requested": {**RECORDED, "use_amp": True},
                                      "effective": {"amp": {"enabled": True,
                                                            "dtype": "float16"}}})
    with pytest.raises(RunPlanRefused, match="mixed precision on"):
        _plan(trained_on_cuda, learning_rate=1e-3, use_amp=True)
    # Nothing changed: it must take effect as it did, as a resume's must.
    with pytest.raises(RunPlanRefused, match="mixed precision on"):
        _plan(trained_on_cuda)
    assert _plan(trained_on_cuda, device="cuda").changes == {}
    plan = _plan(trained_on_cuda, learning_rate=1e-3, use_amp=False)
    assert plan.changes["use_amp"] == {"parent": True, "new": False}
    # Kept on with only its dtype changed: it must still take effect here.
    with pytest.raises(RunPlanRefused, match="mixed precision on"):
        _plan(trained_on_cuda, amp_dtype="bfloat16", use_amp=True)
    assert _plan(trained_on_cuda, amp_dtype="bfloat16", use_amp=True,
                 device="cuda").changes == {"amp_dtype": {"parent": "float16",
                                                          "new": "bfloat16"}}
    # Turned on: recorded only where it takes effect.
    with pytest.raises(RunPlanRefused, match="use_amp turned on would not take effect on cpu"):
        _plan(_parent(), use_amp=True)
    assert _plan(_parent(), use_amp=True, device="cuda").changes == {
        "use_amp": {"parent": False, "new": True}}


def test_the_closed_list_is_the_stages_reference_settings_and_companions():
    base, gene = branch_settings("base"), branch_settings("finetune", "gene")
    assert {"learning_rate", "seed", "num_epochs", "batch_size", "min_lr_ratio", "amp_dtype",
            "gradient_accumulation_steps", "early_stopping_patience"} <= base
    assert not {"split_seed", "model_spec", "eval_every_n_epochs", "save_top_k",
                "task_loss_weight", "frozen_parts"} & base
    assert "frozen_parts" not in branch_settings("finetune", "disease"), "kept (O4a)"
    assert {"task_loss_weight", "lp_loss_weight", "candidate_upsample_size"} <= gene
    assert "candidate_sample_size" not in gene


def test_a_resume_that_changes_a_setting_points_to_the_branch():
    with pytest.raises(RunPlanRefused, match=r"branch the parent \(branch_from\)"):
        plan_continuation(_parent("paused"), Path("last.pt"),
                          {**RECORDED, "learning_rate": 1e-3, "device": "cpu"},
                          {"learning_rate"}, device="cpu", record_version=2)


@pytest.mark.parametrize("request_, settings, fragment", [
    (dict(learning_rate=-1.0), True, "learning_rate must be"),
    (dict(seed="5"), True, "seed must be"),
    (dict(split_seed=1), True, "but those on its list"),
    (dict(extend_epochs=2), False, "a branch starts a new run"),
])
def test_a_refusal_of_the_requests_settings_is_said_as_such(request_, settings, fragment):
    """Decision B2: what a branch changes is judged as a new run's settings
    are, each new value by its field spec (`stated_values_problem`), and a
    refusal of the request's settings themselves is a `SettingsRefused`, the
    runner's invalid settings; a refusal about the parent or the operation is
    not."""
    from src.utils.run_plan import SettingsRefused

    with pytest.raises(RunPlanRefused, match=fragment) as refused:
        _plan(_parent(), **request_)
    assert isinstance(refused.value, SettingsRefused) is settings
