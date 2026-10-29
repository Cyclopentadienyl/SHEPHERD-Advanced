"""How a branch is planned from its parent's record (`run_plan.plan_branch`;
contract §9.2, BACKLOG 23), without torch.

Any end state may be branched; each explicit setting that differs from the
parent's and is on the stage's closed list is a change, recorded from the
parent's value to the new one, and one set to the parent's own value is not.
Refused: a setting off the list (the refusal names the list), no change, a
parent without recorded settings or with weights only, another record
version, an extension, and mixed precision that would take effect otherwise
on this device. A resume that changes a setting points to the branch.

Module: tests/unit/test_branch_plan.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.parametrize("request_, fragment", [
    (dict(eval_every_n_epochs=2, learning_rate=1e-3), "but those on its list"),
    (dict(split_seed=1), "but those on its list"),
    (dict(model_spec={"kind": "other"}), "but those on its list"),
    (dict(early_stopping_patience=3), "but those on its list"),
    (dict(learning_rate=1e-4), "changes at least one setting"),
    (dict(extend_epochs=2, learning_rate=1e-3), "a branch starts a new run"),
])
def test_a_request_off_the_list_or_without_a_change_is_refused(request_, fragment):
    with pytest.raises(RunPlanRefused, match=fragment) as refused:
        _plan(_parent(), **request_)
    if "list" in fragment:
        assert ", ".join(sorted(branch_settings("base"))) in str(refused.value)


@pytest.mark.parametrize("parent, fragment", [
    (_parent(checkpoint_kind="weights"), "holds weights only"),
    (_parent(record_version=1), "written before the model spec"),
    (_parent(config={}), "does not record the settings"),
])
def test_a_parent_without_its_settings_is_refused(parent, fragment):
    with pytest.raises(RunPlanRefused, match=fragment):
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
    # The ShepherdGNN path (record version 1) records no bundle training rules
    # and keeps resuming under M3a's until K7 retires it.
    shepherd = _parent(end="paused", record_version=1)
    shepherd.pop("training_rules")
    plan = plan_continuation(shepherd, Path("/runs/p/checkpoints/last.pt"), dict(RECORDED),
                             set(), device="cpu")
    assert plan.mode == "resume"


def test_mixed_precision_must_take_effect_alike_unless_it_is_changed():
    trained_on_cuda = _parent(config={"requested": {**RECORDED, "use_amp": True},
                                      "effective": {"amp": {"enabled": True,
                                                            "dtype": "float16"}}})
    with pytest.raises(RunPlanRefused, match="mixed precision on"):
        _plan(trained_on_cuda, learning_rate=1e-3, use_amp=True)
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
            "gradient_accumulation_steps"} <= base
    assert not {"early_stopping_patience", "split_seed", "model_spec", "eval_every_n_epochs",
                "save_top_k", "task_loss_weight"} & base
    assert {"task_loss_weight", "lp_loss_weight", "candidate_upsample_size"} <= gene
    assert "candidate_sample_size" not in gene


def test_a_resume_that_changes_a_setting_points_to_the_branch():
    with pytest.raises(RunPlanRefused, match=r"branch the parent \(branch_from\)"):
        plan_continuation(_parent("paused"), Path("last.pt"),
                          {**RECORDED, "learning_rate": 1e-3, "device": "cpu"},
                          {"learning_rate"}, device="cpu", record_version=2)
