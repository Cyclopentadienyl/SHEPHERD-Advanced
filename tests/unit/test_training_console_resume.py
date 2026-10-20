"""
Unit tests for architecture-aware resume in the Training Console.

Covers the resume read path aligning with the run layout (contract M3a-2):
refresh lists the selected conv_type's runs through the shared listing, and the
_on_resume containment guard.
"""
from pathlib import Path

import pytest

# training_console imports gradio (and the api/service stack) at module load.
pytest.importorskip("gradio")

from src.api.services.training_manager import training_manager  # noqa: E402
from src.webui.components import training_console as tc  # noqa: E402
from tests.fixtures.operator_access import gradio_request, operator_mode  # noqa: E402


#: A request D8 allows once the deployment is in mode M1 (`_operator_mode`):
#: the console's controls are operator controls since M5-1.
OP = gradio_request()


@pytest.fixture(autouse=True)
def _operator_mode(monkeypatch):
    operator_mode(monkeypatch)


def _run_file(ws, arch, run, name):
    path = ws / "runs" / arch / run / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


# --------------------------------------------------------------- refresh relink
def test_refresh_lists_the_selected_architectures_runs(tmp_path):
    ws = tmp_path / "ws"
    hgt = _run_file(ws, "hgt", "20261001-000000_a_111111", "last.pt")
    gat = _run_file(ws, "gat", "20261002-000000_b_222222", "model-1-0.9.pt")

    update = tc._refresh_checkpoints(str(ws), "hgt", OP)
    assert [value for _, value in update["choices"]] == [str(hgt)]
    assert update["choices"][0][0].startswith("20261001-000000_a_111111 / last.pt")

    update = tc._refresh_checkpoints(str(ws), "gat", OP)  # conv_type change -> relink
    assert [value for _, value in update["choices"]] == [str(gat)]
    assert update["value"] == str(gat)


def test_refresh_does_not_list_the_old_layout(tmp_path):
    ws = tmp_path / "ws"
    (ws / "checkpoints" / "hgt").mkdir(parents=True)
    (ws / "checkpoints" / "hgt" / "last.pt").write_text("x")

    update = tc._refresh_checkpoints(str(ws), "hgt", OP)
    assert update["choices"] == [] and update["value"] is None


def test_the_manager_lists_every_architecture_with_its_run(tmp_path):
    ws = tmp_path / "ws"
    import os

    newer = _run_file(ws, "hgt", "20261001-000000_a_111111", "last.pt")
    older = _run_file(ws, "gat", "20261002-000000_b_222222", "last.pt")
    os.utime(newer, (2_000, 2_000))
    os.utime(older, (1_000, 1_000))

    listed = training_manager.get_checkpoints(str(ws))
    assert [(c["architecture"], c["run"]) for c in listed] == [
        ("hgt", "20261001-000000_a_111111"), ("gat", "20261002-000000_b_222222")]


# --------------------------------------------------------------- docstring
def test_cli_resume_docstring_names_the_run_layout():
    text = Path("scripts/train_model.py").read_text(encoding="utf-8")
    assert "--resume checkpoints/last.pt" not in text  # stale flat example gone
    assert "/checkpoints/hgt/last.pt" not in text      # the M2 layout is gone too
    assert "runs/hgt/<run>/checkpoints/last.pt" in text


# --------------------------------------------------------------- continue block (M3a)
_TINY = dict(device="cpu", hidden_dim=16, num_layers=1, num_heads=2, num_workers=0,
             use_amp=False, num_epochs=2, batch_size=5, warmup_steps=1, seed=42)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """A paused and a finished checkpoint, trained for real on the CPU."""
    import scripts.train_model as train_model
    from tests.fixtures.trainable_workspace import write_trainable_workspace

    root = tmp_path_factory.mktemp("console")
    ws = write_trainable_workspace(root / "ws")
    train_model.train(train_model.TrainConfig(data_dir=str(ws), run_dir=str(root / "paused"),
                                              pause_after_epoch=1, **_TINY))
    train_model.train(train_model.TrainConfig(data_dir=str(ws), run_dir=str(root / "done"),
                                              **_TINY))
    return {"ws": ws, "paused": root / "paused" / "checkpoints" / "last.pt",
            "finished": root / "done" / "checkpoints" / "last.pt"}


def test_the_listed_and_the_typed_checkpoint_go_through_one_check(checkpoints):
    path = str(checkpoints["paused"])
    from_list, _ = tc._on_check_continue(tc.CONTINUE_FROM_LIST, path, "", "cpu", None, OP)
    typed, _ = tc._on_check_continue(tc.CONTINUE_FROM_PATH, "ignored", path, "cpu", None, OP)

    assert "(run list)" in from_list and "(manual path)" in typed
    for text in (from_list, typed):
        assert "Resume the original training (1/2 done, 1 left)" in text
        assert "| learning_rate |" in text and "start a new run" in text


def test_a_finished_checkpoint_offers_extend_and_says_why_it_cannot_resume(checkpoints):
    path = str(checkpoints["finished"])
    text, extend = tc._on_check_continue(tc.CONTINUE_FROM_PATH, None, path, "cpu", None, OP)
    assert "Cannot continue" in text and "extend_epochs" in text
    assert extend["visible"] is True

    text, _ = tc._on_check_continue(tc.CONTINUE_FROM_PATH, None, path, "cpu", 3, OP)
    assert "Extend training by 3 epochs at a fixed rate" in text


def test_continue_sends_only_operational_fields_and_never_bypasses_the_check(
    checkpoints, monkeypatch
):
    started = []
    monkeypatch.setattr(training_manager, "start_training",
                        lambda config: started.append(config) or {"success": True, "pid": 1,
                                                                  "run_dir": "/r"})
    refused = tc._on_resume(str(checkpoints["ws"]), "", "cpu", tc.CONTINUE_FROM_PATH, None,
                            str(checkpoints["finished"]), None, OP)
    assert started == [], "a refused checkpoint was started"
    assert "Not started" in refused[0] and "Cannot continue" in refused[-2]

    tc._on_resume(str(checkpoints["ws"]), "pilot", "cpu", tc.CONTINUE_FROM_LIST,
                  str(checkpoints["paused"]), "", None, OP)
    assert started and set(started[0]) == {"data_dir", "run_name", "device", "resume_from",
                                           "compile"}
    assert started[0]["resume_from"] == str(checkpoints["paused"])


def test_stop_says_what_it_loses_before_it_happens(checkpoints, monkeypatch):
    monkeypatch.setattr(training_manager, "run_dir", checkpoints["paused"].parent.parent)
    visible, text = tc._on_stop_request()
    assert visible["visible"] is True
    assert "epoch 1" in text and "Progress since then is lost" in text

    monkeypatch.setattr(training_manager, "run_dir", checkpoints["ws"] / "no-run")
    _, text = tc._on_stop_request()
    assert "No checkpoint has been saved yet" in text


def test_a_stopped_run_shows_stopped_and_where_it_can_be_continued(checkpoints):
    snapshot = training_manager.__class__()
    snapshot.run_dir = checkpoints["paused"].parent.parent
    outcome = snapshot.latest_snapshot()
    assert outcome["resumable"] is True and outcome["end"] == "paused"

    text = tc._format_status({"status": "stopped", "outcome": outcome})
    assert text.startswith("**Status**: STOPPED")
    assert "Paused at epoch 1 of 2" in text
    assert "No checkpoint was saved" in tc._format_status({"status": "stopped", "outcome": None})


def test_the_service_refuses_an_explicit_conflict_before_launching(checkpoints, monkeypatch):
    import subprocess as sp
    from unittest.mock import patch

    manager = training_manager.__class__()
    with patch.object(sp, "Popen") as popen:
        result = manager.start_training({"data_dir": str(checkpoints["ws"]),
                                         "resume_from": str(checkpoints["paused"]),
                                         "device": "cpu", "learning_rate": 0.5})
    assert result["success"] is False and "learning_rate" in result["error"]
    popen.assert_not_called()


def test_the_continue_inputs_pass_gradios_own_preprocessing_when_empty():
    """Found by driving the page: an empty, hidden extra-epochs Number is sent as
    0, and a widget minimum of 1 rejected the whole Check event before the
    handler ran. The handler tests call it directly and could not see that."""
    import gradio as gr

    with gr.Blocks() as demo:
        tc.create_training_tab()
    widgets = {getattr(block, "elem_id", None): block for block in demo.blocks.values()}
    for elem_id, empty in (("extend_epochs", 0), ("extend_epochs", None),
                           ("manual_checkpoint", ""), ("checkpoint_dropdown", None)):
        widgets[elem_id].preprocess(empty)


# --------------------------------------------------------------- stale Extra epochs (review of 094b90d, P2-2)
def _apply(state, update):
    """What Gradio does with a returned update: set the fields it names."""
    for key in ("value", "visible"):
        if key in update:
            state[key] = update[key]
    return state


@pytest.mark.parametrize("source", [tc.CONTINUE_FROM_LIST, tc.CONTINUE_FROM_PATH])
def test_extra_epochs_left_from_a_finished_checkpoint_never_block_a_resume(
    checkpoints, monkeypatch, source
):
    """Finished checkpoint, Extra epochs 3; then the unfinished one; Check;
    Continue. The selection change clears the field and the old check, and the
    resume starts without extend_epochs. Both sources, with the component state
    the returned updates leave."""
    started = []
    monkeypatch.setattr(training_manager, "start_training",
                        lambda config: started.append(config) or {"success": True, "pid": 1,
                                                                  "run_dir": "/r"})

    def chosen(path):
        return (path, "") if source == tc.CONTINUE_FROM_LIST else (None, path)

    extra = {"value": None, "visible": False}
    listed, manual = chosen(str(checkpoints["finished"]))
    text, update = tc._on_check_continue(source, listed, manual, "cpu", extra["value"], OP)
    _apply(extra, update)
    assert extra["visible"] is True and "Enter **Extra epochs**" in text
    extra["value"] = 3
    text, update = tc._on_check_continue(source, listed, manual, "cpu", extra["value"], OP)
    _apply(extra, update)
    assert "Extend training by 3 epochs" in text and extra == {"value": 3, "visible": True}

    # The operator picks the unfinished checkpoint: the change event fires.
    listed, manual = chosen(str(checkpoints["paused"]))
    update, preview = tc._on_continue_selection_changed()
    _apply(extra, update)
    assert extra == {"value": None, "visible": False} and preview == tc._CONTINUE_PROMPT

    text, update = tc._on_check_continue(source, listed, manual, "cpu", extra["value"], OP)
    _apply(extra, update)
    assert "Resume the original training" in text and extra["visible"] is False
    tc._on_resume(str(checkpoints["ws"]), "", "cpu", source, listed, manual, extra["value"], OP)
    assert len(started) == 1 and "extend_epochs" not in started[0]
    assert started[0]["resume_from"] == str(checkpoints["paused"])


def test_a_stale_value_that_reaches_the_handlers_is_not_sent_for_an_unfinished_checkpoint(
    checkpoints, monkeypatch
):
    """Without the change event (a value left however it was), Check and
    Continue still treat Extra epochs as the finished checkpoint's field."""
    started = []
    monkeypatch.setattr(training_manager, "start_training",
                        lambda config: started.append(config) or {"success": True, "pid": 1,
                                                                  "run_dir": "/r"})
    paused = str(checkpoints["paused"])
    text, update = tc._on_check_continue(tc.CONTINUE_FROM_PATH, None, paused, "cpu", 3, OP)
    assert "Resume the original training" in text
    assert update["visible"] is False and update["value"] is None
    tc._on_resume(str(checkpoints["ws"]), "", "cpu", tc.CONTINUE_FROM_PATH, None, paused, 3, OP)
    assert len(started) == 1 and "extend_epochs" not in started[0]

    # A finished checkpoint still extends by the epochs entered.
    tc._on_resume(str(checkpoints["ws"]), "", "cpu", tc.CONTINUE_FROM_PATH, None,
                  str(checkpoints["finished"]), 3, OP)
    assert started[1]["extend_epochs"] == 3


def test_the_service_still_refuses_extend_epochs_for_an_unfinished_parent(checkpoints):
    """The console's rule does not reach other callers: an explicit
    extend_epochs on an unfinished parent is refused."""
    preview = training_manager.preview_continuation(str(checkpoints["paused"]), device="cpu",
                                                    extend_epochs=3)
    assert preview["ok"] is False and "applies only to a finished stage" in preview["reason"]


def test_every_selection_change_clears_the_check_and_extra_epochs():
    """Each component that is part of the check is wired to clear it: the
    source, the run list and the typed path (each only when it is the source),
    and the device (the check only)."""
    import gradio as gr

    with gr.Blocks() as demo:
        tc.create_training_tab()
    by_id = {block._id: getattr(block, "elem_id", None) for block in demo.blocks.values()}
    wiring = {}
    for fn in demo.fns.values():
        if fn.fn in (tc._on_continue_selection_changed, tc._on_listed_changed,
                     tc._on_manual_changed, tc._on_continue_device_changed):
            for block, event in fn.targets:
                wiring[(by_id[block], event)] = (
                    fn.fn.__name__, [getattr(o, "elem_id", None) for o in fn.outputs])
    both = ["extend_epochs", "continue_preview"]
    assert wiring == {
        ("continue_source", "change"): ("_on_continue_selection_changed", both),
        ("checkpoint_dropdown", "change"): ("_on_listed_changed", both),
        ("manual_checkpoint", "change"): ("_on_manual_changed", both),
        ("device", "change"): ("_on_continue_device_changed", ["continue_preview"]),
    }


def test_a_refreshed_run_list_does_not_clear_a_typed_paths_check():
    """A Refresh or a conv_type change repopulates the run list, whose change
    event fires whatever the source; it clears only when the run list is the
    source (found by the verification round after the P2-2 fix)."""
    untouched = tc._on_listed_changed(tc.CONTINUE_FROM_PATH)
    assert all(update == gr_update() for update in untouched)
    extend, preview = tc._on_listed_changed(tc.CONTINUE_FROM_LIST)
    assert extend["value"] is None and preview == tc._CONTINUE_PROMPT

    untouched = tc._on_manual_changed(tc.CONTINUE_FROM_LIST)
    assert all(update == gr_update() for update in untouched)
    extend, preview = tc._on_manual_changed(tc.CONTINUE_FROM_PATH)
    assert extend["value"] is None and preview == tc._CONTINUE_PROMPT

    assert tc._on_continue_device_changed() == tc._CONTINUE_PROMPT


def gr_update():
    import gradio as gr

    return gr.update()


def test_continue_without_check_on_a_finished_checkpoint_shows_extra_epochs(
    checkpoints, monkeypatch
):
    started = []
    monkeypatch.setattr(training_manager, "start_training",
                        lambda config: started.append(config) or {"success": True})
    result = tc._on_resume(str(checkpoints["ws"]), "", "cpu", tc.CONTINUE_FROM_PATH, None,
                           str(checkpoints["finished"]), None, OP)
    assert started == [] and "Not started" in result[0]
    assert "Enter **Extra epochs** below" in result[-2]
    assert result[-1]["visible"] is True
