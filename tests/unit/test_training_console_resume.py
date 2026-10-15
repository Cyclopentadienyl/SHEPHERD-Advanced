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

    update = tc._refresh_checkpoints(str(ws), "hgt")
    assert [value for _, value in update["choices"]] == [str(hgt)]
    assert update["choices"][0][0].startswith("20261001-000000_a_111111 / last.pt")

    update = tc._refresh_checkpoints(str(ws), "gat")  # conv_type change -> relink
    assert [value for _, value in update["choices"]] == [str(gat)]
    assert update["value"] == str(gat)


def test_refresh_does_not_list_the_old_layout(tmp_path):
    ws = tmp_path / "ws"
    (ws / "checkpoints" / "hgt").mkdir(parents=True)
    (ws / "checkpoints" / "hgt" / "last.pt").write_text("x")

    update = tc._refresh_checkpoints(str(ws), "hgt")
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
    from_list, _ = tc._on_check_continue(tc.CONTINUE_FROM_LIST, path, "", "cpu", None)
    typed, _ = tc._on_check_continue(tc.CONTINUE_FROM_PATH, "ignored", path, "cpu", None)

    assert "(run list)" in from_list and "(manual path)" in typed
    for text in (from_list, typed):
        assert "Resume the original training (1/2 done, 1 left)" in text
        assert "| learning_rate |" in text and "start a new run" in text


def test_a_finished_checkpoint_offers_extend_and_says_why_it_cannot_resume(checkpoints):
    path = str(checkpoints["finished"])
    text, extend = tc._on_check_continue(tc.CONTINUE_FROM_PATH, None, path, "cpu", None)
    assert "Cannot continue" in text and "extend_epochs" in text
    assert extend["visible"] is True

    text, _ = tc._on_check_continue(tc.CONTINUE_FROM_PATH, None, path, "cpu", 3)
    assert "Extend training by 3 epochs at a fixed rate" in text


def test_continue_sends_only_operational_fields_and_never_bypasses_the_check(
    checkpoints, monkeypatch
):
    started = []
    monkeypatch.setattr(training_manager, "start_training",
                        lambda config: started.append(config) or {"success": True, "pid": 1,
                                                                  "run_dir": "/r"})
    refused = tc._on_resume(str(checkpoints["ws"]), "", "cpu", tc.CONTINUE_FROM_PATH, None,
                            str(checkpoints["finished"]), None)
    assert started == [], "a refused checkpoint was started"
    assert "Not started" in refused[0] and "Cannot continue" in refused[-1]

    tc._on_resume(str(checkpoints["ws"]), "pilot", "cpu", tc.CONTINUE_FROM_LIST,
                  str(checkpoints["paused"]), "", None)
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
