"""
Continuing a run from the Training page (contract M3a; M5-2, UI-C).

Covers the Continue drawer's read path aligning with the run layout (contract
M3a-2) and the server's lists (M5-2): the parent list is the run bar's
workspace's runs only, through the shared listing; a listed parent is resolved
by the server from its list again (never a client path) and its bytes compared
with the digest listed; the listed and the typed checkpoint go through one
check; Continue Training sends only operational fields and the dataset
fingerprint the session showed.
"""
from pathlib import Path

import pytest

# training_console imports gradio (and the api/service stack) at module load.
pytest.importorskip("gradio")

from src.api.services.training_manager import training_manager  # noqa: E402
from src.webui.components import training_console as tc  # noqa: E402
from src.webui.shell import standalone_context  # noqa: E402
from tests.fixtures.operator_access import gradio_request, operator_mode  # noqa: E402


#: A request D8 allows once the deployment is in mode M1 (`_operator_mode`):
#: the page's controls are operator controls since M5-1.
OP = gradio_request()


@pytest.fixture(autouse=True)
def _operator_mode(monkeypatch):
    operator_mode(monkeypatch)


def _run_file(ws, arch, run, name):
    path = ws / "runs" / arch / run / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def _manifest(ws):
    """A workspace the server lists by name (a readable manifest is not needed
    to list its runs)."""
    ws.mkdir(parents=True, exist_ok=True)
    return ws


# --------------------------------------------------------------- the parent list
def test_the_parent_list_is_the_run_bars_workspaces_runs_only(tmp_path, monkeypatch):
    root = tmp_path / "root"
    ws, other = _manifest(root / "ws"), _manifest(root / "other")
    hgt = _run_file(ws, "hgt", "20261001-000000_a_111111", "last.pt")
    gat = _run_file(ws, "gat", "20261002-000000_b_222222", "model-1-0.9.pt")
    _run_file(other, "gat", "20261003-000000_c_333333", "last.pt")
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))

    parents, update, note = tc._refresh_checkpoints(OP, {"name": "ws"}, {}, None)
    assert set(parents) == {"hgt/20261001-000000_a_111111/last.pt",
                            "gat/20261002-000000_b_222222/model-1-0.9.pt"}
    labels = dict((value, label) for label, value in update["choices"])
    assert labels["hgt/20261001-000000_a_111111/last.pt"].startswith("HGT · run 2026-10-01 00:00")
    # A file the read refuses is listed, never offered as loadable.
    assert "metadata not readable" in labels["gat/20261002-000000_b_222222/model-1-0.9.pt"]
    assert "ws’s runs" in note and "· 2" in note
    # No client path anywhere: the values are keys the server resolves again.
    assert all(str(root) not in value for value in labels)
    assert {hgt.name, gat.name} == {key.rsplit("/", 1)[1] for key in parents}


def test_the_parent_list_does_not_list_the_old_layout(tmp_path, monkeypatch):
    root = tmp_path / "root"
    ws = _manifest(root / "ws")
    (ws / "checkpoints" / "hgt").mkdir(parents=True)
    (ws / "checkpoints" / "hgt" / "last.pt").write_text("x")
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))

    parents, update, _note = tc._refresh_checkpoints(OP, {"name": "ws"}, {}, None)
    assert parents == {} and update["choices"] == [] and update["value"] is None


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
PAUSED_RUN = "20261001-000000_paused_aaaaaa"
DONE_RUN = "20261002-000000_done_bbbbbb"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A paused and a finished checkpoint, trained for real on the CPU in the
    run layout of a workspace a workspaces root lists (``data``)."""
    import scripts.train_model as train_model
    from tests.fixtures.trainable_workspace import write_trainable_workspace

    root = tmp_path_factory.mktemp("console")
    ws = write_trainable_workspace(root)
    runs = ws / "runs" / "gat"
    train_model.train(train_model.TrainConfig(data_dir=str(ws), run_dir=str(runs / PAUSED_RUN),
                                              pause_after_epoch=1, **_TINY))
    train_model.train(train_model.TrainConfig(data_dir=str(ws), run_dir=str(runs / DONE_RUN),
                                              **_TINY))
    return {"root": root, "ws": ws,
            "paused": runs / PAUSED_RUN / "checkpoints" / "last.pt",
            "finished": runs / DONE_RUN / "checkpoints" / "last.pt"}


@pytest.fixture
def checkpoints(trained, monkeypatch):
    """The trained workspace listed by the server, with the session's state
    as the page holds it: the run bar's workspace and the parent list."""
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(trained["root"]))
    ws = tc._workspace_state("data")
    parents, _problem = tc._list_parents(ws)
    return {**trained, "state": ws, "parents": parents,
            "paused_key": f"gat/{PAUSED_RUN}/last.pt", "done_key": f"gat/{DONE_RUN}/last.pt"}


def _check(c, source, listed, manual, device="cpu", extend=None, mode=None):
    return tc._on_check_continue(source, listed, manual, device, extend, c["state"],
                                 c["parents"], mode or {}, OP)


def _resume(c, source, listed, manual, extend=None, mode=None, device="cpu"):
    return tc._on_resume(c["state"], device, source, listed, manual, extend, c["parents"],
                         mode or {}, OP)


def _starts(started):
    def start(config, **kwargs):
        started.append({**config, "__kwargs__": kwargs})
        return {"success": True, "pid": 1, "run_dir": "/r"}
    return start


def test_the_listed_and_the_typed_checkpoint_go_through_one_check(checkpoints):
    from_list, _extend, mode, compare = _check(checkpoints, tc.CONTINUE_FROM_LIST,
                                               checkpoints["paused_key"], "")
    typed, *_rest = _check(checkpoints, tc.CONTINUE_FROM_PATH, "ignored",
                           str(checkpoints["paused"]))

    assert "(run list)" in from_list and "(manual path)" in typed
    for text in (from_list, typed):
        assert "Resume the original training (1/2 done, 1 left)" in text
        assert "<td>learning_rate</td>" in text and "start a new run" in text
    # The listed parent was resolved by the server from its list: the path
    # shown is the one inside the listed workspace.
    assert f"data/runs/gat/{PAUSED_RUN}/checkpoints/last.pt" in from_list
    # The parent recorded this workspace's dataset: the same, so Continue
    # Training is on.
    assert mode["parent"]["fingerprint"] == checkpoints["state"]["row"]["dataset_fingerprint"]
    assert "trained on this workspace’s current dataset" in compare
    assert tc.continue_refusal(mode["parent"], checkpoints["state"]) is None


def test_a_listed_parent_whose_bytes_changed_since_it_was_listed_is_refused(checkpoints):
    """The list carries each file's digest; the check reads the file again and
    compares (a `last.pt` rewritten since the listing is not what was shown)."""
    parents = {key: dict(item) for key, item in checkpoints["parents"].items()}
    parents[checkpoints["paused_key"]]["sha256"] = "0" * 64
    text, _e, mode, _c = tc._on_check_continue(
        tc.CONTINUE_FROM_LIST, checkpoints["paused_key"], "", "cpu", None, checkpoints["state"],
        parents, {}, OP)
    assert "changed since it was listed" in text and mode["parent"]["refused"] is True


def test_a_parent_the_list_no_longer_has_is_not_opened(checkpoints, monkeypatch):
    opened = []
    monkeypatch.setattr(training_manager, "preview_continuation",
                        lambda *a, **k: opened.append(a) or {"ok": False})
    text, *_rest = _check(checkpoints, tc.CONTINUE_FROM_LIST, "gat/../../etc/last.pt", "")
    assert "Nothing to check" in text and opened == []


@pytest.mark.parametrize("where", ["copy_outside", "server_file", "other_workspace"])
def test_a_typed_path_only_locates_a_checkpoint_the_server_lists(checkpoints, monkeypatch,
                                                                  tmp_path, where):
    """D8 (2026-10-19): a typed parent "only locates an item the server lists
    and is checked like the rest". A file the run bar's workspace does not
    list, even the same bytes copied elsewhere, is refused unopened, for
    Check checkpoint and Continue Training alike."""
    import shutil

    if where == "copy_outside":
        typed = tmp_path / "elsewhere" / "last.pt"
        typed.parent.mkdir()
        shutil.copy2(checkpoints["paused"], typed)
    elif where == "server_file":
        typed = Path(__file__).resolve()
    else:
        other = checkpoints["root"] / "other" / "runs" / "gat" / PAUSED_RUN / "checkpoints"
        other.mkdir(parents=True)
        typed = other / "last.pt"
        shutil.copy2(checkpoints["paused"], typed)
    opened, started = [], []
    monkeypatch.setattr(training_manager, "preview_continuation",
                        lambda *a, **k: opened.append(a) or {"ok": False})
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    text, *_rest = _check(checkpoints, tc.CONTINUE_FROM_PATH, None, str(typed))
    assert "Nothing to check" in text and "not a checkpoint the server lists" in text
    assert str(typed) not in text
    said, *_rest = _resume(checkpoints, tc.CONTINUE_FROM_PATH, None, str(typed))
    assert "Not started: the typed path is not a checkpoint the server lists" in said
    assert opened == [] and started == []


def test_a_typed_path_of_a_listed_checkpoint_is_found_again_from_the_list(checkpoints,
                                                                          monkeypatch):
    """The typed path is only a locator: what is previewed and resumed is the
    file the server's list resolves for it (its listed digest compared too)."""
    seen = []
    real = training_manager.preview_continuation
    monkeypatch.setattr(training_manager, "preview_continuation",
                        lambda parent, **k: seen.append(parent) or real(parent, **k))
    relative = Path(checkpoints["paused"]).resolve()
    text, _e, mode, _c = _check(checkpoints, tc.CONTINUE_FROM_PATH, None, f"  {relative}  ")
    assert "Resume the original training" in text
    assert seen and Path(seen[0]).resolve() == relative
    parents = {key: dict(item) for key, item in checkpoints["parents"].items()}
    parents[checkpoints["paused_key"]]["sha256"] = "0" * 64
    text, *_rest = tc._on_check_continue(tc.CONTINUE_FROM_PATH, None, str(relative), "cpu",
                                         None, checkpoints["state"], parents, {}, OP)
    assert "changed since it was listed" in text


def test_a_finished_checkpoint_offers_extend_and_says_why_it_cannot_resume(checkpoints):
    path = str(checkpoints["finished"])
    text, extend, _mode, _compare = _check(checkpoints, tc.CONTINUE_FROM_PATH, None, path)
    assert "Cannot continue" in text and "extend_epochs" in text
    assert extend["visible"] is True

    text, *_rest = _check(checkpoints, tc.CONTINUE_FROM_PATH, None, path, extend=3)
    assert "Extend training by 3 epochs at a fixed rate" in text


def test_continue_sends_only_operational_fields_and_never_bypasses_the_check(
    checkpoints, monkeypatch
):
    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    refused = _resume(checkpoints, tc.CONTINUE_FROM_PATH, None, str(checkpoints["finished"]))
    assert started == [], "a refused checkpoint was started"
    assert "Not started" in refused[0] and "Cannot continue" in refused[1]

    _resume(checkpoints, tc.CONTINUE_FROM_LIST, checkpoints["paused_key"], "")
    assert started and set(started[0]) - {"__kwargs__"} == {
        "data_dir", "run_name", "device", "resume_from", "compile"}
    assert started[0]["resume_from"] == str(checkpoints["paused"])
    assert started[0]["data_dir"] == str(checkpoints["ws"].resolve())
    # The start carries the dataset fingerprint the run bar showed.
    assert started[0]["__kwargs__"] == {
        "expected_dataset_fingerprint": checkpoints["state"]["row"]["dataset_fingerprint"]}


def test_continue_training_is_refused_when_the_parents_dataset_is_not_the_workspaces(
    checkpoints, monkeypatch
):
    """The parent recorded another dataset (the workspace was rebuilt since):
    the run bar's Continue Training is disabled with the reason, and a direct
    call starts nothing."""
    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    other = dict(checkpoints["state"])
    other["row"] = {**other["row"], "dataset_fingerprint": "e" * 64}
    out = tc._on_resume(other, "cpu", tc.CONTINUE_FROM_LIST, checkpoints["paused_key"], "", None,
                        checkpoints["parents"], {}, OP)
    assert started == []
    assert out[0] == tc.notice_html(
        "Not started: the parent’s dataset is not this workspace’s current one.")
    assert "cannot continue on the current dataset" in out[1]
    assert tc.continue_refusal(out[3]["parent"], other) is not None
    compare = tc.compare_html(out[3]["parent"], other)
    assert "The parent’s dataset is not this workspace’s current one" in compare


def test_stop_says_what_it_loses_before_it_happens(checkpoints, monkeypatch):
    monkeypatch.setattr(training_manager, "get_status",
                        lambda: {"status": "running", "job_id": "j1"})
    monkeypatch.setattr(training_manager, "run_dir", checkpoints["paused"].parent.parent)
    text, mode = tc._on_stop_request(OP, {})
    assert mode["confirm"] == "j1"                          # asked for the job running now
    assert "epoch 1" in text and "Progress since then is lost" in text
    assert str(checkpoints["root"]) not in text            # the file's name, not its path

    monkeypatch.setattr(training_manager, "run_dir", checkpoints["ws"] / "no-run")
    text, _mode = tc._on_stop_request(OP, {})
    assert "No checkpoint has been saved yet" in text

    # Nothing running: nothing is asked, and no checkpoint is read.
    monkeypatch.setattr(training_manager, "get_status", lambda: {"status": "completed",
                                                                  "job_id": "j1"})
    monkeypatch.setattr(training_manager, "latest_snapshot",
                        lambda *a: pytest.fail("read a checkpoint with nothing to stop"))
    text, mode = tc._on_stop_request(OP, {"confirm": "j0"})
    assert mode["confirm"] is None and "no training job is running" in text


def test_a_stopped_run_shows_stopped_and_where_it_can_be_continued(checkpoints):
    snapshot = training_manager.__class__()
    snapshot.run_dir = checkpoints["paused"].parent.parent
    outcome = snapshot.latest_snapshot()
    assert outcome["resumable"] is True and outcome["end"] == "paused"
    assert outcome["dataset_fingerprint"] == checkpoints["state"]["row"]["dataset_fingerprint"]

    kind, words = tc.pill_state({"status": "stopped", "current_epoch": 1, "total_epochs": 2})
    assert kind == "stopped" and words.startswith("Stopped · epoch 1 of 2")
    assert tc.PILL_KIND[kind] == "grey"
    sentence = tc._snapshot_sentence(outcome)
    assert sentence.startswith("Paused at epoch 1 of 2")
    assert f"data/runs/gat/{PAUSED_RUN}/checkpoints/last.pt" in sentence
    assert "No checkpoint was saved" in tc._snapshot_sentence(None)


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
        tc.create_training_tab(standalone_context(demo))
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
    monkeypatch.setattr(training_manager, "start_training", _starts(started))

    def chosen(which):
        if source == tc.CONTINUE_FROM_LIST:
            return checkpoints[f"{'done' if which == 'finished' else 'paused'}_key"], ""
        return None, str(checkpoints[which])

    extra = {"value": None, "visible": False}
    listed, manual = chosen("finished")
    text, update, _m, _c = _check(checkpoints, source, listed, manual, extend=extra["value"])
    _apply(extra, update)
    assert extra["visible"] is True and "Enter <b>Extra epochs</b> above" in text
    extra["value"] = 3
    text, update, _m, _c = _check(checkpoints, source, listed, manual, extend=extra["value"])
    _apply(extra, update)
    assert "Extend training by 3 epochs" in text and extra == {"value": 3, "visible": True}

    # The operator picks the unfinished checkpoint: the change event fires.
    listed, manual = chosen("paused")
    update, preview = tc._on_continue_selection_changed()
    _apply(extra, update)
    assert extra == {"value": None, "visible": False} and preview == tc._CONTINUE_PROMPT

    text, update, _m, _c = _check(checkpoints, source, listed, manual, extend=extra["value"])
    _apply(extra, update)
    assert "Resume the original training" in text and extra["visible"] is False
    _resume(checkpoints, source, listed, manual, extend=extra["value"])
    assert len(started) == 1 and "extend_epochs" not in started[0]
    assert started[0]["resume_from"] == str(checkpoints["paused"])


def test_a_stale_value_that_reaches_the_handlers_is_not_sent_for_an_unfinished_checkpoint(
    checkpoints, monkeypatch
):
    """Without the change event (a value left however it was), Check and
    Continue still treat Extra epochs as the finished checkpoint's field."""
    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    paused = str(checkpoints["paused"])
    text, update, _m, _c = _check(checkpoints, tc.CONTINUE_FROM_PATH, None, paused, extend=3)
    assert "Resume the original training" in text
    assert update["visible"] is False and update["value"] is None
    _resume(checkpoints, tc.CONTINUE_FROM_PATH, None, paused, extend=3)
    assert len(started) == 1 and "extend_epochs" not in started[0]

    # A finished checkpoint still extends by the epochs entered.
    _resume(checkpoints, tc.CONTINUE_FROM_PATH, None, str(checkpoints["finished"]), extend=3)
    assert started[1]["extend_epochs"] == 3


def test_the_service_still_refuses_extend_epochs_for_an_unfinished_parent(checkpoints):
    """The console's rule does not reach other callers: an explicit
    extend_epochs on an unfinished parent is refused."""
    preview = training_manager.preview_continuation(str(checkpoints["paused"]), device="cpu",
                                                    extend_epochs=3)
    assert preview["ok"] is False and "applies only to a finished stage" in preview["reason"]
    # The preview names the dataset the parent recorded (the drawer compares it).
    assert preview["dataset_fingerprint"] == checkpoints["state"]["row"]["dataset_fingerprint"]


def test_every_selection_change_clears_the_check_and_extra_epochs():
    """Each component that is part of the check is wired to clear it: the
    source, the run list and the typed path (each only when it is the source),
    and the drawer's own device (the check only)."""
    import gradio as gr

    with gr.Blocks() as demo:
        tc.create_training_tab(standalone_context(demo))
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
        ("continue_device", "change"): ("_on_continue_device_changed", ["continue_preview"]),
    }


def test_a_refreshed_run_list_does_not_clear_a_typed_paths_check():
    """A Refresh repopulates the run list, whose change event fires whatever
    the source; it clears only when the run list is the source (found by the
    verification round after the P2-2 fix)."""
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
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    result = _resume(checkpoints, tc.CONTINUE_FROM_PATH, None, str(checkpoints["finished"]))
    assert started == [] and "Not started" in result[0]
    assert "Enter <b>Extra epochs</b> above" in result[1]           # the field is above the check
    assert result[2]["visible"] is True
