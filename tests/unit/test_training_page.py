"""
Research › Prepare › Training (M5-2, UI-C): the run bar, the run view, the drawer.
==================================================================================
`src/webui/components/training_console.py`. What these pin:

- **the page under the contract**: built from the page context alone; its
  timers are the shell's (outside ``gr.Tabs``), read without the queue and
  doing nothing while another page is shown; its style sheet registered and
  never under 14 px; no module-level UI state; the old Data Paths, Epoch
  Window, native plots and the Datasets bridge are gone;
- **the run bar**: the select over the server's list with the shared labels;
  the chip and its Details popover (escaped, no absolute path, "All
  datasets ›"); the status pill (never green for a failed or stopped run);
  one reason line, D8's short words for a read-only request, the GPU job
  holding the slot; a running job's dataset shown and the select disabled;
  Datasets' "Use for training" selects the workspace, a hand choice writes the
  session's workspace;
- **the run view**: which run is shown, its used dataset fingerprint for an
  operator only, the snapshot sentence without a path for a read-only request;
  the chart token (rows counted again only when the log changed) and the
  server functions (a run the server lists, never a path);
- **what a tick sends**: only what changed, everything every few ticks, and
  never the session's selection back over a newer one;
- **the banner**: escaped, red for a failure, amber for PyG, dismissed per
  session; no absolute path for a read-only request.

Module: tests/unit/test_training_page.py
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from src.webui import shell  # noqa: E402
from src.webui.components import training_console as tc  # noqa: E402
from tests.fixtures.operator_access import gradio_request, operator_mode  # noqa: E402

ODD = "<b>odd&name"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    base = tmp_path_factory.mktemp("trainpage")
    return (write_demo_workspace(base / "demo"),
            write_demo_workspace(base / "resampled", sample_seed=7))


def _metrics(n):
    return {"train": [{"epoch": e, "total": 1.0 / (e + 1), "learning_rate": 1e-4 if e % 3 else 0.0}
                      for e in range(n)],
            "validation": [{"epoch": e, "val_loss": 1.2 / (e + 1), "val_mrr": 0.1 * e,
                            "val_hits@1": 0.05, "val_hits@10": 0.2} for e in range(n)]}


@pytest.fixture
def root(built, tmp_path, monkeypatch):
    demo, resampled = built
    root = tmp_path / "root"
    root.mkdir()
    shutil.copytree(demo, root / "demo")
    shutil.copytree(resampled, root / "resampled")
    shutil.copytree(demo, root / ODD)
    (root / "scratch_ws").mkdir()
    run = root / "demo" / "runs" / "gat" / "20261018-142000_run_d41f8b"
    (run / "logs").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    (run / "logs" / "train_20261018_142000.json").write_text(json.dumps(_metrics(6)))
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    return root


@pytest.fixture
def operator(monkeypatch):
    operator_mode(monkeypatch)
    return gradio_request()


READ_ONLY = gradio_request()  # without mode M1, D8 refuses every request


def _page():
    with gr.Blocks() as demo:
        ctx = shell.standalone_context(demo)
        tc.create_training_tab(ctx)
    return demo, ctx, {getattr(b, "elem_id", None): b for b in demo.blocks.values()}


class _Access:
    def __init__(self, allowed, condition="enabled"):
        self.allowed, self.condition = allowed, condition


# --------------------------------------------------------------------------- the page
def test_the_page_is_the_run_bar_the_parameters_the_drawer_and_the_run_view():
    _demo, _ctx, by_id = _page()
    for elem_id in ("shep-train-runbar", "shep-train-dataset", "shep-train-chip",
                    "shep-train-pill", "run_name", "shep-train-reason", "shep-train-start",
                    "shep-train-continue", "shep-train-stop", "shep-train-resume",
                    "shep-train-cancel", "shep-train-stop-confirm", "shep-train-stop-keep",
                    "shep-train-banner", "shep-train-params", "shep-train-drawer",
                    "shep-train-runview", "shep-train-run-head", "shep-train-run-snap",
                    "shep-train-charts", "resource_display"):
        assert elem_id in by_id, elem_id
    # What M5-2 retired: the Data Paths section, the Epoch Window, the native plots.
    for gone in ("data_dir", "epoch_window", "loss_plot", "mrr_plot", "hits_plot", "lr_plot",
                 "export_csv_file", "status_display", "error_banner"):
        assert gone not in by_id, gone
    assert not [b for b in by_id.values() if isinstance(b, gr.LinePlot)]
    for name in ("_from_datasets", "_keep_or_clear_workspace", "_load_training_config",
                 "_reset_training_config", "_export_metrics_csv", "_poll_status"):
        assert not hasattr(tc, name), name


def test_the_parameters_keep_their_groups_defaults_and_order():
    demo, _ctx, by_id = _page()
    accordions = [b for b in demo.blocks.values() if isinstance(b, gr.Accordion)]
    opened = {b.label: b.open for b in accordions}
    assert opened == {"Basic Training Parameters": True, "Advanced Parameters": False,
                      "Expert Parameters": False}
    assert by_id["num_epochs"].value == 100 and by_id["learning_rate"].value == 1e-4
    assert by_id["batch_size"].value == "32" and by_id["seed"].value == 42


def test_the_page_uses_the_shells_timers_without_the_queue_and_none_of_its_own():
    demo, ctx, _by_id = _page()
    assert not [b for b in demo.blocks.values() if isinstance(b, gr.Timer)
                and b not in (ctx.timer_fast, ctx.timer_slow)]
    ticks = {fn.fn.__name__: fn for fn in demo.fns.values()
             if any(t[1] == "tick" for t in fn.targets)}
    assert set(ticks) == {"_poll", "_poll_resources"}
    for fn in ticks.values():
        assert fn.queue is False and fn.show_progress == "hidden"
        assert fn.inputs[0] is ctx.location            # does nothing on another page
    assert ticks["_poll"].targets == [(ctx.timer_fast._id, "tick")]
    assert ticks["_poll_resources"].targets == [(ctx.timer_slow._id, "tick")]


def test_a_tick_while_another_page_is_shown_reads_nothing(monkeypatch):
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: pytest.fail("a tick on another page read the status"))
    monkeypatch.setattr(tc.training_manager, "get_system_resources",
                        lambda: pytest.fail("a tick on another page read the resources"))
    out = tc._poll("clinical", {}, {}, {}, [], gradio_request())
    assert len(out) == len(tc.VIEW_KEYS) + 1 and all(u == gr.update() for u in out)
    assert tc._poll_resources("datasets") == gr.update()


def test_the_page_keeps_no_session_state_in_the_module():
    for name in ("_poll_cache_key", "_dismissed_banner_keys", "_current_banner_key",
                 "_pyg_warning_cache"):
        assert not hasattr(tc, name), name


def test_the_page_id_and_style_sheet_follow_the_contract():
    from src.webui.app import PAGE_CSS, mount_options

    assert tc.TRAINING == shell.TRAINING
    assert tc._PAGE == "#" + shell.page_elem_id(shell.TRAINING)
    assert tc.PAGE_CSS in PAGE_CSS and tc.PAGE_CSS in mount_options()["head"]
    for value in re.findall(r"font-size:\s*([0-9.]+)px", tc.PAGE_CSS):
        assert float(value) >= 14, value
    assert "font-size" not in tc._format_resources({"gpu": {}, "ram": {}})


def test_every_gated_act_checks_d8_and_reads_the_chips_after():
    demo, _ctx, _by_id = _page()
    gated = ("_on_start", "_on_stop", "_on_stop_request", "_on_continue_open",
             "_on_check_continue", "_on_resume", "_refresh_checkpoints")
    import inspect

    for name in gated:
        assert "operator_access(request)" in inspect.getsource(getattr(tc, name)), name
    fns = list(demo.fns.values())
    after = {fn.trigger_after: fn for fn in fns if fn.trigger_after is not None}

    def reads_chips(fn):
        current = fn
        for _ in range(4):
            current = after.get(current._id)
            if current is None:
                return False
            if current.fn.__name__ == "read_chips":
                return True
        return False

    acts = [fn for fn in fns if fn.fn is not None and fn.fn.__name__ in gated]
    assert {fn.fn.__name__ for fn in acts} == set(gated)
    for fn in acts:
        assert reads_chips(fn), fn.fn.__name__


# --------------------------------------------------------------------------- the run bar's workspace
def test_the_select_offers_the_servers_list_with_the_shared_labels(root):
    from src.webui.components import renderers as r
    from src.webui.components.workspaces import read_workspaces

    state = tc._workspace_state("demo")
    assert [tuple(c) for c in state["choices"]] == r.workspace_choices(read_workspaces()["rows"])
    assert state["name"] == "demo" and r.is_digest(state["row"]["dataset_fingerprint"])
    assert state["latest_run"] == "gat/20261018-142000_run_d41f8b"
    assert tc._workspace_state("not-listed")["row"] is None


def test_the_chip_shows_the_dataset_and_its_details_escaped_without_a_path(root):
    state = tc._workspace_state(ODD)
    html = tc.chip_html(state)
    fingerprint = state["row"]["dataset_fingerprint"]
    assert "<b>odd" not in html and "&lt;b&gt;odd&amp;name" in html
    assert fingerprint[:12] in html and 'data-copy="' + fingerprint + '"' in html
    assert "record verified" in html and "synthetic" in html
    assert 'data-act="details"' in html and 'data-act="datasets"' in html
    assert "The manifest binds" in html and "allocation seed" in html
    assert "(in the server’s workspace root)" in html
    assert str(root) not in html
    # Running: the select is the job's, and says so.
    assert "in use by the running job" in tc.chip_html(state, in_use=True)
    assert "No dataset selected" in tc.chip_html({"choices": [["a", "a"]]})


def test_page_open_keeps_datasets_choice_then_the_sessions_then_the_first_trainable(root):
    state, dropdown, chip, chosen, cache, _res = tc._open({}, "resampled")
    assert chosen == "resampled" == state["name"] and dropdown["value"] == "resampled"
    assert cache == {}
    state, _d, _c, chosen, _cache, _r = tc._open({"name": "demo"}, None)
    assert chosen == "demo"
    # Datasets' "Use for training" wins over the session's earlier choice.
    state, _d, _c, chosen, _cache, _r = tc._open({"name": "demo"}, "resampled")
    assert chosen == "resampled" == state["name"]
    state, _d, _c, chosen, _cache, _r = tc._open({}, None)
    trainable = [row for row in __import__("src.webui.components.workspaces", fromlist=["x"])
                 .read_workspaces()["rows"] if row.get("trainable")]
    assert chosen == trainable[0]["name"]


def test_a_hand_choice_writes_the_sessions_workspace_and_datasets_selects_it(root, operator):
    state, chip, written, mode, notice = tc._on_dataset("resampled", {"parent": {"x": 1}},
                                                        operator)
    assert written == "resampled" == state["name"] and mode["parent"] == {} and notice == ""
    ws, dropdown, chip, mode = tc._on_training_workspace("demo", state, {})
    assert ws["name"] == "demo" and dropdown["value"] == "demo"
    # Already the run bar's, or not listed: nothing changes.
    assert tc._on_training_workspace("demo", ws, {}) == (gr.update(),) * 4
    assert tc._on_training_workspace("gone", ws, {}) == (gr.update(),) * 4


def test_datasets_use_for_training_reaches_the_run_bar():
    demo, ctx, by_id = _page()
    listeners = [fn for fn in demo.fns.values()
                 if (ctx.training_workspace._id, "change") in fn.targets]
    assert [fn.fn.__name__ for fn in listeners] == ["_on_training_workspace"]
    assert by_id["shep-train-dataset"] in listeners[0].outputs
    hand = [fn for fn in demo.fns.values()
            if (by_id["shep-train-dataset"]._id, "input") in fn.targets]
    assert [fn.fn.__name__ for fn in hand] == ["_on_dataset"]
    assert ctx.training_workspace in hand[0].outputs


# --------------------------------------------------------------------------- the pill and the reason line
@pytest.mark.parametrize("status, kind, words", [
    ({"status": "idle"}, "idle", "Idle · no training job since the server started"),
    ({"status": "running", "phase": "initializing", "elapsed_seconds": 4}, "accent",
     "Starting · loading data and the model · 00:00:04"),
    ({"status": "running", "current_epoch": 12, "total_epochs": 100, "batch": 340,
      "total_batches": 812, "elapsed_seconds": 872}, "running",
     "RUNNING · epoch 12/100 · batch 340/812 · 00:14:32"),
    ({"status": "stopping", "current_epoch": 3, "total_epochs": 10}, "accent",
     "Stopping · epoch 3 of 10"),
    ({"status": "completed", "current_epoch": 60, "total_epochs": 60,
      "start_time": "2026-10-18T14:20:00", "end_time": "2026-10-18T15:01:07"}, "ok",
     "Completed · epoch 60 of 60 · 00:41:07"),
    ({"status": "stopped", "current_epoch": 7, "total_epochs": 60}, "grey",
     "Stopped · epoch 7 of 60"),
    ({"status": "failed", "current_epoch": 7, "total_epochs": 60,
      "start_time": "2026-10-19T08:15:00", "end_time": "2026-10-19T08:24:12"}, "red",
     "FAILED · epoch 7 of 60 · 00:09:12"),
])
def test_the_status_pill(status, kind, words):
    state, said = tc.pill_state(status)
    assert said == words and tc.PILL_KIND[state] == kind
    assert f"shep-pill-{kind}" in tc.pill_html(status)


def test_a_failed_or_stopped_run_is_never_green():
    for status in ("failed", "stopped"):
        html = tc.pill_html({"status": status})
        assert "shep-pill-ok" not in html


def _view(status=None, ws=None, mode=None, access=None, active=None, gpu=None):
    ws = ws if ws is not None else {"row": {"name": "demo", "trainable": True,
                                            "dataset_fingerprint": "a" * 64}}
    return tc.run_bar_view(status or {"status": "idle"}, ws, mode or {},
                           access=access or _Access(True), active=active, gpu=gpu)


def test_the_run_bars_actions_and_their_one_reason():
    idle = _view()
    assert (idle["start"], idle["cont"], idle["stop"]) == (True, True, False)
    assert "Stop disabled: no training is running." in idle["reason"]

    running = _view({"status": "running", "current_epoch": 3, "total_epochs": 12})
    assert (running["start"], running["cont"], running["stop"]) == (False, False, True)
    assert "a training job is running" in running["reason"] and running["progress"] == 0.25
    confirming = _view({"status": "running"}, mode={"confirm": True})
    assert confirming["stop"] is False

    read_only = _view(access=_Access(False, "not_loopback"))
    assert not any(read_only[k] for k in ("start", "cont", "stop", "resume"))
    assert "Operator controls only: request not local" in read_only["reason"]

    test_run = _view(gpu={"text": "GPU job: test run (mygene2-2026-09 · 00:01:42)", "test": True})
    assert test_run["start"] is False and test_run["cont"] is False
    assert "GPU job: test run (mygene2-2026-09 · 00:01:42)" in test_run["reason"]
    assert 'data-act="open-test"' in test_run["reason"]
    cli = _view(gpu={"text": "GPU job: train_model.py (CLI, pid 41822, since 09:58 UTC)",
                     "test": False})
    assert "train_model.py (CLI, pid 41822" in cli["reason"] and "open-test" not in cli["reason"]

    none = _view(ws={"row": None})
    assert none["start"] is False and "select a dataset" in none["reason"]
    untrainable = _view(ws={"row": {"name": "scratch_ws", "trainable": False,
                                    "not_trainable": "x", "manifest": "absent"}})
    assert "scratch_ws is not trainable: no split manifest" in untrainable["reason"]


def test_with_the_drawer_open_continue_training_follows_the_chosen_parent():
    ws = {"row": {"name": "demo", "trainable": True, "dataset_fingerprint": "a" * 64}}
    nothing = _view(ws=ws, mode={"drawer": True, "parent": {}})
    assert nothing["start"] is False and nothing["resume"] is False
    assert "close Continue… to start a new run" in nothing["reason"]
    same = _view(ws=ws, mode={"drawer": True, "parent": {"chosen": True, "fingerprint": "a" * 64}})
    assert same["resume"] is True
    differs = _view(ws=ws, mode={"drawer": True,
                                 "parent": {"chosen": True, "fingerprint": "e" * 64}})
    assert differs["resume"] is False
    assert "the parent’s dataset is not this workspace’s current one" in differs["reason"]
    assert "Continue Training, in the run bar above, is disabled" in differs["drawer_reason"]
    refused = _view(ws=ws, mode={"drawer": True, "parent": {"chosen": True, "refused": True}})
    assert refused["resume"] is False


def test_the_gpu_line_comes_from_the_runner_and_the_lock_holder():
    test = tc._gpu_line({"job_type": "test", "start_time": "2026-10-20T10:00:00",
                         "status": "running"}, None,
                        {"inputs": {"version_label": "mygene2-2026-09"}}, operator=True)
    assert test["test"] is True and test["text"].startswith("GPU job: test run (mygene2-2026-09 · ")
    held = {"configured": True, "problem": None, "held": True, "holder_alive": True,
            "holder": {"command": "train_model.py", "pid": 41822,
                       "started_at": "2026-10-20T09:58:00+00:00"},
            "description": "train_model.py (pid 41822)"}
    cli = tc._gpu_line(None, held, None, operator=False)
    assert cli == {"text": "GPU job: train_model.py (CLI, pid 41822, since 09:58 UTC)",
                   "test": False}
    assert tc._gpu_line(None, {"configured": True, "held": False}, None, operator=True) is None
    problem = tc._gpu_line(None, {"configured": True, "problem": "/srv/lock is a directory"},
                           None, operator=False)
    assert problem["text"] == "the GPU lock cannot be used"            # no path to read-only


# --------------------------------------------------------------------------- the run view
def test_the_run_shown_is_the_runners_else_the_workspaces_latest(root, monkeypatch):
    ws = tc._workspace_state("demo")
    cache = {}
    run = tc._current_run({"status": "idle"}, ws, cache)
    assert run["id"] == "demo/gat/20261018-142000_run_d41f8b" and run["latest"] is True
    job_run = root / "resampled" / "runs" / "sage" / "20261020-090000_run_aaaaaa"
    job_run.mkdir(parents=True)
    monkeypatch.setattr(tc.training_manager, "run_dir", job_run)
    run = tc._current_run({"status": "running", "job_id": "j1"}, ws, cache)
    assert run["id"] == "resampled/sage/20261020-090000_run_aaaaaa" and run["latest"] is False
    assert run["workspace"] == "resampled"
    # A workspace the server does not list: the job names the run.
    outside = Path(str(root) + "-elsewhere") / "runs" / "gat" / "r"
    monkeypatch.setattr(tc.training_manager, "run_dir", outside)
    assert tc._current_run({"status": "running", "job_id": "j2"}, ws, {})["id"] == "job:j2"


@pytest.mark.parametrize("run_id", [
    "demo/gat/../../../etc", "demo/gat/20261018-142000_run_d41f8b/../x", "../demo/gat/x",
    "demo/gat", "demo/xyz/20261018-142000_run_d41f8b", "nope/gat/20261018-142000_run_d41f8b",
    "/abs/path", "job:none", None, 3, "demo/gat/" + "x" * 500,
])
def test_the_server_functions_answer_only_for_a_run_the_server_lists(root, run_id):
    assert tc.resolve_run(run_id) is None
    payload = tc.training_chart_series({"run": run_id, "rev": 1})
    assert payload["run"] is None and payload["charts"] == []
    assert tc.training_chart_csv({"run": run_id})["csv"] == ""


def test_a_run_entry_leaving_its_workspace_or_not_a_directory_is_not_read(root, tmp_path):
    """A run directory is found by listing ``runs/<arch>/``; an entry that is a
    link to a directory outside the workspace (another workspace's run, or any
    directory with a metrics log) or a plain file is not a run the server
    lists, so neither server function reads it."""
    outside = tmp_path / "elsewhere" / "run"
    (outside / "logs").mkdir(parents=True)
    (outside / "logs" / "train_1.json").write_text(json.dumps(_metrics(3)))
    arch = root / "demo" / "runs" / "gat"
    (arch / "20261019-000000_run_link").symlink_to(outside, target_is_directory=True)
    (arch / "20261019-000000_run_file").write_text("not a run")
    for run_id in ("demo/gat/20261019-000000_run_link", "demo/gat/20261019-000000_run_file"):
        assert tc.resolve_run(run_id) is None
        assert tc.training_chart_series({"run": run_id, "rev": 1})["charts"] == []
        assert tc.training_chart_csv({"run": run_id})["csv"] == ""
    # The listed run beside them still answers.
    assert tc.resolve_run("demo/gat/20261018-142000_run_d41f8b") is not None


def test_the_series_function_returns_the_four_charts_of_a_listed_run(root):
    run_id = "demo/gat/20261018-142000_run_d41f8b"
    assert tc.resolve_run(run_id) == root / "demo" / "runs" / "gat" / "20261018-142000_run_d41f8b"
    payload = tc.training_chart_series([{"run": run_id, "rev": 12}])
    assert payload["run"] == run_id and payload["rev"] == 12
    by_id = {c["id"]: c for c in payload["charts"]}
    assert list(by_id) == ["loss", "mrr", "hits", "lr"]
    assert [s["label"] for s in by_id["loss"]["series"]] == ["train", "validation"]
    assert by_id["mrr"]["title"] == "MRR (validation, Trainer top-20)"
    assert [s["label"] for s in by_id["hits"]["series"]] == ["Hits@1", "Hits@10"]
    assert by_id["lr"]["yScale"] == "log" and by_id["loss"]["yScale"] == "linear"
    assert by_id["loss"]["x"] == [1, 2, 3, 4, 5, 6]
    assert by_id["lr"]["series"][0]["values"][0] == 0.0     # the browser draws it as a gap
    assert by_id["hits"]["xLabel"] == by_id["lr"]["xLabel"] == "epoch (shared axis)"
    json.dumps(payload)


def test_a_hits_at_k_that_appears_later_joins_its_chart():
    history = {"train": [], "validation": [{"epoch": 0, "val_hits@1": 0.1, "val_hits@10": 0.2},
                                           {"epoch": 1, "val_hits@1": 0.1, "val_hits@3": 0.15,
                                            "val_hits@10": 0.2}]}
    hits = tc.training_charts(history)[2]
    assert [s["label"] for s in hits["series"]] == ["Hits@1", "Hits@3", "Hits@10"]
    assert hits["series"][1]["values"] == [None, 0.15]


def test_export_csv_is_the_whole_log_as_text(root):
    out = tc.training_chart_csv({"run": "demo/gat/20261018-142000_run_d41f8b"})
    lines = out["csv"].splitlines()
    assert lines[0] == ("epoch,split,learning_rate,total,val_hits@1,val_hits@10,val_loss,val_mrr")
    assert len(lines) == 1 + 12 and lines[1].startswith("1,train,0.0,")
    assert out["filename"] == "training_metrics_gat_20261018-142000_run_d41f8b.csv"


def test_the_chart_token_counts_rows_again_only_when_the_log_changed(root, monkeypatch):
    ws = tc._workspace_state("demo")
    cache = {}
    run = tc._current_run({"status": "idle"}, ws, cache)
    reads = []
    original = tc.training_manager.get_metrics_history

    def counted(run_dir=None):
        reads.append(run_dir)
        return original(run_dir)

    monkeypatch.setattr(tc.training_manager, "get_metrics_history", counted)
    assert tc._chart_token(run, cache) == {"run": run["id"], "rev": 12}
    assert tc._chart_token(run, cache) == {"run": run["id"], "rev": 12}
    assert len(reads) == 1
    log = run["path"] / "logs" / "train_20261018_142000.json"
    log.write_text(json.dumps(_metrics(7)))
    assert tc._chart_token(run, cache)["rev"] == 14 and len(reads) == 2
    assert tc._chart_token(None, cache) == {"run": None, "rev": 0, "empty": "No run to show yet."}


def test_the_used_dataset_fingerprint_is_an_operators_and_compared_in_full():
    ws = {"row": {"dataset_fingerprint": "a" * 64}, "digits": 12}
    run = tc.RunRef(id="demo/gat/r", label="gat/r", path=Path("/x"), workspace="demo",
                    latest=True)
    same = tc.run_head_html(run, ws, operator=True,
                            snapshot={"readable": True, "dataset_fingerprint": "a" * 64})
    assert "same as the run bar" in same and "a" * 12 in same
    other = tc.run_head_html(run, ws, operator=True,
                             snapshot={"readable": True, "dataset_fingerprint": "a" * 12 + "b" * 52})
    assert "≠ not the selected dataset" in other            # 12 equal digits are not equal
    assert "recorded at the first checkpoint" in tc.run_head_html(run, ws, operator=True,
                                                                  snapshot=None)
    read_only = tc.run_head_html(run, ws, operator=False,
                                 snapshot={"readable": True, "dataset_fingerprint": "a" * 64})
    assert "not shown · reading the checkpoint is an operator control" in read_only
    assert "a" * 12 not in read_only


def test_the_snapshot_sentence_names_no_path_to_a_read_only_request(root):
    run = tc.RunRef(id="demo/gat/r", label="gat/r", path=root / "demo" / "runs" / "gat" / "r",
                    workspace="demo", latest=False)
    snapshot = {"checkpoint": str(run["path"] / "checkpoints" / "last.pt"), "readable": True,
                "completed_epochs": 60, "target_epochs": 60, "end": "finished"}
    status = {"status": "completed", "current_epoch": 60, "total_epochs": 60}
    operator = tc.run_snapshot_html(run, status, operator=True, snapshot=snapshot)
    assert "Finished (epoch 60 of 60): demo/runs/gat/r/checkpoints/last.pt" in operator
    assert str(root) not in operator
    read_only = tc.run_snapshot_html(run, status, operator=False, snapshot=None)
    assert "Completed (epoch 60 of 60)." in read_only and "last.pt" not in read_only
    assert "reading the run’s checkpoint is checkpoint inspection" in read_only


def test_the_runs_checkpoint_is_read_again_only_when_it_changed(root, monkeypatch):
    run = tc.RunRef(id="demo/gat/r", label="gat/r", path=root / "demo" / "runs" / "gat" / "r",
                    workspace="demo", latest=True)
    (run["path"] / "checkpoints").mkdir(parents=True)
    last = run["path"] / "checkpoints" / "last.pt"
    reads = []
    monkeypatch.setattr(tc.training_manager, "latest_snapshot",
                        lambda run_dir=None: reads.append(run_dir) or {"readable": True})
    cache = {}
    assert tc._snapshot_for(run, cache) is None and reads == []      # no last.pt yet
    last.write_bytes(b"x")
    tc._snapshot_for(run, cache)
    tc._snapshot_for(run, cache)
    assert len(reads) == 1
    last.write_bytes(b"xy")
    tc._snapshot_for(run, cache)
    assert len(reads) == 2


# --------------------------------------------------------------------------- what a tick sends
def _rendered(**pieces):
    base = {key: f"<{key}>" for key in tc.VIEW_KEYS}
    base.update(dataset=((("a", "a"),), "a", True), start=True, cont=True, stop=False,
                resume=False, runbar=("shep-runbar",), page=("shep-page",), run_name=True,
                charts={"run": None, "rev": 0}, ws={"name": "a"}, training_workspace="a")
    base.update(pieces)
    return {"pieces": base, "cache": {}, "ws_changed": False}


def test_a_tick_sends_only_what_changed_and_everything_every_few_ticks():
    first = tc.view_updates(_rendered(), {})
    cache = first[-1]
    again = tc.view_updates(_rendered(), cache)
    assert all(u == gr.update() for u in again[:-1])
    changed = tc.view_updates(_rendered(pill="<other>", start=False), cache)
    by_key = dict(zip(tc.VIEW_KEYS, changed[:-1]))
    assert by_key["pill"] == "<other>" and by_key["start"] == gr.update(interactive=False)
    assert by_key["chip"] == gr.update()
    full = tc.view_updates(_rendered(), cache, full=True)
    by_key = dict(zip(tc.VIEW_KEYS, full[:-1]))
    assert by_key["chip"] == "<chip>" and by_key["page"] == gr.update(elem_classes=["shep-page"])


def test_a_tick_never_writes_the_sessions_selection_unless_it_changed_it():
    out = dict(zip(tc.VIEW_KEYS, tc.view_updates(_rendered(), {}, full=True)[:-1]))
    assert out["ws"] == gr.update() and out["training_workspace"] == gr.update()
    forced = _rendered()
    forced["ws_changed"] = True
    out = dict(zip(tc.VIEW_KEYS, tc.view_updates(forced, {})[:-1]))
    assert out["ws"] == {"name": "a"} and out["training_workspace"] == "a"


def test_the_full_send_comes_every_few_ticks(root, monkeypatch, operator):
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    ws = tc._workspace_state("demo")
    cache = {}
    sent_whole = []
    for tick in range(1, tc.FULL_SEND_TICKS * 2 + 1):
        out = tc._poll(shell.TRAINING, ws, {}, cache, [], operator)
        cache = out[-1]
        chip = dict(zip(tc.VIEW_KEYS, out[:-1]))["chip"]
        sent_whole.append(chip != gr.update())
    assert sent_whole[0] is True
    assert [i + 1 for i, whole in enumerate(sent_whole) if whole] == [
        1, tc.FULL_SEND_TICKS, tc.FULL_SEND_TICKS * 2]


def test_a_running_jobs_dataset_is_shown_and_the_select_disabled(root, monkeypatch, operator):
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    job_run = root / "resampled" / "runs" / "gat" / "20261020-090000_run_bbbbbb"
    job_run.mkdir(parents=True)
    monkeypatch.setattr(tc.training_manager, "run_dir", job_run)
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: {"status": "running", "job_id": "j", "current_epoch": 1,
                                 "total_epochs": 4, "gpu": {"job": {"job_type": "training"}}})
    rendered = tc.render(tc._workspace_state("demo"), {}, {}, [], operator)
    assert rendered["ws_changed"] is True and rendered["pieces"]["ws"]["name"] == "resampled"
    assert rendered["pieces"]["dataset"][1:] == ("resampled", False)
    assert "in use by the running job" in rendered["pieces"]["chip"]
    assert rendered["pieces"]["stop"] is True and rendered["pieces"]["start"] is False


def test_a_read_only_request_sees_every_action_disabled_and_no_checkpoint(root, monkeypatch):
    # The run shown has a saved state, which an operator's render reads.
    run = root / "demo" / "runs" / "gat" / "20261018-142000_run_d41f8b"
    (run / "checkpoints" / "last.pt").write_bytes(b"saved state")
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    reads = []
    monkeypatch.setattr(tc.training_manager, "latest_snapshot",
                        lambda *a, **k: reads.append(a) or {"readable": True})
    operator_mode(monkeypatch)
    tc.render(tc._workspace_state("demo"), {}, {}, [], gradio_request())
    assert len(reads) == 1
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    monkeypatch.setattr(tc.training_manager, "latest_snapshot",
                        lambda *a, **k: pytest.fail("a read-only request read the checkpoint"))
    pieces = tc.render(tc._workspace_state("demo"), {}, {}, [], READ_ONLY)["pieces"]
    assert not any(pieces[k] for k in ("start", "cont", "stop", "resume", "run_name"))
    assert "Operator controls only: deployment mode not set to M1" in pieces["reason"]
    assert "not shown · reading the checkpoint is an operator control" in pieces["head"]
    assert "last.pt" not in pieces["snap"] and "runs/" not in pieces["snap"]


# --------------------------------------------------------------------------- the banner
def test_the_banner_is_dismissed_for_this_session_until_its_content_changes(root, monkeypatch,
                                                                            operator):
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: {"status": "failed", "error_message": "Process exited with code 1",
                                 "gpu": {"job": None}})
    rendered = tc.render(tc._workspace_state("demo"), {}, {}, [], operator)
    assert "Process exited with code 1" in rendered["pieces"]["banner"]
    key = rendered["cache"]["banner_key"]

    class Evt:
        _data = {"act": "dismiss"}

    dismissed, banner = tc._on_banner_click([], rendered["cache"], Evt())
    assert dismissed == [key] and banner == ""
    again = tc.render(tc._workspace_state("demo"), {}, {}, dismissed, operator)
    assert again["pieces"]["banner"] == ""
    other_session = tc.render(tc._workspace_state("demo"), {}, {}, [], operator)
    assert other_session["pieces"]["banner"]               # another session still sees it


def test_the_banner_names_no_absolute_path_to_a_read_only_request(root, monkeypatch):
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    message = ("Process exited with code 1\n\n--- Last output ---\n"
               "FileNotFoundError: /srv/secret/place/kg.json")
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: {"status": "failed", "error_message": message,
                                 "gpu": {"job": None}})
    banner = tc.render(tc._workspace_state("demo"), {}, {}, [], READ_ONLY)["pieces"]["banner"]
    assert "/srv/secret" not in banner and "a path outside the workspaces root" in banner


def test_the_banner_escapes_the_output():
    html, _key = tc._build_error_banner_payload(
        {"error_message": "Process exited\n\n--- Last output ---\n<script>alert(1)</script>"},
        "<img src=x>")
    assert "<script>" not in html and "&lt;script&gt;" in html and "<img" not in html
