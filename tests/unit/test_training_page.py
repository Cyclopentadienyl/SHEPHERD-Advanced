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
- **one writer of the run bar and the run view**: a fast tick writes only the
  status pill (and the chips) and hands every other change to a queued
  render, which sends what changed (everything every few hand-offs), and
  neither draws an older read of the GPU lock or the run list than the last
  render drew (P2-NEW-1's verification, F1); every
  writer of the run bar's workspace and of what is drawn from it (Datasets'
  "Use for training" included) is in one queued group, so a late tick or
  render never writes back the workspace it started from, nor shows or starts
  its dataset (the pages served by Gradio, one read held; the reviewer's
  reproduction is `tests/unit/test_training_poll_display.py`);
- **the banner**: escaped, red for a failure, amber for PyG, dismissed per
  session; no absolute path for a read-only request (each text redacted
  before it is escaped); the last error line the run printed under the
  headline, and where its log is only when it has one.

Module: tests/unit/test_training_page.py
"""
from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

gr = pytest.importorskip("gradio")

from src.webui import shell  # noqa: E402
from src.webui.components import training_console as tc  # noqa: E402
from tests.fixtures.operator_access import gradio_request, operator_mode  # noqa: E402
from tests.fixtures.served_page import ServedSession, served  # noqa: E402

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
    out = tc._poll("clinical", {}, {}, {}, [], {}, gradio_request())
    assert len(out) == len(tc.POLL_KEYS) + 3 and all(u == gr.update() for u in out)
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
def test_the_select_names_the_servers_workspaces_and_leaves_their_identity_to_the_chip(root):
    """The server's list, in the shared order; a trainable workspace's option
    is its name, its identity (fingerprint, synthetic) the chip's beside it
    (the GB10 WebUI recheck: said twice side by side); one that cannot be
    trained keeps its reason in its option, read before choosing."""
    from src.webui.components import renderers as r
    from src.webui.components.workspaces import read_workspaces

    state = tc._workspace_state("demo")
    rows = read_workspaces()["rows"]
    trainable = {x["name"] for x in rows if r.is_digest(x.get("dataset_fingerprint"))}
    assert [tuple(c) for c in state["choices"]] == [
        (name if name in trainable else label, name) for label, name in r.workspace_choices(rows)]
    assert ("demo", "demo") in [tuple(c) for c in state["choices"]]
    assert any(label != name for label, name in state["choices"])      # a reason kept
    assert "dataset fingerprint" in tc.chip_html(state)
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
    assert "What the manifest binds" in html and "allocation seed" in html
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
    confirming = _view({"status": "running", "job_id": "j1"}, mode={"confirm": "j1"})
    assert confirming["stop"] is False
    # A confirmation asked for an earlier job is not this job's.
    earlier = _view({"status": "running", "job_id": "j2"}, mode={"confirm": "j1"})
    assert earlier["stop"] is True

    read_only = _view(access=_Access(False, "not_loopback"))
    assert not any(read_only[k] for k in ("start", "cont", "stop", "resume"))
    assert "Operator controls only: request not local" in read_only["reason"]

    test_run = _view(gpu={"text": "GPU job: test run (mygene2-2026-09 · 00:01:42)", "test": True})
    assert test_run["start"] is False and test_run["cont"] is False
    assert "GPU job: test run (mygene2-2026-09 · 00:01:42)" in test_run["reason"]
    assert 'data-act="open-test"' in test_run["reason"]
    cli = _view(gpu={"text": "GPU job: train_model.py (pid 41822, since 09:58 UTC)",
                     "test": False})
    assert "train_model.py (pid 41822" in cli["reason"] and "open-test" not in cli["reason"]

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
    held = {"configured": True, "problem": None, "holder_alive": True,
            "holder": {"command": "train_model.py", "pid": 41822,
                       "started_at": "2026-10-20T09:58:00+00:00"},
            "held_by": "train_model.py (pid 41822)"}
    cli = tc._gpu_line(None, held, None, operator=False)
    assert cli == {"text": "GPU job: train_model.py (pid 41822, since 09:58 UTC)",
                   "test": False}
    assert tc._gpu_line(None, {"configured": True, "held_by": None}, None,
                        operator=True) is None
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


def test_an_architecture_that_is_not_one_never_lists_another_directory(root):
    """M5-3 review ARCH-10 (d): the run id's architecture must be a supported
    one. ``demo/../runs`` would otherwise list the workspace's own directory
    (``runs/..``) and find ``runs`` there, inside the workspace, as a "run"."""
    assert (root / "demo" / "runs").is_dir()
    for run_id in ("demo/../runs", "demo/./gat", "demo/GAT/20261018-142000_run_d41f8b"):
        assert tc.resolve_run(run_id) is None, run_id
        assert tc.training_chart_series({"run": run_id, "rev": 1})["charts"] == []


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
    assert "same dataset as the run bar" in same and "a" * 12 in same     # the tag; Details
    other = tc.run_head_html(run, ws, operator=True,
                             snapshot={"readable": True, "dataset_fingerprint": "a" * 12 + "b" * 52})
    assert "≠ not the selected dataset" in other            # 12 equal digits are not equal
    assert "recorded at the first checkpoint" in tc.run_head_html(run, ws, operator=True,
                                                                  snapshot=None, live=True)
    read_only = tc.run_head_html(run, ws, operator=False,
                                 snapshot={"readable": True, "dataset_fingerprint": "a" * 64})
    assert "dataset not shown · operator control" in read_only
    assert "not shown: reading the checkpoint is an operator control" in read_only
    assert "a" * 12 not in read_only


def test_the_snapshot_sentence_names_no_path_to_a_read_only_request(root):
    run = tc.RunRef(id="demo/gat/r", label="gat/r", path=root / "demo" / "runs" / "gat" / "r",
                    workspace="demo", latest=False)
    (run["path"] / "checkpoints").mkdir(parents=True)       # a run that saved its last.pt
    snapshot = {"checkpoint": str(run["path"] / "checkpoints" / "last.pt"), "readable": True,
                "completed_epochs": 60, "target_epochs": 60, "end": "finished"}
    status = {"status": "completed", "current_epoch": 60, "total_epochs": 60}
    operator = tc.run_snapshot_html(run, status, operator=True, snapshot=snapshot)
    assert "Finished · 60 of 60 epochs completed · to train further, use Continue…" in operator
    assert str(root) not in operator and "last.pt" not in operator     # the path: Details
    ws = {"row": {"dataset_fingerprint": "a" * 64}, "digits": 12}
    head = tc.run_head_html(run, ws, operator=True, snapshot=snapshot)
    assert "demo/runs/gat/r/checkpoints/last.pt" in head and str(root) not in head
    read_only = tc.run_snapshot_html(run, status, operator=False, snapshot=None)
    assert "Completed · epoch 60 of 60 · checkpoint details are an operator control" in read_only
    early = {**snapshot, "end": "stopped_early", "completed_epochs": 40, "target_epochs": 60}
    assert "Stopped early · 40 of 60 epochs completed · cannot be continued" in tc.run_snapshot_html(
        run, status, operator=True, snapshot=early)
    assert "Stopped early (patience) · " in tc.run_snapshot_html(
        run, status, operator=True, snapshot={**early, "stop_reason": "patience"})
    assert "last.pt" not in read_only
    assert "last.pt" not in tc.run_head_html(run, ws, operator=False, snapshot=None)


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


def test_a_tick_writes_only_the_pill_and_hands_every_other_change_over(
        root, monkeypatch, operator):
    """P2-NEW-1: the fast tick writes the status pill (on every tick: it is
    drawn from the job runner alone), the chips and its own record, never a
    piece drawn from the selection and never the cache. It hands a change
    over only when what it renders now is not what the session was last
    sent: nothing after a render; new epochs of the run shown, yes."""
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    demo, _ctx, by_id = _page()
    poll = next(fn for fn in demo.fns.values() if fn.name == "_poll")
    assert poll.outputs[0] is by_id["shep-train-pill"] and tc.POLL_KEYS == ("pill",)
    cache_state = poll.inputs[3]                               # the renders' record
    assert isinstance(cache_state, gr.State) and cache_state not in poll.outputs
    assert not {by_id[k] for k in ("shep-train-dataset", "shep-train-chip", "shep-train-start",
                                   "shep-train-charts", "shep-train-run-head")} & set(poll.outputs)
    ws = tc._workspace_state("demo")
    cache = tc._render_view(ws, {}, {}, [], operator)[-1]
    pill, handoff, _chips, polled = tc._poll(shell.TRAINING, ws, {}, cache, [], {}, operator)
    assert pill.startswith('<div class="shep-pill') and handoff == gr.update()
    again = tc._poll(shell.TRAINING, ws, {}, cache, [], polled, operator)
    assert again[0] == pill and again[1] == gr.update()       # the pill on every tick
    log = root / "demo" / "runs" / "gat" / "20261018-142000_run_d41f8b" / "logs" / "train_20261018_142000.json"
    log.write_text(json.dumps(_metrics(7)))                    # a new epoch, nobody acted
    _pill, handoff, _chips, _polled = tc._poll(shell.TRAINING, ws, {}, cache, [], polled, operator)
    assert handoff["changed"] == ["charts"]


def test_the_handed_over_render_sends_what_changed_and_everything_every_few(
        root, monkeypatch, operator):
    """The render a tick hands a change over to sends only what changed, and
    everything every `FULL_SEND_TICKS` hand-offs; the render after an action
    sends everything and starts the count again."""
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    ws = tc._workspace_state("demo")
    cache = tc._render_view(ws, {}, {}, [], operator)[-1]
    sent_whole = []
    for _ in range(tc.FULL_SEND_TICKS * 2):
        out = tc._render_changed(ws, {}, cache, [], {}, operator)
        cache = out[-1]
        sent_whole.append(dict(zip(tc.VIEW_KEYS, out[:-1], strict=True))["chip"] != gr.update())
    assert [i + 1 for i, whole in enumerate(sent_whole) if whole] == [
        tc.FULL_SEND_TICKS, tc.FULL_SEND_TICKS * 2]
    cache = tc._render_view(ws, {}, cache, [], operator)[-1]
    assert cache["handed"] == 0


def test_the_header_chips_follow_the_gpu_slot_on_the_tick_that_sees_it_change(
        root, monkeypatch, operator):
    """M5-3: the chip read "GPU jobs: none" while the run bar showed a test
    run (the chips on the slow timer, the run bar on the fast one). The fast
    tick that sees the GPU slot change reads the chips again for its request;
    a tick that sees no change sends nothing (no timer, no request of its own)."""
    from src.webui.components import chips

    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    monkeypatch.setattr(tc, "lock_state", lambda: {"configured": True, "held_by": None})
    read = []
    monkeypatch.setattr(chips, "read_chips", lambda request: read.append(1) or "<chips>")
    status = {"status": "idle"}
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: dict(status))
    monkeypatch.setattr(tc.training_manager, "get_test_status",
                        lambda: {"status": "running", "inputs": {"version_label": "v"}})
    demo, ctx, _by_id = _page()
    poll_fn = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "_poll")
    assert poll_fn.outputs[2] is ctx.chips
    ws = tc._workspace_state("demo")
    first = tc._poll(shell.TRAINING, ws, {}, {}, [], {}, operator)
    assert first[2] == "<chips>" and len(read) == 1         # nothing seen before: read
    same = tc._poll(shell.TRAINING, ws, {}, {}, [], first[-1], operator)
    assert same[2] == gr.update() and len(read) == 1         # unchanged: nothing sent
    status["gpu"] = {"job": {"job_type": "test", "status": "running",
                             "start_time": "2026-10-21T18:25:13"}}
    started = tc._poll(shell.TRAINING, ws, {}, {}, [], same[-1], operator)
    assert started[2] == "<chips>" and len(read) == 2        # a test run took the slot
    later = tc._poll(shell.TRAINING, ws, {}, {}, [], started[-1], operator)
    assert later[2] == gr.update() and len(read) == 2        # its elapsed time is no change


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


# --------------------------------------------------------------------------- the run bar's workspace, in order
RUNNING = {"status": "running", "job_id": "j", "current_epoch": 1, "total_epochs": 4,
           "gpu": {"job": {"job_type": "training", "status": "running"}}}
ENDED = {"status": "completed", "job_id": "j", "current_epoch": 4, "total_epochs": 4}


def _job_in_resampled(root, monkeypatch):
    """The job runner as the page reads it, its training job's run in
    ``resampled``: ``job.status`` is what it says (idle at first), and while
    ``job.hold`` is set the next read is held right after it was taken (the
    read a tick's or a render's view is computed from) until ``job.release``.
    Nothing is served."""
    from src.webui.components import datasets_page, served_status

    monkeypatch.setattr(served_status, "read_status", lambda: (None, "not read"))
    monkeypatch.setattr(datasets_page, "read_status", lambda: (None, "not read"))
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    monkeypatch.setattr(tc, "lock_state", lambda: {"configured": True, "held_by": None})
    job_run = root / "resampled" / "runs" / "gat" / "20261020-090000_run_bbbbbb"
    job_run.mkdir(parents=True)
    monkeypatch.setattr(tc.training_manager, "run_dir", job_run)
    job = SimpleNamespace(status={"status": "idle"}, hold=threading.Event(),
                          entered=threading.Event(), release=threading.Event())

    def get_status():
        now = json.loads(json.dumps(job.status))
        if job.hold.is_set():
            job.hold.clear()
            job.entered.set()
            assert job.release.wait(30), "never released"
        return now

    monkeypatch.setattr(tc.training_manager, "get_status", get_status)
    return job


def _datasets_and_training():
    """Datasets and Training in one Blocks, as the app builds them: both write
    the session's run-bar workspace (``ctx.training_workspace``)."""
    from src.webui.components import datasets_page

    with gr.Blocks() as demo:
        ctx = shell.standalone_context(demo)
        datasets_page.create_datasets_page(ctx)
        tc.create_training_tab(ctx)
    fns = list(demo.fns.values())
    named = {}
    for fn in fns:
        named.setdefault(fn.name, fn)
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    return demo, ctx, named, by_id


def _shown(answers, shown=None):
    """What the browser shows after it applied `answers` in order, each
    ``(listener, its output values)``: ``{elem_id: {"value": …, "interactive":
    …}}``. An update leaves the keys it does not carry as they were, and a
    component that is not one of a listener's outputs is not touched."""
    shown = {} if shown is None else shown
    for fn, data in answers:
        for block, value in zip(fn.outputs, data, strict=True):
            elem_id = getattr(block, "elem_id", None)
            if elem_id is None or isinstance(block, gr.State):
                continue
            now = shown.setdefault(elem_id, {})
            if isinstance(value, dict) and value.get("__type__") == "update":
                now.update({k: v for k, v in value.items() if k != "__type__"})
            else:
                now["value"] = value
    return shown


def _answered(browser, fn, event):
    """A queued event's answer, as `_shown` takes it."""
    return fn, browser.done[event]["output"]["data"]


def _after(demo, fn, name):
    """The step chained after `fn` (``.then``) whose function is `name`."""
    current = fn
    while current is not None:
        current = next((d for d in demo.fns.values() if d.trigger_after == current._id), None)
        if current is not None and current.name == name:
            return current
    raise AssertionError(f"no {name} after {fn.name}")


def test_every_write_of_the_run_bars_workspace_is_in_one_queued_group_of_one():
    """F7 and P2-NEW-1: the session's run-bar workspace (Training's ``ws`` and
    ``ctx.training_workspace``), every piece drawn from it (the select's
    value, the chip, Start, Continue…, Stop and their reason, the run shown,
    its header, snapshot sentence and chart token, the drawer's line) and the
    session's record of what it was sent (the cache) are written only by
    queued listeners of `shell.TRAINING_SELECTION_CONCURRENCY`, one at a
    time: Datasets' "Use for training ›", the page open, the select, the
    change of ``training_workspace``, every render after an action and the
    render a tick hands a change over to. The fast poll writes none of them:
    the pill, the chips, its own record and the hand-off, a hidden component
    (not a State, whose change a ``queue=False`` answer never dispatches)
    whose change runs the queued render. The operator's choices join every
    time they are asked; the page open's render is chained after it, not a
    second listener on the opening (the two ran in either order)."""
    demo, ctx, named, by_id = _datasets_and_training()
    poll = named["_poll"]
    selection = (poll.inputs[1], ctx.training_workspace)
    writers = [fn for fn in demo.fns.values() if set(selection) & set(fn.outputs)]
    assert {fn.name for fn in writers} == {"_use", "_open", "_on_dataset",
                                           "_on_training_workspace", "_render_view",
                                           "_render_changed"}
    drawn = {by_id[elem_id] for elem_id in (
        "shep-train-dataset", "shep-train-chip", "shep-train-reason", "shep-train-start",
        "shep-train-continue", "shep-train-stop", "shep-train-resume", "shep-train-runbar",
        "shep-train-page", "shep-train-run-head", "shep-train-run-snap", "shep-train-charts",
        "shep-train-drawer-why")}
    cache = poll.inputs[3]
    projection = [fn for fn in demo.fns.values() if (drawn | {cache}) & set(fn.outputs)]
    assert {fn.name for fn in projection} == {"_open", "_on_dataset", "_on_training_workspace",
                                              "_render_view", "_render_changed"}
    for fn in writers + projection:
        assert fn.queue is not False, fn.name
        assert (fn.concurrency_id, fn.concurrency_limit) == (
            shell.TRAINING_SELECTION_CONCURRENCY, 1), fn.name
    for name in ("_on_dataset", "_on_training_workspace"):
        assert named[name].trigger_mode == "multiple", name
    changed = by_id[tc.CHANGED_ID]
    polled = poll.inputs[-1]
    assert poll.queue is False
    assert poll.outputs == [by_id["shep-train-pill"], changed, ctx.chips, polled]
    assert not isinstance(changed, gr.State)
    assert [fn.name for fn in demo.fns.values() if polled in fn.outputs] == ["_poll"]
    handed = [fn for fn in demo.fns.values() if (changed._id, "change") in fn.targets]
    assert [fn.name for fn in handed] == ["_render_changed"]
    opening = [fn for fn in demo.fns.values()
               if (ctx.page_tokens[shell.TRAINING]._id, "change") in fn.targets]
    assert [fn.name for fn in opening] == ["_open"]
    assert _after(demo, opening[0], "_render_view").trigger_after == opening[0]._id


def test_the_poll_hands_a_running_jobs_workspace_over_and_writes_no_selection(
        root, monkeypatch, operator):
    """F7: a tick that finds a training job running in another workspace
    than the session's writes no selection (it writes the pill, `POLL_KEYS`)
    and gives the hand-off token a new value each time, naming the change;
    the queued render follows the job. With the job no longer running and
    the page as the session was sent it, a tick hands nothing over and a
    render writes no selection."""
    job = _job_in_resampled(root, monkeypatch)
    ws = tc._workspace_state("demo")
    cache = tc._render_view(ws, {}, {}, [], operator)[-1]
    job.status = dict(RUNNING)
    out = tc._poll(shell.TRAINING, ws, {}, cache, [], {}, operator)
    assert len(out) == len(tc.POLL_KEYS) + 3 and "ws" not in tc.POLL_KEYS
    assert "training_workspace" not in tc.POLL_KEYS
    handoff = out[1]
    assert "ws" in handoff["changed"] and "dataset" in handoff["changed"]
    again = tc._poll(shell.TRAINING, ws, {}, cache, [], out[-1], operator)
    assert again[1]["at"] != handoff["at"], "each hand-off is a new value"
    view = dict(zip(tc.VIEW_KEYS, tc._render_changed(ws, {}, cache, [], out[-1], operator)[:-1],
                    strict=True))
    assert view["ws"]["name"] == "resampled" == view["training_workspace"]
    job.status = dict(ENDED)
    rendered = tc._render_view(ws, {}, {}, [], operator)
    view = dict(zip(tc.VIEW_KEYS, rendered[:-1], strict=True))
    assert view["ws"] == gr.update() and view["training_workspace"] == gr.update()
    assert tc._poll(shell.TRAINING, ws, {}, rendered[-1], [], {}, operator)[1] == gr.update()


def test_a_late_tick_never_writes_back_the_workspace_it_started_from(root, monkeypatch):
    """F7, through the pages served by Gradio, one session: the fast tick's
    read of the job runner is held (the job running in ``resampled``, the
    session on ``demo``); meanwhile the job ends and Datasets' "Use for
    training ›" chooses ``other``. Released, the tick hands its view over
    and the queued render it runs, reading the job ended, leaves the choice.
    When the tick wrote the session's workspace itself, it wrote back the
    job's over the choice. Positive control: a job running in another
    workspace still moves the session's workspace on the next tick, through
    the hand-off and not the tick's own answer."""
    shutil.copytree(root / "demo", root / "other")
    job = _job_in_resampled(root, monkeypatch)
    demo, ctx, named, by_id = _datasets_and_training()
    poll, opened, choose, use, select = (named[n] for n in (
        "_poll", "_open", "_on_dataset", "_use", "_select"))
    ws = poll.inputs[1]
    changed = by_id[tc.CHANGED_ID]
    hand_over = next(fn for fn in demo.fns.values() if (changed._id, "change") in fn.targets)
    from_datasets = named["_on_training_workspace"]
    datasets_open = next(fn for fn in demo.fns.values()
                         if (ctx.page_tokens[shell.DATASETS]._id, "change") in fn.targets)

    with served(demo) as base:
        browser = ServedSession(base, demo)
        assert browser.wait(browser.join(opened, [None, None]))
        assert browser.wait(browser.join(_after(demo, opened, "_render_view"), [None] * 4))
        assert browser.wait(browser.join(choose, ["demo", None]))
        assert browser.state(ws)["name"] == "demo"
        browser.set_state(ctx.location, shell.TRAINING)

        def tick(answers):
            answers.append(browser.predict(poll, [None] * len(poll.inputs)))

        job.status = dict(RUNNING)
        job.hold.set()
        answers = []
        ticking = threading.Thread(target=tick, args=(answers,), daemon=True)
        ticking.start()
        assert job.entered.wait(30), "the tick's read did not start"
        job.status = dict(ENDED)
        # Datasets: the page read, the row chosen, "Use for training ›", and
        # the change of the session's workspace that its answer makes.
        browser.set_state(ctx.location, shell.DATASETS)
        assert browser.wait(browser.join(datasets_open, [None, None]))
        assert browser.wait(browser.join(select, [None, None], {"key": "other"}))
        assert browser.wait(browser.join(use, [None, None]))
        assert browser.state(ctx.training_workspace) == "other"
        assert browser.wait(browser.join(from_datasets, [None, None, None]))
        assert browser.wait(browser.join(_after(demo, from_datasets, "_render_view"), [None] * 4))
        assert browser.state(ws)["name"] == "other"
        job.release.set()
        ticking.join(30)
        (answer,) = answers
        token = answer[poll.outputs.index(changed)]
        if isinstance(token, dict) and token.get("__type__") != "update":
            assert browser.wait(browser.join(hand_over, [None] * len(hand_over.inputs)))
        assert browser.state(ws)["name"] == "other", \
            "the tick wrote back the job's workspace over the one chosen"
        assert browser.state(ctx.training_workspace) == "other"

        # Positive control: a job running in another workspace moves it.
        browser.set_state(ctx.location, shell.TRAINING)
        job.status = dict(RUNNING)
        answer = browser.predict(poll, [None] * len(poll.inputs))
        token = answer[poll.outputs.index(changed)]
        assert "ws" in token["changed"]
        assert browser.state(ws)["name"] == "other", "the tick wrote the selection itself"
        assert browser.wait(browser.join(hand_over, [None] * len(hand_over.inputs)))
        assert browser.state(ws)["name"] == "resampled"
        assert browser.state(ctx.training_workspace) == "resampled"
        browser.close()


def test_a_workspace_chosen_while_a_render_waits_is_applied_after_it(root, monkeypatch):
    """F7, through the pages served by Gradio, one session: the render the
    poll handed a running job's workspace to is held in its read of the job
    runner (the job running in ``resampled``); meanwhile the job ends and
    Datasets' "Use for training ›" chooses ``other``. In the group the choice
    waits for the render and is applied after it, so the session ends on
    ``other``. With each listener its own group, the choice was applied
    first and the held render, ending last, wrote back the job's workspace."""
    shutil.copytree(root / "demo", root / "other")
    job = _job_in_resampled(root, monkeypatch)
    demo, ctx, named, by_id = _datasets_and_training()
    opened, choose, use, select = (named[n] for n in ("_open", "_on_dataset", "_use",
                                                       "_select"))
    ws = named["_poll"].inputs[1]
    changed = by_id[tc.CHANGED_ID]
    hand_over = next(fn for fn in demo.fns.values() if (changed._id, "change") in fn.targets)
    from_datasets = named["_on_training_workspace"]
    datasets_open = next(fn for fn in demo.fns.values()
                         if (ctx.page_tokens[shell.DATASETS]._id, "change") in fn.targets)

    with served(demo) as base:
        browser = ServedSession(base, demo)
        assert browser.wait(browser.join(opened, [None, None]))
        assert browser.wait(browser.join(choose, ["demo", None]))
        assert browser.wait(browser.join(datasets_open, [None, None]))
        assert browser.wait(browser.join(select, [None, None], {"key": "other"}))
        assert browser.state(ws)["name"] == "demo"

        job.status = dict(RUNNING)
        job.hold.set()
        render = browser.join(hand_over, [None] * len(hand_over.inputs))
        assert job.entered.wait(30), "the render's read did not start"
        job.status = dict(ENDED)
        chosen = browser.join(use, [None, None])
        browser.wait(chosen, timeout=1.5)
        job.release.set()
        assert browser.wait(render, chosen)
        # The change of the session's workspace that the choice's answer makes.
        assert browser.wait(browser.join(from_datasets, [None, None, None]))
        assert browser.wait(browser.join(_after(demo, from_datasets, "_render_view"), [None] * 4))
        browser.close()
        assert browser.state(ctx.training_workspace) == "other"
        assert browser.state(ws)["name"] == "other", \
            "a render that read the job running wrote back its workspace over the choice"


class _TrainingSession:
    """One session on the pages served by Gradio, Training shown, as a browser
    drives it: a choice in the select and the render after it, a fast tick,
    and the queued render a tick's hand-off runs (the hidden component's
    change, which the browser dispatches on the tick's answer). Every answer
    is kept in the order the browser applied it (`shown`)."""

    def __init__(self, browser, demo, ctx, named, by_id):
        self.browser, self.ctx, self.by_id = browser, ctx, by_id
        self.poll, self.choose = named["_poll"], named["_on_dataset"]
        self.opened = named["_open"]
        self.after_choice = _after(demo, self.choose, "_render_view")
        self.after_open = _after(demo, self.opened, "_render_view")
        self.token = by_id[tc.CHANGED_ID]
        self.hand_over = next(fn for fn in demo.fns.values()
                              if (self.token._id, "change") in fn.targets)
        self.ws = self.poll.inputs[1]
        self.answers = []

    def _run(self, fn, data):
        event = self.browser.join(fn, data)
        assert self.browser.wait(event)
        self.answers.append(_answered(self.browser, fn, event))

    def open(self):
        self._run(self.opened, [None, None])
        self._run(self.after_open, [None] * len(self.after_open.inputs))
        self.browser.set_state(self.ctx.location, shell.TRAINING)

    def pick(self, name):
        self._run(self.choose, [name, None])
        self._run(self.after_choice, [None] * len(self.after_choice.inputs))

    def tick(self):
        """A fast tick; its hand-off, if it gave one, run. Returns what it
        named as changed (``[]`` for no hand-off)."""
        answer = self.browser.predict(self.poll, [None] * len(self.poll.inputs))
        self.answers.append((self.poll, answer))
        token = answer[self.poll.outputs.index(self.token)]
        if isinstance(token, dict) and token.get("__type__") == "update":
            return []
        self._run(self.hand_over, [None] * len(self.hand_over.inputs))
        return token["changed"]

    def shown(self, elem_id, key="value"):
        return _shown(self.answers).get(elem_id, {}).get(key)


@pytest.fixture
def training_page(root, monkeypatch, operator):
    """The job runner as `_job_in_resampled` gives it (idle at first), and one
    operator's session on Datasets and Training served by Gradio, the page
    opened."""
    job = _job_in_resampled(root, monkeypatch)
    demo, ctx, named, by_id = _datasets_and_training()
    with served(demo) as base:
        browser = ServedSession(base, demo)
        page = _TrainingSession(browser, demo, ctx, named, by_id)
        page.open()
        try:
            yield job, page
        finally:
            browser.close()


def test_a_choice_while_idle_is_drawn_whole_and_a_tick_hands_nothing_over(training_page):
    """The normal path, served: with no job, each choice is drawn by its
    render (the select, the chip, the run shown and Start), and a tick after
    it finds the page as it was sent: it writes the pill and hands nothing
    over."""
    _job, page = training_page
    page.pick("demo")
    page.pick("resampled")
    assert page.tick() == []
    resampled = page.browser.state(page.ws)
    assert resampled["name"] == "resampled"
    assert page.shown("shep-train-dataset") == "resampled"
    assert page.shown("shep-train-dataset", "interactive") is True
    assert resampled["row"]["dataset_fingerprint"][:12] in page.shown("shep-train-chip")
    assert page.shown("shep-train-charts")["run"] == "resampled/gat/20261020-090000_run_bbbbbb"
    assert page.shown("shep-train-start", "interactive") is True
    assert "Idle" in page.shown("shep-train-pill")


def test_a_choice_while_a_job_runs_elsewhere_follows_the_job_only_while_it_runs(training_page):
    """Served: while a training job runs in ``resampled``, the render after a
    choice of ``demo`` shows the job's dataset, the select disabled, "in use
    by the running job" (as before); a tick then hands nothing over. The end
    is handed over once: the select and Start come back, on the job's
    dataset; a choice after it stays."""
    job, page = training_page
    page.pick("demo")
    job.status = dict(RUNNING)
    page.pick("demo")
    assert page.browser.state(page.ws)["name"] == "resampled"
    assert page.shown("shep-train-dataset") == "resampled"
    assert page.shown("shep-train-dataset", "interactive") is False
    assert "in use by the running job" in page.shown("shep-train-chip")
    assert page.shown("shep-train-start", "interactive") is False
    assert page.shown("shep-train-stop", "interactive") is True
    assert page.tick() == []
    job.status = dict(ENDED)
    changed = page.tick()
    assert {"start", "stop", "dataset", "chip", "reason"} <= set(changed)
    assert page.shown("shep-train-dataset") == "resampled"
    assert page.shown("shep-train-dataset", "interactive") is True
    assert "in use by the running job" not in page.shown("shep-train-chip")
    assert page.shown("shep-train-start", "interactive") is True
    assert page.shown("shep-train-stop", "interactive") is False
    assert "Completed" in page.shown("shep-train-pill")
    assert page.tick() == []
    page.pick("demo")
    assert page.tick() == []
    assert page.browser.state(page.ws)["name"] == "demo" == page.shown("shep-train-dataset")


def test_new_epochs_reach_the_charts_and_the_run_head_while_nobody_acts(
        training_page, root, monkeypatch):
    """Served: while the job runs, nobody acting, each new epoch (its log's
    rows, the epoch the runner reports) is handed over and drawn: the chart
    token's rows, the progress in the reason line; the run's first saved
    state puts the used dataset fingerprint in the run head. A tick that
    finds nothing new hands nothing over."""
    job, page = training_page
    job_run = root / "resampled" / "runs" / "gat" / "20261020-090000_run_bbbbbb"
    (job_run / "logs").mkdir()
    log = job_run / "logs" / "train_20261020_090000.json"
    log.write_text(json.dumps(_metrics(1)))
    fingerprint = tc._workspace_state("resampled")["row"]["dataset_fingerprint"]
    monkeypatch.setattr(tc.training_manager, "latest_snapshot", lambda run_dir=None: {
        "readable": True, "dataset_fingerprint": fingerprint, "completed_epochs": 1,
        "checkpoint": str(job_run / "checkpoints" / "last.pt")})
    job.status = dict(RUNNING)
    page.pick("resampled")
    assert page.tick() == []
    rows = page.shown("shep-train-charts")["rev"]
    assert "recorded at the first checkpoint" in page.shown("shep-train-run-head")
    log.write_text(json.dumps(_metrics(2)))
    job.status = {**RUNNING, "current_epoch": 2}
    changed = page.tick()
    assert {"charts", "reason"} <= set(changed)
    assert page.shown("shep-train-charts")["rev"] > rows
    assert 'aria-valuenow="50"' in page.shown("shep-train-reason")
    (job_run / "checkpoints").mkdir()
    (job_run / "checkpoints" / "last.pt").write_bytes(b"saved state")
    assert "head" in page.tick()
    head = page.shown("shep-train-run-head")
    assert fingerprint[:12] in head and "same dataset as the run bar" in head
    assert page.tick() == []


def _clock(monkeypatch):
    """The page's clock, moved on by hand (``ahead``, seconds): the reads a
    render and a tick keep by time (`tc.POLL_READS`) measure their age by
    ``datetime.now()``."""
    from datetime import datetime, timedelta

    class Clock(datetime):
        ahead = 0.0

        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(seconds=cls.ahead)

    monkeypatch.setattr(tc, "datetime", Clock)
    return Clock


HELD_LOCK = {"configured": True,
             "held_by": "a training job (train_model.py, pid 4242 on host, since 2026-10-23T06:00:00)"}


def test_a_tick_never_draws_an_older_lock_read_than_the_render_after_a_choice(
        training_page, monkeypatch):
    """Served (P2-NEW-1's verification, F1): a tick keeps its own read of the
    GPU lock for `LOCK_READ_SECONDS`, the renders theirs. A CLI takes the GPU
    after the tick read the lock free; a dataset chosen then is drawn by a
    render whose own read is too old, so it reads the lock held: Start
    disabled, the GPU job in the reason line. The next tick's own read is
    still young, and older: the tick draws the render's, finds the page as it
    was sent and hands nothing over. When it drew its own, it handed a free
    lock over and its render enabled Start again without the GPU line while
    the lock was held. The tick's own read, when it is the newer, is still
    handed over (the CLI ends while nobody acts)."""
    _job, page = training_page
    lock = {"now": {"configured": True, "held_by": None}}
    monkeypatch.setattr(tc, "lock_state", lambda: lock["now"])
    clock = _clock(monkeypatch)
    page.pick("demo")                                    # the render reads the lock: free
    assert page.shown("shep-train-start", "interactive") is True
    clock.ahead += tc.LOCK_READ_SECONDS + 0.5
    assert page.tick() == []                             # the tick reads it again: free
    clock.ahead += 1
    lock["now"] = HELD_LOCK                              # a CLI takes the GPU
    page.pick("resampled")                               # its render reads it again: held
    assert page.shown("shep-train-start", "interactive") is False
    assert "GPU job" in page.shown("shep-train-reason")
    handed = page.tick()
    assert page.shown("shep-train-start", "interactive") is False, \
        "the tick's hand-off drew Start enabled from an older read of the lock"
    assert "GPU job" in page.shown("shep-train-reason")
    assert handed == []
    # Positive control: the CLI ends while nobody acts; the tick's own read,
    # newer than the render's once it is read again, is handed over.
    lock["now"] = {"configured": True, "held_by": None}
    clock.ahead += tc.LOCK_READ_SECONDS + 0.5
    assert "start" in page.tick()
    assert page.shown("shep-train-start", "interactive") is True
    assert "GPU job" not in page.shown("shep-train-reason")
    assert page.tick() == []


def test_a_tick_never_draws_an_older_run_list_than_the_render_after_a_choice(
        training_page, root, monkeypatch):
    """Served (F1, the run list): a tick keeps its own listing of the
    workspace's runs for `LATEST_READ_SECONDS`, the renders theirs. A run
    written elsewhere (the CLI) after the tick listed them is drawn by the
    render after a choice, whose own listing is too old; the next tick draws
    the render's listing and hands nothing over. When it drew its own, its
    hand-off drew the previous run back. The tick's own listing, when it is
    the newer, is still handed over (a run written while nobody acts)."""
    _job, page = training_page
    clock = _clock(monkeypatch)
    page.pick("demo")                                    # the selection's latest run
    assert page.shown("shep-train-charts")["run"] == "demo/gat/20261018-142000_run_d41f8b"
    clock.ahead += tc.LATEST_READ_SECONDS + 0.5
    assert page.tick() == []                             # the tick lists the runs again
    clock.ahead += 1
    new = root / "demo" / "runs" / "gat" / "20261023-120000_run_cccccc"
    (new / "checkpoints").mkdir(parents=True)
    (new / "checkpoints" / "model-1-0.0100.pt").write_bytes(b"x")
    (new / "logs").mkdir()
    (new / "logs" / "train_20261023_120000.json").write_text(json.dumps(_metrics(1)))
    page.pick("demo")                                    # its render lists them again
    assert page.shown("shep-train-charts")["run"] == "demo/gat/20261023-120000_run_cccccc"
    handed = page.tick()
    assert page.shown("shep-train-charts")["run"] == "demo/gat/20261023-120000_run_cccccc", \
        "the tick's hand-off drew the previous run from an older listing"
    assert handed == []
    # Positive control: another run written while nobody acts; the tick's own
    # listing, newer than the render's once it lists again, is handed over.
    newest = root / "demo" / "runs" / "gat" / "20261023-130000_run_eeeeee"
    (newest / "checkpoints").mkdir(parents=True)
    (newest / "checkpoints" / "model-1-0.0100.pt").write_bytes(b"x")
    clock.ahead += tc.LATEST_READ_SECONDS + 0.5
    assert "charts" in page.tick()
    assert page.shown("shep-train-charts")["run"] == "demo/gat/20261023-130000_run_eeeeee"
    assert page.tick() == []


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
    assert "dataset not shown · operator control" in pieces["head"]
    assert "last.pt" not in pieces["snap"] and "runs/" not in pieces["snap"]
    assert "last.pt" not in pieces["head"]


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


#: A run refused at its top, as train_model.py prints it (logging, then the traceback).
_REFUSED_TAIL = (
    "Process exited with code 1\n\n--- Last output ---\n"
    "2026-10-21 22:01:22,684 - __main__ - INFO - Configuration: {'run_dir': '/srv/secret/r'}\n"
    "2026-10-21 22:01:22,690 - __main__ - ERROR - Training failed: --device auto requires CUDA\n"
    "Traceback (most recent call last):\n"
    '  File "/srv/secret/scripts/train_model.py", line 917, in resolve_training_device\n'
    "    raise TrainingDeviceUnavailable(\n"
    "TrainingDeviceUnavailable: --device auto requires CUDA, which is not available here. "
    "Pass --device cpu explicitly.")


def test_the_last_error_line_is_the_last_error_the_run_printed():
    lines = _REFUSED_TAIL.partition("--- Last output ---\n")[2].splitlines()
    assert tc.last_error_line(lines) == (
        "TrainingDeviceUnavailable: --device auto requires CUDA, which is not available here. "
        "Pass --device cpu explicitly.")
    # A logging line at ERROR or CRITICAL gives its message; a later INFO line
    # does not replace it; nothing is guessed from other lines.
    logged = ["2026-10-21 10:00:00,001 - __main__ - ERROR - Training data not found in w",
              "2026-10-21 10:00:00,002 - __main__ - INFO - done"]
    assert tc.last_error_line(logged) == "Training data not found in w"
    assert tc.last_error_line(["CRITICAL:root:out of disk"]) == "out of disk"
    assert tc.last_error_line(["epoch 1: loss 0.5", "Note: nothing here", ""]) is None
    # A traceback whose header fell out of the tail still ends in its exception.
    assert tc.last_error_line(["    loss.backward()",
                               "torch.OutOfMemoryError: CUDA out of memory."]) == \
        "torch.OutOfMemoryError: CUDA out of memory."


def test_a_failed_runs_banner_says_its_cause_and_where_its_log_is_only_when_it_exists(tmp_path):
    """M5-3: the headline said only "Process exited with code 1", the cause
    was in the tail under a scroll, and "full log in its logs/" was said of a
    run directory that was never created."""
    missing = tmp_path / "runs" / "gat" / "never_made"
    html = tc._render_error_block(_REFUSED_TAIL, run="gat/never_made", run_dir=missing)
    cause = html.index('<div class="shep-banner-cause">')
    assert cause < html.index("<details")                    # right under the headline
    assert ('Last error printed:</span> <span class="shep-banner-v">TrainingDeviceUnavailable: '
            '--device auto requires CUDA') in html
    assert "no run directory, so no log" in html and "full log" not in html
    run = tmp_path / "runs" / "gat" / "made"
    (run / "logs").mkdir(parents=True)
    assert "no log in its <code>logs/</code>" in tc._render_error_block(
        _REFUSED_TAIL, run="gat/made", run_dir=run)
    (run / "logs" / "train_20261021_220122.json").write_text("{}")
    assert "its log: <code>logs/train_20261021_220122.json</code>" in tc._render_error_block(
        _REFUSED_TAIL, run="gat/made", run_dir=run)
    # Without a cause line there is no cause block.
    assert "shep-banner-cause" not in tc._render_error_block("Process exited with code 1")


def test_the_cause_is_redacted_for_a_read_only_request_as_the_tail_is(root, monkeypatch):
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    message = ("Process exited with code 1\n\n--- Last output ---\n"
               "Traceback (most recent call last):\n"
               '  File "/srv/secret/x.py", line 1, in <module>\n'
               "FileNotFoundError: /srv/secret/place/kg.json")
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: {"status": "failed", "error_message": message,
                                 "gpu": {"job": None}})
    banner = tc.render(tc._workspace_state("demo"), {}, {}, [], READ_ONLY)["pieces"]["banner"]
    cause = banner[banner.index("shep-banner-cause"):banner.index("<details")]
    assert "FileNotFoundError:" in cause and "/srv/secret" not in banner
    assert "a path outside the workspaces root" in cause
    # The markup stays whole: each text is redacted before it is escaped (the
    # built markup redacted whole lost its closing tags as "paths").
    assert banner.count("</span>") == banner.count("<span") and "</details></div>" in banner
    assert "<a path" not in banner and "× Dismiss</button>" in banner


def test_a_run_with_no_run_directory_is_not_called_no_longer_listed(root):
    """M5-3: the run view said "This run is no longer listed" of a run that
    never made its directory (refused before it made its run directory)."""
    run = tc.RunRef(id="demo/gat/never", label="gat/never",
                    path=root / "demo" / "runs" / "gat" / "never", workspace="demo", latest=False)
    ended = tc._chart_token(run, {}, live=False)
    assert ended == {"run": None, "rev": 0, "empty": tc.NO_RUN_DIR_ENDED}
    assert "no longer listed" not in tc.NO_RUN_DIR_ENDED and "no run directory" in tc.NO_RUN_DIR_ENDED
    assert tc._chart_token(run, {}, live=True) == {"run": None, "rev": 0,
                                                   "empty": tc.NO_RUN_DIR_LIVE}
    failed = {"status": "failed"}
    for operator in (True, False):
        snap = tc.run_snapshot_html(run, failed, operator=operator, snapshot=None)
        assert "Failed. No run directory exists for this run, so it has no checkpoint and no log." in snap
        assert "logs/" not in snap
    # Running before the run made its directory: no logs pointed to yet, and
    # the header's Details says no checkpoint is written yet.
    live = tc.run_snapshot_html(run, {"status": "running"}, operator=True, snapshot=None)
    assert live == '<p class="shep-rv-snap">Running</p>'
    head = tc.run_head_html(run, {}, operator=True, snapshot=None, live=True)
    assert "Newest checkpoint" in head and "none written yet" in head


def test_an_ended_run_with_no_checkpoint_is_headed_as_its_snapshot_says(root, monkeypatch,
                                                                       operator):
    """M5-3 review: a run that ended with no checkpoint (Device auto refused on
    a host without CUDA) was headed "used dataset fingerprint: recorded at the
    first checkpoint" over "… so it has no checkpoint and no log." (no run
    directory) or "No checkpoint was saved…" (a directory, no last.pt)."""
    ws = {"row": {"dataset_fingerprint": "a" * 64}, "digits": 12}
    run = tc.RunRef(id="demo/gat/r", label="gat/r", path=root / "demo" / "runs" / "gat" / "r",
                    workspace="demo", latest=False)
    ended = tc.run_head_html(run, ws, operator=True, snapshot=None)
    assert "dataset not recorded (no checkpoint)" in ended
    assert "first checkpoint" not in ended
    assert "recorded at the first checkpoint" in tc.run_head_html(run, ws, operator=True,
                                                                  snapshot=None, live=True)
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    for made in (False, True):
        job_run = root / "demo" / "runs" / "gat" / f"20261022-011234_auto_{int(made)}"
        if made:
            (job_run / "logs").mkdir(parents=True)
        monkeypatch.setattr(tc.training_manager, "run_dir", job_run)
        for status, live in (("failed", False), ("running", True)):
            monkeypatch.setattr(tc.training_manager, "get_status", lambda status=status: {
                "status": status, "job_id": "j", "error_message": _REFUSED_TAIL,
                "gpu": {"job": None}})
            pieces = tc.render(tc._workspace_state("demo"), {}, {}, [], operator)["pieces"]
            if live:
                assert "recorded at the first checkpoint" in pieces["head"]
                assert "none written yet" in pieces["head"]          # its Details
            else:
                assert "dataset not recorded (no checkpoint)" in pieces["head"]
                assert ("it has no checkpoint" in pieces["snap"] if not made
                        else "No checkpoint saved, so nothing can be continued" in pieces["snap"])


def test_a_read_only_banner_keeps_the_tails_indentation(root, monkeypatch):
    """M5-3 review: the read-only redaction joined each line's whitespace, so
    the traceback's lines lost their indent and its ``^^^`` markers no longer
    sat under the code; and the tail was drawn in the page's proportional
    font (the app's "* { font-family … !important }" won over its own)."""
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    message = ("Process exited with code 1\n\n--- Last output ---\n"
               "Traceback (most recent call last):\n"
               '  File "/srv/secret/train.py", line 949, in train\n'
               "    device = resolve(config.device)\n"
               "             ^^^^^^^^^^^^^^^^^^^^^^\n"
               "TrainingDeviceUnavailable: --device auto requires CUDA")
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: {"status": "failed", "error_message": message,
                                 "gpu": {"job": None}})
    banner = tc.render(tc._workspace_state("demo"), {}, {}, [], READ_ONLY)["pieces"]["banner"]
    tail = banner[banner.index('<pre class="shep-banner-tail">'):banner.index("</pre>")]
    assert "/srv/secret" not in banner
    assert '\n  File &quot;a path outside the workspaces root&quot;, line 949' in tail
    assert "\n    device = resolve(config.device)\n             ^^^^" in tail
    scrub = tc.read_only_scrub(None)
    assert scrub("\t  x = /srv/a/b.py  y") == "\t  x = a path outside the workspaces root y"
    assert scrub("   ") == "   " and scrub("") == ""
    tail_css = tc.PAGE_CSS[tc.PAGE_CSS.index(".shep-banner-tail {"):]
    assert tail_css[:tail_css.index("}")].count("font-family: var(--shep-mono) !important") == 1


def test_the_banner_escapes_the_output():
    html, _key = tc._build_error_banner_payload(
        {"error_message": "Process exited\n\n--- Last output ---\n<script>alert(1)</script>"},
        "<img src=x>")
    assert "<script>" not in html and "&lt;script&gt;" in html and "<img" not in html


# --------------------------------------------------------------------------- M5-2 review fixes
#: Markup a workspace or run name can hold; never sent to the browser as a tag.
TAG = "<img src=x onerror=window.__xss=1>"
_FORM = [""] + [100, 1e-4, "32", "gat", "cpu", 42, "256", 4, 0.1, 0.01, "cosine", 500, 0.01,
                10, 1.0, 0.5, 0.3, 0.2, 1, 1.0, "8", True, "float16", 0.07, 0.1, 1.0,
                "15, 10, 5", 5000]


def _no_tag(html):
    return "<img" not in html and "<b>odd" not in html


def test_the_notice_escapes_the_server_text_it_carries(root, monkeypatch, operator):
    """The run bar's notice is HTML: a workspace name, a path or the runner's
    error in it is escaped (`notice_html`), for Start, Stop and Continue
    Training alike."""
    assert tc.notice_html("") == "" and tc.notice_html(None) == ""
    assert tc.notice_html(TAG) == f'<span class="shep-notice-t">{tc.r.e(TAG)}</span>'
    # Start: the runner's refusal, and a workspace that is not trainable.
    monkeypatch.setattr(tc.training_manager, "start_training",
                        lambda config, **k: {"success": False, "error": f"{TAG} refused"})
    notice, _seed = tc._on_start(operator, tc._workspace_state("demo"), *_FORM, False)
    assert _no_tag(notice) and "&lt;img src=x" in notice and "refused" in notice
    odd = tc._workspace_state(ODD)
    odd["row"] = {**odd["row"], "trainable": False, "not_trainable": "x", "manifest": "read"}
    notice, _seed = tc._on_start(operator, odd, *_FORM, False)
    assert _no_tag(notice) and "&lt;b&gt;odd&amp;name is not trainable" in notice
    # Stop: the snapshot sentence names the checkpoint inside the workspace.
    last = root / ODD / "runs" / "gat" / "r" / "checkpoints" / "last.pt"
    monkeypatch.setattr(tc.training_manager, "stop_training", lambda: {
        "success": True, "outcome": {"checkpoint": str(last), "readable": True,
                                     "completed_epochs": 3, "target_epochs": 10, "end": None}})
    notice, _mode = tc._on_stop(operator, {})
    assert _no_tag(notice) and "&lt;b&gt;odd&amp;name/runs/gat/r/checkpoints/last.pt" in notice
    # Continue Training: a listed parent the server no longer lists, in that workspace.
    fp = odd["row"]["dataset_fingerprint"]
    parents = {"gat/r/last.pt": {"workspace": ODD, "architecture": "gat", "run": "r",
                                 "filename": "last.pt", "sha256": "c" * 64,
                                 "dataset_fingerprint": fp}}
    notice, _text, _extend, _mode = tc._on_resume(
        tc._workspace_state(ODD), "cpu", tc.CONTINUE_FROM_LIST, "gat/r/last.pt", "", None,
        parents, {}, operator)
    assert "no longer listed" in notice and "&lt;b&gt;odd&amp;name" in notice and _no_tag(notice)


def test_the_run_view_and_drawer_builders_escape_names_and_paths(root):
    """The run header (workspace, run label), the drawer's note, the dataset
    comparison and the check's checkpoint path, each with markup in a name."""
    run = tc.RunRef(id=f"{ODD}/gat/{TAG}", label=f"gat/{TAG}", path=Path("/x"), workspace=ODD,
                    latest=True)
    ws = {"name": ODD, "row": {"name": ODD, "dataset_fingerprint": "a" * 64}, "digits": 12,
          "listed": [{"name": ODD, "dataset_fingerprint": "a" * 64}]}
    head = tc.run_head_html(run, ws, operator=True, snapshot={"readable": True,
                                                               "dataset_fingerprint": "a" * 64})
    assert _no_tag(head) and "&lt;img src=x" in head
    # The run's workspace is named in Details when it is not the run bar's.
    elsewhere = tc.run_head_html(run, {**ws, "name": "demo"}, operator=True,
                                 snapshot={"readable": True, "dataset_fingerprint": "a" * 64})
    assert _no_tag(elsewhere) and "Workspace" in elsewhere and "&lt;b&gt;odd&amp;name" in elsewhere
    assert "Workspace" not in head
    note = tc._parents_note(ws, {"k": {"architecture": "gat", "run": TAG}}, None)
    assert _no_tag(note) and "1 run, 1 checkpoint" in note
    for fingerprint in ("a" * 64, "e" * 64):          # the same dataset, and another
        compare = tc.compare_html({"chosen": True, "fingerprint": fingerprint,
                                   "source": tc.CONTINUE_FROM_LIST}, ws)
        assert _no_tag(compare), fingerprint
    assert "&lt;b&gt;odd&amp;name" in tc.compare_html(
        {"chosen": True, "fingerprint": "e" * 64, "source": tc.CONTINUE_FROM_LIST}, ws)
    preview = {"parent": str(root / ODD / "runs" / "gat" / TAG / "last.pt"), "ok": False,
               "reason": f"refused: {TAG}", "sha256": "c" * 64}
    check = tc._check_html(preview, tc.CONTINUE_FROM_LIST)
    assert _no_tag(check) and "&lt;b&gt;odd&amp;name/runs/gat/&lt;img" in check
    listed = tc._check_html(preview, tc.CONTINUE_FROM_LIST, listed=f"gat/{TAG}/last.pt")
    assert _no_tag(listed) and "<code>gat/&lt;img" in listed


def _rendered_for(status, mode, operator, monkeypatch, ws):
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: status)
    return tc.render(ws, mode, {}, [], operator)["pieces"]


def test_a_stop_confirmation_belongs_to_the_job_it_was_asked_for(root, monkeypatch, operator):
    """Asked for job a1 and left unanswered when a1 completes: when b2 runs,
    the run bar shows no confirmation and Stop is enabled; a "Stop training"
    sent for a1 stops nothing while b2 runs."""
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    monkeypatch.setattr(tc.training_manager, "run_dir", None)
    ws = tc._workspace_state("demo")
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: {"status": "running", "job_id": "a1"})
    monkeypatch.setattr(tc.training_manager, "latest_snapshot", lambda *a: None)
    _text, mode = tc._on_stop_request(operator, {})
    assert mode["confirm"] == "a1"
    asked = _rendered_for({"status": "running", "job_id": "a1"}, mode, operator, monkeypatch, ws)
    assert "shep-rb-confirming" in asked["runbar"] and asked["stop"] is False
    done = _rendered_for({"status": "completed", "job_id": "a1"}, mode, operator, monkeypatch, ws)
    assert "shep-rb-confirming" not in done["runbar"]
    other = _rendered_for({"status": "running", "job_id": "b2"}, mode, operator, monkeypatch, ws)
    assert "shep-rb-confirming" not in other["runbar"] and other["stop"] is True
    stops = []
    monkeypatch.setattr(tc.training_manager, "stop_training",
                        lambda: stops.append(1) or {"success": True, "outcome": None})
    notice, closed = tc._on_stop(operator, mode)                       # b2 runs now
    assert stops == [] and "Not stopped" in notice and closed["confirm"] is None
    tc._on_stop(operator, {"confirm": "b2"})                          # asked for b2
    tc._on_stop(operator, {})                                         # a direct call
    assert stops == [1, 1]


def test_keep_training_only_closes_the_question():
    """Keep training runs `_on_stop_cancel` and then redraws the view; no
    step of its chain is the Stop."""
    demo, _ctx, by_id = _page()
    keep = next(fn for fn in demo.fns.values()
                if (by_id["shep-train-stop-keep"]._id, "click") in fn.targets)
    assert keep.fn is tc._on_stop_cancel
    chain, step = [], keep
    while (step := next((fn for fn in demo.fns.values() if fn.trigger_after == step._id),
                        None)) is not None:
        chain.append(step.fn)
    assert chain and tc._on_stop not in chain and tc._on_stop_request not in chain


def test_start_and_continue_close_a_confirmation_left_from_an_earlier_job():
    demo, _ctx, by_id = _page()
    start = next(fn for fn in demo.fns.values()
                 if (by_id["shep-train-start"]._id, "click") in fn.targets)
    after = next(fn for fn in demo.fns.values() if fn.trigger_after == start._id)
    assert after.fn is tc._on_stop_cancel
    assert tc._on_stop_cancel({"confirm": "a1", "drawer": True}) == {"confirm": None,
                                                                      "drawer": True}


def _listing(items, problem=None):
    from src.api.services import server_lists

    return server_lists.Listing(tuple(items), problem)


def _item(workspace, run="r", filename="last.pt", kind="training_state", epoch=3):
    from src.api.services.server_lists import CheckpointItem

    return CheckpointItem(workspace=workspace, architecture="gat", run=run, filename=filename,
                          sha256="c" * 64, size_bytes=1, checkpoint_kind=kind, epoch=epoch,
                          validation=None, dataset_fingerprint=None,
                          trained_on_this_dataset=None, problem=None)


def test_the_parent_list_follows_the_run_bars_workspace_while_the_drawer_is_open(
        root, monkeypatch, operator):
    """After the run bar's workspace changed with Continue… open, the list is
    the new workspace's runs (D8 first; nothing listed with the drawer
    closed), and the old check, comparison and Extra epochs are cleared."""
    from src.api.services import server_lists

    listed = []
    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda name: listed.append(name) or _listing([_item(name)]))
    ws = tc._workspace_state("resampled")
    assert tc._relist_parents(operator, ws, {"drawer": False}) == (gr.update(),) * 7
    assert listed == []
    parents, runs, select, note, compare, preview, extend = tc._relist_parents(
        operator, ws, {"drawer": True})
    assert listed == ["resampled"] and list(parents) == ["gat/r/last.pt"]
    assert parents["gat/r/last.pt"]["workspace"] == "resampled"
    assert runs["value"] == "gat/r" and select["value"] == "gat/r/last.pt"
    assert "1 run, 1 checkpoint" in note
    assert compare == "" and tc._CONTINUE_PROMPT in preview
    assert extend == gr.update(visible=False, value=None)
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)    # read-only from here
    refused = tc._relist_parents(READ_ONLY, ws, {"drawer": True})
    assert refused[0] == {} and refused[1] == refused[2] == gr.update(choices=[], value=None)
    assert "Refused" in refused[3] and listed == ["resampled"]


def test_a_parent_listed_for_another_workspace_is_never_continued_on_this_one(root):
    """A list read for demo, the run bar now on resampled (a running job's
    dataset, or a change the list did not follow): the reason says so, and
    nothing is resolved in resampled under demo's names."""
    item = {**_item("demo").__dict__}
    parents = {"gat/r/last.pt": item}
    ws = tc._workspace_state("resampled")
    chosen = tc._chosen(tc.CONTINUE_FROM_LIST, "gat/r/last.pt", "", parents, ws)
    assert chosen["workspace"] == "demo"
    why = tc.continue_refusal(chosen, ws)
    assert why.startswith("the run list shows demo’s runs, not resampled’s; refresh the list")
    path, why, _item_ = tc._parent_path(tc.CONTINUE_FROM_LIST, "gat/r/last.pt", "", ws, parents)
    assert path is None and "the run list shows demo’s runs" in why
    assert tc.continue_refusal(chosen, tc._workspace_state("demo")) is None


def test_a_workspace_change_relists_through_the_gated_path_then_reads_the_chips():
    demo, ctx, _by_id = _page()
    fns = list(demo.fns.values())
    after = {fn.trigger_after: fn for fn in fns if fn.trigger_after is not None}
    first = next(fn for fn in fns if (ctx.training_workspace._id, "change") in fn.targets)
    chain, current = [], first
    while current is not None:
        chain.append(current.fn.__name__)
        current = after.get(current._id)
    assert chain == ["_on_training_workspace", "_relist_parents", "_on_parent_chosen",
                     "_render_view", "read_chips"]
    import inspect

    assert "operator_access(request)" in inspect.getsource(tc._relist_parents)


def test_the_run_view_lists_the_workspaces_runs_again_while_the_page_stays_open(
        root, monkeypatch):
    """A run started elsewhere (the CLI) is shown once it writes a checkpoint:
    the latest run is listed again every `LATEST_READ_SECONDS`, not on every
    tick, and not only when the page is opened."""
    import os
    import time

    ws = tc._workspace_state("demo")
    cache = {}
    old = tc._current_run({"status": "idle"}, ws, cache)
    assert old["id"] == "demo/gat/20261018-142000_run_d41f8b"
    new = root / "demo" / "runs" / "gat" / "20261021-150000_run_dddddd"
    (new / "checkpoints").mkdir(parents=True)
    (new / "checkpoints" / "model-1-0.0100.pt").write_bytes(b"x")
    later = time.time() + 5
    os.utime(new / "checkpoints" / "model-1-0.0100.pt", (later, later))
    listings = []
    original = tc._latest_run
    monkeypatch.setattr(tc, "_latest_run", lambda d: listings.append(d) or original(d))
    assert tc._current_run({"status": "idle"}, ws, cache)["id"] == old["id"]   # within the interval
    assert listings == []
    cache["latest"][1] -= tc.LATEST_READ_SECONDS                                 # the interval passed
    run = tc._current_run({"status": "idle"}, ws, cache)
    assert run["id"] == "demo/gat/20261021-150000_run_dddddd" and run["latest"] is True
    assert len(listings) == 1


def test_a_live_run_reads_its_checkpoint_once_for_the_fingerprint(root, monkeypatch, operator):
    """While the job runner's run is live, ``last.pt`` (rewritten each epoch,
    read whole) is read once per run and session for the recorded dataset
    fingerprint, which stays shown; after the run ends the snapshot is read."""
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    job_run = root / "demo" / "runs" / "gat" / "20261020-090000_run_live01"
    (job_run / "checkpoints").mkdir(parents=True)
    last = job_run / "checkpoints" / "last.pt"
    last.write_bytes(b"epoch 1")
    monkeypatch.setattr(tc.training_manager, "run_dir", job_run)
    ws = tc._workspace_state("demo")
    fp = ws["row"]["dataset_fingerprint"]
    reads = []
    monkeypatch.setattr(tc.training_manager, "latest_snapshot", lambda run_dir=None: reads.append(
        run_dir) or {"readable": True, "dataset_fingerprint": fp, "completed_epochs": len(reads),
                     "checkpoint": str(last)})
    status = {"status": "running", "job_id": "j", "current_epoch": 2, "total_epochs": 9}
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: status)
    cache = {}
    for epoch in range(2, 6):
        last.write_bytes(b"epoch %d" % epoch * epoch)                  # a new size each epoch
        rendered = tc.render(ws, {}, cache, [], operator)
        cache = rendered["cache"]
        assert fp[:12] in rendered["pieces"]["head"]
        assert "same dataset as the run bar" in rendered["pieces"]["head"]
    assert len(reads) == 1
    status = {"status": "completed", "job_id": "j", "current_epoch": 9, "total_epochs": 9}
    tc.render(ws, {}, cache, [], operator)
    assert len(reads) == 2


def test_the_reason_lines_sentence_ends_once():
    test_run = _view(gpu={"text": "GPU job: test run (mygene2-2026-09 · 00:00:25)", "test": True})
    text = re.sub(r"<[^>]+>", "", test_run["reason"])
    assert text.endswith("(mygene2-2026-09 · 00:00:25) · Open Test ›") and ")." not in text
    cli = _view(gpu={"text": "GPU job: train_model.py (pid 41822, since 09:58 UTC)",
                     "test": False})
    assert re.sub(r"<[^>]+>", "", cli["reason"]).endswith("since 09:58 UTC).")


def test_a_checkpoints_option_says_what_it_is_how_far_it_trained_and_its_limit():
    """In a few words, from the file's own record (the GB10 WebUI recheck: one
    long line per file read as a wall of text): the latest save or a top-k
    save, the epochs its progress record says it completed (never epoch + 1
    for a file without one), what limits continuing, and the Trainer's
    validation figure, named as the Trainer's."""
    item = {**_item("demo").__dict__, "completed_epochs": 195, "end": "paused",
            "validation": {"metric": "val_mrr", "value": 0.28}}
    assert tc.checkpoint_words(item) == (
        "Latest save · 195 epochs completed · Validation MRR (Trainer, top-20) 0.2800")
    top = {**item, "filename": "model-20-0.2867.pt", "completed_epochs": 21, "end": None}
    assert tc.checkpoint_words(top).startswith("Top-k save · 21 epochs completed")
    assert tc.checkpoint_words({**item, "end": "finished"}).startswith(
        "Latest save · 195 epochs completed · finished")
    assert "stopped early, cannot continue" in tc.checkpoint_words({**item, "end": "stopped_early"})
    weights = _item("demo", filename="model-03-0.2.pt", kind="weights").__dict__
    assert tc.checkpoint_words(weights) == "Top-k save · weights only, cannot continue"
    older = {**_item("demo", epoch=194).__dict__}                     # no progress record
    assert tc.checkpoint_words(older) == "Latest save · recorded epoch 194"
    assert tc.checkpoint_words({**older, "problem": "ValueError: x"}) == "last.pt · cannot be read"
    other = {**item, "filename": "best.pt", "completed_epochs": 1}
    assert tc.checkpoint_words(other).startswith("best.pt · 1 epoch completed")


def test_a_run_reads_as_its_name_architecture_and_start_time():
    assert tc.run_words("gat", "20261024-173909_ui-live-stop-02_c1d9cf") == \
        "ui-live-stop-02 · GAT · 2026-10-24 17:39"
    assert tc.run_words("gat", "20261024-163856_run_559817") == "GAT · 2026-10-24 16:38"
    assert tc.run_words("hgt", "my-old-run") == "HGT · my-old-run"


def test_the_drawer_lists_runs_then_the_chosen_runs_checkpoints():
    """Two steps over the same server list (the same keys, digests and check):
    the runs, latest first; the chosen run's files, newest first. Choosing
    another run offers its files with its latest save chosen; a checkpoint
    kept by key stays chosen; two runs that would read alike are told apart
    by their directory's name."""
    def item(run, filename, completed, mrr=None):
        return {**_item("demo", run=run, filename=filename).__dict__, "completed_epochs": completed,
                "validation": {"metric": "val_mrr", "value": mrr} if mrr else None}
    new, old = "20261024-173909_a_c1d9cf", "20261024-163856_run_559817"
    parents = {f"gat/{new}/last.pt": item(new, "last.pt", 195),
               f"gat/{new}/model-20-0.2867.pt": item(new, "model-20-0.2867.pt", 21, 0.2867),
               f"gat/{old}/last.pt": item(old, "last.pt", 11),
               f"gat/{old}/model-02-0.2160.pt": item(old, "model-02-0.2160.pt", 3, 0.216),
               f"gat/{old}/model-01-0.2160.pt": item(old, "model-01-0.2160.pt", 2, 0.216)}
    runs, files = tc._parents_update(parents)
    assert runs["choices"] == [("a · GAT · 2026-10-24 17:39", f"gat/{new}"),
                               ("GAT · 2026-10-24 16:38", f"gat/{old}")]
    assert runs["value"] == f"gat/{new}" and files["value"] == f"gat/{new}/last.pt"
    assert [label for label, _key in files["choices"]] == [
        "Latest save · 195 epochs completed",
        "Top-k save · 21 epochs completed · Validation MRR (Trainer, top-20) 0.2867"]
    others = tc._on_run_chosen(f"gat/{old}", parents, f"gat/{new}/last.pt")
    assert others["value"] == f"gat/{old}/last.pt"                       # its latest save
    assert [key for _label, key in others["choices"]] == [k for k in parents if old in k]
    labels = [label for label, _key in others["choices"]]
    assert len(set(labels)) == len(labels)                                # same MRR, told apart
    kept = tc._on_run_chosen(f"gat/{old}", parents, f"gat/{old}/model-01-0.2160.pt")
    assert kept["value"] == f"gat/{old}/model-01-0.2160.pt"
    assert tc._parents_update(parents, f"gat/{old}/model-02-0.2160.pt")[0]["value"] == f"gat/{old}"
    twins = {"gat/20261024-163856_run_111111/last.pt": item("20261024-163856_run_111111", "last.pt", 1),
             "gat/20261024-163801_run_222222/last.pt": item("20261024-163801_run_222222", "last.pt", 1)}
    assert [label for label, _key in tc._run_choices(twins)] == [
        "GAT · 2026-10-24 16:38 · 20261024-163856_run_111111",
        "GAT · 2026-10-24 16:38 · 20261024-163801_run_222222"]
    assert tc._on_run_chosen(None, {}, None) == gr.update(choices=[], value=None)


def test_the_chosen_checkpoints_details_say_what_its_option_does_not():
    """A table: the file, what it holds, what continuing does when its stage
    ended, its model fingerprint (copyable) and its run's directory; its
    progress and validation figure are its option's, not repeated."""
    item = {**_item("demo", run="20261024-163856_run_559817").__dict__, "completed_epochs": 11,
            "end": "finished", "validation": {"metric": "val_mrr", "value": 0.214}}
    html = tc.parent_details_html(item)
    for words in ("File", "last.pt", "Holds", "training state", "Continuing",
                  "extends it by the Extra epochs you enter", "Model fingerprint", "cccccccccccc",
                  'data-copy="' + "c" * 64, "gat/20261024-163856_run_559817"):
        assert words in html, words
    assert "epochs completed" not in html and "MRR" not in html
    weights = tc.parent_details_html(_item("demo", kind="weights").__dict__)
    assert "weights only (cannot be continued)" in weights and "Continuing" not in weights
    unread = tc.parent_details_html({**_item("demo").__dict__, "problem": "ValueError: <b>x"})
    assert "shep-ckpt-amber" in unread and "Cannot be read" in unread and "&lt;b&gt;x" in unread
    assert tc.parent_details_html(None) == ""


def test_the_device_help_says_auto_needs_cuda_as_the_trainer_does():
    """train_model.py refuses --device auto without CUDA (contract M3a-1a); the
    help said it fell back to the CPU, so the default could start a run the
    trainer refuses without the page saying why (the WebUI audit, 2026-10-25)."""
    for key in ("device", "continue_device"):
        text = tc.FIELD_HELP[key][1]
        assert "CUDA" in text and "refused" in text, key
        assert "otherwise CPU" not in text and "otherwise the CPU" not in text, key


def test_the_chart_help_leads_with_the_gestures_in_paragraphs():
    title, text = tc.TOPIC_HELP["training-charts"]
    assert text.startswith(tc.charts.GESTURES + "\n\n") and "\n• over the epoch axis" in text
    for key in ("continue_source", "checkpoint_dropdown", "continue_device", "run_name",
                "training-used-dataset", "continue_run"):
        help_text = {**tc.FIELD_HELP, **tc.TOPIC_HELP}[key][1]
        assert "\n\n" in help_text, key                      # a lead, then the rest


def test_continue_has_its_own_device_which_its_check_and_start_use():
    demo, _ctx, by_id = _page()
    device, own = by_id["device"], by_id["continue_device"]
    for name in ("_on_check_continue", "_on_resume"):
        fn = next(f for f in demo.fns.values() if f.fn is getattr(tc, name))
        assert own in fn.inputs and device not in fn.inputs, name
    opened = next(f for f in demo.fns.values()
                  if (by_id["shep-train-continue"]._id, "click") in f.targets)
    copy = next(f for f in demo.fns.values() if f.trigger_after == opened._id)
    assert copy.fn is tc._continue_device and copy.inputs == [device] and copy.outputs == [own]
    assert [tc._continue_device(v) for v in ("cpu", "cuda", "auto", None)] == [
        "cpu", "cuda", "auto", "auto"]
    assert "continue_device" in tc.FIELD_HELP


def test_a_failed_runs_output_tail_is_a_disclosure_fitted_to_the_window():
    html = tc._render_error_block("exit 1\n\n--- Last output ---\n" + "\n".join(
        f"line {i}" for i in range(20)), run="gat/r")
    assert '<details class="shep-banner-more" open><summary>Last output' in html
    assert f"({tc.BANNER_TAIL_LINES} lines)" in html and "line 19" in html and "line 7" not in html
    one = tc._render_error_block("exit 1\n\n--- Last output ---\nonly")
    assert "(1 line)" in one
    assert "<details" not in tc._render_error_block("exit 1")
    assert "window.innerHeight < 1000" in tc._BANNER_JS and "data-tail" in tc._BANNER_JS
    _demo, _ctx, by_id = _page()
    assert by_id["shep-train-banner"].js_on_load == tc._BANNER_JS


def _rules(css, needle):
    """The CSS rules whose selector names `needle`, as (selector, body)."""
    return [(sel.strip(), body) for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)
            if needle in sel]


def test_the_page_styles_hold_the_reviewed_layout():
    css = tc.PAGE_CSS
    # The build time wraps with the chip, never hidden.
    assert not [b for _s, b in _rules(css, "shep-ds-built") if "display: none" in b]
    # The selects: a border, the text clear of the chevron, grey while disabled.
    assert any("border: 1px solid" in b for _s, b in _rules(css, "#shep-train-dataset .wrap:has"))
    assert any("padding-right: 30px" in b for _s, b in _rules(css, "#checkpoint_dropdown input"))
    assert any("padding-right: 30px" in b for _s, b in _rules(css, "#continue_run input"))
    assert any("grey" in b for _s, b in _rules(css, "input:disabled"))
    # Disabled actions: neutral grey, full opacity.
    disabled = [b for _s, b in _rules(css, ".shep-rb-row > button:disabled")]
    assert disabled and "opacity: 1" in disabled[0] and "--shep-grey-50" in disabled[0]
    # The reason line's icon stays beside its sentence.
    why = [b for s, b in _rules(css, ".shep-rb-why") if s == f"{tc._PAGE} .shep-rb-why"]
    assert any("flex-wrap: nowrap" in b for b in why) and not any("flex-wrap: wrap" in b for b in why)
    # A failed run's tail: four lines at a time when wide, never an inner scroll when narrow.
    narrow = css[css.index("@media (max-width: 1399px), (max-height: 759px)"):]
    assert ".shep-banner-tail {{ max-height: none; overflow: visible; }}".replace(
        "{{", "{").replace("}}", "}") in narrow
    wide = css[css.index("@media (min-width: 1400px) and (min-height: 760px)"):]
    assert "#shep-train-runview > #shep-train-charts { flex: 1 1 0 !important; min-height: 0 !important; overflow-y: auto !important;" in wide
    # A failed run's banner takes height: the plots keep their minimum then too
    # (M5-3: one minimum per plot box replaced the charts' 240/370 px).
    assert "min-height: 370px" not in wide and "min-height: 240px" not in wide
    assert f".shep-chart-plot {{ min-height: {tc.PLOT_MIN_HEIGHT + tc.PLOT_AXIS_ROOM}px; }}" in wide



def test_the_continue_drawer_fits_and_says_the_plan_once():
    """M5-3: at 1920 x 1080 the drawer scrolled (821 px in 684) with Extra
    epochs off screen: an empty footer took room, the fields were looser than
    the parameters', and the check's headline repeated the plan line."""
    assert "#shep-train-drawer-why:not(:has(.shep-drawer-why)) { display: none !important; }" \
        in tc.PAGE_CSS
    assert "#shep-train-drawer-bd .block.padded { padding-top: 4px !important;" in tc.PAGE_CSS
    preview = {"parent": "/w/demo/runs/gat/r/checkpoints/last.pt", "ok": True,
               "action": "Extend training by 30 epochs at a fixed rate of 1e-05 (epochs 151-180, "
                         "new target 180)", "sha256": "c" * 64, "inherited": {}}
    check = tc._check_html(preview, tc.CONTINUE_FROM_LIST)
    assert check.count("Extend training by 30 epochs") == 1
    assert "Checkpoint read; it can continue on this workspace." in check


def test_extra_epochs_sits_above_every_check_result():
    """M5-3: on the first Check of a finished parent and when the parent's
    dataset differs, Extra epochs fell below the fold at 1920 x 1080 (under the
    result). It is beside Check checkpoint now, before the comparison and the
    result, and the drawer is denser (fits at 1920 x 1080 and 1920 x 940 in
    every result, measured in a browser)."""
    _demo, _ctx, by_id = _page()

    def row_of(block):
        while block is not None and "shep-train-check-row" not in (block.elem_classes or []):
            block = block.parent
        return block

    row = row_of(by_id["extend_epochs"])
    assert row is not None and row_of(by_id["shep-train-check"]) is row
    body = by_id["shep-train-drawer-bd"].children
    assert body.index(row) < body.index(by_id["shep-train-compare"]) < body.index(
        by_id["continue_preview"])
    wide = tc.PAGE_CSS[tc.PAGE_CSS.index("@media (min-width: 1400px) and (min-height: 760px) {"):]
    wide = wide.replace(f"{tc._PAGE} ", "")
    assert "#continue_source, #continue_device { display: flex !important;" in wide
    assert "#checkpoint_dropdown .container { display: flex;" not in wide   # labels above
    assert "#shep-train-drawer-why { padding: 8px 16px !important; }" in wide
    # When the datasets differ, the comparison says the start refuses; the
    # check does not add its general line, and a rate is never split.
    preview = {"parent": "/w/demo/runs/gat/r/checkpoints/last.pt", "ok": True,
               "action": "Extend training by 10 epochs at a fixed rate of 3e-05 (epochs 9-18, "
                         "new target 18)", "sha256": "c" * 64, "inherited": {}}
    differs = tc._check_html(preview, tc.CONTINUE_FROM_LIST, differs=True)
    assert "The start checks again" not in differs
    assert '<span class="shep-nowrap">3e-05</span>' in differs
    assert "The start checks again" in tc._check_html(preview, tc.CONTINUE_FROM_LIST)


def test_the_details_popover_holds_the_manifest_digests():
    """M5-3: at 460 px the second column's digests ended 2 px past the card's
    inner edge (1920 and 1536, both modes)."""
    assert tc.DETAILS_POPOVER_WIDTH >= 480
    assert f"width: min({tc.DETAILS_POPOVER_WIDTH}px, calc(100vw - 48px));" in tc.PAGE_CSS


def test_the_details_popover_is_drawn_over_the_parameter_labels():
    """M5-3 review: at 1280 x 720 and 1080 x 1920 the labels "Learning Rate",
    "Batch Size", "Device" and "Seed" (Gradio's z-index 40) were drawn on the
    popover, which sits in the chip (z-index 20)."""
    assert tc.CHIP_Z_INDEX > 40
    assert (f"#shep-train-chip {{ flex: 1 1 360px !important; position: relative; "
            f"z-index: {tc.CHIP_Z_INDEX}; }}") in tc.PAGE_CSS


def test_a_two_sentence_reason_breaks_between_its_sentences():
    """M5-3: in the Continue states the run bar's reason left one word alone
    on its second line ("…cannot be" / "continued."): each sentence is a word
    group, so the line breaks between them."""
    ws = {"row": {"name": "demo", "trainable": True, "dataset_fingerprint": "a" * 64}}
    refused = _view(ws=ws, mode={"drawer": True, "parent": {"chosen": True, "refused": True}})
    reason = refused["reason"]
    assert reason.count('class="shep-wg"') == 2
    assert ('<span class="shep-wg">Start disabled: close Continue… to start a new run.</span> '
            '<span class="shep-wg">Continue Training disabled: the check says this checkpoint '
            'cannot be continued.</span>') in reason
    assert ".shep-rb-why .shep-wg { text-wrap: pretty; }" in tc.PAGE_CSS.replace(f"{tc._PAGE} ", "")


def test_the_plots_keep_a_usable_minimum_and_the_resources_line_its_own_row():
    """M5-3: at 1536 x 760 the plots were 70 to 100 px tall; each plot box keeps
    room for a plot area of PLOT_MIN_HEIGHT px. The GB10 WebUI recheck: the
    resources line, kept over the scrolling plots (sticky), hid the last axis
    and its label. The run view is now three rows, and only the charts' row
    scrolls when their minimum does not fit: the header and the resources
    line are rows of their own, nothing is drawn over an axis, and no axis
    label is hidden to make room."""
    assert tc.PLOT_MIN_HEIGHT >= 140
    css = tc.PAGE_CSS
    wide = css[css.index("@media (min-width: 1400px) and (min-height: 760px) {"):]
    wide = wide[:wide.index("\n}\n")]
    plot_box = tc.PLOT_MIN_HEIGHT + tc.PLOT_AXIS_ROOM
    assert f"#shep-train-charts .shep-chart-plot {{ min-height: {plot_box}px; }}" in wide
    assert "#shep-train-charts .shep-charts-grid { grid-auto-rows: minmax(min-content, 1fr); }" in wide
    # Only !important beats the inline "overflow: visible" Gradio writes on the block.
    assert ("#shep-train-runview > #shep-train-charts { flex: 1 1 0 !important; min-height: 0 "
            "!important; overflow-y: auto !important;") in wide
    assert "#shep-train-runview > #resource_display { flex: none !important; }" in wide
    assert "position: sticky; bottom" not in css and "#shep-train-runview { overflow-y" not in css
    assert ".shep-charts-bar { position: sticky; top: 0;" in wide
    # The boxes inside the row take its height, the cards keep their content's minimum.
    assert "#shep-train-charts .shep-charts-grid { min-height: 0; }" in wide
    assert "#shep-train-charts .shep-chart { min-height: auto; }" in wide
    assert ".shep-chart-xl { display: none; }" not in css.replace(f"{tc._PAGE} ", "")
    _demo, _ctx, by_id = _page()
    assert by_id["shep-train-run-head"].js_on_load == tc._POPOVER_JS
    assert by_id["shep-train-chip"].js_on_load == tc._POPOVER_JS
    assert "--shep-rv-top-h" not in css
