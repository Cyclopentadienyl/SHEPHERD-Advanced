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
    assert len(out) == len(tc.VIEW_KEYS) + 2 and all(u == gr.update() for u in out)
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
    held = {"configured": True, "problem": None, "held": True, "holder_alive": True,
            "holder": {"command": "train_model.py", "pid": 41822,
                       "started_at": "2026-10-20T09:58:00+00:00"},
            "description": "train_model.py (pid 41822)"}
    cli = tc._gpu_line(None, held, None, operator=False)
    assert cli == {"text": "GPU job: train_model.py (pid 41822, since 09:58 UTC)",
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
        cache = out[-2]
        chip = dict(zip(tc.VIEW_KEYS, out[:-2]))["chip"]
        sent_whole.append(chip != gr.update())
    assert sent_whole[0] is True
    assert [i + 1 for i, whole in enumerate(sent_whole) if whole] == [
        1, tc.FULL_SEND_TICKS, tc.FULL_SEND_TICKS * 2]


def test_the_header_chips_follow_the_gpu_slot_on_the_tick_that_sees_it_change(
        root, monkeypatch, operator):
    """M5-3: the chip read "GPU jobs: none" while the run bar showed a test
    run (the chips on the slow timer, the run bar on the fast one). The fast
    tick that sees the GPU slot change reads the chips again for its request;
    a tick that sees no change sends nothing (no timer, no request of its own)."""
    from src.webui.components import chips

    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    monkeypatch.setattr(tc, "read_lock", lambda: {"configured": True, "held": False})
    read = []
    monkeypatch.setattr(chips, "read_chips", lambda request: read.append(1) or "<chips>")
    status = {"status": "idle"}
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: dict(status))
    monkeypatch.setattr(tc.training_manager, "get_test_status",
                        lambda: {"status": "running", "inputs": {"version_label": "v"}})
    demo, ctx, _by_id = _page()
    poll_fn = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "_poll")
    assert poll_fn.outputs[-1] is ctx.chips
    ws = tc._workspace_state("demo")
    first = tc._poll(shell.TRAINING, ws, {}, {}, [], operator)
    assert first[-1] == "<chips>" and len(read) == 1        # nothing seen before: read
    same = tc._poll(shell.TRAINING, ws, {}, first[-2], [], operator)
    assert same[-1] == gr.update() and len(read) == 1        # unchanged: nothing sent
    status["gpu"] = {"job": {"job_type": "test", "status": "running",
                             "start_time": "2026-10-21T18:25:13"}}
    started = tc._poll(shell.TRAINING, ws, {}, same[-2], [], operator)
    assert started[-1] == "<chips>" and len(read) == 2       # a test run took the slot
    later = tc._poll(shell.TRAINING, ws, {}, started[-2], [], operator)
    assert later[-1] == gr.update() and len(read) == 2       # its elapsed time is no change


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
    assert _no_tag(head) and "latest in &lt;b&gt;odd&amp;name" in head and "&lt;img src=x" in head
    note = tc._parents_note(ws, {"k": {}}, None)
    assert _no_tag(note) and "&lt;b&gt;odd&amp;name’s runs" in note
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
    assert tc._relist_parents(operator, ws, {"drawer": False}) == (gr.update(),) * 6
    assert listed == []
    parents, select, note, compare, preview, extend = tc._relist_parents(
        operator, ws, {"drawer": True})
    assert listed == ["resampled"] and list(parents) == ["gat/r/last.pt"]
    assert parents["gat/r/last.pt"]["workspace"] == "resampled"
    assert select["value"] == "gat/r/last.pt" and "resampled’s runs" in note
    assert compare == "" and tc._CONTINUE_PROMPT in preview
    assert extend == gr.update(visible=False, value=None)
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)    # read-only from here
    refused = tc._relist_parents(READ_ONLY, ws, {"drawer": True})
    assert refused[0] == {} and refused[1] == gr.update(choices=[], value=None)
    assert "Refused" in refused[2] and listed == ["resampled"]


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
        assert fp[:12] in rendered["pieces"]["head"] and "same as the run bar" in rendered["pieces"]["head"]
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


def test_a_parents_option_names_its_file_and_what_it_holds():
    last = tc._parent_label(_item("demo").__dict__)
    weights = tc._parent_label(_item("demo", filename="model-03-0.2.pt", kind="weights").__dict__)
    assert "epoch 3 · last.pt (training state) · model fingerprint cccccccccccc" in last
    assert "epoch 3 · model-03-0.2.pt (weights only)" in weights and last != weights
    unknown = tc._parent_label({**_item("demo").__dict__, "checkpoint_kind": None, "epoch": None})
    assert "· last.pt · model fingerprint" in unknown


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
    assert "#shep-train-runview { overflow-y: auto !important; }" in wide
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


def test_the_plots_keep_a_usable_minimum_with_the_toolbar_in_view():
    """M5-3: at 1536 x 760 the plots were 70 to 100 px tall. Each plot box keeps
    room for a plot area of PLOT_MIN_HEIGHT px; when that does not fit, the run
    view scrolls inside with its top (header and snapshot) and the charts'
    toolbar staying in view, and the resources line yielding below."""
    assert tc.PLOT_MIN_HEIGHT >= 140
    css = tc.PAGE_CSS
    wide = css[css.index("@media (min-width: 1400px) and (min-height: 760px) {"):]
    plot_box = tc.PLOT_MIN_HEIGHT + tc.PLOT_AXIS_ROOM
    assert f"#shep-train-charts .shep-chart-plot {{ min-height: {plot_box}px; }}" in wide
    assert "#shep-train-charts .shep-charts-grid { grid-auto-rows: minmax(min-content, 1fr); }" in wide
    assert "#shep-train-runview > #shep-train-rv-top { position: sticky; top: 0;" in wide
    assert ".shep-charts-bar { position: sticky; top: var(--shep-rv-top-h" in wide
    _demo, _ctx, by_id = _page()
    assert by_id["shep-train-run-head"].js_on_load == tc._RUN_HEAD_JS
    assert "--shep-rv-top-h" in tc._RUN_HEAD_JS and "ResizeObserver" in tc._RUN_HEAD_JS
