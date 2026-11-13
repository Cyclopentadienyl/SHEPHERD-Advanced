"""
Research › Prepare › Training: the run view (the ops-wiring round's B1b).
=========================================================================
`src/webui/components/training_console.py`. What these pin:

- **the run bar keeps its own height** (BACKLOG 52 F): its column's scale is
  0, so Gradio's inline flex-grow never gives it a share of the page's room;
- **a past run can be viewed** (BACKLOG 52 H): the Viewing select lists the
  run bar's workspace's runs, latest first, by the one rule of a run's
  option (short, name first, valued by the run resolver's full id); choosing
  one changes only the run view (the chart token, Export CSV, the header),
  never the run bar or its status pill, which always show the training job;
  the choice holds while the run bar shows the workspace it was made for and
  no job has started since (back on its workspace, it is shown again); its
  writer is in the run bar's ordered group (the render/poll single-writer
  contract);
- **the header says how the run viewed stands to the job**: the running
  job, the last job, a past run (while another runs), the latest run;
- **empty states say what will appear and the next step** (the skill's §9;
  BACKLOG 69): the chart region's line by cause (who starts a first run, to
  a read-only request), and an empty chart's own line in place of its plot,
  by cause too: a running job waits for its first validation, an ended run
  logged none (the component's own rule: `test_webui_charts`).

Module: tests/unit/test_training_console_runview.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from src.webui.components import model_list as ml  # noqa: E402
from src.webui.components import renderers as r  # noqa: E402
from src.webui.components import training_console as tc  # noqa: E402
from tests.unit.test_training_page import (  # noqa: E402,F401 - fixtures
    READ_ONLY,
    _metrics,
    _page,
    built,
    operator,
    root,
)

LISTED = "demo/gatv2/20261018-142000_run_d41f8b"
PAST = "20261016-090000_first-base_a1b2c3"
JOB = "20261022-101500_new-base_c0ffee"
RUNNING = {"status": "running", "job_id": "j7", "current_epoch": 2, "total_epochs": 5,
           "gpu": {"job": {"job_type": "training", "status": "running"}}}


@pytest.fixture
def runs(root, monkeypatch):  # noqa: F811
    """``demo`` with three runs, latest first: the job runner's (``JOB``,
    its log one epoch), the fixture's listed run, and an older past run
    (``PAST``, three epochs); the PyG probe off."""
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    base = root / "demo" / "runs" / "gatv2"
    for name, epochs, mtime in ((PAST, 3, 1_700_000_000), (JOB, 1, 1_900_000_000)):
        run = base / name
        (run / "logs").mkdir(parents=True)
        (run / "checkpoints").mkdir()
        (run / "logs" / f"train_{name[:8]}_{name[9:15]}.json").write_text(json.dumps(_metrics(epochs)))
        (run / "checkpoints" / "model-0.pt").write_bytes(b"x")
        os.utime(run / "checkpoints" / "model-0.pt", (mtime, mtime))
    listed = base / LISTED.rsplit("/", 1)[1] / "checkpoints"
    (listed / "model-0.pt").write_bytes(b"x")
    os.utime(listed / "model-0.pt", (1_800_000_000, 1_800_000_000))
    return base


def _job(monkeypatch, base, status):
    monkeypatch.setattr(tc.training_manager, "run_dir", base / JOB)
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: json.loads(json.dumps(status)))
    word = status.get("status") or "idle"
    monkeypatch.setattr(tc.training_manager, "get_job_status_word", lambda job_type: word)


def _pieces(ws, viewing, request, mode=None):
    return tc.render(ws, mode or {}, {}, [], request, viewing)["pieces"]


# --------------------------------------------------------------------------- the run bar
def test_the_run_bar_keeps_its_own_height_and_the_run_view_takes_the_rest():
    """BACKLOG 52 F, found at its root: Gradio writes a column's scale inline
    as its flex-grow, which outranks the page's ``flex: none``; at the
    default 1 the run bar took half the room the run view left and drew it
    blank under its two rows (189 px for 97 px of rows at 1850 x 960, more
    with every section collapsed). At 0 it is its rows' height."""
    _demo, _ctx, by_id = _page()
    assert by_id["shep-train-runbar"].scale == 0
    assert by_id["shep-train-runview"].scale == 22 and by_id["shep-train-params"].scale == 10
    assert "#shep-train-runbar {" in tc.PAGE_CSS and "flex: none;" in tc.PAGE_CSS.split(
        "#shep-train-runbar {", 1)[1].split("}", 1)[0]


# --------------------------------------------------------------------------- the Viewing select
def test_the_select_lists_the_workspaces_runs_latest_first_by_the_one_option_rule(runs):
    """Options short and name first (at most 40 characters, one separator,
    unique: the skill's §8), by `model_list.run_options` (the Run select's
    rule); each value the run resolver's full id, which it resolves."""
    ws = tc.workspace_state("demo")
    assert ws["runs"] == [f"gatv2/{JOB}", "gatv2/20261018-142000_run_d41f8b", f"gatv2/{PAST}"]
    choices = tc.view_choices(ws, ws["runs"], None)
    assert [value for _label, value in choices] == [f"demo/{label}" for label in ws["runs"]]
    assert choices == ml.run_options({f"demo/{label}": tuple(label.split("/", 1))
                                      for label in ws["runs"]})
    labels = [label for label, _value in choices]
    assert labels[0].startswith("new-base · ") and labels[2].startswith("first-base · ")
    assert len(set(labels)) == len(labels)
    assert all(len(label) <= r.OPTION_CHARS and label.count(" · ") <= 1 for label in labels)
    assert all(tc.resolve_run(value) is not None for _label, value in choices)
    # The Run select of Continue… and Fine-tune reads the same rule.
    parents = {"k": {"architecture": "gatv2", "run": PAST, "run_key": f"gatv2/{PAST}"}}
    assert ml.run_choices(parents) == [(labels[2], f"gatv2/{PAST}")]


def test_the_viewing_box_holds_a_whole_option_at_any_window_width():
    """The closed select is the header's one name for the run viewed, and an
    option's end is the start time that tells same-name runs apart: its box
    is sized by the option's 40 characters (`renderers.OPTION_CHARS`, as
    40ch beside Gradio's chevron, margins, padding and border), never by the
    window (``clamp(300px, 19vw, 360px)`` cut 33 of them at 1536 x 760). The
    header's words beside it take a line of their own under it where they
    would need more than two lines (their basis, about a state line and
    Details ›). The browser check measures both at each size."""
    rule = tc.PAGE_CSS.split("#shep-train-view-run .wrap {", 1)[1].split("}", 1)[0]
    assert f"width: calc({r.OPTION_CHARS}ch + 64px);" in rule and "vw" not in rule
    assert r.OPTION_CHARS == 40
    head = tc.PAGE_CSS.split("#shep-train-run-head {", 1)[1].split("}", 1)[0]
    assert "flex: 1 1 540px !important;" in head


def test_the_run_viewed_is_offered_first_when_it_is_not_listed(runs, monkeypatch, operator):  # noqa: F811
    """The training job's run before its directory is made, or in a
    workspace the run bar no longer shows, is the run viewed and offered
    first, so the select always holds the run shown."""
    _job(monkeypatch, runs, RUNNING)
    monkeypatch.setattr(tc.training_manager, "run_dir", runs / "20261023-080000_fresh_aaaaaa")
    ws = tc.workspace_state("demo")
    pieces = _pieces(ws, {}, operator)
    choices, value = pieces["view"]
    assert value == "demo/gatv2/20261023-080000_fresh_aaaaaa" == choices[0][1]
    assert [v for _l, v in choices[1:]] == [f"demo/{label}" for label in ws["runs"]]
    assert pieces["charts"] == {"run": None, "rev": 0, "empty": tc.WAITING_FOR_EPOCH}


def test_choosing_a_past_run_changes_only_the_view_and_the_job_drives_the_pill(
        runs, monkeypatch, operator):  # noqa: F811
    """BACKLOG 52 H. While a job runs, choosing a past run in Viewing changes
    the chart token, Export CSV and the header (its relation, its state, its
    Details), and nothing of the run bar: the pill, the reason line, the
    actions, the dataset, the banner are the training job's."""
    _job(monkeypatch, runs, RUNNING)
    ws = tc.workspace_state("demo")
    following = _pieces(ws, {}, operator)
    assert following["charts"]["run"] == f"demo/gatv2/{JOB}"
    assert following["view"][1] == f"demo/gatv2/{JOB}"
    assert tc.RELATION_RUNNING in following["head"]
    viewing = tc._on_view_run(f"demo/gatv2/{PAST}", ws)
    assert viewing == {"run": f"demo/gatv2/{PAST}", "ws": "demo", "job": "j7"}
    chosen = _pieces(ws, viewing, operator)
    assert chosen["charts"] == {"run": f"demo/gatv2/{PAST}", "rev": 6}
    assert chosen["view"][1] == f"demo/gatv2/{PAST}"
    assert tc.RELATION_PAST_WHILE in chosen["head"] and tc.RELATION_RUNNING not in chosen["head"]
    assert f"gatv2/{PAST}" in chosen["head"]                               # its Details
    changed = {key for key in tc.VIEW_KEYS if chosen[key] != following[key]}
    assert changed == {"head", "view", "charts"}
    assert chosen["pill"] == tc.pill_html(RUNNING) and "RUNNING · epoch 2/5" in chosen["pill"]
    # Export CSV names the chosen run, with its own log.
    csv = tc.training_chart_csv({"run": chosen["charts"]["run"]})
    assert csv["filename"] == f"training_metrics_gatv2_{PAST}.csv"
    assert len(csv["csv"].splitlines()) == 1 + 6
    assert tc.training_chart_series(chosen["charts"])["run"] == f"demo/gatv2/{PAST}"


def test_the_choice_holds_while_its_workspace_is_shown_and_no_job_has_started(
        runs, monkeypatch, operator):  # noqa: F811
    """A past run chosen stays viewed when the job it was chosen beside ends.
    Another workspace in the run bar shows its own run; back on the workspace
    the choice was made for, with no job started since, the choice is shown
    again (it is compared with the workspace shown, and only the select
    writes it). A job started since is shown as it starts (from this page or
    any other entry), on any workspace."""
    _job(monkeypatch, runs, RUNNING)
    ws = tc.workspace_state("demo")
    viewing = tc._on_view_run(f"demo/gatv2/{PAST}", ws)
    _job(monkeypatch, runs, {**RUNNING, "status": "completed"})
    ended = _pieces(ws, viewing, operator)
    assert ended["charts"]["run"] == f"demo/gatv2/{PAST}"
    assert tc.RELATION_PAST in ended["head"] and tc.RELATION_PAST_WHILE not in ended["head"]
    assert "Completed" in ended["pill"]
    other = tc.workspace_state("resampled")
    assert tc._chosen_run(viewing, {**RUNNING, "status": "completed"}, other) is None
    away = _pieces(other, viewing, operator)
    assert away["charts"]["run"] != f"demo/gatv2/{PAST}" and away["view"][1] != f"demo/gatv2/{PAST}"
    back = _pieces(tc.workspace_state("demo"), viewing, operator)
    assert back["charts"]["run"] == back["view"][1] == f"demo/gatv2/{PAST}"
    _job(monkeypatch, runs, {**RUNNING, "job_id": "j8"})                   # a new job
    started = _pieces(ws, viewing, operator)
    assert started["charts"]["run"] == f"demo/gatv2/{JOB}"
    assert tc.RELATION_RUNNING in started["head"]
    _job(monkeypatch, runs, {**RUNNING, "job_id": "j8", "status": "completed"})
    assert _pieces(ws, viewing, operator)["charts"]["run"] == f"demo/gatv2/{JOB}"   # not back
    # A choice of a run no longer listed falls back to the run followed.
    gone = {**viewing, "run": "demo/gatv2/20991231-000000_gone_ffffff", "job": "j8"}
    assert _pieces(ws, gone, operator)["charts"]["run"] == f"demo/gatv2/{JOB}"


@pytest.mark.parametrize("job, live, latest, words", [
    (True, True, False, tc.RELATION_RUNNING),
    (True, False, False, tc.RELATION_LAST_JOB),
    (False, True, False, tc.RELATION_PAST_WHILE),
    (False, True, True, tc.RELATION_PAST_WHILE),
    (False, False, True, tc.RELATION_LATEST),
    (False, False, False, tc.RELATION_PAST),
])
def test_the_relation_word_keeps_the_job_and_the_run_viewed_apart(job, live, latest, words):
    run = tc.RunRef(id="demo/gatv2/r", label="gatv2/r", path=Path("/x"), workspace="demo",
                    job=job, latest=latest)
    assert tc.view_relation(run, live) == words


def test_a_past_runs_state_is_its_checkpoints_never_the_jobs(runs, operator):  # noqa: F811
    """The training job's state ("Running · epoch 2 of 5", "Failed") is said
    only of the job's own run; a past run's line is its checkpoint's."""
    past = tc.RunRef(id=f"demo/gatv2/{PAST}", label=f"gatv2/{PAST}", path=runs / PAST,
                     workspace="demo", job=False, latest=False)
    snapshot = {"readable": True, "completed_epochs": 3, "target_epochs": 3, "end": "finished"}
    line = tc.run_snapshot_html(past, RUNNING, operator=True, snapshot=snapshot)
    assert "Running" not in line and "Finished · 3 of 3 epochs completed" in line
    gone = {**past, "path": runs / "never_made"}
    assert tc.run_snapshot_html(tc.RunRef(gone), {"status": "failed"}, operator=True,
                                snapshot=None).startswith('<span class="shep-rv-snap">No run')


def test_the_select_writes_only_the_view_choice_in_the_run_bars_ordered_group():
    """The render/poll single-writer contract: the select's input writes the
    session's view choice alone, in the run bar's queued group of one, then
    the render; the select's options and value are a render's piece (`VIEW_KEYS`
    ``view``), handed over by the poll, never written by it; every render and
    the poll read the choice."""
    from src.webui import shell

    demo, ctx, by_id = _page()
    select = by_id["shep-train-view-run"]
    assert select.label == "Viewing" and "shep-train-view-run" in tc.FIELD_HELP
    fns = list(demo.fns.values())
    chosen = [fn for fn in fns if fn.name == "_on_view_run"]
    assert len(chosen) == 1
    fn = chosen[0]
    assert (select._id, "input") in fn.targets
    assert fn.concurrency_id == shell.TRAINING_SELECTION_CONCURRENCY
    viewing = fn.outputs[0]
    assert len(fn.outputs) == 1 and isinstance(viewing, gr.State)
    writers = [f for f in fns if viewing in f.outputs]
    assert writers == [fn]
    for name in ("_render_view", "_render_changed", "_poll"):
        for f in (f for f in fns if f.name == name):
            assert viewing in f.inputs, name
            assert (select in f.outputs) == (name != "_poll"), name
    assert "view" in tc.HANDED_KEYS and "view" not in tc.POLL_KEYS + tc.SELECTION_KEYS
    assert "snap" not in tc.VIEW_KEYS and "shep-train-run-snap" not in by_id


# --------------------------------------------------------------------------- empty states
def test_the_chart_regions_line_says_what_will_appear_and_the_next_step(runs, root,  # noqa: F811
                                                                        monkeypatch):
    """The skill's §9, by cause: no run yet in the workspace; the job's run
    before its first epoch; a run that ended with no epoch, or with no run
    directory; a run no longer listed. Each says what will appear, or what to
    do instead, with one action at most; with no run yet, the action is said
    to an operator, and who takes it to a read-only request (whose Start is
    locked: a refusal says who can act). The figure's own line before the
    page's first render is the same for every viewer: it loads."""
    empty = tc.workspace_state("scratch_ws")
    assert empty["runs"] == [] and tc._current_run({"status": "idle"}, empty, {}) is None
    assert tc._chart_token(None, {}, operator=True) == {"run": None, "rev": 0,
                                                        "empty": tc.NO_RUN_YET}
    assert tc._chart_token(None, {}, operator=False) == {"run": None, "rev": 0,
                                                         "empty": tc.NO_RUN_YET_READ_ONLY}
    assert "Start one with ▸ Start" in tc.NO_RUN_YET and "appear here" in tc.NO_RUN_YET
    assert tc.NO_RUN_YET_READ_ONLY.endswith("An operator starts one with ▸ Start.")
    assert "Start one" not in tc.NO_RUN_YET_READ_ONLY and "appear here" in tc.NO_RUN_YET_READ_ONLY
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    refused = _pieces(empty, {}, READ_ONLY)
    assert refused["charts"]["empty"] == tc.NO_RUN_YET_READ_ONLY
    assert refused["start"] == {"value": f"🔒 {tc.ACTION_LABELS['start']}", "interactive": False}
    _demo, _ctx, by_id = _page()
    template = by_id["shep-train-charts"].html_template
    assert f'data-empty="{r.e(tc.charts.LOADING)}"' in template and "Start" not in template
    never = tc.RunRef(id="demo/gatv2/never", label="gatv2/never", path=runs / "never",
                      workspace="demo", job=True, latest=False)
    assert tc._chart_token(never, {}, operator=True, live=True)["empty"] == tc.WAITING_FOR_EPOCH
    assert tc._chart_token(never, {}, operator=True, live=False)["empty"] == tc.NO_RUN_DIR_ENDED
    quiet = runs / "20261024-000000_quiet_dddddd"
    (quiet / "logs").mkdir(parents=True)
    (quiet / "logs" / "train_20261024_000000.json").write_text(
        json.dumps({"train": [], "validation": []}))
    token = {"run": f"demo/gatv2/{quiet.name}", "rev": 0}
    monkeypatch.setattr(tc.training_manager, "get_job_status_word", lambda job_type: "completed")
    assert tc.training_chart_series(token)["empty"] == tc.NO_EPOCH_LOGGED
    monkeypatch.setattr(tc.training_manager, "get_job_status_word", lambda job_type: "running")
    monkeypatch.setattr(tc.training_manager, "run_dir", quiet)
    assert tc.training_chart_series(token)["empty"] == tc.WAITING_FOR_EPOCH
    assert tc.training_chart_series({"run": "demo/gatv2/gone", "rev": 1})["empty"] == \
        tc.NO_LONGER_LISTED
    for words in (tc.NO_EPOCH_LOGGED, tc.NO_RUN_DIR_ENDED, tc.NO_LONGER_LISTED):
        assert words.endswith("choose another run in Viewing.")


def test_an_empty_chart_says_its_own_line_and_draws_no_plot_or_axis_label():
    """BACKLOG 69: an empty chart kept its full height, an axis label alone
    under a blank box. Each spec without a series carries its line, which the
    module shows in place of the plot and its axis label (the chart
    component's tests); a drawn chart carries none. Before a first
    validation, the line is the cause's: the running job's two validation
    charts wait for it; a run that ended without one (stopped or failed
    first, or validating less often than it trained) logged none, and no
    promise is made of it."""
    no_validation = {"train": _metrics(2)["train"], "validation": []}
    waiting = {c["id"]: c for c in tc.training_charts(no_validation, live=True)}
    assert waiting["mrr"]["empty"] == waiting["hits"]["empty"] == tc.WAITING_FOR_VALIDATION
    assert "empty" not in waiting["loss"] and "empty" not in waiting["lr"]
    ended = {c["id"]: c for c in tc.training_charts(no_validation, live=False)}
    assert ended["mrr"]["empty"] == ended["hits"]["empty"] == tc.NO_VALIDATION_LOGGED
    assert "once" not in tc.NO_VALIDATION_LOGGED
    no_lr = {c["id"]: c for c in tc.training_charts(
        {"train": [{"epoch": 0, "train_loss": 1.0}], "validation": []})}
    assert no_lr["lr"]["empty"] == tc.NOT_IN_THE_LOG == "Not recorded in this run’s log."
    other = {c["id"]: c for c in tc.training_charts(
        {"train": [{"epoch": 0, "train_loss": 1.0}], "validation": [{"epoch": 0, "val_loss": 1}]})}
    assert "empty" not in other["mrr"] and "charted" in other["mrr"]["title"]  # its title says it
    drawn = {c["id"]: c for c in tc.training_charts(_metrics(3))}
    assert not [c for c in drawn.values() if "empty" in c]


def test_a_listed_run_that_ended_before_its_first_validation_says_it_logged_none(
        runs, monkeypatch):
    """B1b makes any listed run viewable: one that ended with train entries
    and no validation entry draws its Loss and learning rate, and its two
    validation charts say it logged none; the training job's run, while it
    logs, says they are drawn once it validates (`_logging_now`, the
    region's own test)."""
    run = runs / "20261010-000000_novalid_eeeeee"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "train_20261010_000000.json").write_text(
        json.dumps({"train": _metrics(2)["train"], "validation": []}))
    token = {"run": f"demo/gatv2/{run.name}", "rev": 2}
    _job(monkeypatch, runs, {**RUNNING, "status": "completed"})
    ended = {c["id"]: c for c in tc.training_chart_series(token)["charts"]}
    assert ended["mrr"]["empty"] == ended["hits"]["empty"] == tc.NO_VALIDATION_LOGGED
    assert ended["loss"]["series"] and ended["lr"]["series"]
    _job(monkeypatch, runs, RUNNING)
    monkeypatch.setattr(tc.training_manager, "run_dir", run)
    live = {c["id"]: c for c in tc.training_chart_series(token)["charts"]}
    assert live["mrr"]["empty"] == live["hits"]["empty"] == tc.WAITING_FOR_VALIDATION
