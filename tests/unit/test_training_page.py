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
from src.webui.components import model_list as ml  # noqa: E402
from src.webui.components import renderers as r  # noqa: E402
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
    """A task model's log as a finetune writes it: a training loss, and pooled
    MRR and top-k on validation (no validation loss)."""
    return {"train": [{"epoch": e, "train_loss": 1.0 / (e + 1),
                       "learning_rate": 1e-4 if e % 3 else 0.0} for e in range(n)],
            "validation": [{"epoch": e, "val_pooled_mrr": 0.1 * e, "val_pooled_top1": 0.05,
                            "val_pooled_top3": 0.1, "val_pooled_top10": 0.2}
                           for e in range(n)]}


@pytest.fixture
def root(built, tmp_path, monkeypatch):
    demo, resampled = built
    root = tmp_path / "root"
    root.mkdir()
    shutil.copytree(demo, root / "demo")
    shutil.copytree(resampled, root / "resampled")
    shutil.copytree(demo, root / ODD)
    (root / "scratch_ws").mkdir()
    run = root / "demo" / "runs" / "gatv2" / "20261018-142000_run_d41f8b"
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
                    "shep-train-drawer-close", "shep-train-stop-confirm", "shep-train-stop-keep",
                    "shep-train-banner", "shep-train-params", "shep-train-drawer",
                    "shep-train-runview", "shep-train-run-head", "shep-train-view-run",
                    "shep-train-charts", "resource_display"):
        assert elem_id in by_id, elem_id
    # What M5-2 retired: the Data Paths section, the Epoch Window, the native plots.
    # R-49: the drawer's second closer (its footer's Cancel) and the run bar's
    # "Run a new one…" line while it is open.
    for gone in ("data_dir", "epoch_window", "loss_plot", "mrr_plot", "hits_plot", "lr_plot",
                 "export_csv_file", "status_display", "error_banner", "shep-train-cancel",
                 "shep-train-continues"):
        assert gone not in by_id, gone
    assert not [b for b in by_id.values() if isinstance(b, gr.LinePlot)]
    for name in ("_from_datasets", "_keep_or_clear_workspace", "_load_training_config",
                 "_reset_training_config", "_export_metrics_csv", "_poll_status"):
        assert not hasattr(tc, name), name


def test_the_form_is_what_a_new_base_takes_common_settings_first():
    """Ops-wiring round, decisions S1, S3, S5: the Parameters are what a new
    base takes (`run_plan.new_run_settings("base")`), each starting at the
    value the run takes without it; Epochs, Learning rate, Batch size, Seed,
    Split seed and Device shown; the rest under one closed "More training
    settings", a setting a rule reads only under its guard hidden while it
    does not apply; the model spec's fields under a sibling closed "Model";
    integer settings take whole numbers only. The ShepherdGNN form stays
    gone."""
    from src.utils.run_plan import new_run_settings

    demo, _ctx, by_id = _page()
    assert set(tc.BASE_FIELDS) | {"model_spec"} == new_run_settings("base")
    references, _sources = tc.base_references()
    for name in tc.BASE_FIELDS:
        field = by_id[name]
        assert field.label == r.SETTING_LABELS[name]
        assert field.value == tc.form_value(references[name]), name
        if isinstance(field, gr.Number):
            assert field.precision == (0 if isinstance(references[name], int) else None), name
    assert by_id["num_neighbors"].value == "15, 10, 5" and by_id["use_amp"].value is False
    # Early stopping off: a text box reading "off", which the browser keeps (a
    # number box showed None as 0 once mounted, K6 closing smoke).
    assert isinstance(by_id["early_stopping_patience"], gr.Textbox)
    assert by_id["early_stopping_patience"].value == "off"
    assert by_id["scheduler_type"].choices == [(c, c) for c in ("cosine", "onecycle", "linear",
                                                                "none")]
    # Read only under a guard the reference does not hold: hidden.
    hidden = {name for name in tc.BASE_FIELDS if by_id[name].visible is False}
    assert hidden == {"warmup_steps", "min_lr_ratio", "amp_dtype"}
    accordions = {b.label: b.open for b in demo.blocks.values() if isinstance(b, gr.Accordion)}
    assert accordions == {"More training settings": False, "Model": False}
    more, model = by_id["shep-train-more"], by_id["shep-train-model"]

    def inside(block, parent):
        while block is not None:
            if block is parent:
                return True
            block = getattr(block, "parent", None)
        return False

    common = [*tc.BASE_FIELDS[:tc.BASE_SHOWN], "device"]
    assert not any(inside(by_id[name], more) or inside(by_id[name], model) for name in common)
    assert all(inside(by_id[name], more) for name in tc.BASE_FIELDS[tc.BASE_SHOWN:])
    spec_fields = [field for field in tc.spec_form_fields() if field.editable]
    assert spec_fields and all(inside(by_id[field.elem_id], model) for field in spec_fields)
    assert by_id["spec-encoder-relation_edge_dim"].value == "off"
    assert by_id["spec-encoder-layers"].precision == 0
    assert "spec-encoder-kind" not in by_id          # one architecture: a fact
    assert "GATv2" in by_id["shep-train-spec-facts"].value
    for gone in ("conv_type", "hidden_dim", "num_layers", "num_heads",
                 "diagnosis_weight", "link_prediction_weight", "contrastive_weight",
                 "ortholog_weight", "use_ortholog_gate", "amp_mode", "randomize_seed",
                 "temperature", "label_smoothing", "margin", "num_neighbors_str",
                 "max_subgraph_nodes", "shep-train-differs"):
        assert gone not in by_id, gone
    for name in ("_collect_config", "ConfigValidationError", "SEED_PARAM_INDEX", "_random_seed",
                 "_seed_field_lock", "AMP_HGT_REFUSAL", "_on_conv_type_change", "_MAX_SEED",
                 "recipe_differences", "differs_html", "_on_base_changed"):
        assert not hasattr(tc, name), name
    # One visible line on what the operator set (decision S2), beside Reset.
    assert "Set by you: nothing" in by_id["shep-train-setby"].value
    assert by_id["shep-train-reset"].value == "Reset to reference"


def test_each_fields_input_records_it_and_reset_and_the_guards_are_wired():
    """Decision S2: each field's ``.input`` (never ``.change``, which a page's
    own write fires too) adds it to the form's record in the browser
    (`record_touches`: no request, the input's one listener), then, chained
    after it, says the line, its last input always said ("always_last"); the
    record is a hidden JSON that Start's own
    request carries with every field, so what Start sends is read from the
    click's payload alone. Reset writes the record, the line and every
    field; the guards' change shows what depends on them. None of these
    waits for the queue."""
    import json

    demo, _ctx, by_id = _page()
    keys = tc.base_form_keys()
    fields = [by_id[key] if key in by_id else by_id[f"spec-{key.replace('.', '-')}"]
              for key in keys]
    fns = list(demo.fns.values())
    setby = by_id["shep-train-setby"]
    touched = by_id["shep-train-touched"]
    assert isinstance(touched, gr.JSON) and touched.visible is False and touched.value == []
    for key, field in zip(keys, fields, strict=True):
        (record,) = [fn for fn in fns if (field._id, "input") in fn.targets]
        (line,) = [fn for fn in fns if fn.trigger_after == record._id and fn.fn is not None]
        assert record.fn is None and record.inputs == [touched] == record.outputs
        assert record.js == tc.TOUCH_JS.replace("{key}", json.dumps(key))
        assert record.queue is False and record.trigger_mode == "multiple"
        assert line.outputs == [setby] and line.inputs == [touched, *fields]
        assert line.queue is False and line.trigger_mode == "always_last"
        # The line said from the record as it is (it holds the key by then).
        assert line.fn is tc._on_base_input
    assert tc._on_base_input(["x", keys[0]], *tc.base_form_defaults()) == tc.set_by_you_html(
        dict(zip(keys, tc.base_form_defaults(), strict=True)), ["x", keys[0]])
    assert not [fn for fn in fns if any(target[1] == "change" and target[0] in
                                        {f._id for f in fields} for target in fn.targets)
                and fn.fn is not tc._on_base_guards]
    (reset,) = [fn for fn in fns if fn.fn is tc._on_base_reset]
    assert reset.targets == [(by_id["shep-train-reset"]._id, "click")]
    assert reset.outputs == [touched, setby, *fields] and reset.queue is False
    (guards,) = [fn for fn in fns if fn.fn is tc._on_base_guards]
    assert guards.inputs == [by_id["scheduler_type"], by_id["use_amp"]]
    assert guards.outputs == [by_id["warmup_steps"], by_id["min_lr_ratio"], by_id["amp_dtype"]]
    assert {target[1] for target in guards.targets} == {"change"}
    (start,) = [fn for fn in fns if fn.fn is tc._on_start]
    assert start.inputs[3] is touched and start.inputs[4:] == fields
    # No record on the server: no State is written by a field's input.
    assert not [fn for fn in fns if any(isinstance(block, gr.State) for block in fn.outputs)
                and any((f._id, "input") in fn.targets for f in fields)]


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


def test_no_listener_of_the_continue_drawers_controls_shows_a_progress_timer():
    """R-59: the drawer's listeners answer at once; Gradio's progress timer
    ("processing | 0.1/0.1s") flashed over the run select, the parent's
    details and the check when the source, the listed or typed checkpoint or
    the drawer's Device changed. Every listener of those controls (and of the
    run select and the operation) shows none."""
    demo, _ctx, by_id = _page()
    controls = {by_id[name]._id: name for name in (
        "continue_source", "continue_run", "checkpoint_dropdown", "manual_checkpoint",
        "continue_device", "continue_operation")}
    shown = sorted({(controls[block], event, fn.name) for fn in demo.fns.values()
                    for block, event in fn.targets
                    if block in controls and fn.show_progress != "hidden"})
    assert shown == []
    assert {name for fn in demo.fns.values() for block, _event in fn.targets
            if block in controls for name in [controls[block]]} == set(controls.values())


def test_a_tick_while_another_page_is_shown_reads_nothing(monkeypatch):
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: pytest.fail("a tick on another page read the status"))
    monkeypatch.setattr(tc.training_manager, "get_system_resources",
                        lambda: pytest.fail("a tick on another page read the resources"))
    out = tc._poll("clinical", {}, {}, {}, [], {}, {}, gradio_request())
    assert len(out) == 3 and all(u == gr.update() for u in out)
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
    from src.webui.components.renderers import resources_html

    assert "font-size" not in resources_html({"gpu": {}, "ram": {}})


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
            # A step may be the browser's alone (fn None: SHOW_CHECK_JS).
            if getattr(current.fn, "__name__", None) == "read_chips":
                return True
        return False

    acts = [fn for fn in fns if fn.fn is not None and fn.fn.__name__ in gated]
    assert {fn.fn.__name__ for fn in acts} == set(gated)
    for fn in acts:
        assert reads_chips(fn), fn.fn.__name__


# --------------------------------------------------------------------------- the run bar's workspace
def test_the_select_names_the_servers_workspaces_and_leaves_their_identity_to_the_chip(root):
    """The server's list, in the shared order, with every page's options
    (`renderers.workspace_choices`, no option rule of the run bar's own): a
    trainable workspace's option is its name, its identity (fingerprint,
    synthetic) the chip's beside it (the GB10 WebUI recheck: said twice side
    by side); one that cannot be trained keeps its reason in its option, read
    before choosing."""
    from src.webui.components import renderers as r
    from src.webui.components.workspaces import read_workspaces

    state = tc.workspace_state("demo")
    rows = read_workspaces()["rows"]
    trainable = {x["name"] for x in rows if r.is_digest(x.get("dataset_fingerprint"))}
    assert [tuple(c) for c in state["choices"]] == r.workspace_choices(rows)
    assert all(label == name for label, name in state["choices"] if name in trainable)
    assert ("demo", "demo") in [tuple(c) for c in state["choices"]]
    assert any(label != name for label, name in state["choices"])      # a reason kept
    assert "dataset fingerprint" in tc.chip_html(state)
    assert state["name"] == "demo" and r.is_digest(state["row"]["dataset_fingerprint"])
    assert state["runs"] == ["gatv2/20261018-142000_run_d41f8b"]
    assert tc.workspace_state("not-listed")["row"] is None


def test_the_chip_shows_the_dataset_and_its_details_escaped_without_a_path(root):
    state = tc.workspace_state(ODD)
    html = tc.chip_html(state)
    fingerprint = state["row"]["dataset_fingerprint"]
    assert "<b>odd" not in html and "&lt;b&gt;odd&amp;name" in html
    assert fingerprint[:12] in html and 'data-copy="' + fingerprint + '"' in html
    assert "record verified" in html and "synthetic" in html
    assert 'data-act="details"' in html and 'data-act="datasets"' in html
    assert "Manifest contents (7 files)" in html and "Allocation seed" in html
    assert "Built (UTC)" in html and "workspace root" not in html
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
    # The operator's pick (`shell.picked`: one request per option chosen).
    hand = [fn for fn in demo.fns.values()
            if (by_id["shep-train-dataset"]._id, "blur") in fn.targets]
    assert [fn.fn.__name__ for fn in hand] == ["_on_dataset"]
    assert ctx.training_workspace in hand[0].outputs


# --------------------------------------------------------------------------- the pill and the reason line
@pytest.mark.parametrize("status, kind, words, details", [
    ({"status": "idle"}, "idle", "Idle · no training job since the server started", None),
    ({"status": "running", "phase": "initializing", "elapsed_seconds": 4}, "accent",
     "Starting · loading data and the model", "00:00:04 elapsed"),
    ({"status": "running", "current_epoch": 12, "total_epochs": 100, "batch": 340,
      "total_batches": 812, "elapsed_seconds": 872}, "running",
     "RUNNING · epoch 12/100", "batch 340/812, 00:14:32 elapsed"),
    ({"status": "stopping", "current_epoch": 3, "total_epochs": 10}, "accent",
     "Stopping · epoch 3 of 10", None),
    ({"status": "completed", "current_epoch": 60, "total_epochs": 60,
      "start_time": "2026-10-18T14:20:00", "end_time": "2026-10-18T15:01:07"}, "ok",
     "Completed · epoch 60 of 60", "00:41:07 elapsed"),
    ({"status": "stopped", "current_epoch": 7, "total_epochs": 60}, "grey",
     "Stopped · epoch 7 of 60", None),
    ({"status": "failed", "current_epoch": 7, "total_epochs": 60,
      "start_time": "2026-10-19T08:15:00", "end_time": "2026-10-19T08:24:12"}, "red",
     "FAILED · epoch 7 of 60", "00:09:12 elapsed"),
])
def test_the_status_pill(status, kind, words, details):
    """The state and how far it got, with at most one " · " (the WebUI skill
    §5: it read "RUNNING · epoch 12/100 · batch 340/812 · 00:14:32", three);
    the batch and the time are the pill's details, after its words with no
    second separator. The pill is a role="status" region, read as one text:
    its words and details never run together ("epoch 12/100batch …"). Its
    mark is the shared status rule's (`renderers.status_mark`, through
    `job_status_html`): the shape and the word with the colour, a grey ○
    while idle or stopped, the blue spinner only while it runs, a red ✕ only
    for a failure; no page glyph of its own."""
    state, said = tc.pill_state(status)
    assert said == words and tc.PILL_KIND[state] == kind
    assert said.count(" · ") <= 1 and tc.pill_details(status) == details
    html = tc.pill_html(status)
    assert f"shep-pill-{kind}" in html
    assert tc.job_status_html(state, words) in html
    # What a screen reader reads: the words, the shape hidden (and the space after it).
    read = re.sub(r'<span class="shep-st-shape" aria-hidden="true">[^<]*</span>'
                  + re.escape(r.MARK_SPACE), "", html)
    text = re.sub(r"<[^>]+>", "", read)
    assert text == (f"{words}, {details}" if details else words), text
    assert text.count(" · ") <= 1
    if not details:
        assert "shep-pill-sub" not in html
    marks = {"idle": "○", "grey": "○", "ok": "●", "red": "✕", "accent": "■"}
    if kind == "running":
        assert '<span class="shep-spinner" aria-hidden="true"></span>' in html
    else:
        assert f'<span class="shep-st-shape" aria-hidden="true">{marks[kind]}</span>' in html
        assert "shep-spinner" not in html
    assert "shep-dot" not in html


def test_a_failed_or_stopped_run_is_never_green():
    for status in ("failed", "stopped"):
        html = tc.pill_html({"status": status})
        assert "shep-pill-ok" not in html


def test_the_idle_fact_is_said_once_as_plain_muted_text_and_stop_is_not_drawn(root, operator,
                                                                               monkeypatch):
    """R-39: while idle the page said it three times: the header chip ("GPU
    jobs: none", the shell's), the status pill as a bordered, fully rounded
    lozenge beside the 40 px buttons, and the reason line "Stop disabled: no
    training is running." beside a visible, disabled ■ Stop. Now the pill is
    the run bar's one idle fact, plain muted text with its grey ○, and Stop
    is not drawn while no training job runs or stops (nor its reason)."""
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: {"status": "idle"})
    pieces = tc.render(tc.workspace_state("demo"), {}, {}, [], operator)["pieces"]
    assert "shep-rb-nojob" in pieces["runbar"]
    assert "Stop" not in pieces["reason"] and "no training" not in pieces["reason"]
    assert pieces["pill"].count("no training job since the server started") == 1
    assert 'class="shep-pill shep-pill-idle"' in pieces["pill"]
    css = tc.PAGE_CSS
    assert (f"{tc._PAGE} .shep-rb-nojob #shep-train-stop {{ display: none !important; }}") in css
    idle = css.split(f"{tc._PAGE} #shep-train-pill .shep-pill-idle {{", 1)[1].split("}", 1)[0]
    for rule in ("border-color: transparent;", "background: none;", "padding-left: 0;",
                 "font-weight: 400;", "color: var(--shep-muted);"):
        assert rule in idle, rule
    # A running job draws Stop again, with no "nojob" class.
    for status in ({"status": "running", "job_id": "j", "current_epoch": 1, "total_epochs": 2},
                   {"status": "stopping", "job_id": "j"}):
        assert _view(status)["stop_shown"] is True, status
    for status in ({"status": "completed"}, {"status": "failed"}, {"status": "stopped"}):
        assert _view(status)["stop_shown"] is False, status


def _view(status=None, ws=None, mode=None, access=None, active=None, gpu=None):
    ws = ws if ws is not None else {"row": {"name": "demo", "trainable": True,
                                            "dataset_fingerprint": "a" * 64}}
    return tc.run_bar_view(status or {"status": "idle"}, ws, mode or {},
                           access=access or _Access(True), active=active, gpu=gpu)


def test_the_run_bars_actions_and_their_one_reason():
    idle = _view()
    assert (idle["start"], idle["cont"], idle["stop"]) == (True, True, False)
    # R-39: with no job, Stop is not drawn, so no reason line says it is disabled.
    assert idle["reason"] == "" and idle["stop_shown"] is False

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
    """The drawer's start (Continue Training, or Start branch for Branch)
    follows the chosen parent, and its reason is beside it, in the drawer's
    footer, in its own words (ops-wiring round B1c; the skill's §9). The run
    bar draws none of its actions while the drawer is open, so it says no
    reason (R-49: "Start disabled: close Continue… to start a new run.")."""
    ws = {"row": {"name": "demo", "trainable": True, "dataset_fingerprint": "a" * 64}}
    nothing = _view(ws=ws, mode={"drawer": True, "parent": {}})
    assert nothing["start"] is False and nothing["resume"] is False
    assert nothing["reason"] == ""
    assert "Continue Training disabled: choose a checkpoint first." in nothing["drawer_reason"]
    same = _view(ws=ws, mode={"drawer": True, "parent": {"chosen": True, "fingerprint": "a" * 64}})
    assert same["resume"] is True and same["drawer_reason"] == ""
    differs = _view(ws=ws, mode={"drawer": True,
                                 "parent": {"chosen": True, "fingerprint": "e" * 64}})
    assert differs["resume"] is False
    assert "the parent’s dataset" not in differs["reason"]            # not in the run bar
    assert ("Continue Training disabled: the parent’s dataset is not this workspace’s current "
            "one.") in differs["drawer_reason"]
    assert "run bar above" not in differs["drawer_reason"]
    # A refused check, by what it refused and what to do (the B1c review: one
    # sentence blamed the checkpoint for every refusal).
    refused = _view(ws=ws, mode={"drawer": True,
                                 "parent": {"chosen": True, "refused": tc.REFUSED_PARENT}})
    assert refused["resume"] is False
    assert "cannot be continued" in refused["drawer_reason"]
    branch = _view(ws=ws, mode={"drawer": True, "operation": "branch",
                                "parent": {"chosen": True, "refused": tc.REFUSED_PARENT}})
    assert ("Start branch disabled: the check says this checkpoint cannot be branched."
            in branch["drawer_reason"])
    edited = _view(ws=ws, mode={"drawer": True, "operation": "branch",
                                "parent": {"chosen": True, "refused": tc.REFUSED_SETTINGS}})
    assert edited["resume"] is False
    assert ("Start branch disabled: the check refuses a setting you edited; change it, then "
            "check again.") in edited["drawer_reason"]
    for source, action in ((tc.CONTINUE_FROM_LIST, "Refresh the list, then check it again."),
                           (tc.CONTINUE_FROM_PATH,
                            "Click Check checkpoint to read the file there now.")):
        changed = _view(ws=ws, mode={"drawer": True, "parent": {
            "chosen": True, "source": source, "refused": tc.REFUSED_CHANGED}})
        assert changed["resume"] is False
        assert (f"Continue Training disabled: the checkpoint changed since it was chosen. "
                f"{action}") in changed["drawer_reason"]
    # A cause before the drawer's own names the drawer's start too.
    busy = _view({"status": "running"}, ws=ws, mode={"drawer": True, "operation": "branch"})
    assert "Start branch disabled: a training job is running." in busy["drawer_reason"]
    assert busy["reason"] == ""
    # Closed, the run bar says it again.
    closed = _view({"status": "running"}, ws=ws, mode={})
    assert "Start and Continue… disabled: a training job is running." in closed["reason"]


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
    ws = tc.workspace_state("demo")
    cache = {}
    run = tc._current_run({"status": "idle"}, ws, cache)
    assert run["id"] == "demo/gatv2/20261018-142000_run_d41f8b" and run["latest"] is True
    job_run = root / "resampled" / "runs" / "sage" / "20261020-090000_run_aaaaaa"
    job_run.mkdir(parents=True)
    monkeypatch.setattr(tc.training_manager, "run_dir", job_run)
    run = tc._current_run({"status": "running", "job_id": "j1"}, ws, cache)
    assert run["id"] == "resampled/sage/20261020-090000_run_aaaaaa" and run["latest"] is False
    assert run["workspace"] == "resampled"
    # A workspace the server does not list: the job names the run.
    outside = Path(str(root) + "-elsewhere") / "runs" / "gatv2" / "r"
    monkeypatch.setattr(tc.training_manager, "run_dir", outside)
    assert tc._current_run({"status": "running", "job_id": "j2"}, ws, {})["id"] == "job:j2"


@pytest.mark.parametrize("run_id", [
    "demo/gatv2/../../../etc", "demo/gatv2/20261018-142000_run_d41f8b/../x", "../demo/gatv2/x",
    "demo/gatv2", "demo/xyz/20261018-142000_run_d41f8b", "nope/gatv2/20261018-142000_run_d41f8b",
    "/abs/path", "job:none", None, 3, "demo/gatv2/" + "x" * 500,
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
    arch = root / "demo" / "runs" / "gatv2"
    (arch / "20261019-000000_run_link").symlink_to(outside, target_is_directory=True)
    (arch / "20261019-000000_run_file").write_text("not a run")
    for run_id in ("demo/gatv2/20261019-000000_run_link", "demo/gatv2/20261019-000000_run_file"):
        assert tc.resolve_run(run_id) is None
        assert tc.training_chart_series({"run": run_id, "rev": 1})["charts"] == []
        assert tc.training_chart_csv({"run": run_id})["csv"] == ""
    # The listed run beside them still answers.
    assert tc.resolve_run("demo/gatv2/20261018-142000_run_d41f8b") is not None


def test_an_architecture_that_is_not_one_never_lists_another_directory(root):
    """M5-3 review ARCH-10 (d), C1: the run id's directory is matched against
    the names the workspace's runs/ lists, never joined onto a path.
    ``demo/../runs`` would otherwise list the workspace's own directory
    (``runs/..``) and find ``runs`` there, inside the workspace, as a "run"."""
    assert (root / "demo" / "runs").is_dir()
    for run_id in ("demo/../runs", "demo/./gatv2", "demo/GAT/20261018-142000_run_d41f8b",
                   "demo/GATV2/20261018-142000_run_d41f8b"):
        assert tc.resolve_run(run_id) is None, run_id
        assert tc.training_chart_series({"run": run_id, "rev": 1})["charts"] == []


#: A base run's log (`lp_task.validate`, the Trainer's ``val_`` prefix) and a
#: Fine-tune child's (`finetune_task`'s pooled report).
BASE_LOG = {"train": [{"epoch": e, "train_loss": 1.0 / (e + 1), "learning_rate": 1e-4}
                      for e in range(3)],
            "validation": [{"epoch": e, "val_lp_accuracy": 0.5 + 0.1 * e, "val_lp_roc_auc": 0.6,
                            "val_lp_average_precision": 0.55, "val_lp_loss": 1.0 - 0.1 * e,
                            "val_lp_positives": 34} for e in range(3)]}
CHILD_LOG = {"train": [{"epoch": e, "train_loss": 2.0 / (e + 1), "learning_rate": 1e-5}
                       for e in range(2)],
             "validation": [{"epoch": e, "val_pooled_mrr": 0.2 + 0.05 * e, "val_pooled_top1": 0.1,
                             "val_pooled_top10": 0.4, "val_pooled_top3": 0.2,
                             "val_macro_recall@10": 0.3, "val_coverage": 1.0}
                            for e in range(2)]}


def test_a_bundle_run_is_listed_and_charted(root):
    """G2: a model bundle's run (``runs/gatv2/``) is a run the server lists,
    so its charts and CSV answer; before, only a ShepherdGNN conv type was."""
    run = root / "demo" / "runs" / "gatv2" / "20261028-100000_base_abcdef"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "train_20261028_100000.json").write_text(json.dumps(BASE_LOG))
    run_id = "demo/gatv2/20261028-100000_base_abcdef"
    assert tc.resolve_run(run_id) == run
    payload = tc.training_chart_series({"run": run_id, "rev": 6})
    assert payload["run"] == run_id and [c["id"] for c in payload["charts"]] == [
        "loss", "mrr", "hits", "lr"]
    assert tc.training_chart_csv({"run": run_id})["csv"].startswith("epoch,split,")


def test_the_charts_draw_what_a_base_and_a_task_model_log_and_no_empty_chart():
    """G3: the middle two charts follow the run's kind, by the keys its log
    holds: a base's LP accuracy, then its ROC-AUC and AP (and its LP loss as
    the validation loss); a task model's pooled MRR, then its pooled top-k by
    cut-off (no validation loss: train only). Every chart has a series."""
    base = {c["id"]: c for c in tc.training_charts(BASE_LOG)}
    assert base["mrr"]["title"] == "LP accuracy (validation, Trainer)"
    assert [s["label"] for s in base["mrr"]["series"]] == ["LP accuracy"]
    assert base["mrr"]["series"][0]["values"] == [0.5, 0.6, 0.7]
    assert base["hits"]["title"] == "ROC-AUC and AP (validation, Trainer)"
    assert [s["label"] for s in base["hits"]["series"]] == ["ROC-AUC", "AP"]
    assert [s["label"] for s in base["loss"]["series"]] == ["train", "validation"]
    child = {c["id"]: c for c in tc.training_charts(CHILD_LOG)}
    assert child["mrr"]["title"] == "Pooled MRR (validation, Trainer)"
    assert [s["label"] for s in child["mrr"]["series"]] == ["pooled MRR"]
    assert child["hits"]["title"] == "Pooled top-k (validation, Trainer)"
    assert [s["label"] for s in child["hits"]["series"]] == ["top-1", "top-3", "top-10"]
    assert [s["label"] for s in child["loss"]["series"]] == ["train"]
    for charts in (base, child):
        assert all(chart["series"] for chart in charts.values()), charts
    assert tc.validation_kind(BASE_LOG["validation"]) == "base"
    assert tc.validation_kind(CHILD_LOG["validation"]) == "finetune"
    assert tc.validation_kind([{"epoch": 0, "val_mrr": 0.1, "val_hits@1": 0.05,
                                "val_hits@10": 0.2}]) == "other"
    # Before the first validation the two charts wait, titled for any run.
    waiting = {c["id"]: c for c in tc.training_charts({"train": BASE_LOG["train"],
                                                       "validation": []})}
    assert waiting["mrr"]["title"] == "Validation figure (Trainer)"
    assert waiting["mrr"]["series"] == [] and waiting["loss"]["series"]


def test_a_plain_val_loss_log_draws_both_loss_series_and_its_values_unchanged(root):
    """R1 (the reviewer's K8 closeout review, adopted by the owner): a plain log
    with epoch, train_loss, learning_rate and val_loss (no model spec, no
    stage's selection figure, no MRR or Hits@K) still draws the Loss chart with
    its train and validation series, its values and its CSV as logged; nothing
    reads it as a run without validation. B69-T1: its middle two charts say the
    run logs only its loss (they read the generic "No other figure charted:
    Export CSV has every logged figure" and "No further figure charted")."""
    log = {"train": [{"epoch": 0, "train_loss": 1.0, "learning_rate": 0.001}],
           "validation": [{"epoch": 0, "val_loss": 0.8}]}
    charts = {c["id"]: c for c in tc.training_charts(log)}
    assert [s["label"] for s in charts["loss"]["series"]] == ["train", "validation"]
    assert charts["loss"]["x"] == [1]
    assert [s["values"] for s in charts["loss"]["series"]] == [[1.0], [0.8]]
    assert charts["lr"]["series"][0]["values"] == [0.001]
    assert tc.validation_kind(log["validation"]) == "loss"
    assert charts["mrr"]["series"] == [] and charts["hits"]["series"] == []
    assert charts["mrr"]["title"] == "Validation figure (Trainer)"
    assert charts["hits"]["title"] == "More validation figures (Trainer)"
    assert charts["mrr"]["empty"] == tc.LOSS_ONLY_LOGGED == (
        "This run logs only its loss: the Loss chart draws it.")
    assert charts["hits"]["empty"] == tc.NOT_IN_THE_LOG
    assert not [c["title"] for c in charts.values() if "charted" in c["title"]]
    # A base's LP loss alone, beside its epoch and time, is its loss alone too.
    assert tc.validation_kind([{"epoch": 0, "timestamp": "t", "val_lp_loss": 0.7}]) == "loss"
    assert tc._metrics_csv_text(log).splitlines() == [
        "epoch,split,learning_rate,train_loss,val_loss",
        "1,train,0.001,1.0,", "1,validation,,,0.8"]

    run = root / "demo" / "runs" / "gat" / "20261001-000000_plain_aaaaaa"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "train_20261001_000000.json").write_text(json.dumps(log))
    run_id = "demo/gat/20261001-000000_plain_aaaaaa"
    assert tc.resolve_run(run_id) == run, "a run under any directory of runs/ is listed"
    served = {c["id"]: c for c in tc.training_chart_series({"run": run_id, "rev": 2})["charts"]}
    assert [s["values"] for s in served["loss"]["series"]] == [[1.0], [0.8]]
    assert tc.training_chart_csv({"run": run_id})["csv"] == tc._metrics_csv_text(log)


def test_a_log_with_figures_this_build_does_not_chart_gets_the_generic_state():
    """N8 (TM28): the history-only MRR and Hits@K charts retired. A log whose
    validation entries hold only such figures gets one generic state: the
    middle two charts say no figure is charted and draw no series; its Loss
    chart and learning rate are as for any run, and the CSV still holds every
    key it logged."""
    log = {"train": [{"epoch": e, "train_loss": 1.0 - 0.1 * e, "learning_rate": 1e-3}
                     for e in range(2)],
           "validation": [{"epoch": e, "val_mrr": 0.1 * e, "val_hits@1": 0.05,
                           "val_hits@10": 0.2, "val_loss": 0.9 - 0.1 * e}
                          for e in range(2)]}
    assert tc.validation_kind(log["validation"]) == "other"
    charts = {c["id"]: c for c in tc.training_charts(log)}
    for chart_id in ("mrr", "hits"):
        assert charts[chart_id]["series"] == [] and charts[chart_id]["x"] == []
        assert "No" in charts[chart_id]["title"] and "charted" in charts[chart_id]["title"]
    assert [s["label"] for s in charts["loss"]["series"]] == ["train", "validation"]
    assert charts["lr"]["series"]
    header = tc._metrics_csv_text(log).splitlines()[0].split(",")
    assert {"val_mrr", "val_hits@1", "val_hits@10", "val_loss", "train_loss",
            "learning_rate"} <= set(header)


def test_the_series_function_returns_the_four_charts_of_a_listed_run(root):
    run_id = "demo/gatv2/20261018-142000_run_d41f8b"
    assert tc.resolve_run(run_id) == root / "demo" / "runs" / "gatv2" / "20261018-142000_run_d41f8b"
    payload = tc.training_chart_series([{"run": run_id, "rev": 12}])
    assert payload["run"] == run_id and payload["rev"] == 12
    by_id = {c["id"]: c for c in payload["charts"]}
    assert list(by_id) == ["loss", "mrr", "hits", "lr"]
    # A finetune logs no validation loss, so its Loss chart draws training
    # only, and says so in its title: a chart of one series draws no legend
    # key (R-54; the T06 review: its key "train" was the one left).
    assert [s["label"] for s in by_id["loss"]["series"]] == ["train"]
    assert by_id["loss"]["title"] == "Loss (train)"
    assert by_id["mrr"]["title"] == "Pooled MRR (validation, Trainer)"
    assert [s["label"] for s in by_id["hits"]["series"]] == ["top-1", "top-3", "top-10"]
    assert by_id["lr"]["yScale"] == "log" and by_id["loss"]["yScale"] == "linear"
    assert by_id["loss"]["x"] == [1, 2, 3, 4, 5, 6]
    assert by_id["lr"]["series"][0]["values"][0] == 0.0     # the browser draws it as a gap
    # R-54: the learning rate says its scale; the shared epoch axis is labelled
    # once, by the figure, not under each bottom chart.
    assert by_id["lr"]["title"] == "Learning rate (log scale)"
    assert payload["xLabel"] == tc.SHARED_AXIS == "Epoch (shared axis)"
    assert not [c for c in payload["charts"] if "xLabel" in c]
    json.dumps(payload)


def test_every_chart_of_one_series_is_titled_by_it():
    """R-54: a chart of one series draws no legend key (the charts module's
    `keyed`), so its title is what names the series: the Trainer's figure and
    the learning rate by their own titles; the Loss chart of a log with only
    the training loss "Loss (train)" (it kept a key "train" under "Loss", the
    T06 review), with both losses "Loss" and a key each; companions with one
    of theirs logged, that one ("ROC-AUC", "Pooled top-10")."""
    base = {"train": [{"epoch": 0, "total": 0.9, "learning_rate": 1e-3}],
            "validation": [{"epoch": 0, "val_lp_accuracy": 0.6, "val_lp_roc_auc": 0.7,
                            "val_lp_loss": 0.8}]}
    task = {"train": [{"epoch": 0, "train_loss": 0.9, "learning_rate": 1e-3}],
            "validation": [{"epoch": 0, "val_pooled_mrr": 0.2, "val_pooled_top10": 0.4}]}
    seen = {}
    for name, history in (("base", base), ("task", task)):
        for chart in tc.training_charts(history):
            seen[name, chart["id"]] = chart["title"]
            labels = [s["label"] for s in chart["series"]]
            if len(labels) == 1:
                assert labels[0].lower() in chart["title"].lower(), (name, chart)
    assert seen == {
        ("base", "loss"): "Loss", ("base", "mrr"): "LP accuracy (validation, Trainer)",
        ("base", "hits"): "ROC-AUC (validation, Trainer)", ("base", "lr"): "Learning rate (log scale)",
        ("task", "loss"): "Loss (train)", ("task", "mrr"): "Pooled MRR (validation, Trainer)",
        ("task", "hits"): "Pooled top-10 (validation, Trainer)",
        ("task", "lr"): "Learning rate (log scale)"}
    # Both companions logged: the chart's own title, a key each.
    both = {**base, "validation": [{**base["validation"][0], "val_lp_average_precision": 0.5}]}
    assert tc.training_charts(both)[2]["title"] == "ROC-AUC and AP (validation, Trainer)"


def test_a_top_k_that_appears_later_joins_its_chart():
    history = {"train": [], "validation": [{"epoch": 0, "val_pooled_mrr": 0.2,
                                            "val_pooled_top1": 0.1, "val_pooled_top10": 0.2},
                                           {"epoch": 1, "val_pooled_mrr": 0.2,
                                            "val_pooled_top1": 0.1, "val_pooled_top3": 0.15,
                                            "val_pooled_top10": 0.2}]}
    hits = tc.training_charts(history)[2]
    assert [s["label"] for s in hits["series"]] == ["top-1", "top-3", "top-10"]
    assert hits["series"][1]["values"] == [None, 0.15]


def test_export_csv_is_the_whole_log_as_text(root):
    out = tc.training_chart_csv({"run": "demo/gatv2/20261018-142000_run_d41f8b"})
    lines = out["csv"].splitlines()
    assert lines[0] == ("epoch,split,learning_rate,train_loss,val_pooled_mrr,"
                        "val_pooled_top1,val_pooled_top10,val_pooled_top3")
    assert len(lines) == 1 + 12 and lines[1].startswith("1,train,0.0,")
    assert out["filename"] == "training_metrics_gatv2_20261018-142000_run_d41f8b.csv"


def test_the_chart_token_counts_rows_again_only_when_the_log_changed(root, monkeypatch):
    ws = tc.workspace_state("demo")
    cache = {}
    run = tc._current_run({"status": "idle"}, ws, cache)
    reads = []
    original = tc.training_manager.get_metrics_history

    def counted(run_dir=None):
        reads.append(run_dir)
        return original(run_dir)

    monkeypatch.setattr(tc.training_manager, "get_metrics_history", counted)
    assert tc._chart_token(run, cache, operator=True) == {"run": run["id"], "rev": 12}
    assert tc._chart_token(run, cache, operator=False) == {"run": run["id"], "rev": 12}
    assert len(reads) == 1
    log = run["path"] / "logs" / "train_20261018_142000.json"
    log.write_text(json.dumps(_metrics(7)))
    assert tc._chart_token(run, cache, operator=True)["rev"] == 14 and len(reads) == 2
    assert tc._chart_token(None, cache, operator=True) == {"run": None, "rev": 0,
                                                           "empty": tc.NO_RUN_YET}


def test_the_used_dataset_fingerprint_is_an_operators_and_compared_in_full():
    ws = {"row": {"dataset_fingerprint": "a" * 64}, "digits": 12}
    run = tc.RunRef(id="demo/gatv2/r", label="gatv2/r", path=Path("/x"), workspace="demo",
                    latest=True)
    same = tc.run_head_html(run, ws, operator=True,
                            snapshot={"readable": True, "dataset_fingerprint": "a" * 64})
    # The tag; Details says the relation, not the run bar's fingerprint again (R-41).
    assert "same dataset as the run bar" in same and "a" * 12 not in same
    assert "<dt class=\"shep-k\">Dataset used</dt><dd class=\"shep-v\">same as the run bar</dd>" in same
    # By the shared OK mark, never a check glyph of the page's own (B5), and
    # quiet: a normal state is the mark and its word, never a tag (R-57).
    assert r.status_mark("ok", "same dataset as the run bar") in same
    assert "shep-tag" not in same
    other = tc.run_head_html(run, ws, operator=True,
                             snapshot={"readable": True, "dataset_fingerprint": "a" * 12 + "b" * 52})
    # 12 equal digits are not equal: a condition to watch, by the one mark (▲, amber).
    assert tc.r.warn_chip("not the selected dataset") in other
    assert r.fingerprint_value("a" * 12 + "b" * 52, what="dataset fingerprint") in other
    unreadable = tc.run_head_html(run, ws, operator=True, snapshot={"readable": False})
    assert tc.r.warn_chip("checkpoint cannot be read") in unreadable
    assert "recorded at the first checkpoint" in tc.run_head_html(run, ws, operator=True,
                                                                  snapshot=None, live=True)
    # "dataset not recorded" only when the run recorded none (the WebUI skill
    # §6); a run that recorded one beside a run-bar dataset with no
    # fingerprint cannot be compared, which is another state.
    unrecorded = tc.run_head_html(run, ws, operator=True,
                                  snapshot={"readable": True, "dataset_fingerprint": None})
    assert "dataset not recorded" in unrecorded
    bare = tc.run_head_html(run, {"row": {}, "digits": 12}, operator=True,
                            snapshot={"readable": True, "dataset_fingerprint": "a" * 64})
    assert "dataset not recorded" not in bare
    assert "dataset cannot be verified: the run bar&#x27;s dataset has no fingerprint" in bare
    read_only = tc.run_head_html(run, ws, operator=False,
                                 snapshot={"readable": True, "dataset_fingerprint": "a" * 64})
    assert "dataset not shown · operator control" in read_only
    assert "not shown: reading the checkpoint is an operator control" in read_only
    assert "a" * 12 not in read_only


def test_the_snapshot_sentence_names_no_path_to_a_read_only_request(root):
    run = tc.RunRef(id="demo/gatv2/r", label="gatv2/r", path=root / "demo" / "runs" / "gatv2" / "r",
                    workspace="demo", job=True, latest=False)
    (run["path"] / "checkpoints").mkdir(parents=True)       # a run that saved its last.pt
    snapshot = {"checkpoint": str(run["path"] / "checkpoints" / "last.pt"), "readable": True,
                "completed_epochs": 60, "target_epochs": 60, "end": "finished"}
    status = {"status": "completed", "current_epoch": 60, "total_epochs": 60}
    operator = tc.run_snapshot_html(run, status, operator=True, snapshot=snapshot)
    assert "Finished · 60 of 60 epochs completed. To train further, use Continue…" in operator
    assert str(root) not in operator and "last.pt" not in operator     # the path: Details
    ws = {"row": {"dataset_fingerprint": "a" * 64}, "digits": 12}
    head = tc.run_head_html(run, ws, operator=True, snapshot=snapshot)
    assert "<code>demo/runs/gatv2/r</code>" in head and str(root) not in head
    assert "<code>checkpoints/last.pt</code>" in head
    read_only = tc.run_snapshot_html(run, status, operator=False, snapshot=None)
    assert "Completed · epoch 60 of 60. Checkpoint details are an operator control" in read_only
    early = {**snapshot, "end": "stopped_early", "completed_epochs": 40, "target_epochs": 60}
    assert "Stopped early · 40 of 60 epochs completed. It cannot be continued." in tc.run_snapshot_html(
        run, status, operator=True, snapshot=early)
    assert "Stopped early (patience) · " in tc.run_snapshot_html(
        run, status, operator=True, snapshot={**early, "stop_reason": "patience"})
    assert "last.pt" not in read_only
    assert "last.pt" not in tc.run_head_html(run, ws, operator=False, snapshot=None)


def test_the_state_line_has_one_separator_and_no_continue_hint_with_the_drawer_open(root):
    """B72-RV1 and NEW-training-finetune-3: the finished and stopped-early lines
    carried two " · " ("Finished · 1 of 1 epochs completed · to train further,
    use Continue…") and said "use Continue…" while the Continue drawer was
    open and its button hidden. NEW-training-finetune-4: "1 of 1 epochs"."""
    run = tc.RunRef(id="demo/gatv2/r", label="gatv2/r", path=root / "demo" / "runs" / "gatv2" / "r",
                    workspace="demo", job=False, latest=True)
    run["path"].mkdir(parents=True)
    status = {"status": "idle"}
    one = {"readable": True, "completed_epochs": 1, "target_epochs": 1, "end": "finished"}
    lines = {
        "finished": tc.run_snapshot_html(run, status, operator=True, snapshot=one),
        "finished, drawer open": tc.run_snapshot_html(run, status, operator=True, snapshot=one,
                                                      drawer=True),
        "stopped early": tc.run_snapshot_html(run, status, operator=True, snapshot={
            **one, "end": "stopped_early", "completed_epochs": 40, "target_epochs": 60,
            "stop_reason": "patience"}),
        "paused": tc.run_snapshot_html(run, status, operator=True, snapshot={
            **one, "end": "paused", "completed_epochs": 3, "target_epochs": 60, "resumable": True}),
        "read-only": tc.run_snapshot_html({**run, "job": True}, {
            "status": "completed", "current_epoch": 1, "total_epochs": 1}, operator=False,
            snapshot=None),
    }
    for name, line in lines.items():
        assert line.count(" · ") <= 1, (name, line)
    assert lines["finished"] == ('<span class="shep-rv-snap">Finished · 1 of 1 epoch completed. '
                                 'To train further, use Continue…</span>')
    assert lines["finished, drawer open"] == ('<span class="shep-rv-snap">Finished · 1 of 1 epoch '
                                              'completed</span>')
    assert "Continue…" not in lines["finished, drawer open"]
    assert "Stopped early (patience) · 40 of 60 epochs completed. It cannot be continued." in \
        lines["stopped early"]
    assert "3 of 60 epochs completed · can be continued" in lines["paused"]


def test_epochs_done_says_one_epoch_and_more_epochs():
    """NEW-training-finetune-4: '1 of 1 epochs completed'; the unit's number
    follows the total, and with no total the one epoch words of every page."""
    done = tc._epochs_done
    assert done({"completed_epochs": 1, "target_epochs": 1}) == "1 of 1 epoch completed"
    assert done({"completed_epochs": 1, "target_epochs": 60}) == "1 of 60 epochs completed"
    assert done({"completed_epochs": 3, "target_epochs": 3}) == "3 of 3 epochs completed"
    assert done({"completed_epochs": 1, "target_epochs": None}) == "1 epoch completed"
    assert done({"completed_epochs": 5}) == "5 epochs completed"
    assert done({"completed_epochs": None, "target_epochs": None}) == "epochs not recorded"


def test_a_run_that_kept_only_best_pt_names_its_dataset_and_its_state(root):
    """B69-T2: the run view read ``last.pt`` alone, so a run that kept only a
    ``best.pt`` (which its Continue drawer lists) read "dataset not recorded
    (no checkpoint)" and "No checkpoint saved, so nothing can be continued".
    The snapshot is the run's latest saved file (`checkpoint_paths.latest_saved`:
    its ``last.pt``, else its newest file), read once per change of it."""
    from src.utils.checkpoint_io import write_checkpoint

    run = tc.RunRef(id="demo/gatv2/b", label="gatv2/b", path=root / "demo" / "runs" / "gatv2" / "b",
                    workspace="demo", job=False, latest=True)
    best = run["path"] / "checkpoints" / "best.pt"
    write_checkpoint({"checkpoint_kind": "training_state", "epoch": 3,
                      "training_input_digests": {"split_manifest": "d" * 64},
                      "progress": {"completed_epochs": 4, "end": "finished",
                                   "stage": {"start_epoch": 0, "target_epochs": 4}}}, best)
    cache = {}
    snapshot = tc._snapshot_for(run, cache)
    assert snapshot["checkpoint"] == str(best) and snapshot["dataset_fingerprint"] == "d" * 64
    ws = {"row": {"dataset_fingerprint": "d" * 64}, "digits": 12}
    head = tc.run_head_html(run, ws, operator=True, snapshot=snapshot)
    assert "same dataset as the run bar" in head and "no checkpoint" not in head
    line = tc.run_snapshot_html(run, {"status": "idle"}, operator=True, snapshot=snapshot)
    assert "Finished · 4 of 4 epochs completed" in line and "No checkpoint" not in line
    assert not tc._snapshot_sentence(snapshot).startswith("No checkpoint")
    # A last.pt written since is the one read.
    last = best.with_name("last.pt")
    write_checkpoint({"checkpoint_kind": "training_state", "epoch": 4,
                      "training_input_digests": {"split_manifest": "e" * 64},
                      "progress": {"completed_epochs": 5, "end": "paused",
                                   "stage": {"start_epoch": 0, "target_epochs": 6}}}, last)
    again = tc._snapshot_for(run, cache)
    assert again["checkpoint"] == str(last) and again["dataset_fingerprint"] == "e" * 64
    # No file at all is still "no checkpoint".
    empty = tc.RunRef(run, id="demo/gatv2/e", path=root / "demo" / "runs" / "gatv2" / "e")
    (empty["path"] / "checkpoints").mkdir(parents=True)
    assert tc._snapshot_for(empty, {}) is None


def test_a_best_pt_saved_mid_run_is_said_as_that_files_state_not_the_runs(root):
    """The T06 review of B69-T2: a run that trained 4 epochs and kept only a
    ``best.pt`` saved at epoch 2 (a real best or top-k file records no end
    mid-run) read "2 of 4 epochs completed · can be continued", as if the run
    had stopped at epoch 2, while its charts showed 4. The state line names
    the file it read first; a run's own ``last.pt`` is said as before."""
    from src.utils.checkpoint_io import write_checkpoint

    run = tc.RunRef(id="demo/gatv2/m", label="gatv2/m", path=root / "demo" / "runs" / "gatv2" / "m",
                    workspace="demo", job=False, latest=True)
    best = run["path"] / "checkpoints" / "best.pt"
    write_checkpoint({"checkpoint_kind": "training_state", "epoch": 1,
                      "training_input_digests": {"split_manifest": "d" * 64},
                      "progress": {"completed_epochs": 2, "end": None,
                                   "stage": {"start_epoch": 0, "target_epochs": 4}}}, best)
    snapshot = tc._snapshot_for(run, {})
    assert snapshot["checkpoint"] == str(best) and snapshot["end"] is None
    line = tc.run_snapshot_html(run, {"status": "idle"}, operator=True, snapshot=snapshot)
    assert "best.pt: 2 of 4 epochs completed · can be continued" in line, line
    assert line.count(" · ") == 1
    # The run's own last save names no file: its state is the run's.
    last = best.with_name("last.pt")
    write_checkpoint({"checkpoint_kind": "training_state", "epoch": 3,
                      "training_input_digests": {"split_manifest": "d" * 64},
                      "progress": {"completed_epochs": 4, "end": "finished",
                                   "stage": {"start_epoch": 0, "target_epochs": 4}}}, last)
    own = tc.run_snapshot_html(run, {"status": "idle"}, operator=True,
                               snapshot=tc._snapshot_for(run, {}))
    assert "Finished · 4 of 4 epochs completed" in own and ".pt" not in own, own


def test_the_run_details_name_the_folder_once_the_model_and_the_log_with_no_bubble(root):
    """R-41: the Run details popover named the run's folder twice (Run, and
    inside Checkpoint's path), said Logs as a sentence ("the run directory's
    logs/"), repeated the run bar's dataset fingerprint with Copy beside a
    header that already says "same dataset as the run bar", carried an ⓘ
    (a bubble inside a popover) and had no model fingerprint. Now: the folder
    once, the checkpoint and the log by their place in it, the model's 12 hex
    digits with Copy (the snapshot's own read), the dataset "same as the run
    bar" (or its fingerprint where it is not), and no help bubble."""
    from src.utils.checkpoint_io import write_checkpoint

    run = tc.RunRef(id="demo/gatv2/d", label="gatv2/d", path=root / "demo" / "runs" / "gatv2" / "d",
                    workspace="demo", job=False, latest=True)
    digest = write_checkpoint({"checkpoint_kind": "training_state", "epoch": 0,
                               "training_input_digests": {"split_manifest": "a" * 64},
                               "progress": {"completed_epochs": 1, "end": "finished",
                                            "stage": {"start_epoch": 0, "target_epochs": 1}}},
                              run["path"] / "checkpoints" / "last.pt")
    (run["path"] / "logs").mkdir()
    (run["path"] / "logs" / "train_20261116_204731.json").write_text(json.dumps(_metrics(1)))
    snapshot = tc._snapshot_for(run, {})
    assert snapshot["sha256"] == digest
    ws = {"name": "demo", "row": {"dataset_fingerprint": "a" * 64}, "digits": 12}
    head = tc.run_head_html(run, ws, operator=True, snapshot=snapshot)
    details = head.split('id="shep-train-run-details"', 1)[1]
    rows = dict(re.findall(r'<dt class="shep-k">([^<]+)</dt><dd class="shep-v">(.*?)</dd>', details))
    assert list(rows) == ["Run folder", "Checkpoint", "Model fingerprint", "Logs", "Dataset used"]
    assert rows["Run folder"] == "<code>demo/runs/gatv2/d</code>"
    assert details.count("demo/runs/gatv2/d") == 1
    assert rows["Checkpoint"] == "<code>checkpoints/last.pt</code>"
    assert rows["Model fingerprint"] == r.fingerprint_value(digest, what="model fingerprint")
    assert digest[:12] in rows["Model fingerprint"] and "data-copy" in rows["Model fingerprint"]
    assert rows["Logs"] == "<code>logs/train_20261116_204731.json</code>"
    assert rows["Dataset used"] == "same as the run bar" and "a" * 12 not in details
    assert "data-help" not in details and "shep-help" not in head
    assert "training-used-dataset" not in tc.TOPIC_HELP
    # Another dataset: its fingerprint, once, with Copy.
    other = tc.run_head_html(run, {**ws, "row": {"dataset_fingerprint": "b" * 64}}, operator=True,
                             snapshot=snapshot)
    assert other.count(f'<code class="shep-fpv" title="{"a" * 64}">') == 1
    assert f'data-copy="{"a" * 64}"' in other
    # No log yet: said, not left blank.
    (run["path"] / "logs" / "train_20261116_204731.json").unlink()
    assert '<span class="shep-muted">none written</span>' in tc.run_head_html(
        run, ws, operator=True, snapshot=snapshot)


def test_the_runs_checkpoint_is_read_again_only_when_it_changed(root, monkeypatch):
    run = tc.RunRef(id="demo/gatv2/r", label="gatv2/r", path=root / "demo" / "runs" / "gatv2" / "r",
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
def _action(key, interactive, allowed=True):
    """A run-bar action's piece as the render gives it (`chips.gated`)."""
    from src.webui.components.chips import gated

    return gated(tc.ACTION_LABELS[key], allowed, interactive=interactive)


def _rendered(**pieces):
    base = {key: f"<{key}>" for key in tc.VIEW_KEYS}
    base.update(dataset=((("a", "a"),), "a", True), start=_action("start", True),
                cont=_action("cont", True), stop=_action("stop", False),
                resume=_action("resume", False), runbar=("shep-runbar",), page=("shep-page",),
                run_name=True, view=((("r", "a/gatv2/r"),), "a/gatv2/r"),
                charts={"run": None, "rev": 0}, ws={"name": "a"}, training_workspace="a")
    base.update(pieces)
    return {"pieces": base, "cache": {}, "ws_changed": False}


def test_a_tick_sends_only_what_changed_and_everything_every_few_ticks():
    first = tc.view_updates(_rendered(), {})
    cache = first[-1]
    again = tc.view_updates(_rendered(), cache)
    assert all(u == gr.update() for u in again[:-1])
    changed = tc.view_updates(_rendered(pill="<other>", start=_action("start", False)), cache)
    by_key = dict(zip(tc.VIEW_KEYS, changed[:-1]))
    assert by_key["pill"] == "<other>"
    assert by_key["start"] == gr.update(value="▸ Start", interactive=False)
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


def test_a_tick_writes_no_piece_and_hands_every_change_over(
        root, monkeypatch, operator):
    """P2-NEW-1 and B46-races: the fast tick writes the chips and its own
    record, never a piece (the status pill included: a tick that crossed a
    render brought back the pill it read before) and never the cache. It
    hands a change over only when what it renders now is not what the
    session was last sent: nothing after a render; new epochs of the run
    shown, yes."""
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    demo, _ctx, by_id = _page()
    poll = next(fn for fn in demo.fns.values() if fn.name == "_poll")
    assert poll.outputs[0] is by_id[tc.CHANGED_ID] and "pill" in tc.HANDED_KEYS
    cache_state = poll.inputs[3]                               # the renders' record
    assert isinstance(cache_state, gr.State) and cache_state not in poll.outputs
    assert not {by_id[k] for k in ("shep-train-dataset", "shep-train-chip", "shep-train-start",
                                   "shep-train-charts", "shep-train-run-head",
                                   "shep-train-pill")} & set(poll.outputs)
    ws = tc.workspace_state("demo")
    cache = tc._render_view(ws, {}, {}, [], {}, operator)[-1]
    handoff, _chips, polled = tc._poll(shell.TRAINING, ws, {}, cache, [], {}, {}, operator)
    assert handoff == gr.update()
    again = tc._poll(shell.TRAINING, ws, {}, cache, [], {}, polled, operator)
    assert again[0] == gr.update()
    log = root / "demo" / "runs" / "gatv2" / "20261018-142000_run_d41f8b" / "logs" / "train_20261018_142000.json"
    log.write_text(json.dumps(_metrics(7)))                    # a new epoch, nobody acted
    handoff, _chips, _polled = tc._poll(shell.TRAINING, ws, {}, cache, [], {}, polled, operator)
    assert handoff["changed"] == ["charts"]


def test_the_handed_over_render_sends_what_changed_and_everything_every_few(
        root, monkeypatch, operator):
    """The render a tick hands a change over to sends only what changed, and
    everything every `FULL_SEND_TICKS` hand-offs; the render after an action
    sends everything and starts the count again."""
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    ws = tc.workspace_state("demo")
    cache = tc._render_view(ws, {}, {}, [], {}, operator)[-1]
    sent_whole = []
    for _ in range(tc.FULL_SEND_TICKS * 2):
        out = tc._render_changed(ws, {}, cache, [], {}, {}, operator)
        cache = out[-1]
        sent_whole.append(dict(zip(tc.VIEW_KEYS, out[:-1], strict=True))["chip"] != gr.update())
    assert [i + 1 for i, whole in enumerate(sent_whole) if whole] == [
        tc.FULL_SEND_TICKS, tc.FULL_SEND_TICKS * 2]
    cache = tc._render_view(ws, {}, cache, [], {}, operator)[-1]
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
    poll_fn = next(fn for fn in demo.fns.values() if fn.fn is tc._poll)
    assert poll_fn.outputs[1] is ctx.chips
    ws = tc.workspace_state("demo")
    first = tc._poll(shell.TRAINING, ws, {}, {}, [], {}, {}, operator)
    assert first[1] == "<chips>" and len(read) == 1         # nothing seen before: read
    same = tc._poll(shell.TRAINING, ws, {}, {}, [], {}, first[-1], operator)
    assert same[1] == gr.update() and len(read) == 1         # unchanged: nothing sent
    status["gpu"] = {"job": {"job_type": "test", "status": "running",
                             "start_time": "2026-10-21T18:25:13"}}
    started = tc._poll(shell.TRAINING, ws, {}, {}, [], {}, same[-1], operator)
    assert started[1] == "<chips>" and len(read) == 2        # a test run took the slot
    later = tc._poll(shell.TRAINING, ws, {}, {}, [], {}, started[-1], operator)
    assert later[1] == gr.update() and len(read) == 2        # its elapsed time is no change


def test_a_running_jobs_dataset_is_shown_and_the_select_disabled(root, monkeypatch, operator):
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    job_run = root / "resampled" / "runs" / "gatv2" / "20261020-090000_run_bbbbbb"
    job_run.mkdir(parents=True)
    monkeypatch.setattr(tc.training_manager, "run_dir", job_run)
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: {"status": "running", "job_id": "j", "current_epoch": 1,
                                 "total_epochs": 4, "gpu": {"job": {"job_type": "training"}}})
    rendered = tc.render(tc.workspace_state("demo"), {}, {}, [], operator)
    assert rendered["ws_changed"] is True and rendered["pieces"]["ws"]["name"] == "resampled"
    assert rendered["pieces"]["dataset"][1:] == ("resampled", False)
    assert "in use by the running job" in rendered["pieces"]["chip"]
    assert rendered["pieces"]["stop"] == {"value": "■ Stop", "interactive": True}
    assert rendered["pieces"]["start"] == {"value": "▸ Start", "interactive": False}


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
    job_run = root / "resampled" / "runs" / "gatv2" / "20261020-090000_run_bbbbbb"
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
        "shep-train-page", "shep-train-run-head", "shep-train-view-run", "shep-train-charts",
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
    assert poll.outputs == [changed, ctx.chips, polled]
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
    than the session's writes no selection (it writes no piece at all)
    and gives the hand-off token a new value each time, naming the change;
    the queued render follows the job. With the job no longer running and
    the page as the session was sent it, a tick hands nothing over and a
    render writes no selection."""
    job = _job_in_resampled(root, monkeypatch)
    ws = tc.workspace_state("demo")
    cache = tc._render_view(ws, {}, {}, [], {}, operator)[-1]
    job.status = dict(RUNNING)
    out = tc._poll(shell.TRAINING, ws, {}, cache, [], {}, {}, operator)
    assert len(out) == 3
    handoff = out[0]
    assert "ws" in handoff["changed"] and "dataset" in handoff["changed"]
    again = tc._poll(shell.TRAINING, ws, {}, cache, [], {}, out[-1], operator)
    assert again[0]["at"] != handoff["at"], "each hand-off is a new value"
    view = dict(zip(tc.VIEW_KEYS, tc._render_changed(ws, {}, cache, [], {}, out[-1], operator)[:-1],
                    strict=True))
    assert view["ws"]["name"] == "resampled" == view["training_workspace"]
    job.status = dict(ENDED)
    rendered = tc._render_view(ws, {}, {}, [], {}, operator)
    view = dict(zip(tc.VIEW_KEYS, rendered[:-1], strict=True))
    assert view["ws"] == gr.update() and view["training_workspace"] == gr.update()
    assert tc._poll(shell.TRAINING, ws, {}, rendered[-1], [], {}, {}, operator)[0] == gr.update()


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
        assert browser.wait(browser.join(_after(demo, opened, "_render_view"), [None] * 5))
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
        assert browser.wait(browser.join(_after(demo, from_datasets, "_render_view"), [None] * 5))
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


#: The Training page's session state and every piece drawn from it, by
#: elem_id where it has one (B46-races): the run bar and the run view (the
#: pill and a failed job's banner included) and the Continue drawer's list,
#: choice and check.
DRAWN_FROM_THE_SELECTION = (
    "shep-train-dataset", "shep-train-chip", "shep-train-pill", "shep-train-reason",
    "shep-train-start", "shep-train-continue", "shep-train-stop", "shep-train-resume",
    "shep-train-runbar", "shep-train-page", "shep-train-banner", "shep-train-run-head",
    "shep-train-view-run", "shep-train-charts", "shep-train-drawer-why",
    "shep-train-parents-note", "continue_run", "checkpoint_dropdown", "manual_checkpoint",
    "shep-train-parent-details", "shep-train-compare", "continue_preview", "extend_epochs",
    "extend_lr", "shep-train-drawer-notice", "shep-train-branch", "continue_operation")


def test_every_writer_of_the_selection_or_a_piece_drawn_from_it_is_in_the_group():
    """B46-races, the contract: the session's selection (``ws``, ``mode``,
    the run viewed's choice, the banners dismissed, what it was sent) and
    every piece drawn from it, the status pill, a failed job's banner and the
    Continue drawer's list, choice and check included, have one kind of
    writer: a queued listener of `shell.TRAINING_SELECTION_CONCURRENCY`, one
    at a time, first asked first run. So no answer drawn from an older
    selection or an older read of the job is applied after a newer one. The
    fast poll writes none of them (only the chips, its own record and the
    hand-off). On the old wiring the poll wrote the pill, Dismiss the banner,
    and the drawer's listing, its Refresh and its check ran each in its own
    group."""
    demo, ctx, named, by_id = _datasets_and_training()
    poll = named["_poll"]
    ws, mode, cache, dismissed, viewing = poll.inputs[1:6]
    session = {ws, ctx.training_workspace, mode, cache, dismissed, viewing}
    drawn = {by_id[elem_id] for elem_id in DRAWN_FROM_THE_SELECTION}
    writers = [fn for fn in demo.fns.values() if (session | drawn) & set(fn.outputs)]
    outside = sorted({fn.name for fn in writers
                      if (fn.concurrency_id, fn.concurrency_limit)
                      != (shell.TRAINING_SELECTION_CONCURRENCY, 1) or fn.queue is False})
    assert outside == [], outside
    assert not (session | drawn) & set(poll.outputs)


def _waiting_in_group(demo, event_id) -> bool:
    """Whether the queued event `event_id` waits in the run bar's group."""
    group = demo._queue.event_queue_per_concurrency_id.get(shell.TRAINING_SELECTION_CONCURRENCY)
    return group is not None and any(event._id == event_id for event in list(group.queue))


def test_a_tick_that_crosses_a_render_never_brings_back_an_older_pill(root, monkeypatch):
    """B46-races, through the page served by Gradio, one session: the fast
    tick's read of the job runner is held (the job running); meanwhile the
    job ends and a render draws the page (its pill "Completed"); released,
    the tick answers. The browser applies the answers as they come: on the
    old wiring the tick's own pill ("Running", read before the render's) was
    applied last and stayed until the next tick. Now the tick writes no
    pill; a change it finds is handed over, and the render it runs draws the
    job as it is then."""
    job = _job_in_resampled(root, monkeypatch)
    demo, ctx, named, by_id = _datasets_and_training()
    poll, opened = named["_poll"], named["_open"]
    render = _after(demo, opened, "_render_view")
    changed = by_id[tc.CHANGED_ID]
    hand_over = next(fn for fn in demo.fns.values() if (changed._id, "change") in fn.targets)
    with served(demo) as base:
        browser = ServedSession(base, demo)
        assert browser.wait(browser.join(opened, [None, None]))
        browser.set_state(ctx.location, shell.TRAINING)
        job.status = dict(RUNNING)
        first = browser.join(render, [None] * len(render.inputs))
        assert browser.wait(first)
        job.hold.set()
        answers = []
        ticking = threading.Thread(
            target=lambda: answers.append(browser.predict(poll, [None] * len(poll.inputs))),
            daemon=True)
        ticking.start()
        assert job.entered.wait(30), "the tick's read did not start"
        job.status = dict(ENDED)
        later = browser.join(render, [None] * len(render.inputs))
        assert browser.wait(later)
        job.release.set()
        ticking.join(30)
        (tick,) = answers
        shown = _shown([_answered(browser, render, first), _answered(browser, render, later),
                        (poll, tick)])
        assert 'data-state="completed"' in shown["shep-train-pill"]["value"], \
            "the tick brought back the pill it read before the render's"
        # Its hand-off, if any, runs a render that reads the job at its turn.
        token = tick[poll.outputs.index(changed)]
        if isinstance(token, dict) and token.get("__type__") != "update":
            handed = browser.join(hand_over, [None] * len(hand_over.inputs))
            assert browser.wait(handed)
            shown = _shown([_answered(browser, hand_over, handed)], shown)
        assert 'data-state="completed"' in shown["shep-train-pill"]["value"]
        browser.close()


def test_a_workspace_chosen_while_the_drawer_lists_never_shows_the_previous_list(
        root, monkeypatch):
    """B46-races, through the pages served by Gradio, one session: Continue…
    is opened on ``demo`` and its listing is held; meanwhile ``resampled``
    is chosen in the run bar, and the change of the session's workspace and
    the drawer's relisting follow (asked as the browser asks them). On the old
    wiring the listing ran in a group of its own: the relisting ran first
    (the drawer not yet open, it listed nothing) and the held listing, let go,
    was applied last, so the drawer listed ``demo``'s runs under a run bar on
    ``resampled``. In the run bar's group every one of them runs in the order
    asked: the drawer lists ``resampled``."""
    job = _job_in_resampled(root, monkeypatch)
    operator_mode(monkeypatch)
    held, entered = threading.Event(), threading.Event()

    def listing(ws):
        name = (ws or {}).get("name")
        if name == "demo":
            entered.set()
            assert held.wait(30), "never let go"
        return {}, f"listed {name}"

    monkeypatch.setattr(tc, "_list_parents", listing)
    demo, ctx, named, by_id = _datasets_and_training()
    opened, choose, follow = named["_open"], named["_on_dataset"], named["_on_training_workspace"]
    cont = named["_on_continue_open"]
    relist = _after(demo, follow, "_relist_parents")
    del job
    with served(demo) as base:
        browser = ServedSession(base, demo)
        assert browser.wait(browser.join(opened, [None, None]))
        assert browser.wait(browser.join(choose, ["demo", None]))
        opening = browser.join(cont, [None] * len(cont.inputs))
        assert entered.wait(30), "the listing did not start"
        asked = [(choose, browser.join(choose, ["resampled", None])),
                 (follow, browser.join(follow, [None] * len(follow.inputs))),
                 (relist, browser.join(relist, [None] * len(relist.inputs)))]
        last = asked[-1][1]
        # The listing is let go once the relisting has run (each in a group of
        # its own) or waits behind it in the run bar's group.
        for _ in range(3000):
            if last in browser.done or _waiting_in_group(demo, last):
                break
            threading.Event().wait(0.01)
        held.set()
        assert browser.wait(opening, *(event for _fn, event in asked))
        fns = {opening: cont, **{event: fn for fn, event in asked}}
        shown = _shown([_answered(browser, fns[event], event) for event in browser.order
                        if event in fns])
        assert browser.state(named["_poll"].inputs[1])["name"] == "resampled"
        assert "listed resampled" in shown["shep-train-parents-note"]["value"], \
            "the drawer listed the previous workspace's runs"
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
        assert browser.wait(browser.join(_after(demo, from_datasets, "_render_view"), [None] * 5))
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
    assert page.shown("shep-train-charts")["run"] == "resampled/gatv2/20261020-090000_run_bbbbbb"
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
    job_run = root / "resampled" / "runs" / "gatv2" / "20261020-090000_run_bbbbbb"
    (job_run / "logs").mkdir()
    log = job_run / "logs" / "train_20261020_090000.json"
    log.write_text(json.dumps(_metrics(1)))
    fingerprint = tc.workspace_state("resampled")["row"]["dataset_fingerprint"]
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
    assert "same as the run bar</dd>" in head and "same dataset as the run bar" in head
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
    assert page.shown("shep-train-charts")["run"] == "demo/gatv2/20261018-142000_run_d41f8b"
    clock.ahead += tc.LATEST_READ_SECONDS + 0.5
    assert page.tick() == []                             # the tick lists the runs again
    clock.ahead += 1
    new = root / "demo" / "runs" / "gatv2" / "20261023-120000_run_cccccc"
    (new / "checkpoints").mkdir(parents=True)
    (new / "checkpoints" / "model-1-0.0100.pt").write_bytes(b"x")
    (new / "logs").mkdir()
    (new / "logs" / "train_20261023_120000.json").write_text(json.dumps(_metrics(1)))
    page.pick("demo")                                    # its render lists them again
    assert page.shown("shep-train-charts")["run"] == "demo/gatv2/20261023-120000_run_cccccc"
    handed = page.tick()
    assert page.shown("shep-train-charts")["run"] == "demo/gatv2/20261023-120000_run_cccccc", \
        "the tick's hand-off drew the previous run from an older listing"
    assert handed == []
    # Positive control: another run written while nobody acts; the tick's own
    # listing, newer than the render's once it lists again, is handed over.
    newest = root / "demo" / "runs" / "gatv2" / "20261023-130000_run_eeeeee"
    (newest / "checkpoints").mkdir(parents=True)
    (newest / "checkpoints" / "model-1-0.0100.pt").write_bytes(b"x")
    clock.ahead += tc.LATEST_READ_SECONDS + 0.5
    assert "charts" in page.tick()
    assert page.shown("shep-train-charts")["run"] == "demo/gatv2/20261023-130000_run_eeeeee"
    assert page.tick() == []


def test_a_read_only_request_sees_every_action_disabled_and_no_checkpoint(root, monkeypatch):
    # The run shown has a saved state, which an operator's render reads.
    run = root / "demo" / "runs" / "gatv2" / "20261018-142000_run_d41f8b"
    (run / "checkpoints" / "last.pt").write_bytes(b"saved state")
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    reads = []
    monkeypatch.setattr(tc.training_manager, "latest_snapshot",
                        lambda *a, **k: reads.append(a) or {"readable": True})
    operator_mode(monkeypatch)
    tc.render(tc.workspace_state("demo"), {}, {}, [], gradio_request())
    assert len(reads) == 1
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    monkeypatch.setattr(tc.training_manager, "latest_snapshot",
                        lambda *a, **k: pytest.fail("a read-only request read the checkpoint"))
    pieces = tc.render(tc.workspace_state("demo"), {}, {}, [], READ_ONLY)["pieces"]
    assert not pieces["run_name"]
    # Refused: every gated action disabled, with the lock, beside D8's reason
    # (the one gated-button rule, the WebUI skill §3).
    for key, label in tc.ACTION_LABELS.items():
        assert pieces[key] == {"value": f"🔒 {label}", "interactive": False}, key
    assert "Operator controls only: deployment mode not set to M1" in pieces["reason"]
    assert "dataset not shown · operator control" in pieces["head"]
    # The run's folder is a listing's fact (R-41: named once, in Details); its
    # checkpoint and model fingerprint are an operator's.
    assert "checkpoints/" not in pieces["head"] and "Model fingerprint" not in pieces["head"]
    assert "last.pt" not in pieces["head"]


# --------------------------------------------------------------------------- the banner
def test_the_banner_is_dismissed_for_this_session_until_its_content_changes(root, monkeypatch,
                                                                            operator):
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: {"status": "failed", "error_message": "Process exited with code 1",
                                 "gpu": {"job": None}})
    rendered = tc.render(tc.workspace_state("demo"), {}, {}, [], operator)
    assert "Process exited with code 1" in rendered["pieces"]["banner"]
    key = rendered["cache"]["banner_key"]

    class Evt:
        _data = {"act": "dismiss"}

    dismissed = tc._on_banner_click([], rendered["cache"], Evt())
    assert dismissed == [key]
    # The render after the click, the banner's one writer, hides it.
    again = tc.render(tc.workspace_state("demo"), {}, {}, dismissed, operator)
    assert again["pieces"]["banner"] == ""
    other_session = tc.render(tc.workspace_state("demo"), {}, {}, [], operator)
    assert other_session["pieces"]["banner"]               # another session still sees it


def test_the_banner_names_no_absolute_path_to_a_read_only_request(root, monkeypatch):
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    message = ("Process exited with code 1\n\n--- Last output ---\n"
               "FileNotFoundError: /srv/secret/place/kg.json")
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: {"status": "failed", "error_message": message,
                                 "gpu": {"job": None}})
    banner = tc.render(tc.workspace_state("demo"), {}, {}, [], READ_ONLY)["pieces"]["banner"]
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
    missing = tmp_path / "runs" / "gatv2" / "never_made"
    html = tc._render_error_block(_REFUSED_TAIL, run="gatv2/never_made", run_dir=missing)
    cause = html.index('<div class="shep-banner-cause">')
    assert cause < html.index("<details")                    # right under the headline
    assert ('Last error printed:</span> <span class="shep-banner-v">TrainingDeviceUnavailable: '
            '--device auto requires CUDA') in html
    assert "no run directory, so no log" in html and "full log" not in html
    run = tmp_path / "runs" / "gatv2" / "made"
    (run / "logs").mkdir(parents=True)
    assert "no log in its <code>logs/</code>" in tc._render_error_block(
        _REFUSED_TAIL, run="gatv2/made", run_dir=run)
    (run / "logs" / "train_20261021_220122.json").write_text("{}")
    assert "its log: <code>logs/train_20261021_220122.json</code>" in tc._render_error_block(
        _REFUSED_TAIL, run="gatv2/made", run_dir=run)
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
    banner = tc.render(tc.workspace_state("demo"), {}, {}, [], READ_ONLY)["pieces"]["banner"]
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
    run = tc.RunRef(id="demo/gatv2/never", label="gatv2/never",
                    path=root / "demo" / "runs" / "gatv2" / "never", workspace="demo", job=True,
                    latest=False)
    ended = tc._chart_token(run, {}, operator=True, live=False)
    assert ended == {"run": None, "rev": 0, "empty": tc.NO_RUN_DIR_ENDED}
    assert "no longer listed" not in tc.NO_RUN_DIR_ENDED and "no run directory" in tc.NO_RUN_DIR_ENDED
    assert tc._chart_token(run, {}, operator=True, live=True) == {
        "run": None, "rev": 0, "empty": tc.WAITING_FOR_EPOCH}
    failed = {"status": "failed"}
    for operator in (True, False):
        snap = tc.run_snapshot_html(run, failed, operator=operator, snapshot=None)
        assert "Failed. No run directory exists for this run, so it has no checkpoint and no log." in snap
        assert "logs/" not in snap
    # Running before the run made its directory: no logs pointed to yet, and
    # the header's Details says no checkpoint is written yet.
    live = tc.run_snapshot_html(run, {"status": "running"}, operator=True, snapshot=None)
    assert live == '<span class="shep-rv-snap">Running</span>'
    head = tc.run_head_html(run, {}, operator=True, snapshot=None, live=True)
    assert "Newest checkpoint" in head and "none written yet" in head


def test_an_ended_run_with_no_checkpoint_is_headed_as_its_snapshot_says(root, monkeypatch,
                                                                       operator):
    """M5-3 review: a run that ended with no checkpoint (Device auto refused on
    a host without CUDA) was headed "used dataset fingerprint: recorded at the
    first checkpoint" over "… so it has no checkpoint and no log." (no run
    directory) or "No checkpoint was saved…" (a directory, no last.pt)."""
    ws = {"row": {"dataset_fingerprint": "a" * 64}, "digits": 12}
    run = tc.RunRef(id="demo/gatv2/r", label="gatv2/r", path=root / "demo" / "runs" / "gatv2" / "r",
                    workspace="demo", job=True, latest=False)
    ended = tc.run_head_html(run, ws, operator=True, snapshot=None)
    assert "dataset not recorded (no checkpoint)" in ended
    assert "first checkpoint" not in ended
    assert "recorded at the first checkpoint" in tc.run_head_html(run, ws, operator=True,
                                                                  snapshot=None, live=True)
    # "dataset not recorded" only when the run recorded none (the WebUI skill
    # §6); a run that recorded one beside a run-bar dataset with no
    # fingerprint cannot be compared, which is another state.
    unrecorded = tc.run_head_html(run, ws, operator=True,
                                  snapshot={"readable": True, "dataset_fingerprint": None})
    assert "dataset not recorded" in unrecorded
    bare = tc.run_head_html(run, {"row": {}, "digits": 12}, operator=True,
                            snapshot={"readable": True, "dataset_fingerprint": "a" * 64})
    assert "dataset not recorded" not in bare
    assert "dataset cannot be verified: the run bar&#x27;s dataset has no fingerprint" in bare
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    for made in (False, True):
        job_run = root / "demo" / "runs" / "gatv2" / f"20261022-011234_auto_{int(made)}"
        if made:
            (job_run / "logs").mkdir(parents=True)
        monkeypatch.setattr(tc.training_manager, "run_dir", job_run)
        for status, live in (("failed", False), ("running", True)):
            monkeypatch.setattr(tc.training_manager, "get_status", lambda status=status: {
                "status": status, "job_id": "j", "error_message": _REFUSED_TAIL,
                "gpu": {"job": None}})
            pieces = tc.render(tc.workspace_state("demo"), {}, {}, [], operator)["pieces"]
            if live:
                assert "recorded at the first checkpoint" in pieces["head"]
                assert "none written yet" in pieces["head"]          # its Details
            else:
                assert "dataset not recorded (no checkpoint)" in pieces["head"]
                assert ("it has no checkpoint" in pieces["head"] if not made
                        else "No checkpoint saved, so nothing can be continued" in pieces["head"])


@pytest.mark.parametrize("role", ["read-only", "operator"])
def test_a_banner_redacts_paths_and_keeps_the_tails_indentation(root, monkeypatch, role):
    """M5-3 review: the read-only redaction joined each line's whitespace, so
    the traceback's lines lost their indent and its ``^^^`` markers no longer
    sat under the code; and the tail was drawn in the page's proportional
    font (the app's "* { font-family … !important }" won over its own). The
    round's closing check: an operator's banner showed the server's absolute
    path (the round's §8: no error shows a server's path), where the
    Diagnosis drawer, the Load step and Test's job card redact for every
    request; one redaction for every request (`printed_scrub`)."""
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    request = READ_ONLY
    if role == "operator":
        operator_mode(monkeypatch)
        request = gradio_request()
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
    banner = tc.render(tc.workspace_state("demo"), {}, {}, [], request)["pieces"]["banner"]
    tail = banner[banner.index('<pre class="shep-banner-tail">'):banner.index("</pre>")]
    assert "/srv/secret" not in banner
    assert '\n  File &quot;a path outside the workspaces root&quot;, line 949' in tail
    assert "\n    device = resolve(config.device)\n             ^^^^" in tail
    scrub = tc.printed_scrub(None)
    assert scrub("\t  x = /srv/a/b.py  y") == "\t  x = a path outside the workspaces root y"
    assert scrub("   ") == "   " and scrub("") == ""
    tail_css = tc.PAGE_CSS[tc.PAGE_CSS.index(".shep-banner-tail {"):]
    assert tail_css[:tail_css.index("}")].count("font-family: var(--shep-mono) !important") == 1


def test_every_banner_block_redacts_what_it_prints_whoever_calls_it(monkeypatch, tmp_path):
    """The closing review: the redaction of a printed line became one rule
    for every request, but the banner's blocks kept an unredacted default
    (``_as_is``) beside it, so a caller that left the redaction out showed an
    absolute server path again. The blocks redact themselves; no caller
    chooses whether."""
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(tmp_path))
    inside = f"{tmp_path}/demo/node_features.pt"
    message = (f"train_model: refused: {inside} could not be parsed\n\n--- Last output ---\n"
               "Traceback (most recent call last):\n"
               '  File "/srv/secret/train.py", line 9, in train\n'
               f"ValueError: {inside} could not be parsed")
    warning = "PyG native extensions missing: /srv/secret/torch_scatter.so"
    html, _key = tc._build_error_banner_payload({"error_message": message}, warning)
    for shown in (html, tc._render_error_block(message), tc._render_warning_block(warning)):
        assert str(tmp_path) not in shown and "/srv/secret" not in shown
    assert "demo/node_features.pt" in html and "a path outside the workspaces root" in html
    assert not hasattr(tc, "_as_is")


def test_the_banner_escapes_the_output():
    html, _key = tc._build_error_banner_payload(
        {"error_message": "Process exited\n\n--- Last output ---\n<script>alert(1)</script>"},
        "<img src=x>")
    assert "<script>" not in html and "&lt;script&gt;" in html and "<img" not in html


def test_the_banners_lead_with_their_condition_by_the_shared_mark():
    """The skill's §6: a failure is an error's ✕ in red and an advisory a
    condition to watch's ▲ in amber, each with its words (the condition
    first: no "Warning:" before it), by `renderers.status_mark`, never a page
    glyph (it read "⚠ Training failed …" and "⚠ Warning: …")."""
    warning = "PyG native extensions missing: torch_scatter. Training and inference still run"
    html, _key = tc._build_error_banner_payload(
        {"error_message": "Process exited with code 1", "current_epoch": 7, "total_epochs": 60},
        warning)
    assert r.status_mark("error", "Training failed at epoch 7 of 60: Process exited with code 1") \
        in html
    assert r.status_mark("warn", warning) in html
    assert "⚠" not in html and "Warning:" not in html


# --------------------------------------------------------------------------- M5-2 review fixes
#: Markup a workspace or run name can hold; never sent to the browser as a tag.
TAG = "<img src=x onerror=window.__xss=1>"
#: Start's inputs after the workspace: the run name, the device, the fields
#: the operator touched (none) and the base form.
_FORM = ["", "cpu", [], *tc.base_form_defaults()]


def _no_tag(html):
    return "<img" not in html and "<b>odd" not in html


def test_each_notice_is_drawn_by_its_kind_a_refusal_with_its_mark_a_state_plain(
        monkeypatch, operator):
    """B5 (leftover e): the run bar's and Fine-tune's notices said refusals
    and states alike in amber with no mark. Each caller now gives its
    message's kind (`notice_html`'s `mark`, required): a refusal reads with the
    ✕ of a refusal, a plain state ("Stopped. …", nothing to stop, a job that
    had already ended) as plain text, with no mark or colour of its own."""
    import inspect

    from src.webui.components import finetune_page as fp

    assert inspect.signature(tc.notice_html).parameters["mark"].default is \
        inspect.Parameter.empty, "every caller says its notice's kind"
    assert "amber" not in next(line for line in tc.PAGE_CSS.splitlines()
                               if "#shep-train-notice {" in line)
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: {"status": "idle"})
    nothing, _mode = tc.ask_stop(operator_access_allowed(), {})
    assert nothing == tc.notice_html("Nothing to stop: no training job is running.", mark=None)
    assert "shep-st" not in nothing
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: {"status": "running", "job_id": "j1"})
    monkeypatch.setattr(tc.training_manager, "stop_training",
                        lambda: {"success": True, "outcome": None})
    stopped, _mode = tc.confirm_stop(operator_access_allowed(), {"confirm": "j1"})
    assert stopped.startswith('<span class="shep-notice-t">Stopped.') and "shep-st" not in stopped
    monkeypatch.setattr(tc.training_manager, "stop_training",
                        lambda: {"success": False, "error": "<b>busy</b>"})
    failed, _mode = tc.confirm_stop(operator_access_allowed(), {"confirm": "j1"})
    assert failed == tc.notice_html("Not stopped: <b>busy</b>", mark="error")
    assert "✕" in failed and "shep-st-error" in failed and "<b>busy" not in failed
    refused, _mode = tc.ask_stop(operator_access_refused(), {})
    assert "shep-st-error" in refused and "Refused: " in refused
    # Fine-tune's Stop is the same two steps, so its notice by the same kinds.
    assert "tc.ask_stop" in inspect.getsource(fp._on_finetune_stop_request)
    assert "tc.confirm_stop" in inspect.getsource(fp._on_finetune_stop)


def operator_access_allowed():
    class Allowed:
        allowed, condition, refusal = True, None, None
    return Allowed()


def operator_access_refused():
    class Refused:
        allowed, condition, refusal = False, "mode_unset", "Refused: deployment mode not set to M1."
    return Refused()


def test_the_notice_escapes_the_server_text_it_carries(root, monkeypatch, operator):
    """The run bar's notice is HTML: a workspace name, a path or the runner's
    error in it is escaped (`notice_html`), for Start, Stop and Continue
    Training alike."""
    assert tc.notice_html("", mark=None) == "" and tc.notice_html(None, mark="error") == ""
    assert tc.notice_html(TAG, mark=None) == f'<span class="shep-notice-t">{tc.r.e(TAG)}</span>'
    # Start: the runner's refusal, and a workspace that is not trainable.
    monkeypatch.setattr(tc.training_manager, "start_training",
                        lambda config, **k: {"success": False, "error": f"{TAG} refused"})
    notice = tc._on_start(operator, tc.workspace_state("demo"), *_FORM)
    assert _no_tag(notice) and "&lt;img src=x" in notice and "refused" in notice
    odd = tc.workspace_state(ODD)
    odd["row"] = {**odd["row"], "trainable": False, "not_trainable": "x", "manifest": "read"}
    notice = tc._on_start(operator, odd, *_FORM)
    assert _no_tag(notice) and "&lt;b&gt;odd&amp;name is not trainable" in notice
    # Stop: the snapshot sentence names the checkpoint inside the workspace.
    last = root / ODD / "runs" / "gatv2" / "r" / "checkpoints" / "last.pt"
    monkeypatch.setattr(tc.training_manager, "stop_training", lambda: {
        "success": True, "outcome": {"checkpoint": str(last), "readable": True,
                                     "completed_epochs": 3, "target_epochs": 10, "end": None}})
    notice, _mode = tc._on_stop(operator, {})
    assert _no_tag(notice) and "&lt;b&gt;odd&amp;name/runs/gatv2/r/checkpoints/last.pt" in notice
    # Continue Training: a listed parent the server no longer lists, in that
    # workspace, said by the check the start runs first (no notice says it
    # again, B1c review), escaped alike.
    fp = odd["row"]["dataset_fingerprint"]
    parents = {"gatv2/r/last.pt": {"workspace": ODD, "architecture": "gatv2", "run": "r",
                                 "filename": "last.pt", "sha256": "c" * 64,
                                 "dataset_fingerprint": fp}}
    notice, text, *_rest = tc._on_resume(
        operator, tc.CONTINUE, tc.workspace_state(ODD), "", "cpu", tc.CONTINUE_FROM_LIST,
        "gatv2/r/last.pt", "", None, None, parents, {}, [])
    assert notice == ""
    assert "no longer listed" in text and "&lt;b&gt;odd&amp;name" in text and _no_tag(text)


def test_the_run_view_and_drawer_builders_escape_names_and_paths(root):
    """The run header (workspace, run label), the drawer's note, the dataset
    comparison and the check's checkpoint path, each with markup in a name."""
    run = tc.RunRef(id=f"{ODD}/gatv2/{TAG}", label=f"gatv2/{TAG}",
                    path=root / ODD / "runs" / "gatv2" / TAG, workspace=ODD, latest=True)
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
    note = tc._parents_note(ws, {"k": {"architecture": "gatv2", "run": TAG}}, None)
    assert _no_tag(note) and "1 run, 1 checkpoint" in note
    for fingerprint in ("a" * 64, "e" * 64):          # the same dataset, and another
        compare = tc.compare_html({"chosen": True, "fingerprint": fingerprint,
                                   "source": tc.CONTINUE_FROM_LIST}, ws)
        assert _no_tag(compare), fingerprint
    assert "&lt;b&gt;odd&amp;name" in tc.compare_html(
        {"chosen": True, "fingerprint": "e" * 64, "source": tc.CONTINUE_FROM_LIST}, ws)
    preview = {"parent": str(root / ODD / "runs" / "gatv2" / TAG / "last.pt"), "ok": False,
               "reason": f"refused: {TAG}", "sha256": "c" * 64}
    check = tc._check_html(preview, tc.CONTINUE_FROM_LIST)
    assert _no_tag(check) and "&lt;b&gt;odd&amp;name/runs/gatv2/&lt;img" in check
    listed = tc._check_html(preview, tc.CONTINUE_FROM_LIST, listed=f"gatv2/{TAG}/last.pt")
    assert _no_tag(listed) and "<code>gatv2/&lt;img" in listed


def _rendered_for(status, mode, operator, monkeypatch, ws):
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: status)
    return tc.render(ws, mode, {}, [], operator)["pieces"]


def test_a_stop_confirmation_belongs_to_the_job_it_was_asked_for(root, monkeypatch, operator):
    """Asked for job a1 and left unanswered when a1 completes: when b2 runs,
    the run bar shows no confirmation and Stop is enabled; a "Stop training"
    sent for a1 stops nothing while b2 runs."""
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    monkeypatch.setattr(tc.training_manager, "run_dir", None)
    ws = tc.workspace_state("demo")
    monkeypatch.setattr(tc.training_manager, "get_status",
                        lambda: {"status": "running", "job_id": "a1"})
    monkeypatch.setattr(tc.training_manager, "latest_snapshot", lambda *a: None)
    _text, mode = tc._on_stop_request(operator, {})
    assert mode["confirm"] == "a1"
    asked = _rendered_for({"status": "running", "job_id": "a1"}, mode, operator, monkeypatch, ws)
    assert "shep-rb-confirming" in asked["runbar"]
    assert asked["stop"] == {"value": "■ Stop", "interactive": False}
    done = _rendered_for({"status": "completed", "job_id": "a1"}, mode, operator, monkeypatch, ws)
    assert "shep-rb-confirming" not in done["runbar"]
    other = _rendered_for({"status": "running", "job_id": "b2"}, mode, operator, monkeypatch, ws)
    assert "shep-rb-confirming" not in other["runbar"]
    assert other["stop"] == {"value": "■ Stop", "interactive": True}       # no lock: allowed
    stops = []
    monkeypatch.setattr(tc.training_manager, "stop_training",
                        lambda: stops.append(1) or {"success": True, "outcome": None})
    notice, closed = tc._on_stop(operator, mode)                       # b2 runs now
    assert stops == [] and "Not stopped" in notice and closed["confirm"] is None
    tc._on_stop(operator, {"confirm": "b2"})                          # asked for b2
    tc._on_stop(operator, {})                                         # a direct call
    assert stops == [1, 1]


def test_keep_training_only_closes_the_question():
    """Keep training runs `close_stop_question` and then redraws the view; no
    step of its chain is the Stop."""
    demo, _ctx, by_id = _page()
    keep = next(fn for fn in demo.fns.values()
                if (by_id["shep-train-stop-keep"]._id, "click") in fn.targets)
    assert keep.fn is tc.close_stop_question
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
    assert after.fn is tc.close_stop_question
    assert tc.close_stop_question({"confirm": "a1", "drawer": True}) == {"confirm": None,
                                                                      "drawer": True}


def _listing(items, problem=None):
    from src.api.services import server_lists

    return server_lists.Listing(tuple(items), problem)


def _item(workspace, run="r", filename="last.pt", kind="training_state", epoch=3, *,
          record="model", stage="base", task=None):
    """A listed checkpoint: by default a base's model record, as the Continue
    drawer offers (K6 decision 1: model records only)."""
    from src.api.services.server_lists import CheckpointItem

    return CheckpointItem(workspace=workspace, architecture="gatv2", run=run, filename=filename,
                          sha256="c" * 64, size_bytes=1, checkpoint_kind=kind, epoch=epoch,
                          validation=None, dataset_fingerprint=None,
                          trained_on_this_dataset=None, problem=None, record=record,
                          learning_stage=stage if record == "model" else None,
                          task=task if record == "model" else None,
                          encoder="gatv2" if record == "model" else None)


def test_the_parent_list_follows_the_run_bars_workspace_while_the_drawer_is_open(
        root, monkeypatch, operator):
    """After the run bar's workspace changed with Continue… open, the list is
    the new workspace's runs (D8 first; nothing listed with the drawer
    closed), and the old check, comparison and Extra epochs are cleared."""
    from src.api.services import server_lists

    listed = []
    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda name: listed.append(name) or _listing(
                            [_item(name), _item(name, run="h", record="refused"),
                             _item(name, run="x", record=None)]))
    ws = tc.workspace_state("resampled")
    assert tc._relist_parents(operator, ws, {"drawer": False}) == (gr.update(),) * 11
    assert listed == []
    parents, runs, select, note, compare, *cleared = tc._relist_parents(
        operator, ws, {"drawer": True})
    extend, rate, preview, notice, touched, branch = cleared
    # Only the model record is offered (K6 decision 1): a refused record (a
    # file without a model spec among them) and one not read are never
    # continued from here.
    assert listed == ["resampled"] and list(parents) == ["gatv2/r/last.pt"]
    assert parents["gatv2/r/last.pt"]["workspace"] == "resampled"
    assert runs["value"] == "gatv2/r" and select["value"] == "gatv2/r/last.pt"
    assert "1 run, 1 checkpoint" in note
    assert compare == "" and tc._CONTINUE_PROMPT in preview
    assert extend == rate == gr.update(visible=False, value=None)
    assert notice == "" and touched == [] and branch == gr.update(visible=False)
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)    # read-only from here
    refused = tc._relist_parents(READ_ONLY, ws, {"drawer": True})
    assert refused[0] == {} and refused[1] == refused[2] == gr.update(choices=[], value=None)
    assert "Refused" in refused[3] and listed == ["resampled"]


def test_a_parent_listed_for_another_workspace_is_never_continued_on_this_one(root):
    """A list read for demo, the run bar now on resampled (a running job's
    dataset, or a change the list did not follow): the reason says so, and
    nothing is resolved in resampled under demo's names."""
    item = {**_item("demo").__dict__}
    parents = {"gatv2/r/last.pt": item}
    ws = tc.workspace_state("resampled")
    chosen = tc._chosen(tc.CONTINUE_FROM_LIST, "gatv2/r/last.pt", "", parents, ws)
    assert chosen["workspace"] == "demo"
    why = tc.continue_refusal(chosen, ws)
    assert why.startswith("the run list shows demo’s runs, not resampled’s; refresh the list")
    path, why, _item_ = tc._parent_path(tc.CONTINUE_FROM_LIST, "gatv2/r/last.pt", "", ws,
                                        parents)
    assert path is None and "the run list shows demo’s runs" in why
    assert tc.continue_refusal(chosen, tc.workspace_state("demo")) is None


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

    ws = tc.workspace_state("demo")
    cache = {}
    old = tc._current_run({"status": "idle"}, ws, cache)
    assert old["id"] == "demo/gatv2/20261018-142000_run_d41f8b"
    new = root / "demo" / "runs" / "gatv2" / "20261021-150000_run_dddddd"
    (new / "checkpoints").mkdir(parents=True)
    (new / "checkpoints" / "model-1-0.0100.pt").write_bytes(b"x")
    later = time.time() + 5
    os.utime(new / "checkpoints" / "model-1-0.0100.pt", (later, later))
    listings = []
    original = tc._listed_runs
    monkeypatch.setattr(tc, "_listed_runs", lambda d: listings.append(d) or original(d))
    assert tc._current_run({"status": "idle"}, ws, cache)["id"] == old["id"]   # within the interval
    assert listings == []
    cache["runs"][1] -= tc.LATEST_READ_SECONDS                                 # the interval passed
    run = tc._current_run({"status": "idle"}, ws, cache)
    assert run["id"] == "demo/gatv2/20261021-150000_run_dddddd" and run["latest"] is True
    assert len(listings) == 1


def test_a_live_run_reads_its_checkpoint_once_for_the_fingerprint(root, monkeypatch, operator):
    """While the job runner's run is live, ``last.pt`` (rewritten each epoch,
    read whole) is read once per run and session for the recorded dataset
    fingerprint, which stays shown; after the run ends the snapshot is read."""
    monkeypatch.setattr(tc, "_get_pyg_warning", lambda: None)
    job_run = root / "demo" / "runs" / "gatv2" / "20261020-090000_run_live01"
    (job_run / "checkpoints").mkdir(parents=True)
    last = job_run / "checkpoints" / "last.pt"
    last.write_bytes(b"epoch 1")
    monkeypatch.setattr(tc.training_manager, "run_dir", job_run)
    ws = tc.workspace_state("demo")
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
        assert "same as the run bar</dd>" in rendered["pieces"]["head"]
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


def _option_rules(labels, choices=None, values=None):
    """The WebUI skill §8: options unique, at most 40 characters, one
    separator; given the select's `choices`, each value is the full key."""
    assert len(set(labels)) == len(labels), labels
    for label in labels:
        assert len(label) <= r.OPTION_CHARS and label.count(" · ") <= 1, label
    if choices is not None:
        assert [value for _label, value in choices] == list(values)


def test_the_run_checkpoint_helpers_have_one_home():
    """Training's Continue… and Fine-tune's parent choose through one Run →
    Checkpoint pair (`model_list`'s): no page keeps a copy of its own."""
    from src.webui.components import finetune_page as ft

    for name in ("run_words", "checkpoint_words", "run_choices", "checkpoint_choices",
                 "parents_update", "on_run_chosen", "runs_of", "OPTION_CHARS", "_fit",
                 "_choices", "_progress_words", "validation_words", "_TOP_K_FILE"):
        assert not hasattr(tc, name) and not hasattr(ft, name), name
    for name in ("checkpoint_words", "run_choices", "checkpoint_choices",
                 "parents_update", "on_run_chosen", "chosen_in_run", "runs_of"):
        assert callable(getattr(ml, name)), name
    # A run's words are the shared renderers' (every page names a run by
    # them, a file's title too: `renderers.model_name_line`), not a page's.
    from src.webui.components import renderers as r

    assert callable(r.run_words) and not hasattr(ml, "run_words")


def test_a_checkpoints_option_says_what_it_is_and_how_far_it_trained():
    """In a few words, from the file's own record (the WebUI skill §8: at most
    40 characters, one separator): the latest save, a best save of its run by
    the selection its record names, a top-k save by its run's recorded best-k
    (never by its name), and the epochs its progress record says it completed
    (never epoch + 1 for a file without one). How its stage ended and its
    Trainer validation figure are the chosen file's details, not its option
    (the option read "Top-k save · 21 epochs completed · stopped early, cannot
    continue · Validation LP accuracy (Trainer) 0.8123", 107 characters and
    three separators)."""
    item = {**_item("demo").__dict__, "completed_epochs": 195, "end": "paused",
            "validation": {"metric": "val_lp_accuracy", "value": 0.8123},
            "selection": {"monitor": "val_lp_accuracy", "mode": "max"}}
    assert ml.checkpoint_words(item) == "Latest save · 195 epochs completed"
    assert ml.checkpoint_words(item, best=True) == "Latest save · 195 epochs completed"
    top = {**item, "filename": "model-20-0.8123.pt", "completed_epochs": 21,
           "end": "stopped_early", "in_best_k": True}
    assert ml.checkpoint_words(top) == "Top-k save · 21 epochs completed"
    assert ml.checkpoint_words(top, best=True) == "Best LP accuracy · 21 epochs completed"
    task = {**top, "selection": {"monitor": "val_pooled_mrr", "mode": "max"}}
    assert ml.checkpoint_words(task, best=True) == "Best pooled MRR · 21 epochs completed"
    # A top-k file's name is not what makes it one: its run's records are.
    named_alike = {**top, "in_best_k": False}
    assert ml.checkpoint_words(named_alike) == "model-20-0.8123.pt · 21 epochs completed"
    assert ml.checkpoint_words({**top, "in_best_k": None}) == (
        "model-20-0.8123.pt · 21 epochs completed")
    assert ml.checkpoint_words({**item, "end": "finished"}) == "Latest save · 195 epochs completed"
    weights = {**_item("demo", filename="model-03-0.2.pt", kind="weights").__dict__,
               "in_best_k": True}
    assert ml.checkpoint_words(weights) == "Top-k save · weights only"
    older = {**_item("demo", epoch=194).__dict__}                     # no progress record
    assert ml.checkpoint_words(older) == "Latest save · recorded epoch 194"
    assert ml.checkpoint_words({**older, "problem": "ValueError: x"}) == "last.pt · cannot be read"
    other = {**item, "filename": "best.pt", "completed_epochs": 1}
    assert ml.checkpoint_words(other) == "best.pt · 1 epoch completed"
    # A long file name is shortened to fit, its progress kept.
    long = ml.checkpoint_words({**item, "filename": "a-very-long-checkpoint-name-x.pt"})
    assert long.endswith(" · 195 epochs completed") and "…" in long
    _option_rules([ml.checkpoint_words(i) for i in (item, top, weights, older, other)]
                  + [long, ml.checkpoint_words(top, best=True)])


def test_a_run_reads_as_its_name_architecture_and_start_time():
    """The distinguishing token first (the WebUI skill §8): a run's name, then
    its architecture and start time; an unnamed run's start time, never the
    "GATv2 · " every run shares; one separator (`renderers.run_words`, every
    page's)."""
    from src.webui.components import renderers as r

    assert r.run_words("gat", "20261024-173909_ui-live-stop-02_c1d9cf") == \
        "ui-live-stop-02 · GAT 2026-10-24 17:39"
    assert r.run_words("gat", "20261024-163856_run_559817") == "2026-10-24 16:38 · GAT"
    assert r.run_words("hgt", "my-old-run") == "my-old-run · HGT"
    # A model bundle's run reads as Models reads it (`renderers.architecture_label`).
    assert r.run_words("gatv2", "20261028-100000_run_abcdef") == "2026-10-28 10:00 · GATv2"
    long = r.run_words("gatv2", "20261028-100000_smoke-ft-patients_like_me_abcdef")
    assert long.endswith(" · GATv2 2026-10-28 10:00") and long.startswith("smoke") and "…" in long
    # A layout directory is listed whatever its name (K8-7b): its label, the
    # part every run of it shares, is what gets shortened; the name and the
    # start time that tell its runs apart stay, never made longer.
    layout = "experiments_from_october"
    assert r.run_words(layout, "20261024-173909_a_c1d9cf") == "a · EXP…OBER 2026-10-24 17:39"
    assert r.run_words(layout, "20261024-173909_run_c1d9cf") == "2026-10-24 17:39 · EXP…OBER"
    named = r.run_words(layout, "20261024-173909_a_very_long_run_name_for_october_c1d9cf")
    assert named.startswith("a_ver") and named.endswith(" · EXP…OBER 2026-10-24 17:39"), named
    twins = {f"{layout}_v2/20261024-173909_a_c1d9c{i}/last.pt":
             {**_item("demo", run=f"20261024-173909_a_c1d9c{i}").__dict__,
              "architecture": f"{layout}_v2", "run_key": f"{layout}_v2/20261024-173909_a_c1d9c{i}"}
             for i in range(2)}
    chosen = ml.run_choices(twins)
    assert [label for label, _key in chosen] == [
        "a · EXP…R_V2 2026-10-24 17:39:09 c1d9c0", "a · EXP…R_V2 2026-10-24 17:39:09 c1d9c1"]
    _option_rules([long, named, *(r.run_words(layout, run) for run in (
        "20261024-173909_a_c1d9cf", "20261024-173909_run_c1d9cf", "my-very-long-old-run-dir-x"))],
        chosen, [item["run_key"] for item in twins.values()])
    # A run a record names by its directory alone (a parent's, a family's
    # root run: ops-wiring round B0c, where Models said "run 2026-10-01
    # 00:00" and dropped the name): the same words, with no layout said.
    assert r.run_words(None, "20261024-173909_ui-live-stop-02_c1d9cf") == \
        "ui-live-stop-02 · 2026-10-24 17:39"
    assert r.run_words(None, "20261024-163856_run_559817") == "2026-10-24 16:38"
    assert r.run_words(None, "my-old-run") == "my-old-run"
    # The same parts whole where the details name a run (`fit` False: the
    # review of B0c), the architecture too; the option keeps its cut.
    assert r.run_words(layout, "20261024-173909_a_very_long_run_name_for_october_c1d9cf",
                        fit=False) == f"a_very_long_run_name_for_october · {layout.upper()} " \
                                      "2026-10-24 17:39"
    assert r.run_words(None, "my-very-long-old-run-directory-name-past-forty",
                        fit=False) == "my-very-long-old-run-directory-name-past-forty"
    from src.webui.components import models_page

    assert not hasattr(models_page, "run_words")     # its own words retired into these


def test_the_drawer_lists_runs_then_the_chosen_runs_checkpoints():
    """Two steps over the same server list (the same keys, digests and check):
    the runs, latest first; the chosen run's files, newest first. Choosing
    another run offers its files with its latest save chosen; a checkpoint
    kept by key stays chosen; two runs that would read alike are told apart
    by their start time to the second, then their own id, and two files by
    their names. Each value is the listing's key, whatever the words."""
    def item(run, filename, completed, figure=None, top=False):
        return {**_item("demo", run=run, filename=filename).__dict__,
                "completed_epochs": completed, "in_best_k": top or None,
                "validation": {"metric": "val_lp_accuracy", "value": figure} if figure else None}
    new, old = "20261024-173909_a_c1d9cf", "20261024-163856_run_559817"
    parents = {f"gatv2/{new}/last.pt": item(new, "last.pt", 195),
               f"gatv2/{new}/model-20-0.2867.pt": item(new, "model-20-0.2867.pt", 21, 0.2867,
                                                       True),
               f"gatv2/{old}/last.pt": item(old, "last.pt", 3),
               f"gatv2/{old}/model-02-0.2160.pt": item(old, "model-02-0.2160.pt", 3, 0.216, True),
               f"gatv2/{old}/model-01-0.2160.pt": item(old, "model-01-0.2160.pt", 2, 0.216, True)}
    runs, files = ml.parents_update(parents)
    assert runs["choices"] == [("a · GATv2 2026-10-24 17:39", f"gatv2/{new}"),
                               ("2026-10-24 16:38 · GATv2", f"gatv2/{old}")]
    assert runs["value"] == f"gatv2/{new}" and files["value"] == f"gatv2/{new}/last.pt"
    assert [label for label, _key in files["choices"]] == [
        "Latest save · 195 epochs completed", "Top-k save · 21 epochs completed"]
    others = ml.on_run_chosen(f"gatv2/{old}", parents, f"gatv2/{new}/last.pt")
    assert others["value"] == f"gatv2/{old}/last.pt"                     # its latest save
    assert [key for _label, key in others["choices"]] == [k for k in parents if old in k]
    labels = [label for label, _key in others["choices"]]
    # The latest save and a top-k save at the same epochs are told apart by
    # what they are; still unique, short and with one separator.
    assert labels == ["Latest save · 3 epochs completed", "Top-k save · 3 epochs completed",
                      "Top-k save · 2 epochs completed"]
    twins = {f"gatv2/{old}/model-03-0.1.pt": item(old, "model-03-0.1.pt", 3, top=True),
             f"gatv2/{old}/model-03-0.2.pt": item(old, "model-03-0.2.pt", 3, top=True)}
    told = [label for label, _key in ml.checkpoint_choices(twins, list(twins))]
    assert told == ["model-03-0.1.pt · 3 epochs completed", "model-03-0.2.pt · 3 epochs completed"]
    _option_rules(labels, others["choices"], [k for k in parents if old in k])
    _option_rules(told)
    # Long names alike at their start keep the end that tells them apart (a
    # copy, a versioned top-k file); names alike at both ends keep their start
    # through where they differ (`renderers.fit_apart`: the closing check of
    # the ops round, where they were numbered "(1)", "(2)", "(3)", which says
    # nothing of which file is which). Never two options alike.
    for names, epochs in ((("base_model_before_change_epoch10_copy_a.pt",
                            "base_model_before_change_epoch10_copy_b.pt"), 10),
                          (("model-195-0.2867.pt", "model-195-0.2867-v1.pt"), 195),
                          (("checkpoint_run1_of_the_final.pt", "checkpoint_run2_of_the_final.pt",
                            "checkpoint_run3_of_the_final.pt"), 3)):
        files = {f"gatv2/{old}/{name}": item(old, name, epochs) for name in names}
        chosen = ml.checkpoint_choices(files, list(files))
        told = [label for label, _key in chosen]
        _option_rules(told, chosen, list(files))
        assert all(label.endswith(f" · {epochs} epochs completed") for label in told), told
    assert told == ["checkpoint_run1_…pt · 3 epochs completed",
                    "checkpoint_run2_…pt · 3 epochs completed",
                    "checkpoint_run3_…pt · 3 epochs completed"]
    copies = {f"gatv2/{old}/base_model_before_change_epoch10_copy_{x}.pt":
              item(old, f"base_model_before_change_epoch10_copy_{x}.pt", 10) for x in "ab"}
    assert [label for label, _key in ml.checkpoint_choices(copies, list(copies))] == [
        "base_mod…copy_a.pt · 10 epochs completed", "base_mod…copy_b.pt · 10 epochs completed"]
    kept = ml.on_run_chosen(f"gatv2/{old}", parents, f"gatv2/{old}/model-01-0.2160.pt")
    assert kept["value"] == f"gatv2/{old}/model-01-0.2160.pt"
    assert ml.parents_update(parents, f"gatv2/{old}/model-02-0.2160.pt")[0]["value"] == \
        f"gatv2/{old}"
    # The Test page's use of the one pair (review of B3: its own copy
    # retired): nothing chosen until the operator picks a run, a listed
    # choice kept with its run, and each select interactive for an operator
    # while it has options; no flag given leaves interactivity as it is.
    runs, files = ml.parents_update(parents, None, first=False, interactive=True)
    assert (runs["value"], files["value"], files["choices"]) == (None, None, [])
    assert (runs["interactive"], files["interactive"]) == (True, False)
    assert runs["choices"] == ml.parents_update(parents)[0]["choices"]
    runs, files = ml.parents_update(parents, f"gatv2/{old}/last.pt", first=False,
                                    interactive=False)
    assert (runs["value"], files["value"]) == (f"gatv2/{old}", f"gatv2/{old}/last.pt")
    assert (runs["interactive"], files["interactive"]) == (False, False)
    assert "interactive" not in ml.parents_update(parents)[0]
    # Runs alike to the minute: their time to the second; alike to the second
    # (two layouts' runs, or one second's two ids): their own id too.
    twin_runs = {
        "gatv2/20261024-163856_run_111111/last.pt": item("20261024-163856_run_111111", "last.pt", 1),
        "gatv2/20261024-163801_run_222222/last.pt": item("20261024-163801_run_222222", "last.pt", 1),
        "gatv2/20261024-163801_run_333333/last.pt": item("20261024-163801_run_333333", "last.pt", 1)}
    chosen = ml.run_choices(twin_runs)
    assert [label for label, _key in chosen] == [
        "2026-10-24 16:38:56 · GATv2", "2026-10-24 16:38:01 · GATv2 222222",
        "2026-10-24 16:38:01 · GATv2 333333"]
    _option_rules([label for label, _key in chosen], chosen,
                  [f"gatv2/{key.split('/')[1]}" for key in twin_runs])
    assert ml.on_run_chosen(None, {}, None) == gr.update(choices=[], value=None)


def test_a_runs_best_save_is_its_recorded_selections_best_never_a_file_name():
    """The skill's §8: "best" is the best value of the run's own recorded
    selection metric, in its recorded direction, among this run's saved files,
    named by that metric: a base by validation LP accuracy, higher is better;
    a min-mode monitor, the lowest; ties share it; an older file that records
    no selection is never best. A label never changes the value."""
    def item(run, filename, figure, epochs=4, *, metric="val_lp_accuracy", mode="max",
             top=True):
        return {**_item("demo", run=run, filename=filename).__dict__,
                "completed_epochs": epochs, "in_best_k": top,
                "validation": {"metric": metric, "value": figure},
                "selection": {"monitor": metric, "mode": mode}}
    run = "20261024-173909_a_c1d9cf"
    keys = [f"gatv2/{run}/{name}" for name in ("last.pt", "model-03-0.7000.pt",
                                               "model-02-0.9000.pt", "model-01-0.9000.pt")]
    base = dict(zip(keys, (item(run, "last.pt", 0.7, top=False),
                           item(run, "model-03-0.7000.pt", 0.7),
                           item(run, "model-02-0.9000.pt", 0.9, 3),
                           item(run, "model-01-0.9000.pt", 0.9, 2))))
    assert r.best_saves(base) == set(keys[2:])                          # a tie shares it
    chosen = ml.checkpoint_choices(base, keys)
    assert [label for label, _key in chosen] == [
        "Latest save · 4 epochs completed", "Top-k save · 4 epochs completed",
        "Best LP accuracy · 3 epochs completed", "Best LP accuracy · 2 epochs completed"]
    _option_rules([label for label, _key in chosen], chosen, keys)
    # A task model's: named by validation pooled MRR, never an LP figure's name.
    task = {key: {**value, "validation": {**value["validation"], "metric": "val_pooled_mrr"},
                  "selection": {"monitor": "val_pooled_mrr", "mode": "max"}}
            for key, value in base.items()}
    assert [label for label, _key in ml.checkpoint_choices(task, keys)][2:] == [
        "Best pooled MRR · 3 epochs completed", "Best pooled MRR · 2 epochs completed"]
    # A min-mode monitor: the lowest is best, whatever the file names say.
    low = {key: {**value, "selection": {"monitor": "val_lp_accuracy", "mode": "min"}}
           for key, value in base.items()}
    assert r.best_saves(low) == set(keys[:2])
    # An older file without the record, or a figure of another metric, is none.
    older = {key: {**value, "selection": None} for key, value in base.items()}
    assert r.best_saves(older) == set()
    other = {keys[2]: {**base[keys[2]], "validation": {"metric": "val_pooled_mrr",
                                                       "value": 0.99}},
             keys[3]: base[keys[3]]}
    assert r.best_saves(other) == {keys[3]}
    # Two runs are each their own: one best per run.
    second = "20261023-000000_b_aaaaaa"
    both = {**base, f"gatv2/{second}/last.pt": item(second, "last.pt", 0.1)}
    assert r.best_saves(both) == {*keys[2:], f"gatv2/{second}/last.pt"}


def test_the_chosen_checkpoints_details_say_what_its_option_does_not():
    """A summary list: the file, its stage, what it holds, what continuing does when
    its stage ended, its Trainer validation figure (named as the Trainer's),
    its model fingerprint (copyable) and its run's directory; its progress is
    its option's, not repeated. What its run's saves were ranked by is its
    record's selection, named with its direction, or not recorded."""
    item = {**_item("demo", run="20261024-163856_run_559817").__dict__, "completed_epochs": 11,
            "end": "finished", "validation": {"metric": "val_lp_accuracy", "value": 0.8123}}
    assert "Selection</dt><dd class=\"shep-v\">not recorded<" in tc.parent_details_html(item)
    item["selection"] = {"monitor": "val_lp_accuracy", "mode": "max"}
    html = tc.parent_details_html(item)
    assert ("Selection</dt><dd class=\"shep-v\">by validation LP accuracy (Trainer), "
            "higher is better<") in html
    for words in ("File", "last.pt", "Kind", "Base", "Holds", "training state", "Continuing",
                  "extends it by the Extra epochs you enter", "Validation",
                  "LP accuracy (Trainer) 0.8123", "Model fingerprint", "cccccccccccc",
                  'data-copy="' + "c" * 64, "gatv2/20261024-163856_run_559817"):
        assert words in html, words
    # Never a test result: the figure says it is the Trainer's selection
    # metric on val, as every details list's Validation row does (B0c).
    assert "LP accuracy (Trainer) 0.8123 — selection metric on val, not an independent test" \
        in html
    assert "epochs completed" not in html
    assert "test" not in html.lower().replace("not an independent test", "")
    stopped = tc.parent_details_html({**item, "end": "stopped_early"})
    assert "not possible: it stopped early" in stopped
    child = tc.parent_details_html({**item, "learning_stage": "finetune", "task": "gene",
                                    "validation": None})
    assert "<dt class=\"shep-k\">Kind</dt><dd class=\"shep-v\">Gene<" in child
    assert "not recorded in this checkpoint" in child
    # A frozen-backbone child is named so (decision F1: `model_kind_text`),
    # never by its stage words alone.
    from tests.unit.test_webui_renderers import GRAPH_SIDE_FROZEN

    frozen = tc.parent_details_html({**item, "learning_stage": "finetune", "task": "disease",
                                     "weight_transfer": GRAPH_SIDE_FROZEN})
    assert ("<dt class=\"shep-k\">Kind</dt><dd class=\"shep-v\">Disease · frozen graph "
            "backbone<") in frozen
    weights = tc.parent_details_html(_item("demo", kind="weights").__dict__)
    assert ("weights only (cannot be continued or branched)" in weights
            and "Continuing" not in weights)
    # A branch is a new run whatever its parent's end: no "Continuing" row.
    assert "Continuing" not in tc.parent_details_html({**item, "end": "stopped_early"},
                                                      tc.BRANCH)
    unread = tc.parent_details_html({**_item("demo").__dict__, "problem": "ValueError: <b>x"})
    assert "Cannot be read" in unread and "&lt;b&gt;x" in unread
    assert '<span class="shep-st shep-st-warn">' in unread    # ▲ and its words, no amber box
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
    assert text.startswith(tc.charts.GESTURES + "\n\n") and "\n- The wheel over the epoch axis" in text
    for key in ("manual_checkpoint", "checkpoint_dropdown", "continue_device", "run_name",
                "continue_run"):
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


def test_every_device_field_is_the_one_field_with_the_hosts_note(monkeypatch):
    """R-32 and NEW-training-finetune-1: Device defaulted to auto on Training,
    in the Continue drawer and on Fine-tune, and nothing at the field said
    that auto (and cuda) would be refused on a host with no CUDA device. Every
    Device field is now the one field (`device_field`), its info line the one
    note (`DEVICE_NOTE`) on such a host and none where CUDA is visible."""
    import torch

    from src.webui.components import finetune_page as fp

    def fields():
        with gr.Blocks() as demo:
            ctx = shell.standalone_context(demo)
            tc.create_training_tab(ctx)
            fp.create_finetune_page(ctx)
        by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
        return [by_id[key] for key in ("device", "continue_device", "ft_device")]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert tc.DEVICE_NOTE == "No CUDA device here: auto and cuda are refused; choose cpu."
    for field in fields():
        assert isinstance(field, gr.Radio) and field.label == "Device"
        assert [c[1] for c in field.choices] == ["auto", "cuda", "cpu"] and field.value == "auto"
        assert field.info == tc.DEVICE_NOTE, field.elem_id
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert all(not field.info for field in fields())
    # One note in the page's words: no command line flag.
    assert "--device" not in tc.DEVICE_NOTE


def test_the_drawer_says_the_check_is_optional_beside_a_normal_check_button():
    """R-48: the drawer said "Select a checkpoint, then click Check
    checkpoint." while a checkpoint was chosen and the start enabled (the
    start checks first), with Check checkpoint a small button that read as a
    chip, and the parent's dataset a filled green box. The prompt says the
    check is optional, beside the button in step 3's row (so they are in view
    together, NEW-training-finetune-2); the button is a normal secondary one;
    the dataset relation is one line of text."""
    _demo, _ctx, by_id = _page()
    check = by_id["shep-train-check"]
    assert check.size != "sm" and check.variant == "secondary"
    prompt = by_id["shep-train-check-prompt"]
    assert prompt.parent is check.parent and "shep-train-check-row" in check.parent.elem_classes
    said = tc.CHECK_PROMPTS[tc.CONTINUE]
    assert prompt.value == tc.check_prompt_html(tc.CONTINUE) and said in prompt.value
    assert said.startswith("Optional") and "The start checks first" in said
    assert "Select a checkpoint" not in "".join(tc.CHECK_PROMPTS.values()) + tc._CONTINUE_PROMPT
    # Branch: the check is how a setting is changed (its fields are shown by
    # a check only), so its prompt never says "Optional" (the T06 review).
    branch = tc.CHECK_PROMPTS[tc.BRANCH]
    assert "Optional" not in branch and "change a setting" in branch
    assert "keeps the parent’s settings" in branch
    mode, shown, *_cleared = tc._on_operation(tc.BRANCH, {})
    assert mode["operation"] == tc.BRANCH and shown == tc.check_prompt_html(tc.BRANCH)
    assert tc._on_operation(tc.CONTINUE, mode)[1] == tc.check_prompt_html(tc.CONTINUE)
    assert tc.check_prompt_html("<b>") == tc.check_prompt_html(tc.CONTINUE)
    ws = {"name": "demo", "row": {"name": "demo", "dataset_fingerprint": "a" * 64}, "listed": []}
    same = tc.compare_html({"chosen": True, "fingerprint": "a" * 64}, ws)
    assert same == ('<p class="shep-cmp-line">'
                    + r.status_mark("ok", "Same dataset as the run bar") + "</p>")
    unknown = tc.compare_html({"chosen": True, "fingerprint": None}, ws)
    assert unknown.startswith('<p class="shep-cmp-line"><span class="shep-st shep-st-neutral">')
    assert not _rules(tc.PAGE_CSS, "shep-cmp-ok") and not _rules(tc.PAGE_CSS, "shep-cmp-grey")


def test_with_the_drawer_open_the_run_bar_draws_no_action_and_the_drawer_one_closer():
    """R-49: while Continue… was open the run bar kept a disabled Start with
    "Start disabled: close Continue… to start a new run." and the line "Run a
    new one, continued from the checkpoint chosen in …", the drawer had two
    closers (its × and its footer's Cancel), and the rarely used "Manual
    path" took a radio row near its top. Now the run bar draws none of its
    actions and says no reason; the × is the one closer; the typed path is
    behind a link beside the list's note, which switches the source."""
    demo, _ctx, by_id = _page()
    hidden = [sel for sel, body in _rules(tc.PAGE_CSS, ".shep-rb-drawer")
              if "display: none" in body]
    assert any("#shep-train-start" in sel for sel in hidden)
    assert any("#shep-train-continue" in sel for sel in hidden)
    assert any("#shep-train-stop" in sel for sel in hidden)
    ws = {"row": {"name": "demo", "trainable": True, "dataset_fingerprint": "a" * 64}}
    assert _view(ws=ws, mode={"drawer": True, "parent": {}})["reason"] == ""
    # One closer, its word drawn (it was a "×" over a 0 px wide button whose
    # word was drawn at 0 px).
    closers = [f for f in demo.fns.values() if f.fn is tc._on_continue_close]
    assert [f.targets for f in closers] == [[(by_id["shep-train-drawer-close"]._id, "click")]]
    assert by_id["shep-train-drawer-close"].value == "Close"
    (rule,) = [body for sel, body in _rules(tc.PAGE_CSS, ".shep-x-btn") if sel.endswith(".shep-x-btn")]
    assert "font-size: 15px" in rule and "flex: none !important" in rule
    assert not _rules(tc.PAGE_CSS, ".shep-x-btn::before")
    # The source: a radio kept out of view, set by the link beside the note.
    source, link = by_id["continue_source"], by_id["shep-train-source-link"]
    assert source.visible == "hidden"
    assert link.parent is by_id["shep-train-parents-note"].parent
    assert link.value == tc.source_link_html(tc.CONTINUE_FROM_LIST)
    assert "Type a path instead ›" in link.value and 'data-act="source"' in link.value
    (wired,) = [f for f in demo.fns.values() if f.fn is tc._on_source_link]
    assert wired.inputs == [source] and wired.outputs == [source, link]
    assert wired.targets == [(link._id, "click")]

    class _Evt:
        def __init__(self, act):
            self._data = {"act": act}

    assert tc._on_source_link(tc.CONTINUE_FROM_LIST, _Evt("source")) == (
        tc.CONTINUE_FROM_PATH, tc.source_link_html(tc.CONTINUE_FROM_PATH))
    assert tc._on_source_link(tc.CONTINUE_FROM_PATH, _Evt("source"))[0] == tc.CONTINUE_FROM_LIST
    assert "Choose from the list instead ›" in tc.source_link_html(tc.CONTINUE_FROM_PATH)
    assert tc._on_source_link(tc.CONTINUE_FROM_LIST, _Evt("other")) == (gr.update(), gr.update())
    # Its text is the typed path's own help (the radio's went with it).
    assert "continue_source" not in tc.FIELD_HELP
    assert "only finds a checkpoint that list has" in tc.FIELD_HELP["manual_checkpoint"][1]


def test_a_failed_runs_output_tail_is_a_disclosure_fitted_to_the_window():
    html = tc._render_error_block("exit 1\n\n--- Last output ---\n" + "\n".join(
        f"line {i}" for i in range(20)), run="gatv2/r")
    assert '<details class="shep-banner-more" open><summary>Last output' in html
    assert f"({tc.BANNER_TAIL_LINES} lines)" in html and "line 19" in html and "line 7" not in html
    one = tc._render_error_block("exit 1\n\n--- Last output ---\nonly")
    assert "(1 line)" in one
    assert "<details" not in tc._render_error_block("exit 1")
    assert "window.innerHeight < 1000" in tc._BANNER_JS and "data-tail" in tc._BANNER_JS
    _demo, _ctx, by_id = _page()
    assert by_id["shep-train-banner"].js_on_load == r.KEEP_OPEN_JS + tc._BANNER_JS


def _rules(css, needle):
    """The CSS rules whose selector names `needle`, as (selector, body)."""
    return [(sel.strip(), body) for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)
            if needle in sel]


def test_the_page_styles_hold_the_reviewed_layout():
    from src.webui.style import SHELL_CSS

    css = tc.PAGE_CSS
    # R-38: the run bar is one row at 1850 x 960 only without the build time beside the
    # fingerprint: it is the Details popover's (one click away, `details_html`), never cut.
    built = [(sel, b) for sel, b in _rules(css, "shep-ds-built") if "display: none" in b]
    assert built and all(".shep-rb-chip " in sel for sel, _b in built), built
    popover = tc.details_html({"row": {"name": "demo", "build": {
        "state": "recorded", "built_at": "2026-10-15T09:41:00Z"}}, "facts": None})
    assert "Built (UTC)" in popover and "2026-10-15 09:41" in popover
    # The selects: the shared field border (R-8: never a page's own, which was
    # #cdd2da at 1.52:1), the text clear of the chevron, grey while disabled.
    assert not any("border:" in b for _s, b in _rules(css, "#shep-train-dataset .wrap:has"))
    assert "body .gradio-container .wrap:has(> .wrap-inner)" in SHELL_CSS
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
    from tests.unit.test_webui_shell import side, stack

    narrow = stack(css)
    assert ".shep-banner-tail {{ max-height: none; overflow: visible; }}".replace(
        "{{", "{").replace("}}", "}") in narrow
    wide = side(css)
    assert "#shep-train-runview > #shep-train-charts { flex: 1 1 0 !important; min-height: 0 !important; overflow: hidden !important;" in wide
    # A failed run's banner takes height: the plots keep their minimum then too
    # (M5-3: one minimum per plot box replaced the charts' 240/370 px).
    assert "min-height: 370px" not in wide and "min-height: 240px" not in wide
    assert f".shep-chart-plot {{ min-height: {tc.PLOT_MIN_HEIGHT + tc.PLOT_AXIS_ROOM}px; }}" in wide



def test_the_continue_drawer_fits_and_says_the_plan_once():
    """M5-3: at 1920 x 1080 the drawer scrolled (821 px in 684) with Extra
    epochs off screen: an empty footer took room, the fields were looser than
    the parameters', and the check's headline repeated the plan line. (B1c:
    the footer holds the start beside its reason; an empty reason or notice
    takes no room in it.)"""
    assert ("#shep-train-drawer-why:not(:has(.shep-drawer-why)), "
            f"{tc._PAGE} #shep-train-drawer-notice:not(:has(.shep-notice-t)) "
            "{ display: none !important; }") in tc.PAGE_CSS
    # Its blocks as dense as its field row's measure says (D03: one token for
    # both), by the shared dense form (`style.py`'s `shep-form-dense`, the
    # parameters' and Fine-tune's form's too).
    from src.webui.style import SHELL_CSS

    _demo, _ctx, by_id = _page()
    for body in ("shep-train-drawer-bd", "shep-train-params-bd"):
        assert "shep-form-dense" in by_id[body].elem_classes, body
    assert ".shep-form-dense { --shep-field-pad-b: 4px; }" in SHELL_CSS
    assert (".shep-form-dense .block.padded { padding-top: var(--shep-field-pad-b) !important; "
            "padding-bottom: var(--shep-field-pad-b) !important; }") in SHELL_CSS
    assert "#shep-train-drawer-bd .block.padded" not in tc.PAGE_CSS     # no copy of its own
    assert "#shep-train-params-bd .form > .block" not in tc.PAGE_CSS
    preview = {"parent": "/w/demo/runs/gatv2/r/checkpoints/last.pt", "ok": True,
               "action": "Extend training by 30 epochs at a fixed rate of 1e-05 (epochs 151-180, "
                         "new target 180)", "sha256": "c" * 64, "inherited": {}}
    check = tc._check_html(preview, tc.CONTINUE_FROM_LIST)
    assert check.count("Extend training by 30 epochs") == 1
    assert r.status_mark("ok", "Checkpoint read; it can continue on this workspace.") in check


def test_the_drawers_refusals_lead_with_the_shared_status_mark():
    """Ops-wiring round B1c (B0c's leftover): the check's and the dataset
    comparison's heads drew a page glyph, "⚠", for what refuses the start.
    Each is the shared mark's ✕ of a refusal (`renderers.status_mark`, the
    skill's §6), its words beside it, tinted red; none is a page glyph."""
    import inspect

    refused = tc._check_html({"ok": False, "parent": "/w/x.pt", "reason": "no"},
                             tc.CONTINUE_FROM_PATH)
    assert r.status_mark("error", "Cannot continue.") in refused
    assert "shep-check-red" in refused
    branch = tc._check_html({"ok": False, "parent": "/w/x.pt", "reason": "no"},
                            tc.CONTINUE_FROM_PATH, operation=tc.BRANCH)
    assert r.status_mark("error", "Cannot branch.") in branch
    preview = {"parent": "/w/demo/runs/gatv2/r/checkpoints/last.pt", "ok": True,
               "action": "Resume the original training (1/2 done, 1 left)", "sha256": "c" * 64,
               "inherited": {}}
    differs = tc._check_html(preview, tc.CONTINUE_FROM_LIST, differs=True)
    assert r.status_mark("error", "Checkpoint read; cannot continue on the current dataset.") \
        in differs
    ws = {"name": "demo", "row": {"name": "demo", "dataset_fingerprint": "a" * 64}, "listed": []}
    compare = tc.compare_html({"chosen": True, "fingerprint": "e" * 64}, ws)
    assert r.status_mark("error", "The parent’s dataset is not this workspace’s current one") \
        in compare and "shep-cmp-red" in compare
    assert "⚠" not in inspect.getsource(tc)
    assert "shep-cmp-amber" not in tc.PAGE_CSS and "shep-check-amber" not in tc.PAGE_CSS


def test_the_drawers_notice_is_a_refusal_with_the_shared_mark():
    """The B1c review: the drawer's footer notice, which says only refusals
    (the job runner's "The checkpoint changed since it was chosen…", "Not
    started: …", D8's), was amber text with no mark, the colour of a
    condition that does not block. It is the shared mark's ✕ of a refusal
    (`renderers.status_mark`, the skill's §6), its words beside it, red; no
    amber is drawn for it."""
    said = "Not started: the job runner refused it."
    assert tc.notice_html(said, mark="error") == (
        f'<span class="shep-notice-t">{r.status_mark("error", said)}</span>')
    assert tc.notice_html("", mark="error") == ""
    rule = [b for sel, b in _rules(tc.PAGE_CSS, "#shep-train-drawer-notice")
            if sel == f"{tc._PAGE} #shep-train-drawer-notice"]
    assert rule and not any("amber" in body for body in rule)
    # D8's refusal of the start, should one reach it, is the same mark's.
    notice, *_rest = tc._on_resume(READ_ONLY, tc.BRANCH, {}, "", "cpu", tc.CONTINUE_FROM_LIST,
                                   None, "", None, None, {}, {}, [],
                                   *(None,) * len(tc.BRANCH_FIELDS))
    refusal = tc.operator_access(READ_ONLY).refusal
    assert notice == f'<span class="shep-notice-t">{r.status_mark("error", refusal)}</span>'


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
    # The parent's dataset under its details (R-48), then step 3's row, then
    # the check's answer.
    body = by_id["shep-train-drawer-bd"].children
    assert body.index(by_id["shep-train-parent-details"]) < body.index(
        by_id["shep-train-compare"]) < body.index(row) < body.index(by_id["continue_preview"])
    from tests.unit.test_webui_shell import side

    wide = side(tc.PAGE_CSS).replace(f"{tc._PAGE} ", "")
    assert ("#continue_operation, #continue_device { display: flex "
            "!important;") in wide
    assert "#checkpoint_dropdown .container { display: flex;" not in wide   # labels above
    assert "#shep-train-drawer-ft { padding: 8px 16px !important; }" in wide
    # When the datasets differ, the comparison says the start refuses; the
    # check does not add its general line, and a rate is never split.
    preview = {"parent": "/w/demo/runs/gatv2/r/checkpoints/last.pt", "ok": True,
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
    popover, which sits in the chip (z-index 20). The chip is never narrower
    than its widest part (D08: stacked at 960 and 1440 px wide its box shrank
    to 242 px under the 292 px fingerprint line, whose Copy and ⓘ ran past it,
    so the ⓘ's bubble, placed beside the box, covered the ⓘ)."""
    assert tc.CHIP_Z_INDEX > 40
    assert (f"#shep-train-chip {{ flex: 0 1 340px !important; min-width: min-content !important; "
            f"position: relative; z-index: {tc.CHIP_Z_INDEX}; }}") in tc.PAGE_CSS


def test_the_drawers_start_sits_in_its_footer_beside_its_reason():
    """Ops-wiring round B1c (the skill's §9: a refusal beside what it
    disables): Continue Training, or Start branch, and Cancel sat in the run
    bar above the drawer, the drawer saying "Continue Training, in the run
    bar above, is disabled: …", and the run bar's reason said two sentences
    (M5-3 broke them between their word groups). Now they are the drawer's
    footer, under its body, with the start's reason and its last notice; the
    run bar says one sentence, why its own Start waits."""
    _demo, _ctx, by_id = _page()
    footer = by_id["shep-train-drawer-ft"]
    # R-49: one closer, the × in the drawer's header; no Cancel beside the start.
    assert [getattr(child, "elem_id", None) for child in footer.children] == [
        "shep-train-step-start", "shep-train-resume", "shep-train-drawer-why",
        "shep-train-drawer-notice"]
    drawer = by_id["shep-train-drawer"]
    assert drawer.children.index(by_id["shep-train-drawer-bd"]) < drawer.children.index(footer)
    runbar = by_id["shep-train-runbar"]

    def inside(block, parent):
        while block is not None:
            if block is parent:
                return True
            block = getattr(block, "parent", None)
        return False

    assert not inside(by_id["shep-train-resume"], runbar)
    assert "shep-train-cancel" not in by_id
    assert "shep-drawer-only" not in tc.PAGE_CSS
    ws = {"row": {"name": "demo", "trainable": True, "dataset_fingerprint": "a" * 64}}
    refused = _view(ws=ws, mode={"drawer": True, "parent": {"chosen": True, "refused": True}})
    assert refused["reason"] == ""
    assert ("<span>Continue Training disabled: the check says this checkpoint cannot be "
            "continued.</span>") in refused["drawer_reason"]


def test_the_plots_keep_a_usable_minimum_and_the_resources_line_its_own_row():
    """M5-3: at 1536 x 760 the plots were 70 to 100 px tall; each plot box keeps
    room for a plot area of PLOT_MIN_HEIGHT px. The GB10 WebUI recheck: the
    resources line, kept over the scrolling plots (sticky), hid the last axis
    and its label. The run view is now three rows, and only the plots' body
    scrolls when their minimum does not fit: the header and the resources
    line are rows of their own, the chart toolbar a row of its own above the
    body (B1b: it was sticky over the top row's titles and plots once
    scrolled), nothing is drawn over an axis, and no drawn chart's axis label
    is hidden to make room (an empty chart's goes with its plot: BACKLOG 69).
    The floor was raised from 140 with the room B1b took back (measured in a
    browser: all four plots and their axes in view at 1850 x 960). A run view
    too short for one row of plots (`RUN_VIEW_ONE_ROW`, a container query on
    the page row's own height: 288 px at 1280 x 600, where the body was 106 px
    and a focused plot's centre sat under the resources line) scrolls as a
    whole instead, its plots and the resources line in flow, nothing sticky."""
    assert tc.PLOT_MIN_HEIGHT >= 170
    from tests.unit.test_webui_shell import side

    css = tc.PAGE_CSS
    wide = side(css)
    plot_box = tc.PLOT_MIN_HEIGHT + tc.PLOT_AXIS_ROOM
    assert f"#shep-train-charts .shep-chart-plot {{ min-height: {plot_box}px; }}" in wide
    assert "#shep-train-charts .shep-charts-grid { grid-auto-rows: minmax(min-content, 1fr); }" in wide
    # Only !important beats the inline "overflow: visible" Gradio writes on the block.
    assert ("#shep-train-runview > #shep-train-charts { flex: 1 1 0 !important; min-height: 0 "
            "!important; overflow: hidden !important;") in wide
    assert "#shep-train-runview > #resource_display { flex: none !important; }" in wide
    assert "position: sticky; bottom" not in css and "#shep-train-runview { overflow-y" not in wide
    assert "#shep-train-page { container: shep-train-page / size; }" in wide
    short = css.split(f"@container shep-train-page (max-height: {tc.RUN_VIEW_ONE_ROW}px) {{", 1)[1].split("\n}", 1)[0]
    assert "#shep-train-runview { overflow-y: auto !important; }" in short
    assert "#shep-train-runview > #shep-train-charts { flex: none !important; }" in short
    assert "#shep-train-charts .shep-charts-body { flex: none; overflow-y: visible; }" in short
    assert "sticky" not in short and tc.RUN_VIEW_ONE_ROW >= tc.PLOT_MIN_HEIGHT + tc.PLOT_AXIS_ROOM
    assert "#shep-train-charts .shep-charts-bar { flex: none; min-height: 34px; padding-bottom: 4px; }" in wide
    assert ("#shep-train-charts .shep-charts-body { flex: 1 1 0; min-height: 0; overflow-y: auto; }"
            in wide)
    assert not [b for _s, b in _rules(css, "shep-charts-bar") if "sticky" in b]
    # The boxes inside the row take its height, the cards keep their content's minimum, and
    # the grid its rows' (so the epoch label under it is never under a row: R-54).
    assert "#shep-train-charts .shep-charts { min-height: 0; }" in wide
    assert "#shep-train-charts .shep-charts-grid { min-height: min-content; }" in wide
    assert "#shep-train-charts .shep-chart { min-height: auto; }" in wide
    hidden = [sel for sel, b in _rules(css, "shep-chart-empty") if "display: none" in b]
    assert hidden == [".shep-chart-empty .shep-chart-plot"], hidden
    _demo, _ctx, by_id = _page()
    # Drawn again by each render: kept as the reader left them (T08-flow-2),
    # by the keeper every dashboard HTML runs first (`shell.plain_html`).
    assert by_id["shep-train-run-head"].js_on_load == r.KEEP_OPEN_JS + tc._POPOVER_JS
    assert by_id["shep-train-chip"].js_on_load == r.KEEP_OPEN_JS + tc._POPOVER_JS
    assert r.KEEP_OPEN_JS not in tc._POPOVER_JS
    assert "--shep-rv-top-h" not in css


#: The run view header's and the chip's "Details ›" popovers across the
#: redraws a running job makes (the test below): a popover opened with a
#: click, the server's next drawing pushed with no
#: pointer or key of the reader's (an element's `click()`, which sends no
#: pointer event, as the poll's render sends none), read after each drawing.
_POPOVER_REDRAW_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1850, 960], [960, 1080]]) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h },
                                                   deviceScaleFactor: 1 })).newPage();
    // The page's State keeps its heartbeat stream open: wait for what is drawn.
    await page.goto(base, { waitUntil: 'load' });
    await page.waitForSelector('#shep-train-run-head [data-act="details"]');
    await page.waitForSelector('#shep-train-chip [data-act="details"]');
    const read = (id) => page.evaluate((id) => {
      const el = document.querySelector('#' + id);
      const button = el.querySelector('[data-act="details"]');
      const pop = button && el.querySelector('#' + button.getAttribute('aria-controls'));
      const f = document.activeElement;
      return { open: !!pop && !pop.hidden && button.getAttribute('aria-expanded') === 'true',
               shown: !!pop && pop.checkVisibility(),
               focus: f && f !== document.body
                 ? (f.closest('[id]') || f).id + ' ' + (f.getAttribute('data-act') || f.className)
                 : 'body',
               sameFocus: !!window.__focus && f === window.__focus,
               text: el.innerText.replace(/\s+/g, ' ').slice(0, 160) };
    }, id);
    const mark = () => page.evaluate(() => { window.__focus = document.activeElement; });
    // The server's next drawing of both: its step number in the header moves.
    const stepNow = () => page.evaluate(() =>
      document.querySelector('#shep-train-run-head').textContent.match(/step (\d+)/)[1]);
    const push = async (button = '#step') => {
      const before = await stepNow();
      await page.evaluate((b) => document.querySelector(b).click(), button);
      await page.waitForFunction((n) => document.querySelector('#shep-train-run-head')
        .textContent.match(/step (\d+)/)[1] !== n, before, { timeout: 20000 });
    };
    const size = {};
    for (const id of ['shep-train-run-head', 'shep-train-chip']) {
      const seen = {};
      const details = '#' + id + ' [data-act="details"]';
      await page.click(details);
      await page.waitForFunction((d) => document.querySelector(d)
        .getAttribute('aria-expanded') === 'true', details);
      await mark();
      seen.opened = await read(id);
      await push(); seen.redrawn = await read(id);
      await push(); seen.again = await read(id);
      await page.keyboard.press('Escape');
      seen.escaped = await read(id);
      await mark(); await push(); seen.afterEscape = await read(id);
      // Opened, then closed by a click outside it; a drawing leaves it closed.
      await page.click(details);
      await page.mouse.click(Math.floor(w / 2), h - 8);
      seen.outside = await read(id);
      await push(); seen.afterOutside = await read(id);
      // Closed with Escape (focus on its button), then a click on the page
      // itself, nothing focusable: focus is the page's, and the drawings
      // that follow (the chip's lead coming and going) leave it there, so
      // Space opens nothing.
      await page.click(details);
      await page.keyboard.press('Escape');
      await page.mouse.click(Math.floor(w / 2), h - 8);
      seen.leftForPage = await read(id);
      seen.pageRedrawn = [];
      for (let i = 0; i < 2; i++) { await push(); seen.pageRedrawn.push(await read(id)); }
      await page.keyboard.press('Space');
      seen.space = await read(id);
      size[id] = seen;
    }
    // Another run viewed and another dataset: the popover is drawn closed.
    await page.click('#shep-train-run-head [data-act="details"]');
    await push('#other');
    size.otherRun = await read('shep-train-run-head');
    out[w + 'x' + h] = size;
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_details_popovers_stay_as_the_reader_left_them_across_redraws(tmp_path):
    """T08-flow-2: the run view header's "Run details" closed by itself 0.5-1.5 s
    after it was opened while a job ran, when the header was drawn again
    ("dataset recorded at the first checkpoint" became "same dataset as the
    run bar"), and the chip's dataset details closed in the drawing that said
    "Completed": Gradio 6.30 draws ``gr.HTML`` again in place and sets each
    element's attributes to the new HTML's, which draws each popover closed,
    and the popover's open state was kept in those attributes only. In a real
    browser (skipped without Node and Playwright), at 1850 x 960 and
    960 x 1080, both with the page's own ``js_on_load`` and drawn by their
    own renderers, the server pushing each next drawing as a running job's
    render does (the header's relation and state words, the chip's "in use
    by the running job" coming and going): a popover the reader opened stays
    open through each drawing with the focus where it was (its × , where its
    open put it); closed with Escape or a click outside it, it stays closed
    through the next; and a header drawn for another run draws it closed.
    Fails on e0e4237, whose popovers were closed by the first drawing. Focus
    on a popover's button that a drawing replaces stays on it; focus the
    reader gave the page (Escape, then a click on nothing focusable) stays
    the page's through the next drawings, so Space opens nothing (the review
    of 8b75d85: the keeper put focus back on the chip's button, kept from
    before the click)."""
    from tests.fixtures.browser import node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    _demo, _ctx, by_id = _page()
    fingerprint = "ab12" * 16
    ws = {"name": "demo", "digits": 12, "facts": None,
          "row": {"name": "demo", "dataset_fingerprint": fingerprint, "trainable": True}}
    other_ws = {**ws, "name": "other", "row": {**ws["row"], "name": "other"}}
    run = {"label": "probe-run", "path": "/nowhere/runs/probe-run"}
    first = {"step": 0, "run": run, "ws": ws}

    def drawn(shown):
        # Even steps: the job running, its dataset first not yet recorded;
        # odd ones: the job ended, its dataset the run bar's.
        n = shown["step"]
        running = n % 2 == 0
        snapshot = None if n == 0 else {"dataset_fingerprint": fingerprint}
        words = (f"Running · epoch {n + 1} of 40 · step {n}" if running
                 else f"Finished · {n + 1} of 40 epochs completed · step {n}")
        head = tc.run_head_html(shown["run"], shown["ws"], operator=True, snapshot=snapshot,
                                live=running, state=tc._state(words),
                                relation="the running job" if running else "the last job")
        return tc.chip_html(shown["ws"], in_use=running), head

    def step(shown):
        shown = {**shown, "step": shown["step"] + 1}
        return (*drawn(shown), shown)

    def other(shown):
        return step({**shown, "ws": other_ws,
                     "run": {"label": "another-run", "path": "/nowhere/runs/another-run"}})

    with gr.Blocks() as page:
        shown = gr.State(first)
        chip = shell.plain_html(drawn(first)[0], elem_id="shep-train-chip",
                                js_on_load=tc._POPOVER_JS)
        head = shell.plain_html(drawn(first)[1], elem_id="shep-train-run-head",
                                js_on_load=tc._POPOVER_JS)
        gr.Button("Step", elem_id="step").click(step, inputs=[shown], outputs=[chip, head, shown])
        gr.Button("Other", elem_id="other").click(other, inputs=[shown],
                                                  outputs=[chip, head, shown])
    with served(page) as base:
        seen = run_node(_POPOVER_REDRAW_JS, tmp_path, base[:-len("gradio_api")])
    for size, popovers in seen.items():
        for name in ("shep-train-run-head", "shep-train-chip"):
            got = popovers[name]
            opened = got["opened"]
            assert opened["open"] and opened["shown"] and opened["focus"].endswith(" close"), \
                (size, name, got)
            for after in ("redrawn", "again"):
                assert got[after]["open"] and got[after]["shown"], (size, name, after, got)
                assert got[after]["focus"] == opened["focus"], (size, name, after, got)
            assert got["opened"]["text"] != got["redrawn"]["text"], (size, name, got)
            assert not got["escaped"]["open"] and got["escaped"]["focus"].endswith(" details")
            # Its button drawn again under the reader's focus keeps that focus:
            # the chip's is drawn anew there (its lead went), so the keeper
            # put focus back on the new one.
            assert got["afterEscape"]["focus"] == got["escaped"]["focus"], (size, name, got)
            if name == "shep-train-chip":
                assert not got["afterEscape"]["sameFocus"], (size, name, got)
            for closed in ("afterEscape", "outside", "afterOutside"):
                assert not got[closed]["open"] and not got[closed]["shown"], \
                    (size, name, closed, got)
            # Focus the reader gave the page stays the page's through the
            # drawings (review of 8b75d85: the keeper put it back on the
            # chip's button, and Space then opened its popover).
            assert got["leftForPage"]["focus"] == "body", (size, name, got)
            for after in (*got["pageRedrawn"], got["space"]):
                assert after["focus"] == "body" and not after["open"], (size, name, got)
        assert not popovers["otherRun"]["open"], (size, popovers["otherRun"])


#: The Training page as the service mounts it: four settings typed, so the
#: "Set by you" line names three and counts one, with its "Details ›"; the
#: Details opened by the reader; a fifth setting typed (the reader's own act
#: draws the line again); then the Details closed by the reader and one more
#: edit. What the line says and whether its Details is open after each.
_SETBY_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 })).newPage();
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.waitForSelector('#num_epochs input', { state: 'visible', timeout: 60000 });
  const line = '#shep-train-setby';
  const says = (words) => page.waitForFunction(([s, w]) => document.querySelector(s).innerText.includes(w),
                                               [line, words], { timeout: 30000 });
  const type = async (id, value) => {
    await page.locator(`#${id} input`).click();
    await page.keyboard.press('Control+A');
    await page.keyboard.type(value);
  };
  const read = async () => page.evaluate((s) => {
    const d = document.querySelector(`${s} details`);
    return { open: d ? d.open : null, text: document.querySelector(s).innerText.split('\n')[0] };
  }, line);
  const out = {};
  await type('num_epochs', '5');
  await type('learning_rate', '0.001');
  await type('batch_size', '256');
  await type('seed', '3');
  await says('and 1 more');
  await page.click(`${line} details > summary`);
  out.opened = await read();
  await type('split_seed', '4');
  await says('and 2 more');
  out.redrawn = await read();
  await page.click(`${line} details > summary`);
  out.closed = await read();
  await type('num_epochs', '7');
  await says('Epochs 7');
  out.again = await read();
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_details_the_reader_opened_stays_open_when_their_own_edit_draws_it_again(tmp_path):
    """The verification of fcc3dea: the disclosure keeper (T08-flow-2) was
    wired only to regions a poll or a render draws again, so a region the
    reader's own act draws again closed what the reader had opened: the
    "Set by you" line's Details, opened, closed under the next field the
    operator set. Every dashboard HTML now runs the one keeper
    (`shell.plain_html`): opened, it stays open across that edit; closed by
    the reader, it stays closed across the next."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_SETBY_JS, tmp_path, base[:-len("gradio_api")])
    assert seen["opened"]["open"] is True, seen
    assert seen["redrawn"]["text"].endswith("and 2 more"), seen
    assert seen["redrawn"]["open"] is True, seen
    assert seen["closed"]["open"] is False, seen
    assert "Epochs 7" in seen["again"]["text"] and seen["again"]["open"] is False, seen


#: The Training page for an operator with no workspaces folder set, so the
#: run bar's reason is the GPU lock's long refusal: at each window, once the
#: reason says it and the page's requests are answered, what
#: `document.elementFromPoint` finds at the centre of each run bar control
#: drawn, and the reason's and the controls' boxes (a screenshot of each
#: window where a third argument names its path's start).
_RUNBAR_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[960, 1080], [1280, 900], [390, 844], [1850, 960]]) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1 })).newPage();
    await page.addInitScript(QUEUE_JS);
    await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
    await page.click('#shep-tab-research-button', { timeout: 60000 });
    await page.waitForFunction(() => {
      const r = document.querySelector('#shep-train-reason');
      return r && r.innerText.includes('GPU lock') && window.__shepQueue.idle();
    }, null, { timeout: 60000 });
    if (process.argv[3]) { await page.screenshot({ path: `${process.argv[3]}_${w}x${h}.png` }); }
    out[`${w}x${h}`] = await page.evaluate(() => {
      const box = (el) => { const r = el.getBoundingClientRect(); return [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)]; };
      const shown = (el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden'; };
      const controls = ['#shep-train-dataset input', '#run_name input', '#shep-train-start', '#shep-train-continue', '#shep-train-stop']
        .map((sel) => [sel, document.querySelector(sel)]).filter(([, el]) => el && shown(el));
      const reason = document.querySelector('#shep-train-reason .shep-rb-why') || document.querySelector('#shep-train-reason');
      return { reason: box(reason), controls: controls.map(([sel, el]) => {
        el.scrollIntoView({ block: 'center' });
        const r = el.getBoundingClientRect(), at = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        return { sel, box: box(el), hit: el === at || el.contains(at), at: at ? (at.id || at.className.toString().slice(0, 40) || at.tagName) : null };
      }) };
    });
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_no_run_bar_control_is_under_the_reason_at_any_window(monkeypatch, tmp_path):
    """The verification of fcc3dea (pre-existing, seen by the T08 fix lane):
    at 960 x 1080, for an operator with no workspaces folder set, the run
    bar's reason (the GPU lock's refusal, several lines) ran over Start and
    Continue…, and `elementFromPoint` at Start's centre found the reason's
    text: its slot shrank below its words (its longest word and the GPU
    tag), which ran on over the buttons. The slot is never narrower than its
    words, so where they do not fit beside the controls the line takes a row
    of its own (the skill's §2): at 960 x 1080, 1280 x 900, 390 x 844 and
    1850 x 960 every run bar control drawn is the first thing at its
    centre."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_mode(monkeypatch)
    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_RUNBAR_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    assert set(seen) == {"960x1080", "1280x900", "390x844", "1850x960"}, seen
    for size, drawn in seen.items():
        names = [control["sel"] for control in drawn["controls"]]
        assert {"#shep-train-start", "#shep-train-continue"} <= set(names), (size, drawn)
        covered = [control for control in drawn["controls"] if not control["hit"]]
        assert covered == [], (size, drawn)
