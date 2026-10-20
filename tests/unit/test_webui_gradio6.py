"""
The dashboard's Gradio 6.30 adjustments that a browser would otherwise show.
============================================================================
- **Native plots.** A LinePlot fixes the series it can draw only when it
  rebuilds its chart, and a new value alone no longer rebuilds it (Gradio PR
  #11306): a plot that starts empty never draws a series that appears later.
  The Training Console starts each plot from a typed empty DataFrame with
  fixed colours, sends data only on each poll, and follows each data step with
  `_legend_step`, which sends `colors_in_legend` only when the set of series
  differs from what that browser session last received ([] for no rows). No
  placeholder rows. The Epoch Window filters rows and sends no ``x_lim``
  (which made 6.30 rebuild every tick, one poll behind).
- **gr.HTML padding** is off by default in Gradio 6; the components laid out
  with Gradio 5's padding ask for it.
- **Timers in an unopened tab do not tick** in Gradio 6 (gradio#11984,
  #11993): Runtime Settings refreshes its restart lock when its tab is
  selected.

Module: tests/unit/test_webui_gradio6.py
"""
import pytest

gr = pytest.importorskip("gradio")
pd = pytest.importorskip("pandas")

from src.webui.components import training_console as tc  # noqa: E402

NO_CHANGE = gr.update()


def _loss(*rows):
    """A loss frame from (epoch, split) pairs, as `_build_loss_df` shapes it."""
    return pd.DataFrame([{"epoch": e, "loss": 1.0 / e, "split": s} for e, s in rows])


def _metric(label, epochs):
    return pd.DataFrame([{"epoch": e, "value": 0.1 * e, "metric": label} for e in epochs])


# --------------------------------------------------------------- the empty value
def test_the_empty_values_are_typed_empty_frames():
    for frame, columns in ((tc._EMPTY_LOSS_DF, ["epoch", "loss", "split"]),
                           (tc._EMPTY_METRIC_DF, ["epoch", "value", "metric"])):
        assert frame.empty and list(frame.columns) == columns
        assert [str(t) for t in frame.dtypes][:2] == ["float64", "float64"]
        assert frame[columns[2]].dtype == object


@pytest.fixture
def poll(monkeypatch):
    """`_poll_status` with its module state put back afterwards (the refresh
    cache, the banner key) and no PyG probe."""
    monkeypatch.setattr(tc, "_poll_cache_key", None)
    monkeypatch.setattr(tc, "_current_banner_key", tc._current_banner_key)
    monkeypatch.setattr(tc, "_pyg_warning_cache", None)
    monkeypatch.setattr(tc.training_manager, "get_system_resources", lambda: {})
    return tc._poll_status


def test_no_rows_is_sent_as_the_typed_empty_frame_not_a_placeholder(poll, monkeypatch):
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: {"status": "idle"})
    out = poll("All")
    assert out[1] is tc._EMPTY_LOSS_DF
    assert all(frame is tc._EMPTY_METRIC_DF for frame in out[2:5])
    assert tc._build_loss_df([], []).empty and tc._build_metric_df([], "x", "X").empty


# --------------------------------------------------------------- series and legend
def test_the_series_present_in_a_plot_output():
    assert tc._series_present(_loss((1, "val"), (1, "train"), (2, "train")), "split") == [
        "train", "val"]
    assert tc._series_present(tc._EMPTY_LOSS_DF, "split") == []
    assert tc._series_present(NO_CHANGE, "split") is None
    assert tc._series_present(gr.update(value=_loss((1, "train"))), "split") == ["train"]


def test_what_a_data_step_sent_and_a_no_op_step_leaves_alone():
    sent = tc._series_sent([_loss((1, "train")), tc._EMPTY_METRIC_DF,
                            _metric("Hits@10", [1]), NO_CHANGE])
    assert sent == [["train"], [], ["Hits@10"], None]
    # A poll that changes no plot does not overwrite what the last data step
    # sent before its legend step has run.
    assert tc._series_sent([NO_CHANGE] * 4) == NO_CHANGE


def test_the_legend_is_sent_only_when_the_set_of_series_changes():
    seen = tc._NO_SERIES
    timeline = [
        ([[], [], [], []], [None, None, None, None]),                 # no rows yet
        ([["train"], [], [], ["Learning Rate"]], [["train"], None, None, ["Learning Rate"]]),
        ([["train"], [], [], ["Learning Rate"]], [None, None, None, None]),   # same set
        ([["train", "val"], ["MRR"], ["Hits@1", "Hits@10"], ["Learning Rate"]],
         [["train", "val"], ["MRR"], ["Hits@1", "Hits@10"], None]),
        ([None, None, None, None], [None, None, None, None]),         # plots left as they were
        ([[], [], [], []], [[], [], [], []]),                         # a new run: no rows
        ([["train"], [], [], []], [["train"], None, None, None]),
    ]
    for sent, expected in timeline:
        previous = seen
        *updates, seen = tc._legend_step(sent, previous)
        for update, legend in zip(updates, expected, strict=True):
            if legend is None:
                assert update == NO_CHANGE, (sent, updates)
            else:
                # Only the legend: never a value, so no row is ever invented.
                assert update == gr.update(colors_in_legend=legend), (sent, updates)
        assert seen == [s if s is not None else p for s, p in zip(sent, previous, strict=True)]
    assert seen == [["train"], [], [], []]


def test_a_new_browser_session_starts_from_empty_legends():
    *updates, seen = tc._legend_step([["train"], None, None, None], None)
    assert updates[0] == gr.update(colors_in_legend=["train"])
    assert seen == [["train"], [], [], []]
    *updates, seen = tc._legend_step(None, None)
    assert all(update == NO_CHANGE for update in updates) and seen == tc._NO_SERIES


def test_a_data_step_wrapped_for_the_legend_returns_its_outputs_and_what_it_sent():
    def handler(a, b):
        return ("status", _loss((1, "train")), tc._EMPTY_METRIC_DF,
                tc._EMPTY_METRIC_DF, tc._EMPTY_METRIC_DF, "button", a + b)

    wrapped = tc._with_series_sent(handler)
    out = wrapped(1, 2)
    assert len(out) == 8 and (out[0], out[5], out[6]) == ("status", "button", 3)
    assert out[1]["split"].tolist() == ["train"]
    assert out[7] == [["train"], [], [], []]
    assert wrapped.__name__ == "handler"     # the API name Gradio derives stays the same


# --------------------------------------------------------------- the Epoch Window
def test_the_epoch_window_filters_rows_and_sends_no_x_lim(poll, monkeypatch):
    history = {"train": [{"epoch": e, "total": 1.0 + e, "learning_rate": 0.001}
                         for e in range(30)],
               "validation": [{"epoch": e, "val_loss": 2.0 + e, "val_mrr": 0.1,
                               "val_hits@1": 0.1, "val_hits@10": 0.2} for e in range(30)]}
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: {"status": "running"})
    monkeypatch.setattr(tc.training_manager, "get_metrics_history", lambda: history)
    for window, first_epoch in (("Last 10", 21), ("All", 1)):
        out = poll(window)
        for frame in out[1:5]:
            assert isinstance(frame, pd.DataFrame), type(frame)   # data only, no x_lim
            assert frame["epoch"].min() == first_epoch and frame["epoch"].max() == 30


# --------------------------------------------------------------- the console's wiring
def _training_tab():
    with gr.Blocks() as demo:
        tc.create_training_tab()
    by_elem = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    return demo, by_elem


def test_the_plots_start_typed_and_empty_with_fixed_colours_and_a_zero_based_y_axis():
    _demo, by_elem = _training_tab()
    for index, elem_id in enumerate(("loss_plot", "mrr_plot", "hits_plot", "lr_plot")):
        plot = by_elem[elem_id]
        assert isinstance(plot, gr.LinePlot)
        assert plot.color_map == tc._PLOT_SERIES[index][1]
        assert plot.color == tc._PLOT_SERIES[index][0]
        assert plot.y_lim == [0, None]
        assert plot.x_lim is None
        assert plot.value["data"] == [], "the initial value has no rows"


def test_every_event_that_sends_plot_data_is_followed_by_the_legend_step():
    demo, by_elem = _training_tab()
    plots = [by_elem[e] for e in ("loss_plot", "mrr_plot", "hits_plot", "lr_plot")]
    fns = list(demo.fns.values())
    followers = {fn.trigger_after: fn for fn in fns if fn.trigger_after is not None}
    data_steps = [fn for fn in fns if fn.fn is not None and fn.name != "_legend_step"
                  and any(out in plots for out in fn.outputs)]
    names = sorted(fn.name for fn in data_steps)
    assert names == ["_on_resume", "_on_start", "_on_stop", "_poll_status"], names
    for fn in data_steps:
        if fn.name == "_on_stop":
            continue        # sends no data: the plots are left as they are
        follower = followers.get(fn._id)
        assert follower is not None and follower.fn is tc._legend_step, fn.name
        assert follower.outputs[:4] == plots
        assert follower.show_progress == "hidden"
        assert fn.outputs[-1] is follower.inputs[0], "the step's series reach the legend step"
        assert follower.outputs[4] is follower.inputs[1], "the session's legends are kept"


# --------------------------------------------------------------- HTML padding
def test_the_html_laid_out_with_gradio_5_padding_keeps_it():
    from src.webui.app import create_gradio_app

    app = create_gradio_app()
    html = [b for b in app.blocks.values() if isinstance(b, gr.HTML)]
    by_elem = {b.elem_id: b for b in html if b.elem_id}
    for elem_id in ("runtime_restart_lock", "error_banner", "resource_display"):
        assert by_elem[elem_id].padding is True, elem_id
    badges = [b for b in html if "shep-badge-row" in str(b.value)]
    assert len(badges) == 2 and all(b.padding is True for b in badges)


# --------------------------------------------------------------- Runtime Settings
def test_selecting_runtime_settings_refreshes_the_restart_lock(monkeypatch):
    import src.webui.components.runtime_settings as rs
    from src.webui.app import create_gradio_app

    app = create_gradio_app()
    tab = next(b for b in app.blocks.values()
               if isinstance(b, gr.Tab) and b.label == "Runtime Settings")
    on_select = [fn for fn in app.fns.values() if (tab._id, "select") in fn.targets]
    assert len(on_select) == 1 and on_select[0].name == "_refresh_restart_lock"
    assert [o.elem_id for o in on_select[0].outputs] == ["runtime_restart", "runtime_restart_lock"]

    monkeypatch.setattr(rs, "is_training_active", lambda: True)
    button, note = on_select[0].fn()
    assert button["interactive"] is False and note["visible"] is True
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    button, note = on_select[0].fn()
    assert button["interactive"] is True and note["visible"] is False
