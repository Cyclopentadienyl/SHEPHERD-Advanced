"""
The dashboard's Gradio 6.30 adjustments that a browser would otherwise show.
============================================================================
- **gr.HTML padding** is off by default in Gradio 6; the components laid out
  with Gradio 5's padding ask for it.
- **Timers in an unopened tab do not tick** in Gradio 6 (gradio#11984,
  #11993): Runtime Settings refreshes its restart lock when its tab is
  selected, and then on the shell's slow timer (outside the tabs) while it is
  shown; it has no timer of its own.

(The Training Console's native-plot workarounds, the typed empty frames, the
legend step after each data step and the Epoch Window's row filter, went with
the LinePlots in M5-2: Training draws its charts with the shared uPlot
component, `tests/unit/test_webui_charts.py` and `test_training_page.py`.)

Module: tests/unit/test_webui_gradio6.py
"""
import pytest

gr = pytest.importorskip("gradio")


# --------------------------------------------------------------- HTML padding
def test_the_html_laid_out_with_gradio_5_padding_keeps_it():
    from src.webui.app import create_gradio_app

    app = create_gradio_app()
    html = [b for b in app.blocks.values() if isinstance(b, gr.HTML)]
    by_elem = {b.elem_id: b for b in html if b.elem_id}
    for elem_id in ("runtime_restart_lock",):
        assert by_elem[elem_id].padding is True, elem_id
    badges = [b for b in html if "shep-badge-row" in str(b.value)]
    assert len(badges) == 2 and all(b.padding is True for b in badges)


# --------------------------------------------------------------- Runtime Settings
def test_selecting_runtime_settings_refreshes_the_restart_lock(monkeypatch):
    import src.webui.components.runtime_settings as rs
    from src.webui.app import create_gradio_app

    app = create_gradio_app()
    # Runtime Settings is the page under ⚙ Settings (M5-2).
    tab = next(b for b in app.blocks.values()
               if isinstance(b, gr.Tab) and b.elem_id == "shep-tab-settings")
    on_select = [fn for fn in app.fns.values() if (tab._id, "select") in fn.targets
                 and fn.name == "_refresh_restart_lock"]
    assert len(on_select) == 1
    assert [o.elem_id for o in on_select[0].outputs] == [
        "runtime_apply", "runtime_restart", "runtime_restart_lock", "runtime_access_reason"]

    from tests.fixtures.operator_access import operator_request

    monkeypatch.setattr(rs, "is_training_active", lambda: True)
    apply, button, note, reason = on_select[0].fn(operator_request(monkeypatch))
    assert apply["interactive"] is True and button["interactive"] is False
    assert note["visible"] is True and reason["visible"] is False
    # Off while training runs, but not refused: no lock (the WebUI skill's §3).
    assert (apply["value"], button["value"]) == (rs.APPLY_LABEL, rs.RESTART_LABEL)
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    apply, button, note, reason = on_select[0].fn(operator_request(monkeypatch))
    assert button["interactive"] is True and note["visible"] is False


def test_read_only_settings_lock_apply_and_restart_with_the_reason(monkeypatch):
    """The one gated-button rule (`chips.gated`, the WebUI skill's §3): a
    control a read-only request cannot use is disabled with the lock and the
    visible "Operator controls only: …" reason beside it, in D8's short words
    (each handler still checks its own request)."""
    import src.webui.components.runtime_settings as rs
    from tests.fixtures.operator_access import gradio_request

    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    apply, button, note, reason = rs._refresh_restart_lock(gradio_request())
    assert (apply["value"], apply["interactive"]) == (f"🔒 {rs.APPLY_LABEL}", False)
    assert (button["value"], button["interactive"]) == (f"🔒 {rs.RESTART_LABEL}", False)
    assert reason["visible"] is True and reason["value"] == (
        '<div class="shep-reason shep-reason-grey">🔒 Operator controls only: deployment mode '
        "not set to M1.</div>")


def test_the_restart_lock_is_polled_on_the_shells_slow_timer_only_while_shown(monkeypatch):
    """M5-3 review ARCH-8: Settings kept a queued 4 s gr.Timer of its own,
    which, once the page had been opened, ticked on every page for the life
    of the browser tab. The lock is now read on the shell's slow timer, with
    no queue, and only while Settings is shown; the dashboard has no timer
    but the shell's two."""
    import src.webui.components.runtime_settings as rs
    from src.webui import shell
    from src.webui.app import create_gradio_app

    app = create_gradio_app()
    timers = [b for b in app.blocks.values() if isinstance(b, gr.Timer)]
    assert sorted(t.value for t in timers) == [shell.FAST_SECONDS, shell.SLOW_SECONDS]
    (poll,) = [fn for fn in app.fns.values() if fn.name == "_poll_restart_lock"]
    slow = next(t for t in timers if t.value == shell.SLOW_SECONDS)
    assert poll.targets == [(slow._id, "tick")] and poll.queue is False
    assert isinstance(poll.inputs[0], gr.State)
    assert [o.elem_id for o in poll.outputs] == [
        "runtime_apply", "runtime_restart", "runtime_restart_lock", "runtime_access_reason"]
    from tests.fixtures.operator_access import operator_request

    monkeypatch.setattr(rs, "is_training_active", lambda: pytest.fail("read on another page"))
    assert poll.fn("clinical", operator_request(monkeypatch)) == (gr.update(),) * 4
    monkeypatch.setattr(rs, "is_training_active", lambda: True)
    apply, button, note, _reason = poll.fn("settings", operator_request(monkeypatch))
    assert button["interactive"] is False and note["visible"] is True
