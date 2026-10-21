"""
The dashboard's Gradio 6.30 adjustments that a browser would otherwise show.
============================================================================
- **gr.HTML padding** is off by default in Gradio 6; the components laid out
  with Gradio 5's padding ask for it.
- **Timers in an unopened tab do not tick** in Gradio 6 (gradio#11984,
  #11993): Runtime Settings refreshes its restart lock when its tab is
  selected.

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
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    apply, button, note, reason = on_select[0].fn(operator_request(monkeypatch))
    assert button["interactive"] is True and note["visible"] is False


def test_read_only_settings_grey_out_apply_and_restart_with_the_reason(monkeypatch):
    """The general D8 rule: a control a read-only request cannot use is
    greyed out with the reason as visible text (each handler still checks its
    own request)."""
    import src.webui.components.runtime_settings as rs
    from tests.fixtures.operator_access import gradio_request

    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    apply, button, note, reason = rs._refresh_restart_lock(gradio_request())
    assert apply["interactive"] is False and button["interactive"] is False
    assert reason["visible"] is True and "operator controls, off for this request: " in (
        reason["value"]) and "SHEPHERD_DEPLOYMENT_MODE is not set" in reason["value"]
    assert reason["value"].startswith('<div class="shep-reason shep-reason-grey">')
