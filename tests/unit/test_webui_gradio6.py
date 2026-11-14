"""
The dashboard's Gradio 6.30 adjustments that a browser would otherwise show.
============================================================================
- **Timers in an unopened tab do not tick** in Gradio 6 (gradio#11984,
  #11993): Runtime Settings refreshes its restart lock when its tab is
  selected, and then on the shell's slow timer (outside the tabs) while it is
  shown; it has no timer of its own.

(gr.HTML's padding, off by default in Gradio 6, was asked for by the Settings
badges and restart note laid out with Gradio 5's: both went with the one
Settings panel of the ops round's B5, which draws its HTML itself.)

(The Training Console's native-plot workarounds, the typed empty frames, the
legend step after each data step and the Epoch Window's row filter, went with
the LinePlots in M5-2: Training draws its charts with the shared uPlot
component, `tests/unit/test_webui_charts.py` and `test_training_page.py`.)

Module: tests/unit/test_webui_gradio6.py
"""
import pytest

gr = pytest.importorskip("gradio")


# --------------------------------------------------------------- Runtime Settings
def test_selecting_runtime_settings_refreshes_the_restart_lock(monkeypatch):
    import src.webui.components.runtime_settings as rs
    from src.webui.app import create_gradio_app

    app = create_gradio_app()
    # Runtime Settings is the page under ⚙ Settings (M5-2).
    tab = next(b for b in app.blocks.values()
               if isinstance(b, gr.Tab) and b.elem_id == "shep-tab-settings")
    on_select = [fn for fn in app.fns.values() if (tab._id, "select") in fn.targets
                 and fn.name == "_on_open"]
    assert len(on_select) == 1
    assert [o.elem_id for o in on_select[0].outputs] == [
        "runtime_apply", "runtime_restart", "runtime_restart_lock", "runtime_access_reason",
        "allocator_preset", "torch_compile", "runtime_facts"]

    from tests.fixtures.operator_access import operator_request

    monkeypatch.setattr(rs, "is_training_active", lambda: True)
    apply, button, note, reason, alloc, compile_, _facts = on_select[0].fn(
        operator_request(monkeypatch))
    assert apply["interactive"] is True and button["interactive"] is False
    assert note["visible"] is True and reason["visible"] is False
    assert rs.RESTART_BUSY in note["value"] and "shep-st-neutral" in note["value"]
    assert alloc["interactive"] is True and compile_["interactive"] is True
    # Off while training runs, but not refused: no lock (the WebUI skill's §3).
    assert (apply["value"], button["value"]) == (rs.APPLY_LABEL, rs.RESTART_LABEL)
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    apply, button, note, reason, *_fields = on_select[0].fn(operator_request(monkeypatch))
    assert button["interactive"] is True and note["visible"] is False


def test_read_only_settings_lock_apply_and_restart_with_the_reason(monkeypatch, tmp_path):
    """The one gated-button rule (`chips.gated`, the WebUI skill's §3): a
    control a read-only request cannot use is disabled with the lock and the
    visible "Operator controls only: …" reason beside it, in D8's short words
    (each handler still checks its own request)."""
    import src.config.runtime_presets as presets
    import src.webui.components.runtime_settings as rs
    from tests.fixtures.operator_access import gradio_request

    monkeypatch.setattr(presets, "RUNTIME_SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    apply, button, note, reason, alloc, compile_, _facts = rs._on_open(gradio_request())
    assert (apply["value"], apply["interactive"]) == (f"🔒 {rs.APPLY_LABEL}", False)
    assert (button["value"], button["interactive"]) == (f"🔒 {rs.RESTART_LABEL}", False)
    assert reason["visible"] is True and reason["value"] == (
        '<div class="shep-reason"><span aria-hidden="true">🔒</span><span>Operator controls '
        "only: deployment mode not set to M1.</span></div>")
    # The fields that need the operator are not interactive for this request,
    # and show the saved values (none saved: the default preset, compile off).
    assert alloc == gr.update(interactive=False, value="cuda_async")
    assert compile_ == gr.update(interactive=False, value=False)


def test_the_fields_show_what_is_saved_each_time_the_page_is_opened(monkeypatch, tmp_path):
    """B5 review: the fields kept the values read when the app was built, so
    after a save the facts said one saved allocator and the field another,
    and Apply wrote the stale one back. The page's open reads the saved file
    once, for the fields and the facts alike."""
    import src.config.runtime_presets as presets
    import src.webui.components.runtime_settings as rs
    from src.webui.app import create_gradio_app
    from tests.fixtures.operator_access import operator_request

    monkeypatch.setattr(presets, "RUNTIME_SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    app = create_gradio_app()
    built = {b.elem_id: b.value for b in app.blocks.values()
             if getattr(b, "elem_id", None) in ("allocator_preset", "torch_compile")}
    assert built == {"allocator_preset": "cuda_async", "torch_compile": False}
    presets.save_runtime_settings({"allocator_preset": "expandable", "torch_compile": True})
    *_lock, alloc, compile_, facts = rs._on_open(operator_request(monkeypatch))
    assert alloc == gr.update(interactive=True, value="expandable")
    assert compile_ == gr.update(interactive=True, value=True)
    assert "Expandable segments (expandable)" in facts and ">On <span" in facts
    # The fields are the saved values again, so nothing reads "Not saved".
    assert rs._on_change(alloc["value"], compile_["value"]) == ""


def test_a_read_only_request_gets_still_fields_and_escaped_refusals(monkeypatch, tmp_path):
    """The ops round's B5: a request D8 refuses gets the allocator and
    torch.compile as non-interactive fields from the page's open, and every
    text the page interpolates (D8's refusal, the restart's error, an explicit
    allocator in the launch environment) is escaped: markup in it reads as
    literal text."""
    import src.webui.components.runtime_settings as rs
    from src.config.runtime_presets import ALLOC_SOURCE_ENV
    from src.webui.app import create_gradio_app
    from tests.fixtures.operator_access import gradio_request, operator_request

    hostile = "<img src=x onerror=alert(1)>"
    app = create_gradio_app()
    fields = {b.elem_id: b for b in app.blocks.values()
              if getattr(b, "elem_id", None) in ("allocator_preset", "torch_compile")}
    assert all(field.interactive is False for field in fields.values())
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)

    class Refused:
        allowed, condition = False, "mode_unset"
        refusal = f"Refused: {hostile}"

    monkeypatch.setattr(rs, "operator_access", lambda request: Refused())
    for said in (rs._on_restart(gradio_request())["value"],
                 rs._on_apply("cuda_async", False, gradio_request())[0]):
        assert hostile not in said and "&lt;img src=x onerror=alert(1)&gt;" in said
        assert "shep-st-error" in said
    monkeypatch.undo()
    monkeypatch.setattr(rs, "restart_backend", lambda: {"success": False, "error": hostile})
    said = rs._on_restart(operator_request(monkeypatch))["value"]
    assert hostile not in said and "Not restarted: &lt;img" in said
    facts = rs.facts_html({"PYTORCH_ALLOC_CONF": hostile, ALLOC_SOURCE_ENV: "env"},
                          {"allocator_preset": hostile})
    assert hostile not in facts and "PYTORCH_ALLOC_CONF=&lt;img" in facts
    assert "overrides it until the backend starts without it" in facts


def test_settings_say_the_running_allocator_apart_from_the_saved_one():
    """What runs is read back from the backend's own environment by the rule
    that wrote it (`running_allocator`), never taken from the saved file: a
    preset the launcher applied is not an explicit override, and a saved
    preset the backend does not run yet says what applies it."""
    import src.webui.components.runtime_settings as rs
    from src.config.runtime_presets import ALLOC_SOURCE_ENV, ALLOCATOR_PRESETS, allocator_env

    assert set(rs.ALLOCATOR_WORDS) == set(ALLOCATOR_PRESETS)
    started = allocator_env({}, {"allocator_preset": "cuda_async"})
    facts = rs.facts_html(started, {"allocator_preset": "expandable"})
    assert "Driver memory pool (cuda_async) <span class=\"shep-muted\">(the saved preset" in facts
    assert "Expandable segments (expandable)" in facts
    assert "Not running yet: Restart Backend applies it" in facts
    assert "launch environment" not in facts
    same = rs.facts_html(started, {"allocator_preset": "cuda_async"})
    assert "Not running yet" not in same
    unset = rs.facts_html({}, {})
    assert "PyTorch’s default (native)" in unset and "(the default; none saved)" in unset
    assert "Not implemented yet" in unset and "<input" not in unset
    explicit = rs.facts_html({"PYTORCH_ALLOC_CONF": "backend:native", ALLOC_SOURCE_ENV: "env"}, {})
    assert "set in the launch environment" in explicit
    assert "<code>PYTORCH_ALLOC_CONF=backend:native</code>" in explicit
    # B5 review: an empty allocator variable is a setting a restart keeps, so
    # it is named as one, never "no allocator was set" with a restart to apply
    # the preset.
    empty = rs.facts_html({"PYTORCH_ALLOC_CONF": ""}, {"allocator_preset": "expandable"})
    assert "<code>PYTORCH_ALLOC_CONF=</code>" in empty and "no allocator was set" not in empty
    assert "overrides it until the backend starts without it" in empty
    assert "Not running yet" not in empty
    # The acceleration row: its state, then why, in brackets as the other rows
    # say theirs (B5 review: the two ran together as one sentence).
    assert ('<span class="shep-st-w">Not implemented yet</span></span> <span class="shep-muted">'
            f"({rs.ACCELERATION})</span>") in unset


def test_no_settings_text_claims_an_architecture_that_is_not_built():
    """Only GATv2 is built (`builder._ENCODERS`): HGT is named only as the
    earlier architecture a measurement was made on, never as what runs."""
    import src.webui.components.runtime_settings as rs

    texts = [text for _title, text in rs.FIELD_HELP.values()] + [rs.ALLOCATOR_DETAILS,
                                                                  rs.COMPILE_DETAILS]
    for text in texts:
        for sentence in text.replace("\n", " ").split(". "):
            if "HGT" in sentence:
                assert "earlier" in sentence or "no longer built" in sentence, sentence
    assert "GATv2, the one architecture built today" in rs.FIELD_HELP["torch_compile"][1]


def test_torch_compile_is_said_to_be_read_where_it_is_read():
    """B5 review: the page said torch.compile was read "when a page starts a
    training run", but only the Training page reads it (its base Start and
    its Continue or Branch); Fine-tune starts runs and does not. The facts,
    the field's info line, its help and Apply's answer say `COMPILE_READ`,
    and these pins fail when another page starts reading it."""
    import re
    from pathlib import Path

    import src.webui.components.runtime_settings as rs

    src = Path(rs.__file__).resolve().parents[3] / "src"
    readers = sorted({path.relative_to(src).as_posix() for path in src.rglob("*.py")
                      if re.search(r"""get\(\s*["']torch_compile["']""",
                                   path.read_text(encoding="utf-8"))})
    # Settings reads it to show it; the Training page is the one that runs it.
    assert readers == ["webui/components/runtime_settings.py",
                       "webui/components/training_console.py"]
    from src.webui.components import finetune_page

    assert "torch_compile" not in Path(finetune_page.__file__).read_text(encoding="utf-8")
    sentence = rs.COMPILE_READ[:1].upper() + rs.COMPILE_READ[1:]
    assert rs.FIELD_INFO["torch_compile"].startswith(sentence)
    assert rs.COMPILE_READ in rs.FIELD_HELP["torch_compile"][1]
    assert f"({rs.COMPILE_READ})" in rs.facts_html({}, {})
    for text in (rs.FIELD_INFO["torch_compile"], rs.FIELD_HELP["torch_compile"][1],
                 rs.facts_html({}, {})):
        assert "when a page starts" not in text


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
