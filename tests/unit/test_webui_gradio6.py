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
def test_selecting_runtime_settings_refreshes_the_restart_lock(monkeypatch, tmp_path):
    import src.config.runtime_presets as presets
    import src.webui.components.runtime_settings as rs
    from src.config.runtime_presets import allocator_env
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
        "runtime_stop", "allocator_preset", "torch_compile", "runtime_facts"]

    from tests.fixtures.operator_access import operator_request

    # The backend started under the default preset; another one is saved, so a
    # restart would apply it (`restart_applies`).
    monkeypatch.setattr(presets, "RUNTIME_SETTINGS_FILE", tmp_path / "settings.json")
    for name, value in allocator_env({}, {}).items():
        monkeypatch.setenv(name, value)
    presets.save_runtime_settings({"allocator_preset": "expandable"})
    monkeypatch.setattr(rs, "is_training_active", lambda: True)
    apply, button, note, reason, stop, alloc, compile_, _facts = on_select[0].fn(
        operator_request(monkeypatch))
    assert apply["interactive"] is True and button["interactive"] is False
    # Stop service follows Restart's lock: off while a GPU job runs, no lock.
    assert (stop["value"], stop["interactive"]) == (rs.STOP_LABEL, False)
    assert note["visible"] is True and reason["visible"] is False
    assert rs.RESTART_BUSY in note["value"] and "shep-st-neutral" in note["value"]
    assert alloc["interactive"] is True and compile_["interactive"] is True
    # Off while training runs, but not refused: no lock (the WebUI skill's §3).
    assert (apply["value"], button["value"]) == (rs.APPLY_LABEL, rs.RESTART_LABEL)
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    apply, button, note, reason, stop, *_fields = on_select[0].fn(operator_request(monkeypatch))
    assert button["interactive"] is True and note["visible"] is False
    assert stop["interactive"] is True
    # R-12: with the running allocator saved, a restart would change nothing,
    # so Restart is off, said beside it; Stop service stays on.
    presets.save_runtime_settings({"allocator_preset": "cuda_async"})
    apply, button, note, reason, stop, *_fields = on_select[0].fn(operator_request(monkeypatch))
    assert (button["value"], button["interactive"]) == (rs.RESTART_LABEL, False)
    assert note["visible"] is True and rs.RESTART_IDLE in note["value"]
    assert stop["interactive"] is True and apply["interactive"] is True


def test_a_restart_question_left_open_restarts_nothing_once_restart_is_off(
        monkeypatch, tmp_path):
    """T06 review (R-12): "Restart now" checked D8 only, so a question asked
    while another allocator was saved still re-exec'd the backend after the
    running one was saved again, or while a GPU job had started: every page
    reconnected and a loaded model stopped for a restart the page called
    off. The confirmed restart makes the question's check again
    (`_restart_refusal`): nothing is asked of the backend, and the row closes
    with the reason the button shows."""
    import src.config.runtime_presets as presets
    import src.webui.components.runtime_settings as rs
    from src.config.runtime_presets import allocator_env
    from tests.fixtures.operator_access import operator_request

    monkeypatch.setattr(presets, "RUNTIME_SETTINGS_FILE", tmp_path / "settings.json")
    for name, value in allocator_env({}, {}).items():   # started under the default preset
        monkeypatch.setenv(name, value)
    presets.save_runtime_settings({"allocator_preset": "expandable"})
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    restarts = []
    monkeypatch.setattr(rs, "restart_backend", lambda: restarts.append(1) or {"success": True})
    row, _question, restart_now, _stop_now, _said = rs._on_restart_request(
        operator_request(monkeypatch))
    assert row["visible"] is True and restart_now["visible"] is True
    # The running allocator saved again while the question is open.
    presets.save_runtime_settings({"allocator_preset": "cuda_async"})
    row, _question, restart_now, _stop_now, said = rs._on_restart(operator_request(monkeypatch))
    assert restarts == [], "a restart that changes nothing reached the backend"
    assert row["visible"] is False and restart_now["visible"] is False
    assert rs.RESTART_IDLE in said and "shep-st-neutral" in said
    # A GPU job started while the question was open: the same, its reason.
    presets.save_runtime_settings({"allocator_preset": "expandable"})
    monkeypatch.setattr(rs, "is_training_active", lambda: True)
    said = rs._on_restart(operator_request(monkeypatch))[-1]
    assert restarts == [] and rs.RESTART_BUSY in said
    # Still wanted: Restart now restarts, once.
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    said = rs._on_restart(operator_request(monkeypatch))[-1]
    assert restarts == [1] and "Restarting the backend" in said


def test_an_open_restart_question_closes_once_restart_turns_off(monkeypatch, tmp_path):
    """R-12's residual (T06-A review): a Restart question left open stayed on
    screen after Restart turned off (the running allocator saved again with
    Apply, or by another tab and read on the slow timer; a GPU job started),
    until Restart now (which refused) or Cancel. Apply's last step and the
    slow timer now make the question's own check (`_restart_refusal`) and
    close it with the reason; a Stop question, a closed row and a Restart
    still wanted are left as they are."""
    import src.config.runtime_presets as presets
    import src.webui.components.runtime_settings as rs
    from src.config.runtime_presets import allocator_env
    from tests.fixtures.operator_access import operator_request

    monkeypatch.setattr(presets, "RUNTIME_SETTINGS_FILE", tmp_path / "settings.json")
    for name, value in allocator_env({}, {}).items():   # started under the default preset
        monkeypatch.setenv(name, value)
    presets.save_runtime_settings({"allocator_preset": "expandable"})
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    _row, restart_question, *_rest = rs._on_restart_request(operator_request(monkeypatch))
    _row, stop_question, *_rest = rs._on_stop_service_request(operator_request(monkeypatch))
    restart_question, stop_question = restart_question["value"], stop_question["value"]
    unchanged = [gr.update()] * 5
    # Still wanted: the open question stays.
    for fn in (rs._after_apply, lambda q, req: rs._poll_restart_lock("settings", q, req)):
        assert list(fn(restart_question, operator_request(monkeypatch))[5:]) == unchanged
    # The running allocator saved again, or a GPU job started: it closes.
    for save, busy, reason in (("cuda_async", False, rs.RESTART_IDLE),
                               ("expandable", True, rs.RESTART_BUSY)):
        presets.save_runtime_settings({"allocator_preset": save})
        monkeypatch.setattr(rs, "is_training_active", lambda busy=busy: busy)
        for fn in (rs._after_apply, lambda q, req: rs._poll_restart_lock("settings", q, req)):
            row, question, restart_now, stop_now, said = fn(
                restart_question, operator_request(monkeypatch))[5:]
            assert (row["visible"], restart_now["visible"], stop_now["visible"]) == (
                False, False, False)
            assert question["value"] == "" and reason in said
            # A Stop question and a closed row are left as they are.
            assert list(fn(stop_question, operator_request(monkeypatch))[5:]) == unchanged
            assert list(fn("", operator_request(monkeypatch))[5:]) == unchanged


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
    apply, button, note, reason, stop, alloc, compile_, _facts = rs._on_open(gradio_request())
    assert (apply["value"], apply["interactive"]) == (f"🔒 {rs.APPLY_LABEL}", False)
    assert (button["value"], button["interactive"]) == (f"🔒 {rs.RESTART_LABEL}", False)
    assert (stop["value"], stop["interactive"]) == (f"🔒 {rs.STOP_LABEL}", False)
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
    assert "Expandable segments (expandable)" in facts and ">On</dd>" in facts
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
    for said in (rs._on_restart(gradio_request())[-1],
                 rs._on_restart_request(gradio_request())[-1],
                 rs._on_apply("cuda_async", False, gradio_request())[0]):
        assert hostile not in said and "&lt;img src=x onerror=alert(1)&gt;" in said
        assert "shep-st-error" in said
    monkeypatch.undo()
    monkeypatch.setattr(rs, "restart_backend", lambda: {"success": False, "error": hostile})
    said = rs._on_restart(operator_request(monkeypatch))[-1]
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


def test_torch_compile_is_said_to_be_read_where_it_is_read(monkeypatch, tmp_path):
    """B5 review: the page said torch.compile was read "when a page starts a
    training run", but only the Training page reads it (its base Start and
    its Continue or Branch); Fine-tune starts runs and does not. Since T06
    (NEW-settings-datasets-footer-1, R-13) the field's visible line says it,
    once on the screen, after its warning: the facts, the help and Apply's
    answer no longer repeat it. These pins fail when another page starts
    reading it."""
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
    import src.config.runtime_presets as presets
    from tests.fixtures.operator_access import operator_request

    sentence = rs.COMPILE_READ[:1].upper() + rs.COMPILE_READ[1:]
    # R-13: the warning leads the visible line, then where it is read.
    assert rs.FIELD_INFO["torch_compile"] == f"{rs.COMPILE_EXPERIMENTAL} {sentence}."
    assert rs.COMPILE_EXPERIMENTAL.startswith("Experimental: keep it off unless you are "
                                              "benchmarking")
    monkeypatch.setattr(presets, "RUNTIME_SETTINGS_FILE", tmp_path / "settings.json")
    applied, _facts = rs._on_apply("cuda_async", True, operator_request(monkeypatch))
    # Each said once on the screen: the facts, the help, the accordion and
    # Apply's answer repeat neither.
    for text in (rs.FIELD_HELP["torch_compile"][1], rs.facts_html({}, {}), rs.COMPILE_DETAILS,
                 applied):
        assert "read when the training page" not in text.lower(), text
        assert "keep it off unless" not in text.lower(), text
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
    assert poll.inputs[1].elem_id == "runtime_question"
    assert [o.elem_id for o in poll.outputs] == [
        "runtime_apply", "runtime_restart", "runtime_restart_lock", "runtime_access_reason",
        "runtime_stop", "runtime_confirm", "runtime_question", "runtime_restart_now",
        "runtime_stop_now", "runtime_restart_status"]
    from tests.fixtures.operator_access import operator_request

    monkeypatch.setattr(rs, "is_training_active", lambda: pytest.fail("read on another page"))
    assert poll.fn("clinical", "", operator_request(monkeypatch)) == (gr.update(),) * 10
    monkeypatch.setattr(rs, "is_training_active", lambda: True)
    apply, button, note, _reason, stop, *asked = poll.fn("settings", "",
                                                          operator_request(monkeypatch))
    assert button["interactive"] is False and note["visible"] is True
    assert stop["interactive"] is False
    assert asked == [gr.update()] * 5, "no question was open: the row is left as it is"


#: Settings in a browser (R-12, R-13, NEW-settings-datasets-footer-1 and -2):
#: the screen's own words read; Restart Backend clicked, cancelled, clicked and
#: confirmed; then asked again and, with the question open, the running
#: allocator saved again by another hand (the settings file written: the slow
#: timer's read); saved as before and asked again, and with the question
#: open the running allocator saved again with Apply Settings. Each step waits
#: for what it asked to be drawn, never for a fixed time.
_SETTINGS_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [base, saved, mounted, queueJs] = [process.argv[2], process.argv[3], Number(process.argv[4]),
                                           process.argv[5]];
  const save = (preset) => fs.writeFileSync(saved, JSON.stringify({ allocator_preset: preset, torch_compile: false }));
  const browser = await chromium.launch();
  const out = {};
  const page = await browser.newPage({ viewport: { width: 1850, height: 960 } });
  await page.addInitScript(queueJs);
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  // Diagnosis' own select at mount writes the location first (the slow
  // timer's poll reads it): answered before Settings is opened.
  await page.waitForFunction((fn) => window.__shepQueue && window.__shepQueue.answered(fn) >= 1,
                             mounted, { timeout: 60000 });
  await page.click('#shep-tab-settings-button', { timeout: 60000 });
  await page.waitForSelector('#runtime_restart:not([disabled])', { timeout: 60000 });
  const shown = (sel) => page.evaluate((s) => {
    const el = document.querySelector(s);
    return !!el && el.getClientRects().length > 0;
  }, sel);
  const text = (sel) => page.evaluate((s) => document.querySelector(s).innerText, sel);
  const panel = await text('#shep-settings');
  out.compileReadTimes = (panel.match(/read when the training page starts/gi) || []).length;
  out.compileLine = await text('#torch_compile');
  out.variableNamed = /SHEPHERD_[A-Z_]+/.test(panel);
  out.note = await shown('#runtime_restart_note');
  // Restart asks first; Cancel closes the question and asks nothing.
  await page.click('#runtime_restart');
  await page.waitForSelector('#runtime_confirm', { state: 'visible' });
  out.question = await text('#runtime_question');
  out.confirmButtons = [await shown('#runtime_restart_now'), await shown('#runtime_stop_now')];
  await page.click('#runtime_confirm_cancel');
  await page.waitForSelector('#runtime_confirm', { state: 'hidden' });
  // Asked again and confirmed.
  await page.click('#runtime_restart');
  await page.waitForSelector('#runtime_restart_now', { state: 'visible' });
  await page.click('#runtime_restart_now');
  await page.waitForFunction(() => /Restarting the backend/.test(
    document.querySelector('#runtime_restart_status').innerText));
  out.afterConfirm = { status: await text('#runtime_restart_status'),
                       questionShown: await shown('#runtime_confirm') };
  // Asked once more (the status line clears), and with the question open the
  // running allocator saved again by another hand: on the slow timer's read
  // Restart is off, and the open question closes with the reason (R-12's
  // residual: it stayed open until Restart now or Cancel).
  const asked = async () => {
    await page.click('#runtime_restart');
    await page.waitForSelector('#runtime_restart_now', { state: 'visible' });
    await page.waitForFunction(() => document.querySelector('#runtime_restart_status').innerText.trim() === '');
  };
  await asked();
  save('cuda_async');
  await page.waitForSelector('#runtime_confirm', { state: 'hidden', timeout: 30000 });
  out.timer = { status: await text('#runtime_restart_status'),
                restartDisabled: await page.evaluate(() => document.querySelector('#runtime_restart').disabled) };
  // Saved as before: Restart is on again; asked, and with the question open
  // the running allocator saved again with Apply Settings: Restart is off and
  // says why beside it, and the question closes with the reason.
  save('expandable');
  await page.waitForSelector('#runtime_restart:not([disabled])', { timeout: 30000 });
  await asked();
  await page.click('#allocator_preset input');
  await page.click('ul[role=listbox] li:has-text("Driver memory pool")');
  await page.click('#runtime_apply');
  await page.waitForSelector('#runtime_restart[disabled]', { timeout: 30000 });
  await page.waitForSelector('#runtime_confirm', { state: 'hidden', timeout: 30000 });
  out.idle = { reason: await text('#runtime_restart_lock'),
               stopEnabled: await page.evaluate(() => !document.querySelector('#runtime_stop').disabled) };
  out.stale = { status: await text('#runtime_restart_status'),
                questionShown: await shown('#runtime_confirm') };
  await browser.close();
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_settings_asks_before_a_restart_and_says_each_fact_once_in_a_browser(
        monkeypatch, tmp_path):
    """R-12: on ba6e85e one click of Restart Backend restarted at once, with
    nothing waiting for it. Now it asks in the panel's question row (every
    open page reconnects; a loaded model is stopped first) with Restart now
    and Cancel, nothing is asked of the backend before Restart now, and with
    the running allocator saved Restart is off, the reason beside it; a
    question left open then closes with the reason, whether Apply on the page
    or another hand (the slow timer's read) saved it (R-12's residual: it
    stayed open until Restart now, which refuses, or Cancel). R-13 and
    NEW-settings-datasets-footer-1: torch.compile's visible line leads with
    the warning and where it is read is said once on the screen.
    NEW-settings-datasets-footer-2: no standing restart note, no internal
    variable in the panel's text."""
    import src.config.runtime_presets as presets
    import src.webui.components.runtime_settings as rs
    from src.config.runtime_presets import allocator_env
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import node_with_playwright, run_node
    from tests.fixtures.operator_access import operator_request
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_request(monkeypatch)
    monkeypatch.setattr(presets, "RUNTIME_SETTINGS_FILE", tmp_path / "settings.json")
    for name, value in allocator_env({}, {}).items():   # started under the default preset
        monkeypatch.setenv(name, value)
    presets.save_runtime_settings({"allocator_preset": "expandable", "torch_compile": False})
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    restarts = []
    monkeypatch.setattr(rs, "restart_backend", lambda: restarts.append(1) or {"success": True})
    from tests.fixtures.browser import QUEUE_JS

    demo = create_gradio_app()
    mounted = next(fn._id for fn in demo.fns.values() if fn.name == "_shown_clinical")
    with served(demo, **mount_options()) as base:
        seen = run_node(_SETTINGS_JS, tmp_path, base[:-len("gradio_api")],
                        str(tmp_path / "settings.json"), str(mounted), QUEUE_JS)

    assert seen["compileReadTimes"] == 1, seen
    assert seen["compileLine"].splitlines()[-1].startswith(rs.COMPILE_EXPERIMENTAL), seen
    assert seen["variableNamed"] is False and seen["note"] is False, seen
    assert seen["question"].startswith("Restart the backend? Every open page reconnects"), seen
    assert seen["confirmButtons"] == [True, False], seen
    assert seen["afterConfirm"]["questionShown"] is False, seen
    assert "Restarting the backend" in seen["afterConfirm"]["status"]
    assert rs.RESTART_IDLE in seen["idle"]["reason"] and seen["idle"]["stopEnabled"], seen
    # R-12's residual: the question left open while Restart turned off
    # closes with the reason the button shows, on the slow timer's read and
    # after Apply; nothing more is asked of the backend.
    assert rs.RESTART_IDLE in seen["timer"]["status"] and seen["timer"]["restartDisabled"], seen
    assert rs.RESTART_IDLE in seen["stale"]["status"], seen
    assert seen["stale"]["questionShown"] is False, seen
    assert restarts == [1], "the backend was asked other than once, by Restart now"
