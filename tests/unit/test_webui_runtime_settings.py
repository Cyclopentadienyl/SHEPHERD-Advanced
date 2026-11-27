"""
⚙ Settings as a read-only viewer sees it (the owner's D6 of 2026-11-24)
=======================================================================
D6: the full lock reason is said once, in one place (the header's D8 chip,
which each "Operator controls only" phrase opens); beside each locked action
its short phrase; never words that point at a reason drawn elsewhere ("see
above", "see below", "the reason is with": it may have scrolled away or be in
another region). Settings said it three ways: Data locations' notice
repeated who can act ("Folder paths are shown to operator controls only"),
and Runtime settings drew "Read-only for this request: the reason is with the
actions below." between its facts and its fields.

And ⚙ Settings › Display (the owner's revision of D9, 2026-11-25): this
browser's choice of how Training's Parameters start, stacked, read before the
page is painted, kept in the browser alone and offered to every viewer.

Module: tests/unit/test_webui_runtime_settings.py
"""
from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("gradio")

from src.utils.version_checker import NOTICE  # noqa: E402
from tests.fixtures.operator_access import gradio_request  # noqa: E402

#: Words that point at a reason drawn elsewhere (R-WHY; the walk's ``pointer-words``).
POINTER = re.compile(r"\bsee (above|below)\b|\bthe reason is with\b", re.I)


def _html(outputs) -> str:
    """Every value a handler draws, its HTML as drawn."""
    return " ".join(str(o.get("value", "")) for o in outputs if isinstance(o, dict))


def _text(outputs) -> str:
    """Every value a handler draws, as its words."""
    return re.sub(r"<[^>]+>", " ", _html(outputs))


def test_read_only_settings_say_each_lock_where_it_is_and_point_nowhere(monkeypatch, tmp_path):
    """For a request D8 refuses, what Settings draws (both panels, as the page's open draws
    them) holds no words that point at a reason drawn elsewhere; Runtime settings' one phrase,
    beside Apply, Restart and Stop, also names the two fields the request dims, so each is
    described by it (R-WHY's ``data-for``); Data locations' notice says its consequence (the
    paths are hidden), not who can act, which the chip and the phrases say. Fails on 9adb3c4:
    Runtime's "the reason is with the actions below" line, and the notice's "shown to operator
    controls only"."""
    import src.config.runtime_presets as presets
    import src.webui.components.data_locations as dl
    import src.webui.components.runtime_settings as rs

    monkeypatch.setattr(presets, "RUNTIME_SETTINGS_FILE", tmp_path / "runtime.json")
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    refused = gradio_request()
    runtime, locations = rs._on_open(refused), dl._on_locations_open(refused)
    for name, outputs in (("runtime", runtime), ("data locations", locations)):
        found = POINTER.search(_text(outputs))
        assert found is None, (name, found and found.group(0))
    phrases = re.findall(r'<button[^>]*class="shep-why"[^>]*data-for="([^"]+)"', _html(runtime))
    assert len(phrases) == 1, phrases
    assert {"runtime_apply", "runtime_restart", "runtime_stop", "allocator_preset",
            "torch_compile"} <= set(phrases[0].split()), phrases
    notice = _text([locations[len(dl.KEYS) * len(dl.ROW_PARTS)]])
    assert "hidden" in notice and "operator controls" not in notice.lower(), notice


#: The display choices' head script (`display.DISPLAY_HEAD`) on a page of its own (``argv[2]``:
#: the head): for each case (``argv[3]``: ``[name, saved, blocked]``, a value saved before the
#: page loads or none, and whether the browser's storage throws on every read and write), the
#: root element's mark as a script right after the head script reads it (before the body
#: exists), the choice the page reads, then a pick of Folded (what it returns, what the page
#: hears, what the browser keeps, the mark) and a pick of an unknown value.
_DISPLAY_HEAD_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const [KEY, ATTR, EVENT] = [__KEY__, __ATTR__, __EVENT__];
  const html = '<!doctype html><html><head>' + JSON.parse(process.argv[2])
    + '<script>window.__beforeBody = [document.documentElement.getAttribute(' + JSON.stringify(ATTR)
    + '), !!document.body];</script></head><body><p>page</p></body></html>';
  const out = {};
  for (const [name, saved, blocked] of JSON.parse(process.argv[3])) {
    const page = await (await browser.newContext()).newPage();
    await page.route('http://display.test/**', (r) => r.fulfill({ status: 200, contentType: 'text/html', body: html }));
    if (saved !== null) {
      await page.addInitScript(([k, v]) => { window.localStorage.setItem(k, v); }, [KEY, saved]);
    }
    if (blocked) {
      await page.addInitScript(() => {
        Storage.prototype.getItem = () => { throw new Error('blocked'); };
        Storage.prototype.setItem = () => { throw new Error('blocked'); };
      });
    }
    await page.goto('http://display.test/');
    out[name] = await page.evaluate(([k, a, ev]) => {
      const kept = () => { try { return window.localStorage.getItem(k); } catch (e) { return 'unreadable'; } };
      const [first, early] = window.__beforeBody, read = window.ShepDisplay.params();
      let heard = null;
      document.addEventListener(ev, (e) => { heard = e.detail; });
      const picked = window.ShepDisplay.setParams('folded');
      const after = { picked, heard, kept: kept(), mark: document.documentElement.getAttribute(a) };
      const odd = window.ShepDisplay.setParams('sideways');
      return { first, early, read, after, odd, oddMark: document.documentElement.getAttribute(a), oddKept: kept() };
    }, [KEY, ATTR, EVENT]);
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_display_choice_is_read_before_the_page_is_drawn_and_a_missing_unreadable_or_unknown_one_is_shown(
        tmp_path):
    """The owner's revision of D9 (2026-11-25): how Training's Parameters start, stacked, is
    each reader's choice, kept in their browser (``localStorage``, every access wrapped): a
    missing, unreadable or unknown value means Shown. The head script (`display.DISPLAY_HEAD`,
    in the mount's head) writes the choice on the root element before the body exists, so
    the first frame any page paints is already in it: Shown, Folded and Automatic read as
    saved; nothing saved, a value no choice has, and storage that throws read Shown. A pick
    is kept in the browser, written on the root element and announced to the open page
    (`display.PARAMS_EVENT`), and holds for the page session where storage throws; an
    unknown pick is Shown. Fails at bb258d1: no such script."""
    from src.webui.components import display
    from tests.fixtures.browser import node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    cases = [["missing", None, False], ["shown", "shown", False], ["folded", "folded", False],
             ["auto", "auto", False], ["unknown", "sideways", False], ["cased", "Folded", False],
             ["unreadable", "folded", True]]
    script = (_DISPLAY_HEAD_JS.replace("__KEY__", json.dumps(display.PARAMS_KEY))
              .replace("__ATTR__", json.dumps(display.PARAMS_ATTR))
              .replace("__EVENT__", json.dumps(display.PARAMS_EVENT)))
    seen = run_node(script, tmp_path, json.dumps(display.DISPLAY_HEAD), json.dumps(cases))
    read = {"missing": "shown", "shown": "shown", "folded": "folded", "auto": "auto",
            "unknown": "shown", "cased": "shown", "unreadable": "shown"}
    assert display.PARAMS_DEFAULT == display.SHOWN and set(display.PARAMS_MODES) == {"shown", "folded", "auto"}
    for name, want in read.items():
        got = seen[name]
        assert got["first"] == got["read"] == want and got["early"] is False, (name, got)
        blocked = name == "unreadable"
        assert got["after"] == {"picked": "folded", "heard": "folded", "mark": "folded",
                                "kept": "unreadable" if blocked else "folded"}, (name, got)
        assert got["odd"] == got["oddMark"] == "shown", (name, got)
        assert got["oddKept"] == ("unreadable" if blocked else "shown"), (name, got)


def test_the_display_choice_never_reaches_the_server():
    """⚙ Settings › Display is the browser's alone (the owner's revision of D9, 2026-11-25): no
    server setting, no settings-file key, no API, no operator lock. Its one field's events run
    in the browser only (no function on the server): the page load shows the choice the head
    script read, and the reader's pick saves it; no handler outputs to it (nothing dims it for
    a read-only viewer); its words follow the skill (one field, a one-line note under it
    saying where it applies, three short options). The review of 36c0592 (P3): the note in its
    head, two short facts (it was two sentences, 23 words, five lines in the 240 px panel at
    1850 x 960, and "this page's layout" was unclear on Settings); a label that names the
    panel it sets ("Training parameters" read as training hyperparameters); the line under it
    and the bubble's lead front-loaded, the bubble within the budget."""
    import gradio as gr

    import src.webui.components.runtime_settings as rs
    from src.webui.app import create_gradio_app
    from src.webui.components import display
    from src.webui.components.help import TEXT_BUDGET, budget_problems

    app = create_gradio_app()
    radio = next(b for b in app.blocks.values() if getattr(b, "elem_id", None) == rs.DISPLAY_PARAMS)
    assert isinstance(radio, gr.Radio) and radio.interactive is not False
    assert radio.label == rs.DISPLAY_PARAMS_LABEL == "Training's Parameters"
    assert radio.info == rs.FIELD_INFO[rs.DISPLAY_PARAMS] == (
        "Stacked windows only (portrait or narrow). Side by side it is always shown.")
    title, text = rs.FIELD_HELP[rs.DISPLAY_PARAMS]
    assert title == rs.DISPLAY_PARAMS_LABEL and text.startswith("Parameters panel, stacked: ")
    assert budget_problems(text, TEXT_BUDGET) == () and text.count("\n- ") == 4
    assert [label for label, _value in display.PARAMS_CHOICES] == ["Shown", "Folded", "Automatic"]
    assert radio.value == display.PARAMS_DEFAULT
    mine = [fn for fn in app.fns.values()
            if any(t[0] == radio._id for t in fn.targets) or radio in fn.outputs]
    assert sorted((fn.targets[0][1], fn.fn is None, fn.js) for fn in mine) == [
        ("input", True, display.WRITE_PARAMS_JS), ("load", True, display.READ_PARAMS_JS)], mine
    panel = next(b for b in app.blocks.values() if getattr(b, "elem_id", None) == "shep-display-hd")
    assert rs.DISPLAY_NOTE == ("this browser only", "any viewer can change it")
    assert "<h2>Display</h2>" in panel.value and all(part in panel.value for part in rs.DISPLAY_NOTE)
    assert panel.value.count('class="shep-sep"') == 1 and "shep-intro" not in panel.value


#: ⚙ Settings at the owner's window (1850 x 960), the Parameters choice ``argv[3]`` saved in
#: the browser: from the click on Settings, at each painted frame of its Display panel, the
#: option its radio shows checked; once the page is answered and still, whether every option
#: can be used, the page's own scroll and each panel body's (its content's height against its
#: box's), then a pick of ``argv[4]`` and what the browser keeps and the root element's mark.
_SETTINGS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 })).newPage();
  await page.addInitScript(QUEUE_JS);
  await page.addInitScript(([k, v]) => { try { window.localStorage.setItem(k, v); } catch (e) {} }, [__KEY__, process.argv[3]]);
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#shep-tab-settings-button', { timeout: 60000 });
  await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 60000 });
  await page.evaluate(() => {
    const seen = window.__checked = [], after = new MessageChannel();
    after.port1.onmessage = () => {
      const radio = document.getElementById('shep-display-params');
      if (!radio || !radio.getClientRects().length) { return; }
      const on = radio.querySelector('input[type=radio]:checked'), now = on ? on.value : 'none';
      if (seen[seen.length - 1] !== now) { seen.push(now); }
    };
    const tick = () => { after.port2.postMessage(0); if (!window.__done) { requestAnimationFrame(tick); } };
    requestAnimationFrame(tick);
  });
  await page.$eval('#shep-tab-settings-button', (el) => el.click());
  await page.waitForSelector('#shep-display-params label', { state: 'visible', timeout: 60000 });
  await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 60000 });
  await page.waitForTimeout(1500);
  const read = await page.evaluate(() => {
    window.__done = true;
    const body = (id) => { const e = document.getElementById(id); return [e.scrollHeight, e.clientHeight]; };
    const se = document.scrollingElement;
    return { checked: window.__checked,
             usable: [...document.querySelectorAll('#shep-display-params input[type=radio]')].map((i) => !i.disabled),
             page: [se.scrollHeight, se.clientHeight, se.scrollWidth, se.clientWidth],
             bodies: { dl: body('shep-dl-bd'), runtime: body('shep-settings-bd'), display: body('shep-display-bd') },
             layout: document.documentElement.getAttribute('data-shep-layout') };
  });
  await page.click(`#shep-display-params label:has(input[value="${process.argv[4]}"])`);
  await page.waitForTimeout(300);
  const picked = await page.evaluate(([k, a]) => ({ kept: window.localStorage.getItem(k),
                                                     mark: document.documentElement.getAttribute(a) }),
                                     [__KEY__, __ATTR__]);
  console.log(JSON.stringify({ ...read, picked }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.mark.parametrize("who,state", [("operator", "first_use"), ("operator", "configured"),
                                       ("read-only", "configured")])
def test_settings_offers_the_display_choice_to_every_viewer_within_the_owners_window(
        who, state, monkeypatch, tmp_path):
    """⚙ Settings › Display (the owner's revision of D9, 2026-11-25): one choice of how
    Training's Parameters start, stacked, in a small panel of its own ("this browser"),
    offered to every viewer: it changes only the viewer's own page, so a read-only viewer may
    change it too. At the owner's window (1850 x 960), for an operator at first use and
    configured and for a read-only viewer: from the first painted frame of Settings the radio
    shows the choice saved in the browser (Automatic here), never the build's default first;
    every option can be used; the three panels sit side by side and nothing scrolls, the page
    nor any panel's body (the skill's one screen); a pick (Folded) is kept in the browser and
    marked on the page at once. Fails at bb258d1: no Display panel."""
    from src.webui.app import create_gradio_app, mount_options
    from src.webui.components import display
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.operator_access import operator_mode
    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    if who == "operator":
        operator_mode(monkeypatch)
    else:
        monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)
    if state == "configured":
        write_demo_workspace(tmp_path / "root" / "demo")
        monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(tmp_path / "root"))
    script = (_SETTINGS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__KEY__", json.dumps(display.PARAMS_KEY))
              .replace("__ATTR__", json.dumps(display.PARAMS_ATTR)))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")], display.AUTO, display.FOLDED)
    assert seen["checked"] == [display.AUTO], seen
    assert seen["usable"] == [True, True, True], seen
    assert seen["layout"] == "side", seen
    height, view, width, wide = seen["page"]
    assert height <= view and width <= wide, seen
    for name, (content, box) in seen["bodies"].items():
        assert content <= box, (name, seen)
    assert seen["picked"] == {"kept": display.FOLDED, "mark": display.FOLDED}, seen


#: ⚙ Settings for an operator at each window of ``argv[3]`` (``[[w, h], ...]``), its page
#: answered and still: the arrangement, the page's own scroll, whether Display stands under
#: Runtime settings, whether Runtime settings' body has more under its edge at its top, Display's
#: height against its own least, and the bottom of Display's options against its body's view
#: (its top and its height in view). Then what Runtime settings holds, each brought into its
#: body's view as a reader scrolls to it (`scrollIntoView`; the body is the panel's one scroll
#: region): its facts' top and then their foot each in the body's view, and each field's
#: control (Memory allocator's box, Compile's box and its words) in the middle of the body's
#: view, whole in it and in the window, the control itself under its middle (nothing over it),
#: not disabled; and the page's own scroll after each, never moved.
_UNDER_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of JSON.parse(process.argv[3])) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1 })).newPage();
    await page.addInitScript(QUEUE_JS);
    await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('#shep-tab-settings-button', { timeout: 60000 });
    await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 60000 });
    await page.$eval('#shep-tab-settings-button', (el) => el.click());
    await page.waitForSelector('#shep-display-params label', { state: 'visible', timeout: 60000 });
    await page.waitForFunction(() => window.__shepQueue.idle() && document.fonts.status === 'loaded', null, { timeout: 60000 });
    await page.waitForTimeout(1500);
    out[`${w}x${h}`] = await page.evaluate(() => {
      // A field's control: its box (Gradio's own wrap, not the field's status tracker, which covers it whole).
      const CONTROL = (field) => `${field} .wrap:not([data-testid="status-tracker"])`;
      const box = (el) => el.getBoundingClientRect();
      const seen = (bodyId, el) => {
        const b = document.getElementById(bodyId), r = box(b), e = box(el);
        return { bottom: Math.round(e.bottom - r.top), view: b.clientHeight };
      };
      const se = document.scrollingElement, body = document.getElementById('shep-settings-bd');
      const display = document.getElementById('shep-display');
      const runtime = box(document.getElementById('shep-settings')), shown = box(display);
      const first = { layout: document.documentElement.getAttribute('data-shep-layout'),
                      page: [se.scrollHeight, se.clientHeight, se.scrollWidth, se.clientWidth],
                      under: shown.top >= runtime.bottom && Math.abs(shown.left - runtime.left) < 1,
                      more: body.scrollHeight - body.clientHeight > 1,
                      display: [shown.height, parseFloat(getComputedStyle(display).minHeight) || 0],
                      options: seen('shep-display-bd', document.querySelector(CONTROL('#shep-display-params'))) };
      // The body's view, and whether `e` lies in it (within half a pixel).
      const inView = (e) => { const r = box(body), top = r.top + body.clientTop, bottom = top + body.clientHeight;
        return { top: e.top >= top - 0.5 && e.top < bottom, bottom: e.bottom <= bottom + 0.5 && e.bottom > top,
                 whole: e.top >= top - 0.5 && e.bottom <= bottom + 0.5 }; };
      const still = () => se.scrollTop === 0 && se.scrollLeft === 0;
      const facts = document.getElementById('runtime_facts');
      facts.scrollIntoView({ block: 'start', inline: 'nearest' });
      const head = inView(box(facts)).top && still();
      facts.scrollIntoView({ block: 'end', inline: 'nearest' });
      const foot = inView(box(facts)).bottom && still();
      first.facts = { head, foot };
      const reach = (at, input) => {
        const control = document.querySelector(at), own = document.querySelector(input);
        control.scrollIntoView({ block: 'center', inline: 'nearest' });
        const e = box(control), hit = document.elementFromPoint((e.left + e.right) / 2, (e.top + e.bottom) / 2);
        return { whole: inView(e).whole && e.top >= 0 && e.bottom <= innerHeight,
                 hit: !!hit && control.contains(hit), enabled: !!own && !own.disabled, still: still() };
      };
      first.allocator = reach(CONTROL('#allocator_preset'), '#allocator_preset input');
      first.compile = reach('#torch_compile label:has(input[type=checkbox])', '#torch_compile input[type=checkbox]');
      body.scrollTop = 0;
      return first;
    });
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


#: A runtime report of a CPU-only host whose footer warning takes two lines at these windows (the
#: external review of ffb1483's host: no CUDA device, cuVS without cupy), in the probe's own
#: words, so the room the footer leaves is the same on every host (`app._runtime_footer`).
_TWO_LINE_RUNTIME = {
    "status": NOTICE, "python": "3.12", "platform": "Linux aarch64", "pyg_native": {}, "retrieval": {},
    "torch": {"available": True, "version": "2.10.0", "cuda_build": "13.0", "cuda_available": False},
    "torch_geometric": {"available": True, "version": "2.8.0"},
    "issues": ["No CUDA device visible: GPU training is unavailable. torch is built for CUDA 13.0. Check the "
               "driver and CUDA libraries.",
               "The GPU vector backend cannot be constructed: cuVS is installed but cupy is missing (Voyager "
               "on the CPU is unaffected)."]}


@pytest.mark.parametrize("footer", ["this host's", "two lines"])
def test_settings_display_gives_way_first_under_runtime_settings(footer, monkeypatch, tmp_path):
    """The review of 36c0592 (P2): side by side in a window under about 1740 px Display stands
    under Runtime settings, and drawn whole (240 to 259 px) it left Runtime settings' body 156
    of its 588 px at the display floor (1280 x 720), neither of its fields in view, where
    bb258d1 had shown 431 px with Memory allocator; 223 px at 1366 x 768. A choice kept in
    the browser, with no Apply, pushed out the operator's server settings. Display now gives
    way first, down to its head and its choice (the label and the three options), and
    Runtime settings' body takes the rest. For an operator with a workspace, side by side, at
    1366 x 768 and 1280 x 720: Display under Runtime settings, the page not scrolling; while
    Runtime settings' body has more under its edge, Display is at its least height (it gives
    way first), its options in its body's view; and what Runtime settings holds is reached in
    its body, its one scroll region: its facts' top and foot each brought into the body's
    view, and each field's control (Memory allocator's, Compile's) brought into it whole,
    nothing over it, not disabled, the page itself never scrolled. Fails on 36c0592 (Display
    drawn whole while Runtime settings' body scrolled).

    UI-SW-ER3 A (the external review of ffb1483): this test asked Memory allocator's box to be
    in view unscrolled at 1366 x 768. How much of Runtime settings' body is in view first
    depends on the host (its environment's facts and the footer's warning: a CPU-only host's
    two-line warning left the allocator under the edge, reached by one scroll), and under the
    deployment window a region may scroll (the WebUI skill's §1); the product's promise is the
    give-way order and the reach, which the test now asks for. Where the room is is not. Run
    with this host's own footer, and with the reviewer's two-line warning stood in
    (`_TWO_LINE_RUNTIME`), where the old check fails on this host too (Memory allocator's box
    314 px down a 233 px view at 1366 x 768, the reviewer's numbers)."""
    import src.webui.app as webui_app
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.operator_access import operator_mode
    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    if footer == "two lines":
        monkeypatch.setattr(webui_app, "probe_runtime", lambda: _TWO_LINE_RUNTIME)
    operator_mode(monkeypatch)
    write_demo_workspace(tmp_path / "root" / "demo")
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(tmp_path / "root"))
    sizes = [[1366, 768], [1280, 720]]
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_UNDER_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")], json.dumps(sizes))
    for w, h in sizes:
        got = seen[f"{w}x{h}"]
        where = (w, h, got)
        height, view, width, wide = got["page"]
        assert got["layout"] == "side" and got["under"], where
        assert height <= view and width <= wide, where
        assert got["options"]["bottom"] <= got["options"]["view"], where       # Display's choice
        tall, least = got["display"]
        if got["more"]:                                                         # Display gave way first
            assert least > 0 and tall <= least + 1, where
        assert got["facts"] == {"head": True, "foot": True}, where
        for field in ("allocator", "compile"):
            assert got[field] == {"whole": True, "hit": True, "enabled": True, "still": True}, (field, where)
    # At the display floor the two cannot both be whole, so the give-way is measured there.
    assert seen["1280x720"]["more"], seen
