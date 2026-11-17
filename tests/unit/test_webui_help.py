"""
The shared ⓘ (M5-2, UI-A): one help control for every ordinary help text.
=========================================================================
`src/webui/components/help.py`. What these pin:

- **the registry**: one text per key, a key that can be an HTML id, every text
  escaped in the one hidden container, and a ⓘ that names its text by
  ``aria-describedby``;
- **the built dashboard**: every registered field exists (by ``elem_id``) and
  its title is the field's label; no component keeps ordinary help in
  ``info=``, which holds only what must stay visible (a field's format, unit
  or "off"); every ⓘ the shared renderers draw names a registered text;
- **the text grammar and its budget** (`help_lines`, `budget_problems`, the
  WebUI skill's §7): every text the built dashboard registers has a lead of
  about 90 characters, a few items, at most 80 words, no sentence over 25
  words and its codes only in a trailing "Ref:" line; the Test and Models
  texts at most three items, the train/test overlap text its two
  definitions in about 55 words; and every editable field of Training's
  Model accordion has help;
- **the browser side** builds the bubble's parts with ``createElement`` and
  ``textContent`` and is given to the mount's ``head`` (Gradio 6 drops a
  head given to ``gr.Blocks``); in a real headless browser (Node and
  Playwright, skipped without): a hover previews after a delay as a tooltip
  without "Esc to close", a click pins a dialog named by its title with it,
  the pinned bubble never covers the field it explains (beside, then above,
  then below), a card heading's ⓘ never covers its card, at a wide and a
  narrow window, its width stays within the budget, the ⓘ's hit target is at
  least 24 × 24 px, the lead is bold, "- " lines are one list and a "Term:"
  is bold, and a hostile text shows as literal text in each part.

Module: tests/unit/test_webui_help.py
"""
from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from src.webui.components import help as h  # noqa: E402
from tests.fixtures.browser import node_with_playwright  # noqa: E402


# --------------------------------------------------------------------------- the registry
def test_a_field_and_a_topic_are_registered_once_each():
    registry = h.HelpRegistry()
    assert registry.field("num_epochs", "Epochs", "Smoke test: 2-5") == "num_epochs"
    registry.field("num_epochs", "Epochs", "Smoke test: 2-5")          # the same again: fine
    registry.topic("dataset-fingerprint", "Dataset fingerprint", "The SHA-256 …")
    assert registry.targets() == ("num_epochs",)
    assert "dataset-fingerprint" in registry and "num_epochs" in registry
    with pytest.raises(ValueError, match="twice"):
        registry.field("num_epochs", "Epochs", "other words")


@pytest.mark.parametrize("key", ["", "1abc", "a b", 'x"><script>', "a/b", None])
def test_a_key_must_be_able_to_be_an_html_id(key):
    with pytest.raises(ValueError):
        h.HelpRegistry().topic(key, "T", "text")


def test_a_title_and_a_text_are_required():
    with pytest.raises(ValueError, match="title and a text"):
        h.HelpRegistry().topic("k", "", "text")


def test_every_text_is_escaped_once_in_the_hidden_container():
    registry = h.HelpRegistry()
    registry.field("dx_architecture", "Arch <b>", "the latest run under {data}/runs/<arch>/ & more")
    html = registry.texts_html()
    assert html.startswith('<div class="shep-help-texts" hidden>')
    assert 'id="shep-help-text-dx_architecture"' in html
    assert 'data-help-target="dx_architecture"' in html
    assert "{data}/runs/&lt;arch&gt;/ &amp; more" in html
    assert 'data-help-title="Arch &lt;b&gt;"' in html
    assert "<arch>" not in html and "<b>" not in html


def test_a_help_button_is_a_button_that_names_its_text():
    button = h.help_button("dataset-fingerprint", 'Fingerprint "x"')
    assert button.startswith('<button type="button" class="shep-i"')
    assert 'aria-describedby="shep-help-text-dataset-fingerprint"' in button
    assert 'aria-expanded="false"' in button
    assert 'aria-label="Help: Fingerprint &quot;x&quot;"' in button
    own = h.help_button("chip-d8", "Operator controls", describedby="shep-chip-d8-text")
    assert 'aria-describedby="shep-chip-d8-text"' in own
    with pytest.raises(ValueError):
        h.help_button("not a key", "x")


def test_copy_carries_the_full_value_escaped():
    button = h.copy_button('a"b' + "0" * 62, "the full dataset fingerprint")
    assert 'data-copy="a&quot;b' in button and button.startswith('<button type="button"')
    assert "shep-copy-icon" in h.copy_button("0" * 64, "x", icon_only=True)


def test_the_browser_side_reads_texts_as_text_and_goes_to_the_mount():
    from src.webui.app import mount_options

    head = mount_options()["head"]
    assert h.HELP_HEAD in head
    # The bubble's parts are made and filled with textContent: no help text
    # is parsed as HTML (the one innerHTML is the ⓘ's own icon).
    assert "fill(b.querySelector('.shep-bubble-bd'), textOf(btn))" in h.HELP_HEAD
    assert h.HELP_HEAD.count("innerHTML") == 1 and "btn.innerHTML = '<svg" in h.HELP_HEAD
    assert "insertAdjacentHTML" not in h.HELP_HEAD and "outerHTML" not in h.HELP_HEAD
    for behaviour in ("mouseover", "focusin", "'click'", "Escape", "MutationObserver",
                      "aria-expanded", "aria-describedby"):
        assert behaviour in h.HELP_HEAD, behaviour


def test_a_click_on_a_fields_title_still_reaches_the_field():
    """The ⓘ is a <button> inside the field's <label>: without ``for``, it would
    become the label's control and a click on "Epochs" would open the help
    instead of focusing the input. The label names its input by id."""
    assert "function keepLabel(target, key)" in h.HELP_HEAD
    assert "label.htmlFor = input.id" in h.HELP_HEAD
    assert "input.id = 'shep-in-' + key" in h.HELP_HEAD
    # kept on every pass, for a field Gradio renders again
    assert "keepLabel(target, key); describe(target, key, t.id); continue;" in h.HELP_HEAD


def test_the_bubble_is_in_the_dashboards_font_and_never_taller_than_the_window():
    bubble = h.HELP_CSS[h.HELP_CSS.index(".shep-bubble {"):]
    bubble = bubble[:bubble.index("}")]
    assert "font-family: 'IBM Plex Sans'" in bubble        # it lives outside .gradio-container
    assert "max-height: calc(100vh - 16px)" in bubble
    assert "overflow-y: auto" in h.HELP_CSS[h.HELP_CSS.index(".shep-bubble-bd {"):]
    assert "roomAbove > roomBelow" in h.HELP_HEAD and "bubble.style.maxHeight" in h.HELP_HEAD


def test_the_bubble_is_a_tooltip_previewing_and_a_dialog_when_pinned():
    """The skill's §7 and §10: pinned, the bubble is a dialog named by its
    title; a hover or focus preview is a tooltip; a hover opens after a short
    delay; its width is the 320 px budget; the ⓘ's hit target is 24 × 24."""
    head = h.HELP_HEAD
    assert "bubble.setAttribute('role', pinned ? 'dialog' : 'tooltip')" in head
    assert "bubble.setAttribute('aria-labelledby', 'shep-help-bubble-title')" in head
    assert f"HOVER_MS = {h.HOVER_OPEN_MS}" in head and 150 <= h.HOVER_OPEN_MS <= 600
    assert "openSoon(btn)" in head and "__" not in head.replace("__pwned", "")
    assert h.BUBBLE_MAX_WIDTH == 320 and "max-width: 320px" in h.HELP_CSS
    assert f"MAX_W = {h.BUBBLE_MAX_WIDTH}" in head
    assert h.HIT_TARGET >= 24 and "width: 24px; height: 24px" in h.HELP_CSS
    # Beside, then above, then below the field it explains, never over it.
    assert "['right', 'left', 'above', 'below']" in head and "overlaps(" in head


# --------------------------------------------------------------------------- the text grammar
def test_a_help_text_reads_as_a_lead_items_terms_and_a_ref_line():
    """The grammar the bubble shows (and the script reads by the same
    patterns): the first line is the lead, "- " lines are items, a short
    capitalised "Term: " is bold, a trailing "Ref:" line holds the codes; a
    blank line only separates; a lead is never split into a term."""
    lines = h.help_lines("Default: the lead, never a term.\n\n- Item one\n"
                         "- Term: an item with a term\nPlain words: a sentence's colon, "
                         "too long to be a term here: x\nReference: 200.\nRef: SHEPHERD_X")
    assert [(line.kind, line.term) for line in lines] == [
        ("lead", None), ("item", None), ("item", "Term"), ("para", "Plain words"),
        ("para", "Reference"), ("ref", "Ref")]
    assert lines[0].text == "Default: the lead, never a term."
    assert lines[2].text == "an item with a term" and lines[-1].text == "SHEPHERD_X"
    assert h.help_lines("one line") == (h.HelpLine("lead", "one line"),)
    assert h.help_lines("lowercase: not a term\nauto: not one either")[1].term is None
    assert f"new RegExp({json.dumps(h.TERM_PATTERN)})" in h.HELP_HEAD
    assert f"BULLET = {json.dumps(h.BULLET)}" in h.HELP_HEAD


def test_the_budget_names_what_a_text_runs_over():
    assert h.budget_problems("A short lead.\n- one\n- two\nRef: D5; SHEPHERD_DEVICE") == ()
    long_lead = h.budget_problems("x" * 101)
    assert long_lead and "lead of 101 characters" in long_lead[0]
    assert any("81 words" in p for p in h.budget_problems("Lead.\n" + "word " * 80))
    assert any("a sentence of 26 words" in p
               for p in h.budget_problems("Lead.\n" + "word " * 25 + "end."))
    assert any("5 list items" in p for p in h.budget_problems("Lead.\n" + "- a\n" * 5))
    for code in ("SHEPHERD_DEVICE", "(D5)", "R10", "README C6", "repo@e95433a", "train.py",
                 "mode M1"):
        assert any("a code outside the Ref line" in p
                   for p in h.budget_problems(f"Lead.\nIt names {code} here.")), code
    assert any("'•' bullet" in p for p in h.budget_problems("Lead.\n• old bullet"))
    assert any("not the last line" in p for p in h.budget_problems("Lead.\nRef: x\nmore"))
    # A governed note shown elsewhere as one line keeps its words, a sentence a line.
    assert h.sentence_lines("One. Two words? Three!") == "One.\nTwo words?\nThree!"
    assert h.sentence_lines(None) is None


# --------------------------------------------------------------------------- in a real browser
HOSTILE = "<img src=x onerror=window.__pwned=true> & more"
HOST = Path(__file__).resolve().parent.parent / "fixtures" / "help_host.cjs"


def _page(tmp_path: Path) -> Path:
    registry = h.HelpRegistry()
    # As Gradio 6 draws a field: the component's block (its elem_id), its
    # <label> a "block" too, the title in it.
    registry.field("fld", "Epochs", f"How many passes. {HOSTILE}\n\n- Smoke test: 2-5 {HOSTILE}"
                   f"\n- Every pass sees each pair once.\nDefault: 200, {HOSTILE}\n"
                   "Ref: SHEPHERD_X hparams.py:18")
    # A field drawn as a grid row: its block "display: contents", its label's
    # cells (title, ⓘ, box) the grid's columns (Diagnosis' Options, b405973).
    registry.field("gfld", "Top-K", "How many candidates.")
    registry.topic("kv-help", "A key", "What the key means. " + HOSTILE)
    registry.topic("row-help", "A row", "What the row says.")
    registry.topic("card-help", "A card", "What the card shows.\n\n- Its state.\n- When it "
                   "is read.\n- Why it can be grey.")
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<style>:root {{ --shep-accent: #5558e3; --shep-accent-50: #eef0ff; --shep-accent-text: #3a3dbd;
--shep-border: #e1e4ea; --shep-border-2: #cdd2da; --shep-text: #1d2230; --shep-muted: #5b6270; }}
body {{ margin: 0; font: 15px sans-serif; }}
.shep-kv {{ display: grid; grid-template-columns: 10em 1fr; margin: 24px 40px; max-width: 640px; }}
{h.HELP_CSS}</style>{h.HELP_HEAD}</head><body>
<div class="block" id="fld" style="margin: 40px; width: 300px; padding: 16px; border: 1px solid #ccc">
<label class="block" style="display: block"><span data-testid="block-info">Epochs</span><input>
</label></div>
<div style="display: grid; grid-template-columns: max-content auto 1fr; margin: 24px 40px; width: 336px">
<div class="block" id="gfld" style="display: contents"><label class="block"
style="grid-column: 1 / -1; display: grid; grid-template-columns: subgrid; align-items: center">
<span data-testid="block-info">Top-K</span><input type="number" style="width: 72px"></label></div></div>
<div class="shep-kv"><span class="shep-k">A key{h.help_button("kv-help", "A key")}</span>
<span class="shep-v">its value, long enough to fill the row</span></div>
<table id="tbl" style="margin: 24px 40px; width: calc(100% - 80px)"><tr>
<td>A row{h.help_button("row-help", "A row")}</td><td>its cells</td></tr></table>
<div id="card" class="{h.REGION_CLASS}" style="margin: 24px 40px; width: 340px; padding: 8px;
border: 1px solid #ccc"><div style="display: flex"><div class="shep-card-title"
style="flex: 0 0 auto"><span>A card</span>{h.help_button("card-help", "A card")}</div>
<span style="margin-left: auto">read 12:00</span></div><p>Its state, read now, and long lines
that fill the card's width so a bubble beside its title would cover them.</p>
<p>More of the card.</p></div>
<div style="height: 1400px"></div>
{registry.texts_html()}</body></html>"""
    path = tmp_path / "help.html"
    path.write_text(page, encoding="utf-8")
    return path


def test_the_pinned_bubble_never_covers_its_field_and_shows_text_as_text(tmp_path):
    """In a real browser, at the deployment window (1850 × 960) and a narrow
    one (420 × 800), for a field's ⓘ, a grid-row field's (its block
    ``display: contents``, so it has no box of its own: the bubble covered
    the ⓘ and the field, D08), a key-value row's and a table row's: a
    hover opens nothing before the delay and a tooltip after it; a click pins
    a dialog named by the help's title; the bubble's rect does not intersect
    field's (measured here, by the test's own rule) and is not under the
    field's centre (``document.elementsFromPoint``); a card heading's ⓘ (a
    region, R-10) never covers its card; it is at most 320 px wide; the ⓘ's
    hit target is at least 24 × 24; "Esc to close" shows only pinned
    (R-29); the field's text shows a bold lead, its "- " lines as one list,
    a bold "Term:" and a muted Ref line (R-9), a hostile text in each part is
    literal text and runs nothing; Esc closes it back to a tooltip."""
    found = node_with_playwright()
    if found is None:
        pytest.skip("Node with Playwright is not installed")
    node, env = found
    result = subprocess.run([node, str(HOST), str(_page(tmp_path))], env=env,
                            capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stderr
    sizes = json.loads(result.stdout)
    assert set(sizes) == {"1850x960", "420x800"}
    for size, targets in sizes.items():
        assert set(targets) == {"field", "grid", "kv", "row", "region"}
        for which, seen in targets.items():
            where = f"{size} {which}"
            assert seen["hit"]["width"] >= 24 and seen["hit"]["height"] >= 24, where
            assert seen["early"] is False and seen["hover"] == "tooltip", where
            assert seen["hoverFooter"] is False and seen["pinnedFooter"] is True, where
            pinned = seen["pinned"]
            assert not pinned["hidden"] and pinned["role"] == "dialog", where
            assert pinned["labelledby"] and pinned["title"] in ("Epochs", "Top-K", "A key",
                                                                "A row", "A card"), where
            b, f = pinned["bubble"], pinned["field"]
            assert not (b["left"] < f["right"] and f["left"] < b["right"]
                        and b["top"] < f["bottom"] and f["top"] < b["bottom"]), (where, b, f)
            assert pinned["bubbleAtFieldCentre"] is False, where
            assert pinned["width"] <= 320 + 0.5, where
            assert b["left"] >= 0 and b["right"] <= pinned["viewport"]["width"], where
            assert pinned["made"] == 0 and pinned["pwned"] is False, where
            if which in ("field", "kv"):
                assert HOSTILE in pinned["text"], where
            assert seen["closed"] is True, where
            parts = pinned["parts"]
            assert parts[0]["tag"] == "p" and parts[0]["cls"] == "shep-bubble-lead", where
            assert int(parts[0]["weight"]) >= 600, where
        # The structured text, part by part: each hostile text literal in it.
        field = targets["field"]["pinned"]["parts"]
        assert [(p["tag"], p["cls"]) for p in field] == [
            ("p", "shep-bubble-lead"), ("ul", "shep-bubble-list"), ("p", ""),
            ("p", "shep-bubble-ref")], size
        assert field[0]["text"] == f"How many passes. {HOSTILE}"
        assert field[1]["items"] == [f"Smoke test: 2-5 {HOSTILE}", "Every pass sees each pair once."]
        assert field[1]["terms"] == [{"text": "Smoke test:", "weight": field[1]["terms"][0]["weight"]}]
        assert int(field[1]["terms"][0]["weight"]) >= 600
        assert field[2]["text"] == f"Default: 200, {HOSTILE}" and field[2]["terms"][0]["text"] == "Default:"
        assert field[3]["text"] == "Ref: SHEPHERD_X hparams.py:18"
        assert [p["tag"] for p in targets["region"]["pinned"]["parts"]] == ["p", "ul"]


_DESCRIBED_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + process.argv[2]);
  await page.waitForSelector('#fld button.shep-i');
  const read = () => page.evaluate(() => {
    const input = document.querySelector('#fld input');
    const ids = (input.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean);
    return { ids, texts: ids.map((id) => (document.getElementById(id) || {}).textContent || null) };
  });
  const first = await read();
  // Gradio draws an info= line set at runtime again: a new element, no id.
  await page.evaluate(() => {
    const old = document.querySelector('#fld .info-text');
    const fresh = document.createElement('div');
    fresh.className = 'info-text'; fresh.textContent = 'Now: 7 passes';
    old.replaceWith(fresh);
  });
  await page.waitForTimeout(150);
  const again = await read();
  const bare = await page.evaluate(() => document.querySelector('#bare input').getAttribute('aria-describedby'));
  console.log(JSON.stringify({ first, again, bare }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_fields_visible_info_line_is_part_of_its_description(tmp_path):
    """B5 (leftover d): the help set each input's ``aria-describedby`` to the
    bubble's text alone, so a field's visible ``info=`` line (a format, a
    unit, when a setting applies) was neither its name (the input is
    labelled by its title) nor its description. The shared help now
    describes the input by its info line, then its help text, and keeps it
    so when Gradio draws the line again; a field with no info line is
    described by its help text alone."""
    found = node_with_playwright()
    if found is None:
        pytest.skip("Node with Playwright is not installed")
    node, env = found
    registry = h.HelpRegistry()
    registry.field("fld", "Epochs", "How many passes.")
    registry.field("bare", "Seed", "The seed.")
    page = tmp_path / "described.html"
    page.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><style>{h.HELP_CSS}</style>
{h.HELP_HEAD}</head><body>
<div class="block" id="fld"><label class="block"><span data-testid="block-info">Epochs</span>
<div class="info-text">Whole passes, 1 to 10000. {html.escape(HOSTILE)}</div><input></label></div>
<div class="block" id="bare"><label class="block"><span data-testid="block-info">Seed</span>
<input></label></div>
{registry.texts_html()}</body></html>""", encoding="utf-8")
    script = tmp_path / "described.cjs"
    script.write_text(_DESCRIBED_JS, encoding="utf-8")
    result = subprocess.run([node, str(script), str(page)], env=env, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert seen["first"]["ids"] == ["shep-info-fld", h.text_id("fld")]
    assert seen["first"]["texts"][0] == f"Whole passes, 1 to 10000. {HOSTILE}"
    assert seen["first"]["texts"][1].startswith("How many passes.")
    assert seen["again"]["ids"] == ["shep-info-fld", h.text_id("fld")]
    assert seen["again"]["texts"][0] == "Now: 7 passes"
    assert seen["bare"] == h.text_id("bare")


#: A focus preview whose bubble slides under a pointer the reader never moved
#: (T08: the shared walk's Diagnosis load drawer at 390 x 844). The page's
#: clock is paused, so the bubble's hide runs only when the script advances
#: it: what the test reads never depends on how fast the machine is. At each
#: window: the field put mid-window; its ⓘ focused once to learn where its
#: bubble goes (``R``); the pointer rested just above that place; the ⓘ
#: focused again, then left (its preview's hide pending) and the page
#: scrolled 40 px by the wheel, so the bubble, following its ⓘ, comes under
#: the resting pointer, which the browser tells it with a "mouseover" and no
#: "mousemove". Then the clock runs past the hide. Last, a reader's own
#: pointer, moved from the ⓘ into its hover preview, keeps the bubble, and
#: leaving it closes the bubble.
_RESTING_POINTER_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1850, 960], [420, 800]]) {
    const page = await browser.newPage({ viewport: { width: w, height: h } });
    await page.clock.install();
    await page.goto('file://' + process.argv[2]);
    await page.waitForSelector('#fld button.shep-i');
    // The page's clock paused, and seen to stand still.
    for (let i = 0; ; i++) {
      await page.clock.pauseAt((await page.evaluate(() => Date.now())) + 60000);
      const at = await page.evaluate(() => Date.now());
      await new Promise((r) => setTimeout(r, 50));
      if (await page.evaluate(() => Date.now()) === at) { break; }
      if (i > 4) { throw new Error('the page clock did not pause'); }
    }
    const bubble = () => page.evaluate(() => {
      const b = document.getElementById('shep-help-bubble');
      if (!b || b.hidden) { return null; }
      const r = b.getBoundingClientRect();
      return { left: r.left, top: r.top, width: r.width, height: r.height };
    });
    const inBubble = (p) => page.evaluate(([x, y]) => {
      const b = document.getElementById('shep-help-bubble'), e = document.elementFromPoint(x, y);
      return !!b && !!e && b.contains(e);
    }, p);
    await page.evaluate(() => window.scrollTo(0, 300));
    await page.keyboard.press('Tab');
    await page.focus('#fld button.shep-i');
    const R = await bubble();
    await page.evaluate(() => document.activeElement.blur());
    await page.clock.runFor(1000);
    const dryClosed = (await bubble()) === null;
    const P = [Math.round(R.left + R.width / 2), Math.round(R.top - 20)];
    await page.mouse.move(P[0], P[1]);
    await page.focus('#fld button.shep-i');
    const before = { bubble: await bubble(), underPointer: await inBubble(P) };
    await page.evaluate(() => {
      window.__entered = false; window.__moves = 0;
      document.getElementById('shep-help-bubble').addEventListener('mouseover', () => { window.__entered = true; });
      document.addEventListener('mousemove', () => { window.__moves += 1; }, true);
      document.activeElement.blur();
    });
    await page.mouse.wheel(0, 40);
    // The bubble follows its ⓘ on the scroll and the browser then tells it
    // the pointer is over it: both observed, the page's clock run a frame at
    // a time, never as far as the pending hide (250 ms).
    let slid = false, ran = 0;
    for (const end = Date.now() + 5000; Date.now() < end; await new Promise((r) => setTimeout(r, 20))) {
      if (await inBubble(P) && await page.evaluate(() => window.__entered)) { slid = true; break; }
      if (ran < 192) { await page.clock.runFor(16); ran += 16; }
    }
    await page.clock.runFor(1000);
    const after = { bubble: await bubble(), underPointer: await inBubble(P),
                    moves: await page.evaluate(() => window.__moves) };
    // A reader's own pointer: over the ⓘ (its preview after the delay), then
    // into the bubble; the bubble stays while it is there and closes after it leaves.
    const i = await page.locator('#fld button.shep-i').boundingBox();
    await page.mouse.move(i.x + i.width / 2, i.y + i.height / 2);
    await page.clock.runFor(1000);
    const hover = await bubble();
    await page.mouse.move(hover.left + hover.width / 2, hover.top + hover.height / 2, { steps: 8 });
    await page.clock.runFor(1000);
    const kept = (await bubble()) !== null;
    await page.mouse.move(w - 4, h - 4, { steps: 4 });
    await page.clock.runFor(1000);
    const left = (await bubble()) === null;
    out[`${w}x${h}`] = { R, P, dryClosed, before, slid, ran, after, hover: hover !== null, kept, left };
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_focus_preview_closes_though_it_slid_under_a_resting_pointer(tmp_path):
    """T08 (the shared walk's ``ui_walk`` errored at setup): Tab moved
    through the Diagnosis load drawer at 390 x 844, the Validation (Trainer)
    ⓘ's preview opened on focus, and the next control's scroll moved the
    bubble, with its ⓘ, under the pointer the walk had left resting. The
    browser sends the bubble a "mouseover" then (no "mousemove"), and the
    bubble's ``mouseenter`` cancelled the hide its ⓘ's focusout had asked
    for: the preview outlived its focus and, kept inside the window, covered
    the drawer's ×, so its Close never reached the server. A preview now
    stays only for the reader's own pointer moving in it: here it closes and
    the page under the pointer is reached again, while a pointer moved from
    the ⓘ into its hover preview still keeps the bubble until it leaves (at
    1850 x 960 and 420 x 800)."""
    found = node_with_playwright()
    if found is None:
        pytest.skip("Node with Playwright is not installed")
    node, env = found
    registry = h.HelpRegistry()
    registry.field("fld", "Epochs", "How many passes.\n\n- Smoke test: 2-5.\n- Every pass sees "
                   "each pair once.\n- Default: 200.")
    page = tmp_path / "resting.html"
    page.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<style>body {{ margin: 0; font: 15px sans-serif; }} {h.HELP_CSS}</style>{h.HELP_HEAD}</head><body>
<div style="height: 540px"></div>
<div class="block" id="fld" style="margin: 0 40px; width: 300px; padding: 16px; border: 1px solid #ccc">
<label class="block" style="display: block"><span data-testid="block-info">Epochs</span><input>
</label></div><div style="height: 2400px"></div>
{registry.texts_html()}</body></html>""", encoding="utf-8")
    script = tmp_path / "resting.cjs"
    script.write_text(_RESTING_POINTER_JS, encoding="utf-8")
    result = subprocess.run([node, str(script), str(page)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert set(seen) == {"1850x960", "420x800"}
    for size, s in seen.items():
        # The scene as the walk met it: a focus preview, the pointer resting
        # outside it, then the bubble under that pointer with no pointer move.
        assert s["dryClosed"] and s["before"]["bubble"] == s["R"], (size, s)
        assert s["before"]["underPointer"] is False and s["slid"] is True, (size, s)
        assert s["ran"] < 250 and s["after"]["moves"] == 0, (size, s)
        # Its hide was not cancelled: closed, the page under the pointer reached again.
        assert s["after"]["bubble"] is None and s["after"]["underPointer"] is False, (size, s)
        # A reader moving into a hover preview keeps it; leaving closes it.
        assert s["hover"] and s["kept"] and s["left"], (size, s)


# --------------------------------------------------------------------------- the built dashboard
@pytest.fixture(scope="module")
def built():
    """The dashboard as `mount_dashboard` builds it, and its help registry."""
    from src.webui import shell
    from src.webui.app import PAGES

    registry = h.HelpRegistry()
    with gr.Blocks() as demo:
        shell.build_shell(demo, PAGES, help_registry=registry)
        texts = shell.help_texts(registry)
    return demo, registry, texts


def test_every_registered_help_target_exists_in_the_built_app(built):
    """Structural, so a page that adds or removes a field (UI-B, UI-C, UI-D)
    changes no test here: every page module the app builds that declares
    ``FIELD_HELP`` has all of it registered, and every registered target is a
    component of the built app whose label is the help's title."""
    import sys

    from src.webui.app import PAGES

    demo, registry, _texts = built
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    modules = {sys.modules[builder.__module__] for builder in PAGES.values()}
    declared = [m.FIELD_HELP for m in modules if isinstance(getattr(m, "FIELD_HELP", None), dict)]
    assert declared, "no page declares field help"
    for field_help in declared:
        assert set(field_help) <= set(registry.targets())
    for entry in registry.entries():
        if entry.target is None:
            continue
        assert entry.target in by_id, entry.target
        component = by_id[entry.target]
        assert getattr(component, "label", None) == entry.title, entry.target


def test_no_component_keeps_ordinary_help_in_info(built):
    """``info=`` holds only text that must stay visible (the WebUI skill's §7):
    a state set at runtime, the host's Device note (no CUDA device: auto and
    cuda are refused), and a field's format, unit or "off", set at build
    time only on the Training page's base form, its Continue drawer's
    Branch fields and the Fine-tune page's fields, the one field of each
    setting (`FIELD_FORMATS`, by its setting or its model spec field), when
    each of Settings' two fields applies (`runtime_settings.FIELD_INFO`), and
    the Diagnosis fields' rules (`diagnosis_panel.FIELD_INFO`: which HPO ids
    are read, what a blank Patient ID does); no ordinary help is kept there."""
    from src.webui.components import diagnosis_panel as dp
    from src.webui.components import finetune_page as fp
    from src.webui.components import runtime_settings as rs
    from src.webui.components import training_console as tc

    demo, _registry, _texts = built
    with_info = {getattr(b, "elem_id", None): b.info for b in demo.blocks.values()
                 if getattr(b, "info", None)}
    elem_ids = {name: name for name in tc.BASE_FIELDS}
    elem_ids.update({field.path: field.elem_id for field in tc.spec_form_fields()})
    expected = {elem_ids[key]: text for key, text in tc.FIELD_FORMATS.items()}
    expected.update({f"branch_{name}": text for name, text in tc.FIELD_FORMATS.items()
                     if name in tc.BRANCH_FIELDS})
    expected.update({f"ft_{name}": text for name, text in tc.FIELD_FORMATS.items()
                     if name in fp.FINETUNE_FIELDS})
    expected.update(rs.FIELD_INFO)
    expected.update(dp.FIELD_INFO)
    # The one Device note, on every Device field, where no CUDA device is
    # visible (R-32, `training_console.device_note`).
    if tc.device_note():
        expected.update(dict.fromkeys(("device", "continue_device", "ft_device"),
                                      tc.DEVICE_NOTE))
    assert with_info == expected


def test_the_hidden_container_holds_every_registered_text(built):
    _demo, registry, texts = built
    for entry in registry.entries():
        assert f'id="{h.text_id(entry.key)}"' in texts.value


def test_every_renderer_help_key_is_registered(built):
    """The ⓘ the shared renderers draw name registered texts (the shell
    registers RENDERER_HELP)."""
    from src.webui.components import renderers as r

    _demo, registry, _texts = built
    workspace = {"name": "w", "dataset_fingerprint": "a" * 64, "trainable": True,
                 "dataset_record": {"binding": "verified", "origin": "files"},
                 "build": {"state": "not_recorded"}, "declared_releases": ()}
    item = {"workspace": "w", "architecture": "gatv2", "run": "20261017-091500_run_x",
            "filename": "best.pt", "sha256": "b" * 64, "epoch": 3,
            "validation": {"metric": "val_pooled_mrr", "value": 0.3},
            "dataset_fingerprint": "a" * 64,
            "problem": None}
    rendered = (r.dataset_identity_html(workspace)
                + r.model_facts_html(r.facts_from_checkpoint(item, [workspace]))
                + r.fingerprint_html("model fingerprint", "b" * 64, help_key="model-fingerprint"))
    keys = h.registered_keys_in(rendered)
    assert set(keys) >= {"dataset-fingerprint", "validation-trainer", "model-fingerprint"}
    for key in keys:
        assert key in registry, key


# --------------------------------------------------------------------------- the budget, over every text
def test_every_registered_help_text_is_within_the_budget(built):
    """R-9 (the WebUI skill's §7): every text the built dashboard registers,
    whichever page registers it, answers first in a lead of about 90
    characters, has a few items, about 60-80 words without its Ref line, no
    sentence over 25 words and its codes (setting names, decision, review
    and README codes, a source commit or file) only in a trailing Ref line.
    The texts the base held failed it: 89 of its 131."""
    _demo, registry, _texts = built
    over = {entry.key: h.budget_problems(entry.text) for entry in registry.entries()}
    assert {key: problems for key, problems in over.items() if problems} == {}
    # 147 since R-49: the Continue drawer's source radio, and its text, went
    # behind a link; its facts are the Checkpoint path's text. 146 since R-82:
    # the Test page's and Models' R10 texts are the one record's ("record-r10").
    assert len(over) >= 146


def _words(text: str) -> int:
    return sum(len(f"{line.term or ''} {line.text}".split()) for line in h.help_lines(text)
               if line.kind != "ref")


def test_the_test_and_models_texts_are_an_answer_and_three_items_at_most(built):
    """R-74: the Test and Models (Results included) texts are one answer
    sentence and at most three items, about 60 words, their codes in the Ref
    line (the budget above)."""
    _demo, registry, _texts = built
    keys = [e.key for e in registry.entries()
            if e.key.startswith(("shep-test-", "test-", "models-"))]
    assert len(keys) >= 24
    for entry in registry.entries():
        if entry.key in keys:
            lines = h.help_lines(entry.text)
            assert sum(line.kind == "item" for line in lines) <= 3, entry.key
            assert _words(entry.text) <= 70, (entry.key, _words(entry.text))


def test_the_train_test_overlap_text_defines_each_column_once(built):
    """R-69: the overlap labels' text was the four generated definitions in
    one 155-word paragraph, the training and validation ones repeating each
    other; it is one hand-written text (Test's result and Models' records
    alike): a lead naming the rows, one definition line per column, about 55
    words, the code in its Ref line."""
    from src.webui.components import renderers as r

    _demo, registry, _texts = built
    by_key = {e.key: e for e in registry.entries()}
    for key in ("record-r10",):
        entry = by_key[key]
        assert (entry.title, entry.text) == r.R10_HELP and entry.title == r.R10_TITLE
        lines = h.help_lines(entry.text)
        assert "training or validation inputs" in lines[0].text
        assert [line.term for line in lines if line.term and line.kind == "para"] == [
            "Disease overlap", "Phenotype-set overlap"]
        assert "ignore order and repeats" in entry.text and lines[-1].text == "R10"
        assert _words(entry.text) <= 60


def test_every_model_spec_field_has_help(built):
    """NEW-help-1: the Model accordion's 17 editable fields had no help; each
    now has its ⓘ (registered by its elem_id, titled by its label): what it
    is, its reference value and, where the reference recipe names one, its
    source in the Ref line."""
    from src.webui.components import training_console as tc

    _demo, registry, _texts = built
    editable = [field for field in tc.spec_form_fields() if field.editable]
    assert len(editable) == 17
    targets = set(registry.targets())
    by_key = {e.key: e for e in registry.entries()}
    for field in editable:
        assert field.elem_id in targets, field.path
        entry = by_key[field.elem_id]
        assert entry.text.split("\n", 1)[0] == tc.SPEC_FIELD_WORDS[field.path]
        assert "Reference: " in entry.text, field.path
    assert set(tc.SPEC_FIELD_WORDS) == {field.path for field in editable}
    concat = by_key["spec-encoder-concat"].text
    assert "GATv2 needs it on" in concat and concat.endswith("README C1)")
