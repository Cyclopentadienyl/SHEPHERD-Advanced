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
  is bold, and a hostile text shows as literal text in each part; a bubble
  whose ⓘ is scrolled out of view (the window or a region) is not drawn and
  takes no click, and is drawn again with its ⓘ back (T08-help-1); a click on
  a control under a focus preview reaches it (T08-flow-1), and the wheel over
  that preview still scrolls its capped text; and, in the served
  dashboard, Tab meets each checkbox's ⓘ and box in the order drawn
  (T08-flow-3).

Module: tests/unit/test_webui_help.py
"""
from __future__ import annotations

import html
import json
import re
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


def test_the_text_layout_budget_counts_what_a_320_px_bubble_draws():
    """The UI sweep of 2026-11-22 (its text brief's §11 and §14 #8): the text
    layout's budget (`help.TEXT_BUDGET`) adds to the skill's what a 320 px
    bubble draws in lines of about 39 characters: a lead of 90 at most (one
    over 78, three lines, is noted), none or 2-4 items of 76 characters at
    most, all of one form, and plain words (no semicolon, no "please",
    "sorry", "oops", "invalid", "an error occurred" or "we"). The skill's own
    budget (`HELP_BUDGET`) is unchanged: every registered text meets it."""
    tb = h.TEXT_BUDGET
    good = "Says what it is in a short lead.\n- First: one fact\n- Second: another fact\nRef: X"
    assert h.budget_problems(good, tb) == () and h.budget_notes(good) == ()
    assert any("lead of 91" in p for p in h.budget_problems("x" * 91, tb))
    assert h.budget_problems("x" * 91) == ()                       # the skill's: 100
    assert h.budget_notes("x" * 80) == ("a lead of 80 characters (three lines over 78)",)
    assert any("1 list item" in p for p in h.budget_problems("Lead.\n- only one", tb))
    assert any("an item of 77" in p for p in h.budget_problems("Lead.\n- " + "y" * 77 + "\n- b", tb))
    assert any("two forms (term rows" in p for p in h.budget_problems("Lead.\n- A: b\n- plain", tb))
    assert any("two forms (sentences" in p for p in h.budget_problems("Lead.\n- One.\n- two", tb))
    assert any("semicolon" in p for p in h.budget_problems("Lead.\nOne; two.", tb))
    for word in ("please", "Sorry", "invalid", "we", "an error occurred"):
        assert any("never says" in p for p in h.budget_problems(f"Lead.\nThen {word} here.", tb)), word
    assert h.budget_problems("Lead.\nRef: a; b", tb) == ()        # the Ref line holds codes
    assert h.HELP_BUDGET == h.HelpBudget()


def test_the_shared_renderers_texts_meet_the_text_layout_budget():
    """The texts the shared renderers' ⓘ name (`renderers.RENDERER_HELP`, the
    shell registers them) meet the text layout's budget; the governed ones
    (the exposure caveat, R10) keep their words, as paragraphs where a part
    is longer than a two-line item."""
    from src.webui.components import renderers as r

    for key, (_title, text) in r.RENDERER_HELP.items():
        assert h.budget_problems(text, h.TEXT_BUDGET) == (), key


def _dashboard_help(monkeypatch):
    """Every help text the built dashboard registers, by key (`shell.help_texts` is handed the
    one registry after every page has registered its own)."""
    from src.webui import app, shell

    seen, real = [], shell.help_texts
    monkeypatch.setattr(shell, "help_texts", lambda registry: seen.append(registry) or real(registry))
    app.create_gradio_app()
    (registry,) = seen
    return {entry.key: entry.text for entry in registry.entries()}


#: A help text keyed by a field's name carries that field's spec description as its lead (the
#: Training form's, the Continue drawer's branch fields, Fine-tune's).
_FIELD_KEY = re.compile(r"(?:branch_|ft_)?(?P<name>[a-z_]+)")


def _exempt_source(key: str, text: str):
    """**The one exemption from the text budget**: the words a text keeps verbatim from its
    source, which the WebUI may not change. A lead that is its field's API wording
    (`training_fields.FieldSpec.description`, which the schema and the refusals share), or a
    text that is a governed note (`caveats.MODE_RESULT_NOTES`, the Test page's measurement
    notes), as (the exempt words, their source); None for every other text."""
    from src.config import training_fields as tf
    from src.evaluation import caveats

    if key.startswith("test-mode-"):
        mode = key[len("test-mode-"):]
        return " ".join(text.split("\n")), caveats.MODE_RESULT_NOTES.get(mode)
    found = _FIELD_KEY.fullmatch(key)
    spec = {f.name: f for f in tf.FIELDS}.get(found.group("name")) if found else None
    lead = h.help_lines(text)[0].text
    if spec is not None and spec.description and lead == spec.description:
        return lead, spec.description
    return None


def test_every_registered_help_text_meets_the_text_budget_but_its_verbatim_source(monkeypatch):
    """E2 of the sweep's integration (its decision 6): every help text the dashboard registers,
    on every page, meets the text layout's budget (`help.TEXT_BUDGET`: a lead of 90 characters
    at most, none or 2-4 items of one form each 76 at most, 80 words, plain words, Ref last).
    The one exemption is words the WebUI may not change: a lead that is its field's API wording
    (`FieldSpec.description`, equal to it exactly; the rest of its text, its source line and
    term rows, meets the budget) and a governed measurement note (`caveats.MODE_RESULT_NOTES`,
    the whole text equal to it, a sentence a line). A lead over 78 characters (three bubble
    lines) stays a note (`help.budget_notes`), not a failure."""
    texts = _dashboard_help(monkeypatch)
    assert len(texts) >= 140, len(texts)
    exempt, over = {}, {}
    for key, text in texts.items():
        problems = h.budget_problems(text, h.TEXT_BUDGET)
        source = _exempt_source(key, text) if problems else None
        if source is not None:
            words, original = source
            assert original is not None and words == original, (key, words, original)
            exempt[key] = words
            # The governed note is the whole text; a field's text past its API-worded lead (its
            # source line, its term rows) still meets the budget.
            rest = "" if key.startswith("test-mode-") else text.split("\n", 1)[-1]
            problems = h.budget_problems("Lead.\n" + rest, h.TEXT_BUDGET) if rest.strip() else ()
        if problems:
            over[key] = problems
    assert over == {}, over
    assert all(k.startswith("test-mode-") or _FIELD_KEY.fullmatch(k) for k in exempt), sorted(exempt)


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
    literal text and runs nothing; Esc closes it back to a tooltip. The parts
    are set apart by the text tokens (the UI sweep of 2026-11-22): a
    paragraph's space above each part after the first, an item's above each
    list item after the first, none above the first (a reset of ``p`` and
    ``ul`` outweighed the tokens, so every part touched the next)."""
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
        assert [p["above"] for p in field] == [0, 8, 8, 8], size        # --shep-space-para
        assert field[1]["itemsAbove"] == [0, 4], size                    # --shep-space-item
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


def _controls(prefix: str, rows: int) -> str:
    """A band of `rows` x 8 buttons that count their clicks (``data-clicks``)."""
    return ('<div style="display: grid; grid-template-columns: repeat(8, 1fr); grid-auto-rows: 40px">'
            + "".join(f'<button type="button" class="ctl" id="{prefix}{n}">{prefix}{n}</button>'
                      for n in range(rows * 8)) + "</div>")


#: T08-help-1. At each window, for a field's ⓘ whose page scrolls (the
#: window, by 2000 px) and one whose region scrolls inside (by 300 px: its ⓘ
#: then above the region, still in the window), its bubble pinned by a click
#: or previewed by focus: the bubble as drawn; the scroll; whether the bubble
#: is still drawn (waited for); what is at the place a bubble kept inside
#: the window stands (its place beside the field, moved as its ⓘ moved, kept
#: inside the window: what 03aaf86 drew there); the ⓘ back in view (the
#: bubble drawn again, where); the ⓘ away again and one click at that place.
#: Then, pinned, with focus inside the bubble (its ×): the ⓘ scrolled away,
#: where focus is and whether the page moved; Esc. Last, pinned and closed
#: by its ×: what is drawn, and once the ⓘ is scrolled away.
_AWAY_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const out = {};
  const I = '#fld button.shep-i';
  for (const [w, h] of [[1850, 960], [960, 1080]]) {
    for (const where of ['window', 'region']) {
      const away = where === 'window' ? 2000 : 300;
      const file = 'file://' + process.argv[where === 'window' ? 2 : 3];
      const scroll = (page, by) => page.evaluate(([where, by]) => {
        if (where === 'window') { window.scrollTo(0, by); } else { document.getElementById('region').scrollTop = by; }
      }, [where, by]);
      const offset = (page) => page.evaluate((where) => (where === 'window' ? window.scrollY
        : document.getElementById('region').scrollTop), where);
      const drawn = (page, want) => page.waitForFunction((want) => {
        const b = document.getElementById('shep-help-bubble');
        return !!b && (!b.hidden && b.checkVisibility({ visibilityProperty: true })) === want;
      }, want, { timeout: 3000 }).then(() => true, () => false);
      const read = (page) => page.evaluate((I) => {
        const b = document.getElementById('shep-help-bubble'), r = b.getBoundingClientRect();
        const a = document.querySelector(I).getBoundingClientRect(), g = document.getElementById('region');
        return { role: b.getAttribute('role'), bubble: { left: r.left, top: r.top, width: r.width, height: r.height },
                 i: { top: a.top, bottom: a.bottom }, region: g ? g.getBoundingClientRect().top : null };
      }, I);
      for (const how of ['pinned', 'preview']) {
        const page = await browser.newPage({ viewport: { width: w, height: h } });
        await page.goto(file);
        await page.waitForSelector(I);
        if (how === 'pinned') { await page.click(I); } else { await page.focus(I); }
        const shown = await drawn(page, true);
        const before = await read(page);
        await scroll(page, away);
        const hid = await drawn(page, false);
        const moved = await read(page);
        const top = Math.min(Math.max(8, before.bubble.top + moved.i.top - before.i.top), h - before.bubble.height - 8);
        const P = [Math.round(before.bubble.left + before.bubble.width / 2),
                   Math.round(top + Math.min(40, before.bubble.height / 2))];
        const at = await page.evaluate(([x, y]) => {
          const b = document.getElementById('shep-help-bubble'), all = document.elementsFromPoint(x, y);
          const ctl = all.find((e) => e.matches('button.ctl'));
          return { bubble: !!all[0] && b.contains(all[0]), ctl: ctl ? ctl.id : null };
        }, P);
        await scroll(page, 0);
        const back = await drawn(page, true);
        const again = await read(page);
        await scroll(page, away);
        await drawn(page, false);
        await page.mouse.click(P[0], P[1]);
        const closed = await page.waitForFunction(() => document.getElementById('shep-help-bubble').hidden,
                                                  null, { timeout: 3000 }).then(() => true, () => false);
        const clicks = at.ctl ? await page.evaluate((id) => Number(document.getElementById(id).dataset.clicks || 0), at.ctl) : null;
        out[`${w}x${h} ${where} ${how}`] = { shown, before, hid, moved, P, at, back, again, clicks, closed,
                                             offset: await offset(page) };
        await page.close();
      }
      // Focus inside a pinned bubble (its ×, as a press on it leaves it), then its ⓘ scrolled away.
      const page = await browser.newPage({ viewport: { width: w, height: h } });
      await page.goto(file);
      await page.waitForSelector(I);
      await page.click(I);
      await drawn(page, true);
      await page.evaluate(() => {
        window.escSeen = false;
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { window.escSeen = true; } });
        document.querySelector('#shep-help-bubble .shep-bubble-x').focus();
      });
      const inside = await page.evaluate(() => document.activeElement.className);
      await scroll(page, away);
      const hid = await drawn(page, false);
      const focus = await page.evaluate((I) => document.activeElement === document.querySelector(I) ? 'i'
        : document.activeElement.className || document.activeElement.tagName, I);
      const kept = await offset(page);
      await page.keyboard.press('Escape');
      const closed = await page.waitForFunction(() => document.getElementById('shep-help-bubble').hidden,
                                                null, { timeout: 3000 }).then(() => true, () => false);
      out[`${w}x${h} ${where} esc`] = { inside, hid, focus, kept, closed, offset: await offset(page),
                                        escSeen: await page.evaluate(() => window.escSeen) };
      await page.close();
      // Pinned, then closed by its × (the press leaves focus on the ×); then the ⓘ scrolled away.
      const xpage = await browser.newPage({ viewport: { width: w, height: h } });
      await xpage.goto(file);
      await xpage.waitForSelector(I);
      await xpage.click(I);
      await drawn(xpage, true);
      await xpage.click('#shep-help-bubble .shep-bubble-x');
      const gone = await drawn(xpage, false);
      const state = await xpage.evaluate((I) => ({ expanded: document.querySelector(I).getAttribute('aria-expanded'),
        focus: document.activeElement === document.querySelector(I) ? 'i' : document.activeElement.className }), I);
      await scroll(xpage, away);
      await xpage.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
      const after = await xpage.evaluate(() => { const b = document.getElementById('shep-help-bubble');
        return !b.hidden && b.checkVisibility({ visibilityProperty: true }); });
      out[`${w}x${h} ${where} x`] = { gone, ...state, after, offset: await offset(xpage) };
      await xpage.close();
    }
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_bubble_whose_i_left_view_is_not_drawn_over_other_controls(tmp_path):
    """T08-help-1: an open bubble followed its ⓘ on a scroll but was always
    kept inside the window, and nothing asked whether its ⓘ was still in
    view; once the ⓘ was scrolled out of the window, or out of the visible
    part of the region it scrolls in, the bubble stayed drawn over controls it
    has nothing to do with, and a click on one landed on the bubble. Now a
    bubble, pinned or a preview, whose ⓘ is out of view is not drawn (as
    Floating UI's ``hide`` does) and takes no click: the control under its
    place gets it. It stays open: with its ⓘ back in view it is drawn again
    where it was (beside its field, never over it). Focus inside it when it
    hides moves to its ⓘ without scrolling the page, and Esc then closes it
    without moving the page and still reaches the page (an Esc for the drawer
    the reader sees is not eaten by a bubble they cannot see). Closed by its
    ×, it stays closed: focus given back to its ⓘ opened a preview with no
    ⓘ to follow, which a scroll left drawn over the page. At 1850 x 960 and
    960 x 1080, the window scrolled by 2000 px and an inner region by
    300 px."""
    found = node_with_playwright()
    if found is None:
        pytest.skip("Node with Playwright is not installed")
    node, env = found
    registry = h.HelpRegistry()
    registry.field("fld", "Epochs", "How many passes.\n\n- Smoke test: 2-5.\n- Every pass sees "
                   "each pair once.\n- Default: 200.")
    field = ('<div class="block" id="fld" style="width: 300px; padding: 16px; border: 1px solid #ccc; '
             'background: #fff"><label class="block" style="display: block">'
             '<span data-testid="block-info">Epochs</span><input></label></div>')
    head = (f'<!doctype html><html><head><meta charset="utf-8"><style>body {{ margin: 0; font: 15px '
            f'sans-serif; }} button.ctl {{ margin: 0; border: 1px solid #ddd; background: #fafafa; }}'
            f'{h.HELP_CSS}</style>{h.HELP_HEAD}<script>document.addEventListener("click", (e) => {{'
            f' const b = e.target.closest && e.target.closest("button.ctl");'
            f' if (b) {{ b.dataset.clicks = Number(b.dataset.clicks || 0) + 1; }} }});</script></head><body>')
    tail = f"{registry.texts_html()}</body></html>"
    window = tmp_path / "window.html"
    window.write_text(f'{head}<div style="padding: 40px 40px 16px">{field}</div>{_controls("w", 90)}{tail}',
                      encoding="utf-8")
    region = tmp_path / "region.html"
    region.write_text(
        f'{head}{_controls("t", 9)}<div id="region" style="height: 360px; overflow: auto; margin: 0 16px; '
        f'border: 1px solid #ccc"><div style="padding: 8px 24px">{field}</div>{_controls("r", 20)}</div>'
        f'{_controls("b", 20)}{tail}', encoding="utf-8")
    script = tmp_path / "away.cjs"
    script.write_text(_AWAY_JS, encoding="utf-8")
    result = subprocess.run([node, str(script), str(window), str(region)], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert len(seen) == 16, sorted(seen)
    for case, s in seen.items():
        size, where, how = case.split()
        if how == "esc":
            # Focus left the hidden bubble for its ⓘ, the page unmoved; Esc
            # closed the bubble, moved nothing and reached the page.
            assert s["inside"] == "shep-bubble-x" and s["hid"], (case, s)
            assert s["focus"] == "i" and s["kept"] == s["offset"] > 0, (case, s)
            assert s["closed"] and s["escSeen"], (case, s)
            continue
        if how == "x":
            # Closed by its ×, it stays closed (focus back on its ⓘ opens no
            # preview), and nothing is drawn once the page scrolls.
            assert s["gone"] and s["expanded"] == "false" and s["focus"] == "i", (case, s)
            assert s["after"] is False and s["offset"] > 0, (case, s)
            continue
        away = 2000 if where == "window" else 300
        assert s["shown"] and s["before"]["role"] == ("dialog" if how == "pinned" else "tooltip"), (case, s)
        # The ⓘ out of view: above the window, or above its region in the window.
        m = s["moved"]
        if where == "window":
            assert m["i"]["bottom"] < 0, (case, m)
        else:
            assert 0 <= m["i"]["top"] and m["i"]["bottom"] <= m["region"], (case, m)
        # Not drawn, and a control, not the bubble, under its place.
        assert s["hid"] and not s["at"]["bubble"] and s["at"]["ctl"], (case, s)
        # Back in view: drawn again, where it was, still open as it was.
        assert s["back"] and s["again"]["role"] == s["before"]["role"], (case, s)
        for k in ("left", "top", "width", "height"):
            assert abs(s["again"]["bubble"][k] - s["before"]["bubble"][k]) <= 1, (case, s)
        # One click at its place reached that control; the bubble closed; nothing scrolled.
        assert s["clicks"] == 1 and s["closed"] and s["offset"] == away, (case, s)


#: T08-flow-1 (Fine-tune: Epochs typed, Tab, the Seed ⓘ's focus preview over
#: the Graph backbone radios). The page's clock paused, as in the resting
#: pointer's test. At each window: a click in Epochs, a key typed, Tab: focus
#: on Seed's ⓘ, its preview drawn; which radio it covers (``elementFromPoint``
#: at its centre in the bubble). One mouse click there (the pointer moves from
#: Epochs to it, as a mouse does); the clock run past the hide. Then the
#: preview again (focus on the ⓘ) and the pointer moved across it and out of
#: it: the preview, its focus kept, is still drawn after the hide's delay;
#: focus leaving closes it.
_THROUGH_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1850, 960], [420, 800]]) {
    const page = await browser.newPage({ viewport: { width: w, height: h } });
    await page.clock.install();
    await page.goto('file://' + process.argv[2]);
    await page.waitForSelector('#seed button.shep-i');
    for (let i = 0; ; i++) {
      await page.clock.pauseAt((await page.evaluate(() => Date.now())) + 60000);
      const at = await page.evaluate(() => Date.now());
      await new Promise((r) => setTimeout(r, 50));
      if (await page.evaluate(() => Date.now()) === at) { break; }
      if (i > 4) { throw new Error('the page clock did not pause'); }
    }
    const state = () => page.evaluate(() => {
      const b = document.getElementById('shep-help-bubble'), a = document.activeElement;
      return { shown: !!b && !b.hidden && b.checkVisibility({ visibilityProperty: true }),
               role: b ? b.getAttribute('role') : null,
               focus: a.matches('#seed button.shep-i') ? 'seed-i' : (a.id || a.value || a.tagName),
               checked: [...document.querySelectorAll('#backbone input')].filter((i) => i.checked).map((i) => i.value) };
    });
    await page.click('#ep');
    await page.keyboard.type('3');
    await page.keyboard.press('Tab');
    const tabbed = await state();
    const covered = await page.evaluate(() => {
      const b = document.getElementById('shep-help-bubble');
      return [...document.querySelectorAll('#backbone label')].map((l) => {
        const r = l.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
        return { value: l.querySelector('input').value, x, y, under: b.contains(document.elementFromPoint(x, y)) };
      }).filter((l) => l.under && !l.value.startsWith('keep'));
    });
    const target = covered[0];
    if (target) { await page.mouse.click(target.x, target.y); }
    await page.clock.runFor(1000);
    const clicked = await state();
    // The preview again, the pointer across it and out of it.
    await page.focus('#seed button.shep-i');
    const again = await state();
    const box = await page.evaluate(() => {
      const r = document.getElementById('shep-help-bubble').getBoundingClientRect();
      return { left: r.left, top: r.top, width: r.width, height: r.height };
    });
    await page.mouse.move(box.left + box.width / 2, box.top + box.height / 2, { steps: 6 });
    await page.mouse.move(box.left + box.width + 30, box.top + box.height + 30, { steps: 6 });
    await page.clock.runFor(1000);
    const crossed = await state();
    await page.evaluate(() => document.activeElement.blur());
    await page.clock.runFor(1000);
    const left = await state();
    out[`${w}x${h}`] = { tabbed, covered, target: target || null, clicked, again, crossed, left };
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_focus_preview_lets_the_click_through_to_the_control_under_it(tmp_path):
    """T08-flow-1: Fine-tune at 1850 x 960, Epochs typed and Tab pressed,
    focus on the Seed field's ⓘ opened its preview (a tooltip opened by
    focus, not by the pointer) over the Graph backbone radios; the first
    mouse click on Frozen landed on the preview, which only closed it (the
    radio stayed Trained). A preview the pointer did not open now lets the
    reader's pointer through once it moves over it: one click on the control
    under it reaches that control (its state changes) and the preview, its
    focus gone, closes. Its focus still keeps it: the pointer moving across it
    and out does not close it (WCAG 1.4.13, persistent). A preview the pointer
    opened stays the pointer's (the resting pointer's test: the pointer moved
    into it keeps it) and a pinned bubble is a dialog. At 1850 x 960 and
    420 x 800."""
    found = node_with_playwright()
    if found is None:
        pytest.skip("Node with Playwright is not installed")
    node, env = found
    registry = h.HelpRegistry()
    registry.field("epochs", "Epochs", "How many passes over the training pairs.")
    registry.field("seed", "Seed", "The random seed of the weights' start and the sampling.\n"
                   "Reference: 33. A provisional default from the source.\n- Default: 42.")
    radio = ('<label style="display: block; padding: 6px 0"><input type="radio" name="bb" '
             'value="{v}"{c}> {t}</label>')
    page = tmp_path / "through.html"
    page.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<style>body {{ margin: 0; font: 15px sans-serif; }} {h.HELP_CSS}</style>{h.HELP_HEAD}</head><body>
<div style="padding: 20px 40px">
<div class="block" id="epochs" style="width: 300px; padding: 12px; border: 1px solid #ccc">
<label class="block" style="display: block"><span data-testid="block-info">Epochs</span><input id="ep"></label></div>
<div style="display: flex; flex-wrap: wrap; gap: 24px; margin-top: 16px; align-items: flex-start">
<div class="block" id="seed" style="width: 300px; padding: 12px; border: 1px solid #ccc">
<label class="block" style="display: block"><span data-testid="block-info">Seed</span><input id="sd"></label></div>
<fieldset id="backbone" style="width: 300px; margin: 0; padding: 8px 12px; border: 1px solid #ccc">
<legend>Graph backbone</legend>{radio.format(v="keep", c=" checked", t="Trained")}
{radio.format(v="graph_side", c="", t="Frozen")}{radio.format(v="encoder", c="", t="Encoder only")}
{radio.format(v="head", c="", t="Head only")}</fieldset></div></div>
{registry.texts_html()}</body></html>""", encoding="utf-8")
    script = tmp_path / "through.cjs"
    script.write_text(_THROUGH_JS, encoding="utf-8")
    result = subprocess.run([node, str(script), str(page)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert set(seen) == {"1850x960", "420x800"}
    for size, s in seen.items():
        # The scene T08 met: focus on Seed's ⓘ, its preview a tooltip over a radio.
        assert s["tabbed"]["focus"] == "seed-i" and s["tabbed"]["shown"], (size, s)
        assert s["tabbed"]["role"] == "tooltip" and s["tabbed"]["checked"] == ["keep"], (size, s)
        assert s["target"], (size, s)
        # One click: that radio chosen, focus on it, the preview gone.
        assert s["clicked"]["checked"] == [s["target"]["value"]], (size, s)
        assert s["clicked"]["focus"] == s["target"]["value"] and not s["clicked"]["shown"], (size, s)
        # Its focus keeps it while the pointer crosses it; focus leaving closes it.
        assert s["again"]["shown"] and s["crossed"]["shown"] and s["crossed"]["focus"] == "seed-i", (size, s)
        assert not s["left"]["shown"], (size, s)


#: The review of 9c4580f (T08-flow-1). A field's ⓘ focused with the page
#: scrolled part way, its preview taller than the window (its height capped,
#: its text scrolling inside); the pointer moved over it, then the wheel turned
#: over it: by a little, to the text's end, and once more. Each read once the
#: text or the page moved.
_WHEEL_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1850, 960], [420, 360]]) {
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
    await page.goto('file://' + process.argv[2]);
    await page.waitForSelector('#fld button.shep-i');
    await page.evaluate(() => window.scrollTo(0, 520));
    await page.focus('#fld button.shep-i');
    await page.waitForFunction(() => {
      const b = document.getElementById('shep-help-bubble');
      return !!b && !b.hidden;
    });
    const state = () => page.evaluate(() => {
      const b = document.getElementById('shep-help-bubble'), bd = b.querySelector('.shep-bubble-bd');
      const r = b.getBoundingClientRect();
      return { shown: !b.hidden && b.checkVisibility({ visibilityProperty: true }),
               role: b.getAttribute('role'), through: getComputedStyle(b).pointerEvents === 'none',
               box: { left: r.left, top: r.top, width: r.width, height: r.height },
               text: bd.scrollTop, room: bd.scrollHeight - bd.clientHeight, page: window.scrollY,
               focus: document.activeElement.matches('#fld button.shep-i') };
    });
    const wheel = async (dy) => {
      const before = await state();
      await page.mouse.wheel(0, dy);
      await page.waitForFunction((b) => {
        const bd = document.querySelector('#shep-help-bubble .shep-bubble-bd');
        return bd.scrollTop !== b.text || window.scrollY !== b.page;
      }, before);
      await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
      return state();
    };
    const opened = await state();
    const box = opened.box;
    await page.mouse.move(box.left + box.width / 2, box.top + box.height / 2, { steps: 4 });
    const over = await state();
    const wheeled = await wheel(120);
    const seen = { opened, over, wheeled };
    // On to the text's end and past it, where the wheel moved the text.
    if (wheeled.text > opened.text) {
      seen.end = await wheel(Math.ceil(opened.room) + 400);
      seen.past = await wheel(120);
    }
    out[`${w}x${h}`] = seen;
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_wheel_over_a_focus_preview_scrolls_its_text_though_clicks_go_through(tmp_path):
    """The review of 9c4580f (T08-flow-1): a preview focus opened lets the
    reader's pointer through once it moves over it (its pointer events off),
    so the wheel turned over a preview whose height was capped scrolled the
    page behind it, never its text (420 x 360: the text at 0, the page 520 →
    640), which the mouse could then not reach. Now, at 1850 x 960 and
    420 x 360, a field's preview taller than the window, the pointer over it:
    it still lets the pointer through (clicks reach the control under it),
    the wheel over it scrolls its text and not the page, to the text's end,
    and the page past that end (the wheel is never held). Fails on 4b5cee6
    (the page scrolled) and on e0e4237 (the preview took the pointer)."""
    found = node_with_playwright()
    if found is None:
        pytest.skip("Node with Playwright is not installed")
    node, env = found
    registry = h.HelpRegistry()
    registry.field("fld", "Epochs", "How many passes over the training pairs.\n\n" + "\n".join(
        f"- Item {i}: a long line of explanation that wraps over two lines in the bubble."
        for i in range(40)))
    page = tmp_path / "wheel.html"
    page.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<style>body {{ margin: 0; font: 15px sans-serif; }} {h.HELP_CSS}</style>{h.HELP_HEAD}</head><body>
<div style="height: 600px"></div><div style="padding: 0 40px">
<div class="block" id="fld" style="width: 300px; padding: 16px; border: 1px solid #ccc">
<label class="block" style="display: block"><span data-testid="block-info">Epochs</span><input></label>
</div></div><div style="height: 1400px"></div>
{registry.texts_html()}</body></html>""", encoding="utf-8")
    script = tmp_path / "wheel.cjs"
    script.write_text(_WHEEL_JS, encoding="utf-8")
    result = subprocess.run([node, str(script), str(page)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert set(seen) == {"1850x960", "420x360"}
    for size, s in seen.items():
        opened, over = s["opened"], s["over"]
        # A focus preview whose text scrolls inside, more of it than one turn.
        assert opened["shown"] and opened["role"] == "tooltip" and opened["focus"], (size, s)
        assert opened["room"] > 240 and opened["text"] == 0, (size, s)
        # The pointer over it: it lets the pointer through.
        assert over["shown"] and over["through"], (size, s)
        # The wheel moves its text, not the page, and the pointer still goes through.
        assert s["wheeled"]["text"] > 0 and s["wheeled"]["page"] == opened["page"], (size, s)
        assert s["wheeled"]["shown"] and s["wheeled"]["through"] and s["wheeled"]["focus"], (size, s)
        assert abs(s["end"]["text"] - opened["room"]) <= 1 and s["end"]["page"] == opened["page"], \
            (size, s)
        # Past the text's end the page scrolls on.
        assert s["past"]["page"] > opened["page"] and s["past"]["focus"], (size, s)


#: T08-flow-3. The served dashboard at each window: each page opened (its
#: accordions open); for each checkbox given (one with field help) drawn on
#: it: its row's focusable parts (its ⓘ and its box) in the order drawn (left
#: to right), and in the order Tab reaches them from the control before them.
#: A page is read once its panel is shown, the page's own requests are
#: answered (`QUEUE_JS`) and every checkbox the page draws (`expected`, by
#: the page each one is built on; Gradio mounts an accordion's fields only
#: when it is first opened) is drawn with its ⓘ, never by a fixed time.
_CHECKBOX_ORDER_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const base = process.argv[2], ids = JSON.parse(process.argv[3]), expected = JSON.parse(process.argv[4]);
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1850, 960], [960, 1080]]) {
    const page = await browser.newPage({ viewport: { width: w, height: h } });
    await page.addInitScript(QUEUE_JS);
    const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.idle());
    await page.goto(base);
    await page.waitForSelector('#dx_explanations button.shep-i');
    const seen = {};
    for (const id of ['clinical', 'training', 'finetune', 'test', 'settings']) {
      const level1 = id === 'clinical' || id === 'settings';
      if (!level1 && !(await page.$('#shep-tab-research-button[aria-selected="true"]'))) {
        await page.click('#shep-tab-research-button', { force: true });
      }
      const panel = level1 ? `#shep-tab-${id}` : `#shep-page-${id}`;
      await page.click(panel + '-button', { force: true });
      await page.waitForSelector(panel + '-button[aria-selected="true"]');
      // The page drawn: its panel shown and its own requests answered.
      await page.waitForSelector(panel, { state: 'visible' });
      await idle();
      // Training's Parameters, folded on a narrow window, and every accordion.
      for (const fold of await page.$$(panel + ' button.shep-train-fold[aria-expanded="false"]')) {
        if (await fold.isVisible()) { await fold.click(); }
      }
      for (const acc of await page.$$(panel + ' button.label-wrap:not(.open)')) {
        if (await acc.isVisible()) { await acc.click(); }
      }
      await idle();
      // Every checkbox this page draws, drawn, its ⓘ added.
      await page.waitForFunction((mine) => mine.every((k) => {
        const b = document.getElementById(k), box = b && b.querySelector('input[type=checkbox]');
        return !!box && box.getClientRects().length > 0 && !!b.querySelector('button.shep-i');
      }), expected[id] || []);
      const rows = await page.evaluate((ids) => {
        const tabbable = (e) => e.tabIndex >= 0 && !e.disabled && e.getClientRects().length > 0
          && getComputedStyle(e).visibility === 'visible';
        const all = [...document.querySelectorAll('a[href], button, input, select, textarea, summary, [tabindex]')].filter(tabbable);
        return ids.map((k) => {
          const b = document.getElementById(k), box = b && b.querySelector('input[type=checkbox]');
          if (!box || !box.getClientRects().length) { return null; }
          const parts = [b.querySelector('button.shep-i'), box].filter(tabbable);
          const drawn = parts.slice().sort((p, q) => p.getBoundingClientRect().left - q.getBoundingClientRect().left)
            .map((p) => (p === box ? 'box' : 'i'));
          const first = Math.min(...parts.map((p) => all.indexOf(p)));
          if (first > 0) { all[first - 1].setAttribute('data-before', k); }
          const mid = parts.map((p) => { const r = p.getBoundingClientRect(); return r.top + r.height / 2; });
          return { k, drawn, disabled: box.disabled, oneLine: Math.max(...mid) - Math.min(...mid) < 12 };
        }).filter(Boolean);
      }, ids);
      for (const row of rows) {
        await page.focus(`[data-before="${row.k}"]`);
        const order = [];
        for (let n = 0; n < row.drawn.length; n++) {
          await page.keyboard.press('Tab');
          order.push(await page.evaluate((k) => {
            const a = document.activeElement, b = document.getElementById(k);
            return !b.contains(a) ? 'out' : a.matches('button.shep-i') ? 'i' : a.type === 'checkbox' ? 'box' : a.tagName;
          }, row.k));
        }
        await page.evaluate((k) => document.querySelector(`[data-before="${k}"]`).removeAttribute('data-before'), row.k);
        seen[row.k] = { page: id, drawn: row.drawn, tab: order, disabled: row.disabled, oneLine: row.oneLine };
      }
    }
    out[`${w}x${h}`] = seen;
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_tab_reaches_each_checkboxs_i_and_box_in_the_order_they_are_drawn(tmp_path, monkeypatch):
    """T08-flow-3 (WCAG 2.4.3): Diagnosis › Options draws "Explanations ⓘ ☑"
    (R-24: the words in the key column, the box in the control column with
    Top-K's box), but Gradio puts a checkbox's box before its words and the
    shared help puts the ⓘ after the words, so Tab reached the box before the
    ⓘ. In the served dashboard, at 1850 x 960 and 960 x 1080, every checkbox
    with field help drawn on Diagnosis, Training (its Parameters unfolded and
    its accordions open), Fine-tune, Test and Settings: Tab from the control
    before it reaches its ⓘ and its box in the order they are drawn
    (Diagnosis: ⓘ then box; the others, drawn box, words, ⓘ: box then ⓘ).
    Training's Continue drawer's Mixed precision is drawn only with a run to
    continue; it is the same setting field as the base form's, which is
    walked."""
    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, run_node
    from tests.fixtures.operator_access import operator_mode
    from tests.fixtures.served_page import served

    operator_mode(monkeypatch)
    demo = create_gradio_app()
    texts = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "shep-help-texts")
    targets = set(re.findall(r'data-help-target="([^"]+)"', str(texts.value)))
    boxes = {b.elem_id: b for b in demo.blocks.values() if isinstance(b, gr.Checkbox) and b.elem_id in targets}
    helped = set(boxes)
    assert {"dx_explanations", "use_amp", "ft_use_amp", "torch_compile"} <= helped, helped

    def page_of(block):
        """The page (its tab or research page id, as the script names it)
        the block is built on."""
        while block is not None:
            block = getattr(block, "parent", None)
            if isinstance(block, gr.Tab) and block.elem_id.startswith(("shep-page-", "shep-tab-")):
                return block.elem_id.split("-", 2)[2]
        raise AssertionError("not on a page")

    expected: dict[str, list[str]] = {}
    for k in sorted(helped - {"branch_use_amp"}):
        expected.setdefault(page_of(boxes[k]), []).append(k)
    assert set(expected) == {"clinical", "training", "finetune", "test", "settings"}, expected
    with served(demo, **mount_options()) as base:
        seen = run_node(_CHECKBOX_ORDER_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")], json.dumps(sorted(helped)), json.dumps(expected),
                        timeout=600)
    assert set(seen) == {"1850x960", "960x1080"}
    for size, rows in seen.items():
        assert set(rows) == helped - {"branch_use_amp"}, (size, sorted(helped - set(rows)))
        for k, row in rows.items():
            assert row["oneLine"] and set(row["drawn"]) == {"i", "box"}, (size, k, row)
            assert row["tab"] == row["drawn"], (size, k, row)
        assert rows["dx_explanations"]["drawn"] == ["i", "box"], (size, rows["dx_explanations"])


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


#: Where one sentence ends and the next begins (the budget's own split).
_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=\S)")


def _words(text: str) -> int:
    return sum(len(f"{line.term or ''} {line.text}".split()) for line in h.help_lines(text)
               if line.kind != "ref")


def test_the_test_and_models_texts_are_an_answer_and_the_budgets_items_at_most(built):
    """R-74: the Test and Models (Results included) texts are one answer
    sentence and a few items, about 60 words (70 at most), their codes in the
    Ref line. How many items is one rule, in one place: the text budget every
    registered text meets (`help.TEXT_BUDGET`, the skill's §7: none or 2-4).
    R-74 said three before that budget existed; the sweep wrote these texts
    to the budget (lanes L3 and L4), and a page's own count beside it would be
    a second rule for the same thing."""
    _demo, registry, _texts = built
    keys = [e.key for e in registry.entries()
            if e.key.startswith(("shep-test-", "test-", "models-"))]
    assert len(keys) >= 24
    budget = h.TEXT_BUDGET
    for entry in registry.entries():
        if entry.key in keys:
            lines = h.help_lines(entry.text)
            assert len(_SENTENCES.split(lines[0].text)) == 1, (entry.key, lines[0].text)
            items = sum(line.kind == "item" for line in lines)
            assert items == 0 or budget.items_min <= items <= budget.items, (entry.key, items)
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


#: The card's ⓘ pinned with focus on it; the card drawn again (a new ⓘ of
#: the same help); then focus on the page and the card drawn again.
_REPLACED_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1850, height: 960 } });
  await page.goto('file://' + process.argv[2]);
  const I = '#card button.shep-i';
  await page.waitForSelector(I);
  await page.click(I);
  const redraw = () => page.evaluate(() => {
    const card = document.getElementById('card');
    const old = card.querySelector('button.shep-i');
    card.innerHTML = card.innerHTML;
    return old.isConnected;
  });
  const read = () => page.evaluate(() => {
    const b = document.getElementById('shep-help-bubble'), a = document.activeElement;
    const i = document.querySelector('#card button.shep-i');
    return { open: !!b && !b.hidden, role: b && b.getAttribute('role'),
             focus: a === i ? 'its i' : (a === document.body ? 'page' : a.tagName),
             expanded: i.getAttribute('aria-expanded') };
  });
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  const before = await read();
  const oldKept = await redraw();
  await frames();
  const after = await read();
  await page.keyboard.press('Escape');
  await frames();
  const escaped = await read();
  // Focus on the page (a click on text), the bubble pinned again: a drawing
  // never takes the page's focus.
  await page.click(I);
  await page.mouse.click(5, 5);
  await page.click(I);
  await page.evaluate(() => document.activeElement.blur());
  await frames();
  await redraw();
  await frames();
  const unfocused = await read();
  console.log(JSON.stringify({ before, oldKept, after, escaped, unfocused }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_focus_on_a_pinned_i_follows_it_when_a_drawing_replaces_it(tmp_path):
    """Verification of fcc3dea (V-RB1-2): a pinned bubble whose ⓘ a drawing
    replaced (Training's dataset chip at a run's start and end) found the new
    ⓘ and kept it open and expanded, but the reader's focus fell to the page,
    so Escape no longer reached it; the shared disclosure keeper gives focus
    back in the same case. Focus now goes to the new ⓘ, unscrolled, and
    Escape closes the bubble; focus the reader had moved to the page stays
    there. Fails on fcc3dea (focus on the page)."""
    from tests.fixtures.browser import node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    seen = run_node(_REPLACED_JS, tmp_path, str(_page(tmp_path)))
    assert seen["before"] == {"open": True, "role": "dialog", "focus": "its i",
                              "expanded": "true"}, seen
    assert seen["oldKept"] is False
    assert seen["after"] == {"open": True, "role": "dialog", "focus": "its i",
                             "expanded": "true"}, seen
    assert seen["escaped"]["open"] is False and seen["escaped"]["focus"] == "its i", seen
    assert seen["unfocused"]["focus"] == "page" and seen["unfocused"]["open"] is True, seen
