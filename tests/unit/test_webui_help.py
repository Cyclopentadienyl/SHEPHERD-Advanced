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
- **the browser side** fills the bubble with ``textContent`` and is given to
  the mount's ``head`` (Gradio 6 drops a head given to ``gr.Blocks``); in a
  real headless browser (Node and Playwright, skipped without): a hover
  previews after a delay as a tooltip, a click pins a dialog named by its
  title, the pinned bubble never covers the field it explains (beside, then
  above, then below) at a wide and a narrow window, its width stays within
  the budget, the ⓘ's hit target is at least 24 × 24 px, and a hostile text
  shows as literal text.

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
    # The bubble is filled with textContent: no help text is parsed as HTML.
    assert "textContent = textOf(btn)" in h.HELP_HEAD
    assert "innerHTML = textOf" not in h.HELP_HEAD
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


# --------------------------------------------------------------------------- in a real browser
HOSTILE = "<img src=x onerror=window.__pwned=true> & more"
HOST = Path(__file__).resolve().parent.parent / "fixtures" / "help_host.cjs"


def _page(tmp_path: Path) -> Path:
    registry = h.HelpRegistry()
    # As Gradio 6 draws a field: the component's block (its elem_id), its
    # <label> a "block" too, the title in it.
    registry.field("fld", "Epochs", "How many passes. " + HOSTILE)
    registry.topic("kv-help", "A key", "What the key means. " + HOSTILE)
    registry.topic("row-help", "A row", "What the row says.")
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<style>:root {{ --shep-accent: #5558e3; --shep-accent-50: #eef0ff; --shep-accent-text: #3a3dbd;
--shep-border: #e1e4ea; --shep-border-2: #cdd2da; --shep-text: #1d2230; --shep-muted: #5b6270; }}
body {{ margin: 0; font: 15px sans-serif; }}
.shep-kv {{ display: grid; grid-template-columns: 10em 1fr; margin: 24px 40px; max-width: 640px; }}
{h.HELP_CSS}</style>{h.HELP_HEAD}</head><body>
<div class="block" id="fld" style="margin: 40px; width: 300px; padding: 16px; border: 1px solid #ccc">
<label class="block" style="display: block"><span data-testid="block-info">Epochs</span><input>
</label></div>
<div class="shep-kv"><span class="shep-k">A key{h.help_button("kv-help", "A key")}</span>
<span class="shep-v">its value, long enough to fill the row</span></div>
<table id="tbl" style="margin: 24px 40px; width: calc(100% - 80px)"><tr>
<td>A row{h.help_button("row-help", "A row")}</td><td>its cells</td></tr></table>
<div style="height: 1400px"></div>
{registry.texts_html()}</body></html>"""
    path = tmp_path / "help.html"
    path.write_text(page, encoding="utf-8")
    return path


def test_the_pinned_bubble_never_covers_its_field_and_shows_text_as_text(tmp_path):
    """In a real browser, at the deployment window (1850 × 960) and a narrow
    one (420 × 800), for a field's ⓘ, a key-value row's and a table row's: a
    hover opens nothing before the delay and a tooltip after it; a click pins
    a dialog named by the help's title; the bubble's rect does not intersect
    the field's (measured here, by the test's own rule) and is not under the
    field's centre (``document.elementsFromPoint``); it is at most 320 px
    wide; the ⓘ's hit target is at least 24 × 24; a hostile text is literal
    text and runs nothing; Esc closes it back to a tooltip."""
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
        for which, seen in targets.items():
            where = f"{size} {which}"
            assert seen["hit"]["width"] >= 24 and seen["hit"]["height"] >= 24, where
            assert seen["early"] is False and seen["hover"] == "tooltip", where
            pinned = seen["pinned"]
            assert not pinned["hidden"] and pinned["role"] == "dialog", where
            assert pinned["labelledby"] and pinned["title"] in ("Epochs", "A key", "A row"), where
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
    a state set at runtime, and a field's format, unit or "off", set at build
    time only on the Training page's base form, its Continue drawer's
    Branch fields and the Fine-tune page's fields, the one field of each
    setting (`FIELD_FORMATS`, by its setting or its model spec field), and
    when each of Settings' two fields applies (`runtime_settings.FIELD_INFO`);
    no ordinary help is kept there."""
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
