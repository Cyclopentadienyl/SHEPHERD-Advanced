"""
The dashboard's shell (M5-2, UI-A): navigation, the page contract, the head.
============================================================================
`src/webui/shell.py`, `src/webui/app.py`. What these pin:

- **two navigation levels**: "Diagnosis (clinical)" first and selected,
  "Research (pre-clinical)", "⚙ Settings" aligned right; under Research,
  Datasets · Training · Fine-tune · Test · Models with Training selected; both levels wrap,
  never into an overflow menu; each page in its own tab with its elem_id;
- **the page contract**: each builder is called with the one context; Test
  and Models hold the one-line temporary body until their builders exist;
  a cross-link from any page, Diagnosis included, switches both levels
  (`go`); the shared timers are created outside ``gr.Tabs``; a research page
  refreshes when it is shown, and only it: its own tab, or Research while it
  is the page shown (`on_page_open`, the page tokens), once per opening even
  when both selects fire; ``ctx.location`` says
  which page is shown; every gated act is followed by the chips' read;
- **the browser title** follows the location;
- **the head**: the shell's and every page's style sheet (`PAGE_CSS`) go in as
  one ``<style>`` element, since Gradio 6.30 keeps only ``.contain``-prefixed
  selectors inside an ``@media`` rule of the mount's ``css``; the theme's text
  sizes are never under 14 px; the page is always light (Gradio's dark class is
  kept off); Gradio's footer is off; the segmented control's selected option
  is dark text on white;
- the access log keeps the no-queue event path quiet (the chips' timer);
- **one choice, one request** (UI-11): a select's choice is wired through
  `shell.picked` everywhere in the dashboard, and in a browser one pick of a
  Dropdown, by pointer or by keys, is one request and the page's own write of
  its value is none;
- **the shared UI measurement check** (D08, INT-AUDIT; `tests/fixtures/browser.py`):
  the service on the demo workspace, one browser through every page at the
  representative sizes and back (`ui_walk`): the layout decision at each, and
  the check's rules (no sideways page scroll, one scroll region per area,
  nothing over a focused control, text and type scale, borders and focus at
  3:1, the stage hint plain text inside the research pages' row and off its
  tabs, one " · " per option);
- **the stage hint** (the stage-hint package of 2026-11-21, N1-N7): one
  mapping of the research pages to their stage, the hint it feeds after the
  research tablist, the hint per page and along every path that changes the
  page (a click, keys, a cross-link, a reload) whatever job runs, none on
  Diagnosis or Settings, a hovered tab never read as selected, and the hint
  inside the row and off the tabs in a narrow window and with larger text.

Module: tests/unit/test_webui_shell.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from src.webui import shell  # noqa: E402


def _built(pages=None):
    from src.webui.app import PAGES

    with gr.Blocks() as demo:
        ctx = shell.build_shell(demo, pages or PAGES)
    return demo, ctx


@pytest.fixture(scope="module")
def built():
    return _built()


def _tabs(demo):
    return [b for b in demo.blocks.values() if isinstance(b, gr.Tab)]


def test_two_levels_in_order_with_their_defaults(built):
    demo, ctx = built
    assert ctx.nav.selected == shell.CLINICAL and ctx.research_nav.selected == shell.TRAINING
    assert ctx.nav.overflow_behavior == "wrap" and ctx.research_nav.overflow_behavior == "wrap"
    outer = [t for t in _tabs(demo) if t.elem_id and t.elem_id.startswith("shep-tab-")]
    assert [(t.label, t.id) for t in outer] == [
        ("Diagnosis (clinical)", "clinical"), ("Research (pre-clinical)", "research"),
        ("⚙ Settings", "settings")]
    assert [t.alignment for t in outer] == ["left", "left", "right"]
    inner = [t for t in _tabs(demo) if t.elem_id and t.elem_id.startswith("shep-page-")]
    assert [(t.label, t.id, t.elem_id) for t in inner] == [
        ("Datasets", "datasets", "shep-page-datasets"), ("Training", "training", "shep-page-training"),
        ("Fine-tune", "finetune", "shep-page-finetune"),
        ("Test", "test", "shep-page-test"), ("Models", "models", "shep-page-models")]
    assert shell.RESEARCH_TABLIST_LABEL == (
        "Research pages: Prepare (Datasets, Training, Fine-tune), Evaluate (Test, Models)")


def test_the_research_tabs_sit_inside_research(built):
    demo, ctx = built
    config = demo.get_config_file()
    research_tab = ctx.tabs[shell.RESEARCH]

    def find(node, wanted):
        if node["id"] == wanted:
            return node
        for child in node.get("children", []):
            hit = find(child, wanted)
            if hit:
                return hit
        return None

    research = find(config["layout"], research_tab._id)
    assert find(research, ctx.research_nav._id) is not None


def test_every_page_needs_its_builder_and_no_placeholder_is_left():
    """M5-3 review ARCH-6: every page is built (UI-D shipped Test and Models),
    so a missing builder is a KeyError when the shell is built, as Diagnosis'
    and Settings' are, never a "being built in M5" body shown silently."""
    from src.webui.app import PAGES

    for missing in (shell.TEST, shell.MODELS):
        with pytest.raises(KeyError):
            _built({k: v for k, v in PAGES.items() if k != missing})
    assert not hasattr(shell, "TEMPORARY_BODY")
    demo, _ctx = _built(dict(PAGES))
    assert not [b for b in demo.blocks.values()
                if isinstance(b, gr.Markdown) and "being built" in str(b.value)]


def test_every_builder_gets_the_one_context_and_can_cross_link():
    """Diagnosis is built first; it can still link to a research page, both
    levels at once."""
    seen = {}

    def builder(name):
        def build(ctx):
            seen[name] = ctx
            button = gr.Button(f"to models from {name}")
            button.click(lambda: ctx.go(shell.MODELS), None, ctx.nav_outputs)
        return build

    pages = {page: builder(page) for page in (shell.CLINICAL, shell.DATASETS, shell.TRAINING,
                                              shell.FINETUNE, shell.TEST, shell.MODELS,
                                              shell.SETTINGS)}
    demo, ctx = _built(pages)
    assert set(seen) == set(pages) and all(c is ctx for c in seen.values())
    assert ctx.nav_outputs == [ctx.nav, ctx.research_nav, ctx.research_page]
    outer, inner, shown = ctx.go(shell.MODELS)
    assert outer.selected == shell.RESEARCH and inner.selected == shell.MODELS
    assert shown == shell.MODELS          # Research's own refresh is then Models'
    outer, inner, shown = ctx.go(shell.SETTINGS)
    assert outer.selected == shell.SETTINGS and inner == gr.update() and shown == gr.update()
    assert len(ctx.no_nav()) == len(ctx.nav_outputs)
    with pytest.raises(ValueError):
        ctx.go("nowhere")
    links = [fn for fn in demo.fns.values() if fn.outputs == ctx.nav_outputs]
    # Each page's link, and the shell's own: a load opens the page kept (UI-14).
    assert len(links) == 7 + 1 and [fn.name for fn in links].count("_restore") == 1


def test_the_shared_timers_are_outside_the_tabs(built):
    demo, ctx = built
    layout = demo.get_config_file()["layout"]

    def ids(node):
        yield node["id"]
        for child in node.get("children", []):
            yield from ids(child)

    nav_node = next(n for n in _walk(layout) if n["id"] == ctx.nav._id)
    inside = set(ids(nav_node))
    assert ctx.timer_fast._id not in inside and ctx.timer_slow._id not in inside
    assert (ctx.timer_fast.value, ctx.timer_slow.value) == (shell.FAST_SECONDS, shell.SLOW_SECONDS)


def _walk(node):
    yield node
    for child in node.get("children", []):
        yield from _walk(child)


def test_a_research_page_refreshes_only_when_it_is_the_page_shown(built):
    """Returning to Research refreshes the research page shown, never the
    other four (Models' refresh opens every checkpoint)."""
    demo, ctx = built
    tokens = [ctx.page_tokens[p] for p in shell.RESEARCH_PAGES]
    (research,) = [fn for fn in demo.fns.values()
                   if (ctx.tabs[shell.RESEARCH]._id, "select") in fn.targets]
    # Each writer is first given the page shown as it was sent and the time of
    # the last write (`_written`); its own inputs and outputs follow.
    sent, written_at = research.inputs[:2]
    assert research.inputs[2:] == [ctx.location, ctx.research_page, *tokens]
    # queued: Gradio 6.30 dispatches a State's .change only from a queued answer
    assert research.outputs == [ctx.location, *tokens, written_at]
    assert research.queue is not False
    location, *bumped, at = research.fn("datasets 9", 0.0, "clinical", "datasets", 0, 4, 0, 0, 0)
    assert location == "datasets" and at == 9.0 and bumped == [1, gr.update(), gr.update(),
                                                               gr.update(), gr.update()]
    # nothing chosen: Training
    location, *bumped, _ = research.fn("training 9", 0.0, "settings", None, 0, 4, 0, 0, 0)
    assert location == "training" and bumped == [gr.update(), 5, gr.update(), gr.update(),
                                                  gr.update()]
    location, *bumped, _ = research.fn("finetune 9", 0.0, "clinical", "finetune", 0, 0, 2, 0, 0)
    assert location == "finetune" and bumped[2] == 3
    for page in shell.RESEARCH_PAGES:
        (own,) = [fn for fn in demo.fns.values() if (ctx.tabs[page]._id, "select") in fn.targets]
        assert own.inputs == [sent, written_at, ctx.location, ctx.page_tokens[page]]
        assert own.outputs == [ctx.location, ctx.research_page, ctx.page_tokens[page], written_at]
        assert own.fn(f"{page} 9", 0.0, "clinical", 2) == (page, page, 3, 9.0)
        assert own.queue is not False
    # Datasets' refresh hangs on its token, not on any tab's select.
    from src.webui.components import datasets_page

    refresh = [fn for fn in demo.fns.values() if fn.name == "_refresh"
               and (ctx.page_tokens[shell.DATASETS]._id, "change") in fn.targets]
    assert len(refresh) == 1
    assert not [fn for fn in demo.fns.values() if fn.name == "_refresh"
                and any(t[1] == "select" for t in fn.targets)]
    assert datasets_page.create_datasets_page in demo_builders()


def test_a_research_page_refreshes_once_per_opening(built):
    """The first opening of Research, and a cross-link that switches both
    levels, fire Research's select and the inner tab's (Gradio mounts or
    switches it): whichever runs first refreshes the page and sets the
    location; the other finds the page shown and leaves its token. The two
    run one at a time, in order, so the second sees what the first set."""
    demo, ctx = built
    (research,) = [fn for fn in demo.fns.values()
                   if (ctx.tabs[shell.RESEARCH]._id, "select") in fn.targets]
    inner = {page: next(fn for fn in demo.fns.values()
                        if (ctx.tabs[page]._id, "select") in fn.targets)
             for page in shell.RESEARCH_PAGES}
    # first opening: Research's select, then Training's mount select
    location, *bumped, at = research.fn("training 5", 0.0, "clinical", "training", 0, 0, 0, 0, 0)
    assert location == "training" and bumped[1] == 1
    assert inner["training"].fn("training 6", at, location, 1) == (
        "training", "training", gr.update(), 6.0)
    # the other order: the inner select first, then Research's
    page, _, token, at = inner["models"].fn("models 5", 0.0, "clinical", 0)
    assert (page, token) == ("models", 1)
    assert research.fn("models 5", at, page, "models", 0, 0, 0, 0, 1) == (
        "models", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 5.0)
    # within Research, another page's tab still refreshes it
    assert inner["test"].fn("test 7", 6.0, "training", 4) == ("test", "test", 5, 7.0)
    # returning to Research from another page refreshes the page shown
    assert research.fn("test 8", 7.0, "settings", "test", 0, 0, 0, 5, 0)[4] == 6
    for fn in (research, *inner.values()):
        assert fn.concurrency_id == shell.OPEN_PAGE_CONCURRENCY and fn.concurrency_limit == 1


def demo_builders():
    from src.webui.app import PAGES

    return set(PAGES.values())


def test_the_location_follows_the_page_shown(built):
    """Every writer of the location (Diagnosis', Settings', Research's and
    each research tab's select) is queued in one group of one, each select
    sent once (UI-14, B46-mount: Diagnosis' and Settings' ran without the
    queue, so the select Diagnosis' tab fires when it mounts could write the
    location after a page opened later)."""
    demo, ctx = built
    for tab in (shell.CLINICAL, shell.SETTINGS):
        shown = [fn for fn in demo.fns.values() if (ctx.tabs[tab]._id, "select") in fn.targets
                 and fn.outputs[0] is ctx.location]
        assert len(shown) == 1 and shown[0].fn(f"{tab} 3.5", 0.0) == (tab, 3.5)
    writers = [fn for fn in demo.fns.values() if ctx.location in fn.outputs]
    assert len(writers) == 2 + 1 + len(shell.RESEARCH_PAGES)
    for fn in writers:
        assert fn.queue is not False and fn.concurrency_id == shell.OPEN_PAGE_CONCURRENCY
        assert fn.concurrency_limit == 1 and fn.trigger_mode == "multiple", fn.name
        # V-RB1-1: each is given the page shown as it was sent, and when.
        sent, written_at = fn.inputs[:2]
        assert fn.js == shell.SENT_JS and sent.visible is False
        assert fn.outputs[-1] is written_at and written_at is writers[0].inputs[1]


def test_the_brand_is_the_pages_one_level_one_heading():
    """Verification of fcc3dea (V-AUD-1, axe page-has-heading-one on every
    page): the shell's title was a div and the panels' titles h2, so no page
    had a level-1 heading. The brand is the one h1, drawn as before (no
    heading size or margin of its own); nothing else in the dashboard draws
    one. Fails on fcc3dea."""
    from pathlib import Path

    assert shell._brand_html() == f'<h1 class="shep-brand">{shell.BRAND}</h1>'
    webui = Path(shell.__file__).resolve().parent
    drawn = sorted(str(path.relative_to(webui)) for path in webui.rglob("*.py")
                   if "<h1" in path.read_text(encoding="utf-8"))
    assert drawn == ["shell.py"], drawn
    from src.webui.style import SHELL_CSS

    assert "#shep-brand h1.shep-brand { margin: 0; padding: 0;" in SHELL_CSS


def test_a_location_writer_writes_only_for_the_page_shown_as_it_was_sent(built):
    """Verification of fcc3dea (V-RB1-1): after a reload of Research ›
    Training, Diagnosis' tab sent the select it sends as it mounts after the
    restore had opened Training, so the session's location ended on Diagnosis
    while Training was shown, and every Training poll idled. Every writer now
    writes only when the page shown as it was sent is its own, and never over
    a write sent after it; with no head script it writes as asked. Fails on
    fcc3dea, whose writers take no such input."""
    demo, ctx = built
    clinical, research, models = (
        next(fn for fn in demo.fns.values()
             if (ctx.tabs[tab]._id, "select") in fn.targets and ctx.location in fn.outputs)
        for tab in (shell.CLINICAL, shell.RESEARCH, shell.MODELS))
    nothing = (gr.update(), gr.update())
    # Sent by the mount while Training is shown: nothing written.
    assert clinical.fn("training 3110.4", 2997.0) == nothing
    # Sent before a write applied already (it arrived after it): nothing.
    assert clinical.fn("clinical 1400.0", 2997.0) == nothing
    assert models.fn("models 10", 20.0, "clinical", 0) == (gr.update(),) * 4
    assert research.fn("clinical 30", 0.0, "clinical", "models", 0, 0, 0, 0, 0) == (
        gr.update(),) * 7
    # Its own page shown, sent after the last write: written, with its time.
    assert clinical.fn("clinical 3200", 2997.0) == ("clinical", 3200.0)
    # No head script: written as asked.
    assert clinical.fn("", 2997.0) == ("clinical", gr.update())
    assert "sent: sent" in shell.SHELL_HEAD and "performance.now()" in shell.SHELL_HEAD


def test_a_load_opens_the_page_this_tab_kept_and_nothing_else(built):
    """UI-14 (B32-UI8): the shell built level 1 with Diagnosis selected and
    kept no page, so every reload landed on Diagnosis. The page load hands
    the page the head script kept (`KEPT_PAGE`) to `_restore`, which opens a
    research page or Settings as a cross-link does and leaves Diagnosis,
    nothing kept or a value no page has."""
    demo, ctx = built
    (restore,) = [fn for fn in demo.fns.values() if fn.name == "_restore"]
    assert ("load" in {t[1] for t in restore.targets})
    assert restore.outputs == ctx.nav_outputs and "ShepShell.kept" in restore.js
    assert len(restore.inputs) == 1 and restore.inputs[0].visible is False
    nav, inner, page = restore.fn("models")
    assert (nav.selected, inner.selected, page) == ("research", "models", "models")
    assert restore.fn("settings")[0].selected == "settings"
    for kept in ("clinical", "", None, "nowhere", "<script>", ["models"]):
        assert restore.fn(kept) == ctx.no_nav(), kept
    # What is kept, and where: the page id only, for this browser tab.
    assert f"var KEPT = '{shell.KEPT_PAGE}'" in shell.SHELL_HEAD
    assert "sessionStorage" in shell.SHELL_HEAD and "localStorage" not in shell.SHELL_HEAD
    assert "restored: function" in shell.SHELL_HEAD


def test_on_page_open_runs_on_its_own_tab_for_a_first_level_page():
    seen = {}

    def clinical(ctx):
        box = gr.Textbox()
        ctx.on_page_open(shell.CLINICAL, {"fn": lambda: "x", "outputs": [box]})
        seen["ctx"] = ctx

    pages = {**{page: (lambda ctx: None) for page in (*shell.RESEARCH_PAGES, shell.SETTINGS)},
             shell.CLINICAL: clinical}
    demo, ctx = _built(pages)
    assert [fn for fn in demo.fns.values() if (ctx.tabs[shell.CLINICAL]._id, "select") in fn.targets
            and fn.outputs != [ctx.location]]
    with pytest.raises(ValueError):
        ctx.on_page_open("nowhere", {"fn": lambda: None})


def test_every_gated_act_is_followed_by_the_chips_read(built):
    """Training Start, Continue…, Continue Training, Stop (asking and
    confirming), Check checkpoint and the parent list's Refresh, Fine-tune's
    base list and Start, Diagnosis' Change model… (it lists checkpoints),
    Settings Apply and Restart: each act's chain reads the chips again."""
    from src.webui.components.chips import read_chips

    demo, _ctx = built
    by_id = demo.fns

    def chain_reads_chips(fn):
        current = fn
        for _ in range(4):
            nxt = [d for d in by_id.values() if d.trigger_after == current._id]
            if not nxt:
                return False
            current = nxt[0]
            if current.fn is read_chips:
                return True
        return False

    names = {"_on_start", "_on_stop", "_on_resume", "_on_change_model",
             "_on_apply", "_on_restart", "_on_check_continue", "_on_stop_request",
             "_on_continue_open", "_on_finetune_bases", "_on_finetune_start",
             "_on_finetune_stop_request", "_on_finetune_stop"}
    acts = [fn for fn in by_id.values() if fn.name in names
            or getattr(getattr(fn.fn, "__wrapped__", None), "__name__", None) in names]
    found = {fn.name if fn.name in names else fn.fn.__wrapped__.__name__ for fn in acts}
    assert found == names
    for fn in acts:
        assert chain_reads_chips(fn), fn.name
    # the Continue drawer's parent list Refresh
    from src.webui.components import training_console as tc

    refreshes = [fn for fn in by_id.values() if fn.fn is tc._refresh_checkpoints]
    assert any(chain_reads_chips(fn) for fn in refreshes)


def test_the_title_follows_the_location():
    assert shell.TITLES == {
        "clinical": "Diagnosis · Clinical — SHEPHERD-Advanced",
        "settings": "Settings — SHEPHERD-Advanced",
        "datasets": "Datasets · Research — SHEPHERD-Advanced",
        "training": "Training · Research — SHEPHERD-Advanced",
        "finetune": "Fine-tune · Research — SHEPHERD-Advanced",
        "test": "Test · Research — SHEPHERD-Advanced",
        "models": "Models · Research — SHEPHERD-Advanced",
    }
    from src.webui.components.help import as_json_for_script

    assert as_json_for_script(shell.TITLES) in shell.SHELL_HEAD
    assert as_json_for_script(shell.RESEARCH_TABLIST_LABEL) in shell.SHELL_HEAD
    from src.webui.app import create_gradio_app
    assert create_gradio_app().title == shell.TITLES["clinical"]


def test_the_head_carries_the_style_sheets_and_both_scripts():
    from src.webui.app import PAGE_CSS, dashboard_head, mount_options
    from src.webui.components import datasets_page
    from src.webui.components.help import HELP_CSS, HELP_HEAD
    from src.webui.style import SHELL_CSS

    options = mount_options()
    head = options["head"]
    assert head.startswith("<style>") and SHELL_CSS in head and HELP_CSS in head
    assert HELP_HEAD in head and shell.SHELL_HEAD in head
    # Each page's own style sheet is registered beside its builder.
    assert datasets_page.PAGE_CSS in PAGE_CSS and datasets_page.PAGE_CSS in head
    assert head.index(datasets_page.PAGE_CSS) < head.index("</style>")
    extra = "#shep-page-test .x { font-size: 15px; }"
    assert extra in dashboard_head((*PAGE_CSS, extra))
    assert "#shep-ds-table" not in SHELL_CSS          # a page's rules live in its module
    # The one-screen rule reaches the page frame (keyed off the layout pass's
    # attribute on the root element): it is not left to `css`.
    from src.webui.style import SIDE, STACK
    assert SIDE + " {" in head and STACK + " {" in head
    assert "@media" not in options["css"].split("@font-face")[-1]
    assert "</style>" not in SHELL_CSS + HELP_CSS
    assert "IBM Plex Sans" in options["css"]


def test_no_text_size_is_under_14px():
    from src.webui.app import TEXT_SIZE, _dashboard_theme

    sizes = [TEXT_SIZE.xxs, TEXT_SIZE.xs, TEXT_SIZE.sm, TEXT_SIZE.md, TEXT_SIZE.lg]
    assert all(int(s.rstrip("px")) >= 14 for s in sizes)
    assert TEXT_SIZE.md == "15px"
    theme = _dashboard_theme()
    assert (theme.text_xxs, theme.text_xs, theme.text_sm, theme.text_md) == (
        "14px", "14px", "14px", "15px")
    css_sizes = re.findall(r"font-size:\s*([0-9.]+)(px|rem)", shell.SHELL_HEAD)
    assert css_sizes == []           # the shell's script sets no size


def test_the_shell_css_never_shrinks_text_under_14px():
    from src.webui.app import DASHBOARD_CSS, DASHBOARD_HEAD

    for css in (DASHBOARD_CSS, DASHBOARD_HEAD):
        for value, unit in re.findall(r"font-size:\s*([0-9.]+)(px|rem|em)", css):
            px = float(value) * (16 if unit == "rem" else 1)
            assert unit != "em" and px >= 14, (value, unit)


def test_the_page_is_always_light_and_gradios_footer_is_off():
    """Gradio's dark mode (the system's setting, or ?__theme=dark) would leave
    its near-white prose on the dashboard's light page: the head script keeps
    its class off the body and asks for light in the address."""
    from src.webui.app import mount_options
    from src.webui.style import SHELL_CSS

    assert "classList.remove('dark')" in shell.SHELL_HEAD
    assert "attributeFilter: ['class']" in shell.SHELL_HEAD
    assert "searchParams.set('__theme', 'light')" in shell.SHELL_HEAD
    assert "color-scheme: light" in SHELL_CSS
    # No footer: its "Settings" duplicated the ⚙ Settings tab and offered Dark.
    assert mount_options()["footer_links"] == []


def test_the_segmented_controls_selected_option_is_dark_on_white():
    from src.webui.style import SHELL_CSS

    rule = next(line for line in SHELL_CSS.splitlines()
                if line.startswith(".shep-seg .wrap label.selected {"))
    assert "background: #fff" in rule and "color: var(--shep-text) !important" in rule


def test_the_no_queue_event_path_is_kept_out_of_the_access_log():
    from src.api.main import _QUIET_PREFIXES

    assert "/ui/gradio_api/run/" in _QUIET_PREFIXES


def test_a_page_can_be_built_alone_with_a_standalone_context():
    from src.webui.components.datasets_page import create_datasets_page

    with gr.Blocks() as demo:
        ctx = shell.standalone_context(demo)
        create_datasets_page(ctx)
    assert {"shep-ds-table", "shep-ds-details", "shep-ds-use"} <= {
        getattr(b, "elem_id", None) for b in demo.blocks.values()}


def test_plain_html_has_no_prose_css_and_no_default_click():
    """No prose CSS and no default click script: only the shared disclosure
    keeper, then the caller's own script."""
    from src.webui.components.renderers import KEEP_OPEN_JS

    with gr.Blocks():
        block = shell.plain_html("<b>x</b>", elem_id="x", elem_classes="extra")
        own = shell.plain_html("", js_on_load="var own = 1;")
    assert block.apply_default_css is False and block.js_on_load == KEEP_OPEN_JS
    assert own.js_on_load == KEEP_OPEN_JS + "var own = 1;"
    assert block.elem_classes == ["shep-html", "extra"]


def test_every_html_an_event_draws_keeps_its_disclosures_as_the_reader_left_them():
    """Every ``gr.HTML`` of the dashboard that an event draws (any listener's
    output: a poll's, a tick's, a render's or the reader's own act's) runs
    the shared keeper (`renderers.KEEP_OPEN_JS`) once, before its own script,
    so a ``details`` or an ``aria-expanded`` popover it holds stays as the
    reader left it when it is drawn again. What a region holds is its
    renderers' choice at run time, so every one is listed from the built app.
    On fcc3dea 128 of the 133 had none: only the regions a poll or a render
    draws again were wired (T08-flow-2), and those the reader's own acts draw
    again (the "Set by you" lines, the Continue drawer's preview, Models'
    details and records) closed a Details the reader had opened."""
    from src.webui.app import create_gradio_app
    from src.webui.components.renderers import KEEP_OPEN_JS

    demo = create_gradio_app()
    drawn = {block._id: block for fn in demo.fns.values() for block in fn.outputs
             if isinstance(block, gr.HTML)}
    names = {key: getattr(block, "elem_id", None) or f"#{key}" for key, block in drawn.items()}
    assert len(drawn) >= 130, len(drawn)
    unkept = sorted(names[key] for key, block in drawn.items()
                    if not (block.js_on_load or "").startswith(KEEP_OPEN_JS))
    assert unkept == [], unkept
    assert [names[key] for key, block in drawn.items()
            if block.js_on_load.count(KEEP_OPEN_JS) != 1] == []
    # Among them, regions only the reader's own acts draw again.
    assert {"shep-train-setby", "ft_setby", "continue_preview", "shep-mp-detail-top",
            "shep-mp-records"} <= set(names.values())


def _disclosures_named(markup: str) -> list:
    """What `renderers.KEEP_OPEN_JS` keys each disclosure of `markup` by: the
    ``data-keep`` of the disclosure itself or of its nearest element around
    it (``None`` where none names it), in order."""
    from html.parser import HTMLParser

    void = {"area", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}

    class Names(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack, self.found = [], []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            named = attrs.get("data-keep")
            if named is None:
                named = next((n for _t, n in reversed(self.stack) if n is not None), None)
            if tag == "details" or ("aria-expanded" in attrs and "aria-controls" in attrs):
                self.found.append(named)
            if tag not in void:
                self.stack.append((tag, attrs.get("data-keep")))

        def handle_endtag(self, tag):
            while self.stack:
                if self.stack.pop()[0] == tag:
                    break

    parser = Names()
    parser.feed(markup)
    return parser.found


def test_a_region_that_can_show_another_thing_names_what_its_disclosures_are_of():
    """The review of 5033d29: the keeper runs on every drawn HTML (4c72a2d),
    and keys a disclosure by what its region names (`renderers.keep_as`), its
    words and its place; a region that names nothing is its one thing. The
    regions that show one of many named nothing, so a "Details ›" opened for
    one model was drawn open for the next one selected (the same for a
    record, a candidate's "More paths", a workspace's manifest, the Continue
    drawer's settings and a result's folds); before 4c72a2d they closed on
    every drawing. Each such region's renderer names the thing shown, so
    every disclosure it draws for one thing is named, drawn again for the
    same thing by the same names, and for another thing by none of them."""
    from src.webui.components import datasets_page as dsp
    from src.webui.components import diagnosis_panel as dp
    from src.webui.components import models_page as mp
    from src.webui.components import results_segment as rs
    from src.webui.components import training_console as tc

    def model(name):
        return {"sha256": name[0] * 64, "architecture": "gatv2", "run": f"run-{name}",
                "filename": f"{name}.pt"}

    def record(digit, sequence):
        return {"source": digit * 64, "cohort": {"label": "cohort", "role": "test"},
                "registration": {"sequence": sequence, "registered_at": "2026-11-17T00:00:00Z"}}

    def candidate(disease):
        paths = [["HP:1", f"G{n}", disease] for n in range(5)]
        return dp.details_outputs({"candidates": [{"rank": 1, "disease_id": disease,
                                                   "evidence_package": {"direct_paths": paths}}]},
                                  1, "paths")["details_html"]["value"]

    def workspace(name):
        rows = [{"name": n, "trainable": True, "facts": {"bound": [("Graph", "a" * 64)]}}
                for n in ("ws_a", "ws_b")]
        return dsp.details_html({"rows": rows}, name)

    def checked(run):
        preview = {"ok": True, "parent": f"gatv2/{run}/last.pt", "sha256": run[-1] * 64,
                   "inherited": {"learning_rate": 0.001}, "action": "continue 5 more epochs"}
        return tc._check_html(preview, tc.CONTINUE_FROM_LIST, listed=f"gatv2/{run}/last.pt")

    reference = "mims-harvard/SHEPHERD@e95433a disease run_*.sh script (lr 1e-5; README U3, U8)"
    row = {"protocol": "disease", "result": "ok", "data": {"kind": "supplied"},
           "exposure": {"r10": None}, "model": {"reference": reference}}
    regions = {
        "Models' selected checkpoint": (lambda: mp.identity_html(model("alpha"), None),
                                        lambda: mp.identity_html(model("beta"), None)),
        "Models' record": (lambda: mp.record_block_html(record("c", 1), None),
                           lambda: mp.record_block_html(record("d", 2), None)),
        "Diagnosis' selected candidate": (lambda: candidate("MONDO:1"),
                                          lambda: candidate("MONDO:2")),
        "Datasets' selected workspace": (lambda: workspace("ws_a"), lambda: workspace("ws_b")),
        "the Continue drawer's check": (lambda: checked("run-a"), lambda: checked("run-b")),
        "a result's row": (lambda: rs.protocol_table_html("disease", [row], names=["r1"]),
                           lambda: rs.protocol_table_html("disease", [row], names=["r2"])),
        "the results compared": (lambda: rs.compare_html([row], False, names=["r1"]),
                                 lambda: rs.compare_html([row], False, names=["r2"])),
    }
    for region, (one, other) in regions.items():
        named = _disclosures_named(one())
        assert named and None not in named, (region, named)
        assert _disclosures_named(one()) == named, region
        assert set(named).isdisjoint(_disclosures_named(other())), (region, named)


def test_the_dead_code_the_m5_review_found_stays_gone():
    """M5-3 review ARCH-6: a merge helper, a default workspace path, an
    unused import, the temporary body's style and the style of an element the
    dashboard no longer has, none of them referenced."""
    from src.webui import app, style
    from src.webui.components import help as help_module
    from src.webui.components import training_console

    assert not hasattr(help_module, "merge_help")
    assert not hasattr(training_console, "DEFAULT_WORKSPACE")
    assert "math" not in vars(training_console)
    assert "status_display" not in app.DASHBOARD_CSS
    assert "shep-temporary" not in style.SHELL_CSS


def test_no_text_names_the_deleted_amp_refusal_and_the_app_lists_its_pages():
    """K6 closing review: K6b deleted the AMP/HGT refusal, yet the help, shell
    and style texts still gave it as the runtime ``info=`` example, and the
    app's docstring listed the research pages without Fine-tune."""
    from src.webui import app, style
    from src.webui.components import help as help_module

    for text in (help_module.__doc__, shell.__doc__, style.SHELL_CSS,
                 Path(style.__file__).read_text(encoding="utf-8")):
        assert "HGT" not in text and "AMP radio" not in text
    for text in (help_module.__doc__, shell.__doc__, style.SHELL_CSS):
        assert "Reference cohort select" in " ".join(text.split())
    pages = " · ".join(shell.LABELS[page] for page in shell.RESEARCH_PAGES)
    assert pages == "Datasets · Training · Fine-tune · Test · Models"
    assert f"with {pages}," in " ".join(app.__doc__.split())


# --------------------------------------------------------------------------- M5-3 polish
def _rule(css, selector):
    """The declarations of the first rule whose selector list is `selector`."""
    start = css.index(selector + " {")
    return css[start + len(selector) + 2:css.index("}", start)]


def _last_rule(css, selector):
    """The declarations of the last rule whose selector list is `selector` (a
    later rule overrides an earlier one)."""
    start = css.rindex(selector + " {")
    return css[start + len(selector) + 2:css.index("}", start)]


def blocks(css, selector):
    """Every block of rules nested in `selector` (`style.SIDE`, ``STACK``,
    ``SIDE_SHORT``: the layout pass's one decision), each from the selector to
    the brace that closes it, joined."""
    found, at = [], 0
    while True:
        start = css.find(selector + " {", at)
        if start < 0:
            return "\n".join(found)
        depth, i = 0, start
        while True:
            depth += {"{": 1, "}": -1}.get(css[i], 0)
            if depth == 0 and css[i] == "}":
                break
            i += 1
        found.append(css[start:i + 1])
        at = i + 1


def side(css):
    """The side-by-side rules (`style.SIDE`'s blocks)."""
    from src.webui.style import SIDE

    return blocks(css, SIDE)


def stack(css):
    """The stacked rules (`style.STACK`'s blocks)."""
    from src.webui.style import STACK

    return blocks(css, STACK)


def test_a_select_value_cut_to_an_ellipsis_carries_its_whole_text_as_title():
    """M5-3: Training's dataset select, Test's workspace select and Models'
    cohort-version select cut their values with no title. One rule for every
    select: the value ends in an ellipsis, and the shell's head script gives a
    value that does not fit its whole text as its title (and takes it away
    when it fits), on each layout pass and when the pointer or focus arrives."""
    from src.webui.style import SHELL_CSS

    assert "text-overflow: ellipsis" in _rule(SHELL_CSS, '.gradio-container .wrap-inner input[role="combobox"]')
    head = shell.SHELL_HEAD
    assert "var TITLED = 'input[role=\"combobox\"], .shep-seg label, .shep-chip-text';" in head
    assert "box.scrollWidth > box.clientWidth + 1 ? text : ''" in head
    assert "el.removeAttribute('title')" in head
    for event in ("mouseover", "focusin", "input"):
        assert f"document.addEventListener('{event}', titleAt, true);" in head
    assert "layout(); titles();" in head


def test_an_accordion_arrow_turns_inside_a_square_box():
    """M5-3: the turned ▼ (taller than wide) spilled 3 px past the header."""
    from src.webui.style import SHELL_CSS

    rule = _rule(SHELL_CSS, ".gradio-container button.label-wrap > .icon")
    assert "width: 22px" in rule and "height: 22px" in rule and "flex: none" in rule


def test_a_segmented_control_keeps_one_row():
    """M5-3: the candidate details' "Paths 5" wrapped to a second row. The
    segments are grid columns up to their text: short labels stay whole, the
    longest ends in a (titled) ellipsis."""
    from src.webui.style import SHELL_CSS

    rule = _last_rule(SHELL_CSS, ".shep-seg .wrap")
    assert "display: inline-grid !important" in rule and "grid-auto-flow: column" in rule
    assert "grid-auto-columns: minmax(0, max-content)" in rule
    span = _last_rule(SHELL_CSS, ".shep-seg .wrap label span")
    assert "text-overflow: ellipsis" in span and "white-space: nowrap" in span


def test_a_panels_header_sticks_to_the_top_of_the_region_that_scrolls():
    """M5-3: the Test Result panel's title scrolled away with the report."""
    from src.webui.style import SHELL_CSS

    rule = _rule(side(SHELL_CSS), "  .shep-panel .shep-panel-hd")
    assert "position: sticky" in rule and "top: 0" in rule and "background: var(--shep-panel)" in rule
    # never stacked, where the page scrolls
    assert "position: sticky" not in stack(SHELL_CSS)


def test_the_wrapped_chips_sit_on_their_own_line_under_the_area_switch():
    """B5 (leftover b): at 480 px the area switch's two tabs take two lines,
    and the chips, placed one header row down, covered the Research tab
    (measured with ``document.elementsFromPoint``). The header script now
    measures where the chips' and Settings' own line starts (the right tab
    group's top) and the wrapped chips sit there, however many lines the
    switch takes. This pins that wiring; what the browser draws is
    `test_no_header_element_covers_a_tab_at_any_width`'s."""
    from src.webui.style import SHELL_CSS

    head = shell.SHELL_HEAD
    assert "right.getBoundingClientRect().top - shell.getBoundingClientRect().top" in head
    assert "shell.style.setProperty('--shep-chips-top', line + 'px')" in head
    wrapped = SHELL_CSS[SHELL_CSS.index("#shep-shell.shep-nav-wrapped #shep-chips {"):]
    wrapped = wrapped[:wrapped.index("}")]
    assert "top: var(--shep-chips-top, var(--shep-row1-h));" in wrapped


#: Opens the dashboard at 480 px, then 360, 420, 760, 1100 and 1850 (a resize
#: runs the header's layout, as a load does), and reports for each area tab
#: what ``document.elementsFromPoint`` finds on top at its centre and four
#: inset corners: the tab itself or something inside it, or what covers it;
#: and each header chip's left and right edges.
_HEADER_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 480, height: 800 } })).newPage();
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#shep-chips .shep-chips', { timeout: 60000 });
  const out = {};
  for (const width of [480, 360, 390, 420, 760, 1100, 1850]) {
    await page.setViewportSize({ width, height: 800 });
    await page.waitForTimeout(800);
    out[width] = await page.evaluate(() => {
      const ids = ['shep-tab-clinical-button', 'shep-tab-research-button',
                   'shep-tab-settings-button'];
      const tabs = {};
      for (const id of ids) {
        const el = document.getElementById(id);
        if (!el) { tabs[id] = null; continue; }
        const r = el.getBoundingClientRect();
        const points = [[r.left + r.width / 2, r.top + r.height / 2], [r.left + 3, r.top + 3],
                        [r.right - 3, r.top + 3], [r.left + 3, r.bottom - 3],
                        [r.right - 3, r.bottom - 3]];
        tabs[id] = points.map(([x, y]) => {
          const top = document.elementsFromPoint(x, y)[0];
          return !top || el.contains(top) ? null
            : (top.id || String(top.className).slice(0, 60) || top.tagName);
        }).filter(Boolean);
      }
      return { wrapped: document.getElementById('shep-shell').classList.contains('shep-nav-wrapped'),
               chips: document.querySelector('#shep-chips .shep-chips').childElementCount, tabs,
               edges: [...document.querySelectorAll('#shep-chips .shep-chip')].map((c) => {
                 const r = c.getBoundingClientRect(); return [r.left, r.right]; }),
               // Each chip: where it starts and its line, its words whole or not, and its whole
               // width (its words' own width, as an ellipsis does not draw them).
               chipWords: [...document.querySelectorAll('#shep-chips .shep-chip')].map((c) => {
                 const r = c.getBoundingClientRect(), t = c.querySelector('.shep-chip-text');
                 return { left: r.left, top: r.top, whole: !t || t.scrollWidth <= t.clientWidth + 1,
                          natural: r.width + (t ? t.scrollWidth - t.clientWidth : 0) }; }),
               row: document.getElementById('shep-shell').clientWidth - 32,
               width: document.documentElement.clientWidth };
    });
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_no_header_element_covers_a_tab_at_any_width(tmp_path):
    """B5 review: the chips' line was pinned only as source text. In a real
    browser (Node and Playwright, skipped without), the dashboard mounted as
    the service mounts it (`app.mount_options`: its CSS and head script): at
    360, 420, 480, 760, 1100 and 1850 px each area tab is what
    ``document.elementsFromPoint`` finds first at its centre and corners, the
    chips never over it; at 480 the header has wrapped onto its second line;
    every chip is inside the window (UI-02: at 360 px the D8 chip's left edge
    was at -41 px; its words now end in a titled ellipsis inside the room).
    The sweep's L7 (SW-LK9, SW-LK3): at 390 px "Operator controls: on (this
    machine)" ended in an ellipsis beside Settings though a line of the header
    holds it, and the GPU chip, wrapped, floated mid-box. A chip's words are
    whole wherever a header line holds the chip whole (the ellipsis only for a
    chip wider than a line), and chips on more than one line start on one
    left edge."""
    import subprocess

    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import node_with_playwright
    from tests.fixtures.served_page import served

    found = node_with_playwright()
    if found is None:
        pytest.skip("Node with Playwright is not installed")
    node, env = found
    script = tmp_path / "header.cjs"
    script.write_text(_HEADER_JS, encoding="utf-8")
    with served(create_gradio_app(), **mount_options()) as base:
        page = base[:-len("gradio_api")]
        result = subprocess.run([node, str(script), page], env=env, capture_output=True,
                                text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert seen["480"]["wrapped"] is True and seen["1850"]["wrapped"] is False
    for width, facts in seen.items():
        assert facts["chips"] > 0, width
        for tab, covered in facts["tabs"].items():
            assert covered == [], (width, tab, covered)
        for left, right in facts["edges"]:
            assert 0 <= left and right <= facts["width"], (width, facts["edges"])
        for chip in facts["chipWords"]:
            assert chip["whole"] or chip["natural"] > facts["row"], (width, chip, facts["row"])
        tops = {round(chip["top"]) for chip in facts["chipWords"]}
        if len(tops) > 1:
            lefts = [chip["left"] for chip in facts["chipWords"]]
            assert max(lefts) - min(lefts) <= 1, (width, facts["chipWords"])



# --------------------------------------------------------------------------- a reload
def _where_page():
    """The shell with stand-in pages: Models holds a button that answers the
    session's location as the server holds it (what every page's poll reads),
    and Diagnosis' select writes it as late as a busy server would answer it
    (`slow`, in seconds, set by the test)."""
    slow = {"seconds": 0.0}

    def models(ctx):
        where = gr.Textbox(elem_id="where")
        ask = gr.Button("Where", elem_id="ask-where")
        asked = []

        def answer(location):
            asked.append(location)
            return f"{location} #{len(asked)}"

        ask.click(answer, inputs=[ctx.location], outputs=[where])

    pages = {**{page: (lambda ctx: None) for page in (*shell.RESEARCH_PAGES, shell.SETTINGS,
                                                       shell.CLINICAL)},
             shell.MODELS: models}
    with gr.Blocks() as demo:
        ctx = shell.build_shell(demo, pages)
    (shown,) = [fn for fn in demo.fns.values()
                if (ctx.tabs[shell.CLINICAL]._id, "select") in fn.targets]
    original = shown.fn

    def late(*args):
        import time

        time.sleep(slow["seconds"])
        return original(*args)

    late.__name__ = original.__name__
    shown.fn = late
    return demo, slow


#: Research › Models opened, then the page loaded again; then Settings and
#: Diagnosis. Each says which tabs are selected after the load, what the
#: session's location is on the server, and what this browser tab keeps.
_RELOAD_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await page.addInitScript(QUEUE_JS);
  const tab = (nav, id) => `#${nav} > .tab-wrapper [role="tab"][data-tab-id="${id}"]`;
  const chosen = (nav) => page.evaluate((n) => {
    const b = document.querySelector(`#${n} > .tab-wrapper [role="tab"][aria-selected="true"]`);
    return b ? b.getAttribute('data-tab-id') : null; }, nav);
  const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.inflight === 0
                                          && window.__shepQueue.joins.every((j) => j.event_id !== null),
                                          null, { timeout: 20000 });
  // The server's answer to "Where", a new one each time (`location #n`).
  const where = async () => {
    const before = await page.inputValue('#where textarea');
    await page.click('#ask-where');
    await page.waitForFunction((b) => { const v = document.querySelector('#where textarea').value;
                                        return v !== '' && v !== b; }, before, { timeout: 20000 });
    return (await page.inputValue('#where textarea')).replace(/ #\d+$/, '');
  };
  // The stage hint's words as drawn once the shell's pass after the last change has run (null: none drawn).
  const hint = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => {
    const h = document.querySelector('#shep-research-nav > .tab-wrapper > .shep-stage');
    r(h && !h.hidden && h.getClientRects().length ? h.innerText.replace(/\s+/g, ' ').trim() : null); }))));
  const facts = async () => ({ outer: await chosen('shep-nav'), inner: await chosen('shep-research-nav'),
                               kept: await page.evaluate((k) => sessionStorage.getItem(k), '__KEPT__'),
                               title: await page.title(), hint: await hint() });
  const loaded = async (outer) => {
    await page.waitForSelector(tab('shep-nav', outer) + '[aria-selected="true"]', { timeout: 30000 });
    await page.waitForFunction(() => window.ShepShell !== undefined);
    await idle();
  };
  const out = {};
  await page.goto(base, { waitUntil: 'load' });
  await loaded('clinical');
  await page.click(tab('shep-nav', 'research'));
  await page.click(tab('shep-research-nav', 'models'));
  await page.waitForSelector(tab('shep-research-nav', 'models') + '[aria-selected="true"]');
  await idle();
  out.opened = { ...(await facts()), location: await where() };
  await page.reload({ waitUntil: 'load' });
  await loaded('research');
  await page.waitForSelector(tab('shep-research-nav', 'models') + '[aria-selected="true"]',
                             { timeout: 20000 });
  await idle();
  out.reloaded = { ...(await facts()), location: await where() };
  await page.click(tab('shep-nav', 'settings'));
  await idle();
  await page.reload({ waitUntil: 'load' });
  await loaded('settings');
  out.settings = await facts();
  await page.click(tab('shep-nav', 'clinical'));
  await idle();
  await page.reload({ waitUntil: 'load' });
  await loaded('clinical');
  out.clinical = await facts();
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_reload_comes_back_on_the_page_shown(tmp_path):
    """UI-14 (B32-UI8), in a real browser (Node and Playwright, skipped
    without), the shell mounted as the service mounts it: with Research ›
    Models open, a reload came back on Diagnosis. It comes back on Research ›
    Models now, with the session's location the server holds there too (what
    Models' poll reads), and the title; Settings and Diagnosis come back as
    themselves. This browser tab keeps the page id alone. The stage hint
    comes back with the page: "Stage: Evaluate" on Models after the reload,
    none on Settings or Diagnosis (the stage-hint package, N03)."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    demo, _slow = _where_page()
    script = (_RELOAD_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__KEPT__", shell.KEPT_PAGE))
    with served(demo, **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")])
    models = {"outer": "research", "inner": "models", "kept": "models",
              "title": shell.TITLES["models"], "location": "models", "hint": "Stage: Evaluate"}
    assert seen["opened"] == models, seen
    assert seen["reloaded"] == models, seen
    assert seen["settings"]["outer"] == "settings" and seen["settings"]["kept"] == "settings"
    assert seen["settings"]["title"] == shell.TITLES["settings"]
    assert seen["clinical"]["outer"] == "clinical" and seen["clinical"]["kept"] == "clinical"
    assert seen["settings"]["hint"] is None and seen["clinical"]["hint"] is None


# --------------------------------------------------------------------------- the stage hint
def _stage_page():
    """The shell with stand-in pages and two cross-links, each a page's own
    button that opens another as every cross-link does (`PageContext.go`):
    Models › "Open Training", Training › "Open Test"."""
    def link(target):
        def build(ctx):
            gr.Button(f"Open {shell.LABELS[target]}", elem_id=f"go-{target}").click(
                lambda: ctx.go(target), outputs=ctx.nav_outputs, show_progress="hidden")
        return build

    pages = {**{page: (lambda ctx: None) for page in (*shell.RESEARCH_PAGES, shell.SETTINGS,
                                                       shell.CLINICAL)},
             shell.MODELS: link(shell.TRAINING), shell.TRAINING: link(shell.TEST)}
    with gr.Blocks() as demo:
        shell.build_shell(demo, pages)
    return demo


#: The stage hint along every path that changes the page shown, at the owner's
#: window, while the job runner's stand-in says a training job runs: what the
#: two levels select, the hint's words as drawn (null when none is drawn) and
#: the GPU chip's words, after each act once the page's requests are answered
#: and the shell's next pass has run. Then each tab's look with the pointer on
#: Test beside the selected Models, the hint's look under the pointer, where
#: the keyboard goes after the last tab, and the hint taken out of the row (as
#: a redraw of Gradio's does). Then the row in four windows: the owner's, a
#: narrow one, the owner's at a browser zoom of 200 % (half the CSS pixels,
#: twice the device pixels) and a narrow one with larger and wider text (as a
#: browser's minimum font size and a text-spacing style sheet give: the text
#: alone, every box's own size kept). Then, on Training and on Models, the
#: row at the edge of its room: a window a few pixels narrower than the row
#: needs for the tabs and the hint side by side (as drawn: the tablist with
#: its right padding, the hint with its right margin), then a few wider.
_STAGE_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const ready = async (page) => {
    await page.addInitScript(QUEUE_JS);
    await page.goto(base, { waitUntil: 'load' });
    await page.waitForSelector('#shep-tab-clinical-button[aria-selected="true"]', { timeout: 30000 });
    await page.waitForFunction(() => window.ShepShell !== undefined);
  };
  const idle = (page) => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.inflight === 0
                                              && window.__shepQueue.joins.every((j) => j.event_id !== null),
                                              null, { timeout: 20000 });
  const frame = (page) => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  const tab = (nav, id) => `#${nav} > .tab-wrapper [role="tab"][data-tab-id="${id}"]`;
  const seen = async (page) => {
    await idle(page);
    await frame(page);
    return page.evaluate(() => {
      const chosen = (n) => { const b = document.querySelector(`#${n} > .tab-wrapper [role="tab"][aria-selected="true"]`);
                              return b ? b.getAttribute('data-tab-id') : null; };
      const h = document.querySelector('#shep-research-nav > .tab-wrapper > .shep-stage');
      const drawn = h && !h.hidden && h.getClientRects().length > 0;
      const gpu = document.querySelector('#shep-chips .shep-chip-gpu .shep-chip-text');
      return { outer: chosen('shep-nav'), inner: chosen('shep-research-nav'),
               hint: drawn ? h.innerText.replace(/\s+/g, ' ').trim() : null,
               hints: document.querySelectorAll('.shep-stage').length, gpu: gpu ? gpu.textContent : null,
               focused: document.activeElement ? document.activeElement.id : null };
    });
  };
  const out = { click: {}, keys: [], links: [] };
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await ready(page);
  await page.waitForSelector('#shep-chips .shep-chip-gpu[data-state="busy"]', { timeout: 30000 });
  out.clinical = await seen(page);
  await page.click(tab('shep-nav', 'research'));
  out.research = await seen(page);
  for (const id of ['datasets', 'training', 'finetune', 'test', 'models']) {
    await page.click(tab('shep-research-nav', id));
    out.click[id] = await seen(page);
  }
  // Keys: the focus on Fine-tune, Enter; Tab to Test, Space; Shift+Tab back, Enter.
  await page.focus(tab('shep-research-nav', 'finetune'));
  for (const key of ['Enter', 'Tab', 'Space', 'Shift+Tab', 'Enter']) {
    await page.keyboard.press(key);
    out.keys.push({ key, ...(await seen(page)) });
  }
  // Cross-links: from Models to Training, from Training to Test.
  await page.click(tab('shep-research-nav', 'models'));
  await seen(page);
  for (const id of ['training', 'test']) {
    await page.click(`#go-${id}`);
    out.links.push({ to: id, ...(await seen(page)) });
  }
  await page.click(tab('shep-nav', 'settings'));
  out.settings = await seen(page);
  await page.click(tab('shep-nav', 'research'));
  out.back = await seen(page);
  // A tab under the pointer beside the selected one: each tab's look.
  await page.click(tab('shep-research-nav', 'models'));
  await seen(page);
  const looks = () => page.evaluate(() => [...document.querySelectorAll('#shep-research-nav > .tab-wrapper [role="tab"]')]
    .map((t) => { const s = getComputedStyle(t);
      return { id: t.getAttribute('data-tab-id'), selected: t.getAttribute('aria-selected') === 'true',
               line: s.borderBottomColor, background: s.backgroundColor, color: s.color, weight: s.fontWeight }; }));
  await page.mouse.move(2, 950);
  out.rest = await looks();
  await page.hover(tab('shep-research-nav', 'test'));
  out.hovered = await looks();
  // The hint: no control's look under the pointer, no focus stop, no role.
  const hintLook = () => page.evaluate(() => { const h = document.querySelector('#shep-research-nav > .tab-wrapper > .shep-stage');
    if (!h) { return null; }
    const s = getComputedStyle(h);
    return { color: s.color, background: s.backgroundColor, border: s.borderBottomColor, cursor: s.cursor,
             decoration: s.textDecorationLine, tag: h.tagName, role: h.getAttribute('role'),
             tabindex: h.getAttribute('tabindex'), live: h.getAttribute('aria-live'), tabIndex: h.tabIndex,
             controls: h.querySelectorAll('a, button, input, [role], [tabindex]').length,
             parent: h.parentElement.className.includes('tab-wrapper'),
             after: h.previousElementSibling && h.previousElementSibling.getAttribute('role'),
             inTablist: !!h.closest('[role="tablist"]') }; });
  await page.mouse.move(2, 950);
  out.hintRest = await hintLook();
  if (out.hintRest) { await page.hover('#shep-research-nav > .tab-wrapper > .shep-stage'); }
  out.hintHover = await hintLook();
  await page.focus(tab('shep-research-nav', 'models'));
  await page.keyboard.press('Tab');
  out.afterLastTab = await page.evaluate(() => ({ id: document.activeElement.id, tag: document.activeElement.tagName,
    inHint: !!document.activeElement.closest('.shep-stage') }));
  // Gradio draws the row again: the hint is put back, with its words.
  await page.evaluate(() => { const h = document.querySelector('#shep-research-nav > .tab-wrapper > .shep-stage');
                              if (h) { h.remove(); } });
  out.redrawn = await seen(page);
  await page.close();
  // The row as drawn: its box and the end of its content, the hint, its
  // right margin, the tablist's right padding, the tabs; null with no hint.
  const rowFacts = () => {
    const box = (e) => { const r = e.getBoundingClientRect();
      return { left: r.left, top: r.top, right: r.right, bottom: r.bottom }; };
    const row = document.querySelector('#shep-research-nav > .tab-wrapper');
    const h = row && row.querySelector(':scope > .shep-stage');
    if (!h) { return null; }
    const r = box(h), rs = getComputedStyle(row);
    const list = row.querySelector('[role="tablist"]');
    const at = document.elementsFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2)[0];
    const tabs = [...row.querySelectorAll('[role="tab"]')].map(box);
    const f = { row: box(row), end: box(row).right - parseFloat(rs.paddingRight) - parseFloat(rs.borderRightWidth),
                hint: r, words: h.innerText.replace(/\s+/g, ' ').trim(),
                margin: parseFloat(getComputedStyle(h).marginRight),
                listPad: parseFloat(getComputedStyle(list).paddingRight), tabs,
                hit: !!at && h.contains(at), cut: h.scrollWidth > h.clientWidth + 1 || h.scrollHeight > h.clientHeight + 1,
                overX: document.scrollingElement.scrollWidth - document.scrollingElement.clientWidth,
                width: innerWidth };
    f.needed = Math.max(...tabs.map((t) => t.right)) + f.listPad + (r.right - r.left) + f.margin;
    return f;
  };
  // The row in four windows.
  out.windows = {};
  const LARGER = '#shep-research-nav > .tab-wrapper button[role="tab"], #shep-research-nav > .tab-wrapper > .shep-stage,'
    + ' #shep-research-nav > .tab-wrapper > .shep-stage > * { font-size: 24px !important; letter-spacing: 0.12em !important; }';
  for (const [name, w, h, dpr, css] of [['owner', 1850, 960, 1, ''], ['narrow', 420, 900, 1, ''],
                                        ['zoom200', 925, 480, 2, ''], ['larger', 640, 900, 1, LARGER]]) {
    const p = await (await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: dpr })).newPage();
    await ready(p);
    if (css) { await p.addStyleTag({ content: css }); }
    await p.click(tab('shep-nav', 'research'));
    out.windows[name] = {};
    for (const id of ['training', 'models']) {
      await p.click(tab('shep-research-nav', id));
      await seen(p);
      out.windows[name][id] = await p.evaluate(rowFacts);
    }
    await p.close();
  }
  // The edge of the row's room, on each page: the window resized until the
  // row's content ends 6 px short of what the tabs and the hint need side by
  // side ('under'), then 6 px past it ('over').
  out.edges = {};
  const e = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await ready(e);
  await e.click(tab('shep-nav', 'research'));
  for (const id of ['training', 'models']) {
    await e.setViewportSize({ width: 1850, height: 960 });
    await e.click(tab('shep-research-nav', id));
    await seen(e);
    out.edges[id] = {};
    for (const [side, by] of [['under', -6], ['over', 6]]) {
      let f = await e.evaluate(rowFacts);
      for (let i = 0; f && i < 5 && Math.abs(f.end - f.needed - by) > 1; i += 1) {
        await e.setViewportSize({ width: Math.round(f.width - (f.end - f.needed - by)), height: 960 });
        await seen(e);
        f = await e.evaluate(rowFacts);
      }
      out.edges[id][side] = f;
    }
  }
  await e.close();
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def stage_hint_seen(tmp_path_factory):
    """`_STAGE_JS` once for this module (Node and Playwright, skipped
    without), on the shell mounted as the service mounts it with stand-in
    pages (`_stage_page`), while the job runner says a training job runs (the
    runner's stand-in: its `active_job`, which the GPU chip reads)."""
    from src.api.services.training_manager import training_manager
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    running = {"job_type": "training", "job_id": "stand-in", "status": "running", "pid": 12345,
               "start_time": "2026-11-21T09:00:00+00:00"}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(training_manager, "active_job", lambda: dict(running))
        with served(_stage_page(), **mount_options()) as base:
            return run_node(_STAGE_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)),
                            tmp_path_factory.mktemp("stage-hint"), base[:-len("gradio_api")])


@pytest.mark.parametrize("page, words", [
    ("datasets", "Stage: Prepare"), ("training", "Stage: Prepare"), ("finetune", "Stage: Prepare"),
    ("test", "Stage: Evaluate"), ("models", "Stage: Evaluate")])
def test_each_research_page_shows_its_stage_at_the_right_of_its_row(stage_hint_seen, page, words):
    """N01-N02 (the stage-hint package of 2026-11-21), in a real browser: the
    page's own tab clicked (the shell's switch), the row shows that page's
    stage, once: "Stage: Prepare" for Datasets, Training and Fine-tune,
    "Stage: Evaluate" for Test and Models. On 056b8b5 no hint was drawn (the
    two captions stood above the tabs instead)."""
    from tests.fixtures.browser import STAGE_HINTS

    seen = stage_hint_seen["click"][page]
    assert (seen["outer"], seen["inner"], seen["hint"], seen["hints"]) == ("research", page, words, 1), seen
    assert STAGE_HINTS[page] == words


def test_the_stage_hint_follows_keys_cross_links_and_returns_and_none_off_research(stage_hint_seen):
    """N03 and N05 (package rows N03, N05): the hint follows every path that
    changes the page shown, as the tab underline does: Research opened (its
    default, Training), the keyboard (Enter on Fine-tune, Tab to Test and
    Space, Shift+Tab back and Enter), a page's cross-link (Models › Training,
    Training › Test) and a return to Research from Settings (the page it
    showed). Diagnosis and Settings draw no hint. When the hint is taken out
    of the row, as a redraw of Gradio's takes it, the shell's next pass puts it
    back with its words. (After a reload: `test_a_reload_comes_back_on_the_page_shown`.)"""
    seen = stage_hint_seen
    assert (seen["clinical"]["outer"], seen["clinical"]["hint"]) == ("clinical", None)
    assert (seen["research"]["inner"], seen["research"]["hint"]) == ("training", "Stage: Prepare")
    keys = [(k["key"], k["focused"], k["inner"], k["hint"]) for k in seen["keys"]]
    assert keys == [
        ("Enter", "shep-page-finetune-button", "finetune", "Stage: Prepare"),
        ("Tab", "shep-page-test-button", "finetune", "Stage: Prepare"),
        ("Space", "shep-page-test-button", "test", "Stage: Evaluate"),
        ("Shift+Tab", "shep-page-finetune-button", "test", "Stage: Evaluate"),
        ("Enter", "shep-page-finetune-button", "finetune", "Stage: Prepare")], keys
    links = [(k["to"], k["outer"], k["inner"], k["hint"]) for k in seen["links"]]
    assert links == [("training", "research", "training", "Stage: Prepare"),
                     ("test", "research", "test", "Stage: Evaluate")], links
    assert (seen["settings"]["outer"], seen["settings"]["hint"]) == ("settings", None)
    assert (seen["back"]["inner"], seen["back"]["hint"]) == ("test", "Stage: Evaluate")
    redrawn = seen["redrawn"]
    assert (redrawn["inner"], redrawn["hint"], redrawn["hints"]) == ("models", "Stage: Evaluate", 1)


def test_the_stage_hint_says_the_page_shown_never_the_job_that_runs(stage_hint_seen):
    """N4 (package row N04): a training job runs the whole time (the GPU chip
    says so on every page, as it said before) and the hint says the stage of
    the page shown: Evaluate on Test and Models, Prepare on Training. The
    hint reads no job state and gates nothing (its script reads the selected
    tab alone: `test_one_stage_mapping_names_the_tablist_and_feeds_the_hint_and_the_captions_are_gone`)."""
    seen = stage_hint_seen
    shown = [seen["clinical"], seen["research"], *seen["click"].values(), *seen["keys"], *seen["links"],
             seen["settings"], seen["back"]]
    assert {s["gpu"] for s in shown} == {"GPU job: training (job runner)"}
    assert seen["click"]["test"]["hint"] == seen["click"]["models"]["hint"] == "Stage: Evaluate"
    assert seen["click"]["training"]["hint"] == "Stage: Prepare"


def test_a_tab_under_the_pointer_never_reads_as_selected_and_the_hint_is_no_control(stage_hint_seen):
    """N7 (package row N05, its hover half; the owner's screenshot of Fine-tune
    hovered beside Models selected showed two underlined tabs): with the
    pointer on Test beside the selected Models, Test has a fill it lacks at
    rest and keeps its rest line (no underline), weight and a colour that is
    not the selected one; Models alone has the accent line, the accent text and
    the strong weight. The hint (N2) is plain text after the tablist (a
    paragraph, outside the tablist, no role, focus stop, live region or
    control in it); under the pointer it looks as at rest with no pointer
    hand, and the keyboard goes from the last tab past it."""
    seen = stage_hint_seen
    rest = {t["id"]: t for t in seen["rest"]}
    hovered = {t["id"]: t for t in seen["hovered"]}
    assert [t["id"] for t in seen["hovered"] if t["selected"]] == ["models"]
    test, models = hovered["test"], hovered["models"]
    assert test["background"] != rest["test"]["background"]
    assert (test["line"], test["weight"]) == (rest["test"]["line"], rest["test"]["weight"])
    assert test["line"] != models["line"] and test["weight"] != models["weight"]
    assert test["color"] != models["color"]
    selected_look = {(t["line"], t["weight"]) for t in seen["hovered"] if t["selected"]}
    assert [t["id"] for t in seen["hovered"] if (t["line"], t["weight"]) in selected_look] == ["models"]
    hint = seen["hintRest"]
    assert hint is not None, "no stage hint drawn"
    assert (hint["tag"], hint["role"], hint["tabindex"], hint["live"], hint["tabIndex"], hint["controls"]) == (
        "P", None, None, None, -1, 0), hint
    assert hint["parent"] and hint["after"] == "tablist" and not hint["inTablist"], hint
    assert seen["hintHover"] == hint and hint["cursor"] != "pointer" and hint["decoration"] == "none"
    assert not seen["afterLastTab"]["inHint"], seen["afterLastTab"]


def _hint_placed_by_the_rows_room(facts):
    """N6 (package §六: the room as drawn decides, never a window size): the
    stage hint lies inside the research pages' row, over no tab, at the row's
    right (its own right margin from the end of the row's content), and it is
    on a line of its own exactly when the row has no room for the tabs and the
    hint side by side: when the tabs alone take more than one line, or when
    the tablist as drawn (its last tab and its right padding) and the hint
    (its width and its right margin) need more than the row's content. Else
    it is on the tabs' line, after the last tab. Within a pixel of the edge
    either is right (rounding). Returns whether it is on a line of its own."""
    hint, row, tabs = facts["hint"], facts["row"], facts["tabs"]
    assert (row["left"] - 0.5 <= hint["left"] and hint["right"] <= row["right"] + 0.5
            and row["top"] - 0.5 <= hint["top"] and hint["bottom"] <= row["bottom"] + 0.5), facts
    assert not any(hint["left"] < t["right"] - 0.5 and hint["right"] > t["left"] + 0.5
                   and hint["top"] < t["bottom"] - 0.5 and hint["bottom"] > t["top"] + 0.5 for t in tabs), facts
    assert abs(hint["right"] - (facts["end"] - facts["margin"])) <= 0.5, facts
    last = max(t["right"] for t in tabs)
    tabs_one_line = max(t["top"] for t in tabs) - min(t["top"] for t in tabs) < 0.5
    needed = last + facts["listPad"] + (hint["right"] - hint["left"]) + facts["margin"]
    own_line = hint["top"] >= max(t["bottom"] for t in tabs) - 0.5
    if not tabs_one_line or needed > facts["end"] + 1:
        assert own_line, ("no room on the tabs' line, yet the hint is on it", needed, facts)
    elif needed < facts["end"] - 1:
        assert not own_line, ("room on the tabs' line, yet the hint is on a line of its own", needed, facts)
        assert hint["left"] >= last - 0.5, facts                                   # after the tabs
        assert hint["top"] < min(t["bottom"] for t in tabs) and hint["bottom"] > max(t["top"] for t in tabs)
    return own_line


@pytest.mark.parametrize("window", ["owner", "narrow", "zoom200", "larger"])
def test_the_stage_hint_stays_inside_the_row_and_off_the_tabs_in_every_window(stage_hint_seen, window):
    """N6 (package row N05, its layout half; package §六: relative geometry,
    no window size decides), on Training and on Models: at the owner's
    window, in a narrow one, at a browser zoom of 200 % of the owner's window
    and in a narrow one with larger, wider text (the text alone, every box its
    own size). In each the hint says its page's stage, is the first thing
    drawn at its centre, is not cut, and the page does not scroll sideways;
    it lies inside the research pages' row, over no tab, at the row's right,
    on the tabs' line after them where the row has room for both as drawn and
    on a line of its own where it has not (`_hint_placed_by_the_rows_room`).
    The windows are inputs, never the pass condition: which line the hint
    takes is judged by the row's room in each, and both outcomes are walked at
    the edge of that room (`test_the_stage_hint_takes_a_line_of_its_own_exactly_when_the_row_has_no_room`)."""
    for page, words in (("training", "Stage: Prepare"), ("models", "Stage: Evaluate")):
        facts = stage_hint_seen["windows"][window][page]
        assert facts is not None, (window, page, "no stage hint drawn")
        assert facts["words"] == words, facts
        assert facts["hit"] and not facts["cut"] and facts["overX"] <= 0, facts
        _hint_placed_by_the_rows_room(facts)


def test_the_stage_hint_takes_a_line_of_its_own_exactly_when_the_row_has_no_room(stage_hint_seen):
    """N6 at the edge of the row's room (the external review of b0d5fba: the
    window test judged the line by a named window, and its own-line check left
    out the tablist's 16 px right padding, so in a band of window widths 16 px
    wide it would fail the product's correct wrap): on Training and on
    Models, with the window 6 px short of the row's room for the tabs and the
    hint side by side, the hint is on a line of its own; 6 px past it, on the
    tabs' line after them. Both at the row's right, inside it, over no tab,
    uncut, with no sideways page scroll. The edge is found from the row as
    drawn at the owner's window, never from a window size."""
    for page, words in (("training", "Stage: Prepare"), ("models", "Stage: Evaluate")):
        edge = stage_hint_seen["edges"][page]
        for side, own in (("under", True), ("over", False)):
            facts = edge[side]
            assert facts is not None, (page, side, "no stage hint drawn")
            assert facts["words"] == words and facts["hit"] and not facts["cut"] and facts["overX"] <= 0, facts
            short = facts["needed"] - facts["end"]
            assert (short > 1) if own else (short < -1), (page, side, "the edge was not reached", facts)
            assert _hint_placed_by_the_rows_room(facts) is own, (page, side, facts)


#: Models opened while Diagnosis' mount-time select is still being answered
#: (the server answers it late), then the session's location asked once every
#: request has been answered.
_MOUNT_RACE_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const page = await (await browser.newContext()).newPage();
  await page.addInitScript(QUEUE_JS);
  const cfg = await (await fetch(base + 'config')).json();
  const shownIndex = cfg.dependencies.find((d) => d.api_name === '_shown_clinical').id;
  let sent = false, answered = false;
  const isShown = (req) => { try { return JSON.parse(req.postData() || '{}').fn_index === shownIndex; }
                             catch (e) { return false; } };
  page.on('request', (req) => { if (isShown(req)) { sent = true; } });
  page.on('requestfinished', (req) => {
    if (req.url().includes('/run/predict') && isShown(req)) { answered = true; } });
  const tab = (nav, id) => `#${nav} > .tab-wrapper [role="tab"][data-tab-id="${id}"]`;
  await page.goto(base, { waitUntil: 'load' });
  // Observable completion only: the select sent, then every request answered.
  for (let i = 0; !sent; i++) {
    if (i > 400) { throw new Error('Diagnosis never fired its mount-time select'); }
    await page.waitForTimeout(25);
  }
  await page.click(tab('shep-nav', 'research'));
  await page.click(tab('shep-research-nav', 'models'));
  await page.waitForSelector(tab('shep-research-nav', 'models') + '[aria-selected="true"]');
  for (let i = 0; ; i++) {
    if (i > 800) { throw new Error('the requests were never all answered'); }
    const queued = await page.evaluate((n) => window.__shepQueue.answered(n) > 0
                                        && window.__shepQueue.inflight === 0, shownIndex);
    if (answered || queued) {
      if (await page.evaluate(() => window.__shepQueue.inflight === 0)) { break; }
    }
    await page.waitForTimeout(25);
  }
  await page.click('#ask-where');
  await page.waitForFunction(() => document.querySelector('#where textarea').value !== '', null,
                             { timeout: 20000 });
  console.log(JSON.stringify({
    location: (await page.inputValue('#where textarea')).replace(/ #\d+$/, ''),
    queued: !answered }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_diagnosis_mount_select_never_writes_over_a_page_opened_after_it(tmp_path):
    """UI-14 (B46-mount): about 1.5 s after the page loads, Diagnosis' tab
    fires its select as it mounts; it wrote the location without the queue,
    so when the server answered it after Research › Models had been opened,
    the location read Diagnosis while Models was shown and Models' poll did
    nothing. Here the server answers that select 1.5 s late (a busy server);
    the location the server holds after every answer is Models, since the
    select is queued with the other location writers, in the order sent."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    demo, slow = _where_page()
    slow["seconds"] = 1.5
    with served(demo, **mount_options()) as base:
        seen = run_node(_MOUNT_RACE_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    assert seen == {"location": "models", "queued": True}, seen


#: Diagnosis' mount-time select held on its way to the server until Research ›
#: Models is opened and answered, then let go: what the server's location is
#: once every request is answered.
_MOUNT_LATE_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const page = await (await browser.newContext()).newPage();
  await page.addInitScript(QUEUE_JS);
  const cfg = await (await fetch(base + 'config')).json();
  const shownIndex = cfg.dependencies.find((d) => d.api_name === '_shown_clinical').id;
  let held = null;
  await page.route('**/queue/join**', (route) => {
    let body = {};
    try { body = JSON.parse(route.request().postData() || '{}'); } catch (e) { body = {}; }
    if (body.fn_index === shownIndex && held === null) { held = route; return; }
    route.continue();
  });
  const tab = (nav, id) => `#${nav} > .tab-wrapper [role="tab"][data-tab-id="${id}"]`;
  await page.goto(base, { waitUntil: 'load' });
  for (let i = 0; held === null; i++) {
    if (i > 400) { throw new Error('Diagnosis never sent its mount-time select'); }
    await page.waitForTimeout(25);
  }
  await page.click(tab('shep-nav', 'research'));
  await page.click(tab('shep-research-nav', 'models'));
  await page.waitForSelector(tab('shep-research-nav', 'models') + '[aria-selected="true"]');
  // Every request but the held one answered.
  await page.waitForFunction(() => window.__shepQueue.inflight === 1
                             && window.__shepQueue.direct === 0, null, { timeout: 20000 });
  await held.continue();
  await page.waitForFunction((n) => window.__shepQueue.answered(n) > 0
                             && window.__shepQueue.idle(), shownIndex, { timeout: 20000 });
  await page.click('#ask-where');
  await page.waitForFunction(() => document.querySelector('#where textarea').value !== '', null,
                             { timeout: 20000 });
  console.log(JSON.stringify({
    location: (await page.inputValue('#where textarea')).replace(/ #\d+$/, ''),
    shown: await page.evaluate(() => document.querySelector(
      '#shep-research-nav > .tab-wrapper [role="tab"][aria-selected="true"]').dataset.tabId) }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_mount_time_select_that_arrives_late_never_moves_the_location(tmp_path):
    """Verification of fcc3dea (V-RB1-1), the arrival half: Diagnosis' select
    sent as its tab mounts, held on its way until Research › Models has been
    opened and answered, wrote "clinical" when it arrived, so Models' poll
    did nothing while Models was shown. It was sent before Models' writes,
    so it no longer writes over them (`_written`). Fails on fcc3dea."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    demo, _slow = _where_page()
    with served(demo, **mount_options()) as base:
        seen = run_node(_MOUNT_LATE_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    assert seen == {"location": "models", "shown": "models"}, seen


# --------------------------------------------------------------------------- one choice, one request
def test_every_select_choice_in_the_dashboard_is_its_one_event_per_pick():
    """UI-11 (review G6): no listener of the dashboard takes a single-select
    Dropdown's ``input`` (Gradio 6.30 sends it twice for one pick and once
    for a focus that chose nothing) or its ``change`` or ``select`` as the
    operator's choice; every choice the ledger named is its blur
    (`shell.picked`). A multiselect and a Radio keep their input."""
    from src.webui.app import create_gradio_app

    demo = create_gradio_app()
    taken = {}
    for fn in demo.fns.values():
        for block_id, event in fn.targets:
            block = demo.blocks.get(block_id)
            if isinstance(block, gr.Dropdown):
                taken.setdefault(block.elem_id, set()).add((event, fn.name))
    single = {block.elem_id for block in demo.blocks.values()
              if isinstance(block, gr.Dropdown) and not block.multiselect}
    wrong = {elem: events for elem, events in taken.items()
             if elem in single and any(event == "input" for event, _name in events)}
    assert wrong == {}, wrong
    for elem in ("shep-test-workspace", "shep-test-version", "shep-test-model",
                 "shep-test-model-run", "shep-train-view-run", "shep-mp-task",
                 "shep-models-research-workspace", "shep-models-clinical-workspace"):
        assert any(event == "blur" for event, _name in taken[elem]), (elem, taken[elem])
    chosen = next(b for b in demo.blocks.values()
                  if isinstance(b, gr.Dropdown) and b.multiselect)
    assert shell.picked(chosen) == chosen.input
    radio = next(b for b in demo.blocks.values() if isinstance(b, gr.Radio))
    assert shell.picked(radio) == radio.input


def _picks_page():
    """A Dropdown and a Radio, each choice wired through `shell.picked`, each
    answer counting its calls into the page; a button writes the Dropdown's
    value from the server."""
    calls = {"select": [], "radio": []}
    with gr.Blocks() as demo:
        select = gr.Dropdown(choices=[("Alpha", "a"), ("Beta", "b"), ("Gamma", "c")], value="a",
                             label="Pick", elem_id="pick", interactive=True)
        radio = gr.Radio(choices=[("One", "1"), ("Two", "2")], value="1", label="Radio",
                         elem_id="radio")
        seen = gr.Textbox(elem_id="seen")
        write = gr.Button("Write", elem_id="write")

        def count(name):
            def fn(value):
                calls[name].append(value)
                return f"{name} {len(calls[name])}"
            fn.__name__ = f"_{name}_picked"
            return fn

        shell.picked(select)(count("select"), inputs=[select], outputs=[seen])
        shell.picked(radio)(count("radio"), inputs=[radio], outputs=[seen])
        write.click(lambda: gr.update(value="a"), inputs=None, outputs=[select])
    return demo, calls


#: Each act, then its answer drawn (the counter's words), then the requests it
#: sent counted (`QUEUE_JS`: counted when asked, so a second request one act
#: sends is counted before the first one's answer is drawn).
_PICKS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const page = await (await browser.newContext()).newPage();
  await page.addInitScript(QUEUE_JS);
  await page.goto(base, { waitUntil: 'networkidle' });
  const cfg = await page.evaluate(async (b) => (await fetch(b + 'config')).json(), base);
  const fnOf = (name) => cfg.dependencies.find((d) => d.api_name === name).id;
  const ids = { select: fnOf('_select_picked'), radio: fnOf('_radio_picked') };
  const count = (id) => page.evaluate((i) => window.__shepQueue.count(i), id);
  const seen = (words) => page.waitForFunction((w) => document.querySelector('#seen textarea').value === w,
                                               words, { timeout: 20000 });
  const out = {};
  let before = await count(ids.select);
  await page.click('#pick input');
  await page.click('ul[role=listbox] li:has-text("Beta")');
  await seen('select 1');
  out.pointer = (await count(ids.select)) - before;
  before = await count(ids.select);
  await page.click('#pick input');
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  await seen('select 2');
  out.keys = (await count(ids.select)) - before;
  out.value = await page.inputValue('#pick input');
  before = await count(ids.select);
  await page.click('#write');
  await page.waitForFunction(() => document.querySelector('#pick input').value === 'Alpha');
  await page.waitForFunction(() => window.__shepQueue.inflight === 0);
  // A listener the write had started would have been asked by now (in the
  // answer's own handling); the Radio's pick after it is the marker.
  before = { select: before, radio: await count(ids.radio) };
  await page.click('#radio label:has-text("Two")');
  await seen('radio 1');
  out.server_write = (await count(ids.select)) - before.select;
  out.radio = (await count(ids.radio)) - before.radio;
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_one_pick_of_a_select_is_one_request_and_the_pages_write_is_none(tmp_path):
    """UI-11 in Gradio 6.30's own page (Node and Playwright, skipped without):
    a Dropdown's option clicked, and one chosen with the arrow key and Enter,
    each send its listener once (``input`` sent the click's twice); the
    server's write of its value sends none (``change`` would); a Radio's pick
    sends its listener once."""
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    demo, calls = _picks_page()
    with served(demo) as base:
        seen = run_node(_PICKS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    assert seen == {"pointer": 1, "keys": 1, "value": "Gamma", "server_write": 0, "radio": 1}, seen
    assert calls == {"select": ["b", "c"], "radio": ["2"]}


# --------------------------------------------------------------------------- D01 and the shared tokens (T04)
def test_one_stage_mapping_names_the_tablist_and_feeds_the_hint_and_the_captions_are_gone():
    """N1-N3 (the stage-hint package of 2026-11-21): Prepare and Evaluate were
    captions above the research tabs, kept twice (the tablist's hand-written
    name and the style sheet's generated content on the Datasets and Test
    tabs), with a divider, a group gap and a wrap measure of their own
    (``researchLayout``, ``shep-rnav-wrapped``). One mapping (`shell.STAGES`)
    now gives each research page its stage: the tablist's name is derived
    from it, word for word as before, and the head script's hint reads it.
    The captions, the divider, the wrap measure and their tokens are gone,
    the style sheet names no stage, and the area switch (level 1) is named
    "Area", so "Stage" names one thing on the screen."""
    from src.webui.components.help import as_json_for_script
    from src.webui.style import SHELL_CSS

    assert shell.STAGES == {"datasets": "Prepare", "training": "Prepare", "finetune": "Prepare",
                            "test": "Evaluate", "models": "Evaluate"}
    assert set(shell.STAGES) == set(shell.RESEARCH_PAGES)
    assert shell.RESEARCH_TABLIST_LABEL == (
        "Research pages: Prepare (Datasets, Training, Fine-tune), Evaluate (Test, Models)")
    head = shell.SHELL_HEAD
    assert f"var STAGES = {as_json_for_script(shell.STAGES)}, STAGE_WORD = \"Stage\";" in head
    assert "title(); labels(); stage(); layout(); titles();" in head
    assert "var TABLIST_LABELS = {'shep-nav': 'Area'," in head
    # The hint's words by textContent, never markup; after the tablist, never in it.
    stage = head[head.index("function stage()"):head.index("// Row 1 holds")]
    assert "innerHTML" not in stage and "textContent" in stage
    assert "#shep-research-nav > .tab-wrapper" in stage and "row.appendChild(hint)" in stage
    assert "setAttribute('role'" not in stage and "tabindex" not in stage.lower()
    assert not re.search(r"job|gpu|chip", stage, re.I)                 # N4: no job state read
    for gone in ("Prepare", "Evaluate", "--shep-caption-h", "--shep-group-gap", "--shep-group-caption",
                 "shep-rnav-wrapped", "#shep-page-test-button", "#shep-page-datasets-button",
                 "--shep-tab-hover-line"):
        assert gone not in SHELL_CSS, gone
    for gone in ("researchLayout", "shep-rnav-wrapped", "--shep-group-gap", "'Stage'"):
        assert gone not in head, gone
    # The row wraps by itself: no window-size rule decides where the hint goes.
    hint = _rule(SHELL_CSS, "#shep-research-nav > .tab-wrapper > .shep-stage")
    assert "margin: 0 16px 0 auto" in hint and "cursor: default" in hint
    assert "flex-wrap: wrap" in _rule(SHELL_CSS, "#shep-research-nav > .tab-wrapper")
    assert "min-width: 0" in _rule(SHELL_CSS, "#shep-research-nav > .tab-wrapper > .tab-container")


#: The dashboard as the service mounts it, at 1850 x 960 then 420 x 900 and
#: back: the research row's geometry, stage hint and cues, level 1's selected
#: look against a form's chosen radio, the accordion's chevron, the header
#: chips, and on every page the type scale and every field's border, as
#: drawn (`tests/fixtures/browser.py`, LOOK_JS). It waits for the page's own
#: events to be answered (QUEUE_JS), never for a fixed time.
_SHELL_LOOK_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__, LOOK_JS = __LOOK_JS__;
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await page.addInitScript(QUEUE_JS);
  await page.addInitScript(LOOK_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.inflight === 0
                                          && window.__shepQueue.joins.every((j) => j.event_id !== null),
                                          null, { timeout: 30000 });
  const open = async (id) => {
    if (id === 'clinical' || id === 'settings') {
      await page.click(`#shep-tab-${id}-button`);
      await page.waitForSelector(`#shep-tab-${id}-button[aria-selected="true"]`);
    } else {
      if (!(await page.$('#shep-tab-research-button[aria-selected="true"]'))) {
        await page.click('#shep-tab-research-button');
      }
      await page.click(`#shep-page-${id}-button`);
      await page.waitForSelector(`#shep-page-${id}-button[aria-selected="true"]`);
    }
    await idle();
  };
  const rect = (sel) => page.evaluate((s) => { const r = document.querySelector(s).getBoundingClientRect();
    return { left: r.left, top: r.top, right: r.right, bottom: r.bottom, height: r.height }; }, sel);
  const css = (sel, props, pseudo) => page.evaluate(([s, p, ps]) => {
    const el = document.querySelector(s); if (!el) { return null; }
    const c = getComputedStyle(el, ps || null), o = {};
    for (const k of p) { o[k] = c.getPropertyValue(k); }
    return o; }, [sel, props, pseudo || null]);
  const row = async () => {
    const tabs = {};
    for (const id of ['datasets', 'training', 'finetune', 'test', 'models']) {
      tabs[id] = await rect(`#shep-page-${id}-button`);
    }
    return page.evaluate((tabs) => {
      const shell = document.getElementById('shep-shell');
      const wrap = document.querySelector('#shep-research-nav > .tab-wrapper');
      const box = (e) => { const r = e.getBoundingClientRect();
        return { left: r.left, top: r.top, right: r.right, bottom: r.bottom, height: r.height }; };
      const hint = wrap.querySelector(':scope > .shep-stage');
      const ws = getComputedStyle(wrap);
      return { tabs, row: box(wrap), hint: hint && box(hint), margin: hint && parseFloat(getComputedStyle(hint).marginRight),
               end: box(wrap).right - parseFloat(ws.paddingRight) - parseFloat(ws.borderRightWidth),
               listPad: parseFloat(getComputedStyle(wrap.querySelector('[role="tablist"]')).paddingRight),
               tabFont: parseFloat(getComputedStyle(document.getElementById('shep-page-training-button')).fontSize),
               generated: [...wrap.querySelectorAll('[role="tab"]')].flatMap((t) => ['::before', '::after'].map((p) =>
                 getComputedStyle(t, p).content)).filter((c) => !['none', 'normal', '""'].includes(c)),
               headerAndTabs: wrap.getBoundingClientRect().bottom - shell.getBoundingClientRect().top,
               row1: document.querySelector('#shep-nav > .tab-wrapper').getBoundingClientRect().height,
               hScroll: document.documentElement.scrollWidth > document.documentElement.clientWidth };
    }, tabs);
  };
  const out = {};
  await page.goto(base, { waitUntil: 'load' });
  await page.waitForSelector('#shep-chips .shep-chip', { timeout: 60000 });
  await idle();
  out.chips = await page.evaluate(() => [...document.querySelectorAll('#shep-chips .shep-chip')].map((c) => {
    const s = getComputedStyle(c), d = getComputedStyle(c.querySelector('.shep-dot') || c);
    return { state: c.getAttribute('data-state'), background: s.backgroundColor, border: s.borderTopColor,
             dot: d.backgroundColor, ring: d.borderTopColor }; }));
  await open('settings');
  await page.mouse.move(2, 950);
  out.settingsSelected = await css('#shep-tab-settings-button', ['background-color', 'color', 'font-weight']);
  await open('training');
  await page.mouse.move(2, 950);
  out.wide = await row();
  const TAB = ['background-color', 'color', 'font-weight', 'border-bottom-color', 'border-bottom-width'];
  out.level1 = await css('#shep-tab-research-button', ['background-color', 'color', 'font-weight']);
  out.radio = await css('#device label.selected', ['background-color', 'color', 'border-top-color']);
  out.radioRest = await css('#device label:not(.selected)', ['background-color', 'color']);
  out.rest = await css('#shep-page-test-button', TAB);
  out.selected = await css('#shep-page-training-button', TAB);
  await page.hover('#shep-page-test-button');
  out.hover = await css('#shep-page-test-button', TAB);
  await page.mouse.move(2, 950);
  await page.focus('#shep-page-training-button');
  await page.keyboard.press('Shift');                    // keyboard focus: :focus-visible
  out.focus = await page.evaluate(() => { const s = getComputedStyle(document.activeElement);
    return { id: document.activeElement.id, outline: s.outlineStyle + ' ' + s.outlineWidth + ' ' + s.outlineColor }; });
  await page.click('#shep-page-training-button');
  await idle();
  const chevron = () => page.evaluate(() => {
    const b = [...document.querySelectorAll('#shep-page-training button.label-wrap')]
      .find((x) => x.offsetParent);
    return { open: b.classList.contains('open'), transform: getComputedStyle(b.querySelector('.icon')).transform };
  });
  out.closed = await chevron();
  await page.locator('#shep-page-training button.label-wrap:visible').first().click();
  await page.waitForFunction(() => [...document.querySelectorAll('#shep-page-training button.label-wrap')]
    .some((x) => x.offsetParent && x.classList.contains('open')));
  out.opened = await page.evaluate(async () => {
    const icon = [...document.querySelectorAll('#shep-page-training button.label-wrap.open')][0]
      .querySelector('.icon');
    await Promise.all(icon.getAnimations().map((a) => a.finished));   // its turn, finished
    return { open: true, transform: getComputedStyle(icon).transform }; });
  const frame = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  await page.setViewportSize({ width: 420, height: 900 });
  await frame();
  out.narrow = await row();
  await page.setViewportSize({ width: 1850, height: 960 });
  await frame();
  out.back = await row();
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""

#: What a read-only viewer's Settings and header draw (no deployment mode):
#: the look at 1850 x 960; then at 1850 x 960 and 390 x 844, Runtime
#: settings' accordions open (the external review of 6a2d52e measured the
#: reason 734 px from the buttons so), where D8's reason and the fields'
#: line are, and each locked action brought into view as a reader brings it
#: (`scrollIntoView` nearest): how far below the reason it is and whether
#: both are wholly on screen (inside the window and every box that clips).
_READ_ONLY_LOOK_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__, LOOK_JS = __LOOK_JS__;
(async () => {
  const browser = await chromium.launch();
  const out = { sizes: {} };
  for (const [w, h] of [[1850, 960], [390, 844]]) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h } })).newPage();
    await page.addInitScript(QUEUE_JS);
    await page.addInitScript(LOOK_JS);
    await page.goto(process.argv[2], { waitUntil: 'load' });
    await page.waitForSelector('#shep-chips .shep-chip-readonly', { timeout: 60000 });
    await page.click('#shep-tab-settings-button');
    await page.waitForSelector('#allocator_preset input:disabled', { timeout: 30000 });
    // The page's open answered: D8's reason drawn for this request.
    await page.waitForSelector('#runtime_access_reason .shep-reason', { state: 'visible', timeout: 30000 });
    await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
    if (w === 1850) {
      out.look = await page.evaluate(() => {
        const s = (sel) => { const c = getComputedStyle(document.querySelector(sel));
          return { background: c.backgroundColor, border: c.borderTopColor, color: c.color }; };
        return { chip: s('#shep-chips .shep-chip-readonly'),
                 fields: window.__shepLook.fields(document.getElementById('shep-tab-settings')),
                 select: s('#allocator_preset .wrap:has(> .wrap-inner)'), value: s('#allocator_preset input'),
                 box: s('#torch_compile input[type=checkbox]'), label: s('#torch_compile label') };
      });
    }
    for (const acc of await page.$$('#shep-settings button.label-wrap:not(.open)')) { await acc.click(); }
    await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
    out.sizes[w + 'x' + h] = await page.evaluate(async () => {
      const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      // The accordions opened: their transitions finished, so nothing moves while it is read.
      await Promise.all(document.getAnimations().map((a) => a.finished));
      await frame();
      const T = (e) => (e.innerText || '').replace(/\s+/g, ' ').trim();
      const box = (e) => { const r = e.getBoundingClientRect();
        return { top: r.top, bottom: r.bottom, left: r.left, right: r.right }; };
      const clip = (e) => { let x = { left: 0, top: 0, right: innerWidth, bottom: innerHeight };
        for (let a = e.parentElement; a; a = a.parentElement) { const s = getComputedStyle(a);
          if (/(auto|scroll|hidden|clip)/.test(s.overflowY + s.overflowX)) { const p = box(a);
            x = { left: Math.max(x.left, p.left), top: Math.max(x.top, p.top),
                  right: Math.min(x.right, p.right), bottom: Math.min(x.bottom, p.bottom) }; } }
        return x; };
      const whole = (e) => { const q = box(e), x = clip(e);
        return q.left >= x.left - 1 && q.top >= x.top - 1 && q.right <= x.right + 1 && q.bottom <= x.bottom + 1; };
      const reason = document.querySelector('#runtime_access_reason .shep-reason');
      const fields = document.querySelector('#runtime_fields_lock .shep-reason');
      const actions = {};
      for (const id of ['runtime_apply', 'runtime_restart', 'runtime_stop']) {
        const b = document.getElementById(id);
        b.scrollIntoView({ block: 'nearest' });
        await frame();
        actions[id] = { text: T(b), disabled: b.disabled, below: box(b).top - box(reason).bottom,
                        whole: whole(b), reasonWhole: whole(reason), at: box(b), clip: clip(b) };
      }
      const panel = document.getElementById('shep-settings');
      return { reason: T(reason), inFooter: !!document.querySelector('#shep-settings-ft #runtime_access_reason'),
               fields: fields && T(fields), fieldsAt: fields && box(fields),
               facts: box(document.getElementById('runtime_facts')),
               allocator: box(document.getElementById('allocator_preset')), actions,
               said: (panel.innerText.match(/Operator controls only/g) || []).length };
    });
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _rgb(hex_colour):
    return "rgb({}, {}, {})".format(*(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)))


def test_the_research_navigation_and_shared_tokens_as_drawn(tmp_path, monkeypatch):
    """D01 and UI-10 in a real browser (Node and Playwright, skipped without),
    the dashboard mounted as the service mounts it, for an operator:

    - **the research row** (the stage-hint package, N1 and N6; it was D01 /
      R-22+R-37's captions over the tabs): at 1850 x 960 no caption tier and
      no generated text on a tab (less room above the tabs than a line of
      their text), the research tabs 32 px, the stage hint inside the row at
      its right, and at 420 px too, on the tabs' line where the row has room
      for both and on a line of its own where it has not
      (`_hint_placed_by_the_rows_room`: the room as drawn decides, never the
      window); back at 1850 as at first;
    - **cues** (N7): a tab under the pointer has a fill it lacks at rest and
      no line under it (its bottom border as at rest), so the selected tab
      alone has the accent line, the accent text and the strong weight;
      keyboard focus a 2 px ring in the focus token;
    - **R-20**: level 1's selected item (Research, and Settings) is the accent
      fill with white text; a form's chosen radio (Training's Device) is not;
    - **R-31**: a closed accordion's chevron points right, an open one down;
    - **R-21**: a normal header chip is plain text, its mark in a status token.
    The type scale (R-17), one field-label style per page (R-16) and every
    field's border at 3:1 (R-8), on every page at every representative size,
    are the shared check's (`test_every_page_at_every_size_passes_the_ui_measurement_check`)."""
    from src.webui import style
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import LOOK_JS, QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.operator_access import operator_mode
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_mode(monkeypatch)
    script = (_SHELL_LOOK_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__LOOK_JS__", json.dumps(LOOK_JS)))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")])
    tokens = dict(re.findall(r"(--shep-[a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{6})", style.SHELL_CSS))

    wide = seen["wide"]
    tabs = wide["tabs"].values()
    assert {round(t["height"]) for t in tabs} == {32}
    assert wide["generated"] == [] and not wide["hScroll"]
    assert min(t["top"] for t in tabs) - wide["row"]["top"] < wide["tabFont"]          # no caption tier
    assert wide["headerAndTabs"] - wide["row1"] <= wide["row"]["bottom"] - wide["row"]["top"] + 0.5
    narrow = seen["narrow"]
    assert not narrow["hScroll"] and narrow["generated"] == []
    for row in (wide, narrow, seen["back"]):  # the hint inside the row at its right, its line by the row's room
        assert row["hint"] is not None, ("no stage hint drawn", row)
        _hint_placed_by_the_rows_room({**row, "tabs": list(row["tabs"].values())})
    def drawn_as(row):                       # the row and its hint, against the row's own box
        hint, box = row["hint"], row["row"]
        return (row["headerAndTabs"], hint["top"] - box["top"], hint["left"] - box["left"],
                hint["right"] - box["right"], box["bottom"] - box["top"])
    assert drawn_as(seen["back"]) == drawn_as(wide)

    rest, hover, selected = seen["rest"], seen["hover"], seen["selected"]
    assert hover["background-color"] != rest["background-color"]
    assert hover["border-bottom-color"] == rest["border-bottom-color"]                 # no line under it
    assert hover["font-weight"] == rest["font-weight"] != selected["font-weight"]
    assert hover["color"] != selected["color"]
    assert selected["border-bottom-color"] == _rgb(tokens["--shep-accent"])
    assert selected["color"] == _rgb(tokens["--shep-accent-text"]) and selected["font-weight"] == "600"
    assert seen["focus"]["id"] == "shep-page-training-button"
    assert seen["focus"]["outline"] == f"solid 2px {_rgb(tokens['--shep-accent-600'])}"

    accent = _rgb(tokens["--shep-accent"])
    for level1 in (seen["level1"], seen["settingsSelected"]):
        assert level1["background-color"] == accent and level1["color"] == "rgb(255, 255, 255)"
    assert seen["radio"]["background-color"] != accent
    assert seen["radio"]["color"] != "rgb(255, 255, 255)"
    assert seen["radio"]["border-top-color"] == accent           # chosen: its border and dot

    assert seen["closed"]["transform"] == "matrix(0, -1, 1, 0, 0, 0)" and not seen["closed"]["open"]
    assert seen["opened"]["transform"] == "none"

    normal = [c for c in seen["chips"] if c["state"] in ("on", "none")]
    assert normal, seen["chips"]
    for chip in normal:
        assert chip["background"] == "rgba(0, 0, 0, 0)" and chip["border"] == "rgba(0, 0, 0, 0)"
    on = next(c for c in seen["chips"] if c["state"] == "on")
    assert on["dot"] == _rgb(tokens["--shep-green-800"])


def test_a_read_only_viewer_sees_disabled_fields_and_the_read_only_lozenge(tmp_path, monkeypatch):
    """NEW-settings-datasets-footer-3 and R-21, in a real browser for a request
    D8 refuses (no deployment mode): Settings' allocator select and
    torch.compile checkbox are drawn off (the disabled fill, border and grey
    words, the select's value at 4.5:1 on it), not like the enabled ones an
    operator sees; the header's read-only chip is the amber lozenge.

    Where Runtime settings says why (the external review of 6a2d52e): the
    "Operator controls only: …" reason is said once in the panel, beside the
    actions it turns off, by the one gated-button rule: the actions' row's
    line just above Apply Settings, Restart Backend and Stop service, each
    within 120 px under it and both wholly on screen when the action is
    brought into view, at 1850 x 960 and 390 x 844 with the accordions open;
    next to the fields, between the facts and the allocator select, a short
    relation to it (`FIELDS_LOCKED`). Fails on 6a2d52e: the reason was that
    body line, 734 px above the buttons at 1850 x 960."""
    from src.webui import style
    from src.webui.app import create_gradio_app, mount_options
    from src.webui.components import runtime_settings as rs
    from tests.fixtures.browser import LOOK_JS, QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.operator_access import DEPLOYMENT_MODE
    from tests.fixtures.served_page import served
    from tests.unit.test_webui_button_contrast import contrast

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    monkeypatch.delenv(DEPLOYMENT_MODE, raising=False)
    script = (_READ_ONLY_LOOK_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__LOOK_JS__", json.dumps(LOOK_JS)))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")])
    tokens = dict(re.findall(r"(--shep-[a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{6})", style.SHELL_CSS))
    off, line, words = (tokens["--shep-grey-50"], tokens["--shep-grey-200"], tokens["--shep-grey-600"])
    look = seen["look"]
    assert look["select"]["background"] == _rgb(off) and look["select"]["border"] == _rgb(line)
    assert look["value"]["color"] == _rgb(words) and contrast(words, off) >= 4.5
    assert look["box"]["background"] == _rgb(off) and look["box"]["border"] == _rgb(line)
    assert look["label"]["color"] == _rgb(words)
    assert look["chip"]["background"] == _rgb(tokens["--shep-amber-50"])
    assert look["chip"]["color"] == _rgb(tokens["--shep-amber-800"])
    assert any(f["disabled"] for f in look["fields"])
    reason = ["🔒", *"Operator controls only: deployment mode not set to M1.".split()]
    for size, at in seen["sizes"].items():
        assert at["reason"].split() == reason and at["inFooter"] and at["said"] == 1, (size, at)
        for name, action in at["actions"].items():
            assert action["text"].startswith("🔒 ") and action["disabled"], (size, name, action)
            assert 0 <= action["below"] <= 120, (size, name, action)
            assert action["whole"] and action["reasonWhole"], (size, name, action)
        # Next to the fields: the relation, between the facts and the select.
        fields, allocator = at["fieldsAt"], at["allocator"]
        assert at["fields"] == f"🔒 {rs.FIELDS_LOCKED}", (size, at)
        assert at["facts"]["bottom"] <= fields["top"] + 0.5, (size, at)
        assert fields["bottom"] <= allocator["top"] + 0.5 <= fields["bottom"] + 24, (size, at)


#: Settings' scroll region (its level-1 tab item, side by side) with the
#: evidence folders box placed partly under the region's foot, as a walk or a
#: reader's scrolling can leave it, then focused as a keyboard focuses it:
#: where the box is and what is over its centre, at each overlap.
_PARTLY_HIDDEN_FOCUS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1366, height: 620 } })).newPage();
  await page.addInitScript(QUEUE_JS);
  await page.goto(process.argv[2], { waitUntil: 'load' });
  await page.waitForSelector('#shep-chips .shep-chip', { timeout: 60000 });
  await page.click('#shep-tab-settings-button');
  const box = '#dl_box_training_evidence_dirs textarea';
  await page.waitForSelector(box + ':enabled', { timeout: 30000 });
  await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
  await page.keyboard.press('Tab');
  const seen = [];
  for (const overlap of [6, 10, 20, 30]) {
    seen.push(await page.evaluate(async ([sel, overlap]) => {
      const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const ta = document.querySelector(sel);
      let region = ta.parentElement;
      while (region && !(/(auto|scroll)/.test(getComputedStyle(region).overflowY)
                         && region.scrollHeight > region.clientHeight + 1)) { region = region.parentElement; }
      if (document.activeElement) { document.activeElement.blur(); }
      const foot = region.getBoundingClientRect().bottom;
      region.scrollTop += ta.getBoundingClientRect().top - (foot - overlap);
      await frame();
      const placed = foot - ta.getBoundingClientRect().top;
      ta.focus();
      await frame();
      const r = ta.getBoundingClientRect();
      const top = document.elementsFromPoint(r.left + r.width / 2, r.top + r.height / 2)[0];
      return { overlap, placed, region: region.id || region.className, foot, top: r.top, bottom: r.bottom,
               focused: document.activeElement === ta,
               over: top === ta || ta.contains(top) ? 'itself' : (top.closest('[id]') || {}).id };
    }, [box, overlap]));
  }
  console.log(JSON.stringify(seen));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_box_partly_under_a_pages_region_foot_is_focused_whole_into_view(tmp_path, monkeypatch):
    """The shared walk's open problem on Settings at 1366 x 620 (T06-D): side by
    side, Settings scrolls in its level-1 tab item, and a text box partly under
    that region's foot was focused by bringing its first line just into view,
    its centre left under the runtime line below ("focused
    #dl_box_training_evidence_dirs textarea under #runtime_footer"): the
    caret's room under what a scroll region brings into view (R-25) was set on
    the page and on the panels, not on a page's own region. In a real
    browser, for an operator, the evidence folders box placed 6 to 30 px
    into the region's foot and focused is wholly inside the region, nothing
    over its centre. On the code before (5137694) it stays at 561-619 px,
    under the runtime line at 588."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.operator_access import DEPLOYMENT_MODE
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    monkeypatch.setenv(DEPLOYMENT_MODE, "M1")
    script = _PARTLY_HIDDEN_FOCUS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")])
    for s in seen:
        assert s["region"] == "shep-tab-settings" and abs(s["placed"] - s["overlap"]) < 1, s
        assert s["focused"] and s["over"] == "itself", s
        assert s["bottom"] <= s["foot"] + 0.5, s


# --------------------------------------------------------------------------- D02 and D03 (T04)
def test_one_layout_decision_keys_every_style_sheet_and_no_window_size_gate_is_left():
    """D02 (FL-UI-02): side by side against stacked was decided by
    ``(min-width: 1400px) and (min-height: 760px)``, written in 13 media
    queries and 2 ``matchMedia`` calls over 8 modules, so 1280 x 900 and
    1920 x 700 stacked and 1440 x 1600 (portrait) did not. The shell's layout
    pass now makes one decision (the window's shape, then whether the page's
    panels fit by their own ``--shep-panel-min``) and writes it on the root
    element; every style sheet and the two scripts read that attribute. No
    media query or ``matchMedia`` on the window's size decides it any more
    (the media queries left are content breakpoints and reduced motion).
    What the browser does with it is
    `test_one_layout_decision_at_every_size_and_back`'s."""
    from src.webui.app import dashboard_head
    from src.webui.components import models_page, training_console
    from src.webui.style import LAYOUT_ATTR, SHORT_BELOW

    head = dashboard_head()
    for css in (head, models_page.SCROLL_JS, training_console._BANNER_JS):
        assert not re.search(r"@media[^{]*(min-width: 1400px|max-width: 1399px|min-height: 760px|"
                             r"max-height: 759px|max-height: 819px|max-height: 899px)", css), css[:60]
        assert "matchMedia" not in css
    assert "document.documentElement.getAttribute('data-shep-layout')" in models_page.SCROLL_JS
    assert "document.documentElement.getAttribute('data-shep-layout') === 'side'" in training_console._BANNER_JS
    js = shell.SHELL_HEAD
    assert f"var LAYOUT = '{LAYOUT_ATTR}'" in js and f"SHORT_BELOW = {SHORT_BELOW}" in js
    assert "var want = w > h && sideFits(shownTab()) ? 'side' : 'stack';" in js
    # Each panel's minimum read in px whatever unit its page writes (M4, the UI sweep of
    # 2026-11-22): the property is registered as a length, and where a browser has no @property
    # a rem or em value is converted by the font size (`lengthPx`), never by a probe.
    assert "lengthPx(kids[j], '--shep-panel-min')" in js and "getPropertyValue(prop)" in js
    assert "@property --shep-panel-min { syntax: '<length>'; inherits: false; initial-value: 0px; }" in head
    for page_css in (head,):
        assert side(page_css) and stack(page_css)
    # Every page with panels side by side says what its panels need.
    from src.webui import app

    for css in app.PAGE_CSS[:6]:
        assert "--shep-panel-min:" in css or "--shep-mp-list-min:" in css, css[:80]
    # Nor does a module still describe the gate, or the sticky card R-25
    # removed (the T04 review: Diagnosis', Models' and the charts' docstrings
    # still named 1400 x 760, and the load confirmation's comment 820 px).
    stale = re.compile(r"1400 [×x] 760|wide at 1400|below 820 px|sticky at its bottom")
    for path in sorted(Path(shell.__file__).parent.rglob("*.py")):
        assert not stale.search(path.read_text(encoding="utf-8")), path


#: What the layout pass decided at a stop of the shared walk (`ui_walk`), on
#: the page shown: the decision and the short flag, what the decision measured
#: (the panels' own minimum widths with their gaps, against the room), how the
#: panels sit, the page's own scroll both ways, and Training's run bar.
_LAYOUT_FACTS_JS = r"""() => {
  const { shownPage, words } = window.__shepLook;
  const { box, panels: kids } = shownPage();
  const panels = kids.map((k) => ({ min: parseFloat(getComputedStyle(k).getPropertyValue('--shep-panel-min')) || 0,
                                    top: Math.round(k.getBoundingClientRect().top), w: k.getBoundingClientRect().width }))
    .filter((p) => p.min > 0);
  const gap = box ? parseFloat(getComputedStyle(box).columnGap) || 0 : 0;
  const se = document.scrollingElement;
  const bar = document.getElementById('shep-train-runbar');
  let runbar = null;
  if (bar && bar.getClientRects().length) {
    // The run bar's items as drawn: one not drawn (Stop while no job runs, R-39; Continue… and
    // Stop while the drawer is open) has no box, and no line.
    const items = ['shep-train-dataset', 'shep-train-chip', 'run_name', 'shep-train-start',
                   'shep-train-continue', 'shep-train-stop', 'shep-train-pill']
      .map((i) => document.getElementById(i).getBoundingClientRect())
      .filter((r) => r.width > 0 && r.height > 0);
    // On one line: one horizontal band crosses every item (the lowest top above the highest
    // bottom), whatever each item's own height.
    runbar = { h: Math.round(bar.getBoundingClientRect().height), items: items.length,
               oneLine: items.length > 0 && Math.max(...items.map((r) => r.top)) < Math.min(...items.map((r) => r.bottom)) };
  }
  // A box's top and bottom as drawn, or null when it is not drawn.
  const at = (e) => { if (!e) { return null; } const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 ? [Math.round(r.top), Math.round(r.bottom)] : null; };
  // Training's Continue drawer, open: its body, step 3's row (Check checkpoint and its prompt),
  // and what the run bar draws of its actions and says beside them (R-48, R-49,
  // NEW-training-finetune-2), the reason read by the walk's `words` (never Gradio's status
  // tracker in its block, which showed a queued event's elapsed time, "1.5s", under load).
  let drawer = null;
  const dr = document.getElementById('shep-train-drawer');
  if (dr && dr.getClientRects().length) {
    drawer = { body: at(document.getElementById('shep-train-drawer-bd')),
               check: at(document.getElementById('shep-train-check')),
               prompt: at(document.getElementById('shep-train-check-prompt')),
               actions: ['shep-train-start', 'shep-train-continue', 'shep-train-stop']
                 .filter((i) => at(document.getElementById(i))),
               reason: words(document.getElementById('shep-train-reason')),
               closers: [...dr.querySelectorAll('button')].filter((b) => at(b)
                 && /^(close|cancel)$/i.test((b.getAttribute('aria-label') || b.textContent).trim())).length };
  }
  // Fine-tune's form with nothing expanded: what its body leaves to scroll, and the Device,
  // Run name and "More settings" header against the body's bottom and the footer row's top
  // (B27-1, NEW-training-finetune-5); its job panel's Stop (NEW-training-finetune-7).
  let finetune = null;
  const fb = document.getElementById('ft_form_bd');
  if (fb && fb.getClientRects().length) {
    finetune = { scroll: fb.scrollHeight - fb.clientHeight, bottom: at(fb)[1],
                 footer: at(document.getElementById('ft_start_ft'))[0],
                 device: at(document.getElementById('ft_device')),
                 runName: at(document.getElementById('ft_run_name')),
                 more: at(document.querySelector('#ft_more > button')),
                 stop: at(document.getElementById('ft_stop')) };
  }
  return { layout: document.documentElement.getAttribute('data-shep-layout'),
           short: document.documentElement.hasAttribute('data-shep-short'),
           need: panels.reduce((a, p) => a + p.min, 0) + Math.max(0, panels.length - 1) * gap,
           width: box ? box.getBoundingClientRect().width + innerWidth - document.documentElement.clientWidth : 0,
           tops: panels.map((p) => p.top), narrowest: Math.min(...panels.map((p) => p.w - p.min), 0),
           overX: se.scrollWidth - se.clientWidth, overY: se.scrollHeight - innerHeight, runbar,
           drawer, finetune };
}"""

#: The walk's windows, narrow to wide and back on each page: the shared
#: check's representative sizes (`UI_SIZES`), checked, and the layout's other
#: shapes (a narrow column, a square, a portrait screen), laid out only;
#: 1536 x 760, short and the owner's laptop, and 1280 x 600, a short laptop's
#: window, checked too (side by side below about 660 px of height, Models
#: scrolled its list around the rows' own region and Training's plots' body was
#: shorter than one plot: the T04 review).
_WALK_SIZES = ((760, 1000, False), (1850, 960, True), (1280, 900, True), (1920, 700, True),
               (1440, 1600, True), (1000, 1000, False), (1536, 760, True), (1366, 620, True),
               (1280, 600, True), (1080, 1920, False), (960, 1080, True), (390, 844, True),
               (1850, 960, False))

#: The Diagnosis load drawer, opened at a stop of the walk (`browser.UI_WALK_JS`), a
#: row of its list chosen (the served model's, unfolded and located), so its
#: details show beside the list (UI-07: two regions, each its own scroll).
_CHANGE_DRAWER = {"click": "#shep-dx-change-btn", "shown": "#shep-dx-change",
                  "close": "#shep-dx-change-close button",
                  "pick": "#shep-models-clinical-rows tr[data-key]:not([hidden])",
                  "picked": "#shep-models-clinical-facts [data-facts-key]"}
#: Training's Continue drawer, opened at a stop of the walk; its one closer is
#: the × in its header (R-49).
_CONTINUE_DRAWER = {"click": "#shep-train-continue", "shown": "#shep-train-drawer",
                    "close": "#shep-train-drawer-close"}
#: The drawers the layout walks open (`_layout_walk_stops`), as a reader first sees each: the
#: Diagnosis Load drawer (no row chosen), and where a run and a registry let them open
#: (configured), Training's Continue drawer and the Test page's import drawer.
_LOAD_DRAWER = {"click": "#shep-dx-change-btn", "shown": "#shep-dx-change",
                "close": "#shep-dx-change-close button"}
_IMPORT_DRAWER = {"click": "#shep-test-import-open", "shown": "#shep-test-import",
                  "close": "#shep-test-import-close"}
#: The windows a drawer is opened at in a layout walk: the owner's, a laptop, a portrait column
#: and a phone (where the drawers' foot, header and fields change their layout).
_DRAWER_SIZES = ((1850, 960), (1366, 768), (960, 1080), (390, 844))


#: The windows the configured walk opens the Load drawer at with a row chosen (its Model Facts
#: beside the list, or under it on a phone), last, so the drawer's other stops see no choice.
_PICKED_SIZES = ((1850, 960), (390, 844))


def _layout_walk_stops(configured: bool):
    """**A state's layout walk**: every page at `browser.LAYOUT_SIZES` (`layout_stops`), then
    its drawers open at `_DRAWER_SIZES` (the Load drawer in every state; Continue and import
    where `configured`), each stop's layout rules recorded, no focus walk. Configured, the
    Load drawer is opened last with a row chosen (`_CHANGE_DRAWER`, at `_PICKED_SIZES`): its
    chosen model's facts start on the drawer's edge (they sat 13 px in, in boxes of their own,
    at 390 x 844)."""
    from tests.fixtures.browser import LAYOUT_SIZES, UI_PAGES, layout_stops

    drawers = [("clinical", _LOAD_DRAWER)] + ([("training", _CONTINUE_DRAWER), ("test", _IMPORT_DRAWER)]
                                             if configured else [])
    picked = ([{"size": [w, h], "page": "clinical", "open": _CHANGE_DRAWER, "focus": False}
               for w, h in _PICKED_SIZES] if configured else [])
    return layout_stops(UI_PAGES, LAYOUT_SIZES) + [
        {"size": [w, h], "page": page, "open": drawer, "focus": False}
        for w, h in _DRAWER_SIZES for page, drawer in drawers] + picked

#: A static page for the shared focus walk itself (`browser.LOOK_JS`'s
#: `focusWalk`): two buttons that draw a 3 px ring on keyboard focus, with no
#: transition, the first the walk's first Tab press lands on, and one that
#: draws no cue; the walk run after that one Tab press, as `UI_WALK_JS` runs it.
_FOCUS_WALK_PAGE_JS = r"""
const { chromium } = require('playwright');
const LOOK_JS = __LOOK_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 800, height: 400 } });
  await page.setContent(`<!doctype html><html><head><style>
    body { background: #fff; }
    button { outline: none; border: 1px solid #767676; background: #fff; width: 120px; height: 32px; }
    button.ring:focus-visible { outline: 3px solid rgb(67, 74, 88); }
  </style></head><body><button class="ring" id="first">First</button>
    <button class="ring" id="second">Second</button><button id="bare">No cue</button></body></html>`);
  await page.addScriptTag({ content: LOOK_JS });
  await page.keyboard.press('Tab');
  const tabbed = await page.evaluate(() => document.activeElement.id);
  const seen = await page.evaluate(() => window.__shepLook.focusWalk(null));
  const after = await page.evaluate(() => document.activeElement === document.body);
  console.log(JSON.stringify({ tabbed, seen, after }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_focus_walk_reads_a_controls_cue_from_it_unfocused(tmp_path):
    """The T06 review: the shared walk presses Tab once and then walks every
    control, reading each one's look before focusing it; the control that Tab
    had focused (Training's Continue drawer's Close, its first control) was
    read while focused, so its 3 px ring changed nothing and read "none"
    (`ui_problems`: "focus cue under 3:1: #shep-train-drawer-close "Close"
    none"), failing the check on a cue the page draws. Each control's cue is
    what its focus changes from it unfocused: the first ring measures as the
    second does, and a control with no cue still reads "none"."""
    from tests.fixtures.browser import LOOK_JS, node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    seen = run_node(_FOCUS_WALK_PAGE_JS.replace("__LOOK_JS__", json.dumps(LOOK_JS)), tmp_path,
                    timeout=120)
    assert seen["tabbed"] == "first", seen
    assert seen["seen"]["n"] == 3 and seen["seen"]["covered"] == [], seen
    assert seen["seen"]["cues"] == ['#bare "No cue" none'], seen
    assert seen["after"], seen                     # focus left as the walk found the page


def _walk_stops():
    from tests.fixtures.browser import UI_PAGES

    stops = [{"size": [w, h], "page": page, "check": checked}
             for page in UI_PAGES for w, h, checked in _WALK_SIZES]
    for stop in stops[len(_WALK_SIZES) - 1::len(_WALK_SIZES)]:
        stop["options"] = True                   # each page's selects, once, at its last stop
    # The Diagnosis load drawer open, its head a field row (D03): side by side
    # from 1280 px its workspace select was squeezed to 26 px, its label cut
    # to "W" (the T04 review), and to 71 px on a phone; and in a short window;
    # and at the owner's window, its list and details side by side (UI-07).
    stops += [{"size": [w, h], "page": "clinical", "open": _CHANGE_DRAWER}
              for w, h in ((1850, 960), (1280, 900), (1366, 620), (390, 844))]
    # Training's Continue drawer open, at the owner's window (step 3 in view
    # without scrolling: NEW-training-finetune-2) and stacked.
    stops += [{"size": [w, h], "page": "training", "open": _CONTINUE_DRAWER}
              for w, h in ((1850, 960), (960, 1080))]
    # A page opened at a size, not only resized to it: at 1280 x 900 Datasets
    # (its own minimum is wider) between two pages that fit.
    stops += [{"size": [1280, 900], "page": page, "check": False}
              for page in ("training", "datasets", "finetune")]
    return stops


@pytest.fixture(scope="module")
def ui_world(tmp_path_factory, _suite_gpu_lock):
    """The configured service, once for this module: in a process of its own
    (`served_page.served_service`, mode M1, an operator) on the demo
    workspace with a base and a disease child trained on it where the server
    lists them (`trained_children.train_children`, 1 epoch each on the CPU),
    the disease child served (the Diagnosis card as an operator sees it).
    Yields its page and its folder; the walks of this state (`ui_walk`,
    `ui_walk_configured`) share it."""
    from tests.fixtures.browser import node_with_playwright
    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.served_page import served_service
    from tests.fixtures.trained_children import train_children

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    root = tmp_path_factory.mktemp("ui-walk")
    demo = write_demo_workspace(root / "workspaces" / "demo")
    served = train_children(demo, ("disease",))["disease"]
    (root / "registry").mkdir()
    settings = {"SHEPHERD_WORKSPACES_ROOT": str(root / "workspaces"),
                "SHEPHERD_REGISTRY": str(root / "registry"), "SHEPHERD_DEPLOYMENT_MODE": "M1",
                "SHEPHERD_DEVICE": "cpu", "SHEPHERD_KG_PATH": str(demo / "kg.json"),
                "SHEPHERD_CHECKPOINT_PATH": str(served), "SHEPHERD_DATA_DIR": str(demo)}
    with served_service(root / "service", settings) as page:
        yield {"page": page, "root": root}


@pytest.fixture(scope="module")
def ui_walk(ui_world):
    """The shared UI measurement walk on the configured service (`ui_world`):
    one browser through `_walk_stops` (`browser.walk_ui`), the layout facts at
    every stop, the shared check and a screenshot at the checked ones (under
    the module's ``ui-walk*/shots``)."""
    from tests.fixtures.browser import walk_ui

    root = ui_world["root"]
    (root / "shots").mkdir()
    return walk_ui(ui_world["page"], root, _walk_stops(), extra_js=_LAYOUT_FACTS_JS,
                   shots=root / "shots")


@pytest.fixture(scope="module")
def ui_walk_configured(ui_world):
    """**The configured walk at the layout sizes** (the sweep's integration, its
    decision 4): every page of the configured service (`ui_world`) at
    `browser.LAYOUT_SIZES` and its drawers open (`_layout_walk_stops`), the
    layout rules recorded at each stop (no focus walk), screenshots under the
    module's ``ui-walk*/layout/shots``."""
    from tests.fixtures.browser import walk_ui

    root = ui_world["root"] / "layout"
    (root / "shots").mkdir(parents=True)
    return walk_ui(ui_world["page"], root, _layout_walk_stops(True), shots=root / "shots")


def test_one_layout_decision_at_every_size_and_back(ui_walk):
    """D02 (FL-UI-02, PTR-LAYOUT in part) in a real browser (Node and
    Playwright, skipped without), the service on the demo workspace, for an
    operator, every page through narrow, wide, short, portrait, square and
    phone windows and back (the shared walk, `ui_walk`):

    - no wider than tall (1440 x 1600, 1000 x 1000, 1080 x 1920, 760 x 1000,
      960 x 1080, 390 x 844): every page stacked, its panels one under the
      other;
    - wider: side by side exactly when the page's panels fit by their own
      minimum widths (1280 x 900: every page but Datasets, whose table and
      digest grid need about 1400 px, whether resized to it or opened at it
      between two pages that fit; 1920 x 700, 1366 x 620 and 1280 x 600: every
      page but Datasets below its width, a short window only denser,
      ``data-shep-short``); side by side, the panels on one line and
      none narrower than its minimum, and the page itself never scrolls;
    - never a horizontal page scroll;
    - back at 1850 x 960, the same as the first time; there the Training run
      bar is one row (R-38).
    Nothing covering a focused control, at every size, is the shared check's
    (`test_every_page_at_every_size_passes_the_ui_measurement_check`)."""
    portrait = {"1440x1600", "1000x1000", "1080x1920", "760x1000", "960x1080", "390x844"}
    # Side by side exactly when the shown panels' own minimums and gaps fit the row (relative:
    # a page's panels sized by what they hold, the UI sweep's M4, move no pin here). Datasets at
    # 1280 x 900, 1366 x 620 and 1280 x 600 stacked by that measure when this was written.
    datasets_stacked = {"1280x900", "1366x620", "1280x600"}
    for s in ui_walk:
        where, facts = (s["size"], s["page"]), s["extra"]
        assert facts["overX"] == 0, (where, facts)
        assert facts["short"] == (int(s["size"].split("x")[1]) < 900), (where, facts)
        narrow_datasets = s["page"] == "datasets" and s["size"] in datasets_stacked
        want = "stack" if s["size"] in portrait or facts["need"] > facts["width"] + 0.5 else "side"
        assert facts["layout"] == want, (where, facts)
        if want == "side":
            assert facts["need"] <= facts["width"] + 0.5, (where, facts)
            assert len(set(facts["tops"])) <= 1 and facts["narrowest"] >= -0.5, (where, facts)
            assert facts["overY"] <= 0, (where, facts)
        elif len(facts["tops"]) > 1:
            assert facts["tops"] == sorted(facts["tops"]) and len(set(facts["tops"])) == len(facts["tops"]), (where, facts)
        if narrow_datasets and want == "stack":
            assert facts["need"] > facts["width"], s                  # its own measure, no global gate
    wide = [s for s in ui_walk if s["size"] == "1850x960" and not s.get("open")]
    # The layout's facts, not a page's own (`drawer`, `finetune`: their test's).
    layout = lambda facts: {k: v for k, v in facts.items() if k not in ("drawer", "finetune")}  # noqa: E731
    for page in {s["page"] for s in wide}:
        first, *_, last = [layout(s["extra"]) for s in wide if s["page"] == page]
        assert first == last, (page, first, last)
    # Opened at a size, not only resized to it: each page decided by its own panels as drawn.
    assert [s["page"] for s in ui_walk[-3:]] == ["training", "datasets", "finetune"]
    for s in ui_walk[-3:]:
        facts = s["extra"]
        assert facts["layout"] == ("stack" if facts["need"] > facts["width"] + 0.5 else "side"), s
    # R-38: at the owner's window the Training run bar is one row (it was two, 119 px): every
    # item drawn (Dataset, its chip, Run name, Start, Continue…, the pill) on one line.
    (training, *_) = [s for s in wide if s["page"] == "training"]
    assert training["extra"]["runbar"]["items"] >= 5 and training["extra"]["runbar"]["oneLine"], training


def test_every_page_at_every_size_passes_the_ui_measurement_check(ui_walk):
    """D08 (INT-AUDIT): the shared UI measurement check (`browser.check`,
    `browser.focusWalk`, `browser.ui_problems`) on every page at every
    representative window (`browser.UI_SIZES`: 1850 x 960, 1280 x 900,
    1920 x 700, a 1366 x 620 short laptop, 1440 x 1600, 960 x 1080, a
    390 x 844 phone; and 1536 x 760 and 1280 x 600), and with the Diagnosis
    load drawer open at 1850 x 960, 1280 x 900, 1366 x 620 and 390 x 844, a
    row of it chosen (its focus walk inside the drawer), the service on the
    demo workspace for an operator (`ui_walk`):

    - no horizontal page scroll; one scroll region per area (none inside
      another, the page included; never two in one panel, or in one area a
      page marks in it: the drawer's list and its details, each its own);
    - every control drawn, focused as a keyboard focuses it, in view with
      nothing over it (another control's own overlay, a help bubble or an
      open list it leaves behind, aside), its focus cue at 3:1;
    - no text under 14 px, only the type scale's sizes and weights, nothing
      upper case; every enabled field's border at 3:1; one field-label style
      per page, and no label cut by its own field;
    - on a research page the stage hint in its page's words (`STAGE_HINTS`),
      inside the research pages' row, over no tab, not cut, outside the
      tablist, at 14 px or more and nothing a control would be (the
      stage-hint package, N2, N5, N6); on Diagnosis and Settings none;
    - no choice or select option with more than one " · ".
    It failed before T04's measure part on the browser's own focus ring (the
    accent buttons' a faint grey) on every page and on Diagnosis' Top-K and
    Explanations ⓘ, which their own bubble covered (since the Options grid,
    b405973); on c1e57c6 it also fails on the captions on the tabs' line, the
    type scale and the field borders. On eafa8a2 it fails on the drawer's
    Workspace label (its select squeezed to 26 px at 1280 x 900, 71 px at
    390 x 844), on Models' list scrolling around its rows' own region
    (1366 x 620, 1280 x 600) and on Training's focused plot with its centre
    under the resources line (1280 x 600)."""
    from tests.fixtures.browser import UI_PAGES, UI_SIZES, ui_problems

    checked = [s for s in ui_walk if "check" in s]
    assert {(s["page"], s["size"]) for s in checked} >= {
        (page, f"{w}x{h}") for page in UI_PAGES for w, h in UI_SIZES}
    assert ui_problems(ui_walk) == []
    # What the check looked at, so that an empty list means something.
    for s in checked:
        assert s["focus"]["n"] >= (5 if s.get("open") else 10), (s["page"], s["size"], s["focus"])
        research = s["page"] not in ("clinical", "settings")
        assert (s["check"]["stage"] is not None) == research, (s["page"], s["size"])
    options = {s["page"]: s["options"] for s in ui_walk if "options" in s}
    assert all(options[page] for page in ("training", "finetune", "test", "models", "settings")), options
    drawer = [s for s in checked if s.get("open") == "#shep-dx-change"]
    assert {s["size"] for s in drawer} == {"1850x960", "1280x900", "1366x620", "390x844"}
    for s in drawer:                                  # its head's select measured
        assert "Workspace" in {label["text"] for label in s["check"]["labels"]}, s
    labels = {s["page"]: {label["text"] for label in s["check"]["labels"]}
              for s in checked if s["size"] == "1850x960" and not s.get("open")}
    assert labels["clinical"] >= {"Model in use", "Options", "HPO terms"}
    assert labels["training"] >= {"Dataset", "Run name", "Epochs"}


def test_the_continue_drawer_and_the_fine_tune_form_fit_the_owners_window(ui_walk):
    """The shared walk's facts (`_LAYOUT_FACTS_JS`) for Training's Continue
    drawer, opened at 1850 x 960 and stacked at 960 x 1080, and Fine-tune's
    form with nothing expanded at 1850 x 960:

    - R-49: with the drawer open the run bar draws none of its actions
      (Start, Continue…, Stop) and says no reason (it kept a disabled Start
      beside "Start disabled: close Continue… to start a new run."); the
      drawer has one closer (its × and its footer's Cancel did the same);
    - NEW-training-finetune-2: at 1850 x 960 step 3's Check checkpoint and
      its prompt are inside the drawer's body as it opens (its row was cut
      at the body's bottom edge, the prompt under it, while the start was
      enabled);
    - B27-1 and NEW-training-finetune-5: Fine-tune's form leaves nothing to
      scroll, and its Device, Run name and "More settings" header are inside
      its body, above the footer row (the body scrolled 183 px; Device and
      Run name at 925-1017 under a body ending at 846, the header cut);
    - NEW-training-finetune-7: with no Fine-tune job its Stop is not drawn
      (a disabled Stop with no reason)."""
    opened = {s["size"]: s["extra"]["drawer"] for s in ui_walk
              if s.get("open") == "#shep-train-drawer"}
    assert set(opened) == {"1850x960", "960x1080"}
    for size, drawer in opened.items():
        assert drawer["actions"] == [] and drawer["reason"] == "", (size, drawer)
        assert drawer["closers"] == 1, (size, drawer)
    wide = opened["1850x960"]
    top, bottom = wide["body"]
    for part in ("check", "prompt"):
        assert wide[part] and top <= wide[part][0] and wide[part][1] <= bottom, (part, wide)
    forms = [s["extra"]["finetune"] for s in ui_walk
             if s["page"] == "finetune" and s["size"] == "1850x960"]
    assert forms
    for form in forms:
        assert form["scroll"] <= 0, form
        for part in ("device", "runName", "more"):
            assert form[part] and form[part][1] <= form["bottom"] <= form["footer"], (part, form)
        assert form["stop"] is None, form


# --------------------------------------------------------------------------- PTR-LAYOUT and INT-AUDIT (T06-D)
#: Pages whose one fingerprint is said twice on purpose, with the reason, in
#: the shared walk's repeat count (`browser.repeat_problems`): Datasets'
#: selected workspace (its table row and its details beside it: B51-h,
#: recorded, not changed here); Models' model in use (its row and the Load
#: box's "in use for diagnosis" line, the page the shared model list's lane
#: changes). Every other screen says each fingerprint once.
_SAID_TWICE = {"datasets": 2, "models": 2}

#: Diagnosis' largest Top-K (20) on the long-named workspace
#: (`registered_cohort.write_long_named_workspace`: 25 diseases, 20 of them
#: with names of 120 to 190 characters).
_LONG_DIAGNOSIS = {"hpo": "HP:0001250\nHP:0001263", "top_k": 20, "rows": 20}

#: What the acceptance walk reads at each stop besides the shared check: the
#: header's first chip (D8's), the layout decision, Diagnosis' results as
#: drawn (whether the table has a box, its rows, those with a long name, and
#: its width against its box's), the card's Change model… as drawn (its
#: words, whether it is disabled, and the reason beside it if one is drawn),
#: and whether the card is folded (`style.FOLDED_ATTR`, OPS-FIX-N1).
#: Every text is read by the walk's `words` (`browser.LOOK_JS`), never Gradio's
#: status tracker (V-24).
_ACCEPTANCE_FACTS_JS = r"""() => {
  const { words } = window.__shepLook;
  const chip = document.querySelector('#shep-chips .shep-chip');
  const table = document.querySelector('#shep-dx-results-body table');
  const rows = table ? [...table.querySelectorAll('tbody tr[data-key]')] : [];
  const change = document.querySelector('#shep-dx-change-btn');
  const why = document.querySelector('#shep-dx-change-why');
  const card = document.querySelector('#shep-dx-card');
  const drawn = (e) => !!e && e.getClientRects().length > 0;
  return { chip: chip ? words(chip) : null,
           layout: document.documentElement.getAttribute('data-shep-layout'),
           shown: drawn(table),
           rows: rows.length, long: rows.filter((r) => /, type \d+/.test(words(r))).length,
           table: table ? [table.scrollWidth, table.parentElement.clientWidth] : null,
           change: drawn(change) ? { text: words(change), off: change.disabled,
                                     why: drawn(why) ? words(why) : '' } : null,
           folded: drawn(card) && card.hasAttribute('__FOLDED__') };
}""".replace("__FOLDED__", shell.FOLDED_ATTR)


def _acceptance_stops():
    """Every page at PLAN_TEST_RESULTS §6's sizes the representative windows
    leave out (`browser.PTR_SIZES`), then Diagnosis answered with its largest
    Top-K at the owner's window, the acceptance sizes, a short laptop, a
    narrow column and a phone (the diagnosis asked at the first of them)."""
    from tests.fixtures.browser import PTR_SIZES, UI_PAGES

    stops = [{"size": [w, h], "page": page} for page in UI_PAGES for w, h in PTR_SIZES]
    answered = [{"size": [w, h], "page": "clinical", "tag": "results"}
                for w, h in ((1850, 960), *PTR_SIZES, (1366, 620), (960, 1080), (390, 844))]
    answered[0]["diagnose"] = _LONG_DIAGNOSIS
    return stops + answered


#: The read-only viewer's windows: on a high-DPI screen (2 device pixels per
#: CSS pixel) the owner's window, the acceptance sizes a high-DPI screen gives
#: (a 3840 x 2160 one at 200 % is 1920 x 1080) and a phone; at 1 the two
#: largest, whose screens are their CSS size.
_READ_ONLY_HIGH_DPI = ((1850, 960), (1920, 1080), (1280, 720), (1440, 900), (1707, 870), (1080, 1920),
                       (960, 540), (390, 844))
_READ_ONLY_LARGE = ((2560, 1600), (3840, 2160))


def _read_only_stops(sizes):
    """Every page, for a viewer D8 refuses, at `sizes`; then Diagnosis
    answered with its largest Top-K on the long names at each of them (the
    diagnosis asked at the first): the diagnose route asks for no operator,
    and such a viewer's Diagnosis is drawn apart (Change model locked, its
    reason beside it)."""
    from tests.fixtures.browser import UI_PAGES

    stops = [{"size": [w, h], "page": page, "tag": "readonly"} for page in UI_PAGES for w, h in sizes]
    answered = [{"size": [w, h], "page": "clinical", "tag": "results"} for w, h in sizes]
    answered[0]["diagnose"] = _LONG_DIAGNOSIS
    return stops + answered


def _answered_problems(answered, *, refused=False):
    """What is wrong with Diagnosis' answered stops (`_ACCEPTANCE_FACTS_JS`):
    each draws the results table with the 20 results, 16 or more with a long
    name, and side by side the table is no wider than its box. Change model…
    is drawn by the gated-button rule: for a viewer D8 refuses (`refused`),
    disabled with the lock and "Operator controls only: …" beside it; for an
    operator, usable, with no lock and no reason."""
    from src.webui.components.chips import LOCK, OPERATOR_ONLY

    problems = []
    for s in answered:
        facts = s["extra"]
        if not facts["shown"] or facts["rows"] != 20 or facts["long"] < 16:
            problems.append((s["size"], "rows", facts))
        if facts["layout"] == "side" and facts["table"][0] > facts["table"][1] + 1:
            problems.append((s["size"], "table wider than its box", facts))
        change = facts["change"] or {}
        locked = (change.get("off") is True and change.get("text", "").startswith(LOCK)
                  and change.get("why", "").startswith(f"{LOCK} {OPERATOR_ONLY}"))
        unlocked = (change.get("off") is False and not change.get("text", LOCK).startswith(LOCK)
                    and change.get("why") == "")
        if not (locked if refused else unlocked):
            problems.append((s["size"], "Change model", change))
    return problems


@pytest.fixture(scope="module")
def acceptance_walk(tmp_path_factory, _suite_gpu_lock):
    """PLAN_TEST_RESULTS §6's layout acceptance, the parts the shared walk
    (`ui_walk`) leaves out (PTR-LAYOUT, INT-AUDIT), on the same check: the
    service (mode M1) on the long-named workspace, a base and its disease
    child trained on it and the child served; one browser as an operator
    through `_acceptance_stops` (Diagnosis answered with 20 long-named
    candidates), and one as a read-only viewer, every request forwarded
    (`browser.READ_ONLY_HEADERS`: D8 refuses it, the page as such a viewer
    sees it), through `_read_only_stops` (every page, then Diagnosis answered
    with the same 20 long-named candidates) at `_READ_ONLY_HIGH_DPI` on a
    high-DPI screen (2 device pixels per CSS pixel) and at `_READ_ONLY_LARGE`.
    Screenshots only when ``SHEPHERD_UI_SHOTS`` names a folder (its ``shots``
    and ``shots-read-only``): they are evidence for a person, no assertion
    reads them, and a 3840 x 2160 page's capture beside the full suite's
    other browsers has crashed Chromium's renderer mid-walk."""
    import os

    from tests.fixtures.browser import READ_ONLY_HEADERS, node_with_playwright, walk_ui
    from tests.fixtures.registered_cohort import write_long_named_workspace
    from tests.fixtures.served_page import served_service
    from tests.fixtures.trained_children import train_children

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    root = tmp_path_factory.mktemp("acceptance-walk")
    demo = write_long_named_workspace(root / "workspaces" / "demo")
    served = train_children(demo, ("disease",))["disease"]
    for name in ("registry", "operator", "read-only", "read-only-large"):
        (root / name).mkdir()
    keep = os.environ.get("SHEPHERD_UI_SHOTS")
    shots = Path(keep) / "shots" if keep else None
    shots_read_only = Path(keep) / "shots-read-only" if keep else None
    for folder in (shots, shots_read_only):
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
    settings = {"SHEPHERD_WORKSPACES_ROOT": str(root / "workspaces"),
                "SHEPHERD_REGISTRY": str(root / "registry"), "SHEPHERD_DEPLOYMENT_MODE": "M1",
                "SHEPHERD_DEVICE": "cpu", "SHEPHERD_KG_PATH": str(demo / "kg.json"),
                "SHEPHERD_CHECKPOINT_PATH": str(served), "SHEPHERD_DATA_DIR": str(demo)}
    with served_service(root / "service", settings) as page:
        operator = walk_ui(page, root / "operator", _acceptance_stops(),
                           extra_js=_ACCEPTANCE_FACTS_JS, shots=shots)
        read_only = walk_ui(page, root / "read-only", _read_only_stops(_READ_ONLY_HIGH_DPI),
                            extra_js=_ACCEPTANCE_FACTS_JS, shots=shots_read_only,
                            headers=READ_ONLY_HEADERS, device_scale=2)
        read_only += walk_ui(page, root / "read-only-large", _read_only_stops(_READ_ONLY_LARGE),
                             extra_js=_ACCEPTANCE_FACTS_JS, shots=shots_read_only,
                             headers=READ_ONLY_HEADERS)
    return {"operator": operator, "read_only": read_only}


def test_the_acceptance_sizes_and_twenty_long_named_results_pass_the_ui_measurement_check(
        acceptance_walk):
    """PTR-LAYOUT: every page at 1440 x 900, 1280 x 720, 1707 x 870,
    1920 x 1080, 2560 x 1600, 3840 x 2160, 1080 x 1920 and 960 x 540 (200 %
    zoom), and Diagnosis with its largest Top-K answered (20 candidates, 16
    of them with a long name) at those, at 1850 x 960, 1366 x 620,
    960 x 1080 and 390 x 844, pass the shared check (`browser.ui_problems`: no sideways page scroll, one
    scroll region per area, every control focused in view with nothing over
    it, the type scale, borders and focus cues). Side by side the results
    table is no wider than its box: a long name's 46-character token widened
    it to 723 px in a 562 px box at 1440 x 900 (699 at 1707 x 870, 520 at
    1366 x 620), Phenotypes and Evidence out of view in its sideways scroll,
    whose bar was under the twentieth row; stacked, a table wider than its
    panel scrolls sideways in its own box (the layout rule). Change model…
    is usable, with no lock and no reason (a read-only viewer's is locked:
    `test_a_read_only_viewers_walk_passes_the_ui_measurement_check`)."""
    from tests.fixtures.browser import PTR_SIZES, UI_PAGES, ui_problems

    seen = acceptance_walk["operator"]
    assert {(s["page"], s["size"]) for s in seen if "check" in s} >= {
        (page, f"{w}x{h}") for page in UI_PAGES for w, h in PTR_SIZES}
    assert ui_problems(seen) == []
    answered = [s for s in seen if s.get("tag") == "results"]
    assert len(answered) == 4 + len(PTR_SIZES)
    assert _answered_problems(answered) == []


def test_a_read_only_viewers_walk_passes_the_ui_measurement_check(acceptance_walk):
    """INT-AUDIT and PTR-LAYOUT: the shared walk as a viewer D8 refuses
    (every request forwarded by a proxy, on a service in mode M1): every page
    at the owner's window, every acceptance size and a phone, the smaller on
    a high-DPI screen, passes the shared check, the D8 chip says
    read-only and why, and the walk focused the controls such a viewer can
    use (the disabled ones are not focusable). Diagnosis answered with 20
    long-named results passes it too at each of those sizes, its results
    table no wider than its box side by side, Change model… locked with its
    reason beside it: such a viewer can diagnose (the diagnose route asks for
    no operator), and their page differs from an operator's by that lock."""
    from tests.fixtures.browser import UI_PAGES, ui_problems

    seen = acceptance_walk["read_only"]
    sizes = (*_READ_ONLY_HIGH_DPI, *_READ_ONLY_LARGE)
    assert {(s["page"], s["size"]) for s in seen if "check" in s} == {
        (page, f"{w}x{h}") for page in UI_PAGES for w, h in sizes}
    assert ui_problems(seen) == []
    for s in seen:
        assert s["extra"]["chip"].startswith("Read-only: forwarded by a proxy"), (s["page"], s["extra"])
        assert s["focus"]["n"] >= 3, (s["page"], s["size"], s["focus"])
    answered = [s for s in seen if s.get("tag") == "results"]
    assert sorted(s["size"] for s in answered) == sorted(f"{w}x{h}" for w, h in sizes)
    assert _answered_problems(answered, refused=True) == []


#: The acceptance walk's reader (`_ACCEPTANCE_FACTS_JS`) on Diagnosis while the
#: card's poll is in flight (V-24), for an operator and then a read-only viewer
#: (`browser.READ_ONLY_HEADERS`), at 1280 x 720: the page served and its card
#: read, then every request a ``queue=False`` listener sends (the card's 5 s
#: poll) held until the page has been read, and the page read once Gradio's
#: status tracker in Change model's reason block shows its timer: the reader's
#: facts, the block's whole ``innerText`` and the reason the page drew in it.
_HELD_POLL_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__, LOOK_JS = __LOOK_JS__, FACTS = __FACTS_JS__, VIEWERS = __VIEWERS__;
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [viewer, headers] of Object.entries(VIEWERS)) {
    const context = await browser.newContext({ viewport: { width: 1280, height: 720 },
                                               ...(headers ? { extraHTTPHeaders: headers } : {}) });
    const page = await context.newPage();
    await page.addInitScript(QUEUE_JS);
    await page.addInitScript(LOOK_JS);
    await page.goto(process.argv[2], { waitUntil: 'load' });
    // The card read: Change model… usable, or locked with its reason beside it.
    await page.waitForSelector('#shep-dx-change-btn:not([disabled]), #shep-dx-change-why .shep-dx-why',
                               { timeout: 60000 });
    await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 60000 });
    let release = null;
    const released = new Promise((resolve) => { release = resolve; });
    await page.route('**/run/predict', async (route) => { await released; await route.continue().catch(() => {}); });
    await page.waitForFunction(() => {
      const tracker = document.querySelector('#shep-dx-change-why [data-testid="status-tracker"]');
      return !!tracker && /\d\.\ds/.test(tracker.textContent); }, null, { timeout: 30000 });
    const facts = await page.evaluate(FACTS);
    const block = await page.evaluate(() => {
      const why = document.getElementById('shep-dx-change-why'), own = why.querySelector('.shep-dx-why');
      return { text: why.innerText.trim(), own: own ? own.innerText.trim() : '' }; });
    release();
    await page.unroute('**/run/predict');
    await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 60000 });
    out[viewer] = { facts, block };
    await context.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_walks_reader_never_reads_gradios_status_tracker_as_a_blocks_words(tmp_path, monkeypatch):
    """V-24: the acceptance walk once read Change model's reason as "0.2s" at
    1280 x 720. While the card's 5 s poll (``queue=False``, its progress
    hidden) is in flight, Gradio keeps its status tracker in each block the
    poll writes, hidden (opacity 0, a 0 px box) but rendered, and the
    tracker's timer is in the block's ``innerText``. With the poll held in
    flight and the timer in the reason's block (`_HELD_POLL_JS`; the panel,
    its wiring and the shell real, the served status and the card's facts
    stand-ins), the walk's reader reads the reason the page drew, by the
    walk's one ``words`` (`browser.LOOK_JS`): none for an operator, the
    lock's words alone for a read-only viewer, never the timer.

    A tracker Gradio draws is no such case (the review of 6a14d6b, R3-2): the
    reader skipped every tracker, so a shown timer or a failed event's red
    "Error" in a block read as nothing, and a check of that block passed. Read
    on a page of its own (`_DRAWN_TRACKER_JS`), a drawn tracker's words are
    read like any other: the timer beside a block's own words, the timer
    alone, and "Error". Fails on 6a14d6b ("Own words", "", "")."""
    from src.webui.app import mount_options
    from src.webui.components import diagnosis_panel as dp
    from src.webui.components.chips import LOCK, OPERATOR_ONLY
    from tests.fixtures.browser import (
        LOOK_JS,
        QUEUE_JS,
        READ_ONLY_HEADERS,
        node_with_playwright,
        run_node,
    )
    from tests.fixtures.operator_access import operator_mode
    from tests.fixtures.served_page import served
    from tests.unit.test_diagnosis_panel import _status_serving, _task_identity

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_mode(monkeypatch)
    status = _status_serving(_task_identity("disease_task"))
    monkeypatch.setattr(dp.card, "read_status_within", lambda *a, **k: (status, None))
    monkeypatch.setattr(dp.card, "served_facts", lambda s, *, operator, **k: {
        "key": dp.card.served_key(s), "operator": operator, "workspaces": [], "serving": None,
        "item": None, "item_problem": None, "recorded": None, "trained": None, "cohorts": None})
    with gr.Blocks() as demo:
        shell.build_shell(demo, {**{page: (lambda ctx: None)
                                    for page in (*shell.RESEARCH_PAGES, shell.SETTINGS)},
                                 shell.CLINICAL: dp.create_diagnosis_tab})
    script = (_HELD_POLL_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__LOOK_JS__", json.dumps(LOOK_JS))
              .replace("__FACTS_JS__", _ACCEPTANCE_FACTS_JS)
              .replace("__VIEWERS__", json.dumps({"operator": None, "read_only": READ_ONLY_HEADERS})))
    with served(demo, **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")])
    for viewer, read in seen.items():
        # Gradio's timer is in the block's innerText while the poll is held.
        assert re.search(r"\d\.\ds", read["block"]["text"]), (viewer, read)
    operator, read_only = seen["operator"], seen["read_only"]
    assert operator["block"]["own"] == "", operator
    assert operator["facts"]["change"] == {"text": "Change model…", "off": False, "why": ""}, operator
    change = read_only["facts"]["change"]
    assert read_only["block"]["own"].startswith(f"{LOCK} {OPERATOR_ONLY}"), read_only
    assert change["off"] is True and change["text"] == f"{LOCK} Change model…", read_only
    assert change["why"] == read_only["block"]["own"], read_only
    # A tracker Gradio draws is read like any other words (the review of
    # 6a14d6b, R3-2): `_DRAWN_TRACKER_JS` on a page of its own.
    with gr.Blocks() as drawn:
        own = gr.HTML("<p>Own words</p>", elem_id="shep-t-own")
        only = gr.HTML("", elem_id="shep-t-only")
        gr.Button("Go", elem_id="shep-t-go").click(
            lambda: ("<p>Own words</p>", ""), None, [own, only], show_progress="full", queue=False)
        failed = gr.HTML("", elem_id="shep-t-err")

        def fail():
            raise gr.Error("The reason could not be read")

        gr.Button("Fail", elem_id="shep-t-fail").click(fail, None, [failed], show_progress="full")
    script = (_DRAWN_TRACKER_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__LOOK_JS__", json.dumps(LOOK_JS)))
    with served(drawn) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")])
    for block, read in seen.items():
        # Drawn: not hidden, opaque, with a box, its words in view.
        assert read["tracker"]["hide"] is False and read["tracker"]["opacity"] == "1", (block, read)
        assert min(read["tracker"]["box"]) > 0, (block, read)
    assert re.search(r"\d\.\ds", seen["own"]["words"]) and "Own words" in seen["own"]["words"], seen
    assert re.search(r"\d\.\ds", seen["only"]["words"]), seen
    assert "Error" in seen["err"]["words"], seen


#: A drawn status tracker, read by the walk's ``words`` (the review of
#: 6a14d6b, R3-2): at 1280 x 720, a block whose own words stay beside the
#: tracker of an event in flight (``show_progress="full"``, its request held),
#: a block whose only drawn words are that tracker's timer, then a block whose
#: event failed, Gradio's red "Error" in its tracker. Each block's tracker (its
#: ``hide`` class, opacity, box) and what ``words`` reads of the block.
_DRAWN_TRACKER_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__, LOOK_JS = __LOOK_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 720 } })).newPage();
  await page.addInitScript(QUEUE_JS);
  await page.addInitScript(LOOK_JS);
  await page.goto(process.argv[2], { waitUntil: 'load' });
  await page.waitForSelector('#shep-t-go');
  await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 60000 });
  const shownWith = (ids, pattern) => page.waitForFunction(([ids, pattern]) => ids.every((id) => {
    const t = document.querySelector(`#${id} [data-testid="status-tracker"]`);
    return !!t && !t.classList.contains('hide') && getComputedStyle(t).opacity === '1'
      && new RegExp(pattern).test(t.textContent); }), [ids, pattern], { timeout: 30000 });
  const read = (id) => page.evaluate((id) => {
    const el = document.getElementById(id), t = el.querySelector('[data-testid="status-tracker"]');
    const box = t.getBoundingClientRect();
    return { words: window.__shepLook.words(el), text: el.innerText.trim(),
             tracker: { hide: t.classList.contains('hide'), opacity: getComputedStyle(t).opacity,
                        box: [Math.round(box.width), Math.round(box.height)] } };
  }, id);
  const out = {};
  let release = null;
  const released = new Promise((resolve) => { release = resolve; });
  await page.route('**/run/predict', async (route) => { await released; await route.continue().catch(() => {}); });
  await page.click('#shep-t-go');
  await shownWith(['shep-t-own', 'shep-t-only'], '\\d\\.\\ds');
  out.own = await read('shep-t-own');
  out.only = await read('shep-t-only');
  release();
  await page.unroute('**/run/predict');
  await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 60000 });
  await page.click('#shep-t-fail');
  await shownWith(['shep-t-err'], 'Error');
  out.err = await read('shep-t-err');
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_each_fingerprint_is_said_once_on_a_screen(ui_walk):
    """INT-AUDIT's repeat count (the skill's §1.2 and §12, B27-5), on the
    shared walk (`ui_walk`): on every page at every size each fingerprint
    drawn is said once in the page's visible text, except where
    `_SAID_TWICE` gives the second place's reason. The Diagnosis load drawer
    is counted apart: its dataset fingerprint is said by its workspace line
    and by each row on this base (B27-5; the shared model list's lane says
    it once)."""
    from tests.fixtures.browser import repeat_problems

    counted = [s for s in ui_walk if "repeats" in s and s.get("open") != "#shep-dx-change"]
    assert {s["page"] for s in counted if s["repeats"]} >= {
        "clinical", "datasets", "training", "finetune", "test", "models"}
    assert repeat_problems(counted, _SAID_TWICE) == []


def test_each_fingerprint_is_said_once_with_twenty_results_and_for_a_read_only_viewer(
        acceptance_walk):
    """The same repeat count on the acceptance walk: Diagnosis answered (the
    result's own identity said against the model in use where they differ,
    not again where they are the same: the result's model fingerprint and the
    card's once), every page at the acceptance sizes, and every page a
    read-only viewer sees. Where the card is folded (OPS-FIX-N1, option b: a
    short window, 1366 x 620 and the read-only 1280 x 720 here) its
    fingerprint is in its Details, one click away, and the result says
    "Produced by the model in use": the screen draws no fingerprint, and the
    count is kept by the stops where the card is whole
    (`test_diagnosis_panel`'s short-window test opens Details to the
    fingerprint)."""
    from tests.fixtures.browser import repeat_problems

    seen = acceptance_walk["operator"] + acceptance_walk["read_only"]
    answered = [s for s in seen if s.get("tag") == "results"]
    whole = [s for s in answered if not s["extra"]["folded"]]
    assert whole and all(s["repeats"] for s in whole)
    assert any(s["extra"]["folded"] for s in answered)
    assert repeat_problems(seen, _SAID_TWICE) == []


#: The four field rows' boxes and actions as drawn (D03): Fine-tune's base
#: picker and Test's Run, at 1850 x 960 and at 1440 x 900 (where Fine-tune's
#: two selects wrap, Checkpoint under Run).
_ROWS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await page.addInitScript(QUEUE_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.inflight === 0
                                          && window.__shepQueue.joins.every((j) => j.event_id !== null),
                                          null, { timeout: 30000 });
  await page.goto(process.argv[2], { waitUntil: 'load' });
  await page.waitForSelector('#shep-chips .shep-chip', { timeout: 60000 });
  await idle();
  await page.click('#shep-tab-research-button'); await idle();
  const out = {};
  for (const [w, h] of [[1850, 960], [1440, 900]]) {
    await page.setViewportSize({ width: w, height: h });
    for (const [name, fields, button] of [['finetune', ['ft_run', 'ft_checkpoint'], 'ft_refresh'],
                                          ['test', ['shep-test-model-run'], 'shep-test-model-reload']]) {
      await page.click(`#shep-page-${name}-button`); await idle();
      await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
      out[`${name}@${w}`] = await page.evaluate(([fields, button]) => {
        const r = (e) => { const b = e.getBoundingClientRect(); return [+b.top.toFixed(1), +b.bottom.toFixed(1), +b.left.toFixed(1)]; };
        return { boxes: fields.map((f) => r(document.querySelector(`#${f} .wrap:has(> .wrap-inner)`))),
                 button: r(document.getElementById(button)) };
      }, [fields, button]);
    }
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_field_rows_action_is_its_boxes_height_and_stands_on_their_line(tmp_path, monkeypatch):
    """D03 (FL-UI-03): Gradio draws a select as a padded block, its box the
    block's last line; the rows lined their small action button up with the
    block's outer bottom (Fine-tune's Refresh 15 px under the boxes' centre,
    Test's ↻ patched by 4 px and still 12 px off). The shared field row
    (`style.py`'s ``.shep-field-row``, two tokens) makes the action the box's
    height on the box's line, an icon-only one a square of it; with the
    selects wrapped the button stays beside the last box. Fine-tune's form
    is two columns at 1850 x 960 (B27-1), its base's selects wrapped in the
    left one, and one column at 1440 x 900, the selects side by side.
    Measured in a browser (Node and Playwright, skipped without)."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.operator_access import operator_mode
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_mode(monkeypatch)
    script = _ROWS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")])
    for where, row in seen.items():
        last = max(row["boxes"], key=lambda b: b[0])
        assert abs(row["button"][0] - last[0]) <= 1 and abs(row["button"][1] - last[1]) <= 1, (where, row)
        assert row["button"][2] > max(b[2] for b in row["boxes"] if b[0] == last[0]), (where, row)
    assert seen["finetune@1440"]["boxes"][0][0] == seen["finetune@1440"]["boxes"][1][0]
    assert seen["finetune@1850"]["boxes"][1][0] > seen["finetune@1850"]["boxes"][0][1]   # wrapped
    test = seen["test@1850"]["button"]
    assert abs((test[1] - test[0]) - (seen["test@1850"]["boxes"][0][1] - seen["test@1850"]["boxes"][0][0])) <= 1


def test_the_field_rows_share_one_class_and_two_tokens(built):
    """D03: the four rows of a select and an action (Fine-tune's base picker,
    Training's Continue drawer, Test's Run, the Diagnosis load drawer's head)
    carry the one field-row class; their own alignment patches are gone; a
    dense form's blocks (the shared `shep-form-dense`: Training's parameters
    and its drawer, Fine-tune's form) set the row's padding token with their
    padding (one value, never two that drift). Test's Run row is a bar of
    whole groups since the sweep (L3): its select and ↻ are that bar's one
    group, and that group is the field row; Diagnosis' Task segment keeps
    the aside class beside a whole-label one (P2). Both are read from the
    built page, not from their source's words."""
    from src.webui.components import finetune_page, model_list, test_page, training_console
    from src.webui.style import SHELL_CSS

    assert "--shep-field-box-h: 41px; --shep-field-pad-b: 10px;" in SHELL_CSS
    button = _rule(SHELL_CSS, ".shep-field-row > button")
    assert "height: var(--shep-field-box-h) !important" in button
    assert "margin: 0 0 var(--shep-field-pad-b) !important" in button
    assert '"shep-ft-parent-row", "shep-field-row"' in Path(finetune_page.__file__).read_text()
    demo, _ctx = built
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}

    def row_of(block):          # the nearest gr.Row (Gradio puts a field in a Form of its own)
        block = block.parent
        while not isinstance(block, gr.Row):
            block = block.parent
        return block

    run_row = row_of(by_id["shep-test-model-run"])
    assert row_of(by_id["shep-test-model-reload"]) is run_row
    assert {"shep-field-row", "shep-grp"} <= set(run_row.elem_classes or [])
    assert "shep-bar" in (run_row.parent.elem_classes or [])
    assert '"shep-train-parent-row", "shep-field-row"' in Path(training_console.__file__).read_text()
    assert '["shep-field-row"] if chosen.layout == "drawer"' in Path(model_list.__file__).read_text()
    # The Diagnosis Load drawer's head: its Task segment stands in the shared
    # list's field row as an aside (its classes read from the built page).
    task = by_id["shep-dx-change-task"]
    assert {"shep-seg", "shep-field-aside"} <= set(task.elem_classes or [])
    assert "shep-field-row" in (row_of(task).elem_classes or [])
    assert "margin-bottom: 4px" not in test_page.PAGE_CSS
    assert "#ft_refresh {" not in finetune_page.PAGE_CSS
    assert "#shep-train-refresh {" not in training_console.PAGE_CSS
    assert ".shep-form-dense { --shep-field-pad-b: 4px; }" in SHELL_CSS
    assert (".shep-form-dense .block.padded { padding-top: var(--shep-field-pad-b) !important; "
            "padding-bottom: var(--shep-field-pad-b) !important; }") in SHELL_CSS
    assert "--shep-field-pad-b: 4px;" not in training_console.PAGE_CSS     # no copy of its own


def test_every_button_link_and_disclosure_takes_the_focus_tokens_ring():
    """D08 (R-52): the shared check found Gradio's buttons (the accent Run,
    Start and Apply among them), every accordion's head and the Diagnosis
    card's link buttons drawing the browser's own ring on keyboard focus
    (``outline: auto``: no token, a faint grey on the accent buttons), on
    every page at every size. One rule at no specificity gives every button,
    link and disclosure the focus token's 2 px ring, so a control with a ring
    of its own (a tab, a table row, the ⓘ) keeps it; the pages' own summary
    rings, a second colour of the same cue, are gone. As drawn:
    `test_every_page_at_every_size_passes_the_ui_measurement_check`."""
    from src.webui.components import models_page, test_page
    from src.webui.style import SHELL_CSS

    assert (":where(button, a[href], summary):focus-visible { outline: 2px solid var(--shep-focus-ring); "
            "outline-offset: 2px; }") in SHELL_CSS
    assert ".shep-link:focus-visible" not in SHELL_CSS and ".shep-more > summary:focus-visible" not in SHELL_CSS
    for css in (models_page.PAGE_CSS, test_page.PAGE_CSS):
        assert "summary:focus-visible" not in css


#: The shared scroll cue and Compare's bounded box on a page of their own (the
#: shell's style sheet and head script, and the Models page's style sheet):
#: a region with more below says so until its end; a Compare taller and wider
#: than the detail's body keeps its sideways scrollbar in view.
_CUE_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const [css, js] = [process.argv[2], process.argv[3]].map((p) => require('fs').readFileSync(p, 'utf8'));
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1200, height: 900 } })).newPage();
  const rows = Array.from({ length: 300 }, (_, i) => `<tr><th scope="row">Field ${i}</th><td>${i}</td><td>${i}</td></tr>`).join('');
  await page.setContent(`<!doctype html><html><head><style>${css}</style></head><body>
    <div id="cue" style="height: 200px; overflow: auto;"><div style="height: 900px"></div></div>
    <div id="cuex" style="width: 200px; overflow-x: auto; overflow-y: hidden;"><div style="width: 900px; height: 20px"></div></div>
    <div id="fits" style="height: 200px; overflow: auto;"><div style="height: 100px"></div></div>
    <div id="shep-page-models"><div id="shep-mp-detail" style="height: 600px; display: flex; flex-direction: column;">
      <div id="shep-mp-detail-bd" style="flex: 1 1 auto; min-height: 0; overflow: auto; padding: 12px 16px;">
        <div class="shep-tbl-wrap shep-rs-wrap"><table class="shep-tbl shep-rs-cmp" style="min-width: 1600px"><thead><tr><th>Field</th><th>A</th><th>B</th></tr></thead><tbody>${rows}</tbody></table></div>
      </div></div></div>
    <script>${js}</script></body></html>`);
  // A layout pass, as the page's own changes start one in the dashboard.
  await page.evaluate(() => window.ShepShell.update());
  await page.waitForFunction(() => document.getElementById('cue').hasAttribute('data-shep-more'), null, { timeout: 5000 }).catch(() => {});
  const cue = () => page.evaluate(() => { const e = document.getElementById('cue');
    return { more: e.hasAttribute('data-shep-more'), mask: getComputedStyle(e).maskImage !== 'none' }; });
  const out = { top: await cue() };
  out.sideways = await page.evaluate(() => { const e = document.getElementById('cuex');
    return { more: e.hasAttribute('data-shep-more-x'), mask: getComputedStyle(e).maskImage !== 'none' }; });
  out.fits = await page.evaluate(() => { const e = document.getElementById('fits');
    return { more: e.hasAttribute('data-shep-more') || e.hasAttribute('data-shep-more-x'),
             mask: getComputedStyle(e).maskImage !== 'none' }; });
  await page.evaluate(() => { const e = document.getElementById('cue'); e.scrollTop = e.scrollHeight; });
  await page.waitForFunction(() => !document.getElementById('cue').hasAttribute('data-shep-more'), null, { timeout: 5000 }).catch(() => {});
  out.end = await cue();
  for (const layout of ['side', 'stack']) {
    out[layout] = await page.evaluate((layout) => {
      document.documentElement.setAttribute('data-shep-layout', layout);
      const detail = document.getElementById('shep-mp-detail'), body = document.getElementById('shep-mp-detail-bd');
      if (layout === 'stack') {   // stacked, the detail flows in the page, which scrolls
        detail.style.height = 'auto'; body.style.overflow = 'visible';
      }
      const wrap = body.querySelector('.shep-rs-wrap');
      wrap.scrollIntoView({ block: 'start' });
      const b = body.getBoundingClientRect(), r = wrap.getBoundingClientRect();
      const bottom = layout === 'side' ? Math.min(b.bottom, innerHeight) : innerHeight;
      return { wrapH: Math.round(r.height), bodyH: Math.round(b.height), sideways: wrap.scrollWidth > wrap.clientWidth,
               barInView: r.bottom <= bottom + 1 };
    }, layout);
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_scroll_cue_while_more_is_below_and_compares_sideways_bar_in_reach(tmp_path):
    """R-73: Test's Result body scrolled with no cue (overlay scrollbars draw
    nothing at rest). Every region that scrolls fades its last lines while
    more is below and draws them whole at its end, and fades its right edge
    while more lies past it (the head script keeps ``data-shep-more`` and
    ``data-shep-more-x``); a region that holds its content draws no cue. No
    page opts in (the sweep's L7: Settings, Fine-tune's form, Training's charts
    and the Diagnosis fields scrolled with no cue because their pages had not).
    B72-C3: Compare, 12,342 px tall, had its sideways scrollbar at its foot,
    out of reach mid-table; it is now a box no taller than the region it is
    read in, so its scrollbar is in view whenever the box is. Both measured
    in a browser on a page of their own (Node and Playwright, skipped without)."""
    from src.webui.components import models_page, test_page
    from src.webui.style import SHELL_CSS
    from tests.fixtures.browser import node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    assert "shep-scroll-cue" not in Path(test_page.__file__).read_text()
    css_path, js_path = tmp_path / "page.css", tmp_path / "head.js"
    css_path.write_text(SHELL_CSS + models_page.PAGE_CSS, encoding="utf-8")
    js_path.write_text(shell._SHELL_JS.replace("__TITLES__", "{}").replace("__RESEARCH_LABEL__", '""')
                       .replace("__STAGES__", "{}").replace("__STAGE_WORD__", '""')
                       .replace("__LAYOUT__", "data-shep-layout").replace("__SHORT__", "data-shep-short")
                       .replace("__SHORT_BELOW__", "900").replace("__GO__", "{}"), encoding="utf-8")
    seen = run_node(_CUE_JS, tmp_path, str(css_path), str(js_path))
    assert seen["top"] == {"more": True, "mask": True}, seen
    assert seen["end"] == {"more": False, "mask": False}, seen
    assert seen["sideways"] == {"more": True, "mask": True}, seen
    assert seen["fits"] == {"more": False, "mask": False}, seen
    for layout in ("side", "stack"):
        box = seen[layout]
        assert box["sideways"] and box["barInView"], (layout, box)
        assert box["wrapH"] <= 900, (layout, box)
    assert seen["side"]["wrapH"] <= seen["side"]["bodyH"], seen


def test_the_training_parameters_fold_when_stacked_and_show_side_by_side():
    """R-60: at 960 x 960 (stacked) the full Parameters panel pushed the run
    view's first plot below the fold (page scroll 1386 px). Stacked, the
    parameters start folded under their title, "Show" opening them (its
    ``aria-expanded``, which the style sheet reads); side by side the body is
    always shown and the toggle is not. Measured on the demo workspace at
    960 x 960: the first plot at y 607-867, the page not scrolled."""
    from src.webui.components import training_console as tc

    assert 'aria-expanded="false" aria-controls="shep-train-params-bd"' in Path(tc.__file__).read_text()
    assert "b.setAttribute('aria-expanded', open ? 'true' : 'false');" in tc.FOLD_JS
    assert ('#shep-train-params:has(.shep-train-fold[aria-expanded="false"]) > #shep-train-params-bd '
            "{ display: none !important; }") in stack(tc.PAGE_CSS)
    assert ".shep-train-fold { display: none; }" in side(tc.PAGE_CSS)


# --------------------------------------------------------------------------- the tab groups and the go link
def test_the_tab_groups_flatten_to_the_research_pages_and_their_style_is_generated():
    """The owner's decision of 2026-11-22: "Datasets │ Training Fine-tune │
    Test Models", a grouping drawn by position, never a stage (`STAGES` and
    the tablist's name are unchanged) and never a node of the tablist: each
    later group's first tab keeps a gap (`TAB_GROUP_GAP`) in whose middle an
    empty ``::before`` draws the separator, through the documented
    ``<elem_id>-button`` ids only; no role, tab stop or text is added. Not
    drawn while the tabs wrap (`TABS_WRAPPED_ATTR`, the head script's mark);
    the gap stays, so the mark changes nothing the pass measures."""
    from src.webui.app import dashboard_head

    assert tuple(page for group in shell.TAB_GROUPS for page in group) == shell.RESEARCH_PAGES
    assert shell.TAB_GROUPS == (("datasets",), ("training", "finetune"), ("test", "models"))
    css = shell.TAB_GROUP_CSS
    assert css in dashboard_head()
    for first in ("training", "test"):
        tab = f"#shep-research-nav > .tab-wrapper > .tab-container > #shep-page-{first}-button"
        assert f"{tab} {{ position: relative; margin-inline-start: var(--shep-tabgroup-gap) !important; }}" in css
        sep = _rule(css, f"{tab}::before")
        assert 'content: ""' in sep and "pointer-events: none" in sep and "width: 1px" in sep
        assert (f"#shep-research-nav[{shell.TABS_WRAPPED_ATTR}] > .tab-wrapper > .tab-container > "
                f"#shep-page-{first}-button::before {{ display: none; }}") in css
    assert "#shep-page-datasets-button" not in css and "#shep-page-finetune-button" not in css
    assert not re.search(r"role|tabindex|content: \"[^\"]", css)
    head = shell.SHELL_HEAD
    lines = head[head.index("function tabLines()"):head.index("var FOLD = ")]
    # The pass reads the tabs' own tops and writes only the mark, which no rule of a tab's box reads.
    assert "offsetTop" in lines and "toggleAttribute(TABS_WRAPPED, wrapped)" in lines
    assert "style" not in lines and "margin" not in lines
    assert "    tabLines();" in head[head.index("function layout()"):]


def test_the_fit_pass_measures_each_box_with_its_own_mark_off():
    """M3 and M6 (the UI sweep): a box the pass marks narrow (a wide table
    whose optional columns the mark hides, a choice group the mark stacks)
    is measured with that mark taken off and put back before anything is
    painted, so what it measures never depends on what it set; a choice group
    is a fieldset block that is not a segmented control, its options on one
    line for the measure alone."""
    head = shell.SHELL_HEAD
    fits = head[head.index("function fits()"):head.index("var TABS_WRAPPED")]
    assert fits.count("removeAttribute(NARROW)") == 2 and fits.count("setAttribute(NARROW, '')") == 2
    for block in fits.split("var groups")[0], fits.split("var groups")[1]:
        off, measure, back = (block.index("removeAttribute(NARROW)"), block.index("measured("),
                              block.index("setAttribute(NARROW, '')"))
        assert off < measure < back
    assert "var CHOICE = 'fieldset.block:not(.shep-seg), fieldset.block.shep-seg-whole';" in head
    assert "':scope > .wrap:has(> label)'" in fits
    from src.webui.style import SHELL_CSS

    assert "fieldset.block[data-shep-narrow]:not(.shep-seg) > .wrap:has(> label)" in SHELL_CSS
    assert "[data-shep-narrow] :is(th, td, col).shep-col-opt { display: none !important; }" in SHELL_CSS


def test_a_notices_link_opens_only_a_listed_place():
    """The notice's next step (`renderers.notice_block`'s ``data-shep-go``):
    the head script opens a page only from `GO_TARGETS` (Settings, focusing
    only its Data locations panel; the research pages, no focus), never a
    value from the page itself."""
    assert shell.GO_TARGETS == {"settings": ("shep-dl",), **{p: () for p in shell.RESEARCH_PAGES}}
    head = shell.SHELL_HEAD
    go = head[head.index("var GO = "):head.index("window.ShepShell = {")]
    assert "var GO = {\"settings\": [\"shep-dl\"], \"datasets\": []" in go
    assert "if (!Object.prototype.hasOwnProperty.call(GO, page)) { return; }" in go
    assert "if (GO[page].indexOf(focusId) < 0) { focusId = ''; }" in go
    assert "innerHTML" not in go and "eval(" not in go and "location.href" not in go
    assert "head.focus({preventScroll: true})" in go


def _groups_page():
    """The shell with stand-in pages: Datasets draws a notice (`notice_block`,
    its link to Settings › Data locations) and two links the script must not
    follow as asked (a place it does not list; an element it does not list);
    Settings draws a stand-in Data locations panel with its heading and a
    link to a research page (Models)."""
    from src.config import server_settings as ss
    from src.webui.components import renderers as r
    from src.webui.components import workspaces as w

    notice = r.notice_block(w.setting_notice(ss.workspaces_root({})))
    stray = ('<button type="button" id="go-evil" data-shep-go="evil">Elsewhere</button>'
             '<button type="button" id="go-unlisted" data-shep-go="settings" '
             'data-shep-focus="shep-settings">Settings, unlisted focus</button>')

    def datasets(ctx):
        shell.plain_html(notice + stray, elem_id="groups-notice")

    def settings(ctx):
        with gr.Column(elem_id="shep-dl", elem_classes=["shep-panel"]):
            shell.plain_html('<div class="shep-panel-hd"><h2>Data locations</h2></div>'
                             '<button type="button" id="go-models" data-shep-go="models">'
                             'Models</button>')
            gr.Textbox(label="Workspaces folder", elem_id="stand-in-field")
            with gr.Column(elem_id="groups-col"):
                gr.CheckboxGroup(["Disease", "Gene"], label="Tasks", elem_id="groups-col-choice")
            with gr.Row(elem_id="groups-row"):
                gr.Textbox(label="Epochs", elem_id="groups-row-a")
                gr.Textbox(label="Seed", elem_id="groups-row-b")
                gr.Radio(["Trained", "Frozen"], label="Graph backbone", elem_id="groups-row-choice")

    pages = {**{page: (lambda ctx: None) for page in (*shell.RESEARCH_PAGES, shell.SETTINGS,
                                                       shell.CLINICAL)},
             shell.DATASETS: datasets, shell.SETTINGS: settings}
    with gr.Blocks() as demo:
        shell.build_shell(demo, pages)
    return demo


_GROUPS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1850, height: 960 } });
  await page.addInitScript(QUEUE_JS);
  await page.goto(base, { waitUntil: 'load' });
  await page.waitForSelector('#shep-tab-clinical-button[aria-selected="true"]', { timeout: 30000 });
  await page.waitForFunction(() => window.ShepShell !== undefined);
  const frame = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  await page.click('#shep-tab-research-button');
  await page.click('#shep-page-datasets-button');
  await page.waitForSelector('#groups-notice .shep-nb');
  const row = () => page.evaluate(() => {
    const list = document.querySelector('#shep-research-nav > .tab-wrapper > .tab-container');
    const tabs = [...list.children];
    return { kids: tabs.map((t) => [t.tagName, t.getAttribute('role'), t.getAttribute('data-tab-id')]),
             wrapped: document.getElementById('shep-research-nav').hasAttribute('data-shep-tabs-wrapped'),
             tabs: tabs.map((t) => { const r = t.getBoundingClientRect(), s = getComputedStyle(t, '::before');
               const drawn = s.content !== 'none' && s.content !== 'normal' && s.display !== 'none';
               return { id: t.getAttribute('data-tab-id'), left: r.left, right: r.right, top: Math.round(r.top),
                        sep: drawn ? { x: r.left + parseFloat(s.left), content: s.content, top: parseFloat(s.top),
                                       bottom: parseFloat(s.bottom), width: parseFloat(s.width) } : null }; }) };
  });
  const out = {};
  for (const [name, w, h] of [['wide', 1850, 960], ['narrow', 390, 844], ['back', 1850, 960]]) {
    await page.setViewportSize({ width: w, height: h });
    await frame(); await frame();
    out[name] = await row();
  }
  // The keyboard: from Datasets' tab, Tab moves tab to tab, past no separator.
  await page.focus('#shep-page-datasets-button');
  const keys = [];
  for (let i = 0; i < 4; i++) {
    await page.keyboard.press('Tab');
    keys.push(await page.evaluate(() => document.activeElement.getAttribute('data-tab-id')));
  }
  await page.click('#shep-page-datasets-button');
  await page.waitForSelector('#groups-notice .shep-nb');
  // A link to a place the script does not list does nothing.
  await page.click('#go-evil');
  await frame(); await frame();
  out.evil = await page.evaluate(() => document.querySelector('#shep-nav > .tab-wrapper [role="tab"][aria-selected="true"]').getAttribute('data-tab-id'));
  // The notice's link: Settings opens and Data locations' heading takes the focus.
  await page.click('#groups-notice .shep-nb-go');
  await page.waitForSelector('#shep-tab-settings-button[aria-selected="true"]');
  await page.waitForFunction(() => document.activeElement && document.activeElement.tagName === 'H2', null, { timeout: 10000 });
  out.landed = await page.evaluate(() => ({ text: document.activeElement.textContent.trim(),
    inDl: !!document.activeElement.closest('#shep-dl'), tabindex: document.activeElement.getAttribute('tabindex'),
    title: document.title }));
  // Fields in a panel: a choice group in a column keeps the column's width on its edge; a form's
  // row keeps a gap between its fields' boxes, its own choice group at its one-line width; no
  // form shows a ground of its own.
  out.fields = await page.evaluate(() => {
    const box = (sel) => { const r = document.querySelector(sel).getBoundingClientRect(); return { left: r.left, right: r.right, width: r.width }; };
    const input = (sel) => box(sel + ' :is(input, textarea)');
    return { col: box('#groups-col'), colChoice: box('#groups-col-choice'),
             a: input('#groups-row-a'), b: input('#groups-row-b'), rowChoice: box('#groups-row-choice'),
             options: box('#groups-row-choice > .wrap'),
             grounds: [...document.querySelectorAll('#shep-dl .form')].map((f) => getComputedStyle(f).backgroundColor) };
  });
  // A link to a research page from Settings: the research area opens on that page.
  await page.click('#go-models');
  await page.waitForSelector('#shep-tab-research-button[aria-selected="true"]', { timeout: 10000 });
  await page.waitForSelector('#shep-page-models-button[aria-selected="true"]', { timeout: 10000 });
  out.research = await page.evaluate(() => document.querySelector('#shep-nav > .tab-wrapper [role="tab"][aria-selected="true"]').getAttribute('data-tab-id'));
  // An element the script does not list for the page: the page opens, the focus stays.
  await page.click('#shep-page-datasets-button');
  await page.waitForSelector('#groups-notice .shep-nb');
  await page.click('#go-unlisted');
  await page.waitForSelector('#shep-tab-settings-button[aria-selected="true"]');
  await frame(); await frame();
  out.unlisted = await page.evaluate(() => document.activeElement.id);
  out.keys = keys;
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_tab_groups_and_a_notices_link_in_a_browser(tmp_path):
    """The tab groups and the notice's link as drawn (Node and Playwright,
    skipped without), the shell mounted as the service mounts it:

    - at the owner's window the tabs are on one line, a separator drawn in
      the gap before Training and before Test (between the tabs, never over
      one, empty, 1 px), none before Datasets or the second tab of a group;
    - at a phone's width the tabs wrap: the row is marked and no separator
      is drawn (a line never starts with one); back at the owner's window,
      as before;
    - the tablist holds the five tabs only, in `RESEARCH_PAGES`' order; the
      keyboard's Tab moves from tab to tab, meeting no separator;
    - the notice's link opens Settings and moves the focus to the Data
      locations heading; a link to a research page from Settings opens the
      research area on that page (its outer tab was looked up by a page's
      id, so nothing opened); a link to a place the script does not list
      does nothing, and an element it does not list for the page gets no
      focus;
    - fields in a panel (style.py's M3 and M5): a choice group in a column
      keeps the column's width on its edge (Gradio centred it at its
      one-line width, a form's grey ground beside it); in a row the fields'
      boxes keep a gap (they touched once a block lost its side padding) and
      the row's choice group its one-line width; no form draws a ground."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    with served(_groups_page(), **mount_options()) as base:
        seen = run_node(_GROUPS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    wide = seen["wide"]
    assert wide["kids"] == [["BUTTON", "tab", page] for page in shell.RESEARCH_PAGES], wide
    assert not wide["wrapped"] and len({t["top"] for t in wide["tabs"]}) == 1, wide
    drawn = {t["id"]: t["sep"] for t in wide["tabs"] if t["sep"]}
    assert set(drawn) == {"training", "test"}, wide
    by = {t["id"]: t for t in wide["tabs"]}
    for first, before in (("training", "datasets"), ("test", "finetune")):
        sep = drawn[first]
        assert sep["content"] == '""' and sep["width"] == 1, sep
        assert by[before]["right"] < sep["x"] < by[first]["left"], (sep, by[before], by[first])
    narrow = seen["narrow"]
    assert narrow["wrapped"] and len({t["top"] for t in narrow["tabs"]}) > 1, narrow
    assert not any(t["sep"] for t in narrow["tabs"]), narrow
    assert seen["back"] == wide
    assert seen["keys"] == ["training", "finetune", "test", "models"], seen["keys"]
    assert seen["evil"] == "research"
    assert seen["landed"] == {"text": "Data locations", "inDl": True, "tabindex": "-1",
                              "title": shell.TITLES["settings"]}, seen["landed"]
    assert seen["research"] == "research"
    assert seen["unlisted"] != "shep-settings"
    fields = seen["fields"]
    col, choice = fields["col"], fields["colChoice"]
    assert abs(choice["left"] - col["left"]) < 1, fields
    assert abs(choice["width"] - col["width"]) < 1, fields
    assert fields["b"]["left"] - fields["a"]["right"] >= 15.5, fields
    assert fields["rowChoice"]["left"] - fields["b"]["right"] >= 15.5, fields
    assert fields["rowChoice"]["width"] < 2 * fields["options"]["width"], fields
    assert fields["grounds"] and set(fields["grounds"]) == {"rgba(0, 0, 0, 0)"}, fields


# --------------------------------------------------------------------------- the first-use and empty walks
#: The header's facts at a stop of a state walk: whether the chips and Settings went onto a
#: second line, and the GPU chip's words.
_HEADER_FACTS_JS = r"""() => {
  const box = (s) => { const e = document.querySelector(s);
    return e && e.getClientRects().length ? e.getBoundingClientRect() : null; };
  // Training's run view: its resources line and the panel's content bottom (its padding box).
  const view = document.getElementById('shep-train-runview'), res = box('#shep-train-runview > #resource_display');
  const foot = view && res ? view.getBoundingClientRect().bottom - (parseFloat(getComputedStyle(view).borderBottomWidth) || 0)
                              - (parseFloat(getComputedStyle(view).paddingBottom) || 0) : null;
  return { wrapped: document.getElementById('shep-shell').classList.contains('shep-nav-wrapped'),
           gpu: ((document.querySelector('#shep-chips .shep-chip-gpu .shep-chip-text') || {}).textContent || '').trim(),
           runview: res && foot !== null && view.scrollHeight <= view.clientHeight + 1
             ? { resBottom: res.bottom, foot } : null };
}"""
@pytest.fixture(scope="module")
def ui_walk_first_use(tmp_path_factory):
    """**The first-use walk** (the UI sweep of 2026-11-22): the service as a
    first start serves it (no workspaces folder, no results registry, the
    launcher's local start: `browser.state_settings("first_use")`), every
    page at `browser.LAYOUT_SIZES` and the Load drawer open (`_layout_walk_stops`), the
    layout rules recorded at each stop (no focus walk), screenshots under the module's
    ``ui-walk-first-use*/shots``."""
    from tests.fixtures.browser import node_with_playwright, walk_state

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    root = tmp_path_factory.mktemp("ui-walk-first-use")
    (root / "shots").mkdir()
    return walk_state("first_use", root, _layout_walk_stops(False), shots=root / "shots",
                      extra_js=_HEADER_FACTS_JS)


@pytest.fixture(scope="module")
def ui_walk_empty(tmp_path_factory):
    """**The empty-folder walk**: an empty workspaces folder and an empty
    registry (`browser.state_settings("empty")`), every page at
    `browser.LAYOUT_SIZES` and the Load drawer open (`_layout_walk_stops`)."""
    from tests.fixtures.browser import node_with_playwright, walk_state

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    root = tmp_path_factory.mktemp("ui-walk-empty")
    (root / "shots").mkdir()
    return walk_state("empty", root, _layout_walk_stops(False),
                      shots=root / "shots", extra_js=_HEADER_FACTS_JS)


def _state_walk_holds_the_shells_rules(seen, gpu_words):
    from tests.fixtures.browser import UI_PAGES, layout_problems

    assert {s["page"] for s in seen} == set(UI_PAGES)
    for s in seen:
        width = int(s["size"].split("x")[0])
        assert s["extra"]["gpu"] == gpu_words, s["extra"]
        if width >= 1280:                                 # one header line from a laptop's width
            assert not s["extra"]["wrapped"], (s["page"], s["size"])
        assert "layout" in s["check"] and s["check"]["layout"]["tabs"] is not None \
            or s["page"] in ("clinical", "settings"), (s["page"], s["size"])
    assert layout_problems(seen, rules=("tab-groups", "help-on-page", "notice-repeat")) == []


def test_at_first_use_trainings_resources_line_ends_its_run_view(ui_walk_first_use):
    """The independent check of the UI sweep (C10): at 1280 x 720 at first use
    (no run to draw) the run view's charts row shrank to its empty message in
    a run view too short for one row of plots, so the resources line stood
    mid-panel over 190 px of blank; at 1366 x 768 it was at the panel's foot.
    Wherever the run view does not scroll, its resources line is its last row,
    on the panel's content bottom, at every layout size."""
    stops = [s for s in ui_walk_first_use
             if s["page"] == "training" and not s.get("open") and s["extra"].get("runview")]
    assert {s["size"] for s in stops} >= {"1850x960", "1366x768", "1280x720"}, stops
    for stop in stops:
        facts = stop["extra"]["runview"]
        assert abs(facts["resBottom"] - facts["foot"]) <= 1, (stop["size"], facts)


def test_at_first_use_the_header_and_tab_groups_hold_on_every_page_and_size(ui_walk_first_use):
    """B-11 and the owner's tab groups, at first use (no workspaces folder, no
    registry; the state of the owner's six screenshots): the GPU chip says
    "GPU jobs: not set up" and the header is one line from 1280 px wide up
    (it wrapped at 1280 on "GPU jobs: refused, no workspaces folder"); the
    research tabs keep their order, a separator is never drawn at a line's
    start or while they wrap, nothing but a tab is in the tablist; no help
    text is drawn on the page and no notice twice on a screen. The page
    rules (`browser.layout_problems`: beside, narrow-box, orphan, edge, …)
    are held at zero by `test_every_page_of_every_state_has_no_layout_finding`."""
    _state_walk_holds_the_shells_rules(ui_walk_first_use, "GPU jobs: not set up")


def test_with_an_empty_workspaces_folder_the_header_and_tab_groups_hold(ui_walk_empty):
    """The same shell rules with an empty workspaces folder and an empty
    registry: the GPU lock is set up, so the chip says "GPU jobs: none"."""
    _state_walk_holds_the_shells_rules(ui_walk_empty, "GPU jobs: none")


# --------------------------------------------------------------------------- the shared pieces as drawn
def _pieces_page():
    """The shell with one stand-in research page (Datasets) drawing the shared pieces the sweep's
    lanes reported on (each in its own element, measured by `_PIECES_JS`): a relation led by a
    mark in a box wider than its words, and one its page places at the line's end; a bar with an
    empty note and a note that says something, and a search row in a bar; a panel's header in a
    column too short for it; a row of fields and a grid of fields (their fields in Gradio's one
    form), a choice group with an ⓘ in a field row; a wrapped name in the shared tree; and a wide
    table in a fit box above a tall page."""
    from src.config import server_settings as ss
    from src.webui.components import renderers as r
    from src.webui.components import workspaces as w
    from src.webui.style import CELL_WHOLE, FIT, FIT_WIDE

    notice = w.setting_notice(ss.workspaces_root({}))
    # One identity line (Test's and Models', `renderers.dataset_identity_html`) in boxes of
    # every width from 160 to 552 px, so some line ends at each of its separators.
    ident = r.dataset_identity_html(
        {"name": "demo", "dataset_fingerprint": "3f563f74bb04" + "0" * 52, "trainable": True,
         "dataset_record": {"binding": "verified", "origin": "synthetic", "sha256": None,
                            "problem": None},
         "build": {"state": "recorded", "built_at": "2026-11-22T12:10:00Z"},
         "declared_releases": (), "train_samples": 20, "val_samples": 5}, name=False)
    seps = "".join(f'<div class="pc-sepbox" style="width: {w}px">{ident}</div>'
                   for w in range(160, 560, 8))
    prose = "Words of a column that wraps, as a model's name and start do in its own column. " * 2
    cells = "".join(f'<td class="shep-nowrap">a cell that never breaks {i:02d}</td>' for i in range(12))
    tree = ('<table class="shep-tbl shep-mtree" id="pc-tree"><tbody><tr><td><div class="shep-tree-name">'
            '<button type="button" class="shep-fold" aria-expanded="true">▾</button>'
            '<span class="shep-tree-label">a run named at length so that it wraps over three lines</span>'
            '</div></td></tr></tbody></table>')

    def datasets(ctx):
        with gr.Column(elem_id="pc-panel", elem_classes=["shep-panel"]):
            shell.plain_html('<div class="shep-panel-hd"><h2>Shared pieces</h2></div>')
            shell.plain_html(r.see_above("Start", notice), elem_id="pc-rel")
            shell.plain_html(r.see_above("Import", notice).replace(
                'class="shep-reason', 'style="justify-content: end; max-width: none" class="shep-reason', 1),
                elem_id="pc-rel-end")
            with gr.Row(elem_id="pc-bar", elem_classes=["shep-bar"]):
                with gr.Row(elem_id="pc-search", elem_classes=["shep-search-row", "shep-grp", "shep-grp-grow"]):
                    shell.plain_html('<span class="shep-search-lbl">Search</span>', elem_id="pc-search-lbl")
                    gr.Textbox(show_label=False, container=False, elem_id="pc-search-box", lines=1, max_lines=1)
                    gr.Button("Clear", size="sm", scale=0, min_width=0, elem_id="pc-search-clear")
                shell.plain_html("", elem_id="pc-note-empty", elem_classes=["shep-bar-note"])
                shell.plain_html("<p>A note that says something.</p>", elem_id="pc-note-said",
                                 elem_classes=["shep-bar-note"])
            with gr.Column(elem_id="pc-squeeze"):
                with gr.Row(elem_id="pc-squeeze-hd", elem_classes=["shep-panel-hd"]):
                    shell.plain_html("<h2>A header whose title runs over two or three lines in a narrow "
                                     "column</h2>", elem_id="pc-squeeze-title")
                shell.plain_html('<div style="height: 600px">A tall body.</div>', elem_id="pc-squeeze-bd")
            with gr.Row(elem_id="pc-fields", elem_classes=["shep-fields"]):
                for name in ("Epochs", "Seed", "Learning rate"):
                    gr.Textbox(label=name, elem_id=f"pc-fields-{name[:2].lower()}")
            with gr.Row(elem_id="pc-grid", elem_classes=["shep-field-grid"]):
                for name in ("Warmup", "Decay", "Patience", "Clip"):
                    gr.Textbox(label=name, elem_id=f"pc-grid-{name.lower()}")
            with gr.Row(elem_id="pc-frow", elem_classes=["shep-field-row"]):
                gr.Textbox(label="Name", elem_id="pc-frow-name")
                gr.Radio(["Trained", "Frozen"], label="Graph backbone", elem_id="pc-frow-choice")
            ctx.help.field("pc-frow-choice", "Graph backbone", "What the base's graph layers do here.")
            shell.plain_html(tree, elem_id="pc-tree-box")
            shell.plain_html(f'<div id="pc-c12">{seps}<div style="width: 320px"><table class="shep-tbl" '
                             f'id="pc-whole"><tbody><tr><td class="{CELL_WHOLE}">Record not readable</td>'
                             f'<td>{prose}</td></tr></tbody></table></div></div>', elem_id="pc-c12-box")
            shell.plain_html(f'<div class="shep-tbl-wrap {FIT}" id="pc-fit"><table class="shep-tbl {FIT_WIDE}">'
                             f'<tbody><tr>{cells}</tr></tbody></table></div>'
                             '<div style="height: 2400px"></div>', elem_id="pc-fit-box")

    pages = {**{page: (lambda ctx: None) for page in (*shell.RESEARCH_PAGES, shell.SETTINGS,
                                                       shell.CLINICAL)},
             shell.DATASETS: datasets}
    with gr.Blocks() as demo:
        ctx = shell.build_shell(demo, pages)
        shell.help_texts(ctx.help)
    return demo


#: The stand-in page's facts as drawn (`_pieces_page`), at 800 x 900 (stacked: the page itself
#: scrolls) with the panel held to 640 px, the squeezed column to 240 px with its body a scroll region, and the field row's and
#: grid's floors set (16rem, 12rem).
_PIECES_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 800, height: 900 } });
  await page.addInitScript(QUEUE_JS);
  await page.goto(process.argv[2], { waitUntil: 'load' });
  await page.waitForFunction(() => window.ShepShell !== undefined, null, { timeout: 30000 });
  await page.click('#shep-tab-research-button');
  await page.click('#shep-page-datasets-button');
  await page.waitForSelector('#pc-panel');
  await page.addStyleTag({ content: `
    #pc-panel { max-width: 640px !important; }
    #pc-squeeze { height: 240px !important; max-height: 240px !important; flex: none !important; width: 260px !important; }
    #pc-squeeze-bd { flex: 1 1 auto !important; min-height: 0 !important; overflow: auto !important; }
    #pc-fields { --shep-field-min: 16rem; }
    #pc-grid { --shep-field-min: 12rem; }
    #pc-tree-box { width: 180px; }` });
  const frames = (n) => page.evaluate((n) => new Promise((r) => { const f = (k) => k ? requestAnimationFrame(() => f(k - 1)) : r(); f(n); }), n);
  await page.waitForSelector('#pc-frow-choice .shep-i');
  await page.waitForSelector('#pc-fit[data-shep-narrow]');
  await frames(4);
  const out = await page.evaluate(() => {
    const q = (s) => document.querySelector(s);
    const box = (e) => e.getBoundingClientRect();
    const ink = (e) => { const r = document.createRange(); r.selectNodeContents(e);
      return [...r.getClientRects()].filter((x) => x.width > 0); };
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    const rel = (id) => { const p = q(`#${id} .shep-reason`), mark = ink(p.querySelector('.shep-nb-mark'))[0],
      words = ink(p.querySelector('.shep-nb-text'));
      return { gap: words[0].left - mark.right, colGap: parseFloat(getComputedStyle(p).columnGap),
               wordsRight: Math.max(...words.map((x) => x.right)), boxRight: box(p).right,
               boxLeft: box(p).left, markLeft: mark.left }; };
    const mid = (e) => { const r = box(e); return (r.top + r.bottom) / 2; };
    const title = q('#pc-frow-choice [data-testid="block-info"]'), info = q('#pc-frow-choice .shep-i');
    const lines = ink(q('#pc-tree .shep-tree-label')), fold = box(q('#pc-tree .shep-fold'));
    const hd = q('#pc-squeeze-hd');
    return {
      rel: rel('pc-rel'), relEnd: rel('pc-rel-end'),
      notes: { empty: getComputedStyle(q('#pc-note-empty')).display, said: getComputedStyle(q('#pc-note-said')).display },
      search: { box: mid(q('#pc-search-box input, #pc-search-box textarea')), label: mid(q('#pc-search-lbl .shep-search-lbl')),
                clear: mid(q('#pc-search-clear')) },
      header: { client: hd.clientHeight, scroll: hd.scrollHeight, title: box(q('#pc-squeeze-title')).height,
                height: box(hd).height },
      fields: { floor: 16 * rem, boxes: [...document.querySelectorAll('#pc-fields .block:has(input, textarea)')]
        .map((e) => ({ width: box(e).width, top: Math.round(box(e).top), left: Math.round(box(e).left) })),
        row: box(q('#pc-fields')).width },
      grid: { floor: 12 * rem, boxes: [...document.querySelectorAll('#pc-grid .block:has(input, textarea)')]
        .map((e) => ({ width: box(e).width, left: Math.round(box(e).left) })), row: box(q('#pc-grid')).width },
      choice: { title: box(title), info: box(info) },
      c12: (() => {
        // Each separator's "·" as drawn: where its line starts (its box's left) and whether the
        // fact after it starts the next line; the whole cell's and the prose cell's lines.
        const tops = (e) => new Set(ink(e).map((x) => Math.round(x.top))).size;
        const seps = [];
        for (const b of document.querySelectorAll('#pc-c12 .pc-sepbox')) {
          for (const sep of b.querySelectorAll('.shep-sep')) {
            const t = sep.firstChild, at = t.data.indexOf('·'), rg = document.createRange();
            rg.setStart(t, at); rg.setEnd(t, at + 1);
            const dot = rg.getBoundingClientRect(), next = sep.nextElementSibling;
            const after = next ? ink(next)[0] : null;
            seps.push({ width: box(b).width, fromStart: dot.left - box(b).left,
                        endsLine: !!after && after.top > dot.bottom - 1,
                        drawn: getComputedStyle(sep).visibility !== 'hidden' });
          }
        }
        const cells = [...document.querySelectorAll('#pc-whole td')];
        return { seps, whole: tops(cells[0]), prose: tops(cells[1]) };
      })(),
      tree: { lines: lines.length, first: { top: lines[0].top, bottom: lines[0].bottom },
              fold: { top: fold.top, bottom: fold.bottom } },
    };
  });
  // The fit box scrolled sideways and the page scrolled to its end, then anything drawn: the
  // layout pass runs and measures the box, and neither scroll moves.
  await page.evaluate(() => { const fit = document.getElementById('pc-fit'); fit.scrollLeft = 120;
    window.scrollTo(0, document.scrollingElement.scrollHeight); });
  await frames(2);
  out.scroll = { before: await page.evaluate(() => ({ fit: document.getElementById('pc-fit').scrollLeft, page: window.scrollY,
                                                      wide: document.getElementById('pc-fit').scrollWidth > document.getElementById('pc-fit').clientWidth })) };
  await page.evaluate(() => document.body.appendChild(document.createElement('div')));
  await frames(4);
  out.scroll.after = await page.evaluate(() => ({ fit: document.getElementById('pc-fit').scrollLeft, page: window.scrollY }));
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def pieces_seen(tmp_path_factory):
    """`_PIECES_JS` on `_pieces_page`, served as the service mounts it, once for the module."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    with served(_pieces_page(), **mount_options()) as base:
        return run_node(_PIECES_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)),
                        tmp_path_factory.mktemp("pieces"), base[:-len("gradio_api")])


def test_a_separator_never_starts_a_line_and_a_whole_cell_keeps_its_line(pieces_seen):
    """C12 (the L6 check): Test's identity line wrapped with its separator at
    the next line's start ("· synthetic …", 1550 px), and the Models table
    broke "Fine-tune, full" and "Record not readable" while its Model column
    had room. A separator is joined to the fact before it (`renderers.SEP`:
    a no-break space, then "· "), so in a box of any width a line ends at it
    and never starts with it; a cell whose value reads as one
    (`style.CELL_WHOLE`) keeps its line while a column of prose beside it
    wraps. The sweep's L7 (SW-LK2, SW-LK7: Fine-tune's facts line ended in
    a dangling " · "): a separator the line breaks after is not drawn (the
    shell's layout pass marks it, its box kept), so no drawn separator ends
    a line either."""
    facts = pieces_seen["c12"]
    ends = [x for x in facts["seps"] if x["endsLine"]]
    assert ends, facts["seps"]                                 # some line breaks after one
    assert not [x for x in ends if x["drawn"]], ends           # and draws no separator there
    assert [x for x in facts["seps"] if not x["endsLine"] and x["drawn"]], facts["seps"]
    starts = [x for x in facts["seps"] if x["fromStart"] < 1]
    assert not starts, starts
    assert facts["whole"] == 1 and facts["prose"] > 1, facts


def test_a_relations_words_start_beside_its_mark_and_its_page_still_places_it(pieces_seen):
    """S1 of the sweep's integration (the lanes' L2 R-a, L3 R1, L5 R2): a relation led by a mark
    (`renderers.see_above`) in a box wider than its words starts its words one column gap after
    its mark (the mark's ``auto`` column took the room the words left: 74 px on Fine-tune, 209 px
    on Test), and a page that places it at its line's end (``justify-content``) still does."""
    rel, end = pieces_seen["rel"], pieces_seen["relEnd"]
    assert rel["gap"] <= rel["colGap"] + 1, rel
    assert abs(rel["markLeft"] - rel["boxLeft"]) <= 1, rel
    assert end["gap"] <= end["colGap"] + 1 and abs(end["wordsRight"] - end["boxRight"]) <= 1, end


def test_an_empty_bar_note_takes_no_room_and_a_note_that_says_something_is_drawn(pieces_seen):
    """S2 (L1 R1): a bar's note (``shep-bar-note``, a ``gr.HTML``, which always draws its
    container and the element in it) is hidden while it says nothing and drawn once it does;
    the shared rule tested the container, so no empty note was ever hidden."""
    assert pieces_seen["notes"] == {"empty": "none", "said": "block"}, pieces_seen["notes"]


def test_a_panels_header_keeps_its_words_in_a_column_too_short_for_it(pieces_seen):
    """S3 (L1 R3): a panel's header (``shep-panel-hd``, a row in a column) never shrinks under
    its words when the column is too short for its body (a drawer's header was squeezed to its
    48 px floor at 1366 x 768, its two lines of title under the body)."""
    hd = pieces_seen["header"]
    assert hd["scroll"] <= hd["client"] + 1 and hd["height"] >= hd["title"], hd


def test_a_search_rows_label_and_clear_stand_on_the_boxs_middle_line_in_a_bar(pieces_seen):
    """S5 (L4 b): in a bar, whose groups stand on one bottom line, a search row's "Search" and
    Clear stay on its box's middle line (they sank to its bottom; Models set it by its id)."""
    s = pieces_seen["search"]
    assert abs(s["label"] - s["box"]) <= 1.5 and abs(s["clear"] - s["box"]) <= 1.5, s


def test_a_row_of_fields_and_a_grid_of_fields_reach_the_fields_in_gradios_form(pieces_seen):
    """S7 (L2 R-b): Gradio draws a row's adjacent fields in one ``.form``; the shared row of
    fields (``shep-fields``) and grid of fields (``shep-field-grid``) reach them there: each
    field at least its floor (``--shep-field-min``), a row wrapping whole fields where one line
    cannot hold them at their floors (they stood on one line in equal shares under it), a grid's
    fields in columns of one width (the form stood in the grid's first column, one field a
    line)."""
    f, g = pieces_seen["fields"], pieces_seen["grid"]
    assert len(f["boxes"]) == 3 and all(b["width"] >= f["floor"] - 0.5 for b in f["boxes"]), f
    assert len({b["top"] for b in f["boxes"]}) > 1, f                      # wrapped, whole fields
    assert len(g["boxes"]) == 4 and all(b["width"] >= g["floor"] - 0.5 for b in g["boxes"]), g
    assert len({b["left"] for b in g["boxes"]}) > 1, g
    assert max(b["width"] for b in g["boxes"]) - min(b["width"] for b in g["boxes"]) <= 1, g


def test_a_choice_groups_title_and_its_help_stay_on_one_line_in_a_field_row(pieces_seen):
    """S8 (L2 R-c): in a field row (``shep-field-row``), a choice group's title and the ⓘ after it
    (its own children) stand on one line (the row drew each field as a column, so the ⓘ took a
    line of its own under the title)."""
    c = pieces_seen["choice"]
    assert c["info"]["top"] < c["title"]["bottom"] and c["info"]["bottom"] > c["title"]["top"], c
    assert c["info"]["left"] >= c["title"]["right"] - 1, c


def test_a_wrapped_names_fold_toggle_stands_on_its_first_line(pieces_seen):
    """S4 (L4 a): in the shared tree (``shep-mtree``) the fold toggle of a name that wraps stands
    on its first line (it was centred on the middle of three or four lines), and the first line
    lies within the toggle's height."""
    t = pieces_seen["tree"]
    assert t["lines"] >= 2, t
    assert t["fold"]["top"] <= t["first"]["top"] + 1 and t["first"]["bottom"] <= t["fold"]["bottom"] + 1, t


def test_the_layout_pass_keeps_a_fit_boxs_sideways_scroll_and_the_pages_scroll(pieces_seen):
    """S9 (L5 R1): the layout pass measures a fit box's table out of flow at its narrowest; it
    holds the scroll of the box's ancestors across the measure and puts it back, so a table
    scrolled sideways stays where it was and a page scrolled to its end does not move when
    anything is drawn (the box went back to 0 on every pass, the page jumped, and the shared
    focus check found Datasets' focused "In use" ⓘ out of view at 390 x 844)."""
    before, after = pieces_seen["scroll"]["before"], pieces_seen["scroll"]["after"]
    assert before["wide"] and before["fit"] > 0 and before["page"] > 0, before
    assert after == {"fit": before["fit"], "page": before["page"]}, (before, after)


#: A static page for the layout rules themselves (`browser.LOOK_JS`'s ``layout``): one page shown
#: as the shell shows one (its tab selected, its panels), with a panel and, outside it, a bar
#: (Training's run bar is no panel) whose long words stand beside its button; in the panel a
#: group whose "Details ›" link wraps under its box, a button in a box drawn as no box (``display:
#: contents``) that hides its overflow, a drawer (absolute, over the panel) whose button lies over
#: a control of the panel, and one button drawn over another.
_LAYOUT_RULES_PAGE_JS = r"""
const { chromium } = require('playwright');
const LOOK_JS = __LOOK_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  const words = 'Words that explain the button beside them at length, more than a hundred characters, '
    + 'as a long refusal once stood beside Start in the run bar.';
  await page.setContent(`<!doctype html><html><head><style>
    body { margin: 0; font: 15px sans-serif; background: #fff; }
    button, input { font: inherit; height: 32px; }
    .shep-panel { position: relative; width: 900px; height: 420px; border: 1px solid #ccc; }
    .shep-bar { display: flex; align-items: flex-start; gap: 16px; width: 900px; margin-top: 16px; }
    #g1 { display: flex; flex-wrap: wrap; width: 260px; gap: 4px; }
    #g1 input { width: 220px; }
    #dc { display: contents; overflow: hidden; }
    #under { position: absolute; left: 520px; top: 200px; width: 160px; }
    #drawer { position: absolute; z-index: 5; top: 60px; left: 480px; width: 400px; height: 340px; background: #fff; }
    #in-drawer { position: absolute; left: 60px; top: 136px; width: 160px; }
    #o2 { position: absolute; left: 40px; top: 4px; }
    .c-row { display: flex; height: 105px; margin-top: 16px; }
    .c-row button { height: 40px; }
  </style></head><body>
    <div id="shep-nav"><div class="tab-wrapper"><button role="tab" aria-selected="true" data-tab-id="x">X</button></div></div>
    <div id="shep-tab-x"><div class="shep-page">
      <div class="shep-panel" id="p1">
        <div class="shep-panel-hd"><h2>A panel</h2></div>
        <div class="shep-grp" id="g1"><input type="text" id="g1-in"><button type="button" class="shep-link" id="g1-link">Details ›</button></div>
        <div id="dc"><button type="button" id="dc-btn">In a box drawn as no box</button></div>
        <button type="button" id="under">Under the drawer</button>
        <div id="drawer"><button type="button" id="in-drawer">The drawer's</button></div>
        <div style="position: relative; margin-top: 16px"><button type="button" id="o1">The first button</button><button type="button" id="o2">Over it</button></div>
      </div>
      <div class="shep-bar" id="bar"><button type="button" id="bar-btn">Start</button><p id="bar-text">${words}</p></div>
      <div class="shep-bar" id="bar2"><button type="button" id="bar2-btn">Start</button><p id="bar2-text" style="width: 12em">Start disabled: GPU jobs need a workspaces folder.</p></div>
      <div class="shep-bar" id="bar3"><button type="button" id="bar3-btn">Start</button><p id="bar3-text">Start disabled: no folder.</p></div>
      <div class="c-row" style="align-items: center"><button type="button" id="c-mid">Start</button></div>
      <div class="c-row" style="align-items: flex-start"><button type="button" id="c-top">Start</button></div>
    </div></div></body></html>`);
  await page.addScriptTag({ content: LOOK_JS });
  const layout = await page.evaluate(() => window.__shepLook.check().layout);
  console.log(JSON.stringify(layout));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_layout_rules_read_a_bar_a_link_a_box_of_no_box_and_an_open_drawer(tmp_path):
    """S10 of the sweep's integration (the lanes' L1 4 and 5, L5 3, L3 R5): the layout rules
    (`browser.LOOK_JS`'s ``layout``), on a static page shown as the shell shows one:

    - ``beside`` reads a bar outside every panel as a region of its own (Training's run bar: its
      long words beside Start were never measured);
    - ``split-group`` does not count a group's "Details ›" link (``shep-link``) as one of its
      controls, as it does not count a ⓘ or a Copy;
    - ``clipped`` passes over a box drawn as no box (``display: contents``), which clips nothing
      (Gradio's hidden overflow on Diagnosis' Top-K block was read as a cut);
    - ``overlap`` does not count a control an open drawer lies over (the Test page's import
      drawer: 13 pairs, each a panel control under the drawer), and still counts a button drawn
      over another;
    - ``beside`` reads words drawn on two lines beside a button however few they are (the
      independent check, C4: it counted 100 characters, so a 51-character reason on two lines
      beside Start passed), and passes a one-line relation;
    - ``centred`` finds a button centred in a row about two and a half times its height (Fine-tune's
      Start, 40 px, in its 105 px row at 1280, which the rule's earlier form found: three times
      the button's height let it pass) and passes the same button on the row's top."""
    from tests.fixtures.browser import LOOK_JS, node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    found = run_node(_LAYOUT_RULES_PAGE_JS.replace("__LOOK_JS__", json.dumps(LOOK_JS)), tmp_path,
                     timeout=120)
    assert [b for b in found["beside"] if "#bar-text" in b and "#bar-btn" in b], found["beside"]
    assert not [s for s in found["split"] if "#g1" in s], found["split"]
    assert not [c for c in found["clipped"] if "#dc-btn" in c], found["clipped"]
    assert not [o for o in found["overlap"] if "#under" in o], found["overlap"]
    assert [o for o in found["overlap"] if "#o1" in o and "#o2" in o], found["overlap"]
    assert [b for b in found["beside"] if "#bar2-text" in b and "#bar2-btn" in b], found["beside"]
    assert not [b for b in found["beside"] if "#bar3-text" in b], found["beside"]
    assert [c for c in found["centred"] if "#c-mid" in c], found["centred"]
    assert not [c for c in found["centred"] if "#c-top" in c], found["centred"]


#: A static page for the rules the after screenshots asked for (the sweep's L7; `browser.LOOK_JS`'s
#: ``layout``, `seeAll`): one page shown as the shell shows one, in panels placed side by side, each
#: rule's bad case beside its good one (each named in the test).
_LAYOUT_L7_PAGE_JS = r"""
const { chromium } = require('playwright');
const LOOK_JS = __LOOK_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const lines = [1, 2, 3, 4, 5, 6].map((i) => `<p>Line ${i} of a region taller than its box.</p>`).join('');
  const wide = '<table style="width: 420px; table-layout: fixed"><tr><td>Name</td><td>Fingerprint</td>'
    + '<td>Not admitted: Gene, Patients-like-me ›</td></tr></table>';
  const long = 'serving reads this workspace · in your Training run bar, said in more words than one line holds';
  await page.setContent(`<!doctype html><html><head><style>
    body { margin: 0; font: 15px sans-serif; background: #fff; }
    p { margin: 0; }
    button, input { font: inherit; height: 32px; }
    .shep-panel { position: absolute; width: 440px; border: 1px solid #ccc; }
    .shep-scroll-cue[data-shep-more] { -webkit-mask-image: linear-gradient(to bottom, #000 calc(100% - 28px), transparent);
      mask-image: linear-gradient(to bottom, #000 calc(100% - 28px), transparent); }
    .rg { overflow: auto; height: 44px; margin-bottom: 6px; }
    .sw { overflow-x: auto; overflow-y: hidden; width: 200px; }
    .shep-bar { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 8px; width: 400px; margin-bottom: 8px; }
    .shep-grp { display: flex; gap: 4px; }
    .rt { display: flex; gap: 16px; margin-bottom: 8px; }
    .rt .block { width: 180px; }
    .mono { font: 14px monospace; margin-bottom: 6px; }
    .mono code { font: inherit; }
  </style></head><body>
    <div id="shep-nav"><div class="tab-wrapper"><button role="tab" aria-selected="true" data-tab-id="x">X</button></div></div>
    <div id="shep-tab-x"><div class="shep-page">
      <div class="shep-panel" id="p1" style="left: 0; top: 40px; height: 440px">
        <div class="rg" id="cue-bad">${lines}</div>
        <div class="rg shep-scroll-cue" data-shep-more id="cue-good">${lines}</div>
        <div class="rg" id="cue-pad" style="padding-bottom: 80px"><p>One short line, then room.</p></div>
        <div class="rg" id="cue-nobox"><div style="display: contents; overflow: hidden">${lines}</div></div>
        <div class="sw" id="side-bad">${wide}</div>
        <div class="shep-rs-cue">Wider than the panel: scroll sideways.</div>
        <div class="sw" id="side-good">${wide}</div>
        <div id="clip-bad" style="width: 120px; overflow: hidden; white-space: nowrap">A long sentence that runs past its box</div>
        <span id="clip-ell" title="A long value cut to an ellipsis, whole here"
          style="display: inline-block; width: 120px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis">A long value cut to an ellipsis, whole here</span>
        <span id="clip-ell-bad"
          style="display: inline-block; width: 120px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis">Another value cut, with no title</span>
        <span id="clip-sr" style="position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap">Copy the value</span>
      </div>
      <div class="shep-panel" id="p2" style="left: 460px; top: 40px; height: 440px">
        <div class="shep-bar" id="bar-end"><div class="shep-grp" style="width: 300px"><input type="text" style="width: 290px"></div>
          <div class="shep-grp" style="width: 200px; margin-left: auto"><button type="button">Start</button></div></div>
        <div class="shep-bar" id="bar-start"><div class="shep-grp" style="width: 300px"><input type="text" style="width: 290px"></div>
          <div class="shep-grp" style="width: 200px"><button type="button">Start</button></div></div>
        <div class="rt" id="rt-bad" style="align-items: flex-end">
          <div class="block" id="rt-bad-a"><span data-testid="block-info">Device</span><div>No CUDA device here.</div><input type="text"></div>
          <div class="block" id="rt-bad-b"><span data-testid="block-info">Run name</span><input type="text"></div></div>
        <div class="rt" id="rt-good" style="align-items: flex-start">
          <div class="block" id="rt-good-a"><span data-testid="block-info">Device</span><div>No CUDA device here.</div><input type="text"></div>
          <div class="block" id="rt-good-b"><span data-testid="block-info">Run name</span><input type="text"></div></div>
        <div class="shep-bar" id="bar-note"><div class="shep-grp"><button type="button" class="primary" id="dis-bar" disabled>Start</button></div>
          <div class="shep-bar-note" style="flex-basis: 100%"><p>Start disabled: no dataset is listed yet.</p></div></div>
        <div style="display: flex; gap: 16px">
          <div class="shep-area"><button type="button" id="area-btn">Show older runs</button></div>
          <div class="shep-area"><p id="area-words" style="width: 160px">Not a certified medical device; decision support for a clinician.</p></div></div>
      </div>
      <div class="shep-panel" id="p3" style="left: 920px; top: 40px; height: 440px">
        <p class="mono" style="width: 20ch"><code id="tok-flag"><span style="white-space: nowrap">run.py --generate</span><wbr><span style="white-space: nowrap">-samples</span></code></p>
        <p class="mono" style="width: 14ch; overflow-wrap: anywhere"><code id="tok-hyphen">/tmp/abc-0/-home-user-aaaa-bbbb-cccc</code></p>
        <p class="mono" style="width: 12ch"><code id="tok-path"><span style="white-space: nowrap">/very_long</span><wbr><span style="white-space: nowrap">/path_to</span><wbr><span style="white-space: nowrap">/a_file.py</span></code></p>
        <p class="mono" style="width: 30ch"><code id="tok-fp">3f563f74bb04</code></p>
        <p class="mono" id="sep-bad" style="width: 14ch; font-size: 15px">alpha&nbsp;· beta&nbsp;· gamma delta</p>
        <p class="mono" id="sep-good" style="width: 40ch; font-size: 15px">alpha&nbsp;· beta&nbsp;· gamma delta</p>
        <p class="mono" id="chain-bad" style="width: 26ch; font-size: 15px">trained on demo&nbsp;· dataset fingerprint</p>
        <p class="mono" id="chain-long" style="width: 12ch; font-size: 15px">a&nbsp;· a fact much longer than its box</p>
      </div>
      <div class="shep-panel" id="p4" style="left: 0; top: 500px; height: 260px">
        <button type="button" class="primary" id="dis-bad" disabled>Register this report</button>
        <div style="height: 24px"></div>
        <button type="button" id="dis-reason" disabled>Import a new version</button>
        <p class="shep-reason" style="margin-top: 4px">Import disabled: no results registry is set.</p>
        <button type="button" class="primary" id="dis-under" disabled style="position: absolute; left: 260px; top: 120px">Commit</button>
        <div id="layer" style="position: absolute; left: 240px; top: 100px; width: 180px; height: 70px; background: #fff; z-index: 3"></div>
        <div style="position: absolute; left: 16px; top: 200px"><button type="button" class="primary" id="dis-own" disabled>Use for training</button>
          <span class="shep-note" style="margin-left: 12px">No workspace is listed to use yet.</span></div>
      </div>
      <div class="shep-panel" id="p5" style="left: 460px; top: 500px; height: 180px">
        <div data-notice="workspaces_root">No workspaces folder is set.</div>
        <div style="height: 60px"></div>
        <button type="button" id="dis-sec" disabled>Refresh</button>
        <button type="button" class="primary" id="dis-prim" disabled style="margin-left: 120px">Load this model</button>
      </div>
      <div class="shep-panel" id="p6" style="left: 920px; top: 500px; height: 180px">
        <div class="sw" id="rh-bad"><table style="width: 520px; table-layout: fixed"><colgroup><col style="width: 150px"><col style="width: 150px"><col style="width: 220px"></colgroup>
          <tr id="rh-bad-row"><td>demo</td><td>3f56</td><td>${long}</td></tr></table></div>
        <div class="sw" id="rh-good"><table style="width: 520px; table-layout: fixed"><colgroup><col style="width: 150px"><col style="width: 150px"><col style="width: 220px"></colgroup>
          <tr id="rh-good-row"><td>demo</td><td>3f56</td><td>in use</td></tr></table></div>
        <div class="rg shep-scroll-cue" data-shep-more id="iv-region"><div style="height: 1200px"></div><button type="button" id="iv-in">Apply</button></div>
        <button type="button" id="iv-out" style="position: absolute; left: 16px; top: 1000px">Apply Settings</button>
        <div style="display: contents; overflow: auto"><button type="button" id="iv-nobox" style="position: absolute; left: 200px; top: 1000px">Restart</button></div>
      </div>
    </div></div></body></html>`);
  await page.addScriptTag({ content: LOOK_JS });
  const layout = await page.evaluate(() => window.__shepLook.check().layout);
  console.log(JSON.stringify(layout));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def l7_rules_seen(tmp_path_factory):
    """The layout rules' findings on `_LAYOUT_L7_PAGE_JS`, once for the module."""
    from tests.fixtures.browser import LOOK_JS, node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    return run_node(_LAYOUT_L7_PAGE_JS.replace("__LOOK_JS__", json.dumps(LOOK_JS)),
                    tmp_path_factory.mktemp("l7-rules"), timeout=120)


def _hits(found, key, element):
    return [line for line in found[key] if element in line]


def test_the_in_view_and_scroll_cue_rules_find_a_page_cut_at_its_foot(l7_rules_seen):
    """SW-LK1, SW-LK2, SW-LK5 and SW-LK8 (the L6 after screenshots): Settings' Data locations,
    Fine-tune's form, Training's charts and Diagnosis' inputs ran past their scroll region's foot
    with no cue, and the walk found nothing. ``in-view`` finds a control drawn past the window
    that no scroll region in view holds (and passes one a region in view holds); ``scroll-cue``
    finds a region whose words run past its foot with no cue (and passes the R-73 cue's mask and a
    region whose overflow is only its padding), and a box whose table runs past its right edge with
    no sideways cue (and passes one with Compare's drawn cue before it). A box drawn as no box
    (``display: contents``, Gradio's blocks around Diagnosis' Top-K) hides and scrolls nothing:
    words in one are judged, not taken as hidden for a screen reader or held by a region."""
    found = l7_rules_seen
    assert _hits(found, "inview", "#iv-out") and _hits(found, "inview", "#iv-nobox"), found["inview"]
    assert not _hits(found, "inview", "#iv-in"), found["inview"]
    assert _hits(found, "cue", "#cue-bad holds") and _hits(found, "cue", "#cue-nobox holds"), found["cue"]
    assert not _hits(found, "cue", "#cue-good") and not _hits(found, "cue", "#cue-pad"), found["cue"]
    assert _hits(found, "cue", "#side-bad holds"), found["cue"]
    assert not _hits(found, "cue", "#side-good"), found["cue"]
    assert not _hits(found, "cue", "#iv-region"), found["cue"]


def test_the_clipped_rule_finds_words_a_box_cuts(l7_rules_seen):
    """SW-LK3 and SW-LK4: words cut by a box that hides its overflow were never measured (only
    controls were). ``clipped`` finds a text cut by its box and an ellipsis with no title, and
    passes an ellipsis whose whole words are its title and a screen reader's hidden words."""
    found = l7_rules_seen
    assert [c for c in found["clipped"] if "cut by #clip-bad" in c], found["clipped"]
    assert [c for c in found["clipped"] if "cut by #clip-ell-bad" in c], found["clipped"]
    assert not [c for c in found["clipped"] if "cut by #clip-ell " in c or "#clip-sr" in c], found["clipped"]


def test_the_wrap_end_rule_finds_a_group_pushed_to_the_end_of_its_line(l7_rules_seen):
    """SW-LK5: Training's run bar wrapped its action group to a second line at the line's end,
    the start of that line empty. ``wrap-end`` finds a bar's later line that does not start on
    the bar's start edge, and passes a bar whose wrapped group starts it."""
    found = l7_rules_seen
    assert _hits(found, "wrapEnd", "#bar-end line 2"), found["wrapEnd"]
    assert not _hits(found, "wrapEnd", "#bar-start"), found["wrapEnd"]


def test_the_disabled_reason_rule_finds_a_disabled_button_with_nothing_said(l7_rules_seen):
    """SW-LK6: Test's Register this report and Models' Load this model were disabled with no
    visible reason. ``disabled-reason`` finds a disabled button with nothing said in its group or
    under it, one whose words beside it are a page's own form rather than the shared reason line
    (C11), and a region's primary action whose region has only the notice; it passes a button with
    its reason line under it, one whose bar's note says why, a secondary button in a region that
    draws the notice, and a button an open layer (a drawer) covers."""
    found = l7_rules_seen
    assert [d for d in _hits(found, "disabled", "#dis-bad") if "nothing said" in d], found["disabled"]
    assert _hits(found, "disabled", "#dis-prim"), found["disabled"]
    assert [d for d in _hits(found, "disabled", "#dis-own") if "own form" in d], found["disabled"]
    for quiet in ("#dis-reason", "#dis-sec", "#dis-under", "#dis-bar"):
        assert not _hits(found, "disabled", quiet), (quiet, found["disabled"])


def test_the_token_separator_and_chain_rules_find_what_reads_broken(l7_rules_seen):
    """SW-LK7 and SW-LK2: Datasets' build command broke inside its flag ("--generate" /
    "-samples"), Settings' path broke after its hyphens, Fine-tune's facts line ended at a
    separator, and Diagnosis split "dataset / fingerprint". ``token-break`` finds a word of a
    token broken though it fits its box and a line ending at a hyphen inside a word, and passes a
    path longer than its box that breaks before its separators and a fingerprint on one line;
    ``sep-end`` finds a line ending at " · " and passes the chain on one line; ``chain-break``
    finds a fact of a chain broken though it fits a line of its own and passes a fact longer than
    its box."""
    found = l7_rules_seen
    assert _hits(found, "token", "#tok-flag"), found["token"]
    assert [t for t in _hits(found, "token", "#tok-hyphen") if "hyphen" in t], found["token"]
    assert not _hits(found, "token", "#tok-path") and not _hits(found, "token", "#tok-fp"), found["token"]
    assert _hits(found, "sepEnd", "#sep-bad"), found["sepEnd"]
    assert not _hits(found, "sepEnd", "#sep-good"), found["sepEnd"]
    assert [c for c in _hits(found, "chain", "#chain-bad") if "dataset fingerprint" in c], found["chain"]
    assert not _hits(found, "chain", "#chain-long"), found["chain"]


def test_the_row_rules_find_labels_off_one_line_and_a_row_a_hidden_cell_sets(l7_rules_seen):
    """SW-LK2 and SW-LK3: Fine-tune's Run name label stood 23 px under Device's in one row of
    fields, and the Datasets table's selected row at 390 px was 2.5 times the other's height, set
    by a cell scrolled out of view. ``row-top`` finds two labelled fields side by side whose labels
    do not share a top line and passes the same row on its top line; ``row-height`` finds the row
    a cell out of view sets and passes one whose cells in view set it. ``beside`` passes the words
    of one area a page marks (``shep-area``) beside the button of another (a list and its chosen
    item's details)."""
    found = l7_rules_seen
    assert _hits(found, "rowTop", "#rt-bad-a"), found["rowTop"]
    assert not _hits(found, "rowTop", "#rt-good-a"), found["rowTop"]
    assert _hits(found, "rowHeight", "#rh-bad-row"), found["rowHeight"]
    assert not _hits(found, "rowHeight", "#rh-good-row"), found["rowHeight"]
    assert not [b for b in found["beside"] if "#area-words" in b], found["beside"]


def test_the_layout_problems_hold_in_view_at_the_owners_windows_and_tab_groups_whole():
    """``in-view`` is held at the owner's windows only (`browser.OWNER_SIZES`: a page scrolls
    on a phone). ``tab-groups`` does not count a tab group the row's width wraps apart (SW-LK9:
    Models alone on the second line at 390 px): the tabs wrap by width alone and no group is
    broken to a line of its own, the sweep's decision 3 (the integrator's, 2026-11-22; the
    owner may reopen it). It still finds a separator drawn while the tabs wrap."""
    from tests.fixtures.browser import OWNER_SIZES, layout_problems

    def stop(size, inview=(), lines=None):
        layout = {"inview": list(inview), "tabs": None if lines is None else
                  {"lines": lines, "seps": [], "others": [], "focusable": []}}
        return {"size": size, "page": "settings", "check": {"layout": layout}}

    owner = "x".join(map(str, OWNER_SIZES[0]))
    found = layout_problems([stop(owner, ["#x \"Apply\" drawn 40 px under the window"]),
                             stop("390x844", ["#x \"Apply\" drawn 900 px under the window"])],
                            rules=("in-view",))
    assert found == [f"{owner} settings: in-view: #x \"Apply\" drawn 40 px under the window"], found
    split = layout_problems([stop("390x844", lines=[["datasets", "training", "finetune", "test"], ["models"]])],
                            rules=("tab-groups",))
    assert split == [], split
    whole = layout_problems([stop("390x844", lines=[["datasets", "training", "finetune"], ["test", "models"]])],
                            rules=("tab-groups",))
    assert whole == [], whole
    seps = {"lines": [["datasets", "training", "finetune", "test"], ["models"]],
            "seps": [{"tab": "test", "lineStart": False}], "others": [], "focusable": []}
    drawn = layout_problems([{"size": "390x844", "page": "settings", "check": {"layout": {"tabs": seps}}}],
                            rules=("tab-groups",))
    assert drawn and all("a separator before test" in p for p in drawn), drawn


@pytest.mark.parametrize("state", ["first_use", "empty", "configured"])
def test_every_page_of_every_state_has_no_layout_finding(state, request):
    """**The layout rules held at zero** (the sweep's integration, its decision 4; no allow-list):
    every page as an operator sees it, at every layout size (`browser.LAYOUT_SIZES`: the owner's
    window, the owner's Firefox width, two laptops, a portrait column and a phone, then the
    owner's again), at first use (no workspaces folder, no results registry: the owner's six
    screenshots), with an empty workspaces folder and registry, and configured (the demo
    workspace, a child served), and its drawers open as a reader first sees them (the Diagnosis
    Load drawer in every state; Training's Continue drawer and the Test page's import drawer
    configured), has none of the layout rules' findings (`browser.layout_problems`: beside,
    narrow-box, overlap, orphan, centred, split-group, edge, clipped, empty-region,
    help-on-page, notice-repeat, tab-groups, and since L7 in-view at the owner's windows,
    scroll-cue, wrap-end, disabled-reason, token-break, sep-end, chain-break, row-top,
    row-height). One service a state, one browser a walk."""
    from tests.fixtures.browser import layout_problems

    seen = request.getfixturevalue({"first_use": "ui_walk_first_use", "empty": "ui_walk_empty",
                                    "configured": "ui_walk_configured"}[state])
    # What was measured, so that none means something: every page at every size, every drawer,
    # and (configured) the Load drawer with a row chosen at each of its sizes.
    assert _walk_coverage(seen, state) == []
    assert layout_problems(seen) == []


def _walk_coverage(seen: list, state: str) -> list:
    """**What a layout walk of `state` must have measured** (E1's coverage), as the
    problems found: every page at every layout size; every drawer the state opens; and,
    configured, the Load drawer with a row chosen at each of `_PICKED_SIZES`, its chosen
    model's facts drawn (B-L6-4; the sweep's L7, FX-2: the walk recorded only which drawers
    opened, so removing the row-chosen stops passed unseen); no other state chooses a row."""
    from tests.fixtures.browser import LAYOUT_SIZES, UI_PAGES

    problems = []
    pages = [s for s in seen if not s.get("open") and "check" in s and "layout" in s["check"]]
    missing = {(page, f"{w}x{h}") for page in UI_PAGES for w, h in LAYOUT_SIZES} - {
        (s["page"], s["size"]) for s in pages}
    if missing:
        problems.append(f"pages not measured: {sorted(missing)}")
    drawers = {"#shep-dx-change"} | ({"#shep-train-drawer", "#shep-test-import"}
                                     if state == "configured" else set())
    opened = {s["open"] for s in seen if s.get("open")}
    if opened != drawers:
        problems.append(f"drawers opened {sorted(opened)}, wanted {sorted(drawers)}")
    picked = {s["size"] for s in seen if s.get("open") == "#shep-dx-change" and s.get("picked")}
    wanted = {f"{w}x{h}" for w, h in _PICKED_SIZES} if state == "configured" else set()
    if picked != wanted:
        problems.append(f"Load drawer stops with a row chosen at {sorted(picked)}, wanted "
                        f"{sorted(wanted)}")
    return problems


def test_the_walks_coverage_names_a_row_chosen_stop_that_did_not_run():
    """FX-2 (the sweep's L7): E1 checked which drawers opened, not that the configured
    walk's row-chosen Load drawer stops ran; with them removed it still passed. The coverage
    now names each `_PICKED_SIZES` stop that did not choose a row and draw its facts, and a
    state that chose one where none is asked."""
    from tests.fixtures.browser import LAYOUT_SIZES, UI_PAGES

    pages = [{"page": page, "size": f"{w}x{h}", "check": {"layout": {}}}
             for page in UI_PAGES for w, h in LAYOUT_SIZES]
    drawers = [{"page": page, "size": "1850x960", "open": shown}
               for page, shown in (("clinical", "#shep-dx-change"), ("training", "#shep-train-drawer"),
                                   ("test", "#shep-test-import"))]
    picked = [{"page": "clinical", "size": f"{w}x{h}", "open": "#shep-dx-change", "picked": "a/b.pt"}
              for w, h in _PICKED_SIZES]
    assert _walk_coverage(pages + drawers + picked, "configured") == []
    gone = _walk_coverage(pages + drawers + picked[:-1], "configured")
    assert len(gone) == 1 and "row chosen" in gone[0] and "390x844" in gone[0], gone
    blank = [{**stop, "picked": ""} for stop in picked]           # facts never drawn
    assert _walk_coverage(pages + drawers + blank, "configured"), blank
    assert _walk_coverage(pages + drawers[:1], "first_use") == []
    assert _walk_coverage(pages + drawers[:1] + picked, "first_use")


def test_prose_in_a_region_on_a_panels_edge_keeps_the_measure_and_a_reason_draws_no_box(built):
    """The independent check of the UI sweep (C3): Models' details column is
    a region on a panel's edge (`shep-edge`), not a `.shep-panel`, so the
    shared prose rule never reached its paragraphs: "No result is registered
    beside this checkpoint…" ran 138 characters a line at 1850 x 960 (169 for
    a read-only viewer), against the text brief's 56ch. The shared rule now
    covers a region on a panel's edge as it covers a panel, so Models'
    details and the shared list's columns keep the measure. The neutral
    reason (`shep-reason-grey`, beside Load on Models and in the Diagnosis
    drawer) was the one reason drawn in a filled box; it is a reason line
    like every other."""
    from src.webui.style import SHELL_CSS

    prose = _rule(SHELL_CSS, ":where(.shep-panel p, .shep-edge p)")
    assert "max-width: var(--shep-measure)" in prose
    demo, _ctx = built
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    assert "shep-edge" in (by_id["shep-mp-detail"].elem_classes or [])
    grey = _rule(SHELL_CSS, ".shep-reason-grey")
    assert not re.search(r"\b(border|background|padding)\b", grey), grey


def test_a_state_said_as_a_sentence_and_a_note_in_a_span_keep_the_measure():
    """The independent check of the UI sweep (C7): Settings' first-use line
    ("Nothing is set yet. Each folder is needed only by what its row names,
    …", 118 characters) was a status mark, an inline span the measure never
    reached: one line 816 px wide at 1550 x 830, where every go link lands.
    Datasets' note beside Use for training was an inline span too, its
    max-width ignored (493 px). A state said as a sentence is a reason line
    led by its mark (`renderers.marked_line`: a block at the measure, the mark
    in a column of its own); a note in a span is an inline box that keeps the
    measure (both 470 px, measured)."""
    from src.webui.components import data_locations as dl
    from src.webui.components import renderers as r
    from src.webui.style import SHELL_CSS

    line = r.marked_line("neutral", dl.NOTHING_SET)
    assert line.startswith('<p class="shep-reason">') and 'aria-hidden="true">○' in line
    assert r.e(dl.NOTHING_SET) in line
    assert "display: inline-block" in _rule(SHELL_CSS, "span.shep-note")
    assert "max-width: var(--shep-measure)" in _rule(SHELL_CSS, ".shep-note")
    assert "max-width: var(--shep-measure)" in _rule(SHELL_CSS, ".shep-reason")
