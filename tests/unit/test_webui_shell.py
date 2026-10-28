"""
The dashboard's shell (M5-2, UI-A): navigation, the page contract, the head.
============================================================================
`src/webui/shell.py`, `src/webui/app.py`. What these pin:

- **two navigation levels**: "Diagnosis (clinical)" first and selected,
  "Research (pre-clinical)", "⚙ Settings" aligned right; under Research,
  Datasets · Training · Test · Models with Training selected; both levels wrap,
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
- the access log keeps the no-queue event path quiet (the chips' timer).

Module: tests/unit/test_webui_shell.py
"""
from __future__ import annotations

import re

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
    assert len(links) == 7


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
    assert research.inputs == [ctx.location, ctx.research_page, *tokens]
    # queued: Gradio 6.30 dispatches a State's .change only from a queued answer
    assert research.outputs == [ctx.location, *tokens] and research.queue is not False
    location, *bumped = research.fn("clinical", "datasets", 0, 4, 0, 0, 0)
    assert location == "datasets" and bumped == [1, gr.update(), gr.update(), gr.update(),
                                                 gr.update()]
    location, *bumped = research.fn("settings", None, 0, 4, 0, 0, 0)  # nothing chosen: Training
    assert location == "training" and bumped == [gr.update(), 5, gr.update(), gr.update(),
                                                  gr.update()]
    location, *bumped = research.fn("clinical", "finetune", 0, 0, 2, 0, 0)
    assert location == "finetune" and bumped[2] == 3
    for page in shell.RESEARCH_PAGES:
        (own,) = [fn for fn in demo.fns.values() if (ctx.tabs[page]._id, "select") in fn.targets]
        assert own.inputs == [ctx.location, ctx.page_tokens[page]]
        assert own.outputs == [ctx.location, ctx.research_page, ctx.page_tokens[page]]
        assert own.fn("clinical", 2) == (page, page, 3) and own.queue is not False
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
    location, *bumped = research.fn("clinical", "training", 0, 0, 0, 0, 0)
    assert location == "training" and bumped[1] == 1
    assert inner["training"].fn(location, 1) == ("training", "training", gr.update())
    # the other order: the inner select first, then Research's
    page, _, token = inner["models"].fn("clinical", 0)
    assert (page, token) == ("models", 1)
    assert research.fn(page, "models", 0, 0, 0, 0, 1) == (
        "models", gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    # within Research, another page's tab still refreshes it
    assert inner["test"].fn("training", 4) == ("test", "test", 5)
    # returning to Research from another page refreshes the page shown
    assert research.fn("settings", "test", 0, 0, 0, 5, 0)[4] == 6
    for fn in (research, *inner.values()):
        assert fn.concurrency_id == shell.OPEN_PAGE_CONCURRENCY and fn.concurrency_limit == 1


def demo_builders():
    from src.webui.app import PAGES

    return set(PAGES.values())


def test_the_location_follows_the_page_shown(built):
    demo, ctx = built
    for tab in (shell.CLINICAL, shell.SETTINGS):
        shown = [fn for fn in demo.fns.values() if (ctx.tabs[tab]._id, "select") in fn.targets
                 and fn.outputs == [ctx.location]]
        assert len(shown) == 1 and shown[0].fn() == tab and shown[0].queue is False


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
    # The one-screen rule reaches the page frame: it is not left to `css`.
    assert "@media (min-width: 1400px) and (min-height: 760px)" in head
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
    with gr.Blocks():
        block = shell.plain_html("<b>x</b>", elem_id="x", elem_classes="extra")
    assert block.apply_default_css is False and block.js_on_load is None
    assert block.elem_classes == ["shep-html", "extra"]


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


def _wide(css):
    """The wide layout's block (``@media (min-width: 1400px) and (min-height: 760px)``)."""
    start = css.index("@media (min-width: 1400px) and (min-height: 760px) {")
    return css[start:css.index("\n}\n", start)]


def test_a_select_value_cut_to_an_ellipsis_carries_its_whole_text_as_title():
    """M5-3: Training's dataset select, Test's workspace select and Models'
    cohort-version select cut their values with no title. One rule for every
    select: the value ends in an ellipsis, and the shell's head script gives a
    value that does not fit its whole text as its title (and takes it away
    when it fits), on each layout pass and when the pointer or focus arrives."""
    from src.webui.style import SHELL_CSS

    assert "text-overflow: ellipsis" in _rule(SHELL_CSS, '.gradio-container .wrap-inner input[role="combobox"]')
    head = shell.SHELL_HEAD
    assert "var TITLED = 'input[role=\"combobox\"], .shep-seg label';" in head
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

    wide = _wide(SHELL_CSS)
    rule = _rule(wide, "  .shep-panel .shep-panel-hd")
    assert "position: sticky" in rule and "top: 0" in rule and "background: var(--shep-panel)" in rule
    # never at a narrow size, where the page scrolls
    narrow = SHELL_CSS[SHELL_CSS.index("@media (max-width: 1399px), (max-height: 759px) {"):]
    assert "position: sticky" not in narrow
