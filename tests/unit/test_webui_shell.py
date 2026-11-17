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
  3:1, the group captions apart from the tabs, one " · " per option).

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
    """Every writer of the location (Diagnosis', Settings', Research's and
    each research tab's select) is queued in one group of one, each select
    sent once (UI-14, B46-mount: Diagnosis' and Settings' ran without the
    queue, so the select Diagnosis' tab fires when it mounts could write the
    location after a page opened later)."""
    demo, ctx = built
    for tab in (shell.CLINICAL, shell.SETTINGS):
        shown = [fn for fn in demo.fns.values() if (ctx.tabs[tab]._id, "select") in fn.targets
                 and fn.outputs == [ctx.location]]
        assert len(shown) == 1 and shown[0].fn() == tab
    writers = [fn for fn in demo.fns.values() if ctx.location in fn.outputs]
    assert len(writers) == 2 + 1 + len(shell.RESEARCH_PAGES)
    for fn in writers:
        assert fn.queue is not False and fn.concurrency_id == shell.OPEN_PAGE_CONCURRENCY
        assert fn.concurrency_limit == 1 and fn.trigger_mode == "multiple", fn.name


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


def test_the_wrapped_chips_sit_on_their_own_line_under_the_stage_switch():
    """B5 (leftover b): at 480 px the stage switch's two tabs take two lines,
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
#: runs the header's layout, as a load does), and reports for each stage tab
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
  for (const width of [480, 360, 420, 760, 1100, 1850]) {
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
    360, 420, 480, 760, 1100 and 1850 px each stage tab is what
    ``document.elementsFromPoint`` finds first at its centre and corners, the
    chips never over it; at 480 the header has wrapped onto its second line;
    every chip is inside the window (UI-02: at 360 px the D8 chip's left edge
    was at -41 px; its words now end in a titled ellipsis inside the room)."""
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

    def late():
        import time

        time.sleep(slow["seconds"])
        return original()

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
  const facts = async () => ({ outer: await chosen('shep-nav'), inner: await chosen('shep-research-nav'),
                               kept: await page.evaluate((k) => sessionStorage.getItem(k), '__KEPT__'),
                               title: await page.title() });
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
    themselves. This browser tab keeps the page id alone."""
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
              "title": shell.TITLES["models"], "location": "models"}
    assert seen["opened"] == models, seen
    assert seen["reloaded"] == models, seen
    assert seen["settings"]["outer"] == "settings" and seen["settings"]["kept"] == "settings"
    assert seen["settings"]["title"] == shell.TITLES["settings"]
    assert seen["clinical"]["outer"] == "clinical" and seen["clinical"]["kept"] == "clinical"


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
def test_the_group_captions_are_sentence_case_overlines_without_fitted_margins():
    """R-22+R-37: PREPARE and EVALUATE were upper-case ::before labels on the
    tabs' own line, set off by margins fitted to their widths (84 and 116 px).
    They are captions on a tier of their own above the first tab of each
    group, in sentence case at 14 px / 600, never hit (pointer-events none),
    and the groups are a group gap apart; what the browser draws is
    `test_the_research_navigation_and_shared_tokens_as_drawn`'s."""
    from src.webui.style import SHELL_CSS

    caption = _rule(SHELL_CSS, "#shep-page-datasets-button::before, #shep-page-test-button::before")
    assert "bottom: calc(100% + 2px)" in caption and "pointer-events: none" in caption
    assert "font-size: 14px" in caption and "font-weight: 600" in caption
    assert "uppercase" not in caption and "letter-spacing" not in caption
    assert "margin-left: 84px" not in SHELL_CSS and "margin-left: 116px" not in SHELL_CSS
    assert "#shep-page-test-button { margin-left: var(--shep-group-gap); }" in SHELL_CSS
    # The wrapped Evaluate line has no divider at its start (NEW-shell-nav-theme-2).
    assert ("#shep-shell.shep-rnav-wrapped #shep-research-nav > .tab-wrapper > .tab-container"
            " > #shep-page-test-button::after { display: none !important; }") in SHELL_CSS
    head = shell.SHELL_HEAD
    assert "shell.classList.toggle('shep-rnav-wrapped', need > row.clientWidth);" in head
    assert "if (shell) { researchLayout(shell); }" in head


#: The dashboard as the service mounts it, at 1850 x 960 then 420 x 900 and
#: back: the research row's geometry, captions and cues, level 1's selected
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
      const shell = document.getElementById('shep-shell'), L = window.__shepLook;
      const caps = {};
      for (const id of ['datasets', 'test']) {
        const c = L.pseudo(document.getElementById(`shep-page-${id}-button`), '::before');
        c.hit = L.topAt(c.left + c.width / 2, c.top + c.height / 2);
        caps[id] = c;
      }
      const divider = L.pseudo(document.getElementById('shep-page-test-button'), '::after');
      return { tabs, caps, divider: { display: divider.display, left: divider.left },
               wrapped: shell.classList.contains('shep-rnav-wrapped'),
               headerAndTabs: document.querySelector('#shep-research-nav > .tab-wrapper')
                 .getBoundingClientRect().bottom - shell.getBoundingClientRect().top,
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
  await page.setViewportSize({ width: 420, height: 900 });
  await page.waitForFunction(() => document.getElementById('shep-shell').classList.contains('shep-rnav-wrapped'),
                             null, { timeout: 10000 });
  out.narrow = await row();
  await page.setViewportSize({ width: 1850, height: 960 });
  await page.waitForFunction(() => !document.getElementById('shep-shell').classList.contains('shep-rnav-wrapped'),
                             null, { timeout: 10000 });
  out.back = await row();
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""

#: What a read-only viewer's Settings and header draw (no deployment mode).
_READ_ONLY_LOOK_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__, LOOK_JS = __LOOK_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await page.addInitScript(QUEUE_JS);
  await page.addInitScript(LOOK_JS);
  await page.goto(process.argv[2], { waitUntil: 'load' });
  await page.waitForSelector('#shep-chips .shep-chip-readonly', { timeout: 60000 });
  await page.click('#shep-tab-settings-button');
  await page.waitForSelector('#allocator_preset input:disabled', { timeout: 30000 });
  await page.waitForFunction(() => window.__shepQueue.inflight === 0, null, { timeout: 30000 });
  console.log(JSON.stringify(await page.evaluate(() => {
    const s = (sel) => { const c = getComputedStyle(document.querySelector(sel));
      return { background: c.backgroundColor, border: c.borderTopColor, color: c.color }; };
    return { chip: s('#shep-chips .shep-chip-readonly'),
             fields: window.__shepLook.fields(document.getElementById('shep-tab-settings')),
             select: s('#allocator_preset .wrap:has(> .wrap-inner)'), value: s('#allocator_preset input'),
             box: s('#torch_compile input[type=checkbox]'), label: s('#torch_compile label') };
  })));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _rgb(hex_colour):
    return "rgb({}, {}, {})".format(*(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)))


def test_the_research_navigation_and_shared_tokens_as_drawn(tmp_path, monkeypatch):
    """D01 and UI-10 in a real browser (Node and Playwright, skipped without),
    the dashboard mounted as the service mounts it, for an operator:

    - **D01 / R-22+R-37**: at 1850 x 960 header and tabs take about 100 px
      (102), the research tabs are 32 px, each group's caption sits above its
      first tab (not on its line), aligned with its text, in sentence case at
      14 px / 600, and ``elementsFromPoint`` at a caption finds no tab; at
      420 px the Evaluate group wraps whole onto its own line with no divider
      at its start (NEW-shell-nav-theme-2), and back at 1850 one row again;
    - **cues**: a tab under the pointer has a fill and an underline it lacks
      at rest, the selected one the accent line and text, keyboard focus a
      2 px ring in the focus token;
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
    assert 96 <= wide["headerAndTabs"] <= 104, wide["headerAndTabs"]
    assert {round(t["height"]) for t in wide["tabs"].values()} == {32}
    assert not wide["wrapped"] and not wide["hScroll"]
    for group, first in (("datasets", "datasets"), ("test", "test")):
        cap, tab = wide["caps"][group], wide["tabs"][first]
        assert cap["top"] + cap["height"] <= tab["top"], (group, cap, tab)     # above, not on its line
        assert abs(cap["left"] - (tab["left"] + 12)) <= 1, (group, cap, tab)   # with the tab's text
        assert (cap["fontSize"], cap["fontWeight"], cap["textTransform"]) == ("14px", "600", "none")
        assert cap["pointerEvents"] == "none" and cap["hit"] and not cap["hit"]["tab"], cap
    assert wide["divider"]["display"] == "block"
    assert wide["divider"]["left"] < wide["tabs"]["test"]["left"]
    assert wide["divider"]["left"] > wide["tabs"]["finetune"]["right"]
    narrow = seen["narrow"]
    assert narrow["wrapped"] and not narrow["hScroll"]
    assert abs(narrow["tabs"]["test"]["left"] - narrow["tabs"]["datasets"]["left"]) <= 1
    assert narrow["tabs"]["test"]["top"] > narrow["tabs"]["finetune"]["bottom"]
    assert narrow["tabs"]["models"]["top"] == narrow["tabs"]["test"]["top"]          # the group whole
    assert narrow["caps"]["test"]["top"] >= narrow["tabs"]["finetune"]["bottom"]
    assert narrow["divider"]["display"] == "none"
    assert seen["back"]["headerAndTabs"] == wide["headerAndTabs"] and not seen["back"]["wrapped"]

    rest, hover, selected = seen["rest"], seen["hover"], seen["selected"]
    assert hover["background-color"] != rest["background-color"]
    assert hover["border-bottom-color"] != rest["border-bottom-color"]
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
    operator sees; the header's read-only chip is the amber lozenge."""
    from src.webui import style
    from src.webui.app import create_gradio_app, mount_options
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
    assert seen["select"]["background"] == _rgb(off) and seen["select"]["border"] == _rgb(line)
    assert seen["value"]["color"] == _rgb(words) and contrast(words, off) >= 4.5
    assert seen["box"]["background"] == _rgb(off) and seen["box"]["border"] == _rgb(line)
    assert seen["label"]["color"] == _rgb(words)
    assert seen["chip"]["background"] == _rgb(tokens["--shep-amber-50"])
    assert seen["chip"]["color"] == _rgb(tokens["--shep-amber-800"])
    assert any(f["disabled"] for f in seen["fields"])


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
    assert "getPropertyValue('--shep-panel-min')" in js
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
  const { box, panels: kids } = window.__shepLook.shownPage();
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
    runbar = { h: Math.round(bar.getBoundingClientRect().height),
               lines: new Set(items.map((r) => Math.round((r.top + r.bottom) / 2 / 24))).size };
  }
  // A box's top and bottom as drawn, or null when it is not drawn.
  const at = (e) => { if (!e) { return null; } const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 ? [Math.round(r.top), Math.round(r.bottom)] : null; };
  // Training's Continue drawer, open: its body, step 3's row (Check checkpoint and its prompt),
  // and what the run bar draws of its actions and says beside them (R-48, R-49,
  // NEW-training-finetune-2).
  let drawer = null;
  const dr = document.getElementById('shep-train-drawer');
  if (dr && dr.getClientRects().length) {
    drawer = { body: at(document.getElementById('shep-train-drawer-bd')),
               check: at(document.getElementById('shep-train-check')),
               prompt: at(document.getElementById('shep-train-check-prompt')),
               actions: ['shep-train-start', 'shep-train-continue', 'shep-train-stop']
                 .filter((i) => at(document.getElementById(i))),
               reason: document.getElementById('shep-train-reason').innerText.trim(),
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

#: The Diagnosis load drawer, opened at a stop of the walk (`browser.UI_WALK_JS`).
_CHANGE_DRAWER = {"click": "#shep-dx-change-btn", "shown": "#shep-dx-change",
                  "close": "#shep-dx-change-close button"}
#: Training's Continue drawer, opened at a stop of the walk; its one closer is
#: the × in its header (R-49).
_CONTINUE_DRAWER = {"click": "#shep-train-continue", "shown": "#shep-train-drawer",
                    "close": "#shep-train-drawer-close"}

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
    # to "W" (the T04 review), and to 71 px on a phone; and in a short window.
    stops += [{"size": [w, h], "page": "clinical", "open": _CHANGE_DRAWER}
              for w, h in ((1280, 900), (1366, 620), (390, 844))]
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
def ui_walk(tmp_path_factory, _suite_gpu_lock):
    """The shared UI measurement walk, once for this module: the service in a
    process of its own (`served_page.served_service`, mode M1, an operator) on
    the demo workspace with a base and a disease child trained on it where the
    server lists them (`trained_children.train_children`, 1 epoch each on the
    CPU), the disease child served (the Diagnosis card as an operator sees
    it); one browser through `_walk_stops` (`browser.walk_ui`), the layout
    facts at every stop, the shared check and a screenshot at the checked ones
    (under the module's ``ui-walk*/shots``)."""
    from tests.fixtures.browser import node_with_playwright, walk_ui
    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.served_page import served_service
    from tests.fixtures.trained_children import train_children

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    root = tmp_path_factory.mktemp("ui-walk")
    demo = write_demo_workspace(root / "workspaces" / "demo")
    served = train_children(demo, ("disease",))["disease"]
    (root / "registry").mkdir()
    (root / "shots").mkdir()
    settings = {"SHEPHERD_WORKSPACES_ROOT": str(root / "workspaces"),
                "SHEPHERD_REGISTRY": str(root / "registry"), "SHEPHERD_DEPLOYMENT_MODE": "M1",
                "SHEPHERD_DEVICE": "cpu", "SHEPHERD_KG_PATH": str(demo / "kg.json"),
                "SHEPHERD_CHECKPOINT_PATH": str(served), "SHEPHERD_DATA_DIR": str(demo)}
    with served_service(root / "service", settings) as page:
        return walk_ui(page, root, _walk_stops(), extra_js=_LAYOUT_FACTS_JS, shots=root / "shots")


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
    datasets_stacked = {"1280x900", "1366x620", "1280x600"}          # its own minimum, about 1400 px
    for s in ui_walk:
        where, facts = (s["size"], s["page"]), s["extra"]
        assert facts["overX"] == 0, (where, facts)
        assert facts["short"] == (int(s["size"].split("x")[1]) < 900), (where, facts)
        narrow_datasets = s["page"] == "datasets" and s["size"] in datasets_stacked
        want = "stack" if s["size"] in portrait or narrow_datasets else "side"
        assert facts["layout"] == want, (where, facts)
        if want == "side":
            assert facts["need"] <= facts["width"] + 0.5, (where, facts)
            assert len(set(facts["tops"])) <= 1 and facts["narrowest"] >= -0.5, (where, facts)
            assert facts["overY"] <= 0, (where, facts)
        elif len(facts["tops"]) > 1:
            assert facts["tops"] == sorted(facts["tops"]) and len(set(facts["tops"])) == len(facts["tops"]), (where, facts)
        if narrow_datasets:
            assert facts["need"] > facts["width"], s                  # its own measure, no global gate
    wide = [s for s in ui_walk if s["size"] == "1850x960" and not s.get("open")]
    # The layout's facts, not a page's own (`drawer`, `finetune`: their test's).
    layout = lambda facts: {k: v for k, v in facts.items() if k not in ("drawer", "finetune")}  # noqa: E731
    for page in {s["page"] for s in wide}:
        first, *_, last = [layout(s["extra"]) for s in wide if s["page"] == page]
        assert first == last, (page, first, last)
    assert [(s["page"], s["extra"]["layout"]) for s in ui_walk[-3:]] == [
        ("training", "side"), ("datasets", "stack"), ("finetune", "side")]
    # R-38: at the owner's window the Training run bar is one row (it was two, 119 px).
    (training, *_) = [s for s in wide if s["page"] == "training"]
    assert training["extra"]["runbar"]["lines"] == 1 and training["extra"]["runbar"]["h"] < 90, training


def test_every_page_at_every_size_passes_the_ui_measurement_check(ui_walk):
    """D08 (INT-AUDIT): the shared UI measurement check (`browser.check`,
    `browser.focusWalk`, `browser.ui_problems`) on every page at every
    representative window (`browser.UI_SIZES`: 1850 x 960, 1280 x 900,
    1920 x 700, a 1366 x 620 short laptop, 1440 x 1600, 960 x 1080, a
    390 x 844 phone; and 1536 x 760 and 1280 x 600), and with the Diagnosis
    load drawer open at 1280 x 900, 1366 x 620 and 390 x 844 (its focus walk
    inside the drawer), the service on the demo workspace for an operator
    (`ui_walk`):

    - no horizontal page scroll; one scroll region per area (none inside
      another, the page included; never two in one panel);
    - every control drawn, focused as a keyboard focuses it, in view with
      nothing over it (another control's own overlay, a help bubble or an
      open list it leaves behind, aside), its focus cue at 3:1;
    - no text under 14 px, only the type scale's sizes and weights, nothing
      upper case; every enabled field's border at 3:1; one field-label style
      per page, and no label cut by its own field;
    - the Prepare / Evaluate captions above their tabs' line, outside the
      tabs' hit area, taking no pointer, at 14 px or more;
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
        assert len(s["check"]["captions"]) == (2 if research else 0), (s["page"], s["size"])
    options = {s["page"]: s["options"] for s in ui_walk if "options" in s}
    assert all(options[page] for page in ("training", "finetune", "test", "models", "settings")), options
    drawer = [s for s in checked if s.get("open") == "#shep-dx-change"]
    assert {s["size"] for s in drawer} == {"1280x900", "1366x620", "390x844"}
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


def test_the_field_rows_share_one_class_and_two_tokens():
    """D03: the four rows of a select and an action (Fine-tune's base picker,
    Training's Continue drawer, Test's Run, the Diagnosis load drawer's head)
    carry the one field-row class; their own alignment patches are gone; a
    dense form's blocks (the shared `shep-form-dense`: Training's parameters
    and its drawer, Fine-tune's form) set the row's padding token with their
    padding (one value, never two that drift)."""
    from src.webui.components import (
        diagnosis_panel,
        finetune_page,
        model_list,
        test_page,
        training_console,
    )
    from src.webui.style import SHELL_CSS

    assert "--shep-field-box-h: 41px; --shep-field-pad-b: 10px;" in SHELL_CSS
    button = _rule(SHELL_CSS, ".shep-field-row > button")
    assert "height: var(--shep-field-box-h) !important" in button
    assert "margin: 0 0 var(--shep-field-pad-b) !important" in button
    assert '"shep-ft-parent-row", "shep-field-row"' in Path(finetune_page.__file__).read_text()
    assert '"shep-test-model-row", "shep-field-row"' in Path(test_page.__file__).read_text()
    assert '"shep-train-parent-row", "shep-field-row"' in Path(training_console.__file__).read_text()
    assert '["shep-field-row"] if chosen.layout != "table"' in Path(model_list.__file__).read_text()
    assert '"shep-seg", "shep-field-aside"' in Path(diagnosis_panel.__file__).read_text()
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
    <div class="shep-scroll-cue" id="cue" style="height: 200px; overflow: auto;"><div style="height: 900px"></div></div>
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
    nothing at rest). A region with the shared ``shep-scroll-cue`` fades its
    last lines while more is below and draws them whole at its end (the head
    script keeps ``data-shep-more``); Test's Result body has the class.
    B72-C3: Compare, 12,342 px tall, had its sideways scrollbar at its foot,
    out of reach mid-table; it is now a box no taller than the region it is
    read in, so its scrollbar is in view whenever the box is. Both measured
    in a browser on a page of their own (Node and Playwright, skipped without)."""
    from src.webui.components import models_page, test_page
    from src.webui.style import SHELL_CSS
    from tests.fixtures.browser import node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    assert '"shep-test-bd", "shep-test-bd-flush", "shep-scroll-cue"' in Path(test_page.__file__).read_text()
    css_path, js_path = tmp_path / "page.css", tmp_path / "head.js"
    css_path.write_text(SHELL_CSS + models_page.PAGE_CSS, encoding="utf-8")
    js_path.write_text(shell._SHELL_JS.replace("__TITLES__", "{}").replace("__RESEARCH_LABEL__", '""')
                       .replace("__LAYOUT__", "data-shep-layout").replace("__SHORT__", "data-shep-short")
                       .replace("__SHORT_BELOW__", "900"), encoding="utf-8")
    seen = run_node(_CUE_JS, tmp_path, str(css_path), str(js_path))
    assert seen["top"] == {"more": True, "mask": True}, seen
    assert seen["end"] == {"more": False, "mask": False}, seen
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
