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
  is the page shown (`on_page_open`, the page tokens); ``ctx.location`` says
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
        ("Test", "test", "shep-page-test"), ("Models", "models", "shep-page-models")]


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


def test_test_and_models_hold_the_temporary_body_until_built():
    from src.webui.app import PAGES

    pages = {k: v for k, v in PAGES.items() if k not in (shell.TEST, shell.MODELS)}
    demo, _ctx = _built(pages)
    bodies = [b.value for b in demo.blocks.values()
              if isinstance(b, gr.Markdown) and "being built in M5" in str(b.value)]
    assert bodies == ["Test: being built in M5.", "Models: being built in M5."]


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
                                              shell.TEST, shell.MODELS, shell.SETTINGS)}
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
    assert len(links) == 6


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
    other three (Models' refresh opens every checkpoint)."""
    demo, ctx = built
    tokens = [ctx.page_tokens[p] for p in shell.RESEARCH_PAGES]
    (research,) = [fn for fn in demo.fns.values()
                   if (ctx.tabs[shell.RESEARCH]._id, "select") in fn.targets]
    assert research.inputs == [ctx.research_page, *tokens]
    # queued: Gradio 6.30 dispatches a State's .change only from a queued answer
    assert research.outputs == [ctx.location, *tokens] and research.queue is not False
    location, *bumped = research.fn("datasets", 0, 4, 0, 0)
    assert location == "datasets" and bumped == [1, gr.update(), gr.update(), gr.update()]
    location, *bumped = research.fn(None, 0, 4, 0, 0)        # nothing chosen yet: Training
    assert location == "training" and bumped == [gr.update(), 5, gr.update(), gr.update()]
    for page in shell.RESEARCH_PAGES:
        (own,) = [fn for fn in demo.fns.values() if (ctx.tabs[page]._id, "select") in fn.targets]
        assert own.outputs == [ctx.location, ctx.research_page, ctx.page_tokens[page]]
        assert own.fn(2) == (page, page, 3) and own.queue is not False
    # Datasets' refresh hangs on its token, not on any tab's select.
    from src.webui.components import datasets_page

    refresh = [fn for fn in demo.fns.values() if fn.name == "_refresh"
               and (ctx.page_tokens[shell.DATASETS]._id, "change") in fn.targets]
    assert len(refresh) == 1
    assert not [fn for fn in demo.fns.values() if fn.name == "_refresh"
                and any(t[1] == "select" for t in fn.targets)]
    assert datasets_page.create_datasets_page in demo_builders()


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

    pages = {shell.CLINICAL: clinical, shell.SETTINGS: lambda ctx: None}
    demo, ctx = _built(pages)
    assert [fn for fn in demo.fns.values() if (ctx.tabs[shell.CLINICAL]._id, "select") in fn.targets
            and fn.outputs != [ctx.location]]
    with pytest.raises(ValueError):
        ctx.on_page_open("nowhere", {"fn": lambda: None})


def test_every_gated_act_is_followed_by_the_chips_read(built):
    """Training Start, Continue, Stop, Check checkpoint, Save Path and the run
    list's Refresh, Diagnosis' Change model… (it lists checkpoints), Settings
    Apply and Restart: each act's chain reads the chips again."""
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
             "_on_apply", "_on_restart", "_on_check_continue", "_save_training_config"}
    acts = [fn for fn in by_id.values() if fn.name in names
            or getattr(getattr(fn.fn, "__wrapped__", None), "__name__", None) in names]
    found = {fn.name if fn.name in names else fn.fn.__wrapped__.__name__ for fn in acts}
    assert found == names
    for fn in acts:
        assert chain_reads_chips(fn), fn.name
    # the run list's Refresh (its conv_type listener refreshes on the slow timer)
    from src.webui.components import training_console as tc

    refreshes = [fn for fn in by_id.values() if fn.fn is tc._refresh_checkpoints]
    assert any(chain_reads_chips(fn) for fn in refreshes)


def test_the_title_follows_the_location():
    assert shell.TITLES == {
        "clinical": "Diagnosis · Clinical — SHEPHERD-Advanced",
        "settings": "Settings — SHEPHERD-Advanced",
        "datasets": "Datasets · Research — SHEPHERD-Advanced",
        "training": "Training · Research — SHEPHERD-Advanced",
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
