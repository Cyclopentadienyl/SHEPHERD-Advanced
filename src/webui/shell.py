"""
The dashboard's shell: navigation, header chips, shared timers, the page contract.
==================================================================================
M5-2, work package UI-A (`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's
boundaries": navigation in two levels; Settings at the header's right;
Diagnosis is the first screen).

Navigation
----------
Two clickable levels, each a ``gr.Tabs`` (so ``gr.Tabs(selected=…)`` switches
them from Python, and Gradio's tablist semantics and keyboard stay):

- **Level 1** (``#shep-nav``): "Diagnosis (clinical)" (the first screen),
  "Research (pre-clinical)", and "⚙ Settings" (``alignment="right"``). Drawn as a
  segmented control: the selected stage is filled in the accent colour and bold.
  Settings, selected, has a light fill, a 3 px underline and bold.
- **Level 2** (``#shep-research-nav``, inside Research): Datasets · Training ·
  Test · Models under the group labels PREPARE and EVALUATE with a divider. The
  labels are CSS generated content with empty alternative text and
  ``pointer-events: none``: not clickable and not part of a tab's name; the
  tablist's ``aria-label`` gives the grouping to a screen reader. The selected
  page has a 3 px underline and is bold, at normal size. The default inner page is
  Training.
- In-page tabs use a third style: give a ``gr.Radio`` the class ``shep-seg``
  (a grey track with a raised white segment), never a ``gr.Tabs``.
- Both levels use ``overflow_behavior="wrap"``: no tab ever goes into an overflow
  menu. Row 1 holds the brand, the D8 chip, the GPU chip and Settings: the brand
  and the chips are drawn over the level-1 tab row, and when they do not fit on
  one row (the head script measures them) the chips and Settings move onto a
  second line, so nothing overlaps or is cut.
- **The browser title follows the location**: "Diagnosis · Clinical —
  SHEPHERD-Advanced", "Training · Research — SHEPHERD-Advanced", "Settings —
  SHEPHERD-Advanced" (`TITLES`).
- **The layout rule** (the mockup's README): wide (at least 1400 × 760 CSS px)
  the page never scrolls, the selected page fills the space under the header and
  a region too long scrolls inside its own panel (a page that is taller than the
  space scrolls inside it rather than the page); narrow, everything flows and the
  page scrolls, with no region scrolling inside.

The page contract (UI-B, UI-C and UI-D edit only their own modules)
------------------------------------------------------------------
Each page is built by one function, called inside its own ``gr.Tab`` with the
`PageContext` as its one argument:

=========================  ============================================  =======
Page                       Builder                                       Owner
=========================  ============================================  =======
Diagnosis (clinical)       ``diagnosis_panel.create_diagnosis_tab``      UI-B
Research › Datasets        ``datasets_page.create_datasets_page``        UI-A
Research › Training        ``training_console.create_training_tab``      UI-C
Research › Test            ``test_page.create_test_page``                UI-D
Research › Models          ``models_page.create_models_page``            UI-D
⚙ Settings                 ``runtime_settings.create_runtime_settings_tab``  UI-A
=========================  ============================================  =======

(``test_page`` and ``models_page`` are created by UI-D; until then the shell
shows their one-line temporary body, `TEMPORARY_BODY`.) **Registering a page**
is one line each in `src/webui/app.py`: its builder in ``PAGES`` (by the page
ids here) and, when it has one, its style sheet in ``PAGE_CSS``. The context
gives a page everything it shares, and a page reaches nothing else of another
page:

- ``ctx.blocks``: the ``gr.Blocks`` (``ctx.blocks.load(...)`` for a page-load
  read);
- ``ctx.help``: the shared ⓘ registry (`HelpRegistry`): ``ctx.help.field(elem_id,
  title, text)`` for a component's help, ``ctx.help.topic(key, title, text)`` for
  a ⓘ inside rendered HTML (`help_button`). No ordinary help stays in ``info=``;
  only a state-bearing text does (a refusal, the HGT/AMP one);
- ``ctx.on_page_open(page, kwargs)``: run an event (``fn``, ``inputs``,
  ``outputs``, …) **each time the page is shown**: Diagnosis and Settings on
  their tab's ``select``; a research page when its own tab is selected and when
  Research is selected while it is the research page shown (only that page:
  returning to Research never refreshes the other three). The research pages
  are refreshed through ``ctx.page_tokens[page]`` (a `gr.State` the shell
  bumps from queued events, since Gradio 6.30 dispatches a State's ``.change``
  only from a queued event's answer; ``on_page_open`` listens to its
  ``.change``). Diagnosis is shown at page load with no ``select``: read it on
  ``ctx.blocks.load`` too;
- ``ctx.timer_fast`` (1.5 s) and ``ctx.timer_slow`` (5 s): the shared
  ``gr.Timer``\\ s, created outside ``gr.Tabs`` (a timer inside an unopened tab
  never ticks in Gradio 6.30), so **they tick whichever page is shown**: a
  listener takes ``ctx.location`` (a `gr.State`, the id of the page shown:
  ``clinical``, ``settings`` or a research page id) as an input and returns
  ``gr.update()``\\ s, doing no work, while its page is not shown. Read-only polls
  use ``queue=False`` (``/ui/gradio_api/run/`` is kept out of the access log);
- ``ctx.nav`` and ``ctx.research_nav``: the two ``gr.Tabs``; a cross-link returns
  ``ctx.go(page)`` for ``outputs=ctx.nav_outputs`` (both levels at once, and the
  research page shown, so only the target page refreshes; ``ctx.no_nav()``
  leaves them). Use the two together: ``go`` returns one value per output;
- ``ctx.chips`` and ``ctx.chip_refresh``: the header's chips; chain
  ``.then(**ctx.chip_refresh)`` after every gated act (D8's answer and the GPU
  jobs are read again for that request);
- ``ctx.create_model_list(preset, head=…, foot=…)``: the one shared model list
  and its Load (`src/webui/components/model_list.py`; ``"clinical"`` for
  Diagnosis' drawer: rows, "In use now" pinned, the latest 4 runs;
  ``"research"`` for Models: a table, every run). A page shapes it through the
  returned handle's ``view`` (`gr.State`: an architecture, groups such as
  "Ordered by …", one extra column such as the regime, a note beside Load) and
  ``ordering`` (a Load from an ordered list writes a usage history entry);
  ``head``/``foot`` place the page's own components in the list's header and
  footer rows. Each preset has its own element ids (``shep-models-<preset>``),
  so both can sit in one page;
- ``ctx.served_model_changed``: a ``gr.State`` the Load sets (a new token)
  whenever what is served changed (any part of the served identity or the
  served workspace); Diagnosis listens to its ``.change`` to re-read its card;
- ``ctx.training_workspace``: this session's Training run-bar workspace (a
  ``gr.State``, a listed workspace name or None). Datasets' "Use for training ›"
  sets it and opens Training; the run bar reads and writes it, and sets it to
  None when the run bar's workspace is changed by hand to another. (Until UI-C's
  run bar exists, the Training form's listeners do that with the Data
  Directory, `training_console._from_datasets` and `_keep_or_clear_workspace`;
  the run bar replaces both, with their test file
  ``tests/unit/test_training_console_datasets_bridge.py``.)

**Layout classes.** A page's top ``gr.Row`` takes ``elem_classes=["shep-page"]``
and each of its columns ``["shep-panel"]``: wide (at least 1400 × 760 CSS px)
the row fills the space under the header and each panel scrolls inside its own
region; narrow, the panels stack and only the page scrolls. ``shep-panel-hd``,
``shep-panel-bd`` and ``shep-panel-ft`` are a panel's header, body and footer
(`src/webui/style.py`). In-page tabs: a ``gr.Radio`` with the class
``shep-seg``.

**A page's CSS** is its module's ``PAGE_CSS`` string, its selectors scoped to
its own element ids (``#shep-page-<page>``, ``#shep-tab-<tab>``, or ids and
classes the page sets), listed in ``app.PAGE_CSS``. It goes into the head's
``<style>`` with the shell's (never the mount's ``css=``, which Gradio 6.30
rewrites inside ``@media``); the test that no text is under 14 px reads the
whole head. Shared pieces (panels, tags, the model list) stay in
`src/webui/style.py`.

Shared renderers (`src/webui/components/renderers.py`: the dataset identity,
the workspace select's options ``workspace_choices``, the model row and table
row, Model Facts), the workspace reader (`workspaces.py`, ``read_workspaces``:
the server's list with each manifest's state and no absolute path, for every
page that offers a workspace), chips (`chips.py`) and the status reader
(`served_status.py`) are plain imports.
**Per browser session:** a page keeps its state in ``gr.State``, never in module
globals. **The page is always drawn light**: Gradio's dark mode (from the
system's setting or ``?__theme=dark``) is switched off by the head script, since
the dashboard's colours have no dark values; Gradio's footer (with its own
"Settings" and theme switch) is not shown.

Module: src/webui/shell.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import gradio as gr

from src.webui.components.chips import CHIPS_PLACEHOLDER, read_chips
from src.webui.components.help import HelpRegistry, as_json_for_script
from src.webui.components.renderers import RENDERER_HELP

BRAND = "SHEPHERD-Advanced"

#: Level 1 (outer) tab ids and labels, in order.
CLINICAL, RESEARCH, SETTINGS = "clinical", "research", "settings"
#: Level 2 (Research) page ids.
DATASETS, TRAINING, TEST, MODELS = "datasets", "training", "test", "models"
RESEARCH_PAGES = (DATASETS, TRAINING, TEST, MODELS)
DEFAULT_RESEARCH_PAGE = TRAINING

LABELS = {
    CLINICAL: "Diagnosis (clinical)",
    RESEARCH: "Research (pre-clinical)",
    SETTINGS: "⚙ Settings",
    DATASETS: "Datasets",
    TRAINING: "Training",
    TEST: "Test",
    MODELS: "Models",
}

#: The browser title for each location.
TITLES = {
    CLINICAL: f"Diagnosis · Clinical — {BRAND}",
    SETTINGS: f"Settings — {BRAND}",
    **{page: f"{LABELS[page]} · Research — {BRAND}" for page in RESEARCH_PAGES},
}

#: The research tablist's name for a screen reader (the visual group labels are
#: generated content with empty alternative text).
RESEARCH_TABLIST_LABEL = ("Research pages: Prepare (Datasets, Training), "
                          "Evaluate (Test, Models)")

#: The one-line body Test and Models hold until UI-D builds them.
TEMPORARY_BODY = "being built in M5"

#: The shared timers' periods, in seconds.
FAST_SECONDS = 1.5
SLOW_SECONDS = 5.0


def plain_html(value: Any = "", *, js_on_load: Optional[str] = None, **kwargs: Any) -> Any:
    """A ``gr.HTML`` for HTML the dashboard renders itself: without Gradio's
    default prose CSS (which restyles tables, buttons and paragraphs inside
    it) and without its default click script; pass ``js_on_load`` for one of
    your own (``trigger('click', {...})``)."""
    classes = kwargs.pop("elem_classes", None) or []
    classes = [classes] if isinstance(classes, str) else list(classes)
    return gr.HTML(value, apply_default_css=False, js_on_load=js_on_load,
                   elem_classes=["shep-html", *classes], **kwargs)


def tab_elem_id(tab: str) -> str:
    return f"shep-tab-{tab}"


def page_elem_id(page: str) -> str:
    return f"shep-page-{page}"


@dataclass
class PageContext:
    """What every page builder is given (the page contract, module docstring)."""

    blocks: Any
    help: HelpRegistry
    nav: Any
    research_nav: Any
    timer_fast: Any
    timer_slow: Any
    chips: Any
    served_model_changed: Any
    training_workspace: Any
    #: The id of the page shown (`gr.State`): clinical, settings or a research page.
    location: Any = None
    #: The research page shown under Research (`gr.State`), Training at first.
    research_page: Any = None
    #: One `gr.State` per research page, bumped each time that page is shown.
    page_tokens: Dict[str, Any] = field(default_factory=dict)
    tabs: Dict[str, Any] = field(default_factory=dict)

    @property
    def nav_outputs(self) -> list:
        """The outputs a cross-link returns `go` for: both levels and the
        research page shown (so Research's own refresh, when the cross-link
        switches to it, is the target page's and no other's)."""
        return [self.nav, self.research_nav, self.research_page]

    @staticmethod
    def go(page: str) -> Tuple[Any, Any, Any]:
        """Updates for `nav_outputs` that open `page`: a level-1 id (clinical,
        research, settings) or a research page id."""
        if page in RESEARCH_PAGES:
            return gr.Tabs(selected=RESEARCH), gr.Tabs(selected=page), page
        if page in (CLINICAL, RESEARCH, SETTINGS):
            return gr.Tabs(selected=page), gr.update(), gr.update()
        raise ValueError(f"no page {page!r}")

    @staticmethod
    def no_nav() -> Tuple[Any, Any, Any]:
        return gr.update(), gr.update(), gr.update()

    @property
    def chip_refresh(self) -> Dict[str, Any]:
        """Keyword arguments for ``.then(**ctx.chip_refresh)``: both chips read
        again for the request that made the gated act."""
        return {"fn": read_chips, "inputs": None, "outputs": [self.chips], "queue": False,
                "show_progress": "hidden"}

    def on_page_open(self, page: str, kwargs: Dict[str, Any]) -> None:
        """Run an event (``fn``, ``inputs``, ``outputs``, …) each time `page` is
        shown: Diagnosis' or Settings' tab ``select``; for a research page, the
        change of its token, which the shell bumps when its tab is selected and
        when Research is selected while it is the page shown (returning to
        Research shows that page as it is now, and only that page)."""
        if page in RESEARCH_PAGES:
            self.page_tokens[page].change(**kwargs)
        elif page in self.tabs:
            self.tabs[page].select(**kwargs)
        else:
            raise ValueError(f"no page {page!r}")

    def create_model_list(self, preset: str = "clinical", **kwargs: Any) -> Any:
        """The one shared model list and its Load, built here (`model_list`)."""
        from src.webui.components.model_list import create_model_list

        return create_model_list(self, preset, **kwargs)


def standalone_context(blocks: Any, help_registry: Optional[HelpRegistry] = None) -> PageContext:
    """A context for building one page alone (a test, a prototype): real
    components for everything a page may use, created in the current
    ``gr.Blocks``; no navigation is shown around the page."""
    with gr.Column(visible=False):
        chips = plain_html(CHIPS_PLACEHOLDER)
        with gr.Tabs() as nav:
            with gr.Tab("Research", id=RESEARCH) as research_tab:
                with gr.Tabs() as research_nav:
                    pass
    return PageContext(
        blocks=blocks, help=help_registry or HelpRegistry(), nav=nav, research_nav=research_nav,
        timer_fast=gr.Timer(FAST_SECONDS, active=False), timer_slow=gr.Timer(SLOW_SECONDS, active=False),
        chips=chips, served_model_changed=gr.State(None), training_workspace=gr.State(None),
        location=gr.State(CLINICAL), research_page=gr.State(DEFAULT_RESEARCH_PAGE),
        page_tokens={page: gr.State(0) for page in RESEARCH_PAGES},
        tabs={RESEARCH: research_tab})


def _shown(page: str) -> Callable[[], str]:
    def shown() -> str:
        return page
    shown.__name__ = f"_shown_{page}"
    return shown


def _open_research(research_page: Any, *tokens: Any) -> Tuple[Any, ...]:
    """Research selected: the research page shown is the one to refresh (its
    token bumped); the other pages' tokens stay, so they do no work."""
    page = research_page if research_page in RESEARCH_PAGES else DEFAULT_RESEARCH_PAGE
    bumped = tuple((t or 0) + 1 if p == page else gr.update()
                   for p, t in zip(RESEARCH_PAGES, tokens, strict=True))
    return (page, *bumped)


def _open_page(page: str) -> Callable[[Any], Tuple[Any, Any, Any]]:
    """A research page's tab selected: it is the page shown and is refreshed."""
    def open_page(token: Any) -> Tuple[Any, Any, Any]:
        return page, page, (token or 0) + 1
    open_page.__name__ = f"_open_{page}"
    return open_page


def _brand_html() -> str:
    return f'<div class="shep-brand">{BRAND}</div>'


def build_shell(blocks: Any, pages: Dict[str, Callable[[PageContext], None]],
                *, help_registry: Optional[HelpRegistry] = None) -> PageContext:
    """Lay out the header, the two navigation levels and every page.

    `pages` maps a page id (clinical, datasets, training, test, models,
    settings) to its builder; a missing Test or Models builder shows
    `TEMPORARY_BODY`. Returns the context the pages were given."""
    registry = help_registry or HelpRegistry()
    registry.topics(RENDERER_HELP)
    served_model_changed = gr.State(None)
    training_workspace = gr.State(None)
    location = gr.State(CLINICAL)
    research_page = gr.State(DEFAULT_RESEARCH_PAGE)
    page_tokens = {page: gr.State(0) for page in RESEARCH_PAGES}
    # Timers outside gr.Tabs: one in an unopened tab never ticks (Gradio 6.30).
    timer_fast = gr.Timer(FAST_SECONDS)
    timer_slow = gr.Timer(SLOW_SECONDS)
    # Created first and placed later, so every page, Diagnosis included, can
    # cross-link to a research page (`ctx.go`).
    research_nav = gr.Tabs(selected=DEFAULT_RESEARCH_PAGE, overflow_behavior="wrap",
                           elem_id="shep-research-nav", render=False)
    with gr.Column(elem_id="shep-shell", elem_classes=["shep-shell"]):
        plain_html(_brand_html(), elem_id="shep-brand")
        chips = plain_html(CHIPS_PLACEHOLDER, elem_id="shep-chips")
        with gr.Tabs(selected=CLINICAL, overflow_behavior="wrap", elem_id="shep-nav") as nav:
            tabs: Dict[str, Any] = {}
            ctx = PageContext(blocks=blocks, help=registry, nav=nav, research_nav=research_nav,
                              timer_fast=timer_fast, timer_slow=timer_slow, chips=chips,
                              served_model_changed=served_model_changed,
                              training_workspace=training_workspace, location=location,
                              research_page=research_page, page_tokens=page_tokens, tabs=tabs)
            with gr.Tab(LABELS[CLINICAL], id=CLINICAL, elem_id=tab_elem_id(CLINICAL),
                        elem_classes=["shep-tab"]) as clinical_tab:
                tabs[CLINICAL] = clinical_tab
                pages[CLINICAL](ctx)
            with gr.Tab(LABELS[RESEARCH], id=RESEARCH, elem_id=tab_elem_id(RESEARCH),
                        elem_classes=["shep-tab"]) as research_tab:
                tabs[RESEARCH] = research_tab
                research_nav.render()
                with research_nav:
                    for page in RESEARCH_PAGES:
                        with gr.Tab(LABELS[page], id=page, elem_id=page_elem_id(page),
                                    elem_classes=["shep-page-tab"]) as page_tab:
                            tabs[page] = page_tab
                            builder = pages.get(page)
                            if builder is None:
                                gr.Markdown(f"{LABELS[page]}: {TEMPORARY_BODY}.",
                                            elem_classes=["shep-temporary"])
                            else:
                                builder(ctx)
            with gr.Tab(LABELS[SETTINGS], id=SETTINGS, alignment="right",
                        elem_id=tab_elem_id(SETTINGS), elem_classes=["shep-tab"]) as settings_tab:
                tabs[SETTINGS] = settings_tab
                pages[SETTINGS](ctx)
    # Where this session is: the page shown, and the research page to refresh
    # when Research is selected (only that one; Gradio also selects a research
    # tab when it first mounts and when a cross-link switches to it). The
    # events that bump a token go through the queue: Gradio 6.30 dispatches a
    # State's ``.change`` only from a queued event's answer (an answer to a
    # ``queue=False`` POST carries no changed states).
    for tab in (CLINICAL, SETTINGS):
        tabs[tab].select(_shown(tab), None, [location], queue=False, show_progress="hidden")
    tokens = [page_tokens[page] for page in RESEARCH_PAGES]
    tabs[RESEARCH].select(_open_research, [research_page, *tokens], [location, *tokens],
                          show_progress="hidden")
    for page in RESEARCH_PAGES:
        tabs[page].select(_open_page(page), [page_tokens[page]],
                          [location, research_page, page_tokens[page]], show_progress="hidden")
    # The chips: on load, on the slow timer and after every gated act, each
    # read for the request that asked, with no queue (plain POST, no SSE).
    blocks.load(**ctx.chip_refresh)
    timer_slow.tick(**ctx.chip_refresh)
    return ctx


def help_texts(registry: HelpRegistry) -> Any:
    """The hidden container of every registered help text: created after every
    page has registered its help (it may sit anywhere in the page)."""
    return plain_html(registry.texts_html(), elem_id="shep-help-texts",
                   elem_classes=["shep-help-texts-block"])


# ---------------------------------------------------------------------------
# The browser side: the title and the header row
# ---------------------------------------------------------------------------
_SHELL_JS = r"""
(function () {
  if (window.ShepShell) { return; }
  // Always light: the dashboard's colours have no dark values, so Gradio's
  // dark mode (the system's setting, or ?__theme=dark) would leave light text
  // on the light page. Its class is kept off the body, and the address asks
  // for light, so a reload (Gradio reloads after a lost connection) starts so.
  function light() {
    if (document.body && document.body.classList.contains('dark')) { document.body.classList.remove('dark'); }
  }
  try {
    var here = new URL(window.location.href);
    if (here.searchParams.get('__theme') !== 'light') {
      here.searchParams.set('__theme', 'light');
      window.history.replaceState(window.history.state, '', here.toString());
    }
  } catch (err) { /* the address stays; the class is still kept off */ }
  light();
  if (document.body) {
    new MutationObserver(light).observe(document.body, {attributes: true, attributeFilter: ['class']});
  }
  var TITLES = __TITLES__, RESEARCH = 'research', DEFAULT_PAGE = '__DEFAULT__';
  var TABLIST_LABELS = {'shep-nav': 'Stage', 'shep-research-nav': __RESEARCH_LABEL__};
  function selected(navId) {
    var b = document.querySelector('#' + navId + ' > .tab-wrapper [role="tab"][aria-selected="true"]');
    return b ? b.getAttribute('data-tab-id') : null;
  }
  function title() {
    var outer = selected('shep-nav') || 'clinical';
    var key = outer === RESEARCH ? (selected('shep-research-nav') || DEFAULT_PAGE) : outer;
    var want = TITLES[key];
    if (want && document.title !== want) { document.title = want; }
  }
  function labels() {
    for (var id in TABLIST_LABELS) {
      var list = document.querySelector('#' + id + ' > .tab-wrapper [role="tablist"]');
      if (list && list.getAttribute('aria-label') !== TABLIST_LABELS[id]) {
        list.setAttribute('aria-label', TABLIST_LABELS[id]);
      }
    }
  }
  // Row 1 holds the brand, the stage switch, the chips and Settings. When they
  // do not fit on one row, the chips and Settings move onto a second line.
  function layout() {
    var shell = document.getElementById('shep-shell');
    var row = document.querySelector('#shep-nav > .tab-wrapper > .tab-container');
    var chips = document.querySelector('#shep-chips .shep-chips');
    var brand = document.querySelector('#shep-brand .shep-brand');
    var right = row ? row.querySelector('.right-tab-group') : null;
    if (!shell || !row || !chips || !right) { return; }
    var left = 0;
    var buttons = row.querySelectorAll(':scope > button');
    for (var i = 0; i < buttons.length; i++) { left += buttons[i].offsetWidth; }
    // The Settings tab's own width: on a second line its group spans the row.
    var settingsTab = right.querySelector('[role="tab"]');
    var settings = settingsTab ? settingsTab.offsetWidth : right.offsetWidth;
    var need = 16 + (brand ? brand.offsetWidth : 180) + 24 + left + 24 + chips.scrollWidth + 16 + settings + 16;
    shell.style.setProperty('--shep-settings-w', settings + 'px');
    shell.classList.toggle('shep-nav-wrapped', need > shell.clientWidth);
    shell.style.setProperty('--shep-chips-h', chips.offsetHeight + 'px');
  }
  var queued = false;
  function update() {
    if (queued) { return; }
    queued = true;
    requestAnimationFrame(function () { queued = false; title(); labels(); layout(); });
  }
  new MutationObserver(update).observe(document.documentElement,
    {childList: true, subtree: true, attributes: true, attributeFilter: ['aria-selected']});
  window.addEventListener('resize', update);
  if (window.ResizeObserver) {
    var ro = new ResizeObserver(update);
    var watch = setInterval(function () {
      var chips = document.querySelector('#shep-chips');
      var shell = document.getElementById('shep-shell');
      if (chips && shell) { ro.observe(chips); ro.observe(shell); clearInterval(watch); }
    }, 200);
  }
  window.ShepShell = {update: update, title: title};
})();
"""

#: The shell's script, for the mount's ``head=`` (with the help script).
SHELL_HEAD = ("<script>" + _SHELL_JS
              .replace("__TITLES__", as_json_for_script(TITLES))
              .replace("__DEFAULT__", DEFAULT_RESEARCH_PAGE)
              .replace("__RESEARCH_LABEL__", as_json_for_script(RESEARCH_TABLIST_LABEL))
              + "</script>")


__all__ = [
    "BRAND",
    "CLINICAL",
    "DATASETS",
    "DEFAULT_RESEARCH_PAGE",
    "LABELS",
    "MODELS",
    "PageContext",
    "RESEARCH",
    "RESEARCH_PAGES",
    "RESEARCH_TABLIST_LABEL",
    "SETTINGS",
    "SHELL_HEAD",
    "TEMPORARY_BODY",
    "TEST",
    "TITLES",
    "TRAINING",
    "build_shell",
    "help_texts",
    "page_elem_id",
    "standalone_context",
    "tab_elem_id",
]
