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
shows their one-line temporary body, `TEMPORARY_BODY`.) The context gives a page
everything it shares, and a page reaches nothing else of another page:

- ``ctx.blocks``: the ``gr.Blocks`` (``ctx.blocks.load(...)`` for a page-load
  read);
- ``ctx.help``: the shared ⓘ registry (`HelpRegistry`): ``ctx.help.field(elem_id,
  title, text)`` for a component's help, ``ctx.help.topic(key, title, text)`` for
  a ⓘ inside rendered HTML (`help_button`). No ordinary help stays in ``info=``;
  only a state-bearing text does (a refusal, the HGT/AMP one);
- ``ctx.timer_fast`` (1.5 s) and ``ctx.timer_slow`` (5 s): the shared
  ``gr.Timer``\\ s, created outside ``gr.Tabs`` (a timer inside an unopened tab
  never ticks in Gradio 6.30). Read-only polls use ``queue=False``; a timer's
  listener is the page's own. A page refreshes on ``ctx.on_page_open(page,
  kwargs)`` too (its tab's ``select``, and Research's for its pages), since a
  timer's first tick may come later;
- ``ctx.nav`` and ``ctx.research_nav``: the two ``gr.Tabs``; a cross-link returns
  ``ctx.go(page)`` for ``outputs=ctx.nav_outputs`` (both levels at once;
  ``ctx.no_nav()`` leaves them);
- ``ctx.chips`` and ``ctx.chip_refresh``: the header's chips; chain
  ``.then(**ctx.chip_refresh)`` after every gated act;
- ``ctx.create_model_list(preset)``: the one shared model list and its Load
  (`src/webui/components/model_list.py`; ``"clinical"`` for Diagnosis' drawer,
  ``"research"`` for Models);
- ``ctx.served_model_changed``: a ``gr.State`` the Load sets (a new token)
  whenever the served model changed; Diagnosis listens to its ``.change`` to
  re-read its card;
- ``ctx.training_workspace``: this session's Training run-bar workspace (a
  ``gr.State``, a listed workspace name or None). Datasets' "Use for training ›"
  sets it and opens Training; the run bar reads and writes it. (Until UI-C's
  run bar exists, the Training form's ``.change`` listener copies it into this
  session's Data Directory, `training_console._from_datasets`; the run bar
  replaces that listener.)

Shared renderers (`src/webui/components/renderers.py`), chips
(`chips.py`) and the status reader (`served_status.py`) are plain imports.
**Per browser session:** a page keeps its state in ``gr.State``, never in module
globals.

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
    tabs: Dict[str, Any] = field(default_factory=dict)

    @property
    def nav_outputs(self) -> list:
        """The outputs a cross-link returns `go` for."""
        return [self.nav, self.research_nav]

    @staticmethod
    def go(page: str) -> Tuple[Any, Any]:
        """Updates for `nav_outputs` that open `page`: a level-1 id (clinical,
        research, settings) or a research page id."""
        if page in RESEARCH_PAGES:
            return gr.Tabs(selected=RESEARCH), gr.Tabs(selected=page)
        if page in (CLINICAL, RESEARCH, SETTINGS):
            return gr.Tabs(selected=page), gr.update()
        raise ValueError(f"no page {page!r}")

    @staticmethod
    def no_nav() -> Tuple[Any, Any]:
        return gr.update(), gr.update()

    @property
    def chip_refresh(self) -> Dict[str, Any]:
        """Keyword arguments for ``.then(**ctx.chip_refresh)``: both chips read
        again for the request that made the gated act."""
        return {"fn": read_chips, "inputs": None, "outputs": [self.chips], "queue": False,
                "show_progress": "hidden"}

    def on_page_open(self, page: str, kwargs: Dict[str, Any]) -> None:
        """Run an event (``fn``, ``inputs``, ``outputs``, …) when `page` is
        opened: its tab's ``select``, and, for a research page, Research's
        ``select`` too (returning to Research shows the page as it is now)."""
        for tab in (page, RESEARCH if page in RESEARCH_PAGES else None):
            if tab is not None and tab in self.tabs:
                self.tabs[tab].select(**kwargs)

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
        tabs={RESEARCH: research_tab})


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
                              training_workspace=training_workspace, tabs=tabs)
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
