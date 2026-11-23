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

- **Level 1** (``#shep-nav``, the tablist named "Area"): "Diagnosis (clinical)"
  (the first screen), "Research (pre-clinical)", and "⚙ Settings"
  (``alignment="right"``). Drawn as a segmented control: the selected area is
  filled in the accent colour with white text in the strong weight, and so is
  Settings when it is selected (one look for level 1's selected item, which no
  other level has, R-20).
- **Level 2** (``#shep-research-nav``, inside Research): Datasets · Training ·
  Fine-tune · Test · Models, 32 px tabs in one row, which wraps only when
  the tabs alone do not fit. The selected page has a 3 px accent underline,
  the accent text and the strong weight; a tab under the pointer has a grey
  fill and darker text and never an underline, so only the selected tab ever
  reads as selected. The default inner page is Training.
- **The stage hint** (the stage-hint package of 2026-11-21, N1-N7): at the
  right of the research pages' row, after their tablist (never inside it),
  plain text says which stage of the research work the page shown belongs
  to, "Stage: Prepare" (Datasets, Training, Fine-tune) or "Stage: Evaluate"
  (Test, Models), from one mapping (`STAGES`, which also names the tablist,
  `RESEARCH_TABLIST_LABEL`). The head script writes it (``textContent``) on
  each of its passes from the tab selected, so a click, a key, a cross-link
  and a reload's restore all move it, and puts it back when Gradio draws the
  row again; it reads no job state and gates nothing. It is no control: no
  role, focus stop, pointer or hover. When the row has no room for both, the
  hint goes onto a line of its own, still at the right (the row wraps by
  itself: no measure, no window size). Diagnosis and Settings show none.
- Every tab of both levels shows keyboard focus as a 2 px ring in the focus
  token (``--shep-focus-ring``); hover, selected and focus cues are tokens in
  `src/webui/style.py`.
- In-page tabs use a third style: give a ``gr.Radio`` the class ``shep-seg``
  (a grey track with a raised white segment), never a ``gr.Tabs``. A form's
  own radio is a fourth look, a tinted option with its filled dot.
- Both levels use ``overflow_behavior="wrap"``: no tab ever goes into an overflow
  menu. Row 1 holds the brand, the D8 chip, the GPU chip and Settings: the brand
  and the chips are drawn over the level-1 tab row, and when they do not fit on
  one row (the head script measures them) the chips and Settings move onto a
  second line, so nothing overlaps or is cut.
- **The browser title follows the location**: "Diagnosis · Clinical —
  SHEPHERD-Advanced", "Training · Research — SHEPHERD-Advanced", "Settings —
  SHEPHERD-Advanced" (`TITLES`).
- **A reload comes back on the page shown** (UI-14): the head script keeps
  the page shown (``clinical``, ``settings`` or a research page id) for this
  browser tab in ``sessionStorage`` (`KEPT_PAGE`), and the page load asks the
  shell to open it again (`_restore`, the cross-links' own `PageContext.go`),
  so a reload, Gradio's reload after a lost connection included, lands where
  it was, not on Diagnosis. Only the page id is kept: never the address,
  case data, a path, a choice or anything that starts, loads or registers;
  the page opened reads its state as any opening does (a running job is read
  back, never started).
- **The layout rule, one decision** (D02, the WebUI ledger's UI-02): the head
  script's layout pass writes ``data-shep-layout`` on the root element, and
  every style sheet keys its side-by-side and stacked rules off it
  (`src/webui/style.py`'s ``SIDE`` and ``STACK``; no media query on the
  window's size anywhere). The browser's usable area, the window's inner
  size, decides: no wider than tall, ``stack``; wider, ``side`` when the page
  shown fits by its panels' own minimum widths (each panel's
  ``--shep-panel-min`` in its page's style sheet, read from the panels as
  drawn: the shown ``.shep-page`` row's shown children, under any wrapper
  that holds one alone, with the gaps between them), else ``stack``. Side by
  side the page never scrolls, the selected page fills the space under the
  header and a region too long scrolls inside its own panel; stacked,
  everything flows and the page scrolls, with no region scrolling inside
  (a table wider than its panel scrolls sideways in its own box). A short
  window (``data-shep-short``, under ``style.SHORT_BELOW`` px tall) only
  makes headers denser: height never stacks a page. A page with no
  ``.shep-page`` row (Settings) is decided by the window's shape alone.
- **A wide form only while it fits** (OPS-FIX-R2): the same pass measures
  each box of class ``style.FIT`` (the Fine-tune job card): its wide form
  (``style.FIT_WIDE``, a table) laid out at its own narrowest, its
  min-content width in the type drawn, for the measure alone and put back
  before anything is painted, against the box's width. When the form does
  not fit, the box gets ``style.NARROW_ATTR`` and its page draws the narrow
  form (each child's summary list). The measure never reads the state it
  sets, and it is taken again when the page changes, the window resizes or
  a font arrives: no width fitted to one host's text decides it.
- **A box whole only while its column has the room** (OPS-FIX-N1): the same
  pass measures, side by side, each box of class ``style.FOLD`` (the
  Diagnosis card) drawn whole, with what else its column holds and the
  column's region of class ``style.FOLD_ROOM`` (the Diagnosis fields) at
  its floor (its custom property ``style.FOLD_FLOOR``). When the column has
  not that room, the box gets ``style.FOLDED_ATTR`` and its parts of class
  ``style.FOLD_PART`` are not drawn (they stay in the box's own details), so
  the region keeps its floor; stacked, nothing folds. The floor is not a
  CSS minimum: where even the folded box leaves less, the region takes what
  is left and stays the column's one scroll region. Like the fit, the
  measure never reads the state it sets: no window size decides it, the
  box's own height (a notice, a long name, the type drawn), the column's
  and the floor do.

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
Research › Fine-tune       ``finetune_page.create_finetune_page``        K6
Research › Test            ``test_page.create_test_page``                UI-D
Research › Models          ``models_page.create_models_page``            UI-D
⚙ Settings                 ``runtime_settings.create_runtime_settings_tab``  UI-A
=========================  ============================================  =======

Every page id has its builder (one missing is a ``KeyError`` when the shell is
built, never a placeholder). **Registering a page**
is one line each in `src/webui/app.py`: its builder in ``PAGES`` (by the page
ids here) and, when it has one, its style sheet in ``PAGE_CSS``. The context
gives a page everything it shares, and a page reaches nothing else of another
page (one exception, recorded after this list):

- ``ctx.blocks``: the ``gr.Blocks`` (``ctx.blocks.load(...)`` for a page-load
  read);
- ``ctx.help``: the shared ⓘ registry (`HelpRegistry`): ``ctx.help.field(elem_id,
  title, text)`` for a component's help, ``ctx.help.topic(key, title, text)`` for
  a ⓘ inside rendered HTML (`help_button`). No ordinary help stays in ``info=``;
  only a state-bearing text does (Diagnosis' Reference cohort select: the
  chosen version's fingerprint and whether it was mapped against the served
  graph, or why it cannot be read);
- ``ctx.on_page_open(page, kwargs)``: run an event (``fn``, ``inputs``,
  ``outputs``, …) **each time the page is shown**: Diagnosis and Settings on
  their tab's ``select``; a research page when its own tab is selected and when
  Research is selected while it is the research page shown (only that page:
  returning to Research never refreshes the other four). The research pages
  are refreshed through ``ctx.page_tokens[page]`` (a `gr.State` the shell
  bumps from queued events, since Gradio 6.30 dispatches a State's ``.change``
  only from a queued event's answer; ``on_page_open`` listens to its
  ``.change``). **Once per opening**: the first opening of Research, and a
  cross-link that switches both levels, fire Research's select and the inner
  tab's (on mounting it, or on switching it); whichever runs first refreshes
  the page, and the other finds it already shown (`_open_research`).
  Diagnosis is shown at page load with no ``select``: read it on
  ``ctx.blocks.load`` too. It returns the event, so a page chains a step
  after it with ``.then`` (never a second listener on the same opening: the
  two would run in either order);
- ``ctx.timer_fast`` (1.5 s) and ``ctx.timer_slow`` (5 s): the shared
  ``gr.Timer``\\ s, created outside ``gr.Tabs`` (a timer inside an unopened tab
  never ticks in Gradio 6.30), so **they tick whichever page is shown**: a
  listener takes ``ctx.location`` (a `gr.State`, the id of the page shown:
  ``clinical``, ``settings`` or a research page id) as an input and returns
  ``gr.update()``\\ s, doing no work, while its page is not shown. Read-only polls
  use ``queue=False`` (``/ui/gradio_api/run/`` is kept out of the access log).
  **A listener costs one request per tick whatever it returns**, on every
  open page, whichever page is shown: returning ``gr.update()`` saves the
  work, not the request. So a poll goes on the slow timer unless it must be
  fast (Training's run view is the one fast listener; Test reads its job and
  Settings its restart lock on the slow timer, Test at once after Start,
  Cancel and Register too). A page that needs a fast poll only while something
  runs would need a timer of its own, outside ``gr.Tabs`` and made active only
  then; no page has a timer of its own;
- ``ctx.nav`` and ``ctx.research_nav``: the two ``gr.Tabs``; a cross-link returns
  ``ctx.go(page)`` for ``outputs=ctx.nav_outputs`` (both levels at once, and the
  research page shown, so only the target page refreshes; ``ctx.no_nav()``
  leaves them). Use the two together: ``go`` returns one value per output;
- ``ctx.chips`` and ``ctx.chip_refresh``: the header's chips; chain
  ``.then(**ctx.chip_refresh)`` after every gated act (D8's answer and the GPU
  jobs are read again for that request). A poll that reads the GPU job state
  (Training's run bar, Test's job) also outputs ``ctx.chips``, with
  ``chips.chips_on_change(seen, now, request)``: the chips are read again on
  the tick that sees the state change, so the header never lags the page, and
  no timer or request is added;
- ``ctx.create_model_list(preset, head=…, foot=…)``: the one shared model list
  and its Load (`src/webui/components/model_list.py`; ``"clinical"`` for
  Diagnosis' drawer: rows, "In use now" pinned, the latest 4 runs;
  ``"research"`` for Models: a table, every run). A page shapes it through the
  returned handle's ``view`` (`gr.State`: a task filter, groups such as a
  family, one extra column such as the lineage, a note beside Load);
  ``head``/``foot`` place the page's own components in the list's header and
  footer rows. Each preset has its own element ids (``shep-models-<preset>``),
  so both can sit in one page;
- ``ctx.served_model_changed``: a ``gr.State`` the Load sets (a new token)
  whenever what is served changed (any part of the served identity or the
  served workspace); Diagnosis listens to its ``.change`` to re-read its card;
- ``ctx.training_workspace``: this session's Training run-bar workspace (a
  ``gr.State``, a listed workspace name or None). Datasets' "Use for training ›"
  sets it and opens Training; Training's run bar reads it and writes the
  workspace it shows (the page open's, a hand choice, a running job's).
  **Every listener that writes it** (and Training's own record of its run
  bar's workspace) takes ``**ctx.training_workspace_writes``: one queued group
  of one (`TRAINING_SELECTION_CONCURRENCY`), first asked first run, so none
  writes back a workspace it started from over one chosen meanwhile;
- ``ctx.test_model``: a model a cross-link hands the Test page (a
  ``gr.State``, ``{"workspace", "key"}`` of a listed checkpoint, or None).
  Models' "Test this model ›" sets it and opens Test (``ctx.go``); the Test
  page's opening that follows chooses that workspace and checkpoint and
  clears it, so a later opening keeps the reader's own choice.

**The one exception to "nothing else of another page"**: Fine-tune is
Training's second form (README §1: a base, then its task children; §8: "base
training and Fine-tune separately"). `finetune_page` uses `training_console`'s
public training-form pieces (the workspace state, the recipe settings' fields,
the parent selects, the stop question, the job status marks) and no other
page's; `tests/unit/test_webui_datasets.py` pins both.

**Recognise a component by its ``elem_id``, never by identity.** Once an
answer updates a component's properties, Gradio 6.30 keeps a rebuilt copy in
the session under the same key (a new object, with a new ``_id``;
``Blocks.postprocess_data``), and ``gr.EventData.target`` is that copy: a
handler that must tell which component fired compares ``elem_id``, never the
object or its ``_id``.

**Layout classes.** A page's top ``gr.Row`` takes ``elem_classes=["shep-page"]``
and each of its columns ``["shep-panel"]``, with its minimum width as
``--shep-panel-min`` in the page's style sheet (what its content needs;
the layout pass sums them): side by side the row fills the space under the
header and each panel scrolls inside its own region; stacked, the panels
stack and only the page scrolls. ``shep-panel-hd``,
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
from src.webui.components.renderers import KEEP_OPEN_JS, RENDERER_HELP
from src.webui.style import (FIT, FIT_WIDE, FOLD, FOLD_FLOOR, FOLD_ROOM, FOLDED_ATTR, LAYOUT_ATTR,
                              NARROW_ATTR, SEG_WHOLE, SHORT_ATTR, SHORT_BELOW)

BRAND = "SHEPHERD-Advanced"

#: Level 1 (outer) tab ids and labels, in order.
CLINICAL, RESEARCH, SETTINGS = "clinical", "research", "settings"
#: Level 2 (Research) page ids.
DATASETS, TRAINING, FINETUNE, TEST, MODELS = "datasets", "training", "finetune", "test", "models"
RESEARCH_PAGES = (DATASETS, TRAINING, FINETUNE, TEST, MODELS)
DEFAULT_RESEARCH_PAGE = TRAINING

LABELS = {
    CLINICAL: "Diagnosis (clinical)",
    RESEARCH: "Research (pre-clinical)",
    SETTINGS: "⚙ Settings",
    DATASETS: "Datasets",
    TRAINING: "Training",
    FINETUNE: "Fine-tune",
    TEST: "Test",
    MODELS: "Models",
}

#: The browser title for each location.
TITLES = {
    CLINICAL: f"Diagnosis · Clinical — {BRAND}",
    SETTINGS: f"Settings — {BRAND}",
    **{page: f"{LABELS[page]} · Research — {BRAND}" for page in RESEARCH_PAGES},
}

#: **The stage of the research work each research page belongs to** (the
#: stage hint, module docstring): the one mapping, read by the research
#: tablist's name and by the head script's hint.
PREPARE, EVALUATE = "Prepare", "Evaluate"
STAGES = {DATASETS: PREPARE, TRAINING: PREPARE, FINETUNE: PREPARE, TEST: EVALUATE, MODELS: EVALUATE}
#: The hint's word before the stage ("Stage: Prepare").
STAGE_WORD = "Stage"


def _research_tablist_label() -> str:
    groups: Dict[str, list] = {}
    for page in RESEARCH_PAGES:
        groups.setdefault(STAGES[page], []).append(LABELS[page])
    return "Research pages: " + ", ".join(f"{stage} ({', '.join(pages)})"
                                          for stage, pages in groups.items())


#: The research tablist's name for a screen reader, its pages by stage
#: (`STAGES`): "Research pages: Prepare (Datasets, Training, Fine-tune),
#: Evaluate (Test, Models)".
RESEARCH_TABLIST_LABEL = _research_tablist_label()

#: **The research tabs in groups** (the owner's decision of 2026-11-22): data
#: (Datasets), training (Training, Fine-tune), evaluation (Test, Models),
#: drawn "Datasets │ Training Fine-tune │ Test Models". A visual grouping only:
#: not the stages (`STAGES`: Datasets is still Prepare), no tab moves, and the
#: tab order, the tablist's name and the stage hint are unchanged. The first
#: tab of each later group keeps a gap before it (`TAB_GROUP_GAP`), in which a
#: decorative line is drawn while the tabs sit on one line (`TAB_GROUP_CSS`).
TAB_GROUPS = ((DATASETS,), (TRAINING, FINETUNE), (TEST, MODELS))
#: The gap before a later group's first tab (the style token it reads).
TAB_GROUP_GAP = "var(--shep-tabgroup-gap)"
#: Set on ``#shep-research-nav`` by the head script's layout pass while its
#: tabs take more than one line: the separators are not drawn then (a line
#: never starts with one); the gaps stay, so the mark changes nothing the
#: pass measures.
TABS_WRAPPED_ATTR = "data-shep-tabs-wrapped"


def _tab_group_css() -> str:
    """The tab groups' style, generated from `TAB_GROUPS` only, through the
    documented ``<elem_id>-button`` ids: the gap before each later group's
    first tab, and its separator, an empty ``::before`` (no node of the
    tablist, no role, no focus stop, no text: nothing a screen reader or the
    keyboard meets), drawn in the gap's middle, never over the tab's hit
    target, and not drawn while the tabs wrap (`TABS_WRAPPED_ATTR`)."""
    rules = []
    for group in TAB_GROUPS[1:]:
        tab = f"#shep-research-nav > .tab-wrapper > .tab-container > #{page_elem_id(group[0])}-button"
        rules.append(f"{tab} {{ position: relative; margin-inline-start: {TAB_GROUP_GAP} !important; }}")
        rules.append(f"{tab}::before {{ content: \"\"; position: absolute; top: 8px; bottom: 8px; "
                     f"inset-inline-start: calc({TAB_GROUP_GAP} / -2 - 0.5px); width: 1px; "
                     "background: var(--shep-border-2); pointer-events: none; }")
        rules.append(f"#shep-research-nav[{TABS_WRAPPED_ATTR}] > .tab-wrapper > .tab-container > "
                     f"#{page_elem_id(group[0])}-button::before {{ display: none; }}")
    return "\n".join(rules) + "\n"


#: **Where a notice's next step may lead** (`renderers.notice_block`'s
#: ``data-shep-go`` and ``data-shep-focus``): each page by its id, with the
#: elements on it whose heading the link may focus. The head script opens
#: only a page listed here and focuses only an element listed for it.
GO_TARGETS = {SETTINGS: ("shep-dl",), **{page: () for page in RESEARCH_PAGES}}

#: The shared timers' periods, in seconds.
FAST_SECONDS = 1.5
SLOW_SECONDS = 5.0


def plain_html(value: Any = "", *, js_on_load: Optional[str] = None, **kwargs: Any) -> Any:
    """A ``gr.HTML`` for HTML the dashboard renders itself: without Gradio's
    default prose CSS (which restyles tables, buttons and paragraphs inside
    it) and without its default click script; pass ``js_on_load`` for one of
    your own (``trigger('click', {...})``). **Its disclosures stay as the
    reader left them across every drawing** (`renderers.KEEP_OPEN_JS`, before
    any script of its own): whatever event draws it again, a poll's, a
    tick's, a render's or the reader's own act's (a field edited, a Load, an
    order changed), and whatever it then holds, since what an HTML holds is
    its renderers' choice at run time. The chart figures take the same."""
    classes = kwargs.pop("elem_classes", None) or []
    classes = [classes] if isinstance(classes, str) else list(classes)
    return gr.HTML(value, apply_default_css=False, js_on_load=KEEP_OPEN_JS + (js_on_load or ""),
                   elem_classes=["shep-html", *classes], **kwargs)


def picked(control: Any) -> Any:
    """**The one event of an operator's choice in a select**, for a listener
    to wire (``picked(select)(fn, …)``, or a trigger of ``gr.on``): once per
    option chosen, by pointer or by keys, never for the page's own write of
    its value.

    Gradio 6.30's single-select Dropdown (measured, the WebUI ledger's UI-11)
    sends ``input`` on every blur and once more for an option clicked (the
    click, then the blur it causes): two requests for one pick, and one for a
    focus that chose nothing. Its ``select`` is not sent for an option chosen
    with the arrow keys and Enter, and ``change`` follows the server's own
    writes of the value too (a page that writes a select back would start its
    listener again). Its ``blur`` comes once per option chosen either way and
    never from a write, so a Dropdown's choice is its blur; a focus that
    chose nothing is one request, as with ``input``. Every other control
    (a Radio, a multiselect Dropdown, a Checkbox) keeps its ``input``, which
    it sends once per change.

    **A listener keeps every choice** (``trigger_mode="multiple"``): with
    Gradio's default for a blur ("once") the browser drops a choice made
    while the listener still answers the one before it, and the session
    keeps that one under the select showing the next (OPS-C-R2: A, then B
    while A's read ran, then A again, left B's read for good). An action
    that uses the choice checks it first (`unread_choice`)."""
    if isinstance(control, gr.Dropdown) and not getattr(control, "multiselect", False):
        return control.blur
    return control.input


#: **What an action says of a choice not read yet** (`unread_choice`;
#: ``{what}`` is the select's label), after the page's own lead ("Not
#: started: "; on the Test page, before its "Nothing was started.").
UNREAD_CHOICE = ("the {what} you chose is still being read. Try again once it shows, or choose "
                 "it again.")


def unread_choice(*choices: Tuple[str, Any, Any]) -> Optional[str]:
    """**The submit boundary of a select's choice** (OPS-C-R2, the external
    review of 6a2d52e): an action that uses what a select chose (a start)
    takes the select's value in its own request, as the reader sees it, and
    is refused while the session's state was read for another choice.

    A choice is read by later events (its `picked` listener, then the lists
    written after it, each a request of its own), and a ``gr.State`` is read
    when the action runs: a Start pressed while that reading was on its way,
    or after it failed, read the state of the choice before (Fine-tune
    started on the dataset and the base shown before the one chosen; the
    Training run bar's Start on its dataset before). Each of `choices` is
    ``(what, shown, read)``: the select's label, its value as the action's
    own request carries it, and the value the state was read for. Returns
    `UNREAD_CHOICE` for the first whose two differ (the action then does
    nothing), else None. Nothing waits and nothing is locked: once the
    reading lands the two agree and the same action goes through; a reading
    that failed is made again by choosing again."""
    for what, shown, read in choices:
        if shown != read:
            return UNREAD_CHOICE.format(what=what)
    return None


def tab_elem_id(tab: str) -> str:
    return f"shep-tab-{tab}"


def page_elem_id(page: str) -> str:
    return f"shep-page-{page}"


#: The tab groups' style (`_tab_group_css`), for the head's style element
#: (`app.dashboard_head`).
TAB_GROUP_CSS = _tab_group_css()


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
    #: A model handed to the Test page by a cross-link (`gr.State`, the module
    #: docstring), taken and cleared by its next opening.
    test_model: Any = None
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

    @property
    def training_workspace_writes(self) -> Dict[str, Any]:
        """Keyword arguments for every listener that writes
        ``training_workspace`` (or Training's own record of its run bar's
        workspace): one queued group of one (`TRAINING_SELECTION_CONCURRENCY`)."""
        return {"concurrency_id": TRAINING_SELECTION_CONCURRENCY, "concurrency_limit": 1}

    def on_page_open(self, page: str, kwargs: Dict[str, Any]) -> Any:
        """Run an event (``fn``, ``inputs``, ``outputs``, …) each time `page` is
        shown: Diagnosis' or Settings' tab ``select``; for a research page, the
        change of its token, which the shell bumps when its tab is selected and
        when Research is selected while it is the page shown (returning to
        Research shows that page as it is now, and only that page). Returns the
        event, so a step can follow it (``.then``)."""
        if page in RESEARCH_PAGES:
            return self.page_tokens[page].change(**kwargs)
        if page in self.tabs:
            return self.tabs[page].select(**kwargs)
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
        location=gr.State(CLINICAL), test_model=gr.State(None),
        research_page=gr.State(DEFAULT_RESEARCH_PAGE),
        page_tokens={page: gr.State(0) for page in RESEARCH_PAGES},
        tabs={RESEARCH: research_tab})


def _shown(page: str) -> Callable[[], str]:
    def shown() -> str:
        return page
    shown.__name__ = f"_shown_{page}"
    return shown


#: **Every event that writes the session's location** (Diagnosis', Settings',
#: Research's and each research tab's select) runs one at a time, in the order
#: they were sent, each sent once per selection (`LOCATION_WRITES`): the
#: second of a pair sees the location the first set (`_open_research`) (UI-14,
#: B46-mount: it ran without the queue, so a slow answer set the location to
#: Diagnosis while Models was shown, and Models' poll did nothing). A select
#: that a page's mount fires (Diagnosis' tab, about 1.5 s after the page
#: loads, later after a reload's restore) never writes its page over one
#: opened after it: each writer writes only for the page shown as it was sent
#: (`SENT_JS`, `_written`).
OPEN_PAGE_CONCURRENCY = "shep-open-research-page"
#: The keyword arguments of every location writer (`OPEN_PAGE_CONCURRENCY`).
#: ``trigger_mode="multiple"``: with Gradio's default a select asked while
#: the same listener's event waits in the group would be dropped.
LOCATION_WRITES = {"concurrency_id": OPEN_PAGE_CONCURRENCY, "concurrency_limit": 1,
                   "trigger_mode": "multiple"}
#: Where the head script keeps this browser tab's page (``sessionStorage``).
KEPT_PAGE = "shep-location"
#: **What every location writer hands over as it is sent** (`ShepShell.sent`):
#: the page this browser tab shows then and when ("<page> <ms>", the page's
#: own clock), in a hidden box in front of the writer's own inputs. Ordering
#: the writers (`LOCATION_WRITES`) is not enough: Gradio sends a tab's select
#: when the tab mounts, and Diagnosis' mounts late, after a reload's restore
#: (or a hand) has already opened another page (UI-14, V-RB1-1). So a writer
#: writes only for its own page while it is the one shown, and never over a
#: write sent after it (`_written`).
SENT_JS = "(...args) => [window.ShepShell ? window.ShepShell.sent() : '', ...args.slice(1)]"

#: The one queued group of every listener that writes a session's Training
#: run-bar workspace (``PageContext.training_workspace`` and Training's own
#: record of it; `PageContext.training_workspace_writes`): Datasets' "Use for
#: training ›" and, on Training, the page open, the select, the change of
#: ``training_workspace`` and every render of the run bar, the one the fast
#: poll hands a running job's workspace to included. Gradio 6.30 reads a
#: ``gr.State`` when an event starts and writes it when the event ends and
#: orders no two events otherwise, so a late one wrote back the workspace it
#: started from. The group is the server's, as the Test page's selection group
#: is: two sessions' events wait for each other too.
TRAINING_SELECTION_CONCURRENCY = "shep-training-selection"


def _open_research(location: Any, research_page: Any, *tokens: Any) -> Tuple[Any, ...]:
    """Research selected: the research page shown is the one to refresh (its
    token bumped); the other pages' tokens stay, so they do no work.

    **Once per opening.** On the first opening of Research, and on a
    cross-link that changes the research page too, Gradio also fires the
    inner tab's select (on mounting it, or on switching it). Whichever of the
    two runs first refreshes the page and sets the location to it; the other
    finds the page already shown (`location`) and leaves its token. The two
    run one at a time (`OPEN_PAGE_CONCURRENCY`)."""
    page = research_page if research_page in RESEARCH_PAGES else DEFAULT_RESEARCH_PAGE
    if location == page:
        return (page, *(gr.update() for _ in tokens))
    bumped = tuple((t or 0) + 1 if p == page else gr.update()
                   for p, t in zip(RESEARCH_PAGES, tokens, strict=True))
    return (page, *bumped)


def _open_page(page: str) -> Callable[[Any, Any], Tuple[Any, Any, Any]]:
    """A research page's tab selected: it is the page shown and is refreshed,
    unless it already is (Research's select refreshed it for this opening)."""
    def open_page(location: Any, token: Any) -> Tuple[Any, Any, Any]:
        return page, page, (gr.update() if location == page else (token or 0) + 1)
    open_page.__name__ = f"_open_{page}"
    return open_page


def _sent(value: Any) -> Tuple[Optional[str], float]:
    """The page the browser showed when a writer was sent, and when
    (`SENT_JS`); ``(None, 0.0)`` when the head script gave nothing."""
    parts = value.split() if isinstance(value, str) else []
    if len(parts) == 2:
        try:
            return parts[0], float(parts[1])
        except ValueError:
            pass
    return None, 0.0


def _written(writer: Callable[..., Any], ours: Callable[[str], bool],
             outputs: int) -> Callable[..., Tuple[Any, ...]]:
    """`writer` as a location writer: given what the browser handed over as it
    was sent (`SENT_JS`) and the time of the last write applied, it writes
    (its own outputs, then that time) only when the page shown then is its
    own (`ours`) and it was sent no earlier than the write applied last; else
    it changes nothing. A select a tab's mount sends while another page is
    shown, or one that arrives after a later one, never moves the session's
    location off the page shown. With no head script it writes as asked."""
    def write(sent: Any, last: Any, *inputs: Any) -> Tuple[Any, ...]:
        shown, at = _sent(sent)
        if shown is not None and (not ours(shown) or at < (last or 0.0)):
            return tuple(gr.update() for _ in range(outputs + 1))
        answer = writer(*inputs)
        answer = answer if isinstance(answer, tuple) else (answer,)
        return (*answer, at if shown is not None else gr.update())
    write.__name__ = writer.__name__
    return write


def _restore(kept: Any) -> Tuple[Any, Any, Any]:
    """**The page this browser tab showed before it was loaded again**
    (UI-14): `kept` is the page id the head script kept (`KEPT_PAGE`). A
    research page or Settings is opened as a cross-link opens it
    (`PageContext.go`: both levels and the research page shown, so only that
    page refreshes); Diagnosis, nothing kept or a value no page has leaves
    the first screen. It opens a page and nothing else: the page reads its
    state as any opening does."""
    if isinstance(kept, str) and (kept in RESEARCH_PAGES or kept == SETTINGS):
        return PageContext.go(kept)
    return PageContext.no_nav()


def _brand_html() -> str:
    """The product's name: the page's one level-1 heading (a page title for a
    screen reader's heading list; the panels' titles are h2), drawn as it was
    (`.shep-brand`)."""
    return f'<h1 class="shep-brand">{BRAND}</h1>'


def build_shell(blocks: Any, pages: Dict[str, Callable[[PageContext], None]],
                *, help_registry: Optional[HelpRegistry] = None) -> PageContext:
    """Lay out the header, the two navigation levels and every page.

    `pages` maps every page id (clinical, datasets, training, finetune, test,
    models, settings) to its builder; a missing one is a ``KeyError``. Returns the
    context the pages were given."""
    registry = help_registry or HelpRegistry()
    registry.topics(RENDERER_HELP)
    served_model_changed = gr.State(None)
    training_workspace = gr.State(None)
    test_model = gr.State(None)
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
                              test_model=test_model,
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
                            pages[page](ctx)
            with gr.Tab(LABELS[SETTINGS], id=SETTINGS, alignment="right",
                        elem_id=tab_elem_id(SETTINGS), elem_classes=["shep-tab"]) as settings_tab:
                tabs[SETTINGS] = settings_tab
                pages[SETTINGS](ctx)
    # Where this session is: the page shown, and the research page to refresh
    # when Research is selected (only that one, and once: Gradio also selects
    # a research tab when it first mounts and when a cross-link switches to
    # it; `_open_research`). The events that bump a token go through the
    # queue: Gradio 6.30 dispatches a State's ``.change`` only from a queued
    # event's answer (an answer to a ``queue=False`` POST carries no changed
    # states). Each writer is given the page shown as it was sent, and writes
    # only for its own page and never over a later write (`_written`).
    sent = gr.Textbox(visible=False, elem_id="shep-sent-page")
    written_at = gr.State(0.0)
    for tab in (CLINICAL, SETTINGS):
        tabs[tab].select(_written(_shown(tab), tab.__eq__, 1), [sent, written_at],
                         [location, written_at], show_progress="hidden", js=SENT_JS,
                         **LOCATION_WRITES)
    tokens = [page_tokens[page] for page in RESEARCH_PAGES]
    tabs[RESEARCH].select(_written(_open_research, RESEARCH_PAGES.__contains__, 1 + len(tokens)),
                          [sent, written_at, location, research_page, *tokens],
                          [location, *tokens, written_at], show_progress="hidden", js=SENT_JS,
                          **LOCATION_WRITES)
    for page in RESEARCH_PAGES:
        tabs[page].select(_written(_open_page(page), page.__eq__, 3),
                          [sent, written_at, location, page_tokens[page]],
                          [location, research_page, page_tokens[page], written_at],
                          show_progress="hidden", js=SENT_JS, **LOCATION_WRITES)
    # A reload comes back on the page this browser tab showed (`_restore`):
    # the head script hands over the page it kept, the answer opens it as a
    # cross-link does, and the script keeps the page shown again from then.
    kept = gr.Textbox(visible=False, elem_id="shep-kept-page")
    blocks.load(_restore, [kept], ctx.nav_outputs, show_progress="hidden",
                js="() => (window.ShepShell ? window.ShepShell.kept : '')").then(
        None, js="() => { if (window.ShepShell) { window.ShepShell.restored(); } }")
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
  // The page this browser tab showed (`KEPT_PAGE`), read before Gradio draws
  // the first screen: the page load hands it to the shell (`_restore`), and
  // nothing is kept again until that answer is in (`restored`) or the
  // operator picks a tab, so the first screen drawn never overwrites it.
  var KEPT = '__KEPT__', kept = '', restoring = false;
  try { kept = window.sessionStorage.getItem(KEPT) || ''; } catch (err) { kept = ''; }
  restoring = kept !== '';
  function keep(key) {
    if (restoring || !key) { return; }
    try { if (window.sessionStorage.getItem(KEPT) !== key) { window.sessionStorage.setItem(KEPT, key); } }
    catch (err) { /* nothing kept: a reload starts on Diagnosis */ }
  }
  document.addEventListener('click', function (e) {
    if (restoring && e.isTrusted && e.target && e.target.closest && e.target.closest('[role="tab"]')) {
      restoring = false;
    }
  }, true);
  var TABLIST_LABELS = {'shep-nav': 'Area', 'shep-research-nav': __RESEARCH_LABEL__};
  function selected(navId) {
    var b = document.querySelector('#' + navId + ' > .tab-wrapper [role="tab"][aria-selected="true"]');
    return b ? b.getAttribute('data-tab-id') : null;
  }
  function shownKey() {
    var outer = selected('shep-nav') || 'clinical';
    return outer === RESEARCH ? (selected('shep-research-nav') || DEFAULT_PAGE) : outer;
  }
  // What a location writer hands over as it is sent (`SENT_JS`): the page
  // shown now and when, by this page's clock.
  function sent() { return shownKey() + ' ' + performance.now(); }
  function title() {
    var key = shownKey();
    var want = TITLES[key];
    if (want && document.title !== want) { document.title = want; }
    if (selected('shep-nav')) { keep(key); }
  }
  function labels() {
    for (var id in TABLIST_LABELS) {
      var list = document.querySelector('#' + id + ' > .tab-wrapper [role="tablist"]');
      if (list && list.getAttribute('aria-label') !== TABLIST_LABELS[id]) {
        list.setAttribute('aria-label', TABLIST_LABELS[id]);
      }
    }
  }
  // **The stage hint** (module docstring): plain text after the research
  // pages' tablist, never inside it, saying which stage (STAGES) the page
  // shown belongs to; hidden when no research page is shown. Written from
  // the tab selected on every pass (a click, a key, a cross-link, a reload's
  // restore), its words by textContent, and put back when Gradio draws the
  // row again. It reads no job state and is no control.
  var STAGES = __STAGES__, STAGE_WORD = __STAGE_WORD__;
  function stage() {
    var row = document.querySelector('#shep-research-nav > .tab-wrapper');
    if (!row) { return; }
    var hint = row.querySelector(':scope > .shep-stage');
    if (!hint) {
      hint = document.createElement('p');
      hint.className = 'shep-stage';
      var word = document.createElement('span');
      word.className = 'shep-stage-k';
      word.textContent = STAGE_WORD + ':';
      var value = document.createElement('span');
      value.className = 'shep-stage-v';
      hint.appendChild(word);
      hint.appendChild(document.createTextNode(' '));
      hint.appendChild(value);
    }
    if (row.lastElementChild !== hint) { row.appendChild(hint); }
    var want = STAGES[shownKey()] || '', shown = hint.querySelector('.shep-stage-v');
    if (shown.textContent !== want) { shown.textContent = want; }
    if (hint.hidden !== !want) { hint.hidden = !want; }
  }
  // Row 1 holds the brand, the area switch, the chips and Settings. When they
  // do not fit on one row, the chips and Settings move onto a line of their
  // own, under the area switch however many lines it takes (at 480 px its
  // two tabs take two): the chips sit where that line starts, measured, so
  // they never cover a tab.
  // **The one layout decision** (the layout rule, module docstring): the
  // window's inner size first (no wider than tall: stacked), then whether the
  // page shown fits side by side by its panels' own minimum widths. Read from
  // the style sheets (each panel's --shep-panel-min) and the window, never
  // from the state it sets, so a pass never undoes the one before it; a
  // scrollbar the stacked page shows is counted back (side by side has none).
  var LAYOUT = '__LAYOUT__', SHORT = '__SHORT__', SHORT_BELOW = __SHORT_BELOW__;
  function shownTab() {
    var outer = selected('shep-nav') || 'clinical';
    if (outer === RESEARCH) {
      return document.getElementById('shep-page-' + (selected('shep-research-nav') || DEFAULT_PAGE));
    }
    return document.getElementById('shep-tab-' + outer);
  }
  function drawn(el) { return el.getClientRects().length > 0; }
  // A length a style sheet gives in a custom property, in px: registered (style.py's @property),
  // it computes to px; where a browser has no @property, a value in rem or em is converted by
  // the root's or the element's own font size (no probe element, no state read).
  function lengthPx(el, prop) {
    var raw = getComputedStyle(el).getPropertyValue(prop).trim(), n = parseFloat(raw) || 0;
    if (/rem$/.test(raw)) { return n * (parseFloat(getComputedStyle(document.documentElement).fontSize) || 16); }
    if (/em$/.test(raw)) { return n * (parseFloat(getComputedStyle(el).fontSize) || 16); }
    return n;
  }
  function sideFits(tab) {
    var rows = tab ? tab.querySelectorAll('.shep-page') : [], box = null, kids = [];
    for (var i = 0; i < rows.length && !box; i++) { if (drawn(rows[i])) { box = rows[i]; } }
    if (!box) { return true; }
    for (;;) {
      kids = Array.prototype.filter.call(box.children, drawn);
      if (kids.length === 1 && kids[0].children.length) { box = kids[0]; } else { break; }
    }
    var s = getComputedStyle(box), n = 0;
    var need = (parseFloat(s.paddingLeft) || 0) + (parseFloat(s.paddingRight) || 0);
    for (var j = 0; j < kids.length; j++) {
      var min = lengthPx(kids[j], '--shep-panel-min');
      if (min > 0) { need += min; n += 1; }
    }
    if (n > 1) { need += (n - 1) * (parseFloat(s.columnGap) || 0); }
    var bar = window.innerWidth - document.documentElement.clientWidth;
    return need <= box.getBoundingClientRect().width + bar + 0.5;
  }
  function orientation() {
    var root = document.documentElement, w = window.innerWidth, h = window.innerHeight;
    var want = w > h && sideFits(shownTab()) ? 'side' : 'stack';
    if (root.getAttribute(LAYOUT) !== want) { root.setAttribute(LAYOUT, want); }
    if (root.hasAttribute(SHORT) !== (h < SHORT_BELOW)) { root.toggleAttribute(SHORT, h < SHORT_BELOW); }
  }
  // **A wide form only while it fits** (the layout rule's fit, module
  // docstring): each box of class FIT drawn shows its wide form (FIT_WIDE)
  // while that form's own narrowest, its min-content width, fits the box's
  // content width, and is marked NARROW when it does not, for its page's
  // style sheet to draw its narrow form. The form is laid out at its
  // narrowest for the measure alone (its own inline style, put back before
  // anything is painted; a style change is no mutation this pass listens
  // to), so the measure never depends on the state it sets.
  var FIT = '__FIT__', FIT_WIDE = '__FIT_WIDE__', NARROW = '__NARROW__';
  var AT_NARROWEST = 'position:absolute !important;visibility:hidden !important;'
    + 'width:min-content !important;min-width:0 !important;max-width:none !important;';
  // A choice group's options laid out on one line, for the measure alone.
  var ON_ONE_LINE = 'flex-wrap:nowrap !important;flex-direction:row !important;'
    + 'width:max-content !important;max-width:none !important;';
  // **A choice group whole or one per line** (style.py's M3): a radio or checkbox group (a
  // fieldset block, not a segmented control, or a segmented control whose labels stay whole,
  // SEG_WHOLE) whose options do not fit one line of its room is marked NARROW, and its options
  // stand one per line, never one alone under the others.
  var CHOICE = 'fieldset.block:not(.shep-seg), fieldset.block.__SEG_WHOLE__';
  // The scroll positions of `el`'s ancestors are held across the measure and put back: drawn
  // out of flow at its narrowest, `el` no longer fills a box that scrolls around it, and reading
  // its width clamps that box's scroll (a FIT table's sideways scroll went back to 0 on every
  // pass, the page near its end jumped to its top, a focused control left the view).
  function measured(el, extra, read) {
    var held = [];
    for (var a = el.parentElement; a; a = a.parentElement) {
      if (a.scrollLeft || a.scrollTop) { held.push([a, a.scrollLeft, a.scrollTop]); }
    }
    var kept = el.getAttribute('style');
    el.setAttribute('style', (kept ? kept + ';' : '') + extra);
    var value = read(el);
    if (kept === null) { el.removeAttribute('style'); } else { el.setAttribute('style', kept); }
    for (var i = 0; i < held.length; i++) {
      var h = held[i];
      if (h[0].scrollLeft !== h[1]) { h[0].scrollLeft = h[1]; }
      if (h[0].scrollTop !== h[2]) { h[0].scrollTop = h[2]; }
    }
    return value;
  }
  function roomOf(box) {
    var s = getComputedStyle(box);
    return box.clientWidth - (parseFloat(s.paddingLeft) || 0) - (parseFloat(s.paddingRight) || 0);
  }
  function width(el) { return el.getBoundingClientRect().width; }
  // Each box is measured with its own mark taken off (put back before anything is painted; an
  // attribute this pass does not listen to), so what it measures never depends on the state it
  // sets: a table whose optional columns the mark hides, a group the mark stacks.
  function fits() {
    var boxes = document.querySelectorAll('.' + FIT);
    for (var i = 0; i < boxes.length; i++) {
      var box = boxes[i], wide = box.querySelector('.' + FIT_WIDE);
      if (!wide || !drawn(box)) { continue; }
      var was = box.hasAttribute(NARROW);
      if (was) { box.removeAttribute(NARROW); }
      var need = measured(wide, AT_NARROWEST + 'display:' + (wide.tagName === 'TABLE' ? 'table' : 'block')
        + ' !important;', width);
      var room = roomOf(box);
      if (was) { box.setAttribute(NARROW, ''); }
      var narrow = need > room + 0.5;
      if (box.hasAttribute(NARROW) !== narrow) { box.toggleAttribute(NARROW, narrow); }
    }
    var groups = document.querySelectorAll(CHOICE);
    for (var g = 0; g < groups.length; g++) {
      var group = groups[g], options = group.querySelector(':scope > .wrap:has(> label)'), stack = false;
      if (options && options.children.length > 1 && drawn(group)) {
        var had = group.hasAttribute(NARROW);
        if (had) { group.removeAttribute(NARROW); }
        var space = roomOf(group);
        stack = measured(options, ON_ONE_LINE, width) > space + 0.5;
        if (had) { group.setAttribute(NARROW, ''); }
      } else if (!options || !drawn(group)) { continue; }
      if (group.hasAttribute(NARROW) !== stack) { group.toggleAttribute(NARROW, stack); }
    }
  }
  // **The tab groups' separators only while the research tabs sit on one line** (TAB_GROUPS):
  // the pass marks the research tabs' row when its tabs take more than one line, from the
  // tabs' own tops; the gaps before each group stay in both states, so the mark moves nothing.
  var TABS_WRAPPED = '__TABS_WRAPPED__';
  function tabLines() {
    var nav = document.getElementById('shep-research-nav');
    if (!nav || !drawn(nav)) { return; }
    var tabs = nav.querySelectorAll(':scope > .tab-wrapper > .tab-container > button[role="tab"]'), tops = {};
    for (var i = 0; i < tabs.length; i++) {
      if (drawn(tabs[i])) { tops[Math.round(tabs[i].offsetTop / 4)] = true; }
    }
    var wrapped = Object.keys(tops).length > 1;
    if (nav.hasAttribute(TABS_WRAPPED) !== wrapped) { nav.toggleAttribute(TABS_WRAPPED, wrapped); }
  }
  // **A box whole only while its column has the room** (the layout rule's
  // fold, module docstring): side by side, each box of class FOLD drawn is
  // whole while its column (its nearest ancestor holding a region of class
  // FOLD_ROOM) holds what it holds with the box whole and that region at its
  // floor (its FOLD_FLOOR property, or its whole content when less), and is marked
  // FOLDED when it does not, for its long parts to give way. The box is
  // measured whole (the mark taken off for the measure alone and put back
  // before anything is painted; an attribute this pass does not listen to),
  // so the measure never depends on the state it sets. Stacked, nothing folds.
  var FOLD = '__FOLD__', FOLD_ROOM = '__FOLD_ROOM__', FOLD_FLOOR = '__FOLD_FLOOR__', FOLDED = '__FOLDED__';
  function px(v) { return parseFloat(v) || 0; }
  function folds() {
    var side = document.documentElement.getAttribute(LAYOUT) === 'side';
    var boxes = document.querySelectorAll('.' + FOLD);
    for (var i = 0; i < boxes.length; i++) {
      var box = boxes[i], column = box.parentElement, room = null;
      while (column && !(room = column.querySelector(':scope > .' + FOLD_ROOM))) {
        column = column.parentElement;
      }
      var fold = false;
      if (side && room && drawn(box) && drawn(room)) {
        var was = box.hasAttribute(FOLDED);
        if (was) { box.removeAttribute(FOLDED); }
        var c = getComputedStyle(column), r = getComputedStyle(room);
        var space = column.clientHeight - px(c.paddingTop) - px(c.paddingBottom);
        var floor = px(r.getPropertyValue(FOLD_FLOOR));
        var need = Math.min(floor, room.scrollHeight), kids = 0;
        for (var k = column.firstElementChild; k; k = k.nextElementSibling) {
          var s = getComputedStyle(k);
          if (!drawn(k) || s.position === 'absolute' || s.position === 'fixed') { continue; }
          kids += 1;
          need += px(s.marginTop) + px(s.marginBottom) + (k === room ? 0 : k.getBoundingClientRect().height);
        }
        need += Math.max(kids - 1, 0) * px(c.rowGap);
        if (was) { box.setAttribute(FOLDED, ''); }
        fold = need > space + 0.5;
      }
      if (box.hasAttribute(FOLDED) !== fold) { box.toggleAttribute(FOLDED, fold); }
    }
  }
  // **The scroll cue on every region that scrolls** (R-73; the sweep's L7, SW-LK1..8): an element
  // whose own overflow scrolls (auto or scroll) and whose content runs past its foot keeps
  // MORE while more lies below, and MORE_X while more lies past its right edge, which
  // `style.py` draws as a fade at that edge; at its end the words are drawn whole again. No page
  // names its regions (a region a page forgot scrolled with no cue: Settings, Fine-tune's form,
  // Training's charts, the Diagnosis fields). Not a region: a field's own box, a select's open
  // list, the help bubble, and a box that draws its own sideways cue (Compare's, CUED).
  var MORE = 'data-shep-more', MORE_X = 'data-shep-more-x';
  var NOT_REGION = 'textarea, select, input, ul.option-list, #shep-help-bubble, #shep-help-bubble *, '
    + '.shep-rs-top, .shep-rs-wrap';
  function scrolls(v) { return v === 'auto' || v === 'scroll'; }
  function cue(el) {
    var s = getComputedStyle(el);
    var more = scrolls(s.overflowY) && el.scrollHeight - el.scrollTop - el.clientHeight > 1;
    var side = scrolls(s.overflowX) && el.scrollWidth - el.scrollLeft - el.clientWidth > 1;
    if (el.hasAttribute(MORE) !== more) { el.toggleAttribute(MORE, more); }
    if (el.hasAttribute(MORE_X) !== side) { el.toggleAttribute(MORE_X, side); }
    // Its content may change size with no change this pass listens to (a chart drawn again at
    // its new width): a region marked once is watched for that too.
    if ((more || side) && ro && !watched.has(el)) {
      watched.add(el);
      for (var k = el.firstElementChild; k; k = k.nextElementSibling) { ro.observe(k); }
    }
  }
  var watched = typeof WeakSet !== 'undefined' ? new WeakSet() : { has: function () { return true; } };
  // Every element that holds more than its box (a cheap read first; most hold none), and every
  // element still marked (its content shrank), is looked at again on each pass.
  function cues() {
    var all = document.body ? document.body.getElementsByTagName('*') : [];
    for (var i = 0; i < all.length; i++) {
      var e = all[i];
      if ((e.scrollHeight > e.clientHeight + 1 || e.scrollWidth > e.clientWidth + 1
           || e.hasAttribute(MORE) || e.hasAttribute(MORE_X)) && !e.matches(NOT_REGION)) { cue(e); }
    }
  }
  document.addEventListener('scroll', function (e) {
    var t = e.target;
    if (t && t.nodeType === 1 && !t.matches(NOT_REGION)) { cue(t); }
  }, true);
  // **A separator never ends a line either** (`renderers.SEP`; the sweep's L7, SW-LK2 and
  // SW-LK7): SEP joins its "·" to the fact before it, so no line starts with one; where the line
  // breaks after it (the fact after it starts the next line), the pass marks it SEP_END and the
  // style sheet does not draw it. Only its visibility changes, never its box, so the mark never
  // changes what it measured.
  var SEP_END = 'data-shep-end';
  function blockOf(el) {
    var b = el.parentElement;
    while (b && /^(inline|contents)/.test(getComputedStyle(b).display)) { b = b.parentElement; }
    return b;
  }
  // The box of the first character drawn after `sep` in its block.
  function nextGlyph(sep) {
    var block = blockOf(sep);
    if (!block) { return null; }
    var walk = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
    walk.currentNode = sep;
    for (var n = walk.nextNode(); n; n = walk.nextNode()) {
      if (sep.contains(n)) { continue; }
      var at = n.data.search(/\S/);
      if (at < 0) { continue; }
      var range = document.createRange();
      range.setStart(n, at); range.setEnd(n, at + 1);
      var boxes = range.getClientRects();
      for (var k = 0; k < boxes.length; k++) { if (boxes[k].width > 0.1) { return boxes[k]; } }
    }
    return null;
  }
  function seps() {
    var all = document.querySelectorAll('.shep-sep');
    for (var i = 0; i < all.length; i++) {
      var sep = all[i], end = false;
      if (drawn(sep)) {
        var own = sep.getClientRects(), last = own[own.length - 1], next = nextGlyph(sep);
        end = !!next && next.top >= last.bottom - last.height / 2;
      }
      if (sep.hasAttribute(SEP_END) !== end) { sep.toggleAttribute(SEP_END, end); }
    }
  }
  function layout() {
    orientation();
    fits();
    folds();
    tabLines();
    cues();
    seps();
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
    var line = Math.round(right.getBoundingClientRect().top - shell.getBoundingClientRect().top);
    shell.style.setProperty('--shep-chips-top', line + 'px');
  }
  // The layout rule's titled ellipsis, on every page: a select's value or a
  // segmented control's label cut to an ellipsis has its whole text as its
  // title, and loses the title when it fits again. Gradio sets a select's
  // value without an event to listen for, so each element is looked at again
  // on each layout pass (after the page changes) and when the pointer or the
  // focus reaches it.
  var TITLED = 'input[role="combobox"], .shep-seg label, .shep-chip-text';
  function titleOne(el) {
    var box = el, text;
    if (el.tagName === 'INPUT') { text = el.value || ''; }
    else { box = el.querySelector('span') || el; text = (box.textContent || '').trim(); }
    var want = text && box.scrollWidth > box.clientWidth + 1 ? text : '';
    if (want) { if (el.getAttribute('title') !== want) { el.setAttribute('title', want); } }
    else if (el.hasAttribute('title')) { el.removeAttribute('title'); }
  }
  function titles() {
    var all = document.querySelectorAll(TITLED);
    for (var i = 0; i < all.length; i++) { titleOne(all[i]); }
  }
  function titleAt(e) {
    var el = e.target && e.target.closest ? e.target.closest(TITLED) : null;
    if (el) { titleOne(el); }
  }
  document.addEventListener('mouseover', titleAt, true);
  document.addEventListener('focusin', titleAt, true);
  document.addEventListener('input', titleAt, true);
  var queued = false;
  function update() {
    if (queued) { return; }
    queued = true;
    requestAnimationFrame(function () { queued = false; title(); labels(); stage(); layout(); titles(); });
  }
  // A page shown, a panel shown or hidden (a drawer; Gradio's hidden class),
  // the page's own class that changes its panels: each lays the page out again.
  new MutationObserver(update).observe(document.documentElement,
    {childList: true, subtree: true, attributes: true, attributeFilter: ['aria-selected', 'class']});
  window.addEventListener('resize', update);
  // A font arriving redraws text in its own widths: the fit is measured again.
  if (document.fonts && document.fonts.addEventListener) {
    document.fonts.addEventListener('loadingdone', update);
  }
  if (window.ResizeObserver) {
    var ro = new ResizeObserver(update);
    var watch = setInterval(function () {
      var chips = document.querySelector('#shep-chips');
      var shell = document.getElementById('shep-shell');
      if (chips && shell) { ro.observe(chips); ro.observe(shell); clearInterval(watch); }
    }, 200);
  }
  // **A notice's next step** (`renderers.notice_block`'s link): a click on a button with
  // data-shep-go opens that page as a tab click does (the research area first for a research
  // page), then brings the element named by data-shep-focus into view and moves the focus to
  // its heading. Only the places GO_TARGETS lists are opened; any other value does nothing.
  var GO = __GO__;
  function tabButton(page) {
    var outer = page === 'settings' || page === 'clinical' || page === RESEARCH;
    return document.getElementById((outer ? 'shep-tab-' : 'shep-page-') + page + '-button');
  }
  function land(focusId, tries) {
    var el = focusId ? document.getElementById(focusId) : null;
    if (!el || !drawn(el)) {
      if (focusId && tries > 0) { requestAnimationFrame(function () { land(focusId, tries - 1); }); }
      return;
    }
    var head = el.querySelector('h1, h2, h3') || el;
    if (!head.hasAttribute('tabindex')) { head.setAttribute('tabindex', '-1'); }
    head.scrollIntoView({block: 'start'});
    head.focus({preventScroll: true});
  }
  function goTo(page, focusId, tries) {
    var research = page !== 'settings' && page !== 'clinical';
    var outer = tabButton(research ? RESEARCH : page);
    if (research && outer && outer.getAttribute('aria-selected') !== 'true') { outer.click(); }
    var tab = tabButton(page);
    if (!tab || !drawn(tab)) {
      if (tries > 0) { requestAnimationFrame(function () { goTo(page, focusId, tries - 1); }); }
      return;
    }
    if (tab.getAttribute('aria-selected') !== 'true') { tab.click(); }
    land(focusId, 120);
  }
  document.addEventListener('click', function (e) {
    var b = e.target && e.target.closest ? e.target.closest('button[data-shep-go]') : null;
    if (!b) { return; }
    var page = b.getAttribute('data-shep-go') || '', focusId = b.getAttribute('data-shep-focus') || '';
    if (!Object.prototype.hasOwnProperty.call(GO, page)) { return; }
    if (GO[page].indexOf(focusId) < 0) { focusId = ''; }
    e.preventDefault();
    restoring = false;
    goTo(page, focusId, 120);
  });
  window.ShepShell = {update: update, title: title, titles: titles, kept: kept, sent: sent, go: goTo,
                     restored: function () { restoring = false; update(); }};
})();
"""

#: The shell's script, for the mount's ``head=`` (with the help script).
SHELL_HEAD = ("<script>" + _SHELL_JS
              .replace("__TITLES__", as_json_for_script(TITLES))
              .replace("__DEFAULT__", DEFAULT_RESEARCH_PAGE)
              .replace("__KEPT__", KEPT_PAGE)
              .replace("__RESEARCH_LABEL__", as_json_for_script(RESEARCH_TABLIST_LABEL))
              .replace("__STAGES__", as_json_for_script(STAGES))
              .replace("__STAGE_WORD__", as_json_for_script(STAGE_WORD))
              .replace("__LAYOUT__", LAYOUT_ATTR).replace("__SHORT__", SHORT_ATTR)
              .replace("__SHORT_BELOW__", str(SHORT_BELOW))
              .replace("__FIT__", FIT).replace("__FIT_WIDE__", FIT_WIDE)
              .replace("__NARROW__", NARROW_ATTR).replace("__SEG_WHOLE__", SEG_WHOLE)
              .replace("__FOLD__", FOLD).replace("__FOLD_ROOM__", FOLD_ROOM)
              .replace("__FOLD_FLOOR__", FOLD_FLOOR).replace("__FOLDED__", FOLDED_ATTR)
              .replace("__TABS_WRAPPED__", TABS_WRAPPED_ATTR)
              .replace("__GO__", as_json_for_script({k: list(v) for k, v in GO_TARGETS.items()}))
              + "</script>")


__all__ = [
    "BRAND",
    "CLINICAL",
    "DATASETS",
    "DEFAULT_RESEARCH_PAGE",
    "FINETUNE",
    "GO_TARGETS",
    "KEPT_PAGE",
    "LABELS",
    "LOCATION_WRITES",
    "MODELS",
    "OPEN_PAGE_CONCURRENCY",
    "PageContext",
    "RESEARCH",
    "RESEARCH_PAGES",
    "RESEARCH_TABLIST_LABEL",
    "SETTINGS",
    "SHELL_HEAD",
    "STAGES",
    "STAGE_WORD",
    "TABS_WRAPPED_ATTR",
    "TAB_GROUPS",
    "TAB_GROUP_CSS",
    "TEST",
    "TITLES",
    "TRAINING",
    "UNREAD_CHOICE",
    "build_shell",
    "help_texts",
    "page_elem_id",
    "picked",
    "standalone_context",
    "tab_elem_id",
    "unread_choice",
]
