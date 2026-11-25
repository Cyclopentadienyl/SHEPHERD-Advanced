"""
⚙ Settings › Data locations: the folders this service uses, chosen here (UI-06)
===============================================================================
The owner's decision of 2026-11-16: a clinician starts the dashboard by double
click, the page opens with nothing configured, and the data locations are
chosen and saved here, then reused at the next start. A start-time setting
(an environment variable) still wins, and the page says which value is in
effect and where it came from. One panel beside Runtime settings, built from
the one view of the data locations (`src.utils.deployment_view`: what
`src.config.server_settings` resolves, what the saved file holds), every value
it shows escaped:

- **Each row** (Workspaces folder, Results registry, Import folder, Training
  evidence folders) names the folder in plain words and what needs it, then:
  *In effect* (the value, or ○ Not set) with its source in plain words
  ("Saved in Settings", "Set when the service started"); *Saved* only when it
  differs from what is in effect (pending, or overridden); a path box with
  **Check** (the same rule a job's read applies, the resolved path and what it
  holds: "2 workspaces") and **Save**, open on a row with nothing in effect
  (the next step) and behind **Change…** on a set row, which opens it in its
  place with **Cancel** (the owner's "Set to / Change to" decision of
  2026-11-24, `key_html`); for a required folder that is not set,
  **Create suggested folder**, naming it (``~/SHEPHERD-data/workspaces``:
  `deployment_settings.suggested_folder`, the one suggestion the not-set
  words name too), created only on that click (0700 for the registry and the
  import folder, which hold patient-level data); never a guessed old folder
  or a slot's.
  The variable is named only under "Advanced: source ›".
- **A box is the reader's**: it is set when the panel first opens and when
  its own row's Save succeeds (`box_value`: the evidence folders' saved list,
  else empty); every other act and answer, another row's Check or Save, a
  refusal, Settings shown again, leaves what the reader typed (OPS-C-R5), and
  a box Change… opened stays open with it. Cancel (or Escape) closes the box
  and sets it back (`_on_cancel`), and a Save that changes the row closes it.
- **Overridden** (Syncthing's form): a row the environment sets keeps its box
  and Save, which change the saved value only; its Saved line says it applies
  once the start-up setting is removed, and the page never calls it in effect.
  It is not applied while the setting is in place (`settings_store.held_back`),
  so a command run by hand never reads it before the service does.
- **Pending**: a saved change waits while something could use the old
  locations (a GPU job, a command run by hand: `data_locations.apply_saved`).
  One notice says why, from the same check Apply makes (`data_locations.view`:
  the job runner busy, then a GPU lock the old values name), with **Apply**;
  it also applies at the next start. A change a start-up setting holds back
  is said in its row only.
- **An unreadable settings file**: one ✕ notice naming the file, with
  "Replace the unreadable file (a copy is kept)"; Save is off and Create is
  hidden until then (said once, in the notice). Folders set at start still
  apply.
- **Read-only**: for a request D8 refuses, every path reads "Set (path
  hidden)" or ○ Not set (sources stay), a refused value's problem is said in
  fixed words that never quote the value (`deployment_view.problem_words`),
  every other text is redacted by the one redaction, every box and button is
  disabled with the lock, one notice above the rows says the consequence
  (`PATHS_OPERATOR_ONLY`; D8's whole reason is the header chip's), and each
  row says R-WHY's d8 phrase (`ROW_LOCKED`, "🔒 Operator controls only")
  once: beside its locked Change… on a set row, else beside its Check and
  Save, in their bar (the owner's decision of 2026-11-19), its Create on a
  line of its own; the phrase names every control of its row it dims
  (`row_controls`), so each is described by it.

These are folders on the server machine; the page says so. Nothing here
blocks another page: a page that needs a folder that is not set says so in
its own place and works for everything else.

Operator controls (D8): Check, Save, Create suggested folder, Apply and
Replace each check their own Gradio request first (`operator_access`) and are
refused, changing and reading nothing, unless D8 allows it. The page's open
draws them by the one gated-button rule (`chips.gated`). Check reads the
server's folders, so a read-only viewer cannot probe them either.

Module: src/webui/components/data_locations.py
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import gradio as gr

from src.api.services import data_locations
from src.config import server_settings
from src.config.deployment_settings import (
    KEY_INTAKE_DIR,
    KEY_REGISTRY,
    KEY_TRAINING_EVIDENCE_DIRS,
    KEY_WORKSPACES_ROOT,
    KEYS,
    suggested_folder,
)
from src.utils import deployment_view
from src.utils.deployment_view import DeploymentView, FieldView
from src.webui.components import renderers as r
from src.webui.components.chips import LOCK, OPERATOR_ONLY, gated, operator_only
from src.webui.operator_access import operator_access
from src.webui.shell import SETTINGS as SETTINGS_PAGE
from src.webui.shell import plain_html, tab_elem_id

#: The panel's intro: where these folders are.
INTRO = ("These are folders on the server machine: this computer when SHEPHERD-Advanced was "
         "started from its desktop.")
#: A first start: nothing is set (the notice's state), what that means here (its line), and that
#: nothing else is blocked by it (one click away, under its "Details ›": the spec's B, R-LOAD).
NOTHING_SET = "Nothing is set yet."
NOTHING_SET_THEN = "Each folder is needed only by what its row names."
NOTHING_SET_DETAILS = ("Pages: Settings and every other page open as usual",)
#: Each source in plain words (`server_settings.SOURCE_*`).
SOURCE_WORDS = {
    server_settings.SOURCE_SAVED: "Saved in Settings",
    server_settings.SOURCE_ENVIRONMENT: "Set when the service started",
}
#: A saved value the environment overrides.
OVERRIDDEN = "Saved, applies when the start-up setting is removed"
#: A path a read-only viewer is not shown.
PATH_HIDDEN = "Set (path hidden)"
#: The read-only notice: its consequence in plain words (the owner's D6 of 2026-11-24). Who can
#: act is the header chip's whole reason, which each row's phrase opens (R-WHY), never here.
PATHS_OPERATOR_ONLY = "Folder paths are hidden in this read-only view."
#: Each row's short reason beside its locked box and buttons, for a request
#: D8 refuses: R-WHY's ``d8`` phrase (`chips.OPERATOR_ONLY`, drawn by
#: `chips.operator_only`), which opens the D8 chip's whole reason.
ROW_LOCKED = OPERATOR_ONLY
#: The unreadable file's notice (Save is off until it is replaced: said here once).
UNREADABLE = ("Settings file cannot be read: the folders saved in it are unknown and not used. "
              "Save is off until the file is replaced. Folders set when the service started "
              "still apply.")
#: A row whose value would come from an unreadable file (the notice says why).
UNKNOWN = "Unknown: the settings file cannot be read"
#: The box's placeholder: a hint, never information the page needs.
PLACEHOLDER = "Folder path on the server"
EVIDENCE_PLACEHOLDER = "One folder path on the server per line"
#: The evidence row with nothing set: what that means.
EVIDENCE_NONE = "None set: the check reads the measured workspace only"

#: The folders that hold patient-level data: created 0700 (only the account
#: that runs the service can open them), as the deployment guide asks.
PRIVATE_KEYS = (KEY_REGISTRY, KEY_INTAKE_DIR)

#: The buttons' words, shown with no lock while D8 allows the request.
CHECK_LABEL = "Check"
SAVE_LABEL = "Save"
#: **A set row's control in its box's place** (the owner's "Set to / Change to" decision of
#: 2026-11-24): it opens the row's box there, and Cancel closes it (`key_html`).
CHANGE_LABEL = "Change…"
CANCEL_LABEL = "Cancel"
#: The key before a row's box: nothing set yet, or a value to change (a set row opened by
#: Change…, or a value set and not usable, whose box stays open).
SET_KEY = "Set to"
CHANGE_KEY = "Change to"
CREATE_LABEL = "Create suggested folder"
APPLY_LABEL = "Apply"
REPAIR_LABEL = "Replace the unreadable file (a copy is kept)"
#: What Replace does, said beside it.
REPAIR_THEN = ("Replace renames it beside itself and keeps it. Then choose the folders here "
               "again.")

#: The order of the per-row outputs (`_panel`).
ROW_PARTS = ("state", "key", "box", "check", "save", "cancel", "lock", "create", "suggest",
             "result", "adv")
PANEL_PARTS = ("notice", "repair", "apply", "status", "where")


# ---------------------------------------------------------------------------
# What a row says
# ---------------------------------------------------------------------------
def _tilde(path: Path) -> str:
    """`path` with the account's home shown as ``~``."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except (ValueError, RuntimeError, KeyError):
        return str(path)


def _title(path: Path, shown: bool) -> str:
    """A suggested folder's full path as its title, for an operator only."""
    return f' title="{r.e(path)}"' if shown else ""


def _value_html(value: Any) -> str:
    """A path, or a list of them, in code style, each on its own line."""
    values = value if isinstance(value, (tuple, list)) else (value,)
    return "<br>".join(r.token_html(v, classes="shep-dl-path") for v in values)


def effect_html(effect: str, source: str, *, path: bool) -> str:
    """**In effect's value and its source, two facts**: after a path (`path`) the
    source starts a line of its own, so it never runs on after the path's
    last line (the sweep's L7, D13: "…/registry Set when the service
    started" at 1080 x 1920); after words ("Set (path hidden)") it follows
    on their line. Both HTML strings, already escaped."""
    value = f'<span class="shep-dl-v">{effect}</span>'
    if not source:
        return value
    return f"{value}{'<br>' if path else ' '}{source}"


def _source_html(item: FieldView) -> str:
    words = SOURCE_WORDS.get(item.source)
    return f'<span class="shep-dl-src">{r.e(words)}</span>' if words else ""


def _words(text: Optional[str], shown: bool) -> str:
    """A note as the viewer may see it (paths hidden when not `shown`),
    escaped. A problem goes through `deployment_view.problem_words`."""
    return r.e(text if shown else deployment_view.hide_paths(text))


def state_html(item: FieldView, view: DeploymentView, *, shown: bool) -> str:
    """A row's facts (the module docstring): In effect and its source, Saved
    when it differs (pending, overridden), the problem or the note. The
    start-up setting is named under its "Advanced: source ›" only
    (`advanced_html`)."""
    rows: List[Tuple[str, str]] = []
    # The saved file cannot be read and nothing else sets this: the notice
    # says why, once; the row says only that its value is unknown.
    unknown = view.file.problem is not None and item.configured is None
    path = False
    if item.value is not None:
        effect, path = (_value_html(item.value), True) if shown else (r.e(PATH_HIDDEN), False)
    elif item.configured is not None and item.problem:
        path = shown
        effect = (r.status_mark("error", "Not usable") + " "
                  + (_value_html(item.configured) if shown else r.e(PATH_HIDDEN)))
    elif item.optional and item.problem is None:
        effect = r.status_mark("neutral", EVIDENCE_NONE)
    elif unknown:
        effect = r.status_mark("neutral", UNKNOWN)
    else:
        effect = r.status_mark("neutral", "Not set")
    # The value and its source are two facts in the text too, not only on
    # screen (T08 flow: "…/ws-rootSaved in Settings" to a copy, find-in-page
    # and a screen reader, the gap being the value's margin alone).
    rows.append(("In effect", effect_html(effect, _source_html(item), path=path)))
    differs = item.saved is not None and (item.pending or item.overridden
                                          or item.saved != item.configured)
    if differs:
        saved = _value_html(item.saved) if shown else r.e(PATH_HIDDEN)
        if item.overridden:
            mark = f'<span class="shep-muted">{r.e(OVERRIDDEN)}</span>'
        elif item.pending:
            mark = r.status_mark("warn", "Saved, not in effect yet")
        else:
            mark = ""
        rows.append(("Saved", f"{saved}<br>{mark}" if mark else saved))
    if item.problem and not unknown and (item.configured is not None or not item.unset):
        # A refused value in fixed words for a viewer whose paths are hidden:
        # its refusal quotes the value as given (`deployment_view.problem_words`).
        problem = r.e(deployment_view.problem_words(item, paths_shown=shown))
        rows.append(("Problem", f'<span class="shep-dl-problem">{problem}</span>'))
    if item.note:
        rows.append(("Note", r.status_mark("warn", _words(item.note, shown), words_html=True)))
    return r.summary_list(rows, classes="shep-dl-kv")


def advanced_html(item: FieldView) -> str:
    """A row's "Advanced: source ›", one level deep: the start-up setting
    that would override it, and whether it is set. The variable is named
    here only."""
    is_set = item.source == server_settings.SOURCE_ENVIRONMENT
    advanced = r.summary_list([
        ("Start-up setting", f"<code>{r.e(item.variable)}</code> "
                             + ("set" if is_set else "not set")),
    ], classes="shep-dl-kv")
    return (f'<details class="shep-more shep-dl-adv"><summary>Advanced: source ›'
            f"</summary>{advanced}</details>")


def name_html(key: str) -> str:
    """A row's plain name and what needs it (static)."""
    name, purpose = deployment_view.FIELD_WORDS[key]
    return (f'<div class="shep-dl-name"><h3>{r.e(name)}</h3>'
            f'<p class="shep-sub">{r.e(purpose)}</p></div>')


def where_html(view: DeploymentView, *, shown: bool) -> str:
    """Where the saved values are kept and the order a value is read in, once
    for the panel, one click away."""
    file = view.file
    if file.path is None:
        place = r.e(deployment_view.file_problem_words(file, paths_shown=shown)
                    or "No place for a saved settings file in this service's environment")
    elif shown:
        place = f"<code>{r.e(file.path)}</code>"
    else:
        place = r.e(PATH_HIDDEN)
    rows = r.summary_list([
        ("Settings file", place),
        ("Order", r.e("A start-up setting first, then the value saved here. Nothing else is read.")),
    ], classes="shep-dl-kv")
    return (f'<details class="shep-more shep-dl-adv"><summary>Where these are kept ›</summary>'
            f"{rows}</details>")


# ---------------------------------------------------------------------------
# The panel as one read draws it
# ---------------------------------------------------------------------------
def reason_line(text: str = "", *, phrase: str = "") -> str:
    """**A refusal's line beside what it turns off** on Settings (the lock
    before it): in this panel the notice's consequence once, above its rows,
    and each row's ``d8`` phrase (`ROW_LOCKED`, `chips.operator_only`)
    beside its locked Change… or its Check and Save; in Runtime settings the
    phrase beside its actions. `text`:
    words (escaped); `phrase`: a dimmed control's phrase (R-WHY, HTML)."""
    words = " ".join(part for part in (r.e(text) if text else "", phrase) if part)
    return (f'<div class="shep-reason"><span aria-hidden="true">{LOCK}</span>'
            f"<span>{words}</span></div>")


def notice_html(view: DeploymentView, access: Any) -> str:
    """The panel's one notice: read-only (D8's reason), an unreadable file,
    a pending change and why, or a first start; nothing otherwise."""
    shown = access.allowed
    if not shown:
        return reason_line(PATHS_OPERATOR_ONLY)
    if view.file.problem:
        then = (REPAIR_THEN if view.file.repairable else
                "It cannot be opened: correct its permissions or the disk where it is. It is "
                "never renamed.")
        return (f'<div class="shep-dl-notice shep-dl-error">'
                f'{r.status_mark("error", UNREADABLE)}<p class="shep-sub">'
                f"{r.e(view.file.problem)}. {r.e(then)}</p></div>")
    if view.applicable:
        # A change a start-up setting holds back is said in its own row
        # (`OVERRIDDEN`), not here: Apply cannot apply it.
        names = ", ".join(view.field(key).name for key in view.applicable)
        return (f'<div class="shep-dl-notice">'
                f'{r.status_mark("warn", f"Saved, not in effect yet: {names}")}'
                f'<p class="shep-sub">{r.e(view.pending_reason)}</p></div>')
    if view.nothing_set:
        first = {"kind": "neutral", "summary": NOTHING_SET, "details": NOTHING_SET_DETAILS}
        return (f'<div class="shep-dl-notice">'
                f'{r.notice_block(first, consequence=NOTHING_SET_THEN)}</div>')
    return ""


def changeable(item: FieldView) -> bool:
    """**Whether a row shows "Change…" in its box's place** (the owner's "Set to / Change to"
    decision of 2026-11-24): a row with a value in effect, its box opened only to change it. A
    row with nothing in effect keeps its box open, the next step: nothing set, a value set and
    not usable, or one the unreadable settings file would hold. The evidence folders' box
    replaces their whole list (`box_value`, `_value_of`), so a set list follows the same rule."""
    return item.value is not None


def row_controls(key: str, *, change: bool, create: bool) -> str:
    """**The controls of a row a read-only request dims**, as element ids, which its one phrase
    names (R-WHY's ``data-for``, so each is described by it): its box, Check and Save, then its
    Change… on a set row (`changeable`) or its Create where it is offered."""
    ids = [f"dl_box_{key}", f"dl_check_{key}", f"dl_save_{key}"]
    ids += [f"dl_change_{key}"] if change else []
    ids += [f"dl_create_{key}"] if create else []
    return " ".join(ids)


def _drawn_as(item: FieldView) -> str:
    """What a set row's Change… is drawn for (`renderers.keep_as`): the row's values, so the
    shared keeper (`renderers.KEEP_OPEN_JS`) keeps its box open across the panel's redraws only
    while the row says the same, and a Save that changes them closes it."""
    facts = repr((item.value, item.configured, item.saved, item.source, item.pending,
                  item.overridden))
    return hashlib.sha256(facts.encode("utf-8")).hexdigest()[:12]


def key_html(item: FieldView, *, allowed: bool) -> str:
    """**What stands before a row's box** (the owner's "Set to / Change to" decision of
    2026-11-24). A row with nothing in effect: its key ("Set to", or "Change to" for a value
    set and not usable), its box open beside it. A set row (`changeable`): "Change…" in its
    box's place, a button whose ``aria-expanded`` and ``aria-controls`` name the "Change to"
    key it shows (its own HTML: the shared keeper holds it open across the panel's redraws
    while the row says the same, `_drawn_as`); its script (`CHANGE_JS`) opens the box there,
    the focus in it, and Cancel or Escape closes it. For a request D8 refuses, a set row's
    Change… is locked (`chips.gated`'s lock) and its phrase beside it names what it dims."""
    key = item.key
    if not changeable(item):
        words = SET_KEY if item.configured is None else CHANGE_KEY
        return f'<span class="shep-dl-key">{r.e(words)}</span>'
    if not allowed:
        phrase = operator_only(control=row_controls(key, change=True, create=False))
        return (f'<div class="shep-dl-chg"><button type="button" class="shep-dl-change" '
                f'id="dl_change_{key}" disabled>{LOCK} {r.e(CHANGE_LABEL)}'
                f"</button>{reason_line(phrase=phrase)}</div>")
    return (f'<div class="shep-dl-chg"{r.keep_as(_drawn_as(item))}>'
            f'<span class="shep-dl-key" id="dl_changing_{key}" hidden>{r.e(CHANGE_KEY)}</span>'
            f'<button type="button" class="shep-dl-change" id="dl_change_{key}" '
            f'aria-expanded="false" aria-controls="dl_changing_{key}">{r.e(CHANGE_LABEL)}'
            "</button></div>")


#: **Change… and Cancel, element-scoped** (the key's own HTML, after the shared keeper,
#: `shell.plain_html`): Change… opens the row's box in its place (the row's style sheet draws
#: the box and its buttons while it is expanded) with the focus in the box; Cancel, and Escape
#: in the box (which presses Cancel, so its handler empties the box), close it and give the
#: focus back to Change…. The keeper holds the reader's choice across the panel's redraws.
#: **Whether the row's facts stand beside its name** (`BESIDE`, the owner's request of
#: 2026-11-25): the row's head wraps by what each side holds, so only the boxes it draws say so.
#: Each time the row or its facts change size, the head is measured as it wraps (the mark off,
#: put back before anything is painted, so the mark never changes what it measured) and the row
#: is marked with its name column's width (`NAME_WIDTH`); the style sheet then gives a set row's
#: actions the line its facts end on (`PAGE_CSS`). A key no longer in the page stops measuring.
BESIDE = "data-dl-beside"
NAME_WIDTH = "--dl-name"
CHANGE_JS = r"""
function changer() { return element.querySelector('button.shep-dl-change[aria-controls]'); }
function rowOf() { return element.closest('.shep-dl-row'); }
function setOpen(open) {
  var b = changer();
  if (!b) { return; }
  var k = document.getElementById(b.getAttribute('aria-controls'));
  if (k && element.contains(k)) { k.hidden = !open; }
  b.setAttribute('aria-expanded', open ? 'true' : 'false');
  var row = rowOf(), box = open && row ? row.querySelector('.shep-dl-field :is(input, textarea)') : null;
  if (open) { if (box) { box.focus(); } } else { b.focus(); }
}
element.addEventListener('click', function (e) {
  var b = e.target.closest ? e.target.closest('button.shep-dl-change[aria-controls]') : null;
  if (b && !b.disabled && b.getAttribute('aria-expanded') !== 'true') { setOpen(true); }
});
document.addEventListener('click', function (e) {
  var row = rowOf(), c = e.target.closest ? e.target.closest('button[id^="dl_cancel_"]') : null;
  if (c && row && row.contains(c) && changer()) { setOpen(false); }
});
document.addEventListener('keydown', function (e) {
  var row = rowOf(), b = changer();
  if (e.key !== 'Escape' || !row || !b || b.getAttribute('aria-expanded') !== 'true') { return; }
  if (!e.target.closest || !e.target.closest('.shep-dl-field') || !row.contains(e.target)) { return; }
  var c = row.querySelector('button[id^="dl_cancel_"]');
  if (c) { e.preventDefault(); e.stopPropagation(); c.click(); }
});
var queued = false, sizes = null;
function arrange() {
  queued = false;
  if (!element.isConnected) { if (sizes) { sizes.disconnect(); } return; }
  var row = rowOf();
  if (!row) { return; }
  var name = row.querySelector('.shep-dl-head > .shep-dl-left');
  var facts = row.querySelector('.shep-dl-head > .shep-dl-facts');
  var was = row.hasAttribute('__BESIDE__');
  if (was) { row.removeAttribute('__BESIDE__'); }
  var beside = false, width = '';
  if (name && facts && name.getClientRects().length && facts.getClientRects().length) {
    var n = name.getBoundingClientRect(), f = facts.getBoundingClientRect();
    beside = f.left >= n.right - 1 && f.top < n.bottom - 1;
    width = n.width + 'px';
  }
  if (beside && row.style.getPropertyValue('__NAME_WIDTH__') !== width) {
    row.style.setProperty('__NAME_WIDTH__', width);
  }
  if (beside) { row.setAttribute('__BESIDE__', ''); }
}
function later() { if (!queued) { queued = true; requestAnimationFrame(arrange); } }
if (window.ResizeObserver && rowOf()) {
  sizes = new ResizeObserver(later);
  sizes.observe(rowOf());
  var state = rowOf().querySelector('[id^="dl_state_"]');
  if (state) { sizes.observe(state); }
}
""".replace("__BESIDE__", BESIDE).replace("__NAME_WIDTH__", NAME_WIDTH)


def box_value(item: FieldView) -> str:
    """**What a row's box holds when the panel sets it** (`_panel`'s `fill`):
    the evidence folders' saved list, one per line (the list is edited as a
    whole); empty for a one-folder row, whose box holds only a change to
    make."""
    if item.key == KEY_TRAINING_EVIDENCE_DIRS:
        return "\n".join(item.saved or ())
    return ""


def _row_updates(item: FieldView, view: DeploymentView, access: Any,
                 result: str, fill: bool = False) -> List[Any]:
    """One row's outputs, in `ROW_PARTS` order. The box's value is set only
    when `fill` (`box_value`) or for a request D8 refuses (emptied: no path
    is shown); otherwise it is left as the reader typed it."""
    allowed, shown = access.allowed, access.allowed
    readable = view.file.problem is None
    suggestion = suggested_folder(item.key)
    creatable = (suggestion is not None and item.unset and readable
                 and view.file.path is not None)
    change = changeable(item)
    # A read-only row's one phrase: beside its locked Change… on a set row (`key_html`), else
    # beside its locked Check and Save, naming every control of the row it dims.
    lock = "" if allowed or change else reason_line(
        phrase=operator_only(control=row_controls(item.key, change=False, create=creatable)))
    box: Dict[str, Any] = {"interactive": allowed}
    if not allowed:
        box["value"] = ""
    elif fill:
        box["value"] = box_value(item)
    return [
        gr.update(value=state_html(item, view, shown=shown)),
        gr.update(value=key_html(item, allowed=allowed)),
        gr.update(**box),
        gr.update(**gated(CHECK_LABEL, allowed)),
        gr.update(**gated(SAVE_LABEL, allowed, interactive=readable)),
        gr.update(visible=allowed and change),
        gr.update(value=lock),
        gr.update(visible=creatable, **gated(CREATE_LABEL, allowed)),
        gr.update(value=(f'<span{_title(suggestion, shown)}>'
                         f'{r.token_html(_tilde(suggestion), classes="shep-dl-path")}</span>')
                  if creatable else ""),
        gr.update(value=result),
        gr.update(value=advanced_html(item)),
    ]


def _panel(request: gr.Request, *, results: Optional[Mapping[str, str]] = None,
           status: str = "", fill: Iterable[str] = ()) -> Tuple[Any, ...]:
    """Every output of the panel from one read of the view and of D8's
    answer for this request: the rows' (`ROW_PARTS`, in `KEYS` order), then
    the panel's (`PANEL_PARTS`). `results` holds a row's outcome line,
    `status` the panel's; `fill` names the rows whose box is set
    (`box_value`): every row when the panel first opens, a row whose own
    Save succeeded. Every other row's box keeps what the reader typed, for
    every act and every answer, a refusal too (OPS-C-R5: each answer set the
    evidence folders box back to the saved list, so another row's Check
    lost a draft typed there)."""
    access = operator_access(request)
    view = data_locations.view()
    results = results or {}
    fill = frozenset(fill)
    out: List[Any] = []
    for item in view.fields:
        out += _row_updates(item, view, access, results.get(item.key, ""),
                            fill=item.key in fill)
    repairable = access.allowed and view.file.problem is not None and view.file.repairable
    out += [
        gr.update(value=notice_html(view, access)),
        gr.update(visible=repairable, **gated(REPAIR_LABEL, access.allowed)),
        gr.update(visible=bool(view.applicable) and access.allowed,
                  **gated(APPLY_LABEL, access.allowed)),
        gr.update(value=status),
        gr.update(value=where_html(view, shown=access.allowed)),
    ]
    return tuple(out)


def _refused(request: gr.Request, access: Any, key: Optional[str] = None) -> Tuple[Any, ...]:
    """The panel redrawn with the refusal where the act was asked for."""
    said = r.status_mark("error", access.refusal)
    if key is None:
        return _panel(request, status=said)
    return _panel(request, results={key: said})


# ---------------------------------------------------------------------------
# What a folder holds (Check)
# ---------------------------------------------------------------------------
def _count(number: int, one: str, many: str) -> str:
    return f"{number} {one if number == 1 else many}"


#: **What each folder holds, by the server's own list** (`holds_words`): the list, the item's
#: name (one, many), and, while it holds none, the next step: where its contents will appear.
#: "Built" names no builder: the dashboard builds no graph (the owner's decision of
#: 2026-11-24 on "(no workspace yet…)").
HOLDS = {
    KEY_WORKSPACES_ROOT: ("list_workspaces", "workspace", "workspaces",
                          "Each workspace built in it appears in Datasets."),
    KEY_REGISTRY: ("list_cohort_versions", "cohort version", "cohort versions",
                   "Each one imported into it appears in Research\u00a0› Test."),
    KEY_INTAKE_DIR: ("list_intake_files", "import file", "import files",
                     "Each case file placed in it appears in Test\u00a0› Import."),
}


def holds_words(key: str, resolved: Any) -> Tuple[str, Optional[str]]:
    """**What a checked folder holds, then the next step** (the owner's decision of 2026-11-24
    on "(no workspace yet…)": a short state and the next step, apart, never in a
    parenthesis): read by the server's own lists with that folder alone (an explicit mapping:
    no saved file is read). The state ("Holds 2 workspaces.", "Holds no workspace yet."),
    and while it holds none, where its contents will appear (`HOLDS`); None for no step. A
    list that cannot be read says so, with its reason, and no step."""
    from src.api.services import server_lists

    if key == KEY_TRAINING_EVIDENCE_DIRS:
        return f"{_count(len(resolved), 'folder', 'folders')} listed.", None
    reader, one, many, step = HOLDS[key]
    environ = {server_settings.SAVED_VARIABLES[key]: str(resolved)}
    listing = getattr(server_lists, reader)(environ=environ)
    if listing.items:
        return f"Holds {_count(len(listing.items), one, many)}.", None
    if listing.problem:
        return f"Its contents cannot be listed: {listing.problem.rstrip('.')}.", None
    return f"Holds no {one} yet.", step


def _value_of(key: str, text: Any) -> Any:
    """The box's text as the value of `key`: a path, or the evidence
    folders one per line (blank lines skipped). Empty: None."""
    text = "" if text is None else str(text)
    if key == KEY_TRAINING_EVIDENCE_DIRS:
        entries = [line.strip() for line in text.splitlines() if line.strip()]
        return entries or None
    return text.strip() or None


def check_words(key: str, text: Any) -> str:
    """Check: the box's value by the rule its setting is read by
    (`server_settings.check_data_location`): "Usable" and the resolved path, then a line of
    its own saying what the folder holds and, while it holds nothing, the next step
    (`holds_words`), and a note under them where the rule has one."""
    value = _value_of(key, text)
    if value is None:
        return r.status_mark("neutral", "Nothing to check: type a folder path first")
    try:
        resolved, note = server_settings.check_data_location(key, value)
    except ValueError as exc:
        return r.status_mark("error", f"Not usable: {exc}")
    shown = (", ".join(str(p) for p in resolved) if isinstance(resolved, tuple)
             else str(resolved))
    state, step = holds_words(key, resolved)
    holds = " ".join(r.e(words) for words in (state, step) if words)
    said = (f'<div>{r.status_mark("ok", "Usable")} <code>{r.e(shown)}</code></div>'
            f'<p class="shep-dl-holds">{holds}</p>')
    if note:
        said += f'<div>{r.status_mark("warn", note)}</div>'
    return said


#: An act's outcome when the panel's notice already says why a change waits
#: (each fact once: the notice reads the same check Apply makes).
WAITS_ABOVE = "Saved, not in effect yet: the notice above says why"


def _waiting_words(message: str) -> str:
    """A pending apply's outcome: the notice's reference when the notice says
    why (a change Apply could apply waits), else the outcome's own words (a
    change only a start-up setting holds back, said by no notice)."""
    return r.status_mark("warn", WAITS_ABOVE if data_locations.view().applicable else message)


def _outcome_words(done: Any, key: Optional[str] = None) -> str:
    """A save's (and its apply's) outcome as one status line."""
    save, apply = done.save, done.apply
    if not save.saved:
        reason = save.refusals.get(key) if key else None
        return r.status_mark("error", f"Not saved: {reason}" if reason else save.message)
    if key is not None:
        item = data_locations.view().field(key)
        if item.source == server_settings.SOURCE_ENVIRONMENT:
            # Not applied either while it differs (`settings_store.held_back`),
            # so no other process reads it before the setting is removed.
            return r.status_mark("neutral", "Saved. The start-up setting stays in effect. The "
                                            "saved value applies at a start without it.")
    if apply is not None and apply.applied:
        return r.status_mark("ok", "Saved and in effect")
    if apply is not None and apply.pending:
        return _waiting_words(apply.message)
    return r.status_mark("ok", "Saved")


# ---------------------------------------------------------------------------
# Handlers (each an operator control, D8, checked on its own request)
# ---------------------------------------------------------------------------
def _on_locations_open(request: gr.Request, filled: bool = False):
    """The panel each time Settings is shown; the boxes are set at its first
    open for an operator only (`filled`: this session's boxes were set, the
    panel's state, `_on_locations_filled`), so a draft survives Settings
    shown again."""
    return _panel(request, fill=() if filled else KEYS)


def _on_locations_filled(request: gr.Request) -> bool:
    """After an open: whether the boxes are set now (an operator's open set
    them; a request D8 refuses emptied them, so a later open by an operator
    sets them)."""
    return operator_access(request).allowed


def _on_check(key: str, text: Any, request: gr.Request):
    """Check: reads a server folder, so an operator control (D8)."""
    access = operator_access(request)
    if not access.allowed:
        return _refused(request, access, key)
    return _panel(request, results={key: check_words(key, text)})


def _on_save(key: str, text: Any, request: gr.Request):
    """Save the box's value for `key`, then try to apply it at once
    (`data_locations.save_and_apply`). An empty box saves nothing, except for
    the evidence folders, whose list it clears."""
    access = operator_access(request)
    if not access.allowed:
        return _refused(request, access, key)
    value = _value_of(key, text)
    if value is None and key != KEY_TRAINING_EVIDENCE_DIRS:
        return _panel(request, results={key: r.status_mark(
            "neutral", "Nothing saved: type a folder path first")})
    done = data_locations.save_and_apply({key: value})
    return _panel(request, results={key: _outcome_words(done, key)},
                  fill=(key,) if done.save.saved else ())


def _on_cancel(key: str, request: gr.Request):
    """Cancel, or Escape in the box, on a row Change… opened (`CHANGE_JS` closes it): the box
    back to what the panel sets in it (`box_value`: empty, or the evidence folders' saved list),
    so a draft never waits behind Change…, and the row's outcome line emptied (a Check of that
    draft said nothing of the value in effect). It changes nothing; a request D8 refuses gets an
    empty box, no path."""
    item = data_locations.view().field(key)
    value = box_value(item) if operator_access(request).allowed else ""
    return gr.update(value=value), gr.update(value="")


def create_folder(key: str) -> Tuple[Optional[Path], str]:
    """Create `key`'s suggested folder (`deployment_settings.suggested_folder`), 0700 for a
    patient-level one and for ``~/SHEPHERD-data`` itself when this creates it;
    a folder already there is used as it is. Returns it, or None and why."""
    path = suggested_folder(key)
    if path is None:
        return None, "there is no suggested folder for this account"
    try:
        if not path.parent.exists():
            path.parent.mkdir(mode=0o700, parents=True)
        if path.is_dir():
            return path, f"{_tilde(path)} was already there and is used as it is"
        path.mkdir(mode=0o700 if key in PRIVATE_KEYS else 0o777)
        if key in PRIVATE_KEYS:
            os.chmod(path, 0o700)  # the mode, whatever the umask
    except OSError as exc:
        return None, f"{path} cannot be created ({type(exc).__name__}: {exc.strerror or exc})"
    private = " (only this account can open it)" if key in PRIVATE_KEYS else ""
    return path, f"Created {_tilde(path)}{private}"


def _on_create(key: str, request: gr.Request):
    """Create suggested folder: created now, on this click only, then saved
    and applied as a Save would."""
    access = operator_access(request)
    if not access.allowed:
        return _refused(request, access, key)
    path, said = create_folder(key)
    if path is None:
        return _panel(request, results={key: r.status_mark("error", f"Not created: {said}")})
    done = data_locations.save_and_apply({key: str(path)})
    line = _outcome_words(done, key)
    return _panel(request, results={key: f'{r.e(said)}. {line}'})


def _on_apply_saved(request: gr.Request):
    """Apply: the saved values become the ones in use, when nothing could use
    the old ones (`data_locations.apply_saved`); else it says why."""
    access = operator_access(request)
    if not access.allowed:
        return _refused(request, access)
    outcome = data_locations.apply_saved()
    if outcome.pending and not outcome.applied:
        return _panel(request, status=_waiting_words(outcome.message))
    kind = "ok" if outcome.applied else "neutral"
    return _panel(request, status=r.status_mark(kind, outcome.message))


def _on_replace_unreadable(request: gr.Request):
    """Replace the unreadable file: renamed beside it and kept
    (`data_locations.replace_unreadable`), so Settings can save again."""
    access = operator_access(request)
    if not access.allowed:
        return _refused(request, access)
    outcome = data_locations.replace_unreadable()
    return _panel(request, status=r.status_mark("ok" if outcome.replaced else "error",
                                                outcome.message))


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------
def create_data_locations_panel(ctx: Any) -> Dict[str, Any]:
    """Build the Data locations panel (call inside the Settings page, beside
    Runtime settings). Read each time Settings is shown (``ctx.on_page_open``);
    every act redraws the whole panel from one read, each box but the ones
    `_panel` fills left as the reader typed it, then refreshes the header's
    chips (the GPU chip follows the workspaces folder). The controls
    start disabled, as each page's gated controls do, until the open reads
    D8's answer for this request. Returns the components by name, for tests."""
    parts: Dict[str, Any] = {}
    rows: Dict[str, Dict[str, Any]] = {}
    with gr.Column(elem_classes=["shep-panel", "shep-dl", "shep-sized"], elem_id="shep-dl"):
        # The heading takes the focus when a notice's link leads here (the
        # shell's go link, `shell.GO_TARGETS`): a heading, never a control.
        plain_html('<div class="shep-panel-hd"><h2 tabindex="-1">Data locations</h2>'
                   '<span class="shep-panel-right">where this service keeps its data</span>'
                   f'</div><p class="shep-intro">{r.e(INTRO)}</p>', elem_id="shep-dl-hd")
        with gr.Column(elem_id="shep-dl-bd"):
            with gr.Row(elem_id="shep-dl-top"):
                parts["notice"] = plain_html("", elem_id="dl_notice")
                parts["repair"] = gr.Button(REPAIR_LABEL, visible=False, interactive=False,
                                            scale=0, elem_id="dl_repair")
                parts["apply"] = gr.Button(APPLY_LABEL, visible=False, interactive=False,
                                           scale=0, elem_id="dl_apply")
            for key in KEYS:
                row: Dict[str, Any] = {}
                with gr.Row(elem_classes=["shep-dl-row"], elem_id=f"dl_row_{key}"):
                    # The row's head: its name and what needs it, and beside them what is in
                    # effect and where it comes from (words and a disclosure: no field or button).
                    with gr.Row(elem_classes=["shep-dl-head"]):
                        plain_html(name_html(key), elem_classes=["shep-dl-left"])
                        with gr.Column(elem_classes=["shep-dl-facts"]):
                            row["state"] = plain_html("", elem_id=f"dl_state_{key}")
                            row["adv"] = plain_html("", elem_id=f"dl_adv_{key}")
                    with gr.Column(elem_classes=["shep-dl-main"]):
                        with gr.Row(elem_classes=["shep-dl-set"]):
                            # Its key, or a set row's Change… in its box's place (`key_html`).
                            row["key"] = plain_html(f'<span class="shep-dl-key">{SET_KEY}</span>',
                                                    elem_classes=["shep-dl-keycell"],
                                                    js_on_load=CHANGE_JS)
                            # The box and its buttons: their own bar (`shep-bar`), so a group
                            # that wraps starts on the box's edge, never under the key, and the
                            # bar's note (a read-only row's lock) says why its buttons are off.
                            with gr.Row(elem_classes=["shep-dl-field", "shep-bar"]):
                                evidence = key == KEY_TRAINING_EVIDENCE_DIRS
                                row["box"] = gr.Textbox(
                                    label=f"{deployment_view.FIELD_WORDS[key][0]}: folder path",
                                    show_label=False, container=False, interactive=False,
                                    placeholder=EVIDENCE_PLACEHOLDER if evidence else PLACEHOLDER,
                                    lines=2 if evidence else 1, max_lines=6 if evidence else 1,
                                    elem_id=f"dl_box_{key}")
                                # Check and Save, one group (`shep-grp`): they move to
                                # the next line together, never Save alone.
                                with gr.Row(elem_classes=["shep-dl-acts", "shep-grp"]):
                                    row["check"] = gr.Button(CHECK_LABEL, interactive=False,
                                                             scale=0, elem_id=f"dl_check_{key}")
                                    row["save"] = gr.Button(SAVE_LABEL, interactive=False,
                                                            scale=0, elem_id=f"dl_save_{key}")
                                    # A set row opened by Change…: closes it (`_on_cancel`).
                                    row["cancel"] = gr.Button(CANCEL_LABEL, visible=False,
                                                              scale=0, elem_id=f"dl_cancel_{key}")
                                # Read-only: why the box and the buttons are locked (`ROW_LOCKED`),
                                # a whole group of the bar after Check and Save (the owner's
                                # decision of 2026-11-19: beside each row's buttons): on their
                                # line where it holds them, else a line of its own on the box's
                                # edge, never taking Check and Save off the box's line.
                                row["lock"] = plain_html("", elem_id=f"dl_lock_{key}",
                                                         elem_classes=["shep-dl-lock"])
                                # Create and its suggestion, one more whole group of the bar:
                                # after Check and Save on their line, or a line of its own.
                                with gr.Row(elem_classes=["shep-dl-create", "shep-grp"]):
                                    row["create"] = gr.Button(CREATE_LABEL, visible=False,
                                                              interactive=False, scale=0,
                                                              elem_id=f"dl_create_{key}")
                                    row["suggest"] = plain_html("", elem_id=f"dl_suggest_{key}")
                        row["result"] = plain_html("", elem_id=f"dl_result_{key}")
                rows[key] = row
            parts["status"] = plain_html("", elem_id="dl_status")
            parts["where"] = plain_html("", elem_id="dl_where")

    outputs = [rows[key][part] for key in KEYS for part in ROW_PARTS]
    outputs += [parts[name] for name in PANEL_PARTS]
    # Whether this session's boxes were set (`_on_locations_open`).
    filled = gr.State(False)
    ctx.on_page_open(SETTINGS_PAGE, {"fn": _on_locations_open, "inputs": [filled],
                                     "outputs": outputs, "queue": False,
                                     "show_progress": "hidden"}).then(
        fn=_on_locations_filled, outputs=[filled], queue=False, show_progress="hidden")
    for key in KEYS:
        row = rows[key]
        key_state = gr.State(key)
        row["check"].click(fn=_on_check, inputs=[key_state, row["box"]], outputs=outputs,
                           show_progress="hidden")
        row["save"].click(fn=_on_save, inputs=[key_state, row["box"]], outputs=outputs,
                          show_progress="hidden").then(**ctx.chip_refresh)
        row["create"].click(fn=_on_create, inputs=[key_state], outputs=outputs,
                            show_progress="hidden").then(**ctx.chip_refresh)
        row["cancel"].click(fn=_on_cancel, inputs=[key_state], outputs=[row["box"], row["result"]],
                            queue=False, show_progress="hidden")
    parts["apply"].click(fn=_on_apply_saved, outputs=outputs,
                         show_progress="hidden").then(**ctx.chip_refresh)
    parts["repair"].click(fn=_on_replace_unreadable, outputs=outputs,
                          show_progress="hidden").then(**ctx.chip_refresh)
    return {"rows": rows, **parts, "outputs": outputs}


_PAGE = "#" + tab_elem_id(SETTINGS_PAGE)

#: The panel's style sheet (listed with the Settings page's, `runtime_settings.PAGE_CSS`).
#: **A row reads top to bottom** (the UI sweep of 2026-11-22): its head, then
#: "Set to", its box and its buttons: the row's words are never a column
#: beside its box and buttons (a message never shares a row with fields or
#: buttons; the registry's 120 characters stood beside its box at 1850 x 960).
#: **The head** holds the row's name and what needs it at the measure and,
#: beside them while the panel has the room (else under them), what is in
#: effect and its "Advanced: source ›": words and a disclosure, no field or
#: button (the sweep's L7, SW-LK1: the name on a line of its own took a line
#: per row, its right half empty, and the four rows no longer fit the owner's
#: window). The facts start on one edge, the keys ("In effect", "Set to") in
#: one 8em column. **Set to** wraps whole groups, by the room the panel gives
#: it, never by the window: the box keeps room for a path (16em, or its whole
#: line when that is narrower); Check and Save are one group (`shep-grp`) that
#: moves to the next line together, at the box's end, never Save alone (at 390
#: x 844 Save wrapped alone and Check stood centred in a row three lines
#: tall). **The rows share their columns** (the sweep's L7, C3: each box took
#: what Create and its suggestion left on its line, so the boxes and their
#: Check and Save stood at a different x in every row): where the Set-to bar
#: holds a box of `SET_BOX` and its buttons (`SET_WIDE`, the bar's own width,
#: a container query) every box is `SET_BOX` wide and Create, with its
#: suggestion under it where the line is short, follows Save on the line while
#: its button fits, else takes a line of its own; in a narrower bar every box
#: takes the line its buttons leave and Create a line of its own under it.
#: Either way the box, Check, Save and Create start at one x in every row, and
#: "In effect"'s value and the box share their left edge (the keys one 8em
#: column at 14 px, 12 px before the value). **The head's facts hold a path
#: at a readable width** (LOOK2-2: at 1550 x 830 the path in effect was 159 px
#: wide, 8 lines a row): while a row shows a path in effect, every row's facts
#: take at least 28em beside the name or go under it, so the rows keep one
#: form; the name column is 50ch, so at the owner's window the facts beside it
#: hold a path in 3 lines and the four rows stay in view. The panel is at least
#: about 700 px wide at the display floor, so the key always stands beside its
#: box (the query that put it above its box served windows under the floor and
#: is retired with it). The evidence folders' box, a path per line, takes its whole line there: its
#: hint ("One folder path on the server per line", 250.5 px in the dashboard's
#: type) is longer than 16em. A read-only row's short reason (`ROW_LOCKED`)
#: is a whole group of the bar after Check and Save (the owner's decision of
#: 2026-11-19: beside each row's buttons): beside Save where the line holds
#: it, so it adds no line there (the sweep's L7 check, D6: as the bar's note
#: under them it made each read-only row a line taller and the panel's last
#: line fell under its body's edge at 1850 x 960), else a line of its own on
#: the box's edge (K4: bound into Check and Save's group, it took them off the
#: box's line wherever the three did not fit beside the box, from 1550 x 830
#: down, each read-only row 17 px taller than before and shaped unlike an
#: operator's), its words on the buttons' baseline beside a box of two lines
#: too; Create then
#: takes a line of its own on the box's edge: the bar's buttons, Create
#: included, are off for the reason in their bar (C5: Create's lock had no
#: reason beside or under it).
#: Create and its suggestion share their first baseline (a suggestion that
#: wrapped pushed a centred Create down; the recheck of d4209c4); the
#: suggestion asks for its whole path's width (a basis of its content's width,
#: not a Gradio block's 100%): beside Create where the row holds that, else on
#: a line of its own under Create, breaking only before its own separators
#: where a path is longer than the row (OPS-RECHECK-DL1: given a basis of 0,
#: "~/SHEPHERD-data/workspaces" broke inside its words over 4 lines at 1366 x
#: 768). A path is a token (`renderers.token_html`): it breaks only before its
#: own separators, never after a hyphen (SW-LK7: an in-effect path read
#: hyphenated at every size).
#: **A set row's Change…** (the owner's "Set to / Change to" decision of 2026-11-24): while it
#: is not expanded the row's Set-to bar is not drawn and its key cell takes the line, Change… on
#: the boxes' column (past the keys' 8em column and the gap), a text button as the panel's
#: other disclosures are ("Advanced: source ›"), so a set row is shorter than an open one;
#: expanded, the bar is an unset row's, "Change to" its key and Cancel after Save. A read-only
#: set row draws its locked Change… and its phrase there.
#: **A set row's actions share the line its facts end on** (the owner's request of 2026-11-25:
#: "Change…" on the left stood a line under "Advanced: source ›" on the right, though the two
#: parts stand side by side): while the row's facts stand beside its name (`BESIDE`, measured
#: by `CHANGE_JS` as the head wraps) and its Change… is closed, the row is one grid of the head's
#: two columns (the name column as wide as the head drew it, `NAME_WIDTH`, and the gap): the
#: name beside what is in effect, then one line for both parts' last lines, Change… (with a
#: read-only row's phrase, under it on the box's edge where the column cannot hold both) on
#: the boxes' column and "Advanced: source ›" on the facts' edge, on one baseline, under
#: whichever part above them is taller; the outcome line under both. Where the facts go under
#: the name, and while Change… is open, the row is drawn as before.
#: **The name column's width** (the row's name and what needs it): it grows from `NAME_LEAST` to
#: its measure, `NAME_MOST`, as the head's facts beside it take the rest. Whether the facts go
#: beside the name or under it is judged from the least width (a basis of the most made the
#: facts' two short lines, "In effect ○ Not set" and "Advanced: source ›", drop under the name
#: at 1366 x 768 and at the display floor for want of 29 px, each row two lines taller and
#: the panel's edge falling under the "Import folder" heading: E1 at the head 9adb3c4). The
#: least is about 45 characters of its 14 px words a line, the measure's floor. The facts are
#: never narrower than what they cannot break ("Set when the service started" is one unit: the
#: value column's least is its words' least, `min-content`), so where that does not fit beside
#: the name they go under it (read-only at 1280 x 720 they ran 9 px past the panel's edge).
NAME_LEAST = "40ch"
NAME_MOST = "50ch"
#: A Set-to box's width where its bar holds it and its buttons (`SET_WIDE`): one for every row.
SET_BOX = "30em"
#: The Set-to bar's width (its container's) from which every box is `SET_BOX` wide: the box and
#: Check and Save (about 11em) on one line; under it each box takes what its buttons leave.
SET_WIDE = "42em"
#: A set row with Change… closed while its facts stand beside its name: its one grid.
_SHARED = f'{_PAGE} .shep-dl-row[{BESIDE}]:has(.shep-dl-change:not([aria-expanded="true"]))'
PAGE_CSS = f"""
{_PAGE} #shep-dl-bd {{ padding: 4px 16px 12px !important; gap: 0 !important; }}
{_PAGE} #shep-dl-top {{ gap: 8px !important; align-items: center; flex-wrap: wrap; }}
{_PAGE} #shep-dl-top > .shep-html {{ flex: 1 1 320px !important; min-width: 0 !important; }}
{_PAGE} #shep-dl-top > button {{ flex: none !important; min-width: max-content !important; }}
{_PAGE} #shep-dl-top:not(:has(.shep-dl-notice, .shep-reason, button:not([hidden]))) {{ display: none !important; }}
{_PAGE} .shep-dl-notice {{ display: flex; flex-direction: column; gap: 2px; padding: 8px 0; }}
{_PAGE} .shep-dl-notice p {{ margin: 0; }}
{_PAGE} .shep-dl-error {{ border: 1px solid var(--shep-red-200); background: var(--shep-red-50); border-radius: 8px; padding: 8px 12px; }}
{_PAGE} .shep-dl-row {{ gap: 6px 16px !important; padding: 10px 0 !important; border-top: 1px solid var(--shep-border); flex-wrap: wrap; align-items: flex-start; }}
{_PAGE} #shep-dl-top + .shep-dl-row, {_PAGE} .shep-dl-row:first-of-type {{ border-top: 0; }}
{_PAGE} .shep-dl-row > :is(.shep-dl-head, .shep-dl-main) {{ flex: 1 1 100% !important; min-width: 0 !important; }}
{_PAGE} .shep-dl-head {{ gap: 6px 24px !important; flex-wrap: wrap; align-items: flex-start; }}
{_PAGE} .shep-dl-head > .shep-dl-left {{ flex: 1 1 {NAME_LEAST} !important; max-width: {NAME_MOST} !important; min-width: min(16em, 100%) !important; }}
{_PAGE} .shep-dl-head > .shep-dl-facts {{ flex: 1 1 18em !important; min-width: min-content !important; gap: 4px !important; }}
{_PAGE} #shep-dl-bd:has(.shep-dl-facts .shep-dl-path) .shep-dl-head > .shep-dl-facts {{ min-width: min(28em, 100%) !important; }}
{_PAGE} .shep-dl-name h3 {{ margin: 0 0 var(--shep-space-head-below); font-size: 15px; font-weight: 600; }}
{_PAGE} .shep-dl-name p {{ margin: 0; }}
{_SHARED} {{ display: grid !important; grid-template-columns: var({NAME_WIDTH}) minmax(0, 1fr); gap: 4px 24px !important; align-items: start; }}
{_SHARED} :is(.shep-dl-head, .shep-dl-facts, .shep-dl-main) {{ display: contents !important; }}
{_SHARED} .shep-dl-head > .shep-dl-left {{ grid-area: 1 / 1; }}
{_SHARED} [id^="dl_state_"] {{ grid-area: 1 / 2; }}
{_SHARED} .shep-dl-set {{ grid-area: 2 / 1; align-self: baseline; }}
{_SHARED} [id^="dl_adv_"] {{ grid-area: 2 / 2; align-self: baseline; }}
{_SHARED} [id^="dl_result_"] {{ grid-area: 3 / 1 / auto / -1; margin-top: 2px; }}
{_PAGE} .shep-dl-main {{ gap: 6px !important; }}
{_PAGE} .shep-dl-set {{ gap: 8px 12px !important; align-items: baseline; flex-wrap: wrap; }}
{_PAGE} .shep-dl-key {{ font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-dl-set:has(.shep-dl-change:not([aria-expanded="true"])) > .shep-dl-field {{ display: none !important; }}
{_PAGE} .shep-dl-set:has(.shep-dl-change:not([aria-expanded="true"])) > .block.shep-dl-keycell {{ flex: 1 1 100% !important; }}
{_PAGE} .shep-dl-chg {{ display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; padding-inline-start: calc(8em + 12px); }}
{_PAGE} .shep-dl-chg:has(> .shep-dl-change[aria-expanded="true"]) {{ padding-inline-start: 0; }}
{_PAGE} button.shep-dl-change {{ flex: none; display: inline-flex; align-items: center; min-height: 24px; margin: 0; padding: 0; border: 0; background: none; font: inherit; font-size: 15px; color: var(--shep-accent-text); cursor: pointer; white-space: nowrap; }}
{_PAGE} button.shep-dl-change:hover {{ text-decoration: underline; }}
{_PAGE} button.shep-dl-change[aria-expanded="true"] {{ display: none; }}
{_PAGE} button.shep-dl-change:disabled {{ color: var(--shep-disabled-text); cursor: default; text-decoration: none; }}
{_PAGE} .shep-dl-set > .block.shep-dl-keycell {{ flex: 0 0 8em !important; min-width: 8em !important; font-size: 14px; }}
{_PAGE} .shep-dl-set > .shep-dl-field {{ flex: 1 1 16em !important; min-width: min(16em, 100%) !important; gap: 8px !important; align-items: flex-start !important; flex-wrap: wrap; container-type: inline-size; }}
{_PAGE} .shep-dl-field > .block {{ flex: 1 1 16em !important; min-width: min(16em, 100%) !important; margin-inline: 0 !important; }}
{_PAGE} .shep-dl-field > .shep-dl-acts {{ flex: none !important; width: auto !important; gap: 8px !important; flex-wrap: nowrap !important; }}
{_PAGE} .shep-dl-set input, {_PAGE} .shep-dl-set textarea {{ border: 1px solid var(--shep-control-border) !important; border-radius: 6px !important; padding: 8px 12px !important; font-size: 15px; }}
{_PAGE} .shep-dl-set input:focus-visible, {_PAGE} .shep-dl-set textarea:focus-visible {{ outline: 2px solid var(--shep-accent-600) !important; outline-offset: 1px; }}
{_PAGE} .shep-dl-set input:disabled, {_PAGE} .shep-dl-set textarea:disabled {{ background: var(--shep-grey-50) !important; border-color: var(--shep-grey-200) !important; }}
{_PAGE} .shep-dl-acts > button, {_PAGE} .shep-dl-create > button {{ flex: none !important; min-width: max-content !important; }}
{_PAGE} .shep-dl-field > .shep-dl-create {{ flex: 1 1 100% !important; width: auto !important; min-width: auto !important; max-width: 100%; gap: 8px !important; align-items: baseline !important; flex-wrap: wrap !important; }}
@container (min-width: {SET_WIDE}) {{
  {_PAGE} .shep-dl-field > .block {{ flex: 0 1 {SET_BOX} !important; }}
  {_PAGE} .shep-dl-field > .shep-dl-create {{ flex: 1 1 0% !important; }}
}}
{_PAGE} .shep-dl-create > [id^="dl_suggest_"] {{ flex: 1 1 max-content !important; min-width: 0 !important; }}
{_PAGE} .shep-dl-create:not(:has(button:not([hidden]))) {{ display: none !important; }}
{_PAGE} .shep-dl-field > .block[id^="dl_lock_"] {{ flex: none !important; width: auto !important; min-width: max-content !important; }}
{_PAGE} .shep-dl-field > :is(.shep-dl-acts, .block[id^="dl_lock_"]) {{ align-self: baseline !important; }}
{_PAGE} .shep-dl-field:has([id^="dl_lock_"] .shep-reason) > .shep-dl-create {{ flex: 1 1 100% !important; }}
{_PAGE} [id^="dl_lock_"] .shep-reason {{ white-space: nowrap; }}
{_PAGE} .shep-dl-kv {{ gap: 4px 12px; grid-template-columns: max-content minmax(min-content, 1fr); }}
{_PAGE} .shep-dl-kv > dt {{ width: 8em; }}
{_PAGE} .shep-dl-v {{ margin-right: 8px; }}
{_PAGE} .shep-dl-src {{ font-size: 14px; color: var(--shep-text-2); white-space: nowrap; }}
{_PAGE} .shep-dl-problem {{ color: var(--shep-text-2); font-size: 14px; }}
{_PAGE} .shep-dl-adv {{ font-size: 14px; margin-top: 2px; }}
{_PAGE} .shep-dl-holds {{ margin: 2px 0 0; font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} [id^="dl_result_"]:has(.shep-html:empty), {_PAGE} #dl_status:has(.shep-html:empty),
{_PAGE} [id^="dl_suggest_"]:has(.shep-html:empty),
{_PAGE} [id^="dl_lock_"]:has(.shep-html:empty) {{ display: none !important; }}
{_PAGE} #dl_box_{KEY_TRAINING_EVIDENCE_DIRS} {{ min-width: min(100%, 22em) !important; }}
"""


__all__ = ["APPLY_LABEL", "CANCEL_LABEL", "CHANGE_JS", "CHANGE_KEY", "CHANGE_LABEL",
           "CHECK_LABEL", "CREATE_LABEL", "INTRO", "NOTHING_SET", "NOTHING_SET_DETAILS",
           "NOTHING_SET_THEN", "SET_KEY",
           "OVERRIDDEN",
           "PAGE_CSS", "PANEL_PARTS", "PATH_HIDDEN", "PRIVATE_KEYS", "REPAIR_LABEL",
           "NAME_LEAST", "NAME_MOST",
           "ROW_LOCKED", "ROW_PARTS", "SAVE_LABEL", "SET_BOX", "SET_WIDE", "SOURCE_WORDS",
           "BESIDE", "HOLDS", "NAME_WIDTH", "UNKNOWN", "UNREADABLE",
           "box_value", "changeable", "check_words", "create_data_locations_panel", "create_folder",
           "key_html", "row_controls",
           "effect_html", "holds_words", "name_html", "notice_html", "reason_line", "state_html",
           "where_html"]
