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
  holds: "2 workspaces") and **Save**; for a required folder that is not set,
  **Create suggested folder**, naming it (``~/SHEPHERD-data/workspaces``:
  `deployment_settings.suggested_folder`, the one suggestion the not-set
  words name too), created only on that click (0700 for the registry and the
  import folder, which hold patient-level data); never a guessed old folder
  or a slot's.
  The variable is named only under "Advanced: source ›".
- **A box is the reader's**: it is set when the panel first opens and when
  its own row's Save succeeds (`box_value`: the evidence folders' saved list,
  else empty); every other act and answer, another row's Check or Save, a
  refusal, Settings shown again, leaves what the reader typed (OPS-C-R5).
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
  disabled with the lock, and one reason line in D8's words says why
  (`chips.operator_only`).

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
from src.webui.components.chips import LOCK, gated, operator_only
from src.webui.operator_access import operator_access
from src.webui.shell import SETTINGS as SETTINGS_PAGE
from src.webui.shell import plain_html, tab_elem_id

#: The panel's intro: where these folders are.
INTRO = ("These are folders on the server machine: this computer when SHEPHERD-Advanced was "
         "started from its desktop.")
#: A first start: nothing is set, and nothing is blocked by it.
NOTHING_SET = ("Nothing is set yet. Each folder is needed only by what its row names; "
               "Settings and every other page open as usual.")
#: Each source in plain words (`server_settings.SOURCE_*`).
SOURCE_WORDS = {
    server_settings.SOURCE_SAVED: "Saved in Settings",
    server_settings.SOURCE_ENVIRONMENT: "Set when the service started",
}
#: A saved value the environment overrides.
OVERRIDDEN = "Saved, applies when the start-up setting is removed"
#: A path a read-only viewer is not shown.
PATH_HIDDEN = "Set (path hidden)"
#: The read-only reason's second half (the first is D8's, `operator_only`).
PATHS_OPERATOR_ONLY = "Folder paths are shown to operator controls only."
#: The unreadable file's notice (Save is off until it is replaced: said here once).
UNREADABLE = ("Settings file cannot be read: the folders saved in it are unknown and not used; "
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
CREATE_LABEL = "Create suggested folder"
APPLY_LABEL = "Apply"
REPAIR_LABEL = "Replace the unreadable file (a copy is kept)"
#: What Replace does, said beside it.
REPAIR_THEN = ("Replace renames it beside itself and keeps it; then choose the folders here "
               "again.")

#: The order of the per-row outputs (`_panel`).
ROW_PARTS = ("state", "key", "box", "check", "save", "create", "suggest", "result", "adv")
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
    return "<br>".join(f'<code class="shep-dl-path">{r.e(v)}</code>' for v in values)


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
    if item.value is not None:
        effect = _value_html(item.value) if shown else r.e(PATH_HIDDEN)
    elif item.configured is not None and item.problem:
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
    rows.append(("In effect", " ".join(part for part in (
        f'<span class="shep-dl-v">{effect}</span>', _source_html(item)) if part)))
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
        ("Order", r.e("A start-up setting first, then the value saved here; nothing else")),
    ], classes="shep-dl-kv")
    return (f'<details class="shep-more shep-dl-adv"><summary>Where these are kept ›</summary>'
            f"{rows}</details>")


# ---------------------------------------------------------------------------
# The panel as one read draws it
# ---------------------------------------------------------------------------
def reason_line(text: str) -> str:
    """**A refusal's line beside what it turns off** on Settings (D8's
    reason, the lock before it): one per panel, in this panel above its rows
    (each row's box, Check and Save), in Runtime settings beside its
    actions, where the line above its two fields says where it is."""
    return (f'<div class="shep-reason"><span aria-hidden="true">{LOCK}</span>'
            f"<span>{r.e(text)}</span></div>")


def notice_html(view: DeploymentView, access: Any) -> str:
    """The panel's one notice: read-only (D8's reason), an unreadable file,
    a pending change and why, or a first start; nothing otherwise."""
    shown = access.allowed
    if not shown:
        return reason_line(f"{operator_only(access.condition)}. {PATHS_OPERATOR_ONLY}")
    if view.file.problem:
        then = (REPAIR_THEN if view.file.repairable else
                "It cannot be opened: correct its permissions or the disk where it is; it is "
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
        return f'<div class="shep-dl-notice">{r.status_mark("neutral", NOTHING_SET)}</div>'
    return ""


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
    key_words = "Set to" if item.value is None and item.configured is None else "Change to"
    if item.key == KEY_TRAINING_EVIDENCE_DIRS:
        key_words = "Set to"
    box: Dict[str, Any] = {"interactive": allowed}
    if not allowed:
        box["value"] = ""
    elif fill:
        box["value"] = box_value(item)
    return [
        gr.update(value=state_html(item, view, shown=shown)),
        gr.update(value=f'<span class="shep-dl-key">{r.e(key_words)}</span>'),
        gr.update(**box),
        gr.update(**gated(CHECK_LABEL, allowed)),
        gr.update(**gated(SAVE_LABEL, allowed, interactive=readable)),
        gr.update(visible=creatable, **gated(CREATE_LABEL, allowed)),
        gr.update(value=(f'<code class="shep-dl-path"{_title(suggestion, shown)}>'
                         f"{r.e(_tilde(suggestion))}</code>") if creatable else ""),
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


def holds_words(key: str, resolved: Any) -> str:
    """What a checked folder holds, read by the server's own lists with that
    folder alone (an explicit mapping: no saved file is read)."""
    from src.api.services import server_lists

    if key == KEY_TRAINING_EVIDENCE_DIRS:
        return _count(len(resolved), "folder", "folders")
    variable = server_settings.SAVED_VARIABLES[key]
    environ = {variable: str(resolved)}
    if key == KEY_WORKSPACES_ROOT:
        listing = server_lists.list_workspaces(environ=environ)
        what = (_count(len(listing.items), "workspace", "workspaces") if listing.items else
                "no workspace yet; Datasets lists each one built in it")
    elif key == KEY_REGISTRY:
        listing = server_lists.list_cohort_versions(environ=environ)
        what = (_count(len(listing.items), "cohort version", "cohort versions")
                if listing.items else "no cohort version yet")
    else:
        listing = server_lists.list_intake_files(environ=environ)
        what = (_count(len(listing.items), "import file", "import files") if listing.items
                else "no import file yet")
    if listing.problem and not listing.items:
        return f"{what} ({listing.problem})"
    return what


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
    (`server_settings.check_data_location`), as one status line."""
    value = _value_of(key, text)
    if value is None:
        return r.status_mark("neutral", "Nothing to check: type a folder path first")
    try:
        resolved, note = server_settings.check_data_location(key, value)
    except ValueError as exc:
        return r.status_mark("error", f"Not usable: {exc}")
    shown = (", ".join(str(p) for p in resolved) if isinstance(resolved, tuple)
             else str(resolved))
    said = r.status_mark("ok", "Usable") + f" <code>{r.e(shown)}</code>"
    said += f' <span class="shep-muted">({r.e(holds_words(key, resolved))})</span>'
    if note:
        said += "<br>" + r.status_mark("warn", note)
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
            return r.status_mark("neutral", "Saved. The start-up setting stays in effect; the "
                                            "saved value applies at a start without it")
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
    with gr.Column(elem_classes=["shep-panel", "shep-dl"], elem_id="shep-dl"):
        plain_html('<div class="shep-panel-hd"><h2>Data locations</h2>'
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
                    plain_html(name_html(key), elem_classes=["shep-dl-left"])
                    with gr.Column(elem_classes=["shep-dl-main"]):
                        row["state"] = plain_html("", elem_id=f"dl_state_{key}")
                        with gr.Row(elem_classes=["shep-dl-set"]):
                            row["key"] = plain_html('<span class="shep-dl-key">Set to</span>',
                                                    elem_classes=["shep-dl-keycell"])
                            evidence = key == KEY_TRAINING_EVIDENCE_DIRS
                            row["box"] = gr.Textbox(
                                label=f"{deployment_view.FIELD_WORDS[key][0]}: folder path",
                                show_label=False, container=False, interactive=False,
                                placeholder=EVIDENCE_PLACEHOLDER if evidence else PLACEHOLDER,
                                lines=2 if evidence else 1, max_lines=6 if evidence else 1,
                                elem_id=f"dl_box_{key}")
                            row["check"] = gr.Button(CHECK_LABEL, interactive=False, scale=0,
                                                     elem_id=f"dl_check_{key}")
                            row["save"] = gr.Button(SAVE_LABEL, interactive=False, scale=0,
                                                    elem_id=f"dl_save_{key}")
                        with gr.Row(elem_classes=["shep-dl-create"]):
                            row["create"] = gr.Button(CREATE_LABEL, visible=False,
                                                      interactive=False, scale=0,
                                                      elem_id=f"dl_create_{key}")
                            row["suggest"] = plain_html("", elem_id=f"dl_suggest_{key}")
                        row["result"] = plain_html("", elem_id=f"dl_result_{key}")
                        row["adv"] = plain_html("", elem_id=f"dl_adv_{key}")
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
    parts["apply"].click(fn=_on_apply_saved, outputs=outputs,
                         show_progress="hidden").then(**ctx.chip_refresh)
    parts["repair"].click(fn=_on_replace_unreadable, outputs=outputs,
                          show_progress="hidden").then(**ctx.chip_refresh)
    return {"rows": rows, **parts, "outputs": outputs}


_PAGE = "#" + tab_elem_id(SETTINGS_PAGE)

#: The panel's style sheet (listed with the Settings page's, `runtime_settings.PAGE_CSS`):
#: a name column and a facts column per row, the box as tall as its buttons,
#: rows divided by one rule; narrow, the name goes above, and a box keeps
#: room for a path (16em, or its whole row when that is narrower): where its
#: row cannot hold it beside its key and buttons, it and then the buttons
#: wrap (a phone's column squeezed it to 40 px; verification of fcc3dea,
#: V-RA-1). The evidence folders' box, a path per line, takes its whole row:
#: its hint ("One folder path on the server per line", 250.5 px in the
#: dashboard's type) is longer than 16em leaves it (215 px at 390 x 844).
PAGE_CSS = f"""
{_PAGE} #shep-dl-bd {{ padding: 4px 16px 12px !important; gap: 0 !important; }}
{_PAGE} #shep-dl-top {{ gap: 8px !important; align-items: center; flex-wrap: wrap; }}
{_PAGE} #shep-dl-top > .shep-html {{ flex: 1 1 320px !important; min-width: 0 !important; }}
{_PAGE} #shep-dl-top > button {{ flex: none !important; min-width: max-content !important; }}
{_PAGE} #shep-dl-top:not(:has(.shep-dl-notice, .shep-reason, button:not([hidden]))) {{ display: none !important; }}
{_PAGE} .shep-dl-notice {{ display: flex; flex-direction: column; gap: 2px; padding: 8px 0; }}
{_PAGE} .shep-dl-notice p {{ margin: 0; }}
{_PAGE} .shep-dl-error {{ border: 1px solid var(--shep-red-200); background: var(--shep-red-50); border-radius: 8px; padding: 8px 12px; }}
{_PAGE} .shep-dl-row {{ gap: 16px !important; padding: 10px 0 !important; border-top: 1px solid var(--shep-border); flex-wrap: wrap; align-items: flex-start; }}
{_PAGE} #shep-dl-top + .shep-dl-row, {_PAGE} .shep-dl-row:first-of-type {{ border-top: 0; }}
{_PAGE} .shep-dl-left {{ flex: 0 1 260px !important; min-width: 200px !important; }}
{_PAGE} .shep-dl-name h3 {{ margin: 0 0 2px; font-size: 15px; font-weight: 600; }}
{_PAGE} .shep-dl-name p {{ margin: 0; }}
{_PAGE} .shep-dl-main {{ flex: 1 1 420px !important; min-width: 0 !important; gap: 6px !important; }}
{_PAGE} .shep-dl-set {{ gap: 8px !important; align-items: center; flex-wrap: nowrap; }}
{_PAGE} .shep-dl-key {{ font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-dl-set > .block:not(.shep-dl-keycell) {{ flex: 1 1 auto !important; min-width: 0 !important; }}
{_PAGE} .shep-dl-set > .block.shep-dl-keycell {{ flex: 0 0 8em !important; min-width: 8em !important; }}
{_PAGE} .shep-dl-set input, {_PAGE} .shep-dl-set textarea {{ border: 1px solid var(--shep-control-border) !important; border-radius: 6px !important; padding: 8px 12px !important; font-size: 15px; }}
{_PAGE} .shep-dl-set input:focus-visible, {_PAGE} .shep-dl-set textarea:focus-visible {{ outline: 2px solid var(--shep-accent-600) !important; outline-offset: 1px; }}
{_PAGE} .shep-dl-set input:disabled, {_PAGE} .shep-dl-set textarea:disabled {{ background: var(--shep-grey-50) !important; border-color: var(--shep-grey-200) !important; }}
{_PAGE} .shep-dl-set > button, {_PAGE} .shep-dl-create > button {{ flex: none !important; min-width: max-content !important; }}
{_PAGE} .shep-dl-create {{ gap: 8px !important; align-items: center; padding-left: calc(8em + 8px); }}
{_PAGE} .shep-dl-create:not(:has(button:not([hidden]))) {{ display: none !important; }}
{_PAGE} .shep-dl-kv {{ gap: 4px 12px; }}
{_PAGE} .shep-dl-kv > dt {{ width: 8em; }}
{_PAGE} .shep-dl-v {{ margin-right: 8px; }}
{_PAGE} .shep-dl-src {{ font-size: 14px; color: var(--shep-text-2); white-space: nowrap; }}
{_PAGE} .shep-dl-path {{ overflow-wrap: anywhere; }}
{_PAGE} .shep-dl-problem {{ color: var(--shep-text-2); font-size: 14px; }}
{_PAGE} .shep-dl-adv {{ font-size: 14px; margin-top: 2px; }}
{_PAGE} [id^="dl_result_"]:has(.html-container:empty), {_PAGE} #dl_status:has(.html-container:empty),
{_PAGE} [id^="dl_suggest_"]:has(.html-container:empty) {{ display: none !important; }}
@media (max-width: 899px) {{
  {_PAGE} .shep-dl-left {{ flex-basis: 100% !important; }}
  {_PAGE} .shep-dl-set {{ flex-wrap: wrap; }}
  {_PAGE} .shep-dl-set > .block:not(.shep-dl-keycell) {{ flex: 1 1 16em !important; min-width: min(16em, 100%) !important; }}
  {_PAGE} #dl_box_{KEY_TRAINING_EVIDENCE_DIRS} {{ flex-basis: 100% !important; }}
}}
"""


__all__ = ["APPLY_LABEL", "CHECK_LABEL", "CREATE_LABEL", "INTRO", "NOTHING_SET", "OVERRIDDEN",
           "PAGE_CSS", "PANEL_PARTS", "PATH_HIDDEN", "PRIVATE_KEYS", "REPAIR_LABEL",
           "ROW_PARTS", "SAVE_LABEL", "SOURCE_WORDS", "UNKNOWN", "UNREADABLE", "box_value",
           "check_words", "create_data_locations_panel", "create_folder", "holds_words",
           "name_html", "notice_html", "reason_line", "state_html", "where_html"]
