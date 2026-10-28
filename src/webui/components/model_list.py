"""
The shared model list and the one Load (M5-2, UI-A).
====================================================
One component, two presets: Diagnosis' "Change model…" drawer uses the
``clinical`` preset (UI-B), Research › Models the ``research`` preset (UI-D).
Both list the same checkpoints and load through the same handler, so the two
views cannot drift apart (PLAN §6, "M5's boundaries": one shared model list,
one gated Load, no second implementation).

**The list** is WP1's checkpoint list for a workspace the server lists
(`server_lists.list_checkpoints`). Listing opens every checkpoint of the
workspace (SPEC_4 C3), so it is an operator control: D8's helper is asked on the
triggering request first, and a read-only request gets the reason and no list.
Each row's recorded tests are a state read from the ledger beside the
checkpoint (`read_ledger`, `records_for`) and, for contested, from the cohort
version's report store (`record_states.contest_states`, the one implementation
the API's route shares): never a number of a result. An
unreadable entry is greyed "metadata not readable · cannot be loaded". "In use"
is matched by the **full** model fingerprint the status serves and marks the
checkpoint only.

**The two presets** (`PRESETS`):

- ``clinical`` (the drawer): rows (`renderers.model_row_html`), "In use now"
  pinned on top, the latest 4 runs, then "Show older runs"; **models
  finetuned for a task only** (``Preset.rows``; README §5, "a base offers no
  diagnosis"), whatever view the page, or a direct call, gives;
- ``research`` (Models): a table (`renderers.model_table_row_html`: model,
  fingerprints, the Trainer's validation figure, recorded tests), every run,
  "In use (Diagnosis)" on its own row, not pinned; its listing carries the
  family view too (``listing["family"]``, `family_views`).

**A page shapes its view** through ``ModelList.view`` (a `gr.State` the page
sets; the list redraws on its ``.change``), never by drawing rows itself:
``{"task": "disease"}`` lists the rows of one filter (`TASK_FILTERS`: all,
base, disease, gene, patients_like_me, history, task_models; `matches`, on the
server's one predicate `server_lists.record_matches`); ``"keys"`` lists exactly
those rows instead (a family view's, its context rows included); ``"groups":
[{"title", "keys"}]`` draws the rows in those groups, in that order
(checkpoints in no group follow under ``"rest_title"``); ``"column"`` and
``"cells": {key: html}`` add one column the page fills (HTML it built with the
shared renderers, which escape); ``"note"`` is plain text beside the Load
button. Models orders the list this way (its groups "Ordered by …",
"Contested on this version · not ordered", "Not tested on this cohort version ·
newest first", its regime column; its families and their lineage column) and
sets ``ModelList.ordering`` too, so that a Load from it writes a usage history
entry. ``create_model_list(head=…, foot=…)`` places a page's own components (a
filter, links) in the list's header and footer rows.

**The Load**, in three steps, each a handler of its own:

1. "🔒 Load this model" (`request_load`): D8 on that request; **a row that is
   never loaded is refused before anything of its file is read**
   (`load_refusal`: a base offers no diagnosis, a file without a model spec is
   history and never run, a record this build refused is said with why; the
   research Load button is off for such a row, its reason beside it); the item
   is found in the server's list again (`resolve_checkpoint`, never a client
   path) and its bytes hashed again, so a list read earlier is caught (a file
   rewritten since is read again from the server's list and shown as it is
   now); one inline confirmation, "Reload the pipeline with <12 hex> for every
   session?", from → to: model, model fingerprint, trained on, serving
   workspace and graph digest, task and scoring mode (the one table,
   `core.types.TASK_SCORING_MODES`), SP table (a state: "not used by this
   task", "computed from the model's graph view"), synthetic flag, the changed
   rows marked;
2. "🔒 Reload with <new>" (`finish_load`): D8 again, on that request; **both
   columns read again**: what is served (the whole served identity,
   `served_key`) and what would be loaded (the file hashed again, the
   workspace's graph read again, the refusal checked again on what the file
   holds now). A change in either since the confirmation was shown is said
   with the new values, and nothing is reloaded until the operator confirms
   what is shown. Then WP2's one reload (`run_reload`, in process, on the event
   loop as the API route runs it) with the workspace, the checkpoint the
   server resolves from the item and the task its record names, **bound to the
   model fingerprint shown** (`expected_checkpoint_sha256`): the file
   can still change between this hash and the reload's own read, seconds later
   (the build reads the workspace's graph first), and the reload compares its
   read with the fingerprint before the bytes are parsed, so other bytes, or
   none, are refused as not the model shown, never served; a usage history
   entry (WP4
   `record_model_load`) only when the list was ordered by a cohort version's
   score; then the chips and the "served model changed" trigger
   (`PageContext.served_model_changed`), set whenever any part of what is
   served changed;
3. "Keep <old>" closes the confirmation; nothing is done.

**While the reload runs the confirmation shows only its status and Close**
(`loading_buttons`, before the reload; the mockup's footer): Reload goes, at
once in the browser too, so a second click cannot ask for a second reload, and
Keep becomes Close, since closing does not stop it. **Close works at once**,
although nothing on the server answers meanwhile: the page itself hides the
confirmation (or the page's drawer, with its ×) and, once the reload has
answered, sends the close to the server in order after that answer
(`load_guard`, `start_wait_js`, `load_answered_js`); a close sent while the
reload held the server would be handled after it in no fixed order with its
answer, and could leave the box open and empty. A closed load's outcome is not
shown again: what is served then is on the page (the card, the list's "In use").

**The page waits 180 s.** The reload holds the event loop (it never runs beside
a diagnosis, BACKLOG 28), so nothing answers meanwhile: the page itself counts
(a script built for this list's own confirmation, `start_wait_js`) and after
180 s says "The page stopped waiting after 180 s; the server may still finish
the reload." with "Read again" (a status read). When the reload's answer
arrives later, it says so ("The reload finished after the page stopped
waiting.").

**Per browser session**: the listing, the selection, the view and the
confirmation are `gr.State`; nothing here is module state.

Module: src/webui/components/model_list.py
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import gradio as gr

# D5's state of a record key: one implementation, shared with the API's route
# (BACKLOG 36).
from src.api.services.record_states import CohortVersions, contest_states, key_text
from src.api.services.server_lists import record_matches
from src.core.types import SCORING_MODE_TASKS, TASK_SCORING_MODES
from src.evaluation.caveats import TASK_NAMES
from src.webui.components import renderers as r
from src.webui.components.served_status import (
    read_status,
    served_checkpoint_sha,
    served_key,
    serving_workspace,
)
from src.webui.operator_access import operator_access
from src.webui.shell import plain_html

logger = logging.getLogger(__name__)

#: How long the page waits for a reload before it says it stopped waiting.
PAGE_WAIT_SECONDS = 180

#: What a confirmation says it replaces (the mockup's words).
REPLACES_NOTE = (
    "Replaces what every session is served; results already produced keep their own "
    "identity. Diagnoses wait while the pipeline reloads."
)
#: The note "In use" carries.
IN_USE_NOTE = "matched by the full model fingerprint (the checkpoint only, not the whole pipeline)"

#: Said when what is served changed between the confirmation and its confirm.
FROM_CHANGED = ("What every session is served changed since this was shown (another session "
                "reloaded): the From column is what is served now.")
#: Said when what would be loaded changed between the confirmation and its confirm.
TO_CHANGED = ("What would be loaded changed since this was shown: the To column is what the "
              "file and its workspace hold now.")
#: Ends either notice.
NOTHING_RELOADED = "Nothing was reloaded; check the rows again, then reload or keep."

#: The title of the checkpoints a page's groups leave out.
REST_TITLE = "Other checkpoints · newest first"

#: **The list's one filter vocabulary** (``view["task"]``, `matches`): every
#: row; the bases; one task's finetuned models; the files without a model spec
#: (history, never run); every model finetuned for a task.
TASK_FILTERS = ("all", "base", "disease", "gene", "patients_like_me", "history", "task_models")
#: Each filter as the count line ("· Disease only, of 9 listed") and the empty
#: state ("No disease model is listed in this workspace.") name it.
FILTER_WORDS: Dict[str, Tuple[str, str]] = {
    "all": ("All", "checkpoint"),
    "base": ("Base", "base"),
    **{task: (name, f"{name.lower()} model") for task, name in TASK_NAMES.items()},
    "history": ("History", "file without a model spec"),
    "task_models": ("Task models", "model finetuned for a task"),
}

#: Why a listed row is never loaded (`load_refusal`), said before anything of
#: its file is read: README §5 ("a base offers no diagnosis") and §4 (a file
#: without a model spec is listed as history and never run; K6 decision 1).
BASE_REFUSAL = "A base offers no diagnosis: load a model finetuned for a task."
HISTORY_REFUSAL = "No model spec: listed as history, never run."

#: What the confirmation's SP row says of a model whose task computes its
#: shortest-path term from its own graph view (the served identity's
#: ``computed`` state, `InferencePipeline.served_identity`), by scoring mode;
#: every other task model uses none ("not used by this task").
_SP_OF_MODE = {TASK_SCORING_MODES["gene"]: "computed"}


@dataclass(frozen=True)
class Preset:
    """What differs between the two views of the one list."""

    key: str
    #: Runs shown before "Show older runs" (None: every run).
    latest_runs: Optional[int]
    #: The words of the "In use" mark.
    in_use_label: str
    #: Whether the model in use is pinned on top ("In use now").
    pin_in_use: bool = True
    #: ``rows`` (the drawer's list) or ``table`` (Models' columns).
    layout: str = "rows"
    #: The rows this preset ever shows (a `TASK_FILTERS` name), whatever the
    #: view asks: the drawer lists models finetuned for a task only, so a view
    #: of None (a direct call) never lists a base or a history file there.
    rows: str = "all"
    #: Whether its listing carries the family view (`family_views`).
    family: bool = False


PRESETS: Dict[str, Preset] = {
    "clinical": Preset("clinical", 4, "In use", pin_in_use=True, layout="rows",
                       rows="task_models"),
    "research": Preset("research", None, "In use (Diagnosis)", pin_in_use=False, layout="table",
                       family=True),
}


# ---------------------------------------------------------------------------
# The listing (server side)
# ---------------------------------------------------------------------------
def item_key(item: Mapping[str, Any]) -> str:
    """A listed checkpoint's key in this workspace: architecture/run/file."""
    return f"{item.get('architecture')}/{item.get('run')}/{item.get('filename')}"


def ledger_records(workspace_dir: Path, items: Sequence[Mapping[str, Any]]
                   ) -> Dict[str, Dict[str, Any]]:
    """For each listed checkpoint with a digest, what the ledger beside it holds
    for its weights, read through the one ledger reader (`read_ledger`, once per
    directory) and `records_for`: ``state`` (``records``, ``none``,
    ``no_ledger`` or ``not_read``), the records as the ledger holds them (kept
    on the server, in the session's state; never sent to the browser as they
    are), ``problem`` (naming the file, never its directory) and ``ledger``
    (the ledger's resolved path, which only tells two ledgers apart).

    The checkpoint file is never opened: its path is the run layout's, resolved
    to find the directory its ledger is in, as registration writes it."""
    from src.api.routes.evaluations import _ledger_problem
    from src.evaluation.sidecar import LEDGER_FILENAME, read_ledger, records_for
    from src.utils.checkpoint_paths import RUN_CHECKPOINTS_DIRNAME, RUNS_DIRNAME

    workspace_dir = Path(workspace_dir).resolve()
    read: Dict[Path, Tuple[Optional[dict], Optional[str], bool]] = {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not loadable(item):
            continue
        key = item_key(item)
        file = (workspace_dir / RUNS_DIRNAME / str(item.get("architecture")) / str(item.get("run"))
                / RUN_CHECKPOINTS_DIRNAME / str(item.get("filename")))
        try:
            directory = file.resolve().parent
        except (OSError, RuntimeError):
            out[key] = {"state": "not_read", "records": [], "ledger": None,
                        "problem": "the checkpoint's directory cannot be resolved"}
            continue
        if directory != workspace_dir and workspace_dir not in directory.parents:
            out[key] = {"state": "not_read", "records": [], "ledger": None,
                        "problem": "the checkpoint resolves outside the workspace"}
            continue
        ledger_path = directory / LEDGER_FILENAME
        if directory not in read:
            try:
                ledger, digest = read_ledger(ledger_path)
                read[directory] = (ledger, None, digest is not None)
            except (OSError, ValueError) as exc:
                read[directory] = (None, _ledger_problem(exc, ledger_path), True)
        ledger, problem, present = read[directory]
        if problem is not None:
            out[key] = {"state": "not_read", "records": [], "ledger": str(ledger_path),
                        "problem": problem}
        elif not present:
            out[key] = {"state": "no_ledger", "records": [], "ledger": str(ledger_path),
                        "problem": None}
        else:
            records = records_for(ledger, item["sha256"])
            out[key] = {"state": "records" if records else "none", "records": records,
                        "ledger": str(ledger_path), "problem": None}
    return out


def recorded_tests(ledgers: Mapping[str, Mapping[str, Any]],
                   contest: Mapping[str, Mapping[str, Any]]) -> Dict[str, r.RecordedTests]:
    """Each checkpoint's recorded tests as a state, from what `ledger_records`
    and `contest_states` read: how many records, how many contested and how
    many whose contested state cannot be checked, counted apart (M5-3 review
    EVAL-3: one uncheckable record hid a known contested one)."""
    from src.evaluation.sidecar import record_key

    out: Dict[str, r.RecordedTests] = {}
    for key, entry in ledgers.items():
        if entry.get("state") != "records":
            out[key] = r.RecordedTests(entry.get("state") or "not_read",
                                       problem=entry.get("problem"))
            continue
        contested = unchecked = 0
        for record in entry.get("records") or []:
            try:
                found = contest.get(key_text(record_key(dict(record))))
            except ValueError:
                found = None
            state = (found or {}).get("state")
            if state == "contested":
                contested += 1
            elif state != "uncontested":
                unchecked += 1
        out[key] = r.RecordedTests("records", len(entry.get("records") or []), contested,
                                   unchecked=unchecked)
    return out


def read_recorded(workspace_dir: Path, items: Sequence[Mapping[str, Any]], *,
                  environ: Optional[Mapping[str, str]] = None
                  ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """``(ledgers, contest)`` for the listed checkpoints: the ledger beside each
    (`ledger_records`, read once per directory) and every record key's state in
    its cohort version's report store (`contest_states`, one scan per version).
    The one read behind the rows, Models and the Diagnosis card."""
    ledgers = ledger_records(workspace_dir, items)
    records = [rec for entry in ledgers.values() for rec in entry["records"]]
    contest = contest_states(records, CohortVersions.listed(environ)) if records else {}
    return ledgers, contest


def recorded_tests_for(workspace_dir: Path, items: Sequence[Mapping[str, Any]], *,
                       environ: Optional[Mapping[str, str]] = None) -> Dict[str, r.RecordedTests]:
    """Each listed checkpoint's recorded tests, as states (`read_recorded`,
    then `recorded_tests`). Never raises: a problem is a state."""
    return recorded_tests(*read_recorded(workspace_dir, items, environ=environ))


def empty_listing(problem: Optional[str] = None) -> Dict[str, Any]:
    return {"workspace": None, "items": [], "workspaces": [], "recorded": {}, "ledgers": {},
            "contest": {}, "served_sha": None, "serving": None, "problem": problem,
            "status_read": False}


#: The filters a family view is drawn for (`family_views`): the history group
#: is the whole view's (``all``), and a family holds no history file.
FAMILY_FILTERS = ("all", "base", *TASK_NAMES)


def family_views(workspace: str, checkpoints: Any, root: Any, *,
                 environ: Optional[Mapping[str, str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """The workspace's family view (`server_lists.list_family`) for each of
    `FAMILY_FILTERS`, **from the one listing** the list was drawn from
    (`checkpoints`, given as ``listing=``: nothing is listed again), as plain
    data: per row its key, group, family, depth, parent key and where the exact
    parent file is (``parent_state``, ``parent_workspace``, ``parent_problem``,
    redacted: a file's problem may name a server path) and whether it is a
    context row. A filter that kept nothing has no rows."""
    from src.api.services.server_lists import list_family
    from src.webui.components.workspaces import redact

    views: Dict[str, List[Dict[str, Any]]] = {}
    for name in FAMILY_FILTERS:
        record_filter = ({} if name == "all" else {"learning_stage": "base"} if name == "base"
                         else {"task": name})
        listed = list_family(workspace, listing=checkpoints, environ=environ, **record_filter)
        views[name] = [{
            "key": item_key(asdict(row.item)), "group": row.group, "family_id": row.family_id,
            "depth": row.depth, "parent": row.parent, "parent_state": row.parent_state,
            "parent_workspace": row.parent_workspace,
            "parent_problem": redact(row.parent_problem, root) if row.parent_problem else None,
            "context": row.context,
        } for row in listed.items]
    return views


def list_models(workspace: Any, request: Any, *, family: bool = False,
                environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The listing for one session: D8 on the triggering request first (listing
    opens checkpoints), then the server's lists and the status, as plain data
    for a `gr.State`. A refused request lists nothing and says why. With
    `family` (the research preset), the family view of the same listing too
    (``listing["family"]``, `family_views`)."""
    from src.api.services.server_lists import list_checkpoints
    from src.webui.components.workspaces import read_workspaces, redact

    access = operator_access(request)
    if not access.allowed:
        return {**empty_listing(access.refusal), "refused": True, "condition": access.condition}
    listed = read_workspaces(environ)
    rows, directories = listed["rows"], listed["directories"]
    status, status_problem = read_status()
    serving = serving_workspace(status, rows, directories=directories)
    names = [w["name"] for w in rows]
    chosen = workspace if workspace in names else (serving if serving in names else
                                                   (names[0] if names else None))
    listing = {**empty_listing(listed["problem"]), "workspaces": rows, "workspace": chosen,
               "served_sha": served_checkpoint_sha(status), "serving": serving,
               "status_problem": status_problem, "status_read": True}
    if chosen is None:
        return listing
    checkpoints = list_checkpoints(chosen, environ=environ)
    # No absolute server path in what the list shows (a workspace that is a
    # link out of the root names its target; a read's error may name a file):
    # the dashboard's one redaction, as Datasets says the same reasons.
    root = listed["root"]
    items = [asdict(i) for i in checkpoints.items]
    for item in items:
        for field in ("problem", "record_problem"):
            if item.get(field):
                item[field] = redact(item[field], root)
    listing.update(items=items, problem=(redact(checkpoints.problem, root)
                                         if checkpoints.problem else None))
    if family and checkpoints.available:
        listing["family"] = family_views(chosen, checkpoints, root, environ=environ)
    # The ledgers beside the checkpoints and their records' contested states,
    # read once for the rows and kept in the listing (server side, in the
    # session's state) for a page that shows them (Models).
    ledgers: Dict[str, Dict[str, Any]] = {}
    contest: Dict[str, Dict[str, Any]] = {}
    try:
        if chosen in directories:
            ledgers, contest = read_recorded(directories[chosen], items, environ=environ)
    except Exception:  # noqa: BLE001 - the recorded tests then read "not read"
        logger.exception("Recorded tests could not be read")
    listing["recorded"] = {k: v._asdict() for k, v in recorded_tests(ledgers, contest).items()}
    listing["ledgers"], listing["contest"] = ledgers, contest
    return listing


def _recorded(listing: Mapping[str, Any], key: str) -> Optional[r.RecordedTests]:
    raw = (listing.get("recorded") or {}).get(key)
    return r.RecordedTests(**raw) if isinstance(raw, Mapping) else None


def _runs_in_order(items: Sequence[Mapping[str, Any]]) -> List[Tuple[str, List[Mapping[str, Any]]]]:
    """Items grouped by run, in the listing's order (latest run first)."""
    runs: Dict[str, List[Mapping[str, Any]]] = {}
    for item in items:
        runs.setdefault(f"{item.get('architecture')}/{item.get('run')}", []).append(item)
    return list(runs.items())


def _view(view: Any) -> Mapping[str, Any]:
    return view if isinstance(view, Mapping) else {}


def matches(item: Mapping[str, Any], name: str) -> bool:
    """Whether a listed checkpoint is a row of the filter `name`
    (`TASK_FILTERS`), judged by its model record through the server's one
    predicate (`server_lists.record_matches`): a base, a model finetuned for
    that task, any model finetuned for a task, or a file without a model spec.
    A file whose record was refused or not read is a row of ``all`` only."""
    if name == "all":
        return True
    if name == "history":
        return item.get("record") == "history"
    if name == "base":
        return record_matches(item, learning_stage="base")
    if name == "task_models":
        return record_matches(item, learning_stage="finetune")
    return record_matches(item, task=name, learning_stage="finetune")


def task_filter(view: Any) -> str:
    """The filter a view names (``view["task"]``), ``all`` when it names none
    this list knows."""
    chosen = _view(view).get("task")
    return chosen if chosen in TASK_FILTERS else "all"


def _rows_filter(view: Any, preset: Optional[Preset]) -> str:
    """The filter the rows are said by: the view's, or the preset's own rule
    when the view asks for every row (the drawer: task models)."""
    chosen = task_filter(view)
    return preset.rows if chosen == "all" and preset is not None else chosen


def shown_items(listing: Mapping[str, Any], view: Any = None,
                preset: Optional[Preset] = None) -> List[Mapping[str, Any]]:
    """The listed checkpoints the view shows: its ``keys`` when it gives them
    (a family view's rows), else its filter's (``task``, `matches`); never a
    row the preset does not show (``Preset.rows``)."""
    items = list(listing.get("items") or [])
    if preset is not None and preset.rows != "all":
        items = [i for i in items if matches(i, preset.rows)]
    keys = _view(view).get("keys")
    if isinstance(keys, (list, tuple)):
        wanted = set(keys)
        return [i for i in items if item_key(i) in wanted]
    chosen = task_filter(view)
    return [i for i in items if matches(i, chosen)]


def refused_line(listing: Mapping[str, Any]) -> str:
    """Why a refused listing lists nothing, as every page says an operator
    control that is off for this request: "Operator controls only: <the
    condition D8 named, in short words>."."""
    from src.webui.components.chips import D8_SHORT

    return f"Operator controls only: {D8_SHORT.get(listing.get('condition'), 'request not readable')}."


def kept_selection(previous: Any, listing: Mapping[str, Any], key: Any,
                   view: Any = None, preset: Optional[Preset] = None) -> Optional[str]:
    """The selection a new listing keeps: `key` when the listing is of the same
    workspace as `previous` and still shows that checkpoint; else None."""
    if not key or not isinstance(previous, Mapping) or listing.get("refused") \
            or previous.get("workspace") != listing.get("workspace"):
        return None
    item = find_item(listing, key)
    return key if item is not None and item in shown_items(listing, view, preset) else None


def _groups(items: Sequence[Mapping[str, Any]], view: Any
            ) -> Optional[List[Tuple[str, List[Mapping[str, Any]]]]]:
    """The view's groups over the shown items, in its order, with the rest
    after them; None when the view has none."""
    groups = _view(view).get("groups")
    if not isinstance(groups, (list, tuple)):
        return None
    by_key = {item_key(i): i for i in items}
    placed: set = set()
    sections = []
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        members = []
        for key in group.get("keys") or ():
            if key in by_key and key not in placed:
                members.append(by_key[key])
                placed.add(key)
        sections.append((str(group.get("title") or ""), members))
    rest = [i for i in items if item_key(i) not in placed]
    if rest:
        sections.append((str(_view(view).get("rest_title") or REST_TITLE), rest))
    return sections


def _no_items_html(listing: Mapping[str, Any], view: Any, preset: Optional[Preset]) -> str:
    chosen = _rows_filter(view, preset)
    if listing.get("items") and chosen != "all":
        why = f"No {FILTER_WORDS[chosen][1]} is listed in this workspace."
    else:
        why = listing.get("problem") or "No checkpoint is listed in this workspace."
    return f'<div class="shep-reason">{r.e(why)}</div>'


def list_html(listing: Mapping[str, Any], selected: Optional[str], preset: Preset, *,
              show_older: bool = False, view: Any = None) -> str:
    """The list as the preset draws it: rows (the drawer: "In use now" pinned,
    runs newest first, the latest runs then "Show older runs") or a table
    (Models: every checkpoint, the model in use marked on its own row); the
    view's filter or rows, groups and extra column when it gives them."""
    if listing.get("refused"):
        # A listing is no act: the short line every page uses for an operator
        # control off for this request, not "Refused … Nothing was done."
        return f'<div class="shep-reason shep-reason-grey">{r.e(refused_line(listing))}</div>'
    items = shown_items(listing, view, preset)
    if not items:
        return _no_items_html(listing, view, preset)
    workspaces = listing.get("workspaces") or []
    digits = r.distinguishing_digits(i.get("sha256") for i in items)
    served = listing.get("served_sha")
    cells = _view(view).get("cells") if isinstance(_view(view).get("cells"), Mapping) else {}
    column = _view(view).get("column")

    def is_served(item: Mapping[str, Any]) -> bool:
        return bool(served) and item.get("sha256") == served

    pinned = [i for i in items if is_served(i)][:1] if preset.pin_in_use else []
    groups = _groups(items, view)
    if groups is not None:
        sections = groups
        hidden = 0
    else:
        runs = _runs_in_order(items)
        shown = runs if (preset.latest_runs is None or show_older) else runs[:preset.latest_runs]
        hidden = len(runs) - len(shown)
        if preset.layout == "table":
            sections = [("", [i for _, run_items in shown for i in run_items])]
        else:
            sections = []
            for run, run_items in shown:
                started = r.run_started(run_items[0].get("run"))
                sections.append((f"Run {run}" + (f" · started {started}" if started else ""),
                                 run_items))

    if preset.layout == "table":
        return _table_html(sections, pinned, listing, selected, preset, workspaces, digits,
                           served, cells, column, hidden)

    def row(item: Mapping[str, Any], *, in_use: bool) -> str:
        key = item_key(item)
        html_ = r.model_row_html(item, workspaces, recorded=_recorded(listing, key),
                                 in_use=in_use, selected=key == selected, digits=digits,
                                 key=key, in_use_label=preset.in_use_label)
        extra = cells.get(key)
        if extra:
            html_ = html_.replace("</button>", f'<span class="shep-row-sub">{extra}</span></button>', 1)
        return html_

    parts = []
    if pinned:
        parts.append(f'<div class="shep-list-hd"><b>In use now</b> · {r.e(IN_USE_NOTE)}</div>')
        parts.append(row(pinned[0], in_use=True))
    for title, members in sections:
        if title:
            parts.append(f'<div class="shep-list-hd"><b>{r.e(title)}</b></div>')
        for item in members:
            if pinned and is_served(item):
                parts.append(f'<div class="shep-list-note">epoch {r.e(item.get("epoch"))} · model '
                             f'fingerprint {r.e(r.short_hex(served, digits))} — in use, pinned '
                             'above</div>')
                continue
            parts.append(row(item, in_use=not pinned and is_served(item)))
    if hidden > 0:
        parts.append(f'<button type="button" class="shep-row shep-row-more" data-key="__older__">'
                     f'Show older runs ({hidden})</button>')
    return '<div class="shep-list" role="list">' + "".join(parts) + "</div>"


def _table_html(sections: Sequence[Tuple[str, Sequence[Mapping[str, Any]]]],
                pinned: Sequence[Mapping[str, Any]], listing: Mapping[str, Any],
                selected: Optional[str], preset: Preset, workspaces: Sequence[Any], digits: int,
                served: Optional[str], cells: Mapping[str, Any], column: Any, hidden: int) -> str:
    heads = list(r.MODEL_TABLE_COLUMNS)
    if column:
        heads.insert(3, str(column))
    span = len(heads)
    head = "".join(f'<th scope="col"{" class=shep-num" if h.startswith("Validation") else ""}>'
                   f'{r.e(h)}</th>' for h in heads)

    def row(item: Mapping[str, Any]) -> str:
        key = item_key(item)
        return r.model_table_row_html(
            item, workspaces, recorded=_recorded(listing, key),
            in_use=bool(served) and item.get("sha256") == served, selected=key == selected,
            digits=digits, key=key, in_use_label=preset.in_use_label,
            extra=(str(cells.get(key) or '<span class="shep-muted">—</span>') if column else None))

    body = []
    if pinned:
        body.append(f'<tr class="shep-grp"><th colspan="{span}" scope="rowgroup">In use now · '
                    f'{r.e(IN_USE_NOTE)}</th></tr>' + row(pinned[0]))
    for title, members in sections:
        if title:
            body.append(f'<tr class="shep-grp"><th colspan="{span}" scope="rowgroup">'
                        f'{r.e(title)}</th></tr>')
        body.extend(row(i) for i in members if not (pinned and i is pinned[0]))
    more = (f'<button type="button" class="shep-row shep-row-more" data-key="__older__">'
            f'Show older runs ({hidden})</button>' if hidden > 0 else "")
    return ('<div class="shep-tbl-wrap shep-mtbl-wrap"><table class="shep-tbl shep-mtbl" '
            f'aria-label="Checkpoints"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}'
            f'</tbody></table></div>{more}')


def count_line(listing: Mapping[str, Any], preset: Preset, show_older: bool,
               view: Any = None) -> str:
    if listing.get("refused") or not listing.get("items"):
        return ""
    items = shown_items(listing, view, preset)
    runs = _runs_in_order(items)
    of = ""
    chosen = _rows_filter(view, preset)
    if chosen != "all":
        of = f" · {FILTER_WORDS[chosen][0]} only, of {len(listing.get('items') or [])} listed"
    if not items:
        return f"no checkpoint shown{of}"
    order = "newest first" if _view(view).get("groups") is None else "in the groups below"
    if (preset.latest_runs is not None and not show_older and len(runs) > preset.latest_runs
            and _view(view).get("groups") is None):
        shown = sum(len(v) for _, v in runs[:preset.latest_runs])
        return (f"{order} · latest {preset.latest_runs} runs · {shown} checkpoints "
                f"(of {len(items)} in {len(runs)} runs){of}")
    return f"{order} · {len(items)} checkpoints in {len(runs)} runs{of}"


def find_item(listing: Mapping[str, Any], key: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(key, str):
        return None
    for item in listing.get("items") or []:
        if item_key(item) == key:
            return item
    return None


def loadable(item: Optional[Mapping[str, Any]]) -> bool:
    return bool(item) and item.get("problem") is None and r.is_digest(item.get("sha256"))


def load_refusal(item: Mapping[str, Any]) -> Optional[str]:
    """Why a listed checkpoint is never loaded, from its listed model record
    alone (nothing of its file is read again): a base offers no diagnosis; a
    file without a model spec is history, never run (K6 decision 1); a record
    this build refused, or never read, says so. None for a model finetuned for
    a task, which the reload serves under that task's scoring mode."""
    record = item.get("record")
    if record == "model":
        if item.get("learning_stage") == "base":
            return BASE_REFUSAL
        if item.get("task") in TASK_SCORING_MODES:
            return None
        return f"{r.record_kind_text(item)}: its record names no task this build serves."
    if record == "history":
        return HISTORY_REFUSAL
    if record == "refused":
        return f"{r.record_kind_text(item)}: {item.get('record_problem') or 'refused'}."
    return f"{r.record_kind_text(item)}."


def can_load(item: Optional[Mapping[str, Any]]) -> bool:
    """Whether Load is offered for a listed checkpoint: its file was read and
    its record is a model finetuned for a task (`loadable`, `load_refusal`)."""
    return loadable(item) and load_refusal(item) is None


def load_reason_html(item: Optional[Mapping[str, Any]]) -> str:
    """Beside the Load button, why it is off for the selected row when its
    record is never loaded (`load_refusal`); nothing otherwise (an unreadable
    file says so in its row)."""
    if not loadable(item):
        return ""
    why = load_refusal(item)
    return f'<div class="shep-reason shep-reason-grey">{r.e(why)}</div>' if why else ""


def facts_html(listing: Mapping[str, Any], key: Any) -> str:
    item = find_item(listing, key)
    if item is None:
        return '<div class="shep-muted">Select a model to see its facts.</div>'
    workspaces = listing.get("workspaces") or []
    serving = next((w for w in workspaces if w.get("name") == listing.get("serving")), None)
    facts = r.facts_from_checkpoint(item, workspaces, recorded=_recorded(listing, key),
                                    serving_workspace=serving)
    title = (f'<div class="shep-facts-title"><b>{r.e(facts.name_line)}</b> '
             f'{r.chip("selected", "neutral")}</div>')
    return title + r.model_facts_html(facts)


def note_html(view: Any) -> str:
    """The view's note beside the Load button (plain text, escaped)."""
    note = _view(view).get("note")
    return f'<span class="shep-note">{r.e(note)}</span>' if note else ""


# ---------------------------------------------------------------------------
# The Load (server side)
# ---------------------------------------------------------------------------
def _graph_digest(workspace_dir: Path) -> Optional[str]:
    from src.kg.artifacts import read_split_manifest

    try:
        artifacts = read_split_manifest(workspace_dir).manifest.get("artifacts")
    except (ValueError, OSError):
        return None
    kg = artifacts.get("kg") if isinstance(artifacts, Mapping) else None
    return kg if r.is_digest(kg) else None


def _from_side(status: Optional[Mapping[str, Any]], listing: Mapping[str, Any]) -> Dict[str, Any]:
    """What is served now, as the confirmation's "From" column needs it.

    `listing` gives the checkpoints to name the served one by (``items``), the
    listed workspaces (``workspaces``) and the serving workspace's name
    (``serving``)."""
    identity = (status or {}).get("served_identity") or {}
    sha = served_checkpoint_sha(status)
    item = next((i for i in listing.get("items") or [] if sha and i.get("sha256") == sha), None)
    meta = (status or {}).get("checkpoint_meta") or {}
    if item is not None:
        model = r.model_name_line(item)
    elif sha:
        model = f"epoch {meta.get('epoch', '—')} (not in this workspace's list)"
    else:
        model = "no trained model" if (status or {}).get("initialized") else "no model loaded"
    sp = identity.get("shortest_paths") or {}
    record = identity.get("dataset_record") or {}
    return {
        "model": model,
        "sha256": sha,
        "trained_on": item.get("dataset_fingerprint") if item else None,
        "trained_on_known": item is not None,
        "workspace": listing.get("serving"),
        "graph": (identity.get("graph") or {}).get("kg_sha256"),
        "sp": sp.get("state"),
        "sp_build": sp.get("build_id"),
        "mode": identity.get("scoring_mode"),
        "synthetic": record.get("origin") == "synthetic",
    }


def _to_side(item: Mapping[str, Any], workspace: str, workspace_dir: Path,
             workspaces: Sequence[Any]) -> Dict[str, Any]:
    """What would be served, as the confirmation's "To" column needs it: the
    scoring mode is the one the task the model's record names is served under
    (`core.types.TASK_SCORING_MODES`, the pipeline's own table), and its
    shortest-path term a state of that task (`_SP_OF_MODE`); neither is
    guessed from the workspace, which a task model's serving never reads for
    them, and no mixing weight is predicted."""
    origin = next((r.origin_of(w) for w in workspaces if r._get(w, "name") == workspace), None)
    mode = TASK_SCORING_MODES.get(item.get("task"))
    return {
        "model": r.model_name_line(item),
        "sha256": item.get("sha256"),
        "trained_on": item.get("dataset_fingerprint"),
        "trained_on_known": True,
        "workspace": workspace,
        "graph": _graph_digest(workspace_dir),
        "sp": _SP_OF_MODE.get(mode, "absent"),
        "sp_build": None,
        "mode": mode,
        "synthetic": origin == "synthetic",
    }


def _target_now(item: Mapping[str, Any], workspace: Any, *,
                environ: Optional[Mapping[str, str]] = None
                ) -> Tuple[Dict[str, Any], Path, Path]:
    """The checkpoint `item` names, as it is on disk now: found in the server's
    list again (`resolve_checkpoint`) and its bytes hashed again. The same bytes
    are the same item; other bytes are read again through the server's list
    (`list_checkpoints`), so what is shown is what the file holds now. Returns
    ``(item, workspace directory, checkpoint path)``.

    Raises:
        ServerListRefused: the workspace or the checkpoint is no longer listed,
            or its bytes cannot be read.
    """
    from src.api.services.server_lists import (
        ServerListRefused,
        list_checkpoints,
        resolve_checkpoint,
        resolve_workspace,
    )
    from src.evaluation.sidecar import checkpoint_matches

    directory = resolve_workspace(workspace, environ=environ)
    checkpoint = resolve_checkpoint(workspace, item.get("architecture"), item.get("run"),
                                    item.get("filename"), environ=environ)
    # The one comparison of a file's bytes with a digest (`checkpoint_matches`,
    # the registration's): it records nothing. It decides what is shown; the
    # reload compares its own read with the digest shown before it parses it.
    try:
        same = r.is_digest(item.get("sha256")) and checkpoint_matches(checkpoint, item["sha256"])
    except OSError as exc:
        raise ServerListRefused(
            f"{item.get('run')}/{item.get('filename')} cannot be read now "
            f"({type(exc).__name__})") from None
    if same:
        return dict(item), directory, checkpoint
    listed = list_checkpoints(workspace, item.get("architecture"), environ=environ)
    for fresh in listed.items:
        if fresh.run == item.get("run") and fresh.filename == item.get("filename"):
            return asdict(fresh), directory, checkpoint
    raise ServerListRefused(
        f"{item.get('run')}/{item.get('filename')} is no longer listed"
        + (f" ({listed.problem})" if listed.problem else ""))


def _cell(side: Mapping[str, Any], row: str, workspaces: Sequence[Any], *, to: bool,
          other: Optional[Mapping[str, Any]] = None) -> str:
    if row == "model":
        return r.e(side["model"])
    if row == "sha":
        return f'<code class="shep-fpv">{r.e(r.short_hex(side["sha256"]))}</code>'
    if row == "trained":
        if not side.get("trained_on_known"):
            return '<span class="shep-muted">not in this workspace\'s list</span>'
        html_ = r.trained_on_html(side["trained_on"], workspaces)
        if to and other is not None and other.get("trained_on") and \
                other.get("trained_on") == side.get("trained_on"):
            html_ += ' <span class="shep-muted">same dataset</span>'
        return html_
    if row == "workspace":
        if to and other is not None and other.get("workspace") == side["workspace"] and \
                other.get("graph") == side["graph"]:
            return "same"
        name = f"<b>{r.e(side['workspace'])}</b>" if side["workspace"] else \
            '<span class="shep-muted">not a listed workspace</span>'
        return f'{name} · graph <code>{r.e(r.short_hex(side["graph"]))}</code>'
    if row == "sp":
        state = side.get("sp")
        if state == "verified":
            build = f": build <code>{r.e(side['sp_build'])}</code>" if side.get("sp_build") else ""
            return f"present{build}"
        if state == "computed":
            return "computed from the model's graph view"
        return "not used by this task" if side.get("mode") in SCORING_MODE_TASKS else "absent"
    if row == "mode":
        mode = side.get("mode")
        task = SCORING_MODE_TASKS.get(mode)
        code = f'<code>{r.e(mode or "—")}</code>'
        return f"{r.e(TASK_NAMES[task])} ({code})" if task else code
    if row == "data":
        return r.chip("synthetic", "amber") if side["synthetic"] else "files"
    return ""


_ROWS = (("model", "Model"), ("sha", "Model fingerprint"), ("trained", "Trained on"),
         ("workspace", "Serving workspace"), ("mode", "Task and scoring"), ("sp", "SP table"),
         ("data", "Data"))
_COMPARED = {"model": ("sha256",), "sha": ("sha256",), "trained": ("trained_on",),
             "workspace": ("workspace", "graph"), "sp": ("sp", "sp_build"),
             "mode": ("mode",), "data": ("synthetic",)}


def confirmation_html(plan: Mapping[str, Any], *, notice: Optional[str] = None) -> str:
    """The one confirmation: "Reload the pipeline with <12 hex> for every
    session?", from → to with the changed rows marked, and what it replaces."""
    frm, to = plan["from"], plan["to"]
    workspaces = plan.get("workspaces") or []
    rows = []
    for key, label in _ROWS:
        changed = any(frm.get(f) != to.get(f) for f in _COMPARED[key])
        rows.append(
            f'<tr class="{"shep-changed" if changed else ""}"><th scope="row">{r.e(label)}'
            f'{"<span class=shep-vh> (changes)</span>" if changed else ""}</th>'
            f'<td>{_cell(frm, key, workspaces, to=False)}</td>'
            f'<td>{_cell(to, key, workspaces, to=True, other=frm)}</td></tr>')
    note = (f'<div class="shep-notice shep-notice-amber" role="alert">{r.e(notice)}</div>'
            if notice else "")
    new, old = r.short_hex(to["sha256"]), r.short_hex(frm.get("sha256"))
    return (
        f'<div class="shep-confirm" data-new="{r.e(new)}">{note}'
        f'<h3 class="shep-confirm-q">Reload the pipeline with <code>{r.e(new)}</code> for every '
        f'session?</h3>'
        '<table class="shep-ftbl"><thead><tr><th></th><th>From — in use now</th><th>To</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        f'<p class="shep-confirm-note">{r.e(REPLACES_NOTE)}</p>'
        f'<div class="shep-load-wait" hidden data-wait="{PAGE_WAIT_SECONDS}">'
        f'<span class="shep-spinner" aria-hidden="true"></span> <b>Loading <code>{r.e(new)}</code> '
        'for every session… <span class="shep-elapsed">00:00:00</span></b> Diagnoses wait until '
        'the reload finishes. Closing this does not stop it.</div>'
        '<div class="shep-load-timeout" hidden><b>The page stopped waiting after '
        f'{PAGE_WAIT_SECONDS} s; the server may still finish the reload.</b> '
        f'<button type="button" class="shep-btn shep-read-again" data-key="__read_again__">'
        'Read again</button></div>'
        f'<span class="shep-vh" data-old="{r.e(old)}"></span></div>'
    )


#: **What a plan keeps of a listed checkpoint**, the one tuple for the plan's
#: listing and its "To" item: its place, bytes and dataset, and the record
#: fields `load_refusal` checks again and the reload's task comes from.
_ITEM_KEYS = ("workspace", "architecture", "run", "filename", "sha256", "epoch",
              "dataset_fingerprint", "problem", "record", "record_problem", "learning_stage",
              "task", "encoder", "reference", "differences")


def _kept(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: item.get(k) for k in _ITEM_KEYS}


def _listed_context(listing: Mapping[str, Any]) -> Dict[str, Any]:
    """What a plan keeps of the listing to read the From column again: the
    checkpoints to name the served one by, and the listed workspaces."""
    return {"items": [_kept(i) for i in listing.get("items") or []],
            "workspaces": list(listing.get("workspaces") or [])}


def request_load(listing: Mapping[str, Any], key: Any, ordering: Any, request: Any, *,
                 environ: Optional[Mapping[str, str]] = None
                 ) -> Tuple[Optional[Dict[str, Any]], str]:
    """Step 1: D8 on this request; a row that is never loaded refused before
    anything of its file is read (`load_refusal`); the item found in the
    server's list again and its bytes read again, and the confirmation.
    Returns ``(plan, html)``; a refusal returns no plan and says why."""
    from src.api.services.server_lists import ServerListRefused

    access = operator_access(request)
    if not access.allowed:
        return None, f'<div class="shep-reason shep-reason-grey">{r.e(access.refusal)}</div>'
    item = find_item(listing, key)
    if not loadable(item):
        return None, ('<div class="shep-reason">Select a model that can be loaded first. '
                      'Nothing was done.</div>')
    refusal = load_refusal(item)
    if refusal is not None:
        return None, f'<div class="shep-reason">{r.e(refusal)} Nothing was done.</div>'
    workspace = listing.get("workspace")
    try:
        target, directory, _checkpoint = _target_now(item, workspace, environ=environ)
    except ServerListRefused as exc:
        return None, f'<div class="shep-reason">Refused: {r.e(exc)} Nothing was done.</div>'
    if not loadable(target):
        return None, (f'<div class="shep-reason">It cannot be loaded now: '
                      f'{r.e(target.get("problem") or "its digest was not read")}. Nothing was '
                      'done.</div>')
    refusal = load_refusal(target)
    if refusal is not None:
        return None, (f'<div class="shep-reason">The file changed since the list was read: '
                      f'{r.e(refusal)} Nothing was done.</div>')
    status, problem = read_status()
    if status is None:
        return None, (f'<div class="shep-reason">Cannot load now: {r.e(problem)}. Nothing was '
                      'done.</div>')
    workspaces = listing.get("workspaces") or []
    context = _listed_context(listing)
    plan = {"workspace": workspace, "architecture": item["architecture"], "run": item["run"],
            "filename": item["filename"],
            "from": _from_side(status, {**context, "serving": listing.get("serving")}),
            "from_key": served_key(status),
            "to": _to_side(target, workspace, directory, workspaces),
            "to_item": _kept(target),
            **context,
            "ordering": dict(ordering) if isinstance(ordering, Mapping) else None}
    notice = None
    if target.get("sha256") != item.get("sha256"):
        notice = (f"The file changed since the list was read: it holds model "
                  f"{r.short_hex(target.get('sha256'))} now, not {r.short_hex(item.get('sha256'))}"
                  " as listed. The To column is what it holds now; read the list again to see "
                  "the rest.")
    return plan, confirmation_html(plan, notice=notice)


def _record_usage(plan: Mapping[str, Any], loaded: bool, why: Optional[str],
                  environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The usage history entry for a Load from an ordered list (D4), or why it
    could not be written; None when the list was not ordered or it was written.
    The version is read bound to the digest the list was ordered by
    (`resolve_version(expected_sha256=)`, compared before decoding), so an entry
    is never written against other bytes."""
    ordering = plan.get("ordering")
    if not ordering:
        return None
    try:
        from src.config import server_settings
        from src.evaluation.cohort_registry import CohortVersionChanged, resolve_version
        from src.evaluation.report_store import record_model_load

        setting = server_settings.cohort_registry(environ)
        if not setting.available:
            return f"the usage history was not written: {setting.problem}"
        try:
            version = resolve_version(setting.value, ordering["version"],
                                      expected_sha256=ordering.get("version_sha256") or None)
        except CohortVersionChanged:
            return ("the usage history was not written: the cohort version changed since the "
                    "list was ordered")
        record_model_load(version, checkpoint_sha256=plan["to"]["sha256"],
                          metric=ordering["metric"],
                          scoring_semantics_digest=ordering["scoring_semantics_digest"],
                          loaded=loaded, why=why)
    except Exception as exc:  # noqa: BLE001 - the load stands; the page says what failed
        logger.exception("The usage history entry could not be written")
        return f"the usage history was not written ({type(exc).__name__})"
    return None


def _outcome(kind: str, headline: str, text: str = "") -> str:
    waited = ('<span class="shep-if-waited">The reload finished after the page stopped waiting. '
              '</span>')
    return (f'<div class="shep-outcome shep-outcome-{kind}" role="status">{waited}<b>'
            f'{r.e(headline)}</b> {r.e(text)}</div>')


class LoadStep(NamedTuple):
    """What step 2 of a Load leaves on the page.

    `plan` stays set (with the new from- or to-values) when the confirmation is
    shown again; `confirm_html` is that confirmation or the outcome;
    `reason_html` a refusal beside the buttons; `served` the model fingerprint
    served after it; `changed` whether what is served changed (any part of the
    served identity, or the served workspace: the "served model changed"
    trigger)."""

    plan: Optional[Dict[str, Any]]
    confirm_html: str
    reason_html: str
    served: Optional[str]
    changed: bool


def _sides_differ(one: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    keys = set(one) | set(other)
    return any(one.get(k) != other.get(k) for k in keys)


def finish_load(plan: Optional[Mapping[str, Any]], request: Any, *,
                reload: Any = None, environ: Optional[Mapping[str, str]] = None) -> LoadStep:
    """Step 2, synchronous (the caller runs it on the event loop, as the API
    route runs its reload): D8 on this request; what is served and what would be
    loaded both read again, and the confirmation shown again with a notice when
    either changed; WP2's reload with what the server resolves from the item,
    bound to the model fingerprint shown (other bytes are refused by the
    reload's own read, and nothing is published); the usage history."""
    from src.api.services.server_lists import ServerListRefused

    access = operator_access(request)
    if not access.allowed:
        return LoadStep(dict(plan) if plan else None,
                        confirmation_html(plan) if plan else "",
                        f'<div class="shep-reason shep-reason-grey">{r.e(access.refusal)}</div>',
                        None, False)
    if not plan:
        return LoadStep(None, "", '<div class="shep-reason">Nothing to load. Nothing was done.</div>',
                        None, False)
    status, problem = read_status()
    if status is None:
        return LoadStep(dict(plan), confirmation_html(plan),
                        f'<div class="shep-reason">Not reloaded: {r.e(problem)}. Nothing was '
                        'done.</div>', None, False)
    now_sha = served_checkpoint_sha(status)
    now_key = served_key(status)
    workspaces = plan.get("workspaces") or []
    new_plan = dict(plan)
    notices = []
    if now_key != plan.get("from_key"):
        # What is served changed since this was shown (another session loaded
        # a model or another workspace): said with the new from-values, read
        # as the confirmation reads them; never reloaded over unasked.
        new_plan["from"] = _from_side(status, {
            "items": plan.get("items") or [], "workspaces": workspaces,
            "serving": serving_workspace(status, workspaces, environ=environ)})
        new_plan["from_key"] = now_key
        notices.append(FROM_CHANGED)
    shown = plan.get("to_item") or {k: plan.get(k) for k in ("architecture", "run", "filename")}
    try:
        target, directory, checkpoint = _target_now({**shown, "sha256": plan["to"].get("sha256")},
                                                    plan["workspace"], environ=environ)
    except ServerListRefused as exc:
        why = str(exc)
        note = _record_usage(plan, False, why, environ)
        return LoadStep(None, _outcome("grey", "Not reloaded.", f"Refused: {why} Nothing was done."
                                       + (f" Also, {note}." if note else "")), "", now_sha, False)
    # The refusal again on what the file holds now (rewritten into a base
    # since the confirmation, say): never reloaded, never shown to confirm.
    refusal = load_refusal(target) if loadable(target) else None
    if refusal is not None:
        note = _record_usage(plan, False, refusal, environ)
        return LoadStep(None, _outcome("grey", "Not reloaded.", f"{refusal} Nothing was done."
                                       + (f" Also, {note}." if note else "")), "", now_sha, False)
    to_now = _to_side(target, plan["workspace"], directory, workspaces)
    if _sides_differ(to_now, plan["to"]):
        new_plan["to"] = to_now
        new_plan["to_item"] = _kept(target)
        notices.append(TO_CHANGED)
    if notices:
        if not loadable(target):
            note = _record_usage(plan, False, "it cannot be loaded now", environ)
            return LoadStep(None, _outcome(
                "grey", "Not reloaded.",
                f"It cannot be loaded now: {target.get('problem') or 'its digest was not read'}. "
                "Nothing was done." + (f" Also, {note}." if note else "")), "", now_sha, False)
        return LoadStep(new_plan, confirmation_html(new_plan, notice=" ".join(
            [*notices, NOTHING_RELOADED])), "", now_sha, False)

    from fastapi.encoders import jsonable_encoder

    from src.api.routes import pipeline as pipeline_routes

    run = reload if reload is not None else pipeline_routes.run_reload
    try:
        # The task the confirmed model's record names: the reload refuses a
        # build that serves another (README §1, switching task is loading
        # another model).
        answer = jsonable_encoder(run(pipeline_routes.PipelineReloadRequest(
            data_dir=str(directory), checkpoint_path=str(checkpoint),
            expected_checkpoint_sha256=plan["to"]["sha256"], task=plan["to_item"]["task"])))
    except Exception as exc:  # noqa: BLE001 - a reload that raises is a refused reload
        logger.exception("Pipeline reload failed")
        answer = {"success": False, "message": f"Reload failed ({type(exc).__name__})."}
    after = answer.get("status")
    served = served_checkpoint_sha(after)
    changed = isinstance(after, Mapping) and served_key(after) != now_key
    if not answer.get("success"):
        why = str(answer.get("message") or "the reload was refused")
        note = _record_usage(plan, False, why, environ)
        return LoadStep(None, _outcome("grey", "Not reloaded; the model in use is unchanged.",
                                       why + (f" Also, {note}." if note else "")),
                        "", served if isinstance(after, Mapping) else now_sha, changed)
    note = _record_usage(plan, True, None, environ)
    text = f"The model in use for every session is now {r.short_hex(served)}."
    if plan.get("ordering") and note is None:
        text += " The usage history records this load as made from the ordered list."
    elif note:
        text += f" But {note}."
    return LoadStep(None, _outcome("ok", "Reloaded.", text), "", served, changed)


def read_again() -> str:
    """"Read again" after the page stopped waiting: what is served now."""
    status, problem = read_status()
    at = r.clock_utc(datetime.now())
    if status is None:
        return _outcome("grey", f"Status read again at {at}:", f"{problem}.")
    sha = served_checkpoint_sha(status)
    return _outcome("grey", f"Status read again at {at}:",
                    f"the model in use is {r.short_hex(sha) if sha else 'none'}.")


# ---------------------------------------------------------------------------
# The component
# ---------------------------------------------------------------------------
#: The page side: row clicks (and Enter or Space on a focused table row) and
#: "Read again" become the component's ``click`` event with ``{"key": …}``.
MODEL_LIST_JS = r"""
element.addEventListener('click', function (e) {
  if (e.target.closest && e.target.closest('button.shep-copy, button.shep-i')) { return; }
  var row = e.target.closest ? e.target.closest('[data-key]') : null;
  if (!row || row.getAttribute('aria-disabled') === 'true') { return; }
  trigger('click', {key: row.getAttribute('data-key')});
});
element.addEventListener('keydown', function (e) {
  var row = e.target.closest ? e.target.closest('tr[data-key]') : null;
  if (!row || e.target !== row || row.getAttribute('aria-disabled') === 'true') { return; }
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); trigger('click', {key: row.getAttribute('data-key')}); }
});
"""

_START_WAIT_JS = r"""
(...args) => {
  // The Reload button goes at once, before the server answers: a second click
  // cannot ask for a second reload (the answer then hides it).
  var again = document.getElementById(__RELOAD_ID__);
  if (again) { again.disabled = true; }
  // Until the reload answers nothing on the server answers (it holds the event
  // loop): a close (`guard.closers`) is done here at once, by hiding
  // `guard.hide`, and sent to the server only after the answer
  // (`load_answered_js`); while it is closed, an opener shows it again.
  var guard = __GUARD__;
  if (guard) {
    var loads = window.shepLoads = window.shepLoads || {};
    var held = document.getElementById('shep-load-closed-' + guard.list);
    if (held) { held.remove(); }
    loads[guard.list] = {busy: true, closed: false, guard: guard};
    if (!window.shepLoadGuard) {
      window.shepLoadGuard = true;
      document.addEventListener('click', function (e) {
        var target = e.target;
        if (!target || !target.closest) { return; }
        var all = window.shepLoads || {};
        Object.keys(all).forEach(function (id) {
          var st = all[id], g = st.guard;
          if (!st.busy) { return; }
          var css = document.getElementById('shep-load-closed-' + id);
          var hit = function (sel) { return !!target.closest(sel); };
          if (g.closers.some(hit)) {
            e.preventDefault(); e.stopImmediatePropagation();
            if (!css) { css = document.createElement('style'); css.id = 'shep-load-closed-' + id; document.head.appendChild(css); }
            // Three ids more than the selector: above a page's own rules for it
            // (Models: "#page #list > .shep-model-confirm-box" sets its display).
            css.textContent = g.hide + ':not(#shep-load-x1):not(#shep-load-x2):not(#shep-load-x3) { display: none !important; }';
            st.closed = true;
          } else if (st.closed && g.openers.some(hit)) {
            e.preventDefault(); e.stopImmediatePropagation();
            if (css) { css.remove(); }
            st.closed = false;
          }
        });
      }, true);
    }
  }
  var root = document.getElementById(__CONFIRM_ID__);
  if (root) {
    root.classList.remove('shep-waited');
    var wait = root.querySelector('.shep-load-wait'), out = root.querySelector('.shep-load-timeout');
    if (wait) {
      wait.hidden = false;
      var started = Date.now(), limit = (parseInt(wait.getAttribute('data-wait'), 10) || 180) * 1000;
      var timer = setInterval(function () {
        if (!wait.isConnected) { clearInterval(timer); return; }
        var s = Math.floor((Date.now() - started) / 1000);
        var hh = String(Math.floor(s / 3600)).padStart(2, '0'), mm = String(Math.floor(s / 60) % 60).padStart(2, '0'), ss = String(s % 60).padStart(2, '0');
        var el = wait.querySelector('.shep-elapsed'); if (el) { el.textContent = hh + ':' + mm + ':' + ss; }
        // The same count wherever the page shows this load (Diagnosis' card).
        document.querySelectorAll('[data-shep-load-elapsed]').forEach(function (e) { e.textContent = hh + ':' + mm + ':' + ss; });
        if (Date.now() - started >= limit) {
          wait.hidden = true; if (out) { out.hidden = false; } root.classList.add('shep-waited'); clearInterval(timer);
          document.querySelectorAll('[data-shep-load-card]').forEach(function (c) {
            var on = c.querySelector('.shep-load-on'), off = c.querySelector('.shep-load-off');
            if (on) { on.hidden = true; } if (off) { off.hidden = false; }
          });
        }
      }, 1000);
    }
  }
  return args;
}
"""


def start_wait_js(confirm_elem_id: str, reload_elem_id: Optional[str] = None,
                  guard: Optional[Mapping[str, Any]] = None) -> str:
    """The script "Reload with …" runs in the browser before its handler: the
    Reload button (`reload_elem_id`) disabled at once, so a second click cannot
    ask for a second reload before the answer hides it; the 180 s count in
    **this** list's confirmation (its element id), so a list placed twice on a
    page (Diagnosis' drawer and Models) counts in the one whose Reload was
    clicked, and in any element of the page marked ``data-shep-load-elapsed``
    (the count) or ``data-shep-load-card`` (its words when the page stops
    waiting: Diagnosis' card); with `guard` (`load_guard`), until the reload
    answers, a close done in the page at once and held for the server."""
    from src.webui.components.help import as_json_for_script

    return (_START_WAIT_JS.replace("__CONFIRM_ID__", as_json_for_script(confirm_elem_id))
            .replace("__RELOAD_ID__", as_json_for_script(reload_elem_id or ""))
            .replace("__GUARD__", as_json_for_script(dict(guard) if guard else None)))


def load_guard(elem_id: str, close: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """What a close does while this list's reload runs (`start_wait_js`), as
    CSS selectors: the ``closers`` (this list's Close, and `close`'s
    ``also``: a drawer's × and Cancel) hide ``hide`` at once in the page (this
    list's confirmation box, or `close`'s ``hide``: the drawer), and when the
    reload answers ``then`` is clicked (Close itself, or `close`'s ``then``:
    the drawer's ×), its close now handled after that answer; while it is
    closed, an ``opener`` (this list's Load, or `close`'s ``openers``: the
    button that opens the drawer) shows it again instead, on the reload's
    status, since nothing it would ask can be answered before the reload."""
    close = close or {}
    keep = f"#{elem_id}-keep"
    return {"list": elem_id,
            "closers": [keep, *(close.get("also") or ())],
            "hide": close.get("hide") or f"#{elem_id}-confirm-box",
            "then": close.get("then") or keep,
            "openers": list(close.get("openers") or [f"#{elem_id}-load"])}


_LOAD_ANSWERED_JS = r"""
() => {
  // The reload answered: a close done in the page meanwhile goes to the
  // server now, in order after the answer (its own click, no longer held).
  var st = (window.shepLoads || {})[__LIST__];
  if (st && st.busy) {
    st.busy = false;
    if (st.closed) {
      st.closed = false;
      var el = document.querySelector(st.guard.then);
      if (el) { el.click(); }
      else { var css = document.getElementById('shep-load-closed-' + __LIST__); if (css) { css.remove(); } }
    }
  }
  return [];
}
"""

_LOAD_CLOSED_JS = r"""
() => {
  // After a close's server steps: what the page hid while the reload ran is
  // hidden by the server now; the page's own hiding goes.
  var st = (window.shepLoads || {})[__LIST__];
  if (!st || !st.busy) { var css = document.getElementById('shep-load-closed-' + __LIST__); if (css) { css.remove(); } }
  return [];
}
"""


def load_answered_js(elem_id: str) -> str:
    """The page's step after the reload's answer (`load_guard`): a close done
    in the page while the reload ran is clicked again, now for the server."""
    from src.webui.components.help import as_json_for_script

    return _LOAD_ANSWERED_JS.replace("__LIST__", as_json_for_script(elem_id))


def load_closed_js(elem_id: str) -> str:
    """The page's step at the end of a close (this list's Close; a page's
    drawer close: ``ModelList.after_close_js``): the page's own hiding of what
    the server has now hidden goes."""
    from src.webui.components.help import as_json_for_script

    return _LOAD_CLOSED_JS.replace("__LIST__", as_json_for_script(elem_id))


@dataclass
class ModelList:
    """The components of one model list, for the page that places it.

    - ``workspace``: the workspace select (the server's list, the shared
      option labels `renderers.workspace_choices`);
    - ``listing``, ``selected``, ``plan``: this session's listing, selection and
      open confirmation (`gr.State`);
    - ``view``: `gr.State`, None or the page's view of the list
      (``task``, ``keys``, ``groups``, ``rest_title``, ``column``, ``cells``,
      ``note``: the module docstring); the list redraws on its ``.change``;
    - ``ordering``: `gr.State`, None, or ``{"version", "version_sha256",
      "metric", "scoring_semantics_digest"}`` when the page orders the list by a
      cohort version's score (Models): a Load then writes a usage history entry;
    - ``phase``: `gr.State`, ``"idle"``, ``"confirming"`` or ``"loading"``
      (Diagnosis disables Run Diagnosis while it is not idle);
    - ``refresh``: keyword arguments for an event that (re)reads the list:
      ``ctx.on_page_open("models", handle.refresh)``;
    - the rendered parts: ``count``, ``rows``, ``facts``, ``reason``, ``note``,
      ``load_button``, ``confirm``, ``confirm_html``, ``reload_button``,
      ``keep_button``; ``head_extra`` and ``foot_extra``, what the page's
      ``head`` and ``foot`` builders returned;
    - ``reload_done``: the event that ends a Reload (its answer: reloaded or
      not, the phase back to idle or confirming), for a page to chain
      ``.then(…)`` on, as Diagnosis clears its card's "Loading …" line;
    - ``after_close_js``: the page step a page chains at the end of its own
      close named in ``close_while_loading`` (`load_closed_js`).
    """

    preset: Preset
    elem_id: str
    workspace: Any
    listing: Any
    selected: Any
    plan: Any
    view: Any
    ordering: Any
    phase: Any
    show_older: Any
    count: Any
    rows: Any
    facts: Any
    reason: Any
    note: Any
    load_button: Any
    confirm: Any
    confirm_html: Any
    reload_button: Any
    keep_button: Any
    refresh: Dict[str, Any]
    head_extra: Any = None
    foot_extra: Any = None
    reload_done: Any = None
    after_close_js: str = ""


def loading_buttons(plan: Any) -> Tuple[Any, Any, Any]:
    """The step "Reload with …" runs first, before the reload: ``(phase, Reload,
    Keep)``. With a plan, the phase is "loading", Reload goes (a second click
    could ask for a second reload) and Keep becomes Close (closing does not
    stop the reload): while it runs the confirmation shows only its status
    and Close, then, after 180 s, "Read again" and Close. Without one,
    nothing changes."""
    if not plan:
        return "idle", gr.update(), gr.update()
    return "loading", gr.update(visible=False), gr.update(value="Close")


def create_model_list(ctx: Any, preset: str = "clinical", *, elem_id: Optional[str] = None,
                      head: Optional[Callable[[], Any]] = None,
                      foot: Optional[Callable[[], Any]] = None,
                      on_loading: Optional[Mapping[str, Any]] = None,
                      close_while_loading: Optional[Mapping[str, Any]] = None) -> ModelList:
    """Build the shared list in the current layout context and wire its events.

    `ctx` is the page context (`src.webui.shell.PageContext`): its chips are
    refreshed and its ``served_model_changed`` trigger set after a Load.
    `elem_id` defaults to ``shep-models-<preset>``, so both presets can sit in
    one page. `head` and `foot`, when given, are called inside the list's
    header and footer rows to place the page's own components there.
    `on_loading` (``{"fn", "inputs", "outputs"}``) adds the page's own outputs
    to the step "Reload with …" runs before the reload, in the same answer (so
    they show before the reload holds the server): ``fn(plan, *inputs)``
    returns one value per output, `plan` None when there was nothing to load
    (Diagnosis: its card's "Loading … for every session" line and Run
    Diagnosis' reason). `close_while_loading` (``{"hide", "then", "also",
    "openers"}``, `load_guard`) is what Close does while a reload runs when
    the list sits in a page's drawer: the drawer hides at once, and its own
    close is sent when the reload answers; the page chains
    ``after_close_js`` at the end of that close."""
    chosen = PRESETS[preset]
    elem_id = elem_id or f"shep-models-{preset}"
    listing = gr.State(empty_listing())
    selected = gr.State(None)
    plan = gr.State(None)
    view = gr.State(None)
    ordering = gr.State(None)
    phase = gr.State("idle")
    show_older = gr.State(False)
    # The table preset (Models) keeps its header and its rows in one column
    # (``shep-model-main``), so the page can scroll them as one region when
    # they do not fit together (M5-3 review UI-1); the drawer's rows need none.
    with gr.Column(elem_id=elem_id, elem_classes=["shep-model-list", f"shep-preset-{preset}"]):
        main = (gr.Column(elem_classes=["shep-model-main"]) if chosen.layout == "table"
                else contextlib.nullcontext())
        with main:
            with gr.Row(elem_classes=["shep-model-head"]):
                workspace = gr.Dropdown(label="Workspace", choices=[], value=None,
                                        elem_id=f"{elem_id}-workspace", interactive=True,
                                        scale=3)
                count = plain_html("", elem_id=f"{elem_id}-count")
                head_extra = head() if head is not None else None
            with gr.Row(elem_classes=["shep-model-body"]):
                rows = plain_html(
                    '<div class="shep-muted">The list is read when it is opened.</div>',
                    elem_id=f"{elem_id}-rows", js_on_load=MODEL_LIST_JS,
                    elem_classes=["shep-model-rows"])
                facts = plain_html(facts_html(empty_listing(), None),
                                   elem_id=f"{elem_id}-facts",
                                   elem_classes=["shep-model-facts"])
        with gr.Row(elem_classes=["shep-model-foot"]):
            load_button = gr.Button("🔒 Load this model", variant="primary", interactive=False,
                                    elem_id=f"{elem_id}-load", scale=0)
            reason = plain_html("", elem_id=f"{elem_id}-reason")
            note = plain_html("", elem_id=f"{elem_id}-note")
            foot_extra = foot() if foot is not None else None
        with gr.Column(visible=False, elem_id=f"{elem_id}-confirm-box",
                       elem_classes=["shep-model-confirm-box"]) as confirm:
            confirm_html = plain_html("", elem_id=f"{elem_id}-confirm", js_on_load=MODEL_LIST_JS,
                                      elem_classes=["shep-model-confirm"])
            with gr.Row():
                reload_button = gr.Button("🔒 Reload", variant="primary", scale=0,
                                          elem_id=f"{elem_id}-reload")
                keep_button = gr.Button("Keep", scale=0, elem_id=f"{elem_id}-keep")

    def _refresh(name, older, current_view, previous, key, request: gr.Request):
        # A selection still listed in the same workspace stays selected (M5-3
        # review UI-3: the re-read after a Load dropped a row picked meanwhile).
        data = list_models(name, request, family=chosen.family)
        key = kept_selection(previous, data, key, current_view, chosen)
        item = find_item(data, key)
        choices = r.workspace_choices(data.get("workspaces") or [])
        # Reading the list is no act: a refused read says why in the list
        # (`refused_line`); Load's reason is said when Load is clicked, or
        # beside it at once for a row whose record is never loaded.
        return (data, key, gr.update(choices=choices, value=data.get("workspace")),
                count_line(data, chosen, older, current_view),
                list_html(data, key, chosen, show_older=older, view=current_view),
                facts_html(data, key), gr.update(interactive=can_load(item)),
                load_reason_html(item), note_html(current_view))

    refresh_outputs = [listing, selected, workspace, count, rows, facts, load_button, reason, note]
    refresh = {"fn": _refresh, "inputs": [workspace, show_older, view, listing, selected],
               "outputs": refresh_outputs, "show_progress": "minimal"}
    workspace.input(**refresh)

    def _select(data, older, current_view, evt: gr.EventData):
        raw = getattr(evt, "_data", None)
        key = raw.get("key") if isinstance(raw, Mapping) else None
        if key == "__older__":
            return (gr.update(), True,
                    list_html(data, None, chosen, show_older=True, view=current_view),
                    count_line(data, chosen, True, current_view), gr.update(), gr.update(),
                    gr.update())
        item = find_item(data, key)
        if item is None or item not in shown_items(data, current_view, chosen):
            return (gr.update(),) * 7
        return (key, older, list_html(data, key, chosen, show_older=older, view=current_view),
                count_line(data, chosen, older, current_view), facts_html(data, key),
                gr.update(interactive=can_load(item)), load_reason_html(item))

    rows.click(_select, inputs=[listing, show_older, view],
               outputs=[selected, show_older, rows, count, facts, load_button, reason],
               show_progress="hidden")

    def _redraw(data, key, older, current_view):
        # The page changed its view (a filter, an order): the selection stays
        # only while the view still shows it.
        item = find_item(data, key)
        if item is None or item not in shown_items(data, current_view, chosen):
            key, item = None, None
        return (key, list_html(data, key, chosen, show_older=older, view=current_view),
                count_line(data, chosen, older, current_view), facts_html(data, key),
                gr.update(interactive=can_load(item)), load_reason_html(item),
                note_html(current_view))

    view.change(_redraw, inputs=[listing, selected, show_older, view],
                outputs=[selected, rows, count, facts, load_button, reason, note],
                show_progress="hidden")

    def _request(data, key, order, request: gr.Request):
        new_plan, html_ = request_load(data, key, order, request)
        if new_plan is None:
            return None, gr.update(visible=False), "", html_, gr.update(), gr.update(), "idle"
        new, old = r.short_hex(new_plan["to"]["sha256"]), r.short_hex(new_plan["from"].get("sha256"))
        return (new_plan, gr.update(visible=True), html_, "",
                gr.update(value=f"🔒 Reload with {new}", visible=True),
                gr.update(value=f"Keep {old}" if new_plan["from"].get("sha256") else "Cancel"),
                "confirming")

    load_button.click(_request, inputs=[listing, selected, ordering],
                      outputs=[plan, confirm, confirm_html, reason, reload_button, keep_button,
                               phase]).then(**ctx.chip_refresh)

    page_inputs = list((on_loading or {}).get("inputs") or [])
    page_outputs = list((on_loading or {}).get("outputs") or [])

    def _loading(current, *page_values):
        own = loading_buttons(current)
        if not page_outputs:
            return own
        extra = on_loading["fn"](current or None, *page_values)
        return (*own, *extra)

    async def _confirm(current, request: gr.Request):
        # **D8, the status and the reload run here, on the event loop, with
        # nothing awaited before the reload**: it waits behind a running
        # diagnosis and never runs beside one, as the API route's does.
        try:
            step = finish_load(current, request)
        except Exception as exc:  # noqa: BLE001 - the page must leave "loading"
            logger.exception("The Load stopped with an error")
            step = LoadStep(None, _outcome("grey", "The Load stopped with an error.",
                                           f"({type(exc).__name__}) Read the status again: "
                                           "the Diagnosis card shows what is served."),
                            "", None, False)
        token = (f"{step.served}@{datetime.now().isoformat(timespec='seconds')}"
                 if step.changed else gr.update())
        still_open = step.plan is not None
        old = r.short_hex(step.plan["from"].get("sha256")) if still_open else ""
        new = r.short_hex(step.plan["to"].get("sha256")) if still_open else ""
        return (step.plan, gr.update(visible=True), step.confirm_html, step.reason_html, token,
                "confirming" if still_open else "idle",
                gr.update(visible=still_open,
                          **({"value": f"🔒 Reload with {new}"} if still_open else {})),
                gr.update(value=(f"Keep {old}" if still_open and old != "—" else
                                 "Cancel" if still_open else "Close")))

    reload_done = reload_button.click(
        _loading, inputs=[plan, *page_inputs],
        outputs=[phase, reload_button, keep_button, *page_outputs], queue=False,
        js=start_wait_js(f"{elem_id}-confirm", f"{elem_id}-reload",
                         load_guard(elem_id, close_while_loading)),
        show_progress="hidden").then(
        _confirm, inputs=[plan],
        outputs=[plan, confirm, confirm_html, reason, ctx.served_model_changed, phase,
                 reload_button, keep_button],
        show_progress="hidden")
    # A close done in the page while the reload ran goes to the server now.
    reload_done.then(None, js=load_answered_js(elem_id)).then(**ctx.chip_refresh).then(**refresh)

    # Keep (or Cancel, or Close after an outcome) in two steps: the first hides
    # the confirmation box and drops the plan, and changes no component inside
    # the box; the second empties the box and puts Reload back for the next
    # Load. A response that hides a column and updates a component inside it
    # leaves detached page elements and listeners in the browser on every
    # cycle (Gradio 6.30, measured: about 113 elements and 2 listeners per
    # Load-then-Keep), as Diagnosis' Cancel and × do (`reset_change`).
    def _keep():
        return None, gr.update(visible=False), "", "idle"

    def _after_keep(current):
        # Only while nothing was asked since: a Load clicked meanwhile has put
        # its own confirmation in the box.
        if current is not None:
            return gr.update(), gr.update()
        return "", gr.update(visible=True)

    after_close_js = load_closed_js(elem_id)
    keep_button.click(_keep, inputs=None, outputs=[plan, confirm, reason, phase],
                      show_progress="hidden").then(
        _after_keep, inputs=[plan], outputs=[confirm_html, reload_button],
        show_progress="hidden").then(None, js=after_close_js)

    def _read_again(evt: gr.EventData):
        raw = getattr(evt, "_data", None)
        if not (isinstance(raw, Mapping) and raw.get("key") == "__read_again__"):
            return gr.update()
        return read_again()

    confirm_html.click(_read_again, inputs=None, outputs=[reason], show_progress="hidden")

    return ModelList(chosen, elem_id, workspace, listing, selected, plan, view, ordering, phase,
                     show_older, count, rows, facts, reason, note, load_button, confirm,
                     confirm_html, reload_button, keep_button, refresh, head_extra, foot_extra,
                     reload_done, after_close_js)


__all__ = [
    "BASE_REFUSAL",
    "FAMILY_FILTERS",
    "FILTER_WORDS",
    "FROM_CHANGED",
    "HISTORY_REFUSAL",
    "IN_USE_NOTE",
    "LoadStep",
    "MODEL_LIST_JS",
    "ModelList",
    "NOTHING_RELOADED",
    "PAGE_WAIT_SECONDS",
    "PRESETS",
    "Preset",
    "REPLACES_NOTE",
    "REST_TITLE",
    "TASK_FILTERS",
    "TO_CHANGED",
    "can_load",
    "confirmation_html",
    "count_line",
    "create_model_list",
    "facts_html",
    "family_views",
    "find_item",
    "finish_load",
    "item_key",
    "kept_selection",
    "ledger_records",
    "list_html",
    "list_models",
    "load_answered_js",
    "load_closed_js",
    "load_guard",
    "load_reason_html",
    "load_refusal",
    "loadable",
    "loading_buttons",
    "matches",
    "note_html",
    "read_again",
    "read_recorded",
    "refused_line",
    "recorded_tests",
    "recorded_tests_for",
    "request_load",
    "shown_items",
    "start_wait_js",
    "task_filter",
]
