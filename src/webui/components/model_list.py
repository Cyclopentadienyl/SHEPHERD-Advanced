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
checkpoint (`read_ledger`, then the readers' one listing, `record_views.read_records`:
the records that read as results, and apart from them those that cannot, never
counted as results) and, for how each record's condition
group compares (runs that differ, a comparison incomplete), from the cohort
version's report store (`record_states.contest_states`, the one implementation
the API's route shares): never a number of a result, and a description that
hides, orders or refuses nothing. An
unreadable entry is greyed "Metadata not readable: <why>; cannot be loaded"
("Not read: …" for one never opened, `renderers.unreadable_words`), its
name said once, in its Model cell. "In use"
is matched by the **full** model fingerprint the status serves and marks the
checkpoint only.

**One tree, two presets** (`PRESETS`; D04: the two views may differ in
their defaults, never in how a model is drawn). Both draw **the tree**
(`tree_view`, `_tree_html`): every run a summary row (its run words, task,
operation with its freeze words, epochs completed and saved files; no score:
a file's figures are its details, so validations of different tasks are
never put on one scale), indented under the run of the exact parent file it
records where the page gives its families, its saved files folded under it,
each a row of its own (what it is, by the run's recorded selection, how far
it trained, its model fingerprint's 12 hex digits, the dataset only where it
is not the workspace's current one, `renderers.dataset_exception`: the
workspace's dataset is said once, under its select, `workspace_line_html`),
the served file marked in its own row; each group its own ``<tbody>``
(B72-C5). Where they differ:

- ``clinical`` (Diagnosis' Load for inference drawer): **models finetuned
  for a task only** (``Preset.rows``; README §5, "a base offers no
  diagnosis"), whatever view the page, or a direct call, gives; the latest 4
  runs and the run of the model in use, then "Show older runs"
  (`Preset.latest_runs`, BACKLOG 39); the chosen model's facts beside the
  rows (`facts_html`), each its own scroll region side by side, under them
  stacked, where choosing a row brings the facts into view and "↑ Back to the
  list" returns to the row (`FACTS_JS`; UI-07);
- ``research`` (Models): every run, its listing carrying the family view
  too (``listing["family"]``, `family_views`); its page lays the head out and
  shows a detail of its own.

**Search, folds and the selection** (D04), the list's own: a search box
(`search_hit`: names ignoring case, a model fingerprint by its first hex
digits or whole; a match keeps the files it comes from, as context; while
the search is shown, a run holding a file found by its own name or
fingerprint is unfolded as the list is drawn, the operator's folds
untouched, and clearing it brings them back: G7, `fold_open`; searched on
every change of its text, whatever made it, the box's ``change``), Clear
(the box emptied in the page), and the fold state (`gr.State`,
``{node: bool}``, what the operator folded or unfolded; a fold toggle folds
in the page at once, `MODEL_LIST_JS`, `FAMILY_JS`, and the server only keeps
it). By default every run is folded but the selection's and the model in
use's, which is located (scrolled into view once per view). **Nothing
changes what is selected but a click on a row**: a filter, a search, a fold
or a refresh keeps a selection still listed (`kept_selection`), and one the
list does not show is said (`outside_html`); one a refresh no longer lists is
cleared and said beside Load (`SELECTION_GONE`), never replaced by another
file. A lone candidate is chosen when a workspace is first listed (R-27).

**A page shapes its view** through ``ModelList.view`` (a `gr.State` the page
sets; the list is drawn again on its ``.change``), never by drawing rows itself:
``{"task": "disease"}`` lists the rows of one filter (`TASK_FILTERS`: all,
base, disease, gene, patients_like_me, task_models; `matches`, on the
server's one predicate `server_lists.record_matches`); ``"keys"`` lists exactly
those rows instead (a family view's, its context rows included); ``"groups":
[{"node", "title", "keys", "runs", "family"}]`` draws the rows in those
groups, in that order, each run in the order ``runs`` gives (checkpoints in
no group follow after them); ``"context"`` names the rows kept
only for a shown row that descends from them, ``"parents"`` each row's exact
parent file (a search keeps a match's ancestors by them); a view that gives
``"edges"`` (each run under the exact parent file it records, ``{"from":
<file key>, "to": <run key>}``) and ``"bounds"`` (a run whose parent is not
listed here, where it is) is drawn as **the family graph** (`_graph_html`:
Models' By family), its families as nodes and lines, the files of no family
under them as the table; ``"note"`` is plain
text beside the Load
button; ``"shaped_for"`` (a short string, the page's own words for the control
values it shaped the view for) is said by the rows drawn with it
(`view_marker`), so the page can tell rows that answer its controls from rows
drawn before (Models' "Applying …" line). What a view names by key (``keys``,
``groups``, ``context``, ``parents``, ``edges``, ``bounds``) belongs to the
workspace it was read from (the view's ``workspace``): another workspace's
listing is drawn with the view's filter and note only, and not as shaped for
anything (`view_for_listing`), until the page shapes its view again.
``create_model_list(head=…, tools=…, foot=…)`` places a page's own components
(a filter, links) in the list's header row, in its tools row after the
search (Models' Order: one bar, `create_model_list`) and in its footer row.

**The Load**, in three steps, each a handler of its own:

1. "Load this model" (`request_load`; labelled by the one gated-button rule,
   `load_button`: the lock and D8's reason beside it only for a request D8
   refuses): D8 on that request; **a row that is
   never loaded is refused before anything of its file is read**
   (`load_refusal`: a base offers no diagnosis, a record this build refused,
   a file without a model spec among them, is said with why; the
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
2. "Reload with <new>" (`finish_load`; shown only after a Load D8 allowed, so
   with no lock): D8 again, on that request; **both
   columns read again**: what is served (the whole served identity,
   `served_key`) and what would be loaded (the file hashed again, the
   workspace's graph read again, the refusal checked again on what the file
   holds now). A change in either since the confirmation was shown is said
   with the new values, and nothing is reloaded until the operator confirms
   what is shown. Then WP2's one reload (`run_reload`, in process, in a worker
   thread as the API route runs it) with the workspace, the checkpoint the
   server resolves from the item and the task its record names, **bound to the
   model fingerprint shown** (`expected_checkpoint_sha256`): the file
   can still change between this hash and the reload's own read, seconds later
   (the build reads the workspace's graph first), and the reload compares its
   read with the fingerprint before the bytes are parsed, so other bytes, or
   none, are refused as not the model shown, never served; then the chips and
   the "served model changed" trigger
   (`PageContext.served_model_changed`), set whenever any part of what is
   served changed;
3. "Keep <old>" closes the confirmation; nothing is done.

**While the reload runs the confirmation shows only its status and Close**
(`loading_buttons`, before the reload; the mockup's footer): Reload goes, at
once in the browser too, so a second click cannot ask for a second reload, and
Keep becomes Close, since closing does not stop it. **Close works at once**:
the page itself hides the confirmation (or the page's drawer, with its ×) and,
once the reload has answered, sends the close to the server in order after
that answer (`load_guard`, `start_wait_js`, `load_answered_js`); a close sent
while the reload ran would be handled beside it in no fixed order with its
answer, and could leave the box open and empty. A closed load's outcome is not
shown again: what is served then is on the page (the card, the list's "In use").

**The page waits 180 s.** The reload's build runs in a worker thread (BACKLOG
28: the server answers other requests meanwhile) and can take minutes on a
large graph, so the page itself counts (a script built for this list's own
confirmation, `start_wait_js`) and after 180 s says "The page stopped waiting
after 180 s; the server may still finish the reload." with "Read again" (a
status read). When the reload's answer arrives later, it says so ("The reload
finished after the page stopped waiting.").

**Run → Checkpoint** (`run_choices`, `checkpoint_choices`, `parents_update`,
`on_run_chosen`, `chosen_in_run`): the one pair of cascading selects by which a
page chooses a listed checkpoint, Training's Continue…, Fine-tune's parent and
the Test page's model alike; a run's
option is the one rule (`run_options`), which Training's run view also names
its runs by. Each
option is short (the WebUI skill §8: `renderers.fit_option`, at most 40
characters, one separator), the distinguishing token first (a run's name, or
its start time; what a file is), and **its value is the listing's key**
(`item_run_key`, `item_key`), so its words never change what is chosen. A
file's words come from its own record: how far it trained
(`renderers.epoch_words`), a best save of its run by the run's recorded
selection (`renderers.best_saves`), a best-k member by its run's records
(`CheckpointItem.in_best_k`), never from its name. A page that names a file
other than the chosen one names it by the same two options (`pair_words`).

**One draw, from the session's state** (G07; the review of 00258f4,
SL-R1: a pick answered after a newer search drew the list for no search,
under the search box still saying it): every event that changes what the list
shows (a pick, "Show older runs", a search and its Clear, a refresh, the
page's view) changes the session's state only, and one listener draws the
rows, the count and what describes the selection (the facts, Load, its
reason, the note) from that state and the search box as they are when it
runs (`list_drawing`), one draw of a list at a time and the last one asked
for always run. What an event was asked with (the search box's text as a row
was clicked) chooses what it changes, never what is drawn. What an act said
beside Load (a refusal, the status read again) is state too, said while its
selection is drawn, so no drawing drops it (the review of 5171661).

**What a handler commits is checked against what was committed meanwhile**
(the review of dcaed68, SL-R3: a pick answered after another workspace was
chosen selected that workspace's file at the same relative key; a re-read
answered after a newer pick set the selection back): the list's handlers
commit through the session's one `ListClock`, which says whether an answer
still belongs to the workspace the list now shows (the listing's generation,
`GENERATION`, said by the rows a pick is made in) and keeps a newer
selection, fold or "Show older runs" from being written over. A handler reads
everything first and commits last (the review of bd5e678, SL-R3c: a page's
own read after the commit wrote back what it had committed), so a page that
reads more for a listing does so between the refresh's two steps
(``ModelList.read_list``, ``ModelList.commit_list``).

**Per browser session**: the listing, the selection, the view, what was said
beside Load, the confirmation and the list's clock are `gr.State`; nothing a
session sees is module state (the draw tokens are one count for the process,
`draw_token`).

Module: src/webui/components/model_list.py
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import count as _count
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

# How a record's condition group compares (a description): one implementation,
# shared with the API's route (BACKLOG 36).
from src.api.services.record_states import CohortVersions, contest_states, key_text
from src.api.services.server_lists import record_matches
from src.core.types import SCORING_MODE_SP_STATES, SCORING_MODE_TASKS, TASK_SCORING_MODES
from src.evaluation.caveats import TASK_NAMES
from src.webui.components import renderers as r
from src.webui.components.served_status import (
    read_status,
    served_checkpoint_sha,
    served_key,
    serving_workspace,
)
from src.webui.operator_access import operator_access
from src.webui.shell import picked, plain_html
from src.webui.style import CELL_ROOMY, CELL_WHOLE, COL_OPT, COL_TWIN, FIT, FIT_WIDE

logger = logging.getLogger(__name__)

#: How long the page waits for a reload before it says it stopped waiting.
PAGE_WAIT_SECONDS = 180
#: What the confirmation says while this page waits for its reload's answer:
#: the page's own wait, never the load's state, which the server's status
#: says (the Diagnosis card's "Loading"), whichever page or client started it.
PAGE_WAIT_WORDS = "Reload sent. Waiting for the server's answer…"

#: What a confirmation says it replaces (the mockup's words).
REPLACES_NOTE = (
    "Replaces what every session is served once the new model is checked and published. "
    "Until then nothing changes, and results already produced keep their own identity."
)
#: Said when what is served changed between the confirmation and its confirm.
FROM_CHANGED = ("What every session is served changed since this was shown (another session "
                "reloaded): the From column is what is served now.")
#: Said when what would be loaded changed between the confirmation and its confirm.
TO_CHANGED = ("What would be loaded changed since this was shown: the To column is what the "
              "file and its workspace hold now.")
#: Ends either notice.
NOTHING_RELOADED = "Nothing was reloaded. Check the rows again, then reload or keep."

#: **The list's one filter vocabulary** (``view["task"]``, `matches`): every
#: row; the bases; one task's finetuned models; every model finetuned for a
#: task.
TASK_FILTERS = ("all", "base", "disease", "gene", "patients_like_me", "task_models")
#: Each filter as the count line ("Disease: 3 of 10 runs") and the empty state
#: ("No disease model is listed in this workspace.") name it.
FILTER_WORDS: Dict[str, Tuple[str, str]] = {
    "all": ("All", "checkpoint"),
    "base": ("Base", "base"),
    **{task: (name, f"{name.lower()} model") for task, name in TASK_NAMES.items()},
    "task_models": ("Task models", "model finetuned for a task"),
}

#: Why a base row is never loaded (`load_refusal`), said before anything of
#: its file is read: README §5 ("a base offers no diagnosis").
BASE_REFUSAL = "A base offers no diagnosis: load a model finetuned for a task."


@dataclass(frozen=True)
class Preset:
    """What differs between the two views of the one list."""

    key: str
    #: Runs shown before "Show older runs", with the run of the model in use
    #: (None: every run).
    latest_runs: Optional[int]
    #: The words of the "In use" mark.
    in_use_label: str
    #: How the list's own pieces are placed (the tree is drawn alike in
    #: both): ``drawer`` (Diagnosis: its head and search a field row each,
    #: the chosen model's facts beside the rows) or ``page`` (Models: its page
    #: lays the head out in its grid and shows its own detail).
    layout: str = "drawer"
    #: The rows this preset ever shows (a `TASK_FILTERS` name), whatever the
    #: view asks: the drawer lists models finetuned for a task only, so a view
    #: of None (a direct call) never lists a base or a refused file there.
    rows: str = "all"
    #: Whether its listing carries the family view (`family_views`).
    family: bool = False
    #: Where its page says the workspaces folder's notice itself (None: the
    #: list's empty state says it): Diagnosis' card, "Model in use", where the
    #: screen's first action is (L7 SW-LK8); the drawer's empty list then says
    #: its state and the notice's phrase (R-WHY, `NO_MODEL_TO_LIST`), never
    #: the notice again and never a pointer to its place.
    notice_where: Optional[str] = None


PRESETS: Dict[str, Preset] = {
    "clinical": Preset("clinical", 4, "In use", layout="drawer", rows="task_models",
                       notice_where="Model in use"),
    "research": Preset("research", None, "In use (Diagnosis)", layout="page", family=True),
}


# ---------------------------------------------------------------------------
# The listing (server side)
# ---------------------------------------------------------------------------
def item_key(item: Mapping[str, Any]) -> str:
    """A listed checkpoint's key in this workspace: the server's list's
    identity (`CheckpointItem.key`: architecture/run/file for names that are
    text, a form no text names give otherwise, BACKLOG 66), never its shown
    names, so two entries shown alike are never one key; an item that carries
    none (one built by hand) by its names, as the list keys them."""
    from src.api.services.record_views import item_key as key

    return key(item)


def item_run_key(item: Mapping[str, Any]) -> str:
    """A listed checkpoint's run's identity, by which a list groups its
    entries: the server's list's (`CheckpointItem.run_key`: architecture/run
    for names that are text, BACKLOG 66), never its shown names, so two runs
    shown alike are never one group; an item that carries none (one built by
    hand) by its names, as `item_key`."""
    key = item.get("run_key")
    if isinstance(key, str) and key:
        return key
    from src.utils.checkpoint_paths import listing_key

    return listing_key((str(item.get("architecture")), str(item.get("run"))))


# ---------------------------------------------------------------------------
# Run → Checkpoint: the two cascading selects' options
# ---------------------------------------------------------------------------
def runs_of(parents: Mapping[str, Any]) -> Dict[str, List[str]]:
    """The listed checkpoints by run, in the list's order (latest run first,
    newest file first): ``{run identity: [key, ...]}`` (`item_run_key`:
    ``"<architecture>/<run>"`` for names that are text)."""
    runs: Dict[str, List[str]] = {}
    for key, item in parents.items():
        runs.setdefault(item_run_key(item), []).append(key)
    return runs


def run_options(runs: Mapping[Any, Tuple[Any, Any]]) -> List[Tuple[str, Any]]:
    """**Runs as a select's options**, in `runs`' order (latest first), from
    ``{value: (architecture, run directory's name)}``: each run's words told
    apart from the others' (`renderers.run_names`: a name cut alike with
    another's keeps its start through where they differ; runs that would
    still read alike get their start time to the second, then their own id),
    and numbered only where nothing recorded tells them apart. Each value is
    its key, whatever the words: the Run select's run identity
    (`run_choices`), the Training run view's run id."""
    names = r.run_names(runs)
    return r.fit_options([(names[key], None, key) for key in runs])


def run_choices(parents: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """The Run select's options, latest run first (`runs_of`), by the one
    rule of a run's option (`run_options`); each value is the run's identity
    (`item_run_key`), whatever the words."""
    first = {run: parents[keys[0]] for run, keys in runs_of(parents).items()}
    return run_options({run: (item.get("architecture"), item.get("run"))
                        for run, item in first.items()})


def _checkpoint_parts(item: Mapping[str, Any], best: bool) -> Tuple[str, str]:
    """A checkpoint as an option's two parts: what the file is and how far it
    trained (`checkpoint_words`)."""
    filename = str(item.get("filename") or "")
    if item.get("problem") or not r.is_digest(item.get("sha256")):
        return filename, "cannot be read"
    what = ("Latest save" if filename == "last.pt" else
            f"Best {r.metric_words((item.get('selection') or {}).get('monitor'))}" if best else
            "Top-k save" if item.get("in_best_k") else filename)
    return what, ("weights only" if item.get("checkpoint_kind") == "weights"
                  else r.epoch_words(item))


def checkpoint_words(item: Mapping[str, Any], *, best: bool = False) -> str:
    """A checkpoint's option in its run, in a few words (the WebUI skill §8:
    at most `renderers.OPTION_CHARS` characters, one separator): what it is
    and how far it trained (`renderers.epoch_words`), "Latest save · 195
    epochs completed" (``last.pt``, the run's last training state), "Best LP
    accuracy · 21 epochs completed" (`best`: a best save of its run by the
    selection its record names, `renderers.best_saves`, named by that metric),
    "Top-k save · 20 epochs completed" (a file a record of its run names among
    its best-k, `CheckpointItem.in_best_k`: never from a file's name), else
    the file's own name; a file that holds weights only, or cannot be read,
    says so. How its stage ended, its stage, its Trainer validation figure,
    its selection, what it holds and its model fingerprint are the chosen
    file's details."""
    return r.fit_option(*_checkpoint_parts(item, best))


def _checkpoint_entries(parents: Mapping[str, Any],
                        keys: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    """**The one rule of what a run's files are called**, by key in `keys`'
    order: each file's two parts (`_checkpoint_parts`: what it is, how far it
    trained), the run's best saves by its recorded selection named so
    (`renderers.best_saves`, ties all); only files whose two parts read
    exactly alike are named by their file instead. `keys` are all of one
    run's files, so a file's words never depend on which of them a page
    shows. The Run → Checkpoint select (`checkpoint_choices`) and the model
    tree's rows (`_file_names`) both read it."""
    best = r.best_saves({key: parents[key] for key in keys})
    parts = {key: _checkpoint_parts(parents[key], key in best) for key in keys}
    seen = Counter(r.fit_option(*part) for part in parts.values())
    return {key: ((str(parents[key].get("filename")) if seen[r.fit_option(*part)] > 1
                   else part[0]), part[1]) for key, part in parts.items()}


def checkpoint_choices(parents: Mapping[str, Any], keys: Sequence[str]) -> List[Tuple[str, str]]:
    """A run's checkpoints as options, in `keys`' order: their words
    (`_checkpoint_entries`: `checkpoint_words`, the run's best saves by its
    recorded selection named so, ties all; two that would read alike named by
    their file instead, unique in a run, `renderers.fit_option` keeping both
    its ends), and files whose shortened names still read alike are numbered
    in the list's order, "(2)"; still one separator and
    `renderers.OPTION_CHARS` at most. No two options read the same (the WebUI
    skill §8); each value is the file's key, whatever the words; the chosen
    file's full name is in its details."""
    return r.fit_options([(what, rest, key) for key, (what, rest)
                          in _checkpoint_entries(parents, keys).items()])


def pair_words(parents: Mapping[str, Any]) -> Dict[str, Tuple[str, str]]:
    """**Each listed checkpoint by the words of its Run → Checkpoint pair**,
    by its key: its run's option (`run_choices`) and its own option in that
    run (`checkpoint_choices`). No two listed files share both (each select's
    options are unique, the WebUI skill's §8), so a page that names a file
    other than the chosen one, where two files of one run would otherwise
    read alike (the Test page's reports table), names it by what the
    operator would choose it by."""
    runs = {value: label for label, value in run_choices(parents)}
    words: Dict[str, Tuple[str, str]] = {}
    for run, keys in runs_of(parents).items():
        for label, key in checkpoint_choices(parents, keys):
            words[key] = (runs[run], label)
    return words


def parents_update(parents: Mapping[str, Any], keep: Optional[str] = None, *,
                   first: bool = True, interactive: Optional[bool] = None) -> Tuple[Any, Any]:
    """The two selects for `parents`: the runs, and the chosen run's
    checkpoints. `keep` (a checkpoint's key) stays chosen while it is listed,
    and sets its run; otherwise the first run and its first checkpoint (its
    latest save), or with `first` False nothing until the operator picks a
    run (the Test page, which never measures a model it picked itself).
    `interactive`, when given, makes each select interactive while it has
    options (an operator's), else neither (None leaves them as they are)."""
    runs = runs_of(parents)
    listed = isinstance(keep, str) and keep in parents
    run = item_run_key(parents[keep]) if listed else (next(iter(runs), None) if first else None)
    keys = runs.get(run, []) if run else []
    value = keep if keep in keys else (keys[0] if keys else None)

    def allowed(has: bool) -> Dict[str, Any]:
        return {} if interactive is None else {"interactive": bool(interactive) and has}

    return (gr.update(choices=run_choices(parents), value=run, **allowed(bool(parents))),
            gr.update(choices=checkpoint_choices(parents, keys), value=value,
                      **allowed(bool(keys))))


def chosen_in_run(parents: Mapping[str, Any], run: Any, current: Any) -> Optional[str]:
    """**The checkpoint a run's choice selects**: `current` while it is one of
    the run's, else the run's first in the list's order (its latest save);
    None for a run not listed."""
    keys = runs_of(parents or {}).get(run or "", [])
    return current if current in keys else (keys[0] if keys else None)


def on_run_chosen(run, parents, current):
    """A run chosen: its checkpoints, the latest save first. The checkpoint
    chosen stays when it is this run's (opening the drawer sets both;
    `chosen_in_run`)."""
    keys = runs_of(parents or {}).get(run or "", [])
    return gr.update(choices=checkpoint_choices(parents or {}, keys),
                     value=chosen_in_run(parents, run, current))


def recorded_tests(ledgers: Mapping[str, Mapping[str, Any]],
                   contest: Mapping[str, Mapping[str, Any]]) -> Dict[str, r.RecordedTests]:
    """Each checkpoint's recorded tests as a state, from what
    `record_views.ledger_records` and `contest_states` read: how many records read as results, how many of
    them in a group whose runs differ and how many whose comparison is
    incomplete, counted apart (M5-3 review EVAL-3: one uncheckable record hid a
    known difference), and how many records cannot be read as results, counted
    apart from the results and never as one (K8-6e.1). Counts that describe;
    nothing is hidden or refused by them."""
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
                                   unchecked=unchecked,
                                   unreadable=len(entry.get("unreadable") or []))
    return out


def read_recorded(workspace_dir: Path, items: Sequence[Mapping[str, Any]], *,
                  environ: Optional[Mapping[str, str]] = None
                  ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """``(ledgers, contest)`` for the listed checkpoints: the ledger beside each
    (`record_views.ledger_records`, read once per directory, the shared
    implementation the results service reads by too) and every record key's
    state in its cohort version's report store (`contest_states`, one scan per
    version). The one read behind the rows, Models and the Diagnosis card."""
    from src.api.services.record_views import ledger_records

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
            "status_read": False, "runs_unread": 0}


#: The filters a family view is drawn for (`family_views`): the refused group
#: is the whole view's (``all``), and a family holds no refused file.
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


def _latest_checkpoints(workspace: str, latest: int, served: Optional[str], *,
                        environ: Optional[Mapping[str, str]] = None) -> Tuple[Any, int]:
    """**The latest runs' checkpoints, the others not opened** (BACKLOG 39,
    M5-3 review UI-2: the drawer read every checkpoint of the workspace to
    show four runs): the runs as listed without a read
    (`server_lists.listed_runs`), then `latest` of them at a time read
    (`list_checkpoints`' `run_keys`) until `latest` runs hold a model
    finetuned for a task (the runs the drawer shows; a base run is read and
    not shown) and the model in use (`served`, its full fingerprint) is among
    what was read, or every run is. Returns ``(listing, runs not read)``; the
    listing holds every entry that cannot be listed, as a full one does."""
    from src.api.services.server_lists import (
        Listing,
        ServerListRefused,
        list_checkpoints,
        listed_runs,
    )

    try:
        order = [key for key, _files in listed_runs(workspace, environ=environ)]
    except ServerListRefused:
        return list_checkpoints(workspace, environ=environ), 0
    read, wanted = 0, latest
    found: Dict[str, Any] = {}
    while True:
        # The runs still wanted: as many as the shown runs lack (a base run
        # read is not shown), or `latest` more while the model in use is not
        # among them.
        batch = order[read:read + wanted]
        read += len(batch)
        listing = list_checkpoints(workspace, environ=environ, run_keys=batch)
        if not listing.available:
            return listing, 0
        for item in listing.items:
            found.setdefault(item.key, item)
        task_runs = {i.run_key for i in found.values() if matches(asdict(i), "task_models")}
        has_served = not served or any(i.sha256 == served for i in found.values())
        if read >= len(order) or (len(task_runs) >= latest and has_served):
            break
        wanted = latest - len(task_runs) if len(task_runs) < latest else latest
    rank = {key: index for index, key in enumerate(order)}
    items = sorted(found.values(), key=lambda i: rank.get(i.run_key, len(order)))
    return Listing(tuple(items)), len(order) - read


def list_models(workspace: Any, request: Any, *, family: bool = False,
                latest_runs: Optional[int] = None,
                environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The listing for one session: D8 on the triggering request first (listing
    opens checkpoints), then the server's lists and the status, as plain data
    for a `gr.State`. A refused request lists nothing and says why. With
    `family` (the research preset), the family view of the same listing too
    (``listing["family"]``, `family_views`), which needs every run's records.
    With `latest_runs` (the drawer, until "Show older runs"), only the latest
    runs' files are opened (`_latest_checkpoints`), and ``runs_unread`` says
    how many runs were not."""
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
    listing = {**empty_listing(listed["problem"]), "notice": listed.get("notice"),
               "workspaces": rows, "workspace": chosen,
               "served_sha": served_checkpoint_sha(status), "serving": serving,
               "status_problem": status_problem, "status_read": True}
    if chosen is None:
        return listing
    unread = 0
    if latest_runs is not None and not family:
        served = listing["served_sha"] if serving == chosen else None
        checkpoints, unread = _latest_checkpoints(chosen, latest_runs, served, environ=environ)
    else:
        checkpoints = list_checkpoints(chosen, environ=environ)
    listing["runs_unread"] = unread
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
    # The ledgers beside the checkpoints and their records' group descriptions,
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


def _view(view: Any) -> Mapping[str, Any]:
    return view if isinstance(view, Mapping) else {}


def matches(item: Mapping[str, Any], name: str) -> bool:
    """Whether a listed checkpoint is a row of the filter `name`
    (`TASK_FILTERS`), judged by its model record through the server's one
    predicate (`server_lists.record_matches`): a base, a model finetuned for
    that task, or any model finetuned for a task. A file whose record was
    refused or not read is a row of ``all`` only."""
    if name == "all":
        return True
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


#: The parts of a view that name rows by key, read from one workspace's listing.
_KEYED_VIEW_PARTS = ("keys", "groups", "context", "parents", "edges", "bounds")


def view_for_listing(view: Any, previous: Any, listing: Mapping[str, Any]) -> Any:
    """The page's view as it applies to `listing` (a listing just read is
    drawn before the page shapes a new view for it): what the view names
    by key (a family's rows, an order's groups, its context rows, parents, edges and
    boundaries) was read from the workspace the view names (its
    ``workspace``), else from the `previous` listing's, so for another
    workspace's listing it is left out and the view's filter and note apply.
    Another workspace's keys would hide every row of this one. Rows drawn so
    are not what the page shaped the view for, so its ``shaped_for`` goes too
    (`view_marker`): the page's controls are not applied until it shapes the
    view for this listing."""
    current = _view(view)
    if not any(part in current for part in _KEYED_VIEW_PARTS):
        return view
    read_from = current.get("workspace", previous.get("workspace")
                            if isinstance(previous, Mapping) else None)
    if read_from == listing.get("workspace"):
        return view
    return {k: v for k, v in current.items() if k not in (*_KEYED_VIEW_PARTS, "shaped_for")}


def view_marker(view: Any) -> str:
    """What the rows say they were drawn for: the view's ``shaped_for`` as a
    hidden element (``span.shep-view-for``, its ``data-for``, escaped);
    nothing for a view that names nothing."""
    shaped = _view(view).get("shaped_for")
    if not isinstance(shaped, str) or not shaped:
        return ""
    return f'<span class="shep-view-for" hidden data-for="{r.e(shaped)}"></span>'


#: **The key a listing committed to a session carries its generation under**
#: (`ListClock`): which reading of a workspace in this session it is, a new
#: one each time another workspace's listing is committed. A count of the
#: session's own events only, never part of a checkpoint, a ledger or a record.
GENERATION = "generation"


def list_generation(value: Any) -> Optional[int]:
    """The generation a listing carries (``listing[GENERATION]``) or a page's
    act was sent with (the rows' ``data-gen``, as text), or None where none
    is said or it is no whole number."""
    if isinstance(value, Mapping):
        value = value.get(GENERATION)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def generation_marker(listing: Mapping[str, Any]) -> str:
    """What the rows say they were drawn from: the generation of their
    listing as a hidden element (``span.shep-list-gen``, its ``data-gen``),
    which every act made in them is sent with (`MODEL_LIST_JS`, `FAMILY_JS`),
    so the server refuses one made in rows of another workspace
    (`ListClock.current`); nothing for a listing no session committed."""
    generation = list_generation(listing)
    if generation is None:
        return ""
    return f'<span class="shep-list-gen" hidden data-gen="{generation}"></span>'


def refused_line(listing: Mapping[str, Any], *, control: Optional[str] = None) -> str:
    """Why a refused listing lists nothing, as every page says an operator
    control that is off for this request (`chips.operator_only`, R-WHY's
    ``d8`` phrase, HTML): "Operator controls only", which opens the D8
    chip's whole reason (the condition D8 named is said there, once).
    `control`: the element id of the control it explains."""
    from src.webui.components.chips import operator_only

    return operator_only(listing.get("condition"), control=control)


#: Said beside Load when the list read again no longer lists the checkpoint
#: selected (`kept_selection`, `ListClock.commit_listing`): what the list
#: shows now, never why (the file may be elsewhere, or unreadable).
SELECTION_GONE = "The checkpoint selected before is not in the list now. Select one again."


def kept_selection(previous: Any, listing: Mapping[str, Any], key: Any,
                   view: Any = None, preset: Optional[Preset] = None, *,
                   search: Any = "") -> Optional[str]:
    """**The selection a new listing keeps** (D04; the owner's rule after the
    recheck of bd5e678). Of the same workspace as `previous` (the list read
    again): `key` while the listing still lists that checkpoint among the rows
    its preset ever shows, whatever the view's filter, the search or the fold
    state show (a selection the list does not show says so, `outside_html`,
    and keeps its details); else None: a selection gone is cleared, never
    replaced by another file, not even a lone candidate (the clock says so
    beside Load, `SELECTION_GONE`), and a re-read with nothing selected
    chooses nothing. A first listing (`previous` None, or another workspace):
    the lone row shown, when exactly one is and it can be loaded (R-27: a
    lone candidate is no dead end; a preselection loads nothing); else None."""
    if listing.get("refused"):
        return None
    if isinstance(previous, Mapping) and previous.get("workspace") == listing.get("workspace"):
        item = find_item(listing, key) if key else None
        return key if item is not None and item in shown_items(listing, None, preset) else None
    shown = [i for i in shown_items(listing, view, preset) if search_hit(i, search)]
    return lone_key(shown, can_load)


def lone_key(items: Sequence[Mapping[str, Any]],
             admits: Callable[[Mapping[str, Any]], bool]) -> Optional[str]:
    """**A lone candidate is chosen** (R-27, the one rule): the key of the
    only entry of `items` when `admits` takes it (the list's: it can be
    loaded, `can_load`; the Test page's: it can be tested), else None. With
    one choice there is nothing to choose between, so a page that otherwise
    waits for the operator's pick does not leave a dead end; with two or more
    nothing is chosen for them."""
    return item_key(items[0]) if len(items) == 1 and admits(items[0]) else None


def runs_unread(listing: Mapping[str, Any], show_older: bool) -> int:
    """How many runs of the workspace the list has not read (the drawer's
    first listing, BACKLOG 39: ``runs_unread``) while "Show older runs" was
    not asked: what the list shows, searches and filters is its latest runs
    only, and its words say so (`count_line`, the empty state)."""
    return 0 if show_older else int(listing.get("runs_unread") or 0)


#: **The list's empty states, per cause** (the skill's §9: what will appear and
#: the next step, in the flow, `.shep-empty`): no workspace in the workspaces
#: folder, none chosen, no checkpoint in the workspace chosen. The workspaces
#: folder not set (or not usable) is its notice instead (`_no_items_html`).
NO_WORKSPACE = "No workspace in the workspaces folder yet. Build a dataset first (Research › Datasets)."
NO_CHECKPOINT = "No checkpoint is listed in this workspace."
#: The list's state where its page says the workspaces folder's notice elsewhere
#: (`Preset.notice_where`): followed by that notice's phrase (`renderers.reason_phrase`).
NO_MODEL_TO_LIST = "No model to list."


def _no_items_html(listing: Mapping[str, Any], view: Any, preset: Optional[Preset],
                   search: Any = "", show_older: bool = True) -> str:
    """Why the list shows no row, in its own words per cause, in the flow
    (`.shep-empty`, the skill's §9): the search, the filter (each "in this
    workspace", or "in the latest runs" with what "Show older runs" does,
    drawn under it, while runs are not read), nothing listed. The workspaces
    folder not set or not usable is said by its one notice
    (`renderers.notice_block`, from the listing's ``notice``: its state, the
    way to Settings › Data locations, the rest under Details), the list's
    empty state on every page that draws it; an empty folder and a workspace
    with no checkpoint in their own words (`NO_WORKSPACE`, `NO_CHECKPOINT`)."""
    chosen = _rows_filter(view, preset)
    query = str(search or "").strip()
    unread = bool(runs_unread(listing, show_older))
    what = FILTER_WORDS[chosen][1]
    if listing.get("items") and query:
        why = (f"No {what} matches “{query}” in the latest runs. Show older runs searches the "
               "rest." if unread else f"No {what} matches “{query}” in this workspace.")
    elif listing.get("items") and chosen != "all":
        why = (f"No {what} is listed in the latest runs. Show older runs reads the rest."
               if unread else f"No {what} is listed in this workspace.")
    elif listing.get("notice") and not listing.get("items"):
        notice = listing["notice"]
        phrase = r.reason_phrase(notice) if preset is not None and preset.notice_where else ""
        if phrase:
            # Said once on the screen, where the preset's page says it: here
            # the list's state and the notice's own phrase (R-WHY), which
            # brings that notice into view and focuses its link, never a
            # pointer to a place that may have scrolled away (the owner's D6).
            return (f'<p class="shep-empty" data-notice-rel="{r.e(notice.get("key") or "")}">'
                    f'{r.e(NO_MODEL_TO_LIST)} {phrase}</p>')
        return r.notice_block(notice)
    elif listing.get("problem"):
        why = listing["problem"]
    elif listing.get("status_read") and listing.get("workspace") is None \
            and not listing.get("workspaces"):
        why = NO_WORKSPACE
    else:
        why = NO_CHECKPOINT
    return f'<p class="shep-empty">{r.e(why)}</p>'


def list_html(listing: Mapping[str, Any], selected: Optional[str], preset: Preset, *,
              show_older: bool = False, view: Any = None, search: Any = "",
              fold: Optional[Mapping[str, Any]] = None) -> str:
    """The list as the preset draws it: the one tree (D04: `tree_view`,
    `_tree_html`), the latest runs then "Show older runs" where the preset
    shows the latest only (`older_html`); the view's filter or rows and
    groups, the search (`search_hit`), the fold state, a selection the list
    does not show said (`outside_html`), what the view was shaped for
    (`view_marker`), an empty list's too, and the listing's generation
    (`generation_marker`)."""
    if listing.get("refused"):
        # A listing is no act: the short line every page uses for an operator
        # control off for this request, not "Refused … Nothing was done."
        return f'<div class="shep-reason shep-reason-grey">{refused_line(listing)}</div>'
    return (_list_body(listing, selected, preset, show_older=show_older, view=view,
                       search=search, fold=fold) + view_marker(view)
            + generation_marker(listing))


def _list_body(listing: Mapping[str, Any], selected: Optional[str], preset: Preset, *,
               show_older: bool, view: Any, search: Any = "",
               fold: Optional[Mapping[str, Any]] = None) -> str:
    tree = tree_view(listing, view, preset, search=search, fold=fold, selected=selected,
                     show_older=show_older)
    # "Show older runs" under an empty list too: a search or a filter that
    # finds nothing in the latest runs may find it in the runs not read yet.
    # A view that carries its runs' edges (Models' By family) is drawn as the
    # family graph, from the same tree (`_graph_html`).
    if not tree.groups:
        body = _no_items_html(listing, view, preset, search, show_older)
    elif isinstance(_view(view).get("edges"), (list, tuple)):
        body = _graph_html(listing, selected, preset, view, tree, fold)
    else:
        body = _tree_html(listing, selected, preset, view, tree)
    return outside_html(listing, tree, view) + body + older_html(listing, tree, show_older)


def older_html(listing: Mapping[str, Any], tree: TreeView, show_older: bool) -> str:
    """**"Show older runs"** under the tree of a preset that shows its latest
    runs only (the drawer, BACKLOG 39): how many runs it leaves out, "up to N
    more" while some were not read yet (a run not read may hold no task model,
    a base run, so its count is the most it can add); nothing once every run
    is shown."""
    unread = runs_unread(listing, show_older)
    if not tree.cut and not unread:
        return ""
    more = f"up to {tree.cut + unread} more" if unread else str(tree.cut)
    return (f'<button type="button" class="shep-row shep-row-more" data-key="__older__">'
            f'Show older runs ({more})</button>')


# ---------------------------------------------------------------------------
# The tree (D04): families, runs folded, their saved files
# ---------------------------------------------------------------------------
#: The tree's columns (Models; the WebUI skill §8: one or two words each). No
#: score column: validations of different tasks are never put on one scale,
#: and a file's figures are its details.
TREE_COLUMNS = ("Model", "Task", "Operation", "Epochs", "Files", "Fingerprint")
#: **The tree's optional columns** (the UI sweep's M6: a table drops them
#: before it scrolls sideways): what the selected model's details say again
#: (its epochs, its run's saved files, its fingerprint), so the table needs
#: only Model, Task and Operation. Their cells carry `COL_OPT`; the table is
#: a fit box's wide form (`style.FIT`, `style.FIT_WIDE`): while it does not
#: fit with every column, the shell's layout pass marks its frame narrow and
#: the shared style sheet draws no optional cell; when even that does not fit,
#: the frame scrolls sideways, its own region.
OPTIONAL_COLUMNS = ("Epochs", "Files", "Fingerprint")
#: A hex query this long or longer is also read as a fingerprint's start.
FINGERPRINT_PREFIX = 4
_HEX = re.compile(r"[0-9a-f]+")


def run_node(run_key: str) -> str:
    """A run's fold node (the fold state's key, `TreeView`)."""
    return f"r:{run_key}"


def group_node(name: Any) -> str:
    """A group's node: a family by its id, any other group by its title."""
    return f"g:{name}"


def subtree_node(run_key: str) -> str:
    """A run's subtree in the family graph (`_graph_html`): unfolded, its
    child runs are shown. Apart from `run_node` (a run's files)."""
    return f"t:{run_key}"


def search_node(node: str) -> str:
    """The fold state's key of what the operator folded or unfolded while a
    search showed the list: kept for that search only (`fold_open`; the
    search box's handler drops every such key when the search changes), so
    clearing the search brings the operator's own folds back (G7)."""
    return f"q:{node}"


#: The act a page's region sends to pick a listed checkpoint through the
#: list's own selection, as a row's click does (`renderers.TRIGGER_JS`:
#: ``{act: "pick", key}``; Models' details, their "Saved files" row).
PICK_ACT = "pick"


def fold_open(fold: Mapping[str, Any], node: str, query: str, found: bool,
              default: bool) -> bool:
    """**Whether a node is unfolded**, one rule for the tree's runs and the
    graph's subtrees: under a search, what the operator folded during it
    (`search_node`), else unfolded where the search found something it
    hides (`found`: revealed at drawing, never written to the fold state);
    else what the operator folded (`fold[node]`), else `default`."""
    if query and isinstance(fold.get(search_node(node)), bool):
        return fold[search_node(node)]
    if query and found:
        return True
    value = fold.get(node)
    return value if isinstance(value, bool) else default


def search_hit(item: Mapping[str, Any], query: Any) -> Optional[str]:
    """**How a listed checkpoint matches a search** (D04): ``"fingerprint"``
    when the query, at least `FINGERPRINT_PREFIX` hex digits, starts its model
    fingerprint (or is all of it); ``"file"`` when the file's name holds it;
    ``"run"`` when its run's directory name or its run words hold it; each
    compared ignoring case, as plain text (no pattern, no fuzzy service).
    None when it does not match; ``"all"`` for an empty query."""
    text = str(query or "").strip().lower()
    if not text:
        return "all"
    sha = item.get("sha256")
    if (len(text) >= FINGERPRINT_PREFIX and _HEX.fullmatch(text) and isinstance(sha, str)
            and sha.lower().startswith(text)):
        return "fingerprint"
    if text in str(item.get("filename") or "").lower():
        return "file"
    run = str(item.get("run") or "")
    words = r.run_words(item.get("architecture"), run, fit=False)
    if text in run.lower() or text in words.lower():
        return "run"
    return None


class TreeRun(NamedTuple):
    """One run of the tree: its key, its files shown (keys, the listing's
    order), how many files it lists and whether it is shown only for a shown
    run that descends from it (`context`). Where it sits in its family is the
    view's (its ``edges``, `_graph_html`)."""

    key: str
    files: Tuple[str, ...]
    total: int
    context: bool


class TreeGroup(NamedTuple):
    """One group of the tree (a family, or the files whose record is not
    readable or not read, titled by their kind's word): its node, its title
    and its runs; a group with no title is the list itself (Newest first) or a
    family, whose root node names it."""

    node: str
    title: str
    runs: Tuple[TreeRun, ...]
    family: bool = False


class TreeView(NamedTuple):
    """What the tree shows for one listing, view, search and fold state: its
    groups, the keys of its matches (context rows apart), the runs and groups
    unfolded, and the selection when the tree does not show it; where the
    preset shows its latest runs only (`Preset.latest_runs`), how many runs
    the listing holds that it leaves out (`cut`) and whether the run of the
    model in use is shown beyond them (`served_run`)."""

    groups: Tuple[TreeGroup, ...]
    matched: Tuple[str, ...]
    context: Tuple[str, ...]
    open_nodes: frozenset
    outside: Optional[str]
    query: str
    cut: int = 0
    served_run: bool = False


def _runs_of_keys(keys: Sequence[str], by_key: Mapping[str, Mapping[str, Any]]
                  ) -> List[Tuple[str, List[str]]]:
    runs: Dict[str, List[str]] = {}
    for key in keys:
        runs.setdefault(item_run_key(by_key[key]), []).append(key)
    return list(runs.items())


def tree_view(listing: Mapping[str, Any], view: Any = None, preset: Optional[Preset] = None, *,
              search: Any = "", fold: Optional[Mapping[str, Any]] = None,
              selected: Any = None, show_older: bool = True) -> TreeView:
    """**The tree the shared list draws** (D04), from the one listing and the
    page's view: its groups (the view's, each with its runs in lineage order;
    else one group of every run, the listing's order), each run's files, the
    search's matches with the files they come from (`search_hit`; the view's
    ``parents``, the exact parent files, so a match keeps its ancestors, as
    context), and which runs' files are unfolded (`fold_open`): under a search,
    a run holding a file the search found by its own name or fingerprint
    (revealed while it is shown, the operator's own folds untouched: G7);
    else where the fold state names it (`fold`, ``{node: bool}``, what the
    operator unfolded or folded), else where it holds the selection or the
    model in use. A preset that shows its latest runs only (the drawer's
    `Preset.latest_runs`, until `show_older`) keeps the first of them in the
    listing's order (newest first) and the run of the model in use, and
    counts the others (`TreeView.cut`). Nothing here changes what is
    selected: a selection the tree does not show is `outside`. The family
    graph (`_graph_html`) draws this same tree."""
    items = shown_items(listing, view, preset)
    by_key = {item_key(i): i for i in items}
    order = {key: index for index, key in enumerate(by_key)}
    query = str(search or "").strip()
    hits = {key: hit for key, item in by_key.items()
            if (hit := search_hit(item, query)) is not None}
    parents = _view(view).get("parents") if isinstance(_view(view).get("parents"), Mapping) else {}
    shown = set(hits)
    if query:
        for key in list(hits):
            parent = parents.get(key)
            while isinstance(parent, str) and parent in by_key and parent not in shown:
                shown.add(parent)
                parent = parents.get(parent)
    view_context = set(_view(view).get("context") or ())
    context = {key for key in shown if key not in hits or key in view_context}
    listed_runs: Dict[str, int] = Counter(item_run_key(i) for i in listing.get("items") or [])

    def run_of(key: str, files: Sequence[str]) -> TreeRun:
        return TreeRun(key, tuple(sorted(files, key=order.get)), listed_runs.get(key, 0),
                       all(f in context for f in files))

    groups: List[TreeGroup] = []
    placed: set = set()
    for group in (_view(view).get("groups") or []) if isinstance(
            _view(view).get("groups"), (list, tuple)) else []:
        if not isinstance(group, Mapping):
            continue
        keys = [k for k in group.get("keys") or () if k in shown and k not in placed]
        placed.update(keys)
        runs_given = [g for g in group.get("runs") or () if isinstance(g, Mapping)]
        files = dict(_runs_of_keys(keys, by_key))
        runs = [run_of(g["run"], files.pop(g["run"]))
                for g in runs_given if g.get("run") in files]
        runs += [run_of(run, keys_) for run, keys_ in files.items()]
        if runs:
            groups.append(TreeGroup(str(group.get("node") or group_node(group.get("title"))),
                                    str(group.get("title") or ""), tuple(runs),
                                    bool(group.get("family"))))
    rest = [k for k in by_key if k in shown and k not in placed]
    if rest:
        groups.append(TreeGroup(group_node("list"), "",
                                tuple(run_of(run, keys) for run, keys in
                                      _runs_of_keys(rest, by_key))))

    served = listing.get("served_sha")
    cut, served_run = 0, False
    latest = preset.latest_runs if preset is not None else None
    if latest is not None and not show_older:
        # The latest runs (the listing's order), and the run of the model in
        # use wherever it is: the drawer locates it (BACKLOG 39, D04).
        order = list(dict.fromkeys(run.key for group in groups for run in group.runs))
        in_use = {run.key for group in groups for run in group.runs
                  if served and any(by_key[k].get("sha256") == served for k in run.files)}
        keep = set(order[:latest]) | in_use
        served_run = bool(in_use - set(order[:latest]))
        cut = len(order) - len(keep)
        groups = [g._replace(runs=tuple(run for run in g.runs if run.key in keep))
                  for g in groups]
        groups = [g for g in groups if g.runs]
    drawn = {key for group in groups for run in group.runs for key in run.files}
    fold = fold if isinstance(fold, Mapping) else {}
    open_nodes = set()
    for group in groups:
        for run in group.runs:
            holds = any(key == selected or (served and by_key[key].get("sha256") == served)
                        for key in run.files)
            found = any(hits.get(key) in ("fingerprint", "file") and key not in context
                        for key in run.files)
            if fold_open(fold, run_node(run.key), query, found, holds):
                open_nodes.add(run_node(run.key))
    outside = None
    item = find_item(listing, selected)
    if item is not None and selected not in drawn:
        outside = selected
    return TreeView(tuple(groups), tuple(k for k in by_key if k in drawn and k not in context),
                    tuple(k for k in by_key if k in drawn and k in context),
                    frozenset(open_nodes), outside, query, cut, served_run)


def _run_summary(listing: Mapping[str, Any], run: TreeRun) -> Mapping[str, Any]:
    """The file a run's summary is read from, among the files it shows (a
    group of files whose record is not read shows those only): its latest
    save, else its newest read file (a run's files share its record: task,
    operation, family), else its first."""
    items = [find_item(listing, key) or {} for key in run.files]
    read = [i for i in items if loadable(i)]
    latest = [i for i in read if i.get("filename") == "last.pt"]
    return (latest or read or items or [{}])[0]


def _epochs_cell(items: Sequence[Mapping[str, Any]]) -> str:
    """How far a run (or one file) trained: the most epochs a file records
    completed (`completed_epochs`), else the words its file says
    (`renderers.epoch_words`: "recorded epoch 3", "epochs not recorded"),
    on one line while the table has the room (`style.CELL_ROOMY`)."""
    counts = [i.get("completed_epochs") for i in items
              if isinstance(i.get("completed_epochs"), int)
              and not isinstance(i.get("completed_epochs"), bool)]
    if counts:
        return str(max(counts))
    return (f'<span class="shep-muted {CELL_ROOMY}">'
            f'{r.bound_html(r.epoch_words(items[0] if items else {}))}</span>')


def _file_names(items: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, str]:
    """Each file of one run by what it is, the words its Run → Checkpoint
    option leads with (`_checkpoint_entries`: "Latest save", "Best pooled
    MRR" by the run's recorded selection, ties all, "Top-k save", else its
    name; a file's name only where two files would read exactly alike, how far
    they trained included), without how far it trained, which the Epochs
    column says. `keys` are every file the listing lists in the run, so a
    filter or a search never changes what a shown file is called."""
    return {key: what for key, (what, _rest) in _checkpoint_entries(items, keys).items()}


def _fold_button(node: str, opened: bool, label: str) -> str:
    return (f'<button type="button" class="shep-fold" data-fold="{r.e(node)}" '
            f'aria-expanded="{"true" if opened else "false"}" '
            f'aria-label="{r.e(label)}"><span aria-hidden="true">'
            f'{"▾" if opened else "▸"}</span></button>')


class _Files(NamedTuple):
    """What the list draws its saved files' rows from (`_file_cells`), read
    once per drawing: the listing's items by key, its runs' files, the hex
    digits that tell the shown fingerprints apart, the workspaces and the
    digest in use."""

    items: Mapping[str, Mapping[str, Any]]
    run_files: Mapping[str, List[str]]
    digits: int
    workspaces: Sequence[Mapping[str, Any]]
    served: Any


def _files(listing: Mapping[str, Any], shown_keys: Sequence[str]) -> _Files:
    items = {item_key(i): i for i in listing.get("items") or []}
    return _Files(items, runs_of(items),
                  r.distinguishing_digits(items[k].get("sha256") for k in shown_keys),
                  listing.get("workspaces") or [], listing.get("served_sha"))


def _file_cells(files: _Files, key: str, words: str, selected: Optional[str], preset: Preset,
                within: Optional[str] = None, hidden: bool = False
                ) -> Tuple[str, str, Optional[Tuple[str, str]]]:
    """**One saved file as the list draws it** (the tree's file rows): its
    row's attributes (``data-key``, the one mark of a row the page may
    choose, a tab stop, ``aria-current`` on the selection: never
    ``aria-selected`` on a plain table's row, the skill's §8), its name cell
    (what it is, `_file_names`, and "In use" on the served
    one), and its other cells' contents: ``(epochs, fingerprint with the
    dataset only where it is not the workspace's, or None)`` for a file read,
    None for one not readable, whose state and why the caller says in words
    (`renderers.unreadable_words`, `renderers.reason_lead`). An unreadable
    file is chosen like any other (never Loaded: `can_load`), so its details
    say its reason whole, the loader's reference with it
    (NEW-models-results-4)."""
    item = files.items[key]
    unreadable = item.get("problem") is not None or not r.is_digest(item.get("sha256"))
    in_use = bool(files.served) and item.get("sha256") == files.served
    tag = f' {r.chip(preset.in_use_label, "accent")}' if in_use else ""
    classes = ["shep-tree-file"] + (["shep-on"] if key == selected else []) + (
        ["shep-row-dis"] if unreadable else [])
    attrs = (f'class="{" ".join(classes)}" data-key="{r.e(key)}"'
             + (f' data-in="{r.e(within)}"' if within else "") + ' tabindex="0"'
             + (' aria-current="true"' if key == selected else "") + (" hidden" if hidden else ""))
    name = f'<div class="shep-tree-name shep-tree-leaf">{r.whole_words(words)}{tag}</div>'
    if unreadable:
        return attrs, name, None
    exception = r.dataset_exception(item)
    if exception and r.is_synthetic_training(item.get("dataset_fingerprint"), files.workspaces):
        exception += " " + r.chip("synthetic", "amber")
    dataset = f'<div class="shep-sub">{exception}</div>' if exception else ""
    fingerprint = r.fingerprint_value(item.get("sha256"), what="model fingerprint",
                                      digits=files.digits, copy=False)
    return attrs, name, (_epochs_cell([item]), fingerprint + dataset)


def _unreadable_words(item: Mapping[str, Any]) -> str:
    """A file not readable, in its row's words: its state and why, the
    loader's reference left to its details (`renderers.reason_lead`;
    Models' detail, the drawer's facts), one click away."""
    state = r.unreadable_words(item)
    why = r.reason_lead(item.get("problem")) or "its digest was not read"
    return (f'<span class="shep-sub">{r.e(state[:1].upper() + state[1:])}: {r.e(why)}. '
            'It cannot be loaded.</span>')


def _tree_html(listing: Mapping[str, Any], selected: Optional[str], preset: Preset,
               view: Any, tree: TreeView, *, locate: bool = True) -> str:
    """**The tree as a table** (D04: Newest first, the drawer, and the files
    the family graph cannot place): each group its own ``<tbody>`` (B72-C5);
    in it each run's summary row (its run words, its task, operation, epochs
    completed and saved files, `operation_words`), its fold toggle unfolding
    its files: each a row of its own, selectable one by one (`_file_cells`),
    by what it is (`_file_names`), how far it trained and its model
    fingerprint, "In use" on the served one. A folded row is in the table,
    hidden, so nothing is dropped by folding. An entry the listing could not
    list reads its own name and where it sits (`renderers.entry_place`) and
    why, and is no choice: a plain row, with no key the page could choose, no
    hover and no tab stop (V-21: it looked choosable, and a click on it left
    the details on the model chosen before). Epochs, Files and Fingerprint
    are optional columns (`OPTIONAL_COLUMNS`): the table drops them while it
    does not fit its frame, before it scrolls sideways. The table names the row it
    locates (``data-locate``: the selection, else the model in use) unless
    `locate` is off and it shows neither."""
    shown_keys = [k for g in tree.groups for run in g.runs for k in run.files]
    files = _files(listing, shown_keys)
    items = files.items
    # Each run by its whole words (the table has room for them: no option's
    # cut), told apart from the others' where they would read alike
    # (`renderers.distinct_names`: the start to the second, then its own id).
    names = r.distinct_names(
        {run.key: (items[run.files[0]].get("architecture"), items[run.files[0]].get("run"))
         for g in tree.groups for run in g.runs},
        lambda parts, level: " · ".join(
            p for p in r.run_parts(parts[0], parts[1], level, fit=False) if p))
    span = len(TREE_COLUMNS)

    def th_classes(h: str) -> str:
        found = (["shep-num"] if h in ("Epochs", "Files") else []) + (
            [COL_OPT] if h in OPTIONAL_COLUMNS else [])
        return f' class="{" ".join(found)}"' if found else ""

    head = "".join(f'<th scope="col"{th_classes(h)}>{r.e(h)}</th>' for h in TREE_COLUMNS)
    opt, num_opt = f' class="{COL_OPT}"', f' class="shep-num {COL_OPT}"'
    bodies = []
    for group in tree.groups:
        rows = []
        for run in group.runs:
            node = run_node(run.key)
            run_open = node in tree.open_nodes
            first = items[run.files[0]]
            place = r.entry_place(first)
            if place is not None:
                # A directory the listing could not list: its own name and
                # where it sits, the reason in its row (BACKLOG 66, 69). It
                # holds no model to show or load, so it is no choice (V-21):
                # no ``data-key``, the one mark of a row the page may choose.
                why = first.get("problem") or "not read"
                rows.append(
                    f'<tr class="shep-tree-entry shep-row-dis">'
                    f'<td><span class="shep-w600">{r.whole_words(first.get("run"))}</span>'
                    f'<div class="shep-sub">{r.e(place)}</div></td>'
                    # Why, in words on the page (B69 L3: said once, never only
                    # on hover), naming no server path (the listing's redacted).
                    f'<td colspan="{span - 1}"><span class="shep-sub">{r.e(why)}</span></td></tr>')
                continue
            summary = _run_summary(listing, run)
            # The run by its words, told apart from the other runs' (as every
            # Run select names them): its name first, then its layout and start.
            # Its layout and start one fact, kept on one line (a time never parts from its date),
            # after the shared separator (never drawn at a line's end).
            what, _sep, rest = names[run.key].partition(" · ")
            label = (f'<span class="shep-w600">{r.whole_words(what)}</span>'
                     + (f'{r.SEP}<span class="shep-muted shep-nowrap">{r.e(rest)}</span>'
                        if rest else ""))
            count = (f"{len(run.files)} of {run.total}" if len(run.files) < run.total
                     else str(run.total))
            rows.append(
                f'<tr class="shep-tree-run" data-fold="{r.e(node)}">'
                f'<td><div class="shep-tree-name">'
                f'{_fold_button(node, run_open, "Show the saved files of " + what)}'
                f'<span class="shep-tree-label">{label}</span></div></td>'
                f'<td class="{CELL_WHOLE}">{r.whole_words(r.record_kind_text(summary))}</td>'
                f'<td class="{CELL_WHOLE}"><span class="shep-op">'
                f'{r.whole_words(r.operation_words(summary))}</span></td>'
                f'<td{num_opt}>{_epochs_cell([items[k] for k in run.files])}</td>'
                f'<td{num_opt}>{r.e(count)}</td><td{opt}></td></tr>')
            words = _file_names(items, files.run_files[run.key])
            for key in run.files:
                attrs, name, cells = _file_cells(files, key, words[key], selected, preset,
                                                 node, not run_open)
                if cells is None:
                    rows.append(f'<tr {attrs}><td>{name}</td><td colspan="{span - 1}">'
                                f'{_unreadable_words(items[key])}</td></tr>')
                    continue
                epochs, fingerprint = cells
                # How far it trained and its fingerprint, under its name while the Epochs and
                # Fingerprint columns are dropped: each a unit of one chain (the sweep's L7
                # check, SW-LK4: at 1366 and 1280 no fingerprint was in the list, which is
                # searched by name or fingerprint).
                print_ = r.fingerprint_value(items[key].get("sha256"), what="model fingerprint",
                                             digits=files.digits, copy=False)
                twin = (f'<div class="{COL_TWIN} shep-sub">'
                        f'<span class="shep-fact">{r.bound_html(r.epoch_words(items[key]))}</span>'
                        f'{r.SEP}<span class="shep-fact">{print_}</span></div>')
                rows.append(f'<tr {attrs}><td>{name}{twin}</td><td></td><td></td>'
                            f'<td{num_opt}>{epochs}</td><td{opt}></td>'
                            f'<td{opt}>{fingerprint}</td></tr>')
        bodies.append(f'<tbody data-node="{r.e(group.node)}">{"".join(rows)}</tbody>')
    served = files.served
    found = selected if selected in shown_keys else next(
        (k for k in shown_keys if served and items[k].get("sha256") == served), "")
    mark = (f' data-locate="{r.e(found or "")}"' if locate or found else "") + (
        ' data-searched="1"' if tree.query else "")
    return (f'<div class="shep-tbl-wrap shep-mtbl-wrap {FIT}"{mark}>'
            f'<table class="shep-tbl shep-mtbl shep-mtree {FIT_WIDE}" aria-label="Models">'
            f'<thead><tr>{head}</tr></thead>{"".join(bodies)}</table></div>')


# ---------------------------------------------------------------------------
# The family graph (Models' By family; G1-G9, I1-I10)
# ---------------------------------------------------------------------------
#: The name the graph's arrow is read by: what it shows or hides, a run's
#: child runs, counted; one name for both states, ``aria-expanded`` saying
#: which (the APG disclosure).
SUBTREE_LABEL = "Child runs of {name} ({count})"
#: On a folded node holding the selected model's run (I10): said where the
#: fold is, under the node's own text and arrow, so nothing moves under the
#: pointer that folded it.
HOLDS_SELECTION = "Selected model in this folded branch"
#: Above the graph while the selected model's run is in a folded branch (I10).
FOLDED_SELECTION = "Selected model is in a folded branch under"
#: The heading of the files the graph cannot place (no readable model record).
NOT_IN_A_FAMILY = "Not in a family"
#: The chevron of a node's arrow, turned by its state and the graph's
#: orientation (`style.SHELL_CSS`); drawn, never a glyph that a font changes.
_CHEVRON = ('<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" '
            'focusable="false"><path d="M6 3.5 10.5 8 6 12.5" fill="none" '
            'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
            'stroke-linejoin="round"/></svg>')


class _Graph(NamedTuple):
    """The family graph's shape for one drawing (`graph_shape`): the drawn
    runs, each run's exact parent file (the view's ``edges``) and its parent
    run as drawn, each run's children in the tree's order, each family's
    roots, the subtrees unfolded and the selection's run."""

    runs: Mapping[str, TreeRun]
    parent_key: Mapping[str, str]
    parent_run: Mapping[str, str]
    children: Mapping[str, List[str]]
    roots: Tuple[Tuple[TreeGroup, Tuple[str, ...]], ...]
    open_runs: frozenset
    selected_run: Optional[str]


def graph_shape(listing: Mapping[str, Any], view: Any, tree: TreeView,
                fold: Optional[Mapping[str, Any]], selected: Any) -> _Graph:
    """**The family graph from the one tree** (`tree_view`: its families, runs,
    files, search and context) and the view's ``edges`` (each run under the
    run of the exact parent file it records, `models_page.family_view`): a run
    whose parent run is not drawn (a boundary, a filter, a record naming its
    own descendant, G9) is a root; a cycle in a hand-made view is cut where it
    is met, never followed. A subtree is unfolded by the one fold rule
    (`fold_open`, `subtree_node`): a root by default, so each family shows its
    root and its direct children (G6); a run that holds the selection or the
    model in use is reached with its ancestors unfolded; under a search, the
    ancestors of every run it found, at drawing only (G7)."""
    items = {item_key(i): i for i in listing.get("items") or []}
    families = [g for g in tree.groups if g.family]
    runs = {run.key: run for g in families for run in g.runs}
    parent_key: Dict[str, str] = {}
    for edge in _view(view).get("edges") or ():
        if isinstance(edge, Mapping) and edge.get("to") in runs \
                and isinstance(edge.get("from"), str) and edge["from"] in items:
            parent_key.setdefault(edge["to"], edge["from"])
    parent_run = {run: item_run_key(items[key]) for run, key in parent_key.items()
                  if item_run_key(items[key]) in runs and item_run_key(items[key]) != run}
    listed: Dict[str, List[str]] = {}
    for key in runs:
        if key in parent_run:
            listed.setdefault(parent_run[key], []).append(key)
    # Each family drawn from its roots, every run once: a run under its parent
    # run, a run no root reaches (a cycle in a hand-made view) a root itself.
    children: Dict[str, List[str]] = {}
    roots = []
    seen: set = set()
    for group in families:
        order = [run.key for run in group.runs]
        tops: List[str] = []
        for key in order:
            if key in seen:
                continue
            if key in parent_run:
                up, path = key, set()
                while up in parent_run and up not in path:
                    path.add(up)
                    up = parent_run[up]
                if up not in path:
                    continue        # reached from its root, drawn there
            tops.append(key)
            seen.add(key)
            queue = [key]
            for at in queue:
                for child in listed.get(at, ()):
                    if child not in seen:
                        seen.add(child)
                        children.setdefault(at, []).append(child)
                        queue.append(child)
        roots.append((group, tuple(tops)))
    root_set = {key for _group, tops in roots for key in tops}
    # Each drawn run's parent run, as drawn (a tree: no cycle is followed).
    parent_run = {child: at for at, kids in children.items() for child in kids}

    def ancestors(key: str) -> set:
        found, at = set(), key
        while at in parent_run:
            at = parent_run[at]
            found.add(at)
        return found

    served = listing.get("served_sha")
    selected_run = item_run_key(items[selected]) if selected in items else None
    wanted: set = set()
    for key, run in runs.items():
        if key == selected_run or (served and any(items[f].get("sha256") == served
                                                  for f in run.files)):
            wanted |= ancestors(key)
    matched = set(tree.matched)
    found: set = set()
    if tree.query:
        for key, run in runs.items():
            if any(f in matched for f in run.files):
                found |= ancestors(key)
    fold = fold if isinstance(fold, Mapping) else {}
    open_runs = frozenset(key for key in runs if children.get(key) and fold_open(
        fold, subtree_node(key), tree.query, key in found, key in root_set or key in wanted))
    return _Graph(runs, parent_key, parent_run, children, tuple(roots), open_runs,
                  selected_run if selected_run in runs else None)


def drawn_files(listing: Mapping[str, Any], view: Any, preset: Optional[Preset], run: Any, *,
                search: Any = "", show_older: bool = True) -> Tuple[str, ...]:
    """**A run's files as the list draws them** (`tree_view`: the view's rows,
    a Task filter's context, a search's matches and the context it keeps), in
    the list's order: each one a pick the list's one selection accepts and
    shows. What is folded or selected changes none of them. Empty for a run
    the list does not draw. A run's node selects among these (`run_choice`),
    and the Models details offer these as its "Saved files"
    (`models_page.detail_parts`)."""
    tree = tree_view(listing, view, preset, search=search, show_older=show_older)
    return next((drawn.files for group in tree.groups for drawn in group.runs
                 if drawn.key == run), ())


def run_choice(listing: Mapping[str, Any], view: Any, preset: Optional[Preset], run: Any,
               current: Any, *, search: Any = "", show_older: bool = True) -> Optional[str]:
    """**The checkpoint a run's node selects** (the family graph's body: G2,
    I8): the run as the list draws it (`drawn_files`), looked up there and
    never taken on trust, chosen as a Run → Checkpoint choice chooses
    (`chosen_in_run`): `current` while it is one of them, else the first in
    the list's order (the run's latest save). None for a run the list does
    not draw."""
    files = drawn_files(listing, view, preset, run, search=search, show_older=show_older)
    items = {item_key(i): i for i in listing.get("items") or []}
    return chosen_in_run({key: items[key] for key in files}, run, current)


def kept_unfolded(listing: Mapping[str, Any], view: Any, preset: Optional[Preset],
                  fold: Optional[Mapping[str, Any]], before: Any, after: Any, *,
                  search: Any = "", show_older: bool = True) -> Dict[str, bool]:
    """**What a pick in the family graph keeps unfolded** (I5: nothing moves
    under the pointer): the subtrees shown unfolded while `before` is
    selected only because they hold it or lead to it (`graph_shape`'s
    default), and the runs of the table under the graph shown unfolded so
    (`tree_view`), which would fold once `after` is, as fold entries that
    keep them as they are shown (a search's own while one is shown,
    `search_node`). Nothing for a list that draws no graph, or a pick that
    folds nothing."""
    if not isinstance(_view(view).get("edges"), (list, tuple)):
        return {}
    fold = fold if isinstance(fold, Mapping) else {}

    def shown(selected: Any) -> set:
        tree = tree_view(listing, view, preset, search=search, fold=fold, selected=selected,
                         show_older=show_older)
        shape = graph_shape(listing, view, tree, fold, selected)
        nodes = {run_node(run.key) for group in tree.groups if not group.family
                 for run in group.runs} & tree.open_nodes
        for key in shape.open_runs:
            at = key
            while at in shape.parent_run and shape.parent_run[at] in shape.open_runs:
                at = shape.parent_run[at]
            if at not in shape.parent_run:       # every run above it unfolded: shown
                nodes.add(subtree_node(key))
        return nodes

    store = search_node if str(search or "").strip() else str
    return {store(node): True for node in sorted(shown(before) - shown(after))}


def _epochs_words(items: Sequence[Mapping[str, Any]]) -> str:
    """How far a run trained, in the one epoch words (`renderers.epoch_words`):
    its file that completed the most epochs, else its first file's."""
    counted = [i for i in items if isinstance(i.get("completed_epochs"), int)
               and not isinstance(i.get("completed_epochs"), bool)]
    return r.epoch_words(max(counted, key=lambda i: i["completed_epochs"]) if counted
                         else (items[0] if items else {}))


def _parent_file_words(files: _Files, parent: str) -> str:
    """The exact parent file among its run's saved files, by the words its
    row and its Checkpoint option lead with (`_checkpoint_entries`): what it
    is, and how far it trained where that alone tells it apart."""
    run_keys = files.run_files.get(item_run_key(files.items[parent]), [parent])
    entries = _checkpoint_entries(files.items, run_keys)
    what, rest = entries[parent]
    alike = sum(1 for key in run_keys if entries[key][0] == what)
    return what if alike == 1 else r.fit_option(what, rest)


def _bound_lines(bound: Mapping[str, Any]) -> List[str]:
    """A boundary's lines (G5): where the parent its record names is, in the
    family view's words (in another listed workspace, or not available, its
    reason in the selected file's details, `models_page.parent_html`), and
    that run by its words; never a guessed parent, never its digest (the
    details' too)."""
    run = bound.get("run")
    named = (f'<span class="shep-fam-ln">run {r.e(r.run_words(None, run, fit=False))}</span>'
             if isinstance(run, str) and run else "")
    if bound.get("state") == "elsewhere":
        return [f'<span class="shep-fam-ln">Parent in workspace '
                f'<b>{r.e(bound.get("workspace"))}</b></span>', named]
    return [f'<span class="shep-fam-ln">{r.status_mark("warn", "Parent unavailable")}</span>',
            named]


def _graph_html(listing: Mapping[str, Any], selected: Optional[str], preset: Preset,
                view: Any, tree: TreeView, fold: Optional[Mapping[str, Any]]) -> str:
    """**The family graph** (Models' By family; G1-G9, I1-I10): the same tree
    (`tree_view`; its shape, `graph_shape`) drawn as nodes and lines, one DOM
    and one state, real buttons a keyboard and assistive technology read as
    they are: each family's runs as nested lists (``ul``/``li``, a run's
    children in the list under it, however deep: G04), the lines drawn by the
    style sheet from the lists' boxes, so each meets the box it joins.

    **A node is one run** (G2): its name (its words told apart from the
    others', `renderers.distinct_names`) with its layout, its task and its
    operation (`record_kind_text`, `operation_words`: branch, extend,
    fine-tune full or frozen), its saved files and how far it trained, and,
    where its parent run holds several files, which one it came from ("from
    Best LP accuracy", `_parent_file_words`: G3); a run kept for a shown run
    only says so ("Parent of a shown run"), the model in use its chip. One
    " · " a line at most. One look says "selected": the run holding the
    selected model (the accent fill and bar, its body ``aria-current``), the
    only ``aria-current`` in the graph. Its body is a button that **selects
    the run** (G2, I8): the page sends the run (``data-run``) and the list
    chooses its checkpoint as a Run → Checkpoint choice does (`run_choice`),
    through the selection a row's pick goes through; the page's details then
    show that checkpoint, its run's other saved files among them, and the
    exact parent file (`models_page.identity_html`). Nothing is loaded:
    Load still asks (G03, I06). A node with
    child runs has a second button beside its body, never inside it: the
    arrow (``aria-expanded``, ``aria-controls`` its children's list, its name
    its child runs counted), shown while the pointer is over the node or the
    keyboard's focus is in it, always where the pointer is coarse
    (`style.SHELL_CSS`); a node is as large as its text and padding (I1). The
    arrow folds its subtree in the page at once and tells the server
    (`FAMILY_JS`); it never selects or loads. A family's root unfolds by
    default, a deeper subtree where it holds the selection, the model in use
    or, during a search, a run found (`graph_shape`).

    A run whose parent is not listed here hangs from a **boundary**, a box
    that is no choice: where its parent is, in the family view's words (G5).
    The files the graph cannot place (no readable model record) follow under
    `NOT_IN_A_FAMILY`, as the tree draws them (`_tree_html`). The selection
    in a folded branch is said on the folded node (`HOLDS_SELECTION`, under
    its text and arrow) and above the graph with a way to show it (I10). The
    graph decides its orientation by measured fit in the page (`FAMILY_JS`):
    left to right where it fits as drawn, else top to bottom, again only
    when what it is drawn from (``data-shape``), the region's width or the
    fonts change."""
    shape = graph_shape(listing, view, tree, fold, selected)
    shown_keys = [k for run in shape.runs.values() for k in run.files]
    files = _files(listing, shown_keys)
    items = files.items
    bounds = _view(view).get("bounds") if isinstance(_view(view).get("bounds"), Mapping) else {}
    names = r.distinct_names(
        {key: (items[run.files[0]].get("architecture"), items[run.files[0]].get("run"))
         for key, run in shape.runs.items()},
        lambda parts, level: " · ".join(
            p for p in r.run_parts(parts[0], parts[1], level, fit=False) if p))
    served = files.served
    ids: Dict[str, int] = {}
    # The selection in a folded branch (I10): the outermost folded run above
    # the selected model's run, said on that node and above the graph; kept
    # true in the page as the operator folds (`FAMILY_JS`).
    holder, at = None, shape.selected_run
    while at is not None and at in shape.parent_run:
        at = shape.parent_run[at]
        if at not in shape.open_runs:
            holder = at
    # What the graph's size is drawn from: each run, its place and its words,
    # never what is folded, selected or in use (`FAMILY_JS` decides the
    # orientation again only when this changes, the region's width or the
    # fonts do: never on a fold or a pick, I10).
    drawn: List[str] = []

    def node_html(key: str) -> str:
        run = shape.runs[key]
        summary = _run_summary(listing, run)
        what, _sep, rest = names[key].partition(" · ")
        # A start time's date and time on one line ("2026-11-01 00:00").
        rest = re.sub(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})", "\\1\u00a0\\2", rest)
        kids = shape.children.get(key, [])
        opened = key in shape.open_runs
        holds = served and any(items[f].get("sha256") == served for f in run.files)
        count = (f"{len(run.files)} of {_counted(run.total, 'saved file')}"
                 if len(run.files) < run.total else _counted(run.total, "saved file"))
        lines = [
            f'<span class="shep-fam-name"><span class="shep-w600">{r.e(what)}</span>'
            + (f'<span class="shep-muted"> · {r.whole_words(rest)}</span>' if rest else "")
            + "</span>",
            f'<span class="shep-fam-ln">{r.whole_words(r.record_kind_text(summary))} · '
            f'{r.whole_words(r.operation_words(summary))}</span>',
            f'<span class="shep-fam-ln">{r.e(count)}, '
            f'{r.whole_words(_epochs_words([items[f] for f in run.files]))}</span>']
        parent = shape.parent_key.get(key)
        if parent is not None and len(files.run_files.get(item_run_key(items[parent]), ())) > 1:
            lines.append(f'<span class="shep-fam-ln">from '
                         f'{r.whole_words(_parent_file_words(files, parent))}</span>')
        if run.context:
            lines.append(f'<span class="shep-fam-ln">'
                         f'{r.status_mark("neutral", "Parent of a shown run")}</span>')
        drawn.append("\x1f".join((key, shape.parent_run.get(key, ""), *lines)))
        if holds:
            lines.append(f'<span class="shep-fam-ln">{r.chip(preset.in_use_label, "accent")}'
                         '</span>')
        # One look says "selected": the run holding the selected model, its
        # body `aria-current`.
        classes = ["shep-fam-node"] + (["shep-sel"] if key == shape.selected_run else []) + (
            ["shep-ctx"] if run.context else []) + (
            ["shep-fam-parent"] if kids else []) + (
            ["shep-fam-shut"] if kids and not opened else []) + (
            ["shep-fam-holds"] if key == holder else [])
        body = (f'<button type="button" class="shep-fam-body" data-run="{r.e(key)}"'
                + (' aria-current="true"' if key == shape.selected_run else "")
                + f'>{"".join(lines)}</button>')
        arrow = sub = held = ""
        if kids:
            ids[key] = len(ids)
            controls = f"shep-fam-k{ids[key]}"
            label = SUBTREE_LABEL.format(count=len(kids), name=what)
            arrow = (f'<div class="shep-fam-act"><button type="button" class="shep-fam-tog" '
                     f'data-tree="{r.e(subtree_node(key))}" '
                     f'aria-expanded="{"true" if opened else "false"}" '
                     f'aria-controls="{controls}" aria-label="{r.e(label)}">{_CHEVRON}'
                     '</button></div>')
            held = (f'<span class="shep-fam-held"{"" if key == holder else " hidden"}>'
                    f'{r.e(HOLDS_SELECTION)}</span>')
            sub = (f'<ul class="shep-fam-kids" id="{controls}"{"" if opened else " hidden"}>'
                   + "".join(node_html(k) for k in kids) + "</ul>")
        return (f'<li class="shep-fam-li" data-run="{r.e(key)}">'
                f'<div class="{" ".join(classes)}">{body}{arrow}{held}</div>{sub}</li>')

    def root_html(key: str) -> str:
        bound = bounds.get(key) if isinstance(bounds.get(key), Mapping) else None
        if bound is None:
            return node_html(key)
        lines = _bound_lines(bound)
        drawn.append("\x1f".join(("bound", key, *lines)))
        return ('<li class="shep-fam-li shep-fam-bli"><div class="shep-fam-node shep-fam-bound">'
                + "".join(lines) + '</div><ul class="shep-fam-kids">'
                + node_html(key) + "</ul></li>")

    trees = []
    for _group, tops in shape.roots:
        if not tops:
            continue
        first = names[tops[0]].partition(" · ")[0]
        trees.append(f'<ul class="shep-fam-tree" aria-label="{r.e("Family from " + first)}">'
                     + "".join(root_html(key) for key in tops) + "</ul>")

    holder_words = names[holder].partition(" · ")[0] if holder else ""
    hid = (f'<div class="shep-list-outside shep-fam-hid"{"" if holder else " hidden"}>'
           + r.status_mark("neutral", f'{r.e(FOLDED_SELECTION)} <span class="shep-fam-hid-w">'
                                      f'{r.e(holder_words)}</span>', words_html=True)
           + ' <button type="button" class="shep-btn shep-linkish shep-fam-show">Show it'
           '</button></div>')
    others = [g for g in tree.groups if not g.family]
    rest = ""
    if others:
        rest = (f'<h3 class="shep-fam-h">{r.e(NOT_IN_A_FAMILY)}</h3>'
                + _tree_html(listing, selected, preset, view, tree._replace(groups=tuple(others)),
                             locate=False))
    graph = (f'<div class="shep-fam-forest" role="group" aria-label="Model families">'
             f'{"".join(trees)}</div>' if trees else "")
    locate = shape.selected_run or next(
        (key for key, run in shape.runs.items()
         if served and any(items[f].get("sha256") == served for f in run.files)), "")
    inner = hid + graph + rest
    # A token of this drawing (Gradio patches the region's HTML in place: the
    # page's script tells a new drawing by it, `FAMILY_JS`), and one of what
    # the graph's size is drawn from (``data-shape``).
    token = hashlib.sha1(inner.encode("utf-8")).hexdigest()[:12]
    form = hashlib.sha1("\x1e".join(drawn).encode("utf-8")).hexdigest()[:12]
    return (f'<div class="shep-fam" data-shep-folds-own data-draw="{token}" data-shape="{form}" '
            f'data-fam-locate="{r.e(locate)}"'
            + (' data-searched="1"' if tree.query else "") + f'>{inner}</div>')


def outside_html(listing: Mapping[str, Any], tree: TreeView, view: Any = None) -> str:
    """**A selection the list does not show, said** (D04): "Selected model not
    shown by this filter" with what it is and why (the Task filter, the
    search), or "… not shown among the latest runs" where the drawer leaves
    its run out until "Show older runs"; its details kept; one action, Clear
    search, where the search hides it. Nothing for a selection the list
    shows, or none."""
    item = find_item(listing, tree.outside)
    if item is None:
        return ""
    why = []
    chosen = task_filter(view)
    if not matches(item, chosen):
        why.append(f"Task: {FILTER_WORDS[chosen][0]}")
    if tree.query and search_hit(item, tree.query) is None:
        why.append(f"search “{tree.query}”")
    lead = ("Selected model not shown among the latest runs" if not why and tree.cut
            else f"Selected model not shown by this filter ({', '.join(why) or 'the order'})")
    words = (f"{lead}: {r.model_kind_text(item)}, "
             f"{r.run_words(item.get('architecture'), item.get('run'))}. Its details are kept.")
    action = ('<button type="button" class="shep-btn shep-linkish" data-key="__clear_search__">'
              'Clear search</button>' if tree.query and search_hit(item, tree.query) is None
              else "")
    return f'<div class="shep-list-outside">{r.status_mark("neutral", words)} {action}</div>'


def _counted(count: int, word: str) -> str:
    """"1 checkpoint", "3 checkpoints", "2 families": a count with its word in
    its number."""
    if word.endswith("y") and count != 1:
        return f"{count} {word[:-1]}ies"
    return f"{count} {word}{'s' if count != 1 else ''}"


def count_line(listing: Mapping[str, Any], preset: Preset, show_older: bool,
               view: Any = None, *, search: Any = "",
               fold: Optional[Mapping[str, Any]] = None) -> str:
    """**What the list shows, in a few words** (R-27, R-77): never the order,
    which its switch shows, and no " · ". Under every row of the preset, the
    families, runs and saved files shown ("2 families, 10 runs, 20 saved
    files"; the drawer's latest runs, "Latest 4 runs, 7 saved files", "Latest
    4 of 7 runs and the one in use, 10 saved files" when the run of the model
    in use is older); a filter leads with its name and counts its runs
    against the runs listed ("Gene: 2 of 10 runs, with 1 run they come from":
    a family view's context, the runs a shown run descends from, counted
    apart); a search says what matches ("3 runs match “disease”, with 2 runs
    they come from"). Counted against what the preset ever lists (the drawer:
    its task models), an entry the listing could not list never a run (it is
    counted apart, each by the state its row says, in that state's own words:
    "1 entry not read, 1 entry not listed again"); while
    the drawer leaves runs out (not read yet, `runs_unread`, or beyond its
    latest), a search and a filter say they saw the latest runs only ("1 run
    matches “gene” in the latest runs", "Gene: 1 of the latest 4 runs"), never
    the workspace's whole answer. Nothing for a list with no rows."""
    if listing.get("refused") or not listing.get("items"):
        return ""
    chosen = task_filter(view)
    lead = f"{FILTER_WORDS[chosen][0]}: " if chosen not in ("all", preset.rows) else ""
    query = str(search or "").strip()
    every = [i for i in shown_items(listing, None, preset) if not r.entry_place(i)]
    total = len({item_run_key(i) for i in every})
    tree = tree_view(listing, view, preset, search=query, fold=fold, show_older=show_older)
    # A run is counted once, whatever groups it is drawn in; an entry the
    # listing could not list (a directory, B69) is counted apart, never as
    # a run or a saved file, by the state its own row says (V-22: "not
    # listed" beside its own row read as a contradiction), in that state's
    # own words, never merged (the skill's §6; the review of 6a14d6b, R3-1:
    # an alias, whose runs were read, read "not read"): one never opened
    # "not read", a link listed otherwise "not listed again"
    # (`renderers.unreadable_words`, the one word for each state).
    entries: Dict[str, str] = {}
    for key in tree.matched:
        item = find_item(listing, key) or {}
        if r.entry_place(item):
            entries[key] = r.unreadable_words(item)
    runs = list(dict.fromkeys(run.key for g in tree.groups for run in g.runs
                              if not run.context and run.files[0] not in entries))
    context = len({run.key for g in tree.groups for run in g.runs if run.context} - set(runs))
    families = sum(g.family for g in tree.groups)
    files = len(tree.matched) - len(entries)
    unread = runs_unread(listing, show_older)
    # Where a search looked: the latest runs only while the list leaves runs
    # out (not read yet, or beyond its latest: "Show older runs" under it).
    within = " in the latest runs" if unread or tree.cut else ""
    if not runs:
        if query:
            return (f"{lead}no run matches “{query}”{within}" if lead
                    else f"No run matches “{query}”{within}")
        return f"{lead}no checkpoint shown" if lead else "No checkpoint shown"
    if query:
        body = (f"{_counted(len(runs), 'run')} match{'es' if len(runs) == 1 else ''} "
                f"“{query}”{within}")
    elif lead:
        body = f"{len(runs)} of {'the latest ' if unread else ''}{_counted(total, 'run')}"
    elif tree.cut or unread:
        latest = len(runs) - tree.served_run
        of = "" if unread else f" of {latest + tree.cut + tree.served_run}"
        body = (f"Latest {latest}{of} {'run' if latest == 1 and not of else 'runs'}"
                + (" and the one in use" if tree.served_run else "")
                + f", {_counted(files, 'saved file')}")
    else:
        body = (f"{_counted(families, 'family')}, " if families else "") + (
            f"{_counted(len(runs), 'run')}, {_counted(files, 'saved file')}")
    with_ = f", with {_counted(context, 'run')} they come from" if context else ""
    states: Dict[str, int] = {}
    for state in entries.values():
        states[state] = states.get(state, 0) + 1
    unlisted = "".join(f", {_counted(n, 'entry')} {state}" for state, n in states.items())
    return f"{lead}{body}{with_}{unlisted}"


def find_item(listing: Mapping[str, Any], key: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(key, str):
        return None
    for item in listing.get("items") or []:
        if item_key(item) == key:
            return item
    return None


def loadable(item: Optional[Mapping[str, Any]]) -> bool:
    """A listed checkpoint that was read, with its digest (`record_views.item_was_read`)."""
    from src.api.services.record_views import item_was_read

    return item_was_read(item)


def load_refusal(item: Mapping[str, Any]) -> Optional[str]:
    """Why a listed checkpoint is never loaded, from its listed model record
    alone (nothing of its file is read again): a base offers no diagnosis; a
    record this build refused (a file without a model spec among them, with
    the record reader's reason), or never read, says so. None for a model finetuned for
    a task, which the reload serves under that task's scoring mode."""
    record = item.get("record")
    if record == "model":
        if item.get("learning_stage") == "base":
            return BASE_REFUSAL
        if item.get("task") in TASK_SCORING_MODES:
            return None
        return f"{r.record_kind_text(item)}: its record names no task this build serves."
    if record == "refused":
        return f"{r.record_kind_text(item)}: {item.get('record_problem') or 'refused'}."
    return f"{r.record_kind_text(item)}."


def can_load(item: Optional[Mapping[str, Any]]) -> bool:
    """Whether Load is offered for a listed checkpoint: its file was read and
    its record is a model finetuned for a task (`loadable`, `load_refusal`)."""
    return loadable(item) and load_refusal(item) is None


#: The Load button's words, shown with no lock while the operator may use it.
LOAD_LABEL = "Load this model"


def load_button(listing: Mapping[str, Any], item: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Load this model by the one gated-button rule (`chips.gated`): no lock
    while the listing was read for a request D8 allows (offered for a row
    `can_load` admits); for a listing D8 refused, disabled with the lock and
    its reason beside it (`load_reason_html`)."""
    from src.webui.components.chips import gated

    return gated(LOAD_LABEL, not listing.get("refused"), interactive=can_load(item))


#: Load's cause while no model is chosen (the sweep's L7, SW-LK6: Load stood
#: dimmed with nothing said; the skill's §9), led by the need
#: (`renderers.need_words`).
CHOOSE_A_MODEL = r.need_words("choose a model in the list first")


def load_reason_html(item: Optional[Mapping[str, Any]],
                     listing: Optional[Mapping[str, Any]] = None, *,
                     control: Optional[str] = None) -> str:
    """Beside the Load button, why it is off: for a listing D8 refused, its
    reason with the lock and its phrase (`chips.operator_only`, the skill's
    §3); with no model chosen, the listing's notice's phrase when the list is
    empty for want of a setting (`renderers.phrase_line`: the notice is the
    list's, said once), else that a model is to be chosen
    (`CHOOSE_A_MODEL`); for the selected row, when its record is never
    loaded (`load_refusal`) or its file was not read (its state,
    `renderers.unreadable_words`); nothing otherwise. Every dimmed Load says
    why, in the shared form. `control`: the element id of the Load it
    explains, the instance's own (`create_model_list`): a phrase names it
    (R-WHY's ``data-for``), so the dimmed Load is described by its phrase."""
    if (listing or {}).get("refused"):
        from src.webui.components.chips import LOCK

        return (f'<div class="shep-reason shep-reason-grey">{LOCK} '
                f'{refused_line(listing or {}, control=control)}</div>')
    if item is None:
        notice = (listing or {}).get("notice")
        if notice and not (listing or {}).get("items"):
            return r.phrase_line(notice, control=control)
        return r.marked_line("neutral", CHOOSE_A_MODEL)
    if not loadable(item):
        # A file not read, or whose read failed: its state, as its row says
        # it; why is in its details (`facts_html`, Models' detail).
        state = r.unreadable_words(item)
        return (f'<div class="shep-reason shep-reason-grey">'
                f'{r.e(state[:1].upper() + state[1:])}: it cannot be loaded.</div>')
    why = load_refusal(item)
    return f'<div class="shep-reason shep-reason-grey">{r.e(why)}</div>' if why else ""


#: At the end of the chosen model's title, shown stacked only (`style.SHELL_CSS`:
#: its words on one line, on a line of its own when the name and its chip leave
#: it no room): back to the row chosen (`FACTS_JS`).
BACK_TO_LIST = ('<button type="button" class="shep-btn shep-back-list">'
                '↑ Back to the list</button>')


def facts_html(listing: Mapping[str, Any], key: Any) -> str:
    """The chosen model's facts beside the rows (the drawer: Model Facts,
    `renderers.model_facts_html`), under a title that names it and its key
    (``data-facts-key``, for `FACTS_JS`), with "↑ Back to the list"; what
    to do when nothing is chosen."""
    item = find_item(listing, key)
    if item is None:
        return '<div class="shep-muted">Select a model to see its facts.</div>'
    workspaces = listing.get("workspaces") or []
    serving = next((w for w in workspaces if w.get("name") == listing.get("serving")), None)
    facts = r.facts_from_checkpoint(item, workspaces, recorded=_recorded(listing, key),
                                    serving_workspace=serving, listed=True)
    title = (f'<div class="shep-facts-title" data-facts-key="{r.e(key)}" tabindex="-1">'
             f'<b>{r.chain_html(*facts.name_facts) if facts.name_facts else r.e(facts.name_line)}'
             f'</b> {r.chip("selected", "neutral")}{BACK_TO_LIST}</div>')
    return title + r.model_facts_html(facts)


def workspace_line_html(listing: Mapping[str, Any]) -> str:
    """**The chosen workspace's identity, once, under the list's workspace
    select**, on every page that places the shared list (Diagnosis' Change
    model drawer, Models), as Training, Fine-tune and Test say theirs: the
    one dataset identity renderer (`renderers.dataset_identity_html`, its
    name left out, since the select's option is the name,
    `renderers.workspace_choices`: the dataset fingerprint with the digits
    that tell the listed fingerprints apart, "synthetic", the record's
    state). Nothing for a request D8 refuses (no workspace is chosen there)
    or a workspace the listing does not hold."""
    if listing.get("refused"):
        return ""
    rows = listing.get("workspaces") or []
    row = next((w for w in rows if w.get("name") == listing.get("workspace")), None)
    if row is None:
        return ""
    digits = r.distinguishing_digits(w.get("dataset_fingerprint") for w in rows)
    return (f'<div class="shep-ws-line">'
            f'{r.dataset_identity_html(row, digits=digits, name=False)}</div>')


def note_html(view: Any, item: Optional[Mapping[str, Any]] = None) -> str:
    """The view's note beside the Load button (plain text, escaped), **only
    for a selected row Load can serve** (R-78): with nothing selected, or a
    row whose record is never loaded (its reason is said beside Load,
    `load_reason_html`), there is nothing for it to explain."""
    note = _view(view).get("note")
    return (f'<span class="shep-note">{r.e(note)}</span>'
            if note and item is not None and can_load(item) else "")


class ListDrawing(NamedTuple):
    """What the list draws from one state (`list_drawing`): the rows, the
    count line, the facts, Load's arguments (`load_button`), its reason and
    the view's note."""

    rows: str
    count: str
    facts: str
    load: Dict[str, Any]
    reason: str
    note: str


def list_drawing(listing: Mapping[str, Any], key: Any, preset: Preset, *,
                 show_older: Any = False, view: Any = None, search: Any = "",
                 fold: Any = None, said: Any = None,
                 load_id: Optional[str] = None) -> ListDrawing:
    """**Everything the list shows, from one state** (G07; the review of
    00258f4, SL-R1): the rows, the count, and what describes the selection
    (the facts, Load, its reason, the note), all from the same listing,
    selection, "Show older runs", view, search and folds. Pure: nothing is
    read from disk or listed again. A selection the listing does not hold
    (another workspace's) is drawn as none; a view whose keys were read from
    another workspace applies its filter and note only (`view_for_listing`).
    `said` is what an act said beside Load (``{"key", "html"}``: a refusal
    of the Load, the status read again): its words are Load's reason while
    the selection drawn is the one it was said for, else the selection's own
    (`load_reason_html`); a search, a fold or "Show older runs" keeps them
    (the review of 5171661). `load_id`: the element id of this instance's
    Load, which its reason's phrase names (R-WHY)."""
    view = view_for_listing(view, listing, listing)
    fold = fold if isinstance(fold, Mapping) else {}
    item = find_item(listing, key)
    key = key if item is not None else None
    older, search = bool(show_older), search or ""
    reason = (str(said.get("html") or "") if isinstance(said, Mapping)
              and said.get("key") == key else load_reason_html(item, listing, control=load_id))
    return ListDrawing(
        list_html(listing, key, preset, show_older=older, view=view, search=search, fold=fold),
        count_line(listing, preset, older, view, search=search, fold=fold),
        facts_html(listing, key), load_button(listing, item), reason, note_html(view, item))


#: The draw tokens' count (`draw_token`), the process's one.
_DRAW_TOKENS, _DRAW_TOKENS_LOCK = _count(1), threading.Lock()


def draw_token() -> int:
    """**A new draw token** (the list's ``drawn``; Models' view's
    ``redraw``): unlike every token before it in this process, so the change
    it writes is always seen (Gradio fires a state's change only when its
    value differs at the end of an event; a clock can read the same twice:
    Python 3.12 on Windows ticks ``time.monotonic`` every 15.6 ms, the
    review of 5171661). The count starts at 1, a list's token starting at
    0."""
    with _DRAW_TOKENS_LOCK:
        return next(_DRAW_TOKENS)


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
    (``serving``). ``served`` is False when no model is loaded: then nothing is
    served, and the rows that describe what is served do not apply (`_cell`).
    A listed model is named whole (`renderers.model_name_line`, ``fit``
    False), as the To column names its file: two runs compared are never cut
    alike."""
    identity = (status or {}).get("served_identity") or {}
    sha = served_checkpoint_sha(status)
    item = next((i for i in listing.get("items") or [] if sha and i.get("sha256") == sha), None)
    meta = (status or {}).get("checkpoint_meta") or {}
    if item is not None:
        model = r.model_name_line(item, fit=False)
        model_html = r.model_name_html(item, fit=False)
    elif sha:
        model = f"{r.epoch_words(meta)} (not in this workspace's list)"
    else:
        model = "not known" if (status or {}).get("initialized") else "no model loaded"
    sp = identity.get("shortest_paths") or {}
    record = identity.get("dataset_record") or {}
    return {
        "served": bool((status or {}).get("initialized")),
        "model": model,
        "model_html": model_html if item is not None else r.e(model),
        "sha256": sha,
        "trained_on": item.get("dataset_fingerprint") if item else None,
        "trained_on_known": item is not None,
        "workspace": listing.get("serving"),
        "graph": (identity.get("graph") or {}).get("kg_sha256"),
        "sp": sp.get("state"),
        "mode": identity.get("scoring_mode"),
        "synthetic": record.get("origin") == "synthetic",
    }


def _to_side(item: Mapping[str, Any], workspace: str, workspace_dir: Path,
             workspaces: Sequence[Any]) -> Dict[str, Any]:
    """What would be served, as the confirmation's "To" column needs it: the
    scoring mode is the one the task the model's record names is served under
    (`core.types.TASK_SCORING_MODES`, the pipeline's own table), and its
    shortest-path term the state that mode is served with
    (`core.types.SCORING_MODE_SP_STATES`, the rule the served identity is
    checked against); neither is guessed from the workspace, which a task
    model's serving never reads for them, and no mixing weight is predicted."""
    origin = next((r.origin_of(w) for w in workspaces if r._get(w, "name") == workspace), None)
    mode = TASK_SCORING_MODES.get(item.get("task"))
    return {
        "model": r.model_name_line(item, fit=False),
        "model_html": r.model_name_html(item, fit=False),
        "sha256": item.get("sha256"),
        "trained_on": item.get("dataset_fingerprint"),
        "trained_on_known": True,
        "workspace": workspace,
        "graph": _graph_digest(workspace_dir),
        "sp": SCORING_MODE_SP_STATES.get(mode, "absent"),
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


#: The From rows that describe what is served: with no model loaded nothing is
#: served, so they say "not applicable" (their own word, never "absent",
#: "files" or a fingerprint "not recorded", which would describe something
#: served). Otherwise every side names a task: a served model's scoring mode
#: is a task's, and only a model finetuned for a task is loaded
#: (`load_refusal`).
_SERVED_ROWS = ("sha", "trained", "workspace", "mode", "sp", "data")


def _cell(side: Mapping[str, Any], row: str, workspaces: Sequence[Any], *, to: bool,
          other: Optional[Mapping[str, Any]] = None) -> str:
    if side.get("served") is False and row in _SERVED_ROWS:
        return '<span class="shep-muted">not applicable</span>'
    if row == "model":
        return side.get("model_html") or r.e(side["model"])
    if row == "sha":
        return r.fingerprint_value(side["sha256"], what="model fingerprint", copy=False)
    if row == "trained":
        if not side.get("trained_on_known"):
            return '<span class="shep-muted">not in this workspace\'s list</span>'
        html_ = r.trained_on_html(side["trained_on"], workspaces)
        if to and other is not None and other.get("trained_on") and \
                other.get("trained_on") == side.get("trained_on"):
            html_ += ' <span class="shep-muted">same dataset</span>'
        return html_
    if row == "workspace":
        if to and other is not None and other.get("served") is not False and \
                other.get("workspace") == side["workspace"] and \
                other.get("graph") == side["graph"]:
            return "same"
        name = f"<b>{r.e(side['workspace'])}</b>" if side["workspace"] else \
            '<span class="shep-muted">not a listed workspace</span>'
        return f'{name} · graph ' + r.fingerprint_value(side["graph"], what="graph fingerprint",
                                                         copy=False)
    if row == "sp":
        return ("computed from the model's graph view" if side.get("sp") == "computed"
                else "not used by this task")
    if row == "mode":
        mode = side["mode"]
        return f"{r.e(TASK_NAMES[SCORING_MODE_TASKS[mode]])} (<code>{r.e(mode)}</code>)"
    if row == "data":
        return r.chip("synthetic", "amber") if side["synthetic"] else "files"
    return ""


_ROWS = (("model", "Model"), ("sha", "Model fingerprint"), ("trained", "Trained on"),
         ("workspace", "Serving workspace"), ("mode", "Task and scoring"), ("sp", "SP table"),
         ("data", "Data"))
_COMPARED = {"model": ("sha256",), "sha": ("sha256",), "trained": ("trained_on",),
             "workspace": ("workspace", "graph"), "sp": ("sp",),
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
        f'<span class="shep-spinner" aria-hidden="true"></span> <b>{r.e(PAGE_WAIT_WORDS)} '
        '<span class="shep-elapsed">00:00:00</span></b> Closing this does not stop it.</div>'
        '<div class="shep-load-timeout" hidden><b>The page stopped waiting after '
        f'{PAGE_WAIT_SECONDS} s. The server may still finish the reload.</b> '
        f'<button type="button" class="shep-btn shep-read-again" data-key="__read_again__">'
        'Read again</button></div>'
        f'<span class="shep-vh" data-old="{r.e(old)}"></span></div>'
    )


#: **What a plan keeps of a listed checkpoint**, the one tuple for the plan's
#: listing and its "To" item: its place, bytes, how far it trained (both the
#: fields the name line reads, `renderers.epoch_words`) and dataset, and the
#: record fields `load_refusal` checks again and the reload's task comes from.
_ITEM_KEYS = ("workspace", "architecture", "run", "filename", "sha256", "epoch",
              "completed_epochs", "dataset_fingerprint", "problem", "record", "record_problem",
              "learning_stage", "task", "encoder", "reference", "differences")


def _kept(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: item.get(k) for k in _ITEM_KEYS}


def _listed_context(listing: Mapping[str, Any]) -> Dict[str, Any]:
    """What a plan keeps of the listing to read the From column again: the
    checkpoints to name the served one by, and the listed workspaces."""
    return {"items": [_kept(i) for i in listing.get("items") or []],
            "workspaces": list(listing.get("workspaces") or [])}


def _said(text: Any, environ: Optional[Mapping[str, str]] = None) -> str:
    """**A server's text as a Load step shows it**: with no absolute server
    path left, by the dashboard's one redaction against the workspaces root
    (`workspaces.redact`, as the list's own problems are shown). A reload's
    refusal, a list's refusal and a re-read record's problem may name a
    checkpoint's or the root's path (the reload route answers its operator
    with it; the page never shows it)."""
    from src.webui.components.workspaces import redact, workspaces_root

    return redact(text, workspaces_root(environ)[0])


def request_load(listing: Mapping[str, Any], key: Any, request: Any, *,
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
        return None, (f'<div class="shep-reason">Refused: {r.e(_said(exc, environ))} Nothing was '
                      'done.</div>')
    if not loadable(target):
        why = _said(target.get("problem") or "its digest was not read", environ)
        return None, (f'<div class="shep-reason">It cannot be loaded now: {r.e(why)}. Nothing was '
                      'done.</div>')
    refusal = load_refusal(target)
    if refusal is not None:
        return None, (f'<div class="shep-reason">The file changed since the list was read: '
                      f'{r.e(_said(refusal, environ))} Nothing was done.</div>')
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
            **context}
    notice = None
    if target.get("sha256") != item.get("sha256"):
        notice = (f"The file changed since the list was read: it holds model "
                  f"{r.short_hex(target.get('sha256'))} now, not {r.short_hex(item.get('sha256'))}"
                  " as listed. The To column is what it holds now. Read the list again to see "
                  "the rest.")
    return plan, confirmation_html(plan, notice=notice)


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
    """Step 2, synchronous (the caller runs it in a worker thread, as the API
    route runs its reload): D8 on this request; what is served and what would be
    loaded both read again, and the confirmation shown again with a notice when
    either changed; WP2's reload with what the server resolves from the item,
    bound to the model fingerprint shown (other bytes are refused by the
    reload's own read, and nothing is published)."""
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
        return LoadStep(None, _outcome("grey", "Not reloaded.",
                                       f"Refused: {_said(exc, environ)} Nothing was done."),
                        "", now_sha, False)
    # The refusal again on what the file holds now (rewritten into a base
    # since the confirmation, say): never reloaded, never shown to confirm.
    refusal = load_refusal(target) if loadable(target) else None
    if refusal is not None:
        return LoadStep(None, _outcome("grey", "Not reloaded.",
                                       f"{_said(refusal, environ)} Nothing was done."),
                        "", now_sha, False)
    to_now = _to_side(target, plan["workspace"], directory, workspaces)
    if _sides_differ(to_now, plan["to"]):
        new_plan["to"] = to_now
        new_plan["to_item"] = _kept(target)
        notices.append(TO_CHANGED)
    if notices:
        if not loadable(target):
            why = _said(target.get("problem") or "its digest was not read", environ)
            return LoadStep(None, _outcome(
                "grey", "Not reloaded.", f"It cannot be loaded now: {why}. Nothing was done."),
                "", now_sha, False)
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
        # The route answers its operator with the build's own words, which
        # can name the checkpoint's path; the page says them without it.
        why = _said(answer.get("message") or "the reload was refused", environ)
        return LoadStep(None, _outcome("grey", "Not reloaded. The model in use is unchanged.",
                                       why),
                        "", served if isinstance(after, Mapping) else now_sha, changed)
    text = f"The model in use for every session is now {r.short_hex(served)}."
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
#: Clear, a front-end function: the search box emptied in the page (its
#: ``change`` is the search's one path, `create_model_list`).
CLEAR_SEARCH_JS = "() => ''"

#: The page side: row clicks (and Enter or Space on a focused table row) and
#: "Read again" become the component's ``click`` event with ``{"key": …}``,
#: "Clear search" clicks the list's Clear; a row focused under its table's
#: sticky header scrolls clear of it.
MODEL_LIST_JS = r"""
// A run's fold toggle (D04) folds or unfolds its files at once in the page:
// every row is in the table, a folded one hidden, so this only shows or hides
// rows; the server is told, for the next drawing, and draws nothing now.
var foldRows = function (node, open) {
  element.querySelectorAll('button.shep-fold').forEach(function (b) {
    if (b.getAttribute('data-fold') !== node) { return; }
    b.setAttribute('aria-expanded', open ? 'true' : 'false');
    var mark = b.querySelector('span'); if (mark) { mark.textContent = open ? '\u25be' : '\u25b8'; }
  });
  element.querySelectorAll('tr[data-in]').forEach(function (tr) {
    if (tr.getAttribute('data-in') === node) { tr.hidden = !open; }
  });
};
// Every act made in the rows is sent with the generation of the listing they
// were drawn from (`generation_marker`), so the server refuses one made in
// rows of another workspace than the list now shows (`ListClock`); the family
// graph's script, after this one, sends its acts so too.
var listGen = function (data) {
  var mark = element.querySelector('.shep-list-gen');
  if (mark) { data.gen = mark.getAttribute('data-gen'); }
  return data;
};
// A row chosen (a click, Enter or Space) is said to the server and marked on
// the list, so its facts (`FACTS_JS`) know they answer the operator's choice.
// "Clear search" is the list's Clear, clicked here: the box emptied in the
// page, its change the search's one path (`CLEAR_SEARCH_JS`).
var pick = function (key) {
  var list = element.closest('.shep-model-list');
  if (key === '__clear_search__') {
    var clear = list && document.getElementById(list.id + '-search-clear');
    if (clear) { clear.click(); }
    return;
  }
  if (list && key.indexOf('__') !== 0) { list.setAttribute('data-shep-picked', key); }
  trigger('click', listGen({key: key}));
};
element.addEventListener('click', function (e) {
  if (e.target.closest && e.target.closest('button.shep-copy, button.shep-i')) { return; }
  var fold = e.target.closest ? e.target.closest('[data-fold]') : null;
  if (fold && element.contains(fold)) {
    var node = fold.getAttribute('data-fold');
    var button = Array.prototype.find.call(element.querySelectorAll('button.shep-fold'),
      function (b) { return b.getAttribute('data-fold') === node; });
    var open = !(button && button.getAttribute('aria-expanded') === 'true');
    foldRows(node, open);
    // Folded while a search shows the list: kept for that search only (G7).
    var searched = !!element.querySelector('[data-searched]');
    trigger('click', listGen({fold: node, open: open, searched: searched}));
    return;
  }
  var row = e.target.closest ? e.target.closest('[data-key]') : null;
  if (!row) { return; }
  pick(row.getAttribute('data-key'));
});
// The selected row, or else the model in use, is located once it is drawn
// (D04): scrolled into view the first time the list names it under a view
// (an order, a filter: the rows' `view_marker`), never again while it stays
// the one named under the same view, so an operator's own scrolling is kept.
var locate = function () {
  var wrap = element.querySelector('[data-locate]');
  if (!wrap) { return; }
  var key = wrap.getAttribute('data-locate');
  var marker = element.querySelector('.shep-view-for');
  var token = key + '|' + (marker ? marker.getAttribute('data-for') : '');
  if (!key || token === element.__shepLocated) { return; }
  element.__shepLocated = token;
  var row = Array.prototype.find.call(wrap.querySelectorAll('tr[data-key]'),
    function (t) { return t.getAttribute('data-key') === key; });
  if (row && !row.hidden) { row.scrollIntoView({block: 'nearest'}); }
};
new MutationObserver(locate).observe(element, {childList: true, subtree: true});
locate();
element.addEventListener('keydown', function (e) {
  var row = e.target.closest ? e.target.closest('tr[data-key]') : null;
  if (!row || e.target !== row) { return; }
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(row.getAttribute('data-key')); }
});
// A row focused under its table's sticky header scrolls clear of it (the
// layout rule: nothing sticky covers a focused row): the header's height,
// kept on the component as `--shep-tbl-head-h`, is the rows' scroll margin
// (`style.SHELL_CSS`), 0 where the header does not stick. It is in place
// before a row takes focus (a margin set on focus comes too late for the
// browser's scroll), and a redrawn header that changes height changes the
// component's. The region's scroll padding keeps the room of what sticks
// above the table (the Models list's title).
var headRoom = function () {
  var cell = element.querySelector('table > thead th');
  var stuck = cell && getComputedStyle(cell).position === 'sticky';
  element.style.setProperty('--shep-tbl-head-h', stuck ? cell.closest('thead').offsetHeight + 'px' : '0px');
};
if (window.ResizeObserver) { new ResizeObserver(headRoom).observe(element); }
headRoom();
"""

#: **The chosen model's facts, a region of their own** (UI-07; on the facts'
#: ``js_on_load``): a newly chosen model's facts start at their top (side by
#: side the facts scroll on their own, so another model's would open where the
#: last was read); stacked (the shell's one layout decision), a row the
#: operator chose (`MODEL_LIST_JS` marks it on the list) brings its facts into
#: view, their title focused, and "↑ Back to the list" brings the chosen row
#: back into view and focuses it, the list where it was. Waits for the facts
#: that name the row (their title's ``data-facts-key``, which Gradio's HTML may
#: change in place, so the attribute is observed too), never a fixed time.
FACTS_JS = r"""
var region = element.closest('.block') || element;
var list = element.closest('.shep-model-list');
var shown = null;
var stacked = function () { return document.documentElement.getAttribute('data-shep-layout') !== 'side'; };
var drawnKey = function () {
  var t = element.querySelector('[data-facts-key]');
  return t ? t.getAttribute('data-facts-key') : null;
};
new MutationObserver(function () {
  var key = drawnKey();
  if (key === shown) { return; }
  shown = key;
  region.scrollTop = 0;
  if (!key || !list || list.getAttribute('data-shep-picked') !== key) { return; }
  list.removeAttribute('data-shep-picked');
  if (!stacked()) { return; }
  var title = element.querySelector('[data-facts-key]');
  title.scrollIntoView({block: 'start'});
  title.focus({preventScroll: true});
}).observe(element, {childList: true, subtree: true, attributes: true,
                     attributeFilter: ['data-facts-key']});
shown = drawnKey();
element.addEventListener('click', function (e) {
  if (!(e.target.closest && e.target.closest('button.shep-back-list')) || !list) { return; }
  var row = list.querySelector('.shep-model-rows tr.shep-on') || list.querySelector('.shep-model-rows');
  if (!row) { return; }
  row.scrollIntoView({block: 'center'});
  if (row.tagName === 'TR') { row.focus({preventScroll: true}); }
});
"""

#: **The family graph's own script** (`_graph_html`; on the research preset's
#: rows only, after `MODEL_LIST_JS`, so the Diagnosis drawer and the other
#: elements Diagnosis gives `MODEL_LIST_JS` never carry it). Gradio patches the
#: rows' HTML in place (attributes as the server drew them, any the page set
#: dropped), so what the page decides is kept on the region itself and set
#: again after each drawing, and a new drawing is told by its token
#: (``data-draw``):
#:
#: - **the orientation by measured fit** (``data-fam-orient`` on the region):
#:   left to right where the graph, as drawn, fits the region's width, else
#:   top to bottom; decided again only when what the graph is drawn from
#:   changes (``data-shape``: its runs, their places and words; never a fold,
#:   a pick or a hover), the region's own width changes (not a scroll bar
#:   coming or going) or fonts load, and back to left to right only with room
#:   to spare (no flip-flop); no fit-to-screen, no text shrunk. A subtree
#:   unfolded past the region's width is panned to in the region (I6), the
#:   graph's orientation and the arrow under the pointer kept (I5);
#: - **an arrow** folds or unfolds at once in the page and tells the server
#:   (``{tree: {node: open}}``, the fold state's `subtree_node` entries, kept
#:   for the search only while one is shown, `search_node`; like every act of
#:   the rows, with their listing's generation, `MODEL_LIST_JS`'s ``listGen``):
#:   folded, every
#:   subtree under it folds too and the focus, were it inside, moves to the
#:   arrow; the graph's size held until the pointer and the focus have left
#:   the node and the room it gives back is out of view, so nothing it hid
#:   moves what is under the pointer; unfolded, its direct children only show
#:   (I9, I10); one press is one toggle and one request, nothing redrawn,
#:   nothing selected or loaded;
#: - **a node's body selects its run** (a click, Enter or Space: the button's
#:   own click; G2, I8): the server is sent the run once (``{run: key}``) and
#:   chooses its checkpoint by the list's one selection (`run_choice`), the
#:   details following; nothing loads. Nothing moves under the pointer (I5):
#:   the drawing that answers is located already (the region keeps its
#:   scroll), its orientation is not decided again, the server keeps unfolded
#:   what it showed (`kept_unfolded`), and the lines the new selection takes
#:   away above the node (the note of a selection in a folded branch, the
#:   folded node's words, the note of a selection a filter hides) leave their
#:   room above the graph until the pointer and the focus leave the rows (the
#:   room then given back as the region's scroll, so nothing in view moves)
#:   or the next drawing that is no pick's answer. The focus stays on the body:
#:   side by side, as Newest first keeps it on the row picked; stacked, where
#:   Newest first takes it to the details below the list, that would scroll
#:   the graph from under the pointer;
#: - **the selection in a folded branch** is said on the folded node and
#:   above the graph, "Show it" unfolding the way and bringing it into view
#:   (I10); the selected run (else the one in use) is located once per view,
#:   as the tree's rows are, never after a pick made in the graph or in the
#:   details' "Saved files" (a file of the run already selected and located):
#:   the region stays where the pick was made (I5).
#:
#: Hovering reads nothing and changes nothing (the arrow area shows by the
#: style sheet's ``:hover`` and keyboard focus, ``:focus-visible``).
FAMILY_JS = r"""
(function () {
  var ATTR = 'data-fam-orient';
  var root = function () { return element.querySelector('.shep-fam'); };
  // The rows' region (the block the rows are drawn in): its own width is its
  // border box's, which no scroll bar of its own changes.
  var region = element.closest('.block') || element;
  var wide = function () { return region.getBoundingClientRect().width; };
  var viewFor = function () {
    var marker = element.querySelector('.shep-view-for');
    return marker ? marker.getAttribute('data-for') : '';
  };
  var decide = function () {
    element.__shepFamWidth = wide();
    var fam = root(), forest = fam && fam.querySelector('.shep-fam-forest');
    if (!forest) { return; }
    var width = fam.clientWidth;
    if (!width) { return; }
    var was = element.getAttribute(ATTR) || 'v';
    var kept = forest.style.minWidth;
    forest.style.minWidth = '';
    element.setAttribute(ATTR, 'h');
    var need = forest.offsetWidth;
    forest.style.minWidth = kept;
    var slack = was === 'h' ? 0 : Math.max(24, 1.5 * parseFloat(getComputedStyle(fam).fontSize));
    element.setAttribute(ATTR, need <= width - slack ? 'h' : 'v');
  };
  // The outermost folded list a node is in, or null.
  var shut = function (node) {
    var found = null;
    for (var ul = node.closest('.shep-fam-kids'); ul; ul = ul.parentElement.closest('.shep-fam-kids')) {
      if (ul.hidden) { found = ul; }
    }
    return found;
  };
  var nodeOf = function (li) { return li.querySelector(':scope > .shep-fam-node'); };
  var liOf = function (run) {
    var fam = root();
    return fam ? Array.prototype.find.call(fam.querySelectorAll('li[data-run]'),
      function (x) { return x.getAttribute('data-run') === run; }) : null;
  };
  var note = function () {
    var fam = root(), line = fam && fam.querySelector('.shep-fam-hid');
    if (!line) { return; }
    fam.querySelectorAll('.shep-fam-holds').forEach(function (n) { n.classList.remove('shep-fam-holds'); });
    fam.querySelectorAll('.shep-fam-held').forEach(function (s) { s.hidden = true; });
    var sel = fam.querySelector('.shep-fam-node.shep-sel'), ul = sel ? shut(sel) : null;
    line.hidden = !ul;
    if (!ul) { return; }
    var holder = nodeOf(ul.parentElement);
    holder.classList.add('shep-fam-holds');
    var held = holder.querySelector(':scope > .shep-fam-held');
    if (held) { held.hidden = false; }
    var name = holder.querySelector('.shep-fam-name .shep-w600'), words = line.querySelector('.shep-fam-hid-w');
    var text = name ? name.textContent : '';
    if (words && words.textContent !== text) { words.textContent = text; }
  };
  var locate = function () {
    var fam = root();
    var run = fam ? fam.getAttribute('data-fam-locate') : '';
    var token = run + '|' + viewFor();
    if (!run || token === element.__shepFamLocated) { return; }
    element.__shepFamLocated = token;
    var li = liOf(run);
    if (!li) { return; }
    var ul = shut(li), node = ul ? nodeOf(ul.parentElement) : nodeOf(li);
    if (node) { node.scrollIntoView({block: 'nearest', inline: 'nearest'}); }
  };
  // What scrolls the graph on an axis: the nearest box around it that does, else the page.
  var scroller = function (down) {
    for (var e = element; e && e !== document.body; e = e.parentElement) {
      var cs = getComputedStyle(e), o = down ? cs.overflowY : cs.overflowX;
      if ((o === 'auto' || o === 'scroll') && (down ? e.scrollHeight > e.clientHeight : e.scrollWidth > e.clientWidth)) { return e; }
    }
    return down ? document.scrollingElement : null;
  };
  // A body's pick (I5): the node picked and where it was drawn, while the
  // answer is awaited, and whether a line above it would go with the
  // selection it replaces (`room`, kept above the graph once it has). The
  // operator's own scrolling meanwhile is theirs: nothing is made up for.
  var picked = null, room = 0;
  var scrolled = function (e) {
    if (picked && (e.type !== 'keydown' || /^(Arrow|Page|Home|End)/.test(e.key))) { picked.user = true; }
  };
  ['wheel', 'touchmove', 'keydown'].forEach(function (t) { window.addEventListener(t, scrolled, true); });
  var gone = function (node) {
    var lines = Array.prototype.slice.call(element.querySelectorAll('.shep-fam-hid, .shep-fam-held, .shep-list-outside'));
    return lines.some(function (l) {
      return !l.hidden && l.getClientRects().length && (l.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING);
    });
  };
  var pick = function (body) {
    var run = body.getAttribute('data-run'), node = body.closest('.shep-fam-node');
    // The answer's drawing is located already: the region stays where the pick was made.
    element.__shepFamLocated = run + '|' + viewFor();
    picked = {run: run, at: node.getBoundingClientRect().top, lines: gone(node), user: false};
    trigger('click', listGen({run: run}));
  };
  var forestOf = function () { var fam = root(); return fam && fam.querySelector('.shep-fam-forest'); };
  // The room given back once the pointer and the focus have left the rows (a
  // pick in the details' "Saved files" takes the focus there) as what scrolls
  // the graph scrolls back by it, so nothing in view moves, as the hold gives
  // its room back; only where the region is scrolled less than the room does
  // the rest of it close.
  var letRoomGo = function () {
    if (!room || element.matches(':hover') || element.contains(document.activeElement)) { return; }
    var forest = forestOf();
    if (forest && forest.style.marginTop) {
      var v = scroller(true), to = v.scrollTop - Math.min(room, v.scrollTop);
      forest.style.marginTop = '';
      v.scrollTop = to;
    }
    room = 0;
  };
  element.addEventListener('pointerleave', letRoomGo);
  element.addEventListener('focusout', function () { queueMicrotask(letRoomGo); });
  // The graph's size held after a fold (`hold`), or null.
  var held = null;
  // A new drawing: what the page set on the graph put back first (the hold;
  // a pick's room, the answer's own included), before anything is measured.
  var drawn = function () {
    var fam = root(), token = fam ? fam.getAttribute('data-draw') : null;
    if (token === element.__shepFamDraw) { return; }
    element.__shepFamDraw = token;
    var answer = picked;
    picked = null;
    var forest = forestOf();
    if (held && forest && held.forest === forest) {
      forest.style.minHeight = held.h;
      forest.style.minWidth = held.w;
    }
    answer = answer && fam && fam.getAttribute('data-fam-locate') === answer.run ? answer : null;
    if (!answer) { room = 0; }
    if (forest && room) { forest.style.marginTop = room + 'px'; }
    if (!fam) { element.__shepFamShape = null; return; }
    var form = fam.getAttribute('data-shape');
    if (form !== element.__shepFamShape || wide() !== element.__shepFamWidth) {
      element.__shepFamShape = form;
      decide();
    }
    note();
    var li = answer && answer.lines && !answer.user && forest ? liOf(answer.run) : null;
    var node = li && nodeOf(li);
    if (node && node.getClientRects().length) {
      room = Math.max(0, room - (node.getBoundingClientRect().top - answer.at));
      forest.style.marginTop = room ? room + 'px' : '';
    }
    locate();
  };
  new MutationObserver(drawn).observe(element, {childList: true, subtree: true,
                                               attributes: true, attributeFilter: ['data-draw']});
  if (window.ResizeObserver) {
    // The region's own width (its border box: a scroll bar an unfolded subtree
    // brings is no new width).
    new ResizeObserver(function () {
      if (root() && wide() !== element.__shepFamWidth) { decide(); }
    }).observe(region, {box: 'border-box'});
  }
  if (document.fonts && document.fonts.addEventListener) {
    document.fonts.addEventListener('loadingdone', decide);
  }
  var searched = function () { var fam = root(); return !!fam && fam.hasAttribute('data-searched'); };
  var kidsOf = function (b) { return document.getElementById(b.getAttribute('aria-controls')); };
  var setOpen = function (b, open) {
    b.setAttribute('aria-expanded', open ? 'true' : 'false');
    var kids = kidsOf(b);
    if (kids) { kids.hidden = !open; }
    var node = b.closest('.shep-fam-node');
    if (node) { node.classList.toggle('shep-fam-shut', !open); }
  };
  // The graph's size held after a fold: what it hid would shorten (or, left to
  // right, narrow) what the region scrolls, whose scroll would then be pulled
  // back and move what is under the pointer (I5). Let go only once the pointer
  // and the focus have left the node and giving the room back moves nothing in
  // view: the room it held lies wholly beyond the view (else again as the
  // operator scrolls).
  // The forest's own size without the hold, read from what it draws.
  var natural = function (forest) {
    var box = forest.getBoundingClientRect(), last = forest.lastElementChild, w = 0;
    if (forest.style.minWidth) {
      forest.querySelectorAll('.shep-fam-node').forEach(function (n) {
        var r = n.getBoundingClientRect();
        if (r.height) { w = Math.max(w, r.right + parseFloat(getComputedStyle(n).marginRight) - box.left); }
      });
    }
    return {h: last ? last.getBoundingClientRect().bottom - box.top : 0, w: w};
  };
  var letGo = function (e) {
    if (!held) { return; }
    var forest = held.forest, node = held.node;
    if (forest.isConnected && node.isConnected) {
      if (e && e.type === 'focusout' ? node.contains(e.relatedTarget) || node.matches(':hover')
          : node.matches(':hover') || node.contains(document.activeElement)) { return; }
      var v = scroller(true), x = scroller(false), size = held.size || (held.size = natural(forest));
      var below = v.scrollHeight - v.scrollTop - v.clientHeight;
      var beyond = x ? x.scrollWidth - x.scrollLeft - x.clientWidth : Infinity;
      if (forest.offsetHeight - size.h > below + 0.5) { return; }
      if (forest.style.minWidth && forest.offsetWidth - size.w > beyond + 0.5) { return; }
      forest.style.minHeight = '';
      forest.style.minWidth = '';
    }
    node.removeEventListener('pointerleave', letGo);
    node.removeEventListener('focusout', letGo);
    window.removeEventListener('scroll', letGo, true);
    held = null;
  };
  var hold = function (b) {
    var forest = forestOf(), node = b.closest('.shep-fam-node');
    if (!forest || !node) { return; }
    if (held) {
      held.node.removeEventListener('pointerleave', letGo);
      held.node.removeEventListener('focusout', letGo);
    }
    forest.style.minHeight = forest.offsetHeight + 'px';
    if (element.getAttribute(ATTR) === 'h') { forest.style.minWidth = forest.offsetWidth + 'px'; }
    held = {forest: forest, node: node, size: null, h: forest.style.minHeight, w: forest.style.minWidth};
    node.addEventListener('pointerleave', letGo);
    node.addEventListener('focusout', letGo);
    window.addEventListener('scroll', letGo, true);
  };
  var toggle = function (b) {
    var open = b.getAttribute('aria-expanded') !== 'true', kids = kidsOf(b), set = {};
    if (!open) { hold(b); }
    if (kids) {
      kids.querySelectorAll('button.shep-fam-tog[aria-expanded="true"]').forEach(function (d) {
        setOpen(d, false);
        set[d.getAttribute('data-tree')] = false;
      });
      if (!open && kids.contains(document.activeElement)) { b.focus({preventScroll: true}); }
    }
    setOpen(b, open);
    set[b.getAttribute('data-tree')] = open;
    trigger('click', listGen({tree: set, searched: searched()}));
    note();
  };
  var reveal = function () {
    var fam = root(), sel = fam && fam.querySelector('.shep-fam-node.shep-sel');
    if (!sel) { return; }
    var set = {}, any = false;
    for (var ul = sel.closest('.shep-fam-kids'); ul; ul = ul.parentElement.closest('.shep-fam-kids')) {
      var b = ul.hidden ? nodeOf(ul.parentElement).querySelector('button.shep-fam-tog') : null;
      if (b) { setOpen(b, true); set[b.getAttribute('data-tree')] = true; any = true; }
    }
    if (any) { trigger('click', listGen({tree: set, searched: searched()})); }
    note();
    sel.scrollIntoView({block: 'nearest', inline: 'nearest'});
    var body = sel.querySelector('.shep-fam-body');
    if (body) { body.focus({preventScroll: true}); }
  };
  element.addEventListener('click', function (e) {
    var at = e.target && e.target.closest ? e.target : null;
    if (!at || !root() || !root().contains(at)) { return; }
    var b = at.closest('button.shep-fam-tog');
    if (b) { toggle(b); return; }
    b = at.closest('button.shep-fam-body');
    if (b) { pick(b); return; }
    if (at.closest('button.shep-fam-show')) { reveal(); }
  });
  drawn();
})();
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
        if (Date.now() - started >= limit) {
          wait.hidden = true; if (out) { out.hidden = false; } root.classList.add('shep-waited'); clearInterval(timer);
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
    clicked (a page says the load itself from the server's status, as the
    Diagnosis card does, never from this count); with `guard` (`load_guard`),
    until the reload answers, a close done in the page at once and held for
    the server."""
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


class Refreshed(NamedTuple):
    """What a re-read of the list committed (`ListClock.commit_listing`): the
    listing (carrying its generation), the selection kept, "Show older runs",
    and what it says beside Load (``said``, `list_drawing`): None, or, where
    it cleared a selection the listing no longer lists, `SELECTION_GONE`
    for no selection."""

    listing: Dict[str, Any]
    selected: Optional[str]
    older: bool
    said: Optional[Dict[str, Any]] = None


class ListRead(NamedTuple):
    """**A re-read of the list, read and not committed yet** (the review of
    bd5e678, SL-R3c): the refresh's read step (``ModelList.read_list``)
    returns it and its commit step (``ModelList.commit_list``) commits it, so
    a page that reads more for the same listing (Models' cohort versions and
    usage history) does so between the two, never after the commit. The
    session's clock, what the clock said as the read began, the listing read,
    the view, search and folds it is judged with."""

    session: ListClock
    began: Tuple[int, int]
    listing: Dict[str, Any]
    view: Any
    search: Any
    fold: Dict[str, Any]


class ListClock:
    """**One list's clock in one session** (G07; the review of dcaed68,
    SL-R3): what the list's handlers have committed to the session's state,
    so that an answer is written only where it is still the newest of its
    part. A handler reads its inputs as Gradio gives them when the event is
    processed, then reads files or waits; another handler of the same
    session may commit meanwhile (another workspace chosen, a newer pick, a
    fold, "Show older runs"), and an answer computed from what was read at
    the start would write over it. Every commit goes through this clock,
    under its lock:

    - **the generation**: a count bumped each time a listing of another
      workspace than the one before is committed (A, then B, then A again
      are three), carried by the listing (`GENERATION`) and said by the rows
      drawn from it (`generation_marker`). A pick, a fold or "Show older
      runs" is refused unless the listing it was judged on and the rows it
      was made in are of the generation committed now (`current`): the same
      relative key in another workspace, or in the same one read again after
      a switch, is not the file that was picked;
    - **a re-read** (`commit_listing`) changes nothing when another
      workspace's listing was committed since it began; of the same
      workspace, it writes the listing and keeps the selection committed
      last (a pick made while it read), judged by the one rule
      (`kept_selection`: kept while still listed, else cleared and said
      beside Load, never replaced by another file), never the selection it
      began with; a first listing, or another workspace's, preselects a lone
      candidate (R-27); it keeps "Show older runs", and its answer is
      dropped where "Show older runs" was asked while it read only the
      latest runs;
    - **the folds** are the session's one dict (the ``fold`` State), changed
      in place under this lock by each handler that changes them and
      returned by none, so no answer writes back an older copy.

    One per session: Gradio gives each session a deep copy of a State's
    first value (`gradio.state_holder.SessionState`) and passes that object
    itself to every handler that reads the State, and no handler returns it,
    so every handler of a session holds the same clock; a deep copy is a new
    clock with its own lock.

    **The rule every handler keeps** (the review of bd5e678, SL-R3c: Models'
    opening committed the list's re-read, then read its cohort versions and
    usage history, then wrote what it had committed; a pick or another
    workspace committed during those reads was set back): a handler does
    every read that can wait (files, `list_models`, a page's own data)
    *before* it commits here, and builds its outputs after the commit from
    what it read and what the commit returned only, with no read and no
    wait (a refresh's two steps, `ListRead`).

    **The limit**: the check runs once a handler's reads are done, and Gradio
    writes the handler's outputs to the session just after it returns. Two
    commits can still cross between a handler's commit and Gradio writing
    its outputs (one checked, the other checked and written, then the first
    written); the gap stays that short only because every handler keeps the
    rule above, and a handler that read or waited after its commit would
    widen it by as long as that takes. Nothing here holds or delays an event
    to close it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        #: Bumped each time a listing of another workspace is committed.
        self.generation = 0
        #: Whether a listing was committed, and its workspace.
        self.listed = False
        self.workspace: Any = None
        #: The selection committed last.
        self.selected: Optional[str] = None
        #: "Show older runs" as committed last, and how many times it was asked.
        self.older = False
        self.older_asked = 0

    def __deepcopy__(self, memo: Any) -> "ListClock":
        return type(self)()

    @classmethod
    def seeded(cls, listing: Any = None, selected: Any = None, older: Any = False) -> "ListClock":
        """A clock holding what a handler's inputs say: for a handler called
        with no session clock (a test, a direct call), so it judges by the
        listing and the selection it is given, as one session would."""
        clock = cls()
        if isinstance(listing, Mapping):
            clock.listed, clock.workspace = True, listing.get("workspace")
            clock.generation = list_generation(listing) or 0
        clock.selected = selected if isinstance(selected, str) else None
        clock.older = bool(older)
        return clock

    def began(self) -> Tuple[int, int]:
        """What a re-read's commit is judged against: the generation and the
        times "Show older runs" was asked, as the re-read begins."""
        with self.lock:
            return self.generation, self.older_asked

    def _current(self, listing: Any, sent: Any) -> bool:
        own = listing.get(GENERATION) if isinstance(listing, Mapping) else None
        if own is not None and list_generation(own) != self.generation:
            return False
        if sent is not None and list_generation(sent) != self.generation:
            return False
        return not (self.listed and isinstance(listing, Mapping)
                    and listing.get("workspace") != self.workspace)

    def current(self, listing: Any, sent: Any = None) -> bool:
        """Whether an act judged on `listing` (and made in rows that said the
        generation `sent`, when the page sent one) is of the list as it is
        committed now."""
        with self.lock:
            return self._current(listing, sent)

    def commit_listing(self, began: Tuple[int, int], listing: Mapping[str, Any], *,
                       view: Any, preset: Optional[Preset], search: Any,
                       fold: Dict[str, Any]) -> Optional[Refreshed]:
        """Commit a re-read (`began` its start), or None where its answer is
        stale (the class's rules). Of the same workspace: the selection
        committed last while still listed; one the listing no longer lists is
        cleared, never replaced, and said beside Load (`SELECTION_GONE`).
        Another workspace's listing, or the first, starts afresh: a new
        generation, nothing selected but a lone candidate, no fold, "Show
        older runs" off."""
        with self.lock:
            generation, asked = began
            if generation != self.generation:
                return None
            workspace = listing.get("workspace")
            gone = False
            if self.listed and workspace == self.workspace:
                if self.older and listing.get("runs_unread"):
                    if asked != self.older_asked:
                        return None
                    # Read with the latest runs only: what it holds is shown so.
                    older = False
                else:
                    older = self.older
                key = kept_selection({"workspace": workspace}, listing, self.selected, view,
                                     preset, search=search)
                gone = self.selected is not None and key is None and not listing.get("refused")
            else:
                self.generation += 1
                self.listed, self.workspace = True, workspace
                older = False
                key = kept_selection(None, listing, None, view, preset, search=search)
                fold.clear()
            self.selected, self.older = key, older
            said = ({"key": None, "html": f'<div class="shep-reason shep-reason-grey">'
                                          f'{r.e(SELECTION_GONE)}</div>'} if gone else None)
            return Refreshed({**listing, GENERATION: self.generation}, key, older, said)

    def commit_pick(self, listing: Any, sent: Any, key: str, fold: Dict[str, Any],
                    unfold: Mapping[str, Any]) -> bool:
        """Commit a pick of `key` judged on `listing` (and what it keeps
        unfolded, `kept_unfolded`), or refuse it (`current`)."""
        with self.lock:
            if not self._current(listing, sent):
                return False
            self.selected = key
            fold.update(unfold)
            return True

    def commit_folds(self, listing: Any, sent: Any, fold: Dict[str, Any],
                     changes: Mapping[str, Any]) -> bool:
        """Keep what the page folded or unfolded in `listing`'s rows, or
        refuse it (`current`)."""
        with self.lock:
            if not self._current(listing, sent):
                return False
            fold.update(changes)
            return True

    def commit_older(self, generation: int, listing: Any, sent: Any,
                     read: Optional[Mapping[str, Any]]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Commit "Show older runs" asked in `listing`'s rows, with `read`
        (every run of its workspace, or None where none was unread): refused
        where another workspace's listing was committed since it was asked
        (`generation`) or `read` is of another workspace than `listing`.
        Returns whether it
        was, and the listing to write."""
        with self.lock:
            if generation != self.generation or not self._current(listing, sent):
                return False, None
            if read is not None and (not isinstance(listing, Mapping)
                                     or read.get("workspace") != listing.get("workspace")):
                return False, None
            self.older = True
            self.older_asked += 1
            return True, ({**read, GENERATION: self.generation} if read is not None else None)

    def reselect(self, listing: Any, key: str) -> bool:
        """Select `key` again after a Load (Models), only while nothing is
        selected and `listing` is the one committed now: a pick made
        meanwhile stays."""
        with self.lock:
            if self.selected is not None or not self._current(listing, None):
                return False
            self.selected = key
            return True

    def drop_search_folds(self, fold: Dict[str, Any]) -> None:
        """Drop what was folded while a search showed the list (`search_node`)."""
        with self.lock:
            for node in [k for k in fold if str(k).startswith(search_node(""))]:
                del fold[node]

    def folds(self, fold: Any) -> Dict[str, Any]:
        """The folds as they are now, a copy read under the lock."""
        with self.lock:
            return dict(fold) if isinstance(fold, Mapping) else {}


@dataclass
class ModelList:
    """The components of one model list, for the page that places it.

    - ``workspace``: the workspace select (the server's list, the shared
      option labels `renderers.workspace_choices`);
    - ``listing``, ``selected``, ``plan``: this session's listing, selection and
      open confirmation (`gr.State`);
    - ``view``: `gr.State`, None or the page's view of the list
      (``task``, ``keys``, ``groups``, ``context``, ``parents``, ``edges``,
      ``bounds``, ``workspace``, ``note``, ``shaped_for``: the module
      docstring); the list is drawn again on its ``.change``;
    - ``search``, ``search_clear``: the search box and its Clear (`search_hit`);
      ``fold``: `gr.State`, what the operator folded and unfolded
      (`tree_view`; the family graph's subtrees, `subtree_node`; what was
      folded during a search, `search_node`), the session's one dict,
      changed in place under the list's clock and returned by no handler;
    - ``drawn``: `gr.State`, the draw token: an event that changes what the
      list shows writes a new one, and the list draws on its change;
    - ``searched``: `gr.State`, the search token: each change of the search
      writes a new one once the list has handled it; a page that follows the
      search listens to its change, never to the box's (`create_model_list`);
    - ``clock``: `gr.State`, the session's `ListClock`, which no handler
      returns: a page handler that writes the list's selection commits it
      there (Models' selection again after a Load, `ListClock.reselect`);
    - ``phase``: `gr.State`, ``"idle"``, ``"confirming"`` or ``"loading"``
      (Diagnosis disables Run Diagnosis while it is not idle);
    - ``refresh``: keyword arguments for an event that (re)reads the list:
      ``ctx.on_page_open("models", handle.refresh)``; it changes the
      session's state and the draw token, and the list is drawn from that
      state by its one draw (``drawn``, `list_drawing`);
    - ``read_list`` and ``commit_list``: the refresh's two steps, for a page
      that reads more for the same listing (Models' opening, `ListRead`):
      ``read_list`` takes the refresh's inputs (the request after the
      selection, the clock last) and reads the list, committing nothing;
      ``commit_list`` commits that read through the clock and returns the
      refresh's outputs, or None where another workspace was committed since
      it began (then nothing may change). The page reads between the two,
      never after the commit (`ListClock`);
    - the rendered parts: ``count``, ``rows``, ``facts``, ``reason``, ``note``,
      ``load_button``, ``confirm``, ``confirm_html``, ``reload_button``,
      ``keep_button``; ``head_extra`` and ``foot_extra``, what the page's
      ``head`` and ``foot`` builders returned;
    - ``reload_done``: the event that ends a Reload (its answer: reloaded or
      not, the phase back to idle or confirming), for a page to chain
      ``.then(…)`` on, as Diagnosis puts Run Diagnosis back and, after an
      answer that changed nothing served (no ``served_model_changed``),
      reads its card again;
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
    search: Any = None
    search_clear: Any = None
    fold: Any = None
    drawn: Any = None
    searched: Any = None
    clock: Any = None
    read_list: Optional[Callable[..., ListRead]] = None
    commit_list: Optional[Callable[[ListRead], Optional[Tuple[Any, ...]]]] = None


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
                      tools: Optional[Callable[[], Any]] = None,
                      foot: Optional[Callable[[], Any]] = None,
                      on_loading: Optional[Mapping[str, Any]] = None,
                      close_while_loading: Optional[Mapping[str, Any]] = None,
                      picks: Optional[Callable[[], Sequence[Any]]] = None) -> ModelList:
    """Build the shared list in the current layout context and wire its events.

    `ctx` is the page context (`src.webui.shell.PageContext`): its chips are
    refreshed and its ``served_model_changed`` trigger set after a Load.
    `elem_id` defaults to ``shep-models-<preset>``, so both presets can sit in
    one page. `head` and `foot`, when given, are called inside the list's
    header and footer rows to place the page's own components there; `tools`
    inside its tools row, after the search group: on a page (the ``page``
    layout) the tools row is one bar (`style.py`'s ``.shep-bar``) whose
    groups wrap whole, the search group (label, box, Clear) growing from its
    basis and never under its box's floor, so a page's own group (Models'
    Order, a ``.shep-grp``) shares the line while both fit and moves under
    it whole when they do not; a message the page adds there, and the count
    line, are the bar's notes (``.shep-bar-note``), each a row of its own
    under the groups. The drawer layout keeps its field row (no tools).
    `on_loading` (``{"fn", "inputs", "outputs"}``) adds the page's own outputs
    to the step "Reload with …" runs before the reload, in the same answer (so
    they show before the reload holds the server): ``fn(plan, *inputs)``
    returns one value per output, `plan` None when there was nothing to load
    (Diagnosis: Run Diagnosis off with its reason; its card says the load
    from the server's status). `close_while_loading` (``{"hide", "then", "also",
    "openers"}``, `load_guard`) is what Close does while a reload runs when
    the list sits in a page's drawer: the drawer hides at once, and its own
    close is sent when the reload answers; the page chains
    ``after_close_js`` at the end of that close. `picks`, when given, is
    called once `head` and `foot` have placed the page's components and
    returns the page's regions whose ``pick`` acts (`PICK_ACT`, sent by
    `renderers.TRIGGER_JS`) select a listed checkpoint as a row's click does,
    by the list's one selection (Models' details: their "Saved files")."""
    chosen = PRESETS[preset]
    elem_id = elem_id or f"shep-models-{preset}"
    listing = gr.State(empty_listing())
    selected = gr.State(None)
    plan = gr.State(None)
    view = gr.State(None)
    phase = gr.State("idle")
    show_older = gr.State(False)
    #: What the operator folded and unfolded, ``{node: bool}`` (`tree_view`):
    #: the session's one dict, changed in place under the clock, never an output.
    fold = gr.State({})
    #: The session's clock (`ListClock`): never an output, so every handler of
    #: a session holds the same one.
    clock = gr.State(ListClock())
    #: What an act said beside Load, ``{"key", "html"}`` (`list_drawing`).
    said = gr.State(None)
    #: The draw token: each event that changes what the list shows writes a new
    #: one (`draw_token`), and the list is drawn on its change (`_draw`).
    drawn = gr.State(0)
    #: The search token: each change of the search writes a new one, so a page
    #: follows the search once the list has handled it (Models' details) and
    #: never listens to the box's change itself (`_searched`).
    searched = gr.State(0)
    # The page preset (Models) keeps its header and its rows in one column
    # (``shep-model-main``), so the page can scroll them as one region when
    # they do not fit together (M5-3 review UI-1); the drawer's rows need none.
    # The column is a panel's (``shep-edge``, `style.py`'s one content edge):
    # its fields' labels and boxes stand on the edge its title and text do.
    with gr.Column(elem_id=elem_id, elem_classes=["shep-model-list", f"shep-preset-{preset}"]):
        main = (gr.Column(elem_classes=["shep-model-main", "shep-edge"]) if chosen.layout == "page"
                else contextlib.nullcontext())
        with main:
            # A row of fields (the drawer's head) lines up on the boxes (D03,
            # `style.py`'s `.shep-field-row`); the page preset lays its head out
            # in its page's grid.
            head_classes = ["shep-model-head"] + (["shep-field-row"] if chosen.layout == "drawer"
                                                  else [])
            with gr.Row(elem_classes=head_classes):
                workspace = gr.Dropdown(label="Workspace", choices=[], value=None,
                                        elem_id=f"{elem_id}-workspace", interactive=True,
                                        scale=3)
                head_extra = head() if head is not None else None
            # The search and what the list shows, on a line of their own under
            # the head (R-27: the count line never wraps beside a select).
            drawer = chosen.layout == "drawer"
            tools_classes = ["shep-model-tools"] + (["shep-field-row"] if drawer else ["shep-bar"])
            with gr.Row(elem_classes=tools_classes):
                # "Search" said beside its box, as "Order" is (no line of its
                # own above it), its Clear after it: one group of the bar.
                with gr.Row(elem_id=f"{elem_id}-searchrow",
                            elem_classes=["shep-search-row"] + (
                                [] if drawer else ["shep-grp", "shep-grp-grow"])):
                    plain_html('<span class="shep-search-lbl">Search</span>',
                               elem_id=f"{elem_id}-search-lbl")
                    search = gr.Textbox(label="Search", show_label=False, container=False,
                                        placeholder="Name or fingerprint",
                                        elem_id=f"{elem_id}-search", lines=1, max_lines=1,
                                        scale=1, min_width=96, autoscroll=False)
                    clear = gr.Button("Clear", size="sm", scale=0, min_width=0,
                                      elem_id=f"{elem_id}-search-clear")
                if tools is not None:
                    tools()
                count = plain_html("", elem_id=f"{elem_id}-count",
                                   elem_classes=["shep-field-aside" if drawer else "shep-bar-note"])
            with gr.Row(elem_classes=["shep-model-body"]):
                # Two areas, each its own region where it scrolls (UI-07; the
                # shared UI check's ``shep-area``): the rows and the facts.
                # The family graph's script only where a page can draw it
                # (Models' By family: `Preset.family`), never in the drawer.
                rows = plain_html(
                    '<div class="shep-muted">The list is read when it is opened.</div>',
                    elem_id=f"{elem_id}-rows",
                    js_on_load=MODEL_LIST_JS + (FAMILY_JS if chosen.family else ""),
                    elem_classes=["shep-model-rows", "shep-area"])
                facts = plain_html(facts_html(empty_listing(), None),
                                   elem_id=f"{elem_id}-facts", js_on_load=FACTS_JS,
                                   elem_classes=["shep-model-facts", "shep-area"])
        with gr.Row(elem_classes=["shep-model-foot"]):
            load = gr.Button(LOAD_LABEL, variant="primary", interactive=False,
                             elem_id=f"{elem_id}-load", scale=0)
            reason = plain_html("", elem_id=f"{elem_id}-reason")
            note = plain_html("", elem_id=f"{elem_id}-note")
            foot_extra = foot() if foot is not None else None
        with gr.Column(visible=False, elem_id=f"{elem_id}-confirm-box",
                       elem_classes=["shep-model-confirm-box"]) as confirm:
            confirm_html = plain_html("", elem_id=f"{elem_id}-confirm", js_on_load=MODEL_LIST_JS,
                                      elem_classes=["shep-model-confirm"])
            with gr.Row():
                reload_button = gr.Button("Reload", variant="primary", scale=0,
                                          elem_id=f"{elem_id}-reload")
                keep_button = gr.Button("Keep", scale=0, elem_id=f"{elem_id}-keep")

    # **The list is drawn in one place, from the session's state** (G07; the
    # review of 00258f4, SL-R1). Every event that changes what the list shows
    # (a pick, "Show older runs", a search and its Clear, a refresh: the page's
    # opening, a workspace chosen, the Load's and Diagnosis' re-reads; what
    # an act says beside Load) only changes the state and writes a new draw
    # token (`draw_token`); a page changes the list's view, which is a state
    # too. `_draw` runs on the token's and the view's change and draws the
    # rows, the count and what describes the selection (`list_drawing`) from
    # the state and the search box as they are when it runs. Why the last draw
    # is never stale: Gradio reads a `gr.State` input when the event is
    # processed, and the page reads a component's value (the search box) when
    # it asks for the event, not before; a change's outputs (a state) are
    # written before its answer is applied and its state changes fire, so the
    # page asks for the draw only once the change has landed, and the draw
    # that follows the last change reads it. The box is read as it is then:
    # a change of its text (whatever made it; Clear empties it in the page)
    # writes no state but the tokens (the draw's, the search's) and the folds,
    # the search's dropped (a search answered late asks for one more draw,
    # which reads the box as it is, never the text it was typed with; the
    # review of 5171661). Two draws of
    # one list can be asked at once (Models' opening writes the token and the
    # view in one answer, and Gradio 6.30 checks "always_last" for both before
    # either is sent). When only the server is slow, what keeps the last
    # answer drawn the last one asked is the queue: one concurrency id per
    # list, limit 1, so the server runs a list's draws one at a time in the
    # order they were asked and sends their answers in that order on the
    # page's one event stream; "always_last" keeps one draw asked while one is
    # pending and sends it once that one is answered. Two gaps are left
    # (OPS-SL-R3): a network delay that holds back the join answer of the
    # first draw past a later draw can leave the list stale until the next
    # change, and a draw asked while the pending one fails is not sent until
    # the next change. A draw sent outside the queue
    # (``queue=False``) was tried and withdrawn: its answers can arrive out
    # of the order asked (the adversarial check of 37912a9). A handler's own
    # inputs (the search box's text as a pick was clicked) choose what it
    # changes, never what is drawn: an answer that comes late (a pick held
    # while a search was answered) draws nothing from them. What a handler
    # commits is judged by the session's clock (`ListClock`; the review of
    # dcaed68, SL-R3) once its reads are done: an answer for another workspace
    # than the list now shows (the listing's generation) is refused, a re-read
    # keeps a pick made while it read, and the folds are changed in place,
    # never written back. Every handler reads first (files, the listing, a
    # page's own data) and commits last, building its outputs from what it
    # read and what the commit returned only (the review of bd5e678, SL-R3c:
    # Models' opening read its own data after the commit and wrote back what
    # it had committed over a pick made meanwhile). Gradio writes a handler's
    # outputs just after it returns: two commits can still cross between a
    # commit and that write, a gap only that short because no handler reads
    # or waits after its commit (`ListClock`).

    def _read(name, older, current_view, previous, key, request: gr.Request, query="",
              fold_state=None, session=None) -> ListRead:
        # A selection still listed in the same workspace stays selected (M5-3
        # review UI-3: the re-read after a Load dropped a row picked meanwhile;
        # D04: whatever the filter, the search or the folds show), and so does
        # what the operator unfolded; another workspace starts afresh. The
        # drawer opens the latest runs only until "Show older runs" (BACKLOG 39),
        # asked of one workspace: another workspace picked opens its latest
        # runs (it opened every checkpoint of it, the session's "Show older
        # runs" kept); the same one read again keeps it, as it keeps the folds.
        # The selection kept is the one committed last (a pick made while
        # this read, the review of dcaed68, SL-R3b), never the one it began
        # with; an answer read before another workspace was committed changes
        # nothing (`ListClock.commit_listing`). This step reads and commits
        # nothing; `_commit` commits it.
        session = session if isinstance(session, ListClock) else ListClock.seeded(
            previous, key, older)
        fold_state = fold_state if isinstance(fold_state, dict) else {}
        began = session.began()
        older = bool(older) and isinstance(previous, Mapping) and previous.get(
            "workspace") == name
        data = list_models(name, request, family=chosen.family,
                           latest_runs=None if older else chosen.latest_runs)
        return ListRead(session, began, data, view_for_listing(current_view, previous, data),
                        query, fold_state)

    def _commit(read: ListRead) -> Optional[Tuple[Any, ...]]:
        # The refresh's outputs, from the read and the commit only (no read,
        # no wait after it); None where its answer is stale.
        done = read.session.commit_listing(read.began, read.listing, view=read.view,
                                           preset=chosen, search=read.search, fold=read.fold)
        if done is None:
            return None
        choices = r.workspace_choices(read.listing.get("workspaces") or [])
        # Reading the list is no act: a refused read says why in the list
        # (`refused_line`); Load's reason is said when Load is clicked, or
        # beside it at once for a row whose record is never loaded. What was
        # said beside Load was said of the list read before; a selection the
        # list no longer lists is cleared and said (`Refreshed.said`).
        return (done.listing, done.selected,
                gr.update(choices=choices, value=read.listing.get("workspace")), done.older,
                done.said, draw_token())

    def _refresh(name, older, current_view, previous, key, request: gr.Request, query="",
                 fold_state=None, session=None):
        # The list read, then committed (`_read`, `_commit`); a stale answer
        # changes nothing.
        done = _commit(_read(name, older, current_view, previous, key, request, query,
                             fold_state, session))
        return done if done is not None else (gr.update(),) * len(refresh_outputs)

    # What the list draws, where the page shows that it is being read.
    read_into = [workspace, count, rows, facts, load, reason, note]
    refresh_outputs = [listing, selected, workspace, show_older, said, drawn]
    refresh = {"fn": _refresh,
               "inputs": [workspace, show_older, view, listing, selected, search, fold, clock],
               "outputs": refresh_outputs, "show_progress": "minimal",
               "show_progress_on": read_into}
    picked(workspace)(**refresh)

    def _select(data, current, older, current_view, query, fold_state, evt: gr.EventData,
                request: gr.Request, session=None):
        # What it changes: (selected, show_older, listing, what was said
        # beside Load, the draw token; the folds in place); what it was asked
        # with (`query`: the box's text as the pick was made) chooses, never
        # draws. The list's "Clear search" is no act of it: the page clicks
        # the list's Clear (`MODEL_LIST_JS`). Each act is committed only for
        # the listing the list shows now: judged on that listing, made in rows
        # drawn from it (the generation the page sends, ``gen``;
        # `ListClock.current`), so an act answered after another workspace
        # was chosen, or one waiting in the queue across the switch, selects
        # nothing there (the review of dcaed68, SL-R3a).
        session = session if isinstance(session, ListClock) else ListClock.seeded(
            data, current, older)
        raw = getattr(evt, "_data", None)
        raw = raw if isinstance(raw, Mapping) else {}
        sent = raw.get("gen")
        same = gr.update()
        nothing = (same,) * 5
        fold_state = fold_state if isinstance(fold_state, dict) else {}
        tree = raw.get("tree") if isinstance(raw.get("tree"), Mapping) else None
        if isinstance(raw.get("fold"), str) or tree is not None:
            # The page folded or unfolded it already (`MODEL_LIST_JS`; the
            # family graph's arrows, `FAMILY_JS`); the state keeps it for the
            # next drawing, which nothing asks for now. What is folded while a
            # search shows the list is that search's (`search_node`): clearing
            # it brings the operator's own back.
            store = search_node if raw.get("searched") else str
            changes = {}
            if isinstance(raw.get("fold"), str):
                changes[store(raw["fold"])] = bool(raw.get("open"))
            for node, opened in (tree or {}).items():
                if isinstance(node, str) and node.startswith("t:"):
                    changes[store(node)] = bool(opened)
            session.commit_folds(data, sent, fold_state, changes)
            return nothing
        key = raw.get("key")
        if "act" in raw and (raw["act"] != PICK_ACT or not isinstance(key, str)
                             or key.startswith("__")):
            # A page's region picks a listed file only (`PICK_ACT`), never
            # the list's own acts.
            return nothing
        if isinstance(raw.get("run"), str):
            # A family graph node's body (`FAMILY_JS`): its run's checkpoint,
            # as a Run → Checkpoint choice chooses it (`run_choice`), through
            # the selection a row's pick goes through below.
            key = run_choice(data, current_view, chosen, raw["run"], current, search=query,
                             show_older=older)
        if key == "__older__":
            # The runs not opened yet are read now (BACKLOG 39).
            began, _asked = session.began()
            if not session.current(data, sent):
                return nothing
            read = (list_models(data.get("workspace"), request, family=chosen.family)
                    if data.get("runs_unread") else None)
            done, listed = session.commit_older(began, data, sent, read)
            if not done:
                return nothing
            return (same, True, listed if listed is not None else same, same, draw_token())
        item = find_item(data, key)
        if item is None or item not in shown_items(data, current_view, chosen):
            return nothing
        # In the family graph a pick folds nothing it shows (I5): what was
        # unfolded only for the selection before it stays unfolded.
        kept = kept_unfolded(data, current_view, chosen, session.folds(fold_state), current, key,
                             search=query, show_older=older)
        if not session.commit_pick(data, sent, key, fold_state, kept):
            return nothing
        # A pick is a new selection: what was said beside Load is not its.
        return (key, same, same, None, draw_token())

    # A row's pick, and a page region's (`picks`: ``{act: "pick", key}``), are
    # one selection.
    gr.on([rows.click, *(region.click for region in (picks() if picks else ()))], _select,
          inputs=[listing, selected, show_older, view, search, fold, clock],
          outputs=[selected, show_older, listing, said, drawn],
          show_progress="hidden", trigger_mode="multiple")

    def _searched(fold_state, session=None):
        # The search is the list's to draw, from the box as it is when the
        # list is drawn (never the text this was typed with: an answer that
        # comes late, after a Clear, would draw that text again under the
        # empty box; the review of 5171661); the selection stays. What it finds
        # is revealed as the list is drawn (`fold_open`: a run holding a file
        # found by its own name or fingerprint, the graph's ancestors of a run
        # found), the operator's own folds untouched; what was folded while
        # the search before this one showed the list is dropped, so a cleared
        # search brings the operator's folds back (G7); in place, under the
        # session's clock.
        session = session if isinstance(session, ListClock) else ListClock()
        if isinstance(fold_state, dict):
            session.drop_search_folds(fold_state)
        token = draw_token()
        return token, token

    # **One path for every change of the search** (the adversarial check of
    # 37912a9, F2): the box's ``change``, which Gradio 6.30's Textbox sends
    # for each new value, whatever made it (a key typed, Backspace or Delete,
    # a cut, a paste, an input method's composition, Clear, the list's "Clear
    # search"), and never as the page loads. Its ``input`` was sent from the
    # box's keypress handler alone, so a deleted or pasted text was never
    # searched: the list kept drawing the old search under the box. This is
    # the box's one listener: a page follows the search on the search token
    # (`searched`), which this writes. Gradio 6.30 sends a listener it held
    # back ("always_last") by dispatching its whole event again, so two such
    # listeners of the box's change (this and Models' `_sync`, as it was) set
    # each other off without end (measured: `_searched`, `_sync` and a draw,
    # again and again, while the page stayed open); a state's change is
    # dispatched to each of its listeners alone.
    search.change(_searched, inputs=[fold, clock], outputs=[drawn, searched],
                  show_progress="hidden", trigger_mode="always_last")

    # Clear empties the box in the page, with no request of its own: the box's
    # change is the search's one path (Gradio 6.30 applies a front-end
    # function's return to its outputs). The list's "Clear search" clicks it.
    clear.click(None, inputs=None, outputs=[search], js=CLEAR_SEARCH_JS,
                show_progress="hidden")

    def _draw(data, key, older, current_view, query, fold_state, said_now, session=None):
        # The page changed its view (a filter, an order), or the state
        # changed (the token): the selection stays, said where the view does
        # not show it (D04: a filter never changes what is selected). A draw
        # that fails changes nothing (the list keeps its last drawing): Gradio
        # 6.30 drops a deferred call after a failed one, so the failure must
        # not escape (as Models' `_sync`). The folds are read under the
        # session's clock (another handler changes them in place).
        session = session if isinstance(session, ListClock) else ListClock()
        try:
            drawing = list_drawing(data, key, chosen, show_older=older, view=current_view,
                                   search=query, fold=session.folds(fold_state), said=said_now,
                                   load_id=load.elem_id)
        except Exception:  # noqa: BLE001 - the next change draws the list again
            logger.exception("The model list could not be drawn")
            return (gr.update(),) * 6
        return (drawing.rows, drawing.count, drawing.facts, gr.update(**drawing.load),
                drawing.reason, drawing.note)

    gr.on([drawn.change, view.change], _draw,
          inputs=[listing, selected, show_older, view, search, fold, said, clock],
          outputs=[rows, count, facts, load, reason, note], show_progress="hidden",
          trigger_mode="always_last", concurrency_id=f"{elem_id}-draw", concurrency_limit=1)

    def _request(data, key, request: gr.Request):
        # A refusal is said beside Load for the selection asked (`said`), by
        # the list's draw, so a later drawing keeps it while that selection is
        # drawn; a plan opened says nothing there.
        new_plan, html_ = request_load(data, key, request)
        if new_plan is None:
            return (None, gr.update(visible=False), "", {"key": key, "html": html_},
                    gr.update(), gr.update(), "idle", draw_token())
        new, old = r.short_hex(new_plan["to"]["sha256"]), r.short_hex(new_plan["from"].get("sha256"))
        return (new_plan, gr.update(visible=True), html_, None,
                gr.update(value=f"Reload with {new}", visible=True),
                gr.update(value=f"Keep {old}" if new_plan["from"].get("sha256") else "Cancel"),
                "confirming", draw_token())

    load.click(_request, inputs=[listing, selected],
                      outputs=[plan, confirm, confirm_html, said, reload_button, keep_button,
                               phase, drawn]).then(**ctx.chip_refresh)

    page_inputs = list((on_loading or {}).get("inputs") or [])
    page_outputs = list((on_loading or {}).get("outputs") or [])

    def _loading(current, *page_values):
        own = loading_buttons(current)
        if not page_outputs:
            return own
        extra = on_loading["fn"](current or None, *page_values)
        return (*own, *extra)

    def _confirm(current, request: gr.Request):
        # **D8, the status and the reload run here, in Gradio's worker thread
        # (a plain ``def``), never on the event loop** (BACKLOG 28), as the
        # API route's reload runs: a diagnosis accepted on the served model
        # meanwhile finishes on it, and the server answers meanwhile.
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
        served = step.plan["from"].get("sha256") if still_open else None
        old = r.short_hex(served) if served else ""
        new = r.short_hex(step.plan["to"].get("sha256")) if still_open else ""
        return (step.plan, gr.update(visible=True), step.confirm_html, step.reason_html, token,
                "confirming" if still_open else "idle",
                gr.update(visible=still_open,
                          **({"value": f"Reload with {new}"} if still_open else {})),
                gr.update(value=(f"Keep {old}" if still_open and old else
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
        return None, gr.update(visible=False), None, draw_token(), "idle"

    def _after_keep(current):
        # Only while nothing was asked since: a Load clicked meanwhile has put
        # its own confirmation in the box.
        if current is not None:
            return gr.update(), gr.update()
        return "", gr.update(visible=True)

    after_close_js = load_closed_js(elem_id)
    keep_button.click(_keep, inputs=None, outputs=[plan, confirm, said, drawn, phase],
                      show_progress="hidden").then(
        _after_keep, inputs=[plan], outputs=[confirm_html, reload_button],
        show_progress="hidden").then(None, js=after_close_js)

    def _read_again(key, evt: gr.EventData):
        raw = getattr(evt, "_data", None)
        if not (isinstance(raw, Mapping) and raw.get("key") == "__read_again__"):
            return gr.update(), gr.update()
        # Said beside Load, by the list's draw (as `_request`'s refusal).
        return {"key": key, "html": read_again()}, draw_token()

    confirm_html.click(_read_again, inputs=[selected], outputs=[said, drawn],
                       show_progress="hidden")

    return ModelList(chosen, elem_id, workspace, listing, selected, plan, view, phase,
                     show_older, count, rows, facts, reason, note, load, confirm,
                     confirm_html, reload_button, keep_button, refresh, head_extra, foot_extra,
                     reload_done, after_close_js, search, clear, fold, drawn, searched, clock,
                     _read, _commit)


__all__ = [
    "BASE_REFUSAL",
    "BACK_TO_LIST",
    "CLEAR_SEARCH_JS",
    "FACTS_JS",
    "FAMILY_FILTERS",
    "FAMILY_JS",
    "FILTER_WORDS",
    "FROM_CHANGED",
    "GENERATION",
    "ListClock",
    "ListDrawing",
    "ListRead",
    "LoadStep",
    "MODEL_LIST_JS",
    "ModelList",
    "NOTHING_RELOADED",
    "PAGE_WAIT_SECONDS",
    "PAGE_WAIT_WORDS",
    "PICK_ACT",
    "PRESETS",
    "Preset",
    "REPLACES_NOTE",
    "SELECTION_GONE",
    "TASK_FILTERS",
    "TO_CHANGED",
    "can_load",
    "confirmation_html",
    "count_line",
    "create_model_list",
    "drawn_files",
    "facts_html",
    "family_views",
    "fold_open",
    "generation_marker",
    "graph_shape",
    "find_item",
    "finish_load",
    "item_key",
    "kept_unfolded",
    "kept_selection",
    "older_html",
    "list_drawing",
    "list_generation",
    "list_html",
    "list_models",
    "load_answered_js",
    "lone_key",
    "load_closed_js",
    "load_guard",
    "load_reason_html",
    "CHOOSE_A_MODEL",
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
    "run_choice",
    "search_node",
    "shown_items",
    "start_wait_js",
    "subtree_node",
    "task_filter",
    "view_for_listing",
    "view_marker",
    "workspace_line_html",
]
