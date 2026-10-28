"""
Research › Prepare › Fine-tune (K6)
===================================
`docs/working/scorer-retraining/README.md` §1 and §8: a base's task models,
trained as one Fine-tune start of one to three children (Disease, Gene,
Patients-like-me), each an independent run of its own, in order, from one
listed base ("children never chain"). Built by `create_finetune_page` under
the page contract (`src/webui/shell.py`).

Module: src/webui/components/finetune_page.py

**The form** (left panel; K6 decision 11):

- **Dataset**: its own select over the server's workspace list
  (`workspaces.read_workspaces`, the shared option label), starting from the
  Training run bar's selection the first time the page opens and never
  writing it; the dataset identity under it (`renderers.dataset_identity_html`)
  is the one place its fingerprint is shown.
- **The parent**: Run → Checkpoint over the workspace's **listed bases only**
  (`server_lists.list_checkpoints(learning_stage="base")`): a task model (a
  child never chains), a file without a model spec and an unreadable file are
  never offered; a weights-only or early-stopped base is. Listing opens every
  checkpoint, so it is an operator control (D8). The chosen base's summary
  (key-value rows) says its run's directory, file, configuration, Trainer
  validation figure, the dataset it trained on and the model fingerprint
  Start sends.
- **Tasks**: a checkbox group in `TASKS` order whose option text says whether
  each is ready on this workspace (`server_lists.task_readiness`, small JSON
  files only, no D8), and why not in the line under it.
- **Epochs and Seed only**: the other common settings have a reference value
  per task, so one value would overwrite every child's. Each starts at the
  recipe's and is sent only when it differs (`training_console.
  recipe_differences`).
- Device and Run name; **Start**, its reason line, and any refusal inline
  beside it (escaped).

**Start** (D8 first): the workspace and the parent are resolved by the server
from its lists again (`resolve_workspace`, `resolve_checkpoint`; never a
client path), then the job runner's one Fine-tune start
(`TrainingManager.start_finetune`) with the parent's listed digest and the
dataset fingerprint shown. Its checks before anything starts (the parent a
base with weights, its graph, each task's fixture, the settings) are the
runner's alone (decision 3); this page adds none.

**The job card** (right panel): `get_status()["finetune"]`, one row per child
(Task | Status | Run | Best save | Validation (Trainer)). Each status is a mark
of colour, shape and word (`training_console.PILL_KIND`, the
``--shep-status-*`` tokens). A child's products, the best save by the Trainer
validation figure its saved files record and that figure (never a test
result), are read for an operator and only once the child has ended
(`list_checkpoints(run=)`, opening that run's files; kept per session by the
child's run and status). Stop asks first: the Training page's stop logic
(`training_console.ask_stop`, `confirm_stop`) behind this page's own D8-checked
handlers, reaching a running child of the start shown only (`running_child`),
never another page's job; the question names the queued children. Once the
next base or continued run cleared the Fine-tune status, the card says so in
its own words.

**Polls**: the shell's slow timer, ``queue=False``, doing nothing while another
page is shown (``ctx.location``); the card again after every action. Per
browser session, no module state.

Architecture Note:
    The training service is the job runner (`src.api.services.training_manager`,
    allowed: webui -> api); nothing here imports `src.training`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import gradio as gr

from src.api.services.training_manager import FinetuneInputs, training_manager
from src.config.recipe import FINETUNE, FINETUNE_SOURCES
from src.evaluation.caveats import TASK_NAMES
from src.models.bundle.spec import TASKS
from src.webui.components import renderers as r
from src.webui.components import training_console as tc
from src.webui.components.chips import D8_SHORT
from src.webui.components.help import help_button
from src.webui.components.workspaces import read_workspaces, redact, workspaces_root
from src.webui.operator_access import operator_access
from src.webui.shell import FINETUNE as PAGE
from src.webui.shell import page_elem_id, plain_html

logger = logging.getLogger(__name__)

#: The settings this page offers (decision 11), each a recipe setting with its
#: finetune reference value (`recipe.FINETUNE`).
FINETUNE_FIELDS = ("num_epochs", "seed")
#: A child's states after which its run's saved files are final.
ENDED = ("completed", "failed", "stopped")
#: Each status kind's mark (the WebUI skill §6): its shape and the
#: ``--shep-status-*`` token its colour reads. The spinner (None) is Running's
#: alone; a stopping child, still busy but asked to stop, is blue with the
#: Stop button's own ■.
_MARKS = {"ok": ("●", "ok"), "grey": ("○", "neutral"), "idle": ("○", "neutral"),
          "red": ("✕", "error"), "running": (None, "info"), "accent": ("■", "info")}


# =============================================================================
# The form's pieces
# =============================================================================
def task_choices(readiness: Mapping[str, Optional[str]]) -> List[Tuple[str, str]]:
    """The Tasks options, in `TASKS` order: the task's name and whether it is
    ready on the workspace ("Gene · ready", "Gene · not ready"; why not is the
    line under the group, `tasks_note_html`). Each at most
    `training_console.OPTION_CHARS` characters, one separator. A task the
    readiness does not name (before a dataset is read) is not ready."""
    def ready(task: str) -> bool:
        return task in readiness and readiness[task] is None

    return [(f"{TASK_NAMES[task]} · {'ready' if ready(task) else 'not ready'}", task)
            for task in TASKS]


def tasks_note_html(readiness: Mapping[str, Optional[str]]) -> str:
    """Why each task that is not ready is not, grouped by reason ("Not ready:
    Gene and Patients-like-me: no functional fixture."), escaped; empty when
    every task is ready."""
    reasons: Dict[str, List[str]] = {}
    for task in TASKS:
        why = readiness.get(task)
        if why is not None:
            reasons.setdefault(why, []).append(TASK_NAMES[task])
    if not reasons:
        return ""
    parts = []
    for why, names in reasons.items():
        joined = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
        parts.append(f"{joined}: {why}")
    return f'<p class="shep-ft-why">Not ready: {r.e("; ".join(parts))}.</p>'


def _readiness(ws: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    from src.api.services.server_lists import task_readiness

    return task_readiness(ws.get("name"))


def _tasks_update(ws: Mapping[str, Any], chosen: Any) -> Tuple[Any, str]:
    readiness = _readiness(ws)
    kept = [task for task in TASKS if task in (chosen or ())]
    return gr.update(choices=task_choices(readiness), value=kept), tasks_note_html(readiness)


def identity_html(ws: Mapping[str, Any]) -> str:
    """The chosen dataset's identity (the one place its fingerprint is
    shown), or why there is none."""
    row = ws.get("row")
    if not row:
        why = ws.get("problem") or ("No dataset selected: choose one from the server's list."
                                    if ws.get("choices") else "The server lists no workspace.")
        return f'<p class="shep-muted">{r.e(why)}</p>'
    return (f'<div class="shep-ft-ds">'
            f'{r.dataset_identity_html(row, name=False, digits=ws.get("digits") or r.SHORT_HEX)}'
            "</div>")


def bases_note_html(parents: Mapping[str, Any], problem: Optional[str]) -> str:
    """The base list's line: why it is empty, or how many bases it holds."""
    if problem:
        return f'<p class="shep-muted">{r.e(problem)}</p>'
    if not parents:
        return ('<p class="shep-muted">No base checkpoint in this dataset\'s runs: train one on '
                'Research › Training.</p>')
    runs = len(tc.runs_of(parents))
    return (f'<p class="shep-muted">Server list: {len(parents)} base checkpoint'
            f'{"s" if len(parents) != 1 else ""} in {runs} run{"s" if runs != 1 else ""}, '
            'latest first.</p>')


def base_summary_html(item: Optional[Mapping[str, Any]]) -> str:
    """The chosen base's summary as key-value rows, saying what the two
    selects above do not: its run's directory (the Run select gives its
    name, architecture and time), file, configuration, what it holds, its
    Trainer validation figure, how its dataset stands to the one selected,
    and the model fingerprint Start sends (copyable). Escaped; empty when
    none is chosen."""
    if not item:
        return ""
    figure = tc.validation_words(item)
    trained = item.get("trained_on_this_dataset")
    if trained is True:
        on = f'<span class="shep-ok-t">{r.CHECK} the dataset selected</span>'
    elif trained is False:
        on = ('<span class="shep-amber-t">▲ another dataset than the one selected (the start '
              'refuses a parent of another knowledge graph)</span>')
    else:
        on = '<span class="shep-muted">not recorded</span>'
    kind = item.get("checkpoint_kind")
    rows = [
        ("Run", f"<code>{r.e(item.get('architecture'))}/{r.e(item.get('run'))}</code>"),
        ("File", f"<code>{r.e(item.get('filename'))}</code>"),
        ("Configuration", r.e(r.configuration_text(item))),
        ("Holds", r.e(tc.CHECKPOINT_KINDS.get(kind or "", "not recorded"))),
        ("Validation", r.e(figure.removeprefix("Validation ") if figure else "not recorded")),
        ("Trained on", on),
        ("Model fingerprint", tc.digest_html(item.get("sha256"), what="model fingerprint")),
    ]
    return f'<div class="shep-ft-base"><div class="shep-kv">{r.kv_cells(rows)}</div></div>'


# =============================================================================
# The job card
# =============================================================================
def status_mark_html(status: str, words: Optional[str] = None) -> str:
    """A child's status as colour, shape and word together (the WebUI skill
    §6): its kind from `training_console.PILL_KIND`, never green for a failed,
    stopped, cancelled or refused child."""
    kind = tc.PILL_KIND.get(status, "grey")
    shape, token = _MARKS.get(kind, ("○", "neutral"))
    mark = ('<span class="shep-ft-spin" aria-hidden="true"></span>' if shape is None else
            f'<span class="shep-ft-shape" aria-hidden="true">{shape}</span>')
    return (f'<span class="shep-ft-st shep-ft-st-{token}" data-status="{r.e(status)}">{mark}'
            f'<span>{r.e(words or str(status).capitalize())}</span></span>')


def _run_parts(path: Any) -> Tuple[Optional[str], Optional[str]]:
    """``(architecture, run)`` of a run directory, by its last two parts."""
    if not path:
        return None, None
    parts = Path(str(path)).parts
    return (parts[-2], parts[-1]) if len(parts) >= 2 else (None, None)


def parent_line_html(finetune: Mapping[str, Any]) -> str:
    """The card's parent: its run and file as a person reads them and its
    model fingerprint's 12 hex (it can differ from the base chosen now). No
    absolute path, for anyone."""
    parent = Path(str(finetune.get("parent") or ""))
    parts = parent.parts
    if len(parts) >= 4 and parts[-2] == "checkpoints":
        named = (f'{r.e(tc.run_words(parts[-4], parts[-3]))} '
                 f'<code>{r.e(parent.name)}</code>')
    else:
        named = f"<code>{r.e(parent.name)}</code>"
    return (f'<p class="shep-ft-from"><span class="shep-ft-k">From</span> {named} '
            f'{r.fingerprint_html("model fingerprint", finetune.get("parent_sha256"), copy=False)}'
            "</p>")


def child_products(child: Mapping[str, Any]) -> Dict[str, Any]:
    """An ended child's products, read once: its run's saved files' best by
    the Trainer validation figure each records (the list's metadata, never a
    file's name), ties shared, as ``{"best": [{"words", "validation"}],
    "problem"}``. It opens that run's files (`list_checkpoints(run=)`): an
    operator's read only."""
    from src.api.services.server_lists import list_checkpoints

    architecture, run = _run_parts(child.get("run_dir"))
    workspace = tc.workspace_of_run(child.get("run_dir"))
    if workspace is None or run is None:
        return {"best": [], "problem": "its workspace is not listed now"}
    items, problem = tc.checkpoint_items(
        list_checkpoints(workspace, architecture, task=child.get("task"), run=run))
    if problem:
        return {"best": [], "problem": problem}
    scored = [item for item in items.values()
              if isinstance((item.get("validation") or {}).get("value"), (int, float))]
    if not scored:
        return {"best": [], "problem": None}
    top = max(item["validation"]["value"] for item in scored)
    return {"best": [{"words": tc.checkpoint_words(item), "validation": item["validation"]}
                     for item in scored if item["validation"]["value"] == top],
            "problem": None}


def products_of(finetune: Optional[Mapping[str, Any]], held: Mapping[str, Any]
                ) -> Dict[str, Any]:
    """Every ended child's products by its run and status
    (`child_products`), each read only when this session has not read it for
    that ending (`held`)."""
    out: Dict[str, Any] = {}
    for child in (finetune or {}).get("children") or []:
        if child.get("status") not in ENDED or not child.get("run_dir"):
            continue
        key = f"{child['run_dir']}|{child['status']}"
        out[key] = held[key] if key in held else child_products(child)
    return out


def _products_cells(child: Mapping[str, Any], *, operator: bool,
                    products: Mapping[str, Any]) -> str:
    status = child.get("status")
    if status in ("queued", "running", "stopping"):
        return '<td colspan="2" class="shep-muted">after it ends</td>'
    if status in ("cancelled", "refused") or not child.get("run_dir"):
        return '<td colspan="2" class="shep-muted">no run</td>'
    if not operator:
        return (f'<td colspan="2" class="shep-muted">{tc.LOCK_ICON} not shown: reading the '
                'checkpoints is an operator control</td>')
    found = products.get(f"{child['run_dir']}|{status}") or {}
    if found.get("problem"):
        return f'<td colspan="2" class="shep-muted">not read: {r.e(found["problem"])}</td>'
    best = found.get("best") or []
    if not best:
        return ('<td colspan="2" class="shep-muted">no saved file records a validation '
                'figure</td>')
    names = "".join(f"<div>{r.e(entry['words'])}</div>" for entry in best)
    return f"<td>{names}</td><td>{r.validation_cell(best[0]['validation'])}</td>"


def child_row_html(child: Mapping[str, Any], *, operator: bool, products: Mapping[str, Any],
                   scrub: Any = str) -> str:
    """One child's row: Task | Status | Run | Best save | Validation
    (Trainer), escaped (a failure's first line through `scrub`, a read-only
    request's redaction). The run is its name part (`renderers.run_name`; the
    parent line and the Task column say the rest), its whole id in its title."""
    status = str(child.get("status") or "queued")
    task = TASK_NAMES.get(child.get("task"), str(child.get("task")))
    why = ""
    if status in ("failed", "refused") and child.get("error"):
        first = str(child["error"]).strip().split("\n", 1)[0]
        why = f'<div class="shep-sub">{r.e(r.short_reason(scrub(first)))}</div>'
    architecture, run = _run_parts(child.get("run_dir"))
    if run:
        whole = f"{architecture}/{run}"
        run_cell = (f'<code class="shep-ident" title="{r.e(whole)}">'
                    f'{r.e(r.run_name(run) or whole)}</code>')
    else:
        run_cell = ('<span class="shep-muted">not started yet</span>' if status == "queued" else
                    '<span class="shep-muted">never started</span>')
    return (f"<tr><td>{r.e(task)}</td><td>{status_mark_html(status)}{why}</td>"
            f"<td>{run_cell}</td>{_products_cells(child, operator=operator, products=products)}"
            "</tr>")


def job_card_html(status: Mapping[str, Any], *, operator: bool,
                  products: Optional[Mapping[str, Any]] = None, scrub: Any = str) -> str:
    """The Fine-tune job card from the job runner's status (`get_status()`):
    its parent and one row per child; or, with no Fine-tune status, why
    (none since the server started, or the next base or continued run cleared
    it), in words of its own."""
    finetune = status.get("finetune")
    if not finetune:
        if (status.get("status") or "idle") == "idle":
            text = ("No Fine-tune job since the server started. Choose a base and its tasks, "
                    "then Start.")
        else:
            text = ("No Fine-tune job to show: the latest training job is a base or continued "
                    "run (Research › Training shows it). The next Fine-tune start shows here.")
        return f'<div class="shep-ft-card"><p class="shep-ft-empty">{r.e(text)}</p></div>'
    rows = "".join(child_row_html(child, operator=operator, products=products or {},
                                  scrub=scrub)
                   for child in finetune.get("children") or [])
    head = ("<thead><tr><th>Task</th><th>Status</th><th>Run</th>"
            f'<th>Best save{help_button("finetune-best-save", "Best save")}</th>'
            f'<th>Validation (Trainer){help_button("validation-trainer", "Validation (Trainer)")}'
            "</th></tr></thead>")
    link = ('<button type="button" class="shep-link" data-act="training">Charts on Research › '
            'Training ›</button>')
    return (f'<div class="shep-ft-card">{parent_line_html(finetune)}'
            f'<table class="shep-ft-table">{head}<tbody>{rows}</tbody></table>'
            f'<p class="shep-ft-links">{link}</p></div>')


# =============================================================================
# Start, Stop and the reason line
# =============================================================================
def _why(icon: str, text: str) -> str:
    return f'<span class="shep-ft-reason">{icon}<span>{r.e(text)}</span></span>'


def running_child(status: Mapping[str, Any]) -> Optional[str]:
    """**The one job the card's Stop reaches**: the id of the training job
    running now when it is a child of the Fine-tune start the status shows,
    else None. A base or continued run started on Research › Training is
    never asked about or stopped from here (the run bar stops it)."""
    job = tc.running_job(status)
    children = (status.get("finetune") or {}).get("children") or []
    if job is not None and any(child.get("status") == "running" and child.get("job_id") == job
                               for child in children):
        return job
    return None


def actions_view(status: Mapping[str, Any], *, access: Any,
                 gpu: Optional[Mapping[str, Any]], confirming: bool) -> Dict[str, Any]:
    """Start and Stop (interactive or not) and the one reason line, beside
    Start, as one decision for the tick and after every action, in the states
    the Training run bar uses (`training_console.run_bar_view`): D8, a training
    job running or stopping, the GPU slot held elsewhere. Stop stops a running
    Fine-tune child (and so cancels the queued ones), never another page's
    job. An enabled Start has no reason beside it: Stop, off then, is in the
    job panel under the card, whose statuses say that nothing runs."""
    s = status.get("status") or "idle"
    child_running = running_child(status) is not None
    view = {"start": False, "stop": False}
    if not access.allowed:
        short = D8_SHORT.get(access.condition, "request not readable")
        view["reason"] = _why(tc.LOCK_ICON, f"Operator controls only: {short}")
    elif s == "stopping":
        view["reason"] = _why(tc.NO_ICON, "Stopping: the job is ending; Start waits for it.")
    elif child_running:
        view["stop"] = not confirming
        view["reason"] = _why(tc.NO_ICON, "Start disabled: this Fine-tune job is running.")
    elif s == "running":
        view["reason"] = _why(tc.NO_ICON, "Start disabled: a training job is running "
                                          "(Research › Training).")
    elif gpu:
        view["reason"] = _why(tc.NO_ICON, f"Start disabled — {gpu['text']}.")
    else:
        view["start"] = True
        view["reason"] = ""
    return view


def _card_view(mode: Optional[Mapping[str, Any]], cache: Optional[Mapping[str, Any]],
               request: Any) -> Tuple[Any, ...]:
    """The card, Start, Stop, the reason line, the stop question's row and
    the session's cache, read now for this request. The products are read
    for an operator only (they open checkpoints)."""
    cache = dict(cache or {})
    access = operator_access(request)
    status = training_manager.get_status()
    active = status.get("gpu", {}).get("job") if isinstance(status.get("gpu"), Mapping) else None
    busy = (status.get("status") or "idle") in ("running", "stopping")
    gpu = None if busy else tc.gpu_view(active, cache, access.allowed)
    asking = tc.confirming(status, dict(mode or {}))
    view = actions_view(status, access=access, gpu=gpu, confirming=asking)
    products: Dict[str, Any] = {}
    if access.allowed:
        products = products_of(status.get("finetune"), cache.get("products") or {})
        cache["products"] = products
    scrub = str if access.allowed else tc.read_only_scrub(workspaces_root()[0])
    card = job_card_html(status, operator=access.allowed, products=products, scrub=scrub)
    return (card, gr.update(interactive=view["start"]), gr.update(interactive=view["stop"]),
            f'<div class="shep-ft-reasons">{view["reason"]}</div>', gr.update(visible=asking),
            cache)


def _refresh_card(mode, cache, request: gr.Request):
    """After every action: the card and the actions as they are now."""
    return _card_view(mode, cache, request)


def _poll(location, mode, cache, request: gr.Request):
    """The shell's slow timer (``queue=False``), while Fine-tune is shown: the
    card and the actions; nothing while another page is shown."""
    if location != PAGE:
        return (gr.update(),) * 6
    try:
        return _card_view(mode, cache, request)
    except Exception:  # noqa: BLE001 - a tick that raises would stop the page's updates
        logger.exception("The Fine-tune job could not be read")
        return (gr.update(),) * 6


# =============================================================================
# The handlers
# =============================================================================
def _open(ws, preferred, chosen):
    """The page shown: the server's list read again, the dataset chosen: this
    session's own, else the Training run bar's (``ctx.training_workspace``;
    read, never written), else the first trainable; the tasks' readiness."""
    listed = read_workspaces()
    names = [row["name"] for row in listed["rows"]]
    choice = next((name for name in ((ws or {}).get("name"), preferred) if name in names), None)
    if choice is None:
        trainable = [row["name"] for row in listed["rows"] if row.get("trainable")]
        choice = trainable[0] if trainable else (names[0] if names else None)
    state = tc.workspace_state(choice, listed)
    tasks, note = _tasks_update(state, chosen)
    return state, tc.dataset_choices_update(state), identity_html(state), tasks, note


def _on_dataset(name, chosen):
    """A dataset chosen here: its record read now, the tasks' readiness for
    it; the base list follows (`_on_finetune_bases`)."""
    state = tc.workspace_state(name)
    tasks, note = _tasks_update(state, chosen)
    return state, identity_html(state), tasks, note


def _on_finetune_bases(request: gr.Request, ws, current):
    """The parent list: the workspace's listed bases (`list_checkpoints(
    learning_stage="base")`). It opens every checkpoint, so D8 is checked
    first, on this event's own request; refused, the list is emptied, nothing
    is opened, and the line says why.

    Returns the parents (state), the run and checkpoint selects and their line.
    """
    access = operator_access(request)
    empty = gr.update(choices=[], value=None)
    if not access.allowed:
        return {}, empty, empty, f'<p class="shep-muted">{r.e(access.refusal)}</p>'
    name = (ws or {}).get("name")
    if not name:
        return {}, empty, empty, bases_note_html({}, "Select a dataset first.")
    from src.api.services.server_lists import list_checkpoints

    parents, problem = tc.checkpoint_items(list_checkpoints(name, learning_stage="base"))
    return (parents, *tc.parents_update(parents, current), bases_note_html(parents, problem))


def _on_base_chosen(key, parents):
    """A checkpoint chosen: the base's summary."""
    return base_summary_html((parents or {}).get(key) if key else None)


def _on_finetune_start(request: gr.Request, ws, parents, key, tasks, epochs, seed, device,
                       run_name):
    """Start: an operator control (D8), checked first on this event's own
    request; then the Fine-tune start (`_start`).

    Returns the notice: the refusal, inline and escaped, or nothing once
    started (the card shows the job).
    """
    access = operator_access(request)
    if not access.allowed:
        return tc.notice_html(access.refusal)
    return tc.notice_html(_start(ws, parents, key, tasks, epochs, seed, device, run_name))


def _on_finetune_stop_request(request: gr.Request, mode):
    """Stop, on the card: D8 checked first, on this event's own request; then
    the Training page's own question (`training_console.ask_stop`, which
    names the queued children), for a running child only (`running_child`)."""
    return tc.ask_stop(operator_access(request), mode, reaches=running_child,
                       what="Fine-tune child")


def _on_finetune_stop(request: gr.Request, mode):
    """The confirmed Stop, on the card: an operator control (D8), checked on
    this event's own request; then the Training page's own stop
    (`training_console.confirm_stop`), which reaches a running child only
    (`running_child`): it stops it and so cancels the queued ones, and never
    stops another page's job."""
    return tc.confirm_stop(operator_access(request), mode, reaches=running_child,
                           what="Fine-tune child")


def _start(ws, parents, key, tasks, epochs, seed, device, run_name) -> str:
    """Start's work after D8, as a plain sentence ("" once started): the
    chosen base and tasks (in `TASKS` order), the settings that differ from
    the recipe's, the workspace and the parent resolved by the server from
    its lists again, and the runner's Fine-tune start with the parent's listed
    digest and the dataset fingerprint this session showed."""
    row = (ws or {}).get("row") or {}
    fingerprint = tc.workspace_fingerprint(ws)
    if not row or not fingerprint:
        return "Not started: select a dataset from the server's list first."
    if not row.get("trainable"):
        return f"Not started: {row.get('name')} is not trainable: {r.not_trainable_short(row)}."
    item = (parents or {}).get(key) if key else None
    if not item:
        return "Not started: choose a base checkpoint from the list."
    if item.get("workspace") != row.get("name"):
        return (f"Not started: the list shows {item.get('workspace')}’s bases, not "
                f"{row.get('name')}’s; list them again (Refresh).")
    chosen = tuple(task for task in TASKS if task in (tasks or ()))
    if not chosen:
        return "Not started: choose at least one task."
    settings, problems = tc.recipe_differences(
        {"num_epochs": epochs, "seed": seed}, {name: FINETUNE[name] for name in FINETUNE_FIELDS})
    if problems:
        return "Not started: " + " ".join(problems)
    from src.api.services.server_lists import (
        ServerListRefused,
        resolve_checkpoint,
        resolve_workspace,
    )

    try:
        directory = resolve_workspace(row["name"])
        path = resolve_checkpoint(row["name"], item["architecture"], item["run"],
                                  item["filename"])
    except ServerListRefused as exc:
        return (f"Not started: {redact(exc, workspaces_root()[0])}; list the bases again "
                "(Refresh).")
    result = training_manager.start_finetune(
        FinetuneInputs(data_dir=str(directory), finetune_from=str(path), tasks=chosen,
                       expected_parent_sha256=item.get("sha256"), device=device or "auto",
                       run_name=(run_name or "").strip(), settings=settings),
        expected_dataset_fingerprint=fingerprint)
    if result.get("success"):
        return ""
    return tc.not_started(result.get("error"))


# =============================================================================
# Help
# =============================================================================
#: The fields' help, behind the shared ⓘ: ``{elem_id: (title, text)}``, each
#: title its field's label.
FIELD_HELP = {
    "ft_dataset": (
        "Dataset",
        "The workspace the children train on, from the server's list as on Research › "
        "Datasets. It starts from the Training page's dataset and never changes it.\n\n"
        "Start sends the dataset fingerprint shown, and the server refuses the start if the "
        "workspace changed since.",
    ),
    "ft_run": (
        "Run",
        "The runs holding a base checkpoint, latest first: name, architecture and start time.\n\n"
        "Only bases are offered: a task model is never a parent (children never chain), and a "
        "file without a model spec is never trained from here.",
    ),
    "ft_checkpoint": (
        "Checkpoint",
        "The base to fine-tune, from the chosen run, newest first. Its summary is under the "
        "list.\n\n"
        "• A weights-only or early-stopped base can be fine-tuned.\n"
        "• Start sends its model fingerprint; the server reads the file against it.\n"
        "• Listing opens each checkpoint, so it is an operator control.",
    ),
    "ft_tasks": (
        "Tasks",
        "The task models to train from the base: one child run each, in this order.\n\n"
        "• Ready: the dataset can be trained, and for Gene and Patients-like-me the gene "
        "task's functional fixture was derived from this build of it.\n"
        "• Before anything starts, the server checks the parent, its graph and each fixture "
        "again; the training process checks the fixture's files.",
    ),
    **{f"ft_{name}": tc.setting_help(name, FINETUNE, FINETUNE_SOURCES)
       for name in FINETUNE_FIELDS},
    "ft_device": (
        "Device",
        "auto: the GPU (CUDA). With no CUDA device visible, Start is refused: training never "
        "falls back to the CPU on its own.\n\n"
        "cpu: for development, recorded as asked.",
    ),
    "ft_run_name": (
        "Run name",
        "Optional. Each child is saved in a new folder of its own, named by the start time, "
        "this name and its task:\n"
        "<workspace>/runs/<architecture>/<date-time>_<name>-<task>_<id>/",
    ),
}

#: The ⓘ inside the card.
TOPIC_HELP = {
    "finetune-best-save": (
        "Best save",
        "The saved file of a child's run with the highest Trainer validation figure (pooled "
        "MRR) its record holds; files that tie share it.\n\n"
        "It is the Trainer's selection figure on the val samples, not a test result: test the "
        "model on Research › Test. Shown to the operator once the child has ended (reading it "
        "opens the run's checkpoints).",
    ),
}


# =============================================================================
# The page
# =============================================================================
def create_finetune_page(ctx: Any) -> None:
    """Build Research › Prepare › Fine-tune (the page contract:
    `src/webui/shell.py`). `ctx` gives the help registry, the shared timers,
    the cross-links, the chips' read after every gated act and the Training
    run bar's workspace (read only)."""
    ctx.help.fields(FIELD_HELP)
    ctx.help.topics(TOPIC_HELP)

    ws = gr.State(tc.empty_workspace())
    parents = gr.State({})
    mode = gr.State({})
    cache = gr.State({})

    with gr.Row(elem_id="shep-ft-row", elem_classes=["shep-page", "shep-ft"], equal_height=False):
        # ------------------------------------------------------------- the form
        with gr.Column(elem_id="shep-ft-form", elem_classes=["shep-panel"], scale=10,
                       min_width=360):
            plain_html('<div class="shep-panel-hd"><h2>Fine-tune a base</h2>'
                       '<span class="shep-panel-right">one child run per task</span></div>',
                       elem_id="ft_form_hd")
            with gr.Column(elem_id="ft_form_bd", elem_classes=["shep-ft-bd"]):
                dataset = gr.Dropdown(label="Dataset", choices=[], value=None,
                                      elem_id="ft_dataset")
                identity = plain_html(identity_html({}), elem_id="ft_identity")
                with gr.Row(elem_classes=["shep-ft-parent-row"]):
                    run_select = gr.Dropdown(label="Run", choices=[], value=None,
                                             elem_id="ft_run")
                    refresh = gr.Button("Refresh", size="sm", scale=0, elem_id="ft_refresh")
                checkpoint = gr.Dropdown(label="Checkpoint", choices=[], value=None,
                                         elem_id="ft_checkpoint")
                bases_note = plain_html("", elem_id="ft_bases_note")
                base = plain_html("", elem_id="ft_base")
                tasks = gr.CheckboxGroup(label="Tasks", choices=task_choices({}), value=[],
                                         elem_id="ft_tasks")
                tasks_note = plain_html("", elem_id="ft_tasks_note")
                with gr.Row(elem_classes=["shep-ft-settings"]):
                    epochs = tc.setting_field("num_epochs", FINETUNE["num_epochs"],
                                              elem_id="ft_num_epochs")
                    seed = tc.setting_field("seed", FINETUNE["seed"], elem_id="ft_seed")
                with gr.Row(elem_classes=["shep-ft-settings"]):
                    device = gr.Radio(label="Device", choices=["auto", "cuda", "cpu"],
                                      value="auto", elem_id="ft_device")
                    run_name = gr.Textbox(label="Run name", value="", max_lines=1,
                                          elem_id="ft_run_name")
            with gr.Column(elem_id="ft_start_ft", elem_classes=["shep-panel-ft"]):
                with gr.Row(elem_classes=["shep-ft-start-row"]):
                    start = gr.Button("▸ Start", variant="primary", interactive=False, scale=0,
                                      elem_id="ft_start")
                    reason = plain_html("", elem_id="ft_reason")
                notice = plain_html("", elem_id="ft_notice")
        # ------------------------------------------------------------- the job
        with gr.Column(elem_id="shep-ft-job", elem_classes=["shep-panel"], scale=12,
                       min_width=360):
            plain_html('<div class="shep-panel-hd"><h2>Fine-tune job</h2>'
                       '<span class="shep-panel-right">the latest start</span></div>',
                       elem_id="ft_job_hd")
            with gr.Column(elem_id="ft_job_bd", elem_classes=["shep-ft-bd"]):
                card = plain_html(job_card_html({"status": "idle"}, operator=False),
                                  elem_id="ft_card", js_on_load=tc.TRIGGER_JS)
                with gr.Row(elem_classes=["shep-ft-stop-row"]):
                    stop = gr.Button("■ Stop", variant="secondary", interactive=False, scale=0,
                                     elem_id="ft_stop")
                with gr.Row(visible=False, elem_id="ft_confirm_row") as confirm_row:
                    stop_text = plain_html("", elem_id="ft_stop_text")
                    stop_confirm = gr.Button("■ Stop training", variant="stop", scale=0,
                                             elem_id="ft_stop_confirm")
                    stop_keep = gr.Button("Keep training", scale=0, elem_id="ft_stop_keep")

    # =========================================================================
    # Wiring
    # =========================================================================
    card_step = {"fn": _refresh_card, "inputs": [mode, cache],
                 "outputs": [card, start, stop, reason, confirm_row, cache],
                 "show_progress": "hidden"}
    bases_step = {"fn": _on_finetune_bases, "inputs": [ws, checkpoint],
                  "outputs": [parents, run_select, checkpoint, bases_note],
                  "show_progress": "hidden"}
    base_step = {"fn": _on_base_chosen, "inputs": [checkpoint, parents], "outputs": [base],
                 "show_progress": "hidden"}

    # The page shown: the list and the tasks, the bases (gated), the card.
    ctx.on_page_open(PAGE, {
        "fn": _open, "inputs": [ws, ctx.training_workspace, tasks],
        "outputs": [ws, dataset, identity, tasks, tasks_note],
        "show_progress": "hidden"}).then(**bases_step).then(**base_step).then(
        **card_step).then(**ctx.chip_refresh)
    # The shell's slow timer, a read-only poll without the queue.
    ctx.timer_slow.tick(_poll, inputs=[ctx.location, mode, cache],
                        outputs=[card, start, stop, reason, confirm_row, cache],
                        queue=False, show_progress="hidden")

    dataset.input(_on_dataset, inputs=[dataset, tasks],
                  outputs=[ws, identity, tasks, tasks_note],
                  show_progress="hidden").then(**bases_step).then(**base_step).then(
        **ctx.chip_refresh)
    refresh.click(**bases_step).then(**base_step).then(**ctx.chip_refresh)
    run_select.change(tc.on_run_chosen, inputs=[run_select, parents, checkpoint],
                      outputs=[checkpoint], show_progress="hidden")
    checkpoint.change(**base_step)

    start.click(_on_finetune_start,
                inputs=[ws, parents, checkpoint, tasks, epochs, seed, device, run_name],
                outputs=[notice], show_progress="hidden").then(
        tc.close_stop_question, inputs=[mode], outputs=[mode], show_progress="hidden").then(
        **card_step).then(**ctx.chip_refresh)
    # Stop asks first: the Training page's stop logic, D8 on each request.
    stop.click(_on_finetune_stop_request, inputs=[mode], outputs=[stop_text, mode],
               show_progress="hidden").then(**card_step).then(**ctx.chip_refresh)
    stop_confirm.click(_on_finetune_stop, inputs=[mode], outputs=[notice, mode],
                       show_progress="hidden").then(**card_step).then(**ctx.chip_refresh)
    stop_keep.click(tc.close_stop_question, inputs=[mode], outputs=[mode],
                    show_progress="hidden").then(**card_step)

    def _on_card_click(evt: gr.EventData):
        """The card's "Charts on Research › Training ›"."""
        return ctx.go("training") if tc.event_act(evt) == "training" else ctx.no_nav()

    card.click(_on_card_click, inputs=None, outputs=ctx.nav_outputs, show_progress="hidden")


#: This page's element id (`shell.page_elem_id`), the scope of `PAGE_CSS`.
_PAGE = "#" + page_elem_id(PAGE)

#: This page's style sheet (listed in `src/webui/app.py`'s ``PAGE_CSS``).
#: Nothing under 14 px; status colours only through the ``--shep-status-*``
#: tokens.
PAGE_CSS = f"""
{_PAGE} .shep-ft-bd {{ padding: 8px 16px 12px !important; gap: 10px !important; }}
{_PAGE} .shep-ft-parent-row {{ align-items: flex-end !important; flex-wrap: nowrap !important; }}
{_PAGE} #ft_run {{ flex: 1 1 auto !important; min-width: 0 !important; }}
{_PAGE} #ft_refresh {{ flex: none !important; width: auto !important; min-width: max-content !important; }}
{_PAGE} .shep-ft-settings {{ flex-wrap: wrap !important; gap: 8px 16px !important; }}
{_PAGE} .shep-ft-ds {{ font-size: 15px; display: flex; flex-wrap: wrap; gap: 4px 6px; align-items: center; }}
{_PAGE} .shep-ft-base {{ border: 1px solid var(--shep-border); border-radius: 8px; background: var(--shep-panel); padding: 8px 12px; }}
{_PAGE} .shep-ft-base .shep-kv {{ font-size: 14px; gap: 4px 14px; }}
{_PAGE} .shep-ft-base .shep-v {{ overflow-wrap: anywhere; min-width: 0; }}
{_PAGE} #ft_base:not(:has(.shep-ft-base)), {_PAGE} #ft_tasks_note:not(:has(p)) {{ display: none !important; }}
{_PAGE} .shep-ft-why {{ margin: 0; font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-ft-start-row {{ align-items: center !important; flex-wrap: wrap !important; gap: 8px 14px !important; }}
{_PAGE} .shep-ft-start-row > button, {_PAGE} .shep-ft-stop-row > button {{ flex: none !important; width: auto !important; min-width: max-content !important; padding: 0 14px !important; height: 40px; }}
{_PAGE} #ft_reason {{ flex: 1 1 240px !important; }}
{_PAGE} .shep-ft-reason {{ display: inline-flex; align-items: flex-start; gap: 6px; font-size: 15px; color: var(--shep-text-2); }}
{_PAGE} .shep-ft-reason > svg {{ flex: none; margin-top: 3px; }}
{_PAGE} #ft_notice {{ font-size: 15px; color: var(--shep-amber-800); overflow-wrap: anywhere; }}
{_PAGE} #ft_notice:has(.html-container:empty) {{ display: none !important; }}
{_PAGE} .shep-ft-card {{ display: flex; flex-direction: column; gap: 10px; font-size: 15px; }}
{_PAGE} .shep-ft-empty {{ margin: 0; color: var(--shep-text-2); }}
{_PAGE} .shep-ft-from {{ margin: 0; overflow-wrap: anywhere; }}
{_PAGE} .shep-ft-k {{ font-size: 14px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: var(--shep-text-2); margin-right: 8px; }}
{_PAGE} .shep-ft-table {{ width: 100%; border-collapse: collapse; font-size: 15px; }}
{_PAGE} .shep-ft-table th {{ text-align: left; font-size: 14px; font-weight: 600; color: var(--shep-text-2); padding: 6px 8px; border-bottom: 1px solid var(--shep-border); white-space: nowrap; }}
{_PAGE} .shep-ft-table td {{ padding: 8px; border-bottom: 1px solid var(--shep-border); vertical-align: top; min-width: 0; }}
{_PAGE} .shep-ft-table code.shep-ident {{ max-width: 40ch; }}
{_PAGE} .shep-ft-st {{ display: inline-flex; align-items: center; gap: 6px; font-weight: 600; white-space: nowrap; }}
{_PAGE} .shep-ft-st-ok {{ color: var(--shep-status-ok); }}
{_PAGE} .shep-ft-st-neutral {{ color: var(--shep-status-neutral); }}
{_PAGE} .shep-ft-st-error {{ color: var(--shep-status-error); }}
{_PAGE} .shep-ft-st-info {{ color: var(--shep-status-info); }}
{_PAGE} .shep-ft-shape {{ font-size: 15px; line-height: 1; }}
{_PAGE} .shep-ft-spin {{ width: 12px; height: 12px; border: 2px solid currentColor; border-right-color: transparent; border-radius: 50%; animation: shep-ft-spin 0.9s linear infinite; }}
@keyframes shep-ft-spin {{ to {{ transform: rotate(360deg); }} }}
@media (prefers-reduced-motion: reduce) {{ {_PAGE} .shep-ft-spin {{ animation: none; }} }}
{_PAGE} .shep-ft-links {{ margin: 0; }}
{_PAGE} .shep-link {{ border: 0; background: none; padding: 0; color: var(--shep-accent-text); font-size: 15px; cursor: pointer; font-weight: 500; }}
{_PAGE} .shep-link:hover {{ text-decoration: underline; }}
{_PAGE} #ft_confirm_row {{ align-items: center !important; flex-wrap: wrap !important; gap: 8px 14px !important; }}
{_PAGE} #ft_stop_text {{ flex: 1 1 320px !important; }}
{_PAGE} .shep-rb-confirm-q {{ background: var(--shep-red-50); border: 1px solid var(--shep-red-200); border-radius: 8px; padding: 8px 12px; font-size: 15px; }}
{_PAGE} .shep-html svg {{ display: inline-block; vertical-align: -2px; }}
@media (min-width: 1400px) and (min-height: 760px) {{
  {_PAGE} > .column {{ height: 100%; min-height: 0 !important; }}
  {_PAGE} #ft_form_bd, {_PAGE} #ft_job_bd {{ flex: 1 1 auto !important; min-height: 0; overflow: auto !important; flex-wrap: nowrap !important; }}
  {_PAGE} #ft_form_bd > *, {_PAGE} #ft_job_bd > * {{ flex-shrink: 0 !important; }}
}}
"""


__all__ = ["PAGE_CSS", "create_finetune_page", "job_card_html", "task_choices"]
