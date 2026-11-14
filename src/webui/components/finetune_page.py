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
  validation figure, the dataset it trained on, its split seed and the model
  fingerprint Start sends: **its split seed and its model are the base's
  facts, never fields** (`BASE_FACTS`): a finetune trains on its base's split
  and keeps its base's graph side.
- **Tasks**: a checkbox group in `TASKS` order whose option text says whether
  each is ready on this workspace (`server_lists.task_readiness`, small JSON
  files only, no D8), and why not in the line under it.
- **The settings** (ops-wiring round, decisions S1, S2, F1): what every
  chosen task's run takes, the intersection of `run_plan.new_run_settings`
  over the tasks (`offered_settings`), each the one field of its setting
  (`training_console.setting_field`). **Epochs, Seed and Graph backbone**
  first (`COMMON_SHOWN`; Graph backbone is the freeze, Trained or Frozen, the
  field spec's values, and with Frozen chosen a visible note under the fields
  says what then trains, `freeze_note_html`); the rest of the common settings
  under **More settings** (`MORE_FIELDS`), a setting a rule across settings
  reads only under its guard shown only while the run would read it (Warmup
  steps with the cosine scheduler: `training_console.guard_updates`). **A
  task's own settings** (`TASK_OWN`: the recipe gives each task its own
  reference, its learning rate, sample sizes, loss weights and candidate
  size) are offered under **"<Task> settings"** only when exactly one task is
  chosen, each starting from that task's reference (`task_starts`); with
  several, a visible line says each child takes its own task's reference
  (`several_note_html`). **Neighbours per hop** is offered for any task set
  (`ANY_TASKS`), in a row of its own under the common settings: its length
  is the base's encoder's to decide, so a base whose encoder another hop
  count takes than a task's reference can be fine-tuned from here;
  untouched, each child takes its task's reference. When the start's own
  check (`spec.hops_problem`, on the base's listed spec, of the list each
  child samples, `run_plan.new_run_hops`) says the reference, or the list
  typed, cannot run on the chosen base, that sentence is beside the field
  (`hops_note_html`), in view without opening anything. **Start sends what
  the operator touched** (decision S2: `sent_settings`, the form's record of
  its fields' input, `training_console.touched_record`, held in the
  browser; a change of tasks drops a task's own settings from it there,
  `record_drops`), whatever its value, of the settings the chosen tasks
  take; untouched, the run fills it (the recipe's or this build's default)
  and records it so. Under the settings, **the base form's one line on what
  the operator set** ("Set by you: Epochs 5 (reference 100)",
  `set_by_html`, `training_console.set_by_line_html`) and **Reset to
  reference** (`training_console.reset_outputs`), which empties the record
  and puts every field where it starts for the chosen tasks.
- Device and Run name; **Start**, its reason line, and any refusal beside it
  (the shared mark of a refusal, escaped): the page's own check of a value,
  or the job runner's refusal in its words, a setting it names first by the
  field's label (`training_console.labelled_problem`).

**Start** (D8 first): the workspace and the parent are resolved by the server
from its lists again (`resolve_workspace`, `resolve_checkpoint`; never a
client path), then the job runner's one Fine-tune start
(`TrainingManager.start_finetune`) with the parent's listed digest and the
dataset fingerprint shown. Its checks before anything starts (the parent a
base with weights, its graph, each task's fixture, the settings, the freeze
by `recipe.frozen_parts_problem`, and each child's hop list and a stated split
seed against its base among them) are the runner's alone (decision 3); this
page refuses nothing of its own but a value its field spec refuses, and its
hop note asks the start's own check.

**The job card** (right panel): `get_status()["finetune"]`, its parent, the
start's graph backbone (`freeze_line_html`, the runner's own record of the
settings it started with), then one row per child
(Task | Status | Run | Best save | Validation (Trainer)). Each status is a mark
of colour, shape and word (`training_console.job_status_html`, the shared
`renderers.status_mark`). A child's products, its best saves by the selection
its saved files record (`renderers.best_saves`: the recorded metric in its
recorded direction, named under the file, `renderers.selection_words`) and
that figure (the Trainer's, never a test result), are read for an operator
and only once the child has ended
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
from src.config.recipe import (FINETUNE, FINETUNE_SOURCES, FINETUNE_TASKS, FROZEN_PARTS,
                               reference_settings)
from src.config.training_fields import by_name, not_read_by_rules
from src.config.training_fields import names as field_names
from src.evaluation.caveats import TASK_NAMES, task_list_text
from src.models.bundle.record import listed_spec
from src.models.bundle.spec import TASKS, hops_problem
from src.utils.run_plan import new_run_hops, new_run_settings, new_run_values
from src.webui.components import model_list as ml
from src.webui.components import renderers as r
from src.webui.components import training_console as tc
from src.webui.components.chips import gated, operator_only
from src.webui.components.help import help_button
from src.webui.components.workspaces import read_workspaces, redact, workspaces_root
from src.webui.operator_access import operator_access
from src.webui.shell import FINETUNE as PAGE
from src.webui.shell import page_elem_id, plain_html

logger = logging.getLogger(__name__)

#: **The base's facts, never the page's fields** (ops-wiring round, B2;
#: decision S4): a finetune trains on its base's split and keeps its base's
#: graph side, so a split seed or a model spec stated here could only repeat
#: the base's or be refused by the start's check against it. The base's
#: summary says both (`base_summary_html`: its split seed and configuration).
BASE_FACTS = frozenset({"split_seed", "model_spec"})
#: **The common settings shown first** (decision F1 for the freeze): Epochs,
#: Seed and Graph backbone, one value for every child.
COMMON_SHOWN = ("num_epochs", "seed", "frozen_parts")
#: **A task's own settings**: those the recipe gives each task its own
#: reference for (`recipe.FINETUNE_TASKS`: its learning rate, hop list, sample
#: sizes, loss weights and candidate size), so one value would overwrite each
#: child's. Offered when exactly one task is chosen, but `ANY_TASKS`.
TASK_OWN = frozenset().union(*FINETUNE_TASKS.values())
#: **The task setting offered for any set of tasks**: the hop list, one number
#: per hop of its base's encoder (`spec.hops_problem`, the start's check of
#: every child against its base), so a base whose encoder takes another hop
#: count than a task's reference can be fine-tuned from this page, one task or
#: several. Untouched, each child takes its own task's reference.
ANY_TASKS = ("num_neighbors",)


def _ordered(settings: Any) -> Tuple[str, ...]:
    """`settings` in the forms' order: the base form's
    (`training_console.BASE_FIELDS`), then the field spec's."""
    order = tuple(tc.BASE_FIELDS) + tuple(name for name in field_names()
                                          if name not in tc.BASE_FIELDS)
    return tuple(name for name in order if name in settings)


def _takes(tasks: Any) -> frozenset:
    """What every run of `tasks` takes (the intersection of
    `run_plan.new_run_settings` over them, the job runner's own rule of a
    Fine-tune start), but the base's facts (`BASE_FACTS`)."""
    return frozenset.intersection(*(new_run_settings("finetune", task)
                                    for task in tasks)) - BASE_FACTS


#: **The rest of the common settings**, under "More settings": what every
#: task's run takes but a task's own settings and those shown first.
MORE_FIELDS = _ordered(_takes(TASKS) - TASK_OWN - set(COMMON_SHOWN))
#: **The task settings**, under "<Task> settings": the hop list first
#: (`ANY_TASKS`), then each task's own, in the forms' order.
TASK_FIELDS = ANY_TASKS + _ordered(
    (frozenset().union(*(_takes((task,)) for task in TASKS)) & TASK_OWN) - set(ANY_TASKS))
#: **Every field of the page, in the order its handlers take their values**
#: (Start's, the guards', the line's), each the one field of its setting
#: (`training_console.setting_field`).
FINETUNE_FIELDS = COMMON_SHOWN + MORE_FIELDS + TASK_FIELDS
#: **The task settings a change of tasks starts again** (`_on_tasks_input`):
#: a task's own, whose reference is the task's, so they leave the form's
#: record (`training_console.record_drops`, in the browser); the hop list set
#: stays, the operator's for every child of any task set (`ANY_TASKS`).
TASK_OWN_FIELDS = tuple(name for name in TASK_FIELDS if name not in ANY_TASKS)
#: What a Frozen graph backbone leaves to train, said under the fields while
#: it is chosen (the WebUI skill §7: a fact that changes what happens is
#: visible text, not only a bubble).
FROZEN_NOTE = ("Frozen graph backbone: only the patient encoder and the task head train; the LP "
               "term trains nothing.")
#: A child's states after which its run's saved files are final.
ENDED = ("completed", "failed", "stopped")
#: Start's and Stop's words, shown with no lock while the operator may use
#: them (`chips.gated`). Stop training, the question's answer, is reached only
#: through a Stop D8 allowed, so it shows none.
START_LABEL = "▸ Start"
STOP_LABEL = "■ Stop"


# =============================================================================
# The form's pieces
# =============================================================================
def task_choices(readiness: Mapping[str, Optional[str]]) -> List[Tuple[str, str]]:
    """The Tasks options, in `TASKS` order: the task's name and whether it is
    ready on the workspace ("Gene · ready", "Gene · not ready"; why not is the
    line under the group, `tasks_note_html`). Each at most
    `renderers.OPTION_CHARS` characters, one separator. A task the
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
        parts.append(f"{task_list_text(names)}: {why}")
    return f'<p class="shep-ft-why">Not ready: {r.e("; ".join(parts))}.</p>'


def _readiness(ws: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    from src.api.services.server_lists import task_readiness

    return task_readiness(ws.get("name"))


def _tasks_update(ws: Mapping[str, Any], chosen: Any) -> Tuple[Any, str]:
    readiness = _readiness(ws)
    kept = [task for task in TASKS if task in (chosen or ())]
    return gr.update(choices=task_choices(readiness), value=kept), tasks_note_html(readiness)


# ----------------------------------------------------------------- the settings
def chosen_tasks(tasks: Any) -> Tuple[str, ...]:
    """The chosen tasks in `TASKS` order (the children's order)."""
    return tuple(task for task in TASKS if task in (tasks or ()))


def offered_settings(tasks: Any) -> Tuple[str, ...]:
    """**The settings Start offers for the chosen `tasks`** (decision S1), in
    `FINETUNE_FIELDS` order: what every child's run takes (`_takes`: the
    intersection of `run_plan.new_run_settings` over them, as the job
    runner's start takes it), but the base's facts; with several tasks a
    task's own setting only where any set of tasks takes it (`ANY_TASKS`),
    since each child's reference is its own. None for no task."""
    chosen = chosen_tasks(tasks)
    if not chosen:
        return ()
    takes = _takes(chosen)
    if len(chosen) > 1:
        takes -= TASK_OWN - set(ANY_TASKS)
    return tuple(name for name in FINETUNE_FIELDS if name in takes)


def built_values() -> Dict[str, Any]:
    """Each field's value when the page is built, in `FINETUNE_FIELDS` order:
    the one fill of a new run (`run_plan.new_run_values`: the reference, else
    this build's default) of the first task whose run takes it, which also
    decides its field's kind (`training_console.value_field`: early
    stopping's patience, None, is a text box reading "off"). The common
    settings' are every task's."""
    fills = [new_run_values("finetune", task, {}) for task in TASKS]
    return {name: next(fill[name] for fill in fills if name in fill) for name in FINETUNE_FIELDS}


def task_starts(tasks: Any) -> Dict[str, Any]:
    """**Where each task setting's field starts for the chosen `tasks`**, as
    its field holds it (`training_console.form_value`): with one task, each
    of its own settings at its reference (`run_plan.new_run_values`); with
    several, the hop list at the reference every chosen task shares, else
    empty (untouched, each child takes its own: `hops_note_html` says which).
    A field not named keeps its value."""
    chosen = chosen_tasks(tasks)
    fills = [new_run_values("finetune", task, {}) for task in chosen]
    if len(chosen) == 1:
        return {name: tc.form_value(fills[0][name]) for name in TASK_FIELDS if name in fills[0]}
    starts = {}
    for name in ANY_TASKS if chosen else ():
        values = [fill[name] for fill in fills]
        starts[name] = tc.form_value(values[0]) if all(v == values[0] for v in values) else ""
    return starts


def start_values(tasks: Any) -> List[Any]:
    """**Where every field starts for the chosen `tasks`**, in
    `FINETUNE_FIELDS` order, as its field holds it
    (`training_console.form_value`): the page's built value
    (`built_values`), each task setting's from the tasks (`task_starts`).
    Reset to reference writes them (`_on_finetune_reset`)."""
    starts = task_starts(tasks)
    built = built_values()
    return [starts[name] if name in starts else tc.form_value(built[name])
            for name in FINETUNE_FIELDS]


def sent_settings(tasks: Any, values: Mapping[str, Any], touched: Any
                  ) -> Tuple[Dict[str, Any], List[str]]:
    """**What Start sends of the form** (decisions S1, S2): ``(settings,
    problems)``, of the settings the chosen `tasks` take (`offered_settings`)
    those whose field the operator touched (`touched`, the form's record),
    whatever their value, each checked by its field spec, as the base form
    sends its own (`training_console.sent_settings`): an emptied task
    setting is refused ("Task loss weight must be set, not null."), never
    sent as a null nor said to take the task's reference (OPS-R2); untouched,
    the task's reference fills it. `values` are the fields', by name.
    Nothing for no task. Start (`_start`) and the line on what the operator
    set (`set_by_html`) read it alike."""
    chosen = chosen_tasks(tasks)
    if not chosen:
        return {}, []
    return tc.sent_settings({name: values[name] for name in offered_settings(chosen)},
                            tc.touched_keys(touched))


#: The value the run takes without a setting whose reference is each chosen
#: task's own and differs between them (the hop list for Disease and Gene).
EACH_OWN = "each task’s own reference"


def _without(name: str, chosen: Tuple[str, ...]) -> str:
    """What every child of `chosen` takes without the setting `name`, in the
    line's words (`training_console.without_words`): the one value the tasks'
    fills share (`run_plan.new_run_values`), a reference where the recipe
    gives each a source; else `EACH_OWN`."""
    fills = [new_run_values("finetune", task, {})[name] for task in chosen]
    if any(fill != fills[0] for fill in fills):
        return EACH_OWN
    return tc.without_words(name, fills[0], all(name in reference_settings("finetune", task)[1]
                                                for task in chosen))


def set_by_html(tasks: Any, values: Mapping[str, Any], touched: Any) -> str:
    """**The Fine-tune form's line on what the operator set** (decision S2;
    the base form's one line, `training_console.set_by_line_html`): each
    setting Start sends for the chosen `tasks` (`sent_settings`), in the
    form's order, by its label, with what the children take without it
    (`_without`: "reference 100", "default 0", or each task's own); with
    none, that each child takes its task's reference values (and this
    build's defaults where the recipe sets none); then what Start would
    refuse of a value. Escaped. Reset to reference, beside it, brings it
    back to nothing (`_on_finetune_reset`)."""
    chosen = chosen_tasks(tasks)
    settings, problems = sent_settings(chosen, values, touched)
    rows = [(r.SETTING_LABELS[name], r.setting_words(name, settings[name]),
             _without(name, chosen)) for name in FINETUNE_FIELDS if name in settings]
    defaults = any(name not in reference_settings("finetune", task)[1]
                   for task in chosen for name in offered_settings(chosen))
    nothing = ("each child takes its task’s reference values"
               + (" and this build’s defaults." if defaults else "."))
    return tc.set_by_line_html(rows, problems, nothing)


def hops_problems(tasks: Any, item: Optional[Mapping[str, Any]], touched: Any = (),
                  value: Any = None) -> List[str]:
    """**Why the hop lists the chosen tasks' children would sample cannot
    run on the base `item`** (a listed base, its encoder read from its listed
    spec, `record.listed_spec`), by the start's own check of each child
    against its base (`spec.hops_problem`) on the list each child samples
    (`run_plan.new_run_hops`, as the job runner asks it), given what Start
    sends of the hop field (`value`, sent when the operator touched it:
    `touched`; `training_console.sent_settings`); each sentence once, the
    field named by its label (`training_console.labelled_problem`). Empty
    when every one fits, no task or base is chosen, the base's spec is not
    read or the text typed is no value Start sends (the line on what the
    operator set says why)."""
    chosen = chosen_tasks(tasks)
    spec, _unread = listed_spec((item or {}).get("spec"))
    if spec is None or not chosen:
        return []
    settings, refused = tc.sent_settings({"num_neighbors": value}, tc.touched_keys(touched))
    if refused:
        return []
    out: List[str] = []
    for task in chosen:
        problem = hops_problem(new_run_hops("finetune", task, settings), spec.encoder)
        if problem and tc.labelled_problem(problem) not in out:
            out.append(tc.labelled_problem(problem))
    return out


#: The lead of the hop note when the reference does not fit the chosen base.
HOPS_REFERENCE_LEAD = "The reference does not fit this base"


def hops_note_html(tasks: Any, item: Optional[Mapping[str, Any]], touched: Any = (),
                   value: Any = None) -> str:
    """**The line beside Neighbours per hop**, escaped: when the hop list a
    child would sample cannot run on the chosen base (`hops_problems`), that
    sentence beside the ▲ of a condition to watch, led by
    `HOPS_REFERENCE_LEAD` while it is a task's reference; else, with several
    tasks whose references differ and the field untouched, each child's
    ("Untouched, each child takes its task's reference: Disease 15, 10, 5;
    Gene all neighbours, 10, 5."; once touched, Reset to reference makes it
    so again); else empty."""
    problems = hops_problems(tasks, item, touched, value)
    touched_hops = "num_neighbors" in tc.touched_keys(touched)
    if problems:
        words = " ".join(problems)
        if not touched_hops:
            words = f"{HOPS_REFERENCE_LEAD}: {words}"
        return f'<p class="shep-ft-hops">{r.status_mark("warn", words)}</p>'
    chosen = chosen_tasks(tasks)
    references = {task: new_run_hops("finetune", task, {}) for task in chosen}
    if touched_hops or len({tuple(hops) for hops in references.values()}) < 2:
        return ""
    each = "; ".join(f"{TASK_NAMES[task]} {r.setting_words('num_neighbors', hops)}"
                     for task, hops in references.items())
    text = f"Untouched, each child takes its task’s reference: {each}."
    return f'<p class="shep-ft-why">{r.e(text)}</p>'


def hops_warning_html(tasks: Any, item: Optional[Mapping[str, Any]], touched: Any = (),
                      value: Any = None) -> str:
    """**The hop note's pointer beside Start** (review of B2: in view at the
    deployment window, where the field and its sentence may be under the
    form's fold): while the hop list a child would sample cannot run on the
    chosen base (`hops_problems`), the ▲ of a condition to watch and where
    the field is, the sentence itself said once, beside the field
    (`hops_note_html`); else empty. Start stays enabled: the start's own
    check refuses it, in its words."""
    if not hops_problems(tasks, item, touched, value):
        return ""
    words = (f"{r.SETTING_LABELS['num_neighbors']} does not fit this base: see the field under "
             f"{r.SETTING_LABELS[COMMON_SHOWN[0]]}.")
    return f'<span class="shep-ft-hops-warn">{r.status_mark("warn", words)}</span>'


def several_note_html(tasks: Any) -> str:
    """**The visible line with several tasks**, escaped: each child takes its
    own task's reference for the task settings not offered then (`TASK_OWN`
    of the chosen tasks but `ANY_TASKS`), each named by its label; empty for
    one task or none."""
    chosen = chosen_tasks(tasks)
    if len(chosen) < 2:
        return ""
    own = _ordered((frozenset().union(*(_takes((task,)) for task in chosen)) & TASK_OWN)
                   - set(ANY_TASKS))
    names = task_list_text([r.SETTING_LABELS[name] for name in own])
    text = (f"Each child takes its own task’s reference for these: {names}. Choose one task to "
            "set them.")
    return f'<p class="shep-ft-why">{r.e(text)}</p>'


def task_settings_label(tasks: Any) -> str:
    """The task's own settings' title: "Disease settings" (one task)."""
    chosen = chosen_tasks(tasks)
    return f"{TASK_NAMES[chosen[0]]} settings" if len(chosen) == 1 else "Task settings"


def task_settings_view(tasks: Any, item: Optional[Mapping[str, Any]], touched: Any = (),
                       value: Any = None) -> List[Any]:
    """**The task settings for the chosen `tasks` and base `item`**: the
    section of a task's own settings (shown with exactly one task, titled by
    `task_settings_label`), the hop row under the common settings (shown for
    any task), each task field shown while the tasks take it
    (`offered_settings`: Neighbours per hop for any), the hop note
    (`hops_note_html`, beside the hop field, not behind an accordion), the
    several-task line and the hop note's pointer beside Start
    (`hops_warning_html`, in view at the deployment window). `touched` and
    `value` are the form's record and the hop field's text."""
    chosen = chosen_tasks(tasks)
    offered = set(offered_settings(chosen))
    return [gr.update(visible=len(chosen) == 1, label=task_settings_label(chosen)),
            gr.update(visible=bool(chosen)),
            *(gr.update(visible=name in offered) for name in TASK_FIELDS),
            hops_note_html(chosen, item, touched, value), several_note_html(chosen),
            hops_warning_html(chosen, item, touched, value)]


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
    runs = len(ml.runs_of(parents))
    return (f'<p class="shep-muted">Server list: {len(parents)} base checkpoint'
            f'{"s" if len(parents) != 1 else ""} in {runs} run{"s" if runs != 1 else ""}, '
            'latest first.</p>')


def base_summary_html(item: Optional[Mapping[str, Any]]) -> str:
    """The chosen base's summary in the one summary list
    (`renderers.summary_list`), saying what the two selects above do not:
    its run's directory (the Run select gives its name, architecture and
    time), file, configuration (its model, a fact the children keep, never a
    field: `BASE_FACTS`: its configuration line, `renderers.configuration_text`,
    and beside it its whole spec under "Details ›", each field that differs
    from the reference base spec with the reference's value, the encoder's
    layers the hop list must fit among them: the one renderer of a recorded
    spec, `renderers.model_spec_html`, which reads a listed spec), what it
    holds, its Trainer validation figure, what
    its run's saves were ranked by (`renderers.selection_value`), how its
    dataset stands to the one selected (the shared status mark), the split
    seed its record names, which every child trains on (a fact too), and the
    model fingerprint Start sends (copyable). Escaped; empty when none is
    chosen."""
    if not item:
        return ""
    trained = item.get("trained_on_this_dataset")
    if trained is True:
        on = r.status_mark("ok", "the dataset selected")
    elif trained is False:
        on = r.status_mark("warn", "another dataset than the one selected (the start "
                                   "refuses a parent of another knowledge graph)")
    else:
        on = r.status_mark("neutral", r.NOT_RECORDED.lower())
    seed = item.get("split_seed")
    split = (f"{r.setting_words('split_seed', seed)} (every child trains on this split)"
             if seed is not None else r.NOT_RECORDED.lower())
    kind = item.get("checkpoint_kind")
    rows = [
        ("Run", f"<code>{r.e(item.get('architecture'))}/{r.e(item.get('run'))}</code>"),
        ("File", f"<code>{r.e(item.get('filename'))}</code>"),
        ("Configuration", r.e(r.configuration_text(item)) + r.model_spec_html(item.get("spec"))[1]),
        ("Holds", r.e(tc.CHECKPOINT_KINDS.get(kind or "", "not recorded"))),
        ("Validation", r.e(r.validation_value(r.validation_figure(item),
                                              read=r.record_read(item)))),
        ("Selection", r.e(r.selection_value(item))),
        ("Trained on", on),
        (r.SETTING_LABELS["split_seed"], r.e(split)),
        ("Model fingerprint", r.fingerprint_value(item.get("sha256"), what="model fingerprint")),
    ]
    return f'<div class="shep-ft-base">{r.summary_list(rows)}</div>'


# =============================================================================
# The job card
# =============================================================================
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
        named = (f'{r.e(r.run_words(parts[-4], parts[-3]))} '
                 f'<code>{r.e(parent.name)}</code>')
    else:
        named = f"<code>{r.e(parent.name)}</code>"
    return (f'<p class="shep-ft-from"><span class="shep-ft-k">From</span> {named} '
            f'{r.fingerprint_html("model fingerprint", finetune.get("parent_sha256"), copy=False)}'
            "</p>")


def freeze_line_html(finetune: Mapping[str, Any]) -> str:
    """The card's graph backbone: the freeze the start sent to every child
    (the runner's own record of its settings, `_Finetune.describe`), in the
    field's words ("Frozen"; `renderers.setting_words`), or "Trained
    (reference)" when it sent none, so the recipe's filled it. Escaped."""
    settings = finetune.get("settings") or {}
    if "frozen_parts" in settings:
        words = r.setting_words("frozen_parts", settings["frozen_parts"])
    else:
        words = f"{r.setting_words('frozen_parts', FINETUNE['frozen_parts'])} (reference)"
    return (f'<p class="shep-ft-from"><span class="shep-ft-k">'
            f'{r.e(r.SETTING_LABELS["frozen_parts"])}</span> {r.e(words)}</p>')


def freeze_note_html(value: Any) -> str:
    """The note under the fields while a Frozen graph backbone is chosen
    (`FROZEN_NOTE`): what then trains. Empty for a value that keeps nothing
    from training (the recipe's rule, `recipe.FROZEN_PARTS`; no value of the
    page's own)."""
    if not isinstance(value, str) or not FROZEN_PARTS.get(value):
        return ""
    return f'<p class="shep-ft-why">{r.e(FROZEN_NOTE)}</p>'


def child_products(child: Mapping[str, Any]) -> Dict[str, Any]:
    """An ended child's products, read once: its run's best saves by the
    selection its files record (`renderers.best_saves`: the recorded monitor,
    in its recorded direction, from the list's metadata, never a file's name),
    ties shared, as ``{"best": [{"words", "validation", "selection"}],
    "recorded", "problem"}``, `recorded` whether any of its files records a
    selection. It opens that run's files (`list_checkpoints(run=)`): an
    operator's read only."""
    from src.api.services.server_lists import list_checkpoints

    architecture, run = _run_parts(child.get("run_dir"))
    workspace = tc.workspace_of_run(child.get("run_dir"))
    if workspace is None or run is None:
        return {"best": [], "recorded": False, "problem": "its workspace is not listed now"}
    items, problem = tc.checkpoint_items(
        list_checkpoints(workspace, architecture, task=child.get("task"), run=run))
    if problem:
        return {"best": [], "recorded": False, "problem": problem}
    best = r.best_saves(items)
    return {"best": [{"words": ml.checkpoint_words(item, best=True),
                      "validation": item["validation"],
                      "selection": r.selection_words(item)}
                     for key, item in items.items() if key in best],
            "recorded": any(item.get("selection") for item in items.values()),
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
        why = ("no saved file records its selection metric's figure" if found.get("recorded")
               else f"{r.selection_words({}).lower()} in its saved files")
        return f'<td colspan="2" class="shep-muted">{r.e(why)}</td>'
    names = "".join(f"<div>{r.e(entry['words'])}</div>" for entry in best)
    by = f'<div class="shep-sub">{r.e(best[0]["selection"])}</div>'
    return f"<td>{names}{by}</td><td>{r.validation_cell(best[0]['validation'])}</td>"


def child_row_html(child: Mapping[str, Any], *, operator: bool, products: Mapping[str, Any]) -> str:
    """One child's row: Task | Status | Run | Best save | Validation
    (Trainer), escaped (a failure's first line through the one redaction of a
    printed line, `training_console.printed_scrub`, whoever calls). The run is
    its name part (`renderers.run_name`; the parent line and the Task column
    say the rest), its whole id in its title."""
    status = str(child.get("status") or "queued")
    task = TASK_NAMES.get(child.get("task"), str(child.get("task")))
    why = ""
    if status in ("failed", "refused") and child.get("error"):
        first = str(child["error"]).strip().split("\n", 1)[0]
        scrub = tc.printed_scrub(workspaces_root()[0])
        why = f'<div class="shep-sub">{r.e(r.short_reason(scrub(first)))}</div>'
    architecture, run = _run_parts(child.get("run_dir"))
    if run:
        whole = f"{architecture}/{run}"
        run_cell = r.ident_html(r.run_name(run) or whole, title=whole)
    else:
        run_cell = ('<span class="shep-muted">not started yet</span>' if status == "queued" else
                    '<span class="shep-muted">never started</span>')
    return (f"<tr><td>{r.e(task)}</td><td>{tc.job_status_html(status)}{why}</td>"
            f"<td>{run_cell}</td>{_products_cells(child, operator=operator, products=products)}"
            "</tr>")


def job_card_html(status: Mapping[str, Any], *, operator: bool,
                  products: Optional[Mapping[str, Any]] = None) -> str:
    """The Fine-tune job card from the job runner's status (`get_status()`):
    its parent, the start's graph backbone (`freeze_line_html`) and one row
    per child; or, with no Fine-tune status, why
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
    rows = "".join(child_row_html(child, operator=operator, products=products or {})
                   for child in finetune.get("children") or [])
    head = ("<thead><tr><th>Task</th><th>Status</th><th>Run</th>"
            f'<th>Best save{help_button("finetune-best-save", "Best save")}</th>'
            f'<th>Validation (Trainer){help_button("validation-trainer", "Validation (Trainer)")}'
            "</th></tr></thead>")
    link = ('<button type="button" class="shep-link" data-act="training">Charts on Research › '
            'Training ›</button>')
    return (f'<div class="shep-ft-card">{parent_line_html(finetune)}{freeze_line_html(finetune)}'
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
    """Start and Stop (interactive or not) and their reason lines, as one
    decision for the tick and after every action, in the states the Training
    run bar uses (`training_console.run_bar_view`): D8, a training job running
    or stopping, the GPU slot held elsewhere. ``reason`` is beside Start;
    ``stop_reason`` is beside Stop, which is in the job panel under the card:
    D8's refusal while it refuses (the lock is never without its reason, the
    one gated-button rule), else nothing, since the card's statuses say
    whether a child runs. Stop stops a running Fine-tune child (and so
    cancels the queued ones), never another page's job. An enabled Start has
    no reason beside it."""
    s = status.get("status") or "idle"
    child_running = running_child(status) is not None
    view = {"start": False, "stop": False, "stop_reason": ""}
    if not access.allowed:
        view["reason"] = view["stop_reason"] = _why(tc.LOCK_ICON, operator_only(access.condition))
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
    """The card, Start, Stop, Start's and Stop's reason lines, the stop
    question's row and the session's cache, read now for this request. The products are read
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
    card = job_card_html(status, operator=access.allowed, products=products)
    # Start and Stop by the one gated-button rule (`chips.gated`): no lock
    # while D8 allows this request; disabled with the lock while it refuses
    # it, each beside its own reason line.
    return (card, gr.update(**gated(START_LABEL, access.allowed, interactive=view["start"])),
            gr.update(**gated(STOP_LABEL, access.allowed, interactive=view["stop"])),
            f'<div class="shep-ft-reasons">{view["reason"]}</div>',
            f'<div class="shep-ft-reasons">{view["stop_reason"]}</div>',
            gr.update(visible=asking), cache)


def _refresh_card(mode, cache, request: gr.Request):
    """After every action: the card and the actions as they are now."""
    return _card_view(mode, cache, request)


def _poll(location, mode, cache, request: gr.Request):
    """The shell's slow timer (``queue=False``), while Fine-tune is shown: the
    card and the actions; nothing while another page is shown."""
    if location != PAGE:
        return (gr.update(),) * 7
    try:
        return _card_view(mode, cache, request)
    except Exception:  # noqa: BLE001 - a tick that raises would stop the page's updates
        logger.exception("The Fine-tune job could not be read")
        return (gr.update(),) * 7


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


#: The base list's Refresh, by the one gated-button rule (`chips.gated`).
REFRESH_LABEL = "Refresh"


def _on_finetune_bases(request: gr.Request, ws, current):
    """The parent list: the workspace's listed bases (`list_checkpoints(
    learning_stage="base")`). It opens every checkpoint, so D8 is checked
    first, on this event's own request; refused, the list is emptied, nothing
    is opened, its Refresh is disabled with the lock and the line beside it
    says why in the gated-button rule's words (`chips.operator_only`, as
    Start's reason line), the rule every gated button follows (`chips.gated`).

    Returns the parents (state), the run and checkpoint selects, their line
    and the Refresh button.
    """
    access = operator_access(request)
    empty = gr.update(choices=[], value=None)
    refresh = gr.update(**gated(REFRESH_LABEL, access.allowed))
    if not access.allowed:
        why = _why(tc.LOCK_ICON, f"Not listed: listing the bases opens them. "
                                 f"{operator_only(access.condition)}")
        return {}, empty, empty, f'<div class="shep-ft-reasons">{why}</div>', refresh
    name = (ws or {}).get("name")
    if not name:
        return {}, empty, empty, bases_note_html({}, "Select a dataset first."), refresh
    from src.api.services.server_lists import list_checkpoints

    parents, problem = tc.checkpoint_items(list_checkpoints(name, learning_stage="base"))
    return (parents, *ml.parents_update(parents, current), bases_note_html(parents, problem),
            refresh)


def _on_base_chosen(key, parents):
    """A checkpoint chosen: the base's summary."""
    return base_summary_html((parents or {}).get(key) if key else None)


def _on_task_view(tasks, key, parents, touched, hops):
    """The tasks or the base changed (by the operator or by a page's own
    write): the task settings for them (`task_settings_view`), `hops` the hop
    field's text."""
    return task_settings_view(tasks, (parents or {}).get(key) if key else None, touched, hops)


def _on_tasks_input(tasks, touched, *values):
    """**The operator changed the tasks**: each of a task's own settings
    starts again for them (`task_starts`: one task's reference, or for
    several the hop list they share), having left the form's record in the
    browser (`TASK_OWN_FIELDS`, `training_console.record_drops`, chained
    after it; `touched_without` drops them all the same), so a value set for
    one task is never sent for another; a hop list the operator set stays,
    theirs for every child of any task set (`ANY_TASKS`). A page's write: no field's
    input fires. It never writes the record, which is the browser's: an
    answer that arrives late cannot undo a touch made since. `values` are
    the fields', in `FINETUNE_FIELDS` order. Returns each task field's update
    in `TASK_FIELDS` order, then the line on what the operator set
    (`set_by_html`)."""
    starts = task_starts(tasks)
    record = tc.touched_without(touched, TASK_OWN_FIELDS)
    kept = {name for name in ANY_TASKS if name in record}
    written = {name: value for name, value in starts.items() if name not in kept}
    fields = {**dict(zip(FINETUNE_FIELDS, values, strict=True)), **written}
    return [*(gr.update(value=written[name]) if name in written else gr.update()
              for name in TASK_FIELDS), set_by_html(tasks, fields, record)]


def _on_field_input(tasks: Any, touched: Any, *values: Any) -> str:
    """The line on what the operator set (`set_by_html`), said again on a
    field's input, chained after the field's touch listener
    (`training_console.record_touches`), as the base form's fields say
    theirs: the record it reads holds the field's key, and, run again alone
    for the last input ("always_last"), it reads the record as it is then
    (after Reset or a task switch, without what they dropped). Its values
    are the fields', in `FINETUNE_FIELDS` order. It writes only the line,
    never the record, which is the browser's."""
    return set_by_html(tasks, dict(zip(FINETUNE_FIELDS, values, strict=True)), touched)


def _on_finetune_reset(tasks):
    """**Reset to reference** (`training_console.reset_outputs`, the base
    form's): the record empty, so with several tasks each child takes its own
    task's reference again; every field back where it starts for the chosen
    tasks (`start_values`); the line said for nothing set."""
    starts = start_values(tasks)
    return tc.reset_outputs(starts, set_by_html(tasks, dict(zip(FINETUNE_FIELDS, starts,
                                                                strict=True)), []))


def _on_guards(*values):
    """A guard of a rule across settings changed (the scheduler, mixed
    precision): `training_console.guard_updates` of this page's settings
    (`FINETUNE_FIELDS`), as the base form's guards are answered."""
    return tc.guard_updates(FINETUNE_FIELDS, values)


def _on_hops_input(tasks, key, parents, touched, *values):
    """The hop field's input, chained after its touch listener
    (`training_console.record_touches`), so the record it reads holds its key
    (run again alone for the last input, the record as it is then): its one
    listener besides it, the line on what the operator set (`set_by_html`, as
    every field's listener says it), the hop note and its pointer beside
    Start, for the list typed (`hops_note_html`, `hops_warning_html`).
    `values` are the fields', in `FINETUNE_FIELDS` order. It writes only
    them, never the record, which is the browser's.
    One listener, not two: two input listeners of one field that each send
    its last value ("always_last") retriggered each other in the browser
    without end (seen on the page while it had two; Gradio runs a deferred
    listener again by dispatching its trigger again)."""
    item = (parents or {}).get(key) if key else None
    fields = dict(zip(FINETUNE_FIELDS, values, strict=True))
    hops = fields["num_neighbors"]
    return (set_by_html(tasks, fields, touched), hops_note_html(tasks, item, touched, hops),
            hops_warning_html(tasks, item, touched, hops))


def _on_finetune_start(request: gr.Request, ws, parents, key, tasks, device, run_name,
                       touched, *values):
    """Start: an operator control (D8), checked first on this event's own
    request; then the Fine-tune start (`_start`) with the settings' fields'
    `values`, in `FINETUNE_FIELDS` order, and `touched`, the form's record of
    the fields the operator touched, as this click's own request carries it
    (`training_console.touched_record`).

    Returns the notice beside Start, a refusal with the shared mark of a
    refusal, escaped (`training_console.notice_html`), or nothing once started
    (the card shows the job); and the Stop's notice cleared, which spoke of a
    job before this start.
    """
    access = operator_access(request)
    if not access.allowed:
        return tc.notice_html(access.refusal, mark="error"), ""
    said = _start(ws, parents, key, tasks, device, run_name,
                  dict(zip(FINETUNE_FIELDS, values, strict=True)), touched)
    return tc.notice_html(said, mark="error"), ""


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


def _start(ws, parents, key, tasks, device, run_name, values: Mapping[str, Any],
           touched: Any) -> str:
    """Start's work after D8, as a plain sentence ("" once started): the
    chosen base and tasks (in `TASKS` order), the settings the chosen tasks
    take (`offered_settings`) of `values` (the fields', by `FINETUNE_FIELDS`)
    that the operator touched (`touched`; decision S2: `sent_settings`),
    whatever their value, sent as the
    start's settings (`FinetuneInputs.settings`, which the runner checks; a
    field the tasks do not take, a task setting hidden with several tasks, is
    never sent), the workspace and the parent resolved by the server from its
    lists again, and the runner's Fine-tune start with the parent's listed
    digest and the dataset fingerprint this session showed. The runner's
    refusal is said in its words."""
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
    chosen = chosen_tasks(tasks)
    if not chosen:
        return "Not started: choose at least one task."
    settings, problems = sent_settings(chosen, values, touched)
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
    error = result.get("error")
    # A setting the runner names first by its record name, by the field's
    # label instead, as the Training page's Branch says the runner's
    # refusal of its settings (`labelled_problem`).
    return tc.not_started(tc.labelled_problem(error) if result.get("invalid_settings")
                          else error)


# =============================================================================
# Help
# =============================================================================
def task_setting_help(name: str) -> Tuple[str, str]:
    """A task setting's help, ``(title, text)``: its label, what it is (its
    field spec's description, as every form's help says it), each task's
    reference (the one fill, `run_plan.new_run_values`, of each task whose
    run takes it), and where its field starts and what is sent."""
    each = "; ".join(f"{TASK_NAMES[task]} {r.setting_words(name, values[name])}"
                     for task in TASKS
                     for values in [new_run_values("finetune", task, {})] if name in values)
    if name in ANY_TASKS:
        where = ("With one task it starts at that task's reference. With several, left "
                 "untouched each child takes its own; a value you set is every child's, until "
                 "Reset to reference.")
    else:
        where = ("Offered with one task, starting at its reference; with several, each child "
                 "takes its own.")
    return (r.SETTING_LABELS[name],
            f"{by_name(name).description}\n\nReference, by task: {each}.\n\n{where} Only a "
            "value you edit is sent. Each reference is a provisional default from the source.")


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
        "The runs holding a base checkpoint, latest first, each led by its name or start "
        "time.\n\n"
        "Its architecture and start time follow; its directory is in the summary. Only bases "
        "are offered: a task model is never a parent (children never chain), and a file "
        "without a model spec is never trained from here.",
    ),
    "ft_checkpoint": (
        "Checkpoint",
        "The base to fine-tune, from the chosen run, newest first. Its summary is under the "
        "list.\n\n"
        "• Latest save, Best (by the selection its files record, named by its metric) or "
        "Top-k save.\n"
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
    **{f"ft_{name}": tc.setting_help(name, built_values(), FINETUNE_SOURCES)
       for name in COMMON_SHOWN + MORE_FIELDS},
    **{f"ft_{name}": task_setting_help(name) for name in TASK_FIELDS},
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
        "The saved file of a child's run with the best value of the selection metric its "
        "files record, in the direction they record; files that tie share it.\n\n"
        "• The metric and its direction are named under the file (a task model selects on "
        "validation pooled MRR, higher is better).\n"
        "• It is the Trainer's figure on the val samples, not a test result: test the model "
        "on Research › Test.\n"
        "• Shown to the operator once the child has ended (reading it opens the run's "
        "checkpoints).",
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
                # The dataset and its identity beside it (under it on a narrow
                # form), so the common settings below come into view sooner.
                with gr.Row(elem_classes=["shep-ft-ds-row"]):
                    dataset = gr.Dropdown(label="Dataset", choices=[], value=None,
                                          elem_id="ft_dataset")
                    identity = plain_html(identity_html({}), elem_id="ft_identity")
                # Run → Checkpoint, two cascading selects side by side (the
                # skill's §4), then the list's Refresh; on a narrow form they
                # wrap, one under the other.
                with gr.Row(elem_classes=["shep-ft-parent-row"]):
                    run_select = gr.Dropdown(label="Run", choices=[], value=None,
                                             elem_id="ft_run")
                    checkpoint = gr.Dropdown(label="Checkpoint", choices=[], value=None,
                                             elem_id="ft_checkpoint")
                    refresh = gr.Button(REFRESH_LABEL, size="sm", scale=0, elem_id="ft_refresh")
                bases_note = plain_html("", elem_id="ft_bases_note")
                base = plain_html("", elem_id="ft_base")
                tasks = gr.CheckboxGroup(label="Tasks", choices=task_choices({}), value=[],
                                         elem_id="ft_tasks")
                tasks_note = plain_html("", elem_id="ft_tasks_note")
                # What every chosen task's run takes (decision S1), each field
                # starting at the value the run takes without it: the common
                # settings first, the rest one click away (a setting the run
                # would not read hidden, decision S3), then the task settings.
                values = built_values()
                unread = not_read_by_rules(values)
                fields: Dict[str, Any] = {}
                with gr.Row(elem_classes=["shep-ft-settings"]):
                    for name in COMMON_SHOWN:
                        fields[name] = tc.setting_field(name, values[name], elem_id=f"ft_{name}")
                freeze_note = plain_html(freeze_note_html(values["frozen_parts"]),
                                         elem_id="ft_freeze_note")
                # The hop list, offered for any task set, under the common
                # settings with its note beside it (`task_settings_view`): the
                # start's sentence when the reference does not fit the base is
                # in view without opening anything (2b).
                with gr.Row(visible=False, elem_id="ft_hops_row",
                            elem_classes=["shep-ft-settings", "shep-ft-hops-row"]) as hops_row:
                    for name in ANY_TASKS:
                        fields[name] = tc.setting_field(name, values[name],
                                                        elem_id=f"ft_{name}", visible=False)
                    hops_note = plain_html("", elem_id="ft_hops_note")
                with gr.Accordion("More settings", open=False, elem_id="ft_more"):
                    with gr.Row(elem_classes=["shep-ft-settings", "shep-ft-grid"]):
                        for name in MORE_FIELDS:
                            fields[name] = tc.setting_field(name, values[name],
                                                            elem_id=f"ft_{name}",
                                                            visible=name not in unread)
                # A task's own settings (`task_settings_view`), with one task.
                with gr.Accordion(task_settings_label(()), open=False, visible=False,
                                  elem_id="ft_task_more") as task_more:
                    with gr.Row(elem_classes=["shep-ft-settings", "shep-ft-grid"]):
                        for name in TASK_OWN_FIELDS:
                            fields[name] = tc.setting_field(name, values[name],
                                                            elem_id=f"ft_{name}", visible=False)
                several_note = plain_html("", elem_id="ft_several_note")
                # What the operator set (decision S2), and Reset to reference:
                # the base form's one line and Reset.
                with gr.Row(elem_classes=["shep-ft-setby-row"]):
                    set_by = plain_html(set_by_html((), {name: tc.form_value(values[name])
                                                         for name in FINETUNE_FIELDS}, []),
                                        elem_id="ft_setby")
                    reset = gr.Button("Reset to reference", size="sm", scale=0,
                                      elem_id="ft_reset")
                with gr.Row(elem_classes=["shep-ft-settings"]):
                    device = gr.Radio(label="Device", choices=["auto", "cuda", "cpu"],
                                      value="auto", elem_id="ft_device")
                    run_name = gr.Textbox(label="Run name", value="", max_lines=1,
                                          elem_id="ft_run_name")
            with gr.Column(elem_id="ft_start_ft", elem_classes=["shep-panel-ft"]):
                # Start, its reason and its refusal on one row (the skill's §9:
                # a refusal beside what it refuses).
                with gr.Row(elem_classes=["shep-ft-start-row"]):
                    start = gr.Button(START_LABEL, variant="primary", interactive=False,
                                      scale=0, elem_id="ft_start")
                    reason = plain_html("", elem_id="ft_reason")
                    hops_warn = plain_html("", elem_id="ft_hops_warn")
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
                    stop = gr.Button(STOP_LABEL, variant="secondary", interactive=False,
                                     scale=0, elem_id="ft_stop")
                    stop_reason = plain_html("", elem_id="ft_stop_reason")
                    stop_notice = plain_html("", elem_id="ft_stop_notice")
                with gr.Row(visible=False, elem_id="ft_confirm_row") as confirm_row:
                    stop_text = plain_html("", elem_id="ft_stop_text")
                    stop_confirm = gr.Button("■ Stop training", variant="stop", scale=0,
                                             elem_id="ft_stop_confirm")
                    stop_keep = gr.Button("Keep training", scale=0, elem_id="ft_stop_keep")

    # =========================================================================
    # Wiring
    # =========================================================================
    card_step = {"fn": _refresh_card, "inputs": [mode, cache],
                 "outputs": [card, start, stop, reason, stop_reason, confirm_row, cache],
                 "show_progress": "hidden"}
    bases_step = {"fn": _on_finetune_bases, "inputs": [ws, checkpoint],
                  "outputs": [parents, run_select, checkpoint, bases_note, refresh],
                  "show_progress": "hidden"}
    base_step = {"fn": _on_base_chosen, "inputs": [checkpoint, parents], "outputs": [base],
                 "show_progress": "hidden"}
    # Explicit is what the operator touched (decision S2): the form's record,
    # the browser's (`training_console.touched_record`), which Start's own
    # request carries; each field's input adds it there, and a change of
    # tasks drops a task's own settings from it, there too.
    touched = tc.touched_record("ft_touched")
    field_list = [fields[name] for name in FINETUNE_FIELDS]
    hops = fields[ANY_TASKS[0]]
    task_view_step = {"fn": _on_task_view, "inputs": [tasks, checkpoint, parents, touched, hops],
                      "outputs": [task_more, hops_row, *(fields[name] for name in TASK_FIELDS),
                                  hops_note, several_note, hops_warn],
                      "show_progress": "hidden"}

    # The page shown: the list and the tasks, the bases (gated), the task
    # settings for them, the card.
    ctx.on_page_open(PAGE, {
        "fn": _open, "inputs": [ws, ctx.training_workspace, tasks],
        "outputs": [ws, dataset, identity, tasks, tasks_note],
        "show_progress": "hidden"}).then(**bases_step).then(**base_step).then(
        **task_view_step).then(**card_step).then(**ctx.chip_refresh)
    # The shell's slow timer, a read-only poll without the queue.
    ctx.timer_slow.tick(_poll, inputs=[ctx.location, mode, cache],
                        outputs=[card, start, stop, reason, stop_reason, confirm_row, cache],
                        queue=False, show_progress="hidden")

    dataset.input(_on_dataset, inputs=[dataset, tasks],
                  outputs=[ws, identity, tasks, tasks_note],
                  show_progress="hidden").then(**bases_step).then(**base_step).then(
        **ctx.chip_refresh)
    refresh.click(**bases_step).then(**base_step).then(**ctx.chip_refresh)
    run_select.change(ml.on_run_chosen, inputs=[run_select, parents, checkpoint],
                      outputs=[checkpoint], show_progress="hidden")
    # The base or the tasks changed: the task settings for them (shown, titled,
    # opened when the hop list cannot run on the base, with its note).
    checkpoint.change(**base_step).then(**task_view_step)
    tasks.change(**task_view_step)
    fields["frozen_parts"].change(freeze_note_html, inputs=[fields["frozen_parts"]],
                                  outputs=[freeze_note], show_progress="hidden")
    # Each field's input records it, then (chained after it) says the line
    # again, as the base form's do; its last input is always said
    # ("always_last"). The hop list's one listener says its note and the
    # note's pointer too.
    for name in FINETUNE_FIELDS:
        touch = tc.record_touches(fields[name], name, touched)
        if name in ANY_TASKS:
            touch.then(_on_hops_input, inputs=[tasks, checkpoint, parents, touched, *field_list],
                       outputs=[set_by, hops_note, hops_warn], queue=False,
                       trigger_mode="always_last", show_progress="hidden")
        else:
            touch.then(_on_field_input, inputs=[tasks, touched, *field_list],
                       outputs=[set_by], queue=False, trigger_mode="always_last",
                       show_progress="hidden")
    # The operator's own change of the tasks: a task's own settings leave the
    # record in the browser, then (chained after it) start again at their
    # reference, a hop list set kept; the line again.
    tc.record_drops(tasks, TASK_OWN_FIELDS, touched).then(
        _on_tasks_input, inputs=[tasks, touched, *field_list],
        outputs=[*(fields[name] for name in TASK_FIELDS), set_by], queue=False,
        show_progress="hidden")
    # Reset to reference: the record empty, every field where it starts for
    # the tasks, the line; then the hop note for them.
    reset.click(_on_finetune_reset, inputs=[tasks], outputs=[touched, set_by, *field_list],
                queue=False, show_progress="hidden").then(**task_view_step)
    # A setting a rule across settings reads only under its guard is shown while
    # the run would read it (decision S3), as on the base form.
    dependent, guards = tc.rule_fields(FINETUNE_FIELDS)
    gr.on(triggers=[fields[name].change for name in guards], fn=_on_guards,
          inputs=[fields[name] for name in guards],
          outputs=[fields[name] for name in dependent], queue=False, show_progress="hidden")

    start.click(_on_finetune_start,
                inputs=[ws, parents, checkpoint, tasks, device, run_name, touched, *field_list],
                outputs=[notice, stop_notice], show_progress="hidden").then(
        tc.close_stop_question, inputs=[mode], outputs=[mode], show_progress="hidden").then(
        **card_step).then(**ctx.chip_refresh)
    # Stop asks first: the Training page's stop logic, D8 on each request.
    stop.click(_on_finetune_stop_request, inputs=[mode], outputs=[stop_text, mode],
               show_progress="hidden").then(**card_step).then(**ctx.chip_refresh)
    stop_confirm.click(_on_finetune_stop, inputs=[mode], outputs=[stop_notice, mode],
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
{_PAGE} .shep-ft-bd {{ padding: 8px 16px 12px !important; gap: 8px !important; }}
{_PAGE} .shep-ft-parent-row {{ align-items: flex-end !important; flex-wrap: wrap !important; gap: 8px 12px !important; }}
{_PAGE} #ft_run, {_PAGE} #ft_checkpoint {{ flex: 1 1 300px !important; min-width: 0 !important; }}
{_PAGE} #ft_refresh {{ flex: none !important; width: auto !important; min-width: max-content !important; }}
{_PAGE} .shep-ft-settings {{ flex-wrap: wrap !important; gap: 8px 16px !important; }}
{_PAGE} .shep-ft-ds {{ font-size: 15px; display: flex; flex-wrap: wrap; gap: 4px 6px; align-items: center; }}
{_PAGE} .shep-ft-ds-row {{ align-items: flex-end !important; flex-wrap: wrap !important; gap: 8px 12px !important; }}
{_PAGE} .shep-ft-ds-row > .form:has(#ft_dataset) {{ flex: 1 1 280px !important; min-width: 0 !important; }}
{_PAGE} #ft_identity {{ flex: 2 1 420px !important; min-width: 0 !important; padding-bottom: 6px; }}
/* The chosen base's details: the one summary list in the parameters' panel, no box of its own
   (`base_summary_html`; the skill's §2). */
{_PAGE} .shep-ft-base .shep-kv {{ font-size: 14px; gap: 2px 14px; }}
{_PAGE} #ft_base:not(:has(.shep-ft-base)), {_PAGE} #ft_tasks_note:not(:has(p)), {_PAGE} #ft_freeze_note:not(:has(p)), {_PAGE} #ft_hops_note:not(:has(p)), {_PAGE} #ft_several_note:not(:has(p)) {{ display: none !important; }}
/* The line on what the operator set beside Reset to reference (the base form's; its text is the
   shared `.shep-setby`). */
{_PAGE} .shep-ft-setby-row {{ align-items: flex-start !important; flex-wrap: nowrap !important; gap: 8px 12px !important; }}
{_PAGE} #ft_setby {{ flex: 1 1 auto !important; min-width: 0 !important; }}
{_PAGE} #ft_reset {{ flex: none !important; width: auto !important; min-width: max-content !important; }}
{_PAGE} .shep-ft-why {{ margin: 0; font-size: 14px; color: var(--shep-text-2); }}
/* The hop list and its note beside it (`hops_note_html`): the note takes the rest of the row and
   wraps under the field when the row is narrow. */
{_PAGE} .shep-ft-hops-row {{ align-items: flex-start !important; }}
{_PAGE} .shep-ft-hops-row > .form:has(#ft_num_neighbors) {{ flex: 1 1 280px !important; min-width: 0 !important; }}
{_PAGE} #ft_hops_note {{ flex: 2 1 240px !important; min-width: 0 !important; align-self: center; }}
/* The settings under an accordion in columns of one width, as the base form's grid lays them. */
{_PAGE} .shep-ft-grid > .form {{ display: grid !important; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px 16px !important; background: transparent !important; border: 0 !important; box-shadow: none !important; }}
{_PAGE} .shep-ft-hops {{ margin: 0; font-size: 14px; overflow-wrap: anywhere; }}
{_PAGE} .shep-ft-start-row, {_PAGE} .shep-ft-stop-row {{ align-items: center !important; flex-wrap: wrap !important; gap: 8px 14px !important; }}
{_PAGE} .shep-ft-start-row > button, {_PAGE} .shep-ft-stop-row > button {{ flex: none !important; width: auto !important; min-width: max-content !important; padding: 0 14px !important; height: 40px; }}
{_PAGE} #ft_reason, {_PAGE} #ft_stop_reason, {_PAGE} #ft_notice, {_PAGE} #ft_stop_notice, {_PAGE} #ft_hops_warn {{ flex: 1 1 240px !important; min-width: 0 !important; }}
/* The hop note's pointer beside Start (`hops_warning_html`); it, and Start's reason, take no room
   while empty, so whichever is said sits next to Start. */
{_PAGE} #ft_hops_warn {{ font-size: 15px; }}
{_PAGE} #ft_hops_warn:not(:has(.shep-ft-hops-warn)), {_PAGE} #ft_reason:not(:has(.shep-ft-reason)) {{ display: none !important; }}
/* The configuration's "Details ›" beside its line (`base_summary_html`); opened, the spec's list
   takes the lines under it. */
{_PAGE} .shep-ft-base .shep-v > .shep-more {{ display: inline-block; margin-left: 8px; vertical-align: top; }}
{_PAGE} .shep-ft-reason {{ display: inline-flex; align-items: flex-start; gap: 6px; font-size: 15px; color: var(--shep-text-2); }}
{_PAGE} .shep-ft-reason > svg {{ flex: none; margin-top: 3px; }}
/* Start's and Stop's notices, each beside its button: a refusal carries the shared mark of a
   refusal (`training_console.notice_html`); an empty one takes no room. */
{_PAGE} #ft_notice, {_PAGE} #ft_stop_notice {{ font-size: 15px; overflow-wrap: anywhere; }}
{_PAGE} #ft_notice:not(:has(.shep-notice-t)), {_PAGE} #ft_stop_notice:not(:has(.shep-notice-t)) {{ display: none !important; }}
{_PAGE} .shep-ft-card {{ display: flex; flex-direction: column; gap: 10px; font-size: 15px; }}
{_PAGE} .shep-ft-empty {{ margin: 0; color: var(--shep-text-2); }}
{_PAGE} .shep-ft-from {{ margin: 0; overflow-wrap: anywhere; }}
{_PAGE} .shep-ft-k {{ font-size: 14px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: var(--shep-text-2); margin-right: 8px; }}
{_PAGE} .shep-ft-table {{ width: 100%; border-collapse: collapse; font-size: 15px; }}
{_PAGE} .shep-ft-table th {{ text-align: left; font-size: 14px; font-weight: 600; color: var(--shep-text-2); padding: 6px 8px; border-bottom: 1px solid var(--shep-border); white-space: nowrap; }}
{_PAGE} .shep-ft-table td {{ padding: 8px; border-bottom: 1px solid var(--shep-border); vertical-align: top; min-width: 0; }}
{_PAGE} .shep-ft-table code.shep-ident {{ max-width: 40ch; }}
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
