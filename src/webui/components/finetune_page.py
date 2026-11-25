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

- **The setup notice** (the UI sweep of 2026-11-22; the WebUI skill's §9):
  while no workspaces folder can be used, the folder's notice is said once,
  at the top of the form it empties (`setup_html`: its state, what it means
  for Start, its next step a link to Settings › Data locations, the rest
  under "Details ›"), the GPU lock's merged into it; the identity, base and
  task lines then say nothing of it and Start's reason is its phrase
  (R-WHY, `renderers.phrase_line`).
- **Dataset**: its own select over the server's workspace list
  (`workspaces.read_workspaces`, the shared option label), starting from the
  Training run bar's selection the first time the page opens and never
  writing it; the dataset identity under it (`renderers.dataset_identity_html`)
  is the one place its fingerprint is shown.
- **The parent**: Run → Checkpoint over the workspace's **listed bases only**
  (`server_lists.list_checkpoints(learning_stage="base")`): a task model (a
  child never chains), a file without a model spec and an unreadable file are
  never offered; a weights-only or early-stopped base is. Listing opens every
  checkpoint, so it is an operator control (D8). **The chosen base's summary
  is one line** (D3, D8 and R2, the owner's decisions of 2026-11-24: one
  glance tells whether the parent is right): its stage's facts
  (`PARENT_STAGE_FACTS`: the model fingerprint Start sends, how its dataset
  stands to the one chosen, a warning of its own record), its name and save
  being the selects' over it; its run's directory, file, configuration,
  Trainer validation figure, selection, split seed and the list's count
  behind its one "Details ›" (`base_summary_html`): **its split seed and its
  model are the base's facts, never fields** (`BASE_FACTS`): a finetune
  trains on its base's split and keeps its base's graph side. The list's
  own line says only what is to act on (`bases_note_html`).
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
- Device and Run name; **Start**, a bar (the skill's §2): its reason, a
  one-line relation beside it (under it on a narrow form), and under it the
  notes the bar's width, what Start would refuse of a value and a refusal
  (the shared mark of a refusal, escaped): the page's own check of a value,
  or the job runner's refusal in its words, a setting it names first by the
  field's label (`training_console.labelled_problem`).

**Layout** (the UI sweep of 2026-11-22): each panel sized by what it holds
(`shep-sized`: a basis, a share and a minimum in rem, no `scale=`); the
form's two columns side by side while it holds both at their basis (the
base's, and the children's widest row: the three tasks, or Epochs, Seed and
Graph backbone), else one under the other; field rows wrap whole fields and a
choice group is whole on its line or one option per line (`shep-fields`, the
shared rule).

**Start** (D8 first): the Dataset and Run selects as Start's own request
carries them, against what the session read for them (`shell.unread_choice`):
a choice whose reading has not landed, or failed, is refused beside Start,
never started on the choice before it (OPS-C-R2); the workspace and the
parent are resolved by the server from its lists again (`resolve_workspace`,
`resolve_checkpoint`; never a client path), then the job runner's one
Fine-tune start
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
from src.webui.shell import page_elem_id, picked, plain_html, unread_choice
from src.webui.style import FIT, FIT_WIDE, NARROW_ATTR, SIDE

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
FROZEN_NOTE = ("Frozen graph backbone: only the patient encoder and the task head train. The LP "
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
    """Why each task that is not ready is not, one sentence per reason, its
    tasks named together ("Gene and Patients-like-me are not ready: no
    functional fixture."), escaped; empty when every task is ready."""
    reasons: Dict[str, List[str]] = {}
    for task in TASKS:
        why = readiness.get(task)
        if why is not None:
            reasons.setdefault(why, []).append(TASK_NAMES[task])
    if not reasons:
        return ""
    parts = [f"{task_list_text(names)} {'is' if len(names) == 1 else 'are'} not ready: {why}."
             for why, names in reasons.items()]
    return f'<p class="shep-ft-why">{r.e(" ".join(parts))}</p>'


def _readiness(ws: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    from src.api.services.server_lists import task_readiness

    return task_readiness(ws.get("name"))


def _tasks_update(ws: Mapping[str, Any], chosen: Any) -> Tuple[Any, str]:
    """The Tasks options for the workspace `ws` and why a task is not ready,
    said once a dataset is listed; with none, the form's setup notice or the
    identity line says why (`setup_html`, `identity_html`), never this line."""
    readiness = _readiness(ws)
    kept = [task for task in TASKS if task in (chosen or ())]
    note = tasks_note_html(readiness) if ws.get("row") else ""
    return gr.update(choices=task_choices(readiness), value=kept), note


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
    setting is refused ("Task loss weight must be set."), never
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
    nothing = ("Each child takes its task’s reference values"
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


#: The lead of the hop note when the reference does not fit the chosen base
#: (a sentence of its own before the start's: one colon a sentence).
HOPS_REFERENCE_LEAD = "The reference does not fit this base"
#: The lead of the hop note when several tasks' references differ (each
#: task's reference under it, as term rows).
HOPS_EACH_LEAD = "Untouched, each child takes its task’s reference:"


def hops_note_html(tasks: Any, item: Optional[Mapping[str, Any]], touched: Any = (),
                   value: Any = None) -> str:
    """**The line beside Neighbours per hop**, escaped: when the hop list a
    child would sample cannot run on the chosen base (`hops_problems`), that
    sentence beside the ▲ of a condition to watch, led by
    `HOPS_REFERENCE_LEAD` while it is a task's reference; else, with several
    tasks whose references differ and the field untouched, each child's
    (`HOPS_EACH_LEAD`, then each task's reference as a term row: "Disease
    15, 10, 5", "Gene all neighbours, 10, 5"; once touched, Reset to
    reference makes it so again); else empty."""
    problems = hops_problems(tasks, item, touched, value)
    touched_hops = "num_neighbors" in tc.touched_keys(touched)
    if problems:
        words = " ".join(problems)
        if not touched_hops:
            words = f"{HOPS_REFERENCE_LEAD}. {words}"
        return f'<p class="shep-ft-hops">{r.status_mark("warn", words)}</p>'
    chosen = chosen_tasks(tasks)
    references = {task: new_run_hops("finetune", task, {}) for task in chosen}
    if touched_hops or len({tuple(hops) for hops in references.values()}) < 2:
        return ""
    rows = [(TASK_NAMES[task], r.e(r.setting_words("num_neighbors", hops)))
            for task, hops in references.items()]
    return (f'<div class="shep-ft-why"><p>{r.e(HOPS_EACH_LEAD)}</p>'
            f'{r.summary_list(rows, classes="shep-ft-each")}</div>')


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


#: Why no dataset can be chosen with a usable workspaces folder that holds
#: none yet (the Datasets page says how one is built).
NO_WORKSPACE_YET = "No workspace in the folder yet: Research › Datasets says how to build one."


def identity_html(ws: Mapping[str, Any]) -> str:
    """The chosen dataset's identity (the one place its fingerprint is
    shown), or why there is none; nothing while the form's setup notice says
    why (the workspaces folder's notice, `setup_html`: said once a screen)."""
    row = ws.get("row")
    if not row:
        if ws.get("notice"):
            return ""
        why = ws.get("problem") or ("No dataset selected: choose one from the server's list."
                                    if ws.get("choices") else NO_WORKSPACE_YET)
        return f'<p class="shep-muted">{r.e(why)}</p>'
    return (f'<div class="shep-ft-ds">'
            f'{r.dataset_identity_html(row, name=False, digits=ws.get("digits") or r.SHORT_HEX)}'
            "</div>")


#: The element ids of Start and Stop, which a dimmed one's reason names (R-WHY's ``data-for``).
START_ID, STOP_ID = "ft_start", "ft_stop"


def _said_above(ws_notice: Any, gpu: Optional[Mapping[str, Any]]) -> bool:
    """Whether the form's setup notice says the GPU lock's notice: the lock
    cannot be used because no workspaces folder is set (`gpu`'s ``unset``,
    `training_console.gpu_view`) while the folder's notice is shown."""
    return bool(ws_notice and gpu and gpu.get("unset") and gpu.get("notice"))


def setup_html(ws_notice: Any, gpu: Optional[Mapping[str, Any]] = None) -> str:
    """**The form's setup notice** (the WebUI skill's §9, the text brief's
    §10): the workspaces folder's notice (`workspaces.setting_notice`, kept
    with the page's workspace, `_workspace`), said once, at the top of the
    form it empties (no dataset, base or task to choose): its state, the next
    step as a link to Settings › Data locations, the rest under "Details ›"
    (`renderers.notice_block`). With it the GPU lock's notice when the lock
    cannot be used because of it (`_said_above`; it follows from the
    folder's, ``also``). What that means for Start is said once, by Start's
    relation to it beside Start (`actions_view`); the identity,
    base and task lines say nothing of it. Empty while the folder can be
    used."""
    if not ws_notice:
        return ""
    merged = _said_above(ws_notice, gpu)
    return r.notice_block(ws_notice, also=(gpu["notice"],) if merged else ())


def listed_words(parents: Mapping[str, Any]) -> str:
    """How many bases the server's list holds and in how many runs, in the
    order the selects offer them ("6 base checkpoints in 4 runs, latest
    first"): the base summary's "Server list" row (`base_summary_html`)."""
    runs = len(ml.runs_of(parents))
    return (f'{len(parents)} base checkpoint{"s" if len(parents) != 1 else ""} in {runs} '
            f'run{"s" if runs != 1 else ""}, latest first')


def bases_note_html(parents: Mapping[str, Any], problem: Optional[str]) -> str:
    """**The base list's line, only while it says something to act on** (the
    owner's D3/D8 of 2026-11-24): why the list is empty, or could not be
    read; nothing for a list that holds bases, whose count is a fact of the
    chosen base's Details (`listed_words`, `base_summary_html`)."""
    if problem:
        return f'<p class="shep-muted">{r.e(problem)}</p>'
    if not parents:
        return ('<p class="shep-muted">No base checkpoint in this dataset\'s runs: train one on '
                'Research › Training.</p>')
    return ""


#: **The parent's facts this stage shows** (D3, D8 and R2, the owner's
#: decisions of 2026-11-24: one glance tells whether the chosen parent is
#: right), in the order its line says them, each a ``data-fact`` of the
#: summary's line (`base_summary_html`): its model fingerprint, which Start
#: sends; how its dataset stands to the one chosen (`TRAINED_ON_TAGS`); and
#: what its own record says that changes the choice (`parent_warnings`).
#: Its name and its save are the Run and Checkpoint selects' over the line
#: (the WebUI skill's §1.2: each fact once); everything else is behind the
#: summary's one "Details ›".
PARENT_STAGE_FACTS = ("fingerprint", "trained_on", "warnings")
#: **How the parent's dataset stands to the one chosen**, a self-naming tag
#: by the shared status mark (`listed_record`'s ``trained_on_this_dataset``:
#: the dataset it records against the workspace's now). Another dataset is a
#: condition to watch: the start refuses a parent of another knowledge graph
#: (`TrainingManager.start_finetune`), which another dataset may or may not
#: be.
TRAINED_ON_TAGS = {True: ("ok", "same dataset"),
                   False: ("warn", "another dataset: Start refuses it if its graph differs"),
                   None: ("neutral", "dataset not recorded")}


def parent_warnings(item: Mapping[str, Any]) -> List[str]:
    """**What the base's own record says that changes the choice**, each a
    condition to watch: a record this build does not read (its validation and
    selection are then not read either, `renderers.record_read`), and a model
    spec it cannot read (the start's hop check reads it, `record.listed_spec`).
    A hop list that cannot run on the base is the chosen tasks' (it is said
    beside the hop field, `hops_note_html`, and pointed at beside Start),
    never here."""
    out = []
    if not r.record_read(item):
        out.append("record not readable")
    if listed_spec(item.get("spec"))[0] is None:
        out.append("model spec not readable")
    return out


def base_summary_html(item: Optional[Mapping[str, Any]],
                      parents: Optional[Mapping[str, Any]] = None) -> str:
    """**The chosen base's summary: one line of its stage's facts, the rest
    behind one "Details ›"** (D3, D8 and R2, the owner's decisions of
    2026-11-24: nine rows and the list's line became one line). The line says
    `PARENT_STAGE_FACTS` as tags, each a ``data-fact``, with group gaps and no
    " · " chain: the model fingerprint Start sends (copyable), how its
    dataset stands to the one chosen (`TRAINED_ON_TAGS`) and any warning of
    its own record (`parent_warnings`). Its "Details ›" closes the line and
    opens under it (the one summary list, `renderers.details_list`, kept open
    across redraws while the same base is shown), with no disclosure inside
    it (the skill's §1.5): its run's directory (the Run select gives its name,
    architecture and time), file, configuration (its model, a fact the
    children keep, never a field, `BASE_FACTS`: the one renderer of a recorded
    spec in its rows form, `renderers.model_spec_rows`, each field that
    differs from the reference base spec a row with the reference's value,
    none counted away), what it holds, its
    Trainer validation figure, what its run's saves were ranked by
    (`renderers.selection_value`), the split seed its record names, which
    every child trains on (a fact too), and how many bases the server lists
    (`listed_words`, from `parents`). Escaped; empty when none is chosen."""
    if not item:
        return ""
    seed = item.get("split_seed")
    split = (f"{r.setting_words('split_seed', seed)} (every child trains on this split)"
             if seed is not None else r.NOT_RECORDED.lower())
    kind = item.get("checkpoint_kind")
    warnings = parent_warnings(item)
    facts = {
        "fingerprint": r.fingerprint_html("model fingerprint", item.get("sha256")),
        "trained_on": r.status_mark(*TRAINED_ON_TAGS[item.get("trained_on_this_dataset")]),
        "warnings": "".join(r.status_mark("warn", words) for words in warnings),
    }
    tags = "".join(f'<span class="shep-ft-tag" data-fact="{key}">{facts[key]}</span>'
                   for key in PARENT_STAGE_FACTS if facts[key])
    # Its configuration as rows (`renderers.model_spec_rows`): every field that
    # differs from the reference on a row of its own, none counted away and no
    # disclosure inside this one.
    spec_rows, spec_titles = r.model_spec_rows(item.get("spec"), "Configuration")
    rows = [
        # Tokens break only before their own separators (L7 SW-LK7: the run
        # ended a line at a hyphen at 390 px, reading hyphenated).
        ("Run", r.token_html(f"{item.get('architecture')}/{item.get('run')}")),
        ("File", r.token_html(item.get("filename"))),
        *spec_rows,
        ("Holds", r.e(tc.CHECKPOINT_KINDS.get(kind or "", "not recorded"))),
        ("Validation", r.e(r.validation_value(r.validation_figure(item),
                                              read=r.record_read(item)))),
        ("Selection", r.e(r.selection_value(item))),
        (r.SETTING_LABELS["split_seed"], r.e(split)),
    ]
    if parents:
        rows.append(("Server list", r.e(listed_words(parents))))
    details = r.details_list(rows, of=item.get("key") or item.get("sha256"), titles=spec_titles)
    return (f'<div class="shep-ft-base"><div class="shep-ft-tags">{tags}{details}</div>'
            "</div>")


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


def _products(child: Mapping[str, Any], *, operator: bool,
              products: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    """An ended child's products, each HTML, as ``(best save, validation)``;
    or ``(why, None)``: one line said once for both (the table's one cell over
    the two columns, the narrow card's one row)."""
    status = child.get("status")
    if status in ("queued", "running", "stopping"):
        return "after it ends", None
    if status in ("cancelled", "refused") or not child.get("run_dir"):
        return "no run", None
    if not operator:
        return f"{tc.LOCK_ICON} not shown: reading the checkpoints is an operator control", None
    found = products.get(f"{child['run_dir']}|{status}") or {}
    if found.get("problem"):
        return f"not read: {r.e(found['problem'])}", None
    best = found.get("best") or []
    if not best:
        why = ("no saved file records its selection metric's figure" if found.get("recorded")
               else f"{r.selection_words({}).lower()} in its saved files")
        return r.e(why), None
    names = "".join(f"<div>{r.e(entry['words'])}</div>" for entry in best)
    by = f'<div class="shep-sub">{r.e(best[0]["selection"])}</div>'
    return f"{names}{by}", r.validation_cell(best[0]["validation"])


#: The job card's two product columns and their ⓘ: the table's heads, the
#: narrow card's keys.
BEST_SAVE, VALIDATION = "Best save", "Validation (Trainer)"
_PRODUCT_HELPS = {BEST_SAVE: ("finetune-best-save", BEST_SAVE),
                  VALIDATION: ("validation-trainer", VALIDATION)}


def _child_view(child: Mapping[str, Any], *, operator: bool,
                products: Mapping[str, Any]) -> Dict[str, Any]:
    """**One child as the card says it, whichever form draws it**: its task,
    status, run and products (`_products`), each HTML, escaped (a failure's
    first line through the one redaction of a printed line,
    `training_console.printed_scrub`, whoever calls). The run is its name part
    (`renderers.run_name`; the parent line and the task say the rest), its
    whole id in its title. The table's row and the narrow card's summary list
    are both drawn from it (`_row_html`, `_list_html`)."""
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
        run_html = (f'<div class="shep-ft-run">{r.ident_html(r.run_name(run) or whole, title=whole)}'
                    "</div>")
    else:
        run_html = ('<span class="shep-muted">not started yet</span>' if status == "queued" else
                    '<span class="shep-muted">never started</span>')
    best, validation = _products(child, operator=operator, products=products)
    return {"task": r.e(task), "status": f"{tc.job_status_html(status)}{why}", "run": run_html,
            "best": best, "validation": validation}


def _row_html(view: Mapping[str, Any]) -> str:
    """A child's table row (`_child_view`): Task | Status | Run | Best save |
    Validation (Trainer); a line said for both products is one cell over the
    two columns."""
    if view["validation"] is None:
        products = f'<td colspan="2" class="shep-muted">{view["best"]}</td>'
    else:
        products = f"<td>{view['best']}</td><td>{view['validation']}</td>"
    return (f"<tr><td>{view['task']}</td><td>{view['status']}</td><td>{view['run']}</td>"
            f"{products}</tr>")


def _list_html(view: Mapping[str, Any]) -> str:
    """**A child on a card narrower than the table's narrowest** (UI ledger
    V-20; the review of 6a2d52e, OPS-C-R6): the same child (`_child_view`) as
    the one summary list (`renderers.summary_list`, the skill's §4: one thing,
    several attributes), its task as its title and Status, Run, Best save and
    Validation (Trainer) as its rows, each product key with its table head's
    ⓘ; a line said for both products is one row named by both."""
    rows = [("Status", view["status"]), ("Run", view["run"])]
    if view["validation"] is None:
        rows.append((f"{BEST_SAVE}, {VALIDATION}",
                     f'<span class="shep-muted">{view["best"]}</span>'))
    else:
        rows += [(BEST_SAVE, view["best"]), (VALIDATION, view["validation"])]
    return (f'<section class="shep-ft-child"><h3 class="shep-ft-child-t">{view["task"]}</h3>'
            f"{r.summary_list(rows, helps=_PRODUCT_HELPS)}</section>")


def child_row_html(child: Mapping[str, Any], *, operator: bool, products: Mapping[str, Any]) -> str:
    """One child's table row (`_row_html` of `_child_view`)."""
    return _row_html(_child_view(child, operator=operator, products=products))


def job_card_html(status: Mapping[str, Any], *, operator: bool,
                  products: Optional[Mapping[str, Any]] = None) -> str:
    """The Fine-tune job card from the job runner's status (`get_status()`):
    its parent, the start's graph backbone (`freeze_line_html`) and its
    children, each drawn once from its one view (`_child_view`) in both forms:
    a row of the table, and a summary list (`_list_html`) that the card shows
    instead of the table while the table's own narrowest does not fit the
    card (the shell's fit, `style.FIT`: measured in the type drawn, never a
    fixed width or the window's); or, with no Fine-tune status, why (none
    since the server started, or the next base or continued run cleared it),
    in words of its own."""
    finetune = status.get("finetune")
    if not finetune:
        if (status.get("status") or "idle") == "idle":
            text = ("No Fine-tune job since the server started. Choose a base and its tasks, "
                    "then Start.")
        else:
            text = ("No Fine-tune job to show: the latest training job is a base or continued "
                    "run (Research › Training shows it). The next Fine-tune start shows here.")
        return f'<div class="shep-ft-card"><p class="shep-ft-empty">{r.e(text)}</p></div>'
    views = [_child_view(child, operator=operator, products=products or {})
             for child in finetune.get("children") or []]
    rows = "".join(_row_html(view) for view in views)
    lists = "".join(_list_html(view) for view in views)
    head = ("<thead><tr><th>Task</th><th>Status</th><th>Run</th>"
            f"<th>{BEST_SAVE}{help_button(*_PRODUCT_HELPS[BEST_SAVE])}</th>"
            f"<th>{VALIDATION}{help_button(*_PRODUCT_HELPS[VALIDATION])}"
            "</th></tr></thead>")
    link = ('<button type="button" class="shep-link" data-act="training">Charts on Research › '
            'Training ›</button>')
    return (f'<div class="shep-ft-card {FIT}">{parent_line_html(finetune)}'
            f'{freeze_line_html(finetune)}'
            f'<table class="shep-ft-table {FIT_WIDE}">{head}<tbody>{rows}</tbody></table>'
            f'<div class="shep-ft-children">{lists}</div>'
            f'<p class="shep-ft-links">{link}</p></div>')


# =============================================================================
# Start, Stop and the reason line
# =============================================================================
def refused_html(tasks: Any, values: Mapping[str, Any], touched: Any, *,
                 tried: bool = False) -> str:
    """**What Start would refuse of the form, said beside Start** (UI ledger
    NEW-training-finetune-6: an emptied task setting was said only in the
    line under the fields, out of view under the form's fold, while Start
    beside it said nothing): each value Start would refuse (`sent_settings`,
    the line's "Not valid: …"), by its label in the shared words ("Task loss
    weight must be set."), beside the ▲ of a condition to watch; else empty.
    Start stays enabled: its own check refuses it, in the same words.

    **Start's own refusal of them is this same line** (`tried`: Start was
    clicked and refused the form's values), said as the ✕ of a refusal ("✕
    Not started: Learning rate must be set."), never a second notice beside
    it (the T06 review: "▲ Start would refuse: …" and "✕ Not started: …", one
    fact twice in Start's row). The next input says it again as what Start
    would refuse. Escaped."""
    _settings, problems = sent_settings(tasks, values, touched)
    if not problems:
        return ""
    words = ("Not started: " if tried else "Start would refuse: ") + " ".join(problems)
    return ('<span class="shep-ft-refused">'
            + r.status_mark("error" if tried else "warn", words) + "</span>")


def _why(icon: str, text: str = "", *, phrase: str = "") -> str:
    """A reason in the shared form (`shep-reason`: its mark, hidden from a screen reader, in a
    column of its own; the sweep's L7 check, C5: a page's own form, so the walk read Start and
    Refresh as off with no reason for a read-only viewer): the page's words (escaped), then a
    dimmed control's phrase (R-WHY, `renderers.reason_phrase`), either or both."""
    words = " ".join(part for part in (r.e(text), phrase) if part)
    return f'<span class="shep-reason shep-ft-reason">{icon}<span>{words}</span></span>'


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


def child_job(status: Mapping[str, Any]) -> bool:
    """Whether the training job the status shows (running or stopping) is a
    child of the Fine-tune start it shows: the job the card's Stop reaches."""
    job = status.get("job_id")
    children = (status.get("finetune") or {}).get("children") or []
    return job is not None and any(child.get("job_id") == job for child in children)


def actions_view(status: Mapping[str, Any], *, access: Any,
                 gpu: Optional[Mapping[str, Any]], confirming: bool,
                 said_above: bool = False) -> Dict[str, Any]:
    """Start and Stop (interactive or not, and whether Stop is drawn) and
    their reason lines, as one decision for the tick and after every action,
    in the states the Training run bar uses (`training_console.run_bar_view`):
    D8, a training job running or stopping, the GPU slot held elsewhere.
    ``reason`` is beside Start; ``stop_reason`` is beside Stop, which is in
    the job panel under the card. **A dimmed control's reason is its cause's
    phrase or one line in the page's words** (R-WHY, SW-C11): a cause with a
    key of its own (D8, the GPU lock's notice, the GPU held elsewhere) by its
    phrase (`renderers.reason_phrase`, keyed to the one place its whole
    reason is drawn: the header chip, or the form's setup notice, which is
    not in Start's bar, so the phrase stands beside Start); a local cause (a
    job running or stopping) by one line led by the need
    (`renderers.need_words`: no control's name, no pointer). **Stop is drawn only while a child of this
    page's Fine-tune start runs or is stopping** (``stop_shown``, the run
    bar's rule, `training_console.stop_shown`; NEW-training-finetune-7: an
    idle panel showed a disabled Stop with no reason): drawn, it says D8's
    refusal while that refuses (the lock is never without its reason, the
    one gated-button rule) and that the job is ending while it stops; not
    drawn, nothing. Stop stops a running Fine-tune child (and so cancels the
    queued ones), never another page's job. An enabled Start has no reason
    beside it.

    **The GPU lock's reason is one line** (the text brief's §12): with its
    notice (`gpu`'s ``notice``), its phrase beside Start (R-WHY,
    `renderers.phrase_line`: "Needs a workspaces folder", which brings the
    notice into view), the notice itself said once, in the form's setup
    notice; a GPU holder, the GPU chip's phrase ("GPU in use", which opens
    the chip's reason)."""
    s = status.get("status") or "idle"
    child_running = running_child(status) is not None
    shown = tc.stop_shown(status, child_job)
    view = {"start": False, "stop": False, "stop_reason": "", "stop_shown": shown}
    if not access.allowed:
        view["reason"] = _why(tc.LOCK_ICON, phrase=operator_only(control=START_ID))
        view["stop_reason"] = _why(tc.LOCK_ICON, phrase=operator_only(control=STOP_ID)) if shown else ""
    elif s == "stopping":
        view["reason"] = _why(r.DISABLED_MARK, r.need_words("a training job is stopping"))
        if shown:
            view["stop_reason"] = _why(r.DISABLED_MARK, r.need_words("this Fine-tune job is stopping"))
    elif child_running:
        view["stop"] = not confirming
        view["reason"] = _why(r.DISABLED_MARK, r.need_words("this Fine-tune job is running"))
    elif s == "running":
        view["reason"] = _why(r.DISABLED_MARK, r.need_words(
            "a training job is running: Research › Training shows it"))
    elif gpu and gpu.get("notice"):
        view["reason"] = r.phrase_line(gpu["notice"], control=START_ID)
    elif gpu:
        view["reason"] = _why(r.DISABLED_MARK, phrase=r.reason_phrase("gpu_busy", control=START_ID))
    else:
        view["start"] = True
        view["reason"] = ""
    return view


def _card_view(mode: Optional[Mapping[str, Any]], cache: Optional[Mapping[str, Any]],
               request: Any, ws: Optional[Mapping[str, Any]] = None) -> Tuple[Any, ...]:
    """The card, Start, Stop, Start's and Stop's reason lines, the stop
    question's row, the form's setup notice (`setup_html`, from the page's
    workspace `ws`, and the GPU line read here, so Start's relation and the
    notice it points to are one decision) and the session's cache, read now
    for this request. The products are read for an operator only (they open
    checkpoints)."""
    cache = dict(cache or {})
    access = operator_access(request)
    status = training_manager.get_status()
    active = status.get("gpu", {}).get("job") if isinstance(status.get("gpu"), Mapping) else None
    busy = (status.get("status") or "idle") in ("running", "stopping")
    gpu = None if busy else tc.gpu_view(active, cache, access.allowed)
    asking = tc.confirming(status, dict(mode or {}))
    ws_notice = (ws or {}).get("notice")
    view = actions_view(status, access=access, gpu=gpu, confirming=asking,
                        said_above=_said_above(ws_notice, gpu))
    products: Dict[str, Any] = {}
    if access.allowed:
        products = products_of(status.get("finetune"), cache.get("products") or {})
        cache["products"] = products
    card = job_card_html(status, operator=access.allowed, products=products)
    # Start and Stop by the one gated-button rule (`chips.gated`): no lock
    # while D8 allows this request; disabled with the lock while it refuses
    # it, each beside its own reason line.
    return (card, gr.update(**gated(START_LABEL, access.allowed, interactive=view["start"])),
            gr.update(**gated(STOP_LABEL, access.allowed, interactive=view["stop"]),
                      visible=view["stop_shown"]),
            f'<div class="shep-ft-reasons">{view["reason"]}</div>',
            f'<div class="shep-ft-reasons">{view["stop_reason"]}</div>',
            gr.update(visible=asking), setup_html(ws_notice, gpu), cache)


#: How many outputs the card step answers (`_card_view`).
CARD_OUTPUTS = 8


def _refresh_card(mode, cache, request: gr.Request, ws=None):
    """After every action: the card, the actions and the setup notice as
    they are now."""
    return _card_view(mode, cache, request, ws)


def _poll(location, mode, cache, request: gr.Request, ws=None):
    """The shell's slow timer (``queue=False``), while Fine-tune is shown: the
    card, the actions and the setup notice; nothing while another page is
    shown."""
    if location != PAGE:
        return (gr.update(),) * CARD_OUTPUTS
    try:
        return _card_view(mode, cache, request, ws)
    except Exception:  # noqa: BLE001 - a tick that raises would stop the page's updates
        logger.exception("The Fine-tune job could not be read")
        return (gr.update(),) * CARD_OUTPUTS


# =============================================================================
# The handlers
# =============================================================================
def _workspace(name: Optional[str], listed: Optional[Mapping[str, Any]] = None
               ) -> Dict[str, Any]:
    """The page's workspace (`training_console.workspace_state`) with the
    workspaces folder's notice (``notice``: `read_workspaces`' own, None while
    the folder can be used), both from one read of the server's list."""
    listed = listed if listed is not None else read_workspaces()
    state = tc.workspace_state(name, listed)
    state["notice"] = listed.get("notice")
    return state


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
    state = _workspace(choice, listed)
    tasks, note = _tasks_update(state, chosen)
    return state, tc.dataset_choices_update(state), identity_html(state), tasks, note


def _on_dataset(name, chosen):
    """A dataset chosen here: its record read now, the tasks' readiness for
    it; the base list follows (`_on_finetune_bases`)."""
    state = _workspace(name)
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
        why = _why(tc.LOCK_ICON, "Not listed: listing the bases opens them.",
                   phrase=operator_only(control="ft_refresh"))
        return {}, empty, empty, f'<div class="shep-ft-reasons">{why}</div>', refresh
    name = (ws or {}).get("name")
    if not name:
        # With no dataset listed the setup notice says why, once, above
        # (`setup_html`); the empty Run and Checkpoint say their relation to it
        # (its phrase, `renderers.phrase_line`, as Test's do: the sweep's L7 check, C7: they
        # were two empty boxes that said nothing; the skill's §9), else ask
        # for a choice.
        notice = (ws or {}).get("notice")
        if notice:
            return {}, empty, empty, r.phrase_line(notice, control="ft_run"), refresh
        listed = (ws or {}).get("choices")
        note = bases_note_html({}, "Select a dataset first.") if listed else ""
        return {}, empty, empty, note, refresh
    from src.api.services.server_lists import list_checkpoints

    parents, problem = tc.checkpoint_items(list_checkpoints(name, learning_stage="base"))
    return (parents, *ml.parents_update(parents, current), bases_note_html(parents, problem),
            refresh)


def _on_base_chosen(key, parents):
    """A checkpoint chosen (or the list read again): the base's summary, the
    list's count among its Details (`base_summary_html`)."""
    return base_summary_html((parents or {}).get(key) if key else None, parents)


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
    (`set_by_html`) and what Start would refuse, beside it (`refused_html`)."""
    starts = task_starts(tasks)
    record = tc.touched_without(touched, TASK_OWN_FIELDS)
    kept = {name for name in ANY_TASKS if name in record}
    written = {name: value for name, value in starts.items() if name not in kept}
    fields = {**dict(zip(FINETUNE_FIELDS, values, strict=True)), **written}
    return [*(gr.update(value=written[name]) if name in written else gr.update()
              for name in TASK_FIELDS), set_by_html(tasks, fields, record),
            refused_html(tasks, fields, record)]


def _on_field_input(tasks: Any, touched: Any, *values: Any) -> Tuple[str, str]:
    """The line on what the operator set (`set_by_html`) and what Start would
    refuse, beside it (`refused_html`), said again on a field's input,
    chained after the field's touch listener
    (`training_console.record_touches`), as the base form's fields say
    theirs: the record it reads holds the field's key, and, run again alone
    for the last input ("always_last"), it reads the record as it is when it
    runs (once Reset or a task switch has landed, without what they
    dropped). Its values are the fields', in `FINETUNE_FIELDS` order. It
    writes only the two lines, never the record, which is the browser's; as
    the base form's line (`training_console._on_base_input`), a run read
    before Reset or a task switch landed can be answered after their line."""
    fields = dict(zip(FINETUNE_FIELDS, values, strict=True))
    return set_by_html(tasks, fields, touched), refused_html(tasks, fields, touched)


def _on_finetune_reset(tasks):
    """**Reset to reference** (`training_console.reset_outputs`, the base
    form's): the record empty, so with several tasks each child takes its own
    task's reference again; every field back where it starts for the chosen
    tasks (`start_values`); the line said for nothing set, and nothing beside
    Start that it would refuse."""
    starts = start_values(tasks)
    record, line, *fields = tc.reset_outputs(
        starts, set_by_html(tasks, dict(zip(FINETUNE_FIELDS, starts, strict=True)), []))
    return (record, line, "", *fields)


def _on_guards(*values):
    """A guard of a rule across settings changed (the scheduler, mixed
    precision): `training_console.guard_updates` of this page's settings
    (`FINETUNE_FIELDS`), as the base form's guards are answered."""
    return tc.guard_updates(FINETUNE_FIELDS, values)


def _on_hops_input(tasks, key, parents, touched, *values):
    """The hop field's input, chained after its touch listener
    (`training_console.record_touches`), so the record it reads holds its key
    (run again alone for the last input, the record as it is when it runs):
    its one listener besides it, the line on what the operator set
    (`set_by_html`, as every field's listener says it), the hop note and its
    pointer beside Start, for the list typed (`hops_note_html`,
    `hops_warning_html`).
    `values` are the fields', in `FINETUNE_FIELDS` order. It writes only
    them and what Start would refuse (`refused_html`), never the record,
    which is the browser's.
    One listener, not two: two input listeners of one field that each send
    its last value ("always_last") retriggered each other in the browser
    without end (seen on the page while it had two; Gradio runs a deferred
    listener again by dispatching its trigger again)."""
    item = (parents or {}).get(key) if key else None
    fields = dict(zip(FINETUNE_FIELDS, values, strict=True))
    hops = fields["num_neighbors"]
    return (set_by_html(tasks, fields, touched), refused_html(tasks, fields, touched),
            hops_note_html(tasks, item, touched, hops), hops_warning_html(tasks, item, touched, hops))


def _on_finetune_start(request: gr.Request, dataset, run, ws, parents, key, tasks, device,
                       run_name, touched, *values):
    """Start: an operator control (D8), checked first on this event's own
    request; then the selection it uses, as this click's own request carries
    the Dataset and Run selects (`dataset`, `run`; the Checkpoint is `key`),
    against what the session read for them (`shell.unread_choice`: the
    workspace `ws`, the chosen base's run in `parents`), so a choice whose
    reading has not landed (or failed) is refused, never started on the one
    before it (OPS-C-R2); then the Fine-tune start (`_start`) with the
    settings' fields' `values`, in `FINETUNE_FIELDS` order, and `touched`, the
    form's record of the fields the operator touched, as this click's own
    request carries it (`training_console.touched_record`).

    Returns the notice beside Start, a refusal with the shared mark of a
    refusal, escaped (`training_console.notice_html`), or nothing once started
    (the card shows the job); the Stop's notice cleared, which spoke of a job
    before this start; and the line on what Start would refuse of the form
    (`refused_html`): a refusal of the form's own values is said there, as
    Start's ✕, with no notice (the T06 review: the same sentence twice in
    Start's row); otherwise it is left as it is.
    """
    access = operator_access(request)
    if not access.allowed:
        return tc.notice_html(access.refusal, mark="error"), "", gr.update()
    item = (parents or {}).get(key) if key else None
    unread = unread_choice(("Dataset", dataset, (ws or {}).get("name")),
                           *([("Run", run, ml.item_run_key(item))] if item else []))
    if unread:
        return tc.notice_html(f"Not started: {unread}", mark="error"), "", gr.update()
    fields = dict(zip(FINETUNE_FIELDS, values, strict=True))
    said, of_the_form = _start(ws, parents, key, tasks, device, run_name, fields, touched)
    if of_the_form:
        return "", "", refused_html(tasks, fields, touched, tried=True)
    return tc.notice_html(said, mark="error"), "", gr.update()


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
           touched: Any) -> Tuple[str, bool]:
    """Start's work after D8, as a plain sentence ("" once started) and
    whether it is a refusal of the form's own values (`sent_settings`'
    problems, which the line beside Start says: `refused_html`): the
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
        return "Not started: select a dataset from the server's list first.", False
    if not row.get("trainable"):
        return (f"Not started: {row.get('name')} is not trainable: "
                f"{r.not_trainable_short(row)}.", False)
    item = (parents or {}).get(key) if key else None
    if not item:
        return "Not started: choose a base checkpoint from the list.", False
    if item.get("workspace") != row.get("name"):
        return (f"Not started: the list shows {item.get('workspace')}’s bases, not "
                f"{row.get('name')}’s. List them again (Refresh).", False)
    chosen = chosen_tasks(tasks)
    if not chosen:
        return "Not started: choose at least one task.", False
    settings, problems = sent_settings(chosen, values, touched)
    if problems:
        return "Not started: " + " ".join(problems), True
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
        why = str(redact(exc, workspaces_root()[0]))
        return (f"Not started: {why}{'' if why.endswith('.') else '.'} List the bases again "
                "(Refresh).", False)
    result = training_manager.start_finetune(
        FinetuneInputs(data_dir=str(directory), finetune_from=str(path), tasks=chosen,
                       expected_parent_sha256=item.get("sha256"), device=device or "auto",
                       run_name=(run_name or "").strip(), settings=settings),
        expected_dataset_fingerprint=fingerprint)
    if result.get("success"):
        return "", False
    error = result.get("error")
    # A setting the runner names first by its record name, by the field's
    # label instead, as the Training page's Branch says the runner's
    # refusal of its settings (`labelled_problem`).
    return tc.not_started(tc.labelled_problem(error) if result.get("invalid_settings")
                          else error), False


# =============================================================================
# Help
# =============================================================================
def task_setting_help(name: str) -> Tuple[str, str]:
    """A task setting's help, ``(title, text)``: its label, what it is (its
    field spec's description, as every form's help says it, the API's words
    and so its lead), where its field starts and what is sent, then each
    task's reference (the one fill, `run_plan.new_run_values`, of each task
    whose run takes it), a term row each, or one sentence for one task (the
    bubble's grammar, `help.help_lines`; the text brief's §11)."""
    references = [(TASK_NAMES[task], r.setting_words(name, values[name]))
                  for task in TASKS
                  for values in [new_run_values("finetune", task, {})] if name in values]
    if name in ANY_TASKS:
        where = ("With one task it starts at that task's reference.\n"
                 "Untouched with several tasks, each child takes its own.\n"
                 "A value you set is every child's until Reset to reference. Only a value you "
                 "edit is sent.")
    else:
        where = ("Offered with one task, starting at its reference.\n"
                 "With several tasks, each child takes its own. Only a value you edit is sent.")
    if len(references) > 1:
        listed = "Each task's reference, a provisional default from the source:\n" + "\n".join(
            f"- {task}: {words}" for task, words in references)
    else:
        listed = ("Reference: " + ", ".join(f"{task} {words}" for task, words in references)
                  + ", a provisional default from the source.")
    return (r.SETTING_LABELS[name], f"{by_name(name).description}\n\n{where}\n{listed}")


#: The fields' help, behind the shared ⓘ: ``{elem_id: (title, text)}``, each
#: title its field's label; each text a lead that answers what the field is,
#: then a few term rows (the text brief's §11, `help.TEXT_BUDGET`).
FIELD_HELP = {
    "ft_dataset": (
        "Dataset",
        "The workspace the children train on, from the server's list.\n\n"
        "- List: the same as on Research › Datasets\n"
        "- Starts at: the Training page's dataset, never changing it\n"
        "- Start sends: the dataset fingerprint shown\n"
        "- Refused: a start whose workspace changed since it was shown",
    ),
    "ft_run": (
        "Run",
        "The runs holding a base checkpoint, latest first.\n\n"
        "- Option: its name or start time, then its architecture and time\n"
        "- Directory: in the base's Details under the lists\n"
        "- Offered: bases only, never a task model (children never chain)\n"
        "- Never offered: a file without a model spec",
    ),
    "ft_checkpoint": (
        "Checkpoint",
        "The chosen run's base to fine-tune, newest first, its summary under the list.\n\n"
        "- Kinds: Latest save, Best (named by its recorded metric) or Top-k save\n"
        "- Also offered: a weights-only or early-stopped base\n"
        "- Start sends: its model fingerprint, the file read against it\n"
        "- Listing: opens each checkpoint, so it is an operator control",
    ),
    "ft_tasks": (
        "Tasks",
        "The task models to train from the base: one child run each, in this order.\n\n"
        "- Ready: the dataset can be trained\n"
        "- Gene, Patients-like-me: the gene task's fixture is of this build too\n"
        "- Before a start: the server checks parent, graph and fixtures again\n"
        "- In training: the process checks the fixture's files",
    ),
    **{f"ft_{name}": tc.setting_help(name, built_values(), FINETUNE_SOURCES)
       for name in COMMON_SHOWN + MORE_FIELDS},
    **{f"ft_{name}": task_setting_help(name) for name in TASK_FIELDS},
    "ft_device": (
        "Device",
        "Where the children train. They never fall back to the CPU on their own.\n\n"
        "- auto: the GPU (CUDA). With no CUDA device visible, Start is refused.\n"
        "- cpu: for development, recorded as asked.",
    ),
    "ft_run_name": (
        "Run name",
        "Optional. Each child's new folder is named by start time, this name and task:\n"
        "<workspace>/runs/<architecture>/<date-time>_<name>-<task>_<id>/",
    ),
}

#: The ⓘ inside the card.
TOPIC_HELP = {
    "finetune-best-save": (
        "Best save",
        "The child's saved file with the best recorded selection metric.\n\n"
        "- Metric: validation pooled MRR, higher is better, named under the file\n"
        "- Source: the Trainer's figure on val, not a Research › Test result\n"
        "- Ties: several saved files can share it\n"
        "- Shown: to the operator once the child has ended",
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
        # Each panel sized by what it holds (`shep-sized`, `PAGE_CSS`: a basis,
        # a share of the room left and a minimum, in rem, no `scale=`): the
        # form grows the most (its two columns, B27-1).
        with gr.Column(elem_id="shep-ft-form", elem_classes=["shep-panel", "shep-sized"],
                       min_width=0):
            plain_html('<div class="shep-panel-hd"><h2>Fine-tune a base</h2>'
                       '<span class="shep-panel-right">one child run per task</span></div>',
                       elem_id="ft_form_hd")
            with gr.Column(elem_id="ft_form_bd",
                           elem_classes=["shep-ft-bd", "shep-form-dense", "shep-stack"]):
                # The setup notice (`setup_html`): what empties the form, said
                # once over both columns, its next step a link; hidden while
                # empty.
                setup = plain_html("", elem_id="ft_setup")
                # Two columns where the form is wide enough for both (B27-1: one
                # column scrolled 183 px at 1850 x 960 with nothing expanded),
                # one under the other where it is not, by their own basis
                # (`PAGE_CSS`, in rem; no window size): the base (its dataset,
                # Run and Checkpoint, its summary), then its children (the
                # tasks, their settings, the device and the run name).
                with gr.Row(elem_classes=["shep-ft-cols"]):
                    with gr.Column(elem_id="ft_base_col", elem_classes=["shep-ft-col"]):
                        # The dataset and its identity beside it (under it on a narrow
                        # form), so the common settings below come into view sooner.
                        with gr.Row(elem_classes=["shep-ft-ds-row"]):
                            dataset = gr.Dropdown(label="Dataset", choices=[], value=None,
                                                  elem_id="ft_dataset")
                            identity = plain_html(identity_html({}), elem_id="ft_identity",
                                                  elem_classes=["shep-beside"])
                        # Run → Checkpoint, two cascading selects side by side (the
                        # skill's §4), then the list's Refresh; on a narrow form they
                        # wrap, one under the other.
                        with gr.Row(elem_classes=["shep-ft-parent-row", "shep-field-row"]):
                            run_select = gr.Dropdown(label="Run", choices=[], value=None,
                                                     elem_id="ft_run")
                            checkpoint = gr.Dropdown(label="Checkpoint", choices=[], value=None,
                                                     elem_id="ft_checkpoint")
                            refresh = gr.Button(REFRESH_LABEL, size="sm", scale=0, elem_id="ft_refresh")
                        bases_note = plain_html("", elem_id="ft_bases_note")
                        base = plain_html("", elem_id="ft_base")
                    with gr.Column(elem_id="ft_children_col", elem_classes=["shep-ft-col"]):
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
                        # A row of fields (`shep-fields`): Graph backbone a
                        # choice group, whole on its line or one option per line.
                        with gr.Row(elem_classes=["shep-ft-settings", "shep-fields"]):
                            for name in COMMON_SHOWN:
                                fields[name] = tc.setting_field(name, values[name], elem_id=f"ft_{name}")
                        freeze_note = plain_html(freeze_note_html(values["frozen_parts"]),
                                                 elem_id="ft_freeze_note")
                        # The hop list, offered for any task set, under the common
                        # settings with its note under it (`task_settings_view`): the
                        # start's sentence when the reference does not fit the base is
                        # in view without opening anything (2b); a sentence takes the
                        # row's width, never a column beside the field.
                        with gr.Row(visible=False, elem_id="ft_hops_row",
                                    elem_classes=["shep-ft-settings", "shep-ft-hops-row",
                                                  "shep-fields"]) as hops_row:
                            for name in ANY_TASKS:
                                fields[name] = tc.setting_field(name, values[name],
                                                                elem_id=f"ft_{name}", visible=False)
                            hops_note = plain_html("", elem_id="ft_hops_note")
                        with gr.Accordion("More settings", open=False, elem_id="ft_more"):
                            with gr.Row(elem_classes=["shep-ft-settings", "shep-ft-grid", "shep-field-grid"]):
                                for name in MORE_FIELDS:
                                    fields[name] = tc.setting_field(name, values[name],
                                                                    elem_id=f"ft_{name}",
                                                                    visible=name not in unread)
                        # A task's own settings (`task_settings_view`), with one task.
                        with gr.Accordion(task_settings_label(()), open=False, visible=False,
                                          elem_id="ft_task_more") as task_more:
                            with gr.Row(elem_classes=["shep-ft-settings", "shep-ft-grid", "shep-field-grid"]):
                                for name in TASK_OWN_FIELDS:
                                    fields[name] = tc.setting_field(name, values[name],
                                                                    elem_id=f"ft_{name}", visible=False)
                        several_note = plain_html("", elem_id="ft_several_note")
                        # What the operator set (decision S2), and Reset to reference:
                        # the base form's one line and Reset.
                        with gr.Row(elem_classes=["shep-ft-setby-row", "shep-beside-row"]):
                            set_by = plain_html(set_by_html((), {name: tc.form_value(values[name])
                                                                 for name in FINETUNE_FIELDS}, []),
                                                elem_id="ft_setby", elem_classes=["shep-beside"])
                            reset = gr.Button("Reset to reference", size="sm", scale=0,
                                              elem_id="ft_reset")
                        with gr.Row(elem_classes=["shep-ft-settings", "shep-fields"]):
                            device = tc.device_field("ft_device")
                            # A name's width by its role (R-WIDTH, the owner's D12 of
                            # 2026-11-24: the shared ``shep-w-name``, `style.py`), not
                            # the row's share.
                            run_name = gr.Textbox(label="Run name", value="", max_lines=1,
                                                  elem_id="ft_run_name",
                                                  elem_classes=["shep-w-name"])
            with gr.Column(elem_id="ft_start_ft", elem_classes=["shep-panel-ft"]):
                # Start and why it is off, a bar (the skill's §2 and §9): its
                # reason a one-line relation beside it, on Start's first line
                # (under it when the row is narrow); what Start would refuse
                # of a value and the runner's refusal, sentences, a note the
                # bar's width under it, never a column beside the button.
                with gr.Row(elem_classes=["shep-ft-start-row", "shep-bar"]):
                    start = gr.Button(START_LABEL, variant="primary", interactive=False,
                                      scale=0, elem_id="ft_start")
                    with gr.Column(elem_classes=["shep-ft-start-why", "shep-beside"], min_width=0):
                        reason = plain_html("", elem_id="ft_reason")
                        hops_warn = plain_html("", elem_id="ft_hops_warn")
                    refused = plain_html("", elem_id="ft_refused", elem_classes=["shep-bar-note"])
                    notice = plain_html("", elem_id="ft_notice", elem_classes=["shep-bar-note"])
        # ------------------------------------------------------------- the job
        with gr.Column(elem_id="shep-ft-job", elem_classes=["shep-panel", "shep-sized"],
                       min_width=0):
            plain_html('<div class="shep-panel-hd"><h2>Fine-tune job</h2>'
                       '<span class="shep-panel-right">the latest start</span></div>',
                       elem_id="ft_job_hd")
            with gr.Column(elem_id="ft_job_bd", elem_classes=["shep-ft-bd"]):
                card = plain_html(job_card_html({"status": "idle"}, operator=False),
                                  elem_id="ft_card", js_on_load=r.TRIGGER_JS)
                # Stop as Start: its reason beside it, a notice under it.
                with gr.Row(elem_classes=["shep-ft-stop-row", "shep-bar"]):
                    # Drawn only while a child of this page's start runs or
                    # stops (`actions_view`'s ``stop_shown``): none at first.
                    stop = gr.Button(STOP_LABEL, variant="secondary", interactive=False,
                                     scale=0, elem_id="ft_stop", visible=False)
                    stop_reason = plain_html("", elem_id="ft_stop_reason",
                                             elem_classes=["shep-beside"])
                    stop_notice = plain_html("", elem_id="ft_stop_notice",
                                             elem_classes=["shep-bar-note"])
                with gr.Row(visible=False, elem_id="ft_confirm_row") as confirm_row:
                    stop_text = plain_html("", elem_id="ft_stop_text")
                    stop_confirm = gr.Button("■ Stop training", variant="stop", scale=0,
                                             elem_id="ft_stop_confirm")
                    stop_keep = gr.Button("Keep training", scale=0, elem_id="ft_stop_keep")

    # =========================================================================
    # Wiring
    # =========================================================================
    card_step = {"fn": _refresh_card, "inputs": [mode, cache, ws],
                 "outputs": [card, start, stop, reason, stop_reason, confirm_row, setup, cache],
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

    # The page shown: the list and the tasks, then the card and the actions
    # (Start's lock and reason, D8's answer for this request, which needs no
    # base: a read-only viewer saw an unlocked Start with no reason until the
    # last step, verification of fcc3dea V-RB2-2), then the bases (gated),
    # and the task settings for them.
    ctx.on_page_open(PAGE, {
        "fn": _open, "inputs": [ws, ctx.training_workspace, tasks],
        "outputs": [ws, dataset, identity, tasks, tasks_note],
        "show_progress": "hidden"}).then(**card_step).then(**bases_step).then(
        **base_step).then(**task_view_step).then(**ctx.chip_refresh)
    # The shell's slow timer, a read-only poll without the queue.
    ctx.timer_slow.tick(_poll, inputs=[ctx.location, mode, cache, ws],
                        outputs=[card, start, stop, reason, stop_reason, confirm_row, setup, cache],
                        queue=False, show_progress="hidden")

    # Every choice is kept ("multiple", `shell.picked`): one made while the
    # one before it is read is read after it, never dropped.
    picked(dataset)(_on_dataset, inputs=[dataset, tasks],
                    outputs=[ws, identity, tasks, tasks_note],
                    show_progress="hidden", trigger_mode="multiple").then(**bases_step).then(
        **base_step).then(**ctx.chip_refresh)
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
                       outputs=[set_by, refused, hops_note, hops_warn], queue=False,
                       trigger_mode="always_last", show_progress="hidden")
        else:
            touch.then(_on_field_input, inputs=[tasks, touched, *field_list],
                       outputs=[set_by, refused], queue=False, trigger_mode="always_last",
                       show_progress="hidden")
    # The operator's own change of the tasks: a task's own settings leave the
    # record in the browser, then (chained after it) start again at their
    # reference, a hop list set kept; the line again.
    tc.record_drops(tasks, TASK_OWN_FIELDS, touched).then(
        _on_tasks_input, inputs=[tasks, touched, *field_list],
        outputs=[*(fields[name] for name in TASK_FIELDS), set_by, refused], queue=False,
        show_progress="hidden")
    # Reset to reference: the record empty, every field where it starts for
    # the tasks, the line; then the hop note for them.
    reset.click(_on_finetune_reset, inputs=[tasks], outputs=[touched, set_by, refused, *field_list],
                queue=False, show_progress="hidden").then(**task_view_step)
    # A setting a rule across settings reads only under its guard is shown while
    # the run would read it (decision S3), as on the base form.
    dependent, guards = tc.rule_fields(FINETUNE_FIELDS)
    gr.on(triggers=[fields[name].change for name in guards], fn=_on_guards,
          inputs=[fields[name] for name in guards],
          outputs=[fields[name] for name in dependent], queue=False, show_progress="hidden")

    # Start carries the Dataset and Run selects as the reader sees them, which
    # it checks against what the session read for them (`_on_finetune_start`).
    start.click(_on_finetune_start,
                inputs=[dataset, run_select, ws, parents, checkpoint, tasks, device, run_name,
                        touched, *field_list],
                outputs=[notice, stop_notice, refused], show_progress="hidden").then(
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
        return ctx.go("training") if r.event_act(evt) == "training" else ctx.no_nav()

    card.click(_on_card_click, inputs=None, outputs=ctx.nav_outputs, show_progress="hidden")


#: This page's element id (`shell.page_elem_id`), the scope of `PAGE_CSS`.
_PAGE = "#" + page_elem_id(PAGE)

#: The job card's Run column at its narrowest: nine characters of the code
#: font (14 px), the widest word every child's run name holds, its task's
#: ("patients-": the runner names a child's run after its task), so no
#: child's own task word is cut (it was at seven, "patien…", whatever the
#: card's width). Past it the column takes its share of the card's room; a
#: name's token wider than that share is cut, never widening the table (the
#: review of a6aae90).
RUN_FLOOR = "76px"

#: This page's style sheet (listed in `src/webui/app.py`'s ``PAGE_CSS``).
#: Nothing under 14 px; status colours only through the ``--shep-status-*``
#: tokens.
PAGE_CSS = f"""
{_PAGE} .shep-ft-bd {{ padding: var(--shep-space-2) var(--shep-inset) var(--shep-space-3) !important; }}
{_PAGE} #ft_job_bd {{ gap: var(--shep-space-2) !important; }}
/* The two panels sized by what they hold (`shep-sized`, the shared rule; no `scale=`): the form's
   basis and the larger share of the room left, the job panel's basis; each never under its
   minimum (the layout rule's measure, `src/webui/shell.py`): the form's base summary list keeps
   its key column whole from 30rem; from 27.5rem the job's card holds the ended children's table
   whole in the dashboard's type (else the card shows their lists: the shell's fit). The job panel
   takes no share of the room left: the form does (L7 SW-LK2: a quarter of it to a job panel of
   one sentence left the tasks one a line and Run name under the fold at 1550 x 830). */
{_PAGE} #shep-ft-form {{ --shep-panel-basis: 32rem; --shep-panel-grow: 3; --shep-panel-min: 30rem; }}
{_PAGE} #shep-ft-job {{ --shep-panel-basis: 24rem; --shep-panel-grow: 0; --shep-panel-min: 27.5rem; }}
/* The setup notice over both columns (`setup_html`), hidden while it says nothing. */
{_PAGE} #ft_setup:not(:has(.shep-nb)) {{ display: none !important; }}
/* The form's two columns (B27-1): side by side while the form holds both at their basis, else
   one under the other (they wrap: no window size decides it), each its fields with the body's
   gap. Each basis is what its column holds on a line: the base's selects and summary list, and
   the children's widest row, the three tasks or Epochs, Seed and Graph backbone, each whole
   (37.5rem: the tasks need 37.1rem on one line; at 30rem they stood one a line, L7 SW-LK2). */
{_PAGE} .shep-ft-cols {{ flex-wrap: wrap !important; align-items: flex-start !important; gap: var(--shep-space-2) var(--shep-space-5) !important; }}
{_PAGE} .shep-ft-cols > .shep-ft-col {{ flex: 1 1 24rem !important; min-width: 0 !important; gap: var(--shep-space-2) !important; }}
{_PAGE} .shep-ft-cols > #ft_children_col {{ flex-basis: 37.5rem !important; }}
/* Run and Checkpoint side by side where the column holds both, else one under the other; their
   group keeps a select's floor beside Refresh, so Refresh stays on the last box's line. */
{_PAGE} .shep-ft-parent-row > .form {{ flex: 1 1 14rem !important; min-width: min(14rem, 100%) !important; }}
/* A read-only viewer's line on the list (it is not listed, Refresh is locked) ends under Refresh,
   the row's last control, as a refusal's reason stands under what it turns off (the sweep's L7
   check, C5: from the row's start it ended 330 px left of Refresh at 1366 x 768). */
{_PAGE} #ft_bases_note .shep-ft-reasons {{ display: flex; justify-content: flex-end; }}
{_PAGE} #ft_run, {_PAGE} #ft_checkpoint {{ flex: 1 1 18rem !important; min-width: 0 !important; }}
{_PAGE} .shep-ft-ds {{ font-size: 15px; display: flex; flex-wrap: wrap; gap: 4px 6px; align-items: center; }}
{_PAGE} .shep-ft-ds-row {{ align-items: flex-end !important; flex-wrap: wrap !important; gap: var(--shep-space-2) var(--shep-space-3) !important; }}
{_PAGE} .shep-ft-ds-row > .form:has(#ft_dataset) {{ flex: 1 1 17.5rem !important; min-width: 0 !important; }}
/* The dataset's identity beside its select only while it is one line (the shared `shep-beside`). */
{_PAGE} #ft_identity {{ padding-bottom: 6px; }}
/* The chosen base's summary (`base_summary_html`; D3, D8 and R2): its stage's facts as tags on one
   line, a group's gap between them and no " · " chain, then its one "Details ›" closing the line
   (one unit, moved to the next line whole where the line has no room for it); opened, its one
   summary list goes under the line, on the line's edge, with no box of its own (the skill's §2). */
{_PAGE} .shep-ft-tags {{ margin: 0; font-size: 15px; line-height: var(--shep-lh-prose); }}
{_PAGE} .shep-ft-tag {{ display: inline-block; margin-inline-end: var(--shep-space-5); }}
{_PAGE} .shep-ft-tag > .shep-st + .shep-st {{ margin-inline-start: var(--shep-space-3); }}
{_PAGE} .shep-ft-tags > .shep-more {{ display: inline; }}
{_PAGE} .shep-ft-tags > .shep-more > summary {{ display: inline-block; white-space: nowrap; min-height: 0; line-height: inherit; }}
{_PAGE} .shep-ft-tags > .shep-more::details-content {{ display: none; }}
{_PAGE} .shep-ft-tags > .shep-more[open]::details-content {{ display: block; }}
{_PAGE} .shep-ft-tags > .shep-more[open] > summary + * {{ margin-top: var(--shep-space-item); }}
{_PAGE} .shep-ft-base .shep-kv {{ font-size: 14px; gap: 2px 14px; }}
/* Lines that say nothing take no room (and no gap of their column). */
{_PAGE} #ft_identity:not(:has(.shep-ft-ds, p)), {_PAGE} #ft_bases_note:not(:has(p, .shep-ft-reason)), {_PAGE} #ft_base:not(:has(.shep-ft-base)), {_PAGE} #ft_tasks_note:not(:has(p)), {_PAGE} #ft_freeze_note:not(:has(p)), {_PAGE} #ft_hops_note:not(:has(p)), {_PAGE} #ft_several_note:not(:has(p)) {{ display: none !important; }}
/* The line on what the operator set and Reset to reference: the shared row of words beside a
   control only while they are one line (`shep-beside-row`; its text the shared `.shep-setby`). */
{_PAGE} .shep-ft-why {{ margin: 0; font-size: 14px; line-height: var(--shep-lh-prose); color: var(--shep-text-2); }}
{_PAGE} .shep-ft-why > .shep-ft-each {{ margin-top: var(--shep-space-1); font-size: 14px; gap: var(--shep-space-1) 12px; }}
/* The hop list, its note under it the row's width (`hops_note_html`): a sentence never stands as a
   column beside the field. */
{_PAGE} .shep-ft-hops-row > .form:has(#ft_num_neighbors) {{ flex: 0 1 20rem !important; }}
{_PAGE} #ft_hops_note {{ flex: 1 1 100% !important; min-width: 0 !important; }}
/* The settings under an accordion in columns of one width (the shared grid of fields,
   ``shep-field-grid``), as the base form's grid lays them. */
{_PAGE} .shep-ft-grid {{ --shep-field-min: 14rem; }}
{_PAGE} .shep-ft-hops {{ margin: 0; font-size: 14px; overflow-wrap: anywhere; }}
/* Start and Stop (`shep-bar`): the reason beside each on the button's first line (baselines
   aligned, so the button stands at the bar's top whatever follows), under it when the row cannot
   hold both; a note (what Start would refuse, a refusal) the bar's width under them. Each button is
   its words' width by the shared rule (`style.py`; a bar with a reason slot beside its action is never
   spanned, `shell.py`'s ``spans``), so it keeps one width whether or not its reason shows. */
{_PAGE} .shep-ft-start-row, {_PAGE} .shep-ft-stop-row {{ align-items: baseline !important; }}
{_PAGE} .shep-ft-start-row > button, {_PAGE} .shep-ft-stop-row > button {{ padding: 0 14px !important; height: 40px; }}
/* Each reason beside its button only while it is one line (the shared `shep-beside`). */
{_PAGE} .shep-ft-start-why, {_PAGE} #ft_stop_reason {{ gap: var(--shep-space-1) !important; }}
{_PAGE} .shep-ft-start-why:not(:has(.shep-ft-reason, .shep-nb-rel, .shep-ft-hops-warn)) {{ display: none !important; }}
{_PAGE} #ft_hops_warn:not(:has(.shep-ft-hops-warn)), {_PAGE} #ft_reason:not(:has(.shep-ft-reason, .shep-nb-rel)), {_PAGE} #ft_refused:not(:has(.shep-ft-refused)) {{ display: none !important; }}
{_PAGE} #ft_hops_warn {{ font-size: 14px; line-height: var(--shep-lh-prose); }}
{_PAGE} #ft_refused {{ font-size: 15px; line-height: var(--shep-lh-prose); }}
{_PAGE} .shep-ft-reason > svg {{ flex: none; margin-top: 3px; }}
{_PAGE} .shep-ft-reason > .shep-nb-mark {{ flex: none; }}
/* Start's and Stop's notices, each a note under its button: a refusal carries the shared mark of
   a refusal (`training_console.notice_html`); an empty one takes no room. */
{_PAGE} #ft_notice, {_PAGE} #ft_stop_notice {{ font-size: 15px; overflow-wrap: anywhere; }}
{_PAGE} #ft_notice:not(:has(.shep-notice-t)), {_PAGE} #ft_stop_notice:not(:has(.shep-notice-t)) {{ display: none !important; }}
{_PAGE} .shep-ft-card {{ display: flex; flex-direction: column; gap: 10px; font-size: 15px; }}
{_PAGE} .shep-ft-empty {{ margin: 0; color: var(--shep-text-2); }}
{_PAGE} .shep-ft-from {{ margin: 0; overflow-wrap: anywhere; }}
{_PAGE} .shep-ft-k {{ font-size: var(--shep-label-size); font-weight: var(--shep-label-weight); color: var(--shep-label-color); margin-right: 8px; }}
/* The card's table fits its card (T08-flow-6: kept on one line, its heads and its run name made
   it wider than the card, its last column cut at the panel's edge): a head of a word or two wraps
   at its spaces, a run name after its own hyphens (never inside a token, as Compare's run ids);
   a number never breaks. The run's narrowest is a floor (`RUN_FLOOR`), not its longest token:
   past it the run takes its share of the card, up to its whole name, and a token wider than that
   share (a name with no hyphen, the whole id of a directory not named by its start time) ends in
   the shared identifier's ellipsis, its whole id in its title, never widening the table (the
   review of a6aae90). */
{_PAGE} .shep-ft-table {{ width: 100%; border-collapse: collapse; font-size: 15px; }}
{_PAGE} .shep-ft-table th {{ text-align: left; font-size: 14px; font-weight: 600; color: var(--shep-text-2); padding: 6px; border-bottom: 1px solid var(--shep-border); vertical-align: bottom; }}
{_PAGE} .shep-ft-table td {{ padding: 8px 6px; border-bottom: 1px solid var(--shep-border); vertical-align: top; min-width: 0; }}
{_PAGE} .shep-ft-table :is(th, td):first-child {{ padding-left: 0; }}
{_PAGE} .shep-ft-table :is(th, td):last-child {{ padding-right: 0; }}
/* A card the table's own narrowest does not fit (the shell's fit marks it `NARROW_ATTR`; UI
   ledger V-20, OPS-C-R6; OPS-FIX-R2: measured in the type drawn, so another host's wider text
   never leaves the table past the card, as the fixed 404 px did: 412 px in a 406 px card) shows
   each child as its summary list instead (`_list_html`, the same child's view as its row): its
   task as its title, its rows the table's columns, a line between children as between the
   table's rows. Keyed to the card's own content, never the window's. (At the owner's sizes the
   ended Disease and Patients-like-me children's table is 396.8 px at its narrowest in the
   dashboard's type and the card 406 px side by side: the table; 324 px at 390 x 844: the list.) */
{_PAGE} .shep-ft-children {{ display: none; }}
{_PAGE} .shep-ft-card[{NARROW_ATTR}] .shep-ft-table {{ display: none; }}
{_PAGE} .shep-ft-card[{NARROW_ATTR}] .shep-ft-children {{ display: flex; flex-direction: column; }}
{_PAGE} .shep-ft-child {{ padding: 8px 0; border-bottom: 1px solid var(--shep-border); }}
{_PAGE} .shep-ft-child:first-child {{ border-top: 1px solid var(--shep-border); }}
{_PAGE} .shep-ft-child-t {{ margin: 0 0 6px; font-size: 15px; font-weight: 600; color: var(--shep-text); }}
{_PAGE} .shep-ft-run {{ display: grid; grid-template-columns: minmax({RUN_FLOOR}, max-content); }}
{_PAGE} .shep-ft-run > code.shep-ident {{ white-space: normal; max-width: none; }}
{_PAGE} .shep-ft-links {{ margin: 0; }}
{_PAGE} .shep-link {{ border: 0; background: none; padding: 0; color: var(--shep-accent-text); font-size: 15px; cursor: pointer; font-weight: 400; }}
{_PAGE} .shep-link:hover {{ text-decoration: underline; }}
{_PAGE} #ft_confirm_row {{ align-items: center !important; flex-wrap: wrap !important; gap: 8px 14px !important; }}
{_PAGE} #ft_stop_text {{ flex: 1 1 320px !important; }}
{_PAGE} .shep-rb-confirm-q {{ background: var(--shep-red-50); border: 1px solid var(--shep-red-200); border-radius: 8px; padding: 8px 12px; font-size: 15px; }}
{_PAGE} .shep-html svg {{ display: inline-block; vertical-align: -2px; }}
{SIDE} {{
  {_PAGE} > .column {{ height: 100%; min-height: 0 !important; }}
  {_PAGE} #ft_form_bd, {_PAGE} #ft_job_bd {{ flex: 1 1 auto !important; min-height: 0; overflow: auto !important; flex-wrap: nowrap !important; }}
  {_PAGE} #ft_form_bd > *, {_PAGE} #ft_job_bd > * {{ flex-shrink: 0 !important; }}
}}
"""


__all__ = ["PAGE_CSS", "create_finetune_page", "job_card_html", "task_choices"]
