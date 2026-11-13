"""
Research › Prepare › Training (M5-2, UI-C)
==========================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries" (Training: the run
bar, the parameters for the next run, Continue, the run view and its charts).
Built by `create_training_tab` under the page contract (`src/webui/shell.py`).

Module: src/webui/components/training_console.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/webui/components/training_console.py

**The run bar** (full width, never scrolls): the Dataset select over the
server's workspace list (`workspaces.read_workspaces`, the shared option label
`renderers.workspace_choices`); choosing one reads its manifest and record and
the dataset chip shows them (`renderers.dataset_identity_html`) with a
"Details ›" popover (the full fingerprint with Copy, what the manifest binds,
the split, the sources, the path relative to the workspace root, "All
datasets ›"). While a training job runs the select is disabled and shows that
job's dataset ("in use by the running job"). Then the status pill (idle,
starting, running with its epoch, then its batch and time as the pill's
details, stopping, completed, stopped, failed in red; never green for a failed
or stopped run), the run name, one
reason line saying why an action is disabled (D8's short words for a read-only
request; the GPU job holding the slot, from the job runner and the GPU lock's
holder record, with "Open Test ›" for a test run), and Start, Continue… and
Stop, each labelled by the one gated-button rule (`chips.gated`: no lock while
the operator may use it; disabled, with the lock, beside D8's reason while a
read-only request is refused). Stop asks first, in the code's words ("Stop
training" / "Keep training"). A failed run's error banner spans the page: under its headline the
last error line the run printed, beside it where the run's log is (only when it
has one), then its output tail.

**Start sends the full dataset fingerprint it showed** (the session's
``gr.State``, never the client's); the job runner reads the manifest again and
refuses a different one before anything is written or started
(`TrainingManager.start_training`'s ``expected_dataset_fingerprint``, the one
start path the API route uses too).

**Parameters "for the next base run"** (ops-wiring round, decisions S1, S2,
S3, S5): what a new base takes (`run_plan.new_run_settings("base")`, in the
presentation order `BASE_FIELDS`), each field the one of its setting
(`setting_field`) and starting at the value the run takes without it (the
one fill, `run_plan.new_run_values`: the recipe's, else this build's
default). Epochs, Learning rate, Batch size, Seed, Split seed and Device are
shown; the rest are under "More training settings", where a setting a rule
across settings reads only under its guard shows only while it applies
(warmup with the cosine scheduler, the minimum ratio with a scheduler, the
dtype with mixed precision on: `training_fields.not_read_by_rules`). Beside
it, "Model": the architecture, a fact read from the encoder declarations (a
choice once two or more are declared), and the reference base spec's input,
encoder and LP head fields generated from their dataclasses
(`spec_form_fields`: numbers and switches to edit, texts as facts). Each
field's help is behind the shared ⓘ (`FIELD_HELP`: its field spec's
description, the value the run takes without it and its source); a format,
a unit or "off" is said under the field (`FIELD_FORMATS`). **Explicit is what
the operator touched** (decision S2): each field's input records it in the
form's record, held in the browser and carried by Start's own request
(`touched_record`), Reset to reference clears it, and the line under the
form says what is set ("Set by you: Epochs 5 (reference 200)", the first
few named and every one under its Details).
Start sends ``learning_stage="base"``, the settings touched that the run
reads, whatever their value, and the model spec whole when a field of it was
touched (`base_start_settings`); each value is checked first by its field
spec, the settings by the rules across settings the start applies, the spec
by its own validation and the hop list against the encoder it builds, a
problem named by its label. A base's task models are trained on
Research › Fine-tune (`src/webui/components/finetune_page.py`).

**Continue…** opens a drawer over the parameters column, "Continue or branch
a run", in numbered steps (the skill's §4): its operation, **Continue**
(resume a paused run or extend a finished one, every setting its own) or
**Branch** (a new run from the checkpoint's weights with a new optimizer;
ops-wiring round, decisions B1-B4); 1. Run and 2. Checkpoint, the parent
list being the run bar's workspace's model records only (bases and task
models; a file without a model spec is never continued or branched from the
dashboard, decision 1), from `server_lists.list_checkpoints` (an operator
control since it opens every checkpoint) in the shared Run → Checkpoint
selects (`model_list`'s, as Fine-tune's parent: short options whose values
are the listing's keys), or a typed path that locates one of them; the
chosen checkpoint's details (its kind by `renderers.model_kind_text`, a
frozen-backbone child named so); 3. Check, which shows what the operation
does (the shared run plan, through the job runner's one preview of each,
`preview_continuation` or `preview_branch`) and compares the parent's
recorded dataset fingerprint with the workspace's current one; 4. Start in
the drawer's footer, Continue Training or Start branch and Cancel beside
the start's reason, which says what the check refused and what to do
(`continue_refusal`; the skill's §9), while the run bar's Start is disabled
("close Continue… to start a new run"). Extending a parent whose rate ended
at 0 asks its Extra-epoch rate (the preview's `zero_rate_groups`), sent as
`extend_lr`. **A branch's fields are the settings its parent's stage and
task may change** (the preview's `changeable`), each the one field of its
setting (`setting_field`), shown after a check at the parent's value; only
the fields the operator edits are sent (`branch_sent`, decision S2), none a
same-settings branch ("no setting changed: a new run from these weights
with a new optimizer", decision B1), and the start is the job runner's one
(`start_training` with `branch_from`). A listed parent is resolved by the
server from its list again (never a client path) and its bytes compared
with the digest listed; a Refresh that lists another file under the chosen
option shows it and clears the check (`_on_refreshed`), so the start sends
the digest the details show and the preview read, and the runner and the
training process still decide at start.

**The run view** (three rows: its header, the charts, the resources line)
shows the run chosen in its **Viewing** select: the run bar's workspace's
runs, latest first, each named by the one rule of a run's option
(`model_list.run_options`: short, name first) and valued by the run
resolver's full id (`resolve_run`). A choice holds while the run bar shows
the workspace it was made for and no job has started since (`_chosen_run`:
another workspace shown in between does not end it, and back on its own it
is shown again); without one, the run the job runner started last (a
Fine-tune child included), or, before any since the server started, the
workspace's latest run (`viewed_run`). **The training job and
the run viewed are kept apart**: the run bar, its status pill, its reason
line, its actions and a failed job's banner are always the job's; only the
header, the chart token and Export CSV follow the run viewed, and the header
says how it stands to the job (the running job, the last job, a past run
while another runs, the latest run: `view_relation`). Beside it the header
says which dataset fingerprint the run used (from its checkpoint: operators
only; a read-only request sees "not shown"), the run's state in a few words
(operators: where it can be continued from; the job's state only of the
job's own run) and its Details; then the shared chart component
(`src/webui/components/charts.py`: Loss, the Trainer's validation figure and
its companions, chosen by the keys the run's log holds (a base's LP
accuracy, ROC-AUC and AP; a task model's pooled MRR and pooled top-k; titles
that say no figure is charted for a log that holds neither), and the
learning rate on a log axis, one figure sharing the epoch axis, 2 × 2 in
the wide layout and, in the narrow one, from about 1000 px of its own width
(`charts.TWO_COLUMNS_FROM`); the data from the run's metrics log through the
component's server functions), each plot at least `PLOT_MIN_HEIGHT` px, the
plots' body scrolling under the toolbar's own row when they do not fit. An
empty region says what will appear and the next step (`NO_RUN_YET`, or who
takes it to a read-only request, `NO_RUN_YET_READ_ONLY`; `WAITING_FOR_EPOCH`,
`NO_EPOCH_LOGGED`, …), and an empty chart its own line in place of its plot
(`WAITING_FOR_VALIDATION` while the job's run logs, `NO_VALIDATION_LOGGED`
once a run ended without one).

**Per browser session, no module state:** the selection, the run viewed's
choice, the drawer, the last values sent and the banners a session dismissed
are ``gr.State``. The page is
refreshed when it is shown (``ctx.on_page_open``) and by the shell's timers
(outside ``gr.Tabs``: ``ctx.timer_fast`` for the run bar and the run view,
``ctx.timer_slow`` for the resources line), both ``queue=False``, doing no work
while another page is shown (``ctx.location``).
**The run bar and the run view are one projection of the session's
selection, written only by a queued render** (`render`, the pieces of
`VIEW_KEYS`: the run bar's workspace, ``ws`` and ``ctx.training_workspace``;
the select's value, the chip, the run viewed's header, its Viewing select
and its chart token, Start, Continue…, Stop and their reason, the drawer's
line): every listener that writes them, or the selection they are drawn
from, is in one queued group of one (``ctx.training_workspace_writes``),
first asked first run, each from what the one before it wrote: the page open,
then its render; the select; the Viewing select's choice (``viewing``, which
only it writes), then its render; Datasets' "Use for training ›" and the
change of ``ctx.training_workspace`` it makes; the render after every action,
and the render the fast poll hands a change over to. So a workspace or a run
chosen while a render waits is applied after it, and no piece drawn from an
older selection is applied after one drawn from a newer. **The fast poll writes only the
status pill** (`POLL_KEYS`, the job runner's state alone, on every tick, as
the renders do), the header's chips and a hand-off: it renders the page as it
is now and compares it with
what the session was last sent; when a piece differs (new epochs, a job that
started or ended, the GPU slot, a running job's workspace to follow), it gives
a hidden component a new value (`CHANGED_ID`), whose ``.change`` runs the
queued render, which reads the session's selection and the job again at its
own turn and sends what changed (everything again every `FULL_SEND_TICKS`
hand-offs). It writes no cache: what the session was sent, and the reads a
render keeps, have the renders as their one writer (the poll keeps its own
record, its reads throttled by time, `POLL_READS`, and the GPU slot it saw,
in a State only it writes).

**D8:** Start, Continue…, Continue Training and Start branch, Check
checkpoint, the parent list and Stop (its confirmation reads the run's
checkpoint) call the Gradio helper
first, on their own request, then the shared service in process; each is
followed by the chips' read. A read-only request sees every action disabled
with D8's reason as text, and no checkpoint path, snapshot or recorded
fingerprint. The metrics log and the workspaces' JSON records are not
operator-only. Stop's question also names the Fine-tune children queued after
the running one (`stop_question`); the Fine-tune page's Stop runs the same
stop logic (`ask_stop`, `confirm_stop`) behind its own D8-checked handlers.

Architecture Note:
    The training service is the job runner (`src.api.services.training_manager`,
    allowed: webui -> api); nothing here imports `src.training`.

Version: 2.0.0
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import (Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple,
                    get_args)

import gradio as gr

from src.api.services.training_manager import training_manager
from src.config.recipe import FINETUNE_TASKS, branch_settings, reference_settings
from src.config.runtime_presets import load_runtime_settings
from src.config.training_fields import (CROSS_FIELD_RULES, bound_words, by_name,
                                        cross_field_problems, names as field_names,
                                        not_read_by_rules, value_problem)
from src.evaluation.caveats import TASK_NAMES
from src.models.bundle.spec import (ENCODER_DECLARATIONS, SpecError, declared_fields,
                                    hops_problem, reference_spec)
from src.utils.run_plan import new_run_values
from src.utils.gpu_lock import lock_state
from src.webui.components import charts
from src.webui.components import renderers as r
from src.webui.components.chips import (busy_words, chips_on_change, gated, gpu_chip_from,
                                         operator_only)
from src.webui.components.help import help_button
from src.webui.components.workspaces import (
    manifest_facts,
    read_workspaces,
    redact,
    shown_path,
    workspaces_root,
)
from src.webui.operator_access import operator_access

logger = logging.getLogger(__name__)

#: This page's id in the shell (`src.webui.shell.TRAINING`, which a test
#: checks it equals). Not imported from the shell at the top:
#: ``src/webui/components/__init__.py`` imports this module while the shell
#: is still importing its own components, so the shell is imported where it
#: is used (`_shell`).
TRAINING = "training"


def _shell():
    from src.webui import shell

    return shell


def _model_list():
    """The shared model list (`src.webui.components.model_list`: the Run →
    Checkpoint selects' options), imported where it is used, as the shell is
    (`_shell`): it imports the shell."""
    from src.webui.components import model_list

    return model_list


def plain_html(value: Any = "", **kwargs: Any) -> Any:
    """The shell's ``plain_html`` (no prose CSS, no default click script)."""
    return _shell().plain_html(value, **kwargs)


TRAINING_CONFIG_FILE = Path(".shepherd_training_config.json")
DISPLAY_PREFIX = "SHEPHERD-Advanced/"

#: Every this many renders the fast poll hands over, the run bar and the run
#: view are sent whole, whatever the session's record of what it was last sent
#: says (each other one sends what changed; the render after an action, whole).
FULL_SEND_TICKS = 8
#: The GPU lock is read at most this often per session (seconds).
LOCK_READ_SECONDS = 5.0
#: The run bar's workspace's runs are listed again at most this often per
#: session (seconds), for the run view's Viewing select and the run it follows.
LATEST_READ_SECONDS = 10.0
#: Lines of a failed run's output the banner shows.
BANNER_TAIL_LINES = 12
#: The dataset chip's "Details ›" popover is this wide (px, at most the window
#: less 48): "What the manifest binds" lays four columns of 12-hex digests and
#: their labels in it (M5-3: at 460 the second column's digests ended 2 px
#: past the card's inner edge).
DETAILS_POPOVER_WIDTH = 480
#: The dataset chip's stacking level: above the parameter fields' labels
#: (Gradio's 40), which its Details popover opens over in a narrow window
#: (M5-3: at 1280 x 720 "Learning Rate" and "Batch Size" were drawn on it).
CHIP_Z_INDEX = 60


def _save_training_config(data_dir: str, request: gr.Request) -> str:
    """Save training path config: an operator control (D8), checked on this
    event's own request before anything is written.

    **Unused since M5-2**: its only caller, the Data Paths section's Save Path,
    is gone (the workspace is the run bar's select). Kept, gated, for the
    owner's question "Save as the default dataset", which is not built."""
    access = operator_access(request)
    if not access.allowed:
        return access.refusal
    stripped = data_dir.strip()
    if stripped.startswith(DISPLAY_PREFIX):
        stripped = stripped[len(DISPLAY_PREFIX):]
    try:
        with open(TRAINING_CONFIG_FILE, "w") as f:
            json.dump({"data_dir": stripped}, f, indent=2)
        return "Training path config saved."
    except Exception as e:
        return f"Failed to save: {e}"


# =============================================================================
# A new run's settings in a form (decisions S1, S2, S3, S5)
# =============================================================================
#: **The base form's settings, in its order** (presentation only): what a new
#: base takes (`run_plan.new_run_settings("base")`) but its model spec, whose
#: fields the Model accordion offers (`spec_form_fields`); a test holds the two
#: equal. The first `BASE_SHOWN`, with Device, are the common settings, always
#: shown; the rest are under "More training settings", each shown only while
#: the run would read it (`training_fields.not_read_by_rules`).
BASE_FIELDS: Tuple[str, ...] = (
    "num_epochs", "learning_rate", "batch_size", "seed", "split_seed",
    "num_neighbors", "scheduler_type", "warmup_steps", "min_lr_ratio", "weight_decay",
    "max_grad_norm", "gradient_accumulation_steps", "use_amp", "amp_dtype",
    "early_stopping_patience", "eval_every_n_epochs", "save_top_k")
BASE_SHOWN = 5


def _rule_bound_words(name: str) -> str:
    """The bound a rule across settings adds to the setting `name`'s own,
    in words, read from the rules themselves (`CROSS_FIELD_RULES`'
    ``min_exclusive``, its guard's values by their words): "more than 0 with
    onecycle"; empty for none."""
    return "; ".join(
        f"more than {rule.threshold:g} with " + " or ".join(
            r.setting_words(guard, value) for guard, allowed in (rule.when or {}).items()
            for value in allowed)
        for rule in CROSS_FIELD_RULES if rule.kind == "min_exclusive" and rule.subject == name)


#: A field's format, unit or "off", said under it as visible text (``info=``;
#: the WebUI skill's §7: never only in a bubble), by its setting or its model
#: spec field's dotted path. A field not named here takes a plain number, a
#: switch or one of its choices. A bound is read from its root (the field
#: spec's, `bound_words`, and a rule's, `_rule_bound_words`); the warmup words
#: are the Trainer's (`Trainer._create_scheduler`: a number capped at half the
#: steps, 0 the warmup ratio of a tenth), which a test holds to a real Trainer.
FIELD_FORMATS: Dict[str, str] = {
    "num_neighbors": "Whole numbers separated by commas, one per encoder layer; -1 takes all.",
    "warmup_steps": "Optimizer steps, at most half the run's; 0 takes a tenth of them.",
    "min_lr_ratio": "; ".join(filter(None, (
        f"A fraction of the peak rate, {bound_words(by_name('min_lr_ratio').valid)}",
        _rule_bound_words("min_lr_ratio")))) + ".",
    "gradient_accumulation_steps": "Batches per optimizer step.",
    "eval_every_n_epochs": "Epochs between the Trainer's validations.",
    "early_stopping_patience": 'Epochs, or "off": no early stopping.',
    "encoder.relation_edge_dim": 'A whole number, or "off": edges carry no relation.',
}

#: What a setting's help closes with (`setting_help`): a tunable value's
#: reference is the source's provisional default (`src.config.recipe`), so
#: says that; a setting whose values are a study's arms, not a tuned value,
#: says what each arm is instead.
_SETTING_CLOSINGS = {
    "frozen_parts": "Not a tuned value: Trained is the source's full fine-tune, and Frozen is "
                    "the frozen-backbone control.",
}
_PROVISIONAL = "A provisional default from the source, not tuned on this data."


def setting_help(name: str, references: Mapping[str, Any], sources: Mapping[str, str]
                 ) -> Tuple[str, str]:
    """A setting's help, ``(title, text)``: its label, then what it is (its
    field spec's description, `training_fields.FieldSpec.description`, the
    one text of it), then the value a run takes when the operator sets none:
    the reference's, with the source of that value (`src.config.recipe`) and
    what that reference is (`_SETTING_CLOSINGS`, else a provisional default
    from the source), or this build's default where the reference recipe
    sets none (`sources` names no source for it)."""
    lead = by_name(name).description
    shown = r.setting_words(name, references[name])
    if name not in sources:
        return (r.SETTING_LABELS[name],
                f"{lead}\n\nDefault: {shown}, this build's; the reference recipe sets none.")
    return (r.SETTING_LABELS[name],
            f"{lead}\n\n"
            f"Reference: {shown}.\n"
            f"Source: {sources[name]}.\n\n"
            f"{_SETTING_CLOSINGS.get(name, _PROVISIONAL)}")


def form_value(value: Any) -> Any:
    """A value as its field holds it, the text `parse_value` reads back: a
    list as its numbers separated by commas ("15, 10, 5", a ``-1`` as typed),
    and an optional number's None as "off" (`value_field`)."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return r.OFF.lower() if value is None else value


def text_field(kind: str, value: Any) -> bool:
    """Whether the field of a value of `kind` built at `value` is a text box
    (`value_field`): a list of whole numbers, and an optional number whose
    value is None there; every other number is a number box, which takes a
    number or None, never text."""
    return kind == "list[int]" or (value is None and kind in ("int", "float"))


def value_field(kind: str, label: str, value: Any, *, elem_id: str,
                choices: Optional[Sequence[Tuple[str, Any]]] = None,
                info: Optional[str] = None, visible: bool = True) -> Any:
    """**The one field of a value**, by its kind (a field spec's, or a model
    spec field's declared type): a list of whole numbers as text, a switch, a
    few `choices` as a radio (``(words, value)``: each shown by its words and
    sent as the value), else a number box (whole numbers for an integer,
    which `value_problem` and the spec's type rule refuse as a float). **An
    optional number whose value is None** (early stopping's patience, the
    encoder's relation edge dimension) is a text box reading "off", a state
    the browser keeps: a number box shows None as 0 once mounted and sends 0
    after any edit. `info` is its format, unit or "off", said under it
    (`FIELD_FORMATS`); what it is stays behind its ⓘ."""
    common = {"label": label, "elem_id": elem_id, "info": info, "visible": visible}
    if text_field(kind, value):
        return gr.Textbox(value=form_value(value), max_lines=1, **common)
    if kind == "bool":
        return gr.Checkbox(value=bool(value), **common)
    if choices:
        return gr.Radio(choices=list(choices), value=value, **common)
    return gr.Number(value=value, precision=0 if kind == "int" else None, **common)


def setting_field(name: str, value: Any, *, elem_id: str, visible: bool = True) -> Any:
    """**The one field of a learning setting** (the base form's and the
    Fine-tune page's), by its field spec (`value_field`): its kind, and its
    closed values shown by their words (`setting_words`) and sent as the
    spec's. Labelled by `r.SETTING_LABELS`, its format under it
    (`FIELD_FORMATS`), its help behind the ⓘ (`setting_help`)."""
    spec = by_name(name)
    choices = ([(r.setting_words(name, choice), choice) for choice in spec.choices]
               if spec.choices else None)
    return value_field(spec.kind, r.SETTING_LABELS[name], value, elem_id=elem_id,
                       choices=choices, info=FIELD_FORMATS.get(name), visible=visible)


def parse_value(kind: str, value: Any, *, optional: bool = False) -> Tuple[Any, Optional[str]]:
    """A field's value as its kind takes it (`value_field`): a list's text as
    whole numbers; a number's text (an `optional` number's field is a text
    box), "off" or empty as None, else the number it says; a whole float of an
    integer field (a number box's value) as an int; an int of a float field
    as a float. ``(value, why not)``, why not in the words that follow the
    field's name, "or "off"" only where None is taken."""
    if kind == "list[int]":
        try:
            return [int(part) for part in str(value or "").split(",") if part.strip()], None
        except ValueError:
            return None, "must be whole numbers separated by commas, such as 15, 10, 5."
    if kind in ("int", "float") and isinstance(value, str):
        text = value.strip()
        if text.lower() in ("", r.OFF.lower()):
            return None, None
        try:
            return (int(text) if kind == "int" else float(text)), None
        except ValueError:
            off = f' or "{r.OFF.lower()}"' if optional else ""
            return None, (f"must be a whole number{off}." if kind == "int"
                          else f"must be a number{off}.")
    if kind == "int" and isinstance(value, float) and value.is_integer():
        return int(value), None
    if kind == "float" and isinstance(value, int) and not isinstance(value, bool):
        return float(value), None
    return value, None


def _named(name: str, problem: str, label: str) -> str:
    """A problem whose sentence starts with `name` (a check's words), named
    by `label` instead; any other led by the label."""
    return label + problem[len(name):] if problem.startswith(name) else f"{label}: {problem}"


#: A sentence of a check that names a setting first by its record name, at
#: the text's start or after another sentence's end (a check's sentences are
#: joined by one space): "min_lr_ratio must be more than 0 when …".
_LEADING_SETTING = re.compile(
    r"(?:^|(?<=\. ))(" + "|".join(sorted(map(re.escape, r.SETTING_LABELS), key=len,
                                         reverse=True)) + r")(?= )")


def labelled_problem(problem: Any) -> str:
    """**A check's sentences, each that names a setting first led by the
    setting's label instead** (`r.SETTING_LABELS`, as `_named` names a
    value's problem: "Final learning-rate ratio must be more than 0 when …"),
    as the form names its field; any other as it is. The base form's rules
    across settings and a Branch check's refusal of the settings it sends
    (the job runner's ``invalid_settings``, its sentences joined) are said by
    it, so one rule reads alike on the page (the B1c review). Plain text: the
    caller escapes it."""
    return _LEADING_SETTING.sub(lambda match: r.SETTING_LABELS[match.group(1)],
                                str(problem or ""))


def setting_problem(name: str, value: Any, reference: Any) -> Tuple[Any, Optional[str]]:
    """A field's value parsed (`parse_value`) and checked by its field spec
    (`value_problem`, the one check the start makes again): ``(value,
    problem)``, the problem named by the field's label (`r.SETTING_LABELS`).
    Empty is taken only where the reference value is (early stopping's
    patience)."""
    label = r.SETTING_LABELS[name]
    parsed, why = parse_value(by_name(name).kind, value, optional=reference is None)
    if why:
        return None, f"{label} {why}"
    problem = value_problem(name, parsed, allow_none=reference is None)
    return parsed, (_named(name, problem, label) if problem else None)


def touched_keys(touched: Any) -> frozenset:
    """**A form's record of the fields the operator touched, as a request
    carries it** (decision S2: explicit is what the operator touched): its
    keys, each a setting or a model spec field's path. The record is the
    browser's (`touched_record`), so it is read as data: a list's strings;
    anything else is no key."""
    if not isinstance(touched, (list, tuple)):
        return frozenset()
    return frozenset(key for key in touched if isinstance(key, str))


def touched_with(touched: Any, key: str) -> List[str]:
    """A form's record (`touched_keys`) with `key` added: the record a
    field's input leaves (`TOUCH_JS`), which the base form's line on what the
    operator set is said from (`_base_input_listener`)."""
    return sorted({*touched_keys(touched), key})


#: **A field's input adds its key to its form's record, in the browser**
#: (`record_touches`; ``{key}`` is the key as a JSON string): the record's
#: keys and the key, each once, in order. Returned in a list, the listener's
#: one output (Gradio reads a returned array as the outputs).
TOUCH_JS = ("(touched) => [[...new Set([...(Array.isArray(touched) ? touched : []), {key}])]"
            ".sort()]")


def touched_record(elem_id: str) -> Any:
    """**A form's record of the fields the operator touched** (decision S2),
    held in the browser: a hidden JSON list, empty at first and again after
    Reset, that each field's input adds its key to there (`record_touches`),
    with no request of its own, and that Start's own request carries beside
    the fields' values. What Start sends is therefore read from the click's
    own payload alone, never from a state another request writes on the
    server (the base form's and the Fine-tune page's)."""
    return gr.JSON(value=[], visible=False, elem_id=elem_id)


def record_touches(field: Any, key: str, record: Any) -> None:
    """`field`'s input (the operator's own edit: ``.input``, never
    ``.change``, which a page's own write fires too) adds `key` to `record`
    (`touched_record`) in the browser (`TOUCH_JS`), before any later click is
    sent; every input does ("multiple"). Wire it before any other input
    listener of `field`: the browser runs an event's listeners in order, so
    they read the record with `key`."""
    field.input(None, inputs=[record], outputs=[record],
                js=TOUCH_JS.replace("{key}", json.dumps(key)), queue=False,
                trigger_mode="multiple", show_progress="hidden")


def sent_settings(values: Mapping[str, Any], touched: Collection[str],
                  references: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """**What a start sends of a form's settings** (decisions S2, S3):
    ``(settings, problems)``. A setting is sent **when the operator touched
    its field** (`touched`, the form's record of its fields' input), whatever
    its value: the reference's own value typed is sent, and recorded as the
    operator's; an untouched one is not, and the run fills it (the
    reference's, or this build's default) and records it so. A touched
    setting the run would not read on the form's values (a rule across
    settings, `training_fields.not_read_by_rules`: warmup without cosine) is
    hidden and not sent. Each one sent is parsed and checked by its field
    spec (`setting_problem`; `references`, the run's values without it, say
    where empty is taken), every problem named by its label. The base form
    and the Fine-tune page send their settings by it."""
    unread = not_read_by_rules({name: parse_value(by_name(name).kind, value)[0]
                                for name, value in values.items()})
    settings: Dict[str, Any] = {}
    problems: List[str] = []
    for name, value in values.items():
        if name not in touched or name in unread:
            continue
        parsed, problem = setting_problem(name, value, references.get(name))
        if problem:
            problems.append(problem)
        else:
            settings[name] = parsed
    return settings, problems


# ----------------------------------------------------------------- the base form
def base_references() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """What a new base trains with when the operator sets nothing, and the
    source of each value the reference recipe gives: **the one fill of a new
    run** (`run_plan.new_run_values`: the stage's reference, else this
    build's default, the field spec's) and the recipe's sources
    (`recipe.reference_settings`); a setting without a source is a build
    default. Its model spec, None here, is the reference spec
    (`reference_spec`)."""
    return new_run_values("base", None, {}), reference_settings("base")[1]


#: The model spec's sections the base form offers, in its order: the input
#: module, the encoder and the LP head (`spec.InputSpec`, `EncoderSpec`,
#: `LPHeadSpec`). The graph view is not the form's (no relation editor).
SPEC_SECTIONS = ("input", "encoder", "lp_head")
#: The architecture: the encoder's kind (`architecture_choices`).
ARCHITECTURE = "encoder.kind"
#: The form's one name of the architecture (`spec_key_words`).
ARCHITECTURE_LABEL = "Architecture"
#: The Model accordion's first line: what Start sends of it.
MODEL_NOTE = ("A new base builds the reference spec; Start sends it whole, with the fields you "
              "change, and checks it first.")
#: Each declared type a spec field may hold, as the field kinds name it
#: (`value_field`); a text, or any other type, is a fact the form shows.
_SPEC_KINDS = {int: "int", float: "float", bool: "bool"}


@dataclass(frozen=True)
class SpecField:
    """One field of a model spec section in the base form: its dotted path
    (``encoder.layers``, as `ModelSpec.differences_from_reference` names a
    field), its kind (`value_field`'s, or ``"fact"`` for a text the form
    shows and does not edit), and its value in the reference base spec."""

    path: str
    kind: str
    reference: Any

    @property
    def editable(self) -> bool:
        return self.kind != "fact"

    @property
    def elem_id(self) -> str:
        return "spec-" + self.path.replace(".", "-")


def spec_form_fields() -> Tuple[SpecField, ...]:
    """**The base spec's fields the form shows, generated from each
    section's dataclass fields** (`SPEC_SECTIONS`) and their declared types
    (`spec.declared_fields`, what the spec's own type rule reads), in their
    declared order: a number or a switch is a field the operator may edit
    (`Optional[int]`, the relation edge dimension, off or a number), a text
    is a fact (`spec_facts_html`). The encoder's kind is the architecture,
    offered apart (`architecture_choices`)."""
    reference = reference_spec("base")
    out: List[SpecField] = []
    for section in SPEC_SECTIONS:
        part = getattr(reference, section)
        for name, declared in declared_fields(type(part)):
            path = f"{section}.{name}"
            if path == ARCHITECTURE:
                continue
            members = [arg for arg in get_args(declared) if arg is not type(None)]
            kind = _SPEC_KINDS.get(members[0] if len(members) == 1 else declared, "fact")
            out.append(SpecField(path, kind, getattr(part, name)))
    return tuple(out)


def architecture_choices() -> List[Tuple[str, str]]:
    """**The architectures a new base may build** (decision S5): the encoder
    kinds this build implements, each by its declaration
    (`spec.ENCODER_DECLARATIONS`, pinned equal to the builder's
    implementations), by their words (`renderers.architecture_label`); a name
    only `ENCODER_KINDS` holds is never offered. With one, the Model accordion
    states it as a fact (`spec_facts_html`); with two or more it is a choice,
    sent as the spec's encoder kind (`ARCHITECTURE`)."""
    return [(r.architecture_label(kind), kind) for kind in ENCODER_DECLARATIONS]


def _spec_keys() -> Tuple[str, ...]:
    """The base form's spec fields an operator may touch, in its order: the
    architecture, when it is a choice, then each editable spec field."""
    choice = (ARCHITECTURE,) if len(architecture_choices()) > 1 else ()
    return choice + tuple(field.path for field in spec_form_fields() if field.editable)


def base_form_keys() -> Tuple[str, ...]:
    """The base form's fields in the order its handlers take their values:
    its settings (`BASE_FIELDS`), then its spec fields (`_spec_keys`)."""
    return BASE_FIELDS + _spec_keys()


def _spec_reference(key: str) -> Any:
    section, name = key.split(".", 1)
    return getattr(getattr(reference_spec("base"), section), name)


def spec_key_words(key: str) -> str:
    """**The base form's one name of a spec field**, wherever the form names
    it (its field, its facts, the line on what the operator set, a refusal):
    the architecture by `ARCHITECTURE_LABEL`, any other by its path's words
    (`renderers.field_path_words`: "Encoder › Layers")."""
    return ARCHITECTURE_LABEL if key == ARCHITECTURE else r.field_path_words(key)


def base_form_defaults() -> Tuple[Any, ...]:
    """The base form's starting values, in `base_form_keys` order: what a new
    base trains with when the operator sets nothing (`base_references`, and
    the reference base spec's fields), as each field holds it
    (`form_value`). Reset to reference writes them again."""
    references, _ = base_references()
    return tuple(form_value(references[key] if key in references else _spec_reference(key))
                 for key in base_form_keys())


#: A spec field's dotted path at the start of a check's sentence.
_SPEC_PATH = re.compile(rf"\b(?:{'|'.join(SPEC_SECTIONS)})\.[a-z_]+\b")


def edited_spec(values: Mapping[str, Any], touched: Collection[str]
                ) -> Tuple[Optional[Any], Optional[str]]:
    """**The model spec Start sends** (decisions S2, S5): ``(spec, problem)``.
    None and None when the operator touched no spec field: the run builds the
    reference spec and records it so. Otherwise the reference base spec with
    each touched field replaced by its value (`parse_value`, by its kind),
    checked by the spec's own validation (`ModelSpec.validate`: each field's
    declared type, each section's rules and its kind's declaration's), which
    the start makes again; or None and why not, each field named by the
    form's name of it (`spec_key_words`)."""
    keys = [key for key in _spec_keys() if key in touched]
    if not keys:
        return None, None
    fields = {field.path: field for field in spec_form_fields()}
    changes: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        field = fields.get(key)
        parsed, why = parse_value(field.kind if field else "fact", values[key],
                                  optional=field is not None and field.reference is None)
        if why:
            return None, f"{spec_key_words(key)} {why}"
        section, name = key.split(".", 1)
        changes.setdefault(section, {})[name] = parsed
    reference = reference_spec("base")
    spec = replace(reference, **{section: replace(getattr(reference, section), **fields)
                                 for section, fields in changes.items()})
    try:
        spec.validate()
    except SpecError as exc:
        return None, _SPEC_PATH.sub(lambda match: spec_key_words(match.group(0)), str(exc))
    return spec, None


def _base_reading(values: Mapping[str, Any], touched: Collection[str]
                  ) -> Tuple[Dict[str, Any], Optional[Any], List[str]]:
    """The base form read once: ``(settings, spec, problems)``, the settings
    Start sends (`sent_settings`; the model spec among them, whole, when a
    field of it was touched: `edited_spec`), that spec, and every problem
    the start's own checks would find, each named by its label: each value's
    (`sent_settings`), **the rules across settings** on the run's settings as
    the start fills them, for what the form sends (`cross_field_problems` on
    `new_run_values`, as `run_plan.new_run_problem` asks them: onecycle needs
    a minimum ratio above 0), the spec's own (`edited_spec`), and the hop
    list the run would sample (the one sent, else the reference's) against
    the encoder it would build (`hops_problem`), so a changed encoder is
    checked with the hops."""
    references, _ = base_references()
    settings, problems = sent_settings({name: values[name] for name in BASE_FIELDS},
                                       touched, references)
    problems.extend(labelled_problem(problem) for problem in
                    cross_field_problems(new_run_values("base", None, settings), settings))
    spec, problem = edited_spec(values, touched)
    if problem:
        problems.append(problem)
        return settings, None, problems
    if spec is not None:
        settings["model_spec"] = spec.to_dict()
    if not ("num_neighbors" in touched and "num_neighbors" not in settings):
        hops = hops_problem(settings.get("num_neighbors", references["num_neighbors"]),
                            (spec or reference_spec("base")).encoder)
        if hops:
            problems.append(_named("num_neighbors", hops, r.SETTING_LABELS["num_neighbors"]))
    return settings, spec, problems


def base_start_settings(values: Mapping[str, Any], touched: Collection[str]
                        ) -> Tuple[Dict[str, Any], List[str]]:
    """**What Start sends of the base form** (decisions S1, S2, S3, S5):
    ``(settings, problems)``, `values` by `base_form_keys` and `touched` the
    form's record of its fields' input (`_base_reading`)."""
    settings, _spec, problems = _base_reading(values, touched)
    return settings, problems


def _spec_words(key: str, value: Any) -> str:
    return r.architecture_label(value) if key == ARCHITECTURE else r.setting_words(key, value)


#: How many of the settings the operator set the line names (`set_by_you_html`);
#: the rest are counted, and every one is in its Details.
SET_BY_SHOWN = 3


def set_by_you_html(values: Mapping[str, Any], touched: Collection[str]) -> str:
    """**The base form's one line on what the operator set** (decision S2),
    under its fields: "Set by you: Epochs 5 (reference 200), Encoder ›
    Layers 2 (reference 3)", each setting and spec field Start sends, in the
    form's order, by the form's name of it, with the value the run takes
    without it in brackets (the reference's; "default" where the recipe sets
    none and the run takes this build's). The first `SET_BY_SHOWN` are named
    and the rest counted ("and 4 more"; `renderers.named_words`), every one
    a row of its "Details ›" (`renderers.details_list`). With none: "Set by
    you: nothing; the run takes the reference values", and this build's
    defaults where the recipe sets some. Then "Not valid: …", what Start
    would refuse (`_base_reading`), by its label. Escaped."""
    references, sources = base_references()
    settings, spec, problems = _base_reading(values, touched)
    rows = [(r.SETTING_LABELS[name], f"{r.setting_words(name, settings[name])} "
             f"({'reference' if name in sources else 'default'} "
             f"{r.setting_words(name, references[name])})")
            for name in BASE_FIELDS if name in settings]
    if spec is not None:
        for key in _spec_keys():
            if key in touched:
                section, name = key.split(".", 1)
                rows.append((spec_key_words(key),
                             f"{_spec_words(key, getattr(getattr(spec, section), name))} "
                             f"(reference {_spec_words(key, _spec_reference(key))})"))
    if rows:
        text = "Set by you: " + r.named_words([f"{key} {value}" for key, value in rows],
                                              SET_BY_SHOWN)
    else:
        defaults = any(name not in sources for name in BASE_FIELDS)
        text = ("Set by you: nothing; the run takes the reference values"
                + (" and this build's defaults." if defaults else "."))
    details = (r.details_list([(key, r.e(value)) for key, value in rows],
                              classes="shep-setby-kv")
               if len(rows) > SET_BY_SHOWN else "")
    refused = (f'<p class="shep-amber-t">Not valid: {r.e(" ".join(problems))}</p>'
               if problems else "")
    return f'<div class="shep-setby"><p>{r.e(text)}</p>{details}{refused}</div>'


def spec_facts_html() -> str:
    """The Model accordion's facts (decision S5), one summary list (the
    skill's §4): the architecture, when this build implements one
    (`architecture_choices`; with more it is a choice), and each text field
    of the reference base spec (`spec_form_fields`), each by the form's name
    of it (`spec_key_words`), its value kept whole in code style."""
    choices = architecture_choices()
    rows = [(spec_key_words(ARCHITECTURE), r.e(choices[0][0]))] if len(choices) == 1 else []
    rows += [(spec_key_words(field.path), r.ident_html(r.setting_words(field.path,
                                                                       field.reference)))
             for field in spec_form_fields() if not field.editable]
    return r.summary_list(rows, classes="shep-train-spec-facts")


def rule_fields(fields: Sequence[str] = BASE_FIELDS) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """A form's settings (`fields`, in its order: the base form's, the
    Continue drawer's Branch fields or the Fine-tune page's) that a rule
    across settings reads only under its guard (`CROSS_FIELD_RULES`'
    ``read_only_when``), and the settings of those guards, each in the
    form's order."""
    rules = [rule for rule in CROSS_FIELD_RULES if rule.kind == "read_only_when"]
    subjects = {rule.subject for rule in rules}
    guards = {name for rule in rules for name in (rule.when or {})}
    return (tuple(name for name in fields if name in subjects),
            tuple(name for name in fields if name in guards))


def rule_unread(fields: Sequence[str], values: Sequence[Any]
                ) -> Tuple[Tuple[str, ...], frozenset]:
    """**What a form's guards leave unread**: ``(dependent, unread)``, the
    settings of `fields` a rule across settings reads only under its guard
    (`rule_fields`) and those of them the run would not read on the guards'
    field `values` (in `rule_fields(fields)` order), by the rules
    themselves (`not_read_by_rules`). The base form's, the Fine-tune page's
    and the Branch fields' guards are answered by it."""
    dependent, guards = rule_fields(fields)
    return dependent, not_read_by_rules({name: parse_value(by_name(name).kind, value)[0]
                                         for name, value in zip(guards, values, strict=True)})


def guard_updates(fields: Sequence[str], values: Sequence[Any]) -> List[Any]:
    """A guard of a rule across settings changed (the scheduler, mixed
    precision; by the operator or by a page's own write): each setting of a
    form's `fields` that a rule reads only under its guard is shown while the
    run would read it and hidden otherwise (`rule_unread`). `values` are the
    guards', in `rule_fields(fields)` order. The base form's guards
    (`_on_base_guards`) and the Fine-tune page's are answered by it."""
    dependent, unread = rule_unread(fields, values)
    return [gr.update(visible=name not in unread) for name in dependent]


def _on_base_guards(*values: Any) -> List[Any]:
    """The base form's guards changed (by the operator or by Reset):
    `guard_updates` of its settings (`BASE_FIELDS`)."""
    return guard_updates(BASE_FIELDS, values)


def _base_input_listener(key: str) -> Callable[..., str]:
    """The line listener of the base form's field `key` (`set_by_you_html`),
    on the field's input after the record has `key` (`record_touches`, wired
    first; `touched_with` adds it all the same). Its values are the form's,
    in `base_form_keys` order. It writes only the line, never the record,
    which is the browser's: an answer that arrives late cannot undo a later
    touch."""
    def on_input(touched: Any, *values: Any) -> str:
        return set_by_you_html(dict(zip(base_form_keys(), values, strict=True)),
                               touched_with(touched, key))

    return on_input


def _on_base_reset() -> Tuple[Any, ...]:
    """Reset to reference: the form's record empty (`touched_record`), the
    line again, and every field back at the value the run takes without it
    (`base_form_defaults`). The values are a page's writes, so no field's
    input fires and none is recorded as touched; the guards' change shows or
    hides what depends on them."""
    defaults = base_form_defaults()
    return ([], set_by_you_html(dict(zip(base_form_keys(), defaults, strict=True)), []),
            *(gr.update(value=value) for value in defaults))


# =============================================================================
# The run bar's workspace (the session's selection)
# =============================================================================
def empty_workspace() -> Dict[str, Any]:
    return {"name": None, "row": None, "facts": None, "choices": [], "listed": [],
            "digits": r.SHORT_HEX, "problem": None, "runs": []}


def _listed_runs(directory: Optional[Path]) -> List[str]:
    """A workspace's run directories, as "<arch>/<run>", from the run
    layout's one listing (`list_runs`: most recently trained first); none
    when it lists none. Only the runs inside the workspace, as `resolve_run`
    finds one: a run behind a link out of it is never named, so the page
    never names a run it then cannot show."""
    if directory is None:
        return []
    from src.utils.checkpoint_paths import list_runs, run_architecture

    try:
        runs = list_runs(directory)
        home = Path(directory).resolve()
    except (OSError, RuntimeError):  # a link loop: RuntimeError before 3.13
        return []
    named = []
    for run in runs:
        try:
            inside = home in run.resolve().parents
        except (OSError, RuntimeError):
            inside = False
        if inside:
            named.append(f"{run_architecture(run)}/{run.name}")
    return named


def workspace_state(name: Optional[str], listed: Optional[Mapping[str, Any]] = None
                     ) -> Dict[str, Any]:
    """The session's run-bar workspace, read now (or from `listed`, a
    `read_workspaces` answer just read): the server's list (the dashboard's one
    reader, every reason without an absolute path) with the options' labels,
    the selected item, what its manifest binds, the listed fingerprints (to say
    what a parent trained on) and its runs, latest first (`_listed_runs`). The
    full fingerprint kept here is the one Start sends."""
    listed = listed if listed is not None else read_workspaces()
    rows = listed["rows"]
    state = empty_workspace()
    # The options are every page's (`renderers.workspace_choices`: the name
    # first); the identity (fingerprint, record, why it cannot be trained) is
    # the chip's, beside the select.
    state["choices"] = [[label, name] for label, name in r.workspace_choices(rows)]
    state["listed"] = [{"name": x["name"], "dataset_fingerprint": x.get("dataset_fingerprint"),
                        "dataset_record": x.get("dataset_record")} for x in rows]
    state["digits"] = r.distinguishing_digits(x.get("dataset_fingerprint") for x in rows)
    state["problem"] = listed["problem"]
    row = next((x for x in rows if x["name"] == name), None)
    if row is not None:
        state["name"], state["row"] = row["name"], row
        directory = listed["directories"].get(row["name"])
        if row.get("manifest") == "read" and directory is not None:
            state["facts"], _problem = manifest_facts(directory, listed["root"])
        state["runs"] = _listed_runs(directory)
    return state


def workspace_fingerprint(ws: Optional[Mapping[str, Any]]) -> Optional[str]:
    row = (ws or {}).get("row") or {}
    value = row.get("dataset_fingerprint")
    return value if r.is_digest(value) else None


def dataset_choices_update(ws: Mapping[str, Any], *, interactive: bool = True) -> Any:
    choices = [tuple(c) for c in ws.get("choices") or []]
    return gr.update(choices=choices, value=ws.get("name"), interactive=interactive)


# --------------------------------------------------------------------------- the chip
LOCK_ICON = ('<svg class="shep-lock" width="12" height="12" viewBox="0 0 16 16" aria-hidden="true" '
         'focusable="false"><rect x="3" y="7" width="10" height="7.5" rx="1.6" '
         'fill="currentColor"/><path d="M5.2 7V5.2a2.8 2.8 0 0 1 5.6 0V7" fill="none" '
         'stroke="currentColor" stroke-width="1.6"/></svg>')
NO_ICON = ('<svg class="shep-no" width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" '
       'focusable="false"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" '
       'stroke-width="1.5"/><path d="M4.6 8h6.8" stroke="currentColor" stroke-width="1.5"/></svg>')


def details_html(ws: Mapping[str, Any]) -> str:
    """The chip's "Details ›" popover: the workspace's name and record, the
    shared details body (`renderers.dataset_details_html`, as Research ›
    Datasets shows it), and "All datasets ›"."""
    row = ws.get("row") or {}
    name = str(row.get("name") or "")
    head = (f'<div class="shep-pop-hd"><b>{r.e(name)}</b>{r.record_chip(row)}'
            '<button type="button" class="shep-pop-x" data-act="close" '
            'aria-label="Close the dataset details">×</button></div>')
    return (head + r.dataset_details_html(row, ws.get("facts"))
            + '<button type="button" class="shep-link" data-act="datasets">All datasets ›</button>')


def chip_html(ws: Optional[Mapping[str, Any]], *, in_use: bool = False) -> str:
    """The dataset chip beside the select: "in use by the running job" while a
    training job runs, the shared dataset identity (built or sources, the
    dataset fingerprint in bold monospace, synthetic, record verified, not
    trainable), and "Details ›" with its popover."""
    ws = ws or {}
    row = ws.get("row")
    lead = '<span class="shep-rb-inuse">in use by the running job</span>' if in_use else ""
    if not row:
        why = ws.get("problem") or ("No dataset selected: choose one from the server's list."
                                    if ws.get("choices") else "The server lists no workspace.")
        return f'<div class="shep-rb-chip">{lead}<span class="shep-muted">{r.e(why)}</span></div>'
    identity = r.dataset_identity_html(row, name=False, digits=ws.get("digits") or r.SHORT_HEX)
    return (f'<div class="shep-rb-chip">{lead}{identity}'
            '<button type="button" class="shep-link shep-rb-details" data-act="details" '
            'aria-expanded="false" aria-controls="shep-train-details">Details ›</button>'
            '<div class="shep-pop" id="shep-train-details" role="dialog" '
            f'aria-label="Dataset details" hidden>{details_html(ws)}</div></div>')


#: A "Details ›" popover's behaviour, element-scoped (the element is never
#: re-mounted, and a replaced value keeps the listeners, being delegated): the
#: dataset chip's, with its "All datasets ›" cross-link, and the run view
#: header's. Escape, a click outside or focus leaving it closes it.
_POPOVER_JS = r"""
function pop() { return element.querySelector('.shep-pop'); }
function toggle(open) {
  var p = pop(), b = element.querySelector('[data-act="details"]');
  if (!p || !b) { return; }
  p.hidden = !open;
  b.setAttribute('aria-expanded', open ? 'true' : 'false');
  if (open) { var c = p.querySelector('.shep-copy, .shep-pop-x'); if (c) { c.focus(); } }
}
element.addEventListener('click', function (e) {
  var a = e.target.closest ? e.target.closest('[data-act]') : null;
  if (!a) { return; }
  var act = a.getAttribute('data-act');
  if (act === 'details') { var p = pop(); toggle(p && p.hidden); }
  else if (act === 'close') { toggle(false); var b = element.querySelector('[data-act="details"]'); if (b) { b.focus(); } }
  else if (act === 'datasets') { toggle(false); trigger('click', {act: 'datasets'}); }
});
element.addEventListener('keydown', function (e) {
  if (e.key === 'Escape' && pop() && !pop().hidden) { e.stopPropagation(); toggle(false);
    var b = element.querySelector('[data-act="details"]'); if (b) { b.focus(); } }
});
element.addEventListener('focusout', function (e) {
  var p = pop();
  if (p && !p.hidden && e.relatedTarget && !element.contains(e.relatedTarget)) { toggle(false); }
});
element.addEventListener('pointerdown', function (e) { e.stopPropagation(); });
document.addEventListener('pointerdown', function (e) {
  var p = pop();
  if (p && !p.hidden && element.isConnected && !element.contains(e.target)) { toggle(false); }
}, {passive: true});
"""

#: Cross-links in the reason line ("Open Test ›") and the banner's Dismiss.
TRIGGER_JS = r"""
element.addEventListener('click', function (e) {
  var a = e.target.closest ? e.target.closest('[data-act]') : null;
  if (a) { trigger('click', {act: a.getAttribute('data-act'), key: a.getAttribute('data-key') || ''}); }
});
"""

#: A wide window at least this tall shows a failed run's output tail open;
#: a shorter one shows its summary line, so the charts keep their room.
BANNER_OPEN_HEIGHT = 1000

#: The banner: Dismiss (`TRIGGER_JS`) and its output tail's disclosure, sent
#: open and closed here once per banner on a wide window under
#: `BANNER_OPEN_HEIGHT` px; once the user opens or closes it, their choice
#: holds for the banners that follow on this page.
_BANNER_JS = TRIGGER_JS + r"""
function fitTail() {
  var d = element.querySelector('details.shep-banner-more:not([data-fit])');
  if (!d) { return; }
  d.setAttribute('data-fit', '');
  var chosen = element.getAttribute('data-tail');
  var wide = false;
  try { wide = window.matchMedia('(min-width: 1400px) and (min-height: 760px)').matches; } catch (err) { /* keep it open */ }
  d.open = chosen ? chosen === 'open' : !(wide && window.innerHeight < __OPEN__);
}
element.addEventListener('click', function (e) {
  var s = e.target.closest ? e.target.closest('details.shep-banner-more > summary') : null;
  if (s) { element.setAttribute('data-tail', s.parentElement.open ? 'closed' : 'open'); }
});
new MutationObserver(fitTail).observe(element, {childList: true, subtree: true});
fitTail();
""".replace("__OPEN__", str(BANNER_OPEN_HEIGHT))


# =============================================================================
# Status, GPU jobs and the reason line
# =============================================================================
def _clock(seconds: Any) -> Optional[str]:
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds < 0:
        return None
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _elapsed(status: Mapping[str, Any]) -> Optional[str]:
    """How long the job has run (or ran, from its start and end)."""
    if status.get("elapsed_seconds") is not None:
        return _clock(status["elapsed_seconds"])
    try:
        start = datetime.fromisoformat(str(status["start_time"]))
        end = datetime.fromisoformat(str(status["end_time"]))
    except (KeyError, TypeError, ValueError):
        return None
    return _clock((end - start).total_seconds())


def _epochs(status: Mapping[str, Any], sep: str = " of ") -> Optional[str]:
    epoch, total = status.get("current_epoch"), status.get("total_epochs")
    if epoch is None:
        return None
    return f"epoch {epoch}{sep}{total}" if total else f"epoch {epoch}"


#: **The status table of the training jobs** (K6 decision 9): each state's
#: kind, never green for a failed, stopped or cancelled run. The run bar's
#: pill reads it, and so does the Fine-tune job card for each child, whose
#: own states are a Fine-tune start's: queued, cancelled (by a stop) and
#: refused (the slot would not take it).
PILL_KIND = {"idle": "idle", "starting": "accent", "running": "running", "stopping": "accent",
             "completed": "ok", "stopped": "grey", "failed": "red",
             "queued": "grey", "cancelled": "grey", "refused": "red"}
#: Each kind's row of the shared status marks (`renderers.STATUS_MARKS`): a
#: job starting or stopping is busy, not yet or no longer running (the Stop
#: button's own square), and only Running turns.
STATUS_MARK_OF_KIND = {"ok": "ok", "grey": "neutral", "idle": "neutral", "red": "error",
                       "running": "running", "accent": "stopping"}


def job_status_html(status: str, words: Optional[str] = None) -> str:
    """A job's status as colour, shape and word together (the WebUI skill
    §6; `renderers.status_mark`), by its kind in `PILL_KIND`: never green for
    a failed, stopped, cancelled or refused job. `words` default to the
    status itself ("Running"). The Fine-tune job card's statuses."""
    kind = STATUS_MARK_OF_KIND[PILL_KIND.get(status, "grey")]
    return r.status_mark(kind, words or str(status).capitalize())


def pill_state(status: Mapping[str, Any]) -> Tuple[str, str]:
    """``(kind, words)`` for the status pill: the job's state and, after at
    most one " · " (the WebUI skill §5), how far it got ("RUNNING · epoch
    12/100", "Completed · epoch 60 of 60"; a start's phase). The rest, its
    batch and how long it has run, is the pill's details (`pill_details`)."""
    s = status.get("status") or "idle"
    if s == "idle":
        return "idle", "Idle · no training job since the server started"
    if s == "running":
        if status.get("current_epoch") is None:
            phase = ("loading data and the model" if status.get("phase") == "initializing"
                     else "the process is starting")
            return "starting", f"Starting · {phase}"
        return "running", " · ".join(p for p in ("RUNNING", _epochs(status, "/")) if p)
    words = {"stopping": "Stopping", "completed": "Completed", "stopped": "Stopped",
             "failed": "FAILED"}.get(s)
    if words is None:
        return "stopped", str(s)
    return s, " · ".join(p for p in (words, _epochs(status)) if p)


def pill_details(status: Mapping[str, Any]) -> Optional[str]:
    """The pill's details after its words, with no second " · " (its text
    reads "RUNNING · epoch 12/100, batch 340/812, 00:14:32 elapsed",
    `pill_html`): the batch while an epoch runs and how long the job has run
    (or ran), "batch 340/812, 00:14:32 elapsed"; None when it has neither
    (idle)."""
    s = status.get("status") or "idle"
    if s == "idle":
        return None
    parts = []
    batch, batches = status.get("batch"), status.get("total_batches")
    if s == "running" and status.get("current_epoch") is not None and batch is not None \
            and batches:
        parts.append(f"batch {batch}/{batches}")
    elapsed = _elapsed(status)
    if elapsed:
        parts.append(f"{elapsed} elapsed")
    return ", ".join(parts) or None


def pill_html(status: Mapping[str, Any]) -> str:
    """The run bar's status pill: the training job's state by the one status
    rule (`job_status_html`: its shape and word with its colour, a grey ○
    while idle or stopped, the blue spinner only while it runs, a red ✕ only
    for a failure), then its details (`pill_details`). Always the job
    runner's job, whatever run the run view shows."""
    state, words = pill_state(status)
    kind = PILL_KIND.get(state, "grey")
    details = pill_details(status)
    # The details follow the words after a comma only its text carries (the
    # pill's gap shows it): a role="status" region is read as one text.
    more = (f'<span class="shep-pill-sub"><span class="shep-vh">, </span>{r.e(details)}</span>'
            if details else "")
    return (f'<div class="shep-pill shep-pill-{kind}" data-state="{r.e(state)}" role="status">'
            f'{job_status_html(state, words)}{more}</div>')


def _gpu_line(active: Optional[Mapping[str, Any]], lock: Optional[Mapping[str, Any]],
              test: Optional[Mapping[str, Any]], *, operator: bool) -> Optional[Dict[str, Any]]:
    """The GPU job holding the one slot when it is not this page's training
    job, from the job runner (a test run) or the GPU lock's holder record (a
    process this server did not start: a formal CLI, or another server's
    runner on the same lock; or a lock that cannot be used): ``{"text",
    "test"}`` or None."""
    if active and active.get("job_type") == "test":
        inputs = (test or {}).get("inputs") or {}
        label = inputs.get("version_label")
        elapsed = None
        try:
            started = datetime.fromisoformat(str(active.get("start_time")))
            elapsed = _clock((datetime.now() - started).total_seconds())
        except (TypeError, ValueError):
            pass
        detail = " · ".join(str(p) for p in (label, elapsed) if p)
        text = "GPU job: test run" + (f" ({detail})" if detail else " (job runner)")
        if active.get("status") == "stopping":
            text += " · stopping"
        return {"text": text, "test": True}
    if active or lock is None:
        return None
    chip = gpu_chip_from(None, lock, operator=operator)
    if chip.kind == "busy" or lock.get("problem"):
        # The header chip's words, and where to stop this service's inference
        # when that holds it (`busy_words`): the runner refuses a start itself.
        text = busy_words(chip) if chip.kind == "busy" else "the GPU lock cannot be used"
        if lock.get("problem") and operator:
            text += f" ({lock['problem']})"
        return {"text": text, "test": False}
    return None


#: **The Continue drawer's two operations from a parent checkpoint**
#: (ops-wiring round, decisions B1-B4), each through the same run list, parent
#: path, check and start: **continue** it (resume a paused run or extend a
#: finished one, every setting its own) or **branch** it (a new run from its
#: weights with a new optimizer and schedule, its settings the parent's but
#: those changed on its stage's list, or none).
CONTINUE = "continue"
BRANCH = "branch"
#: The drawer's operation choice, a radio of two (the skill's §8), each by its
#: verb, sent as the operation: one line beside its label in the drawer's
#: width (the B1c review: "Continue: resume or extend" and "Branch: a new run
#: from these weights" wrapped to three rows at 1850 x 960 and pushed the
#: check's plan below the fold). What each does is the drawer's title, its
#: help (`FIELD_HELP`) and the check's plan.
OPERATION_CHOICES = [("Continue", CONTINUE), ("Branch", BRANCH)]
#: The drawer's start by the operation, the one button in its footer.
START_LABELS = {CONTINUE: "▸ Continue Training", BRANCH: "▸ Start branch"}
#: What a check says the operation does to a checkpoint: its verb, and its
#: past participle ("cannot be branched").
OPERATION_VERBS = {CONTINUE: ("continue", "continued"), BRANCH: ("branch", "branched")}

#: The run bar's gated actions (the render's keys) and their words, shown with
#: no lock while the operator may use them (`chips.gated`). ``resume`` is the
#: drawer's start, labelled by its operation (`start_label`).
ACTION_LABELS = {"start": "▸ Start", "cont": "Continue…", "stop": "■ Stop",
                 "resume": START_LABELS[CONTINUE]}


def operation_of(mode: Optional[Mapping[str, Any]]) -> str:
    """The drawer's operation as the session records it (`CONTINUE` unless
    Branch is chosen)."""
    return BRANCH if (mode or {}).get("operation") == BRANCH else CONTINUE


def start_label(mode: Optional[Mapping[str, Any]]) -> str:
    """The drawer's start button's words for the session's operation."""
    return START_LABELS[operation_of(mode)]


def run_bar_view(status: Mapping[str, Any], ws: Mapping[str, Any], mode: Mapping[str, Any],
                 *, access: Any, active: Optional[Mapping[str, Any]],
                 gpu: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """What the run bar's actions and the drawer's start are and why, as one
    decision for the tick and every action: ``start``, ``cont``, ``stop`` and
    ``resume`` (interactive or not), ``reason`` (the run bar's, escaped HTML:
    one line), ``drawer_reason`` (the drawer footer's, beside its start, in
    its words: "Start branch disabled: …"; the skill's §9) and ``progress``
    (a fraction while running, else None)."""
    s = status.get("status") or "idle"
    training_busy = s in ("running", "stopping")
    drawer = bool(mode.get("drawer"))
    row = ws.get("row") or {}
    action = start_label(mode).removeprefix("▸ ")
    view: Dict[str, Any] = {"start": False, "cont": False, "stop": False, "resume": False,
                            "progress": None, "drawer_reason": ""}
    epoch, total = status.get("current_epoch"), status.get("total_epochs")
    if s == "running" and isinstance(epoch, int) and isinstance(total, int) and total > 0:
        view["progress"] = max(0.0, min(1.0, epoch / total))

    def line(icon: str, text: str, extra: str = "") -> str:
        return f'<span class="shep-rb-why">{icon}<span>{r.e(text)}</span>{extra}</span>'

    def disabled(cause: str) -> None:
        """One cause, said for the run bar's actions and the drawer's start."""
        view["reason"] = line(NO_ICON, f"Start and Continue… disabled: {cause}")
        view["drawer_reason"] = line(NO_ICON, f"{action} disabled: {cause}")

    if not access.allowed:
        view["reason"] = line(LOCK_ICON, operator_only(access.condition))
        view["drawer_reason"] = view["reason"]
        return view
    if training_busy:
        view["stop"] = s == "running" and not confirming(status, mode)
        if s == "running":
            disabled("a training job is running.")
        else:
            view["reason"] = line(NO_ICON, "Stopping: the job is ending; Start and Continue… "
                                           "wait for it.")
            view["drawer_reason"] = line(NO_ICON, f"Stopping: the job is ending; {action} "
                                                  "waits for it.")
        return view
    if gpu:
        open_test = ('<button type="button" class="shep-link" data-act="open-test">Open Test ›</button>'
                     if gpu.get("test") else "")
        tag = '<span class="shep-tag shep-tag-accent">GPU</span>'
        # The sentence ends with its period, or with "· Open Test ›" instead
        # (the run bar's link; the drawer's reason ends with its period).
        view["reason"] = (f'<span class="shep-rb-why">{tag}'
                          f'<span>Start and Continue… disabled — {r.e(gpu["text"])}'
                          f'{" · " + open_test if open_test else "."}</span></span>')
        view["drawer_reason"] = (f'<span class="shep-rb-why">{tag}<span>{r.e(action)} '
                                 f'disabled — {r.e(gpu["text"])}.</span></span>')
        return view
    if not row:
        disabled("select a dataset.")
        return view
    if not row.get("trainable"):
        disabled(f"{row.get('name')} is not trainable: {r.not_trainable_short(row)}.")
        return view
    if not drawer:
        view["start"] = view["cont"] = True
        view["reason"] = line(NO_ICON, "Stop disabled: no training is running.")
        return view
    # The drawer is open: its start's reason is beside it, in its footer.
    why = continue_refusal(mode.get("parent") or {}, ws, operation=operation_of(mode))
    view["resume"] = why is None
    view["reason"] = line(NO_ICON, "Start disabled: close Continue… to start a new run.")
    if why:
        view["drawer_reason"] = line(NO_ICON, f"{action} disabled: {why}")
    return view


def _other_workspace(item: Mapping[str, Any], ws: Mapping[str, Any]) -> Optional[str]:
    """Why a listed parent is not the run bar's workspace's (the list was read
    for another workspace than the one selected now), or None."""
    listed = item.get("workspace")
    if listed is None or listed == ws.get("name"):
        return None
    return (f"the run list shows {listed}’s runs, not {ws.get('name')}’s; refresh the list "
            "(Refresh beside it).")


#: **What the last check refused**, as the session's parent record keeps it
#: (``refused``, `_refusal_kind`): a setting the operator edited or entered
#: (the page's own value check, or the preview's ``invalid_settings``: a
#: setting not read or off the list, a value, a rule across settings, the hop
#: list), the parent changed since its fingerprint was shown (the preview's
#: ``changed``), the parent itself (the plan, its record, the device), or no
#: parent the run list holds (`_parent_path` found none to read).
REFUSED_SETTINGS, REFUSED_CHANGED, REFUSED_PARENT = "settings", "changed", "parent"
REFUSED_UNLISTED = "unlisted"


def _refusal_kind(preview: Mapping[str, Any]) -> Optional[str]:
    """What a refused preview refused (`REFUSED_SETTINGS`, `REFUSED_CHANGED`,
    `REFUSED_PARENT`), or None when it can go ahead."""
    if preview.get("ok"):
        return None
    if preview.get("invalid_settings"):
        return REFUSED_SETTINGS
    return REFUSED_CHANGED if preview.get("changed") else REFUSED_PARENT


def continue_refusal(parent: Mapping[str, Any], ws: Mapping[str, Any], *,
                     operation: str = CONTINUE) -> Optional[str]:
    """Why the drawer's start (Continue Training, or Start branch for the
    `operation` Branch) is disabled for the chosen parent, or None: said
    beside the start, so by its cause and what to do (the skill's §9; the
    B1c review: one sentence blamed the checkpoint for an edit the check
    refused). A refused check by what it refused (`_refusal_kind`): a setting
    the operator edited, to change and check again; the parent changed since
    its fingerprint was shown, with the action for where it was shown
    (`_changed_parent_action`); no checkpoint the run list holds, with what
    to do for where it was chosen; else the parent itself, which the check
    says why. A branch's data is its parent's as a continued run's is
    (BACKLOG 23), so another dataset is refused for both."""
    if not parent.get("chosen"):
        return "choose a checkpoint first."
    other = _other_workspace(parent, ws) if parent.get("source") == CONTINUE_FROM_LIST else None
    if other:
        return other
    refused = parent.get("refused")
    if refused == REFUSED_SETTINGS:
        return "the check refuses a setting you edited; change it, then check again."
    if refused == REFUSED_CHANGED:
        return (f"the checkpoint changed since it was chosen. "
                f"{_changed_parent_action(parent.get('source'))}")
    if refused == REFUSED_UNLISTED:
        return ("the typed path is not a checkpoint the run list holds. Type one it holds, or "
                "Refresh the list if its run is new." if parent.get("source") == CONTINUE_FROM_PATH
                else "the checkpoint is no longer in the run list. Refresh the list, then "
                     "choose one.")
    if refused:
        return f"the check says this checkpoint cannot be {OPERATION_VERBS[operation][1]}."
    recorded, current = parent.get("fingerprint"), workspace_fingerprint(ws)
    if r.is_digest(recorded) and current and recorded != current:
        return "the parent’s dataset is not this workspace’s current one."
    return None


# =============================================================================
# The error banner (red: a failed run; amber: PyG's slow fallback)
# =============================================================================
def _get_pyg_warning() -> Optional[str]:
    """PyG native-extension fallback warning, probed in this process (cheap
    after its first, logged, probe: `pyg_native_check` caches it)."""
    try:
        from src.utils.pyg_native_check import get_fallback_warning

        return get_fallback_warning()
    except Exception as exc:  # torch stack may be unavailable in this process
        logger.debug("PyG native check unavailable in UI process: %s", exc)
        return None


#: A logging line at ERROR or CRITICAL ("<date> <time> - <logger> - ERROR -
#: <message>", train_model.py's format, or logging's default "ERROR:<logger>:
#: <message>"): its message.
_LOG_ERROR = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d+)? - \S+ - (?:ERROR|CRITICAL) - (?P<m>\S.*)$"
    r"|^(?:ERROR|CRITICAL):[\w.]*:(?P<d>\S.*)$")
#: The last line of a Python traceback: an exception's (dotted) class name,
#: then ": message" or nothing, not indented, under the traceback's indented
#: lines.
_EXCEPTION_LINE = re.compile(r"^(?:[A-Za-z_]\w*\.)*[A-Z]\w*(?::\s+\S.*)?$")


def last_error_line(lines: Sequence[str]) -> Optional[str]:
    """The last error line a run printed in `lines` (its output tail), or None:
    a logging line at ERROR or CRITICAL (its message), or the exception line
    that ends a Python traceback (as printed: the class and the message),
    whichever came last. Nothing is guessed from other lines."""
    found = None
    previous = ""
    for line in lines:
        text = line.rstrip()
        logged = _LOG_ERROR.match(text)
        if logged:
            found = (logged.group("m") or logged.group("d")).strip()
        elif (text and previous[:1].isspace() and _EXCEPTION_LINE.match(text)):
            found = text
        previous = line
    return found


def _log_where(run: Optional[str], run_dir: Optional[Path]) -> str:
    """Where a failed run's log is, said only when it exists: its training log
    (``logs/train_*.json``) named; else that none is there (no log in its
    ``logs/``; or no run directory, as a run refused before it made its run
    directory leaves it). Without `run_dir` (not known here) nothing is said of the log."""
    if not run:
        return ""
    head = f'run <code>{r.e(run)}</code>'
    if run_dir is None:
        return f'<span class="shep-banner-where">{head}</span>'
    try:
        exists = Path(run_dir).is_dir()
    except OSError:
        exists = False
    if not exists:
        said = "no run directory, so no log"
    else:
        log = training_manager.metrics_log_file(Path(run_dir))
        said = (f'its log: <code>logs/{r.e(log.name)}</code>' if log is not None
                else 'no log in its <code>logs/</code>')
    return f'<span class="shep-banner-where">{head} · {said}</span>'


def _as_is(text: str) -> str:
    return text


def read_only_scrub(root: Optional[Path]) -> Callable[[str], str]:
    """A read-only request's redaction of a printed line (`redact`: no
    absolute path), its leading indentation kept: `redact` joins whitespace,
    and a traceback's lines and its ``^^^`` markers line up by their indent."""
    def scrub(text: str) -> str:
        text = str(text)
        return text[:len(text) - len(text.lstrip())] + redact(text, root)
    return scrub


def _render_error_block(message: str, *, headline: str = "Training failed",
                        run: Optional[str] = None, run_dir: Optional[Path] = None,
                        scrub: Callable[[str], str] = _as_is) -> str:
    """The red block: the headline, the run and where its log is (only when
    there is one), Dismiss; under the headline the last error line the run
    printed (`last_error_line`), so the cause is read without scrolling; and
    the output tail (monospace, escaped) in a disclosure, open as sent; the
    banner's script closes it on a wide window under `BANNER_OPEN_HEIGHT` px
    tall, where it would squeeze the charts (`_BANNER_JS`). Every text the run
    printed goes through `scrub` (a read-only request's redaction), the cause
    as the tail's lines, and then is escaped: the redaction reads text, never
    the markup around it."""
    first, _sep, tail = str(message).partition("\n\n--- Last output ---\n")
    every = tail.splitlines() if tail else []
    lines = [scrub(line) for line in every[-BANNER_TAIL_LINES:]]
    cause = last_error_line(every)
    cause_html = (f'<div class="shep-banner-cause"><span class="shep-banner-k">Last error '
                  f'printed:</span> <span class="shep-banner-v">{r.e(scrub(cause))}</span></div>'
                  if cause else "")
    count = f'{len(lines)} line{"" if len(lines) == 1 else "s"}'
    body = ('<details class="shep-banner-more" open><summary>Last output '
            f'<span class="shep-banner-n">({count})</span></summary>'
            f'<pre class="shep-banner-tail">{r.e(chr(10).join(lines))}</pre></details>'
            if lines else "")
    # The condition first, by the shared mark (an error's ✕, red: the skill's §6).
    said = r.status_mark("error", f"{headline}: {scrub(first.strip())}")
    return ('<div class="shep-banner shep-banner-red" role="alert">'
            f'<div class="shep-banner-hd"><span class="shep-banner-h">{said}</span>'
            f'{_log_where(run, run_dir)}'
            '<button type="button" class="shep-btn shep-banner-x" data-act="dismiss">× Dismiss</button>'
            f'</div>{cause_html}{body}</div>')


def _render_warning_block(message: str, scrub: Callable[[str], str] = _as_is) -> str:
    """The amber block for a non-fatal advisory: the condition first, by the
    shared mark (a condition to watch's ▲: the skill's §6)."""
    return ('<div class="shep-banner shep-banner-amber" role="status">'
            f'<div class="shep-banner-hd"><span class="shep-banner-h">'
            f'{r.status_mark("warn", scrub(str(message)))}</span>'
            '<button type="button" class="shep-btn shep-banner-x" data-act="dismiss">× Dismiss</button>'
            '</div></div>')


def _build_error_banner_payload(
    status_info: Mapping[str, Any], pyg_warning: Optional[str], *, run: Optional[str] = None,
    run_dir: Optional[Path] = None, scrub: Callable[[str], str] = _as_is,
) -> Tuple[Optional[str], Optional[str]]:
    """``(html, key)`` for the banner, or ``(None, None)`` when nothing must
    show. ``key`` names the content, so a session's dismissal holds until it
    changes (a new or different error shows again). `scrub` redacts each text
    before it is escaped (a read-only request: no absolute path)."""
    blocks: List[str] = []
    key_parts: List[str] = []
    error_msg = status_info.get("error_message")
    if error_msg:
        epoch = _epochs(status_info)
        headline = f"Training failed at {epoch}" if epoch else "Training failed"
        blocks.append(_render_error_block(error_msg, headline=headline, run=run, run_dir=run_dir,
                                          scrub=scrub))
        key_parts.append("ERR:" + str(error_msg))
    if pyg_warning:
        blocks.append(_render_warning_block(pyg_warning, scrub))
        key_parts.append("PYG:" + str(pyg_warning))
    if not blocks:
        return None, None
    return "".join(blocks), "||".join(key_parts)


# =============================================================================
# The run view
# =============================================================================
def _shown_path(path: Any) -> str:
    """A server path as an operator sees it here, only as the shared
    redaction gives one path (`workspaces.shown_path`; D7): relative to the
    workspaces root when it is inside it (``<workspace>/runs/…``, its names
    whole), and any other absolute path as "a path outside the workspaces
    root", never the path itself."""
    return shown_path(path, workspaces_root()[0])


def _snapshot_sentence(snapshot: Optional[Dict[str, Any]]) -> str:
    """Where a run can be continued from, in one sentence (contract M3a); the
    checkpoint named relative to the workspaces root."""
    if not snapshot:
        return "No checkpoint was saved, so nothing can be continued."
    shown = _shown_path(snapshot.get("checkpoint", ""))
    if not snapshot.get("readable", True):
        return f"The latest checkpoint {shown} cannot be read ({snapshot.get('reason')})."
    epoch, target = snapshot.get("completed_epochs"), snapshot.get("target_epochs")
    where = f"epoch {epoch} of {target}" if target else f"epoch {epoch}"
    end = snapshot.get("end")
    if end == "finished":
        return f"Finished ({where}): {shown}. To train further, use Continue…"
    if end == "stopped_early":
        return f"Stopped early at {where} ({snapshot.get('stop_reason')}); it cannot be continued."
    label = "Paused at" if end == "paused" else "Latest resumable checkpoint:"
    return f"{label} {where}, {shown}."


class RunRef(Dict[str, Any]):
    """A run the view can show: ``id`` (what the chart's token, its server
    functions and the Viewing select name it by: ``<workspace>/<arch>/<run>``,
    `resolve_run`'s ids), ``label`` ("gatv2/<run>"), ``path``, ``workspace``,
    ``job`` (True for the run the job runner started last, the run bar's
    training job) and ``latest`` (True for its workspace's latest listed run,
    when it is not that job's)."""


def workspace_of_run(run_dir: Any) -> Optional[str]:
    """The listed workspace a run directory is inside, by the name the server
    lists it under, or None (one listing of the root: `resolve_workspaces`)."""
    from src.api.services.server_lists import ServerListRefused, resolve_workspaces

    try:
        directories = resolve_workspaces()
    except ServerListRefused:
        return None
    try:
        resolved = Path(run_dir).resolve()
    except OSError:
        resolved = Path(run_dir)
    return next((listed for listed, directory in directories.items()
                 if directory in resolved.parents), None)


def _runner_run(status: Mapping[str, Any], cache: Dict[str, Any]) -> Optional[RunRef]:
    """The run the job runner started last, named by its listed workspace
    (``<workspace>/<arch>/<run>``) or, when the workspace is not listed, by
    its job (``job:<id>``). The workspace's name is looked up once per run
    and session (`workspace_of_run`)."""
    run_dir = training_manager.run_dir
    if run_dir is None or (status.get("status") or "idle") == "idle":
        return None
    key = str(run_dir)
    named = cache.get("runner_ws")
    if not named or named[0] != key:
        named = (key, workspace_of_run(run_dir))
        cache["runner_ws"] = named
    arch = Path(run_dir).parent.name
    label = f"{arch}/{Path(run_dir).name}"
    run_id = f"{named[1]}/{label}" if named[1] else f"job:{status.get('job_id')}"
    return RunRef(id=run_id, label=label, path=Path(run_dir), workspace=named[1], job=True,
                  latest=False)


def _runs_of(ws: Mapping[str, Any], cache: Dict[str, Any]) -> List[str]:
    """The run bar's workspace's runs ("<arch>/<run>", latest first): first
    as its selection read them, then listed again at most every
    `LATEST_READ_SECONDS` per session, so a run started elsewhere (the CLI,
    the API, another browser) is offered and shown once it writes a
    checkpoint, while the page stays open."""
    name = ws.get("name")
    if not name:
        return []
    now = datetime.now().timestamp()
    held = cache.get("runs")
    if held and held[0] == name and now - held[1] < LATEST_READ_SECONDS:
        return list(held[2])
    if held and held[0] == name:
        from src.api.services.server_lists import ServerListRefused

        try:
            runs = _listed_runs(_workspace_path(name))
        except ServerListRefused:
            runs = []
    else:
        runs = list(ws.get("runs") or [])
    cache["runs"] = [name, now, runs]
    return list(runs)


def _listed_run(ws: Mapping[str, Any], label: str, runs: Sequence[str]) -> Optional[RunRef]:
    """The run bar's workspace's run `label` ("<arch>/<run>"), as the server
    finds it again (`resolve_run`), or None when it is no longer listed."""
    name = ws.get("name")
    path = resolve_run(f"{name}/{label}") if name else None
    if path is None:
        return None
    return RunRef(id=f"{name}/{label}", label=label, path=path, workspace=name, job=False,
                  latest=bool(runs) and runs[0] == label)


def _current_run(status: Mapping[str, Any], ws: Mapping[str, Any],
                 cache: Dict[str, Any]) -> Optional[RunRef]:
    """The run the view follows unless the session chose another: the run
    the job runner started last (a Fine-tune child included), or, before any
    since the server started, the run bar's workspace's latest run."""
    runner = _runner_run(status, cache)
    if runner is not None:
        return runner
    runs = _runs_of(ws, cache)
    return _listed_run(ws, runs[0], runs) if runs else None


def _job_id(status: Mapping[str, Any]) -> Optional[str]:
    """The training job the run bar shows (None before any since the server
    started)."""
    return status.get("job_id") if (status.get("status") or "idle") != "idle" else None


def _chosen_run(viewing: Optional[Mapping[str, Any]], status: Mapping[str, Any],
                ws: Mapping[str, Any]) -> Optional[str]:
    """The run id the session chose in the Viewing select (`_on_view_run`),
    while the choice holds: made for the run bar's workspace of now, with no
    training job started since (a job started is shown as it starts, from
    this page or any other entry); else None, and the view follows the run
    it would show without a choice (`_current_run`). The choice is compared
    with the workspace shown now, not ended by another shown in between: back
    on the workspace it was made for, with no job started since, it is shown
    again (only `_on_view_run` writes it)."""
    viewing = viewing or {}
    run = viewing.get("run")
    if not isinstance(run, str) or not run:
        return None
    if viewing.get("ws") != ws.get("name") or viewing.get("job") != _job_id(status):
        return None
    return run


def viewed_run(viewing: Optional[Mapping[str, Any]], status: Mapping[str, Any],
               ws: Mapping[str, Any], cache: Dict[str, Any]) -> Optional[RunRef]:
    """**The run the view shows**: the session's choice while it holds
    (`_chosen_run`) and its run is still listed (or is the job runner's),
    else the run it follows (`_current_run`). Only the view's pieces are drawn
    from it: the header, the chart token and Export CSV; the run bar and its
    status pill are always the training job's."""
    chosen = _chosen_run(viewing, status, ws)
    if chosen is not None:
        runner = _runner_run(status, cache)
        if runner is not None and chosen == runner["id"]:
            return runner
        prefix = f"{ws.get('name')}/"
        if chosen.startswith(prefix):
            ref = _listed_run(ws, chosen[len(prefix):], _runs_of(ws, cache))
            if ref is not None:
                return ref
    return _current_run(status, ws, cache)


def view_choices(ws: Mapping[str, Any], runs: Sequence[str],
                 run: Optional[RunRef]) -> List[Tuple[str, str]]:
    """The Viewing select's options: the run bar's workspace's runs, latest
    first, each named by the one rule of a run's option (`model_list.run_options`:
    short, name first) and valued by its full id; the run shown first when it
    is not among them (the training job's run before it is listed, or in a
    workspace the run bar no longer shows)."""
    entries: Dict[str, Tuple[str, str]] = {}
    if run is not None and run["id"] not in {f"{ws.get('name')}/{label}" for label in runs}:
        entries[run["id"]] = tuple(str(run["label"]).split("/", 1))
    for label in runs:
        entries[f"{ws.get('name')}/{label}"] = tuple(label.split("/", 1))
    return _model_list().run_options(entries)


#: How the run viewed stands to the run bar's training job (`view_relation`):
#: said once, beside the Viewing select; the run bar and its pill always show
#: the job.
RELATION_RUNNING = "the running job"
RELATION_LAST_JOB = "the last job"
RELATION_PAST_WHILE = "a past run, while another runs"
RELATION_LATEST = "the latest run"
RELATION_PAST = "a past run"


def view_relation(run: RunRef, live: bool) -> str:
    """How the run viewed stands to the training job: it is the running job,
    or the last job once that ended; another run is a past run, said "while
    another runs" while a job runs, else "the latest run" when it is its
    workspace's latest."""
    if run.get("job"):
        return RELATION_RUNNING if live else RELATION_LAST_JOB
    if live:
        return RELATION_PAST_WHILE
    return RELATION_LATEST if run.get("latest") else RELATION_PAST


def resolve_run(run_id: Any) -> Optional[Path]:
    """The directory of the run the server lists as `run_id`, or None: a
    listed workspace's run, found by listing its ``runs/`` and then the
    directory of that name again (names compared, never joined onto a path,
    so ``.`` or ``..`` names nothing), inside that workspace, whatever the
    directory's name (the listing reads every one); or ``job:<id>``, the job
    runner's latest training job."""
    if not isinstance(run_id, str) or len(run_id) > 400:
        return None
    if run_id.startswith("job:"):
        status = training_manager.get_job_status_word("training")
        job_id = training_manager.get_status().get("job_id") if status != "idle" else None
        return training_manager.run_dir if job_id and run_id == f"job:{job_id}" else None
    parts = run_id.split("/")
    if len(parts) != 3 or not all(parts):
        return None
    workspace, arch, run = parts
    from src.api.services.server_lists import ServerListRefused, resolve_workspace
    from src.utils.checkpoint_paths import RUNS_DIRNAME

    try:
        directory = resolve_workspace(workspace)
        layouts = [entry for entry in (directory / RUNS_DIRNAME).iterdir() if entry.name == arch]
        entries = list(layouts[0].iterdir()) if layouts else []
    except (ServerListRefused, OSError):
        return None
    for entry in entries:
        if entry.name != run:
            continue
        try:
            resolved = entry.resolve()
            inside = directory in resolved.parents and resolved.is_dir()
        except OSError:
            inside = False
        return entry if inside else None
    return None


def _snapshot_for(run: RunRef, cache: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The run's latest saved state (`TrainingManager.latest_snapshot`, which
    reads its ``last.pt``), read again only when the file changed since this
    session last read it."""
    from src.utils.checkpoint_paths import run_checkpoint_dir

    last = run_checkpoint_dir(run["path"]) / "last.pt"
    try:
        stat = last.stat()
        key = [str(last), stat.st_mtime_ns, stat.st_size]
    except OSError:
        key = [str(last), None, None]
    held = cache.get("snapshot")
    if held and held[0] == key:
        return held[1]
    snapshot = training_manager.latest_snapshot(run["path"]) if key[1] is not None else None
    cache["snapshot"] = [key, snapshot]
    return snapshot


def _used_while_live(run: RunRef, cache: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """While the job runner's run is live: only what the run view's header
    needs, the dataset fingerprint the run records. It is fixed for a run, so
    it is read from the run's first saved state once per run and session and
    then kept (``last.pt`` is rewritten every epoch and read whole); the live
    snapshot sentence names the newest file without reading it."""
    held = cache.get("used")
    if held and held[0] == run["id"]:
        return {"readable": True, "dataset_fingerprint": held[1]}
    snapshot = _snapshot_for(run, cache)
    if snapshot and snapshot.get("readable", True):
        cache["used"] = [run["id"], snapshot.get("dataset_fingerprint")]
    return snapshot


def _newest_checkpoint(run_dir: Path) -> Optional[Path]:
    """The newest checkpoint file of a run, by its time (nothing is read)."""
    from src.utils.checkpoint_paths import run_checkpoints

    best, when = None, -1.0
    for path in run_checkpoints(run_dir):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > when:
            best, when = path, mtime
    return best


def _dataset_relation(ws: Mapping[str, Any], *, operator: bool,
                      snapshot: Optional[Mapping[str, Any]], live: bool) -> Tuple[str, str]:
    """How the run's recorded dataset stands to the run bar's: ``(tag, row)``,
    the header's short tag (always shown: a mismatch, a missing record or a
    refusal changes what the curves mean) and the Details row with the
    fingerprint itself. Reading it opens the run's checkpoint: an operator's
    only."""
    if not operator:
        return (f'<span class="shep-tag shep-tag-grey">{LOCK_ICON} dataset not shown · operator '
                'control</span>',
                '<span class="shep-muted">not shown: reading the checkpoint is an operator '
                'control</span>')
    if snapshot is None:
        words = "recorded at the first checkpoint" if live else "not recorded (no checkpoint)"
        return f'<span class="shep-muted">dataset {words}</span>', f'<span class="shep-muted">{words}</span>'
    if not snapshot.get("readable", True):
        return (r.warn_chip("checkpoint cannot be read"),
                '<span class="shep-muted">not read (the checkpoint cannot be read)</span>')
    recorded, current = snapshot.get("dataset_fingerprint"), workspace_fingerprint(ws)
    row = r.fingerprint_value(recorded, digits=ws.get("digits") or r.SHORT_HEX,
                              what="dataset fingerprint")
    # The skill's §6: "not recorded" only when the run recorded none; a run
    # that recorded one beside a run-bar dataset without a fingerprint cannot
    # be compared, which is another state.
    if not r.is_digest(recorded):
        return r.status_mark("neutral", f"dataset {r.NOT_RECORDED.lower()}"), row
    if not current:
        return (r.status_mark("neutral", f"dataset {r.CANNOT_BE_VERIFIED.lower()}: the run "
                                         "bar's dataset has no fingerprint"), row)
    if recorded == current:
        return f'<span class="shep-tag shep-tag-ok">{r.CHECK} same dataset as the run bar</span>', row
    return r.warn_chip("not the selected dataset"), row


def run_head_html(run: Optional[RunRef], ws: Mapping[str, Any], *, operator: bool,
                  snapshot: Optional[Mapping[str, Any]], live: bool = False,
                  relation: Optional[str] = None, state: str = "",
                  continuing: Optional[str] = None, operation: str = CONTINUE) -> str:
    """The run view's header, beside its Viewing select (which names the run,
    short and name first: `view_choices`), one line where it fits: how the
    run viewed stands to the training job (`relation`, `view_relation`'s
    words; the run bar and its pill always show that job), how its recorded
    dataset stands to the run bar's (`_dataset_relation`), while Continue… is
    open with a listed checkpoint chosen whether this is the run continued
    (or, for the drawer's `operation` Branch, branched) from (`continuing`,
    its "<architecture>/<run>": the drawer's parent and the run viewed are
    different runs unless chosen so), the run's state
    (`state`, `run_snapshot_html`), and Details ›. The rest, read less often
    (the run's directory, its checkpoint and logs, the dataset fingerprint it
    used; its workspace when it is not the run bar's), is in Details. `live`:
    the run viewed is the training job, running. With no run, nothing: the
    chart region says what will appear and how (`NO_RUN_YET`, `NO_RUN_YET_READ_ONLY`)."""
    if run is None:
        return ""
    label = str(run["label"])
    tag, used = _dataset_relation(ws, operator=operator, snapshot=snapshot, live=live)
    said = f'<span class="shep-rv-rel">{r.e(relation)}</span>' if relation else ""
    if continuing:
        done = OPERATION_VERBS[operation][1]
        said += (f'<span class="shep-tag shep-tag-accent">the run {done} from</span>'
                 if continuing == label else
                 f'<span class="shep-tag shep-tag-grey">not the run {done} from</span>')
    rows = [("Run", f"<code>{r.e(label)}</code>")]
    if run.get("workspace") and run.get("workspace") != ws.get("name"):
        # Said here only when it is not the run bar's (which names its own).
        rows.append(("Workspace", r.e(run["workspace"])))
    if operator:
        if live:
            newest = _newest_checkpoint(run["path"])
            rows.append(("Newest checkpoint", f"<code>{r.e(_shown_path(newest))}</code>" if newest
                         else '<span class="shep-muted">none written yet</span>'))
        elif snapshot and snapshot.get("checkpoint"):
            rows.append(("Checkpoint", f"<code>{r.e(_shown_path(snapshot['checkpoint']))}</code>"))
    rows.append(("Logs", "the run directory’s <code>logs/</code>"))
    rows.append(("Dataset used", used + help_button("training-used-dataset",
                                                    "Used dataset fingerprint")))
    details = ('<button type="button" class="shep-link shep-rv-details" data-act="details" '
               'aria-expanded="false" aria-controls="shep-train-run-details">Details ›</button>'
               '<div class="shep-pop shep-rv-pop" id="shep-train-run-details" role="dialog" '
               'aria-label="Run details" hidden><div class="shep-pop-hd"><b>Run details</b>'
               '<button type="button" class="shep-pop-x" data-act="close" '
               f'aria-label="Close the run details">×</button></div>{r.summary_list(rows)}'
               '</div>')
    return (f'<div class="shep-rv-hd">{said}{tag}{state}'
            f'<span class="shep-rv-tools">{details}</span></div>')


def _epochs_done(snapshot: Mapping[str, Any]) -> str:
    """"195 of 1000 epochs completed" from a snapshot's progress record."""
    done, target = snapshot.get("completed_epochs"), snapshot.get("target_epochs")
    return f"{done} of {target} epochs completed" if target else f"{done} epochs completed"


def _state(text: str) -> str:
    return f'<span class="shep-rv-snap">{text}</span>'


def run_snapshot_html(run: Optional[RunRef], status: Mapping[str, Any], *, operator: bool,
                      snapshot: Optional[Mapping[str, Any]]) -> str:
    """The run's state in a few words, in the header's line (`run_head_html`;
    the space it took as a row of its own went to the charts): how far it got
    and whether it can be continued, for an operator from its checkpoint's
    record; for a read-only request its state and that the rest is an
    operator's. The training job's state (`status`) is said only of the job's
    own run; a past run's is its checkpoint's. The checkpoint's and the logs'
    paths are in the header's Details. With no run, nothing."""
    if run is None:
        return ""
    s = (status.get("status") or "idle") if run.get("job") else "past"
    live = s in ("running", "stopping")
    try:
        exists = Path(run["path"]).is_dir()
    except OSError:
        exists = False
    ended = pill_state(status)[1].split(" · ")[0].capitalize()
    if not exists and not live:
        # A run refused before it made its run directory (e.g. "--device
        # auto requires CUDA" on a host without it): there is no
        # checkpoint and no log to point to (for any request: a directory's
        # existence is not checkpoint inspection).
        state = (f"{ended} ({_epochs(status)}). " if _epochs(status) else f"{ended}. ") \
            if s in ("completed", "stopped", "failed") else ""
        return _state(f'{r.e(state)}No run directory exists for this run, so it has no '
                      'checkpoint and no log.')
    if live:
        epochs = _epochs(status)
        return _state(f'Running{" · " + r.e(epochs) if epochs else ""}')
    if not operator:
        state = (f"{ended} · {_epochs(status)}" if s in ("completed", "stopped", "failed")
                 and _epochs(status) else "")
        return _state(f'{r.e(state)}{" · " if state else ""}checkpoint details are an '
                      'operator control')
    if not snapshot:
        return _state("No checkpoint saved, so nothing can be continued")
    if not snapshot.get("readable", True):
        return _state(f'The latest checkpoint cannot be read ({r.e(snapshot.get("reason"))})')
    end = snapshot.get("end")
    if end == "finished":
        line = f"Finished · {_epochs_done(snapshot)} · to train further, use Continue…"
    elif end == "stopped_early":
        reason = snapshot.get("stop_reason")
        line = (f"Stopped early{f' ({reason})' if reason else ''} · {_epochs_done(snapshot)} · "
                "cannot be continued")
    else:
        line = _epochs_done(snapshot) + (" · can be continued" if snapshot.get("resumable", True)
                                         else "")
    return _state(r.e(line))


# =============================================================================
# The charts: data from the run's metrics log (the server function)
# =============================================================================
#: The four charts, in figure order (2×2): Loss, the Trainer's validation
#: figure ("mrr") and its companions ("hits"), the learning rate. The ids stay
#: whatever the run's kind; the middle two take their titles and series from
#: `VALIDATION_CHARTS`.
CHART_TITLES = {
    "loss": "Loss",
    "lr": "Learning rate (log)",
}
SHARED_AXIS = "epoch (shared axis)"
_POOLED_TOP = re.compile(r"^val_pooled_top(\d+)$")
#: The middle two charts by the run's kind, read from the keys its validation
#: log holds (`validation_kind`): ``(the figure's title, its series, the
#: companions' title, theirs)``, each series ``(label, key)``; the companions
#: by cut-off are filled from the log (`_k_series`). A run that has logged no
#: validation yet is "none": its two charts wait, titled by what they will
#: show for any run. A log with validation entries but neither stage's figure
#: is "other": its two charts say no other figure is charted and that Export
#: CSV has every one, and draw no series; its Loss chart, learning rate and
#: Export CSV (every key) are as for any run.
VALIDATION_CHARTS = {
    "base": ("LP accuracy (validation, Trainer)", (("LP accuracy", "val_lp_accuracy"),),
             "ROC-AUC and AP (validation, Trainer)",
             (("ROC-AUC", "val_lp_roc_auc"), ("AP", "val_lp_average_precision"))),
    "finetune": ("Pooled MRR (validation, Trainer)", (("pooled MRR", "val_pooled_mrr"),),
                 "Pooled top-k (validation, Trainer)", _POOLED_TOP),
    "other": ("No other figure charted: Export CSV has every logged figure", (),
              "No further figure charted", ()),
    "none": ("Validation figure (Trainer)", (), "More validation figures (Trainer)", ()),
}


def validation_kind(validation: Sequence[Mapping[str, Any]]) -> str:
    """Which run a validation log is, by its keys: "finetune" (pooled MRR),
    "base" (LP accuracy), "other" for validation entries holding neither (no
    figure this build charts there), or "none" before the first validation."""
    keys = {str(key) for entry in validation for key in entry}
    if "val_pooled_mrr" in keys:
        return "finetune"
    if "val_lp_accuracy" in keys:
        return "base"
    return "other" if keys - {"epoch"} else "none"


def _k_series(validation: Sequence[Mapping[str, Any]], pattern: Any, words: str
              ) -> List[Tuple[str, str]]:
    """Every cut-off k the log names by `pattern`, in order, as ``(label,
    key)``: "top-1", "top-10"."""
    found = {}
    for entry in validation:
        for key in entry:
            match = pattern.match(str(key))
            if match:
                found[int(match.group(1))] = str(key)
    return [(f"{words}{k}", found[k]) for k in sorted(found)]


def _by_epoch(entries: Sequence[Mapping[str, Any]], *keys: str) -> Dict[int, Any]:
    """{1-indexed epoch: the first of `keys` an entry has} (epochs in the log
    are 0-indexed; the status and the charts count from 1)."""
    out: Dict[int, Any] = {}
    for entry in entries:
        epoch = entry.get("epoch") if isinstance(entry, Mapping) else None
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            continue
        for key in keys:
            if entry.get(key) is not None:
                out[epoch + 1] = entry.get(key)
                break
    return out


def _has_value(values: Sequence[Any]) -> bool:
    return any(charts.rounded(v) is not None for v in values)


#: A chart's own line when it has nothing to draw, in place of its plot and
#: axis label (BACKLOG 69: it kept its whole height, its axis label alone
#: under an empty box), by cause: before the running job's first validation,
#: what its two validation charts will draw and when; a run that ended with
#: no validation logged (stopped or failed before its first, or validating
#: less often than it trained), that it logged none; a figure this run's log
#: does not hold, that it is not recorded there. The whole figure's line,
#: when no chart has anything, is the chart region's (`_chart_token`,
#: `training_chart_series`).
WAITING_FOR_VALIDATION = "Drawn once the run logs its first validation."
NO_VALIDATION_LOGGED = "This run logged no validation."
NOT_IN_THE_LOG = f"{r.NOT_RECORDED} in this run’s log."


def training_charts(history: Mapping[str, Any], *, live: bool = False) -> List[Dict[str, Any]]:
    """The four chart specs from a metrics history (`get_metrics_history`):
    Loss (train, and validation where the run logs one: ``val_loss``, a
    base's ``val_lp_loss``), the Trainer's validation figure and its
    companions as the run's kind logs them (`VALIDATION_CHARTS`: a base's LP
    accuracy, then its ROC-AUC and AP; a task model's pooled MRR, then its
    pooled top-k; for a log holding neither, titles that say no figure is
    charted), and the learning rate on a log axis. A series with no value yet
    is left out, so no chart draws a figure the run does not log; a chart
    left with none says why in its own line (`WAITING_FOR_VALIDATION` while
    the run is `live`, the training job's run still logging, else
    `NO_VALIDATION_LOGGED`; `NOT_IN_THE_LOG`; the generic state's titles say
    it themselves)."""
    train = [e for e in history.get("train") or [] if isinstance(e, Mapping)]
    val = [e for e in history.get("validation") or [] if isinstance(e, Mapping)]
    specs = []

    def spec(chart_id, title, columns, *, y_scale="linear", x_label=None, empty=NOT_IN_THE_LOG):
        epochs = sorted(set().union(*[set(c) for _label, c in columns])) if columns else []
        series = [(label, [col.get(x) for x in epochs]) for label, col in columns]
        series = [(label, values) for label, values in series if _has_value(values)]
        return charts.chart_spec(chart_id, title, epochs if series else [],
                                 series, y_scale=y_scale, x_label=x_label,
                                 empty=None if series else empty)

    def columns(pairs):
        return [(label, _by_epoch(val, key)) for label, key in pairs]

    kind = validation_kind(val)
    figure_title, figure, others_title, others = VALIDATION_CHARTS[kind]
    if not isinstance(others, tuple):
        others = _k_series(val, others, "top-")
    waiting = {"none": WAITING_FOR_VALIDATION if live else NO_VALIDATION_LOGGED,
               "other": None}.get(kind, NOT_IN_THE_LOG)
    specs.append(spec("loss", CHART_TITLES["loss"],
                      [("train", _by_epoch(train, "total", "train_loss")),
                       ("validation", _by_epoch(val, "val_loss", "val_lp_loss"))]))
    specs.append(spec("mrr", figure_title, columns(figure), empty=waiting))
    specs.append(spec("hits", others_title, columns(others), x_label=SHARED_AXIS, empty=waiting))
    specs.append(spec("lr", CHART_TITLES["lr"],
                      [("learning rate", _by_epoch(train, "learning_rate"))],
                      y_scale="log", x_label=SHARED_AXIS))
    return specs


def _rows(history: Mapping[str, Any]) -> int:
    return len(history.get("train") or []) + len(history.get("validation") or [])


#: **The chart region's line when it has nothing to draw**, by cause (the
#: skill's §9: what will appear and the next step, one action at most): no
#: run yet in the run bar's workspace (its next step said to an operator, and
#: who takes it to a read-only request, whose Start is locked); the training
#: job's run before its first epoch is logged; a run that ended without
#: logging one, or without making its run directory (refused first); a run the
#: server no longer lists.
NO_RUN_YET = ("No run in this workspace yet: its curves appear here after its first epoch. "
              "Start one with ▸ Start.")
NO_RUN_YET_READ_ONLY = ("No run in this workspace yet: its curves appear here after its first "
                        "epoch. An operator starts one with ▸ Start.")
WAITING_FOR_EPOCH = "No epoch logged yet: the curves appear here when the run’s first epoch ends."
NO_EPOCH_LOGGED = "This run logged no epoch, so it has no curves: choose another run in Viewing."
NO_RUN_DIR_ENDED = ("This run has no run directory, so it logged no epoch: choose another run "
                    "in Viewing.")
NO_LONGER_LISTED = "This run is no longer listed: choose another run in Viewing."


def _logging_now(path: Path) -> bool:
    """Whether `path` is the run of the training job, running now (its log
    still grows)."""
    if training_manager.get_job_status_word("training") not in ("running", "stopping"):
        return False
    try:
        return training_manager.run_dir is not None and \
            Path(training_manager.run_dir).resolve() == Path(path).resolve()
    except OSError:
        return False


def training_chart_series(data: Any) -> Dict[str, Any]:
    """The chart component's server function: the figure for the run its
    token names, if the server lists that run (`resolve_run`; never a path),
    read from the run's metrics log; when it logged no epoch, the region's
    line says whether one is still to come (`WAITING_FOR_EPOCH`) or not
    (`NO_EPOCH_LOGGED`), and so do its validation charts before a first
    validation (`training_charts`). Not an operator control: the metrics log is not
    checkpoint inspection."""
    t = charts.token_from(data)
    run_id = t.get("run")
    path = resolve_run(run_id)
    if path is None:
        return charts.figure_payload(None, 0, [], empty=NO_LONGER_LISTED)
    history = training_manager.get_metrics_history(path)
    live = _logging_now(path)
    return charts.figure_payload(run_id, _rows(history), training_charts(history, live=live),
                                 empty=WAITING_FOR_EPOCH if live else NO_EPOCH_LOGGED)


def _metrics_csv_text(history: Mapping[str, Any]) -> str:
    """Every metric the log holds, one row per entry and split (epochs from 1)."""
    train = [e for e in history.get("train") or [] if isinstance(e, Mapping)]
    val = [e for e in history.get("validation") or [] if isinstance(e, Mapping)]
    keys = sorted({k for entry in train + val for k in entry} - {"epoch"})
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["epoch", "split", *keys], extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for split, entries in (("train", train), ("validation", val)):
        for entry in entries:
            row = {k: entry.get(k, "") for k in keys}
            epoch = entry.get("epoch")
            row["epoch"] = epoch + 1 if isinstance(epoch, int) and not isinstance(epoch, bool) else ""
            row["split"] = split
            writer.writerow(row)
    return out.getvalue()


def training_chart_csv(data: Any) -> Dict[str, Any]:
    """Export CSV, the component's second server function: the run's whole
    metrics log as CSV text, which the browser saves (no file is written on
    the server)."""
    t = charts.token_from(data)
    run_id = t.get("run")
    path = resolve_run(run_id)
    if path is None:
        return {"filename": None, "csv": ""}
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id).split("/", 1)[-1])
    return {"filename": f"training_metrics_{name}.csv",
            "csv": _metrics_csv_text(training_manager.get_metrics_history(path))}


def _chart_token(run: Optional[RunRef], cache: Dict[str, Any], *,
                 operator: bool, live: bool = False) -> Dict[str, Any]:
    """The component's token for the run viewed: its id and how many rows its
    metrics log holds, counted again only when the log file changed (a stat).
    No run, and a run whose directory does not exist, are not asked for (the
    component would say "no longer listed" of a run that was never listed):
    the token's line says why there is nothing and what comes next
    (`NO_RUN_YET` to an `operator`, `NO_RUN_YET_READ_ONLY` to a request D8
    refuses; `WAITING_FOR_EPOCH` while the training job, `live`, has not made
    it yet; `NO_RUN_DIR_ENDED`)."""
    if run is None:
        return charts.token(None, 0, empty=NO_RUN_YET if operator else NO_RUN_YET_READ_ONLY)
    try:
        exists = Path(run["path"]).is_dir()
    except OSError:
        exists = False
    if not exists:
        return charts.token(None, 0, empty=WAITING_FOR_EPOCH if live else NO_RUN_DIR_ENDED)
    log = training_manager.metrics_log_file(run["path"])
    try:
        stat = log.stat() if log is not None else None
        sig = [run["id"], str(log), stat.st_mtime_ns, stat.st_size] if stat else [run["id"], None]
    except OSError:
        sig = [run["id"], None]
    held = cache.get("rows")
    if held and held[0] == sig:
        rows = held[1]
    else:
        rows = _rows(training_manager.get_metrics_history(run["path"])) if sig[1] else 0
        cache["rows"] = [sig, rows]
    return charts.token(run["id"], rows)


# =============================================================================
# The resources line
# =============================================================================
_IDLE_RESOURCE_HTML = '<div class="shep-rv-res"><b>Resources</b> <span class="shep-muted">reading…</span></div>'


def _poll_resources(location: Any = TRAINING) -> Any:
    """The resources line, on the shell's slow timer while Training is shown."""
    if location != TRAINING:
        return gr.update()
    try:
        return r.resources_html(training_manager.get_system_resources())
    except Exception:  # noqa: BLE001 - the line must not break the page
        logger.exception("The resources could not be read")
        return '<div class="shep-rv-res"><b>Resources</b> <span class="shep-muted">not read</span></div>'


# =============================================================================
# The view: one render for the tick and after every action
# =============================================================================
#: The view's pieces, in output order (`create_training_tab` lists the same).
VIEW_KEYS = ("ws", "dataset", "chip", "pill", "reason", "start", "cont", "stop", "resume",
             "runbar", "page", "run_name", "banner", "head", "view", "charts", "drawer_reason",
             "training_workspace")
#: The pieces that are the session's selection (the run bar's workspace):
#: written by a render only when it changed them (a running job's dataset:
#: `render`'s ``ws_changed``).
SELECTION_KEYS = ("ws", "training_workspace")
#: The one piece the fast poll writes (`_poll`), on every tick, besides the
#: renders: the status pill, drawn from the job runner alone, never from the
#: session's selection. A tick that crosses a render can show an older pill
#: until the next tick.
POLL_KEYS = ("pill",)
#: Every other piece but the selection's own, which a queued render alone
#: writes (they are drawn from the session's selection, ``ws``, ``mode``
#: where it shapes them: the drawer, Stop's confirmation; the banner and the
#: run name with them; and the run view's choice, ``viewing``, for the
#: Viewing select, the header and the chart token): when the poll finds one
#: of them not what the session was last sent, it hands the change over
#: (`CHANGED_ID`).
HANDED_KEYS = tuple(key for key in VIEW_KEYS if key not in SELECTION_KEYS + POLL_KEYS)
#: The reads a render keeps per session that are throttled by time, not by a
#: file's change (the GPU lock's probe, `LOCK_READ_SECONDS`; the workspace's
#: runs, `LATEST_READ_SECONDS`): the fast poll keeps its own (``polled``, which
#: only it writes), so it reads them no more often than a render does; the
#: poll and the render it hands a change over to take each from there when it
#: is newer than the last render's (`_with_polled`).
POLL_READS = ("lock", "runs")
#: The hidden component the fast poll hands a change over through (`_poll`),
#: whose ``.change`` runs the queued render (`_render_changed`): not a
#: ``gr.State``, because the answer to a ``queue=False`` request dispatches a
#: component's ``.change``, never a State's.
CHANGED_ID = "shep-train-changed"


def gpu_view(active, cache: Dict[str, Any], operator: bool) -> Optional[Dict[str, Any]]:
    """The GPU line: the job runner each time; the lock (it probes the flock)
    at most every `LOCK_READ_SECONDS` per session, when the runner holds no job."""
    if active:
        test = training_manager.get_test_status() if active.get("job_type") == "test" else None
        return _gpu_line(active, None, test, operator=operator)
    now = datetime.now().timestamp()
    held = cache.get("lock")
    if not held or now - held[0] >= LOCK_READ_SECONDS:
        try:
            lock = lock_state()
        except Exception:  # noqa: BLE001 - a lock that cannot be read is said, not raised
            lock = {"configured": True, "problem": "it could not be read just now"}
        held = [now, lock]
        cache["lock"] = held
    return _gpu_line(None, held[1], None, operator=operator)


def render(ws: Optional[Mapping[str, Any]], mode: Optional[Mapping[str, Any]],
           cache: Optional[Mapping[str, Any]], dismissed: Optional[Sequence[str]],
           request: Optional[gr.Request],
           viewing: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Every piece of the run bar and the run view, computed now for this
    request, as plain values (`VIEW_KEYS`) plus the session's new cache. The
    run bar (its pill, its reason line, its actions, a failed job's banner) is
    the training job's; the run view is the run viewed (`viewed_run`: the
    session's choice in the Viewing select, `viewing`, while it holds)."""
    ws = dict(ws or empty_workspace())
    mode = dict(mode or {})
    cache = dict(cache or {})
    access = operator_access(request)
    status = training_manager.get_status()
    active = status.get("gpu", {}).get("job") if isinstance(status.get("gpu"), Mapping) else None
    s = status.get("status") or "idle"

    # While a training job runs, the select shows that job's dataset.
    job_run = _runner_run(status, cache)
    in_use = s in ("running", "stopping")
    ws_changed = False
    if in_use and job_run is not None and job_run.get("workspace") \
            and job_run["workspace"] != ws.get("name"):
        ws = workspace_state(job_run["workspace"])
        ws_changed = True
    run = viewed_run(viewing, status, ws, cache)
    # The run viewed is the training job's, running: its log and record grow.
    live = in_use and run is not None and bool(run.get("job"))
    gpu = None if in_use else gpu_view(active, cache, access.allowed)
    view = run_bar_view(status, ws, mode, access=access, active=active, gpu=gpu)
    # What this render read of the GPU slot, for the header's chips
    # (`chips_on_change`): the training state, the runner's job, the lock's
    # line. Not the test run's line, whose elapsed time changes every tick.
    job = active or {}
    cache["gpu_seen"] = [s, job.get("job_type"), job.get("status"), job.get("start_time"),
                         None if active or gpu is None else gpu.get("text")]

    # The run's checkpoint record (the used fingerprint, the snapshot): read
    # for an operator only, and again only when its last.pt changed; while
    # the job runner's run is live, its recorded fingerprint only, once.
    if run is None or not access.allowed:
        snapshot = None
    elif live:
        snapshot = _used_while_live(run, cache)
    else:
        snapshot = _snapshot_for(run, cache)
    # The banner is the failed job's, whatever run is viewed. A read-only
    # request sees no absolute path: each printed text is redacted before it
    # is escaped (redacting the built markup took its closing tags for paths:
    # "</span>" became "<a path outside the workspaces root").
    banner_html, banner_key = _build_error_banner_payload(
        status if s == "failed" else {}, _get_pyg_warning(),
        run=job_run["label"] if job_run else None, run_dir=job_run["path"] if job_run else None,
        scrub=_as_is if access.allowed else read_only_scrub(workspaces_root()[0]))
    if banner_key is None or banner_key in (dismissed or ()):
        banner_html = ""
    cache["banner_key"] = banner_key

    progress = view["progress"]
    bar = (f'<span class="shep-rb-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
           f'aria-valuenow="{round(progress * 100)}" aria-label="Training progress">'
           f'<i style="width:{progress * 100:.1f}%"></i></span>' if progress is not None else "")
    classes = ["shep-runbar"]
    if mode.get("drawer"):
        classes.append("shep-rb-drawer")
    if confirming(status, mode):
        classes.append("shep-rb-confirming")
    page = ["shep-page", "shep-train-page"] + (["shep-train-drawer-open"] if mode.get("drawer") else [])
    pieces = {
        "ws": ws,
        "dataset": (tuple(tuple(c) for c in ws.get("choices") or []), ws.get("name"), not in_use),
        "chip": chip_html(ws, in_use=in_use),
        "pill": pill_html(status),
        "reason": f'<div class="shep-rb-reason">{bar}{view["reason"]}</div>',
        # The gated actions by the one rule (`chips.gated`): no lock while D8
        # allows this request; disabled with the lock while it refuses it,
        # beside the reason line's "Operator controls only: …". The drawer's
        # start is labelled by its operation.
        **{key: gated(start_label(mode) if key == "resume" else label, access.allowed,
                      interactive=view[key])
           for key, label in ACTION_LABELS.items()},
        "runbar": tuple(classes),
        "page": tuple(page),
        "run_name": bool(access.allowed),
        "banner": banner_html or "",
        "head": run_head_html(
            run, ws, operator=access.allowed, snapshot=snapshot, live=live,
            relation=view_relation(run, in_use) if run is not None else None,
            state=run_snapshot_html(run, status, operator=access.allowed, snapshot=snapshot),
            continuing=((mode.get("parent") or {}).get("run_label")
                        if mode.get("drawer") else None),
            operation=operation_of(mode)),
        # The Viewing select: its options and the run viewed (any request:
        # listing a workspace's runs is not checkpoint inspection).
        "view": (tuple(view_choices(ws, _runs_of(ws, cache), run)),
                 run["id"] if run is not None else None),
        "charts": _chart_token(run, cache, operator=access.allowed, live=live),
        "drawer_reason": (f'<div class="shep-drawer-why">{view["drawer_reason"]}</div>'
                          if view["drawer_reason"] else ""),
        "training_workspace": ws.get("name"),
    }
    return {"pieces": pieces, "cache": cache, "ws_changed": ws_changed}


def _update_for(key: str, value: Any) -> Any:
    """A piece as the component's update."""
    if key == "dataset":
        choices, name, interactive = value
        return gr.update(choices=[tuple(c) for c in choices], value=name, interactive=interactive)
    if key in ACTION_LABELS:
        return gr.update(**value)
    if key == "view":
        choices, run = value
        return gr.update(choices=[tuple(c) for c in choices], value=run,
                         interactive=bool(choices))
    if key in ("runbar", "page"):
        return gr.update(elem_classes=list(value))
    if key == "run_name":
        return gr.update(interactive=bool(value))
    return value


def _comparable(value: Any) -> str:
    """A piece as the session's record of what it was sent keeps it."""
    return json.dumps(value, sort_keys=True, default=str)


def view_updates(rendered: Mapping[str, Any], cache_before: Optional[Mapping[str, Any]], *,
                 full: bool = False) -> Tuple[Any, ...]:
    """A queued render's outputs (`VIEW_KEYS`) and the cache: a piece that
    equals what this session was last sent is ``gr.update()`` (nothing sent),
    unless `full`."""
    sent = dict((cache_before or {}).get("sent") or {})
    pieces = rendered["pieces"]
    cache = dict(rendered["cache"])
    out = []
    for key in VIEW_KEYS:
        value = pieces[key]
        if key in SELECTION_KEYS:
            # The session's selection is written only when this render changed
            # it (a running job's dataset); a queued render's turn in the run
            # bar's workspace group keeps it from writing back an older one.
            out.append(value if rendered.get("ws_changed") else gr.update())
            continue
        comparable = _comparable(value)
        if not full and sent.get(key) == comparable:
            out.append(gr.update())
        else:
            out.append(_update_for(key, value))
            sent[key] = comparable
    cache["sent"] = sent
    return (*out, cache)


def _render_view(ws, mode, cache, dismissed, viewing, request: gr.Request):
    """Every action's last step: the run bar and the run view as they are now,
    whole (an action changed what they show)."""
    rendered = render(ws, mode, cache, dismissed, request, viewing)
    rendered["cache"]["handed"] = 0
    return view_updates(rendered, cache, full=True)


def _with_polled(cache: Optional[Mapping[str, Any]], polled: Optional[Mapping[str, Any]]
                 ) -> Dict[str, Any]:
    """The session's cache with each of the fast poll's own reads kept by
    time (`POLL_READS`) in place of the last render's only when it is newer:
    the lock's (``[when, lock]``); the workspace's runs (``[workspace, when,
    runs]``) when they are of the same workspace too (another workspace's are
    not comparable, and the render's stay: `_runs_of` reads a workspace the
    renders have not drawn yet from its selection); and either when the
    render kept none. So neither a tick nor the render it hands over to
    draws an older read than the last render drew. The render after an
    action reads the lock or the runs again when its own read is too old,
    while the tick's own may still be young enough to keep and be from
    before a CLI took the GPU or a run was written."""
    merged = dict(cache or {})
    own = polled or {}
    lock, held = own.get("lock"), merged.get("lock")
    if lock and (not held or lock[0] > held[0]):
        merged["lock"] = lock
    runs, held = own.get("runs"), merged.get("runs")
    if runs and (not held or (runs[0] == held[0] and runs[1] > held[1])):
        merged["runs"] = runs
    return merged


def _render_changed(ws, mode, cache, dismissed, viewing, polled, request: gr.Request):
    """The render the fast poll hands a change over to (`CHANGED_ID`), at its
    turn in the run bar's workspace group: the run bar and the run view from
    the session's selection and the job as they are now, what changed since
    the session was last sent them, and everything every `FULL_SEND_TICKS`
    hand-offs. The reads kept by time (`POLL_READS`) are the newer of the
    poll's own and the last render's (`_with_polled`), so it draws what the
    poll found, never an older read than the last render drew."""
    rendered = render(ws, mode, _with_polled(cache, polled), dismissed, request, viewing)
    handed = int((cache or {}).get("handed") or 0) + 1
    rendered["cache"]["handed"] = handed
    return view_updates(rendered, cache, full=handed % FULL_SEND_TICKS == 0)


def _poll(location, ws, mode, cache, dismissed, viewing, polled, request: gr.Request):
    """The shell's fast timer (``queue=False``), while Training is shown: the
    status pill (`POLL_KEYS`) on every tick, the header's chips on the tick
    that sees the GPU slot change (`chips_on_change`), the hand-off, and its
    own record (``polled``: its reads kept by time and the GPU slot it saw);
    nothing while another page is shown.

    It writes no piece drawn from the session's selection, and no cache (a
    tick's answer is applied whenever it arrives, also after a render that
    read a newer selection): it renders the page as it is now and compares
    it with what the session was last sent, the cache's record, which only
    the renders write. When a piece of `HANDED_KEYS` is not what was sent,
    the GPU slot is not what the last render saw, or a training job runs in
    another workspace than the session's (``ws_changed``), it gives the
    hand-off token (`CHANGED_ID`) a new value, naming what it found, and the
    queued render that token's change runs reads the session's selection
    and the job again at its own turn.

    The reads a render keeps by a file's change (a run's log rows and
    record, the runner's workspace) are the last render's; those it keeps by
    time (`POLL_READS`) are the poll's own, which only it writes, so it
    reads them no more often than a render does; it and the render it hands
    over to take each where it is newer than the last render's
    (`_with_polled`)."""
    idle = (gr.update(),) * 4
    if location != TRAINING:
        return idle
    before = cache or {}
    try:
        rendered = render(ws, mode, _with_polled(before, polled), dismissed, request, viewing)
    except Exception:  # noqa: BLE001 - a tick that raises would stop the page's updates
        logger.exception("The Training page could not be read")
        return idle
    sent = before.get("sent") or {}
    pieces = rendered["pieces"]
    changed = [key for key in HANDED_KEYS if sent.get(key) != _comparable(pieces[key])]
    seen = rendered["cache"].get("gpu_seen")
    if seen != before.get("gpu_seen"):
        changed.append("gpu")
    if rendered["ws_changed"]:
        changed.append("ws")
    # The chips: read again on the tick that sees the GPU slot change.
    chips = chips_on_change((polled or {}).get("gpu_seen"), seen, request)
    # Each hand-off is a new value (its own time), so its .change always fires.
    handoff = {"changed": changed, "at": time.monotonic_ns()} if changed else gr.update()
    own = {key: rendered["cache"][key] for key in POLL_READS if key in rendered["cache"]}
    own["gpu_seen"] = seen
    return pieces["pill"], handoff, chips, own


# =============================================================================
# The handlers
# =============================================================================
def _serving_name(rows: Sequence[Mapping[str, Any]], directories: Mapping[str, Path]
                  ) -> Optional[str]:
    """The listed workspace the served pipeline reads, or None (the status
    read in process, through the shared reader)."""
    from src.webui.components.served_status import read_status, serving_workspace

    try:
        status, _problem = read_status()
        return serving_workspace(status, rows, directories=directories) if status else None
    except Exception:  # noqa: BLE001 - a default, never a reason to fail the page
        logger.debug("The serving workspace could not be read", exc_info=True)
        return None


def _open(ws, preferred):
    """The page shown: the server's list read again (once), the run bar's
    workspace chosen: Datasets' "Use for training", else this session's own,
    else the running job's, else the one serving reads, else the first
    trainable; the cache emptied so the next tick sends everything; the
    resources read."""
    status = training_manager.get_status()
    cache: Dict[str, Any] = {}
    runner = _runner_run(status, cache)
    listed = read_workspaces()
    rows = listed["rows"]
    names = [x["name"] for x in rows]
    candidates = (preferred, (ws or {}).get("name"), runner and runner.get("workspace"))
    choice = next((n for n in candidates if n and n in names), None)
    if choice is None:
        serving = _serving_name(rows, listed["directories"])
        trainable = [x["name"] for x in rows if x.get("trainable")]
        choice = (serving if serving in trainable else
                  trainable[0] if trainable else names[0] if names else None)
    state = workspace_state(choice, listed)
    return (state, dataset_choices_update(state), chip_html(state), choice, {},
            _poll_resources(TRAINING))


def _on_dataset(name, mode, request: gr.Request):
    """A dataset chosen in the run bar: its manifest and record read now (the
    chip shows them), the session's run-bar workspace set (Datasets marks
    it), any check of a parent cleared (the drawer lists the new workspace's
    runs on its next refresh)."""
    state = workspace_state(name)
    mode = dict(mode or {})
    if mode.get("parent"):
        mode["parent"] = {}
    return state, chip_html(state), state.get("name"), mode, ""


def _on_view_run(choice, ws):
    """A run chosen in the Viewing select: the session's view choice, held
    while the run bar shows the workspace of now and no job has started
    since (`_chosen_run`: a job started later is shown as it starts). Only
    the run view follows it
    (its header, the chart token, Export CSV), never the run bar or its pill.
    Nothing is read or changed on the server but the job's id."""
    return {"run": choice if isinstance(choice, str) else None, "ws": (ws or {}).get("name"),
            "job": _job_id(training_manager.get_status())}


def _on_training_workspace(name, ws, mode):
    """Datasets' "Use for training ›" (``ctx.training_workspace``): the run
    bar selects that workspace (nothing changes on the server)."""
    if not name or name == (ws or {}).get("name"):
        return gr.update(), gr.update(), gr.update(), gr.update()
    state = workspace_state(name)
    if state.get("name") is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    mode = dict(mode or {})
    mode["parent"] = {}
    return state, dataset_choices_update(state), chip_html(state), mode


def event_act(evt: Any) -> Optional[str]:
    """The ``act`` a page element's trigger sent (`TRIGGER_JS`, `_POPOVER_JS`)."""
    data = getattr(evt, "_data", None)
    return data.get("act") if isinstance(data, Mapping) else None


def notice_html(text: Any, *, mark: Optional[str] = None) -> str:
    """A handler's notice (an HTML component: the run bar's, the drawer's,
    Fine-tune's): the one place a handler's sentence becomes markup. The
    sentence carries server text (a workspace or run name, a path, the job
    runner's error), so it is escaped whole; ``""`` for no notice. With
    `mark`, a `renderers.STATUS_MARKS` row, it is that state's mark and
    colour with its words (`renderers.status_mark`, the skill's §6): the
    drawer's notice says only refusals, so its is the ✕ of a refusal."""
    said = "" if text is None else str(text)
    if not said:
        return ""
    return f'<span class="shep-notice-t">{r.status_mark(mark, said) if mark else r.e(said)}</span>'


def _on_banner_click(dismissed, cache, evt: gr.EventData):
    """Dismiss: this session hides this banner until its content changes."""
    key = (cache or {}).get("banner_key")
    if event_act(evt) != "dismiss" or not key:
        return gr.update(), gr.update()
    return [*(dismissed or []), key][-20:], ""


def _workspace_path(name: Any) -> Path:
    """The directory of the workspace the server lists as `name` (found in its
    list again: `server_lists.resolve_workspace`)."""
    from src.api.services.server_lists import resolve_workspace

    return resolve_workspace(name)


def _on_start(request: gr.Request, ws, run_name, device, touched, *values):
    """Start: an operator control (D8), checked first on this event's own
    request; then a new base run (`_start`). `touched` is the form's record
    of the fields the operator touched, as this click's own request carries
    it (`touched_record`); ``values`` are the base form's, in
    `base_form_keys` order.

    Returns the notice (`notice_html`): the refusal, inline, or nothing once
    started.
    """
    access = operator_access(request)
    if not access.allowed:
        return notice_html(access.refusal)
    return notice_html(_start(ws, run_name, device, values, touched))


def _start(ws, run_name, device, values, touched) -> str:
    """Start's work after D8, as a plain sentence ("" once started): **a new
    base run** on the workspace the server resolves from its list, through
    the job runner's one start (`start_training`) with
    ``learning_stage="base"``, the run keys, and the settings the operator
    touched that the run reads, the model spec whole when a field of it was
    touched (`base_start_settings`; decision S2), with the full dataset
    fingerprint this session showed. The runner checks the settings again
    (`_new_bundle_run`) and refuses what a base does not take."""
    row = (ws or {}).get("row") or {}
    fingerprint = workspace_fingerprint(ws)
    if not row or not fingerprint:
        return "Not started: select a dataset from the server's list first."
    if not row.get("trainable"):
        return f"Not started: {row.get('name')} is not trainable: {r.not_trainable_short(row)}."
    settings, problems = base_start_settings(
        dict(zip(base_form_keys(), values, strict=True)), touched_keys(touched))
    if problems:
        return "Not started: " + " ".join(problems)
    from src.api.services.server_lists import ServerListRefused

    try:
        directory = _workspace_path(row["name"])
    except ServerListRefused:
        return (f"Not started: {row['name']} is no longer listed by the server; select a dataset "
                "again.")
    config = {"data_dir": str(directory), "learning_stage": "base",
              "device": device or "auto", "run_name": (run_name or "").strip(),
              # torch.compile is a runtime setting (Settings), read once here.
              "compile": bool(load_runtime_settings().get("torch_compile", False)),
              **settings}
    result = training_manager.start_training(config, expected_dataset_fingerprint=fingerprint)
    if result.get("success"):
        return ""
    return not_started(result.get("error"))


def not_started(error: Any) -> str:
    """A start the job runner refused, as a page says it: its sentence after
    "Not started: ", unless the sentence already ends by saying nothing was
    started (the runner's refusals before anything starts do)."""
    text = str(error or "unknown error")
    return text if text.endswith("Nothing was started.") else f"Not started: {text}"


def running_job(status: Mapping[str, Any]) -> Optional[str]:
    """The id of the training job running now (the one Stop would stop), or
    None when none is running."""
    if (status.get("status") or "idle") != "running":
        return None
    return str(status.get("job_id") or "running")


def confirming(status: Mapping[str, Any], mode: Mapping[str, Any]) -> bool:
    """Whether the run bar shows Stop's confirmation: only for the job it was
    asked for, while that job runs. A confirmation left unanswered when its
    job ended (completed, failed, or stopped elsewhere) is never shown for, or
    applied to, the next job."""
    asked = mode.get("confirm")
    return isinstance(asked, str) and asked == running_job(status)


#: Stop's question. It names no epoch: the job goes on training while the
#: question is open, so an epoch read when it was asked would be out of date
#: by the answer (the GB10 acceptance: asked at epoch 183, at 186 on screen,
#: 195 saved). What can be continued is said once the job has stopped, from
#: the run's own checkpoint (the snapshot sentence).
STOP_QUESTION = ("<b>Stop training now?</b> The job stops. The checkpoints it has saved are "
                 "kept and progress not yet saved is lost; if it has saved none yet, this run "
                 "cannot be continued. Once it has stopped, the page says where it can be "
                 "continued from.")


def stop_question(status: Mapping[str, Any]) -> str:
    """Stop's question (`STOP_QUESTION`, HTML), and, when the job is a
    Fine-tune child with children queued after it, that those will not start
    (a stop cancels them: `TrainingManager._stop`). Read from the status the
    question is asked with; nothing else is read."""
    finetune = status.get("finetune") or {}
    queued = [TASK_NAMES.get(child.get("task"), str(child.get("task")))
              for child in finetune.get("children") or [] if child.get("status") == "queued"]
    if not queued:
        return STOP_QUESTION
    return (f"{STOP_QUESTION} The queued Fine-tune children ({r.e(', '.join(queued))}) will "
            "not start.")


#: How a job's process ended, as a stop that came after it says so.
_ENDED = ("completed", "failed", "stopped")


def ask_stop(access: Any, mode: Optional[Mapping[str, Any]], *,
             reaches: Callable[[Mapping[str, Any]], Optional[str]] = running_job,
             what: str = "training job") -> Tuple[str, Dict[str, Any]]:
    """**Stop's first step, the one for every page** (the Training run bar,
    the Fine-tune job card): for a request D8 allowed (`access`, read by the
    page's own handler on its own request), ask before anything happens
    (`stop_question`), reading nothing but the job's status. The
    confirmation is recorded for the job this page's Stop reaches now (its
    id, from `reaches`: the run bar's reaches the training job running,
    whichever page started it; the Fine-tune card's only a running child of
    its Fine-tune start). `what` names that job in the words.

    Returns the confirmation's text and the session's mode.
    """
    mode = dict(mode or {})
    mode["confirm"] = None
    if not access.allowed:
        return notice_html(access.refusal), mode
    status = training_manager.get_status()
    job = reaches(status)
    if job is None:
        return notice_html(f"Nothing to stop: no {what} is running."), mode
    mode["confirm"] = job
    return f'<div class="shep-rb-confirm-q">{stop_question(status)}</div>', mode


def confirm_stop(access: Any, mode: Optional[Mapping[str, Any]], *,
                 reaches: Callable[[Mapping[str, Any]], Optional[str]] = running_job,
                 what: str = "training job") -> Tuple[str, Dict[str, Any]]:
    """**The confirmed Stop, the one for every page**: for a request D8
    allowed (`access`; refused, nothing is stopped). A confirmation asked for
    another job than the one this page's Stop reaches now (`reaches`, as for
    `ask_stop`) stops nothing (it described that job), and nor does any
    confirmation while the job running is one it does not reach (another
    page's). The notice names the queued Fine-tune children the stop
    cancelled, and says nothing was stopped when the job had already ended on
    its own (the runner's ``message``: `TrainingManager._stop`).

    Returns the notice and the session's mode (the confirmation closed).
    """
    mode = dict(mode or {})
    asked = mode.get("confirm")
    mode["confirm"] = None
    if not access.allowed:
        return notice_html(access.refusal), mode
    status = training_manager.get_status()
    job = reaches(status)
    if isinstance(asked, str) and asked != job:
        return notice_html(f"Not stopped: the {what} you asked to stop is no longer the one "
                           "running; click Stop again if this one should stop."), mode
    if job is None and running_job(status) is not None:
        return notice_html(f"Not stopped: no {what} is running."), mode
    result = training_manager.stop_training()
    if result.get("success"):
        cancelled = [TASK_NAMES.get(task, str(task)) for task in result.get("cancelled_children")
                     or ()]
        tail = (f" The queued Fine-tune children ({', '.join(cancelled)}) were cancelled."
                if cancelled else "")
        if result.get("message"):
            # The process had ended on its own before the stop arrived: the
            # stop only cancelled the queued children.
            ended = result.get("status")
            how = f" ({ended})" if ended in _ENDED else ""
            return notice_html(f"Nothing was stopped: the job had already ended on its own{how}."
                               + tail), mode
        return notice_html("Stopped. " + _snapshot_sentence(result.get("outcome")) + tail), mode
    return notice_html(f"Not stopped: {result.get('error', 'unknown error')}"), mode


def _on_stop_request(request: gr.Request, mode):
    """Stop, on the run bar: D8 checked first, on this event's own request,
    as for the Stop itself; then the question (`ask_stop`)."""
    return ask_stop(operator_access(request), mode)


def close_stop_question(mode):
    """Keep training (and any Start: a confirmation left from an earlier job
    is closed)."""
    mode = dict(mode or {})
    mode["confirm"] = None
    return mode


def _on_stop(request: gr.Request, mode):
    """The confirmed Stop, on the run bar: an operator control (D8), checked
    on this event's own request (`confirm_stop`)."""
    return confirm_stop(operator_access(request), mode)


# --------------------------------------------------------------------------- Continue…
#: The two sources of a parent checkpoint. Both go through the same check.
CONTINUE_FROM_LIST = "From run list"
CONTINUE_FROM_PATH = "Manual path"

#: The drawer's title: what it does from a checkpoint (`OPERATION_CHOICES`).
DRAWER_TITLE = "Continue or branch a run"
#: The drawer's numbered steps (the skill's §4), each a control's label (the
#: run and checkpoint selects, and their help's title) or a step's own words
#: (the check row, the footer).
STEP_RUN, STEP_CHECKPOINT, STEP_CHECK, STEP_START = (
    "1. Run", "2. Checkpoint", "3. Check", "4. Start")

#: The check's prompt before a check, and after what it would check changed.
_CONTINUE_PROMPT = "Select a checkpoint, then click <b>Check checkpoint</b>."
#: **Step 3's answer brought into view** once a check (Check checkpoint, or
#: the start, which checks first) has written it: the drawer's body, the one
#: region that scrolls (the skill's §1.1), scrolls the least that shows the
#: check result (the B1c review: at 1850 x 960 the run, checkpoint and
#: details above it left the answer under the body's fold). Nothing moves
#: when it already shows, or when the drawer has closed (a start that ran).
SHOW_CHECK_JS = ("() => { const el = document.getElementById('continue_preview'); "
                 "if (el && el.offsetParent) el.scrollIntoView({block: 'nearest'}); }")
#: The check's prompt once a Branch field was edited after the last check.
_BRANCH_EDITED = ("A setting changed since the check: click <b>Check checkpoint</b> to see the "
                  "plan again, or <b>Start branch</b>, which checks first.")


#: What a checkpoint file holds, as its record says (`checkpoint_kind`).
CHECKPOINT_KINDS = {"training_state": "training state", "weights": "weights only"}

#: **The drawer's own fields of an extension**, by their names in a request
#: (the CLI's and the API's): each field's label, and the words the drawer
#: names it by in a plan's sentence (`page_words`).
EXTEND_WORDS = {"extend_epochs": "Extra epochs", "extend_lr": "Extra-epoch rate"}
_EXTEND_NAMES = re.compile(r"\b(" + "|".join(EXTEND_WORDS) + r")\b")


def page_words(text: Any) -> str:
    """A plan's sentence (the shared run plan's action or refusal, which names
    a request's fields as the CLI and the API do) with the drawer's extension
    fields named as the drawer labels them (`EXTEND_WORDS`): "To train
    further, extend it at a fixed rate with Extra epochs", beside the field so
    labelled, never `extend_epochs`. Plain text: the caller escapes it."""
    return _EXTEND_NAMES.sub(lambda match: EXTEND_WORDS[match.group(1)], str(text or ""))


def _branch_field_names() -> Tuple[str, ...]:
    """**The settings a branch may change, of any stage and task**: the
    closed lists a branch is judged by (`recipe.branch_settings`) of a base
    and of each task, one drawer field each, built once; a checked parent's
    preview names those its own stage and task may change (`changeable`),
    and only those show. In the base form's order, then a task model's own
    settings in the field spec's."""
    every = set(branch_settings("base")).union(
        *(branch_settings("finetune", task) for task in FINETUNE_TASKS))
    return (tuple(name for name in BASE_FIELDS if name in every)
            + tuple(name for name in field_names() if name in every and name not in BASE_FIELDS))


#: **The drawer's Branch fields** (decisions B1-B4), in its order.
BRANCH_FIELDS = _branch_field_names()


def _branch_field_value(name: str) -> Any:
    """A Branch field's value before a check: the one fill's of the first
    run that reads the setting (a base's, else a task model's:
    `run_plan.new_run_values`), which decides its field's kind (`value_field`:
    an optional number None there, early stopping's patience, is a text box
    reading "off"). A check writes the parent's value (`_branch_updates`)."""
    for stage, task in (("base", None), *(("finetune", task) for task in FINETUNE_TASKS)):
        values = new_run_values(stage, task, {})
        if name in values:
            return values[name]
    return None


def _branch_form_value(name: str, value: Any) -> Any:
    """A parent's value as its Branch field holds it, by the field built for
    the setting (`_branch_field_value`, `text_field`): a text box's text
    (`form_value`: "4, 4", "off"), a number box's number (None, empty)."""
    return (form_value(value) if text_field(by_name(name).kind, _branch_field_value(name))
            else value)


def branch_setting_help(name: str) -> Tuple[str, str]:
    """A Branch field's help, ``(title, text)``: its label, what it is (its
    field spec's description, as the base form's help says it), and where
    its value starts and what is sent."""
    return (r.SETTING_LABELS[name],
            f"{by_name(name).description}\n\n"
            "It starts at the parent's value once Check checkpoint has read it. Only a field "
            "you edit is sent; a value other than the parent's is the branch's change, and "
            "its record keeps it.")


def branch_sent(item: Optional[Mapping[str, Any]], touched: Any, values: Sequence[Any]
                ) -> Tuple[Dict[str, Any], List[str]]:
    """**What a branch sends of the drawer's fields** (decisions S2, B1):
    ``(settings, problems)``, by the one rule a form's settings are sent by
    (`sent_settings`): each field the operator touched (`touched`, the
    drawer's record of its fields' input), whatever its value, parsed and
    checked by its field spec, None taken where the parent's stage and task
    take it without the operator (`run_plan.new_run_values`: early stopping
    off); a touched field the run would not read on the form's values is
    hidden and not sent (a rule across settings). None touched is a
    same-settings branch (decision B1). `item` is the chosen parent as the
    run list has it (its stage and task), `values` the fields', in
    `BRANCH_FIELDS` order. The plan judges what is sent again, by name and
    by value, at the check and at the start."""
    item = item or {}
    try:
        references = new_run_values(item.get("learning_stage"), item.get("task"), {})
    except ValueError:
        references = {}
    return sent_settings(dict(zip(BRANCH_FIELDS, values, strict=True)), touched_keys(touched),
                         references)


def _run_label(item: Mapping[str, Any]) -> str:
    """A listed checkpoint's run as its details show it: its shown names,
    ``<architecture>/<run>``."""
    return f"{item.get('architecture')}/{item.get('run')}"


def checkpoint_items(listing: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """A checkpoint listing (`server_lists.list_checkpoints`) as the session
    keeps it: ``({key: item}, problem)``, each item by the list's identity
    (`CheckpointItem.key`: ``<architecture>/<run>/<file>`` for names that are
    text; never its shown names, so no entry hides another, BACKLOG 66), the
    problem redacted (no absolute path)."""
    from dataclasses import asdict

    if not listing.available:
        return {}, redact(listing.problem, workspaces_root()[0])
    return {item.key: asdict(item) for item in listing.items}, None


def _list_parents(ws: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """The run bar's workspace's model records from the server's list (it
    opens every checkpoint: the caller checked D8): ``({key: item},
    problem)``. **Only a model record is offered** (K6 decision 1): a file
    without a model spec, or one whose record cannot be read, is never
    continued or branched from the dashboard; Research › Models lists it."""
    from src.api.services.server_lists import list_checkpoints

    name = ws.get("name")
    if not name:
        return {}, "Select a dataset in the run bar first."
    listed, problem = checkpoint_items(list_checkpoints(name))
    if problem:
        return {}, problem
    parents = {key: item for key, item in listed.items() if item.get("record") == "model"}
    if listed and not parents:
        return {}, (f"None of the {len(listed)} entries listed for this dataset's runs holds a "
                    "model record; a file without a model spec is never continued here "
                    "(Research › Models lists it).")
    return parents, None


def _parents_note(ws: Mapping[str, Any], parents: Mapping[str, Any], problem: Optional[str]) -> str:
    if problem:
        return f'<span class="shep-muted">{r.e(problem)}</span>'
    runs = len(_model_list().runs_of(parents))
    return (f'<span class="shep-muted">server list · {runs} run{"s" if runs != 1 else ""}, '
            f'{len(parents)} checkpoint{"s" if len(parents) != 1 else ""} · latest first</span>')


def parent_details_html(item: Optional[Mapping[str, Any]], operation: str = CONTINUE) -> str:
    """The chosen checkpoint's details, as the one summary list
    (`renderers.summary_list`), saying what its option does not
    (`model_list.checkpoint_words` gives what it is and its progress): its
    file, its kind in the words every page describes a model by
    (`renderers.model_kind_text`: "Base", a task's name, "Disease · frozen
    graph backbone" for the frozen-backbone control; decision F1), what it
    holds, how its stage ended where that limits continuing (for the
    `operation` Continue; a branch is a new run, whatever its parent's end),
    its Trainer validation figure (never a test result), what its run's saves
    were ranked by (`renderers.selection_value`: the recorded monitor and
    direction, or not recorded), its model fingerprint (copyable) and its
    run's directory. A file that cannot be read says why, beside the ▲ of a
    condition to watch. Empty when none is chosen from the list."""
    if not item:
        return ""
    filename = str(item.get("filename") or "")
    rows = [("File", f"<code>{r.e(filename)}</code>")]
    if item.get("problem") or not r.is_digest(item.get("sha256")):
        why = item.get("problem") or "it records no metadata"
        rows.append(("Cannot be read", r.status_mark("warn", why)))
        return f'<div class="shep-ckpt">{r.summary_list(rows)}</div>'
    rows.append(("Kind", r.e(r.model_kind_text(item))))
    kind = item.get("checkpoint_kind")
    rows.append(("Holds", "weights only (cannot be continued or branched)" if kind == "weights"
                 else r.e(CHECKPOINT_KINDS.get(kind or "", "not recorded"))))
    going_on = {"finished": "extends it by the Extra epochs you enter",
                "stopped_early": "not possible: it stopped early"}.get(item.get("end") or "")
    if going_on and kind != "weights" and operation == CONTINUE:
        rows.append(("Continuing", r.e(going_on)))
    rows.append(("Validation", r.e(r.validation_value(r.validation_figure(item),
                                                      read=r.record_read(item)))))
    rows.append(("Selection", r.e(r.selection_value(item))))
    rows.append(("Model fingerprint", r.fingerprint_value(item["sha256"],
                                                          what="model fingerprint")))
    rows.append(("Run", f"<code>{r.e(_run_label(item))}</code>"))
    return f'<div class="shep-ckpt">{r.summary_list(rows)}</div>'


def _cleared_check() -> Tuple[Any, ...]:
    """**The check's outputs once what it described changed** (the
    checkpoint, its source, the operation, the workspace; contract M3a review
    of `094b90d`, P2-2): Extra epochs and the Extra-epoch rate hidden and
    cleared, the prompt, the start's last notice gone, the Branch fields'
    record of what was edited emptied and their box hidden, so nothing
    entered for one parent or operation is sent for another. In the order of
    the page's ``check_cleared`` outputs."""
    hidden = gr.update(visible=False, value=None)
    return hidden, hidden, _CONTINUE_PROMPT, "", [], gr.update(visible=False)


def _on_continue_open(request: gr.Request, ws, mode):
    """Continue…: the drawer over the parameters, listing the run bar's
    workspace's checkpoints (an operator control: D8 first; refused, the
    drawer stays closed and nothing is opened), its operation Continue.

    Returns the notice, the session's mode, the parents (state), the run and
    checkpoint selects, their note, the operation, and the check's outputs
    cleared (`_cleared_check`)."""
    access = operator_access(request)
    mode = dict(mode or {})
    if not access.allowed:
        return (notice_html(access.refusal), mode, *(gr.update(),) * 5,
                *(gr.update(),) * len(_cleared_check()))
    parents, problem = _list_parents(ws or {})
    mode.update(drawer=True, confirm=None, parent={}, operation=CONTINUE)
    runs, checkpoints = _model_list().parents_update(parents)
    first = checkpoints.get("value") if isinstance(checkpoints, dict) else None
    mode["parent"] = _chosen(CONTINUE_FROM_LIST, first, "", parents, ws)
    return ("", mode, parents, runs, checkpoints, _parents_note(ws or {}, parents, problem),
            CONTINUE, *_cleared_check())


def _refresh_checkpoints(request: gr.Request, ws, mode, current):
    """Refresh the parent list: the run bar's workspace's checkpoints, again
    from the server's list. It opens every checkpoint (SPEC_4 C3), so D8 is
    checked first, on this event's own request, as for the checkpoint
    routes; refused, the list is emptied, nothing is opened, and the note says
    why. The chosen option stays while it is listed (`_on_refreshed` then
    follows the file under it)."""
    access = operator_access(request)
    return _parents_listed(access, ws, current)


def _on_refreshed(source, listed, manual, parents, ws, mode):
    """**After a Refresh, the chosen listed checkpoint is the file the list
    holds now** (the B1c review: the option kept its key, so no selection
    event ran, and the session's record, the details and the check stayed
    the old file's while the start read the new list's digest, so it started
    from bytes no check read and no detail showed). When the file under the
    chosen option is not the one the session recorded (another fingerprint,
    or another option), the selection's path records it and shows its
    details (`_on_parent_chosen`), and the check, which read the old file,
    is cleared with what was entered for it (`_cleared_check`); the start
    then sends the fingerprint the details show. The same file: nothing
    changes, the check stands. A typed path's source is not the list's.

    Returns the session's mode, the dataset comparison, the details, then
    the check's outputs."""
    keep = (gr.update(),) * (3 + len(_cleared_check()))
    if source == CONTINUE_FROM_PATH or not (mode or {}).get("drawer"):
        return keep
    record = (mode or {}).get("parent") or {}
    item = (parents or {}).get(listed) if listed else None
    if record.get("key") == listed and record.get("sha256") == (item or {}).get("sha256"):
        return keep
    return (*_on_parent_chosen(source, listed, manual, parents, ws, mode), *_cleared_check())


def _parents_listed(access: Any, ws: Optional[Mapping[str, Any]], current: Any):
    """The parent list for `ws` as D8 allows it: ``(parents, run select,
    checkpoint select, note)``."""
    if not access.allowed:
        return ({}, gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                f'<span class="shep-muted">{r.e(access.refusal)}</span>')
    parents, problem = _list_parents(ws or {})
    return (parents, *_model_list().parents_update(parents, current),
            _parents_note(ws or {}, parents, problem))


def _relist_parents(request: gr.Request, ws, mode):
    """The run bar's workspace changed (chosen by hand, by Datasets' "Use for
    training ›", or followed to a job running in another workspace) while
    Continue… is open: the parent list becomes the new workspace's runs,
    through the Refresh's path (it opens every checkpoint, so D8 first, on
    this event's own request; refused, the list is emptied and the note says
    why), and the comparison and the check's outputs, which were about the
    old list, are cleared (`_cleared_check`). With the drawer closed nothing
    is listed (it lists when opened).

    Returns the parents (state), the run and checkpoint selects, their note,
    the comparison and the check's outputs."""
    access = operator_access(request)
    if not (mode or {}).get("drawer"):
        return (gr.update(),) * (5 + len(_cleared_check()))
    parents, runs, checkpoints, note = _parents_listed(access, ws, None)
    return (parents, runs, checkpoints, note, "", *_cleared_check())


def _on_continue_close(mode):
    mode = dict(mode or {})
    mode.update(drawer=False, parent={})
    return mode


def _chosen(source: str, listed: Any, manual: Any, parents: Mapping[str, Any],
            ws: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The chosen parent as the session records it: whether one is chosen,
    its recorded dataset fingerprint when the list says it, the model
    fingerprint shown for it (``sha256``: the list's; none yet for a typed
    path, whose Check shows one, `_checked`), and the run it continues (for
    the run bar's line)."""
    if source == CONTINUE_FROM_PATH:
        path = _strip_display_prefix(manual or "")
        return {"chosen": bool(path), "source": source, "fingerprint": None,
                "run_label": None, "sha256": None}
    item = (parents or {}).get(listed) if listed else None
    if not item:
        return {"chosen": False, "source": source}
    return {"chosen": True, "source": source, "fingerprint": item.get("dataset_fingerprint"),
            "run_label": _run_label(item), "key": listed,
            "workspace": item.get("workspace"), "sha256": item.get("sha256")}


def compare_html(parent: Mapping[str, Any], ws: Mapping[str, Any]) -> str:
    """The parent's recorded dataset fingerprint against the workspace's
    current one: ✓ the same; when it differs, the ✕ of a refusal (the start
    refuses another dataset, so the drawer's start is disabled: the shared
    status mark, `renderers.status_mark`) with the listed workspace it
    matches, or "matches no listed workspace"; or what is not known yet."""
    if not parent.get("chosen"):
        return ""
    recorded, current = parent.get("fingerprint"), workspace_fingerprint(ws)
    digits = ws.get("digits") or r.SHORT_HEX
    if not r.is_digest(recorded):
        why = ("The parent’s recorded dataset is read when you check it."
               if parent.get("source") == CONTINUE_FROM_PATH and not parent.get("checked")
               else "The parent records no dataset fingerprint; the start checks it against "
                    "this workspace’s data.")
        return f'<div class="shep-cmp shep-cmp-grey">{r.e(why)}</div>'
    now = (f'this workspace now: <b>{r.e(ws.get("name"))}</b> · '
           + r.fingerprint_html("dataset fingerprint", current, digits=digits, copy=False))
    if current and recorded == current:
        # Said once, as a relation: the fingerprint itself is the run bar's.
        return (f'<div class="shep-cmp shep-cmp-ok">{r.CHECK} Trained on the dataset selected '
                'in the run bar</div>')
    listed = ws.get("listed") or []
    trained = r.trained_on_html(recorded, listed, digits=digits)
    built = r.built_text((ws.get("row") or {}).get("build"))
    rebuilt = (f"<p>It was {r.e(built)}. The start refuses a different dataset.</p>"
               if built else "<p>The start refuses a different dataset.</p>")
    head = r.status_mark("error", "The parent’s dataset is not this workspace’s current one")
    return (f'<div class="shep-cmp shep-cmp-red"><div class="shep-check-h">{head}</div>'
            f'<p>parent {trained}</p><p>{now}</p>{rebuilt}</div>')


def _check_html(preview: Mapping[str, Any], source: str, *, differs: bool = False,
                listed: Optional[str] = None, operation: str = CONTINUE) -> str:
    """The check result of the drawer's `operation`, what it does, and the
    settings taken from the checkpoint, as a read-only summary (contract
    M3a-3; not disabled inputs, which screen readers skip), each in the one
    summary list (`renderers.summary_list`) and the one value-words rule
    (`renderers.setting_words`, by the parent's stage and task), never
    Python's repr (BACKLOG 72): a refusal and a dataset the start refuses by
    the ✕ of a refusal (`renderers.status_mark`), the plan's sentence in the
    drawer's words (`page_words`: "Extra epochs", never `extend_epochs`);
    for a branch, the plan as its changes, each its parent's value and its
    own (`renderers.change_value`), or "no setting changed: a new run from
    these weights with a new optimizer" (decision B1); the model spec by its
    differences from its stage's reference, the full spec one click away
    (`renderers.model_spec_html`); and the other settings in a closed
    disclosure, each named in words (`renderers.field_words`, its record name
    the title): a continuation's every one, fixed (to train with others, the
    note points to Branch), a branch's those it keeps (the ones it may change
    are its fields, under the check, each at the parent's value). One level
    each, beside the other: no disclosure inside another."""
    verb, done = OPERATION_VERBS[operation]
    # The file read: named in full for a typed path; for a listed one by its
    # key in the server's list (`listed`, "<architecture>/<run>/<file>"), which
    # the server resolved to the file it read, its details shown above.
    if source == CONTINUE_FROM_PATH or not listed:
        lines = [f'<div><b>Checkpoint</b> (manual path): <code>{r.e(_shown_path(preview.get("parent", "")))}</code></div>'
                 if source == CONTINUE_FROM_PATH else
                 f'<div><b>Checkpoint</b> (run list): <code>{r.e(_shown_path(preview.get("parent", "")))}</code></div>']
    else:
        lines = [f'<div><b>Checked</b> <code>{r.e(listed)}</code> from the run list</div>']
    if preview.get("sha256") and source == CONTINUE_FROM_PATH:
        arch = preview.get("architecture")
        lines.append('<div>' + r.fingerprint_html("model fingerprint", preview["sha256"], copy=False)
                     + (f", architecture <b>{r.e(r.architecture_label(arch))}</b>" if arch else "") + "</div>")
    if not preview.get("ok"):
        # The service's reason names the file it read ("/srv/…/x.pt does not
        # exist."): shown only as the shared redaction gives it, as the
        # path line above is (D7). A refusal of the settings sent names each
        # setting it leads with by its field's label, as the base form does
        # (`labelled_problem`).
        reason = page_words(redact(preview.get("reason"), workspaces_root()[0]))
        if preview.get("invalid_settings"):
            reason = labelled_problem(reason)
        head = r.status_mark("error", f"Cannot {verb}.")
        return (f'<div class="shep-check shep-check-red"><div class="shep-check-h">{head}</div>'
                f'{"".join(lines)}<p>{r.e(reason)}</p></div>')
    # The plan is said once, in the line under the headline (M5-3: the
    # headline repeated it, two lines more in a drawer that must fit).
    able = "it can continue" if operation == CONTINUE else "it can be branched"
    head = (r.status_mark("error", f"Checkpoint read; cannot {verb} on the current dataset.")
            if differs else f'<span class="shep-ok-t">{r.CHECK} Checkpoint read; {able} on '
                            'this workspace.</span>')
    head = f'<div class="shep-check-h">{head}</div>'
    inherited = dict(preview.get("inherited") or {})
    # A setting the checkpoint's record does not hold, which the settings
    # decode gave (a rules-1 parent's freeze), is listed once, with the
    # decode's own words as its value: where it came from is the fact shown.
    decoded = preview.get("decoded") or {}
    order = {name: index for index, name in enumerate(field_names())}

    def ordered(names: Collection[str]) -> List[str]:
        return sorted(names, key=lambda name: (order.get(name, len(order)), name))

    if operation == BRANCH:
        changes = preview.get("changes") or {}
        count = len(changes)
        said = (f"{count} setting{'s' if count != 1 else ''} changed" if changes
                else "no setting changed")
        lines.append(f"<p>Plan read from the checkpoint: {said}: a new run from these weights "
                     "with a new optimizer.</p>")
        if changes:
            names = ordered(changes)
            lines.append(r.summary_list(
                [(r.field_words(name), r.change_value(name, changes[name])) for name in names],
                titles={r.field_words(name): name for name in names},
                classes="shep-check-changes"))
        # The settings it may change are its fields, at the parent's value.
        changeable = set(preview.get("changeable") or ())
        inherited = {name: value for name, value in inherited.items() if name not in changeable}
    else:
        # A rate such as 3e-05 is never split at its hyphen.
        action = re.sub(r"(\d(?:\.\d+)?e-\d+)", r'<span class="shep-nowrap">\1</span>',
                        r.e(page_words(preview["action"])))
        lines.append(f"<p>Plan read from the checkpoint: {action}.</p>")
    if "model_spec" in inherited:
        # Its every field in the list took about 3,000 px (BACKLOG 72): said
        # by its differences, the whole spec behind its own Details ›.
        words, details = r.model_spec_html(inherited.pop("model_spec"))
        label = r.field_words("model_spec")
        lines.append(r.summary_list([(label, words + details)], titles={label: "model_spec"},
                                    classes="shep-check-spec"))
    if inherited or decoded:
        stage, task = preview.get("learning_stage"), preview.get("task")
        shown = ordered({*inherited, *decoded})
        rows = [(r.field_words(name), r.e(decoded[name]) if name in decoded else
                 r.e(r.setting_words(name, inherited[name], stage=stage, task=task)))
                for name in shown]
        if operation == BRANCH:
            summary, note = ("Settings this branch keeps", "They stay the parent’s in a branch; "
                             "the ones it may change are the fields below.")
            fixed = ""
        else:
            summary, note = ("Settings taken from this checkpoint",
                             "They cannot be changed when continuing; to train with other "
                             "settings, choose Branch above.")
            fixed = "; fixed when continuing"
        lines.append(
            f'<details class="shep-inherit"><summary>{summary} '
            f'<span class="shep-muted">({len(rows)}{fixed})</span></summary>'
            f'<p class="shep-note">{note}</p>'
            + r.summary_list(rows, titles={r.field_words(name): name for name in shown})
            + '</details>')
    if not differs:
        # When the datasets differ, the comparison above already says the
        # start refuses (M5-3: one line less, so the drawer fits).
        lines.append('<p class="shep-note">The start checks again against this workspace’s '
                     'data and graph.</p>')
    kind = "red" if differs else "ok"
    return f'<div class="shep-check shep-check-{kind}">{head}{"".join(lines)}</div>'


def _extra_epochs_update(preview: Mapping[str, Any], operation: str = CONTINUE):
    """The Extra epochs field belongs to continuing a finished checkpoint:
    shown for one, and hidden and cleared for any other (and for a branch),
    so no value is left behind for a checkpoint it does not apply to
    (contract M3a review of `094b90d`, P2-2)."""
    if operation == CONTINUE and preview.get("end") == "finished":
        return gr.update(visible=True)
    return gr.update(visible=False, value=None)


def _extend_lr_update(preview: Mapping[str, Any], operation: str = CONTINUE):
    """**The Extra-epoch rate belongs to extending a finished checkpoint whose
    rate ended at 0** in a parameter group (the preview's
    ``zero_rate_groups``, the groups `extend_lr` replaces, and only those;
    an extension of such a parent refuses without it and no rate is
    guessed): shown for one, hidden and cleared for any other, whose
    extension's rates are the plan's own sentence."""
    if operation == CONTINUE and preview.get("end") == "finished" \
            and preview.get("zero_rate_groups"):
        return gr.update(visible=True)
    return gr.update(visible=False, value=None)


def _branch_updates(preview: Mapping[str, Any], operation: str, touched: Any,
                    values: Sequence[Any]) -> Tuple[Any, ...]:
    """**The Branch fields after a check** (`BRANCH_FIELDS`, after their box):
    for a branch the preview read (its ``changeable`` and ``inherited``),
    the box shown, each field the parent's stage and task may change shown
    unless its run would not read it on the form's values (a rule across
    settings: the base form hides it so), and each of those not edited
    (`touched`) holding the parent's value in its field's form
    (`_branch_form_value`); every other field hidden and left as it is (a
    base records a task model's settings as unset, which no field of theirs
    shows). A refusal of the settings sent leaves the box and the fields as
    they were, the edit to fix; any other refusal of a branch (the parent
    itself, changed since it was shown, or not found) hides the box, whose
    values were read from a parent the check no longer reads, so no edit can
    stand for that refusal's fix (the B1c review). Continue leaves them."""
    keep = (gr.update(),) * len(BRANCH_FIELDS)
    if operation != BRANCH:
        return (gr.update(), *keep)
    if not preview.get("ok") or "changeable" not in preview:
        return (gr.update() if _refusal_kind(preview) == REFUSED_SETTINGS
                else gr.update(visible=False), *keep)
    changeable = set(preview["changeable"])
    inherited = preview.get("inherited") or {}
    edited = touched_keys(touched)
    given = dict(zip(BRANCH_FIELDS, values, strict=True)) if values else {}
    prefilled = {name for name in BRANCH_FIELDS
                 if name in changeable and name in inherited and name not in edited}
    now = {name: _branch_form_value(name, inherited[name]) if name in prefilled
           else given.get(name) for name in BRANCH_FIELDS}
    unread = not_read_by_rules({name: parse_value(by_name(name).kind, value)[0]
                                for name, value in now.items()})
    return (gr.update(visible=True),
            *(gr.update(visible=name in changeable and name not in unread,
                        **({"value": now[name]} if name in prefilled else {}))
              for name in BRANCH_FIELDS))


def _on_continue_selection_changed():
    """The chosen checkpoint, or where it comes from, changed: the last check
    and what was entered for it no longer apply, and are cleared
    (`_cleared_check`)."""
    return _cleared_check()


def _on_listed_changed(source):
    """The run list changed. It is the chosen checkpoint only when the source
    is the run list: a Refresh repopulates it even while a typed path is the
    source, and must not clear that path's check."""
    if source == CONTINUE_FROM_PATH:
        return (gr.update(),) * len(_cleared_check())
    return _on_continue_selection_changed()


def _on_manual_changed(source):
    """The typed path changed; it is the chosen checkpoint only when the
    source is the typed path."""
    if source != CONTINUE_FROM_PATH:
        return (gr.update(),) * len(_cleared_check())
    return _on_continue_selection_changed()


def _on_operation(operation, mode):
    """The operation chosen (Continue or Branch): the session records it (the
    drawer's start, its reason, the run bar's line and the run view's
    header follow it), and the last check, which was the other operation's,
    is cleared with what was entered for it (`_cleared_check`). Returns the
    session's mode, then the check's outputs."""
    mode = dict(mode or {})
    mode["operation"] = BRANCH if operation == BRANCH else CONTINUE
    return (mode, *_cleared_check())


def _on_branch_edited(mode):
    """A Branch field edited by the operator: the last check no longer
    describes what Start branch sends (its plan, its changes), so the check
    says so, and a refusal of the settings it sent (the edit is the fix) no
    longer disables Start branch, which checks what it sends first. A
    refusal of the parent itself is no edit's to dismiss: it stays, said in
    the check and beside the start (the skill's §9: nothing auto-dismisses;
    the B1c review). Returns the check's text and the session's mode."""
    mode = dict(mode or {})
    parent = dict(mode.get("parent") or {})
    if parent.get("refused") and parent.get("refused") != REFUSED_SETTINGS:
        return gr.update(), mode
    if parent.get("refused"):
        parent["refused"] = None
        mode["parent"] = parent
    return f'<p class="shep-note">{_BRANCH_EDITED}</p>', mode


def _on_branch_guards(mode, *values):
    """A Branch field that a rule across settings reads another by changed
    (the scheduler, mixed precision; by the operator or by a check's
    prefill): each field read only under its guard shows while the parent's
    stage and task may change it (the last check's ``changeable``, which the
    session's parent record keeps) and the run would read it
    (`not_read_by_rules`), as the base form's do (`_on_base_guards`).
    `values` are the guards', in `rule_fields(BRANCH_FIELDS)` order."""
    dependent, unread = rule_unread(BRANCH_FIELDS, values)
    changeable = set(((mode or {}).get("parent") or {}).get("changeable") or ())
    return [gr.update(visible=name in changeable and name not in unread) for name in dependent]


def _continue_device(device):
    """Continue…'s Device starts from the Parameters' each time it opens."""
    return device if device in ("auto", "cuda", "cpu") else "auto"


def _on_continue_device_changed():
    """The device is part of the check (mixed precision takes effect only on
    CUDA): the last check no longer describes what the start will do. What
    was entered stays; it belongs to the checkpoint, not the device."""
    return _CONTINUE_PROMPT


def _on_parent_chosen(source, listed, manual, parents, ws, mode):
    """The session records the chosen parent (for the drawer's start and the
    dataset comparison), and the chosen listed checkpoint's details are
    shown for the drawer's operation (none for a typed path, whose Check
    shows what it read)."""
    mode = dict(mode or {})
    if not mode.get("drawer"):
        return gr.update(), gr.update(), gr.update()
    mode["parent"] = _chosen(source, listed, manual, parents, ws)
    item = (parents or {}).get(listed) if source != CONTINUE_FROM_PATH and listed else None
    return (mode, compare_html(mode["parent"], ws or {}),
            parent_details_html(item, operation_of(mode)))


def _strip_display_prefix(path: str) -> str:
    """Strip the UI display prefix (``SHEPHERD-Advanced/...`` -> ``...``)."""
    p = (path or "").strip()
    return p[len("SHEPHERD-Advanced/"):] if p.startswith("SHEPHERD-Advanced/") else p


def _parent_path(source: str, listed: Any, manual: Any, ws: Mapping[str, Any],
                 parents: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str],
                                                      Optional[Mapping[str, Any]]]:
    """``(path, why_not, listed_item)`` for the chosen parent, always one the
    server lists in the run bar's workspace and resolved from that list again
    (never a client path, `PLAN_TEST_RESULTS.md` D8, 2026-10-19): a listed one
    by its names; a typed one only locates a listed checkpoint
    (`server_lists.locate_checkpoint`), and any other file is refused unopened."""
    from src.api.services.server_lists import (
        ServerListRefused,
        locate_checkpoint,
        resolve_checkpoint,
    )
    from src.utils.checkpoint_paths import listing_key

    if source == CONTINUE_FROM_PATH:
        typed = _strip_display_prefix(manual or "")
        if not typed:
            return None, "enter a checkpoint path", None
        if not ws.get("name"):
            return None, "select a dataset in the run bar first", None
        try:
            architecture, run, filename = locate_checkpoint(ws.get("name"), typed)
        except ServerListRefused as exc:
            return None, redact(exc, workspaces_root()[0]), None
        item = (parents or {}).get(listing_key((architecture, run, filename)))
        if not item:
            # Only a model record the run list offers is continued or
            # branched (K6 decision 1), so a file without a model spec is
            # refused unopened.
            return None, ("the typed path is not a model record the run list offers (a file "
                          "without a model spec is never continued here); refresh the list if "
                          "its run is new"), None
    else:
        item = (parents or {}).get(listed) if listed else None
    if not item:
        return None, "choose a checkpoint from the run list", None
    other = _other_workspace(item, ws)
    if other:
        return None, other.rstrip("."), None

    try:
        path = resolve_checkpoint(ws.get("name"), item["architecture"], item["run"],
                                  item["filename"])
    except ServerListRefused as exc:
        return None, f"it is no longer listed ({redact(exc, workspaces_root()[0])}); refresh the list", None
    return str(path), None, item


def _preview_for(operation: str, parent: str, device: Optional[str], *, extend_epochs=None,
                 extend_lr=None, settings: Optional[Mapping[str, Any]] = None,
                 expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    """The service's preview of the drawer's `operation`; `expected_sha256` is
    the model fingerprint shown, which the service compares on its read. A
    branch's (`preview_branch`) with the `settings` the drawer sends
    (`branch_sent`; none, a same-settings branch). A continuation's
    (`preview_continuation`) with Extra epochs and the Extra-epoch rate only
    where the parent takes them (`extend_if_finished`): the fields are the
    console's, and a value left in one from an earlier selection is not a
    request. Empty or 0 in either is none. The API and the CLI send them as
    given and are refused as before."""
    if operation == BRANCH:
        return training_manager.preview_branch(parent, device=device or "auto",
                                               requested=dict(settings or {}),
                                               expected_sha256=expected_sha256)
    epochs = int(extend_epochs) if extend_epochs else None
    return training_manager.preview_continuation(
        parent, device=device or "auto", extend_epochs=epochs,
        requested={"extend_lr": float(extend_lr)} if extend_lr else {},
        extend_if_finished=True, expected_sha256=expected_sha256)


def _changed_parent_action(source: Any) -> str:
    """What to do when the service refuses the parent as changed (its
    ``changed``), for where its fingerprint was shown: a listed parent's is
    the run list's, sent again until the list is read again, and a Refresh
    shows the file there now under the same option (`_on_refreshed`), which
    the check then reads; a typed path's is its last Check's."""
    if source == CONTINUE_FROM_PATH:
        return "Click Check checkpoint to read the file there now."
    return "Refresh the list, then check it again."


def _checked(operation: str, source, listed, manual, device, ws, parents, mode, *,
             extend_epochs=None, extend_lr=None, touched: Any = None,
             values: Sequence[Any] = (), starting: bool = False):
    """The check every Check checkpoint and the drawer's start run, for the
    drawer's `operation`: the parent resolved, the preview of the parent
    shown, the dataset compared. Returns (preview or None, why_not, the
    session's parent record, the settings a branch sends).

    **The model fingerprint shown goes to the service's preview, which
    compares it on its read** (root-cause round, P22-S1; no comparison here):
    the run list's, which its option names, for a listed parent; for a typed
    path, the one its last Check read and showed for that same file, when
    the start (`starting`) follows it (a Check is what shows it). The record
    keeps the digest shown, which the start sends. A refusal as changed gets
    the action for that source (`_changed_parent_action`).

    **A branch sends what the operator edited of its fields** (`branch_sent`:
    the Branch fields `values`, `touched` their record): each value its field
    spec does not take is refused here, by its label, before the parent is
    read, as the plan would refuse it; the rest the preview judges by the
    plan (`preview_branch`), which names the settings the parent's stage and
    task may change (kept in the record as ``changeable``, for the fields a
    rule shows).

    **The record keeps what a refusal refused** (``refused``,
    `_refusal_kind`), so the reason beside the start says its cause and what
    to do (`continue_refusal`), and an edit dismisses only a refusal of the
    settings (`_on_branch_edited`)."""
    path, why, item = _parent_path(source, listed, manual, ws or {}, parents or {})
    record = dict((mode or {}).get("parent") or _chosen(source, listed, manual, parents, ws))
    if path is None:
        record.update(refused=REFUSED_UNLISTED, checked=True)
        return None, why, record, {}
    if source == CONTINUE_FROM_PATH:
        shown = record.get("sha256") if starting and record.get("path") == path else None
    else:
        shown = (item or {}).get("sha256")
    settings: Dict[str, Any] = {}
    if operation == BRANCH:
        settings, problems = branch_sent(item, touched, values)
        if problems:
            preview = {"ok": False, "invalid_settings": True, "parent": path,
                       "reason": " ".join(problems)}
            record.update(chosen=True, checked=True, refused=_refusal_kind(preview))
            return preview, None, record, settings
    preview = _preview_for(operation, path, device, extend_epochs=extend_epochs,
                           extend_lr=extend_lr, settings=settings, expected_sha256=shown)
    if preview.get("changed"):
        preview = {**preview, "reason": f"{preview['reason']} {_changed_parent_action(source)}"}
    recorded = preview.get("dataset_fingerprint") or (item or {}).get("dataset_fingerprint")
    record.update(chosen=True, checked=True, refused=_refusal_kind(preview),
                  fingerprint=recorded, sha256=shown or preview.get("sha256"), path=path)
    if operation == BRANCH and preview.get("ok"):
        record["changeable"] = list(preview.get("changeable") or ())
    return preview, None, record, settings


def _nothing_to_check_html(why: Any) -> str:
    """The check's text when the drawer's check found no checkpoint the run
    list holds to read (`_parent_path`'s reason), for Check checkpoint and
    for the start, which checks first."""
    return f'<div class="shep-reason">Nothing to check: {r.e(why)}.</div>'


def _on_check_continue(request: gr.Request, operation, source, listed, manual, device,
                       extend_epochs, extend_lr, ws, parents, mode, touched, *values):
    """Check checkpoint: preview what the drawer's operation does from the
    chosen checkpoint (continuing it, or branching it with the settings
    edited, `values` the Branch fields', `touched` their record). It opens
    the checkpoint (SPEC_4 C3), so D8 is checked first, on this event's own
    request; refused, nothing is opened.

    Returns the check's text, the Extra epochs and Extra-epoch rate updates
    (each shown only where continuing the checkpoint takes it), the
    session's mode, the dataset comparison, and the Branch box and fields
    (`_branch_updates`).
    """
    keep = (gr.update(),) * (1 + len(BRANCH_FIELDS))
    access = operator_access(request)
    if not access.allowed:
        return (f'<div class="shep-reason">{r.e(access.refusal)}</div>', gr.update(),
                gr.update(), gr.update(), gr.update(), *keep)
    op = BRANCH if operation == BRANCH else CONTINUE
    mode = dict(mode or {})
    mode["operation"] = op
    preview, why, record, _settings = _checked(
        op, source, listed, manual, device, ws, parents, mode, extend_epochs=extend_epochs,
        extend_lr=extend_lr, touched=touched, values=values)
    mode["parent"] = record
    if preview is None:
        hidden = gr.update(visible=False, value=None)
        return (_nothing_to_check_html(why), hidden, hidden, mode, compare_html(record, ws or {}),
                *_branch_updates({}, op, touched, values))
    differs = continue_refusal(record, ws or {}, operation=op) is not None and preview.get("ok")
    text = _check_html(preview, source, differs=bool(differs), listed=listed, operation=op)
    if op == CONTINUE and preview.get("end") == "finished" and not preview.get("ok") \
            and not extend_epochs:
        text += '<p class="shep-note">Enter <b>Extra epochs</b> above, then click <b>Check checkpoint</b> again.</p>'
    return (text, _extra_epochs_update(preview, op), _extend_lr_update(preview, op), mode,
            compare_html(record, ws or {}), *_branch_updates(preview, op, touched, values))


def _on_resume(request: gr.Request, operation, ws, device, source, listed, manual,
               extend_epochs, extend_lr, parents, mode, touched, *values):
    """The drawer's start, Continue Training or Start branch: an operator
    control (D8, checked first on this event's own request; refused, nothing
    is opened or started), then the same check as the preview, the dataset
    comparison, then the job runner's start with the full dataset
    fingerprint and the parent's model fingerprint this session showed.

    A continuation sends only operational fields (contract M3a-3): every
    learning setting comes from the checkpoint, so the training form's values
    are not used; an extension its Extra epochs and, for a parent whose rate
    ended at 0, its Extra-epoch rate, as the check judged them. A branch
    sends ``branch_from`` and the settings the operator edited of its fields
    (`branch_sent`), none for a same-settings branch (decision B1). The job
    runner and the training process each check again, the process on the
    parent the runner decided on (P22-S1; decision B2).

    Returns the start's notice (`notice_html`, in the drawer's footer beside
    the start: only a refusal, so led by the ✕ of a refusal), the check's
    text, the Extra epochs and Extra-epoch rate updates, the session's mode
    (the drawer closed once started) and the Branch box and fields.
    """
    access = operator_access(request)
    if not access.allowed:
        return (notice_html(access.refusal, mark="error"), gr.update(), gr.update(), gr.update(),
                gr.update(), *(gr.update(),) * (1 + len(BRANCH_FIELDS)))
    said, text, extend, rate, mode, branch = _resume(
        operation, ws, device, source, listed, manual, extend_epochs, extend_lr, parents, mode,
        touched, values)
    return notice_html(said, mark="error"), text, extend, rate, mode, *branch


def _resume(operation, ws, device, source, listed, manual, extend_epochs, extend_lr, parents,
            mode, touched=None, values: Sequence[Any] = ()):
    """The drawer's start's work after D8: ``(sentence, check text, Extra
    epochs update, Extra-epoch rate update, mode, Branch box and fields)``,
    the sentence plain text: "" once started, and "" when the start's own
    check refuses it, whose refusal the drawer's footer says beside the start
    from the same record, by its cause and what to do (`continue_refusal`:
    "Start branch disabled: the check refuses a setting you edited; …", the
    dataset comparison's) and the check says why, a checkpoint the run list
    does not hold included ("Nothing to check: …"): the fact said once (the
    skill's §1.2; the B1c review measured the footer at 174 px when a generic
    reason and this notice said one refusal twice); a refusal nothing else
    says (no dataset, the job runner's own) is the sentence."""
    op = BRANCH if operation == BRANCH else CONTINUE
    mode = dict(mode or {})
    mode["operation"] = op
    keep = (gr.update(),) * (1 + len(BRANCH_FIELDS))
    row = (ws or {}).get("row") or {}
    fingerprint = workspace_fingerprint(ws)
    if not row or not fingerprint:
        return "Not started: select a dataset first.", gr.update(), gr.update(), gr.update(), \
            mode, keep
    preview, why, record, settings = _checked(
        op, source, listed, manual, device, ws, parents, mode, extend_epochs=extend_epochs,
        extend_lr=extend_lr, touched=touched, values=values, starting=True)
    mode["parent"] = record
    hidden = gr.update(visible=False, value=None)
    if preview is None:
        return ("", _nothing_to_check_html(why), hidden, hidden, mode,
                _branch_updates({}, op, touched, values))
    refusal = continue_refusal(record, ws, operation=op)
    text = _check_html(preview, source, differs=preview.get("ok") and refusal is not None,
                       listed=listed, operation=op)
    shown = (_extra_epochs_update(preview, op), _extend_lr_update(preview, op))
    branch = _branch_updates(preview, op, touched, values)
    if not preview.get("ok"):
        if op == CONTINUE and preview.get("end") == "finished" and not extend_epochs:
            text += ('<p class="shep-note">Enter <b>Extra epochs</b> above, then click '
                     '<b>Continue Training</b> again.</p>')
        return "", text, *shown, mode, branch
    if refusal:
        return "", text, *shown, mode, branch
    from src.api.services.server_lists import ServerListRefused

    try:
        directory = _workspace_path(row["name"])
    except ServerListRefused:
        return (f"Not started: {row['name']} is no longer listed by the server.", text, *shown,
                mode, branch)
    config: Dict[str, Any] = {
        "data_dir": str(directory),
        "run_name": "",
        "device": device or "auto",
        "compile": bool(load_runtime_settings().get("torch_compile", False)),
    }
    if op == BRANCH:
        config.update(settings, branch_from=preview["parent"])
    else:
        config["resume_from"] = preview["parent"]
        if preview["mode"] == "extend":
            # What the check judged: its epochs, and the rate it set for
            # the groups at 0 when the parent has one.
            config["extend_epochs"] = preview["stage"]["target_epochs"]
            rate = (preview["stage"].get("schedule") or {}).get("extend_lr")
            if rate is not None:
                config["extend_lr"] = rate
    result = training_manager.start_training(config, expected_dataset_fingerprint=fingerprint,
                                             expected_parent_sha256=record["sha256"])
    if result.get("success"):
        mode.update(drawer=False, parent={}, confirm=None)
        return "", text, *shown, mode, branch
    said = page_words(not_started(result.get("error")))
    if result.get("parent_changed"):  # changed after this check's own read
        said = f"{said} {_changed_parent_action(source)}"
    return said, text, *shown, mode, branch


def continues_html(mode: Optional[Mapping[str, Any]]) -> str:
    """The run bar's line for the run the drawer would start (while it is
    open): a new run of its own (contract M3a-2), continued or branched, by
    the drawer's operation, from the checkpoint the drawer shows chosen
    (named there once; the run view's header says whether the run it shows
    is that one)."""
    done = OPERATION_VERBS[operation_of(mode)][1]
    chosen = ((mode or {}).get("parent") or {}).get("chosen")
    return (f'<span class="shep-rb-k">Run</span> a new one, {done} from the checkpoint chosen '
            f'in {DRAWER_TITLE}' if chosen else
            '<span class="shep-rb-k">Run</span> <span class="shep-muted">a new one, once a '
            f'checkpoint is chosen in {DRAWER_TITLE}</span>')


def _toggle_continue_source(source):
    """The run list's two selects, its Refresh and the chosen checkpoint's
    details, or the typed path."""
    from_list = source != CONTINUE_FROM_PATH
    return (gr.update(visible=from_list), gr.update(visible=from_list),
            gr.update(visible=from_list), gr.update(visible=from_list),
            gr.update(visible=not from_list))

# =============================================================================
# Help
# =============================================================================
#: The fields' help, behind the shared ⓘ (`src/webui/components/help.py`):
#: ``{elem_id: (title, text)}``, registered by the tab builder. Only a
#: state-bearing text stays in ``info=``.
FIELD_HELP = {
    "shep-train-dataset": (
        "Dataset",
        "The server's workspace list, as on Research › Datasets. Choosing one only reads it; "
        "nothing changes on the server.\n\n"
        "Start sends the dataset fingerprint shown, and the server refuses the start if the "
        "workspace changed since.",
    ),
    "run_name": (
        "Run name",
        "Optional. Each run is saved in a new folder of its own:\n"
        "<workspace>/runs/<architecture>/<date-time>_<name>_<id>/\n\n"
        "Letters, digits and hyphens are kept; blank uses \"run\". The run view shows the "
        "name once training starts.",
    ),
    # The base form: each setting's label, what it is, and the value a run
    # takes without it, with its source (`setting_help`).
    **{name: setting_help(name, *base_references()) for name in BASE_FIELDS},
    "device": (
        "Device",
        "auto: the GPU (CUDA). With no CUDA device visible, Start is refused: training never "
        "falls back to the CPU on its own.\n\n"
        "- cpu: for development, recorded as asked.\n"
        "- Continue… has its own Device, which starts from this one.",
    ),
    "continue_operation": (
        "Operation",
        "What the drawer does from the chosen checkpoint. Both train a new run of their own; "
        "the checkpoint is not changed.\n\n"
        "• Continue: resume a paused run, or extend a finished one by Extra epochs. Every "
        "setting stays the checkpoint's.\n"
        "• Branch: a new run from its weights, with a new optimizer and schedule. Change the "
        "settings its stage allows, or none for the same settings.",
    ),
    "continue_source": (
        "Checkpoint source",
        "Where the checkpoint to continue or branch from comes from.\n\n"
        "• From run list: the run bar's dataset's model records (bases and task models), as "
        "the server lists them.\n"
        "• Manual path: a typed path. It only finds a checkpoint that list has; any other "
        "file is refused, unopened.\n\n"
        "A file without a model spec is never continued here. Both are checked the same way. "
        "Listing opens each checkpoint, so it is an operator control.",
    ),
    "continue_run": (
        STEP_RUN,
        "The run to continue or branch from, latest first, each led by its name or start "
        "time.\n\n"
        "Its architecture and start time follow; its directory is in the table. Either trains "
        "a new run of its own; the run chosen here is not changed.",
    ),
    "checkpoint_dropdown": (
        STEP_CHECKPOINT,
        "The chosen run's checkpoints, newest first.\n\n"
        "• Latest save: last.pt, the run's last training state.\n"
        "• Best (its metric named): the run's best value of the selection metric its files "
        "record, in their direction; files that tie share it.\n"
        "• Top-k save: another file the run kept by that metric.\n\n"
        "The metric is the Trainer's validation figure, not a test result. The table under "
        "it shows the file, its stage, what it holds, that figure, the selection and the model "
        "fingerprint. The server finds the file in its list again and compares its bytes with "
        "that fingerprint. Refresh lists them again.",
    ),
    "manual_checkpoint": (
        "Checkpoint path",
        "The path of a model record the run list has, typed or pasted. It is checked the same "
        "way as a listed one.",
    ),
    "continue_device": (
        "Device",
        "Where the new run trains: auto is the GPU (CUDA), and with no CUDA device visible "
        "the start is refused; choose cpu for development.\n\n"
        "Check checkpoint and the start (Continue Training or Start branch) both use it. Mixed "
        "precision works only on CUDA; the check says when the checkpoint's setting would not "
        "hold here.\n\n"
        "It starts from the Parameters' Device each time Continue… opens.",
    ),
    "extend_epochs": (
        EXTEND_WORDS["extend_epochs"],
        "Extend a finished run by this many epochs, trained at the fixed rate shown in the "
        "check result.",
    ),
    "extend_lr": (
        EXTEND_WORDS["extend_lr"],
        "The rate the Extra epochs train at in the parameter groups whose rate ended at 0.\n\n"
        "Shown only for such a checkpoint: extending at 0 would not train those groups, and "
        "no rate is guessed. The other groups keep the rate they ended at.",
    ),
    # The Branch fields: each setting's label, what it is, where it starts.
    **{f"branch_{name}": branch_setting_help(name) for name in BRANCH_FIELDS},
    "shep-train-view-run": (
        "Viewing",
        "The run whose curves and Export CSV the run view shows: the run bar's dataset's runs, "
        "latest first, each led by its name or start time.\n\n"
        "• Choosing one changes only the view. The run bar and its status pill always show the "
        "training job.\n"
        "• Beside it, how the run stands to that job: the running job, or a past run.\n"
        "• A job started later is shown as it starts. Its directory is in Details.",
    ),
}

#: The ⓘ inside the page's rendered HTML.
TOPIC_HELP = {
    "training-charts": (
        "Chart gestures",
        charts.GESTURES + "\n\n"
        "The wheel zooms around the pointer:\n"
        "• over the epoch axis, the epochs (the four charts share them);\n"
        "• over a value axis, that chart's values;\n"
        "• inside a plot, both.\n\n"
        "Keys, with a chart focused: + and − zoom, x and y zoom one axis (Shift zooms out), "
        "the arrows pan, End jumps to the latest epochs, 0 resets.\n\n"
        "The view is kept per run in this browser tab, and the epoch axis never starts below "
        "0. While zoomed in a live run, a button counts the new epochs beyond the view. Away "
        "from the charts the wheel scrolls the page.",
    ),
    "training-used-dataset": (
        "Used dataset fingerprint",
        "The dataset the run trained on, as its checkpoint records it (the split manifest's "
        "fingerprint), compared in full with the run bar's dataset.\n\n"
        "• Shown to the operator only: reading it opens the checkpoint.\n"
        "• Before the first checkpoint nothing records it yet; a run that ended without one "
        "never recorded it.",
    ),
}


# =============================================================================
# The page
# =============================================================================
def create_training_tab(ctx: Any) -> None:
    """Build Research › Prepare › Training (the page contract:
    `src/webui/shell.py`). `ctx` gives the help registry, the shared timers,
    the cross-links, the chips' read after every gated act and this session's
    run-bar workspace (``ctx.training_workspace``)."""
    ctx.help.fields(FIELD_HELP)
    ctx.help.topics(TOPIC_HELP)

    ws = gr.State(empty_workspace())
    mode = gr.State({})
    cache = gr.State({})
    # The fast poll's own record (its reads kept by time, `POLL_READS`, and the
    # GPU slot it saw last, for the chips): only it writes it.
    polled = gr.State({})
    dismissed = gr.State([])
    parents = gr.State({})
    # The run view's choice in its Viewing select (`_on_view_run`; read by
    # every render, `viewed_run`): only the select's input writes it.
    viewing = gr.State({})

    with gr.Column(elem_id="shep-train", elem_classes=["shep-train"]):
        # ------------------------------------------------------------- run bar
        # Its own height, never a share of the page's (scale 0: Gradio writes
        # a column's scale inline as its flex-grow, which the style sheet does
        # not outrank; at 1 the run bar took half the room the run view left
        # and drew it blank under its rows, BACKLOG 52 F).
        with gr.Column(elem_id="shep-train-runbar", elem_classes=["shep-runbar"],
                       scale=0) as runbar:
            with gr.Row(elem_classes=["shep-rb-row", "shep-rb-row1"]):
                dataset = gr.Dropdown(label="Dataset", choices=[], value=None,
                                      elem_id="shep-train-dataset", scale=0, min_width=240,
                                      container=False)
                chip = plain_html(chip_html(None), elem_id="shep-train-chip", js_on_load=_POPOVER_JS)
                pill = plain_html(pill_html({"status": "idle"}), elem_id="shep-train-pill")
            with gr.Row(elem_classes=["shep-rb-row", "shep-rb-row2"]):
                run_name = gr.Textbox(label="Run name", value="", max_lines=1, elem_id="run_name",
                                      scale=0, min_width=240, container=False)
                continues = plain_html("", elem_id="shep-train-continues")
                reason = plain_html('<div class="shep-rb-reason"></div>', elem_id="shep-train-reason",
                                    js_on_load=TRIGGER_JS)
                # Labelled by the render's one gated-button rule (`ACTION_LABELS`).
                start_btn = gr.Button(ACTION_LABELS["start"], variant="primary",
                                      interactive=False, scale=0, elem_id="shep-train-start")
                cont_btn = gr.Button(ACTION_LABELS["cont"], interactive=False, scale=0,
                                     elem_id="shep-train-continue")
                stop_btn = gr.Button(ACTION_LABELS["stop"], variant="secondary",
                                     interactive=False, scale=0, elem_id="shep-train-stop",
                                     elem_classes=["shep-stop"])
            # Stop asks first, in the code's words (contract M3a). The
            # question and Check (in the drawer) are reached only through a
            # Stop or a Continue… D8 allowed, so they show no lock (the
            # skill's §3); their handlers check their own request again.
            with gr.Row(elem_classes=["shep-rb-row", "shep-rb-confirm"]):
                stop_text = plain_html("", elem_id="shep-train-stop-text")
                stop_confirm_btn = gr.Button("■ Stop training", variant="stop", scale=0,
                                             elem_id="shep-train-stop-confirm")
                stop_keep_btn = gr.Button("Keep training", scale=0, elem_id="shep-train-stop-keep")
            notice = plain_html("", elem_id="shep-train-notice")
        banner = plain_html("", elem_id="shep-train-banner", js_on_load=_BANNER_JS)

        with gr.Row(elem_id="shep-train-page", elem_classes=["shep-page", "shep-train-page"],
                    equal_height=False) as page:
            # ------------------------------------------------------- parameters
            with gr.Column(elem_id="shep-train-params", elem_classes=["shep-panel"], scale=10,
                           min_width=360):
                plain_html('<div class="shep-panel-hd"><h2>Parameters</h2>'
                           '<span class="shep-panel-right">for the next base run</span></div>',
                           elem_id="shep-train-params-hd")
                with gr.Column(elem_id="shep-train-params-bd", elem_classes=["shep-train-bd"]):
                    # What a new base takes (decision S1), each field starting at
                    # the value the run takes without it: the common settings
                    # first, the rest one click away, a setting the run would not
                    # read hidden (decision S3); the model spec by its fields.
                    references, _sources = base_references()
                    unread = not_read_by_rules(references)
                    fields: Dict[str, Any] = {}
                    with gr.Group(elem_classes=["shep-train-grid"]):
                        for name in BASE_FIELDS[:BASE_SHOWN]:
                            fields[name] = setting_field(name, references[name], elem_id=name)
                        device = gr.Radio(label="Device", choices=["auto", "cuda", "cpu"],
                                          value="auto", elem_id="device")
                    with gr.Accordion("More training settings", open=False,
                                      elem_id="shep-train-more"):
                        with gr.Group(elem_classes=["shep-train-grid"]):
                            for name in BASE_FIELDS[BASE_SHOWN:]:
                                fields[name] = setting_field(name, references[name],
                                                             elem_id=name,
                                                             visible=name not in unread)
                    with gr.Accordion("Model", open=False, elem_id="shep-train-model"):
                        plain_html(f'<p class="shep-note">{r.e(MODEL_NOTE)}</p>'
                                   f'{spec_facts_html()}', elem_id="shep-train-spec-facts")
                        if len(architecture_choices()) > 1:
                            fields[ARCHITECTURE] = gr.Radio(
                                label=spec_key_words(ARCHITECTURE),
                                choices=architecture_choices(),
                                value=_spec_reference(ARCHITECTURE),
                                elem_id="spec-encoder-kind")
                        for section in SPEC_SECTIONS:
                            plain_html(f'<h3 class="shep-train-sub">'
                                       f'{r.e(r.field_words(section))}</h3>',
                                       elem_id=f"shep-train-spec-{section}")
                            with gr.Group(elem_classes=["shep-train-grid"]):
                                for spec_field in spec_form_fields():
                                    if spec_field.editable and spec_field.path.startswith(
                                            f"{section}."):
                                        fields[spec_field.path] = value_field(
                                            spec_field.kind,
                                            r.field_words(spec_field.path.split(".", 1)[1]),
                                            spec_field.reference, elem_id=spec_field.elem_id,
                                            info=FIELD_FORMATS.get(spec_field.path))
                    # What the operator set (decision S2), and Reset to reference.
                    with gr.Row(elem_classes=["shep-train-setby-row"]):
                        set_by = plain_html(set_by_you_html(dict(zip(
                            base_form_keys(), base_form_defaults(), strict=True)), []),
                            elem_id="shep-train-setby")
                        reset_btn = gr.Button("Reset to reference", size="sm", scale=0,
                                              elem_id="shep-train-reset")
                    # torch.compile is configured in the Runtime Settings tab (it is an
                    # execution/runtime knob, applied per training run).
                base_fields = [fields[key] for key in base_form_keys()]

            # ------------------------------------------------- Continue… drawer
            # Numbered steps (the skill's §4): what to do (Continue or Branch),
            # 1. Run, 2. Checkpoint, 3. Check, then 4. Start in the footer, the
            # start beside its reason (the skill's §9); the body scrolls above
            # it when it does not fit, the footer never.
            with gr.Column(elem_id="shep-train-drawer", elem_classes=["shep-panel"], scale=10,
                           min_width=360):
                with gr.Row(elem_classes=["shep-panel-hd", "shep-drawer-hd"]):
                    plain_html(f'<h2>{r.e(DRAWER_TITLE)}</h2><span class="shep-muted">replaces '
                               'the parameters until closed</span>',
                               elem_id="shep-train-drawer-title")
                    close_btn = gr.Button("Close", scale=0, elem_id="shep-train-drawer-close",
                                          elem_classes=["shep-x-btn"])
                with gr.Column(elem_id="shep-train-drawer-bd", elem_classes=["shep-train-bd"]):
                    continue_operation = gr.Radio(choices=OPERATION_CHOICES, value=CONTINUE,
                                                  label="Operation",
                                                  elem_id="continue_operation")
                    continue_source = gr.Radio(choices=[CONTINUE_FROM_LIST, CONTINUE_FROM_PATH],
                                               value=CONTINUE_FROM_LIST, label="Checkpoint source",
                                               elem_id="continue_source")
                    parents_note = plain_html("", elem_id="shep-train-parents-note")
                    # Two steps (the GB10 WebUI recheck: one option per file, each the
                    # run, epoch, file, kind and fingerprint, read as a wall of text):
                    # the run, then its checkpoint, each in a few words; the chosen
                    # file's details in a table under them.
                    with gr.Row(elem_classes=["shep-train-parent-row"]):
                        continue_run = gr.Dropdown(label=STEP_RUN, choices=[], value=None,
                                                   elem_id="continue_run")
                        refresh_ckpt_btn = gr.Button("Refresh", size="sm", scale=0,
                                                     elem_id="shep-train-refresh")
                    checkpoint_dropdown = gr.Dropdown(label=STEP_CHECKPOINT, choices=[],
                                                      value=None, elem_id="checkpoint_dropdown")
                    parent_details = plain_html("", elem_id="shep-train-parent-details")
                    manual_checkpoint = gr.Textbox(label="Checkpoint path", value="", max_lines=1,
                                                   visible=False, elem_id="manual_checkpoint")
                    # The device is part of the check, so it is here, not only
                    # in the parameters the drawer covers.
                    continue_device = gr.Radio(label="Device", choices=["auto", "cuda", "cpu"],
                                               value="auto", elem_id="continue_device")
                    # Extra epochs (shown for a finished checkpoint) and its rate
                    # (for one whose rate ended at 0) beside Check checkpoint,
                    # above what the check says (M5-3: under a long result it
                    # fell below the fold at 1920 x 1080).
                    with gr.Row(elem_classes=["shep-train-check-row"]):
                        plain_html(f'<span class="shep-step">{STEP_CHECK}</span>',
                                   elem_id="shep-train-step-check")
                        # No widget minimum: hidden and empty, a Number is sent as
                        # 0, which a minimum would reject before any handler runs.
                        # Empty or 0 means "not extending" (and "no rate"); the
                        # run plan refuses below 1 (and a rate of 0 or less).
                        extend_epochs = gr.Number(label=EXTEND_WORDS["extend_epochs"],
                                                  value=None, precision=0, visible=False,
                                                  elem_id="extend_epochs", scale=0, min_width=0)
                        extend_lr = gr.Number(label=EXTEND_WORDS["extend_lr"], value=None,
                                              visible=False, elem_id="extend_lr", scale=0,
                                              min_width=0)
                        check_continue_btn = gr.Button("Check checkpoint", size="sm", scale=0,
                                                       elem_id="shep-train-check")
                    compare = plain_html("", elem_id="shep-train-compare")
                    continue_preview = plain_html(f'<p class="shep-note">{_CONTINUE_PROMPT}</p>',
                                                  elem_id="continue_preview")
                    # Branch: the settings its parent's stage and task may
                    # change, each the one field of its setting, shown after a
                    # check at the parent's value (`_branch_updates`).
                    with gr.Column(visible=False, elem_id="shep-train-branch") as branch_box:
                        plain_html('<h3 class="shep-train-sub">Settings to change</h3>'
                                   '<p class="shep-note">Each starts at the parent’s value; '
                                   'only a field you edit is sent. Edit none for the same '
                                   'settings.</p>', elem_id="shep-train-branch-hd")
                        with gr.Group(elem_classes=["shep-train-grid"]):
                            branch_fields = [setting_field(name, _branch_field_value(name),
                                                           elem_id=f"branch_{name}",
                                                           visible=False)
                                             for name in BRANCH_FIELDS]
                with gr.Row(elem_id="shep-train-drawer-ft", elem_classes=["shep-panel-ft"]):
                    plain_html(f'<span class="shep-step">{STEP_START}</span>',
                               elem_id="shep-train-step-start")
                    # Labelled by the render's one gated-button rule
                    # (`ACTION_LABELS`, by the operation: `start_label`).
                    resume_btn = gr.Button(ACTION_LABELS["resume"], variant="primary",
                                           interactive=False, scale=0,
                                           elem_id="shep-train-resume")
                    cancel_btn = gr.Button("Cancel", scale=0, elem_id="shep-train-cancel")
                    drawer_reason = plain_html("", elem_id="shep-train-drawer-why")
                    drawer_notice = plain_html("", elem_id="shep-train-drawer-notice")

            # ------------------------------------------------------- run view
            with gr.Column(elem_id="shep-train-runview", elem_classes=["shep-panel"], scale=22):
                # Three rows (`PAGE_CSS`): the header (the Viewing select, the
                # run viewed's relation to the training job, its state and
                # Details), the charts, and the resources line.
                with gr.Row(elem_id="shep-train-rv-top", elem_classes=["shep-panel-hd"]):
                    view_run = gr.Dropdown(label="Viewing", choices=[], value=None,
                                           interactive=False, elem_id="shep-train-view-run",
                                           scale=0, min_width=240, container=False)
                    run_head = plain_html("", elem_id="shep-train-run-head",
                                          js_on_load=_POPOVER_JS)
                figure = charts.chart_figure(
                    key="training", elem_id="shep-train-charts", series_fn=training_chart_series,
                    csv_fn=training_chart_csv, help_key="training-charts",
                    empty=charts.LOADING)
                resources = plain_html(_IDLE_RESOURCE_HTML, elem_id="resource_display")

    # =========================================================================
    # Wiring
    # =========================================================================
    # Explicit is what the operator touched (decision S2): the base form's
    # record, the browser's (`touched_record`), which Start's own request
    # carries. Each field's input (never a value a page writes) adds it there,
    # then says the line again; the line's last input is always said
    # ("always_last": an input while the one before is answered is sent after
    # it, never dropped). Reset to reference clears the record and writes every
    # field's starting value.
    touched = touched_record("shep-train-touched")
    for key, field in zip(base_form_keys(), base_fields, strict=True):
        record_touches(field, key, touched)
        field.input(_base_input_listener(key), inputs=[touched, *base_fields],
                    outputs=[set_by], queue=False, trigger_mode="always_last",
                    show_progress="hidden")
    reset_btn.click(_on_base_reset, inputs=None, outputs=[touched, set_by, *base_fields],
                    queue=False, show_progress="hidden")
    # A setting a rule across settings reads only under its guard is shown while
    # the run would read it (decision S3): its guards' change, by the operator
    # or by Reset, shows or hides it.
    dependent, guards = rule_fields()
    gr.on(triggers=[fields[name].change for name in guards], fn=_on_base_guards,
          inputs=[fields[name] for name in guards],
          outputs=[fields[name] for name in dependent], queue=False, show_progress="hidden")
    view_outputs = [ws, dataset, chip, pill, reason, start_btn, cont_btn, stop_btn, resume_btn,
                    runbar, page, run_name, banner, run_head, view_run, figure, drawer_reason,
                    ctx.training_workspace, cache]
    view_inputs = [ws, mode, cache, dismissed, viewing]
    # Every render writes the run bar and the run view drawn from the session's
    # selection, and may write the workspace itself (a running job's dataset),
    # so every one is in the group of the workspace's writers
    # (`shell.TRAINING_SELECTION_CONCURRENCY`): it runs after those asked
    # before it, from what they wrote, and its answer is never applied after
    # the answer of one asked after it.
    in_order = ctx.training_workspace_writes
    view_step = {"fn": _render_view, "inputs": view_inputs, "outputs": view_outputs,
                 "show_progress": "hidden", **in_order}
    # In the group an event can wait as long as the one before it runs, and with
    # Gradio's defaults the browser does not send a listener's trigger while its
    # event waits (an input's "once" is dropped; a change's "always_last" is sent
    # once, with the first trigger's data): with "multiple" each choice of the
    # workspace joins, and the last one asked is the one that stays.
    every_one = {"trigger_mode": "multiple"}
    # The fast poll's hand-off of a change it found (`_poll`).
    changed = gr.JSON(value=None, visible=False, elem_id=CHANGED_ID)

    # The page shown: the list read again, then the run bar whole, in that order.
    ctx.on_page_open(TRAINING, {
        "fn": _open, "inputs": [ws, ctx.training_workspace],
        "outputs": [ws, dataset, chip, ctx.training_workspace, cache, resources],
        "show_progress": "hidden", **in_order}).then(**view_step)

    # The shell's timers (outside gr.Tabs), read-only polls without the queue.
    # The fast one writes the status pill (`POLL_KEYS`) and the chips, and hands
    # every other change to a queued render in the group. That render reads
    # everything at its turn, so the token's default trigger ("always_last") is
    # the one wanted: tokens given while one render waits need one more render,
    # not one each.
    ctx.timer_fast.tick(_poll, inputs=[ctx.location, *view_inputs, polled],
                        outputs=[*(view_outputs[VIEW_KEYS.index(key)] for key in POLL_KEYS),
                                 changed, ctx.chips, polled],
                        queue=False, show_progress="hidden")
    changed.change(**{**view_step, "fn": _render_changed, "inputs": [*view_inputs, polled]})
    ctx.timer_slow.tick(_poll_resources, inputs=[ctx.location], outputs=[resources],
                        queue=False, show_progress="hidden")

    # The run bar's dataset.
    dataset.input(_on_dataset, inputs=[dataset, mode],
                  outputs=[ws, chip, ctx.training_workspace, mode, notice],
                  show_progress="hidden", **in_order, **every_one).then(**view_step)
    # The run view's run (the session's view choice, then the render that
    # draws it), in the same group: a render drawn from an older choice is
    # never applied after a newer one's.
    view_run.input(_on_view_run, inputs=[view_run, ws], outputs=[viewing],
                   show_progress="hidden", **in_order, **every_one).then(**view_step)
    # The session's workspace changed (a hand choice above writes it, and so
    # do Datasets' "Use for training ›" and a render that follows a job
    # running in another workspace): with Continue… open, the parent list
    # follows it (a gated listing, then the chosen parent, the run bar and the
    # chips again). A follow therefore costs its render and this chain.
    chosen_step = {"fn": _on_parent_chosen,
                   "inputs": [continue_source, checkpoint_dropdown, manual_checkpoint, parents, ws,
                              mode], "outputs": [mode, compare, parent_details],
                   "show_progress": "hidden"}
    # The drawer's record of the Branch fields the operator edited, the
    # browser's, as the base form's (`touched_record`).
    branch_touched = touched_record("shep-train-branch-touched")
    # What a check leaves once what it described changed (`_cleared_check`).
    check_cleared = [extend_epochs, extend_lr, continue_preview, drawer_notice, branch_touched,
                     branch_box]
    ctx.training_workspace.change(_on_training_workspace, inputs=[ctx.training_workspace, ws, mode],
                                  outputs=[ws, dataset, chip, mode],
                                  show_progress="hidden", **in_order, **every_one).then(
        _relist_parents, inputs=[ws, mode],
        outputs=[parents, continue_run, checkpoint_dropdown, parents_note, compare,
                 *check_cleared], show_progress="hidden").then(**chosen_step).then(
        **view_step).then(**ctx.chip_refresh)
    def _on_chip_click(evt: gr.EventData):
        """The chip's "All datasets ›": Research › Datasets."""
        return ctx.go("datasets") if event_act(evt) == "datasets" else ctx.no_nav()

    def _on_reason_click(evt: gr.EventData):
        """The reason line's "Open Test ›": Research › Test."""
        return ctx.go("test") if event_act(evt) == "open-test" else ctx.no_nav()

    chip.click(_on_chip_click, inputs=None, outputs=ctx.nav_outputs, show_progress="hidden")
    reason.click(_on_reason_click, inputs=None, outputs=ctx.nav_outputs, show_progress="hidden")
    banner.click(_on_banner_click, inputs=[dismissed, cache], outputs=[dismissed, banner],
                 show_progress="hidden")

    # Start, Stop.
    # Start also closes a Stop confirmation left from an earlier job.
    start_btn.click(_on_start, inputs=[ws, run_name, device, touched, *base_fields],
                    outputs=[notice],
                    show_progress="hidden").then(
        close_stop_question, inputs=[mode], outputs=[mode], show_progress="hidden").then(
        **view_step).then(**ctx.chip_refresh)
    stop_btn.click(_on_stop_request, inputs=[mode], outputs=[stop_text, mode],
                   show_progress="hidden").then(**view_step).then(**ctx.chip_refresh)
    stop_confirm_btn.click(_on_stop, inputs=[mode], outputs=[notice, mode],
                           show_progress="hidden").then(**view_step).then(**ctx.chip_refresh)
    stop_keep_btn.click(close_stop_question, inputs=[mode], outputs=[mode],
                        show_progress="hidden").then(**view_step)

    # Continue… (its operation Continue each time it opens).
    cont_btn.click(_on_continue_open, inputs=[ws, mode],
                   outputs=[notice, mode, parents, continue_run, checkpoint_dropdown,
                            parents_note, continue_operation, *check_cleared],
                   show_progress="hidden").then(
        _continue_device, inputs=[device], outputs=[continue_device],
        show_progress="hidden").then(**chosen_step).then(**view_step).then(**ctx.chip_refresh)
    for closer in (cancel_btn, close_btn):
        closer.click(_on_continue_close, inputs=[mode], outputs=[mode],
                     show_progress="hidden").then(**view_step)
    # Refresh, then the chosen option's file as the list holds it now: a
    # kept option fires no selection event, so the record, the details and
    # the check follow it here (`_on_refreshed`), then the drawer's start.
    refresh_ckpt_btn.click(_refresh_checkpoints, inputs=[ws, mode, checkpoint_dropdown],
                           outputs=[parents, continue_run, checkpoint_dropdown, parents_note],
                           show_progress="hidden").then(
        _on_refreshed, inputs=chosen_step["inputs"],
        outputs=[*chosen_step["outputs"], *check_cleared], show_progress="hidden").then(
        **view_step).then(**ctx.chip_refresh)
    # A run chosen: its checkpoints (the checkpoint select's change then
    # clears the last check and records the parent, as for any choice).
    continue_run.change(_model_list().on_run_chosen, inputs=[continue_run, parents, checkpoint_dropdown],
                        outputs=[checkpoint_dropdown], show_progress="hidden")
    # Check checkpoint and the start take the operation, the parent, the
    # device, what continuing takes (Extra epochs and its rate) and what a
    # branch takes (the Branch fields and the record of those edited).
    taken = [extend_epochs, extend_lr, ws, parents, mode, branch_touched, *branch_fields]
    branch_shown = [branch_box, *branch_fields]
    check_continue_btn.click(
        _on_check_continue,
        inputs=[continue_operation, continue_source, checkpoint_dropdown, manual_checkpoint,
                continue_device, *taken],
        outputs=[continue_preview, extend_epochs, extend_lr, mode, compare, *branch_shown],
        show_progress="hidden").then(**view_step).then(
        None, js=SHOW_CHECK_JS, queue=False).then(**ctx.chip_refresh)
    resume_btn.click(
        _on_resume,
        inputs=[continue_operation, ws, continue_device, continue_source, checkpoint_dropdown,
                manual_checkpoint, extend_epochs, extend_lr, parents, mode, branch_touched,
                *branch_fields],
        outputs=[drawer_notice, continue_preview, extend_epochs, extend_lr, mode,
                 *branch_shown],
        show_progress="hidden").then(**view_step).then(
        None, js=SHOW_CHECK_JS, queue=False).then(**ctx.chip_refresh)
    continue_source.change(fn=_toggle_continue_source, inputs=[continue_source],
                           outputs=[continue_run, checkpoint_dropdown, refresh_ckpt_btn,
                                    parent_details, manual_checkpoint])
    # A changed selection, source or operation makes the last check, and what
    # was entered for it, stale: it is cleared (contract M3a review of
    # 094b90d, P2-2). `.change` fires on updates from other events too
    # (Refresh), so the run list and the typed path clear only when they are
    # the source.
    continue_source.change(fn=_on_continue_selection_changed, inputs=[],
                           outputs=check_cleared)
    checkpoint_dropdown.change(fn=_on_listed_changed, inputs=[continue_source],
                               outputs=check_cleared)
    manual_checkpoint.change(fn=_on_manual_changed, inputs=[continue_source],
                             outputs=check_cleared)
    continue_device.change(fn=_on_continue_device_changed, inputs=[], outputs=[continue_preview])
    # The session's record of the chosen parent (the drawer's start).
    chosen_inputs = [continue_source, checkpoint_dropdown, manual_checkpoint, parents, ws, mode]
    for component in (continue_source, checkpoint_dropdown, manual_checkpoint):
        component.change(_on_parent_chosen, inputs=chosen_inputs,
                         outputs=[mode, compare, parent_details],
                         show_progress="hidden").then(**view_step)
    # The operation chosen: the session's record, then the chosen parent's
    # record and details for it, then the drawer's start and the run bar.
    continue_operation.change(_on_operation, inputs=[continue_operation, mode],
                              outputs=[mode, *check_cleared], show_progress="hidden").then(
        **chosen_step).then(**view_step)
    # The Branch fields: each edit recorded in the browser first (decision
    # S2: what is sent is what the operator touched), then the check says it
    # is stale and the start checks again; a guard of a rule across settings
    # shows or hides what it guards, as on the base form.
    for name, field in zip(BRANCH_FIELDS, branch_fields, strict=True):
        record_touches(field, name, branch_touched)
    gr.on(triggers=[field.input for field in branch_fields], fn=_on_branch_edited,
          inputs=[mode], outputs=[continue_preview, mode], show_progress="hidden",
          trigger_mode="always_last").then(**view_step)
    branch_by_name = dict(zip(BRANCH_FIELDS, branch_fields, strict=True))
    branch_dependent, branch_guards = rule_fields(BRANCH_FIELDS)
    gr.on(triggers=[branch_by_name[name].change for name in branch_guards],
          fn=_on_branch_guards, inputs=[mode, *(branch_by_name[name] for name in branch_guards)],
          outputs=[branch_by_name[name] for name in branch_dependent], queue=False,
          show_progress="hidden")

    # The run bar's line for the run the drawer would start (`continues_html`).
    mode.change(continues_html, inputs=[mode], outputs=[continues], show_progress="hidden")


#: This page's element id (`shell.page_elem_id`), the scope of `PAGE_CSS`.
_PAGE = f"#shep-page-{TRAINING}"

#: The smallest plot area the run view draws at a wide size (M5-3: about 70 to
#: 100 px at 1536 x 760), and the room uPlot takes around it in the plot's box
#: (the epoch axis, 30 px, and its margins). 140 until B1b; raised toward the
#: skill's 200 with the room the state line's own row and the run bar's blank
#: took: the most at which the owner's window (1850 x 960) still shows all
#: four plots with their axes (its chart body holds 519 px; 200 needs 566), so
#: only a smaller window, or a banner, scrolls the plots.
PLOT_MIN_HEIGHT = 170
PLOT_AXIS_ROOM = 47

#: This page's style sheet (the page contract: listed in `src/webui/app.py`'s
#: ``PAGE_CSS``), with the chart component's. Nothing under 14 px.
PAGE_CSS = charts.CHART_CSS + f"""
{_PAGE} .shep-train {{ gap: 12px !important; }}
{_PAGE} #shep-train-runbar {{ background: var(--shep-panel); border: 1px solid var(--shep-border); border-radius: 10px; padding: 10px 16px !important; gap: 8px !important; flex: none; }}
{_PAGE} .shep-rb-row {{ align-items: center !important; gap: 10px 14px !important; flex-wrap: wrap !important; }}
{_PAGE} .shep-rb-row > * {{ min-width: 0; }}
{_PAGE} .shep-rb-row > .block {{ width: auto !important; }}
{_PAGE} #shep-train-dataset, {_PAGE} #run_name, {_PAGE} #shep-train-view-run {{ flex: 0 0 auto !important; width: auto !important; }}
{_PAGE} #shep-train-dataset > div:not(.wrap), {_PAGE} #run_name > label, {_PAGE} #shep-train-view-run > div:not(.wrap) {{ display: flex !important; align-items: center; gap: 8px; }}
{_PAGE} #shep-train-dataset span[id^="shep-lbl-"], {_PAGE} #run_name span[id^="shep-lbl-"], {_PAGE} #shep-train-view-run span[id^="shep-lbl-"] {{ position: static !important; display: block !important; width: 96px !important; height: auto !important; overflow: visible !important; clip: auto !important; clip-path: none !important; margin: 0 !important; padding: 0 !important; background: none !important; border: 0 !important; white-space: nowrap; font-size: 14px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: var(--shep-text-2) !important; }}
{_PAGE} #shep-train-view-run span[id^="shep-lbl-"] {{ width: auto !important; }}
{_PAGE} #shep-train-dataset .wrap {{ width: clamp(220px, 16vw, 320px); min-width: 0 !important; }}
/* A run's option is 40 characters at most (`renderers.OPTION_CHARS`), its end the start time that
   tells same-name runs apart, and the closed select is the header's one name for the run viewed: its
   box is sized by those 40 characters, never by the window (19vw cut 33 of them at 1536 x 760). The
   text has 40ch (40 digits' width: an option of dates, digits and lowercase names fits whole; one of
   mostly capitals may still end in the shell's titled ellipsis), and the box adds what Gradio draws
   around it: the chevron's 30 px in the input, its 2 x 4 px margins, the 2 x 12 px padding and the
   border. The header's words beside it wrap to their own line where it does not fit. */
{_PAGE} #shep-train-view-run .wrap {{ width: calc({r.OPTION_CHARS}ch + 64px); min-width: 0 !important; }}
{_PAGE} #shep-train-dataset input, {_PAGE} #shep-train-view-run input {{ flex: 1 1 auto !important; width: 0 !important; min-width: 0 !important; }}
{_PAGE} #shep-train-dataset .secondary-wrap, {_PAGE} #shep-train-view-run .secondary-wrap {{ min-width: 0 !important; }}
{_PAGE} #shep-train-dataset .icon-wrap, {_PAGE} #shep-train-view-run .icon-wrap {{ flex: none !important; }}
/* The two selects drawn as fields: a border, the text clear of the chevron, grey while disabled
   (a value too long ends in the shell's titled ellipsis, the rule for every select). */
{_PAGE} #shep-train-dataset .wrap:has(> .wrap-inner), {_PAGE} #shep-train-view-run .wrap:has(> .wrap-inner), {_PAGE} #continue_run .wrap:has(> .wrap-inner), {_PAGE} #checkpoint_dropdown .wrap:has(> .wrap-inner) {{ border: 1px solid var(--shep-border-2) !important; border-radius: 8px !important; background: #fff !important; box-shadow: none !important; }}
{_PAGE} #shep-train-dataset .wrap:has(> .wrap-inner):focus-within, {_PAGE} #shep-train-view-run .wrap:has(> .wrap-inner):focus-within, {_PAGE} #continue_run .wrap:has(> .wrap-inner):focus-within, {_PAGE} #checkpoint_dropdown .wrap:has(> .wrap-inner):focus-within {{ border-color: var(--shep-accent) !important; }}
{_PAGE} #shep-train-dataset input, {_PAGE} #shep-train-view-run input, {_PAGE} #continue_run input, {_PAGE} #checkpoint_dropdown input {{ padding-right: 30px !important; }}
{_PAGE} #shep-train-dataset .icon-wrap, {_PAGE} #shep-train-view-run .icon-wrap, {_PAGE} #continue_run .icon-wrap, {_PAGE} #checkpoint_dropdown .icon-wrap {{ color: var(--shep-text-2) !important; }}
{_PAGE} #shep-train-dataset .wrap:has(input:disabled), {_PAGE} #shep-train-view-run .wrap:has(input:disabled), {_PAGE} #continue_run .wrap:has(input:disabled), {_PAGE} #checkpoint_dropdown .wrap:has(input:disabled) {{ background: var(--shep-grey-50) !important; border-color: var(--shep-grey-200) !important; }}
{_PAGE} #shep-train-dataset input:disabled, {_PAGE} #shep-train-view-run input:disabled, {_PAGE} #continue_run input:disabled, {_PAGE} #checkpoint_dropdown input:disabled {{ color: var(--shep-grey-600) !important; -webkit-text-fill-color: var(--shep-grey-600); cursor: not-allowed; }}
{_PAGE} #shep-train-dataset .wrap:has(input:disabled) .icon-wrap, {_PAGE} #shep-train-view-run .wrap:has(input:disabled) .icon-wrap {{ color: var(--shep-grey-600) !important; }}
{_PAGE} #run_name .input-container {{ width: 260px; }}
{_PAGE} #run_name input {{ border: 1px solid var(--shep-border-2) !important; border-radius: 8px; padding: 6px 10px !important; background: #fff; }}
{_PAGE} #run_name input:disabled {{ background: var(--shep-grey-50); color: var(--shep-muted); }}
{_PAGE} .shep-rb-row > button {{ flex: none !important; width: auto !important; min-width: max-content !important; padding: 0 14px !important; height: 40px; white-space: nowrap; }}
{_PAGE} #shep-train-chip {{ flex: 1 1 360px !important; position: relative; z-index: {CHIP_Z_INDEX}; }}
{_PAGE} .shep-rb-chip {{ display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; font-size: 15px; position: relative; }}
/* The identity's parts wrap one by one (never hidden): "built …", the fingerprint, then the marks with "Details ›". */
{_PAGE} .shep-rb-chip > .shep-ds {{ display: contents; }}
{_PAGE} .shep-rb-chip .shep-ds-marks {{ display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; }}
{_PAGE} .shep-rb-chip > .shep-ds > .shep-sep {{ margin: 0 -2px; }}
{_PAGE} .shep-rb-chip > .shep-ds > .shep-sep:has(+ .shep-ds-marks) {{ display: none; }}
{_PAGE} .shep-rb-inuse {{ color: var(--shep-text-2); font-size: 14px; }}
{_PAGE} .shep-link {{ border: 0; background: none; padding: 0; color: var(--shep-accent-text); font-size: 15px; cursor: pointer; font-weight: 500; }}
{_PAGE} .shep-link:hover {{ text-decoration: underline; }}
{_PAGE} .shep-pop {{ position: absolute; top: calc(100% + 8px); left: 0; z-index: 30; width: min({DETAILS_POPOVER_WIDTH}px, calc(100vw - 48px)); background: #fff; border: 1px solid var(--shep-border-2); border-radius: 10px; box-shadow: 0 10px 28px rgba(20,24,40,.16); padding: 12px 14px; display: flex; flex-direction: column; gap: 10px; font-size: 15px; }}
{_PAGE} .shep-pop[hidden] {{ display: none; }}
{_PAGE} .shep-pop-hd {{ display: flex; align-items: center; gap: 8px; overflow-wrap: anywhere; }}
{_PAGE} .shep-pop-x {{ margin-left: auto; width: 28px; height: 28px; border: 0; background: none; font-size: 20px; cursor: pointer; color: var(--shep-muted); }}
{_PAGE} .shep-pop-sub {{ font-weight: 600; font-size: 15px; }}
{_PAGE} .shep-pop .shep-kv-2 {{ grid-template-columns: max-content max-content max-content max-content; font-size: 14px; gap: 2px 12px; }}
{_PAGE} .shep-pop .shep-v {{ overflow-wrap: anywhere; }}
{_PAGE} .shep-pop .shep-link {{ align-self: flex-start; }}
{_PAGE} #shep-train-pill {{ margin-left: auto; flex: none !important; }}
{_PAGE} .shep-pill {{ display: inline-flex; align-items: center; gap: 6px; height: 30px; padding: 0 12px; border-radius: 999px; font-size: 14px; font-weight: 600; white-space: nowrap; border: 1px solid var(--shep-border-2); background: var(--shep-panel-2); color: var(--shep-text-2); }}
{_PAGE} .shep-pill-sub {{ font-weight: 400; }}
/* The pill's tint by its kind; its mark and word are the shared status mark's (`job_status_html`),
   so a running job's spinner stays blue on the light tint, as every page draws it. */
{_PAGE} .shep-pill-running, {_PAGE} .shep-pill-accent {{ background: var(--shep-accent-50); border-color: var(--shep-accent-100); color: var(--shep-accent-text); }}
{_PAGE} .shep-pill-ok {{ background: var(--shep-green-50); border-color: var(--shep-green-200); color: var(--shep-green-800); }}
{_PAGE} .shep-pill-grey {{ background: var(--shep-grey-50); border-color: var(--shep-grey-200); color: var(--shep-grey-800); }}
{_PAGE} .shep-pill-red {{ background: var(--shep-red-50); border-color: var(--shep-red-200); color: var(--shep-red-700); }}
{_PAGE} #shep-train-reason {{ flex: 1 1 260px !important; }}
{_PAGE} .shep-rb-reason {{ display: flex; align-items: center; justify-content: flex-end; gap: 12px; flex-wrap: wrap; }}
{_PAGE} .shep-rb-why {{ display: inline-flex; align-items: flex-start; gap: 6px; flex-wrap: nowrap; font-size: 15px; color: var(--shep-text-2); text-align: right; }}
{_PAGE} .shep-rb-why > svg, {_PAGE} .shep-rb-why > .shep-tag {{ flex: none; margin-top: 3px; }}
{_PAGE} .shep-rb-why > .shep-tag {{ margin-top: 0; }}
{_PAGE} .shep-rb-why > span:not(.shep-tag) {{ min-width: 0; text-wrap: pretty; }}
/* A reason of two sentences breaks between them (`renderers.word_groups`); a sentence wider than its line wraps inside its group without leaving one word alone (text-wrap: pretty). */
{_PAGE} .shep-rb-why .shep-wg {{ text-wrap: pretty; }}
{_PAGE} .shep-rb-why .shep-link {{ white-space: nowrap; }}
{_PAGE} .shep-rb-progress {{ flex: 1 1 160px; max-width: 380px; height: 8px; border-radius: 4px; background: var(--shep-grey-200); overflow: hidden; }}
{_PAGE} .shep-rb-progress > i {{ display: block; height: 100%; background: var(--shep-accent); }}
{_PAGE} button.primary:not(:disabled) {{ background: var(--shep-accent) !important; border-color: var(--shep-accent) !important; color: #fff !important; }}
{_PAGE} button.primary:not(:disabled):hover {{ background: var(--shep-accent-600) !important; border-color: var(--shep-accent-600) !important; }}
{_PAGE} #shep-train-stop:not(:disabled) {{ color: var(--shep-red-700) !important; border-color: var(--shep-red-200) !important; background: #fff !important; }}
{_PAGE} .shep-rb-row > button:disabled {{ background: var(--shep-grey-50) !important; border: 1px solid var(--shep-grey-200) !important; color: var(--shep-grey-600) !important; opacity: 1 !important; box-shadow: none !important; cursor: not-allowed; }}
{_PAGE} #shep-train-continues {{ display: none !important; }}
{_PAGE} .shep-rb-drawer #shep-train-continue, {_PAGE} .shep-rb-drawer #shep-train-stop, {_PAGE} .shep-rb-drawer #run_name {{ display: none !important; }}
{_PAGE} .shep-rb-drawer #shep-train-continues {{ display: block !important; flex: 0 1 auto !important; }}
{_PAGE} .shep-rb-k {{ font-size: 14px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: var(--shep-text-2); margin-right: 8px; }}
{_PAGE} .shep-rb-confirm {{ display: none !important; }}
{_PAGE} .shep-rb-confirming .shep-rb-confirm {{ display: flex !important; }}
{_PAGE} .shep-rb-confirming .shep-rb-row2 {{ display: none !important; }}
{_PAGE} #shep-train-stop-text {{ flex: 1 1 400px !important; }}
{_PAGE} .shep-rb-confirm-q {{ background: var(--shep-red-50); border: 1px solid var(--shep-red-200); border-radius: 8px; padding: 8px 12px; font-size: 15px; }}
{_PAGE} #shep-train-notice:empty, {_PAGE} #shep-train-notice .html-container:empty {{ display: none; }}
{_PAGE} #shep-train-notice:has(.html-container:empty), {_PAGE} #shep-train-banner:not(:has(.shep-banner)) {{ display: none !important; }}
{_PAGE} #shep-train-notice {{ font-size: 15px; color: var(--shep-amber-800); overflow-wrap: anywhere; }}
{_PAGE} .shep-banner {{ border-radius: 8px; border: 1px solid; padding: 10px 14px; margin-bottom: 6px; font-size: 15px; }}
{_PAGE} .shep-banner-red {{ background: var(--shep-red-50); border-color: var(--shep-red-200); border-left: 4px solid var(--shep-red-700); color: var(--shep-red-700); }}
{_PAGE} .shep-banner-amber {{ background: var(--shep-amber-50); border-color: var(--shep-amber-300); border-left: 4px solid var(--shep-amber-600); color: var(--shep-amber-800); }}
{_PAGE} .shep-banner-hd {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
{_PAGE} .shep-banner-h {{ font-weight: 600; overflow-wrap: anywhere; }}
{_PAGE} .shep-banner-where {{ margin-left: auto; color: var(--shep-text-2); font-size: 14px; }}
/* The last error line the run printed, right under the headline: read without opening or scrolling the tail. */
{_PAGE} .shep-banner-cause {{ margin-top: 4px; color: var(--shep-text); line-height: 1.4; overflow-wrap: anywhere; }}
{_PAGE} .shep-banner-k {{ font-weight: 600; color: var(--shep-red-700); }}
{_PAGE} .shep-banner-v {{ font-family: var(--shep-mono); font-size: 14px; }}
{_PAGE} .shep-banner-more {{ margin-top: 6px; }}
{_PAGE} .shep-banner-more > summary {{ cursor: pointer; font-size: 14px; font-weight: 600; color: var(--shep-red-700); width: max-content; max-width: 100%; }}
{_PAGE} .shep-banner-n {{ font-weight: 400; color: var(--shep-text-2); }}
{_PAGE} .shep-banner-tail {{ margin: 6px 0 0; padding: 6px 10px; background: #fff; border: 1px solid var(--shep-red-200); border-radius: 6px; color: var(--shep-text); font-family: var(--shep-mono) !important; font-size: 14px; line-height: 1.35; white-space: pre-wrap; overflow-wrap: anywhere; max-height: calc(4 * 1.35em + 14px); overflow: auto; }}
{_PAGE} #shep-train-params-bd, {_PAGE} #shep-train-drawer-bd {{ padding: 4px 16px 12px !important; gap: 10px !important; }}
{_PAGE} #shep-train-drawer {{ display: none !important; }}
{_PAGE} .shep-train-drawer-open #shep-train-drawer {{ display: flex !important; }}
{_PAGE} .shep-train-drawer-open #shep-train-params {{ display: none !important; }}
{_PAGE} .shep-drawer-hd {{ flex-wrap: nowrap !important; }}
{_PAGE} #shep-train-drawer-title {{ flex: 1 1 auto !important; }}
{_PAGE} #shep-train-drawer-title .html-container > div {{ display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
{_PAGE} #shep-train-drawer-title h2 {{ font-size: 16px; font-weight: 600; margin: 0; }}
{_PAGE} .shep-x-btn {{ font-size: 0 !important; width: 32px; height: 32px; min-width: 0 !important; padding: 0 !important; background: none !important; border: 0 !important; box-shadow: none !important; }}
{_PAGE} .shep-x-btn::before {{ content: "×"; font-size: 22px; color: var(--shep-muted); }}
/* The dataset comparison and the check result sit in the drawer's one container, the panel:
   a tint and the status mark's lead set them apart, never a border of their own (the skill's
   §2: no bordered box inside a box). A result that can continue has no tint. */
{_PAGE} .shep-cmp, {_PAGE} .shep-check {{ border-radius: 8px; padding: 10px 12px; font-size: 15px; }}
{_PAGE} .shep-cmp p, {_PAGE} .shep-check p {{ margin: 4px 0 0; }}
/* A refusal (the ✕ of the shared status mark: the start refuses it) is tinted red, the skill's §6. */
{_PAGE} .shep-cmp-red, {_PAGE} .shep-check-red {{ background: var(--shep-red-50); color: var(--shep-red-700); }}
{_PAGE} .shep-cmp-ok {{ background: var(--shep-green-50); color: var(--shep-green-800); }}
{_PAGE} .shep-cmp-grey {{ background: var(--shep-grey-50); color: var(--shep-grey-800); }}
{_PAGE} .shep-check-h {{ font-weight: 600; margin-bottom: 4px; }}
{_PAGE} .shep-check code {{ overflow-wrap: anywhere; word-break: normal; }}
/* The check's model spec and settings (`_check_html`): summary lists and one closed disclosure in
   the check, none a bordered box of its own (the skill's §2), at the check's own size (15 px; the
   dense drawer below sets the whole check at 14 px to fit). */
{_PAGE} .shep-check .shep-kv {{ font-size: inherit; gap: 4px 14px; margin-top: 6px; }}
{_PAGE} .shep-inherit {{ margin-top: 8px; }}
{_PAGE} .shep-inherit > summary {{ cursor: pointer; font-weight: 600; min-height: 24px; }}
/* The drawer's footer, under its body and never scrolled with it: the step, the start and Cancel,
   then the start's reason (the render's) and its last notice, each taking the line's rest, wrapping
   under the buttons when narrow; an empty one takes no room. */
{_PAGE} #shep-train-drawer-ft {{ flex: none !important; }}
{_PAGE} #shep-train-drawer-ft > button {{ flex: none !important; width: auto !important; min-width: max-content !important; padding: 0 14px !important; height: 40px; white-space: nowrap; }}
{_PAGE} #shep-train-drawer-ft > button:disabled {{ background: var(--shep-grey-50) !important; border: 1px solid var(--shep-grey-200) !important; color: var(--shep-grey-600) !important; opacity: 1 !important; box-shadow: none !important; cursor: not-allowed; }}
{_PAGE} #shep-train-drawer-ft > .block {{ width: auto !important; }}
{_PAGE} #shep-train-drawer-why, {_PAGE} #shep-train-drawer-notice {{ flex: 1 1 240px !important; min-width: 0 !important; }}
{_PAGE} .shep-drawer-why {{ font-size: 15px; color: var(--shep-text-2); }}
{_PAGE} .shep-drawer-why .shep-rb-why {{ text-align: left; flex-wrap: nowrap; align-items: flex-start; }}
{_PAGE} #shep-train-drawer-why:not(:has(.shep-drawer-why)), {_PAGE} #shep-train-drawer-notice:not(:has(.shep-notice-t)) {{ display: none !important; }}
{_PAGE} #shep-train-drawer-notice {{ font-size: 15px; overflow-wrap: anywhere; }}
/* A step's number and name (the skill's §4: numbered steps), as a field's label reads. */
{_PAGE} .shep-step {{ font-size: 14px; font-weight: 600; color: var(--shep-text-2); white-space: nowrap; }}
{_PAGE} #shep-train-step-check, {_PAGE} #shep-train-step-start {{ flex: none !important; width: auto !important; align-self: center; }}
/* The run view's header row: the Viewing select, then how the run viewed stands to the training
   job, its dataset's relation, its state and Details, wrapping as one line of words: beside the
   select while they need two lines there at most (about 540 px: a state line and Details ›), else
   on a line of their own under it, at the row's width (beside a select that holds a whole option,
   1536 x 760 drew them in three lines over the plots' room). */
{_PAGE} #shep-train-rv-top {{ flex: none !important; align-items: center !important; flex-wrap: wrap !important; gap: 6px 14px !important; padding: 6px 16px !important; }}
{_PAGE} #shep-train-rv-top > * {{ min-width: 0; }}
{_PAGE} #shep-train-run-head {{ flex: 1 1 540px !important; }}
{_PAGE} .shep-rv-hd {{ display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; }}
{_PAGE} .shep-rv-rel {{ font-size: 15px; font-weight: 600; color: var(--shep-text); }}
{_PAGE} #shep-train-view-run .options {{ min-width: min(28rem, calc(100vw - 48px)); }}
{_PAGE} #shep-train-view-run .options li {{ min-width: 100%; }}
/* Details, at the header's end: its popover opens under it, aligned to the right. */
{_PAGE} .shep-rv-tools {{ margin-left: auto; position: relative; }}
{_PAGE} .shep-rv-pop {{ left: auto; right: 0; width: min(560px, calc(100vw - 48px)); }}
{_PAGE} .shep-rv-pop .shep-kv {{ font-size: 14px; gap: 4px 14px; }}
{_PAGE} .shep-rv-snap {{ font-size: 15px; color: var(--shep-text-2); overflow-wrap: anywhere; }}
{_PAGE} #shep-train-charts {{ padding: 8px 16px 6px !important; }}
{_PAGE} .shep-rv-res {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 6px; padding: 8px 16px; border-top: 1px solid var(--shep-border); font-size: 15px; color: var(--shep-text-2); }}
{_PAGE} .shep-rv-res b {{ color: var(--shep-text); margin-right: 8px; }}
{_PAGE} .shep-lock {{ flex: none; }}
{_PAGE} .shep-html svg {{ display: inline-block; vertical-align: -2px; }}
{_PAGE} .shep-train-parent-row {{ align-items: flex-end !important; flex-wrap: nowrap !important; }}
{_PAGE} #continue_run {{ flex: 1 1 auto !important; min-width: 0 !important; }}
/* An open list at least as wide as its few words need (Gradio sizes the list and each entry to
   the box, inline), so an option reads on one line; still fixed and opaque, as Gradio draws it,
   and kept inside the window. */
{_PAGE} #continue_run .options, {_PAGE} #checkpoint_dropdown .options {{ min-width: min(28rem, calc(100vw - 48px)); }}
{_PAGE} #continue_run .options li, {_PAGE} #checkpoint_dropdown .options li {{ min-width: 100%; }}
/* The chosen checkpoint's details: the one summary list in the drawer's panel, no box of its own
   (`parent_details_html`). */
{_PAGE} #shep-train-parent-details:not(:has(.shep-ckpt)) {{ display: none !important; }}
{_PAGE} .shep-ckpt .shep-kv {{ font-size: 14px; gap: 4px 14px; }}
{_PAGE} #shep-train-refresh {{ flex: none !important; width: auto !important; min-width: max-content !important; }}
/* Extra epochs (when shown) beside Check checkpoint, above what the check says. */
{_PAGE} .shep-train-check-row {{ align-items: flex-end !important; flex-wrap: wrap !important; gap: 8px 12px !important; }}
{_PAGE} .shep-train-check-row > * {{ flex: 0 0 auto !important; width: auto !important; }}
{_PAGE} .shep-train-check-row > .form {{ background: none !important; border: 0 !important; box-shadow: none !important; }}
/* Each label beside its box, one line with the button: "Extra epochs ⓘ [ 10 ]  Check checkpoint". */
{_PAGE} #extend_epochs, {_PAGE} #extend_lr {{ padding: 0 !important; background: none !important; border: 0 !important; box-shadow: none !important; flex: none !important; min-width: max-content !important; }}
{_PAGE} #extend_epochs label.container, {_PAGE} #extend_lr label.container {{ display: flex; align-items: center; gap: 8px; flex-wrap: nowrap; }}
{_PAGE} #extend_epochs [data-testid="block-info"], {_PAGE} #extend_lr [data-testid="block-info"] {{ margin: 0 !important; }}
{_PAGE} #extend_epochs input[type="number"], {_PAGE} #extend_lr input[type="number"] {{ width: 96px; }}
{_PAGE} #shep-train-check {{ align-self: flex-end; width: auto !important; min-width: max-content !important; }}
{_PAGE} #shep-train-parents-note {{ margin-top: -4px; }}
/* The base form's fields, and the drawer's Branch fields, in columns of at least 240 px (one in a
   narrow panel), the group drawn as no box of its own (the skill's §2: one container per region,
   the panel). */
{_PAGE} #shep-train-params-bd .gr-group, {_PAGE} #shep-train-params-bd .gr-group > .styler, {_PAGE} #shep-train-params-bd .gr-group .styler > .form, {_PAGE} #shep-train-branch .gr-group, {_PAGE} #shep-train-branch .gr-group > .styler, {_PAGE} #shep-train-branch .gr-group .styler > .form {{ background: transparent !important; border: 0 !important; box-shadow: none !important; }}
{_PAGE} #shep-train-params-bd .gr-group .styler > .form, {_PAGE} #shep-train-branch .gr-group .styler > .form {{ display: grid !important; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 8px 16px !important; border: 0 !important; }}
{_PAGE} #shep-train-branch {{ gap: 4px !important; }}
{_PAGE} #shep-train-branch-hd .shep-note {{ margin: 2px 0 0; font-size: 14px; color: var(--shep-text-2); }}
/* Device's three choices on one line: the row is its own. */
{_PAGE} #shep-train-params-bd .gr-group #device {{ grid-column: 1 / -1; }}
/* The Model accordion: its note and facts, then each section's heading over its fields. */
{_PAGE} #shep-train-spec-facts .shep-note {{ margin: 0 0 6px; font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} .shep-train-spec-facts {{ font-size: 14px; gap: 2px 14px; }}
{_PAGE} .shep-train-sub {{ margin: 6px 0 0; font-size: 15px; font-weight: 600; }}
/* The base form's line on what the operator set (decision S2) beside Reset to reference, and a
   value Start would refuse. */
{_PAGE} .shep-train-setby-row {{ align-items: flex-start !important; flex-wrap: nowrap !important; gap: 8px 12px !important; }}
{_PAGE} #shep-train-setby {{ flex: 1 1 auto !important; min-width: 0 !important; }}
{_PAGE} #shep-train-reset {{ flex: none !important; width: auto !important; min-width: max-content !important; }}
{_PAGE} .shep-setby p {{ margin: 0; font-size: 14px; color: var(--shep-text-2); overflow-wrap: anywhere; }}
{_PAGE} .shep-setby p.shep-amber-t {{ margin-top: 4px; color: var(--shep-amber-800); }}
{_PAGE} .shep-setby .shep-more {{ margin-top: 2px; font-size: 14px; }}
{_PAGE} .shep-setby-kv {{ font-size: 14px; }}
@media (min-width: 1400px) and (min-height: 760px) {{
  {_PAGE} > .column {{ height: 100%; min-height: 0 !important; }}
  {_PAGE} #shep-train {{ flex: 1 1 auto; min-height: 0; height: 100%; flex-wrap: nowrap !important; }}
  {_PAGE} #shep-train-page {{ flex: 1 1 auto; min-height: 0; height: auto; }}
  {_PAGE} #shep-train-params-bd, {_PAGE} #shep-train-drawer-bd {{ flex: 1 1 auto !important; min-height: 0; overflow: auto; flex-wrap: nowrap !important; }}
  {_PAGE} #shep-train-params-bd > *, {_PAGE} #shep-train-drawer-bd > * {{ flex-shrink: 0 !important; }}
  /* The fields as dense as drawn, so the three groups fit at 1920 × 1080 without scrolling. */
  {_PAGE} #shep-train-params-bd .form > .block {{ padding-top: 4px !important; padding-bottom: 4px !important; }}
  {_PAGE} #shep-train-params-bd .form > .block [data-testid="block-info"] {{ margin-bottom: 4px !important; padding-top: 2px !important; padding-bottom: 2px !important; }}
  {_PAGE} #shep-train-params-bd .form > .block input[type="number"], {_PAGE} #shep-train-params-bd .form > .block input[type="text"] {{ padding-top: 6px !important; padding-bottom: 6px !important; }}
  /* The Continue drawer as dense: it fits at 1920 x 1080 with Extra epochs on screen, as
     drawn (M5-3: 821 px of content in 684); a shorter window scrolls its body inside. */
  {_PAGE} #shep-train-drawer-bd {{ gap: 8px !important; }}
  {_PAGE} #shep-train-drawer-bd .block.padded {{ padding-top: 4px !important; padding-bottom: 4px !important; }}
  {_PAGE} #shep-train-drawer-bd [data-testid="block-info"] {{ margin-bottom: 4px !important; padding-top: 2px !important; padding-bottom: 2px !important; }}
  {_PAGE} #shep-train-drawer-bd .wrap label {{ padding-top: 4px !important; padding-bottom: 4px !important; }}
  {_PAGE} #shep-train-drawer-bd input[type="number"], {_PAGE} #shep-train-drawer-bd input[type="text"] {{ padding-top: 6px !important; padding-bottom: 6px !important; }}
  {_PAGE} #shep-train-drawer-bd .shep-cmp, {_PAGE} #shep-train-drawer-bd .shep-check {{ padding: 8px 12px; }}
  /* In every check result at 1920 x 1080 (M5-3: the first check of a finished parent and a
     parent whose dataset differs overflowed by 20 and 118 px): the source's label beside its
     choices, and the results' lines closer. */
  {_PAGE} #continue_operation, {_PAGE} #continue_source, {_PAGE} #continue_device {{ display: flex !important; flex-wrap: wrap; align-items: center; gap: 2px 10px; }}
  {_PAGE} #continue_operation > [data-testid="block-info"], {_PAGE} #continue_source > [data-testid="block-info"], {_PAGE} #continue_device > [data-testid="block-info"] {{ margin: 0 !important; }}
  /* The run and checkpoint selects keep their labels above them, so each box has the drawer's
     width for its option's few words (the run's Refresh beside its box). */
  {_PAGE} #shep-train-drawer-bd {{ gap: 6px !important; }}
  {_PAGE} #shep-train-drawer-ft {{ padding: 8px 16px !important; }}
  {_PAGE} #shep-train-drawer-ft .html-container, {_PAGE} #shep-train-drawer-ft .shep-html {{ padding: 0 !important; }}
  {_PAGE} #shep-train-drawer-bd .shep-cmp, {_PAGE} #shep-train-drawer-bd .shep-check {{ font-size: 14px; line-height: 1.4; }}
  {_PAGE} #shep-train-drawer-bd .shep-cmp p, {_PAGE} #shep-train-drawer-bd .shep-check p {{ margin-top: 2px; }}
  {_PAGE} #shep-train-drawer-bd .shep-check-h {{ margin-bottom: 2px; }}
  {_PAGE} #shep-train-drawer-bd .shep-inherit {{ margin-top: 4px; }}
  /* Never clipped, and never cramped: each of the four plots keeps a plot area of at least
     PLOT_MIN_HEIGHT px whatever its title's keys take (the minimum is the plot box's, and each
     box around it grows to hold its content). The run view is three rows: its header, the
     charts, and the resources line. The charts take what the other two leave (the GB10 WebUI
     recheck: the resources line, kept over the plots, hid the last axis). In their row the
     toolbar is a row of its own, never over a plot (it was sticky over the top row's titles
     and plots once scrolled), and when the plots' minimum does not fit, only the plots' body
     under it scrolls: nothing is ever drawn over an axis. */
  /* (Gradio writes "overflow: visible" on the block itself, inline: only !important wins.) */
  {_PAGE} #shep-train-runview > #shep-train-charts {{ flex: 1 1 0 !important; min-height: 0 !important; overflow: hidden !important; display: flex; flex-direction: column; }}
  {_PAGE} #shep-train-charts .shep-chart-plot {{ min-height: {PLOT_MIN_HEIGHT + PLOT_AXIS_ROOM}px; }}
  /* The boxes between the charts' row and the grid take the row's height (min-height 0), so
     the grid's two rows share it, each at least what its own cards need (a card keeps its
     content's minimum: title, keys, the plot's minimum, the axis label); rows that need more
     than the body has overflow it, and the body scrolls. */
  {_PAGE} #shep-train-charts .shep-charts, {_PAGE} #shep-train-charts .shep-charts-grid {{ min-height: 0; }}
  /* The toolbar's row as dense as drawn (its buttons 30 px tall, the skill's §10 asks 24) and
     the grid's rows closer, so the room goes to the plots. */
  {_PAGE} #shep-train-charts {{ padding-top: 6px !important; padding-bottom: 4px !important; }}
  {_PAGE} #shep-train-charts .shep-charts-bar {{ flex: none; min-height: 34px; padding-bottom: 4px; }}
  {_PAGE} #shep-train-charts .shep-charts-btn {{ height: 30px; }}
  {_PAGE} #shep-train-charts .shep-charts-icon {{ height: 28px; }}
  {_PAGE} #shep-train-charts .shep-charts-grid {{ row-gap: 6px; }}
  {_PAGE} #shep-train-charts .shep-charts-body {{ flex: 1 1 0; min-height: 0; overflow-y: auto; }}
  {_PAGE} #shep-train-charts .shep-chart {{ min-height: auto; }}
  {_PAGE} #shep-train-charts .shep-charts-grid {{ grid-auto-rows: minmax(min-content, 1fr); }}
  {_PAGE} #shep-train-runview > #shep-train-rv-top {{ position: relative; z-index: 6; }}
  {_PAGE} #shep-train-runview > #resource_display {{ flex: none !important; }}
  {_PAGE} #shep-train-charts > .html-container {{ flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }}
  {_PAGE} #shep-train-charts > .html-container > div {{ flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }}
  {_PAGE} .shep-rb-row1 {{ flex-wrap: nowrap !important; }}
  {_PAGE} #shep-train-chip {{ flex: 1 1 auto !important; }}
  {_PAGE} #shep-train-charts .shep-charts {{ flex: 1 1 auto; }}
}}
@media (min-width: 1400px) and (min-height: 760px) and (max-height: 899px) {{
  {_PAGE} #shep-train-rv-top {{ min-height: 40px; padding-top: 4px !important; padding-bottom: 4px !important; }}
  {_PAGE} .shep-rv-res {{ padding-top: 6px; padding-bottom: 6px; }}
}}
@media (max-width: 1399px), (max-height: 759px) {{
  {_PAGE} #shep-train-chip {{ flex: 1 1 240px !important; }}
  {_PAGE} .shep-rb-reason {{ justify-content: flex-start; }}
  {_PAGE} .shep-rb-why {{ text-align: left; }}
  {_PAGE} .shep-banner-tail {{ max-height: none; overflow: visible; }}
}}
@media (max-width: 1199px) {{
  {_PAGE} #shep-train-reason {{ flex: 1 1 100% !important; order: 10; }}
  {_PAGE} #shep-train-start {{ margin-left: auto; }}
  {_PAGE} #run_name .input-container {{ width: 200px; }}
}}
"""


__all__ = ["PAGE_CSS", "create_training_tab", "training_chart_csv", "training_chart_series",
           "training_charts"]
