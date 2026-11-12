"""
How a run that starts from a parent checkpoint continues, decided once.
=======================================================================
Contract M3a-3 and M3a-4 (`docs/working/PLAN_PROVENANCE_CONTRACT.md`,
decision 8). A run given a parent (`resume_from`) either **resumes** the
parent's active stage, which is unfinished, or **extends** a finished stage at a
fixed rate. Which of the two, with which settings and from which epoch, is
decided here, before the Trainer and before any output exists, from the
parent's own record, never from its file name or the run it sits in. The CLI,
the training service, the console's preview and the probe all go through
`plan_continuation`; the entry point decides again on the bytes it actually
restores, so a preview is never the check.

A model bundle run given a parent to **branch** (`branch_from`, contract §9.2,
BACKLOG 23) is planned by `plan_branch`: a new run of the parent's stage and
task, with its weights, settings and data, changing some of the settings on
the stage's closed list (`src.config.recipe.branch_settings`) and recording
each change; a new optimizer, schedule and stage.

**What a new run takes** (a base, a finetune; ops-wiring round, decision S1)
is stated here once, `new_run_settings`, filled once, `new_run_values`, and
judged once, `new_run_problem` (a setting a run does not read: `unread_problem`):
the training process, the job runner's starts and the API's start read them.

**What the parent's state allows:**

- an unfinished stage (`progress.end` is `None` or `"paused"`) is resumed:
  only its remaining epochs run;
- a finished stage (`"finished"`) is extended only when `extend_epochs` is
  given; resuming it would have nothing to run;
- a run that stopped early (`"stopped_early"`) is neither: resuming would reset
  the patience that stopped it, and extending would treat a stop as a pause;
- a weights-only checkpoint, or one without the record contract M3a writes,
  holds no state to continue from.

Each refusal says why. Nothing falls back to training from scratch.

**Learning settings are inherited and locked** (decision 8). Every field of
the request outside `OPERATIONAL_FIELDS` takes the parent's value, as **the one
settings decode** reads its record (`decode_learning_settings`; research
comparison Stage 9, review F1): its recorded settings (`config.requested`)
under the current training rules, and under released rules 1 the same with
the one setting rules 1 did not have, `frozen_parts`, read as "none" from the
released semantics the record shows (rules 1 had no freeze; every part
trained). The request holds the current settings only (the `TrainConfig`
fields, `src.config.training_fields.names()` for a preview), so a setting a
parent records and this build no longer has is not inherited, and a current
setting the decode does not give is refused. A field the operator set is the
operator's, whatever its value, a default included: set to another value than
the parent's it is refused, never ignored; one not set is inherited and shown
in the plan, as recorded (`inherited`) or as the decode gave it (`decoded`,
with its source: one source per setting). Changing settings from a parent is
a branch (`plan_branch`), and only the settings on its closed list, which
never holds the freeze. The parent is judged by what this build's one
executor can faithfully run, never by its rules number alone; rules it cannot
run (K4's, none recorded; rules it does not know) are refused first, with the
reason.

**Provenance and mode rules stay apart.** Whether the parent was trained on
this workspace's data and graph is M2.3's check (`train_model.check_resume_
parent`), applied by the entry point; this module decides only what the parent's
state and settings allow.

It sits in `src.utils`, beside the run layout (`checkpoint_paths`), and imports
nothing from the training stack, so the training service and the console can
preview a selection without it (`.import-linter.ini`: the WebUI must not import
`src.training`). It reads no file itself except to see that a named
best-weights file exists.

Module: src/utils/run_plan.py
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional

from src.config.model_types import WRITTEN_BEFORE_SPEC
from src.config.recipe import (
    FROZEN_PARTS, branch_settings, frozen_parts_problem, reference_settings, unread_settings,
)
from src.config.training_fields import FIELDS, cross_field_problems, value_problem
from src.utils.checkpoint_paths import PLAIN_DEPTH, PLAIN_VALUES, plain_data_problem

#: Fields that say where a run reads and writes, how it is run on this host, or
#: how this run is continued: never inherited, free to differ from the parent.
#: Every other `TrainConfig` field is a learning setting.
OPERATIONAL_FIELDS = frozenset({
    "data_dir", "config_file", "run_name", "run_dir", "resume_from",
    "device", "compile",
    "pause_after_epoch", "extend_epochs", "extend_lr",
    # The model bundle path: a finetune's or a branch's parent is in its model
    # record, not a setting a continued run inherits.
    "finetune_from", "branch_from",
})


class RunPlanRefused(ValueError):
    """The parent cannot be continued as requested; the message says why and
    what to do instead."""


#: What names a run's stage and task: the run's identity, not a setting of it.
_RUN_IDENTITY = frozenset({"learning_stage", "task"})

#: Every setting a run of any stage may state: each learning setting (a field
#: of the field spec outside `OPERATIONAL_FIELDS`, `TrainConfig`'s fields) but
#: the run's stage and task, which name the run.
RUN_SETTINGS = frozenset(spec.name for spec in FIELDS) - OPERATIONAL_FIELDS - _RUN_IDENTITY


def new_run_settings(learning_stage: str, task: Optional[str] = None) -> frozenset:
    """**What a new run of `learning_stage` (and `task`) takes** (ops-wiring
    round, decision S1): every setting a run may state (`RUN_SETTINGS`) but
    the recipe's settings this stage and task do not read
    (`recipe.unread_settings`: another task's, or any task setting on a base),
    which are refused, never ignored (`unread_problem`). The one statement of
    that rule: the training process refuses an explicit setting outside it
    (`train_model.train`, the CLI and every spawned job), the job runner's
    start takes exactly what every run it makes takes (the intersection over
    a Fine-tune start's tasks), and the API's base start offers it."""
    return frozenset(RUN_SETTINGS - unread_settings(learning_stage, task))


def _runs_words(learning_stage: str, tasks: Collection[Optional[str]]) -> str:
    """The runs a start makes, in a refusal's words: "a base run", "a
    finetune for disease", "every finetune of disease and gene"."""
    named = [task for task in tasks if task is not None]
    if not named:
        return f"a {learning_stage} run"
    if len(named) == 1:
        return f"a {learning_stage} for {named[0]}"
    return f"every {learning_stage} of {', '.join(named[:-1])} and {named[-1]}"


def unread_problem(learning_stage: str, tasks: Collection[Optional[str]],
                   stated: Collection[str]) -> Optional[str]:
    """Why a start of `learning_stage` that makes a run per one of `tasks`
    (``(None,)`` for a base) cannot take the settings `stated`, or None:
    **a setting is taken only when every run of the start reads it**, the
    intersection of `new_run_settings` over `tasks`; any other is refused,
    never ignored, in this one sentence. Asked by the training process of
    every run (`train_model.train`: one task, the CLI and every spawned job)
    and by the job runner's starts (`new_run_problem`), so the CLI, every
    spawned job and the service refuse one request in the same words."""
    common = frozenset.intersection(*(new_run_settings(learning_stage, task)
                                      for task in tasks))
    unread = sorted(set(stated) - common)
    if not unread:
        return None
    one = len(unread) == 1
    every = len([task for task in tasks if task is not None]) > 1
    return (f"{', '.join(unread)} {'is not a setting' if one else 'are not settings'} "
            f"{_runs_words(learning_stage, tasks)} reads, so "
            f"{'it' if one else 'they'} would not take effect "
            f"({'all of them read' if every else 'it reads'} {', '.join(sorted(common))}). "
            f"Remove {'it' if one else 'them'}.")


def new_run_problem(learning_stage: str, tasks: Collection[Optional[str]],
                    settings: Mapping[str, Any]) -> Optional[str]:
    """**The one check of the settings a new run is started with** (BACKLOG
    54; decisions S1, S3): why `settings`, the operator's explicit settings
    with their values, cannot start the runs of `learning_stage` a start
    makes, one per one of `tasks` (``(None,)`` for a base), or None. Asked by
    the training process of a new run (`train_model.train`, the CLI and every
    spawned job, with its one task) and by the job runner's base start and
    Fine-tune start before anything is spawned (`training_manager`), so one
    request has one judgement in one sentence wherever it is sent:

    - the freeze first, by its one check (`recipe.frozen_parts_problem`),
      before the settings a run does not read, so a base asked to freeze
      hears that nothing would train;
    - a setting is taken only when every run reads it (`unread_problem`);
    - its value only when its field spec takes it (`value_problem`: its kind,
      strictly, its bounds and choices), None only where a run's reference
      value is None or the field's default is;
    - the rules across settings (`training_fields.cross_field_problems`), on
      each run's settings as they are filled (`new_run_values`), for what
      `settings` set: one the run would not read is refused."""
    if "frozen_parts" in settings:
        frozen = frozen_parts_problem(learning_stage, settings["frozen_parts"])
        if frozen:
            return frozen
    problem = unread_problem(learning_stage, tasks, settings)
    if problem:
        return problem
    references = [reference_settings(learning_stage, task)[0] for task in tasks]
    problems = [problem for name, value in sorted(settings.items())
                for problem in [value_problem(name, value, allow_none=any(
                    name in values and values[name] is None for values in references))]
                if problem]
    if problems:
        return " ".join(problems)
    for task in tasks:
        problems = cross_field_problems(new_run_values(learning_stage, task, settings),
                                        settings)
        if problems:
            return " ".join(problems)
    return None


def new_run_values(learning_stage: str, task: Optional[str],
                   given: Mapping[str, Any]) -> Dict[str, Any]:
    """The settings a new run of `learning_stage` and `task` trains with, given
    `given`, the operator's: each of `new_run_settings` the operator's when
    given, else the stage's reference (`recipe.reference_settings`), else this
    build's default (the field spec's, `TrainConfig`'s, pinned equal). **The
    one fill of a new run**: the training process fills its configuration
    with it (`train_model.apply_stage_recipe`), and the job runner judges a
    start's runs by it before it spawns (`cross_field_problems`). A model spec
    left unset is None here, the stage's reference spec, and a finetune's split
    seed is its base's, which the training process takes from the base."""
    defaults = {spec.name: (list(spec.default) if isinstance(spec.default, tuple)
                            else spec.default) for spec in FIELDS}
    references, _ = reference_settings(learning_stage, task)
    return {name: (given[name] if name in given else references[name] if name in references
                   else defaults[name])
            for name in new_run_settings(learning_stage, task)}


def finetune_split_seed_problem(split_seed: Any, base_split: Any) -> Optional[str]:
    """Why a finetune stating `split_seed` cannot train on its base's split,
    recorded as `base_split` (the base's ``data_roles["split"]``), or None:
    **a finetune trains on its base's split** (README G4), so a seed other
    than the base's is refused. The training process (`train_model.train`)
    and the job runner's Fine-tune start, before any child starts, ask it of
    a stated seed alike."""
    recorded = base_split.get("seed") if isinstance(base_split, Mapping) else None
    if recorded is not None and _same(split_seed, recorded):
        return None
    return (f"split_seed {split_seed!r} differs from the base's {recorded!r}; a finetune trains "
            "on its base's split")


@dataclass(frozen=True)
class ContinuationPlan:
    """What a run started from a parent does.

    `settings` is the whole configuration the run uses: the parent's learning
    settings and the request's operational fields. `inherited` lists the
    learning settings taken from the parent's record that the request did not
    set explicitly, for the read-only summary shown before start; `decoded`
    the ones the settings decode gave instead, each with its source, so each
    setting has one source.
    """

    mode: str                       # "resume" | "extend"
    action: str                     # one sentence, for people
    settings: Dict[str, Any]
    inherited: Dict[str, Any]
    stage: Dict[str, Any]           # the active stage this run continues or opens
    start_epoch: int                # the next epoch index to run
    end_epoch: int                  # the stage's end, exclusive
    #: Early stopping's best weights: `None`, `{"kind": "self"}` (the parent's
    #: own weights) or `{"kind": "file", "path": Path, "sha256": str}` (a file
    #: the entry point reads and checks by digest).
    best_weights: Optional[Dict[str, Any]]
    #: The parent's best-k entries, each with its file resolved to a path, and
    #: the parent itself when it is the file of early stopping's best epoch
    #: (`_locate_best_weights`). They belong to the run that wrote them: ranked
    #: here, never deleted.
    best_k: List[Dict[str, Any]] = field(default_factory=list)
    #: The parent's callback state as recorded.
    callbacks: Optional[Dict[str, Any]] = None
    #: Each learning setting the run takes from the settings decode that the
    #: parent's record does not hold as a value, with its source in words
    #: (`decode_learning_settings`); never also in `inherited`, and empty for
    #: a parent of the current rules.
    decoded: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BranchPlan:
    """What a branch of a parent does (`plan_branch`).

    `settings` is the whole configuration the run uses: the parent's learning
    settings, the request's changes and its operational fields. `inherited`
    lists the settings taken from the parent that the request did not set;
    `changes` each changed setting as ``{"parent": old, "new": new}``, as the
    record keeps them (`parent.changes`); `decoded` as a continuation's.
    """

    action: str
    settings: Dict[str, Any]
    inherited: Dict[str, Any]
    changes: Dict[str, Dict[str, Any]]
    decoded: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodedSettings:
    """A parent's learning settings in this build's representation
    (`decode_learning_settings`): `settings`, every setting its record holds
    and every one the decode gave, and `decoded`, each one the decode gave
    with its source in words, as a continued run records it."""

    settings: Dict[str, Any]
    decoded: Dict[str, str]


def _same(a: Any, b: Any) -> bool:
    """Two setting values equal as settings: numbers by value, sequences
    element-wise, anything else exactly."""
    numbers = (int, float)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b and type(a) is type(b)
    if isinstance(a, numbers) and isinstance(b, numbers):
        return float(a) == float(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def recorded_training_settings(training: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """A checkpoint's training settings as it records them, for a list or an
    export (research comparison Stage 8), from the run listing's one read of
    the file (`ListedCheckpoint.training`):

    - ``rules``: the training rules it was written under (`training_rules`),
      as recorded;
    - ``learning``: its learning settings, every requested setting
      (`config.requested`) outside `OPERATIONAL_FIELDS`, the same split a
      continued run inherits by (`_inherit_settings`), whether or not a
      setting exists in this build (a freeze setting among them);
    - ``effective``: what took effect, as recorded (`config.effective`: the
      device, AMP, compile, the loaders and the schedule).

    Each is `None` where the file records none (a weights-only file has no
    configuration), and ``unrecorded`` names each such part, so what has no
    source says so. Read as recorded; nothing is filled from this build's
    recipe or from another file of the run.

    **Each part is shown only as plain data an answer can carry**, by the
    one rule for a value read from a file (`checkpoint_paths.plain_data`,
    review fix RC-R1: text, no value holding itself, within the safe loader's
    bounds, `PLAIN_DEPTH` and `PLAIN_VALUES`: the listing reads a checkpoint
    with `weights_only=True`, which reads any depth and shares values
    freely). A part that is not (a setting or a setting's name that is not
    text, such as a lone surrogate an argument decoded from bytes carries) is
    `None` and ``unshown`` says why by its name, naming the file's field and
    never the value, so one setting never makes the whole list fail and the
    other parts are shown. Valid Unicode is kept exactly. **A number that is
    not finite is a value a setting may record** (``max_grad_norm: inf``,
    which `training_fields.value_problem` accepts), so it is shown, as its
    text (``"inf"``, ``"-inf"``, ``"nan"``: `_as_shown`), the form the export
    has always given it, in every answer alike: the list's, the results
    query's, Compare's and both export formats'."""
    training = training if isinstance(training, Mapping) else {}
    requested = training.get("requested")
    effective = training.get("effective")
    out: Dict[str, Any] = {
        "rules": training.get("rules"),
        "learning": ({name: value for name, value in requested.items()
                      if name not in OPERATIONAL_FIELDS}
                     if isinstance(requested, Mapping) else None),
        "effective": dict(effective) if isinstance(effective, Mapping) else None,
    }
    unshown: Dict[str, str] = {}
    for part, recorded_as in _TRAINING_PARTS_RECORDED_AS.items():
        problem = plain_data_problem(out[part], recorded_as, depth=PLAIN_DEPTH,
                                     values=PLAIN_VALUES, finite=False)
        if problem is not None:
            out[part] = None
            unshown[part] = f"its {problem}, so it is not shown"
        else:
            out[part] = _as_shown(out[part])
    out["unrecorded"] = [name for name in _TRAINING_PARTS_RECORDED_AS
                         if out[name] is None and name not in unshown]
    out["unshown"] = unshown
    return out


#: Where a checkpoint keeps each part of its training settings, by the name
#: `recorded_training_settings` gives it.
_TRAINING_PARTS_RECORDED_AS = {"rules": "training_rules", "learning": "config.requested",
                               "effective": "config.effective"}


def _as_shown(value: Any) -> Any:
    """A part of a checkpoint's training settings as every answer carries it
    (`recorded_training_settings`): each number that is not finite as its
    text (``str``: ``"inf"``, ``"-inf"``, ``"nan"``), everything else as
    recorded. Walked only once the one rule has read the part
    (`plain_data`), so it is bounded and holds no value on its own path."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _as_shown(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_shown(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_as_shown(item) for item in value)
    return value


def _progress_text(completed: int, start: int, target: int) -> str:
    done = completed - start
    return f"{done}/{target} done, {target - done} left"


def resolve_recorded_file(recorded: str, parent_checkpoint_dir: Path) -> Path:
    """A file a checkpoint names relative to its own run's checkpoint
    directory (a best-k file, the best weights), as a path here."""
    return Path(os.path.normpath(parent_checkpoint_dir / recorded))


def plan_continuation(
    parent: Mapping[str, Any],
    parent_path: Path,
    request: Mapping[str, Any],
    explicit: Collection[str],
    *,
    device: str,
    parent_sha256: Optional[str] = None,
    record_version: int,
) -> ContinuationPlan:
    """Decide how a run over `request` continues from `parent`.

    Args:
        parent: the parent checkpoint, as read.
        parent_path: where it was read from; files it names (best-k, the best
            weights) are resolved beside it.
        request: the requested configuration, every `TrainConfig` field.
        explicit: the fields the operator set (configuration file keys, CLI
            flags, request fields); a learning setting among them that differs
            from the parent's is refused.
        device: the device the run will use, resolved.
        parent_sha256: the digest of the read `parent` came from. The parent
            is named by it when it is itself the file of early stopping's best
            epoch, so the child's checkpoints can name it in turn.
        record_version: the record version the continuing run writes, the
            model bundle's (`src.models.bundle.record.MODEL_RECORD_VERSION`,
            which every caller passes: `src.utils` does not import
            `src.models`). A parent of another version is refused.

    Raises:
        RunPlanRefused: with the reason and the remedy.
    """
    where = f"The parent {parent_path.name}"
    _check_record_version(parent, record_version, where)
    if parent.get("checkpoint_kind") != "training_state":
        weights_from = parent.get("weights_from") or {}
        raise RunPlanRefused(
            f"{where} holds weights only ({weights_from.get('kind', 'unknown')} weights of "
            f"epoch {weights_from.get('epoch', '?')}), without the optimizer, schedule or "
            "progress a run continues from. Choose the run's last.pt or another "
            "training-state checkpoint.")

    progress = parent.get("progress") or {}
    stage = dict(progress.get("stage") or {})
    completed = int(progress.get("completed_epochs", 0))
    start = int(stage.get("start_epoch", 0))
    target = int(stage.get("target_epochs", 0))
    end = progress.get("end")
    kind = stage.get("kind", "original")
    stage_name = "extended training" if kind == "extension" else "original training"
    extend_epochs = request.get("extend_epochs")

    if end == "stopped_early":
        raise RunPlanRefused(
            f"{where} stopped early at epoch {completed} ({progress.get('stop_reason')}). "
            "Resuming would reset the patience that stopped it, and extending would treat "
            "a stop as a pause, so neither is offered. Serve or measure it, or start a new "
            "run.")
    if end == "finished":
        if not extend_epochs:
            raise RunPlanRefused(
                f"{where} finished its {stage_name} ({target}/{target} epochs), so there is "
                "nothing left to resume. To train further, extend it at a fixed rate with "
                "extend_epochs.")
        mode = "extend"
    elif end in (None, "paused"):
        if extend_epochs:
            raise RunPlanRefused(
                f"{where} has not finished its {stage_name} "
                f"({_progress_text(completed, start, target)}); extend_epochs applies only to "
                "a finished stage. Resume it without extend_epochs.")
        mode = "resume"
    else:
        raise RunPlanRefused(f"{where} records an unknown end state {end!r}.")

    settings, inherited, _, decoded = _inherit_settings(parent, request, explicit, where)
    _check_amp_takes_effect_alike(parent, settings, device, where)
    if mode == "resume" and request.get("extend_lr") is not None:
        raise RunPlanRefused("extend_lr applies only when extending a finished stage.")

    if mode == "resume":
        if stage.get("schedule", {}).get("type") == "external":
            raise RunPlanRefused(
                f"{where} was trained with a scheduler built outside the Trainer; its "
                "schedule cannot be rebuilt from the record.")
        run_stage = stage
        start_epoch, end_epoch = completed, start + target
        action = (f"Resume the {stage_name} ({_progress_text(completed, start, target)})")
    else:
        extend_epochs = int(extend_epochs)
        if extend_epochs < 1:
            raise RunPlanRefused(f"extend_epochs must be at least 1 (got {extend_epochs}).")
        schedule = _fixed_rates(parent, request.get("extend_lr"), where)
        run_stage = {"kind": "extension", "start_epoch": completed,
                     "target_epochs": extend_epochs, "schedule": schedule}
        start_epoch, end_epoch = completed, completed + extend_epochs
        action = (f"Extend training by {extend_epochs} epochs at {_rates_text(schedule['lrs'])} "
                  f"(epochs {completed + 1}-{completed + extend_epochs}, new target "
                  f"{completed + extend_epochs})")

    pause = request.get("pause_after_epoch")
    if pause is not None and int(pause) <= start_epoch:
        raise RunPlanRefused(
            f"pause_after_epoch {pause} is not after this run's start (epoch {start_epoch} "
            "is already done).")

    best_weights, best_k = _locate_best_weights(parent, parent_path, where, parent_sha256)
    return ContinuationPlan(
        mode=mode, action=action, settings=settings,
        inherited=inherited, stage=run_stage, start_epoch=start_epoch, end_epoch=end_epoch,
        best_weights=best_weights, best_k=best_k, callbacks=parent.get("callbacks"),
        decoded=decoded,
    )


def decoded_sentence(decoded: Mapping[str, str]) -> Optional[str]:
    """The settings a plan took from the settings decode (`plan.decoded`), in
    one sentence for the training process's log, beside the plan's action and
    its inherited settings, or None when it took none. The job runner's preview
    carries the same `decoded`, which the Continue drawer lists with the
    settings taken from the checkpoint, each with this source."""
    if not decoded:
        return None
    return ("Not in the parent's record: "
            + "; ".join(f"{name} {source}" for name, source in sorted(decoded.items())) + ".")


def _check_record_version(parent: Any, record_version: int, where: str) -> None:
    """The record and the rules a continued run keeps, checked before anything
    else of the parent is read."""
    if not isinstance(parent, Mapping):
        # A read that is not a checkpoint record at all (a list, a number):
        # only a direct caller can pass one; the training service's preview
        # reads such a file as an empty record.
        raise RunPlanRefused(
            f"{where} holds no checkpoint record this version can continue (it holds a "
            f"{type(parent).__name__}, not a record). Start a new run.")
    found = parent.get("record_version")
    if found != record_version:
        # The model bundle path (record version 2, contract §9.3 hazard 1): a
        # parent written before the model spec, or at another record version,
        # is not continued across the boundary. Said as the record reader
        # (`read_model_record`) says it, which the training process and the
        # service's preview read every parent with first: its one phrase for a
        # file without a spec, its version sentence for one with a spec.
        if parent.get("model_spec") is None:
            raise RunPlanRefused(f"{where}: {WRITTEN_BEFORE_SPEC}")
        shown = found if found is None or type(found) is int else f"a {type(found).__name__}"
        raise RunPlanRefused(f"{where}: checkpoint record version {shown} is not "
                             f"{record_version}; this build reads only its own")
    _check_training_rules(parent, where)


#: The released training rules a parent may have been written under that this
#: build's executor can faithfully continue, written as the versions
#: themselves: its own (`TRAINING_RULES_VERSION`, 2) and released rules 1,
#: whose record the settings decode reads into this build's settings
#: (`decode_learning_settings`). Never derived from the current constant: a
#: rules change does not drop a released version silently. A pin
#: (tests/unit/test_training_fields.py) holds it equal to the released rules'
#: entries, so a new version fails there until a person decides how each
#: earlier one decodes, or records why one retires.
CONTINUED_RULES = (1, 2)


def _check_training_rules(parent: Mapping[str, Any], where: str) -> None:
    """Refuse a parent of training rules this build's executor cannot run as
    they were: none recorded (K4's, the reviewer's K5 review, P2-1: continuing
    it would change early stopping and the base's validation pairs mid-run,
    beside its restored best value and history) or rules it does not know,
    whose meaning is not guessed. Checked before anything else of the parent
    but its record version."""
    rules = parent.get("training_rules")
    if type(rules) is int and rules in CONTINUED_RULES:
        return
    if rules is None:
        raise RunPlanRefused(
            f"{where} was trained under other training rules (none recorded, K4's): this "
            "build turns early stopping on exactly when a patience is in effect and validates "
            "a base on the pairs its split fixes, so continuing it would change both mid-run. "
            "Start a new run, or finetune from it: a finetune takes its weights, not these "
            "rules.")
    shown = rules if type(rules) is int else f"a {type(rules).__name__}"
    raise RunPlanRefused(
        f"{where} was trained under other training rules (rules {shown}), which this build "
        f"does not know (it continues rules {', '.join(str(r) for r in CONTINUED_RULES)}), "
        "so what its settings mean is not guessed. Start a new run, or finetune from it: a "
        "finetune takes its weights, not these rules.")


def decode_learning_settings(parent: Mapping[str, Any], where: str) -> DecodedSettings:
    """**The one settings decode** (research comparison Stage 9, owner's O4,
    review F1): a parent's recorded learning settings in this build's
    representation, for the one executor this build has. Every resume,
    extension and branch reads its parent through it (`_inherit_settings`),
    in the training process and in the job runner's preview alike.

    - **The current rules** (`TRAINING_RULES_VERSION`): the record's settings
      (`config.requested`) as recorded.
    - **Released rules 1**: the same, and `frozen_parts` "none", because the
      record itself shows rules 1's semantics: rules 1 had no freeze setting,
      so it records none, and a weight transfer, where it records one (a
      finetune, a branch), records every part trainable ("all"). Nothing else
      is filled: a current setting a rules-1 record lacks stays missing and is
      refused as such; no default of this build stands in for it.
    - **Anything else is refused** with its reason: rules this build cannot
      run (`_check_training_rules`), no recorded settings, a rules-1 record
      that names a freeze, a freeze value outside the setting's closed
      values or one that contradicts the record's stage
      (`frozen_parts_problem`), and, under either rules, a weight transfer
      that contradicts the freeze (one check,
      `_check_transfer_matches_freeze`).

    The parent's record is read, never changed; what the decode gave is
    returned with its source (`DecodedSettings.decoded`), which a continued
    run records.

    Raises:
        RunPlanRefused: with the reason.
    """
    _check_training_rules(parent, where)
    recorded = (parent.get("config") or {}).get("requested")
    if not isinstance(recorded, Mapping):
        raise RunPlanRefused(
            f"{where} does not record the settings it was trained with, so they cannot "
            "be inherited.")
    settings = dict(recorded)
    decoded: Dict[str, str] = {}
    transfer = parent.get("weight_transfer")
    if parent.get("training_rules") == 1:
        if "frozen_parts" in recorded:
            raise RunPlanRefused(
                f"{where} records frozen_parts {recorded['frozen_parts']!r} under training "
                "rules 1, which had no freeze setting, so its record contradicts its rules.")
        settings["frozen_parts"] = "none"
        decoded["frozen_parts"] = (
            "none, decoded from rules 1: rules 1 had no freeze and trained every part"
            + (" (its weight transfer records trainable 'all')" if transfer is not None else ""))
    if "frozen_parts" in settings:
        value = settings["frozen_parts"]
        problem = frozen_parts_problem(parent.get("learning_stage"), value)
        if problem:
            raise RunPlanRefused(f"{where} cannot be continued with its recorded freeze: "
                                 f"{problem}")
        _check_transfer_matches_freeze(transfer, value, "frozen_parts" in decoded, where)
    return DecodedSettings(settings=settings, decoded=decoded)


def _check_transfer_matches_freeze(transfer: Any, frozen_parts: str, decoded: bool,
                                   where: str) -> None:
    """**One check of a parent's weight transfer against its freeze, under
    every rules** (`decode_learning_settings`): a transfer, where the record
    holds one (a finetune, a branch), records every part trainable ("all")
    exactly when the freeze keeps nothing from training
    (`FROZEN_PARTS[frozen_parts]` empty), and, where it names each part
    trainable or frozen (`parts`, the shape since research comparison Stage
    9b), exactly the freeze's parts frozen. A record that says otherwise
    contradicts itself or, its freeze decoded from rules 1 (`decoded`), the
    rules it records, which trained every part."""
    if transfer is None:
        return
    trainable = transfer.get("trainable") if isinstance(transfer, Mapping) else transfer
    parts = transfer.get("parts") if isinstance(transfer, Mapping) else None
    frozen = FROZEN_PARTS[frozen_parts]
    recorded_frozen = (sorted(part for part, state in parts.items() if state == "frozen")
                       if isinstance(parts, Mapping) else None)
    if (trainable == "all") == (not frozen) and recorded_frozen in (None, sorted(frozen)):
        return
    if (trainable == "all") == (not frozen):
        raise RunPlanRefused(
            f"{where} records a weight transfer with frozen parts {recorded_frozen} beside "
            f"frozen_parts {frozen_parts!r}, which freezes {sorted(frozen)}, so its record "
            "contradicts itself.")
    if decoded:
        raise RunPlanRefused(
            f"{where} records a weight transfer with trainable {trainable!r} under training "
            "rules 1, which trained every part ('all'), so its record contradicts its rules.")
    trains = "trains every part ('all')" if not frozen else "keeps some parts from training"
    raise RunPlanRefused(
        f"{where} records a weight transfer with trainable {trainable!r} beside frozen_parts "
        f"{frozen_parts!r}, which {trains}, so its record contradicts itself.")


def plan_branch(
    parent: Mapping[str, Any],
    parent_path: Path,
    request: Mapping[str, Any],
    explicit: Collection[str],
    *,
    device: str,
    record_version: int,
) -> BranchPlan:
    """Decide a branch of `parent` (contract §9.2, BACKLOG 23): a new run of
    the parent's stage and task, from its weights, with every setting of the
    parent's but those the request changes, each on the closed list of the
    parent's stage and task (`branch_settings`), and at least one.

    The parent's end state does not matter: a finished, early-stopped, paused
    or running run may be branched, since nothing of its progress continues.
    It must record the settings it trained with (a training-state checkpoint;
    a weights-only file records none), and an extension is not a branch.

    Raises:
        RunPlanRefused: with the reason and the remedy.
    """
    where = f"The parent {parent_path.name}"
    _check_record_version(parent, record_version, where)
    if parent.get("checkpoint_kind") != "training_state":
        raise RunPlanRefused(
            f"{where} holds weights only, without the settings a branch keeps and changes. "
            "Choose a training-state checkpoint of its run (last.pt or a top-k file).")
    if request.get("extend_epochs") or request.get("extend_lr") is not None:
        raise RunPlanRefused(
            "extend_epochs and extend_lr extend a finished run at a fixed rate with every "
            "setting kept, and a branch starts a new run: remove them to branch, or resume "
            "the parent (resume_from, without branch_from) to extend it.")
    changeable = branch_settings(parent.get("learning_stage"), parent.get("task"))
    settings, inherited, changes, decoded = _inherit_settings(parent, request, explicit, where,
                                                              changeable=changeable)
    if not changes:
        raise RunPlanRefused(
            f"A branch changes at least one setting of its parent ({', '.join(sorted(changeable))}), "
            "and this request changes none. To continue the parent with its settings, resume "
            "it (resume_from).")
    if "use_amp" in changes:
        # Mixed precision turned on or off is the change, recorded; on must
        # take effect here, or the record would claim what did not happen.
        if settings.get("use_amp") and not str(device).startswith("cuda"):
            raise RunPlanRefused(
                f"use_amp turned on would not take effect on {device}: mixed precision takes "
                "effect only on CUDA. Branch it on a CUDA device, or keep use_amp off.")
    else:
        # Kept on or off (amp_dtype changed or not): it must take effect as it
        # did for the parent, as a resume's must.
        _check_amp_takes_effect_alike(parent, settings, device, where)
    shown = "; ".join(f"{name} {change['parent']!r} -> {change['new']!r}"
                      for name, change in sorted(changes.items()))
    return BranchPlan(action=f"Branch {parent_path.name}: {shown}",
                      settings=settings, inherited=inherited, changes=changes, decoded=decoded)


def _inherit_settings(
    parent: Mapping[str, Any], request: Mapping[str, Any], explicit: Collection[str],
    where: str, *, changeable: Collection[str] = frozenset(),
) -> "tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, str]]":
    """The parent's learning settings, as the one decode reads them
    (`decode_learning_settings`), with the request's operational fields:
    `(settings, inherited, changes, decoded)`. A setting set explicitly to
    another value is a change when `changeable` names it (a branch), and
    refused otherwise. Each setting the run takes from the parent and the
    request does not set has one source: `inherited` when the parent's record
    holds it, `decoded` (with the decode's words) when the decode gave it.
    Then the rules across settings judge what the request changes
    (`training_fields.cross_field_problems` over `changes`: a branch's; a
    resume and an extension change nothing), in the training process and the
    job runner's preview alike."""
    parent_settings = decode_learning_settings(parent, where)
    recorded = parent_settings.settings
    decoded: Dict[str, str] = {}

    settings = dict(request)
    inherited: Dict[str, Any] = {}
    changes: Dict[str, Dict[str, Any]] = {}
    conflicts: List[str] = []
    missing: List[str] = []
    for name in request:
        if name in OPERATIONAL_FIELDS:
            continue
        if name not in recorded:
            missing.append(name)
            continue
        value = recorded[name]
        if name in explicit and not _same(request[name], value):
            if name in changeable:
                changes[name] = {"parent": value, "new": request[name]}
            else:
                conflicts.append(f"{name}: the parent's {value!r}, requested {request[name]!r}")
            continue
        settings[name] = value
        if name in explicit:
            continue
        if name in parent_settings.decoded:
            decoded[name] = parent_settings.decoded[name]
        else:
            inherited[name] = value
    if missing:
        raise RunPlanRefused(
            f"{where} does not record {', '.join(sorted(missing))}, so the run cannot "
            "inherit them.")
    # The freeze is never changed from a parent (owner's O4a): every
    # continuation and branch keeps it, so another freeze is a new finetune.
    freeze = (" The freeze (frozen_parts) is kept by every continuation and branch: to "
              "train with another, finetune the base again (finetune_from)."
              if any(c.startswith("frozen_parts:") for c in conflicts) else "")
    if conflicts and changeable:
        raise RunPlanRefused(
            f"A branch keeps every setting of its parent but those on its list "
            f"({', '.join(sorted(changeable))}); this request changes others: "
            + "; ".join(conflicts) + "." + freeze)
    if conflicts:
        raise RunPlanRefused(
            f"{where} was trained with other settings than this request sets explicitly: "
            + "; ".join(conflicts) + ". A continued run keeps every learning setting of "
            "its parent; remove these to inherit them, or branch the parent (branch_from) "
            "to change settings on a model bundle's list." + freeze)
    # The rules across settings (decision S3), judged on the settings the run
    # trains with for what the request changes (a branch's changes): a
    # setting changed that the run would not read is refused, never ignored.
    # A setting kept from the parent, inherited or stated at the parent's
    # value, is never refused by them, so a resume and an extension, which
    # change nothing, never meet them.
    problems = cross_field_problems(settings, set(changes))
    if problems:
        raise RunPlanRefused(" ".join(problems))
    return settings, inherited, changes, decoded


def _check_amp_takes_effect_alike(
    parent: Mapping[str, Any], settings: Mapping[str, Any], device: str, where: str,
) -> None:
    """Mixed precision changes how a run learns, and takes effect only on CUDA:
    a device change that turns it on or off is a settings change."""
    amp = ((parent.get("config") or {}).get("effective") or {}).get("amp") or {}
    recorded = amp.get("enabled")
    if recorded is None:
        return
    here = bool(settings.get("use_amp")) and str(device).startswith("cuda")
    if bool(recorded) != here:
        raise RunPlanRefused(
            f"{where} trained with mixed precision {'on' if recorded else 'off'} "
            f"({amp.get('dtype')}); on {device} it would be {'on' if here else 'off'}. "
            "Continue it on a device where mixed precision takes effect as it did.")


def _rates_text(rates: List[float]) -> str:
    """The extension's rate in words: one rate when every parameter group
    holds the same ("a fixed rate of 1e-05"), each group's when they differ
    (M5-3: two groups at one rate read "a fixed rate of 1e-05, 1e-05")."""
    shown = [f"{rate:.3g}" for rate in rates]
    if len(set(shown)) == 1:
        return f"a fixed rate of {shown[0]}"
    return f"fixed rates of {', '.join(shown)} (one per parameter group)"


def _group_rates(parent: Mapping[str, Any]) -> List[float]:
    """Each parameter group's rate as the parent's optimizer holds it."""
    groups = (parent.get("optimizer_state_dict") or {}).get("param_groups") or []
    return [float(group["lr"]) for group in groups]


def zero_rate_groups(parent: Mapping[str, Any]) -> List[int]:
    """The parameter groups (their indices) whose rate ended at 0 in the
    parent's optimizer: **the groups `extend_lr` replaces, and only those**
    (`_fixed_rates`). An extension of a parent with one refuses without
    `extend_lr`, so the job runner's preview states them as a fact of its one
    read of the parent, for a page to offer the rate exactly when the plan
    needs it."""
    return [i for i, rate in enumerate(_group_rates(parent)) if rate == 0.0]


def _fixed_rates(
    parent: Mapping[str, Any], extend_lr: Optional[float], where: str,
) -> Dict[str, Any]:
    """An extension's schedule (contract M3a-4): each parameter group's rate as
    the parent's optimizer holds it, held fixed. No warmup, no recomputed curve,
    no cosine past its end, no OneCycle stepped past its total.

    **A zero rate is never replaced by a guess.** A schedule that decayed to 0
    would extend without training; `extend_lr` replaces a zero rate, and only a
    zero rate, and the groups it applied to are recorded.
    """
    rates = _group_rates(parent)
    if not rates:
        raise RunPlanRefused(f"{where} records no optimizer rates to extend at.")
    zero = zero_rate_groups(parent)
    if extend_lr is not None:
        if not zero:
            raise RunPlanRefused(
                f"extend_lr replaces a rate that ended at 0, and none did (the rates are "
                f"{rates}). Extend without it to continue at those rates.")
        if not float(extend_lr) > 0:
            raise RunPlanRefused(f"extend_lr must be above 0 (got {extend_lr}).")
        rates = [float(extend_lr) if i in zero else rate for i, rate in enumerate(rates)]
    elif zero:
        raise RunPlanRefused(
            f"{where} ended with a rate of 0 in parameter group(s) {zero}; extending at 0 "
            "would not train them. Give extend_lr for those groups; no rate is guessed.")
    return {"type": "constant", "lrs": rates,
            "zero_rate_groups_set": zero if extend_lr is not None else [],
            "extend_lr": float(extend_lr) if extend_lr is not None else None}


def _locate_best_weights(
    parent: Mapping[str, Any], parent_path: Path, where: str,
    parent_sha256: Optional[str] = None,
) -> "tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]":
    """Where early stopping's best weights are, and the parent's best-k
    entries, each resolved to a path.

    **A parent that is itself the best epoch's file is carried as a best-k
    entry** (contract M3a review of `094b90d`, P2-1). `ModelCheckpoint` writes a
    best-k file before listing it, so that file records `best_weights_source`
    `{"kind": "self"}` and a best-k list without itself. Continued from it, the
    child restores the parent's own weights; unless the parent is also listed,
    the child's checkpoints cannot name where those weights are, and a
    grandchild would lose them. The entry names the parent by this read's
    digest, so the next generation checks it like any other best-k file.

    **Best weights the record says must be restored, and cannot be located,
    are refused**, never treated as absent.
    """
    callbacks = parent.get("callbacks") or {}
    checkpoint_dir = parent_path.parent
    model_checkpoint = callbacks.get("model_checkpoint") or {}
    best_k = []
    for entry in (model_checkpoint.get("best_k") or []):
        best_k.append({**entry, "path": resolve_recorded_file(entry["file"], checkpoint_dir)})

    early = callbacks.get("early_stopping") or {}
    restores = bool(early.get("restore_best_weights")) and early.get("best_value") is not None
    source = callbacks.get("best_weights_source")
    if source is None:
        if restores:
            raise RunPlanRefused(
                f"{where} records early stopping's best {early.get('monitor')} "
                f"({early.get('best_value')}, epoch {early.get('best_epoch')}) to be restored "
                "at the end of training, and names no file holding those weights. Continuing "
                "would lose them; choose another checkpoint.")
        return None, best_k
    if source.get("kind") == "self":
        best_epoch = int(early.get("best_epoch", 0))
        if model_checkpoint and not any(int(e["epoch"]) == best_epoch for e in best_k):
            if not parent_sha256:
                raise RunPlanRefused(
                    f"{where} holds early stopping's best weights itself and cannot be named "
                    "for the run's checkpoints without the digest of its read.")
            monitor = model_checkpoint.get("monitor")
            value = (parent.get("logs") or {}).get(monitor)
            if value is None and monitor == early.get("monitor"):
                value = early.get("best_value")
            best_k.append({"file": parent_path.name, "value": value, "epoch": best_epoch,
                           "sha256": parent_sha256, "path": parent_path})
        return {"kind": "self"}, best_k
    path = resolve_recorded_file(source["file"], checkpoint_dir)
    if not source.get("sha256"):
        # The digest is what the file is read against (`train_model.plan_from_parent`).
        raise RunPlanRefused(
            f"{where} names {source['file']} as early stopping's best weights without the "
            "SHA-256 it was written with, so the file cannot be checked; choose another "
            "checkpoint.")
    if not path.is_file():
        raise RunPlanRefused(
            f"{where} names {source['file']} as early stopping's best weights, and it is "
            f"not at {path}. Without it the run could not restore its best weights at the "
            "end; put the file back (its SHA-256 is "
            f"{source.get('sha256')}) or choose another checkpoint.")
    return {"kind": "file", "path": path, "sha256": source.get("sha256")}, best_k


__all__ = [
    "BranchPlan",
    "CONTINUED_RULES",
    "ContinuationPlan",
    "DecodedSettings",
    "OPERATIONAL_FIELDS",
    "RUN_SETTINGS",
    "RunPlanRefused",
    "decode_learning_settings",
    "decoded_sentence",
    "finetune_split_seed_problem",
    "new_run_problem",
    "new_run_settings",
    "new_run_values",
    "plan_branch",
    "plan_continuation",
    "recorded_training_settings",
    "resolve_recorded_file",
    "unread_problem",
    "zero_rate_groups",
]
