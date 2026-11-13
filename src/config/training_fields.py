"""
Training field-spec — single declarative source for the training-start parameter set.
=====================================================================================
B-lite (config-authority decision, docs/CONFIG_AUTHORITY.md). This module is the one
declarative description of the ~30 user-facing training-start parameters, read by three
surfaces:

    - the API request model   (src/api/routes/training.py :: TrainingStartRequest: a base
                                start, or a continuation of any run, taking every learning
                                setting, `run_plan.RUN_SETTINGS`, and the task; a new base
                                only those a base reads, `run_plan.new_run_settings("base")`)
    - the WebUI form           (src/webui/components/training_console.py: the base form of
                                what a new base takes, `run_plan.new_run_settings("base")`,
                                see below)
    - the CLI + projection hub (scripts/train_model.py :: TrainConfig -> downstream configs)

It is read, not only descriptive: a new run's settings are judged by one check
(`src.utils.run_plan.new_run_problem`: every value by `value_problem`, every rule across
settings by `cross_field_problems`), which the training process (`train_model.train`) and the
job runner's starts make alike, and a branch's plan judges what it changes by the same rules;
the API's start checks each value by `value_problem` too and takes its bounds from each
field's ``valid``, and the WebUI's base form and Fine-tune page check values with
`value_problem` before the runner checks them again.

Constraint representation
-------------------------
All bounds are structured dicts using the Pydantic/annotated-types vocabulary
``{"ge"|"gt"|"le"|"lt": value}`` so inclusive and exclusive bounds are distinguishable:

    valid       — the hard validity (PyTorch/compute-lib norms): the one value check of a
                  model bundle start (`value_problem`), and the API request's bounds.
    current_api — the API's constraints, compared mechanically against the live Pydantic model.

``valid`` and ``current_api`` share one shape on purpose, and cross-field rules live in
``CROSS_FIELD_RULES`` as data rather than in prose, judged by one evaluator
(`cross_field_problems`), so every surface consumes the spec instead of re-deriving policy.

What the parity tests mechanically pin:

    - API: field existence, defaults, name coverage, and the exact ge/gt/le/lt constraints.
    - Projection: every ``projects_to`` target attribute exists on the real config class.
    - Self-validation: constraint dicts are well-formed; cross-field rules reference real fields
      and real choices.

Phase status:
    - PR-4b (done): ``temperature`` / ``label_smoothing`` / ``margin`` were wired through
      ``TrainConfig`` into the ShepherdGNN path's loss, which had silently dropped them. They
      retired with that path in K8-7, with its architecture, loader and loss-weight fields: a
      model bundle's architecture is its model spec, its loss the stage's. No field is
      ``effective=False``.

    - PR-4c (done): the API now enforces the target policy — closed enums for
      ``scheduler_type`` / ``amp_dtype``, the PyTorch device grammar for ``device`` (so ``cuda:N``
      stays available on multi-GPU hosts), ``min_lr_ratio <= 1.0`` and the ``num_neighbors``
      non-empty/element policy (its ``conv_type`` enum and the conv-type-conditional
      divisibility rule retired with the ShepherdGNN fields in K8-7). The CLI
      ``--device`` follows the same grammar. The WebUI form of that time kept its conservative
      ranges; its only change was replacing the silent ``num_neighbors`` fallback with a
      validation error.
      The surfaces then hand-wrote their constraints, guarded by the parity tests.

    - The settings projection (ops-wiring round A1a, decisions S1-S6): the API's base start
      takes every setting a new base run reads (`run_plan.new_run_settings`), its bounds from
      each field's ``valid`` (derived, for the schema; the parity tests still compare them),
      each value checked first by `value_problem`, strictly and in its words (so a ``-1`` hop
      is legal, as on the CLI, and a string is never a number); the rules across settings
      that say when a setting is read are data here, and an explicit setting the run would
      not read is refused, never ignored.

    - Checkpoint A's verification (AF1): the start takes every learning setting and the task,
      so a continuation of either stage states its own settings over HTTP as on the CLI; the
      runner judges them against the parent's stage and task (`run_plan.plan_continuation`),
      and a new base still refuses what a base does not read in the one check's sentence.

**The WebUI's forms.** The ShepherdGNN form and its collector (``_collect_config``) retired in
K6. The Training page's base form offers what a new base takes
(`run_plan.new_run_settings("base")`: every setting, a setting a rule across settings reads only
under its guard shown only while it applies, `not_read_by_rules`; and the reference spec's
fields), each starting at the value the run takes without it; the Fine-tune page offers Epochs,
Seed and Graph backbone (the freeze, ``frozen_parts``; ops-wiring round, decision F1). Both send
a setting when the operator touched its field, whatever its value (decision S2), with
``learning_stage="base"`` on the base form. Their widgets follow each field's ``kind`` and
``choices`` (`training_console.setting_field`), their help leads are the fields'
``description``, and their values are checked by `value_problem`. The spec records no form's
widgets or ranges (removed in K7); a divergence note names the WebUI only where a surface in use
today differs.

Dependency-light (no torch / pydantic / gradio): any surface can read it and torch-free unit tests
can exercise it.

Module: src/config/training_fields.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple

SCOPES = ("model", "dataloader", "trainer", "loss", "path", "runtime_setting")
KINDS = ("int", "float", "bool", "str", "list[int]", "dict")
BOUND_KEYS = ("ge", "gt", "le", "lt")
# scopes whose effective fields are NOT required to project into TrainConfig:
_PROJECTION_EXEMPT_SCOPES = ("runtime_setting",)


#: The kinds of rule across settings (`CrossFieldRule.kind`).
RULE_KINDS = ("min_exclusive", "read_only_when")


@dataclass(frozen=True)
class CrossFieldRule:
    """A constraint spanning more than one field, represented structurally.

    ``when`` is the applicability guard: ``{"scheduler_type": ("onecycle",)}`` means the
    rule's condition is that ``scheduler_type`` holds one of those values. Two kinds:

    - ``min_exclusive``: while ``when`` holds, ``subject`` must be above ``threshold``;
    - ``read_only_when``: a run reads ``subject`` only while ``when`` holds, so a
      ``subject`` set explicitly while it does not is refused (it would not take effect;
      the precedent of `recipe.unread_settings`: refused, never ignored).
    """

    name: str
    kind: str                                            # one of RULE_KINDS
    subject: str                                         # field the rule constrains
    threshold: Optional[float] = None                    # bound value (min_exclusive)
    when: Optional[Dict[str, Tuple[Any, ...]]] = None    # applicability guard
    enforced_in: str = ""                                # where it is enforced
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in RULE_KINDS:
            raise ValueError(f"{self.name}: unknown rule kind {self.kind!r}")
        if not self.when:
            raise ValueError(f"{self.name}: a rule across settings names its guard (when)")


#: Where every rule is judged: the one evaluator's callers.
_ENFORCED_IN = ("current (`cross_field_problems`: through `src.utils.run_plan.new_run_problem`, "
                "the training process's `train_model.train` for a new run and the job "
                "runner's starts before they spawn, and a branch's plan on what it changes)")

CROSS_FIELD_RULES: Tuple[CrossFieldRule, ...] = (
    CrossFieldRule(
        name="onecycle_requires_positive_min_lr_ratio",
        kind="min_exclusive",
        subject="min_lr_ratio",
        threshold=0.0,
        when={"scheduler_type": ("onecycle",)},
        enforced_in=_ENFORCED_IN + "; and the Trainer's `_validate_min_lr_ratio`, by this "
                                   "rule's own sentence (`rule_problem`)",
        description=(
            "onecycle computes final_div_factor = 1 / min_lr_ratio, so it needs a strictly "
            "positive value; cosine and linear allow 0 (decay to zero)."
        ),
    ),
    # Read only when (ops-wiring round, decision S3): the Trainer's schedule
    # and precision read these settings only under their guard
    # (`Trainer._create_scheduler`, `Trainer._setup_amp`).
    CrossFieldRule(
        name="warmup_steps_read_only_by_cosine",
        kind="read_only_when",
        subject="warmup_steps",
        when={"scheduler_type": ("cosine",)},
        enforced_in=_ENFORCED_IN,
        description=(
            "The cosine schedule's linear warmup reads it (0 warms up over the Trainer's "
            "warmup_ratio, a tenth of the steps); onecycle warms up over that ratio, and "
            "linear and none have no warmup."
        ),
    ),
    CrossFieldRule(
        name="min_lr_ratio_read_only_by_a_scheduler",
        kind="read_only_when",
        subject="min_lr_ratio",
        when={"scheduler_type": ("cosine", "onecycle", "linear")},
        enforced_in=_ENFORCED_IN,
        description="It is a schedule's final rate as a fraction of the peak; none keeps one "
                    "rate.",
    ),
    CrossFieldRule(
        name="amp_dtype_read_only_with_amp",
        kind="read_only_when",
        subject="amp_dtype",
        when={"use_amp": (True,)},
        enforced_in=_ENFORCED_IN,
        description="It is mixed precision's compute dtype; with use_amp off the run computes "
                    "in float32.",
    ),
)


def _rule_value(value: Any) -> str:
    """A value as a rule's sentence names it: a switch as true or false (the
    request's and the configuration file's words), anything else as written."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)


def _rule_guard_words(rule: CrossFieldRule) -> str:
    """A rule's guard in words: "scheduler_type is 'cosine'", "scheduler_type
    is 'cosine', 'onecycle' or 'linear'"."""
    parts = []
    for name, allowed in (rule.when or {}).items():
        shown = [_rule_value(value) for value in allowed]
        listed = shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} or {shown[-1]}"
        parts.append(f"{name} is {listed}")
    return " and ".join(parts)


def _rule_by_name(name: str) -> CrossFieldRule:
    for rule in CROSS_FIELD_RULES:
        if rule.name == name:
            return rule
    raise KeyError(name)


def _guard_holds(rule: CrossFieldRule, values: Mapping[str, Any]) -> bool:
    """Whether `rule`'s guard (``when``) holds on `values`, which hold every
    field of it."""
    return all(values[name] in allowed for name, allowed in (rule.when or {}).items())


def not_read_by_rules(values: Mapping[str, Any]) -> frozenset:
    """**The settings a run would not read, by the rules across settings**:
    the subject of each ``read_only_when`` rule (`CROSS_FIELD_RULES`) whose
    guard does not hold on `values`, a run's settings as it would train with
    them (warmup_steps unless the scheduler is cosine). A rule whose guard
    `values` does not hold every field of is not judged. The same guard
    `rule_problem` refuses an explicit setting by (`cross_field_problems`), so
    what a form leaves out is what a start would refuse: the Training page's
    base form neither offers nor sends a setting its run would not read
    (decision S3)."""
    return frozenset(rule.subject for rule in CROSS_FIELD_RULES
                     if rule.kind == "read_only_when"
                     and all(name in values for name in (rule.when or {}))
                     and not _guard_holds(rule, values))


def rule_problem(rule: Any, values: Mapping[str, Any]) -> Optional[str]:
    """Why `values`, a run's settings as it would train with them, break the
    rule `rule` (a `CrossFieldRule` or its name), in the rule's one sentence,
    or None. A rule whose fields `values` does not hold is not judged. Whether
    the rule applies to a run (what the operator set) is
    `cross_field_problems`'s; the Trainer asks a rule of its own settings
    here directly, so the rule has one sentence wherever it is refused."""
    rule = _rule_by_name(rule) if isinstance(rule, str) else rule
    fields = (rule.subject, *(rule.when or {}))
    if any(name not in values for name in fields):
        return None
    holds = _guard_holds(rule, values)
    subject = values[rule.subject]
    if rule.kind == "min_exclusive":
        if not holds or subject is None or isinstance(subject, bool) \
                or not isinstance(subject, (int, float)) or subject > rule.threshold:
            return None
        return (f"{rule.subject} must be more than {rule.threshold:g} when "
                f"{_rule_guard_words(rule)}, not {_rule_value(subject)}: {rule.description}")
    if holds:
        return None
    guards = ", ".join(f"{name} is {_rule_value(values[name])}" for name in (rule.when or {}))
    return (f"{rule.subject} is read only when {_rule_guard_words(rule)}, and this run's "
            f"{guards}, so it would not take effect. {rule.description} Remove "
            f"{rule.subject}, or change {' and '.join(rule.when or {})}.")


def cross_field_problems(values: Mapping[str, Any], explicit: Collection[str]) -> List[str]:
    """**The one evaluator of the rules across settings** (`CROSS_FIELD_RULES`):
    every problem of a run whose settings are `values` (as it would train with
    them: the operator's, the stage's reference or this build's default, or a
    parent's) and whose operator set `explicit` (for a branch, what it
    changes), each in its rule's sentence (`rule_problem`). A rule is judged
    only where the operator set one of its fields: a ``read_only_when`` rule
    where the operator set its subject (an explicit setting the run would not
    read is refused, never ignored), a ``min_exclusive`` rule where the
    operator set its subject or a field of its guard. **A value the operator
    did not set is never refused here**: a filled one, or one kept from a
    parent (inherited, or stated at the parent's value), so a resume and an
    extension, which change nothing, never meet these rules. Requested mixed
    precision on a device without it is no problem of these rules: the run
    records the precision requested and the precision in effect apart.

    Called through the one check of a new run's settings
    (`run_plan.new_run_problem`: the training process's `train_model.train`
    and the job runner's starts before they spawn) and by the branch plan
    (`run_plan.plan_branch`, on what it changes), so the CLI, every spawned job
    and the service give one judgement."""
    chosen = set(explicit)
    problems = []
    for rule in CROSS_FIELD_RULES:
        judged = ({rule.subject} if rule.kind == "read_only_when"
                  else {rule.subject, *(rule.when or {})})
        if not judged & chosen:
            continue
        problem = rule_problem(rule, values)
        if problem:
            problems.append(problem)
    return problems


@dataclass(frozen=True)
class FieldSpec:
    """One training-start parameter, described once.

    ``valid`` / ``choices`` / ``valid_pattern`` / ``item_valid`` / ``min_length`` are the TARGET
    canonical policy (enforced in PR-4c). ``current_api`` is the STRUCTURED constraint the API
    enforces TODAY (``{}`` = accepted but unconstrained, e.g. a free string; ``None`` = not an API
    field). ``known_divergence`` / ``divergence_note`` record where the surfaces (CLI, API, the
    WebUI's base form, Fine-tune page and Continue) deliberately differ.
    """

    name: str
    kind: str
    default: Any
    scope: str
    projects_to: Optional[str] = None

    # target policy:
    valid: Optional[Dict[str, Any]] = None
    valid_pattern: Optional[str] = None
    choices: Optional[Tuple[Any, ...]] = None
    item_valid: Optional[Dict[str, Any]] = None   # list[int]: per-element constraint
    min_length: Optional[int] = None              # list[int]: non-empty etc.

    # current API behavior pinned by PR-4a (numeric), extended by PR-4c (enum/pattern). A
    # list's items and length are the policy's (`item_valid`, `min_length`): the API checks
    # each value by `value_problem`, so it has no list constraint of its own.
    current_api: Optional[Dict[str, Any]] = None
    current_api_choices: Optional[Tuple[Any, ...]] = None   # enum the API rejects outside of
    current_api_pattern: Optional[str] = None               # regex the API enforces

    known_divergence: bool = False
    divergence_note: str = ""

    effective: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"{self.name}: unknown scope {self.scope!r}")
        for label in ("valid", "current_api", "item_valid"):
            d = getattr(self, label)
            if d is None:
                continue
            if not isinstance(d, dict):
                raise TypeError(f"{self.name}: {label} must be a dict or None")
            bad = set(d) - set(BOUND_KEYS)
            if bad:
                raise ValueError(f"{self.name}: {label} has unknown bound keys {sorted(bad)}")
            if "ge" in d and "gt" in d:
                raise ValueError(f"{self.name}: {label} cannot set both ge and gt")
            if "le" in d and "lt" in d:
                raise ValueError(f"{self.name}: {label} cannot set both le and lt")


# =============================================================================================
# The spec. current_api is what src/api/routes/training.py::TrainingStartRequest enforces: the run
# keys, the continuation keys, the task and every learning setting (`src.utils.run_plan.
# RUN_SETTINGS`; a new base is refused those it does not read), each setting unset by default (the
# recipe's value, this build's default or a parent's, never the request's), its bounds the
# field's `valid`.
# scope/projects_to from the scripts/train_model.py projection: into TrainConfig, or on into the
# TrainerConfig it builds.
# =============================================================================================
FIELDS: Tuple[FieldSpec, ...] = (
    # ---- paths (effective -> they DO project into TrainConfig) ------------------------------
    FieldSpec("data_dir", "str", "data/workspaces/default", "path", projects_to="TrainConfig.data_dir",
              current_api={},
              description="Workspace directory."),
    # Contract M3a-2: every run writes into a directory of its own under the workspace. The
    # free output, checkpoint and log locations gave way to an optional run name; an explicit
    # directory (absent or empty) stays a CLI option.
    FieldSpec("run_name", "str", "", "path", projects_to="TrainConfig.run_name",
              current_api={},
              description="Optional run name, part of the allocated run directory's name."),
    FieldSpec("run_dir", "str", "", "path", projects_to="TrainConfig.run_dir",
              current_api=None,
              known_divergence=True,
              divergence_note="CLI only (--run-dir). The API and the WebUI name a run and have "
                              "its directory allocated.",
              description="Explicit run directory; must be absent or empty."),
    FieldSpec("resume_from", "str", None, "path", projects_to="TrainConfig.resume_from",
              current_api={},
              known_divergence=True,
              divergence_note="WebUI resume uses a checkpoint dropdown, not this field.",
              description="Checkpoint path to resume from."),

    # ---- the model bundle path (docs/working/scorer-retraining/README.md §3-§4) -------------
    # Each field's divergence note says which surface takes it: the CLI, the WebUI (K6), and
    # the API's two starts, a base or a continuation (POST /training/start) and the Fine-tune
    # children.
    FieldSpec("learning_stage", "str", None, "trainer", projects_to="TrainConfig.learning_stage",
              choices=("base", "finetune"), current_api={},
              current_api_choices=("base", "finetune"),
              known_divergence=True,
              divergence_note="CLI (--stage); the WebUI's Training base form sends 'base'. POST "
                              "/training/start requires 'base' for a new run and takes either "
                              "stage, or none, with resume_from (compared with the parent's "
                              "when given, so a finetune child continues over HTTP); the "
                              "Fine-tune page and POST /training/finetune start finetunes.",
              description="Model bundle stage: base (KG link prediction) or finetune; "
                          "required, a run without one is refused before anything is read."),
    FieldSpec("model_spec", "dict", None, "model", projects_to="TrainConfig.model_spec",
              current_api={},
              known_divergence=True,
              divergence_note="Configuration file, POST /training/start and a Fine-tune start "
                              "of one task (its graph side the base's); unset is the stage's "
                              "reference spec (a finetune's: its base's graph side). The "
                              "WebUI's Training base form sends it whole when a field of its "
                              "Model accordion was touched.",
              description="The model spec as plain data, read strictly."),
    FieldSpec("split_seed", "int", 0, "dataloader", projects_to="TrainConfig.split_seed",
              valid={"ge": 0}, current_api={"ge": 0},
              known_divergence=True,
              divergence_note="CLI (--split-seed), POST /training/start and the WebUI's "
                              "Training base form; a finetune takes its base's, and one stated "
                              "otherwise is refused before any child starts.",
              description=("The seed of the graph's edge split: runs on one split validate on the "
                           "same pairs.")),
    FieldSpec("task", "str", None, "trainer", projects_to="TrainConfig.task",
              choices=("disease", "gene", "patients_like_me"), current_api={},
              current_api_choices=("disease", "gene", "patients_like_me"),
              known_divergence=True,
              divergence_note="CLI (--task), POST /training/finetune and the WebUI's "
                              "Fine-tune page (its Tasks); POST /training/start with "
                              "resume_from, compared with the parent's (a new base is refused "
                              "one).",
              description="A finetune's task."),
    FieldSpec("finetune_from", "str", None, "path", projects_to="TrainConfig.finetune_from",
              current_api=None,
              known_divergence=True,
              divergence_note="CLI (--finetune-from), POST /training/finetune and the WebUI's "
                              "Fine-tune page (a listed base).",
              description="The base checkpoint a finetune starts from."),
    FieldSpec("branch_from", "str", None, "path", projects_to="TrainConfig.branch_from",
              current_api=None,
              known_divergence=True,
              divergence_note="CLI (--branch-from), the job runner's start (start_training "
                              "with branch_from) and POST /training/branch. Not a WebUI "
                              "field yet: the Training page's Branch mode is the ops-wiring "
                              "round's stage B1.",
              description="The checkpoint a branch starts from: its stage, task, weights "
                          "and settings, none or some changed on the stage's list."),
    FieldSpec("task_loss_weight", "float", None, "loss", projects_to="TrainConfig.task_loss_weight",
              valid={"ge": 0}, current_api={"ge": 0},
              known_divergence=True,
              divergence_note="Unset is the task's reference (README C3). POST /training/start takes it for a continuation, judged against the parent; a new base is refused it.",
              description="Weight of the task loss in a finetune."),
    FieldSpec("lp_loss_weight", "float", None, "loss", projects_to="TrainConfig.lp_loss_weight",
              valid={"ge": 0}, current_api={"ge": 0},
              known_divergence=True,
              divergence_note="Unset is the task's reference (README C3). POST /training/start takes it for a continuation, judged against the parent; a new base is refused it.",
              description="Weight of the LP loss in a finetune."),
    FieldSpec("candidate_sample_size", "int", None, "dataloader",
              projects_to="TrainConfig.candidate_sample_size", valid={"ge": 1},
              current_api={"ge": 1},
              known_divergence=True,
              divergence_note="Unset is the task's reference (U9). POST /training/start takes it for a continuation, judged against the parent; a new base is refused it.",
              description="Random candidate diseases per finetune batch, plus the truths."),
    FieldSpec("sparse_sample_size", "int", None, "dataloader",
              projects_to="TrainConfig.sparse_sample_size", valid={"ge": 0},
              current_api={"ge": 0},
              known_divergence=True,
              divergence_note="Unset is the task's reference (U9). POST /training/start takes it for a continuation, judged against the parent; a new base is refused it.",
              description="Random gene, phenotype and disease seeds per finetune batch."),
    FieldSpec("candidate_upsample_size", "int", None, "dataloader",
              projects_to="TrainConfig.candidate_upsample_size", valid={"ge": 0},
              current_api={"ge": 0},
              known_divergence=True,
              divergence_note="Unset is the task's reference (U14). POST /training/start takes it for a continuation, judged against the parent; a new base is refused it.",
              description="Rarely seen genes added to each gene-task patient's list in training."),
    # The freeze setting (research comparison Stage 9, owner's O4a): which parts
    # of a finetune train. Its parts and the stage rule are `src.config.recipe`'s
    # (`FROZEN_PARTS`, `frozen_parts_problem`).
    FieldSpec("frozen_parts", "str", "none", "trainer", projects_to="TrainConfig.frozen_parts",
              choices=("none", "graph_side"), current_api={},
              current_api_choices=("none", "graph_side"),
              known_divergence=True,
              divergence_note="A finetune setting (the recipe's reference 'none'): the CLI "
                              "(--frozen-parts), POST /training/finetune's settings and the "
                              "WebUI's Fine-tune page (Graph backbone: Trained or Frozen); a "
                              "base with any freeze is refused. POST /training/start takes it "
                              "for a continuation, which keeps its parent's freeze; a new base "
                              "is refused it.",
              description=("Whether the base's input, encoder and LP head train with the task or "
                           "stay fixed.")),

    # ---- sampling (the stage samplers, from TrainConfig) ------------------------------------
    FieldSpec("batch_size", "int", 32, "dataloader", projects_to="TrainConfig.batch_size",
              valid={"ge": 1, "le": 2048},
              current_api={"ge": 1, "le": 2048},
              description="Training examples per step: larger is faster and needs more memory."),
    FieldSpec("num_neighbors", "list[int]", (15, 10, 5), "dataloader",
              projects_to="TrainConfig.num_neighbors",
              item_valid={"ge": 1}, min_length=1,
              current_api={},
              known_divergence=True,
              divergence_note="One number per hop the model spec's encoder declares "
                              "(`hops_problem`; GATv2: one per layer), checked by the base "
                              "start, the Fine-tune start (against its base's encoder) and by "
                              "the run. The API checks the list by `value_problem`, so -1, "
                              "every neighbour, is legal there as on the CLI.",
              description=("Neighbours sampled at each hop, one number per encoder layer; -1 "
                           "takes all.")),
    # ---- trainer (TrainerConfig) ------------------------------------------------------------
    FieldSpec("num_epochs", "int", 100, "trainer", projects_to="TrainerConfig.num_epochs",
              valid={"ge": 1, "le": 10000},
              current_api={"ge": 1, "le": 10000},
              description="How many passes over the training data the run makes."),
    FieldSpec("learning_rate", "float", 1e-4, "trainer", projects_to="TrainerConfig.learning_rate",
              valid={"gt": 0, "le": 1.0},
              current_api={"gt": 0, "le": 1.0},
              description="The optimizer's step size; with the scheduler none it stays constant."),
    FieldSpec("weight_decay", "float", 0.01, "trainer", projects_to="TrainerConfig.weight_decay",
              valid={"ge": 0.0},
              current_api={"ge": 0.0},
              description="The optimizer's penalty on large weights; 0 is none."),
    FieldSpec("scheduler_type", "str", "cosine", "trainer", projects_to="TrainerConfig.scheduler_type",
              choices=("cosine", "onecycle", "linear", "none"),
              current_api={}, current_api_choices=("cosine", "onecycle", "linear", "none"),
              description="How the learning rate changes over the run; none keeps it constant."),
    FieldSpec("warmup_steps", "int", 500, "trainer", projects_to="TrainerConfig.warmup_steps",
              valid={"ge": 0},
              current_api={"ge": 0},
              description=("Cosine's warmup steps, read only with cosine; 0 warms up over a tenth "
                           "of the steps.")),
    FieldSpec("min_lr_ratio", "float", 0.01, "trainer", projects_to="TrainerConfig.min_lr_ratio",
              valid={"ge": 0.0, "le": 1.0},
              current_api={"ge": 0.0, "le": 1.0},
              description=("A schedule's final rate as a fraction of its peak; 0 decays to zero "
                           "(not onecycle).")),
    FieldSpec("gradient_accumulation_steps", "int", 1, "trainer",
              projects_to="TrainerConfig.gradient_accumulation_steps",
              valid={"ge": 1},
              current_api={"ge": 1},
              description="Batches accumulated per optimizer step; 1 steps after every batch."),
    FieldSpec("max_grad_norm", "float", 1.0, "trainer", projects_to="TrainerConfig.max_grad_norm",
              valid={"gt": 0},
              current_api={"gt": 0.0},
              description="The largest gradient norm a step takes; a larger one is scaled down."),
    FieldSpec("use_amp", "bool", True, "trainer", projects_to="TrainerConfig.use_amp",
              current_api={},
              description="16-bit arithmetic on CUDA to save memory and time; off keeps float32."),
    FieldSpec("amp_dtype", "str", "float16", "trainer", projects_to="TrainerConfig.amp_dtype",
              choices=("float16", "bfloat16"),
              current_api={}, current_api_choices=("float16", "bfloat16"),
              description="Mixed precision's 16-bit type, read only with mixed precision on."),
    FieldSpec("eval_every_n_epochs", "int", 1, "trainer", projects_to="TrainerConfig.eval_every_n_epochs",
              valid={"ge": 1}, current_api={"ge": 1},
              description="How often the Trainer validates, in epochs."),
    FieldSpec("early_stopping_patience", "int", 10, "trainer", projects_to="TrainerConfig.early_stopping_patience",
              valid={"ge": 1},
              current_api={"ge": 1},
              description=("Epochs with no better validation figure before the run stops; none "
                           "is no early stopping.")),
    FieldSpec("save_top_k", "int", 3, "trainer", projects_to="TrainerConfig.save_top_k",
              valid={"ge": 1}, current_api={"ge": 1},
              description="How many of the run's best saves, by its selection metric, it keeps."),
    FieldSpec("device", "str", "auto", "trainer", projects_to="TrainerConfig.device",
              valid_pattern=r"^(auto|cpu|mps|cuda(:\d+)?)$",
              current_api={}, current_api_pattern=r"^(auto|cpu|mps|cuda(:\d+)?)$",
              known_divergence=True,
              divergence_note="Deliberately asymmetric: API (and CLI --device) accept the PyTorch device "
                              "grammar incl. cuda:N for multi-GPU hosts; the WebUI stays conservative with "
                              "{auto,cuda,cpu}.",
              description="Compute device."),
    FieldSpec("seed", "int", 42, "trainer", projects_to="TrainerConfig.seed",
              valid={"ge": 0, "le": 2**32 - 1},
              current_api={"ge": 0, "le": 2**32 - 1},
              description=("The random seed of the weights' start and the sampling (the split has "
                           "its own).")),

    # ---- continuing from a parent (contract M3a-3, M3a-4) -----------------------------------
    # The parent is `resume_from`; what happens is decided by its record (src/utils/run_plan).
    FieldSpec("extend_epochs", "int", None, "trainer", projects_to="TrainConfig.extend_epochs",
              valid={"ge": 1}, current_api={"ge": 1},
              known_divergence=True,
              divergence_note="CLI (--extend-epochs), POST /training/start with resume_from, "
                              "and the console's continue block, shown only for a finished "
                              "parent; not a training-form field.",
              description="Extend a finished parent by this many epochs at a fixed rate."),
    FieldSpec("extend_lr", "float", None, "trainer", projects_to="TrainConfig.extend_lr",
              valid={"gt": 0}, current_api={"gt": 0},
              known_divergence=True,
              divergence_note="CLI (--extend-lr) and POST /training/start with resume_from: "
                              "replaces a zero rate, and only a zero rate (the preview's "
                              "zero_rate_groups names them). Not a WebUI field yet.",
              description="Rate for parameter groups whose rate ended at 0, when extending."),
    FieldSpec("pause_after_epoch", "int", None, "runtime_setting",
              projects_to="TrainConfig.pause_after_epoch",
              valid={"ge": 1}, current_api=None,
              known_divergence=True,
              divergence_note="CLI and probe only (--pause-after-epoch); no WebUI or HTTP control.",
              description="Stop with a resumable checkpoint after this many epochs of progress."),

    # ---- runtime setting (exempt from the projection requirement) ----------------------------
    FieldSpec("compile", "bool", False, "runtime_setting", projects_to="TrainConfig.compile",
              current_api={},
              known_divergence=True,
              divergence_note="Runtime setting, not a training-form field: an operational field "
                              "of POST /training/start (CLI --compile); the WebUI's base start "
                              "and Continue send it, sourced from the Runtime Settings file.",
              description="torch.compile toggle (Runtime Settings)."),
)


# --- convenience accessors (torch-free) -------------------------------------------------------
def by_name(name: str) -> FieldSpec:
    for f in FIELDS:
        if f.name == name:
            return f
    raise KeyError(name)


def names() -> Tuple[str, ...]:
    return tuple(f.name for f in FIELDS)


def in_scope(scope: str) -> Tuple[FieldSpec, ...]:
    return tuple(f for f in FIELDS if f.scope == scope)


def effective_fields() -> Tuple[FieldSpec, ...]:
    return tuple(f for f in FIELDS if f.effective)


def must_project(f: FieldSpec) -> bool:
    """Projection rule: every effective field must project into TrainConfig unless it is an
    explicit exception (a runtime setting, or an accepted-but-not-effective no-op field)."""
    return f.effective and f.scope not in _PROJECTION_EXEMPT_SCOPES


def bounds(constraint: Optional[Dict[str, Any]]) -> Tuple[Any, bool, Any, bool]:
    """Normalize a constraint dict to ``(lo, lo_exclusive, hi, hi_exclusive)``.

    Missing bounds are ``None``. ``{"gt": 0}`` -> ``(0, True, None, False)``.
    """
    if not constraint:
        return (None, False, None, False)
    lo = constraint.get("ge", constraint.get("gt"))
    hi = constraint.get("le", constraint.get("lt"))
    return (lo, "gt" in constraint, hi, "lt" in constraint)


def bound_words(constraint: Optional[Dict[str, Any]]) -> str:
    """A constraint's bounds in words, from `bounds`: "1 or more", "between 0
    and 1", "more than 0 and at most 1"; empty for none."""
    lo, lo_open, hi, hi_open = bounds(constraint)

    def num(value: Any) -> str:
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)

    if lo is not None and hi is not None:
        if not lo_open and not hi_open:
            return f"between {num(lo)} and {num(hi)}"
        return (f"{'more than' if lo_open else 'at least'} {num(lo)} and "
                f"{'less than' if hi_open else 'at most'} {num(hi)}")
    if lo is not None:
        return f"more than {num(lo)}" if lo_open else f"{num(lo)} or more"
    if hi is not None:
        return f"less than {num(hi)}" if hi_open else f"{num(hi)} or less"
    return ""


#: The sampler's "every neighbour" hop size (the gene task's first hop, README
#: U3), beside the field spec's per-hop floor of 1.
_ALL_NEIGHBOURS = -1
#: Each kind in a refusal's words.
_KIND_WORDS = {"int": "an integer", "float": "a number", "bool": "true or false",
               "str": "text", "list[int]": "a list of integers", "dict": "a mapping"}


def value_problem(name: str, value: Any, *, allow_none: bool = False) -> Optional[str]:
    """Why `value` is not a valid value of the field `name`, by its spec (kind,
    bounds, choices, list items), or None. Strict about kinds: a bool is not a
    number and a string is never coerced, so what is checked is what a run is
    given. `allow_none` admits None where the caller's default is None (a
    recipe value such as early stopping's patience)."""
    spec = by_name(name)
    if value is None:
        return None if allow_none or spec.default is None else f"{name} must be set, not null."
    numbers = (int, float)
    if spec.kind == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif spec.kind == "float":
        ok = isinstance(value, numbers) and not isinstance(value, bool)
    elif spec.kind == "bool":
        ok = isinstance(value, bool)
    elif spec.kind == "str":
        ok = isinstance(value, str)
    elif spec.kind == "list[int]":
        ok = isinstance(value, list) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in value)
    else:
        ok = isinstance(value, dict)
    if not ok:
        return f"{name} must be {_KIND_WORDS.get(spec.kind, spec.kind)}, not {value!r}."
    if spec.choices is not None and value not in spec.choices:
        return f"{name} must be one of {list(spec.choices)}, not {value!r}."

    def outside(number: Any, constraint: Optional[Dict[str, Any]]) -> bool:
        lo, lo_open, hi, hi_open = bounds(constraint)
        return ((lo is not None and (number <= lo if lo_open else number < lo))
                or (hi is not None and (number >= hi if hi_open else number > hi)))

    if spec.kind in ("int", "float") and outside(value, spec.valid):
        return f"{name} must be {bound_words(spec.valid)}, not {value!r}."
    if spec.kind == "list[int]":
        if spec.min_length is not None and len(value) < spec.min_length:
            return f"{name} needs at least {spec.min_length} item(s)."
        every = name == "num_neighbors"
        items = [v for v in value if not (every and v == _ALL_NEIGHBOURS)]
        if any(outside(v, spec.item_valid) for v in items):
            return (f"{name} items must be {bound_words(spec.item_valid)}"
                    f"{f' (or {_ALL_NEIGHBOURS}, every neighbour)' if every else ''}, "
                    f"not {value!r}.")
    return None
