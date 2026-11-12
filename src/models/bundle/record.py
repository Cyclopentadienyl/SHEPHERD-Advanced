"""
The model record a bundle checkpoint carries, and the one way to read it.
========================================================================
`docs/working/PLAN_PROVENANCE_CONTRACT.md` §9.1–§9.3. The training payload
(`src.training.checkpoint_payload`) writes these fields beside M3a's record;
serving, measurement and the operations read them through `read_model_record`
and build through `load_bundle`, so no consumer reads a spec its own way.

**Record version 2** marks a checkpoint the bundle path wrote. A checkpoint
without a `model_spec` was written before it: every reader refuses it with
one phrase (`WRITTEN_BEFORE_SPEC`), a list keeps it listed with that reason,
and nothing is guessed from its weight names (§9.3, hazard 1).

**Task first** (hazard 3): `load_bundle` compares the learning stage and task a
caller needs with the record before it builds anything.

**One input, one claim**: a record that names one input two ways (its
`data_roles` and its `training_input_digests`, the view's digest or rule in
two places) and names it differently is refused by `read_model_record`, so
every entry that reads it refuses it, whichever copy that entry trusts.

**Each value the reader compares is compared as plain data** (`_plain`): by
its kind as well as its content, so a value that is not plain data (a tensor
the safe loader also accepts) is refused naming its field, never compared.
The walk is total over what the safe loader reads: a list or mapping that
holds itself, or one nested or sized past what a record holds, is refused
naming its field, and a value compared with the reader's own (a version, a
stage) is judged by that value's kind without being walked (`_differs`).

**What `graph_exposure` says is read in its writer's words**
(`recorded_graph_use`): the strings `scripts/train_model.py` records there are
named here once (`EXPOSURE_TRAIN_EDGES` and the rest), the writer writes them
through these names, and a reader takes each for the parts of the edge split
it puts in graph use. A string no build wrote is never read as one: the
record's exposure is then not recorded, never "unseen".

Plain data only, so the payload still loads with `weights_only=True`.
Reading and checking a record needs no torch; only `load_bundle` builds.

Module: src/models/bundle/record.py
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.config.model_types import MODEL_RECORD_KEYS, WRITTEN_BEFORE_SPEC
from src.models.bundle.capabilities import model_capabilities
from src.models.bundle.spec import LEARNING_STAGES, TASKS, ModelSpec, SpecError, ViewShape, _shown
from src.utils.checkpoint_paths import (
    PLAIN_DEPTH,
    PLAIN_VALUES,
    NotPlainData,
    plain_data,
    text_problem,
)

#: The checkpoint `record_version` the bundle path writes (M3a's was 1).
MODEL_RECORD_VERSION = 2
OPERATIONS = ("new", "resume", "extend", "branch", "finetune")

#: Top-level payload keys this record owns (§9.1), defined once beside the
#: encoder kinds so the run listing can carry them without importing this.
RECORD_KEYS = MODEL_RECORD_KEYS


class ModelRecordError(ValueError):
    """A checkpoint the bundle path cannot run, with the reason."""


@dataclass
class ModelRecord:
    model_spec: ModelSpec
    graph_view: ViewShape
    operation: str
    family_id: str
    parent: Optional[Dict[str, Any]] = None
    weight_transfer: Optional[Dict[str, Any]] = None
    data_roles: Dict[str, Any] = field(default_factory=dict)
    graph_exposure: Dict[str, Any] = field(default_factory=dict)
    loss_identity: Dict[str, Any] = field(default_factory=dict)
    scoring_identity: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None

    @property
    def learning_stage(self) -> str:
        return self.model_spec.learning_stage

    @property
    def task(self) -> Optional[str]:
        return self.model_spec.task

    @property
    def capabilities(self) -> Dict[str, Any]:
        """What this model offers a measurement and what it needs, read from its
        spec (`src.models.bundle.capabilities.model_capabilities`): the one
        description every list carries (`listed_record`) and every new
        measurement is judged by (`src.evaluation.protocols`)."""
        return model_capabilities(self.model_spec)

    def validate(self) -> "ModelRecord":
        self.model_spec.validate()
        # A name first, so a value with no one answer to `in` (a tensor) is
        # refused, never asked.
        if not isinstance(self.operation, str) or self.operation not in OPERATIONS:
            raise ModelRecordError(f"unknown operation {_shown(self.operation)}")
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ModelRecordError("a model record names its family (the root run)")
        # The names a list shows are text an answer can carry (`text_problem`).
        why = text_problem(self.family_id)
        if why is not None:
            raise ModelRecordError(f"the record's family_id {why}")
        if (self.operation == "new") != (self.parent is None):
            raise ModelRecordError("a parent is recorded for every operation but 'new'")
        if self.parent is not None and not (
                isinstance(self.parent, dict) and isinstance(self.parent.get("sha256"), str)
                and self.parent["sha256"] and isinstance(self.parent.get("run"), str)
                and self.parent["run"]):
            raise ModelRecordError("a parent is named by the SHA-256 of its checkpoint file "
                                   "and its run")
        for key in ("sha256", "run") if self.parent is not None else ():
            why = text_problem(self.parent[key])
            if why is not None:
                raise ModelRecordError(f"the record's parent.{key} {why}")
        if self.operation == "finetune" and self.learning_stage != "finetune":
            raise ModelRecordError("a finetune operation produces a finetune spec")
        if self.operation == "new" and self.learning_stage != "base":
            raise ModelRecordError("a new run is a base; a task model comes from finetune")
        changes = (self.parent or {}).get("changes")
        if self.operation == "branch":
            # BACKLOG 23: a branch is its parent with the recorded changes,
            # each from the parent's value to the new one, its weights loaded;
            # `{}` when no setting changed (a same-settings branch, ops-wiring
            # round, decision B1). Not recorded is not "none changed": a
            # branch without `changes` is refused.
            if not (isinstance(changes, dict) and all(
                    isinstance(c, dict) and set(c) == {"parent", "new"}
                    for c in changes.values())):
                raise ModelRecordError("a branch records each setting it changed as "
                                       "parent.changes {name: {parent, new}}, {} when it "
                                       "changed none")
            if self.weight_transfer is None:
                raise ModelRecordError("a branch records how its parent's weights were loaded")
        elif changes is not None:
            raise ModelRecordError(f"only a branch records changed settings, not a "
                                   f"{self.operation} run")
        return self

    def to_payload(self) -> Dict[str, Any]:
        self.validate()
        return {
            "model_spec": self.model_spec.to_dict(),
            "graph_view": self.graph_view.record(),
            "learning_stage": self.learning_stage,
            "task": self.task,
            "operation": self.operation,
            "parent": dict(self.parent) if self.parent is not None else None,
            "family_id": self.family_id,
            "weight_transfer": (dict(self.weight_transfer)
                                if self.weight_transfer is not None else None),
            "data_roles": dict(self.data_roles),
            "graph_exposure": dict(self.graph_exposure),
            "loss_identity": dict(self.loss_identity),
            "scoring_identity": (dict(self.scoring_identity)
                                 if self.scoring_identity is not None else None),
            "selection": dict(self.selection) if self.selection is not None else None,
        }


def has_model_spec(payload: Mapping[str, Any]) -> bool:
    return isinstance(payload, Mapping) and payload.get("model_spec") is not None


def _plain(name: str, value: Any, consequence: str = "so it is not compared", *,
           finite: bool = True) -> Any:
    """A value the reader compares (the field `name`), as plain data that
    states its kind beside its content, or a `ModelRecordError` naming the
    field and saying `consequence` (what the caller does not do with it: a
    compared claim is not compared, a part a list shows is not shown).

    Read by the project's one rule for a value read from a file that an
    answer carries (`checkpoint_paths.plain_data`, lifted from here in review
    fix RC-R1, its bounds `PLAIN_DEPTH` and `PLAIN_VALUES` with it). A record
    is plain data, but the safe loader (`weights_only=True`) also accepts
    tensors, and Python's `==` gives no one answer for anything else (a
    tensor of several values has no truth value; one of a single value
    answers as that value) nor across kinds (`True == 1`). So each value is
    read as the kinds a record holds: null, a boolean, a finite number, a
    string that is text (`text_problem`: a lone surrogate is not, and no
    answer can carry one), a list (or tuple), a mapping keyed by such
    strings. Two values are the same exactly when their kinds and contents
    are; a boolean is never a number. Anything else, or a number that is not
    finite, is refused.

    **The walk ends for every value the safe loader reads** (the reviewer's
    K8-4r2 recheck: a rule field holding a list that holds itself made this
    walk recurse until `RecursionError`, and the whole model list answered
    HTTP 500): a list or mapping holding itself, nested past `PLAIN_DEPTH`
    levels or walked past `PLAIN_VALUES` values is refused, naming the path
    where the walk stopped. These bounds are the safe loader's, which reads
    any depth and shares values freely, so this reader passes them; the rule
    holds no value parsed from JSON to them (review fix RC-R1). `finite`
    False admits a number that is not finite, for a part a caller shows in
    the one form every answer carries (`run_plan.shown_value`)."""
    try:
        return plain_data(value, name, depth=PLAIN_DEPTH, values=PLAIN_VALUES, finite=finite)
    except NotPlainData as exc:
        raise ModelRecordError(f"the record's {exc}, {consequence}") from None


def _differs(name: str, value: Any, expected: Any) -> bool:
    """Whether the record's field `name` holds other than `expected`, a value
    the reader itself holds, of a known scalar kind (a version number, a
    learning stage, a task or none). Judged directly by that kind: a list or
    a mapping differs without being walked, and any other value is compared
    as plain data (`_plain`, which refuses one that is not)."""
    if isinstance(value, (list, tuple, Mapping)):
        return True
    return _plain(name, value) != _plain(name, expected)


def _as_written(value: Any) -> Any:
    return value


def _view_rule(rule: Any) -> Any:
    """A graph view rule in the form the view stores it: `build_graph_view`
    (`src/kg/graph_view.py`) keeps `exclude_relations` sorted, while the spec
    keeps the operator's order, so the two copies are compared in the view's
    form. Inlined, not imported: that module needs torch, and reading a record
    does not. A part that is not a rule, or exclusions that are not names, stay
    as written: as plain data they are then reported as differing, and
    anything else is refused naming its field (`_plain`)."""
    if not isinstance(rule, Mapping):
        return rule
    excluded = rule.get("exclude_relations", [])
    if isinstance(excluded, (list, tuple)) and all(isinstance(r, str) for r in excluded):
        excluded = sorted(excluded)
    return {**rule, "exclude_relations": excluded}


#: The claims one bundle record makes twice about one input (contract §9.1):
#: the model record's `data_roles` beside the training record's
#: `training_input_digests` (M2.1), the graph view's digest in three places and
#: its rule in two. Entries trust different copies (serving the training record
#: and the top-level view, training `data_roles`), so the one reader refuses a
#: record whose copies disagree, whichever copy is wrong. Each pair is
#: `(claim, claim, the form both are compared in)`, each form then compared as
#: plain data (`_plain`).
_SAME_INPUT = (
    ("data_roles.kg", "training_input_digests.kg", _as_written),
    ("graph_view.digest", "data_roles.graph_view.digest", _as_written),
    ("graph_view.digest", "data_roles.split.view_digest", _as_written),
    ("data_roles.graph_view.rule", "model_spec.graph_view", _view_rule),
    ("data_roles.task_fixture.record", "training_input_digests.task_fixture", _as_written),
    ("parent.sha256", "training_input_digests.resume_checkpoint", _as_written),
)


def _claim(payload: Mapping[str, Any], name: str) -> Any:
    """The value at a dotted `name`, or None where any part is absent or not a
    mapping (an absent claim contradicts nothing)."""
    value: Any = payload
    for key in name.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _shown_claim(name: str, value: Any) -> str:
    """A claim as a contradiction names it: a string by its first characters (a
    digest's prefix), escaped and bounded (`_shown`) unless each is printable."""
    if not isinstance(value, str):
        return name
    head = value[:12]
    return f"{name} {head if head.isprintable() else _shown(head)}..."


def _contradictions(payload: Mapping[str, Any]) -> List[str]:
    """Each pair of this record's claims about one input that disagree, compared
    only where both are present, as plain data (`_plain`, which refuses a
    claim that is not, naming it). The task's case files are a task fixture's
    when the run read one (`data_roles.task_fixture`), the workspace's samples
    otherwise, as `train_model` records them."""
    task = "task_" if _claim(payload, "data_roles.task_fixture") is not None else ""
    pairs = _SAME_INPUT + tuple(
        (f"data_roles.task_data.{part}.samples", f"training_input_digests.{task}{part}_samples",
         _as_written) for part in ("train", "val"))
    found: List[str] = []
    for first, second, form in pairs:
        one, other = _claim(payload, first), _claim(payload, second)
        if one is None or other is None:
            continue
        if _plain(first, form(one)) != _plain(second, form(other)):
            found.append(f"{_shown_claim(first, one)} differs from {_shown_claim(second, other)}")
    return found


def read_model_record(payload: Mapping[str, Any]) -> ModelRecord:
    """The model record of a loaded checkpoint, or the reason it has none,
    a record that contradicts itself among them (`_SAME_INPUT`)."""
    if not has_model_spec(payload):
        raise ModelRecordError(WRITTEN_BEFORE_SPEC)
    if _differs("record_version", payload.get("record_version"), MODEL_RECORD_VERSION):
        raise ModelRecordError(f"checkpoint record version {_shown(payload.get('record_version'))} "
                               f"is not {MODEL_RECORD_VERSION}; this build reads only its own")
    missing = [k for k in RECORD_KEYS if k not in payload]
    if missing:
        raise ModelRecordError(f"the model record lacks {missing}")
    try:
        spec = ModelSpec.from_dict(payload["model_spec"])
        view = ViewShape.from_record(payload["graph_view"])
    except (SpecError, TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ModelRecordError(f"the model spec cannot be read: {exc}") from exc
    if (_differs("learning_stage", payload["learning_stage"], spec.learning_stage)
            or _differs("task", payload["task"], spec.task)):
        raise ModelRecordError("the record's learning stage or task differs from its model spec")
    # Each part as plain data of its kind, or the record is refused as a whole,
    # never half read (a list shows it as refused, with this reason).
    for key in ("data_roles", "graph_exposure", "loss_identity"):
        if not isinstance(payload[key], dict):
            raise ModelRecordError(f"the record's {key} is not a mapping")
    for key in ("parent", "weight_transfer", "scoring_identity", "selection"):
        if payload[key] is not None and not isinstance(payload[key], dict):
            raise ModelRecordError(f"the record's {key} is neither a mapping nor null")
    contradictions = _contradictions(payload)
    if contradictions:
        raise ModelRecordError(
            f"the record contradicts itself: {'; '.join(contradictions)}. One file names one "
            "input two ways, so neither claim is used")
    record = ModelRecord(
        model_spec=spec, graph_view=view, operation=payload["operation"],
        family_id=payload["family_id"], parent=payload["parent"],
        weight_transfer=payload["weight_transfer"], data_roles=dict(payload["data_roles"]),
        graph_exposure=dict(payload["graph_exposure"]),
        loss_identity=dict(payload["loss_identity"]),
        scoring_identity=payload["scoring_identity"], selection=payload["selection"])
    return record.validate()


def check_model_record(record: ModelRecord, *, learning_stage: Optional[str] = None,
                       task: Optional[str] = None, view_digest: Optional[str] = None) -> None:
    """Refuse a record that is not what the caller needs: the stage and task
    first, then the graph view."""
    if learning_stage is not None:
        if learning_stage not in LEARNING_STAGES:
            raise ValueError(f"unknown learning stage {learning_stage!r}")
        if record.learning_stage != learning_stage:
            raise ModelRecordError(f"a {record.learning_stage} checkpoint, not a "
                                   f"{learning_stage} one")
    if task is not None:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        if record.task != task:
            raise ModelRecordError(f"a checkpoint for task {record.task!r}, not {task!r}")
    if view_digest is not None and record.graph_view.digest != view_digest:
        raise ModelRecordError(f"built on graph view {record.graph_view.digest[:12]}, not "
                               f"{view_digest[:12]}")


# ---------------------------------------------------------------------------
# A record's graph exposure, in its writer's words
# ---------------------------------------------------------------------------
#: The words `scripts/train_model.py`, their one writer, records in a model
#: record's ``graph_exposure`` (contract §9.1): what each phase of a run passed
#: messages over and supervised with. The writer writes them through these
#: names and a reader reads them through `recorded_graph_use`, so the two
#: cannot drift apart.
EXPOSURE_TRAIN_EDGES = "train edges"
EXPOSURE_TRAIN_EDGES_DRAWN_PAIRS_REMOVED = "train edges, the step's drawn pairs removed (F5)"
EXPOSURE_VAL_EDGES = "val edges"
EXPOSURE_FULL_VIEW = "the full view"
EXPOSURE_FULL_VIEW_SP = "the full view (README G4, §6)"
EXPOSURE_TRAIN_CASES = "train cases"
EXPOSURE_VAL_CASES = "val cases"

#: Every part of an edge split (`src.kg.graph_view.SPLIT_NAMES`, restated:
#: this module imports no graph code): what a word naming the full view puts
#: in graph use.
EXPOSURE_ALL_PARTS: Tuple[str, ...] = ("train", "val", "test")

#: The parts of the edge split each word puts in graph use: a word naming
#: edges, the part it names; a word naming the full view, every part; a word
#: naming patient cases, no edge.
EXPOSURE_EDGE_PARTS: Dict[str, Tuple[str, ...]] = {
    EXPOSURE_TRAIN_EDGES: ("train",),
    EXPOSURE_TRAIN_EDGES_DRAWN_PAIRS_REMOVED: ("train",),
    EXPOSURE_VAL_EDGES: ("val",),
    EXPOSURE_FULL_VIEW: EXPOSURE_ALL_PARTS,
    EXPOSURE_FULL_VIEW_SP: EXPOSURE_ALL_PARTS,
    EXPOSURE_TRAIN_CASES: (),
    EXPOSURE_VAL_CASES: (),
}

#: The words that name the full view: a use by one of them is full-view
#: exposure, whatever part an edge is in.
EXPOSURE_FULL_VIEW_WORDS: Tuple[str, ...] = (EXPOSURE_FULL_VIEW, EXPOSURE_FULL_VIEW_SP)


def recorded_graph_use(graph_exposure: Any) -> Dict[str, Tuple[str, ...]]:
    """What a record's ``graph_exposure`` says each use of the graph covered,
    read in its writer's words: ``{"<stage>.<phase>.<use>": parts}`` for every
    string the record holds (`EXPOSURE_EDGE_PARTS`), in the record's order.

    The record is one learning stage's block (``base`` or ``finetune``), a
    mapping of phases (and, for a gene finetune, ``sp``) whose every value is
    a mapping or one of the writer's words.

    Raises:
        ValueError: the record says something this build's writer never wrote,
            the reason naming where: no stage block, a stage this build does
            not know, a value that is neither a mapping nor a word, or a word
            no build wrote. Its exposure is then not recorded.
    """
    if not isinstance(graph_exposure, Mapping) or len(graph_exposure) != 1:
        raise ValueError("the record's graph_exposure is not one learning stage's block")
    (stage, block), = graph_exposure.items()
    if not isinstance(stage, str) or stage not in LEARNING_STAGES:
        raise ValueError(f"the record's graph_exposure names the stage {_shown(stage)}, which "
                         "this build does not write")
    uses: Dict[str, Tuple[str, ...]] = {}

    def walk(at: str, value: Any, depth: int) -> None:
        if isinstance(value, str):
            if value not in EXPOSURE_EDGE_PARTS:
                raise ValueError(f"the record's graph_exposure.{at} says {_shown(value)}, which "
                                 "no build's writer wrote")
            uses[at] = EXPOSURE_EDGE_PARTS[value]
            return
        if not isinstance(value, Mapping) or not value or depth > 2:
            raise ValueError(f"the record's graph_exposure.{at} is not a use this build's "
                             "writer wrote")
        for key, part in value.items():
            walk(f"{at}.{key}", part, depth + 1)

    if not isinstance(block, Mapping):
        raise ValueError(f"the record's graph_exposure.{stage} is not a mapping of phases")
    walk(stage, block, 0)
    return uses


def _weights_problem(payload: Any) -> Optional[str]:
    """Why a read checkpoint holds no weights a child could start from, as
    the rest of a sentence naming it, or None: **the one rule of a parent's
    weights**, a finetune's and a branch's alike (`finetune_parent_problem`,
    `branch_parent_problem`)."""
    if not isinstance(payload, Mapping) or payload.get("state_dict") is None:
        return "holds no model weights."
    return None


def finetune_parent_problem(record: ModelRecord, payload: Mapping[str, Any]) -> Optional[str]:
    """Why a read checkpoint cannot be a finetune's parent (contract §9.2), as
    the rest of a sentence naming it, or None: a base, then its weights;
    children never chain. The training process
    (`train_model.read_finetune_parent`) and the job runner's Fine-tune start
    ask it of one read alike; its graph is compared with the workspace's
    where the workspace is read."""
    if record.learning_stage != "base":
        return (f"is a {record.learning_stage} checkpoint; a finetune's parent is a base, "
                "and children never chain.")
    return _weights_problem(payload)


def branch_parent_problem(record: ModelRecord, payload: Mapping[str, Any], learning_stage: Any,
                          task: Any, name: str) -> Optional[str]:
    """Why a read checkpoint, the file `name`, cannot be the parent of a
    branch of `learning_stage` and `task` (contract §9.2), as a sentence, or
    None: a branch keeps its parent's stage and task, then starts from its
    weights, the stage first as a finetune's parent's is
    (`finetune_parent_problem`). The training process
    (`train_model.read_branch_parent`) and the job runner's preview
    (`TrainingManager.preview_branch`) ask it of one read alike, so the two
    refuse a parent in one sentence."""
    if record.learning_stage != learning_stage or record.task != task:
        return (f"The branch parent {name} is a {record.learning_stage} checkpoint"
                + (f" for {record.task}" if record.task else "")
                + "; a branch keeps its parent's stage and task.")
    problem = _weights_problem(payload)
    return f"The branch parent {name} {problem}" if problem else None


#: What a model list shows of a checkpoint's record (`listed_record`).
LISTED_RECORD_FIELDS = ("record", "record_problem", "learning_stage", "task", "operation",
                        "family_id", "parent_sha256", "parent_run", "parent_changes",
                        "encoder", "reference",
                        "differences", "capabilities", "spec", "weight_transfer",
                        "loss_identity")
#: The parts of a record a list shows as the record holds them (`listed_record`),
#: each read as plain data first (`_plain`).
_SHOWN_AS_RECORDED = ("weight_transfer", "loss_identity")


def listed_record(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """What a model list shows of a checkpoint's record, read by
    `read_model_record` alone, so the lists apply no rule of their own.

    `record` is "model" (a record this build reads) or "refused" (a record
    this build cannot read, its reason in `record_problem`; a file without a
    model spec, or a read that is no record, says `WRITTEN_BEFORE_SPEC`, as a
    load of it does). A model adds its learning stage,
    task, operation, family (the root run), parent (the SHA-256 of the exact
    parent file and the parent run), `parent_changes` (a branch's
    ``parent.changes`` as recorded, ``{}`` for a same-settings branch,
    ops-wiring round decision B4; None for every other operation, which
    records none; each number that is not finite as its text, as the
    training settings show it, `run_plan.shown_value`), encoder kind, and how
    its spec stands to
    the reference recipe (README §5, axis 2): `differences`, the sorted names
    (``section.field``) of the spec's fields that differ from it
    (`ModelSpec.differences_from_reference`), and `reference`, true exactly
    when there are none; and `capabilities`, its capability description
    (`ModelRecord.capabilities`), from which the Test page, the job runner,
    the command line and the probe judge a new measurement of it
    (`src.evaluation.protocols`). Reading, comparing and exporting what is
    kept never consult it. `spec` is the model's structure as its record
    holds it, every value (`ModelSpec.to_dict`, without the reference
    recipe's provenance notes, `sources`): two variants that differ on one
    field differ here by value, whatever today's reference recipe is
    (research comparison Stage 8, the results export). `weight_transfer` and
    `loss_identity` are those parts of its record as it holds them (research
    comparison Stage 9b fixes): how its stage's weights began (where the
    record names them, each part trainable or frozen, the frozen parts'
    forward mode and the SHA-256 of the downstream weights training started
    from; none for a base, which loads nothing) and its loss (its terms and
    coefficients, and the terms without a trainable parameter), so two arms
    of one base are checked to start alike where they are listed and
    compared. A part that is not plain data (`_plain`) refuses the summary as
    a whole, with why, never half shown."""
    from src.utils.run_plan import shown_value

    summary: Dict[str, Any] = dict.fromkeys(LISTED_RECORD_FIELDS)
    try:
        record = read_model_record(payload if payload is not None else {})
        for name in _SHOWN_AS_RECORDED:
            _plain(name, getattr(record, name), "so a list does not show the record")
        changes = (record.parent or {}).get("changes") if record.operation == "branch" else None
        _plain("parent.changes", changes, "so a list does not show the record", finite=False)
    except ModelRecordError as exc:
        summary.update(record="refused", record_problem=str(exc))
        return summary
    parent = record.parent or {}
    differences = tuple(sorted(record.model_spec.differences_from_reference()))
    summary.update(record="model", learning_stage=record.learning_stage, task=record.task,
                   operation=record.operation, family_id=record.family_id,
                   parent_sha256=parent.get("sha256"), parent_run=parent.get("run"),
                   parent_changes=shown_value(copy.deepcopy(changes)),
                   encoder=record.model_spec.encoder.kind, reference=not differences,
                   differences=differences, capabilities=record.capabilities,
                   spec={name: value for name, value in record.model_spec.to_dict().items()
                         if name != "sources"},
                   **{name: copy.deepcopy(getattr(record, name)) for name in _SHOWN_AS_RECORDED})
    return summary


def load_bundle(payload: Mapping[str, Any], *, learning_stage: Optional[str] = None,
                task: Optional[str] = None, view_digest: Optional[str] = None) -> tuple:
    """Read, check and build a checkpoint's bundle, its weights loaded strictly.
    Returns `(ModelBundle, ModelRecord)`."""
    from src.models.bundle.builder import build_bundle

    record = read_model_record(payload)
    check_model_record(record, learning_stage=learning_stage, task=task, view_digest=view_digest)
    state = payload.get("state_dict")
    if state is None:
        raise ModelRecordError("the checkpoint holds no model weights")
    bundle = build_bundle(record.model_spec, record.graph_view)
    try:
        bundle.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ModelRecordError(f"the weights do not match the model spec: {exc}") from exc
    return bundle, record


__all__ = [
    "EXPOSURE_ALL_PARTS",
    "EXPOSURE_EDGE_PARTS",
    "EXPOSURE_FULL_VIEW",
    "EXPOSURE_FULL_VIEW_SP",
    "EXPOSURE_FULL_VIEW_WORDS",
    "EXPOSURE_TRAIN_CASES",
    "EXPOSURE_TRAIN_EDGES",
    "EXPOSURE_TRAIN_EDGES_DRAWN_PAIRS_REMOVED",
    "EXPOSURE_VAL_CASES",
    "EXPOSURE_VAL_EDGES",
    "LISTED_RECORD_FIELDS",
    "MODEL_RECORD_VERSION",
    "ModelRecord",
    "ModelRecordError",
    "OPERATIONS",
    "PLAIN_DEPTH",
    "PLAIN_VALUES",
    "RECORD_KEYS",
    "branch_parent_problem",
    "check_model_record",
    "finetune_parent_problem",
    "has_model_spec",
    "listed_record",
    "load_bundle",
    "read_model_record",
    "recorded_graph_use",
]
