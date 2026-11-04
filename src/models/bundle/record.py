"""
The model record a bundle checkpoint carries, and the one way to read it.
========================================================================
`docs/working/PLAN_PROVENANCE_CONTRACT.md` §9.1–§9.3. The training payload
(`src.training.checkpoint_payload`) writes these fields beside M3a's record;
serving, measurement and the operations read them through `read_model_record`
and build through `load_bundle`, so no consumer reads a spec its own way.

**Record version 2** marks a checkpoint the bundle path wrote. A checkpoint
without a `model_spec` was written before it: it is listed as history and
never run, and nothing is guessed from its weight names (§9.3, hazard 1).

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

Plain data only, so the payload still loads with `weights_only=True`.
Reading and checking a record needs no torch; only `load_bundle` builds.

Module: src/models/bundle/record.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.config.model_types import MODEL_RECORD_KEYS
from src.models.bundle.spec import LEARNING_STAGES, TASKS, ModelSpec, SpecError, ViewShape, _shown

#: The checkpoint `record_version` the bundle path writes (M3a's was 1).
MODEL_RECORD_VERSION = 2
OPERATIONS = ("new", "resume", "extend", "branch", "finetune")

#: Top-level payload keys this record owns (§9.1), defined once beside the
#: encoder kinds so the run listing can carry them without importing this.
RECORD_KEYS = MODEL_RECORD_KEYS


class ModelRecordError(ValueError):
    """A checkpoint the bundle path cannot run, with the reason."""


HISTORY_ONLY = ("written before the model spec: listed as history and never run by this "
                "build; start a new run")


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

    def validate(self) -> "ModelRecord":
        self.model_spec.validate()
        # A name first, so a value with no one answer to `in` (a tensor) is
        # refused, never asked.
        if not isinstance(self.operation, str) or self.operation not in OPERATIONS:
            raise ModelRecordError(f"unknown operation {_shown(self.operation)}")
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ModelRecordError("a model record names its family (the root run)")
        if (self.operation == "new") != (self.parent is None):
            raise ModelRecordError("a parent is recorded for every operation but 'new'")
        if self.parent is not None and not (
                isinstance(self.parent, dict) and isinstance(self.parent.get("sha256"), str)
                and self.parent["sha256"] and isinstance(self.parent.get("run"), str)
                and self.parent["run"]):
            raise ModelRecordError("a parent is named by the SHA-256 of its checkpoint file "
                                   "and its run")
        if self.operation == "finetune" and self.learning_stage != "finetune":
            raise ModelRecordError("a finetune operation produces a finetune spec")
        if self.operation == "new" and self.learning_stage != "base":
            raise ModelRecordError("a new run is a base; a task model comes from finetune")
        changes = (self.parent or {}).get("changes")
        if self.operation == "branch":
            # BACKLOG 23: a branch is its parent with the recorded changes,
            # each from the parent's value to the new one, its weights loaded.
            if not (isinstance(changes, dict) and changes and all(
                    isinstance(c, dict) and set(c) == {"parent", "new"}
                    for c in changes.values())):
                raise ModelRecordError("a branch records each setting it changed as "
                                       "parent.changes {name: {parent, new}}")
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


#: How deep a value the reader compares may nest, and how many values it may
#: hold in all, to be walked: the deepest part a record compares, a graph view
#: rule, is a mapping holding a list of names. Past either, the value is
#: refused naming its field (`_plain`), so the walk is bounded however the
#: file was written (the safe loader reads any depth and shares values freely).
PLAIN_DEPTH = 16
PLAIN_VALUES = 10_000


def _plain(name: str, value: Any) -> Any:
    """A value the reader compares (the field `name`), as plain data that
    states its kind beside its content, or a `ModelRecordError` naming the
    field.

    A record is plain data, but the safe loader (`weights_only=True`) also
    accepts tensors, and Python's `==` gives no one answer for anything else
    (a tensor of several values has no truth value; one of a single value
    answers as that value) nor across kinds (`True == 1`). So each value is
    read as the kinds a record holds: null, a boolean, a finite number, a
    string, a list (or tuple), a mapping with string keys. Two values are
    the same exactly when their kinds and contents are; a boolean is never a
    number. Anything else, or a number that is not finite, is refused.

    **The walk ends for every value the safe loader reads** (the reviewer's
    K8-4r2 recheck: a rule field holding a list that holds itself made this
    walk recurse until `RecursionError`, and the whole model list answered
    HTTP 500). A scalar is judged where it stands; a list or mapping is
    walked only while it is not already on the path walked (a value holding
    itself), lies at most `PLAIN_DEPTH` levels down, and the walk has read
    at most `PLAIN_VALUES` values (one value held many times over counts
    each time); otherwise it is refused, naming the path where it stopped."""
    walked = [0]

    def read(at: str, part: Any, path: Tuple[int, ...]) -> Any:
        walked[0] += 1
        if walked[0] > PLAIN_VALUES:
            raise ModelRecordError(f"the record's {name} holds more than {PLAIN_VALUES} values, "
                                   "so it is not compared")
        if part is None:
            return ("null",)
        if isinstance(part, bool):
            return ("boolean", part)
        if isinstance(part, (int, float)):
            if isinstance(part, float) and not math.isfinite(part):
                raise ModelRecordError(f"the record's {at} is {part!r}, not a finite number, "
                                       "so it is not compared")
            return ("number", part)
        if isinstance(part, str):
            return ("string", part)
        if not isinstance(part, (list, tuple, Mapping)):
            raise ModelRecordError(f"the record's {at} is a {type(part).__name__}, not plain "
                                   "data, so it is not compared")
        if id(part) in path:
            raise ModelRecordError(f"the record's {at} is a {type(part).__name__} that contains "
                                   "itself, so it is not compared")
        if len(path) >= PLAIN_DEPTH:
            raise ModelRecordError(f"the record's {at} nests deeper than {PLAIN_DEPTH} levels, "
                                   "so it is not compared")
        path = path + (id(part),)
        if isinstance(part, Mapping):
            for key in part:
                if not isinstance(key, str):
                    raise ModelRecordError(f"the record's {at} has a key that is not a string "
                                           f"({type(key).__name__}), so it is not compared")
            return ("mapping", {k: read(f"{at}.{k}", v, path) for k, v in part.items()})
        return ("list", tuple(read(f"{at}[{i}]", v, path) for i, v in enumerate(part)))

    return read(name, value, ())


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
    return f"{name} {value[:12]}..." if isinstance(value, str) else name


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
        raise ModelRecordError(HISTORY_ONLY)
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


def finetune_parent_problem(record: ModelRecord, payload: Mapping[str, Any]) -> Optional[str]:
    """Why a read checkpoint cannot be a finetune's parent (contract §9.2), as
    the rest of a sentence naming it, or None: a base with weights; children
    never chain. Its graph is compared with the workspace's where the
    workspace is read (`train_model.read_finetune_parent`)."""
    if record.learning_stage != "base":
        return (f"is a {record.learning_stage} checkpoint; a finetune's parent is a base, "
                "and children never chain.")
    if not isinstance(payload, Mapping) or payload.get("state_dict") is None:
        return "holds no model weights."
    return None


#: What a list says of a checkpoint without a model spec (README §4: listed as
#: history, never run). The WebUI never loads, measures or continues one, and
#: serving refuses one at load; only the ShepherdGNN path outside it (the
#: command line's Mode C measurement) still runs such a file, until K8 retires
#: that path (K8-6b; README §7).
LISTED_WITHOUT_SPEC = ("no model spec (a ShepherdGNN checkpoint): listed as history and never "
                       "run from the WebUI; only the ShepherdGNN path outside it (command "
                       "line) still runs it, until that path is retired (K8)")

#: What a model list shows of a checkpoint's record (`listed_record`).
LISTED_RECORD_FIELDS = ("record", "record_problem", "learning_stage", "task", "operation",
                        "family_id", "parent_sha256", "parent_run", "encoder", "reference",
                        "differences")


def listed_record(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """What a model list shows of a checkpoint's record, read by
    `read_model_record` alone, so the lists apply no rule of their own.

    `record` is "model" (a record this build reads), "history" (no model spec:
    `LISTED_WITHOUT_SPEC`) or "refused" (a record this build cannot
    read, its reason in `record_problem`). A model adds its learning stage,
    task, operation, family (the root run), parent (the SHA-256 of the exact
    parent file and the parent run), encoder kind, and how its spec stands to
    the reference recipe (README §5, axis 2): `differences`, the sorted names
    (``section.field``) of the spec's fields that differ from it
    (`ModelSpec.differences_from_reference`), and `reference`, true exactly
    when there are none."""
    summary: Dict[str, Any] = dict.fromkeys(LISTED_RECORD_FIELDS)
    if payload is None or not has_model_spec(payload):
        summary.update(record="history", record_problem=LISTED_WITHOUT_SPEC)
        return summary
    try:
        record = read_model_record(payload)
    except ModelRecordError as exc:
        summary.update(record="refused", record_problem=str(exc))
        return summary
    parent = record.parent or {}
    differences = tuple(sorted(record.model_spec.differences_from_reference()))
    summary.update(record="model", learning_stage=record.learning_stage, task=record.task,
                   operation=record.operation, family_id=record.family_id,
                   parent_sha256=parent.get("sha256"), parent_run=parent.get("run"),
                   encoder=record.model_spec.encoder.kind, reference=not differences,
                   differences=differences)
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
    "HISTORY_ONLY",
    "LISTED_RECORD_FIELDS",
    "LISTED_WITHOUT_SPEC",
    "MODEL_RECORD_VERSION",
    "ModelRecord",
    "ModelRecordError",
    "OPERATIONS",
    "RECORD_KEYS",
    "check_model_record",
    "finetune_parent_problem",
    "has_model_spec",
    "listed_record",
    "load_bundle",
    "read_model_record",
]
